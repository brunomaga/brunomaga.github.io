import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from inference_utils import device, load_model, sample_prompts, validation_loss, model_size_mb, Timer  # also adds GPTlite to the path
from utils import get_batch, get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from kvcache import generate_kvcache


def quantize_tensor(x, bits, dim=-1, symmetric=True):
  """ Integer quantization of x with one scale (and zero-point) for each slice of x along dim, i.e. the values
      along dim share a scale. Returns the integers q, the scales s and the zero-points z, so that x ≈ s * (q - z) """
  if symmetric:  # absmax: integers in [-2^(b-1), 2^(b-1) - 1], zero-point 0
    q_max = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(dim=dim, keepdim=True) / q_max).clamp(min=1e-8)
    zero = torch.zeros_like(scale)
    q = torch.clamp(torch.round(x / scale), -q_max - 1, q_max)
  else:  # min-max: integers in [0, 2^b - 1], the zero-point is the integer that represents 0
    q_max = 2 ** bits - 1
    x_min, x_max = x.amin(dim=dim, keepdim=True), x.amax(dim=dim, keepdim=True)
    scale = ((x_max - x_min) / q_max).clamp(min=1e-8)
    zero = torch.round(-x_min / scale)
    q = torch.clamp(torch.round(x / scale) + zero, 0, q_max)
  return q, scale, zero


def fake_quantize(x, bits, dim=-1, symmetric=True):
  """ x quantized and dequantized: the closest values that the integer format can represent """
  q, scale, zero = quantize_tensor(x, bits, dim, symmetric)
  return scale * (q - zero)


def fake_quantize_groups(W, bits, group_size):
  """ W quantized and dequantized with one scale per group of group_size consecutive weights of a row """
  return fake_quantize(W.unflatten(-1, (-1, group_size)), bits).flatten(-2)


class QuantizedLinear(nn.Module):
  """ Linear layer with symmetric (absmax) weight-only quantization: INT8 with one scale per output channel,
      or INT4 with one scale per group of group_size consecutive weights of a row, packing two weights per byte.
      The scales are computed from the weights, unless given (e.g. by GPTQ). The weights are dequantized on the
      fly, and the computation runs in the precision of the input """

  def __init__(self, linear, bits, group_size=64, scale=None):
    super().__init__()
    assert bits in (4, 8), "only 8 and 4 bits are supported"
    W = linear.weight.data  # [out_features, in_features]
    self.bits, (self.out_features, self.in_features) = bits, W.shape
    self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.clone(), requires_grad=False)
    q_max = 2 ** (bits - 1) - 1  # 127 for INT8, 7 for INT4
    if bits == 8:
      scale = W.abs().amax(dim=1, keepdim=True) / q_max if scale is None else scale.view(-1, 1)  # [out_features, 1]
    else:
      self.group_size = group_size if self.in_features % group_size == 0 else self.in_features
      W = W.view(self.out_features, -1, self.group_size)   # [out_features, n_groups, group_size]
      if scale is None:
        scale = W.abs().amax(dim=2, keepdim=True) / q_max  # [out_features, n_groups, 1]
    scale = scale.clamp(min=1e-8)
    q = torch.clamp(torch.round(W / scale), -q_max - 1, q_max).to(torch.int8)
    if bits == 4:  # shift the values from [-8, 7] to [0, 15] and pack two of them per byte
      q = (q + 8).to(torch.uint8).view(self.out_features, -1)
      q = q[:, 0::2] | (q[:, 1::2] << 4)
    self.register_buffer('qweight', q)
    self.register_buffer('scale', scale)

  def dequantize(self):
    """ Weight matrix in the precision of the scales """
    if self.bits == 8:
      return self.qweight.to(self.scale.dtype) * self.scale
    low, high = (self.qweight & 0x0F).to(torch.int8) - 8, (self.qweight >> 4).to(torch.int8) - 8
    q = torch.stack([low, high], dim=-1).view(self.out_features, -1, self.group_size)  # unpack and interleave
    return (q.to(self.scale.dtype) * self.scale).view(self.out_features, self.in_features)

  def forward(self, x):
    if self.bits == 8:
      # with one scale per output channel, the scale can be applied after the matrix multiplication
      out = F.linear(x, self.qweight.to(x.dtype)) * self.scale.view(-1).to(x.dtype)
      return out if self.bias is None else out + self.bias
    return F.linear(x, self.dequantize().to(x.dtype), self.bias)


def replace_linears(model, new_layer):
  """ Replaces every linear layer of the transformer blocks by new_layer(linear). The embeddings and the
      output layer stay in full precision """
  linears = [(parent, name, child) for block in model.blocks for parent in block.modules()
             for name, child in parent.named_children() if isinstance(child, nn.Linear)]
  for parent, name, child in linears:
    setattr(parent, name, new_layer(child))
  return model


def quantize(model, bits, group_size=64):
  """ Weight-only quantization of the linear layers of the transformer blocks, with round-to-nearest """
  return replace_linears(model, lambda linear: QuantizedLinear(linear, bits, group_size))


@torch.inference_mode()
def calibration_inputs(model, modules, data, n_batches, batch_size, seqlen):
  """ Inputs of the given modules on a calibration set, as [n_tokens, in_features] tensors """
  inputs = {module: [] for module in modules}
  def hook(module, args):
    inputs[module].append(args[0].reshape(-1, args[0].size(-1)))
  hooks = [module.register_forward_pre_hook(hook) for module in modules]
  with torch.random.fork_rng():  # the same calibration batches at every call
    torch.manual_seed(0)
    for _ in range(n_batches):
      x, _ = get_batch(data, batch_size=batch_size, seqlen=seqlen)
      model(x.to(device))
  for hook in hooks:
    hook.remove()
  return {module: torch.cat(xs) for module, xs in inputs.items()}


def gptq(linear, X, bits=4, group_size=64, damp=0.01):
  """ GPTQ: quantizes the weights of a linear layer one column (input feature) at a time, and updates the columns
      not yet quantized to compensate for the error, with the inverse of the Hessian H = 2 X^T X of the layer's
      reconstruction error, for the calibration inputs X: [n_tokens, in_features]. Sets the weights to their
      quantized values, and returns the scales [out_features, n_groups, 1] """
  W = linear.weight.data.float().clone()  # [out_features, in_features]
  n_in = W.size(1)
  group_size = n_in if bits == 8 else group_size  # INT8: one scale per output channel
  H = 2 * X.T.float() @ X.float()  # [in_features, in_features]
  H += damp * H.diagonal().mean() * torch.eye(n_in, device=W.device)  # dampening, for numerical stability
  # upper Cholesky factor of H^-1: row i holds the update of the columns after i once column i is quantized
  Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
  q_max = 2 ** (bits - 1) - 1
  scales = torch.zeros(W.size(0), n_in // group_size, 1, device=W.device)
  for i in range(n_in):
    g = i // group_size
    if i % group_size == 0:  # the scale of a group comes from its weights, as updated by the previous columns
      scales[:, g] = (W[:, i:i + group_size].abs().amax(dim=1, keepdim=True) / q_max).clamp(min=1e-8)
    q = torch.clamp(torch.round(W[:, i] / scales[:, g, 0]), -q_max - 1, q_max) * scales[:, g, 0]
    err = (W[:, i] - q) / Hinv[i, i]
    W[:, i:] -= err[:, None] * Hinv[i, i:][None, :]  # column i becomes q, and the next columns compensate
  linear.weight.data = W.to(linear.weight.dtype)
  return scales


def quantize_gptq(model, data, n_batches, batch_size, seqlen, bits=4, group_size=64):
  """ Quantizes the linear layers with GPTQ, one block at a time, so that the inputs of a block come from the
      blocks already quantized before it """
  for block in model.blocks:
    linears = [(parent, name, child) for parent in block.modules() for name, child in parent.named_children()
               if isinstance(child, nn.Linear)]
    inputs = calibration_inputs(model, [child for _, _, child in linears], data, n_batches, batch_size, seqlen)
    for parent, name, child in linears:
      scale = gptq(child, inputs[child], bits, group_size)
      setattr(parent, name, QuantizedLinear(child, bits, group_size, scale=scale))
  return model


def awq_scales(X, linears, bits, group_size, n_grid=20):
  """ AWQ: searches the scales s = mean|X_j|^alpha of the input channels that minimize the error of the outputs of
      the linears that read X, when their weights W * s are quantized and their input is X / s """
  act = X.abs().mean(dim=0)  # [in_features]
  outputs = [F.linear(X, linear.weight) for linear in linears]
  best_error, best_s = float('inf'), None
  for alpha in torch.linspace(0, 1, n_grid):
    s = act.pow(alpha).clamp(min=1e-4)
    s = s / (s.max() * s.min()).sqrt()  # center the scales around 1
    error = sum((F.linear(X / s, fake_quantize_groups(linear.weight * s, bits, group_size)) - out).pow(2).mean()
                for linear, out in zip(linears, outputs))
    if error < best_error:
      best_error, best_s = error, s
  return best_s


@torch.inference_mode()
def quantize_awq(model, data, n_batches, batch_size, seqlen, bits=4, group_size=64):
  """ AWQ: multiplies the input channels of every linear layer by the scales of awq_scales, divides the output
      of the previous operation by the same scales, then quantizes the weights with round-to-nearest """
  for block in model.blocks:
    mha, ffwd = block.mha, block.ffwd.net
    reads = [mha.query_proj, mha.out_proj, ffwd[0], ffwd[2]]  # the first linear layer to read each input
    inputs = calibration_inputs(model, reads, data, n_batches, batch_size, seqlen)
    # (operation before, linears after): the LayerNorms, the value projection (attention is linear in the values),
    # and the first MLP layer (ReLU(z / s) = ReLU(z) / s for s > 0)
    for before, linears, X in ((block.ln1, [mha.query_proj, mha.key_proj, mha.value_proj], inputs[mha.query_proj]),
                               (mha.value_proj, [mha.out_proj], inputs[mha.out_proj]),
                               (block.ln2, [ffwd[0]], inputs[ffwd[0]]),
                               (ffwd[0], [ffwd[2]], inputs[ffwd[2]])):
      s = awq_scales(X, linears, bits, group_size)
      if isinstance(before, nn.LayerNorm):
        before.weight.div_(s)
      else:  # a linear layer: divide its rows (outputs)
        before.weight.div_(s[:, None])
      before.bias.div_(s)
      for linear in linears:
        linear.weight.mul_(s)
  return quantize(model, bits, group_size)


def int8_matmul(x, W):
  """ x @ W^T with INT8 activations (one scale per token) and INT8 weights (one scale per output channel). The
      integers are multiplied and accumulated in INT32, as in INT8 tensor cores, then scaled back """
  xq, x_scale, _ = quantize_tensor(x, 8, dim=-1)  # [n_tokens, in_features], [n_tokens, 1]
  wq, w_scale, _ = quantize_tensor(W, 8, dim=-1)  # [out_features, in_features], [out_features, 1]
  out = xq.to(torch.int32) @ wq.to(torch.int32).T
  return out * x_scale * w_scale.T


class LLMInt8Linear(nn.Module):
  """ LLM.int8(): the input features with a value above the threshold in any token (the outliers) are multiplied
      in the precision of the input, and all the others in INT8 """

  def __init__(self, linear, threshold=6.0):
    super().__init__()
    self.weight, self.bias, self.threshold = linear.weight, linear.bias, threshold

  def forward(self, x):
    x2 = x.reshape(-1, x.size(-1))  # [n_tokens, in_features]
    outliers = (x2.abs() > self.threshold).any(dim=0)  # [in_features]
    out = int8_matmul(x2[:, ~outliers], self.weight[:, ~outliers])
    out = out + x2[:, outliers] @ self.weight[:, outliers].T
    return (out + self.bias).view(*x.shape[:-1], -1)


class W8A8Linear(nn.Module):
  """ INT8 weights and activations, with the activations quantized on the fly """

  def __init__(self, linear):
    super().__init__()
    self.weight, self.bias = linear.weight, linear.bias

  def forward(self, x):
    out = int8_matmul(x.reshape(-1, x.size(-1)), self.weight) + self.bias
    return out.view(*x.shape[:-1], -1)


@torch.inference_mode()
def smoothquant(model, data, n_batches, batch_size, seqlen, alpha=0.5):
  """ SmoothQuant: divides input channel j of the linear layers after each LayerNorm by
      s_j = max|X_j|^alpha / max|W_j|^(1-alpha), folding the division into the LayerNorm, and multiplies input
      channel j of their weights by s_j. The output does not change, but the activations have smaller outliers """
  for block in model.blocks:
    mha = block.mha
    groups = ((block.ln1, [mha.query_proj, mha.key_proj, mha.value_proj]), (block.ln2, [block.ffwd.net[0]]))
    inputs = calibration_inputs(model, [linears[0] for _, linears in groups], data, n_batches, batch_size, seqlen)
    for ln, linears in groups:
      x_max = inputs[linears[0]].abs().amax(dim=0)  # [in_features]
      w_max = torch.stack([linear.weight.abs().amax(dim=0) for linear in linears]).amax(dim=0)
      s = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
      ln.weight.div_(s)
      ln.bias.div_(s)
      for linear in linears:
        linear.weight.mul_(s)
  return model


def hadamard(n):
  """ Orthonormal rotation of size n built from Hadamard matrices (Sylvester's construction): one Hadamard
      matrix if n is a power of 2, otherwise a block-diagonal matrix of Hadamard matrices, of the largest power
      of 2 that divides n (e.g. 3 blocks of 128 for n = 384) """
  k = n & -n  # largest power of 2 that divides n
  H = torch.ones(1, 1)
  while H.size(0) < k:
    H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)])
  return torch.block_diag(*[H / k ** 0.5] * (n // k))


class FakeQuantLinear(nn.Module):
  """ Linear layer that simulates low-precision weights and activations: they are quantized and dequantized by
      weight_fn and act_fn (fake quantization), and the multiplication runs in the precision of the input.
      With rotate=True, the inputs and the weights are rotated by a Hadamard matrix R first: W x = (W R^T)(R x) """

  def __init__(self, linear, weight_fn, act_fn=None, rotate=False):
    super().__init__()
    W = linear.weight.data
    self.rotation = hadamard(W.size(1)).to(W.device) if rotate else None
    if rotate:
      W = W @ self.rotation.T
    self.register_buffer('weight', weight_fn(W))
    self.bias, self.act_fn = linear.bias, act_fn

  def forward(self, x):
    if self.rotation is not None:
      x = x @ self.rotation.T  # x is a row vector: R x = x R^T
    if self.act_fn is not None:
      x = self.act_fn(x)
    return F.linear(x, self.weight, self.bias)


FP4_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])  # the non-negative values of FP4 E2M1


def round_to_fp4(x):
  """ Rounds every value of x, in [-6, 6], to the nearest value of FP4 E2M1 """
  grid = FP4_E2M1.to(x.device, x.dtype)
  nearest = (x.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
  return torch.sign(x) * grid[nearest]


def fake_quantize_mxfp4(x):
  """ MXFP4: every block of 32 consecutive values shares a power-of-two scale (an E8M0 number). The scale maps
      the largest value of the block to [4, 8), the exponent of the largest FP4 value (6), and larger values
      are clipped to 6 """
  blocks = x.unflatten(-1, (-1, 32))
  amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=2 ** -126)
  scale = 2 ** (torch.floor(torch.log2(amax)) - 2)
  return (round_to_fp4((blocks / scale).clamp(-6, 6)) * scale).flatten(-2)


def fake_quantize_nvfp4(x):
  """ NVFP4: every block of 16 consecutive values shares an FP8 E4M3 scale, which maps its largest value to 6,
      and the tensor has one FP32 scale, which maps the largest block scale to 448, the largest FP8 E4M3 value """
  tensor_scale = (x.abs().amax() / (6 * 448)).clamp(min=1e-12)
  blocks = x.unflatten(-1, (-1, 16))
  block_scale = (blocks.abs().amax(dim=-1, keepdim=True) / 6 / tensor_scale).to(torch.float8_e4m3fn).to(x.dtype)
  scale = (block_scale * tensor_scale).clamp(min=1e-12)
  return (round_to_fp4((blocks / scale).clamp(-6, 6)) * scale).flatten(-2)


def fake_quantize_fp8(x):
  """ FP8 E4M3 with one scale per tensor, which maps the largest value to 448 """
  scale = (x.abs().amax() / 448).clamp(min=1e-12)
  return (x / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale


def quantize_kv_cache(kv_cache, bits=2, group_size=32, residual=32):
  """ KIVI: quantizes the keys per channel and the values per token, in groups of group_size tokens, with
      asymmetric quantization. The last tokens, at least residual of them, stay in full precision until they fill
      a group. Returns the dequantized cache: a list of (keys, values) of shape [B, H, S, D], one per block """
  new_cache = []
  for k, v in kv_cache:
    n = (k.size(2) - residual) // group_size * group_size  # the tokens of the full groups out of the window
    if n > 0:
      k_groups = k[:, :, :n].unflatten(2, (-1, group_size))  # [B, H, n/G, G, D]
      k_old = fake_quantize(k_groups, bits, dim=-2, symmetric=False).flatten(2, 3)  # one scale per channel
      v_old = fake_quantize(v[:, :, :n], bits, dim=-1, symmetric=False)              # one scale per token
      k, v = torch.cat([k_old, k[:, :, n:]], dim=2), torch.cat([v_old, v[:, :, n:]], dim=2)
    new_cache.append((k, v))
  return new_cache


@torch.inference_mode()
def generate_quantized_kvcache(model, prompt, n_tokens, seqlen, quantize_fn):
  """ Greedy generation with a KV cache that quantize_fn quantizes and dequantizes after every step """
  logits, kv_cache = model(prompt, max_seqlen=seqlen)
  tokens = prompt
  for i in range(n_tokens):
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_token], dim=-1)
    if i < n_tokens - 1:
      logits, kv_cache = model(next_token, kv_cache=quantize_fn(kv_cache), max_seqlen=seqlen)
  return tokens


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, train_data, valid_data, _, _ = get_tiny_shakespeare_data()
  batch_size, seqlen = get_gptlite_model_parameters()[4], get_gptlite_model_parameters()[6]
  calibration_batches = 4 # batches of the calibration set of GPTQ, AWQ and SmoothQuant
  model = load_model(GPTlite_KVCache, vocab_size)
  prompt = sample_prompts(valid_data, 8, seqlen // 4)
  n_tokens = seqlen - prompt.size(1)
  calibration = (train_data, calibration_batches, batch_size, seqlen)

  # weight-only quantization: memory, time per token and validation loss
  models = {
    "FP32": model,
    "BF16": copy.deepcopy(model).to(torch.bfloat16),
    "INT8 weights": quantize(copy.deepcopy(model), bits=8),
    "INT4 weights": quantize(copy.deepcopy(model), bits=4),
    "INT4 weights, GPTQ": quantize_gptq(copy.deepcopy(model), *calibration),
    "INT4 weights, AWQ": quantize_awq(copy.deepcopy(model), *calibration),
  }
  for name, model_obj in models.items():
    with torch.inference_mode():
      generate_kvcache(model_obj, prompt, 1, seqlen)  # warm-up
      with Timer() as timer:
        generate_kvcache(model_obj, prompt, n_tokens, seqlen)
    print(f"{name}: {model_size_mb(model_obj):.2f} MB, {timer.elapsed / n_tokens * 1000:.2f} ms per token, "
          f"validation loss {validation_loss(model_obj, valid_data):.3f}")

  # weights and activations, simulated with fake quantization: validation loss
  per_token_int4 = lambda x: fake_quantize(x, 4)  # one scale per token (activations) or output channel (weights)
  models = {
    "W8A8": replace_linears(copy.deepcopy(model), W8A8Linear),
    "W8A8, SmoothQuant": replace_linears(smoothquant(copy.deepcopy(model), *calibration), W8A8Linear),
    "LLM.int8()": replace_linears(copy.deepcopy(model), LLMInt8Linear),
    "W4A4": replace_linears(copy.deepcopy(model), lambda l: FakeQuantLinear(l, per_token_int4, per_token_int4)),
    "W4A4, Hadamard rotation": replace_linears(copy.deepcopy(model),
                                 lambda l: FakeQuantLinear(l, per_token_int4, per_token_int4, rotate=True)),
    "MXFP4 weights and activations": replace_linears(copy.deepcopy(model),
                                       lambda l: FakeQuantLinear(l, fake_quantize_mxfp4, fake_quantize_mxfp4)),
    "NVFP4 weights and activations": replace_linears(copy.deepcopy(model),
                                       lambda l: FakeQuantLinear(l, fake_quantize_nvfp4, fake_quantize_nvfp4)),
  }
  with torch.inference_mode():
    for name, model_obj in models.items():
      print(f"{name}: validation loss {validation_loss(model_obj, valid_data):.3f}")

  # KV cache quantization: generated tokens that match those of the full-precision cache
  reference = generate_quantized_kvcache(model, prompt, n_tokens, seqlen, lambda cache: cache)
  for name, quantize_fn in (
    ("FP8 KV cache", lambda cache: [(fake_quantize_fp8(k), fake_quantize_fp8(v)) for k, v in cache]),
    ("INT2 KV cache, KIVI", quantize_kv_cache),
  ):
    tokens = generate_quantized_kvcache(model, prompt, n_tokens, seqlen, quantize_fn)
    match = (tokens == reference)[:, prompt.size(1):].float().mean().item()
    print(f"{name}: {match * 100:.1f}% of the generated tokens match the full-precision cache")
