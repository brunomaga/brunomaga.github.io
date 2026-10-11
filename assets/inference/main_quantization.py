import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from inference_utils import load_model, sample_prompts, validation_loss, model_size_mb, Timer  # also adds GPTlite to the path
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from main_kvcache import generate_kvcache


class QuantizedLinear(nn.Module):
  """ Linear layer with symmetric (absmax) weight-only quantization: INT8 with one scale per output channel,
      or INT4 with one scale per group of group_size consecutive weights of a row, packing two weights per byte.
      The weights are dequantized on the fly, and the computation runs in the precision of the input """

  def __init__(self, linear, bits, group_size=64):
    super().__init__()
    assert bits in (4, 8), "only 8 and 4 bits are supported"
    W = linear.weight.data  # [out_features, in_features]
    self.bits, (self.out_features, self.in_features) = bits, W.shape
    self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.clone(), requires_grad=False)
    q_max = 2 ** (bits - 1) - 1  # 127 for INT8, 7 for INT4
    if bits == 8:
      scale = W.abs().amax(dim=1, keepdim=True) / q_max  # [out_features, 1]
    else:
      self.group_size = group_size if self.in_features % group_size == 0 else self.in_features
      W = W.view(self.out_features, -1, self.group_size)   # [out_features, n_groups, group_size]
      scale = W.abs().amax(dim=2, keepdim=True) / q_max    # [out_features, n_groups, 1]
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


def quantize(model, bits, group_size=64):
  """ Replaces the linear layers of the transformer blocks by quantized ones. The embeddings and the
      output layer stay in full precision """
  linears = [(parent, name, child) for block in model.blocks for parent in block.modules()
             for name, child in parent.named_children() if isinstance(child, nn.Linear)]
  for parent, name, child in linears:
    setattr(parent, name, QuantizedLinear(child, bits, group_size))
  return model


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, _ = get_tiny_shakespeare_data()
  seqlen = get_gptlite_model_parameters()[6]
  model = load_model(GPTlite_KVCache, vocab_size)
  prompt = sample_prompts(valid_data, 8, seqlen // 4)
  n_tokens = seqlen - prompt.size(1)

  models = {
    "FP32": model,
    "BF16": copy.deepcopy(model).to(torch.bfloat16),
    "INT8 weights": quantize(copy.deepcopy(model), bits=8),
    "INT4 weights": quantize(copy.deepcopy(model), bits=4),
  }
  for name, model_obj in models.items():
    with torch.inference_mode():
      generate_kvcache(model_obj, prompt, 1, seqlen)  # warm-up
      with Timer() as timer:
        generate_kvcache(model_obj, prompt, n_tokens, seqlen)
    print(f"{name}: {model_size_mb(model_obj):.2f} MB, {timer.elapsed / n_tokens * 1000:.2f} ms per token, "
          f"validation loss {validation_loss(model_obj, valid_data):.3f}")

