import torch
import torch.nn as nn
import torch.nn.functional as F
from inference_utils import load_model, sample_prompts, Timer  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from main_kvcache import generate_kvcache


class GPTlite_StaticCache(nn.Module):
  """ GPTlite with a static KV cache: buffers preallocated for max_seqlen tokens, where every step writes the
      keys and values of the new tokens in place. Attention runs over the whole buffer, with a mask for the
      positions not filled yet, so all tensors have the same shape at every decode step """

  def __init__(self, model, batch_size, max_seqlen):
    super().__init__()
    self.model = model
    n_heads, d_head = model.blocks[0].mha.n_heads, model.blocks[0].mha.d_head
    shape = (len(model.blocks), batch_size, n_heads, max_seqlen, d_head)
    weight = model.token_embedding.weight
    self.register_buffer('k_cache', torch.zeros(shape, dtype=weight.dtype, device=weight.device))
    self.register_buffer('v_cache', torch.zeros(shape, dtype=weight.dtype, device=weight.device))
    self.register_buffer('positions', torch.arange(max_seqlen, device=weight.device))

  def forward(self, tokens, input_pos):
    """ tokens: [B, T] new tokens, input_pos: [T] their positions. Returns the logits [B, T, vocab_size] """
    model, (B, T) = self.model, tokens.shape
    x = model.token_embedding(tokens) + model.position_embedding(input_pos)
    mask = input_pos[:, None] >= self.positions[None, :]  # [T, max_seqlen]: attend to the positions up to itself
    for i, block in enumerate(model.blocks):
      mha, H, D = block.mha, block.mha.n_heads, block.mha.d_head
      h = block.ln1(x)
      q = mha.query_proj(h).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
      k = mha.key_proj(h).view(B, T, H, D).transpose(1, 2)
      v = mha.value_proj(h).view(B, T, H, D).transpose(1, 2)
      self.k_cache[i, :, :, input_pos] = k  # write the new keys and values in place
      self.v_cache[i, :, :, input_pos] = v
      out = F.scaled_dot_product_attention(q, self.k_cache[i], self.v_cache[i], attn_mask=mask)
      x = x + mha.out_proj(out.transpose(1, 2).reshape(B, T, H * D))
      x = x + block.ffwd(block.ln2(x))
    return model.fc_out(model.ln(x))


def generate_static(model, prompt, n_tokens, decode_step):
  """ Greedy generation with the static cache: the prompt is processed by the model (prefill), and every
      new token by decode_step, which can be the same model or a compiled version of it """
  P = prompt.size(1)
  logits = model(prompt, torch.arange(P, device=prompt.device))
  tokens = [torch.argmax(logits[:, -1], dim=-1, keepdim=True)]
  for pos in range(P, P + n_tokens - 1):
    logits = decode_step(tokens[-1], torch.tensor([pos], device=prompt.device))
    tokens.append(torch.argmax(logits[:, -1], dim=-1, keepdim=True))
  return torch.cat([prompt] + tokens, dim=1)


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, _ = get_tiny_shakespeare_data()
  seqlen = get_gptlite_model_parameters()[6]
  batch_size, prompt_len = 8, seqlen // 4
  n_tokens = seqlen - prompt_len  # the static cache holds the whole sequence
  prompt = sample_prompts(valid_data, batch_size, prompt_len)

  model_kvcache = load_model(GPTlite_KVCache, vocab_size)
  model_static = GPTlite_StaticCache(load_model(GPTlite, vocab_size), batch_size, seqlen)
  # CUDA graphs ("reduce-overhead") need a GPU; on the CPU, torch.compile only fuses the kernels
  decode_compiled = torch.compile(model_static, mode="reduce-overhead" if torch.cuda.is_available() else "default", fullgraph=True)

  outputs = {}
  with torch.inference_mode():
    for name, generate_fn in (
      ("Dynamic KV cache", lambda: generate_kvcache(model_kvcache, prompt, n_tokens, seqlen)),
      ("Static KV cache", lambda: generate_static(model_static, prompt, n_tokens, model_static)),
      ("Static KV cache, compiled decode step", lambda: generate_static(model_static, prompt, n_tokens, decode_compiled)),
    ):
      generate_fn()  # warm-up, which also compiles and records the CUDA graph
      with Timer() as timer:
        outputs[name] = generate_fn()
      print(f"{name}: {timer.elapsed / n_tokens * 1000:.2f} ms per token")
  reference = outputs["Dynamic KV cache"]
  print(f"Same outputs: {all(torch.equal(reference, output) for output in outputs.values())}")

