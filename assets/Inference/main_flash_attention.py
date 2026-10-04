import copy
import torch
import torch.nn.functional as F
from inference_utils import load_model, sample_prompts, Timer  # also adds GPTlite to the path
from gptlite import GPTlite, MultiHeadAttention
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters


class MultiHeadAttention_SDPA(MultiHeadAttention):
  """ GPTlite's multi-head attention, computed by PyTorch's fused scaled_dot_product_attention, which
      runs a FlashAttention (or memory-efficient) kernel when the GPU and the inputs allow it """

  def forward(self, x, causal_mask=True):
    (B, S, _), H, D = x.shape, self.n_heads, self.d_head
    q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    k = self.key_proj(x).view(B, S, H, D).transpose(1, 2)
    v = self.value_proj(x).view(B, S, H, D).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal_mask,
                                         dropout_p=self.dropout.p if self.training else 0.0)
    out = out.transpose(1, 2).reshape(B, S, H * D)
    return self.dropout(self.out_proj(out))


def use_sdpa(model):
  """ Replaces the attention of every block by the fused one, keeping the weights """
  for block in model.blocks:
    block.mha.__class__ = MultiHeadAttention_SDPA
  return model


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, _ = get_tiny_shakespeare_data()
  seqlen = get_gptlite_model_parameters()[6]
  model = load_model(GPTlite, vocab_size)
  model_sdpa = use_sdpa(copy.deepcopy(model))

  # prefill of a batch of full-length sequences, where attention costs the most
  batch_size, n_runs = 16, 20
  prompt = sample_prompts(valid_data, batch_size, seqlen)
  with torch.inference_mode():
    for name, model_obj in (("GPTlite", model), ("GPTlite with SDPA", model_sdpa)):
      model_obj(prompt)  # warm-up
      if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
      with Timer() as timer:
        for _ in range(n_runs):
          logits = model_obj(prompt)
      memory = f", peak memory {torch.cuda.max_memory_allocated() / 2**20:.1f} MB" if torch.cuda.is_available() else ""
      print(f"{name}: {timer.elapsed / n_runs * 1000:.2f} ms per forward pass{memory}")
    same = torch.allclose(model(prompt), model_sdpa(prompt), atol=1e-4)
  print(f"Same outputs: {same}")

