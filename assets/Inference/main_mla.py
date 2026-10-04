import torch
import torch.nn as nn
from inference_utils import device, load_model, sample_prompts, validation_loss, train_steps, Timer  # also adds GPTlite to the path
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache, scaled_dot_product_attention_kv_cache
from main_kvcache import generate_kvcache


class MultiHeadAttention_MLA(nn.Module):
  """ Multi-head Latent Attention (MLA): the keys and values of all heads are computed from a single latent
      vector per token, and only that vector is cached. The cache of a layer is a tuple with a tensor of
      shape [B, 1, S, d_latent]: the latent vector behaves like a single key-value head shared by all heads """

  def __init__(self, d_model, n_heads, d_head, dropout_p, d_latent):
    super().__init__()
    self.n_heads = n_heads
    self.d_head = d_head
    self.query_proj = nn.Linear(d_model, n_heads * d_head)
    self.kv_down_proj = nn.Linear(d_model, d_latent, bias=False)  # compresses each token into a latent vector
    self.key_up_proj = nn.Linear(d_latent, n_heads * d_head)      # recovers the keys of all heads
    self.value_up_proj = nn.Linear(d_latent, n_heads * d_head)    # recovers the values of all heads
    self.out_proj = nn.Linear(n_heads * d_head, d_model)
    self.dropout = nn.Dropout(dropout_p)
    self.absorb = True  # attention directly over the latent vectors, without recomputing keys and values

  def forward(self, x, kv_cache=None, causal_mask=True, max_seqlen=None):
    (B, S, _), H, D = x.shape, self.n_heads, self.d_head
    dropout = self.dropout if self.training else None
    q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    c = self.kv_down_proj(x).unsqueeze(1)                    # [B, 1, S, d_latent]

    # If a cache is provided, prepend the past latent vectors
    if kv_cache is not None:
      c = torch.cat([kv_cache[0], c], dim=2)  # [B, 1, S_past + S, d_latent]
      if max_seqlen is not None and c.size(2) > max_seqlen:
        c = c[:, :, -max_seqlen:]

    if self.absorb:
      # absorb the key up-projection into the query: q.(W_uk c + b_uk) = (W_uk^T q).c + q.b_uk, where the last
      # term adds the same value to all the scores of a query, so the softmax is not affected by it
      W_uk = self.key_up_proj.weight.view(H, D, -1)            # [H, D, d_latent]
      q_latent = torch.einsum('bhsd,hdc->bhsc', q, W_uk)       # [B, H, S, d_latent]
      # attention with the latent vectors as keys and values; the attention function divides the scores
      # by sqrt(d_latent), so we rescale the queries to divide by sqrt(d_head) as in the original attention
      q_latent = q_latent * (q_latent.size(-1) / D) ** 0.5
      out_latent = scaled_dot_product_attention_kv_cache(q_latent, c, c, causal_mask=causal_mask, dropout=dropout)
      # absorb the value up-projection: sum_t w_t (W_uv c_t + b_uv) = W_uv (sum_t w_t c_t) + b_uv
      W_uv = self.value_up_proj.weight.view(H, D, -1)          # [H, D, d_latent]
      out = torch.einsum('bhsc,hdc->bhsd', out_latent, W_uv) + self.value_up_proj.bias.view(H, 1, D)
    else:
      # recompute the keys and values of all heads from the latent vectors
      k = self.key_up_proj(c[:, 0]).view(B, -1, H, D).transpose(1, 2)    # [B, H, S_past + S, D]
      v = self.value_up_proj(c[:, 0]).view(B, -1, H, D).transpose(1, 2)
      out = scaled_dot_product_attention_kv_cache(q, k, v, causal_mask=causal_mask, dropout=dropout)

    out = out.transpose(1, 2).reshape(B, S, H * D)
    out = self.dropout(self.out_proj(out))
    return out, (c,)


class GPTlite_MLA(GPTlite_KVCache):
  """ GPTlite with a KV cache, whose blocks use Multi-head Latent Attention """

  def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, d_latent):
    super().__init__(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen)
    for block in self.blocks:
      block.mha = MultiHeadAttention_MLA(d_model, n_heads, d_head, dropout_p, d_latent)


def convert_mha_to_mla(state_dict, n_heads, d_head, d_latent):
  """ Converts the weights of a multi-head GPTlite into MLA: the key and value projections are stacked
      in a single matrix [W_k; W_v], and factorized with a truncated singular value decomposition (SVD)
      into an up-projection (U sqrt(S)) times a down-projection (sqrt(S) V^T) of rank d_latent """
  mla_state_dict = {name: param for name, param in state_dict.items()
                    if '.mha.key_proj.' not in name and '.mha.value_proj.' not in name}
  for prefix in {name[:name.index('mha.') + 4] for name in state_dict if '.mha.' in name}:
    W = torch.cat([state_dict[prefix + 'key_proj.weight'], state_dict[prefix + 'value_proj.weight']])  # [2*H*D, d_model]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    sqrt_S = S[:d_latent].sqrt()
    W_up = U[:, :d_latent] * sqrt_S  # [2*H*D, d_latent]
    mla_state_dict[prefix + 'kv_down_proj.weight'] = sqrt_S[:, None] * Vh[:d_latent]  # [d_latent, d_model]
    mla_state_dict[prefix + 'key_up_proj.weight'] = W_up[:n_heads * d_head]
    mla_state_dict[prefix + 'value_up_proj.weight'] = W_up[n_heads * d_head:]
    mla_state_dict[prefix + 'key_up_proj.bias'] = state_dict[prefix + 'key_proj.bias']
    mla_state_dict[prefix + 'value_up_proj.bias'] = state_dict[prefix + 'value_proj.bias']
  return mla_state_dict


def set_absorb(model, absorb):
  for block in model.blocks:
    block.mha.absorb = absorb


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, train_data, valid_data, _, _ = get_tiny_shakespeare_data()
  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gptlite_model_parameters()
  uptrain_iters = 1000 # short uptraining after the conversion
  model_mha = load_model(GPTlite_KVCache, vocab_size)
  prompt = sample_prompts(valid_data, 8, seqlen // 4)
  n_tokens = seqlen - prompt.size(1)

  # [W_k; W_v] has rank at most min(d_model, 2*n_heads*d_head): a latent of that size is an exact conversion
  full_rank = min(d_model, 2 * n_heads * d_head)
  for d_latent in (None, full_rank, full_rank // 2, full_rank // 4):
    if d_latent is None:  # the original multi-head model
      model, name, loss_converted = model_mha, "MHA", None
    else:
      model = GPTlite_MLA(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, d_latent).to(device).eval()
      model.load_state_dict(convert_mha_to_mla(model_mha.state_dict(), n_heads, d_head, d_latent))
      name = f"MLA with d_latent={d_latent}"
      with torch.inference_mode():  # absorbed and explicit attention compute the same output
        logits_absorbed = model(prompt)[0]
        set_absorb(model, False)
        logits_explicit = model(prompt)[0]
        set_absorb(model, True)
      print(f"{name}: absorbed and explicit attention match: {torch.allclose(logits_absorbed, logits_explicit, atol=1e-4)}")
      loss_converted = validation_loss(model, valid_data)
      train_steps(model, train_data, uptrain_iters, lr, batch_size, seqlen)
    loss = validation_loss(model, valid_data)

    with torch.inference_mode():
      _, kv_cache = model(prompt)
      generate_kvcache(model, prompt, 1, seqlen)  # warm-up
      cache_bytes = sum(t.numel() * t.element_size() for layer_cache in kv_cache for t in layer_cache)
      with Timer() as timer:
        generate_kvcache(model, prompt, n_tokens, seqlen)

    loss_info = f"{loss:.3f}" if loss_converted is None else f"{loss_converted:.3f} after conversion, {loss:.3f} after uptraining"
    print(f"{name}: KV cache {cache_bytes / prompt.numel():.0f} bytes per token, "
          f"{timer.elapsed / n_tokens * 1000:.2f} ms per token, validation loss {loss_info}")
