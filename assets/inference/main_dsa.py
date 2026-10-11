import torch
import torch.nn as nn
import torch.nn.functional as F
from inference_utils import device, sample_prompts, validation_loss, Timer, GPTLITE_CKPT_PATH  # also adds GPTlite to the path
from utils import get_batch, get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from main_kvcache import generate_kvcache


def lightning_index_scores(q, w, k):
  """ Index scores I[t, s] = sum_h w[t, h] * ReLU(q[t, h] . k[s]) of the lightning indexer, for the queries
      q [B, Sq, n_heads, d], the head weights w [B, Sq, n_heads] and a single key per entry k [B, Sk, d] """
  dots = F.relu(torch.einsum('bqhd,bkd->bqhk', q, k))  # [B, Sq, n_heads, Sk]
  return torch.einsum('bqhk,bqh->bqk', dots, w)        # [B, Sq, Sk]


def top_k_indices(scores, k):
  """ Indices of the k highest scores of each row. A stable sort breaks ties by keeping the earliest entries, so the
      selection is the same whether a sequence is processed at once or one token at a time """
  return scores.sort(dim=-1, descending=True, stable=True).indices[..., :k]


def indexer_loss(index_scores, attention_scores, visible):
  """ KL divergence between the attention distribution (the target: attention weights summed over heads and
      normalized) and the softmax of the index scores, over the entries each query can see.
      index_scores: [B, Sq, Sk], attention_scores: [B, H, Sq, Sk] logits, visible: [Sq, Sk] or [B, Sq, Sk] """
  with torch.no_grad():
    target = F.softmax(attention_scores.masked_fill(~visible.unsqueeze(-3), -1e9), dim=-1).sum(dim=1) * visible
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-9)
  log_index = F.log_softmax(index_scores.masked_fill(~visible, -1e9), dim=-1).masked_fill(~visible, 0.0)
  return F.kl_div(log_index.flatten(0, -2), target.flatten(0, -2), reduction='batchmean')


class MultiHeadAttention_DSA(nn.Module):
  """ GPTlite's multi-head attention with DeepSeek Sparse Attention (DSA): a small lightning indexer scores the
      past tokens of each query, and all heads attend only to the top_k tokens with the highest scores. The
      projections have the same names as in GPTlite, to load its weights. The KV cache of a layer holds the
      keys, the values and the indexer keys of the past tokens. The selection is applied as a mask over the
      scores of all tokens, which shows its effect on quality; the speedup needs a kernel that reads only the
      keys and values of the selected tokens """

  def __init__(self, d_model, n_heads, d_head, dropout_p, top_k, n_index_heads=4, d_index=16):
    super().__init__()
    self.n_heads, self.d_head, self.top_k = n_heads, d_head, top_k
    self.n_index_heads, self.d_index = n_index_heads, d_index
    self.query_proj = nn.Linear(d_model, n_heads * d_head)
    self.key_proj = nn.Linear(d_model, n_heads * d_head)
    self.value_proj = nn.Linear(d_model, n_heads * d_head)
    self.out_proj = nn.Linear(n_heads * d_head, d_model)
    self.dropout = nn.Dropout(dropout_p)
    # lightning indexer: a few query heads, a single key per token, and a weight per head
    self.index_query_proj = nn.Linear(d_model, n_index_heads * d_index)
    self.index_key_proj = nn.Linear(d_model, d_index)
    self.index_weight_proj = nn.Linear(d_model, n_index_heads)
    self.selection = "indexer"  # "dense" (used to train the indexer), "indexer" (top_k index scores) or "local" (top_k most recent tokens)
    self.indexer_loss = None    # KL divergence between the attention and the indexer, in dense mode

  def forward(self, x, kv_cache=None, causal_mask=True, max_seqlen=None):
    (B, S, _), H, D = x.shape, self.n_heads, self.d_head
    q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    k = self.key_proj(x).view(B, S, H, D).transpose(1, 2)
    v = self.value_proj(x).view(B, S, H, D).transpose(1, 2)
    x_index = x.detach()  # the indexer is trained on its own, without gradients into the rest of the model
    k_index = self.index_key_proj(x_index).unsqueeze(1)      # [B, 1, S, d_index]

    # If a cache is provided, prepend the past keys, values and indexer keys
    if kv_cache is not None:
      past_k, past_v, past_k_index = kv_cache
      k, v = torch.cat([past_k, k], dim=2), torch.cat([past_v, v], dim=2)
      k_index = torch.cat([past_k_index, k_index], dim=2)
      if max_seqlen is not None and k.size(2) > max_seqlen:
        k, v, k_index = k[:, :, -max_seqlen:], v[:, :, -max_seqlen:], k_index[:, :, -max_seqlen:]

    Sk = k.size(2)
    causal = torch.ones(S, Sk, dtype=torch.bool, device=x.device).tril(diagonal=Sk - S)  # [S, Sk]
    scores = q @ k.transpose(-2, -1) / D ** 0.5  # [B, H, S, Sk]
    self.indexer_loss = None
    if self.selection == "local":  # the top_k most recent tokens of each query
      n = min(self.top_k, Sk)
      position = torch.arange(Sk - S, Sk, device=x.device)  # position of each query in the keys
      top = (position[:, None] - torch.arange(n, device=x.device)).clamp(min=0).expand(B, S, n)
      visible = torch.zeros(B, S, Sk, dtype=torch.bool, device=x.device).scatter_(-1, top, True) & causal
    else:
      q_index = self.index_query_proj(x_index).view(B, S, self.n_index_heads, self.d_index)
      index_scores = lightning_index_scores(q_index, self.index_weight_proj(x_index), k_index[:, 0])  # [B, S, Sk]
      if self.selection == "dense":  # dense attention; the indexer learns to imitate it
        visible = causal
        self.indexer_loss = indexer_loss(index_scores, scores, causal)
      else:  # the top_k past tokens with the highest index scores
        top = top_k_indices(index_scores.masked_fill(~causal, float('-inf')), min(self.top_k, Sk))
        visible = torch.zeros(B, S, Sk, dtype=torch.bool, device=x.device).scatter_(-1, top, True) & causal

    weights = F.softmax(scores.masked_fill(~visible.unsqueeze(-3), float('-inf')), dim=-1)
    out = (self.dropout(weights) if self.training else weights) @ v  # [B, H, S, D]
    out = self.out_proj(out.transpose(1, 2).reshape(B, S, H * D))
    out = self.dropout(out) if self.training else out
    return out, (k, v, k_index)


class GPTlite_DSA(GPTlite_KVCache):
  """ GPTlite with a KV cache, whose blocks use DeepSeek Sparse Attention """

  def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, top_k):
    super().__init__(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen)
    for block in self.blocks:
      block.mha = MultiHeadAttention_DSA(d_model, n_heads, d_head, dropout_p, top_k)


def set_selection(model, selection):
  for block in model.blocks:
    block.mha.selection = selection


def warmup_indexer(model, data, n_steps, lr, batch_size, seqlen):
  """ Dense warm-up of DeepSeek-V3.2: the model runs dense attention and only the indexers are trained, to
      imitate the attention distribution of their layer """
  indexer_params = [p for name, p in model.named_parameters() if '.index_' in name]
  optimizer = torch.optim.Adam(indexer_params, lr=lr)
  set_selection(model, "dense")
  for _ in range(n_steps):
    x, _ = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    model(x.to(device))
    loss = sum(block.mha.indexer_loss for block in model.blocks)
    loss.backward()  # only the indexers receive gradients: their input and target are detached
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
  set_selection(model, "indexer")
  return model


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, train_data, valid_data, _, decode_fn = get_tiny_shakespeare_data()
  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gptlite_model_parameters()
  top_k = seqlen // 4 # each query attends to a quarter of the context window
  warmup_iters = 1000 # dense warm-up steps of the indexer

  # GPTlite with the weights of the trained model, plus untrained indexers
  model = GPTlite_DSA(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, top_k).to(device).eval()
  missing, _ = model.load_state_dict(torch.load(GPTLITE_CKPT_PATH, map_location=device), strict=False)
  assert all('.index_' in name for name in missing), "only the indexers should be missing from the checkpoint"

  for selection, name in (("dense", "Dense attention"),
                          ("local", f"The {top_k} most recent tokens"),
                          ("indexer", f"Top {top_k} tokens, untrained indexer")):
    set_selection(model, selection)
    print(f"{name}: validation loss {validation_loss(model, valid_data):.3f}")
  warmup_indexer(model, train_data, warmup_iters, lr, batch_size, seqlen)
  print(f"Top {top_k} tokens, trained indexer: validation loss {validation_loss(model, valid_data):.3f}")

  prompt = sample_prompts(valid_data, 1, seqlen // 4)
  with torch.inference_mode(), Timer() as timer:  # each decoding step attends to the top_k cached tokens
    tokens = generate_kvcache(model, prompt, seqlen - prompt.size(1), seqlen)
  print(f"Generated {tokens.size(1) - prompt.size(1)} tokens in {timer.elapsed:.2f} seconds:", decode_fn(tokens[0].tolist()))
