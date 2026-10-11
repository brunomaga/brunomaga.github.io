import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from inference_utils import device, validation_loss  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_batch, get_tiny_shakespeare_data, get_gptlite_model_parameters
from main_dsa import lightning_index_scores, indexer_loss, top_k_indices


class Compressor(nn.Module):
  """ Compresses the hidden states of every m tokens into one entry: a sum of per-token entries, weighted per
      channel by a softmax over the tokens of the block, with learned positional biases. With overlap (CSA),
      each entry also covers the m tokens of the previous block, through a second series of entries and weights.
      The tokens of a trailing incomplete block are not compressed (the sliding window covers them) """

  def __init__(self, d_model, d_entry, m, overlap):
    super().__init__()
    self.m, self.overlap = m, overlap
    n_series = 2 if overlap else 1
    self.entry_proj = nn.Linear(d_model, n_series * d_entry, bias=False)   # C^a (and C^b)
    self.weight_proj = nn.Linear(d_model, n_series * d_entry, bias=False)  # Z^a (and Z^b)
    self.position_bias = nn.Parameter(torch.zeros(n_series, m, d_entry))  # B^a (and B^b)

  def forward(self, h):
    """ h: [B, S, d_model] -> compressed entries [B, S // m, d_entry] """
    B, n_blocks = h.size(0), h.size(1) // self.m
    h = h[:, :n_blocks * self.m]
    C = self.entry_proj(h).view(B, n_blocks, self.m, -1)  # [B, n_blocks, m, n_series * d_entry]
    Z = self.weight_proj(h).view(B, n_blocks, self.m, -1)
    if self.overlap:
      (Ca, Cb), (Za, Zb) = C.chunk(2, dim=-1), Z.chunk(2, dim=-1)
      Za, Zb = Za + self.position_bias[0], Zb + self.position_bias[1]
      Cb = F.pad(Cb, (0, 0, 0, 0, 1, 0))[:, :n_blocks]  # series b of the previous block (none for block 0)
      Zb = F.pad(Zb, (0, 0, 0, 0, 1, 0), value=float('-inf'))[:, :n_blocks]
      C, Z = torch.cat([Ca, Cb], dim=2), torch.cat([Za, Zb], dim=2)  # [B, n_blocks, 2m, d_entry]
    else:
      Z = Z + self.position_bias[0]
    return (F.softmax(Z, dim=2) * C).sum(dim=2)


class CompressedAttention(nn.Module):
  """ DeepSeek-V4 attention. CSA (top_k given) compresses every m tokens with overlap, and a lightning indexer
      selects the top_k compressed entries of each query. HCA (top_k None) compresses every m tokens without
      overlap, and attends to all of them. Both use shared key-value multi-query attention: each entry is both
      the key and the value of all query heads, together with uncompressed entries of the last n_win tokens.
      The indexer is trained, as in DeepSeek Sparse Attention, to imitate the attention over the compressed entries.
      GPTlite adds positional embeddings to its input, so DeepSeek-V4's partial RoPE is not needed """

  def __init__(self, d_model, n_heads, d_head, dropout_p, m, n_win, top_k=None, n_groups=2, n_index_heads=4, d_index=16):
    super().__init__()
    self.n_heads, self.d_head, self.m, self.n_win, self.top_k = n_heads, d_head, m, n_win, top_k
    self.n_index_heads, self.d_index = n_index_heads, d_index
    d_q_latent = d_model // 2
    self.q_down_proj = nn.Linear(d_model, d_q_latent)  # low-rank queries; the latent is shared with the indexer
    self.q_up_proj = nn.Linear(d_q_latent, n_heads * d_head)
    self.compressor = Compressor(d_model, d_head, m, overlap=top_k is not None)
    self.window_proj = nn.Linear(d_model, d_head)  # uncompressed entries, for the sliding window
    self.q_norm, self.kv_norm = nn.RMSNorm(d_head), nn.RMSNorm(d_head)
    self.sink = nn.Parameter(torch.zeros(n_heads))  # attention sink: a learned logit per head, with no value
    if top_k is not None:  # lightning indexer, with keys compressed like the entries
      self.index_compressor = Compressor(d_model, d_index, m, overlap=True)
      self.index_query_proj = nn.Linear(d_q_latent, n_index_heads * d_index)
      self.index_weight_proj = nn.Linear(d_model, n_index_heads)
    # grouped output projection: each group of heads is projected separately, then all groups together
    self.n_groups, d_group = n_groups, d_model // n_groups
    self.group_proj = nn.Parameter(torch.randn(n_groups, n_heads // n_groups * d_head, d_group) * (n_heads // n_groups * d_head) ** -0.5)
    self.out_proj = nn.Linear(n_groups * d_group, d_model)
    self.dropout = nn.Dropout(dropout_p)
    self.indexer_loss = None  # KL divergence between the attention over the entries and the indexer, in training

  def forward(self, x, causal_mask=True):
    (B, S, _), H, D = x.shape, self.n_heads, self.d_head
    c_q = self.q_down_proj(x)
    q = self.q_norm(self.q_up_proj(c_q).view(B, S, H, D)).transpose(1, 2)  # [B, H, S, D]
    entries = self.kv_norm(self.compressor(x))                             # [B, n_blocks, D]
    window = self.kv_norm(self.window_proj(x))                             # [B, S, D]
    n_blocks, t = entries.size(1), torch.arange(S, device=x.device)

    # each query sees the compressed blocks before its own block, and the uncompressed last n_win tokens
    block_visible = torch.arange(n_blocks, device=x.device)[None, :] < (t // self.m)[:, None]   # [S, n_blocks]
    window_visible = (t[None, :] <= t[:, None]) & (t[None, :] > t[:, None] - self.n_win)        # [S, S]
    block_visible = block_visible.expand(B, S, n_blocks)
    self.indexer_loss = None
    if self.top_k is not None and n_blocks > 0:  # CSA: keep the top_k visible blocks with the highest index scores
      x_index, c_q_index = x.detach(), c_q.detach()  # the indexer is trained only by its own loss
      q_index = self.index_query_proj(c_q_index).view(B, S, self.n_index_heads, self.d_index)
      index_scores = lightning_index_scores(q_index, self.index_weight_proj(x_index), self.index_compressor(x_index))
      if self.training:
        self.indexer_loss = indexer_loss(index_scores, q @ entries.transpose(-2, -1).unsqueeze(1) / D ** 0.5, block_visible)
      top = top_k_indices(index_scores.masked_fill(~block_visible, float('-inf')), min(self.top_k, n_blocks))
      block_visible = torch.zeros_like(block_visible).scatter_(-1, top, True) & block_visible

    kv = torch.cat([entries, window], dim=1)[:, None]  # [B, 1, n_blocks + S, D]: keys and values of all heads
    visible = torch.cat([block_visible, window_visible.expand(B, S, S)], dim=-1)  # [B, S, n_blocks + S]
    scores = (q @ kv.transpose(-2, -1) / D ** 0.5).masked_fill(~visible[:, None], float('-inf'))
    sink = self.sink.view(1, H, 1, 1).expand(B, H, S, 1)
    weights = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]  # the sink takes part of the weight
    out = (self.dropout(weights) if self.training else weights) @ kv        # [B, H, S, D]
    out = out.transpose(1, 2).reshape(B, S, self.n_groups, -1)
    out = torch.einsum('bsgi,gio->bsgo', out, self.group_proj).reshape(B, S, -1)
    return self.dropout(self.out_proj(out))


def use_csa_hca(model, m, m_heavy, n_win, top_k):
  """ Replaces the attention of every block: CSA in even blocks, HCA in odd blocks """
  for i, block in enumerate(model.blocks):
    d_model, n_heads, d_head, dropout_p = block.ln1.normalized_shape[0], block.mha.n_heads, block.mha.d_head, block.mha.dropout.p
    if i % 2 == 0:
      block.mha = CompressedAttention(d_model, n_heads, d_head, dropout_p, m, n_win, top_k=top_k)
    else:
      block.mha = CompressedAttention(d_model, n_heads, d_head, dropout_p, m_heavy, n_win)
  return model.to(device)


def train(model, data, n_steps, lr, batch_size, seqlen):
  """ Trains a model with the cross entropy loss, plus the loss of the indexers of its CSA layers """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model.train()
  for _ in range(n_steps):
    x, y = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    logits = model(x.to(device))
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1))
    loss = loss + sum(block.mha.indexer_loss for block in model.blocks if getattr(block.mha, 'indexer_loss', None) is not None)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
  model.eval()
  return model


def kv_cache_values(model, seqlen):
  """ Values stored in the KV cache for one sequence of seqlen tokens, over all layers """
  total = 0
  for block in model.blocks:
    mha = block.mha
    if isinstance(mha, CompressedAttention):  # compressed entries, indexer keys and the sliding window
      index_values = mha.d_index * (seqlen // mha.m) if mha.top_k is not None else 0
      total += mha.d_head * (seqlen // mha.m) + index_values + mha.d_head * min(mha.n_win, seqlen)
    else:  # one key and one value per head and token
      total += 2 * mha.n_heads * mha.d_head * seqlen
  return total


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, train_data, valid_data, _, _ = get_tiny_shakespeare_data()
  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gptlite_model_parameters()
  train_iters = 3000 # both models are trained from scratch with the same steps
  m, m_heavy = 4, max(8, seqlen // 16) # compression rates of CSA and HCA
  n_win, top_k = max(8, seqlen // 8), max(2, seqlen // 32) # sliding window, and compressed entries per query in CSA

  model_dense = GPTlite(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device)
  model_compressed = use_csa_hca(copy.deepcopy(model_dense), m, m_heavy, n_win, top_k)
  for name, model in (("Dense attention", model_dense), ("CSA and HCA", model_compressed)):
    train(model, train_data, train_iters, lr, batch_size, seqlen)
    print(f"{name}: validation loss {validation_loss(model, valid_data):.3f}, "
          f"KV cache of a {seqlen}-token sequence: {kv_cache_values(model, seqlen)} values")
