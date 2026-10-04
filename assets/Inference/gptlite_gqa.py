import torch
import torch.nn as nn
from gptlite_kvcache import GPTlite_KVCache, scaled_dot_product_attention_kv_cache


class MultiHeadAttention_GQA(nn.Module):
    """ Multi Head Attention with Grouped Query Attention (GQA) and an optional KV cache.
        Every head has its own query, but the heads are split in n_groups groups, and the heads of a
        group share the same key and value. GQA becomes Multi-Head Attention (MHA) when the number of
        groups equals the number of heads, and Multi-Query Attention (MQA) when there is 1 group.
        The cache stores the keys and values of the n_groups groups only.
    """

    def __init__(self, d_model, n_heads, d_head, dropout_p, n_groups):
        super().__init__()
        assert n_heads % n_groups == 0, "the number of heads must be a multiple of the number of groups"
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_groups = n_groups

        # one query per head, but one key and one value per group
        self.query_proj = nn.Linear(d_model, n_heads * d_head)
        self.key_proj = nn.Linear(d_model, n_groups * d_head)
        self.value_proj = nn.Linear(d_model, n_groups * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, kv_cache=None, causal_mask=True, max_seqlen=None):
        (B, S, _), H, G, D = x.shape, self.n_heads, self.n_groups, self.d_head

        q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, n_heads, S, d_head]
        k = self.key_proj(x).view(B, S, G, D).transpose(1, 2)    # [B, n_groups, S, d_head]
        v = self.value_proj(x).view(B, S, G, D).transpose(1, 2)

        # If a cache is provided, prepend the past keys and values (of the groups only)
        if kv_cache is not None:
            past_keys, past_values = kv_cache
            k = torch.cat([past_keys, k], dim=2)  # [B, n_groups, S_past + S, d_head]
            v = torch.cat([past_values, v], dim=2)
            if max_seqlen is not None and k.size(2) > max_seqlen:
                k = k[:, :, -max_seqlen:, :]
                v = v[:, :, -max_seqlen:, :]

        # Each group of n_heads/n_groups consecutive heads uses the key and value of its group,
        # e.g. with 12 heads and 4 groups, heads 0-2 use group 0, heads 3-5 use group 1, etc.
        k_heads = k.repeat_interleave(H // G, dim=1)  # [B, n_heads, S_past + S, d_head]
        v_heads = v.repeat_interleave(H // G, dim=1)
        out = scaled_dot_product_attention_kv_cache(q, k_heads, v_heads, causal_mask=causal_mask,
                                                    dropout=self.dropout if self.training else None)

        out = out.transpose(1, 2).reshape(B, S, H * D)  # [B, S, n_heads*d_head]
        out = self.dropout(self.out_proj(out))
        return out, (k, v)


class GPTlite_GQA(GPTlite_KVCache):
    """ GPTlite with a KV cache, whose blocks use Grouped Query Attention """

    def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, n_groups):
        super().__init__(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen)
        for block in self.blocks:
            block.mha = MultiHeadAttention_GQA(d_model, n_heads, d_head, dropout_p, n_groups)


def convert_mha_to_gqa(state_dict, n_heads, d_head, n_groups):
    """ Converts the weights of a multi-head GPTlite into GQA: the key and value projections of the
        heads of each group are replaced by their average (mean pooling) """
    gqa_state_dict = dict(state_dict)
    for name, param in state_dict.items():
        if '.mha.key_proj.' in name or '.mha.value_proj.' in name:
            heads = param.view(n_groups, n_heads // n_groups, d_head, -1)  # [n_groups, heads per group, d_head, d_model or 1]
            gqa_state_dict[name] = heads.mean(dim=1).reshape(n_groups * d_head, *param.shape[1:])
    return gqa_state_dict

