import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# reuse some modules from the original GPTlite model
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPTlite'))
from gptlite import FeedForward


def scaled_dot_product_attention_kv_cache(Q, K, V, causal_mask=True, dropout=None):
    """
    Attention of Sq new tokens over Sk keys and values, where the last Sq keys and values belong
    to the new tokens and the first Sk-Sq ones are cached. Query i can attend to all cached keys
    and to the new keys up to itself, so the causal mask is shifted by Sk-Sq. This covers the
    whole sequence without a cache (training or prefill, Sq == Sk), one new token with a cache
    (decode, Sq == 1, no mask needed) and several new tokens with a cache (e.g. a prompt suffix
    after a cached prefix).
    """

    # Q: [B, H, Sq, D], K: [B, H, Sk, D], V: [B, H, Sk, D]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # [B, H, Sq, Sk]

    Sq, Sk = Q.size(2), K.size(2)
    if causal_mask and Sq > 1:
        mask = torch.ones(Sq, Sk, device=Q.device).tril(diagonal=Sk - Sq)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)  # [B, H, Sq, Sk]
    if dropout is not None:
        weights = dropout(weights)

    output = weights @ V  # [B, H, Sq, D]
    return output



class MultiHeadAttention_KVCache(nn.Module):
    """
    Multi-head attention with an optional KV cache, given as a tuple (past_keys, past_values).

    Without a cache (training or prefill), it processes the full sequence as usual. With a cache,
    it appends the keys and values of the new tokens to the cached ones, keeps at most the last
    max_seqlen of them, and returns the updated cache.
    """

    def __init__(self, d_model, n_heads, d_head, dropout_p):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        # query, key and value projections of all heads, and output projection
        self.query_proj = nn.Linear(d_model, n_heads * d_head)
        self.key_proj = nn.Linear(d_model, n_heads * d_head)
        self.value_proj = nn.Linear(d_model, n_heads * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, kv_cache=None, causal_mask=True, max_seqlen=None):
        B, S, _ = x.shape
        H, D = self.n_heads, self.d_head

        # Compute Q, K, V of the new tokens
        q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
        k = self.key_proj(x).view(B, S, H, D).transpose(1, 2)
        v = self.value_proj(x).view(B, S, H, D).transpose(1, 2)

        # If a cache is provided, prepend the past keys and values
        if kv_cache is not None:
            past_keys, past_values = kv_cache
            k = torch.cat([past_keys, k], dim=2)  # [B, H, S_past + S, D]
            v = torch.cat([past_values, v], dim=2)
            # Keep at most the last max_seqlen keys and values
            if max_seqlen is not None and k.size(2) > max_seqlen:
                k = k[:, :, -max_seqlen:, :]
                v = v[:, :, -max_seqlen:, :]

        # Compute attention
        out = scaled_dot_product_attention_kv_cache(q, k, v, causal_mask=causal_mask,
                                          dropout=self.dropout if self.training else None)

        # Project output
        out = out.transpose(1, 2).reshape(B, S, H * D)  # [B, S, H*D]
        out = self.out_proj(out)
        out = self.dropout(out) if self.training else out

        # Return output and updated cache
        new_cache = (k, v)
        return out, new_cache
    

class Block_KVCache(nn.Module):
    """
    The block passes the cache to MultiHeadAttention and returns the updated cache alongside the output.
    """

    def __init__(self, d_model, n_heads, d_head, dropout_p):
        super().__init__()
        self.mha = MultiHeadAttention_KVCache(d_model, n_heads, d_head, dropout_p)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, dropout_p)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, kv_cache=None, max_seqlen=None):
        # Pre-layer normalization
        mha_out, new_cache = self.mha(self.ln1(x), kv_cache=kv_cache, 
                                     causal_mask=True, max_seqlen=max_seqlen)
        x = x + mha_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_cache
    

class GPTlite_KVCache(nn.Module):
    """
    GPTlite that handles cache for all layers and adjust positional
    embeddings for inference with caching.

    The model maintains a list of caches (one per layer). During inference, it uses the cache
    to determine the positions of the new tokens (S_past, S_past+1, ... if within seqlen, or
    seqlen-1 if beyond). Positional embeddings are capped at seqlen-1 to match training.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seqlen, d_model)
        self.blocks = nn.ModuleList([
            Block_KVCache(d_model, n_heads, d_head, dropout_p) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.seqlen = seqlen

    def forward(self, x, kv_cache=None, max_seqlen=None):
        B, T = x.shape
        
        # Determine positions
        if kv_cache is not None and kv_cache[0] is not None:
            # During inference, the new tokens follow the S_past cached ones
            S_past = kv_cache[0][0].size(2)  # S_past from first layer's past_keys
            positions = torch.arange(S_past, S_past + T, device=x.device).clamp(max=self.seqlen - 1)
            positions = positions.unsqueeze(0).expand(B, T)
        else:
            # During training or first inference step
            positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        
        # Embeddings
        pos_embeddings = self.position_embedding(positions)  # [B, T, E]
        x = self.token_embedding(x) + pos_embeddings  # [B, T, E]

        # Initialize cache if None
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        # Process through blocks
        new_caches = []
        for block, layer_cache in zip(self.blocks, kv_cache):
            x, new_cache = block(x, kv_cache=layer_cache, max_seqlen=max_seqlen)
            new_caches.append(new_cache)

        # Final layers
        x = self.ln(x)
        x = self.fc_out(x)
        return x, new_caches
