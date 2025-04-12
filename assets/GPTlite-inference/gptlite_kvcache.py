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
    During inference with caching, Q has length 1 (the new token), and K and V include all past tokens
    plus the new one. Since we’re computing attention for the latest position, it can attend to all
    keys/values up to itself, so no mask is applied.
    """

    # Q: [B, H, Sq, D], K: [B, H, Sk, D], V: [B, H, Sk, D]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # [B, H, Sq, Sk]
    
    # IMPORTANT: Apply mask only during training when Sq == Sk
    # During inference (Sq=1, Sk=S_past+1), no mask is needed as the single query attends to all keys
    if causal_mask and Q.size(2) == K.size(2):
        seqlen = Q.size(2)
        mask = torch.tril(torch.ones(seqlen, seqlen, device=Q.device))
        mask = mask.to(scores.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights = F.softmax(scores, dim=-1)  # [B, H, Sq, Sk]
    if dropout is not None:
        weights = dropout(weights)
    
    output = weights @ V  # [B, H, Sq, D]
    return output



class MultiHeadAttention_KVCache(nn.Module):
    """
    MHA with support for a KV cache in the forward method. The cache will be a tuple
    (past_keys, past_values) per layer, and we’ll trim it to a maximum sequence
    length (max_seqlen) during inference.

    During training (kv_cache=None), it processes the full sequence as usual. During inference
    (cache provided), it appends the new token’s keys and values to the cached ones, trims
    to max_seqlen if necessary, and returns the updated cache.
    """

    def __init__(self, d_model, n_heads, d_head, dropout_p):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.query_proj = nn.Linear(d_model, n_heads * d_head)
        self.key_proj = nn.Linear(d_model, n_heads * d_head)
        self.value_proj = nn.Linear(d_model, n_heads * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, kv_cache=None, causal_mask=True, max_seqlen=None):
        B, S, _ = x.shape
        H, D = self.n_heads, self.d_head

        # Compute Q, K, V
        q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
        k = self.key_proj(x).view(B, S, H, D).transpose(1, 2)
        v = self.value_proj(x).view(B, S, H, D).transpose(1, 2)

        # If cache is provided (inference), concatenate past keys/values
        if kv_cache is not None:
            past_keys, past_values = kv_cache
            k = torch.cat([past_keys, k], dim=2)  # [B, H, S_past + S, D]
            v = torch.cat([past_values, v], dim=2)
            # Trim cache to max_seqlen if exceeded
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
        self.ffwd = FeedForward(d_model, dropout_p)  # Assume FeedForward is defined elsewhere
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
    to determine the position of the new token (e.g., S_past if within seqlen, or seqlen-1 if
    beyond). Positional embeddings are capped at seqlen-1 to match training.
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
            # During inference, use position based on past sequence length
            S_past = kv_cache[0][0].size(2)  # S_past from first layer's past_keys
            position = min(S_past, self.seqlen - 1)
            positions = torch.tensor([position], device=x.device).unsqueeze(0).expand(B, T)
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