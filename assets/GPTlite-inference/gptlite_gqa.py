import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# reuse some modules from the original GPTlite model
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPTlite'))
from gptlite import FeedForward, scaled_dot_product_attention

class MultiHeadAttention_GQA(nn.Module):
    """ Multi Head Attention with Grouped Query Attention (GQA)
        GQA: Uses a dedicated projection for queries (per head) but groups keys and values.
        Grouped Query Attention (GQA) becomes Multi-Head Attention (MHA) when number of groups equals
        the number of heads, and the same as Multi-Query Attention (MQA) when number of groups is 1.
    """

    def __init__(self, d_model, n_heads, d_head, dropout_p, n_groups):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_groups = n_groups
        self.group_size = n_heads // n_groups

        # Query projection remains per head:
        self.query_proj = nn.Linear(d_model, n_heads * d_head)
        self.key_proj = nn.Linear(d_model, n_groups * d_head)
        self.value_proj = nn.Linear(d_model, n_groups * d_head)
        self.dropout = nn.Dropout(dropout_p)

        # Output projection remains the same:
        self.out_proj = nn.Linear(n_heads * d_head, d_model)

    def forward(self, x, causal_mask=True):
        (B, S, _), H, D = x.shape, self.n_heads, self.d_head

        # Compute query; shape: [B, S, n_heads, d_head]
        q = self.query_proj(x)       # [B, S, n_heads*d_head]
        # Compute keys and values; shape: [B, S, n_groups, d_head]
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape query to [B, S, n_heads, d_head] then transpose to [B, n_heads, S, d_head]
        q = q.view(B, S, H, D).transpose(1, 2)  # [B, n_heads, S, d_head]
        # Reshape keys and values to [B, S, n_groups, d_head] then transpose to [B, n_groups, S, d_head]
        k = k.view(B, S, self.n_groups, D).transpose(1, 2)  # [B, n_groups, S, d_head]
        v = v.view(B, S, self.n_groups, D).transpose(1, 2)  # [B, n_groups, S, d_head]

        # GQA change: For each query head, assign a key/value group.
        # Create group indices for each head: e.g., if n_heads=12 and n_groups=4, group index = [0,0,0,1,1,1,2,2,2,3,3,3]
        group_indices = torch.arange(H, device=x.device) // self.group_size  # shape: [n_heads]
        # Reshape and expand group_indices to gather from keys/values: shape becomes [B, n_heads, S, D]
        group_indices = group_indices.view(1, H, 1, 1).expand(B, H, S, D)
        # Gather keys and values from the grouped projections
        k = k.gather(dim=1, index=group_indices)  # GQA change: shape [B, n_heads, S, d_head]
        v = v.gather(dim=1, index=group_indices)  # GQA change: shape [B, n_heads, S, d_head]

        # Compute attention using scaled dot product attention
        out = scaled_dot_product_attention(q, k, v, causal_mask=causal_mask, dropout=self.dropout)  # [B, n_heads, S, d_head]

        # Concatenate all heads' outputs: transpose then reshape to [B, S, n_heads*d_head]
        out = out.transpose(1, 2).reshape(B, S, H * D)
        # Project concatenated outputs back to embedding dimension
        out = self.out_proj(out)
        # Apply dropout
        out = self.dropout(out)

        return out
    

class Block_GQA(nn.Module):
    """ a GPT block, with multi-head (now GQA) attention and feed forward network """

    def __init__(self, d_model, n_heads, d_head, dropout_p, n_groups):
        super().__init__()
        self.mha = MultiHeadAttention_GQA(d_model, n_heads, d_head, dropout_p, n_groups)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, dropout_p)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # pre-layer norm
        x = x + self.mha(self.ln1(x), causal_mask=True)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTlite_GQA(nn.Module):

    def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, n_groups):
        super(GPTlite_GQA, self).__init__()
        # vocabulary embedding and positional embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seqlen, d_model)
        
        # all the transformer blocks (each using GQA)
        self.blocks = nn.Sequential(*[
            Block_GQA(d_model, n_heads, d_head, dropout_p, n_groups)
            for _ in range(n_layers)
        ])
        
        # final layer norm and linear layer to project to vocab size
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # SUM Embedding and positional encoding
        seqlen = x.size(1)
        positions = torch.arange(seqlen, device=x.device).unsqueeze(0)  # [1, T]
        pos_embeddings = self.position_embedding(positions)  # B, T, E
        x = self.token_embedding(x) + pos_embeddings  # [B, T, embed_dim]
        
        # Pass through transformer layers
        x = self.blocks(x)  # [B, T, embed_dim]
        
        # Do layer norm and then project to vocab size
        x = self.ln(x)
        x = self.fc_out(x)
        return x  # return logits