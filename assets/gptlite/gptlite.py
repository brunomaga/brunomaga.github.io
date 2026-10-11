import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
  """ the feed forward network (FFN) in the paper"""

  def __init__(self, n_embd, dropout_p):
    super().__init__()
    # Note: in the paper (section 3.3) we have d_{model}=512 and d_{ff}=2048.
    # Therefore the inner layer is 4 times the size of the embedding layer
    self.net = nn.Sequential(
        nn.Linear(n_embd, n_embd*4),
        nn.ReLU(),
        nn.Linear(n_embd*4, n_embd),
        nn.Dropout(dropout_p)
      )

  def forward(self, x):
    return self.net(x)


def scaled_dot_product_attention(Q, K, V, causal_mask=True, dropout=None):
    """ Compute the attention weights and output for all heads """

    # Attention scores: Bx[H]xSxD @ Bx[H]xDxS -> Bx[H]xSxS
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)

    if causal_mask:
      seqlen = scores.size(-1)
      mask = torch.tril(torch.ones(seqlen, seqlen)) # SxS
      mask = mask.unsqueeze(0).unsqueeze(0) # 1x1xSxS
      mask = mask.to(scores.device)
      scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights = F.softmax(scores, dim=-1) # weights or probabilities: BxHxSxS -> BxHxSxS

    if dropout is not None:
        #randomly prevents some tokens from communicating with each other
        weights = dropout(weights) # BxHxSxS

    output = weights @ V # Weighted values: BxHxSxS @ BxHxSxD -> BxHxSxD
    return output


class MultiHeadAttention(nn.Module):
    """ Multi Head Attention. computes all heads in parallel """

    def __init__(self, d_model, n_heads, d_head, dropout_p):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        # Create individual heads
        self.query_proj = nn.Linear(d_model, n_heads*d_head) 
        self.key_proj = nn.Linear(d_model, n_heads*d_head)
        self.value_proj = nn.Linear(d_model, n_heads*d_head)
        self.dropout = nn.Dropout(dropout_p)

        # Output projection
        self.out_proj = nn.Linear(n_heads*d_head, d_model)

    def forward(self, x, causal_mask=True):

        (B, S, _), H, D = x.shape, self.n_heads, self.d_head

        q = self.query_proj(x)  # BxSxE -> BxSx(D*H)
        k = self.key_proj(x) 
        v = self.value_proj(x)

        # from BxSx(H*D) to BxSxHxD to BxHxSxD
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)

        # attention heads
        out = scaled_dot_product_attention(q, k, v, causal_mask=causal_mask, dropout=self.dropout) # BxHxSxD

        # Concatenate all heads' outputs
        out = out.transpose(1,2).reshape(B, S, H*D)  #  BxHxSxD ->  BxSxHxD ->  BxSx(H*D)

        # Project concatenated outputs back to embedding dimension
        out = self.out_proj(out)  # BxSx(H*D) -> BxSxE

        # Apply dropout
        out = self.dropout(out)

        return out
    

class Block(nn.Module):
    """ a GPT block, with multi-head attention and feed forward network """

    def __init__(self, d_model, n_heads, d_head, dropout_p):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, d_head, dropout_p)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, dropout_p)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # pre-layer norm
        x = x + self.mha(self.ln1(x), causal_mask=True)
        x = x + self.ffwd(self.ln2(x))
        return x


# Define a simple GPT-like transformer model from scratch
class GPTlite(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen):
        super(GPTlite, self).__init__()
        # vocabulary embedding and positional embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seqlen, d_model)
        
        # all the transformer blocks
        self.blocks = nn.Sequential(*[Block(d_model, n_heads, d_head, dropout_p) for _ in range(n_layers)])
        
        # final layer norm and linear layer to project to vocab size
        self.ln = nn.LayerNorm(d_model)  # Add this
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # SUM Embedding and positional encoding
        seqlen = x.size(1)
        positions = torch.arange(seqlen, device=x.device).unsqueeze(0) # [1, T]
        pos_embeddings = self.position_embedding(positions) # B, T, E
        x = self.token_embedding(x) +  pos_embeddings # [B, T, embed_dim]
        
        # Pass through transformer layers
        x = self.blocks(x) # [B, T, embed_dim]
        
        # Do layer norm and then project to vocab size
        x = self.ln(x)
        x = self.fc_out(x)
        return x # return logits
    