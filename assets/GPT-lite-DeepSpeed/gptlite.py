import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

# replicate GPT-3 Small in Table 2.1 in "Language Models are Few-Shot Learners, Brown et al, 2021"

# depth of the network as number of decoder blocks.
n_layer = 12

# size of the embeddings (d_model)
n_embd = 768

# number of attention heads in the Multi-Attention mechanism
n_head = 12

# number of heads. this is the $d_k$ in the paper formulation of Attn. Mech
head_size = 64

# block size ie max number of training sequence, the $n_{ctx}$ in the paper .
block_size = 2048

# dropout rate (variable p) for dropout units
dropout = 0.1


class Head(nn.Module):

  def __init__(self, head_size=16):
    super().__init__()
    self.key   = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout) #randomly prevents some tokens from communicating with each other

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) #shape (B,T, head_size)
    q = self.query(x) #shape (B,T, head_size)
    v = self.value(x) #shape (B,T, head_size)

    #compute self-attention scores
    wei = q @ k.transpose(-2, -1) #shape (B,T, head_size) @ (B,head_size,T) --> (B,T,T)
    wei *= head_size**-0.5 #scale by sqrt(d_k) as per paper, so that variance of the wei is 1
    wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B,T,T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)

    #perform weighted aggregation of values
    v = self.value(x) #shape (B,T, head_size)
    out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
    return out


class MultiHeadAttention(nn.Module):
  """ Multi-head attention as described in the paper. Simply a collection of heads with concatenated outputs."""
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj  = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out

class FeedForward(nn.Module):
  """ the feed forward network (FFN) in the paper"""
  def __init__(self, n_embd):
    super().__init__()
    # Note: in the paper (section 3.3) we have d_{model}=512 and d_{ff}=2048.
    # Therefore the inner layer is 4 times the size of the embedding layer
    self.net = nn.Sequential(
        nn.Linear(n_embd, n_embd*4),
        nn.ReLU(),
        nn.Linear(n_embd*4, n_embd),
        nn.Dropout(dropout)
      )
  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  """ Transformer block: comunication (attention) followed by computation (FFN) """
  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension
    # n_heads : the number of heads we'd like to use
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x



class GPTlite(nn.Module):
  def __init__(self, vocab_size, activation_checkpoint_interval=0):
    super().__init__()
    
    self.activation_checkpoint_interval = activation_checkpoint_interval

    # vocabulary embedding and positional embedding
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)

    #sequence of attention heads and feed forward layers
    self.blocks = nn.Sequential( *[Block(n_embd, n_head) for _ in range(n_layer)])

    #one layer normalization layer after transformer blocs and before linear layer that oututs the vocabulary
    self.ln = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
  
  def forward(self, idx, targets=None):
    """ call the model with idx and targets (training) or without targets (generation)"""
    #idx and targets are both of shape (B,T)

    # use case with breakpoint activation
    if self.activation_checkpoint_interval > 0:
      x=idx
      for l, layer in enumerate(self.to_layers()):
        is_checkpoint = l % self.activation_checkpoint_interval == 0 
        x = deepspeed.checkpointing.checkpoint(layer, x) if is_checkpoint else layer(x)
      return x
      
    # regular use case
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #shape (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T).to(idx.device)) #shape (T,C)
    x = tok_emb + pos_emb #shape (B,T,C)
    x = self.blocks(x)
    x = self.ln(x)
    logits = self.lm_head(x) #shape (B,T,C)
    return torch.swapaxes(logits,1,2) #shape (B,C,T)

  def generate(self, idx, max_new_tokens):
    """ given a context idx, generate max_new_tokens tokens and append them to idx """
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:] #we can never have any idx longer than block_size
      logits, _ = self(idx_cond) #call fwd without targets
      logits = logits[:, -1, :] # shape (B, C)
      #convert logits to probabilities
      probs = F.softmax(logits, dim=-1) # shape (B, C)
      #randomly sample the next tokens, 1 for each of the previous probability distributions
      #(one could take instead the argmax, but that would be deterministic and boring)
      idx_next = torch.multinomial(probs, num_samples=1) # shape (B, 1)
      #append next token ix to the solution sequence so far
      idx = torch.cat([idx, idx_next], dim=-1) # shape (B, T+1)
    return idx


  def to_layers(self):  
      layers = [
          lambda idx:
            self.token_embedding_table(idx) +
            self.position_embedding_table(torch.arange(idx.shape[1]).to(idx.device)),
          *self.blocks,
          self.ln,
          self.lm_head,
          lambda logits: torch.swapaxes(logits,1,2)
      ]
      return layers


from deepspeed.pipe import PipelineModule, LayerSpec
class GPTlitePipeSpec(PipelineModule):

  class EmbeddingsSum(nn.Module):
    """ converts tok_emb + pos_emb into an nn.Module. Required for LayerSpec"""

    def __init__(self, vocab_size):
      super().__init__()
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
      B, T = idx.shape
      tok_emb = self.token_embedding_table(idx)
      pos_emb = self.position_embedding_table(torch.arange(T).to(idx.device))
      return tok_emb + pos_emb


  class SwapAxes(nn.Module):
    """ converts  torch.swapaxes(logits,1,2) into an nn.Module. Required for LayerSpec"""

    def __init__(self):
      super().__init__()

    def forward(self, logits):
      return torch.swapaxes(logits,1,2)


  def __init__(self, vocab_size, pipe_kwargs):
    self.specs = \
      [ LayerSpec(GPTlitePipeSpec.EmbeddingsSum, vocab_size) ] + \
      [ LayerSpec(Block, n_embd, n_head) for _ in range(n_layer)] + \
      [ LayerSpec(nn.LayerNorm, n_embd),
        LayerSpec(nn.Linear, n_embd, vocab_size, bias=False) ] + \
      [ LayerSpec(GPTlitePipeSpec.SwapAxes) ]
    super().__init__(layers=self.specs, **pipe_kwargs)

