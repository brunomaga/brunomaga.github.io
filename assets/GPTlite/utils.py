import torch
import torch.nn.functional as F


def get_batch(source, batch_size, seqlen):
  """ get batch of size block_size from source """
  
  # generate `batch_size` random offsets on the data 
  offsets = torch.randint(len(source)-seqlen-1, (batch_size,) )

  # collect `batch_size` subsequences of length `seqlen+1`
  tok_ids = torch.stack([source[i:i+seqlen+1] for i in offsets])

  # target is just x shifted right (ie the next predicted word)
  return tok_ids[:, :-1], tok_ids[:, 1:]


def sin_cos_positional_embeddings(S, E):
  """Generate positional encoding for a given sequence length and embedding size."""

  pos = torch.arange(S) 
  div_term = 1000 ** (torch.arange(E)/E) 
  even_i = torch.arange(0, E, 2)
  odd_i = torch.arange(1, E, 2)

  pe = torch.zeros((S, E))
  # pos unsequeezes to Bx1 to broadcast to SxE, and div_term unsqueezes to 1xE to broadcast to BxE
  pe[:, even_i] = torch.sin( pos.unsqueeze(1) / div_term[even_i].unsqueeze(0) )  # Sine for even indices
  pe[:, odd_i]  = torch.cos( pos.unsqueeze(1) / div_term[odd_i].unsqueeze(0) )  # Cosine for odd indices
  return pe.unsqueeze(0) # from SxE to 1xSxE to sum over batch dimension

