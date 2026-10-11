import torch
import torch.nn.functional as F
import os


def get_batch(source, batch_size, seqlen):
  """ get a GPT-model batch (as token ids) of size block_size from source """
  
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


def get_tiny_shakespeare_data():
  """
  The tiny shakespeare dataset from
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  """ 

  # load tiny shakespeare
  current_dir = os.path.dirname(os.path.realpath(__file__))
  txt_path = os.path.join(current_dir, 'tinyshakespeare.txt')
  with open(txt_path) as f:
      text = f.read()

  #collect all ordered and unique characters in the text
  chars = sorted(list(set(text)))
  print(f"{len(chars)} unique chars: {''.join(chars)}")

  #map characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }
  encode_fn = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long) #encode text to integers
  decode_fn = lambda x: ''.join([itos[i] for i in x]) #decode integers to text
  vocab_size = len(stoi)
  print(encode_fn("Hello world"))
  print(decode_fn(encode_fn("Hello world").tolist()))
  print("character zero is:", decode_fn([0]), "<end>")

  # collect input data, break dataset in train/validation
  data = encode_fn(text)
  data_split = int(0.9*len(data))
  train_data, valid_data = data[:data_split], data[data_split:]
  print(f"Data sizes: total {data.shape[0]}, train {train_data.shape[0]}, valid {valid_data.shape[0]}")

  return vocab_size, train_data, valid_data, encode_fn, decode_fn


def get_gpt3_small_model_parameters():
  """GPT3-small model parameters (Table 2.1 in "Language Models are Few-Shot Learners, Brown et al, 2021") """

  n_layers = 12 # depth of the network as number of decoder blocks.
  d_model = 768 # size of the embeddings
  n_heads = 12 # number of attention heads in the Multi-Attention mechanism
  d_head = 64 # dimensionality of the attn head
  batch_size = 500_000 # batch size (0.5M)
  lr = 6e-4 # learning rate
  seqlen = 2048  # sequence length, context size, or $n_{ctx}$ in the paper    
  dropout_p = 0.1 # dropout rate for dropout units
  return n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p

def get_gptlite_model_parameters():
  """ Parameters of a super small GPT3 model, for debugging and that can be trained on a laptop """

  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gpt3_small_model_parameters()
  d_model = 512
  n_layers //= 2
  n_heads //= 2
  batch_size = 32
  seqlen = 128
  return n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p


def get_gptlite_distilled_model_parameters():
  """
  Parameters of a smaller model than get_gptlite_model_parameters used as e.g.
  student model in distillation or draft model in speculative sampling.
  """
  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gptlite_model_parameters()
  d_model=256
  n_heads=4
  seqlen=64
  return n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p