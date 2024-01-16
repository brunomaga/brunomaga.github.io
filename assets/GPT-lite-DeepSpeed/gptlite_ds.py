import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec


#use base GPTlite model from the GPT-lite post
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
from gptlite import n_layer, n_embd, n_head, block_size, Block

################ BASE MODEL WITH ACTIVATION CHECKPOINTING ######################

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
    return logits

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
      return [
          lambda idx:
            self.token_embedding_table(idx) +
            self.position_embedding_table(torch.arange(idx.shape[1]).to(idx.device)),
          *self.blocks,
          self.ln,
          self.lm_head,
      ]


################ PIPELINE VERSION ######################

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


  def __init__(self, vocab_size, pipe_kwargs):
    self.specs = \
      [ LayerSpec(GPTlitePipeSpec.EmbeddingsSum, vocab_size) ] + \
      [ LayerSpec(Block, n_embd, n_head) for _ in range(n_layer)] + \
      [ LayerSpec(nn.LayerNorm, n_embd),
        LayerSpec(nn.Linear, n_embd, vocab_size, bias=False) ]
    super().__init__(layers=self.specs, **pipe_kwargs)

################ HELPERS ######################

def load_tiny_shakespeare_data():
  rank = dist.get_rank() if dist.is_initialized() else 0 

  #load input data
  current_dir = os.path.dirname(os.path.realpath(__file__))
  txt_path = os.path.join(current_dir, '..', 'GPT-lite', 'tinyshakespeare.txt')
  with open(txt_path) as f:
    text = f.read()
    
  if rank==0: print("input data loaded. Length of text: ", len(text))

  #collect all ordered and unique characters in the text
  chars = sorted(list(set(text)))
  if rank==0: print("unique chars: ", "".join(chars))
  if rank==0: print("length of chars: ", len(chars))

  #map characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }
  encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long) #encode text to integers
  decode = lambda x: ''.join([itos[i] for i in x]) #decode integers to text
  vocab_size = len(stoi)
  if rank==0: print("vocab size: ", vocab_size)
  if rank==0: print(encode("Hello world"))
  if rank==0: print(decode(encode("Hello world").tolist()))
  if rank==0: print("character zero is:", decode([0]), "<end>")

  # collect input data, break dataset in train/validation
  data = encode(text)
  n = int(0.9*len(data))
  train_data, valid_data = data[:n], data[n:]
  if rank==0: print("Train data encoded", data.shape, train_data.shape, valid_data.shape)
  return train_data, valid_data, vocab_size


def get_dataset():
  
  class GPTliteDataset(torch.utils.data.Dataset):

      def __init__(self, train_data, block_size):
        self.train_data = train_data
        self.block_size = block_size

      def __len__(self):
        return len(self.train_data)

      def __getitem__(self, idx):
        # generate 1 random offset on the data
        ix = torch.randint(len(self.train_data)-self.block_size , size=())
        # input is a random subset of tokens
        x = self.train_data[ix   : ix+self.block_size]
        # target is just x shifted right (ie the next predicted word)
        y = self.train_data[ix+1 : ix+1+self.block_size]
        return x, y

  train_data, valid_data, vocab_size = load_tiny_shakespeare_data() 
  train_dataset = GPTliteDataset(train_data, block_size)
  valid_dataset = GPTliteDataset(valid_data, block_size)
  return train_dataset, valid_dataset, vocab_size


def get_model(vocab_size, criterion=None, pipeline_num_stages=0, pipeline_spec_layers=False, activation_checkpoint_interval=0):

  if pipeline_num_stages:
    assert criterion is not None, "for pipeline runs, need to specify criterion"
    pipe_kwargs={
      'num_stages': pipeline_num_stages,
      'activation_checkpoint_interval': activation_checkpoint_interval, 
      'loss_fn': criterion,
    }

    if pipeline_spec_layers:
      model = GPTlitePipeSpec(vocab_size, pipe_kwargs=pipe_kwargs)
    else:
      device_str = f'cuda:{dist.get_rank()}'
      model = GPTlite(vocab_size).to(device_str)
      model = deepspeed.pipe.PipelineModule(layers=model.to_layers(), **pipe_kwargs)
  else:
    model = GPTlite(vocab_size,
              activation_checkpoint_interval=activation_checkpoint_interval)  
  return model


