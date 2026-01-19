import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

#use base GPTlite model from the GPT-lite post
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
import gptlite
from gptlite import block_size, n_embd, dropout

# implements tensor parallelism as described in Megatron-LM: https://arxiv.org/abs/1909.08053
# see https://pytorch.org/docs/stable/notes/extending.html for "how to extend pytorch fwd/backward passes"

if not dist.is_initialized():
  dist.init_process_group()



class Megatron_f(torch.autograd.Function):
  """ The f function in Figure 3 in Megatron paper """

  @staticmethod
  def forward(ctx, x, mp_comm_group=None):
      ctx.mp_comm_group = mp_comm_group
      return x

  @staticmethod
  def backward(ctx, gradient):
      dist.all_reduce(gradient, dist.ReduceOp.SUM, group=ctx.mp_comm_group)
      return gradient, None



class Megatron_g(torch.autograd.Function):
  """ The g function in Figure 3 in Megatron paper """

  @staticmethod
  def forward(ctx, x, mp_comm_group=None):
      dist.all_reduce(x, dist.ReduceOp.SUM, group=mp_comm_group)
      return x

  @staticmethod
  def backward(ctx, gradient):
      return gradient, None



class Megatron_FeedForward(nn.Module):
  """ the feed forward network (FFN) in the paper, with tensor parallelism as in Megatron-LM MLP block"""

  def __init__(self, n_embd, mp_comm_group=None):
    super().__init__()
    self.mp_comm_group = mp_comm_group

    #Fig 3a. MLP: splits first GEMM across colums and second GEMM across rows
    n_embd_mid = n_embd * 4
    if self.mp_comm_group:
        n_embd_mid //= dist.get_world_size()

    self.fc1 = nn.Linear(n_embd, n_embd_mid)
    self.fc2 = nn.Linear(n_embd_mid, n_embd, bias=False)   # <-- no bias here
    self.fc2_bias = nn.Parameter(torch.zeros(n_embd))      # <-- bias added after all-reduce
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    if self.mp_comm_group:
        x = Megatron_f.apply(x, self.mp_comm_group)

    y = F.relu(self.fc1(x))

    z = self.fc2(y)  # matmul only (partial)

    if self.mp_comm_group:
        z = Megatron_g.apply(z, self.mp_comm_group)

    z = z + self.fc2_bias  # <-- bias AFTER all-reduce
    z = self.dropout(z)
    return z


class Megatron_Head(nn.Module):
  """ the attention block with tensor parallelism as in Megatron-LM paper"""

  def __init__(self, head_size, mp_comm_group=None):
    super().__init__()
    self.mp_comm_group = mp_comm_group
    if mp_comm_group:
        #Fig 3b. Self-attention: splits first GEMM across colums and second GEMM across rows
        head_size //= dist.get_world_size()

    self.key   = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape

    if self.mp_comm_group:
      x = Megatron_f.apply(x, self.mp_comm_group)

    k = self.key(x)
    q = self.query(x)
    v = self.value(x)

    # scores
    wei = q @ k.transpose(-2, -1)  # (B,T,T)

    if self.mp_comm_group:
      wei = Megatron_g.apply(wei, self.mp_comm_group)  # <-- reduce BEFORE softmax

    wei *= (q.size(-1) ** -0.5)  # <-- scale by head dim
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    #perform weighted aggregation of values
    out = wei @ v
    return out


def get_megatron_tensor_parallel_model(vocab_size, mp_comm_group=None):

  if mp_comm_group is None:
    mp_comm_group = dist.new_group(range(dist.get_world_size()))

  #replace definition of base model's MLP and Attention head with megatron model parallel version
  gptlite.Head = lambda *args: Megatron_Head(*args, mp_comm_group=mp_comm_group)
  gptlite.FeedForward = lambda *args: Megatron_FeedForward(*args, mp_comm_group=mp_comm_group)
  return gptlite.GPTlite(vocab_size)

