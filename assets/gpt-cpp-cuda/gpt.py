# The GPT blocks in PyTorch, as in the GPTlite post, and the benchmark of their forward pass in eager mode and
# with torch.compile
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# GPT-3 Small, Table 2.1 of "Language Models are Few-Shot Learners", Brown et al., 2020
L, E, H, S, B = 12, 768, 12, 2048, 8   # blocks, size of the embeddings, heads, sequence length, batch size
D = E // H                             # size of each attention head
warmup_iters, benchmark_iters = 5, 20


class FeedForward(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(E, 4*E), nn.ReLU(), nn.Linear(4*E, E))

  def forward(self, x):
    return self.net(x)


class MultiHeadAttention(nn.Module):

  def __init__(self):
    super().__init__()
    self.query = nn.Linear(E, E)
    self.key = nn.Linear(E, E)
    self.value = nn.Linear(E, E)
    self.proj = nn.Linear(E, E)
    self.register_buffer("tril", torch.tril(torch.ones(S, S)))

  def forward(self, x):
    B, S, _ = x.shape
    # the H heads, from (B, S, E) to (B, H, S, D)
    q = self.query(x).view(B, S, H, D).transpose(1, 2)
    k = self.key(x).view(B, S, H, D).transpose(1, 2)
    v = self.value(x).view(B, S, H, D).transpose(1, 2)
    wei = q @ k.transpose(-2, -1) * D**-0.5                         # (B, H, S, S)
    wei = wei.masked_fill(self.tril[:S, :S] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v                                                   # (B, H, S, D)
    return self.proj(out.transpose(1, 2).reshape(B, S, E))          # (B, S, E)


class Block(nn.Module):

  def __init__(self):
    super().__init__()
    self.ln1 = nn.LayerNorm(E)
    self.sa = MultiHeadAttention()
    self.ln2 = nn.LayerNorm(E)
    self.ffwd = FeedForward()

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


rng = np.random.RandomState(19)   # the generator of gpt.h, std::mt19937, with the same seed


def init_values(shape, scale):
  """ the next values of init_value(scale) of gpt.h: libstdc++ turns each 32-bit value r of the generator into the fp32
  value u = r / 2^32, and then into 2u - 1, in [-1, 1) """
  u = rng.randint(0, 2**32, size=shape, dtype=np.uint32).astype(np.float32) / np.float32(2**32)
  return np.float32(scale) * (u * 2 - 1)


@torch.no_grad()
def init_params(blocks):
  """ the parameters of gpt.h, where nn.Linear stores the transposed weights W^T (out x in) """
  for block in blocks:
    for linear in [block.sa.query, block.sa.key, block.sa.value, block.sa.proj, block.ffwd.net[0], block.ffwd.net[2]]:
      n_out, n_in = linear.weight.shape
      linear.weight.copy_(torch.from_numpy(init_values((n_in, n_out), 0.02)).t())
      linear.bias.copy_(torch.from_numpy(init_values(n_out, 0.02)))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, default=B, help="batch size")
  parser.add_argument("--compile", action="store_true", help="compile the model with torch.compile")
  args = parser.parse_args()

  torch.set_default_device("cuda")
  model = nn.Sequential(*[Block() for _ in range(L)])
  init_params(model)
  model = model.to(torch.bfloat16).eval()
  if args.compile:
    model = torch.compile(model)
  x = torch.from_numpy(init_values((args.batch, S, E), 1)).to("cuda", torch.bfloat16)

  with torch.inference_mode():
    for it in range(warmup_iters + benchmark_iters):
      if it == warmup_iters:   # the first iterations compile the model, with torch.compile
        torch.cuda.synchronize()
        start = time.perf_counter()
      y = model(x)
    torch.cuda.synchronize()
  ms = (time.perf_counter() - start) * 1000 / benchmark_iters

  y = y[0].float()   # the output of the first sequence, to compare implementations
  print(f"PyTorch{' + torch.compile' if args.compile else ''}, batch size {args.batch}: {ms:.1f} ms per forward pass")
  print(f"output: {y[0, 0]:.4f} {y[0, 1]:.4f} {y[0, 2]:.4f} ..., mean |y| {y.abs().mean():.4f}")


if __name__ == "__main__":
  main()
