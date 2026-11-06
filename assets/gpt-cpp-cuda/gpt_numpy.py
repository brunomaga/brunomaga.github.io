# The GPT blocks in NumPy, on the CPU, with fp32 values, and the benchmark of their forward pass
import argparse
import time

import numpy as np

# GPT-3 Small, Table 2.1 of "Language Models are Few-Shot Learners", Brown et al., 2020
L, E, H, S, B = 12, 768, 12, 2048, 8   # blocks, size of the embeddings, heads, sequence length, batch size
D = E // H                             # size of each attention head
eps = 1e-5                             # epsilon of the layer normalization, as in PyTorch
warmup_iters, benchmark_iters = 5, 20


def ffn(x, p):
  return np.maximum(x @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"]


def attention(x, p):
  B, S, _ = x.shape
  # the H heads, from (B, S, E) to (B, H, S, D)
  q = (x @ p["Wq"] + p["bq"]).reshape(B, S, H, D).transpose(0, 2, 1, 3)
  k = (x @ p["Wk"] + p["bk"]).reshape(B, S, H, D).transpose(0, 2, 1, 3)
  v = (x @ p["Wv"] + p["bv"]).reshape(B, S, H, D).transpose(0, 2, 1, 3)
  wei = q @ k.transpose(0, 1, 3, 2) * D**-0.5                         # (B, H, S, S)
  wei = np.where(np.tril(np.ones((S, S), dtype=bool)), wei, -np.inf)
  wei = np.exp(wei - wei.max(axis=-1, keepdims=True))                 # softmax of each row
  wei /= wei.sum(axis=-1, keepdims=True)
  out = wei @ v                                                       # (B, H, S, D)
  return out.transpose(0, 2, 1, 3).reshape(B, S, E) @ p["Wo"] + p["bo"]   # (B, S, E)


def layernorm(x, w, b):
  mean = x.mean(axis=-1, keepdims=True)
  var = x.var(axis=-1, keepdims=True)
  return (x - mean) / np.sqrt(var + eps) * w + b


def block(x, p):
  x = x + attention(layernorm(x, p["ln1_w"], p["ln1_b"]), p)
  x = x + ffn(layernorm(x, p["ln2_w"], p["ln2_b"]), p)
  return x


rng = np.random.RandomState(19)   # the generator of gpt.h, std::mt19937, with the same seed


def init_values(shape, scale):
  """ the next values of init_value(scale) of gpt.h: libstdc++ turns each 32-bit value r of the generator into the fp32
  value u = r / 2^32, and then into 2u - 1, in [-1, 1) """
  u = rng.randint(0, 2**32, size=shape, dtype=np.uint32).astype(np.float32) / np.float32(2**32)
  return np.float32(scale) * (u * 2 - 1)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, default=B, help="batch size")
  args = parser.parse_args()

  # the parameters of the L blocks, with the random values of gpt.h, where the weights W are stored as (in x out)
  blocks = []
  for _ in range(L):
    p = {}
    for name, n_in, n_out in [("q", E, E), ("k", E, E), ("v", E, E), ("o", E, E), ("1", E, 4*E), ("2", 4*E, E)]:
      p["W" + name] = init_values((n_in, n_out), 0.02)
      p["b" + name] = init_values(n_out, 0.02)
    for name in ["ln1", "ln2"]:   # the defaults of PyTorch
      p[name + "_w"], p[name + "_b"] = np.ones(E, np.float32), np.zeros(E, np.float32)
    blocks.append(p)
  x = init_values((args.batch, S, E), 1)

  for it in range(warmup_iters + benchmark_iters):
    if it == warmup_iters:
      start = time.perf_counter()
    y = x
    for p in blocks:
      y = block(y, p)
  ms = (time.perf_counter() - start) * 1000 / benchmark_iters

  y = y[0]   # the output of the first sequence, to compare implementations
  print(f"NumPy, batch size {args.batch}: {ms:.1f} ms per forward pass")
  print(f"output: {y[0, 0]:.4f} {y[0, 1]:.4f} {y[0, 2]:.4f} ..., mean |y| {np.abs(y).mean():.4f}")


if __name__ == "__main__":
  main()
