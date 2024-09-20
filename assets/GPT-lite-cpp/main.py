import time
import os
import sys

import torch
import torch.nn.functional as F

#use the GPTlite and Benchmark models from the folder GPT-lite
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
import benchmark
from gptlite import GPTlite, block_size, n_embd

torch_compile_kwargs=dict(mode='default-overhead', fullgraph=False)
warmup_epochs = 30
benchmark_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def output_jit_model(model, model_name, x=None):
  # output the model as binary (with trace or script)
  if x is not None:
    model_jit = torch.jit.trace(model, (x))
  else:
    model_jit = torch.jit.script(model) 
  model_filename = f"{model_name.replace(' ', '_')}.pt"
  model_jit.save(model_filename)
  print(f"model saved to {model_filename}")

def benchmark_train(model, x, label, model_name):

  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  for epoch in range(warmup_epochs+benchmark_epochs):

    if epoch==warmup_epochs:
      start_time = time.time()

    optimizer.zero_grad() 
    output = model(x)
    output = F.softmax(output, dim=1)
    loss = F.cross_entropy(output, label)
    loss.backward()
    optimizer.step()

  benchmark_time = float(time.time() - start_time)
  print(f"{model_name} train runtime: {benchmark_time} seconds")
  print(f"{model_name} train throughput: {benchmark_epochs / benchmark_time} epochs/second")


def benchmark_inference(model, x, model_name, epochs_multiplier=10):

  model.eval()
  with torch.no_grad():
    for _ in range(warmup_epochs*epochs_multiplier): 
      model(x)
    
    start_time = time.time()
    for _ in range(benchmark_epochs*epochs_multiplier):
      model(x)

  benchmark_time = float(time.time() - start_time)
  print(f"{model_name} inference runtime: {benchmark_time} seconds")
  print(f"{model_name} inference throughput: {benchmark_epochs*epochs_multiplier / benchmark_time} epochs/second")


def main():

  assert torch.cuda.is_available(), "CUDA not available"

  # Deep DNN model (W=256, L=2048)
  model_name = "Deep DNN"
  W, L, batch_size = 256, 2048, 2048
  in_size = out_size = W
  x = torch.randn(batch_size, in_size).to(device)
  label = torch.randn(batch_size, out_size).to(device)
  model = benchmark.BenchmarkModel(W, L, in_size, out_size).to(device)
  if torch_compile_kwargs is None:
    output_jit_model(model, model_name)
  else:
    model = torch.compile(model, **torch_compile_kwargs)
  benchmark_train(model, x, label, model_name)
  benchmark_inference(model, x, model_name)

  # Wide DNN Model (W=8192, L=3)
  model_name = "Wide DNN"
  W, L, batch_size = 8192, 3, 2048
  in_size = out_size = W
  x = torch.randn(batch_size, in_size).to(device)
  label = torch.randn(batch_size, out_size).to(device)
  model = benchmark.BenchmarkModel(W, L, in_size, out_size).to(device)
  if torch_compile_kwargs is None:
    output_jit_model(model, model_name)
  else:
    model = torch.compile(model, **torch_compile_kwargs)
  benchmark_train(model, x, label, model_name)
  benchmark_inference(model, x, model_name)

  # GPTlite model: (B, T, C) = (batch_size, block_size, n_embed)
  model_name = "GPTlite"
  vocab_size, batch_size = 65, 1 
  idx = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
  label = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
  model = GPTlite(vocab_size).to(device)
  if torch_compile_kwargs is None:
    output_jit_model(model, model_name)
  else:
    model = torch.compile(model, **torch_compile_kwargs)
  benchmark_train(model, idx, label, model_name)
  benchmark_inference(model, idx, model_name)


if __name__ == "__main__":
  main()
