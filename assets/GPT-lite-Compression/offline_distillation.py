import torch
import torch.nn.functional as F
import os
import time

import gptlite
import benchmark
benchmark_model_W_L = (1024, 1024)

label_filename = lambda batch, folder: os.path.join(folder,f"{batch}.pt") 

def training(model, dataloader, distilation=False):
  # reminder: CrossEntropyLoss = LogSoftmax + NLLLoss
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  start_time = time.time()
  for b, (x, label) in enumerate(dataloader):
    optimizer.zero_grad() 
    output = model(x)
    if distilation:
      # output = F.log_softmax(output, dim=1)
      soft_labels = torch.load(label_filename(b)).to(model.device)
      loss = F.kl_div(output, soft_labels, log_target=False)
    else:
      loss = F.cross_entropy(output, label)
    loss.backward()
    optimizer.step()
  print(f"{b}:: {loss.item()}")
  print(f"train runtime: {float(time.time() - start_time)} seconds")


def inference(model, dataloader, output=False):
  model.eval()
  start_time = time.time()
  with torch.no_grad():
    for b, (x, label) in enumerate(dataloader):
      output = model(x)
      accuracy = (x.argmax(1)==label).sum()/len(x) 
      print(f"{b}: {accuracy}%")
      if output:
        # output = F.log_softmax(output, dim=1)
        torch.save(output, label_filename(b))
  print(f"inference runtime: {float(time.time() - start_time)} seconds")


def main(scale_factor=0.8, output_folder="distilation_output", model='gptlite'):
  
  assert model in ('gptlite', 'benchmark')
  
  # we will force each step to be a complete new run due to mem leak on GPU when
  # allocating and deallocating models on the same run
  
  #offline distilation: output soft labels on first run, use it on second run
  first_round_distilation = not os.path.exists(output_folder)
  os.makedirs(output_folder, exist_ok=True)

  #first run, fully train model and output soft labels
  if first_round_distilation:  
    if model=='benchmark':
      model = benchmark.BenchmarkModel(*benchmark_model_W_L)
    else:
      model = gptlite.GPTlite(vocab_size=1024)
    training(model, dataloader, distilation=False)
    inference(model, dataloader, output=True)

  #second run, load soft labels and train smaller model with it
  else:
    if model=='benchmark':
      model = benchmark.BenchmarkModel(*(benchmark_model_W_L*scale_factor))
    else:
      gptlite.n_layer    = int(gptlite.n_layer*scale_factor)
      gptlite.n_embd     = int(gptlite.n_embd*scale_factor)
      gptlite.n_head     = int(gptlite.n_head*scale_factor)
      gptlite.block_size = int(gptlite.block_size*scale_factor)
      model = gptlite.GPTlite(vocab_size=1024)
    training(model, dataloader, distilation=True)
    inference(model, dataloader, output=False)

  
if __name__ == "__main__":
  main()


  