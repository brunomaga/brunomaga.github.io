import os
import time
import torch
import torch.nn.functional as F
import sys

#use the GPTlite and Benchmark models from the post GPT-lite-DeepSpeed
sys.path.insert(0, os.path.join('..', 'GPT-lite-DeepSpeed'))
import gptlite
import benchmark

label_filename = lambda batch, folder: os.path.join(folder,f"logits_{batch}.pt") 

def training(model, dataloader, teacher_model=False):
  # reminder: CrossEntropyLoss(x) = NLLLoss(LogSoftmax(x))
  # CrossEntropyLog expects unnormalized logits; NLLLoss expects log probabilities
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  start_time = time.time()
  for b, (x, label) in enumerate(dataloader):
    optimizer.zero_grad() 
    output = model(x)
    if teacher_model:
      loss = F.cross_entropy(output, label)
    else:
      student_log_probs = F.log_softmax(output, dim=-1)
      teacher_logits = torch.load(label_filename(b)).to(model.device)
      teacher_probs = F.softmax(teacher_logits, dim=-1)
      loss = F.kl_div(student_log_probs, teacher_probs, log_target=False)
    loss.backward()
    optimizer.step()
  print(f"{b}:: {loss.item()}")
  print(f"train runtime: {float(time.time() - start_time)} seconds")


def inference(model, dataloader, output_labels=False):
  model.eval()
  start_time = time.time()
  with torch.no_grad():
    for b, (x, label) in enumerate(dataloader):
      output = model(x)
      accuracy = (x.argmax(1)==label).sum()/len(x) 
      print(f"{b}: {accuracy}%")
      if output_labels:
        torch.save(output, label_filename(b))
  print(f"inference runtime: {float(time.time() - start_time)} seconds")


def main(output_folder="output", model='gptlite', random_seed=42):
  
  torch.manual_seed(random_seed) 
  
  #if folder exists: it contains labels from the teacher, we're training student
  model_is_teacher = not os.path.exists(output_folder)
  scale_factor = 1.0 if model_is_teacher else 0.8
  os.makedirs(output_folder, exist_ok=True)

  # initialize GPT-model and dataset. Teacher model will be scaled
  if model.lower()=='gptlite':
    gptlite.n_layer    = int(gptlite.n_layer*scale_factor)
    gptlite.n_embd     = int(gptlite.n_embd*scale_factor)
    gptlite.n_head     = int(gptlite.n_head*scale_factor)
    gptlite.block_size = int(gptlite.block_size*scale_factor)
    train_dataset, vocab_size = gptlite.get_dataset()
    model = gptlite.get_model(vocab_size)
  elif model.lower()=='benchmark':
    W, L = 8192, 3 # wide model
    # W, L = 256, 2048 # deep model
    train_dataset = benchmark.get_dataset(W*scale_factor)
    model = benchmark.get_model(W*scale_factor, L*scale_factor)
  else:
    raise NotImplementedError(f"model {model} not implemented")
  
  #first run, fully train model and output soft labels
  #second run, load soft labels and train smaller model with it
  training (model, train_dataset, teacher_model = model_is_teacher)
  inference(model, train_dataset, output_labels = model_is_teacher)

  
if __name__ == "__main__":
  main()


  
