import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import random
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)

#use the GPTlite and Benchmark models from the post GPT-lite-cpp
current_dir = os.path.dirname(os.path.realpath(__file__))
output_folder = "output"
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite-DeepSpeed'))

#input path for the tiny shakespeare dataset
tinyshakespeare_path = os.path.join(current_dir, '..', 'GPT-lite-DeepSpeed', 'tinyshakespeare.txt')

# method that returns the filename of the soft labels of each batch
label_filename = lambda batch: os.path.join(output_folder,f"logits_{batch}.pt") 

def training(model, dataloader, epochs, teacher_model=False):
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  start_time = time.time()
  for epoch in range(epochs):
    running_loss = 0.0
    for b, (x, label) in enumerate(dataloader):
      x, label = x.to(device), label.to(device)
      optimizer.zero_grad() 
      logits = model(x)
      if teacher_model:
        loss = F.cross_entropy(logits, label)
      else:
        student_log_probs = F.log_softmax(logits, dim=-1)
        teacher_logits = torch.load(label_filename(b)).to(device)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_log_probs, log_target=True)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    # print(f"{epoch}:: loss {running_loss / (epoch+1)}")
  print(f"Train loss: {running_loss / (epoch+1)}.")
  print(f"Train runtime: {float(time.time() - start_time)} seconds")


def inference(model, dataloader, output_labels=False):
  model.eval()
  start_time = time.time()
  running_acc=0
  with torch.no_grad():
    for b, (x, label) in enumerate(dataloader):
      x, label = x.to(device), label.to(device)
      output = model(x)
      running_acc += (x.argmax(-1)==label).sum()/len(x) 
      if output_labels:
        torch.save(output, label_filename(b))
  print(f"Inference accuracy: {running_acc/(b+1)*100}%.")
  print(f"Inference runtime: {float(time.time() - start_time)} seconds")


def main(scale_factor=1.0, train_epochs=30, model_name='benchmark'):

  #if folder does not exist: we are training our very first teacher
  teacher_model = not os.path.exists(output_folder)
  os.makedirs(output_folder, exist_ok=True)

  # initialize teacher model or a scaled version of the student model
  if model_name.lower()=='gptlite':
    import gptlite
    batch_size=1
    gptlite.n_layer    = int(gptlite.n_layer*scale_factor)
    gptlite.n_embd     = int(gptlite.n_embd*scale_factor)
    gptlite.n_head     = int(gptlite.n_head*scale_factor)
    gptlite.block_size = int(gptlite.block_size*scale_factor)
    train_dataset, valid_dataset, vocab_size = gptlite.get_dataset(filename=tinyshakespeare_path)
    model = gptlite.get_model(vocab_size).to(device)
  elif model_name.lower()=='benchmark':
    import benchmark
    batch_size=2048
    W, L = 128,128 # deep model
    train_dataset, valid_dataset = benchmark.get_dataset(in_size=W, num_classes=W)
    model = benchmark.get_model(
      W=int(W*scale_factor), L=int(L*scale_factor),
      in_size=W, num_classes=W).to(device)
  else:
    raise NotImplementedError(f"model {model} is not implemented")
  
  def seed_init_fn(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
 
  dataloader_kwargs = {'batch_size': batch_size, 'shuffle': True, 'worker_init_fn': seed_init_fn }
  train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
  valid_dataloader = DataLoader(valid_dataset, **dataloader_kwargs)

  training (model, train_dataloader, epochs=train_epochs, teacher_model=teacher_model) #train teacher
  inference(model, valid_dataloader, output_labels=False) # test accuracy of teacher
  inference(model, train_dataloader, output_labels=True) # output soft labels for next student

  
if __name__ == "__main__":
  import shutil

  # iteratively distill the model to smaller sizes until we reach 1/2 the size
  if os.path.exists(output_folder): shutil.rmtree(output_folder)
  for scale_factor in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    main(scale_factor=scale_factor)

  # try a model that is hald the size directly 
  if os.path.exists(output_folder): shutil.rmtree(output_folder)
  main(scale_factor=0.5)

  
