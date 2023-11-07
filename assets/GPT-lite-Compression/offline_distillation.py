import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys

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
  # reminder: CrossEntropyLoss(x) = NLLLoss(LogSoftmax(x))
  # CrossEntropyLog expects unnormalized logits; NLLLoss expects log probabilities
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
        # docs:  It is recommended to pass certain distributions (like softmax) in the log space to avoid numerical issues caused by explicit log.
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print(f"{epoch}:: loss {running_loss / (epoch+1)}")
  print(f"train runtime: {float(time.time() - start_time)} seconds")


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
  print(f"inference accuracy: {running_acc/(b+1)*100}%")
  print(f"inference runtime: {float(time.time() - start_time)} seconds")


def main(scale_factor=1.0, train_epochs=3, model='benchmark', random_seed=42):
  """
  first run: train teacher model against hard labels and output soft labels
  second run: load soft labels and train smaller model against soft labels
  """
  print(f"==== starting run with scale factor {scale_factor} ====")

  #reset seed so that batches from teacher and student runs are the same
  torch.manual_seed(random_seed)
  
  #if folder does not exist: we are training our first teacher
  teacher_model = not os.path.exists(output_folder)
  os.makedirs(output_folder, exist_ok=True)

  # initialize teacher model or a scaled version of the student model
  if model.lower()=='gptlite':
    import gptlite
    batch_size=1
    gptlite.n_layer    = int(gptlite.n_layer*scale_factor)
    gptlite.n_embd     = int(gptlite.n_embd*scale_factor)
    gptlite.n_head     = int(gptlite.n_head*scale_factor)
    gptlite.block_size = int(gptlite.block_size*scale_factor)
    train_dataset, valid_dataset, vocab_size = gptlite.get_dataset(filename=tinyshakespeare_path)
    model = gptlite.get_model(vocab_size).to(device)
  elif model.lower()=='benchmark':
    import benchmark
    batch_size=2048
    # W, L = 8192, 3 # wide model
    # W, L = 256, 2048 # deep model
    W, L = 256,256 # deep model
    train_dataset, valid_dataset = benchmark.get_dataset(in_size=W, num_classes=W)
    model = benchmark.get_model(
      W=int(W*scale_factor), L=int(L*scale_factor),
      in_size=W, num_classes=W).to(device)
  else:
    raise NotImplementedError(f"model {model} is not implemented")
  
  dataloader_kwargs = {'batch_size': batch_size, 'shuffle': True }
  #reset seed so that batches from teacher and student runs are the same
  torch.manual_seed(random_seed)
  train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
  valid_dataloader = DataLoader(valid_dataset, **dataloader_kwargs)
  training (model, train_dataloader, epochs=train_epochs, teacher_model=teacher_model)
  inference(model, valid_dataloader, output_labels=False) # test accuracy of teacher model

  #reset seed so that batches from teacher and student runs are the same
  torch.manual_seed(random_seed)
  train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
  inference(model, train_dataloader, output_labels=True)  # output soft labels for next student

  
if __name__ == "__main__":
  import shutil
  if os.path.exists(output_folder): shutil.rmtree(output_folder)
  main(scale_factor=1.0) # student of same size as teacher
  main(scale_factor=0.8) # student of 80% size of teacher
  main(scale_factor=0.6) # student of 60% size of teacher


  
