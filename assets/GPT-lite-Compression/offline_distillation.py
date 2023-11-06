import os
import time
import torch
import torch.nn.functional as F
import sys

#use the GPTlite and Benchmark models from the post GPT-lite-cpp
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite-cpp'))
import gptlite
import benchmark

label_filename = lambda batch, folder: os.path.join(folder,f"logits_{batch}.pt") 

def training(model, dataloader, epochs, teacher_model=False):
  # reminder: CrossEntropyLoss(x) = NLLLoss(LogSoftmax(x))
  # CrossEntropyLog expects unnormalized logits; NLLLoss expects log probabilities
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  start_time = time.time()
  for epoch in range(epochs):
    running_loss = 0.0
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
      running_loss += loss.item()
    print(f"{epoch}:: loss {running_loss / (epoch+1)}")
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


def main(scale_factor=1.0, train_epochs=10, output_folder="output", model='gptlite', random_seed=42):
  """
  first run: train teacher model against hard labels and output soft labels
  second run: load soft labels and train smaller model against soft labels
  """
  torch.manual_seed(random_seed) 
  
  #if folder exists: it contains labels from the teacher, so we're training student
  teacher_model = not os.path.exists(output_folder)
  os.makedirs(output_folder, exist_ok=True)

  # initialize teacher model or a scaled version of the student model
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
    W, L = int(W*scale_factor), int(L*scale_factor)
    train_dataset = benchmark.get_dataset(W)
    model = benchmark.get_model(W, L)
  else:
    raise NotImplementedError(f"model {model} not implemented")
  
  training (model, train_dataset, epochs=train_epochs, teacher_model=teacher_model)
  inference(model, train_dataset, output_labels=teacher_model)

  
if __name__ == "__main__":
  main(scale_factor=1.0) # student of same size as teacher
  # main(scale_factor=0.8) # student of 80% size of teacher


  
