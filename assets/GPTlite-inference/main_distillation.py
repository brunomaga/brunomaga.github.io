import os
import sys
import torch
import torch.nn.functional as F

#use the default GPTlite model and utils from the post GPTlite
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPTlite'))
from gptlite import GPTlite
from utils import get_batch, get_tiny_shakespeare_data, get_gptlite_model_parameters, get_gptlite_distilled_model_parameters
GPTLITE_CKPT_PATH = os.path.join(current_dir, '..', 'GPTlite', 'gptlite.pth')

if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, train_data, valid_data, encode_fn, decode_fn = get_tiny_shakespeare_data()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Train parameters
  eval_interval = 100  # evaluation interval
  train_iters = 50000  # number of training iterations
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  temperature = 2 # temperature for distillation

  # Load pre-trained large model parameters
  torch.manual_seed(42) # random seed for model initialization, for reproducibility
  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gptlite_model_parameters()
  model_teacher = GPTlite(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device).eval()
  if os.path.exists(GPTLITE_CKPT_PATH):
    model_teacher.load_state_dict(torch.load(GPTLITE_CKPT_PATH, map_location=device))
    print(f"Loaded model from {GPTLITE_CKPT_PATH} into GPTlite")

  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gptlite_distilled_model_parameters()
  model_student = GPTlite(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device)
  optimizer = torch.optim.Adam(model_student.parameters(), lr=lr)

  # train the model
  for step in range(1, train_iters+1):

    # train step
    model_student.train()
    idx, targets = get_batch(train_data, batch_size=batch_size, seqlen=seqlen)   #get a batch of training data
    idx, targets = idx.to(device), targets.to(device) #move data to GPU
    logits_student = model_student(idx)   #forward pass
    logits_teacher = model_teacher(idx)   #forward pass
    log_softmax_student = F.log_softmax(logits_student/temperature, dim=-1)   #log softmax of student model
    log_softmax_teacher = F.log_softmax(logits_teacher/temperature, dim=-1)   #log softmax of teacher model
    loss = F.kl_div(log_softmax_student, log_softmax_teacher, log_target=True) * (temperature ** 2)  #compute KL divergence loss
    loss.backward()   #backward pass
    torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0) # gradient clipping to avoid exploding gradients
    optimizer.step()   #update parameters
    optimizer.zero_grad(set_to_none=True)  #sets to None instead of 0, to save memory
    if step % 10 == 0:
        print(f"Train step {step}, loss {loss.item():.4f}")

    if step % eval_interval > 0:
        continue

    # evaluation step
    model_student.eval()
    with torch.inference_mode():

      # perform one eval step and compute loss
      idx, targets = get_batch(valid_data, batch_size=batch_size, seqlen=seqlen)
      idx, targets = idx.to(device), targets.to(device) #move data to GPU
      logits_student = model_student(idx)   #forward pass
      logits_teacher = model_teacher(idx)   
      log_softmax_student = F.log_softmax(logits_student/temperature, dim=-1)   #log softmax of student model
      log_softmax_teacher = F.log_softmax(logits_teacher/temperature, dim=-1)   #log softmax of teacher model
      loss = F.kl_div(log_softmax_student, log_softmax_teacher, log_target=True) * (temperature ** 2)  
      print(f"Eval step {step}, eval loss {loss.item():.4f}")

      # Generate a sentence from the current state of the model
      # Begin of String (batch size 1): the token with id 0 (the \n character)
      idx = torch.zeros((1,1), dtype=torch.long, device=device)
      generated_seqlen = 100

      for _ in range(generated_seqlen):
        idx_cond = idx[:, -seqlen:] #we can never a sentence longer than seqlen
        logits = model_student(idx_cond).to(device) #call fwd without targets
        logits = logits[:, -1] # take last token. shape (B=1, C)
        probs = F.softmax(logits, dim=-1) # shape (B=1, C)
        # sample the next token (we could take instead the argmax, but that would be deterministic and boring)
        idx_next = torch.multinomial(probs, num_samples=1) # shape (B=1, 1)
        # append next token idx to the solution sequence so far
        idx = torch.cat([idx, idx_next], dim=-1) # shape (B=1, T+1)

      # print generated string
      idx_without_bos = idx[0, 1:].tolist() # remove batch dim and BOS token (\n)
      print("Generated text:", decode_fn(idx_without_bos))

      # Now save the model
      torch.save(model_student.state_dict(), "gptlite_distilled.pth")
      print("Model saved to gptlite.pth")