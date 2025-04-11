import torch
import torch.nn.functional as F
from gptlite import GPTlite
from utils import get_batch, get_tiny_shakespeare_data, get_gpt2_small_model_parameters


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, train_data, valid_data, encode_fn, decode_fn = get_tiny_shakespeare_data()
  n_layers, d_model, n_heads, d_head, seqlen, dropout_p = get_gpt2_small_model_parameters()

  # Smaller model for debugging
  n_layers = 6
  d_model = 384
  n_heads = 6
  d_head = 64

  # Train parameters
  batch_size = 32   # Batch size
  eval_interval = 100  # evaluation interval
  train_iters = 50000  # number of training iterations
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  learning_rate=2.5e-4 # optimizer's learning rate (Adam's LR on ChatGPT2)

  model = GPTlite(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen)
  model = model.to(device) # move model to GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # train the model
  for step in range(1, train_iters+1):

    # train step
    model.train()
    idx, targets = get_batch(train_data, batch_size=batch_size, seqlen=seqlen)   #get a batch of training data
    idx, targets = idx.to(device), targets.to(device) #move data to GPU
    logits = model(idx)   #forward pass
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()   #backward pass
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to avoid exploding gradients
    optimizer.step()   #update parameters
    optimizer.zero_grad(set_to_none=True)  #sets to None instead of 0, to save memory
    if step % 10 == 0:
        print(f"Train step {step}, loss {loss.item():.2f}")
        
    if step % eval_interval > 0:
        continue

    # evaluation step
    model.eval()
    with torch.inference_mode():

      # perform one eval step and compute loss
      idx, targets = get_batch(valid_data, batch_size=batch_size, seqlen=seqlen)
      idx, targets = idx.to(device), targets.to(device) #move data to GPU
      logits = model(idx)   #forward pass
      loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
      print(f"Eval step {step}, eval loss {loss.item():.2f}")

      # Generate one sample from the current state of the model

      # Begin of String (batch size 1): the token with id 0 (the \n character)
      idx = torch.zeros((1,1), dtype=torch.long, device=device)
      generated_seqlen = 100

      for _ in range(generated_seqlen):
        idx_cond = idx[:, -seqlen:] #we can never a sentence longer than seqlen
        logits = model(idx_cond).to(device) #call fwd without targets
        logits = logits[:, -1] # take last token. shape (B=1, C)
        probs = F.softmax(logits, dim=-1) # shape (B=1, C)
        #randomly sample the next tokens, 1 for each of the previous probability distributions
        #(one could take instead the argmax, but that would be deterministic and boring)
        idx_next = torch.multinomial(probs, num_samples=1) # shape (B=1, 1)
        #append next token ix to the solution sequence so far
        idx = torch.cat([idx, idx_next], dim=-1) # shape (B=1, T+1)

      # print generated string
      idx_without_bos = idx[0, 1:].tolist() # remove batch dim and BOS token (\n)
      print("Generated text:", decode_fn(idx_without_bos))
