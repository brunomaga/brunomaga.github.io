import os
import sys
import torch
import time
import torch.nn.functional as F
from gptlite_inference import GPTlite_Inference


#use the default GPTlite model and utils from the post GPTlite
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPTlite'))
from gptlite import GPTlite
from utils import get_batch, get_tiny_shakespeare_data, get_gpt2_small_model_parameters


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, decode_fn = get_tiny_shakespeare_data()
  n_layers, d_model, n_heads, d_head, seqlen, dropout_p = get_gpt2_small_model_parameters()

  # Smaller model for debugging
  n_layers = 6
  d_model = 384
  n_heads = 6
  d_head = 64

  batch_size = 1 
  generated_seqlen = 1000  # Length of sequence to generate
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  for model_class in (GPTlite, GPTlite_Inference):
    model = model_class(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen)
    model = model.to(device) # move model to GPU
    model.eval()
    n_layers = len(model.blocks)  # Number of transformer layers

    with torch.inference_mode():
      # perform one eval step and compute loss
      idx, targets = get_batch(valid_data, batch_size=batch_size, seqlen=seqlen)
      idx, targets = idx.to(device), targets.to(device) #move data to GPU

      cache = [None] * n_layers  # Initialize cache
      idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start token

      torch.cuda.synchronize()
      start_time = time.time() 
      for _ in range(generated_seqlen):
        x = idx[:, -1:].to(device)  # Process only the latest token
        logits, cache = model(x, cache=cache, max_seqlen=seqlen)
        logits = logits[:, -1]  # Logits for the last position
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)

      torch.cuda.synchronize()
      runtime = time.time() - start_time
      print(f"Runtime for {model_class.__name__}: {runtime:.4f} seconds")

      # print generated string
      idx_without_bos = idx[0, 1:].tolist() # remove batch dim and BOS token (\n)
      print("Generated text:", decode_fn(idx_without_bos))

