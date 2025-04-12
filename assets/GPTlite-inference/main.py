import os
import sys
import torch
import time
import torch.nn.functional as F
from gptlite_kvcache import GPTlite_KVCache


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
  n_layers = 2
  d_model = 384
  n_heads = 8
  d_head = 64

  batch_size = 1 
  generated_seqlen = 1000  # Lenght of sequence to generate
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  kv_cache = None # used by GPTlite_KVCache
  generated_texts = []
  for model_class in (GPTlite, GPTlite_KVCache):
    torch.manual_seed(42) # random seed for model initialization, for reproducibility
    model = model_class(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen)
    model = model.to(device) # move model to GPU
    model.eval()

    if torch.cuda.is_available():
      torch.cuda.synchronize()

    start_time = time.time() 
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start token

    with torch.inference_mode():
      for _ in range(generated_seqlen):
        idx_cond = idx[:, -seqlen:].to(device)  # Use only the last seqlen tokens
        if isinstance(model, GPTlite_KVCache):
          logits, kv_cache = model(idx_cond, max_seqlen=seqlen, kv_cache=kv_cache)
        else:
          logits = model(idx_cond)
        logits = logits[:, -1]  # Logits for the last position
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)

    if torch.cuda.is_available():
      torch.cuda.synchronize()
    runtime = time.time() - start_time
    print(f"Runtime for {model_class.__name__}: {runtime:.4f} seconds")

    # collect generated string
    idx_without_bos = idx[0, 1:].tolist() # remove batch dim and BOS token (\n)
    generated_texts.append(decode_fn(idx_without_bos))

  # sanity check: make sure all generated texts match
  assert all([text==generated_texts[0] for text in generated_texts[1:]]), "Output mismatch!"

