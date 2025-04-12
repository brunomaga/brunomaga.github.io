import os
import sys
import torch
import time
import torch.nn.functional as F
from gptlite_kvcache import GPTlite_KVCache
import numpy as np


#use the default GPTlite model and utils from the post GPTlite
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPTlite'))
from gptlite import GPTlite
from utils import get_tiny_shakespeare_data, get_gpt2_small_model_parameters


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, encode_fn, decode_fn = get_tiny_shakespeare_data()
  n_layers, d_model, n_heads, d_head, seqlen, dropout_p = get_gpt2_small_model_parameters()

  # Smaller model for debugging
  n_layers = 2
  d_model = 384
  n_heads = 8
  d_head = 64

  # inference / sequence generation settings
  batch_size = 4 # number of sequences to generate in parallel
  n_sequences = 20 # total number of sequences to generate
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  bos_token = 0 # Start and end of string is token 0 which is \n
  eos_tokens = encode_fn("EOF\n") # End of string is
  prompts = [ torch.full((1,), bos_token, dtype=torch.long, device=device) for i in range(n_sequences) ] # dummy single-token prompt
  generated_texts = []

  # In this dummy example, use a random generated length. In a trained model this would be matching last token to EOS
  random_answer_seqlens = np.random.randint(seqlen, seqlen*6, size=n_sequences)
  is_completed = lambda toks: len(toks)>=len(eos_tokens) and (toks[-len(eos_tokens):] == eos_tokens).all()
  
  # INFERENCE WITHOUT CONTINUOUS BATCH: parallelize over the batch dimension
  for model_class in (GPTlite, GPTlite_KVCache):
    torch.manual_seed(42) # random seed for model initialization, for reproducibility
    model = model_class(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device).eval()
    generated_texts.append([None]*n_sequences) # list to store generated texts

    if torch.cuda.is_available():
      torch.cuda.synchronize()
    start_time = time.time() 
    kv_cache = None  # Initialize cache for GPTlite_KVCache
    with torch.inference_mode():
      for sequence_id in range(0, n_sequences, batch_size):
        # length of sequence to generate is the length of the longest string to be generated in the batch
        idx = torch.stack(prompts[sequence_id:sequence_id+batch_size], dim=0)  # batch of sequences 
        completed = 0
        while completed < batch_size:
          if isinstance(model, GPTlite_KVCache):
            last_idx = idx[:, -1:].to(device)  # Pass only the latest token
            logits, kv_cache = model(last_idx, max_seqlen=seqlen, kv_cache=kv_cache)
          else:
            idx_cond = idx[:, -seqlen:].to(device)  # Use only the last seqlen tokens
            logits = model(idx_cond)
          logits = logits[:, -1]  # Logits for the last position
          probs = F.softmax(logits, dim=-1)
          idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # argmax insted of multinomial for deterministic results
          idx = torch.cat([idx, idx_next], dim=-1)

          # because model is not trained, we wont get EOS token, so we need to add it manually
          for i, seq_tokens in enumerate(idx):
            if len(seq_tokens) == random_answer_seqlens[sequence_id+i]:
              idx[i][-len(eos_tokens):] = eos_tokens # add EOS token to the sequence
              
            # check if sequence was completed, and increase completed counter if that's the case
            if is_completed(seq_tokens):
              # print(f"Completed sequence {sequence_id+i} with length {len(seq_tokens)} in slot {i}.")	
              completed += 1
              generated_texts[-1][sequence_id+i] = decode_fn(seq_tokens.tolist())

    # sync, measure runtime and collect generated string
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    print(f"Runtime for {model_class.__name__}: {(time.time() - start_time):.4f} seconds")

  # INFERENCE WITH CONTINUOUS BATCHING: continuosly generate strings by filling up batch 
  torch.manual_seed(42) # random seed for model initialization, for reproducibility
  model = GPTlite(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device).eval()

  idx = torch.stack(prompts[:batch_size], dim=0) 
  active = list(range(batch_size)) # list of active sequences as tuples as sequence_id 
  active_seqlens = [0]*batch_size # list of active sequence lengths
  generated_texts.append([None]*n_sequences) # list to store generated texts

  if torch.cuda.is_available():
    torch.cuda.synchronize()
  start_time = time.time() 

  with torch.inference_mode():
    completed = 0
    next_to_be_processed = batch_size
    while completed < n_sequences:
          
      idx_cond = idx[:, -seqlen:].to(device)  # Use only the last seqlen tokens
      logits = model(idx_cond)
      logits = logits[:, -1]  # Logits for the last position
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # for deterministic results
      idx = torch.cat([idx, idx_next], dim=-1)

      # Add new sequences if possible
      for i, (sequence_id, seq_tokens) in enumerate(zip(active, idx)):

        if sequence_id is None:
          continue
        
        # because model is not trained, we wont get EOS token, so we need to add it manually
        active_seqlens[i] += 1 # increase active sequence length
        if active_seqlens[i] == random_answer_seqlens[sequence_id]:
          seq_tokens[-len(eos_tokens):] = eos_tokens # add EOS token to the sequence
          
        if is_completed(seq_tokens):
          generated_texts[-1][sequence_id] = decode_fn(seq_tokens.tolist())
          active[i] = next_to_be_processed if next_to_be_processed < n_sequences else None  # next sequence to be processed 
          active_seqlens[i] = 0 # reset active sequence length
          if active[i] is not None: # load next prompt
            idx[i][-seqlen:-1].fill_(0)  # Reset the sequence
            idx[i][-1] = prompts[active[i]] 
            next_to_be_processed+=1
          completed += 1
          # print(f"Completed sequence {sequence_id} with length {len(seq_tokens)} in slot {i}. Loaded next prompt {active[i]}")

  if torch.cuda.is_available():
    torch.cuda.synchronize()
  print(f"Runtime for GPTlite (continuous batching): {(time.time() - start_time):.4f} seconds")

  # sanity check: make sure all generated texts match
  for sequence_id, text in enumerate(generated_texts[0]):
    for j in range(1, len(generated_texts)):
      eos_offset = len(text) if text.find('\n') == -1 else text.find('\n')
      assert text[:eos_offset]==generated_texts[j][sequence_id][:eos_offset], f"Output mismatch at index {sequence_id}!"
