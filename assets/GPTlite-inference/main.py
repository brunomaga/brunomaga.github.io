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
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters, get_gptlite_distilled_model_parameters
GPTLITE_CKPT_PATH = os.path.join(current_dir, '..', 'GPTlite', 'gptlite.pth')
GPTLITE_DISTILLED_CKPT_PATH = os.path.join(current_dir, '..', 'GPTlite', 'gptlite_distilled.pth')


## helper functions to call on the beginning and end of each model execution

def benchmark_begin():
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  return time.time()

def benchmark_end(name, start_time):
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  print(f"Runtime for {name}: {(time.time() - start_time):.4f} seconds")


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, encode_fn, decode_fn = get_tiny_shakespeare_data()

  # inference / sequence generation settings
  batch_size = 4 # number of sequences to generate in parallel
  n_sequences = 40 # total number of sequences to generate
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  bos_token = 0 # beginning of string is a dummy prompt of token 0 which is \n
  eos_tokens = encode_fn("\nEOF\n") # End of string is string "EOF"
  prompts = [ torch.full((1,), bos_token, dtype=torch.long, device=device) for i in range(n_sequences) ] # input prompts
  generated_texts = []

  n_layers, d_model, n_heads, d_head, _, _, seqlen, dropout_p = get_gptlite_model_parameters()

  # In this dummy example, use a random generated length. In a trained model this would be matching last token to EOS
  random_answer_seqlens = np.random.randint(seqlen, seqlen*4, size=n_sequences)
  is_completed = lambda toks: len(toks)>=len(eos_tokens) and (toks[-len(eos_tokens):] == eos_tokens).all()
  

  ##################################################################################################
  #### REGULAR BATCHED INFERENCE: parallelize requests of diff length over the batch dimension  ####
  #### Variants: GPTlite, GPTlite with KV cache, and GPTlite distilled to a smaller model       ####
  ##################################################################################################	

  torch.manual_seed(42) # random seed for model initialization, for reproducibility
  model = GPTlite(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device).eval()
  if os.path.exists(GPTLITE_CKPT_PATH):
    model.load_state_dict(torch.load(GPTLITE_CKPT_PATH, map_location=device))
    print(f"Loaded model from {GPTLITE_CKPT_PATH} into GPTlite")

  torch.manual_seed(42) # random seed for model initialization, for reproducibility
  model_kvcache = GPTlite_KVCache(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device).eval()
  if os.path.exists(GPTLITE_CKPT_PATH):
    model_kvcache.load_state_dict(torch.load(GPTLITE_CKPT_PATH, map_location=device))
    print(f"Loaded model from {GPTLITE_CKPT_PATH} into GPTlite_KVCache")

  n_layers_d, d_model_d, n_heads_d, d_head_d, _, _, seqlen_d, dropout_p_d = get_gptlite_distilled_model_parameters()
  model_distilled = GPTlite(vocab_size, d_model_d, n_heads_d, d_head_d, n_layers_d, dropout_p_d, seqlen_d).to(device).eval()
  if os.path.exists(GPTLITE_DISTILLED_CKPT_PATH):
    model_distilled.load_state_dict(torch.load(GPTLITE_DISTILLED_CKPT_PATH, map_location=device))
    print(f"Loaded model from {GPTLITE_DISTILLED_CKPT_PATH} into GPTlite_KVCache")


  for (model_obj, model_name) in (
    (model, "original"),
    (model_kvcache, "kv cache"),
    (model_distilled, "distilled"),
  ):
    generated_texts.append([None]*n_sequences) # list to store generated texts
    start_time = benchmark_begin()

    kv_cache = None  # Initialize cache for GPTlite_KVCache
    with torch.inference_mode():
      for batch_start in range(0, n_sequences, batch_size):
        # length of sequence to generate is the length of the longest string to be generated in the batch
        tokens = torch.stack(prompts[batch_start:batch_start+batch_size], dim=0)  # batch of sequences 
        completed = 0
        while completed < batch_size:
          if model_name == "kv cache":
            last_token = tokens[:, -1:].to(device)  # Pass only the latest token
            logits, kv_cache = model_obj(last_token, max_seqlen=seqlen, kv_cache=kv_cache)
          elif model_name == "distilled":
            tokens_in_context = tokens[:, -seqlen_d:].to(device)
            logits = model_obj(tokens_in_context)
          else:
            tokens_in_context = tokens[:, -seqlen:].to(device)  # Use only the last seqlen tokens
            logits = model_obj(tokens_in_context)
          logits = logits[:, -1]  # Logits for the last position
          # for deterministic results use argmax of logits, instead of multinomial of the softmax of logits 
          next_token = torch.argmax(logits, dim=-1, keepdim=True)
          tokens = torch.cat([tokens, next_token], dim=-1)

          for i, seq_tokens in enumerate(tokens):

            # because model is not trained, we wont get EOS token, so we need to add it manually
            if len(seq_tokens) == random_answer_seqlens[batch_start+i]:
              tokens[i][-len(eos_tokens):] = eos_tokens # add EOS token to the sequence
              
            # check if sequence was completed, and increase completed counter if that's the case
            if is_completed(seq_tokens):
              # print(f"Completed sequence {sequence_id+i} with length {len(seq_tokens)} in slot {i}.")	
              completed += 1
              generated_texts[-1][batch_start+i] = decode_fn(seq_tokens.tolist())
    benchmark_end(model_name, start_time)



  #################################################################################################
  #### INFERENCE WITH CONTINUOUS BATCHING: continuosly generate strings by filling up batch    ####
  #################################################################################################

  tokens = torch.stack(prompts[:batch_size], dim=0) 
  active = list(range(batch_size)) # list of active sequences as tuples as sequence_id 
  active_seqlens = [0]*batch_size # list of active sequence lengths
  generated_texts.append([None]*n_sequences) # list to store generated texts
  start_time = benchmark_begin()

  with torch.inference_mode():
    completed = 0
    next_to_be_processed = batch_size
    while completed < n_sequences:
          
      tokens_in_context = tokens[:, -seqlen:].to(device)  # Use only the last seqlen tokens
      logits = model(tokens_in_context)
      logits = logits[:, -1]  # Logits for the last position
      # for deterministic results use argmax of logits, instead of multinomial of the softmax of logits 
      next_token = torch.argmax(logits, dim=-1, keepdim=True)
      tokens = torch.cat([tokens, next_token], dim=-1)

      # Add new sequences if possible
      for i, (sequence_id, seq_tokens) in enumerate(zip(active, tokens)):

        if sequence_id is None: # active slot in batch not being used, ignore
          continue
        
        # because model is not trained, we wont get EOS token, so we need to add it manually
        active_seqlens[i] += 1 # increase active sequence length
        if active_seqlens[i] == random_answer_seqlens[sequence_id]:
          seq_tokens[-len(eos_tokens):] = eos_tokens # add EOS token to the sequence
          
        # if completed, load next prompt in the active slot
        if is_completed(seq_tokens):
          generated_texts[-1][sequence_id] = decode_fn(seq_tokens.tolist())
          if next_to_be_processed < n_sequences:
            active[i] = next_to_be_processed # next sequence id to be processed
            next_prompt = prompts[next_to_be_processed]
            tokens[i][-seqlen:].fill_(0)  # Reset the whole context
            tokens[i][-len(next_prompt):] = next_prompt # load next prompt
            active_seqlens[i] = 0 # reset active sequence length
            next_to_be_processed+=1
            # print(f"Completed sequence {sequence_id} with length {len(seq_tokens)} in slot {i}. Loaded next prompt {active[i]}")
          else:
            active[i] = None # slot is not being used anymore   
          completed += 1
  benchmark_end("continuous batching", start_time)



  #################################################################################################
  #### SPECULATIVE SAMPLING usng distilled model as draft and original model as target         ####
  #################################################################################################

  look_ahead_T = 5
  generated_texts.append([None]*n_sequences) # list to store generated texts 
  start_time = benchmark_begin()

  # Same as GPTlite, but use draft model to generate the look ahead tokens
  dist_draft_p = torch.zeros(look_ahead_T, batch_size, vocab_size, dtype=torch.float32, device=device)

  with torch.inference_mode():
    for batch_start in range(0, n_sequences, batch_size):  # the 'while n<T do' in algorithm 2 in paper
      completed = set()
      while len(completed) < batch_size:
        # Step 1: generate look ahead tokens with draft model
        tokens = torch.stack(prompts[batch_start:batch_start+batch_size], dim=0)  # batch of sequences 

        for t in range(look_ahead_T):  # the first 'for t=1:K do' in algorithm 2 in paper
          tokens_in_context = tokens[:, -seqlen:].to(device)  # Use only the last seqlen tokens
          logits_draft = model_distilled(tokens_in_context)
          logits_draft = logits_draft[:, -1]  # Logits for the last position
          dist_draft_p[t] = F.softmax(logits_draft, dim=-1)  # Store the distribution for the current token

        # Extract speculated tokens (last look_ahead_T tokens) from draft model
        tokens_draft = torch.argmax(dist_draft_p, dim=-1) # or polynomial sampling
        tokens = torch.cat([tokens, tokens_draft], dim=-1)  # append all speculated tokens to the final result

        # Step 2: Perform Main model predictions for the same tokens
        tokens_in_context = tokens[:, -seqlen:].to(device)
        logits_target = model(tokens_in_context)
        # VERY IMPORTANT: compared to before, we use ALL logits, not just the logits of the last token!!!
        dist_target_q = F.softmax(logits_target[:, -look_ahead_T:], dim=-1)
        # tokens_target = torch.argmax(dist_target_q, dim=-1)  # Get the tokens from the target model

        for t in range(look_ahead_T):  # the second 'for t=1:K do' in algorithm 2 in paper
          # sample r from an uniform distribution [0, 1]
          r = torch.rand(batch_size, device=device)
          prob_q = dist_target_q[:, t, tokens_draft[:, t]]  # probability of the token sampled by draft model
          prob_p = dist_draft_p[:, t, tokens_draft[:, t]]  # probability of the token sampled by target model
          if not (r <= min(1, prob_q/prob_p)).all():  # accept the token if all sequences agree
            break # t is the index of the first token rejected

        tokens = tokens[:, :-look_ahead_T+t]  # remove the rejected tokens from the batch

        # Step 3: Finalize remaining tokens with target model
        if t<look_ahead_T: # handle rejected tokens
          # sample from (dist_target_q - dist_draft_p)_+
          q_minus_p = dist_target_q[t:] - dist_draft_p[t:]
          plus_fn = lambda fx: torch.clamp(fx, min=0) / torch.clamp(fx.sum(dim=-1, keepdim=True), min=1e-10)
          new_tokens = torch.argmax(plus_fn(q_minus_p), dim=-1)  # sample from the positive part of the difference
          tokens = torch.cat([tokens, new_tokens], dim=-1)  # append all accepted tokens to the final result
          
        for i, seq_tokens in enumerate(tokens):

          # because model is not trained, we wont get EOS token, so we need to add it manually
          if len(seq_tokens) >= random_answer_seqlens[batch_start+i] and i not in completed:
            tokens[i][-len(eos_tokens):] = eos_tokens # add EOS token to the sequence
            
          # check if sequence was completed, and increase completed counter if that's the case
          if is_completed(seq_tokens):
            # print(f"Completed sequence {sequence_id+i} with length {len(seq_tokens)} in slot {i}.")	
            completed.add(i)
            generated_texts[-1][batch_start+i] = decode_fn(seq_tokens.tolist())

  benchmark_end("GPTlite (speculative sampling)", start_time)



  # final sanity check: make sure all generated texts match
  for sequence_id, text in enumerate(generated_texts[0]):
    for j in range(1, len(generated_texts)):
      eos_offset = len(text) if text.find('\n') == -1 else text.find('\n')
      assert text[:eos_offset]==generated_texts[j][sequence_id][:eos_offset], f"Output mismatch at index {sequence_id}!"
