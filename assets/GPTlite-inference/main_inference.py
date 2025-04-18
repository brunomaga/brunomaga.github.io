import os
import sys
import torch
import time
import torch.nn.functional as F
from gptlite_kvcache import GPTlite_KVCache
from gptlite_gqa import GPTlite_GQA  # Grouped Query Attention
import numpy as np


#use the default GPTlite model and utils from the post GPTlite
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPTlite'))
from gptlite import GPTlite
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters, get_gptlite_distilled_model_parameters
GPTLITE_CKPT_PATH = os.path.join(current_dir, '..', 'GPTlite', 'gptlite.pth')
GPTLITE_MQA_CKPT_PATH = os.path.join(current_dir, '..', 'GPTlite', 'gptlite_mqa.pth')
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


def load_model_ckpt(path, model, model_name, device):
  """ Load state dict into model """

  if os.path.exists(path):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded model from {path} into {model_name}")
  else:
    print(f"Couldnt load model from {path} into {model_name}")
  return model


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, encode_fn, decode_fn = get_tiny_shakespeare_data()

  # inference / sequence generation settings
  batch_size = 4 # number of sequences to generate in parallel
  n_sequences = 20 # total number of sequences to generate
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  breakline = 0 # beginning of string is a dummy prompt of token 0 which is \n
  prompts = [ torch.full((1,), breakline, dtype=torch.long, device=device) for i in range(n_sequences) ] # input prompts
  generated_texts = []

 
  # Load models and states if available

  torch.manual_seed(42) # reset random seed before model initialization, for reproducibility
  n_layers, d_model, n_heads, d_head, _, _, seqlen, dropout_p = get_gptlite_model_parameters()
  model = GPTlite(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device).eval()
  model = load_model_ckpt(GPTLITE_CKPT_PATH, model, "GPTlite", device)

  torch.manual_seed(42) # reset random seed before model initialization, for reproducibility
  model_kvcache = GPTlite_KVCache(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen).to(device).eval()
  model_kvcache = load_model_ckpt(GPTLITE_CKPT_PATH, model_kvcache, "GPTlite_KVCache", device)

  torch.manual_seed(42) # reset random seed before model initialization, for reproducibility
  model_mqa = GPTlite_GQA(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, n_groups=1).to(device).eval()
  model_mqa = load_model_ckpt(GPTLITE_MQA_CKPT_PATH, model_mqa, "GPTlite_GQA", device)

  n_layers_d, d_model_d, n_heads_d, d_head_d, _, _, seqlen_d, dropout_p_d = get_gptlite_distilled_model_parameters()
  model_distilled = GPTlite(vocab_size, d_model_d, n_heads_d, d_head_d, n_layers_d, dropout_p_d, seqlen_d).to(device).eval()
  model_distilled = load_model_ckpt(GPTLITE_DISTILLED_CKPT_PATH, model_distilled, "GPTlite_distilled", device)
  # model_distilled = model # uncomment to force the speculative to always accept the drafted tokens

  # assume sentence to be completed when it output few breaklines (useful when all models are trained)
  # is_completed = lambda seq_toks, seq_id: (seq_toks==breakline).sum() == 50

  # assume sentence to be completed when response reached a random length (good for models that are not trained)
  is_completed = lambda seq_toks, seq_id: len(seq_toks) > seqlen*(seq_id%5+1)


  # ##################################################################################################
  # #### REGULAR BATCHED INFERENCE: parallelize requests of diff length over the batch dimension  ####
  # #### Variants: GPTlite, GPTlite with KV cache, and GPTlite distilled to a smaller model       ####
  # ##################################################################################################	


  for (model_obj, model_name) in (
    (model, "GPTlite"),
    (model_kvcache, "GPTlite_KVCache"),
    (model_mqa, "GPTlite_GQA"),
    (model_distilled, "GPTlite_distilled"),
  ):
    generated_texts.append([None]*n_sequences) # list to store generated texts
    start_time = benchmark_begin()

    kv_cache = None  # Initialize cache for GPTlite_KVCache
    with torch.inference_mode():
      for batch_start in range(0, n_sequences, batch_size):
        # length of sequence to generate is the length of the longest string to be generated in the batch
        tokens = torch.stack(prompts[batch_start:batch_start+batch_size], dim=0)  # batch of sequences 
        completed = set()
        while len(completed) < batch_size:
          if model_name in ["GPTlite", "GPTlite_GQA"] :
            tokens_in_context = tokens[:, -seqlen:].to(device)  # Use only the last seqlen tokens
            logits = model_obj(tokens_in_context)
          elif model_name == "GPTlite_KVCache":
            last_token = tokens[:, -1:].to(device)  # Pass only the latest token
            logits, kv_cache = model_obj(last_token, max_seqlen=seqlen, kv_cache=kv_cache)
          elif model_name in "GPTlite_distilled":
            tokens_in_context = tokens[:, -seqlen_d:].to(device)
            logits = model_obj(tokens_in_context)
          else:
            raise RuntimeError(f"Unknown model name: {model_name}")

          # for deterministic results use argmax of logits, instead of multinomial of the softmax of logits 
          next_token = torch.argmax(logits[:, -1] , dim=-1, keepdim=True) # logits of last predicted token
          tokens = torch.cat([tokens, next_token], dim=-1)

          for i, seq_tokens in enumerate(tokens):

            # check if sequence was completed, and increase completed counter if that's the case
            sequence_id = batch_start+i
            if is_completed(seq_tokens, sequence_id) and sequence_id not in completed:
              print(f"Completed sequence {sequence_id} with length {len(seq_tokens)} in slot {i}.")	
              completed.add(sequence_id)
              generated_texts[-1][sequence_id] = decode_fn(seq_tokens.tolist())
    benchmark_end(model_name, start_time)



  #################################################################################################
  #### INFERENCE WITH CONTINUOUS BATCHING: continuosly append new sequences to input as soon   ####
  #### as a sequence is completed, instead of waiting for the whole batch to be completed.     ####
  #################################################################################################

  tokens = torch.stack(prompts[:batch_size], dim=0) 
  active = list(range(batch_size)) # list of active sequences as tuples as sequence_id 
  generated_texts.append([None]*n_sequences) # list to store generated texts
  start_time = benchmark_begin()

  with torch.inference_mode():
    completed = set()
    next_to_be_processed = batch_size
    while len(completed) < n_sequences:
          
      tokens_in_context = tokens[:, -seqlen:].to(device)  # Use only the last seqlen tokens
      logits = model(tokens_in_context)
      # for deterministic results use argmax of logits, instead of multinomial of the softmax of logits 
      next_token = torch.argmax(logits[:, -1] , dim=-1, keepdim=True) # logits of last predicted token
      tokens = torch.cat([tokens, next_token], dim=-1)

      # Add new sequences if possible
      for i, (sequence_id, seq_tokens) in enumerate(zip(active, tokens)):

        if sequence_id is None: # active slot in batch not being used, ignore
          continue
        
        # if completed, load next prompt in the active slot
        if is_completed(seq_tokens, sequence_id) and sequence_id not in completed:
          generated_texts[-1][sequence_id] = decode_fn(seq_tokens.tolist())
          if next_to_be_processed < n_sequences:
            active[i] = next_to_be_processed # next sequence id to be processed
            next_prompt = prompts[next_to_be_processed]
            tokens[i][-seqlen:].fill_(0)  # Reset the whole context
            tokens[i][-len(next_prompt):] = next_prompt # load next prompt
            next_to_be_processed+=1
          else:
            active[i] = None # free slot: not being used anymore   
          print(f"Completed sequence {sequence_id} with length {len(seq_tokens)} in slot {i}. Loaded next prompt {active[i]}")
          completed.add(sequence_id)
  benchmark_end("continuous batching", start_time)



  #################################################################################################
  #### SPECULATIVE SAMPLING using distilled model as draft and original model as target        ####
  #################################################################################################

  draft_seqlen_K = 5
  generated_texts.append([None]*n_sequences) # list to store generated texts 
  start_time = benchmark_begin()

  # Same as GPTlite, but use draft model to generate the look ahead tokens
  dist_draft_p = torch.zeros(batch_size, draft_seqlen_K, vocab_size, dtype=torch.float32, device=device)

  with torch.inference_mode():
    logits_draft = torch.zeros(batch_size, draft_seqlen_K, vocab_size, dtype=torch.float32, device=device)
    for batch_start in range(0, n_sequences, batch_size):  # the 'while n<T do' in algorithm 2 in paper
      completed = set()
      tokens = torch.stack(prompts[batch_start:batch_start+batch_size], dim=0)  # batch of sequences 
      while len(completed) < batch_size:

        # Step 1: generate K tokens with draft model
        for t in range(draft_seqlen_K):  # the first 'for t=1:K do' in algorithm 2 in paper
          tokens_in_context = tokens[:, -seqlen_d:].to(device)  # Use only the last seqlen tokens
          logits_draft[:, t] = model_distilled(tokens_in_context)[:, -1] # pick last logits
          next_token = torch.argmax(logits_draft[:, t], dim=-1, keepdim=True)
          tokens = torch.cat([tokens, next_token], dim=-1)  # append all speculated tokens to the final result
        tokens_draft = tokens[:, -draft_seqlen_K:]  # tokens generated by draft model

        # Step 2: Perform target model predictions for the same tokens
        tokens_in_context = tokens[:, -seqlen-1:-1].to(device)
        logits_target = model(tokens_in_context)[:, -draft_seqlen_K:] # IMPORTANT: use ALL logits, not just the logits of last token!
        dist_target_q = F.softmax(logits_target, dim=-1)
        dist_draft_p = F.softmax(logits_draft, dim=-1)
        # tokens_target = torch.argmax(logits_target, dim=-1)  # tokens generated by target model

        # Note: if q(x) >= p(x) then accept token
        # otherwise, the paper's rejection sampling scheme accepts token with prob q(x)/p(x)
        # Therefore, combining both, we accept token when r < min( 1, q(x)/p(x) ), where r~Uniform[0, 1]
        # (Why? see Theorem 1: modified rejection sampling recovers the target distribution)
        for b in range(batch_size):

          # iterate over draft tokens and check for rejections
          for t in range(draft_seqlen_K):
            draft_token_picked = tokens[b, -draft_seqlen_K+t]  # tokens generated by draft model for each batch
            prob_q = dist_target_q[b][t][draft_token_picked]
            prob_p = dist_draft_p[b][t][draft_token_picked]
            r = np.random.uniform() # sample r~Uniform[0, 1)
            if r < min( 1, prob_q/prob_p): # draft token accepted
              continue
            
            # token rejected, sample x_{n+t} from (dist_target_q - dist_draft_p)_+
            q_minus_p = dist_target_q[b, t] - dist_draft_p[b, t]
            plus_fn = lambda fx: torch.clamp(fx, min=0) / torch.clamp(fx.sum(dim=-1, keepdim=True), min=1e-10)
            next_token = torch.argmax(plus_fn(q_minus_p), dim=-1, keepdim=True)
            tokens[b, -draft_seqlen_K+t] = next_token  # append this token to the final result
            break # "and exit for loop"

          # Use target model to sample all remaining tokens
          for token_to_generate in range(t+1, draft_seqlen_K):
            # sample x_{n+t} from (dist_target_q - dist_draft_p)_+
            next_token_offset = -draft_seqlen_K+token_to_generate
            tokens_in_context = tokens[[b], -seqlen+next_token_offset : next_token_offset].to(device)
            logits = model(tokens_in_context)
            next_token = torch.argmax(logits[:, -1] , dim=-1, keepdim=True) # logits of last predicted token
            tokens[b, next_token_offset] = next_token
          
        for i, seq_tokens in enumerate(tokens):
          # check if sequence was completed, and increase completed counter if that's the case
          sequence_id = batch_start+i
          if is_completed(seq_tokens, sequence_id) and sequence_id not in completed:
            print(f"Completed sequence {sequence_id} with length {len(seq_tokens)} in slot {i}.")	
            completed.add(sequence_id)
            generated_texts[-1][sequence_id] = decode_fn(seq_tokens.tolist())

  benchmark_end("speculative sampling", start_time)



  # final sanity check: make sure all generated texts match. Will fail for when using small models
  for sequence_id, text in enumerate(generated_texts[0]):
    for j in range(1, len(generated_texts)):
      eos_offset = len(text) if text.find('\n') == -1 else text.find('\n')
      assert text[:eos_offset]==generated_texts[j][sequence_id][:eos_offset], f"Output mismatch at index {sequence_id}!"
