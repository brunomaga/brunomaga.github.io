import torch
from inference_utils import load_model, sample_prompts, Timer  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache


def generate(model, prompt, n_tokens, seqlen):
  """ Greedy generation without a cache: every step processes the last seqlen tokens """
  tokens = prompt
  for _ in range(n_tokens):
    logits = model(tokens[:, -seqlen:])
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_token], dim=-1)
  return tokens


def generate_kvcache(model, prompt, n_tokens, seqlen, kv_cache=None):
  """ Greedy generation with a KV cache: the prompt is processed once (prefill), on top of
      kv_cache if given, then every decode step processes only the newest token """
  logits, kv_cache = model(prompt, kv_cache=kv_cache, max_seqlen=seqlen)
  tokens = prompt
  for i in range(n_tokens):
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_token], dim=-1)
    if i < n_tokens - 1:
      logits, kv_cache = model(next_token, kv_cache=kv_cache, max_seqlen=seqlen)
  return tokens


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, decode_fn = get_tiny_shakespeare_data()
  seqlen = get_gptlite_model_parameters()[6]
  model = load_model(GPTlite, vocab_size)
  model_kvcache = load_model(GPTlite_KVCache, vocab_size) # same checkpoint, so both models share the same weights

  # a batch of prompts, taken from random positions of the validation data
  batch_size, prompt_len = 8, 16
  prompt = sample_prompts(valid_data, batch_size, prompt_len)

  # generate up to twice the context window: both models must match while the sequence
  # fits in the context window (seqlen tokens), and may diverge afterwards
  n_tokens = 2 * seqlen - prompt_len
  outputs = {}
  with torch.inference_mode():
    for name, generate_fn, model_obj in (
      ("GPTlite", generate, model),
      ("GPTlite_KVCache", generate_kvcache, model_kvcache),
    ):
      with Timer() as timer:
        outputs[name] = generate_fn(model_obj, prompt, n_tokens, seqlen)
      print(f"{name}: {timer.elapsed / n_tokens * 1000:.2f} ms per token")

  mismatches = (outputs["GPTlite"] != outputs["GPTlite_KVCache"]).any(dim=0).nonzero()
  n_matching = mismatches[0].item() if len(mismatches) > 0 else outputs["GPTlite"].size(1)
  print(f"Outputs match for the first {n_matching} tokens (context window: {seqlen} tokens)")
  print("Generated text:", decode_fn(outputs["GPTlite_KVCache"][0].tolist()))

