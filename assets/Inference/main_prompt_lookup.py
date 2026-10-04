import torch
from inference_utils import load_model, sample_prompts, Timer  # also adds GPTlite to the path
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from main_kvcache import generate_kvcache
from main_speculative_decoding import generate_speculative


def prompt_lookup(tokens, n_drafts, max_ngram=3):
  """ Drafts the tokens that followed the most recent earlier occurrence of the last n-gram of the sequence,
      trying the longest n-grams first. Returns no drafts if the n-gram never occurred before """
  sequence = tokens[0]
  for n in range(max_ngram, 0, -1):
    if sequence.size(0) <= n:
      continue
    ngram = sequence[-n:]
    windows = sequence[:-1].unfold(0, n, 1)  # all the earlier n-grams of the sequence
    matches = (windows == ngram).all(dim=1).nonzero()
    if len(matches) > 0:
      start = matches[-1].item() + n  # first token after the most recent match
      return sequence[start:start + n_drafts].unsqueeze(0)
  return tokens[:, :0]


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, decode_fn = get_tiny_shakespeare_data()
  seqlen = get_gptlite_model_parameters()[6]
  model = load_model(GPTlite_KVCache, vocab_size)
  n_drafts, n_prompts, prompt_len = 4, 8, seqlen // 2
  n_tokens = seqlen - prompt_len
  prompts = sample_prompts(valid_data, n_prompts, prompt_len)

  with torch.inference_mode():
    with Timer() as timer_greedy:  # reference: greedy decoding with a KV cache
      references = [generate_kvcache(model, prompt[None], n_tokens, seqlen) for prompt in prompts]
    with Timer() as timer:
      results = [generate_speculative(model, prompt[None], n_tokens, prompt_lookup, n_drafts) for prompt in prompts]

  stats = {key: sum(s[key] for _, s in results) for key in results[0][1]}
  print(f"Greedy decoding: {timer_greedy.elapsed / (n_prompts * n_tokens) * 1000:.2f} ms per token")
  print(f"Prompt lookup decoding: {timer.elapsed / (n_prompts * n_tokens) * 1000:.2f} ms per token, "
        f"{100 * stats['accepted drafts'] / max(stats['proposed drafts'], 1):.0f}% of {stats['proposed drafts']} drafts accepted, "
        f"{n_prompts * n_tokens / stats['target forward passes']:.2f} tokens per forward pass")
  print(f"Same outputs as greedy decoding: {all(torch.equal(r, t) for r, (t, _) in zip(references, results))}")
  print("Generated text:", decode_fn(results[0][0][0].tolist()))

