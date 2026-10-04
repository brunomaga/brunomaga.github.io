import torch
from inference_utils import load_model, sample_prompts, Timer, GPTLITE_DISTILLED_CKPT_PATH  # also adds GPTlite to the path
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters, get_gptlite_distilled_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from main_kvcache import generate_kvcache


def truncate_cache(kv_cache, length):
  """ Keeps the first length tokens of the KV cache of every layer """
  return [tuple(t[:, :, :length] for t in layer_cache) for layer_cache in kv_cache]


def generate_speculative(model, prompt, n_tokens, propose, n_drafts):
  """ Greedy speculative decoding of a single sequence (prompt of shape [1, S]). At every iteration,
      propose(tokens, k) returns up to k draft tokens, and the target model checks all of them in a single
      forward pass. The cache of the target holds all the accepted tokens except the last one, which is
      processed together with the drafts. The whole sequence must fit in the context window. """
  assert prompt.size(1) + n_tokens <= model.seqlen, "the sequence must fit in the context window"
  tokens, kv_cache = prompt, None
  if prompt.size(1) > 1:
    _, kv_cache = model(prompt[:, :-1])
  stats = {"target forward passes": 0, "proposed drafts": 0, "accepted drafts": 0}
  while tokens.size(1) < prompt.size(1) + n_tokens:
    drafts = propose(tokens, min(n_drafts, model.seqlen - tokens.size(1)))  # [1, k]
    # target predictions after the last accepted token and after each draft: [1, k+1]
    logits, kv_cache = model(torch.cat([tokens[:, -1:], drafts], dim=1), kv_cache=kv_cache)
    predictions = torch.argmax(logits, dim=-1)
    n_accepted = 0  # accept the drafts while they match the predictions of the target
    while n_accepted < drafts.size(1) and drafts[0, n_accepted] == predictions[0, n_accepted]:
      n_accepted += 1
    # keep the accepted drafts, plus the target's own token at the first mismatch (or the bonus token, after
    # the last draft if all were accepted), and discard the cache entries of the rejected drafts
    tokens = torch.cat([tokens, drafts[:, :n_accepted], predictions[:, n_accepted:n_accepted+1]], dim=1)
    kv_cache = truncate_cache(kv_cache, tokens.size(1) - 1)
    stats["target forward passes"] += 1
    stats["proposed drafts"] += drafts.size(1)
    stats["accepted drafts"] += n_accepted
  return tokens[:, :prompt.size(1) + n_tokens], stats


class DraftModel:
  """ Proposes draft tokens with a small model and its own KV cache. On every call, it reuses the cache
      of the longest prefix of the accepted tokens it has already processed """

  def __init__(self, model):
    self.model = model
    self.kv_cache, self.processed = None, None  # cache and the tokens it holds

  def __call__(self, tokens, n_drafts):
    if n_drafts == 0:
      return tokens[:, :0]
    n_cached = 0
    if self.processed is not None:  # number of leading tokens processed before and still valid
      n = min(self.processed.size(1), tokens.size(1) - 1)
      mismatches = (self.processed[0, :n] != tokens[0, :n]).nonzero()
      n_cached = mismatches[0].item() if len(mismatches) > 0 else n
    kv_cache = truncate_cache(self.kv_cache, n_cached) if n_cached > 0 else None
    new_tokens, drafts = tokens[:, n_cached:], []
    for _ in range(n_drafts):  # greedy generation with the draft model
      logits, kv_cache = self.model(new_tokens, kv_cache=kv_cache)
      new_tokens = torch.argmax(logits[:, -1:], dim=-1)  # [1, 1]
      drafts.append(new_tokens)
    drafts = torch.cat(drafts, dim=1)
    self.kv_cache, self.processed = kv_cache, torch.cat([tokens, drafts[:, :-1]], dim=1)  # the last draft was not processed
    return drafts


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, decode_fn = get_tiny_shakespeare_data()
  seqlen = min(get_gptlite_model_parameters()[6], get_gptlite_distilled_model_parameters()[6])
  model = load_model(GPTlite_KVCache, vocab_size)
  draft_model = load_model(GPTlite_KVCache, vocab_size, ckpt_path=GPTLITE_DISTILLED_CKPT_PATH, distilled=True)
  n_drafts, n_prompts, prompt_len = 4, 8, seqlen // 4
  n_tokens = seqlen - prompt_len
  prompts = sample_prompts(valid_data, n_prompts, prompt_len)

  with torch.inference_mode():
    with Timer() as timer_greedy:  # reference: greedy decoding with the target model and a KV cache
      references = [generate_kvcache(model, prompt[None], n_tokens, seqlen) for prompt in prompts]
    with Timer() as timer:
      results = [generate_speculative(model, prompt[None], n_tokens, DraftModel(draft_model), n_drafts)
                 for prompt in prompts]

  stats = {key: sum(s[key] for _, s in results) for key in results[0][1]}
  print(f"Greedy decoding: {timer_greedy.elapsed / (n_prompts * n_tokens) * 1000:.2f} ms per token")
  print(f"Speculative decoding: {timer.elapsed / (n_prompts * n_tokens) * 1000:.2f} ms per token, "
        f"{100 * stats['accepted drafts'] / stats['proposed drafts']:.0f}% of the drafts accepted, "
        f"{n_prompts * n_tokens / stats['target forward passes']:.2f} tokens per forward pass of the target")
  print(f"Same outputs as greedy decoding: {all(torch.equal(r, t) for r, (t, _) in zip(references, results))}")
  print("Generated text:", decode_fn(results[0][0][0].tolist()))
