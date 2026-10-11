import torch
from inference_utils import device, load_model, sample_prompts, Timer  # also adds GPTlite to the path
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from main_kvcache import generate_kvcache


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, decode_fn = get_tiny_shakespeare_data()
  seqlen = get_gptlite_model_parameters()[6]
  model = load_model(GPTlite_KVCache, vocab_size)

  # a long prefix shared by all requests (e.g. a system prompt or a document), followed by a short
  # request-specific part (e.g. the user's question); the whole sequence fits in the context window
  prefix_len, request_len, n_tokens, n_requests = seqlen // 2, seqlen // 8, seqlen // 4, 16
  prefix = valid_data[:prefix_len].unsqueeze(0).to(device)    # [1, prefix_len]
  requests = sample_prompts(valid_data, n_requests, request_len)  # [n_requests, request_len]

  with torch.inference_mode():
    generate_kvcache(model, torch.cat([prefix, requests[:1]], dim=1), 1, seqlen)  # warm-up

    # without prefix caching: every request processes the shared prefix again
    with Timer() as timer:
      for request in requests:
        generate_kvcache(model, torch.cat([prefix, request[None]], dim=1), 1, seqlen)
    ttft_no_cache = timer.elapsed / n_requests
    outputs_no_cache = [generate_kvcache(model, torch.cat([prefix, request[None]], dim=1), n_tokens, seqlen)[:, prefix_len:]
                        for request in requests]

    # with prefix caching: the keys and values of the prefix are computed once and reused by every
    # request. Attention creates new tensors when it appends to the cache (torch.cat), so the prefix
    # cache is never modified and can be shared by all requests
    with Timer() as timer:
      _, prefix_cache = model(prefix)
    print(f"Prefill of the shared prefix (once): {timer.elapsed * 1000:.2f} ms")
    with Timer() as timer:
      for request in requests:
        generate_kvcache(model, request[None], 1, seqlen, kv_cache=prefix_cache)
    ttft_prefix_cache = timer.elapsed / n_requests
    outputs_prefix_cache = [generate_kvcache(model, request[None], n_tokens, seqlen, kv_cache=prefix_cache)
                            for request in requests]

  print(f"Time to first token without prefix caching: {ttft_no_cache * 1000:.2f} ms per request")
  print(f"Time to first token with prefix caching:    {ttft_prefix_cache * 1000:.2f} ms per request")
  same_outputs = all(torch.equal(a, b) for a, b in zip(outputs_no_cache, outputs_prefix_cache))
  print(f"Outputs with and without prefix caching are identical: {same_outputs}")
  print("Example:", decode_fn(prefix[0].tolist()) + "|" + decode_fn(outputs_prefix_cache[0][0].tolist()))


