import copy
import os
import torch
from inference_utils import device, load_model, sample_prompts, validation_loss, Timer, \
  GPTLITE_DISTILLED_CKPT_PATH, GPTLITE_GQA_CKPT_PATH  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters, get_gptlite_distilled_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from gptlite_gqa import GPTlite_GQA
from main_kvcache import generate, generate_kvcache
from main_flash_attention import use_sdpa
from main_cuda_graphs import GPTlite_StaticCache, generate_static
from main_quantization import quantize
from main_speculative_decoding import generate_speculative, DraftModel
from main_prompt_lookup import prompt_lookup


def benchmark(generate_fn, prompt, n_tokens):
  """ Time to first token (generating a single token), time per output token, throughput and peak GPU memory """
  generate_fn(prompt, n_tokens)  # warm-up, which also compiles the compiled models
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
  with Timer() as first:
    generate_fn(prompt, 1)
  with Timer() as total:
    generate_fn(prompt, n_tokens)
  memory = f"{torch.cuda.max_memory_allocated() / 2**20:.1f}" if torch.cuda.is_available() else "-"
  time_per_token = (total.elapsed - first.elapsed) / (n_tokens - 1)
  return first.elapsed, time_per_token, prompt.numel() // prompt.size(1) * n_tokens / total.elapsed, memory


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, _ = get_tiny_shakespeare_data()
  n_heads, d_head = get_gptlite_model_parameters()[2:4]
  seqlen = min(get_gptlite_model_parameters()[6], get_gptlite_distilled_model_parameters()[6])
  prompt_len, n_drafts = seqlen // 4, 4
  n_tokens = seqlen - prompt_len  # speculative decoding and the static cache need the sequence to fit in the context window
  prompt = sample_prompts(valid_data, 1, prompt_len)  # a single sequence, as in latency-sensitive serving

  model = load_model(GPTlite, vocab_size)
  model_kvcache = load_model(GPTlite_KVCache, vocab_size)
  model_distilled = load_model(GPTlite_KVCache, vocab_size, ckpt_path=GPTLITE_DISTILLED_CKPT_PATH, distilled=True)
  model_sdpa = use_sdpa(copy.deepcopy(model))
  model_static = GPTlite_StaticCache(copy.deepcopy(model), batch_size=1, max_seqlen=seqlen)
  decode_compiled = torch.compile(model_static, mode="reduce-overhead" if torch.cuda.is_available() else "default", fullgraph=True)
  model_int8 = quantize(copy.deepcopy(model_kvcache), bits=8)
  model_int4 = quantize(copy.deepcopy(model_kvcache), bits=4)

  # name: (generation function, model used to compute the validation loss)
  methods = {
    "No KV cache": (lambda p, n: generate(model, p, n, seqlen), model),
    "No KV cache, fused attention (SDPA)": (lambda p, n: generate(model_sdpa, p, n, seqlen), model_sdpa),
    "KV cache": (lambda p, n: generate_kvcache(model_kvcache, p, n, seqlen), model_kvcache),
    "Static KV cache, compiled": (lambda p, n: generate_static(model_static, p, n, decode_compiled), model),
    "KV cache, INT8 weights": (lambda p, n: generate_kvcache(model_int8, p, n, seqlen), model_int8),
    "KV cache, INT4 weights": (lambda p, n: generate_kvcache(model_int4, p, n, seqlen), model_int4),
    "KV cache, distilled model": (lambda p, n: generate_kvcache(model_distilled, p, n, seqlen), model_distilled),
    "Speculative decoding (distilled draft)": (lambda p, n: generate_speculative(model_kvcache, p, n, DraftModel(model_distilled), n_drafts)[0], model_kvcache),
    "Prompt lookup decoding": (lambda p, n: generate_speculative(model_kvcache, p, n, prompt_lookup, n_drafts)[0], model_kvcache),
  }
  if os.path.exists(GPTLITE_GQA_CKPT_PATH):  # created by main_gqa.py
    n_groups = torch.load(GPTLITE_GQA_CKPT_PATH, map_location=device)['blocks.0.mha.key_proj.weight'].size(0) // d_head
    model_gqa = load_model(GPTlite_GQA, vocab_size, ckpt_path=GPTLITE_GQA_CKPT_PATH, n_groups=n_groups)
    methods[f"KV cache, GQA with {n_groups} groups"] = (lambda p, n: generate_kvcache(model_gqa, p, n, seqlen), model_gqa)

  print(f"{'Method':40s} {'TTFT (ms)':>10s} {'TPOT (ms)':>10s} {'tokens/s':>10s} {'memory (MB)':>12s} {'valid loss':>11s}")
  with torch.inference_mode():
    for name, (generate_fn, model_obj) in methods.items():
      ttft, tpot, throughput, memory = benchmark(generate_fn, prompt, n_tokens)
      loss = validation_loss(model_obj, valid_data)
      print(f"{name:40s} {ttft*1000:10.2f} {tpot*1000:10.2f} {throughput:10.1f} {memory:>12s} {loss:11.3f}")

