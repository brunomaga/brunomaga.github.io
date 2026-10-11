import torch
from inference_utils import device, load_model, sample_prompts, validation_loss, train_steps, Timer, GPTLITE_GQA_CKPT_PATH  # also adds GPTlite to the path
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from gptlite_kvcache import GPTlite_KVCache
from gptlite_gqa import GPTlite_GQA, convert_mha_to_gqa
from main_kvcache import generate_kvcache


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, train_data, valid_data, _, _ = get_tiny_shakespeare_data()
  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gptlite_model_parameters()
  uptrain_iters = 1000 # short uptraining after the conversion
  model_mha = load_model(GPTlite_KVCache, vocab_size)
  prompt = sample_prompts(valid_data, 8, seqlen // 4)
  n_tokens = seqlen - prompt.size(1)

  # n_heads groups is multi-head attention (the original model), a single group is multi-query attention
  for n_groups in sorted({n_heads, n_heads // 2, 1} - {0}, reverse=True):
    if n_groups == n_heads:
      model, loss_converted = model_mha, None
    else:
      model = GPTlite_GQA(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, n_groups).to(device).eval()
      model.load_state_dict(convert_mha_to_gqa(model_mha.state_dict(), n_heads, d_head, n_groups))
      loss_converted = validation_loss(model, valid_data)
      train_steps(model, train_data, uptrain_iters, lr, batch_size, seqlen)
    loss = validation_loss(model, valid_data)

    with torch.inference_mode():
      _, kv_cache = model(prompt)
      generate_kvcache(model, prompt, 1, seqlen)  # warm-up
      cache_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv_cache)
      with Timer() as timer:
        generate_kvcache(model, prompt, n_tokens, seqlen)

    loss_info = f"{loss:.3f}" if loss_converted is None else f"{loss_converted:.3f} after conversion, {loss:.3f} after uptraining"
    print(f"n_groups={n_groups}: KV cache {cache_bytes / prompt.numel():.0f} bytes per token, "
          f"{timer.elapsed / n_tokens * 1000:.2f} ms per token, validation loss {loss_info}")
    if n_groups == n_heads // 2:
      torch.save(model.state_dict(), GPTLITE_GQA_CKPT_PATH)
      print(f"Saved the GQA model to {GPTLITE_GQA_CKPT_PATH}")

