import torch
from inference_utils import load_model, Timer  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from main_static_batching import next_tokens, make_requests, static_batching


def continuous_batching(model, requests, batch_size, seqlen):
  """ Keeps batch_size slots busy: as soon as a request finishes, the next waiting request takes its slot """
  outputs, n_steps = [None] * len(requests), 0
  waiting = list(range(len(requests)))  # ids of the requests waiting for a slot
  running = []  # (request id, tokens so far) of the requests in the batch
  while waiting or running:
    while waiting and len(running) < batch_size:  # fill the free slots
      request_id = waiting.pop(0)
      running.append((request_id, requests[request_id][0]))
    tokens = next_tokens(model, [seq for _, seq in running], seqlen)
    running = [(request_id, torch.cat([seq, token[None]])) for (request_id, seq), token in zip(running, tokens)]
    n_steps += 1
    still_running = []
    for request_id, seq in running:  # finished requests leave the batch
      prompt, n_tokens = requests[request_id]
      if len(seq) == len(prompt) + n_tokens:
        outputs[request_id] = seq
      else:
        still_running.append((request_id, seq))
    running = still_running
  return outputs, n_steps


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, valid_data, _, _ = get_tiny_shakespeare_data()
  seqlen = get_gptlite_model_parameters()[6]
  model = load_model(GPTlite, vocab_size)
  batch_size, n_requests = 8, 32
  requests = make_requests(valid_data, n_requests, seqlen)
  n_generated = sum(n_tokens for _, n_tokens in requests)

  outputs = {}
  with torch.inference_mode():
    for name, batching_fn in (("Static batching", static_batching), ("Continuous batching", continuous_batching)):
      with Timer() as timer:
        outputs[name], n_steps = batching_fn(model, requests, batch_size, seqlen)
      print(f"{name}: {timer.elapsed:.2f} seconds, {n_steps} steps, {n_generated / timer.elapsed:.1f} tokens/s, "
            f"{100 * n_generated / (n_steps * batch_size):.1f}% of the slots generating useful tokens")

  same = all(torch.equal(a, b) for a, b in zip(outputs["Static batching"], outputs["Continuous batching"]))
  print(f"Same outputs with both methods: {same}")

