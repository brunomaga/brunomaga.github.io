import torch
from inference_utils import device, load_model, Timer  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from kvcache import generate


def next_tokens(model, sequences, seqlen):
  """ Greedy next token of each sequence, given as a list of 1D tensors of different lengths. The batch
      is padded on the right: with causal attention, real tokens never attend to the padding that
      follows them, so each sequence reads its prediction at its own last position """
  windows = [seq[-seqlen:] for seq in sequences]  # the last seqlen tokens of each sequence
  lengths = torch.tensor([len(w) for w in windows], device=device)
  batch = torch.nn.utils.rnn.pad_sequence(windows, batch_first=True)  # [B, T], padded with token 0
  logits = model(batch)  # [B, T, vocab_size]
  last_logits = logits[torch.arange(len(windows), device=device), lengths - 1]  # [B, vocab_size]
  return torch.argmax(last_logits, dim=-1)  # [B]


def make_requests(data, n_requests, seqlen):
  """ Requests with prompts of different lengths, asking for different numbers of tokens """
  requests = []
  for _ in range(n_requests):
    prompt_len = torch.randint(seqlen // 8, seqlen // 2, (1,)).item()
    offset = torch.randint(len(data) - prompt_len, (1,)).item()
    n_tokens = torch.randint(seqlen // 4, 2 * seqlen, (1,)).item()
    requests.append((data[offset:offset+prompt_len].to(device), n_tokens))
  return requests


def static_batching(model, requests, batch_size, seqlen):
  """ Runs the requests in groups of batch_size, until the longest request of each group finishes """
  outputs, n_steps = [], 0
  for start in range(0, len(requests), batch_size):
    group = requests[start:start+batch_size]
    sequences = [prompt for prompt, _ in group]
    for _ in range(max(n_tokens for _, n_tokens in group)):
      tokens = next_tokens(model, sequences, seqlen)
      sequences = [torch.cat([seq, token[None]]) for seq, token in zip(sequences, tokens)]
      n_steps += 1
    # requests that finished early generated extra tokens, which are thrown away
    outputs += [seq[:len(prompt)+n_tokens] for seq, (prompt, n_tokens) in zip(sequences, group)]
  return outputs, n_steps


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

  with torch.inference_mode():
    # reference: each request generated alone (this also warms up the GPU before the timed runs)
    references = [generate(model, prompt[None], n_tokens, seqlen)[0] for prompt, n_tokens in requests]
    for name, batching in [("Static", static_batching), ("Continuous", continuous_batching)]:
      with Timer() as timer:
        outputs, n_steps = batching(model, requests, batch_size, seqlen)
      print(f"{name} batching: {timer.elapsed:.2f} seconds, {n_steps} steps, {n_generated / timer.elapsed:.1f} tokens/s")
      print(f"  slots generating useful tokens: {100 * n_generated / (n_steps * batch_size):.1f}%")
      print(f"  same outputs as generating each request alone: {all(torch.equal(o, r) for o, r in zip(outputs, references))}")
