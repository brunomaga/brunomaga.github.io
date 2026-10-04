""" Helpers shared by the inference examples: paths, model loading, timing and evaluation """
import os
import sys
import time
import torch
import torch.nn.functional as F

# use the default GPTlite model and utils from the post GPTlite
current_dir = os.path.dirname(os.path.realpath(__file__))
GPTLITE_DIR = os.path.join(current_dir, '..', 'GPTlite')
sys.path.insert(0, GPTLITE_DIR)
from utils import get_batch, get_gptlite_model_parameters, get_gptlite_distilled_model_parameters

GPTLITE_CKPT_PATH = os.path.join(GPTLITE_DIR, 'gptlite.pth')
GPTLITE_DISTILLED_CKPT_PATH = os.path.join(GPTLITE_DIR, 'gptlite_distilled.pth')
GPTLITE_GQA_CKPT_PATH = os.path.join(GPTLITE_DIR, 'gptlite_gqa.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_class, vocab_size, ckpt_path=GPTLITE_CKPT_PATH, distilled=False, **kwargs):
  """ Creates a GPTlite-like model in evaluation mode, and loads its weights if the checkpoint exists.
      Extra keyword arguments are passed to the model constructor (e.g. n_groups for GQA). """
  params = get_gptlite_distilled_model_parameters() if distilled else get_gptlite_model_parameters()
  n_layers, d_model, n_heads, d_head, _, _, seqlen, dropout_p = params
  model = model_class(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, **kwargs).to(device).eval()
  if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded {model_class.__name__} from {ckpt_path}")
  else:
    print(f"Couldn't find {ckpt_path}: {model_class.__name__} has random weights")
  return model


def get_logits(output):
  """ Models with a KV cache return (logits, cache), the others return only the logits """
  return output[0] if isinstance(output, tuple) else output


class Timer:
  """ Context manager that measures the elapsed time in seconds, waiting for the GPU to finish """

  def __enter__(self):
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    self.start = time.perf_counter()
    return self

  def __exit__(self, *args):
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    self.elapsed = time.perf_counter() - self.start


@torch.inference_mode()
def validation_loss(model, data, n_batches=20, batch_size=8, seqlen=None):
  """ Average cross entropy of the model on random batches of data (fixed seed, same batches for all models) """
  seqlen = seqlen or get_gptlite_model_parameters()[6]
  losses = []
  with torch.random.fork_rng():  # the random state is restored at the end
    torch.manual_seed(1234)
    for _ in range(n_batches):
      x, y = get_batch(data, batch_size=batch_size, seqlen=seqlen)
      logits = get_logits(model(x.to(device)))
      losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)).item())
  return sum(losses) / len(losses)


def sample_prompts(data, n_prompts, prompt_len):
  """ Batch of prompts taken from random positions of the data """
  offsets = torch.randint(len(data) - prompt_len, (n_prompts,))
  return torch.stack([data[o:o+prompt_len] for o in offsets]).to(device)


def model_size_mb(model):
  """ Memory taken by the parameters and buffers of a model, in MB """
  tensors = list(model.parameters()) + list(model.buffers())
  return sum(t.numel() * t.element_size() for t in tensors) / 2**20


def train_steps(model, data, n_steps, lr, batch_size=8, seqlen=None):
  """ Trains a model for a few steps with the cross entropy loss (used to uptrain converted models) """
  seqlen = seqlen or get_gptlite_model_parameters()[6]
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model.train()
  for _ in range(n_steps):
    x, y = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    logits = get_logits(model(x.to(device)))
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
  model.eval()
  return model

