import copy
import torch
import torch.nn as nn
from inference_utils import device, load_model, validation_loss, model_size_mb  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_batch, get_tiny_shakespeare_data, get_gptlite_model_parameters
from main_distillation import distillation_loss


@torch.inference_mode()
def compute_importance(model, data, n_batches, batch_size, seqlen):
  """ Importance of every attention head (norm of its output) and every MLP neuron (absolute value of its
      ReLU output), summed over a calibration set """
  heads = [torch.zeros(block.mha.n_heads, device=device) for block in model.blocks]
  neurons = [torch.zeros(block.ffwd.net[0].out_features, device=device) for block in model.blocks]
  def head_hook(i, H, D):  # the input of the output projection concatenates the outputs of all heads
    def hook(module, inputs):
      heads[i] += inputs[0].reshape(-1, H, D).norm(dim=-1).sum(dim=0)
    return hook
  def neuron_hook(i):  # the output of the ReLU holds the activations of the MLP neurons
    def hook(module, inputs, output):
      neurons[i] += output.reshape(-1, output.size(-1)).abs().sum(dim=0)
    return hook
  hooks = []
  for i, block in enumerate(model.blocks):
    hooks.append(block.mha.out_proj.register_forward_pre_hook(head_hook(i, block.mha.n_heads, block.mha.d_head)))
    hooks.append(block.ffwd.net[1].register_forward_hook(neuron_hook(i)))
  for _ in range(n_batches):
    x, _ = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    model(x.to(device))
  for hook in hooks:
    hook.remove()
  return heads, neurons


def prune_linear(linear, rows=None, columns=None):
  """ New linear layer with only the given rows (outputs) and columns (inputs) of the weight matrix """
  weight, bias = linear.weight.data, linear.bias.data
  if rows is not None:
    weight, bias = weight[rows], bias[rows]
  if columns is not None:
    weight = weight[:, columns]
  pruned = nn.Linear(weight.size(1), weight.size(0), device=weight.device)
  pruned.weight.data, pruned.bias.data = weight.clone(), bias.clone()
  return pruned


def prune(model, heads, neurons, n_heads, n_neurons):
  """ Keeps the n_heads most important heads and the n_neurons most important MLP neurons of every block """
  for block, head_importance, neuron_importance in zip(model.blocks, heads, neurons):
    mha, D = block.mha, block.mha.d_head
    keep_heads = head_importance.topk(n_heads).indices.sort().values
    rows = (keep_heads[:, None] * D + torch.arange(D, device=device)).flatten()  # the d_head rows of each kept head
    mha.query_proj = prune_linear(mha.query_proj, rows=rows)
    mha.key_proj = prune_linear(mha.key_proj, rows=rows)
    mha.value_proj = prune_linear(mha.value_proj, rows=rows)
    mha.out_proj = prune_linear(mha.out_proj, columns=rows)
    mha.n_heads = n_heads
    keep_neurons = neuron_importance.topk(n_neurons).indices.sort().values
    block.ffwd.net[0] = prune_linear(block.ffwd.net[0], rows=keep_neurons)     # first MLP layer: one row per neuron
    block.ffwd.net[2] = prune_linear(block.ffwd.net[2], columns=keep_neurons)  # second MLP layer: one column per neuron
  return model


def distill(student, teacher, data, n_steps, lr, batch_size, seqlen, temperature=2):
  """ Trains the student to match the output distribution of the frozen teacher """
  optimizer = torch.optim.Adam(student.parameters(), lr=lr)
  student.train()
  for _ in range(n_steps):
    idx, _ = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    idx = idx.to(device)
    with torch.no_grad():
      logits_teacher = teacher(idx)
    loss = distillation_loss(student(idx), logits_teacher, temperature)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
  student.eval()
  return student


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, train_data, valid_data, _, _ = get_tiny_shakespeare_data()
  n_layers, d_model, n_heads, d_head, batch_size, lr, seqlen, dropout_p = get_gptlite_model_parameters()
  distill_iters = 5000 # distillation steps of each method
  calibration_batches = 20 # batches used to compute the importance of heads and neurons
  keep_ratios = [0.9, 0.8, 0.7, 0.6, 0.5] # iterative pruning, down to half of the heads and neurons
  teacher = load_model(GPTlite, vocab_size)
  n_neurons = teacher.blocks[0].ffwd.net[0].out_features
  target_size = lambda ratio: (max(1, round(ratio * n_heads)), max(1, round(ratio * n_neurons)))
  importance = lambda model: compute_importance(model, train_data, calibration_batches, batch_size, seqlen)
  models = {"Original model (teacher)": teacher}

  # one-shot: prune to the final size at once, then distill
  model = prune(copy.deepcopy(teacher), *importance(teacher), *target_size(keep_ratios[-1]))
  models["Pruned once, before distillation"] = copy.deepcopy(model)
  models["Pruned once, then distilled"] = distill(model, teacher, train_data, distill_iters, lr, batch_size, seqlen)

  # iterative: prune a little, distill, and repeat, with the same total number of distillation steps
  model = copy.deepcopy(teacher)
  for ratio in keep_ratios:
    prune(model, *importance(model), *target_size(ratio))
    distill(model, teacher, train_data, distill_iters // len(keep_ratios), lr, batch_size, seqlen)
  models["Pruned iteratively and distilled"] = model

  # baseline: the same small architecture, randomly initialized, then distilled
  model = copy.deepcopy(model)
  for module in model.modules():
    if hasattr(module, 'reset_parameters'):
      module.reset_parameters()
  models["Random init, then distilled"] = distill(model, teacher, train_data, distill_iters, lr, batch_size, seqlen)

  for name, model in models.items():
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{name}: {n_params:.2f}M parameters ({model_size_mb(model):.1f} MB), "
          f"validation loss {validation_loss(model, valid_data):.3f}")
