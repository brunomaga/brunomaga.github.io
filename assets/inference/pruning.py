import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from inference_utils import device, load_model, validation_loss, model_size_mb, Timer  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_batch, get_tiny_shakespeare_data, get_gptlite_model_parameters
from distillation import distillation_loss


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


def mask_rows(score, sparsity):
  """ Mask that keeps the (1 - sparsity) highest scores of every row """
  prune = score.topk(int(score.size(1) * sparsity), dim=1, largest=False).indices
  return torch.ones_like(score, dtype=torch.bool).scatter_(1, prune, False)


def mask_2_4(score):
  """ Mask that keeps the 2 highest scores in every group of 4 consecutive weights of a row (2:4 sparsity) """
  groups = score.reshape(score.size(0), -1, 4)  # [out_features, in_features/4, 4]
  keep = groups.topk(2, dim=-1).indices
  return torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, keep, True).view_as(score)


def to_semi_structured(model):
  """ Stores the 2:4 sparse weights of the linear layers of the transformer blocks in the compressed format of the
      sparse tensor cores. Needs an NVIDIA GPU of the Ampere generation or newer, and an FP16 or BF16 model """
  from torch.sparse import to_sparse_semi_structured
  for block in model.blocks:
    for module in block.modules():
      if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(to_sparse_semi_structured(module.weight), requires_grad=False)
  return model


@torch.inference_mode()
def wanda(model, data, n_batches, batch_size, seqlen, sparsity=0.5, semi_structured=False):
  """ Wanda: zeros the weights with the lowest |W_ij| * ||X_j||_2 of every row (unstructured), or of every group
      of 4 weights (2:4), in the linear layers of the transformer blocks. The blocks are pruned in order, so that
      the inputs of a block come from the blocks already pruned before it """
  for block in model.blocks:
    linears = [module for module in block.modules() if isinstance(module, nn.Linear)]
    sq_norms = {linear: torch.zeros(linear.in_features, device=device) for linear in linears}
    def hook(module, inputs):  # sum of squares of every input feature j over all calibration tokens
      sq_norms[module] += inputs[0].reshape(-1, module.in_features).pow(2).sum(dim=0)
    hooks = [linear.register_forward_pre_hook(hook) for linear in linears]
    with torch.random.fork_rng():  # the same calibration batches for every block
      torch.manual_seed(0)
      for _ in range(n_batches):
        x, _ = get_batch(data, batch_size=batch_size, seqlen=seqlen)
        model(x.to(device))
    for hook in hooks:
      hook.remove()
    for linear in linears:
      score = linear.weight.abs() * sq_norms[linear].sqrt()  # [out_features, in_features] * [in_features]
      linear.weight.mul_(mask_2_4(score) if semi_structured else mask_rows(score, sparsity))
  return model


def taylor_importance(model, data, n_batches, batch_size, seqlen):
  """ First-order Taylor importance of every attention head and MLP neuron (LLM-Pruner): the sum of
      |w * dL/dw| over the weights of the structure, with the gradient of the loss on a calibration set """
  model.zero_grad(set_to_none=True)
  for _ in range(n_batches):  # the gradients add up over the batches
    x, y = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    logits = model(x.to(device))
    F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1)).backward()
  heads, neurons = [], []
  with torch.no_grad():
    for block in model.blocks:
      mha, ffwd, H, D = block.mha, block.ffwd.net, block.mha.n_heads, block.mha.d_head
      taylor = lambda param: (param * param.grad).abs()
      # the rows (and biases) of the query, key and value projections, and the columns of the output projection
      qkv = sum(taylor(l.weight).view(H, D, -1).sum(dim=(1, 2)) + taylor(l.bias).view(H, D).sum(dim=1)
                for l in (mha.query_proj, mha.key_proj, mha.value_proj))
      heads.append(qkv + taylor(mha.out_proj.weight).view(-1, H, D).sum(dim=(0, 2)))
      # the row (and bias) of the first MLP layer, and the column of the second
      neurons.append(taylor(ffwd[0].weight).sum(dim=1) + taylor(ffwd[0].bias) + taylor(ffwd[2].weight).sum(dim=0))
  model.zero_grad(set_to_none=True)
  return heads, neurons


@torch.inference_mode()
def block_importance(model, data, n_batches, batch_size, seqlen):
  """ Block Influence of ShortGPT: 1 - cosine similarity between the input and the output of every block,
      averaged over the tokens of a calibration set """
  scores = torch.zeros(len(model.blocks), device=device)
  def block_hook(i):
    def hook(module, inputs, output):
      scores[i] += 1 - F.cosine_similarity(inputs[0], output, dim=-1).mean()
    return hook
  hooks = [block.register_forward_hook(block_hook(i)) for i, block in enumerate(model.blocks)]
  for _ in range(n_batches):
    x, _ = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    model(x.to(device))
  for hook in hooks:
    hook.remove()
  return scores / n_batches


def remove_blocks(model, scores, n_blocks):
  """ Removes the n_blocks blocks with the lowest scores """
  keep = scores.topk(len(model.blocks) - n_blocks).indices.sort().values
  model.blocks = nn.Sequential(*[model.blocks[i] for i in keep])
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

  # pruning without retraining: unstructured, 2:4, Taylor importance of heads and neurons, and whole blocks
  models["Wanda 50% unstructured"] = wanda(copy.deepcopy(teacher), train_data, calibration_batches, batch_size, seqlen)
  models["Wanda 2:4"] = wanda(copy.deepcopy(teacher), train_data, calibration_batches, batch_size, seqlen,
                              semi_structured=True)
  heads, neurons = taylor_importance(teacher, train_data, calibration_batches, batch_size, seqlen)
  models["Taylor importance, pruned once"] = prune(copy.deepcopy(teacher), heads, neurons, *target_size(keep_ratios[-1]))
  scores = block_importance(teacher, train_data, calibration_batches, batch_size, seqlen)
  models["ShortGPT, 2 blocks removed"] = remove_blocks(copy.deepcopy(teacher), scores, n_blocks=2)

  for name, model in models.items():
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_zeros = sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1) / 1e6  # zero weights
    print(f"{name}: {n_params:.2f}M parameters ({n_zeros:.2f}M zeros, {model_size_mb(model):.1f} MB), "
          f"validation loss {validation_loss(model, valid_data):.3f}")

  # speedup of the 2:4 sparse tensor cores: forward pass of a batch, dense vs compressed 2:4 weights, in FP16
  if torch.cuda.is_available():
    x = torch.randint(vocab_size, (batch_size, seqlen), device=device)
    dense = models["Wanda 2:4"].half()
    sparse = to_semi_structured(copy.deepcopy(dense))
    for name, model in (("dense", dense), ("2:4 sparse", sparse)):
      with torch.inference_mode():
        model(x)  # warm-up
        with Timer() as timer:
          for _ in range(10):
            model(x)
      print(f"Wanda 2:4, {name} weights in FP16: {timer.elapsed / 10 * 1000:.2f} ms per forward pass")
