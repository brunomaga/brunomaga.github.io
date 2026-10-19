---
layout: post
title:  "Pruning and quantization"
categories: ["Efficient Inference"]
tags: ["LLM", "pruning", "quantization", "FP8", "KV cache"]
---

The most effective way to make a model faster is to make it smaller. [Pruning](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)) removes parts of a large model, and is usually combined with [knowledge distillation](https://en.wikipedia.org/wiki/Knowledge_distillation), which trains the smaller model to behave like the original one. [Quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) keeps all the parts of the model, but stores its numbers with fewer bits.

This post covers which parts of a transformer can be pruned, how to choose them and how distillation recovers the lost accuracy. It then presents the number formats and the quantization of the weights, the activations and the KV cache, and implements the main methods on [GPTlite]({{ site.baseurl }}{% post_url 2023-02-28-gptlite %}). As before, all the code is in the [repository of this post](https://github.com/{{ site.repository }}/tree/master/assets/inference).

## Pruning

Pruning removes parts of a trained model. The idea is old: the 1989 paper [Optimal Brain Damage](https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) already chose the weights to remove with the second derivatives of the loss, which worked much better than removing the smallest weights. Pruning remains a hard problem, for three reasons: one must decide *what* to remove without trying every option; the parts of a network are coupled, so removing one forces changes elsewhere; and removing anything hurts accuracy, which must be recovered. A fourth difficulty is turning the removal into actual speed.

### What can be removed

**Unstructured pruning** removes individual weights by setting them to zero. **Structured pruning** removes whole structures, which shrinks the weight matrices. Unstructured pruning can choose any weight, so it usually loses less accuracy for the same number of removed weights, but the matrices keep their shapes. In GPTlite, a linear layer `nn.Linear(d_in, d_out)` stores a weight matrix of shape `d_out × d_in`, so:

- removing **attention head** $$h$$ removes its $$d_\text{head}$$ rows from the query, key and value projections, and the corresponding $$d_\text{head}$$ columns of the output projection;
- removing **MLP neuron** $$j$$ removes row $$j$$ (and bias $$j$$) of the first MLP layer, and column $$j$$ of the second;
- removing **embedding channel** $$c$$ removes column $$c$$ of the token and position embeddings, entry $$c$$ of every LayerNorm, column $$c$$ of every matrix that reads from the residual stream (query, key and value projections, first MLP layer and final output layer), and row $$c$$ (and bias $$c$$) of every matrix that writes to it (attention output projection and second MLP layer). Every layer changes, which makes this the hardest and most impactful dimension;
- removing a **transformer block** deletes it entirely; thanks to the [residual connections](https://en.wikipedia.org/wiki/Residual_neural_network), the shapes of the other blocks stay valid.

The picture below shows the rows and columns that each structure removes, for an embedding size $$E$$ and an MLP of $$d_\text{ff}$$ neurons:

{: .picture}
<img width="97.4%" height="97.4%" src="{{ site.assets }}/inference/imgs/pruning_structures.svg"/>

All the structured methods below build a smaller linear layer from the rows and columns they keep:

```python
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
```

### Unstructured and 2:4 sparsity

A weight matrix with 50% zeros scattered at random is still multiplied as a dense matrix by GPUs, so it saves no time, and saves memory only with a [sparse](https://en.wikipedia.org/wiki/Sparse_matrix) storage format. The exception is NVIDIA's **2:4 semi-structured sparsity** ([Mishra et al.](https://arxiv.org/abs/2104.08378)): in every group of 4 consecutive weights, 2 are zero, and the sparse tensor cores of Ampere and newer GPUs skip them, doubling the peak throughput of those matrix multiplications. The end-to-end speedups are smaller, since attention, memory traffic and the other operations are not accelerated.

A 2:4 sparse matrix is stored as its non-zero values plus a 2-bit index per value, the position of that value in its group. The following picture illustrates both patterns on the same weights, and this compressed format:

{: .picture}
<img width="77%" height="77%" src="{{ site.assets }}/inference/imgs/sparsity_2_4.svg"/>

{: .caption}
**Unstructured 50%:** zeros anywhere, which a dense matmul does not skip. **2:4 semi-structured:** 2 zeros in every group of 4 consecutive weights. **Values** and **2-bit indices:** half the values, and the position of each one in its group. Inspired by: [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378)

The following functions build the two masks from a score per weight, such as its magnitude. The unstructured mask keeps the highest scores of every row (the picture above keeps the highest of the whole matrix instead), and the 2:4 mask keeps the 2 highest scores of every group of 4. Once the weights follow the 2:4 pattern, PyTorch's `to_sparse_semi_structured` converts them to the compressed format, and the multiplications of those layers then run on the sparse tensor cores:

```python
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
```

### How to decide what to remove

Pruning methods estimate the *importance* of each weight or structure, usually on a small calibration dataset:

- *magnitude*: small weights matter less (simple, but crude);
- *activations*: a neuron, head or channel whose activations are small on average contributes little (used by Minitron, below);
- *weights times activations*: [Wanda](https://arxiv.org/abs/2306.11695) scores each weight by $$\lvert W_{ij} \rvert \cdot \lVert X_j \rVert_2$$, its magnitude times the norm of its input feature, and prunes without any retraining;
- *gradients*: the first-order [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series) $$\lvert w \cdot \partial L / \partial w \rvert$$ estimates how much the loss changes when $$w$$ is removed ([LLM-Pruner](https://arxiv.org/abs/2305.11627), which also groups coupled structures and removes them together);
- *second-order information*: [SparseGPT](https://arxiv.org/abs/2301.00774) prunes and updates the remaining weights to compensate, one column at a time, using the inverse [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) (the same idea as GPTQ, below). It prunes 175B-parameter models to 50-60% unstructured sparsity, or to 2:4 sparsity, in one shot;
- *layer redundancy*: [ShortGPT](https://arxiv.org/abs/2403.03853) removes the blocks whose output is most similar to their input (by [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)), since they change the hidden state the least;
- *learned masks*: [Sheared-LLaMA](https://arxiv.org/abs/2310.06694) learns which heads, neurons, channels and layers to keep to reach a target architecture.

In practice, the magnitude needs only the weights, while the activations, Wanda and ShortGPT need forward passes on the calibration set, and the gradients also need backward passes. The next sections implement Wanda, the Taylor importance and ShortGPT on GPTlite.

### Weights times activations: Wanda

Magnitude pruning ignores the inputs. A small weight that multiplies a large input feature can contribute more to the output than a large weight that multiplies a small one. Wanda therefore multiplies the magnitude of each weight by the norm of its input feature over the calibration tokens, and compares the weights of the same output row.

In the picture below, magnitude pruning removes the weights $$-0.2$$ and $$0.1$$, which multiply the largest inputs. Wanda keeps them, and removes $$0.5$$ and $$-0.6$$ instead:

{: .picture}
<img width="44.5%" height="44.5%" src="{{ site.assets }}/inference/imgs/wanda.svg"/>

{: .caption}
Each output row keeps its 3 highest scores (50%). The magnitude alone removes the weights of the two input features with the largest norms (6 and 8); the Wanda score keeps them. Inspired by: [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)

Wanda prunes much faster than SparseGPT, since it does not update the remaining weights, and reaches a similar accuracy at 50% unstructured sparsity. The following code prunes the linear layers of every block of GPTlite with Wanda, to unstructured or 2:4 sparsity. A forward pre-hook collects the sum of squares of each input feature. Like the original implementation, it prunes one block at a time, so the inputs of a block come from the blocks already pruned before it:

```python
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
```

### Gradients: LLM-Pruner

The Taylor expansion of the loss around the trained weights estimates the change in the loss when a weight is set to zero. The magnitude of its first-order term is $$\lvert w \cdot \partial L / \partial w \rvert$$. LLM-Pruner adds up this term over all the coupled weights of a structure. For a head, these are the rows of the query, key and value projections and the columns of the output projection, and for a neuron, the row of the first MLP layer and the column of the second.

The following code computes these importances with one backward pass per calibration batch. It returns them in the format of `compute_importance` below, so that `prune` can use either:

```python
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
```

### Layer redundancy: ShortGPT

ShortGPT measures how much each block changes its input. The **Block Influence** of a block is one minus the cosine similarity between its input and its output hidden states, averaged over the tokens of a calibration set. The blocks with the lowest influence are removed first. In deep models, these are often the later blocks, but not the last one, as in the following example with 6 blocks:

{: .picture}
<img width="33.3%" height="33.3%" src="{{ site.assets }}/inference/imgs/shortgpt.svg"/>

{: .caption}
The blocks whose output is most similar to their input change the hidden state the least, and are removed first.

The following code computes the Block Influence of every block of GPTlite with a forward hook, and removes the least influential ones:

```python
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
```

### Pruning and distillation: Minitron

NVIDIA's [Minitron](https://arxiv.org/abs/2407.14679) combines structured pruning with distillation:

1. compute the importance of heads, MLP neurons and embedding channels from their activations, and the importance of each layer from how much removing it hurts, on a small calibration set of about a thousand samples;
2. prune the model to the target architecture;
3. retrain the pruned model by distilling from the original one (the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) on the logits), using a few percent of the original training tokens;
4. repeat for smaller sizes.

An illustration of the recipe is displayed below:

{: .picture}
<img width="99.8%" height="99.8%" src="{{ site.assets }}/inference/imgs/minitron.svg"/>

{: .caption}
The importance of each head, neuron, embedding channel and layer is estimated on a small calibration set, the model is pruned to the target size, and the pruned model is retrained by distillation from the trained one. The steps repeat for each smaller size. Adapted from: [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)

Deriving 8B and 4B models from a 15B one this way required up to 40 times fewer training tokens than training them from scratch, and gave better accuracy. A [follow-up](https://arxiv.org/abs/2408.11796) on Llama 3.1 8B found that width pruning (heads, neurons and channels) preserves more accuracy, while depth pruning (whole blocks) gives larger speedups. The key insight is that a pruned teacher is a much better starting point for the student than a random initialization.

This recipe is applied to GPTlite by removing half of the attention heads and MLP neurons of every block, in two ways. The first prunes in one shot, followed by distillation; the second prunes iteratively, removing 10% at a time and distilling after each step, with the same total number of distillation steps. As a baseline, the teacher is distilled into the same small architecture from a random initialization.

The following code computes the importance of heads and neurons, and implements the structured pruning and the experiment. The `distill` function, in the repository, trains the pruned model with a distillation loss:

```python
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


# the experiment: one-shot and iterative pruning, followed by distillation from the original model (teacher)
keep_ratios = [0.9, 0.8, 0.7, 0.6, 0.5]  # iterative pruning, down to half of the heads and neurons
target_size = lambda ratio: (max(1, round(ratio * n_heads)), max(1, round(ratio * n_neurons)))
importance = lambda model: compute_importance(model, train_data, calibration_batches, batch_size, seqlen)

model = prune(copy.deepcopy(teacher), *importance(teacher), *target_size(keep_ratios[-1]))
distill(model, teacher, train_data, distill_iters, lr, batch_size, seqlen)

model = copy.deepcopy(teacher)
for ratio in keep_ratios:
  prune(model, *importance(model), *target_size(ratio))
  distill(model, teacher, train_data, distill_iters // len(keep_ratios), lr, batch_size, seqlen)
```

## Quantization

Quantization stores numbers with fewer bits. Weights take less memory, fewer bytes are read at every decode step (which is what limits decode), and low-precision tensor cores perform more operations per second. The cost is a loss of precision, which must stay small enough not to hurt the quality of the model.

### Number formats

A [floating point](https://en.wikipedia.org/wiki/Floating-point_arithmetic) number has a sign bit, exponent bits that set its *range*, and mantissa (fraction) bits that set its *precision*. The three most common formats are FP32, FP16 and BF16. BF16 (*brain floating point*) keeps the 8 exponent bits of FP32 and only 7 mantissa bits, so it covers the same range as FP32 with the memory and speed of a 16-bit format.

FP16 has more precision than BF16 but a much smaller range (5 exponent bits), so large values can overflow. This is why BF16 is the default format of LLMs today. The picture below compares the bits of the three formats:

{: .picture}
<img width="73.1%" height="73.1%" src="{{ site.assets }}/inference/imgs/floating_point_representation.svg"/>

{: .caption}
The bits of $$0.15625 = 1.01_2 \times 2^{-3}$$ in FP32, FP16 and BF16. BF16 keeps the 8 exponent bits of FP32, and so its range, with fewer fraction bits. Adapted from: [Bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)

Recent GPUs added smaller formats:

| Format | Bits | Sign / exponent / mantissa bits | Largest value | Typical use |
|---|---|---|---|---|
| FP32 | 32 | 1 / 8 / 23 | about 3.4 × 10³⁸ | reference precision |
| FP16 | 16 | 1 / 5 / 10 | 65,504 | weights and activations |
| BF16 | 16 | 1 / 8 / 7 | about 3.4 × 10³⁸ | default for LLM weights and activations |
| FP8 E4M3 | 8 | 1 / 4 / 3 | 448 | weights, activations and KV cache (Hopper, Ada and newer GPUs) |
| FP8 E5M2 | 8 | 1 / 5 / 2 | 57,344 | values that need more range than precision |
| FP4 E2M1 | 4 | 1 / 2 / 1 | 6 | weights and activations with micro-scaling (Blackwell GPUs) |
| INT8 | 8 | integers from -128 to 127 | 127 | weights, activations and KV cache |
| INT4 | 4 | integers from -8 to 7 | 7 | weights, with group-wise scales |

Four bits represent only 16 values: FP4 E2M1 represents $$\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$$. Such small formats only work with fine-grained scaling, described in the section on FP4 below.

### How quantization works

To quantize a tensor $$x$$ to $$b$$-bit integers, one chooses a scale $$s$$ (and optionally a zero-point $$z$$), then rounds and clips, as in the integer-only inference of [Jacob et al.](https://arxiv.org/abs/1712.05877):

$$
x_q = \text{clip}\left(\text{round}\left(\frac{x}{s}\right) + z, \; q_\text{min}, \; q_\text{max} \right), \quad\quad \hat{x} = s \, (x_q - z)
$$

where $$q_\text{min}$$ and $$q_\text{max}$$ are the smallest and the largest integers of the format, and $$\hat{x}$$ is the dequantized approximation of $$x$$. In **symmetric** (*absmax*) quantization, $$z = 0$$ and $$s = \max \lvert x \rvert / q_\text{max}$$, e.g., with $$q_\text{max} = 127$$ for INT8. **Asymmetric** (*min-max*) quantization maps the range $$[\min x, \max x]$$ onto the full integer range, which fits skewed distributions better. In practice, symmetric quantization is the usual choice for the weights, since a zero-point of 0 makes the computation cheaper, while asymmetric quantization suits skewed values, such as the outputs of a ReLU, which are never negative.

The picture below quantizes the same values to 4 bits both ways. Symmetric quantization wastes the levels on the side without values:

{: .picture}
<img width="54.1%" height="54.1%" src="{{ site.assets }}/inference/imgs/quantization_mapping.svg"/>

{: .caption}
**Symmetric (absmax):** $$s = \max \lvert x \rvert / 7$$ and $$z = 0$$, so the levels $$-7$$ to $$-3$$, for values below $$-1$$, are never used. **Asymmetric (min-max):** $$s = (\max x - \min x) / 15$$ and $$z = \text{round}(-\min x / s) = 4$$, so all 16 levels are used.

The following functions implement both. `quantize_tensor` returns the integers, the scales and the zero-points, with one scale for each slice of the tensor along `dim`. `fake_quantize` quantizes and dequantizes, returning the values that the low-precision format can represent. This **fake quantization** is how most of this section simulates formats that the GPU, or PyTorch, cannot multiply directly:

```python
def quantize_tensor(x, bits, dim=-1, symmetric=True):
  """ Integer quantization of x with one scale (and zero-point) for each slice of x along dim, i.e. the values
      along dim share a scale. Returns the integers q, the scales s and the zero-points z, so that x ≈ s * (q - z) """
  if symmetric:  # absmax: integers in [-2^(b-1), 2^(b-1) - 1], zero-point 0
    q_max = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(dim=dim, keepdim=True) / q_max).clamp(min=1e-8)
    zero = torch.zeros_like(scale)
    q = torch.clamp(torch.round(x / scale), -q_max - 1, q_max)
  else:  # min-max: integers in [0, 2^b - 1], the zero-point is the integer that represents 0
    q_max = 2 ** bits - 1
    x_min, x_max = x.amin(dim=dim, keepdim=True), x.amax(dim=dim, keepdim=True)
    scale = ((x_max - x_min) / q_max).clamp(min=1e-8)
    zero = torch.round(-x_min / scale)
    q = torch.clamp(torch.round(x / scale) + zero, 0, q_max)
  return q, scale, zero


def fake_quantize(x, bits, dim=-1, symmetric=True):
  """ x quantized and dequantized: the closest values that the integer format can represent """
  q, scale, zero = quantize_tensor(x, bits, dim, symmetric)
  return scale * (q - zero)


def fake_quantize_groups(W, bits, group_size):
  """ W quantized and dequantized with one scale per group of group_size consecutive weights of a row """
  return fake_quantize(W.unflatten(-1, (-1, group_size)), bits).flatten(-2)
```

The **granularity** of the scales matters as much as the number of bits. A single scale per tensor is cheap but fragile.

One scale per output channel (a row of the weight matrix), per *group* of, e.g., 128 consecutive weights, or per token for the activations, follows the data much more closely, at the cost of storing more scales. Group-wise scales are the standard for 4-bit weights. The following picture compares the three granularities on a weight matrix with one outlier weight:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/inference/imgs/quantization_granularity.svg"/>

{: .caption}
**One scale per tensor:** 1 scale, and the outlier stretches the scale of 47 other weights. **One scale per output channel (row):** 4 scales, and the outlier stretches the scale of 11 other weights. **One scale per group of 4 weights:** 12 scales, and the outlier stretches the scale of 3 other weights.

The main enemy is **outliers**: a single large value stretches the scale, and all the small values collapse onto a few quantization levels. Weights are well-behaved and easy to quantize. Activations are not: [LLM.int8()](https://arxiv.org/abs/2208.07339) showed that large transformers develop a few *outlier feature dimensions*, with values much larger than the rest, that break naive 8-bit quantization of the activations.

Quantization can be applied after training, often with a small calibration dataset (**post-training quantization**, PTQ), which is fast and is what this post does. It can also be simulated during training so that the model learns to tolerate it (**quantization-aware training**, QAT), which gives the best results at 4 bits and below. Finally, there are four things to quantize: the weights, the activations, the KV cache and the attention computation.

### Weight-only quantization

Storing the weights in 4 or 8 bits while computing in 16 bits (e.g., W4A16: 4-bit weights, 16-bit activations) speeds up decode almost in proportion to the bytes saved, because decode is memory-bound. As an example, 4-bit weights are read four times faster than BF16 ones. The kernel must dequantize the weights on the fly, in registers, right before the multiplication; dequantizing the whole matrix to memory first would cancel the gain. Prefill, which is compute-bound, barely benefits.

The picture below shows how INT4 weights are quantized, packed two per byte, and dequantized in the kernel:

{: .picture}
<img width="64.2%" height="64.2%" src="{{ site.assets }}/inference/imgs/int4_packing.svg"/>

Rounding each weight to the nearest level (**round-to-nearest**) works well at 8 bits, but loses too much accuracy at 4 bits. Two methods, GPTQ and AWQ (below), made 4-bit weights practical. Other common formats are NF4, from [QLoRA](https://arxiv.org/abs/2305.14314), whose levels are placed for normally distributed weights, and the GGUF formats of [llama.cpp](https://en.wikipedia.org/wiki/Llama.cpp), for CPUs and edge devices.

To see how it works, this post implements its own quantizer: symmetric INT8 quantization with one scale per output channel, and INT4 quantization with one scale per group of 64 weights, packing two 4-bit weights per byte. It quantizes the linear layers of the transformer blocks of GPTlite, and the memory, the time per token and the validation loss are compared against the FP32 and BF16 models.

With one scale per output channel, the INT8 layer can apply the scale after the matrix multiplication, since all the products that add up to one output share that scale, while the INT4 layer must dequantize the weights first. In plain PyTorch, both convert the weights back to floating point in memory before each multiplication, which saves memory but not time. Actual speedups require kernels that dequantize in registers, such as those of [torchao](https://github.com/pytorch/ao) or [Marlin](https://arxiv.org/abs/2408.11743), or a compiler that fuses the conversion into the multiplication.

The following code implements this weight-only INT8 and INT4 quantization of GPTlite:

```python
class QuantizedLinear(nn.Module):
  """ Linear layer with symmetric (absmax) weight-only quantization: INT8 with one scale per output channel,
      or INT4 with one scale per group of group_size consecutive weights of a row, packing two weights per byte.
      The scales are computed from the weights, unless given (e.g. by GPTQ). The weights are dequantized on the
      fly, and the computation runs in the precision of the input """

  def __init__(self, linear, bits, group_size=64, scale=None):
    super().__init__()
    assert bits in (4, 8), "only 8 and 4 bits are supported"
    W = linear.weight.data  # [out_features, in_features]
    self.bits, (self.out_features, self.in_features) = bits, W.shape
    self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.clone(), requires_grad=False)
    q_max = 2 ** (bits - 1) - 1  # 127 for INT8, 7 for INT4
    if bits == 8:
      scale = W.abs().amax(dim=1, keepdim=True) / q_max if scale is None else scale.view(-1, 1)  # [out_features, 1]
    else:
      self.group_size = group_size if self.in_features % group_size == 0 else self.in_features
      W = W.view(self.out_features, -1, self.group_size)   # [out_features, n_groups, group_size]
      if scale is None:
        scale = W.abs().amax(dim=2, keepdim=True) / q_max  # [out_features, n_groups, 1]
    scale = scale.clamp(min=1e-8)
    q = torch.clamp(torch.round(W / scale), -q_max - 1, q_max).to(torch.int8)
    if bits == 4:  # shift the values from [-8, 7] to [0, 15] and pack two of them per byte
      q = (q + 8).to(torch.uint8).view(self.out_features, -1)
      q = q[:, 0::2] | (q[:, 1::2] << 4)
    self.register_buffer('qweight', q)
    self.register_buffer('scale', scale)

  def dequantize(self):
    """ Weight matrix in the precision of the scales """
    if self.bits == 8:
      return self.qweight.to(self.scale.dtype) * self.scale
    low, high = (self.qweight & 0x0F).to(torch.int8) - 8, (self.qweight >> 4).to(torch.int8) - 8
    q = torch.stack([low, high], dim=-1).view(self.out_features, -1, self.group_size)  # unpack and interleave
    return (q.to(self.scale.dtype) * self.scale).view(self.out_features, self.in_features)

  def forward(self, x):
    if self.bits == 8:
      # with one scale per output channel, the scale can be applied after the matrix multiplication
      out = F.linear(x, self.qweight.to(x.dtype)) * self.scale.view(-1).to(x.dtype)
      return out if self.bias is None else out + self.bias
    return F.linear(x, self.dequantize().to(x.dtype), self.bias)


def replace_linears(model, new_layer):
  """ Replaces every linear layer of the transformer blocks by new_layer(linear). The embeddings and the
      output layer stay in full precision """
  linears = [(parent, name, child) for block in model.blocks for parent in block.modules()
             for name, child in parent.named_children() if isinstance(child, nn.Linear)]
  for parent, name, child in linears:
    setattr(parent, name, new_layer(child))
  return model


def quantize(model, bits, group_size=64):
  """ Weight-only quantization of the linear layers of the transformer blocks, with round-to-nearest """
  return replace_linears(model, lambda linear: QuantizedLinear(linear, bits, group_size))
```

### GPTQ

[GPTQ](https://arxiv.org/abs/2210.17323) quantizes a weight matrix one column at a time and, after each column, updates the columns not yet quantized to compensate for its rounding error. The update uses second-order information: the inverse of the Hessian $$H = 2X^TX$$ of the layer's reconstruction error $$\lVert XW^T - X\hat{W}^T \rVert^2$$, computed from calibration inputs $$X$$ (one input per row), where $$\hat{W}$$ is the quantized weight matrix. Column $$i$$ is rounded to $$\hat{W}_{:,i}$$, and every later column $$j$$ moves by $$-(W_{:,i} - \hat{W}_{:,i}) \, U_{ij} / U_{ii}$$, with $$U$$ the upper [Cholesky factor](https://en.wikipedia.org/wiki/Cholesky_decomposition) of $$H^{-1}$$, the change that minimizes the increase of the reconstruction error.

GPTQ quantizes 175B-parameter models to 3-4 bits in about four GPU hours. One step of the algorithm is illustrated below:

{: .picture}
<img width="58.9%" height="58.9%" src="{{ site.assets }}/inference/imgs/gptq.svg"/>

{: .caption}
GPTQ quantizes the columns of $$W$$ one at a time, and spreads the rounding error of each column over the columns not yet quantized. Inspired by: [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

The following code implements GPTQ for one linear layer, and applies it to GPTlite one block at a time. The rows of the upper Cholesky factor of $$H^{-1}$$ hold the updates in the order the columns are quantized, as in the original implementation. The scale of each group comes from its weights after the updates of the columns before it. For simplicity, the code updates one column at a time; the original implementation processes blocks of 128 columns, which gives the same result much faster on a GPU:

```python
@torch.inference_mode()
def calibration_inputs(model, modules, data, n_batches, batch_size, seqlen):
  """ Inputs of the given modules on a calibration set, as [n_tokens, in_features] tensors """
  inputs = {module: [] for module in modules}
  def hook(module, args):
    inputs[module].append(args[0].reshape(-1, args[0].size(-1)))
  hooks = [module.register_forward_pre_hook(hook) for module in modules]
  with torch.random.fork_rng():  # the same calibration batches at every call
    torch.manual_seed(0)
    for _ in range(n_batches):
      x, _ = get_batch(data, batch_size=batch_size, seqlen=seqlen)
      model(x.to(device))
  for hook in hooks:
    hook.remove()
  return {module: torch.cat(xs) for module, xs in inputs.items()}


def gptq(linear, X, bits=4, group_size=64, damp=0.01):
  """ GPTQ: quantizes the weights of a linear layer one column (input feature) at a time, and updates the columns
      not yet quantized to compensate for the error, with the inverse of the Hessian H = 2 X^T X of the layer's
      reconstruction error, for the calibration inputs X: [n_tokens, in_features]. Sets the weights to their
      quantized values, and returns the scales [out_features, n_groups, 1] """
  W = linear.weight.data.float().clone()  # [out_features, in_features]
  n_in = W.size(1)
  group_size = n_in if bits == 8 else group_size  # INT8: one scale per output channel
  H = 2 * X.T.float() @ X.float()  # [in_features, in_features]
  H += damp * H.diagonal().mean() * torch.eye(n_in, device=W.device)  # dampening, for numerical stability
  # upper Cholesky factor of H^-1: row i holds the update of the columns after i once column i is quantized
  Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
  q_max = 2 ** (bits - 1) - 1
  scales = torch.zeros(W.size(0), n_in // group_size, 1, device=W.device)
  for i in range(n_in):
    g = i // group_size
    if i % group_size == 0:  # the scale of a group comes from its weights, as updated by the previous columns
      scales[:, g] = (W[:, i:i + group_size].abs().amax(dim=1, keepdim=True) / q_max).clamp(min=1e-8)
    q = torch.clamp(torch.round(W[:, i] / scales[:, g, 0]), -q_max - 1, q_max) * scales[:, g, 0]
    err = (W[:, i] - q) / Hinv[i, i]
    W[:, i:] -= err[:, None] * Hinv[i, i:][None, :]  # column i becomes q, and the next columns compensate
  linear.weight.data = W.to(linear.weight.dtype)
  return scales


def quantize_gptq(model, data, n_batches, batch_size, seqlen, bits=4, group_size=64):
  """ Quantizes the linear layers with GPTQ, one block at a time, so that the inputs of a block come from the
      blocks already quantized before it """
  for block in model.blocks:
    linears = [(parent, name, child) for parent in block.modules() for name, child in parent.named_children()
               if isinstance(child, nn.Linear)]
    inputs = calibration_inputs(model, [child for _, _, child in linears], data, n_batches, batch_size, seqlen)
    for parent, name, child in linears:
      scale = gptq(child, inputs[child], bits, group_size)
      setattr(parent, name, QuantizedLinear(child, bits, group_size, scale=scale))
  return model
```

### AWQ

[AWQ](https://arxiv.org/abs/2306.00978) (Activation-aware Weight Quantization) observes that a small fraction of the weights (about 1%) matters much more than the rest: those that multiply large activations. Instead of keeping them in high precision, it multiplies the salient input channels of the weights by a factor before quantization, so that they are rounded more accurately. The corresponding activations are divided by the same factor, which keeps the output of the layer and can be folded into the previous operation.

The factors are searched on calibration data. Unlike GPTQ, AWQ does not update the weights to fit the calibration data, so it is less likely to overfit the calibration set. The following picture illustrates the scaling of a salient channel:

{: .picture}
<img width="62.7%" height="62.7%" src="{{ site.assets }}/inference/imgs/awq.svg"/>

{: .caption}
$$s_j = \text{mean}\lvert X_j \rvert^\alpha$$, with $$\alpha$$ searched on calibration data. The weights of the salient channel become larger, so rounding loses less of them. The division by $$s$$ is folded into the operation before. Inspired by: [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

The following code implements AWQ for GPTlite. For each input of a block, it searches the exponent $$\alpha$$ of the factors $$s_j = \text{mean}\lvert X_j \rvert^\alpha$$ that minimizes the output error of the quantized linear layers that read it. The divisions are folded into the LayerNorms, into the value projection (attention is linear in the values), and into the first MLP layer (since $$\text{ReLU}(y / s) = \text{ReLU}(y) / s$$ for $$s > 0$$):

```python
def awq_scales(X, linears, bits, group_size, n_grid=20):
  """ AWQ: searches the scales s = mean|X_j|^alpha of the input channels that minimize the error of the outputs of
      the linears that read X, when their weights W * s are quantized and their input is X / s """
  act = X.abs().mean(dim=0)  # [in_features]
  outputs = [F.linear(X, linear.weight) for linear in linears]
  best_error, best_s = float('inf'), None
  for alpha in torch.linspace(0, 1, n_grid):
    s = act.pow(alpha).clamp(min=1e-4)
    s = s / (s.max() * s.min()).sqrt()  # center the scales around 1
    error = sum((F.linear(X / s, fake_quantize_groups(linear.weight * s, bits, group_size)) - out).pow(2).mean()
                for linear, out in zip(linears, outputs))
    if error < best_error:
      best_error, best_s = error, s
  return best_s


@torch.inference_mode()
def quantize_awq(model, data, n_batches, batch_size, seqlen, bits=4, group_size=64):
  """ AWQ: multiplies the input channels of every linear layer by the scales of awq_scales, divides the output
      of the previous operation by the same scales, then quantizes the weights with round-to-nearest """
  for block in model.blocks:
    mha, ffwd = block.mha, block.ffwd.net
    reads = [mha.query_proj, mha.out_proj, ffwd[0], ffwd[2]]  # the first linear layer to read each input
    inputs = calibration_inputs(model, reads, data, n_batches, batch_size, seqlen)
    # (operation before, linears after): the LayerNorms, the value projection (attention is linear in the values),
    # and the first MLP layer (ReLU(z / s) = ReLU(z) / s for s > 0)
    for before, linears, X in ((block.ln1, [mha.query_proj, mha.key_proj, mha.value_proj], inputs[mha.query_proj]),
                               (mha.value_proj, [mha.out_proj], inputs[mha.out_proj]),
                               (block.ln2, [ffwd[0]], inputs[ffwd[0]]),
                               (ffwd[0], [ffwd[2]], inputs[ffwd[2]])):
      s = awq_scales(X, linears, bits, group_size)
      if isinstance(before, nn.LayerNorm):
        before.weight.div_(s)
      else:  # a linear layer: divide its rows (outputs)
        before.weight.div_(s[:, None])
      before.bias.div_(s)
      for linear in linears:
        linear.weight.mul_(s)
  return quantize(model, bits, group_size)
```

### Weight and activation quantization

To make the matrix multiplication itself faster, both of its inputs must be in low precision, so that the GPU can use its INT8, FP8 or FP4 tensor cores, which are 2 to 4 times faster than the BF16 ones. This also speeds up the compute-bound prefill and large-batch decode. The difficulty is the outliers of the activations.

**FP8** weights and activations are the most common choice on Hopper and Blackwell GPUs. With per-channel or per-token scales, they are nearly lossless for most LLMs, since a floating point format rounds small and large values with about the same relative error. For INT8 and 4-bit formats, the methods below deal with the outliers.

The following function multiplies INT8 activations, with one scale per token, by INT8 weights, with one scale per output channel, and the `W8A8Linear` layer uses it. The integers are multiplied and accumulated in INT32, as in INT8 tensor cores. PyTorch multiplies integer tensors with `@` only on the CPU; on a GPU, `torch._int_mm` runs this multiplication on the INT8 tensor cores:

```python
def int8_matmul(x, W):
  """ x @ W^T with INT8 activations (one scale per token) and INT8 weights (one scale per output channel). The
      integers are multiplied and accumulated in INT32, as in INT8 tensor cores, then scaled back """
  xq, x_scale, _ = quantize_tensor(x, 8, dim=-1)  # [n_tokens, in_features], [n_tokens, 1]
  wq, w_scale, _ = quantize_tensor(W, 8, dim=-1)  # [out_features, in_features], [out_features, 1]
  out = xq.to(torch.int32) @ wq.to(torch.int32).T
  return out * x_scale * w_scale.T


class W8A8Linear(nn.Module):
  """ INT8 weights and activations, with the activations quantized on the fly """

  def __init__(self, linear):
    super().__init__()
    self.weight, self.bias = linear.weight, linear.bias

  def forward(self, x):
    out = int8_matmul(x.reshape(-1, x.size(-1)), self.weight) + self.bias
    return out.view(*x.shape[:-1], -1)
```

### LLM.int8()

**LLM.int8()** keeps the few outlier feature dimensions in 16 bits and runs everything else in INT8. A feature is an outlier when any of its values is larger in magnitude than a threshold, 6 in the paper. The outliers are rare, so more than 99.9% of the multiplications still run in INT8. The following diagram illustrates this *mixed-precision decomposition*, as the paper calls it:

{: .picture}
<img width="77.8%" height="77.8%" src="{{ site.assets }}/inference/imgs/llm_int8.svg"/>

{: .caption}
The feature columns of $$X$$ with outliers and the matching rows of $$W^T$$ are multiplied in FP16, the other features in INT8, and the two results are added. Adapted from: [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)

The following layer implements LLM.int8() with the `int8_matmul` function above:

```python
class LLMInt8Linear(nn.Module):
  """ LLM.int8(): the input features with a value above the threshold in any token (the outliers) are multiplied
      in the precision of the input, and all the others in INT8 """

  def __init__(self, linear, threshold=6.0):
    super().__init__()
    self.weight, self.bias, self.threshold = linear.weight, linear.bias, threshold

  def forward(self, x):
    x2 = x.reshape(-1, x.size(-1))  # [n_tokens, in_features]
    outliers = (x2.abs() > self.threshold).any(dim=0)  # [in_features]
    out = int8_matmul(x2[:, ~outliers], self.weight[:, ~outliers])
    out = out + x2[:, outliers] @ self.weight[:, outliers].T
    return (out + self.bias).view(*x.shape[:-1], -1)
```

### SmoothQuant

[SmoothQuant](https://arxiv.org/abs/2211.10438) moves the difficulty from the activations to the weights. Dividing the $$j$$-th channel of the activations by a factor $$s_j$$, and multiplying the $$j$$-th input channel of the weights by the same factor, does not change the output of the layer but tames the outliers. With $$s_j = \max \lvert X_j \rvert^\alpha / \max \lvert W_j \rvert^{1-\alpha}$$ and typically $$\alpha = 0.5$$, both the weights and the activations become easy to quantize to INT8 (W8A8), as illustrated below for a layer with one outlier channel:

{: .picture}
<img width="81.8%" height="81.8%" src="{{ site.assets }}/inference/imgs/smoothquant.svg"/>

{: .caption}
Dividing the activations of input channel $$j$$ by $$s_j$$, and multiplying the weights of that channel by $$s_j$$, keeps the output, and moves part of the range of the activation outliers into the weights, so that both are easy to quantize. Inspired by: [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)

Unlike LLM.int8(), SmoothQuant needs no special handling of the outliers at run time, since the factors are folded offline; its paper found it faster than FP16 in all the tested settings, while LLM.int8() was mostly slower. The following code smooths the inputs of the linear layers that follow each LayerNorm of GPTlite, and folds the division into the LayerNorm. The model then uses the `W8A8Linear` layers above:

```python
@torch.inference_mode()
def smoothquant(model, data, n_batches, batch_size, seqlen, alpha=0.5):
  """ SmoothQuant: divides input channel j of the linear layers after each LayerNorm by
      s_j = max|X_j|^alpha / max|W_j|^(1-alpha), folding the division into the LayerNorm, and multiplies input
      channel j of their weights by s_j. The output does not change, but the activations have smaller outliers """
  for block in model.blocks:
    mha = block.mha
    groups = ((block.ln1, [mha.query_proj, mha.key_proj, mha.value_proj]), (block.ln2, [block.ffwd.net[0]]))
    inputs = calibration_inputs(model, [linears[0] for _, linears in groups], data, n_batches, batch_size, seqlen)
    for ln, linears in groups:
      x_max = inputs[linears[0]].abs().amax(dim=0)  # [in_features]
      w_max = torch.stack([linear.weight.abs().amax(dim=0) for linear in linears]).amax(dim=0)
      s = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
      ln.weight.div_(s)
      ln.bias.div_(s)
      for linear in linears:
        linear.weight.mul_(s)
  return model
```

### Rotations: QuaRot and SpinQuant

[QuaRot](https://arxiv.org/abs/2404.00456) and [SpinQuant](https://arxiv.org/abs/2405.16406) multiply the activations by an [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix) $$R$$, such as a normalized [Hadamard matrix](https://en.wikipedia.org/wiki/Hadamard_matrix), and the weights by its transpose. Since $$R^TR = I$$, the output does not change, but the rotation spreads each outlier across all channels. This makes 4-bit weights, activations and KV cache possible (W4A4KV4).

QuaRot uses random Hadamard matrices, while SpinQuant learns the rotations, which gives a higher accuracy, as some rotations quantize much better than others. The picture below rotates a vector with one outlier channel:

{: .picture}
<img width="80.6%" height="80.6%" src="{{ site.assets }}/inference/imgs/hadamard_rotation.svg"/>

{: .caption}
Rotating the weights the other way, $$W R^T$$, keeps the output: $$(W R^T)(R x) = W x$$. After the rotation, the INT4 levels are closer together (5.7 instead of 12.6 apart), so the small values lose much less precision.

The following code builds Hadamard matrices, and a linear layer that simulates low-precision weights and activations with fake quantization, with an optional rotation of its inputs. For simplicity, it rotates the input of every layer on the fly. QuaRot instead folds most rotations into the weights of the previous layers, and applies the few remaining ones with [fast Hadamard](https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform) kernels.

Sylvester's construction builds a Hadamard matrix of size $$n$$ when $$n$$ is a power of 2. For other sizes, such as the 384 features of the attention output of GPTlite, the code uses a block-diagonal matrix of smaller Hadamard matrices, which is also orthogonal:

```python
def hadamard(n):
  """ Orthonormal rotation of size n built from Hadamard matrices (Sylvester's construction): one Hadamard
      matrix if n is a power of 2, otherwise a block-diagonal matrix of Hadamard matrices, of the largest power
      of 2 that divides n (e.g. 3 blocks of 128 for n = 384) """
  k = n & -n  # largest power of 2 that divides n
  H = torch.ones(1, 1)
  while H.size(0) < k:
    H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)])
  return torch.block_diag(*[H / k ** 0.5] * (n // k))


class FakeQuantLinear(nn.Module):
  """ Linear layer that simulates low-precision weights and activations: they are quantized and dequantized by
      weight_fn and act_fn (fake quantization), and the multiplication runs in the precision of the input.
      With rotate=True, the inputs and the weights are rotated by a Hadamard matrix R first: W x = (W R^T)(R x) """

  def __init__(self, linear, weight_fn, act_fn=None, rotate=False):
    super().__init__()
    W = linear.weight.data
    self.rotation = hadamard(W.size(1)).to(W.device) if rotate else None
    if rotate:
      W = W @ self.rotation.T
    self.register_buffer('weight', weight_fn(W))
    self.bias, self.act_fn = linear.bias, act_fn

  def forward(self, x):
    if self.rotation is not None:
      x = x @ self.rotation.T  # x is a row vector: R x = x R^T
    if self.act_fn is not None:
      x = self.act_fn(x)
    return F.linear(x, self.weight, self.bias)
```

### FP4: MXFP4 and NVFP4

**FP4** (NVFP4 and MXFP4) on Blackwell GPUs uses micro-scaled formats, for weights and activations. In the **micro-scaling** ([MX](https://arxiv.org/abs/2310.10537)) formats, every block of 32 consecutive values shares an 8-bit power-of-two scale (MXFP8, MXFP6 and MXFP4), and NVIDIA's **NVFP4** uses blocks of 16 values with an FP8 scale, plus one FP32 scale per tensor. The smaller blocks and finer scales of NVFP4 follow the data more closely, at the cost of a quarter of a bit more per value.

Blackwell tensor cores apply these scales in hardware. OpenAI's gpt-oss models, for example, ship their [mixture-of-experts](https://en.wikipedia.org/wiki/Mixture_of_experts) weights in MXFP4.

The picture below shows the FP4 values and the blocks of both formats:

{: .picture}
<img width="99.5%" height="99.5%" src="{{ site.assets }}/inference/imgs/microscaling.svg"/>

The following code simulates both formats with fake quantization. Used with the `FakeQuantLinear` layer above, it quantizes the weights and activations of GPTlite to FP4:

```python
FP4_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])  # the non-negative values of FP4 E2M1


def round_to_fp4(x):
  """ Rounds every value of x, in [-6, 6], to the nearest value of FP4 E2M1 """
  grid = FP4_E2M1.to(x.device, x.dtype)
  nearest = (x.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
  return torch.sign(x) * grid[nearest]


def fake_quantize_mxfp4(x):
  """ MXFP4: every block of 32 consecutive values shares a power-of-two scale (an E8M0 number). The scale maps
      the largest value of the block to [4, 8), the exponent of the largest FP4 value (6), and larger values
      are clipped to 6 """
  blocks = x.unflatten(-1, (-1, 32))
  amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=2 ** -126)
  scale = 2 ** (torch.floor(torch.log2(amax)) - 2)
  return (round_to_fp4((blocks / scale).clamp(-6, 6)) * scale).flatten(-2)


def fake_quantize_nvfp4(x):
  """ NVFP4: every block of 16 consecutive values shares an FP8 E4M3 scale, which maps its largest value to 6,
      and the tensor has one FP32 scale, which maps the largest block scale to 448, the largest FP8 E4M3 value """
  tensor_scale = (x.abs().amax() / (6 * 448)).clamp(min=1e-12)
  blocks = x.unflatten(-1, (-1, 16))
  block_scale = (blocks.abs().amax(dim=-1, keepdim=True) / 6 / tensor_scale).to(torch.float8_e4m3fn).to(x.dtype)
  scale = (block_scale * tensor_scale).clamp(min=1e-12)
  return (round_to_fp4((blocks / scale).clamp(-6, 6)) * scale).flatten(-2)
```

### KV cache quantization

At long contexts and large batches, the KV cache can be larger than the model weights. Storing it in FP8 or INT8 halves its size and the bytes read at each decode step, with almost no loss of quality; most serving engines enable it with a single configuration flag.

More aggressive methods go down to 2 bits: [KIVI](https://arxiv.org/abs/2402.02750) quantizes the keys per channel, since their outliers live in fixed channels, and the values per token, and keeps the most recent tokens in full precision. Quantized attention kernels, such as [SageAttention](https://github.com/thu-ml/SageAttention) ([Zhang et al.](https://arxiv.org/abs/2410.02367)), apply the same ideas to the attention computation itself. The picture below shows how KIVI groups the keys and the values:

{: .picture}
<img width="88.1%" height="88.1%" src="{{ site.assets }}/inference/imgs/kivi.svg"/>

{: .caption}
The outliers of the keys live in a few fixed channels, so each channel gets its own scale. The outlined cells share one scale and zero-point. Inspired by: [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)

The following code quantizes the KV cache of GPTlite to FP8, or to 2 bits with KIVI, and generates text with it. The loss of quality is measured as the fraction of the generated tokens that match those generated with the full-precision cache:

```python
def fake_quantize_fp8(x):
  """ FP8 E4M3 with one scale per tensor, which maps the largest value to 448 """
  scale = (x.abs().amax() / 448).clamp(min=1e-12)
  return (x / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale


def quantize_kv_cache(kv_cache, bits=2, group_size=32, residual=32):
  """ KIVI: quantizes the keys per channel and the values per token, in groups of group_size tokens, with
      asymmetric quantization. The last tokens, at least residual of them, stay in full precision until they fill
      a group. Returns the dequantized cache: a list of (keys, values) of shape [B, H, S, D], one per block """
  new_cache = []
  for k, v in kv_cache:
    n = (k.size(2) - residual) // group_size * group_size  # the tokens of the full groups out of the window
    if n > 0:
      k_groups = k[:, :, :n].unflatten(2, (-1, group_size))  # [B, H, n/G, G, D]
      k_old = fake_quantize(k_groups, bits, dim=-2, symmetric=False).flatten(2, 3)  # one scale per channel
      v_old = fake_quantize(v[:, :, :n], bits, dim=-1, symmetric=False)              # one scale per token
      k, v = torch.cat([k_old, k[:, :, n:]], dim=2), torch.cat([v_old, v[:, :, n:]], dim=2)
    new_cache.append((k, v))
  return new_cache


@torch.inference_mode()
def generate_quantized_kvcache(model, prompt, n_tokens, seqlen, quantize_fn):
  """ Greedy generation with a KV cache that quantize_fn quantizes and dequantizes after every step """
  logits, kv_cache = model(prompt, max_seqlen=seqlen)
  tokens = prompt
  for i in range(n_tokens):
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_token], dim=-1)
    if i < n_tokens - 1:
      logits, kv_cache = model(next_token, kv_cache=quantize_fn(kv_cache), max_seqlen=seqlen)
  return tokens
```

## Which one to use

For quantization, a reasonable order is to:

- start in BF16;
- quantize the KV cache to FP8 for long contexts;
- use FP8 weights and activations on Hopper or newer GPUs;
- use 4-bit weights (AWQ or GPTQ) when memory is the constraint or the batch size is small;
- use FP4 on Blackwell GPUs, ideally with a model trained or fine-tuned for it (QAT).

Pruning to 2:4 sparsity (with Wanda or SparseGPT) needs no retraining but speeds up only the matrix multiplications. Structured pruning followed by distillation (as in Minitron) needs some retraining, and gives a smaller dense model that is faster on any hardware. The quality must always be measured as well as the speed: the [perplexity](https://en.wikipedia.org/wiki/Perplexity) on held-out data, and the accuracy on the tasks of interest.

To find more about the topic, the survey [Sparsity in Deep Learning](https://arxiv.org/abs/2102.00554) by Hoefler et al. covers pruning beyond LLMs, from the choice of what to remove to the speedups on real hardware. Similarly, the [survey of quantization methods](https://arxiv.org/abs/2103.13630) by Gholami et al. covers the quantization schemes, the granularity of the scales and quantization-aware training.
