---
layout: post
title:  "Mixture-of-Experts and Expert Parallelism"
categories: ["Distributed Training"]
tags: ["MoE", "expert parallelism", "load balancing", "Switch Transformer", "PyTorch"]
---

A [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) layer holds several sub-networks, the *experts*, and a *router* that weights the experts for each input. A sparse MoE runs only the few experts with the largest weights. The number of parameters then grows with the number of experts, while the compute per token stays almost constant. MoEs therefore increase the model capacity per FLOP. A widely shared [leak report](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/) claims that [GPT-4](https://en.wikipedia.org/wiki/GPT-4) is an MoE with 16 experts of about 110B parameters each, about 1.8T parameters in total.

## The First MoEs: A Weighted Sum of Expert Outputs

In 1991, [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) and [Task Decomposition Through Competition in a Modular Connectionist Architecture](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1502_2) introduced early versions of the MoE.

{: .picture}
<img width="88.6%" height="88.6%" src="{{ site.assets }}/distributed-training/imgs/moe_2014_ilya.svg" alt="MoE paper 2014"/>

{: .caption}
**Left:** a Mixture of Experts. **Right:** a Deep Mixture of Experts (with 2 layers). <br/>Adapted from: [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

An MoE has $$N$$ **experts**, the feed-forward networks $$f_1(x), \dots, f_N(x)$$, and a **gating network** $$g(x)$$, also a feed-forward network. The superscripts in the figure number the layers. All experts and the gating network receive the same input $$x$$. The gating network outputs one probability per expert, the relevance of that expert for the input. In its simplest form, $$g(x) = \mathrm{softmax}(x W_g)$$, where the learnable matrix $$W_g$$ maps the input to one logit per expert. The output of the MoE is the sum of the outputs of the experts, weighted by the gate:

$$
z = \sum_{i=1}^{N} g_i(x) \, f_i(x)
$$

In PyTorch, the router is a small feed-forward network with a softmax output, one probability per expert and token. Each expert is a feed-forward network:

```python
class Router(nn.Module):
  """ a DNN to route tokens to experts """

  def __init__(self, n_embd, num_experts, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(n_embd, n_embd*4), nn.ReLU(),
      nn.Linear(n_embd*4, n_embd*4), nn.ReLU(),
      nn.Linear(n_embd*4, num_experts), nn.Softmax(dim=-1)
    )

  def forward(self, x):
    return self.net(x)


class MoE(nn.Module):

  def __init__(self, input_size, output_size, num_experts, dropout=0.1):
    super().__init__()
    self.router = Router(input_size, num_experts, dropout=dropout)
    self.experts = nn.ModuleList([
      FeedForward(input_size, output_size, dropout=dropout) for _ in range(num_experts) ])

  def forward(self, x):
    probs = self.router(x)
    outputs = torch.stack([ expert(x) for expert in self.experts ], dim=-2) # B*S*n_experts*E
    weighted_outputs = outputs * probs.unsqueeze(-1) # B*S*n_experts*E x B*S*n_experts*1 -> B*S*n_experts*E
    weighted_sum = weighted_outputs.sum(dim=-2) # sum over experts: B*S*n_experts*E -> B*S*E
    return weighted_sum, probs, outputs
```

The experts and the gate train together, and the loss decides how the experts share the work. For an input with target $$d$$, Adaptive Mixtures of Local Experts compares three losses:
- **Cooperation**, $$L = \Vert d - \sum_i g_i(x) f_i(x) \Vert^2$$, used in earlier work. The error of an expert depends on the outputs of the other experts, so each expert learns to cancel the residual error of the others. Many experts then contribute to each input, and the experts do not specialize.
- **Competition**, $$L = \sum_i g_i(x) \Vert d - f_i(x) \Vert^2$$. Each expert must produce the whole output on its own, and the gate learns to give each input to a single expert.
- **Negative log-likelihood** of a Gaussian mixture, $$L = -\log \sum_i g_i(x) \, e^{-\frac{1}{2} \Vert d - f_i(x) \Vert^2}$$, used in the experiments of the paper. The gradient of expert $$i$$ is weighted by how well expert $$i$$ fits the target compared with the other experts, not only by $$g_i(x)$$. This weight measures the relevance of the expert better, especially early in training.

The code below implements the competition loss, with the squared error averaged over the classes of a one-hot target:

```python
def loss_per_token_Adaptive_Mixture_1991(probs, outputs, labels):
  """
  outputs: B*S*n_experts*Classes
  probs:   B*S*n_experts
  labels:  B*S (class ids)
  returns: B*S (loss per token)
  """
  one_hot = torch.nn.functional.one_hot(labels, num_classes=outputs.shape[-1]).to(outputs.dtype) # B*S*Classes
  # per-token MSE for each expert: B*S*n_experts
  mse = (one_hot.unsqueeze(-2) - outputs).square().mean(dim=-1)  # mean over Classes
  return (mse * probs).sum(dim=-1)  # weighted sum over experts -> B*S
```

The authors show that each expert learns a part of the task:

> the learning procedure divides up a vowel discrimination task into appropriate subtasks, each of which can be solved by a very simple expert network.

## Deep Mixture of Experts

[Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314) (Eigen et al., 2014) stacks MoE layers into a Deep Mixture of Experts (DMoE), shown on the right of the figure above. Each layer has its own gating network and experts, and the output $$z^1$$ of the first layer is the input of the second:

```python
class MoE_deep(nn.Module):
  def __init__(self, input_size, output_size, num_experts, depth=2, dropout=0.1):
    super().__init__()
    self.stack = nn.ModuleList(
      [MoE(input_size, input_size,  num_experts, dropout) for _ in range(depth-1)] +
      [MoE(input_size, output_size, num_experts, dropout)]
    )

  def forward(self, x):
    # each MoE returns (weighted_sum, probs, outputs); we feed forward only the weighted_sum
    for layer in self.stack:
      x, _, _ = layer(x)
    return x
```

Training often reaches a degenerate local minimum, where "experts at each layer that perform best for the first few examples end up overpowering the remaining experts". The gate then picks these experts more often, so they keep their lead. Eigen et al. therefore constrain the gate in a first training phase. If the cumulative gate value of an expert exceeds the mean of all experts by more than a fixed margin, the expert gets a gate of 0 for the current example. The other gates are renormalized. A second training phase removes the constraint.

The experiments use MNIST digits with random translations and a speech dataset of monophones. The two-layer DMoE is compared with a one-layer MoE and with a fully connected network (DNN) of the same number of parameters. Two baselines change one layer: a single fixed expert at the second layer, and the concatenation of the outputs of all experts of a layer instead of their weighted sum. On the training set, the error of the DMoE lies mostly between those of the two baselines. On the test set, the DMoE overfits: its error is higher than that of the single expert for two of the four model sizes. Its error is close to that of the DNN, which has a less constrained structure.

The DMoE also learns a factored representation:

> we find that the Deep Mixture of Experts automatically learns to develop location-dependent ("where") experts at the first layer, and class-specific ("what") experts at the second layer.

## Sparsely-Gated Mixture of Experts

In the MoEs above, all experts process all inputs, so the compute grows with the number of experts. *Conditional computation* runs only some of the experts for each input: the gate selects the top-$$k$$ experts of each token, and the token goes only to them. The gate thus works as a router. When $$g_i(x) = 0$$, the output $$f_i(x)$$ is not computed. With $$k$$ fixed, the number of experts can grow without a proportional increase in compute, which is the key scaling property of MoEs. $$k$$ trades the compute per token against the number of experts that contribute to it.

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017) places sparse MoE layers between stacked [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) layers of language models and translation models, with up to 131072 experts and 137B parameters. The MoE runs once per position of the text, with a different choice of experts at each position:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/distributed-training/imgs/moe_2017_dean.svg"/>

{: .caption}
A Mixture of Experts (MoE) layer embedded within a recurrent language model. In this case, the sparse gating function selects two experts to perform computations. Their outputs are modulated by the outputs of the gating network. Adapted from: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

### Gating and Sparsity

Shazeer et al. use **noisy top-$$k$$ gating**. Gaussian noise, scaled per expert by a learnable matrix $$W_{noise}$$, is added to the logits of the gate, which gives the noisy logits $$h(x)$$. The logits outside the top-$$k$$ are set to $$-\infty$$ before the softmax, so their gate values are 0:

$$
g(x) = \mathrm{softmax}(\mathrm{KeepTopK}(h(x), k))
$$

$$
h_i(x) = (x W_g)_i + \epsilon \cdot \mathrm{Softplus}((x W_{noise})_i), \quad \epsilon \sim \mathcal{N}(0, 1)
$$

Softplus keeps the scale of the noise positive. On the 1-billion-word benchmark, the flat MoEs use $$k = 4$$. The figure shows $$k = 2$$, and writes $$G(x)$$ for the gate and $$n$$ for the number of experts.

### Load Balancing

Without a constraint, the router sends most tokens to a few experts. These experts train faster and are picked even more often, while the other experts starve. Shazeer et al. add two losses. The **importance** of an expert is the sum of its gate values over a batch $$X$$, $$\mathrm{Importance}(X)_i = \sum_{x \in X} g_i(x)$$. The **importance loss** is the square of the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) $$CV = \sigma / \mu$$ of the importances:

$$
L_{importance} = w_{importance} \cdot CV(\mathrm{Importance}(X))^2
$$

where $$w_{importance}$$ is a hand-tuned weight. Take a batch of 4 inputs and 5 experts with these gate values:

|-|-|-|-|-|-|
| | Expert 1 | Expert 2 | Expert 3 | Expert 4 | Expert 5 |
|-|-|-|-|-|-|
| $$x_1$$ | 0.1 | 0.1 | 0 | 0.8 | 0 |
| $$x_2$$ | 0 | 0 | 0.2 | 0.7 | 0.1 |
| $$x_3$$ | 0.1 | 0 | 0 | 0.9 | 0 |
| $$x_4$$ | 0 | 0 | 0 | 1 | 0 |
| **Importance** | **0.2** | **0.1** | **0.2** | **3.4** | **0.1** |
|-|-|-|-|-|-|

The mean importance is $$\mu = 0.8$$ and the population standard deviation is $$\sigma \approx 1.30$$, so $$CV \approx 1.63$$. Equal importances give $$CV = 0$$.

Equal importances do not guarantee equal numbers of tokens: one expert can receive many tokens with small gate values, and another few tokens with large ones. Unequal loads slow down distributed training, where the device with the most tokens finishes last, and leave some experts under-trained. The **load loss** balances the number of tokens per expert. The token count has no gradient, so the loss uses a smooth estimate of it. The estimate sums, over the batch, the probability $$P(x, i)$$ that expert $$i$$ is in the top-$$k$$ of input $$x$$:

$$
L_{load} = w_{load} \cdot CV(\mathrm{Load}(X))^2, \quad \mathrm{Load}(X)_i = \sum_{x \in X} P(x, i)
$$

$$P(x, i)$$ is the probability that $$h_i(x)$$, with new noise, exceeds the $$k$$-th largest of the other noisy logits, $$\mathrm{kth\_excluding}(h(x), k, i)$$, while the noise of the other experts stays fixed. With $$\Phi$$ the cumulative distribution function of the [standard normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):

$$
P(x, i) = \Phi \left( \frac{(x W_g)_i - \mathrm{kth\_excluding}(h(x), k, i)}{\mathrm{Softplus}((x W_{noise})_i)} \right)
$$

The noise makes this probability a smooth function of the logits, so the load loss has a gradient.

### Batching and Parallelism

If the gate picks $$k$$ of $$N$$ experts for each token, an expert receives about $$kT/N$$ of the $$T$$ tokens of a batch. With many experts, the batch of each expert is small and the hardware is poorly used. Shazeer et al. counter this *shrinking batch problem* in two ways:
- **Mixing data and model parallelism.** The standard layers and the gating network are replicated with data parallelism. Each expert has a single copy, on one device. This split of the experts is now called expert parallelism. With $$W$$ devices of $$T$$ tokens each, an expert receives about $$kTW/N$$ tokens, $$W$$ times more.
- **Convolutional application.** Each expert processes each position of the text independently, so the MoE runs on all time steps of the previous LSTM layer at once, as one large batch.

With thousands of experts, the gate has a large output. A two-level **hierarchical MoE** reduces this branching factor. A primary gating network picks a sparse weighted combination of experts, and each of these experts is itself an MoE with its own gating network. The primary gate uses data parallelism, and each secondary MoE sits on one device.

### Results

The experiments compare MoE models with LSTM baselines of the same compute per token:
- On the 1-billion-word benchmark, at 256 experts, the hierarchical MoE is not better than the flat MoE ([perplexity](https://en.wikipedia.org/wiki/Perplexity) 36.0 against 35.7). It scales to more experts: 34.1 with 4096 experts.
- On 100 billion words of Google News, the perplexity improves up to 65536 experts (68B parameters), 39% lower than the compute-matched baseline, and degrades at 131072 experts.
- On the Google Production translation dataset, the MoE model reaches a 1.01 higher test [BLEU](https://en.wikipedia.org/wiki/BLEU) score after one sixth of the training time.
- The authors claim "greater than 1000x improvements in model capacity with only minor losses in computational efficiency".

The experts specialize in syntax and semantics (Appendix E, Table 9 of the paper):

| Expert 381 | Expert 752 | Expert 2004 |
|:-:|:-:|:-:|
| ... with **researchers** , ... | ... plays **a** core ... | ... with **rapidly** growing ... |
| ... to **innovation** . | ... plays **a** critical ... | ... under **static** conditions ... |
| ... tics **researchers** . | ... provides **a** legislative ... | ... to **swift** ly ... |
| ... the **generation** of ... | ... play **a** leading ... | ... to **dras** tically ... |
| ... technology **innovations** is ... | ... assume **a** leadership ... | ... the **rapid** and ... |
| ... technological **innovations** , ... | ... plays **a** central ... | ... the **fast** est ... |
| ... support **innovation** throughout ... | ... taken **a** leading ... | ... the **Quick** Method ... |
| ... role **innovation** will ... | ... established **a** reconciliation ... | ... rec **urrent** ) ... |
| ... research **scienti** st ... | ... played **a** vital ... | ... provides **quick** access ... |
| ... promoting **innovation** where ... | ... have **a** central ... | ... of **volatile** organic ... |
| ... | ... | ... |

{: .caption}
Contexts corresponding to a few of the 2048 experts in the MoE layer in the encoder portion of the WMT’14 En→Fr translation model. For each expert $$i$$, we sort the inputs in a training batch in decreasing order of $$G(x)_i$$, and show the words surrounding the corresponding positions in the input sentences. Source: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

A PyTorch implementation of this layer is available in [this repository](https://github.com/lucidrains/mixture-of-experts).

## GShard

[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) (Lepikhin et al., 2020) scales a Transformer with sparse MoE layers to 600B parameters. The model translates 100 languages to English and trains on 2048 TPU v3 accelerators in 4 days. The compute grows sub-linearly with the number of parameters.

GShard keeps two ideas of Shazeer et al.: position-wise sparse MoE layers, and a combination of data parallelism and sharding. In GShard, "the attention layer is parallelized by splitting along the batch dimension and replicating its weights to all devices". The experts are too large to replicate, so each device holds a shard of them.

### Compiler and SPMD Partitioning

GShard separates the model code from its parallel partitioning. It is a set of annotation APIs and an extension of the [XLA](https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra) compiler. The user annotates "a few critical tensors in the model with partitioning policies", and XLA partitions the whole computation graph across the devices. The partitioner generates a [single program for all devices](https://en.wikipedia.org/wiki/Single_program,_multiple_data) (SPMD), so the compilation time stays constant as the number of devices grows.

### Model Architecture

The encoder and the decoder replace the feed-forward layer of every second Transformer block with an MoE layer of top-2 gating. An all-to-all collective sends each token to the devices of its experts (dispatch), and a second one returns the outputs (combine):

{: .picture}
<img width="97%" height="97%" src="{{ site.assets }}/distributed-training/imgs/moe_gshard.svg"/>

{: .caption}
Illustration of scaling of Transformer Encoder with MoE Layers. The MoE layer replaces every other Transformer feed-forward layer. Decoder modification is similar. (a) The encoder of a standard Transformer model is a stack of self-attention and feed forward layers interleaved with residual connections and layer normalization. (b) Replacing every other feed-forward layer with an MoE layer gives the MoE Transformer Encoder. (c) When scaling to multiple devices, the MoE layer is sharded across devices, while all other layers are replicated. Adapted from: [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

### Expert Capacity and Random Routing

GShard introduces the **expert capacity**, the maximum number of tokens that an expert processes. The tokens of a batch are split into groups that are dispatched independently. With top-2 gating, a group of $$T$$ tokens sends at most $$2T/N$$ tokens to each expert. Tokens routed to a full expert *overflow*: the MoE layer skips them, and the residual connection passes them to the next layer.

With **random routing**, the first expert of a token always receives it, and the second expert receives it with a probability proportional to its gate value. A second expert with a small gate value contributes little to the output, so skipping it saves compute and capacity.

### Auxiliary Loss

The total loss is $$L = L_{nll} + \alpha L_{aux}$$, with $$L_{nll}$$ the negative log-likelihood, $$\alpha$$ a constant weight, and the **auxiliary loss**:

$$
L_{aux} = \frac{1}{N} \sum_{i=1}^{N} \frac{c_i}{T} \, m_i
$$

$$c_i$$ is the number of tokens whose first choice is expert $$i$$, so $$c_i/T$$ is the fraction of the tokens routed to it. $$m_i$$ is the mean gate value of expert $$i$$ over the $$T$$ tokens. The fractions $$c_i/T$$ sum to 1, and their mean square is minimal when the tokens spread evenly. $$c_i$$ comes from the top-2 selection and has no gradient, so GShard replaces one of the two factors $$c_i/T$$ of the square by the differentiable $$m_i$$.

## Switch Transformers

[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020) found that the loss follows power laws in the model size, the data size and the compute. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021) builds on these results:

> Heeding these results, we investigate a fourth axis: increase the parameter count while keeping the floating point operations (FLOPs) per example constant.

As in GShard, MoE layers replace the feed-forward layers, and the experts are sharded across devices. **Switch routing** sends each token to a single expert (top-1). Shazeer et al. had conjectured that the router needs at least two experts per token to learn, because it must compare them. Switch Transformers show that top-1 routing keeps the model quality. With one expert per token, the expert compute and the all-to-all volume are half those of top-2 routing, and the expert capacity can be at least halved.

Switch Transformers also stabilize the training: the router runs in float32 while the rest of the model runs in [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), and the weights start from a smaller initialization. With these changes, large sparse models train in bfloat16 for the first time. Switch-Base reaches the quality of T5-Base up to 7 times faster with the same compute per token, and the largest model has 1.6 trillion parameters.

{: .picture}
<img width="96.3%" height="96.3%" src="{{ site.assets }}/distributed-training/imgs/moe_switch_transformers.svg"/>

{: .caption}
Illustration of a Switch Transformer encoder block. The dense feed forward network (FFN) layer of the Transformer is replaced by a sparse Switch FFN layer (light blue). The layer operates independently on the tokens in the sequence. Two tokens (<i>x</i><sub>1</sub> = “More” and <i>x</i><sub>2</sub> = “Parameters” below) are routed (solid lines) across four FFN experts; the router routes each token independently. The switch FFN layer returns the output of the selected FFN multiplied by the router gate value (dotted-line). Adapted from: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

### Expert Capacity

The expert capacity is the mean number of tokens per expert multiplied by the **capacity factor**. A factor above 1 leaves a buffer for an uneven routing. Tokens beyond the capacity of their expert skip the layer through the residual connection. A larger factor drops fewer tokens, but adds compute, memory and communication for the padding:

{: .picture}
<img width="96.9%" height="96.9%" src="{{ site.assets }}/distributed-training/imgs/moe_switch_transformers_2.svg"/>

{: .caption}
Illustration of token routing dynamics. Each expert processes a fixed batch-size of tokens modulated by the <i>capacity factor</i>. Each token is routed to the expert with the highest router probability, but each expert has a fixed batch size of (total\_tokens / num\_experts) × capacity\_factor. If the tokens are unevenly dispatched then certain experts will overflow (denoted by dotted red lines), resulting in these tokens not being processed by this layer. A larger capacity factor alleviates this overflow issue, but also increases computation and communication costs (depicted by the empty grey slots). Adapted from: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

### Load Balancing Loss

Switch Transformers replace the importance and load losses of [Shazeer et al.](#sparsely-gated-mixture-of-experts) by one loss, added for every Switch layer. With $$c_i/T$$ the fraction of the tokens dispatched to expert $$i$$ and $$m_i$$ its mean router probability, as in GShard:

$$
L_{balance} = \alpha N \sum_{i=1}^{N} \frac{c_i}{T} \, m_i
$$

Under a uniform routing, $$\sum_i \frac{c_i}{T} m_i = 1/N$$, so the factor $$N$$ keeps the loss at $$\alpha$$ for any number of experts. The paper uses $$\alpha = 10^{-2}$$.

### Distillation

Sparse models have many parameters. Switch Transformers [distill](https://en.wikipedia.org/wiki/Knowledge_distillation) them into small dense models: "We reduce the model size by up to 99% while preserving 30% of the quality gains of the large sparse teacher." A single-device implementation of Switch Transformers is available on [this page](https://nn.labml.ai/transformers/switch/index.html).

## Comparing the Routing Methods

All the methods above store the parameters of all $$N$$ experts. They differ in the number of experts per token, which sets the expert compute per token, with $$F$$ the compute of one expert for one token:

| Routing | Experts per token | Expert compute per token | Load balancing |
|:--|:-:|:-:|:--|
| Dense (Jacobs et al., Eigen et al.) | $$N$$ | $$N F$$ | none, or a gate constraint in a first training phase |
| Noisy top-$$k$$ (Shazeer et al.) | $$k$$ | $$k F$$ | importance and load losses |
| Top-2 with random routing (GShard) | 1 or 2 | at most $$2 F$$ | auxiliary loss, expert capacity |
| Top-1 (Switch Transformers) | 1 | $$F$$ | auxiliary loss, capacity factor |

With expert parallelism, the all-to-all volume per token grows with the number of experts per token, as the compute does. The capacity adds compute and memory for padding. Dense MoEs suit only a few experts, since their compute grows with $$N$$. For a fixed compute and training time, Switch Transformers reach a better quality than top-2 MoEs and dense Transformers.

## Distributed Implementation in PyTorch

An MoE layer runs in four steps, illustrated in the [MegaBlocks paper](https://arxiv.org/abs/2211.15841):

{: .picture}
<img width="94.8%" height="94.8%" src="{{ site.assets }}/distributed-training/imgs/moe_megablocks.svg"/>

{: .caption}
<b>A Mixture-of-Experts Layer.</b> Shown for <i>num\_experts=3</i>, <i>top\_k=1</i> and <i>capacity\_factor=1</i> with the prevalent, token dropping formulation. <b>First (1)</b>, tokens are mapped to experts by the router. Along with expert assignments, the router produces probabilities that reflect the confidence of the assignments. <b>Second (2)</b>, the feature vectors are permuted to group tokens by expert assignment. If the number of tokens assigned to an expert exceeds its capacity, extra tokens are dropped. <b>Third (3)</b>, the expert layers are computed for the set of tokens they were assigned as well as any padding needed for unused capacity. <b>Lastly (4)</b>, the results of the expert computation are un-permuted and weighted by the router probabilities. The outputs for dropped tokens are shown here set to zero. Adapted from: [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)

With expert parallelism, two [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) collectives send the tokens to the devices of their experts and back. In the backward pass, the gradients travel the same path in reverse. `dist.all_to_all_single` has no backward pass, so it is wrapped in an autograd function whose backward pass runs the opposite all-to-all:

```python
class AllToAll(torch.autograd.Function):
  """ all_to_all_single along the first dimension, whose backward pass sends the gradients back """

  @staticmethod
  def forward(ctx, x, output_split_sizes, input_split_sizes):
    ctx.split_sizes = output_split_sizes, input_split_sizes
    out = x.new_empty(sum(output_split_sizes), *x.shape[1:])
    dist.all_to_all_single(out, x.contiguous(), output_split_sizes, input_split_sizes)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    output_split_sizes, input_split_sizes = ctx.split_sizes
    grad_x = grad_out.new_empty(sum(input_split_sizes), *grad_out.shape[1:])
    dist.all_to_all_single(grad_x, grad_out.contiguous(), input_split_sizes, output_split_sizes)
    return grad_x, None, None
```

The MoE block `MoE_dist` has a router, replicated with data parallelism, and one expert per GPU. The number of experts is thus the number of processes, and the rank of a process is the id of its expert. Tokens beyond the capacity of an expert are dropped:

```python
class MoE_dist(nn.Module):
  def __init__(self, n_embd, k=2, capacity_factor=1.25, padding_val=0, local_rank=None):
    super().__init__()
    self.capacity_factor = capacity_factor
    self.padding_val = padding_val
    self.num_experts = dist.get_world_size() # 1 expert per GPU
    self.k = k

    if local_rank is None:
      local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    device = torch.device("cuda", local_rank)

    self.router = DDP(Router(n_embd, self.num_experts).to(device), device_ids=[local_rank])
    self.expert = FeedForward(n_embd).to(device)
```

The **routing step** computes the top-`k` experts of each token, their ids `topk_experts` and their probabilities `topk_probs`:

```python
  def forward(self, x):
    # x: B*S*E
    B, S, E = x.shape
    global_rank = dist.get_rank()
    probs = self.router(x) # get assignments from router, shape B*S*n_experts
    topk_probs, topk_experts = torch.topk(probs, k=self.k, dim=-1) # B*S*k

    ids_per_expert = [ (topk_experts==expert).nonzero() for expert in range(self.num_experts) ]
    probs_per_expert = [ topk_probs[topk_experts==expert] for expert in range(self.num_experts) ]
```

The **permutation step** sends each token to the process of its expert. A first `all_to_all` exchanges the number of tokens that each process sends to each other process. With these counts, an `all_to_all_single` exchanges the metadata of the tokens, and an `AllToAll` exchanges the tokens. The metadata of a token are its position `(b, s)` and the index of the expert in its top-`k`. The row `b` includes an offset by the rank of the sender, so that the rows of different processes differ:

```python
    device = x.device

    # all-to-all to exchange the count of inputs to send/receive to/from each processor
    send_count = [torch.tensor([len(ids)], dtype=torch.int64, device=device) for ids in ids_per_expert]
    recv_count = [torch.tensor([0], dtype=torch.int64, device=device) for _ in ids_per_expert]
    dist.all_to_all(recv_count, send_count)
    fn_count = lambda tensor, scale=1: [x.item()*scale for x in tensor] 

    # send/receive the metadata (b, s, which_topk) to/from the appropriate processors
    M = ids_per_expert[0].shape[-1] # number of columns in metadata
    send_ids = torch.cat(ids_per_expert, dim=0).to(device)
    send_ids[:,0] += global_rank*B # add processor's batch offset
    recv_ids = torch.zeros(sum(recv_count)*M, dtype=send_ids.dtype, device=device)
    dist.all_to_all_single(recv_ids, send_ids.flatten(), fn_count(recv_count,M), fn_count(send_count,M))
    recv_ids = recv_ids.view(-1, M) # reshape to M columns 

    # send/receive input tokens to/from the appropriate processors
    send_toks = torch.cat([ x[ids[:,0], ids[:,1]] for ids in ids_per_expert], dim=0).to(device)
    recv_toks = AllToAll.apply(send_toks, fn_count(recv_count), fn_count(send_count)) # sum(recv_count) rows, E columns
```

{: .picture}
<img width="71%" height="71%" src="{{ site.assets }}/matrix-transpose/all_to_all.svg"/>

{: .caption}
An illustration of the MPI all-to-all collective operation. When each process sends one element to each process, the operation is equivalent to a distributed matrix transpose: the row framed in red before the all-to-all (left), the data of process A, becomes the first column after the all-to-all (right). A single-element all-to-all exchanges the number of items to be sent and received among processes. These counts are then used by two all-to-alls with variable-sized elements in the permutation step, one for the token metadata and one for the tokens. In the un-permutation step, another all-to-all performs the reverse communication by swapping the send and receive counts and buffers. Adapted from: [MPI Tutorial, Shao-Ching Huang, IDRE High Performance Computing Workshop, 2013](https://www.slideshare.net/slideshow/mpi-49244468/49244468)

The **computation step** groups the received tokens by row into a 3D batch of shape `Rows x Capacity x Embedding`. The tokens of each row are cropped or padded to the capacity of the expert:

```python
    # group received metadata by row id 
    uniq_rows, recv_row_lens = recv_ids[:,0].unique(sorted=True, return_counts=True)
    recv_row_offsets = [0] + torch.cumsum(recv_row_lens, dim=0).tolist()
    recv_row_slice = lambda row: slice(recv_row_offsets[row], recv_row_offsets[row+1])

    # crop or pad received items PER SENTENCE to max capacity. Batch shape: Rows * Capacity * E
    capacity = int(S / self.num_experts * self.capacity_factor)
    pad_fn = lambda toks, value = self.padding_val: F.pad(toks, (0,0,0,capacity-toks.shape[0]), value=value) # pad or crop
    batch_row_len = torch.tensor([ min(recv_row_lens[r], capacity) for r in range(len(uniq_rows))], device=device)

    if len(uniq_rows) > 0: # the expert may receive no tokens
      batch_toks = torch.stack([ pad_fn(recv_toks[recv_row_slice(i)][:capacity]) for i in range(len(uniq_rows))], dim=0).to(device)
      batch_toks = self.expert(batch_toks) # Rows * Capacity * E
``` 

The **un-permutation step** writes the outputs of the expert in the order in which the tokens arrived, with the padding value for the dropped tokens. An `AllToAll` with the counts of the permutation step, swapped, sends them back:

```python
    # scatter computed outputs back into recv_toks (only for the non-padding part). out_flat is computed from recv_toks
    # so that every process runs both all-to-alls of the backward pass, even if its expert received no tokens
    out_flat = recv_toks * 0 + self.padding_val
    cursor = 0
    for r in range(len(uniq_rows)):
      n = int(batch_row_len[r].item())
      out_flat[cursor:cursor+n] = batch_toks[r, :n]
      cursor += int(recv_row_lens[r].item())

    # send back
    x = AllToAll.apply(out_flat, fn_count(send_count), fn_count(recv_count)) # sum(send_count) rows, E columns
```

Finally, the **scale step** multiplies each output by the router probability of its expert, and sums the top-`k` outputs of each token. The outputs arrive grouped by expert, so `index_add` adds each one at the position `(b, s)` of its token, given by the metadata of the permutation step:

```python
    x = x * torch.cat(probs_per_expert).view(-1,1) # multiply by router probabilities
    rows = (send_ids[:,0] - global_rank*B) * S + send_ids[:,1] # position b*S+s of each output, without the batch offset
    out = torch.zeros(B*S, E, dtype=x.dtype, device=device).index_add(0, rows, x) # sum the top-k outputs of each token
    return out.view(B,S,E)
```

This implementation has no load balancing loss. The Switch loss can be computed from `probs` and `topk_experts` after the routing step.

### Applying the MoE to an Existing LLM

The MoE block can replace the feed-forward block of existing models. In the [GPT-2-based `GPTlite` model of a previous post]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}), all modules use Distributed Data Parallel, except the feed-forward module of each transformer block, which becomes the MoE:

```python
  # ddp function takes a torch module and distributes it in a data parallel fashion
  ddp = lambda module: DDP(module.to(device), device_ids=[local_rank]) 

  model = GPTlite(vocab_size) # instantiate GPTlite locally
  model.token_embedding_table = ddp(model.token_embedding_table)
  model.position_embedding_table = ddp(model.position_embedding_table)
  model.ln = ddp(model.ln)
  model.lm_head = ddp(model.lm_head)
  for b, block in enumerate(model.blocks): 
    block.sa = ddp(block.sa)
    block.ln1 = ddp(block.ln1)
    block.ln2 = ddp(block.ln2)
    block.ffwd = MoE_dist(n_embd=n_embd) # replace FeedForward with a MoE (parallelism handled inside)
```

The training loop is unchanged. Libraries such as [DeepSpeed Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/) implement these communication steps.

## Further Reading

Selected papers on the training, fine-tuning, scaling and variants of MoEs:

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2022 [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)</summary>
MegaBlocks is "a system for efficient Mixture-of-Experts (MoE) training on GPUs". An MoE layer with a fixed expert capacity must drop tokens or pad the batches of its experts, and the capacity factor trades model quality against hardware efficiency. In the experiments of the paper, the loss decreases as the capacity grows, and is lowest with no dropped tokens, using the dynamic capacity factor of [Tutel](https://arxiv.org/abs/2206.03382). The sparse computation of MoEs also "does not map cleanly to the software primitives supported in major frameworks and libraries".

MegaBlocks writes the MoE layer as block-sparse matrix products, with new GPU kernels for a variable number of tokens per expert. Its *dropless* MoEs (dMoEs) never drop tokens, while regular MoEs use batched matrix products with a fixed batch per expert. The kernels store the sparse matrices in a blocked-CSR-COO format, with *transpose indices* for the products with transposed matrices. Training is up to 40% faster than MoE training with Tutel, and up to 2.4 times faster than dense training with Megatron-LM.
</details>
{::options parse_block_html="false" /}



{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)</summary>
This paper studies "how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model". It analyzes the problem in theory, and tests it on synthetic data with 4 clusters, on CIFAR-10 and on rotated CIFAR-10 images. It finds that the "cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE". The router learns "the cluster-center features", which splits the problem into "simpler linear classification sub-problems that individual experts can conquer".
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)</summary>
GLaM (Generalist Language Model) is a family of dense and sparse decoder-only language models. The largest GLaM has 1.2 trillion parameters, about 7 times more than GPT-3. It needs a third of the energy of GPT-3 for training and half of its FLOPs for inference, and reaches better average zero-shot and one-shot results on 29 NLP tasks. Dense and MoE variants trained on the same data show that:
- the quality of the training data matters for the downstream tasks;
- sparse models scale better than dense models;
- MoE models need less data than dense models of similar FLOPs to reach similar zero-, one- and few-shot results.
</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)</summary>
ST-MoE studies the stability of the training and fine-tuning of large sparse MoEs. It introduces ST-MoE-32B, a sparse model of 269B parameters with the compute of a 32B dense model. It also introduces the **router z-loss**, which removes the instabilities and slightly improves the quality. The z-loss penalizes large logits $$h$$ at the input of the router, for $$T$$ tokens and $$N$$ experts:

$$
L_z = \frac{1}{T} \sum_{t=1}^{T} \left( \log \sum_{i=1}^{N} e^{h_i^{(t)}} \right)^2
$$

Further findings:
- Removing multiplicative operations, the GELU Gated Linear Units (GEGLU) and the Root Mean Square scale parameters, improves the stability but costs quality. So do noise on the input (input jitter), dropout, and the clipping of the updates of the Adafactor optimizer.
- Sparse models are sensitive to roundoff errors, such as those of low-precision formats, because the routers add exponential functions.
- Sparse models overfit more easily: on the small CB task, the dense model is better on the validation set.
- In fine-tuning, a dropout of 0.1 works best, higher rates hurt the quality, and a higher dropout in the experts improves the quality only moderately.
- Fine-tuning only the non-MoE parameters works about as well as fine-tuning all parameters, while fine-tuning only the MoE parameters, about 80% of the model, strongly reduces the quality.
- Sparse models fine-tune best with small batches and high learning rates, and dense models with the opposite.
- Dropping up to 10-15% of the tokens does not hurt the quality of fine-tuning.
- The recommended starting point is top-2 routing, a capacity factor of 1.25 and at most one expert per core. The capacity factor can change at evaluation time, to fit new memory and compute limits. Dense layers stacked with the sparse layers, and a multiplicative bias, improve the quality.

In the encoder, at each layer, at least one expert specializes in sentinel tokens, the mask tokens that mark the blanks to fill. Other encoder experts specialize in punctuation, verbs, proper names or counting (Table 13). The decoder experts specialize far less. The authors relate this to the span corruption objective. The decoder routes fewer tokens together (groups of 456 tokens, against 2048 in the encoder), and more of them are sentinel tokens. The tokens of a group thus cover a smaller semantic space.
</details>
{::options parse_block_html="false" /}



{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2022 [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382)</summary>
The router sets the load of each expert at runtime. Existing MoE systems "suffer inefficient computation due to their static execution, namely static parallelism and pipelining, which does not adapt to the dynamic workload". Tutel switches the parallelism strategy at every iteration (*adaptive parallelism*), and tunes the pipelining degree and the algorithm of the dispatch and combine all-to-alls (*adaptive pipelining*). The table compares the communication of data (DP), model (MP) and expert (EP) parallelism in an MoE layer, alone and combined. $$P$$ is the number of parameters of all experts, $$C_g$$ the token capacity per GPU, $$W$$ the number of GPUs, and $$E$$ the number of experts ($$N$$ above). In ⑦, each expert is split into $$r$$ slices with model parallelism, and each slice is replicated on $$(W/E)/r$$ GPUs with data parallelism.

| Parallelism Method | Communication Complexity | Limitation | Comment |
|:--|:--|:-:|:--|
| ① DP | $$\mathcal{O}(P)$$ | - | Possibly optimal |
| ② MP | $$\mathcal{O}(C_g \cdot W)$$ | - | No better than ⑥ |
| ③ EP | $$\mathcal{O}(C_g)$$ | $$E/W \geq 1$$ | No better than ⑥ |
| ④ DP+MP | $$\mathcal{O}(C_g \cdot r + P/r)$$ | $$1 \leq r \leq W$$ | No better than ⑦ for any $$r$$ |
| ⑤ EP+DP | $$\mathcal{O}(C_g + P/E)$$ | - | A special case of $$r = 1$$ in ⑦ |
| ⑥ EP+MP | $$\mathcal{O}(C_g \cdot max\{1, W/E\})$$ | - | A special case of $$r = W/E$$ in ⑦ |
| ⑦ EP+DP+MP | $$\mathcal{O}(C_g \cdot W/E)$$ – if $$r \geq W/E$$ <br/> $$\mathcal{O}(C_g \cdot r + P/E/r)$$ – if $$1 \leq r < W/E$$ | - | Possibly optimal |

{: .caption}
Analysis on communication complexity of MoE parallelism. Source: [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382)

</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span">2023 [Mixture-of-Experts Meets Instruction Tuning: a Winning Combination for Large Language Models](https://arxiv.org/abs/2305.14705)</summary>
Without instruction tuning, MoE models fine-tuned on single downstream tasks are worse than dense models of the same compute. With instruction tuning, followed or not by fine-tuning on the task, MoE models are better than dense models. The largest model, FLAN-MoE-32B, outperforms FLAN-PaLM-62B on four benchmarks with a third of the FLOPs.
</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2023 [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)</summary>

Soft MoE is a fully differentiable sparse Transformer. A sparse MoE assigns each token to experts, a discrete choice. In a Soft MoE, each expert processes a few *slots*. Each slot is a weighted average of all input tokens, with weights that depend on the tokens and the slot. The compute depends on the number of slots, not on the number of experts. Soft MoE addresses four issues of sparse MoEs: training instability, dropped tokens, the limit on the number of experts, and poor fine-tuning.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/soft_moes.svg"/>

{: .caption}
<b>Sparse and Soft MoE layers.</b> While the router in Sparse MoE layers (left) learns to <i>assign</i> individual input tokens to each of the available slots, in Soft MoE layers (right) each slot is the result of a (different) <i>weighted average</i> of all the input tokens. Learning to make discrete assignments introduces several optimization and implementation issues that Soft MoE sidesteps. Adapted from: [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)

In image classification, Soft MoE outperforms dense Transformers (ViTs) and the sparse Tokens Choice and Experts Choice MoEs. With 128 experts, it has over 40 times the parameters of ViT-H/14, for about 2% more inference time. Each slot mixes all tokens of the input, so Soft MoE does not apply directly to autoregressive decoders, where a token must not depend on later tokens.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2024 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)</summary>
Mixtral 8x7B has the architecture of Mistral 7B, with two differences. It has a fully dense context of 32k tokens, and its feed-forward blocks are MoE layers of 8 feed-forward experts. With top-2 routing, "each token has access to 47B parameters, but only uses 13B active parameters during inference". The gate is $$g(x) = \mathrm{softmax}(\mathrm{TopK}(x W_g))$$, where $$\mathrm{TopK}$$ sets the logits outside the top-$$k$$ to $$-\infty$$. This is the formulation of GShard, with an MoE in every block instead of every second block, and a simpler choice of the second expert. Each expert is a [SwiGLU](https://arxiv.org/abs/2002.05202v1) feed-forward network, as in Llama 2, so the output for an input $$x$$ is:

$$
z = \sum_{i=1}^{N} \mathrm{softmax}(\mathrm{Top2}(x W_g))_i \cdot \mathrm{SwiGLU}_i(x)
$$

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2024 [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)</summary>

Not all tokens need the same compute. Mixture-of-Depths (MoD) uses conditional computation, which "tries to reduce total compute by expending it only when needed". Here, "the network must learn how to dynamically allocate the available compute by making decisions per-token, in each layer, about where to spend compute from the available budget". The budget is a fixed number of FLOPs.

For each token, a router decides whether the token goes through the block (self-attention and MLP) or around it, through the residual connection, which saves compute. Unlike in MoEs, the routing applies to both the MLP and the self-attention. A token routed around a block is not updated, and the other tokens cannot attend to it. "We refer to this strategy as Mixture-of-Depths (MoD) to emphasize how individual tokens pass through different numbers of layers, or blocks, through the depth of the transformer". The router gives each token a scalar weight, and the top-$$k$$ tokens of each sequence go through the block. $$k$$ is fixed, so the computation graph and the tensor sizes are static, and $$k$$ sets the trade-off between quality and speed for a given compute budget.

{: .picture}
<img width="95.4%" height="95.4%" src="{{ site.assets }}/distributed-training/imgs/mixture_of_depths.svg"/>

{: .caption}
<b>Mixture-of-Depths Transformer.</b> As in mixture-of-experts (MoE) transformers, a router chooses among potential computational paths. But unlike in MoE transformers the possible choices are a standard block’s computation (i.e., self-attention and MLP) or a residual connection. Since some tokens take this second route, Mixture-of-Depths (MoD) transformers have a smaller total FLOP footprint compared to vanilla or MoE transformers. On the top right is depicted a trained model’s routing decisions for a short sequence truncated to 64 tokens for visualization purposes. When examining the choices one can find tokens processed by later blocks’ layers, despite passing through relatively few total blocks throughout the model’s depth. This is a unique feature of MoD compared to conventional halting-based, or "early-exit" conditional computation, which instead engage blocks serially, or vanilla transformers, which engage every block. Adapted from: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

The authors compare *token-choice* routing, where each token picks its paths (experts), with *expert-choice* routing, where each path picks its top-$$k$$ tokens. They use expert-choice routing, which needs no load balancing. With it, "routers can try to ensure that the most critical tokens are among the top-$$k$$ by setting their weight appropriately".

{: .picture}
<img width="99.5%" height="99.5%" src="{{ site.assets }}/distributed-training/imgs/mixture_of_depths_2.svg"/>

{: .caption}
<b>Routing schemes.</b> Tokens are funnelled to the computational path of their choice when using token-choice routing (left). If a given path exceeds its capacity (e.g., more than two tokens in this example) then surplus tokens must be dropped (purple token). The exact token that is ultimately dropped depends on the precise implementation in the underlying code. For example, priority is often given to those tokens that come earlier in the sequence or batch order. With expert-choice routing (middle), precisely <i>k</i> (in this case, two) tokens are chosen per path using a top-<i>k</i> mechanism across the tokens’ router weights. Here, tokens are dropped if they are not among the top-<i>k</i> with respect to any given path (orange token), and some tokens may even be funnelled to multiple paths (yellow token). Mixture-of-Depths uses expert-choice routing with a single path (right). Since tokens are dropped whenever <i>k</i> is less than the sequence length, the dropped tokens are routed around the self-attention and MLP computations, and a forward pass spends fewer FLOPs. Adapted from: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

The best models process 12.5% of the tokens in every second block. They match the quality of the baseline for the same training FLOPs and time. They need a fraction of its FLOPs per forward pass, and can be upwards of 50% faster per step in sampling. The top-$$k$$ of a sequence depends on later tokens, which autoregressive sampling does not know yet. An auxiliary loss on the router, or a small auxiliary predictor, decides instead whether a token is in the top-$$k$$.

</details>
{::options parse_block_html="false" /}
