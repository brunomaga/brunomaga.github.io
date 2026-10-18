---
layout: post
title:  "Mixture-of-Experts and Expert Parallelism"
categories: ["Distributed Training"]
tags: ["MoE", "expert parallelism", "load balancing", "Switch Transformer", "PyTorch"]
---

Details of [GPT-4](https://en.wikipedia.org/wiki/GPT-4) have been the subject of much speculation, including a widely shared [leak report](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/) that mentions the possible use of Mixture-of-Experts (MoEs): 16 experts of ~110B parameters each (≈1.8T total parameters). MoEs are attractive because they can increase *model capacity per FLOP*: only a small subset of experts is activated per token. Compared to dense models of similar cost, this can improve scaling, efficiency and quality.

This post covers the development of MoEs since the early days, and provides implementations of MoEs on single and multiple nodes using the PyTorch distributed API. Key concepts include routing, sparsity, load imbalance (and the corresponding loss terms), parallelism, expert capacity, numerical instabilities, overfitting and finetuning.

## Early days: small MoEs as a weighted sum of expert outputs

In 1991, the papers [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) and [Task Decomposition Through Competition in a Modular Connectionist Architecture](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1502_2) introduced some of the earliest architectures similar to what is now called an MoE.

{: .picture}
<img width="88.6%" height="88.6%" src="{{ site.assets }}/distributed-training/imgs/moe_2014_ilya.svg" alt="MoE paper 2014"/>

{: .caption}
**Left:** a Mixture of Experts. **Right:** a Deep Mixture of Experts (with 2 layers). <br/>Adapted from: [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

The MoE was defined as a set of independent **experts** (feed-forward networks, $$f_i(x)$$ in the picture) and a **gating network** (also a feed-forward network, $$g(x)$$). All experts and the gating network receive the same input $$x$$. The gating network outputs a distribution of the relevance (importance) of each expert for the given input. In its simplest form, it is defined as $$g(x) = \mathrm{Softmax}(x \cdot W_g)$$, where $$W_g$$ is an (optional) learnable transformation. The output of the system ($$z$$) is the sum of the outputs of all experts, weighted by the output of the gating network.

An MoE module is simple to implement. The router is a small feed-forward network that outputs, for every token, a probability per expert. Each expert is a regular feed-forward network:

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

The two 1991 approaches differ mainly in the loss function. One uses the error function $$E = \| d - \sum_i p_i \, o_i \| ^2$$, where $$o_i$$ is the output *vector* of expert $$i$$, $$p_i$$ is the router's probability for expert $$i$$, and $$d$$ is the correct label for the input. The other argues that this form can lead to **starvation** of experts (some experts are never assigned any importance) and **overpowering** (one expert gets most of the probability mass). It proposes instead the error function $$E = \sum_i p_i \| d - o_i \| ^2$$ (and variants based on the negative log-likelihood).

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

The authors observed an important property of MoEs:

> the learning procedure divides up a vowel discrimination task into appropriate subtasks, each of which can be solved by a very simple expert network.

## Deep Mixture of Experts

With the rise of deep neural networks, [Learning Factored Representations in a Deep Mixture of Experts (2014)](https://arxiv.org/abs/1312.4314) extended the “Mixture of Experts” to a stacked model (“Deep Mixture of Experts”) with multiple sets of gating networks and experts.

This deep MoE model is implemented by stacking several MoE layers:

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

As before, the authors observed that training led to a **degenerate local minimum**: “experts at each layer that perform best for the first few examples end up overpowering the remaining experts”. Because these experts perform better initially, the gates pick them more often. The authors overcame this by **capping (*max clipping*)** the assignments to a maximum value and re-normalizing all (post max-clipping) gating probabilities. This **cap value is then removed in a second training stage**.

The benchmark compared four models: a single MoE, a two-level MoE, a two-layer MoE where the information passed between layers is the concatenation of the outputs of all layer-1 experts (instead of their weighted sum), and a regular feed-forward network with a similar number of parameters. The results were the following: “In most cases, the deeply stacked experts perform between the single and concatenated experts baselines on the training set, as expected. However, the deep models often suffer from **overfitting**: the mixture’s error on the test set is worse than that of the single expert for two of the four model sizes. Encouragingly, the DMoE performs almost as well as a fully-connected network (DNN) with the same number of parameters, even though this network imposes fewer constraints on its structure”.

An important result shows the relevance of stacking MoEs into a deep model:

> we find that the Deep Mixture of Experts automatically learns to develop location-dependent ("where") experts at the first layer, and class-specific ("what") experts at the second layer.

This led to follow-up work that uses the gating not only to weight the sum of the experts' outputs, but also to route the input to individual experts, as described in the next section.


## Sparsely-Gated Mixture of Experts

Previous MoE implementations had a major drawback: all experts process all inputs. This leads to high training costs, even when some experts have little relevance to some inputs. The solution is **conditional computation** / **sparsity**, where only a subset of experts is picked per input. The distribution output by the gating mechanism is used to select the most relevant experts for each token, and the token is routed only to those top-$$k$$ experts. The gating thus works as a **routing mechanism**. $$k$$ is a hyper-parameter that controls the trade-off between computation and *expertise*.

As a result, the MoE may select a different combination of experts for each token in the text. When $$g_i(x)=0$$ for a given expert $$i$$, $$f_i(x)$$ is not computed. Note that, by picking only the top-$$k$$ experts per token, the number of experts can grow without a proportional increase in compute. This is the key scaling property of MoEs.

Towards better scaling, the 2017 paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) presented a recurrent (LSTM-based) language model where the feed-forward subnetwork is replaced by an MoE structure. The goal was text processing (and translation) with thousands of experts, totalling 137B parameters:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/distributed-training/imgs/moe_2017_dean.svg"/>

{: .caption}
A Mixture of Experts (MoE) layer embedded within a recurrent language model. In this case, the sparse gating function selects two experts to perform computations. Their outputs are modulated by the outputs of the gating network. Adapted from: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

### Processing and batching

**The MoE is called once for each position in the text, selecting a potentially different combination of experts at each position.** This can lead to very small effective batches per expert (**the “shrinking batch problem”**). To overcome it, the authors suggest a very large global batch size.

### Parallelism and Hierarchical MoEs

To enable parallelism, the authors combine **data and model parallelism**: they “distribute the standard layers of the model and the gating network according to conventional data-parallel schemes, but keep only one shared copy of each expert” (i.e., experts are sharded rather than replicated).

When the number of experts is very large (thousands were tested), the branching factor is reduced with a two-level **hierarchical MoE**. In a hierarchical MoE, a primary gating network chooses a sparse weighted combination of experts, and each selected expert is itself an MoE with its own gating network. In the paper's hierarchical MoE, the primary gating network uses data parallelism and the secondary MoEs use model parallelism, with each secondary MoE on one device.

### Improvements specific to language models: convolutionality 

In language tasks, the authors exploit **convolutionality**: each expert is applied in the same way to every time step. The MoE can therefore be applied to all time steps together as one large batch, which increases utilization.

### Gating and Sparsity 

Instead of the typical softmax gating $$G(x) = \mathrm{softmax}(x \cdot W_g)$$, the paper proposes **sparsity** via **noisy top-k gating**. Tunable Gaussian noise, controlled by $$W_{noise}$$, is added to the gating logits, which gives $$H(x)$$. To enforce sparsity, the logits of the experts outside the top-$$k$$ are set to $$-\infty$$ before the softmax, so that their post-softmax value is 0:

$$
G(x) = \mathrm{softmax}(\, \mathrm{KeepTopK}(\, H(x), k) \;)
$$

$$
H(x)_i = (x \cdot W_g)_i + \mathrm{StandardNormal}() \cdot \mathrm{Softplus}((x \cdot W_{noise})_i)
$$

Note: $$\mathrm{Softplus}$$ keeps the noise scale positive.

### Importance loss

To avoid overpowering and starvation of experts, the authors add an **importance loss**, equal to the square of the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) of the set of importance values:

$$
L_{importance} = w_{importance} \cdot CV(\, \mathrm{Importance}(X)\,)^2
$$

where $$CV = \sigma/\mu$$ is the ratio of the standard deviation to the mean. The **importance** of an expert is the batchwise sum of the gate values for that expert, and $$w_{importance}$$ is a hand-tuned scaling factor. As an example, consider a batch of 4 samples $$x_i$$ assigned to 5 experts $$E_j$$ with the following distribution:

|-|-|-|-|-|-|
| | $$E_1$$ | $$E_2$$ | $$E_3$$ | $$E_4$$ | $$E_5$$ |
|-|-|-|-|-|-|
| $$x_1$$ | 0.1 | 0.1 | 0 | 0.8 | 0 |
| $$x_2$$ | 0 | 0 | 0.2 | 0.7 | 0.1 |
| $$x_3$$ | 0.1 | 0 | 0 | 0.9 | 0 |
| $$x_4$$ | 0 | 0 | 0 | 1 | 0 |
| **Importance (sum)** | **0.2** | **0.1** | **0.2** | **3.4** | **0.1** |
|-|-|-|-|-|-|

In this example, the mean importance is $$\mu = 0.8$$ and the population standard deviation is $$\sigma \approx 1.30$$, so $$CV \approx 1.63$$. If all experts had similar importance, the variance and the CV would be smaller, which reduces the importance loss. 

### Load loss

The importance loss balances the overall importance across experts, but experts may still receive different numbers of tokens. This may lead to memory and compute imbalance on distributed hardware, and to undertrained experts. To address this, a second **load loss** $$L_{load}$$ encourages experts to receive a similar number of training samples. However, the hard token count is a constant and cannot be backpropagated. The authors instead use a smooth operator $$\mathrm{Load}(X)$$ that can be backpropagated:

$$
L_{load} = w_{load} \cdot CV(\, \mathrm{Load}(X)\,)^2, \quad \text{with} \quad \mathrm{Load}(X)_i = \sum_{x \in X} P(x,i)
$$

where $$w_{load}$$ is a hand-tuned scaling factor, and $$P(x,i) = Pr(H(x)_i \gt kth\_excluding(H(x), k, i))$$ is the probability that $$G(x)_i$$ is **non-zero**, given a new random choice of noise on element $$i$$, while keeping the already-sampled noise on the other elements (see Appendix A of the paper for details). Note that $$G(x)_i$$ is non-zero iff $$H(x)_i$$ is greater than the $$k$$-th greatest element of $$H(x)$$, excluding $$H(x)_i$$ itself.

### Results 

The experiments fixed a compute budget and compared a baseline model with several MoE configurations, including a hierarchical MoE, on several language tasks. The results show that:
- MoE models outperform baseline LSTM models, and hierarchical MoEs outperform non-hierarchical ones (with a higher parameter count but similar compute per example).
- Perplexity improved significantly when the data size increased (from 10B to 100B words of the Google News Corpus). On a machine translation task with the Google Production dataset, the model achieved a 1.01 higher test BLEU score, even after training for only one sixth of the time.
- In terms of scalability, the authors claim "greater than 1000x improvements in model capacity with only minor losses in computational efficiency".
- Learning to route does not work without comparing at least two experts (due to stability issues). This motivated the use of top-2 routing in that work.

Finally, the input-to-expert assignments show that **different experts specialize in different tasks based on syntax and semantics** (Appendix E, Table 9):

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

A simple Python implementation of this model is available in [this repo](https://github.com/lucidrains/mixture-of-experts).

## Conditional Computing for improved scaling

In 2020, [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) scaled Transformer-based sparsely-gated MoEs up to 600B parameters on a sequence-to-sequence (machine translation) task. The authors claim sub-linear growth of compute with model capacity, and nearly constant compilation time on large TPU pods.

GShard reuses some efficiency techniques from the previous paper. The first is **position-wise Sparsely-Gated MoEs for language tasks** (section 2.2), which deliver sub-linear scaling. The second is **parallel partitioning as a combination of data parallelism and sharding**: (1) "the attention layer is parallelized by splitting along the batch dimension and replicating its weights to all devices"; and (2) "experts in the MoE layer are infeasible to be replicated in all the devices due to its sheer size and the only viable strategy is to shard experts into many devices". 

The main improvements come from compiler optimizations with [Accelerated Linear Algebra (XLA)](https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra) and user-annotated parallelism of tensors, described next.

### Compiler and SPMD optimizations

GShard provides a **module that separates the model description from the parallel partitioning**, consisting "of a set of simple APIs for annotations, and a compiler extension in XLA for automatic parallelization." In practice, GShard "**requires the user to annotate a few critical tensors (in the model) with partitioning policies**". Developers write the module, and the runtime **automatically partitions it across compute units**. This relies on a **compiler technique for Single Program Multiple Data (SPMD) transformation** that generates a single program for all devices, so compilation time is largely independent of the device count (see Figure 2 in the paper).

The XLA SPMD partitioner for GShard "automatically partitions a computation graph
based on sharding annotations. Sharding annotations inform the compiler about how each tensor
should be distributed across devices".

### Model architecture

The model tested was a Transformer-based stack of encoder and decoder layers. The Transformer blocks (self-attention, feed-forward and, in the decoder, cross-attention) were converted into conditional computation modules by **replacing the feed-forward layer in every second block with an MoE layer (with top-2 gating)**, in both the encoder and the decoder.

{: .picture}
<img width="97.2%" height="97.2%" src="{{ site.assets }}/distributed-training/imgs/moe_gshard.svg"/>

{: .caption}
Illustration of scaling of Transformer Encoder with MoE Layers. The MoE layer replaces every other Transformer feed-forward layer. Decoder modification is similar. (a) The encoder of a standard Transformer model is a stack of self-attention and feed forward layers interleaved with residual connections and layer normalization. (b) Replacing every other feed-forward layer with an MoE layer gives the MoE Transformer Encoder. (c) When scaling to multiple devices, the MoE layer is sharded across devices, while all other layers are replicated. Adapted from: [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

### Random routing and expert capacity

The paper introduces **expert capacity**: the maximum number of tokens that can be assigned to an expert per iteration, set to $$O(N/E)$$ for $$N$$ tokens and $$E$$ experts. Tokens assigned to an expert already at maximum capacity are **overflow** tokens, and are passed to the next layer via a residual connection.

The routing mechanism uses **random routing**: in the top-2 setup, the top expert always receives the token, but the second expert receives it only with a probability related to its routing weight. Each token is therefore routed with either top-1 or top-2, depending on whether the 2nd expert is picked. The rationale is that if the 2nd expert has a very small routing weight, its contribution is negligible, so the token only needs to go to the first expert. This reduces compute and overflow.

### Auxiliary loss and load balancing

The total loss is $$L = l_{nll} + k \cdot l_{aux}$$, where $$l_{nll}$$ is the negative log-likelihood loss, $$k$$ is a constant, and $$l_{aux}$$ is an **auxiliary loss** that helps prevent a few experts from taking most tokens early in training. The auxiliary loss is defined as

$$
l_{aux} = \frac{1}{E} \sum^E_{e=1} \frac{c_e}{S} \, m_e
$$

where $$c_e/S$$ is the fraction of the input routed to each expert, and $$m_e$$ is the mean gate value of each expert. The term $$m_e$$ is needed because $$c_e/S$$ comes from the top-2 selection, which is not differentiable.

## Switch Transformers

Training instability is a major issue in large MoEs. The 2021 paper [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) addressed it by simplifying the routing algorithm and improving the model specifications. This resulted in the first large MoE models that could be trained in `bfloat16`.

> The benefit of scale was exhaustively studied in [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) which uncovered power-law scaling with model size, data set size and computational budget,  [and]
advocates training large models on relatively small amounts of data as the computationally optimal approach.
Heeding these results, we investigate a fourth axis: increase the parameter count while
keeping the floating point operations (FLOPs) per example constant.

To scale, the sparsely activated layers split *unique weights* across devices. As in GShard, the feed-forward network is replaced by an MoE. Switch Transformers use **Switch routing**: **each token is routed to only a single expert** (top-1). The authors claim that this simplification preserves model quality, while reducing routing computation and improving stability.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/moe_switch_transformers.svg"/>

{: .caption}
Illustration of a Switch Transformer encoder block. The dense feed forward network (FFN) layer of the Transformer is replaced by a sparse Switch FFN layer (light blue). The layer operates independently on the tokens in the sequence. Two tokens (<i>x</i><sub>1</sub> = “More” and <i>x</i><sub>2</sub> = “Parameters” below) are routed (solid lines) across four FFN experts; the router routes each token independently. The switch FFN layer returns the output of the selected FFN multiplied by the router gate value (dotted-line). Adapted from: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

### Expert Capacity

The paper introduces the **expert capacity factor**. The number of tokens each expert computes (its expert capacity) is the average number of tokens per expert multiplied by this capacity factor. If the capacity is exceeded (i.e. the router sends too many inputs to a given expert), the extra tokens skip the computation and are passed to the next layer via a residual connection.

{: .picture}
<img width="96.7%" height="96.7%" src="{{ site.assets }}/distributed-training/imgs/moe_switch_transformers_2.svg"/>

{: .caption}
Illustration of token routing dynamics. Each expert processes a fixed batch-size of tokens modulated by the <i>capacity factor</i>. Each token is routed to the expert with the highest router probability, but each expert has a fixed batch size of (total\_tokens / num\_experts) × capacity\_factor. If the tokens are unevenly dispatched then certain experts will overflow (denoted by dotted red lines), resulting in these tokens not being processed by this layer. A larger capacity factor alleviates this overflow issue, but also increases computation and communication costs (depicted by the empty grey slots). Adapted from: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

On the choice of the capacity factor, the paper notes:

> Increasing the capacity factor increases the quality but also increases the communication overhead and the memory in the activations. A recommended initial setup is to use an MoE with a top-2 routing with 1.25 capacity factor, with one expert per core. At evaluation time, the capacity factor can be changed to reduce compute.


### Load Balancing loss

The loss function includes a simplified version of the auxiliary loss from [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](#sparsely-gated-mixture-of-experts), which originally had separate load-balancing and importance-weighting losses. This auxiliary loss is added to the total model loss **for every Switch layer**.

Given $$N$$ experts and a batch $$B$$ with $$T$$ tokens, the auxiliary loss is the scaled dot product between the vector $$f$$, the fraction of tokens dispatched to each expert, and the vector $$P$$, the fraction of the router probability allocated to each expert:

$$
loss = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

where $$\alpha$$ is a user-defined hyper-parameter. Ideally, both $$P$$ and $$f$$ are uniform, i.e. all their values equal $$1/N$$.

### Model size reduction via distillation

A downside of these MoEs is their large number of parameters. However, distillation was shown to be feasible when a smaller network is needed: 

> Successful distillation of sparse pre-trained and specialized fine-tuned models into
small dense models. We reduce the model size by up to 99% while preserving 30% of
the quality gains of the large sparse teacher.

The paper includes further details on training techniques and experiments, omitted here for brevity. A single-device implementation of Switch Transformers is available on [this page](https://nn.labml.ai/transformers/switch/index.html). 

## Distributed implementation in PyTorch

The MoE workflow is a four-step algorithm, illustrated in the [MegaBlocks paper](https://arxiv.org/abs/2211.15841) (detailed later): 

{: .picture}
<img width="94.7%" height="94.7%" src="{{ site.assets }}/distributed-training/imgs/moe_megablocks.svg"/>

{: .caption}
<b>A Mixture-of-Experts Layer.</b> Shown for <i>num\_experts=3</i>, <i>top\_k=1</i> and <i>capacity\_factor=1</i> with the prevalent, token dropping formulation. <b>First (1)</b>, tokens are mapped to experts by the router. Along with expert assignments, the router produces probabilities that reflect the confidence of the assignments. <b>Second (2)</b>, the feature vectors are permuted to group tokens by expert assignment. If the number of tokens assigned to an expert exceeds its capacity, extra tokens are dropped. <b>Third (3)</b>, the expert layers are computed for the set of tokens they were assigned as well as any padding needed for unused capacity. <b>Lastly (4)</b>, the results of the expert computation are un-permuted and weighted by the router probabilities. The outputs for dropped tokens are shown here set to zero. Adapted from: [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)

The tokens travel to the experts and back via all-to-all collectives, and their gradients follow the same path in reverse during the backward pass. Since `dist.all_to_all_single` has no backward pass, it is wrapped in an autograd function whose backward pass runs the opposite all-to-all:

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

The distributed mixture-of-experts block `MoE_dist` has a replicated (data-parallel) router and a local expert. For simplicity, there is a single expert per GPU, and tokens that exceed the capacity of an expert are dropped:

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

The `Router` is a feed-forward network that takes the embedding of a token as input and outputs a distribution of allocations (probabilities) over the experts. Since there is only one expert per process, the process id matches the expert id, and the number of experts equals the number of processes. 

The **routing step** computes the top-`k` expert assignments of each token: the expert ids in `topk_experts` and the corresponding probabilities in `topk_probs`: 
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

The **permutation step** sends each token to the processor of its expert, using collective communication in two steps. First, an `all_to_all` exchanges the number of items to send and receive. Then, based on those counts, an `all_to_all_single` call exchanges the metadata of the tokens, and an `AllToAll` call exchanges the tokens themselves: 

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
<img width="70.7%" height="70.7%" src="{{ site.assets }}/matrix-transpose/all_to_all.svg"/>

{: .caption}
An illustration of the MPI all-to-all collective operation. When each process sends one element to each process, the operation is equivalent to a distributed matrix transpose: the row framed in red before the all-to-all (left), the data of process A, becomes the first column after the all-to-all (right). A single-element all-to-all exchanges the number of items to be sent and received among processes. These counts are then used by two all-to-alls with variable-sized elements in the permutation step, one for the token metadata and one for the tokens. In the un-permutation step, another all-to-all performs the reverse communication by swapping the send and receive counts and buffers. Adapted from: [MPI Tutorial, Shao-Ching Huang, IDRE High Performance Computing Workshop, 2013](https://www.slideshare.net/slideshow/mpi-49244468/49244468)

The **computation step** rearranges the received data into a 3D batch of size `Rows x Capacity x Embedding`, cropping or padding each sequence to the capacity of the expert:

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

The output of the expert is stored in `batch_toks`. The **un-permutation step** then performs an `AllToAll` with the opposite communication of the permutation step. This fills the all-to-all output buffer with the partial results of all experts.

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

Finally, the **scale step** computes, for each token, the sum of the outputs of its top-k experts, weighted by the router probabilities. The outputs arrive grouped by expert, so each output is added at the position of its token, given by the metadata `(b, s)` sent in the permutation step:

```python
    x = x * torch.cat(probs_per_expert).view(-1,1) # multiply by router probabilities
    rows = (send_ids[:,0] - global_rank*B) * S + send_ids[:,1] # position b*S+s of each output, without the batch offset
    out = torch.zeros(B*S, E, dtype=x.dtype, device=device).index_add(0, rows, x) # sum the top-k outputs of each token
    return out.view(B,S,E)
```


### Applying the MoE to an existing LLM

The MoE module can replace the feed-forward module in existing architectures. This section applies it to the [GPT2-based `GPTlite` model introduced in a previous post]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}). All modules are distributed with Distributed Data Parallel, except the feed-forward module of the transformer block, which is replaced by the model-parallel mixture of experts:

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

The rest of the training and optimization algorithm is unchanged.

Writing all communication steps from scratch requires some HPC background. Libraries that implement these steps can be used instead, such as [DeepSpeed Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/).

## Further reading

MoE finetuning, scaling and model variants are active research topics. Some useful pointers follow:

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2022 [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)</summary>
MegaBlocks is "a system for efficient Mixture-of-Experts (MoE) training on GPUs", which addresses the **model quality vs hardware efficiency trade-off** in the dynamic routing of MoE layers. The load-imbalanced computation in MoEs forces a choice between (1) dropping tokens from the computation and (2) spending computation and memory on padding. This trade-off is usually set via hyperparameter tuning. However, "The loss reached by the MoE models decreases significantly as expert capacity is increased, but at the cost of additional computation" and "the lowest loss is achieved by the 'max expert capacity value', which avoids dropping tokens through the dynamic
capacity factor mechanism proposed by [Adaptive
mixture-of-experts at scale (2022)](https://arxiv.org/abs/2206.03382)". Not dropping tokens is therefore ideal.

A second motivation is that "the sparse computation in MoEs does not map cleanly to the software primitives supported in major frameworks and libraries". 
To overcome these limitations, MegaBlocks "reformulates MoE computation in terms of block-sparse operations and develops new block-sparse GPU kernels that efficiently handle the dynamism present in MoEs".
It proposes an MoE routing approach that is based on **sparse primitives, maps efficiently to GPUs and never drops tokens**, "enabling end-to-end training speedups of up to 40% and 2.4× over state of the art". This is achieved with block-sparse operations in the MoE layers, which accommodate the imbalanced assignment of tokens to experts. The resulting models are called **dropless-MoEs (dMoEs)**, in contrast with regular MoEs, which use batched matrix multiplication. The dropless-MoE kernels use blocked-CSR-COO encoding and transpose indices to enable efficient matrix products with sparse inputs.
</details>
{::options parse_block_html="false" /}



{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)</summary>
This paper formally studies "how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model". It experiments on a 4-cluster vision classification task and claims that: (1) "cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE"; (2) the experts must be non-linear for an MoE system to work; and (3) "the router can learn the cluster-center features, which helps divide the input complex problem into simpler linear classification sub-problems that individual experts can conquer".
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)</summary>
The paper introduces GLaM (Generalist Language Model), a family of dense and sparse decoder-only language models, "which uses a sparsely activated mixture-of-experts architecture to scale the model capacity while also incurring substantially less training cost compared to dense variants". GLaM has 1.2 trillion parameters (7x larger than GPT-3's 175B), but requires 1/3 of the energy for training and half of the computation FLOPs for inference, while "achieving better overall zero, one and few-shot performance across 29 NLP tasks" (Table 1, Figure 1).

Several variants of GLaM were trained to compare the behavior of MoE and dense models on the same training data (Table 4). The study highlights that:
- "the quality of the pretrained data also plays a critical role, specifically, in the performance of downstream tasks" (section 6.2);
- sparse models scale better (section 6.3);
- "MoE models require significantly less data than dense models of comparable FLOPs to achieve similar zero, one, and few-shot performance" (section 6.4).
</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)</summary>
Presents a study with experiments, results and techniques on the stability of training and fine-tuning of very large sparse MoEs. Also introduces the **router z-loss**, which resolves instability issues and slightly improves model quality, and a 269B sparse model, **the Stable Transferable Mixture-of-Experts** or ST-MoE-32B, which achieves state-of-the-art performance across several NLP benchmarks. The z-loss is defined as:

$$
L_z(x) = \frac{1}{B} \sum_{i=1}^B \left( \log \sum_{j=1}^N e^{x_j^{(i)}} \right)^2
$$

where $$B$$ is the number of tokens, $$N$$ is the number of experts, and $$x \in \mathbb{R}^{B\times N}$$ are the logits going into the router. The z-loss penalizes large logits going into the gating network. The paper also contains an extensive analysis of quality-stability trade-off techniques and results:

- "Sparse models often suffer from training instabilities (Figure 1) worse than those observed in standard densely-activated Transformers. [...] Some architectural improvements involve more multiplications than additions or do not sum many items at once": the paper studies GELU Gated Linear Units (GEGLU) and Root Mean Square Scale Parameters.
- Adding noise (input-jitter) and dropout improves stability, but leads to a significant loss of model quality (Section 3.2).
- Gradient clipping in updates (in the Adafactor optimizer) improves stability, but at a high cost in quality (Section 3.3).
- Sparse expert models are sensitive to roundoff errors (e.g. from lower-precision representations) because the routers add more exponential functions (Section 3.4).
- Sparse models are prone to overfitting (Fig. 3).
- Dropout at 0.1 gives modest improvements, while higher rates reduce performance. Selectively increasing the dropout of the experts increases accuracy (Fig. 4).
- Updating only the non-MoE parameters (while freezing the others) yields an accuracy similar to updating all parameters, and updating only the FFN parameters works slightly better (Fig. 5).
- "Sparse models benefit from noisier hyperparameters including small batch sizes and high learning rates. Dense models behave nearly oppositely" (Fig. 6).
- "Sparse models are robust to dropped tokens when fine-tuning. [...] dropping 10-15% of tokens can perform approximately as well as models that drop < 1%" (Table 5).

The paper makes 3 recommendations for designing sparse systems:
> 
- top-2 routing with 1.25 capacity factor and at most one expert per core.
- the capacity factor can be changed during evaluation to adjust to new memory/compute requirements.
- dense layer stacking and a multiplicative bias can boost quality (Appendix C).

Regarding how MoEs work, the authors show that **encoder experts learn very shallow tasks**, e.g. punctuation, nouns, etc. In other words, **MoEs subdivide a problem into smaller problems that can be solved by different experts combined**. This is shown by "visualizing how tokens are routed among (encoder) experts, [...] by passing a batch of tokens to the model and manually inspecting token assignment at each layer". At each layer, at least one expert specializes in **sentinel tokens** (mask tokens that represent blanks to fill in), and some encoder experts show clear specialization, operating mainly on punctuation, verbs, proper names, counting, etc. Details are in Tables 13 and 15.

For the decoder, the authors note that "expert specialization is far less noticeable in the decoder", and no meaningful specialization (semantics or syntax) is visible in decoder experts:
  > We hypothesize that this lack of meaningful expert specialization is caused by the distribution of
target tokens induced by the span corruption objective. In particular, (a) a smaller number of tokens
are routed jointly in the decoder due to longer sequence lengths in the encoder (e.g. group size
is 2048 in the encoder vs 456 in the decoder in our setup) and (b) a higher proportion of tokens
are sentinel tokens in the decoder. As a result, target tokens in each group typically cover a smaller
semantic space (compared to the encoder), perhaps explaining the lack of expert specialization in the
decoder.
</details>
{::options parse_block_html="false" /}



{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2022 [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382)</summary>
According to the authors, existing sparsely-gated mixture-of-experts (MoE) systems "suffer inefficient computation due to their
static execution, namely static parallelism and pipelining, which does not adapt to the dynamic workload". To address this, the authors present Tutel, a system with two main novelties: adaptive parallelism for optimal expert execution, and adaptive pipelining for the inefficient and non-scalable dispatch/combine operations in MoE layers.

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
This paper claims that **instruction tuning supplemented by further fine-tuning on individual downstream tasks outperforms fine-tuning or instruction tuning alone**. To show this, the authors compare single-task fine-tuning, multi-task instruction tuning, and multi-task instruction tuning followed by single-task fine-tuning. MoEs benefit more from instruction tuning than other models, and benefit more from a higher number of tasks. In practice:
  > in the absence of instruction tuning, MoE models fall short in performance when compared to dense models on downstream tasks. [...] When supplemented with instruction tuning, MoE models exceed the performance of dense models on downstream tasks, as well as on held-out zero-shot and few-shot tasks.
</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2023 [From sparse to soft mixture of experts](https://arxiv.org/abs/2308.00951)</summary>

Soft MoE is a fully-differentiable sparse Transformer. Regular MoEs use a sparse
and discrete router that searches for a good hard assignment between tokens and experts. Soft MoEs instead perform a soft assignment by mixing tokens.

Soft MoE computes several weighted
averages of all tokens (with weights that depend on both tokens and experts), and each expert processes its corresponding weighted average.

It addresses the main challenges of current sparse MoEs: training instability, token dropping, inability to scale the number of experts, and ineffective finetuning.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/soft_moes.svg"/>

{: .caption}
<b>Sparse and Soft MoE layers.</b> While the router in Sparse MoE layers (left) learns to <i>assign</i> individual input tokens to each of the available slots, in Soft MoE layers (right) each slot is the result of a (different) <i>weighted average</i> of all the input tokens. Learning to make discrete assignments introduces several optimization and implementation issues that Soft MoE sidesteps. Adapted from: [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)

In the benchmark, Soft MoEs outperform dense Transformers (ViTs) and popular MoEs (Token Choice and Experts Choice) by a large margin on a vision task, with shorter training time and faster inference. 

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2024 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)</summary>
Mixtral 8x7B, the current state-of-the-art MoE model, is a sparse mixture of experts with the same architecture as Mistral 7B. The differences are that it supports a fully dense context length of 32k tokens, and its feed-forward blocks are replaced by a mixture of 8 feed-forward network experts. The architecture is detailed in Table 1. It uses top-2 routing, and because tokens can be allocated to different experts, "each token has access to 47B parameters, but only uses 13B active parameters during inference". 

The model formulation, in section 2.1, matches the GShard architecture (except that the MoE is in every layer instead of every second one): a sparse top-$$k$$ MoE with gating $$G(x) = \mathrm{Softmax}(\mathrm{TopK}(x \cdot W_g))$$, where $$\mathrm{TopK}(\ell)_i = -\infty$$ if the logit $$\ell_i$$ is not among the top-$$k$$. As before, "in a Transformer model, the MoE layer is applied independently per token and replaces the feed-forward (FFN) sub-block of the transformer block". Mixtral uses the [SwiGLU activation function](https://arxiv.org/abs/2002.05202v1) (as Llama-2), so for a given input $$x$$, the output is:

$$
y = \sum_{i=0}^{n-1} \mathrm{Softmax}(\mathrm{Top2}(x \cdot W_g))_i\cdot \mathrm{SwiGLU}_i(x).
$$
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2024 [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)</summary>

MoEs subdivide the problem domain into smaller problems, each solved by a different expert. The authors of this paper claim that not all problems require the same amount of effort to train a model accurately. They therefore use **conditional computation**, a technique that "tries to reduce total compute by expending it only when needed". Here, "the network must learn how to dynamically allocate the available compute by making decisions per-token, in each layer, about where to spend compute from the available budget" - where the compute budget is a total number of FLOPs.

For every token, the model decides to either apply computation (as in the standard transformer) or skip it and pass the token through a residual connection (leaving it unchanged and saving compute). Unlike in MoEs, the **routing is applied to both the feedforward network and the multi-head attention**. In multi-head attention routing, the router decides not only which tokens to update, but also which tokens are available to attend to. "We refer to this strategy as Mixture-of-Depths (MoD) to emphasize how individual tokens pass through different numbers of layers, or blocks, through the depth of the transformer". During training, for every input, the router produces a scalar weight (importance) per token. The gating picks the top-$$k$$ tokens per sentence per block to participate in the transformer block computation. $$k$$ is a hyper-parameter that defines the maximum number of tokens passed to a block, so the computation graph and tensor sizes remain static throughout the execution. The value of $$k$$ sets the trade-off between performance and speed for a given compute budget.

{: .picture}
<img width="95.3%" height="95.3%" src="{{ site.assets }}/distributed-training/imgs/mixture_of_depths.svg"/>

{: .caption}
<b>Mixture-of-Depths Transformer.</b> As in mixture-of-experts (MoE) transformers, a router chooses among potential computational paths. But unlike in MoE transformers the possible choices are a standard block’s computation (i.e., self-attention and MLP) or a residual connection. Since some tokens take this second route, Mixture-of-Depths (MoD) transformers have a smaller total FLOP footprint compared to vanilla or MoE transformers. On the top right is depicted a trained model’s routing decisions for a short sequence truncated to 64 tokens for visualization purposes. When examining the choices one can find tokens processed by later blocks’ layers, despite passing through relatively few total blocks throughout the model’s depth. This is a unique feature of MoD compared to conventional halting-based, or "early-exit" conditional computation, which instead engage blocks serially, or vanilla transformers, which engage every block. Adapted from: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

The main challenge is the routing scheme. The authors considered (1) token-choice routing, where a router produces per-token probability distributions across computational paths (experts), and the orthogonal approach, (2) expert-choice routing, where each expert picks the top-$$k$$ tokens based on the tokens’ preferences, instead of tokens choosing their preferred expert. They adopted expert-choice routing because it does not require load balancing, and "routers can try to ensure that the most critical tokens are among the
top-$$k$$ by setting their weight appropriately".

{: .picture}
<img width="99.4%" height="99.4%" src="{{ site.assets }}/distributed-training/imgs/mixture_of_depths_2.svg"/>

{: .caption}
<b>Routing schemes.</b> Tokens are funnelled to the computational path of their choice when using token-choice routing (left). If a given path exceeds its capacity (e.g., more than two tokens in this example) then surplus tokens must be dropped (purple token). The exact token that is ultimately dropped depends on the precise implementation in the underlying code. For example, priority is often given to those tokens that come earlier in the sequence or batch order. With expert-choice routing (middle), precisely <i>k</i> (in this case, two) tokens are chosen per path using a top-<i>k</i> mechanism across the tokens’ router weights. Here, tokens are dropped if they are not among the top-<i>k</i> with respect to any given path (orange token), and some tokens may even be funnelled to multiple paths (yellow token). Mixture-of-Depths uses expert-choice routing with a single path (right). Since tokens are dropped whenever <i>k</i> is less than the sequence length, the dropped tokens are routed around the self-attention and MLP computations, and a forward pass spends fewer FLOPs. Adapted from: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

At the time of writing, this work is very recent, and it is not yet clear whether MoDs will be useful for the general use case.

</details>
{::options parse_block_html="false" /}
