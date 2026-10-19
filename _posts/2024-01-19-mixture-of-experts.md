---
layout: post
title:  "Mixture-of-Experts and Expert Parallelism"
categories: ["Distributed Training"]
tags: ["MoE", "expert parallelism", "load balancing", "Switch Transformer", "PyTorch"]
---

Details of [GPT-4](https://en.wikipedia.org/wiki/GPT-4) have been the subject of much speculation and reporting, including a widely shared [leak report](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/) that mentions the possible use of [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoEs): 16 experts of ~110B parameters each (≈1.8T total parameters). In practice, MoEs are attractive because they can increase *model capacity per FLOP*: only a small subset of experts is activated per token, which can improve scaling, efficiency, and quality compared to similarly expensive dense models.

So this post discusses the development of MoEs since the early days, and provides implementations of MoEs on single and multiple nodes using the PyTorch distributed API. It covers key concepts such as routing, sparsity, load imbalance (and the corresponding loss terms), parallelism, expert capacity, numerical instabilities, overfitting and finetuning.

## Early days: small MoEs as a weighted sum of expert outputs

In 1991, the papers [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) and [Task Decomposition Through Competition in a Modular Connectionist Architecture](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1502_2) introduced some of the earliest architectures similar to what is now called an MoE.

{: .picture}
<img width="88.6%" height="88.6%" src="{{ site.assets }}/distributed-training/imgs/moe_2014_ilya.svg" alt="MoE paper 2014"/>

{: .caption}
**Left:** a Mixture of Experts. **Right:** a Deep Mixture of Experts (with 2 layers). <br/>Adapted from: [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

The MoE was defined as a set of independent **experts** (feed-forward networks, $$f_i(x)$$ in the picture) alongside a **gating network** (also a feed-forward network, $$g(x)$$). All the experts and the gating network receive the same input $$x$$. The gating network outputs a distribution of each expert's relevance/importance for the given input and can be defined in its simplest form as $$g(x) = \mathrm{Softmax}(x \cdot W_g)$$, where $$W_g$$ is a learnable transformation. Finally, the output of the system ($$z$$) is the sum of the outputs of all experts weighted by the output of the gating network.

The implementation of an MoE module is straightforward. The router is a small feed-forward network that outputs, for every token, a probability per expert, and each expert is a regular feed-forward network:

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

A major difference between the two 1991 approaches is the loss function. One uses the error function $$E = \| d - \sum_i g_i(x) \, f_i(x) \| ^2$$, where $$f_i(x)$$ is the output *vector* of expert $$i$$, $$g_i(x)$$ is the router's probability for expert $$i$$, and $$d$$ is the correct label for the input. The other argues that this form makes the experts *cooperate*: each expert must cancel the residual error left by the others, so many experts are used for each input. It suggests instead the error function $$E = \sum_i g_i(x) \| d - f_i(x) \| ^2$$ (and variants based on the negative log-likelihood), in which each expert produces the whole output, so that the experts *compete* and a single expert tends to be used for each input. The following code implements the second error function:

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

The authors made an important observation about the behavior of MoEs:

> the learning procedure divides up a vowel discrimination task into appropriate subtasks, each of which can be solved by a very simple expert network.

## Deep Mixture of Experts

With the rise of deep neural networks, [Learning Factored Representations in a Deep Mixture of Experts (2014)](https://arxiv.org/abs/1312.4314) extended the “Mixture of Experts” to a stacked model (“Deep Mixture of Experts”) with multiple sets of gating networks and experts.

This deep MoE model can be coded simply by stacking several MoE layers:

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

The authors noticed that training led to a **degenerate local minimum**: “experts at each layer that perform best for the first few examples end up overpowering the remaining experts”, because, being better initially, they end up being picked more often by the gates. This was overcome by **capping (*max clipping*)** the running sum of the assignments of each expert to a maximum value (the mean over all experts plus a margin): above it, the expert gets a gate value of 0, and the remaining gating probabilities are re-normalized. This **cap value is then removed in a second training stage**.

The benchmark compared a single MoE, a two-level MoE, two-layer baselines whose second layer is either a single expert or the concatenation of the outputs of all its experts (instead of the weighted sum of outputs), and a regular feed-forward network with the same number of parameters. Results showed the following: “In most cases, the deeply stacked experts perform between the single and concatenated experts baselines on the training set, as expected. However, the deep models often suffer from **overfitting**: the mixture’s error on the test set is worse than that of the single expert for two of the four model sizes. Encouragingly, the DMoE performs almost as well as a fully-connected network (DNN) with the same number of parameters, even though this network imposes fewer constraints on its structure”.

An important result emphasizes the relevance of stacking MoEs into a deep model:

> we find that the Deep Mixture of Experts automatically learns to develop location-dependent ("where") experts at the first layer, and class-specific ("what") experts at the second layer.

This led to follow-up work that explored using the gating not only to weight the sum of the experts' outputs, but also to route the input to individual experts, as described in the following section.


## Sparsely-Gated Mixture of Experts

The previous MoE implementations had a big drawback: all experts process all inputs, which leads to high training costs, even when some experts are of little relevance to some inputs. This led to **conditional computation** / **sparsity**, where only a subset of experts is picked per input. In practice, the distribution output by the gating mechanism is now used to pick only the most relevant experts for each token and to route the token only to those top-$$k$$ experts, effectively working as a **routing mechanism**. $$k$$ is a hyper-parameter that drives the trade-off between computation and *expertise*.

This way, the MoE may select a different combination of experts for each token in the text. When $$g_i(x)=0$$ for a given expert $$i$$, $$f_i(x)$$ is not computed. Note that, by picking only the top-$$k$$ experts per token, one can increase the number of experts without a proportional increase in compute, which is the key scaling property of MoEs.

Towards better scaling, the 2017 paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) presented a recurrent ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)-based) language model where an MoE structure was inserted between stacked LSTM layers. The objective was to do text processing (and translation) with thousands of experts, totalling up to 137B parameters:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/distributed-training/imgs/moe_2017_dean.svg"/>

{: .caption}
A Mixture of Experts (MoE) layer embedded within a recurrent language model. In this case, the sparse gating function selects two experts to perform computations. Their outputs are modulated by the outputs of the gating network. Adapted from: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

### Processing and batching

**The MoE is called once for each position in the text, selecting a potentially different combination of experts at each position.** Because this can lead to very small effective per-expert batches (**the “shrinking batch problem”**), they suggest using a very large global batch size to overcome it.

### Parallelism and Hierarchical MoEs

To enable parallelism, the authors implement **data and model parallelism**: they “distribute the standard layers of the model and the gating network according to conventional data-parallel schemes, but keep only one shared copy of each expert” (i.e., experts are sharded rather than replicated, so that each expert receives a combined batch with the relevant tokens of all devices). This scheme is now called **expert parallelism**.

When the number of experts is very large (they tested thousands), they also reduce the branching factor by using a two-level **hierarchical MoE**. A hierarchical MoE is a structure where a primary gating network chooses a sparse weighted combination of experts, and each selected expert is itself an MoE with its own gating network. In the case of their hierarchical MoE, the primary gating network employs data parallelism, and the secondary MoEs employ model parallelism; each secondary MoE resides on one device.

### Improvements specific to language models: convolutionality 

In language tasks, they take advantage of **convolutionality**: each expert is applied in the same way to every time step. This allows the MoE to be applied to all time steps together as one big batch, increasing effective utilization.

### Gating and Sparsity 

Instead of the typical softmax gating $$g(x) = \mathrm{Softmax}(x \cdot W_g)$$, the paper proposes **sparsity** via **noisy top-k gating**, where they add tunable [Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) noise, controlled by $$W_{noise}$$, to the gating logits, which gives $$H(x)$$. To enforce sparsity, the logits of the non-top-$$k$$ experts are set to $$-\infty$$ before the softmax, so that their post-softmax value is 0:

$$
G(x) = \mathrm{softmax}(\, \mathrm{KeepTopK}(\, H(x), k) \;)
$$

$$
H(x)_i = (x \cdot W_g)_i + \mathrm{StandardNormal}() \cdot \mathrm{Softplus}((x \cdot W_{noise})_i)
$$

Note: $$\mathrm{Softplus}$$ keeps the noise scale positive, and the noise helps with load balancing (see the load loss below).

### Importance loss

To avoid **overpowering** (a few experts get most of the probability mass) and **starvation** of experts (some experts are never assigned any importance), they add an **importance loss**, equal to the square of the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) of the set of importance values:

$$
L_{importance} = w_{importance} \cdot CV(\, \mathrm{Importance}(X)\,)^2
$$

where $$CV = \sigma/\mu$$ is the ratio of the standard deviation to the mean. The **importance** of an expert is computed as the batchwise sum of the gate values for that expert, and $$w_{importance}$$ is a hand-tuned scaling factor. As an example, take a batch of 4 samples $$x_i$$ assigned across 5 experts $$E_j$$, with the following distribution:

|-|-|-|-|-|-|
| | $$E_1$$ | $$E_2$$ | $$E_3$$ | $$E_4$$ | $$E_5$$ |
|-|-|-|-|-|-|
| $$x_1$$ | 0.1 | 0.1 | 0 | 0.8 | 0 |
| $$x_2$$ | 0 | 0 | 0.2 | 0.7 | 0.1 |
| $$x_3$$ | 0.1 | 0 | 0 | 0.9 | 0 |
| $$x_4$$ | 0 | 0 | 0 | 1 | 0 |
| **Importance (sum)** | **0.2** | **0.1** | **0.2** | **3.4** | **0.1** |
|-|-|-|-|-|-|

In this example, the mean of the importance is $$\mu = 0.8$$ and the population standard deviation is $$\sigma \approx 1.30$$, thus $$CV \approx 1.63$$. If experts were assigned similar importance, the variance would be smaller and the CV also smaller, thus reducing the importance loss. 

### Load loss

The importance loss tries to balance the overall importance across experts, but experts may still receive different numbers of tokens (e.g., one expert may receive a few tokens with large gate values, and another many tokens with small ones). This may lead to memory and compute imbalance on distributed hardware, and to undertrained experts. To solve this, a second **load loss** $$L_{load}$$ is introduced, which encourages experts to receive a similar number of training samples. However, the hard token count is a discrete quantity and cannot be backpropagated, so instead they use a smooth operator $$\mathrm{Load}(X)$$ that can be backpropagated:

$$
L_{load} = w_{load} \cdot CV(\, \mathrm{Load}(X)\,)^2, \quad \text{with} \quad \mathrm{Load}(X)_i = \sum_{x \in X} P(x,i)
$$

where $$w_{load}$$ is a hand-tuned scaling factor, and $$P(x,i) = Pr(H(x)_i \gt kth\_excluding(H(x), k, i))$$ denotes the probability that $$G(x)_i$$ is **non-zero**, given a new random choice of noise on element $$i$$, but keeping the already-sampled choices of noise on the other elements (see Appendix A of the paper for details). Note that $$G(x)_i$$ is non-zero iff $$H(x)_i$$ is greater than the $$k$$-th greatest element of $$H(x)$$, excluding $$H(x)_i$$ itself.

### Results 

The experiments fixed a compute budget and compared a baseline model with several configurations of MoE, including a hierarchical MoE, on several language tasks. Results show that:
- MoE models beat baseline LSTM models, with hierarchical MoEs beating non-hierarchical ones (with a higher parameter count but similar compute per example).
- [Perplexity](https://en.wikipedia.org/wiki/Perplexity) improved significantly when increasing the data size (from 10B to 100B words of the Google News Corpus). In a machine translation task, on the Google Production dataset, the model achieved a 1.01 higher test [BLEU](https://en.wikipedia.org/wiki/BLEU) score even after training for only one sixth of the time.
- In terms of scalability, the authors claim "greater than 1000x improvements in model capacity with only minor losses in computational efficiency".
- Learning to route does not work without the ability to compare at least two experts (with $$k = 1$$, the gate value of the selected expert is always 1, so it has no gradient), motivating routing to $$k > 1$$ experts in that work.

Finally, they look at input-to-expert assignments, which show that **different experts specialize in different tasks based on syntax and semantics** (Appendix E, Table 9):

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

This post does not provide an implementation, but for the record, a simple Python implementation of this model is available in [this repo](https://github.com/lucidrains/mixture-of-experts).

## Conditional Computing for improved scaling

In 2020, [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) explored the scaling of Transformer-based sparsely-gated MoEs up to 600B parameters on a sequence-to-sequence (machine translation) task. They claim sub-linear compute growth with model capacity and essentially constant compilation time on large [TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) pods.

It reuses some efficiency techniques from the previous paper. The first is **position-wise Sparsely-Gated MoEs for language tasks** (section 2.2), which deliver sub-linear scaling. The second is **parallel partitioning as a combination of data parallelism and sharding**: (1) "the attention layer is parallelized by splitting along the batch dimension and replicating its weights to all devices"; and (2) "experts in the MoE layer are infeasible to be replicated in all the devices due to its sheer size and the only viable strategy is to shard experts into many devices". 

The main improvements come from compiler optimizations with [Accelerated Linear Algebra (XLA)](https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra) and user-flagged parallelism of tensors, as described next.

### Compiler and SPMD optimizations

GShard provides a **module that separates the model description from the parallel partitioning**, consisting "of a set of simple APIs for annotations, and a compiler extension in XLA for automatic parallelization." In practice, GShard "**requires the user to annotate a few critical tensors in the model with partitioning policies**". This allows developers to write the module, while the compiler **automatically partitions it across compute units**. This enables a **compiler technique for [Single Program Multiple Data](https://en.wikipedia.org/wiki/Single_program,_multiple_data) (SPMD) transformation** that generates a single program to run on all devices, keeping compilation time largely independent of the device count (see Figure 2 in the paper).

The XLA SPMD partitioner for GShard "automatically partitions a computation graph
based on sharding annotations. Sharding annotations inform the compiler about how each tensor
should be distributed across devices".

### Model architecture

The model tested was a Transformer-based stack of encoder and decoder layers. The Transformer blocks (self-attention, feed-forward and, in the decoder's case, cross-attention) were converted into conditional computation modules by **replacing the feed-forward layer in every second block with an MoE layer (with top-2 gating)**, in both the encoder and the decoder.

{: .picture}
<img width="97%" height="97%" src="{{ site.assets }}/distributed-training/imgs/moe_gshard.svg"/>

{: .caption}
Illustration of scaling of Transformer Encoder with MoE Layers. The MoE layer replaces every other Transformer feed-forward layer. Decoder modification is similar. (a) The encoder of a standard Transformer model is a stack of self-attention and feed forward layers interleaved with residual connections and layer normalization. (b) Replacing every other feed-forward layer with an MoE layer gives the MoE Transformer Encoder. (c) When scaling to multiple devices, the MoE layer is sharded across devices, while all other layers are replicated. Adapted from: [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

### Random routing and expert capacity

The paper introduces the concept of **expert capacity**, which is the maximum number of tokens that can be assigned to an expert per iteration, set to be of order $$O(T/E)$$ for $$T$$ tokens and $$E$$ experts. Tokens whose selected experts are all at maximum capacity are considered **overflow** tokens and are passed to the next layer via a [residual connection](https://en.wikipedia.org/wiki/Residual_neural_network).

The routing mechanism applies **random routing**: in the top-2 setup, the top expert always receives the token, but the second expert only receives it with a probability proportional to its routing weight. In other words, it is effectively a top-1 or top-2 setup, depending on whether the 2nd expert is picked. The rationale is that if the 2nd expert has a very small routing weight, its contribution is negligible, so the token only needs to be passed to the first expert, which reduces compute and overflow.

### Auxiliary loss and load balancing

The total loss is described as $$L = l_{nll} + \alpha \cdot l_{aux}$$, where $$l_{nll}$$ is the negative log-likelihood loss, $$\alpha$$ is a constant, and $$l_{aux}$$ is an **auxiliary loss** that helps prevent a few experts from stealing most tokens early in training. This auxiliary loss is formulated as

$$
l_{aux} = \frac{1}{E} \sum^E_{e=1} \frac{c_e}{S} \, m_e
$$

where the term $$c_e/S$$ represents the fraction of the input routed to each expert (out of the $$S$$ tokens of a group, as GShard routes the tokens of a batch in independent groups), and $$m_e$$ is the mean gate value of each expert. The loss approximates the mean square of $$c_e/S$$, which is smallest when the tokens are spread evenly across the experts (note that $$m_e$$ replaces one factor $$c_e/S$$ because $$c_e/S$$ comes from the top-2 selection, which is not differentiable).

## Switch Transformers

A big issue with large MoEs is training instability. With that in mind, the 2021 paper [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) tackled this problem by simplifying the routing algorithm and improving the model specifications, delivering the first large MoE models that could be trained in [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).

> The benefit of scale was exhaustively studied in [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) which uncovered power-law scaling with model size, data set size and computational budget,  [and]
advocates training large models on relatively small amounts of data as the computationally optimal approach.
Heeding these results, we investigate a fourth axis: increase the parameter count while
keeping the floating point operations (FLOPs) per example constant.

To achieve scaling, the sparsely activated layers split *unique weights* across devices. As in GShard, the authors replace the feed-forward network with an MoE. In Switch Transformers, they use **Switch routing**: **each token is routed to only a single expert** (top-1), whose output is multiplied by its router probability (a softmax over all experts, so the router still receives a gradient). The claim is that this simplification preserves model quality while reducing routing computation and performing better. Compared to the top-2 routing of GShard, the capacity of each expert can also be at least halved, and the communication costs are reduced.

{: .picture}
<img width="96.3%" height="96.3%" src="{{ site.assets }}/distributed-training/imgs/moe_switch_transformers.svg"/>

{: .caption}
Illustration of a Switch Transformer encoder block. The dense feed forward network (FFN) layer of the Transformer is replaced by a sparse Switch FFN layer (light blue). The layer operates independently on the tokens in the sequence. Two tokens (<i>x</i><sub>1</sub> = “More” and <i>x</i><sub>2</sub> = “Parameters” below) are routed (solid lines) across four FFN experts; the router routes each token independently. The switch FFN layer returns the output of the selected FFN multiplied by the router gate value (dotted-line). Adapted from: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

### Expert Capacity

They introduce the **expert capacity factor**: the number of tokens that each expert computes (its expert capacity) is the average number of tokens per expert times this capacity factor. If the capacity is exceeded (i.e. if the router sends too many inputs to a given expert), the extra tokens skip the computation and are instead passed to the next layer via a residual connection.

{: .picture}
<img width="96.9%" height="96.9%" src="{{ site.assets }}/distributed-training/imgs/moe_switch_transformers_2.svg"/>

{: .caption}
Illustration of token routing dynamics. Each expert processes a fixed batch-size of tokens modulated by the <i>capacity factor</i>. Each token is routed to the expert with the highest router probability, but each expert has a fixed batch size of (total\_tokens / num\_experts) × capacity\_factor. If the tokens are unevenly dispatched then certain experts will overflow (denoted by dotted red lines), resulting in these tokens not being processed by this layer. A larger capacity factor alleviates this overflow issue, but also increases computation and communication costs (depicted by the empty grey slots). Adapted from: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

On the choice of the capacity factor, the [ST-MoE paper](https://arxiv.org/abs/2202.08906) (detailed later) notes that increasing the capacity factor increases the quality but also increases the communication overhead and the memory in the activations. A recommended initial setup is to use an MoE with a top-2 routing with 1.25 capacity factor, with at most one expert per core. At evaluation time, the capacity factor can be changed to reduce compute.


### Load Balancing loss

The loss function includes a simplified version of the auxiliary loss from [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](#sparsely-gated-mixture-of-experts), which originally had separate load-balancing and importance-weighting losses. This auxiliary loss is added to the total model loss, **for every Switch layer**.

Given $$N$$ experts and a batch $$B$$ with $$T$$ tokens, the auxiliary loss is computed as the scaled dot product between the vector $$f$$ of the fraction of tokens dispatched to each expert and the vector $$P$$ of the fraction of the router probability allocated to each expert:

$$
loss = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

where $$\alpha$$ is a user-picked hyper-parameter ($$10^{-2}$$ in the paper). Both the $$P$$ and $$f$$ vectors should ideally be uniform, i.e. have all values equal to $$1/N$$, and the factor $$N$$ keeps the loss constant as the number of experts varies. This loss has the same form as the auxiliary loss of GShard, with $$f_i$$ in the role of $$c_e/S$$ and $$P_i$$ in the role of $$m_e$$.

### Model size reduction via distillation

One of the downsides of these MoEs is their large number of parameters. However, [distillation](https://en.wikipedia.org/wiki/Knowledge_distillation) was shown to be feasible when a smaller network is needed: 

> Successful distillation of sparse pre-trained and specialized fine-tuned models into
small dense models. We reduce the model size by up to 99% while preserving 30% of
the quality gains of the large sparse teacher.

Finally, the paper includes several details on training techniques and experiments, which are omitted here for brevity. A single-device implementation of Switch Transformers is available on [this page](https://nn.labml.ai/transformers/switch/index.html). 

## Distributed implementation in PyTorch

The MoE workflow is a four-step algorithm, best illustrated in the [MegaBlocks paper](https://arxiv.org/abs/2211.15841) (detailed later): 

{: .picture}
<img width="94.8%" height="94.8%" src="{{ site.assets }}/distributed-training/imgs/moe_megablocks.svg"/>

{: .caption}
<b>A Mixture-of-Experts Layer.</b> Shown for <i>num\_experts=3</i>, <i>top\_k=1</i> and <i>capacity\_factor=1</i> with the prevalent, token dropping formulation. <b>First (1)</b>, tokens are mapped to experts by the router. Along with expert assignments, the router produces probabilities that reflect the confidence of the assignments. <b>Second (2)</b>, the feature vectors are permuted to group tokens by expert assignment. If the number of tokens assigned to an expert exceeds its capacity, extra tokens are dropped. <b>Third (3)</b>, the expert layers are computed for the set of tokens they were assigned as well as any padding needed for unused capacity. <b>Lastly (4)</b>, the results of the expert computation are un-permuted and weighted by the router probabilities. The outputs for dropped tokens are shown here set to zero. Adapted from: [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)

The tokens travel to the experts and back with [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) collectives, and their gradients must make the same trip in reverse during the backward pass. `dist.all_to_all_single` has no backward pass, so it is wrapped in an autograd function whose backward pass runs the opposite all-to-all:

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

The distributed mixture-of-experts block is then defined as `MoE_dist`, with a replicated (data-parallel) router and a local expert. For the sake of simplicity, there is a single expert per GPU and no load balancing loss, and the tokens that exceed each expert's capacity are dropped:

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

Here, the `Router` is a feed-forward network that takes the embedding of a token as input and outputs a distribution of allocations (probabilities) over the experts. Also, the code assumes there is only one expert per process, so the process id matches the expert id, and the number of experts matches the number of processes. 

The **routing step** computes the top-`k` expert assignments of each token, i.e. the expert ids in `topk_experts` and the corresponding probabilities in `topk_probs`, and groups them by expert: 
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

The following **permutation step** selectively sends the tokens to each expert's processor. It uses collective communication, in two steps: an `all_to_all` operation to exchange the number of items to send/receive (needed to allocate the receive buffers), followed by an `all_to_all_single` call to exchange the metadata of the tokens and an `AllToAll` call to exchange the tokens themselves (based on those counts): 

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

The **computation step** reshuffles the received data into a 3D batch of size `Rows x Capacity x Embedding`, cropping or padding each sequence to the capacity of the expert:

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

The output of the expert is stored in `batch_toks`. The following **un-permutation step** performs an `AllToAll` with the opposite communication of the permutation step. This populates the all-to-all output buffer with the partial results of all experts.

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

Finally, the un-permutation step also **scales** the outputs: it computes, for each token, the sum of the outputs of its top-k experts, weighted by the router probabilities. The outputs arrive grouped by expert, so each one is added at the position of its token, given by the metadata `(b, s)` sent in the permutation step:

```python
    x = x * torch.cat(probs_per_expert).view(-1,1) # multiply by router probabilities
    rows = (send_ids[:,0] - global_rank*B) * S + send_ids[:,1] # position b*S+s of each output, without the batch offset
    out = torch.zeros(B*S, E, dtype=x.dtype, device=device).index_add(0, rows, x) # sum the top-k outputs of each token
    return out.view(B,S,E)
```


### Applying the MoE to an existing LLM

Remember that the MoE module can replace the feed-forward module in existing architectures. So here it is applied to the [GPT2-based `GPTlite` model built in a previous post]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}). This is straightforward: all that is needed is to distribute its modules in a Distributed Data Parallel fashion, except the feed-forward module of the transformer block, which is replaced with the model-parallel mixture of experts:

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

The rest of the training and optimization algorithm needs no changes.

Finally, note that the previous code is a bit complex for readers who do not come from an HPC background. To avoid the hassle of writing all the communication steps from scratch, one can use a library that does it, such as [DeepSpeed Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/).

## Further reading

There is plenty of ongoing research on MoE finetuning, scaling and model variants. For more on these topics, here are some useful pointers:

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2022 [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)</summary>
MegaBlocks is "a system for efficient Mixture-of-Experts (MoE) training on GPUs", which addresses the **model quality vs hardware efficiency trade-off** in the dynamic routing of MoE layers. In detail, the load-imbalanced computation in MoEs forces one to either (1) drop tokens from the computation or (2) waste computation and memory on padding, a trade-off that is usually set via hyperparameter tuning. However, the authors observe that "The loss reached by the MoE models decreases significantly as expert capacity is increased, but at the cost of additional computation" and that "the lowest loss is achieved by the 'max' capacity factor model, which avoids dropping tokens through the dynamic
capacity factor mechanism proposed by [Adaptive
mixture-of-experts at scale (2022)](https://arxiv.org/abs/2206.03382)". So not dropping tokens is ideal.

As a second motivation, "the sparse computation in MoEs does not map cleanly to the software primitives supported in major frameworks and libraries". 
To overcome these limitations, the authors "reformulate MoE computation in terms of block-sparse operations and develop new block-sparse GPU kernels that efficiently handle the dynamism present in MoEs".
In practice, MegaBlocks proposes an approach to MoE routing that is based on **sparse primitives, maps efficiently to GPUs and never drops tokens**, "enabling end-to-end training speedups of up to 40% and 2.4× over state-of-the-art frameworks for MoE and DNN training, respectively". This is achieved by performing block-sparse operations in the MoE layers to accommodate the imbalanced assignment of tokens to experts. The resulting models are called **dropless-MoEs (dMoEs)**, in contrast with regular MoEs, which use batched matrix multiplication instead. These dropless-MoE kernels use blocked-CSR-COO encoding and transpose indices to enable efficient matrix products with sparse inputs.
</details>
{::options parse_block_html="false" /}



{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)</summary>
This paper formally studies "how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model". It experiments on a synthetic 4-cluster classification task and on vision tasks, and claims that: (1) "cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE"; (2) there needs to be non-linearity in the experts for an MoE system to work; and (3) "the router can learn the cluster-center features, which helps divide the input complex problem into simpler linear classification sub-problems that individual experts can conquer".
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)</summary>
The paper introduces GLaM (Generalist Language Model), a family of dense and sparse decoder-only language models, "which uses a sparsely activated mixture-of-experts architecture to scale the model capacity while also incurring substantially less training cost compared to dense variants". GLaM has 1.2 trillion parameters (7x larger than GPT-3's 175B) but requires 1/3 of the energy for training, and half of the computation FLOPs for inference, while "achieving better overall zero, one and few-shot performance across 29 NLP tasks" (Table 1, Figure 1).

Results were collected by training several variants of GLaM to study the behavior of MoE and dense models on the same training data (Table 4). The study highlights that:
- "the quality of the pretrained data also plays a critical role, specifically, in the performance of downstream tasks" (section 6.2);
- sparse models scale better (section 6.3);
- "MoE models require significantly less data than dense models of comparable FLOPs to achieve similar zero, one, and few-shot performance" (section 6.4).
</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)</summary>
The paper presents a study with experiments, results and techniques related to the stability of training and fine-tuning of very large sparse MoEs. It also introduces the **router z-loss**, which resolves instability issues and slightly improves model quality, and a 269B sparse model, **the Stable Transferable Mixture-of-Experts** or ST-MoE-32B, which achieves state-of-the-art performance across several NLP benchmarks. The z-loss formulation is:

$$
L_z(x) = \frac{1}{B} \sum_{i=1}^B \left( \log \sum_{j=1}^N e^{x_j^{(i)}} \right)^2
$$

where $$B$$ is the number of tokens, $$N$$ is the number of experts, and $$x \in \mathbb{R}^{B\times N}$$ are the logits going into the router. The rationale behind z-loss is that it penalizes large logits into the gating network. The paper also contains a very extensive analysis of quality-stability trade-off techniques and results:

- "Sparse models often suffer from training instabilities (Figure 1) worse than those observed in standard densely-activated Transformers. [...] Some architectural improvements involve more multiplications than additions or do not sum many items at once": the paper studies GELU Gated Linear Units (GEGLU) and Root Mean Square Scale Parameters.
- Adding noise (input-jitter) and adding dropout improve stability, but lead to a significant loss of model quality (Section 3.2).
- Constraining the update clipping (in the Adafactor optimizer) improves stability, but at a high cost in quality (Section 3.3).
- Sparse expert models are sensitive to [roundoff errors](https://en.wikipedia.org/wiki/Round-off_error) (e.g. due to lower-precision representations) because the routers add more exponential functions (Section 3.4).
- Sparse models are prone to overfitting (Fig. 3).
- Adding dropout at 0.1 provides modest boosts, while higher rates penalize performance; selectively increasing the dropout of the experts increases accuracy (Fig. 4).
- Updating the non-MoE parameters (while freezing the others) yields an accuracy similar to updating all the parameters, and updating only the FFN parameters works a bit better (Fig. 5).
- "Sparse models benefit from noisier hyperparameters including small batch sizes and high learning rates. Dense models behave nearly oppositely" (Fig. 6).
- "Sparse models are robust to dropped tokens when fine-tuning. [...] dropping 10-15% of tokens can perform approximately as well as models that drop < 1%" (Table 5).

The paper makes the following 3 recommendations for designing sparse systems:
> 
- top-2 routing with 1.25 capacity factor and at most one expert per core.
- the capacity factor can be changed during evaluation to adjust to new memory/compute requirements.
- dense layer stacking and a multiplicative bias can boost quality (Appendix C).

Regarding how MoEs work, the authors showed that **encoder experts learn very shallow tasks**, e.g. punctuation, nouns, etc. In other words, **MoEs subdivide a problem into smaller problems that can be solved by different experts combined**. The authors show this by "visualizing how tokens are routed among the experts [...] by passing a batch of tokens to the model and manually inspecting token assignment at each layer". They observed that, at each layer, at least one expert specializes in **sentinel tokens** (mask tokens that represent blanks to fill in), and that some encoder experts exhibit clear specialization, with some experts primarily operating on punctuation, verbs, proper names, counting, etc. These are detailed in Tables 13 and 15.

At the decoder level, "expert specialization is far less noticeable in the decoder", and meaningful specialization (semantics or syntax) is not visible in decoder experts:
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
static execution, namely static parallelism and pipelining, which does not adapt to the dynamic workload". To that end, they present Tutel, a system that delivers two main novelties: adaptive parallelism for optimal expert execution and adaptive pipelining for tackling inefficient and non-scalable dispatch/combine operations in MoE layers. The following table compares the communication complexity of data (DP), model (MP) and expert (EP) parallelism, for $$P$$ expert parameters, a token capacity $$C_g$$ per GPU, $$W$$ GPUs, $$E$$ experts and a degree $$r$$ of model parallelism:

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
This paper claims that **MoEs benefit more from instruction tuning than dense models**. To show this, they perform single-task fine-tuning, multi-task instruction tuning, and multi-task instruction tuning followed by single-task fine-tuning. They also show that MoEs benefit more from a higher number of tasks than from a higher number of experts. In practice:
  > in the absence of instruction tuning, MoE models fall short in performance when compared to dense models on downstream tasks. [...] When supplemented with instruction tuning, MoE models exceed the performance of dense models on downstream tasks, as well as on held-out zero-shot and few-shot tasks.
</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2023 [From sparse to soft mixture of experts](https://arxiv.org/abs/2308.00951)</summary>

Soft MoE is a fully-differentiable sparse Transformer. Rather than employing a sparse
and discrete router that tries to find a good hard assignment between tokens and experts (like regular MoEs), Soft MoEs perform a soft assignment by mixing tokens.

In practice, it computes several weighted
averages of all tokens (with weights depending on both tokens and experts) and then processes each weighted average by its corresponding expert.

It addresses the challenges of current sparse MoEs: training instability, token dropping, inability to scale the number of experts, and ineffective finetuning.

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/distributed-training/imgs/soft_moes.svg"/>

{: .caption}
<b>Sparse and Soft MoE layers.</b> While the router in Sparse MoE layers (left) learns to <i>assign</i> individual input tokens to each of the available slots, in Soft MoE layers (right) each slot is the result of a (different) <i>weighted average</i> of all the input tokens. Learning to make discrete assignments introduces several optimization and implementation issues that Soft MoE sidesteps. Adapted from: [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)

In the benchmark, Soft MoEs greatly outperform dense Transformers ([ViTs](https://en.wikipedia.org/wiki/Vision_transformer)) and popular MoEs (Tokens Choice and Experts Choice) on a vision task, while training in a reduced time frame and delivering faster inference. However, Soft MoEs are hard to use in auto-regressive decoders, because each weighted average mixes all input tokens, including future ones.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2024 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)</summary>
At the time of writing, the state-of-the-art MoE model was Mixtral 8x7B, a sparse mixture of experts with the same architecture as Mistral 7B, except that it supports a fully dense context length of 32k tokens and its feed-forward blocks are replaced by a mixture of 8 feed-forward network experts. The model architecture is detailed in Table 1. It performs top-2 routing, and because tokens can be allocated to different experts, "each token has access to 47B parameters, but only uses 13B active parameters during inference". 

The formulation of the model, in section 2.1, matches the GShard architecture (except that the MoE is in every layer instead of every second one, and that the second expert is not picked at random), with a sparse top-$$k$$ MoE with gating $$G(x) = \mathrm{Softmax}(\mathrm{TopK}(x \cdot W_g))$$, where $$\mathrm{TopK}(\ell)_i = -\infty$$ if the logit $$\ell_i$$ is not among the top-$$k$$. As before, "in a Transformer model, the MoE layer is applied independently per token and replaces the feed-forward (FFN) sub-block of the transformer block". They use the [SwiGLU activation function](https://arxiv.org/abs/2002.05202v1) (same as Llama-2), and therefore, for a given input $$x$$, the output is computed as:

$$
y = \sum_{i=0}^{n-1} \mathrm{Softmax}(\mathrm{Top2}(x \cdot W_g))_i\cdot \mathrm{SwiGLU}_i(x).
$$

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2024 [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)</summary>

Remember that MoEs work by subdividing the problem domain into smaller problems that are solved individually by different experts. In this paper, the authors claim that not all tokens require the same amount of effort to accurately make a prediction. To that end, they use **conditional computation**, a technique that "tries to reduce total compute by expending it only when needed". Here, "the network must learn how to dynamically allocate the available compute by making decisions per-token, in each layer, about where to spend compute from the available budget", where this compute budget is a total number of FLOPs.

For every token, the model decides to either apply computation (as in the standard transformer) or skip computation and pass the token through a residual connection (leaving it unchanged and saving compute). Moreover, unlike in MoEs, the **routing is applied to both the feedforward network and the multi-head attention**. In the multi-head attention routing, the router decides not only which tokens to update, but also which tokens are made available to attend to. "We refer to this strategy as Mixture-of-Depths (MoD) to emphasize how individual tokens pass through different numbers of layers, or blocks, through the depth of the transformer". In practice, during training, for every input, the router produces a scalar weight (importance) per token. The gating picks the top-$$k$$ tokens per sentence per block to participate in the transformer block computation. $$k$$ is a hyper-parameter that defines the maximum number of tokens passed to a block, so the computation graph and tensor sizes remain static throughout the execution. Setting $$k$$ allows one to trade off between performance and speed, for a given compute budget.

{: .picture}
<img width="95.4%" height="95.4%" src="{{ site.assets }}/distributed-training/imgs/mixture_of_depths.svg"/>

{: .caption}
<b>Mixture-of-Depths Transformer.</b> As in mixture-of-experts (MoE) transformers, a router chooses among potential computational paths. But unlike in MoE transformers the possible choices are a standard block’s computation (i.e., self-attention and MLP) or a residual connection. Since some tokens take this second route, Mixture-of-Depths (MoD) transformers have a smaller total FLOP footprint compared to vanilla or MoE transformers. On the top right is depicted a trained model’s routing decisions for a short sequence truncated to 64 tokens for visualization purposes. When examining the choices one can find tokens processed by later blocks’ layers, despite passing through relatively few total blocks throughout the model’s depth. This is a unique feature of MoD compared to conventional halting-based, or "early-exit" conditional computation, which instead engage blocks serially, or vanilla transformers, which engage every block. Adapted from: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

The big challenge is the routing scheme. The authors considered (1) token-choice routing, where a router produces per-token probability distributions across computational paths (experts), and the orthogonal approach, (2) expert-choice routing ([Mixture-of-Experts with Expert Choice Routing (2022)](https://arxiv.org/abs/2202.09368)), where, instead of tokens choosing the expert they prefer, each expert picks the top-$$k$$ tokens based on the tokens’ preferences. They ultimately adopted expert-choice routing because it does not require a load balancing loss, and "routers can try to ensure that the most critical tokens are among the
top-$$k$$ by setting their weight appropriately".

{: .picture}
<img width="99.5%" height="99.5%" src="{{ site.assets }}/distributed-training/imgs/mixture_of_depths_2.svg"/>

{: .caption}
<b>Routing schemes.</b> Tokens are funnelled to the computational path of their choice when using token-choice routing (left). If a given path exceeds its capacity (e.g., more than two tokens in this example) then surplus tokens must be dropped (purple token). The exact token that is ultimately dropped depends on the precise implementation in the underlying code. For example, priority is often given to those tokens that come earlier in the sequence or batch order. With expert-choice routing (middle), precisely <i>k</i> (in this case, two) tokens are chosen per path using a top-<i>k</i> mechanism across the tokens’ router weights. Here, tokens are dropped if they are not among the top-<i>k</i> with respect to any given path (orange token), and some tokens may even be funnelled to multiple paths (yellow token). Mixture-of-Depths uses expert-choice routing with a single path (right). Since tokens are dropped whenever <i>k</i> is less than the sequence length, the dropped tokens are routed around the self-attention and MLP computations, and a forward pass spends fewer FLOPs. Adapted from: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

At the time of writing, this is still very recent work, so time will tell if MoDs become useful for the general use case.

</details>
{::options parse_block_html="false" /}
