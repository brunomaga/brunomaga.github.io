---
layout: post
title:  "Mixture-of-Experts and Expert Parallelism"
categories: ["Distributed Training"]
tags: ["MoE", "expert parallelism", "load balancing", "Switch Transformer", "PyTorch"]
---

In a dense neural network, every parameter takes part in the processing of every input, so the compute per token grows with the size of the model. A [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) breaks this link. An MoE layer holds several sub-networks, the *experts*, and a small *router* that decides, for each input, how much each expert should contribute. A sparse MoE runs only the few experts with the largest weights, so the number of parameters grows with the number of experts, while the compute per token stays almost constant. In other words, MoEs increase the *model capacity per FLOP*, and compared with dense models of similar cost, this can improve scaling, efficiency and quality. MoEs are therefore a common choice for large language models, such as Mixtral, described at the end of the post, and a widely shared [leak report](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/) claims that [GPT-4](https://en.wikipedia.org/wiki/GPT-4) is an MoE too, with 16 experts of about 110B parameters each, about 1.8T parameters in total.

The idea is much older than these models, and its history introduces the key concepts one at a time: the weighted sum of the first MoEs, then routing and sparsity, the losses that balance the load of the experts, expert parallelism and the expert capacity. An implementation of an MoE layer with expert parallelism in PyTorch follows, and further work on numerical instabilities, overfitting and fine-tuning closes the post.

## The First MoEs: A Weighted Sum of Expert Outputs

In 1991, the papers [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) and [Task Decomposition Through Competition in a Modular Connectionist Architecture](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1502_2) introduced some of the earliest architectures similar to what is now called an MoE. The first of these papers starts from a problem of learning. When a single network is trained with backpropagation to perform different subtasks on different inputs, "there will generally be strong interference effects that lead to slow learning and poor generalization". A system of several networks, each of which learns to handle a subset of the inputs, avoids this interference.

{: .picture}
<img width="88.6%" height="88.6%" src="{{ site.assets }}/distributed-training/imgs/moe_2014_ilya.svg" alt="MoE paper 2014"/>

{: .caption}
**Left:** a Mixture of Experts. **Right:** a Deep Mixture of Experts (with 2 layers). <br/>Adapted from: [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

The left side of the figure shows the structure of such a system. An MoE has $$N$$ **experts**, the feed-forward networks $$f_1(x), \dots, f_N(x)$$, and a **gating network** $$g(x)$$, which is also a feed-forward network. All the experts and the gating network receive the same input $$x$$, and the gating network outputs one probability per expert, which says how relevant that expert is for the input. In its simplest form, the gate is $$g(x) = \mathrm{softmax}(x W_g)$$: the learnable matrix $$W_g$$ maps the input to one logit per expert, and the [softmax](https://en.wikipedia.org/wiki/Softmax_function) turns the logits into probabilities that sum to 1. The output $$z$$ of the MoE is then the sum of the outputs of the experts, weighted by the gate:

$$
z = \sum_{i=1}^{N} g_i(x) \, f_i(x)
$$

An MoE module is easy to implement. In PyTorch, the router is a feed-forward network that ends with a softmax and returns one probability per expert for every token, and each expert is a regular feed-forward network. Besides the weighted sum, the forward pass returns the probabilities and the outputs of all the experts, which the loss functions below need:

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

How do the experts learn to divide the work among them? They train together with the gate, and the answer lies in the loss function. For an input $$x$$ with target $$d$$, Adaptive Mixtures of Local Experts compares three losses:
- **Cooperation**, $$L = \Vert d - \sum_i g_i(x) f_i(x) \Vert^2$$, the loss of earlier work, compares the target with the weighted sum of the outputs. It couples the experts strongly, because the error of one expert depends on the outputs of all the others: each expert learns to cancel the residual error that the other experts leave. As a result, many experts contribute to each input, and they do not specialize.
- **Competition**, $$L = \sum_i g_i(x) \Vert d - f_i(x) \Vert^2$$, is the expected error when the gate picks a single expert at random, with the gate values as probabilities. Each expert must now produce the whole output on its own, not a residual, so its error no longer depends on the other experts. An expert that does better than the weighted average of the experts on an input receives a larger gate value for it, and one that does worse a smaller one, so the gate learns to give each input to a single expert.
- **Negative log-likelihood** of a Gaussian mixture, $$L = -\log \sum_i g_i(x) \, e^{-\frac{1}{2} \Vert d - f_i(x) \Vert^2}$$, is the loss that the paper uses in its experiments, because it works better. In its gradient, the update of expert $$i$$ is weighted not by $$g_i(x)$$ alone, but by how well expert $$i$$ fits the target compared with the other experts. This is a better measure of the relevance of an expert for an input, especially early in training. When all gate values are still equal and the errors are larger than 1, the competition loss changes the expert that fits best the least, since its error is the smallest, while the log-likelihood changes it the most.

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

Trained with the log-likelihood on a vowel discrimination task, the system shows the property that makes MoEs attractive. In the words of the authors:

> the learning procedure divides up a vowel discrimination task into appropriate subtasks, each of which can be solved by a very simple expert network.

## Deep Mixture of Experts

With the rise of deep neural networks, the next step was to stack MoEs. [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314) (Eigen et al., 2014) builds a Deep Mixture of Experts (DMoE), shown on the right of the figure above. Each layer has its own gating network and its own experts, and the output $$z^1$$ of the first layer is the input of the second; the superscripts of the figure number the layers, and the second layer has $$M$$ experts. Each input is processed by a combination of experts, one mixture per layer, so the number of effective experts grows with the product of the numbers of experts of the layers, while the size of the model grows only with their sum. In code, the DMoE is a stack of the MoE layers of the previous section, each of which passes its weighted sum to the next:

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

Training such a stack is not straightforward. Eigen et al. observe that it often reaches a degenerate local minimum, in which the "experts at each layer that perform best for the first few examples end up overpowering the remaining experts". The cause is a feedback loop: the experts that happen to do best on the first examples are picked more often by the gate, so they receive more training, and they keep their lead. To break this loop, Eigen et al. constrain the gate in a first training phase. They keep the running sum of the gate values of each expert, and when this sum exceeds the mean of all experts by more than a fixed margin, the expert gets a gate value of 0 for the current example, and the other gate values are renormalized. A second training phase then removes the constraint.

The experiments use MNIST digits with random translations, and a speech dataset of monophones. The two-layer DMoE is compared with a one-layer MoE and with a fully connected network (DNN) that has the same number of parameters. Two further baselines keep the mixture of the first layer and change the second one: one uses a single fixed expert, and the other concatenates the outputs of all the experts instead of weighting them. These two baselines bracket the DMoE, since a gate that selects among the experts should fit the training data better than a single expert, but not as well as a layer that uses all the experts at once. In most cases, the training errors confirm this. On the test set, however, the DMoE overfits: its error is higher than that of the single expert for two of the four model sizes. On the other hand, the DMoE performs almost as well as the DNN, even though its structure is more constrained.

An important result concerns what the experts learn. The DMoE develops a factored representation of the data:

> we find that the Deep Mixture of Experts automatically learns to develop location-dependent ("where") experts at the first layer, and class-specific ("what") experts at the second layer.

## Sparsely-Gated Mixture of Experts

The MoEs above have a major drawback: every expert processes every input, so the compute grows with the number of experts, even for experts of little relevance to an input. The solution is **conditional computation**, also called sparsity, where only a few experts run for each input. The gate still computes a weight per expert, but these weights now select the top-$$k$$ experts of each token, and the token is sent only to them, so the gate works as a **router**. The weights of the other experts are zero, and when $$g_i(x) = 0$$, the output $$f_i(x)$$ is not computed. With $$k$$ fixed, the number of experts, and with it the number of parameters, can therefore grow without a proportional increase in compute, which is the key scaling property of MoEs. The hyper-parameter $$k$$ sets the trade-off between the compute per token and the number of experts that contribute to it.

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017) builds on this idea. It places sparse MoE layers between stacked [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) layers of language models and translation models, with up to 131072 experts and 137B parameters. The MoE layer runs once for each position of the text, and it selects a potentially different combination of experts at each position:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/distributed-training/imgs/moe_2017_dean.svg"/>

{: .caption}
A Mixture of Experts (MoE) layer embedded within a recurrent language model. In this case, the sparse gating function selects two experts to perform computations. Their outputs are modulated by the outputs of the gating network. Adapted from: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

### Gating and Sparsity

How does the gate become sparse? Shazeer et al. replace the softmax gate of the previous sections by **noisy top-$$k$$ gating**, which makes two changes to the logits $$x W_g$$ before the softmax. First, it adds Gaussian noise to the logits, with a scale that a second learnable matrix, $$W_{noise}$$, sets for each expert; the results are the noisy logits $$h(x)$$. Second, only the $$k$$ largest noisy logits are kept, and the others are set to $$-\infty$$, so that the softmax gives them a gate value of exactly 0:

$$
g(x) = \mathrm{softmax}(\mathrm{KeepTopK}(h(x), k))
$$

$$
h_i(x) = (x W_g)_i + \epsilon \cdot \mathrm{Softplus}((x W_{noise})_i), \quad \epsilon \sim \mathcal{N}(0, 1)
$$

The function $$\mathrm{Softplus}(a) = \log(1 + e^a)$$ keeps the scale of the noise positive. As the next section shows, the noise also helps to balance the load of the experts. The value of $$k$$ is small. On the 1-billion-word benchmark, the flat MoEs use $$k = 4$$, and the figure above shows $$k = 2$$, in the notation of the paper, with $$G(x)$$ for the gate and $$n$$ for the number of experts.

### Load Balancing

Left to itself, the router does not spread the tokens evenly. Shazeer et al. observe that it tends to converge to a state where it always gives large weights to the same few experts, and this imbalance reinforces itself: the favoured experts train faster, so they are selected even more often, while the other experts starve. Two losses counter this.

The first one balances the **importance** of the experts. The importance of an expert is the sum of its gate values over a batch $$X$$, $$\mathrm{Importance}(X)_i = \sum_{x \in X} g_i(x)$$, and the **importance loss** is the square of the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) $$CV = \sigma / \mu$$ of the importances, the ratio of their standard deviation to their mean, multiplied by a hand-tuned weight $$w_{importance}$$:

$$
L_{importance} = w_{importance} \cdot CV(\mathrm{Importance}(X))^2
$$

As an example, take a batch of 4 inputs and 5 experts, with these gate values:

|-|-|-|-|-|-|
| | Expert 1 | Expert 2 | Expert 3 | Expert 4 | Expert 5 |
|-|-|-|-|-|-|
| $$x_1$$ | 0.1 | 0.1 | 0 | 0.8 | 0 |
| $$x_2$$ | 0 | 0 | 0.2 | 0.7 | 0.1 |
| $$x_3$$ | 0.1 | 0 | 0 | 0.9 | 0 |
| $$x_4$$ | 0 | 0 | 0 | 1 | 0 |
| **Importance** | **0.2** | **0.1** | **0.2** | **3.4** | **0.1** |
|-|-|-|-|-|-|

Expert 4 dominates, with an importance of 3.4 out of a total of 4. The mean importance is $$\mu = 0.8$$ and the population standard deviation is $$\sigma \approx 1.30$$, so $$CV \approx 1.63$$, and the loss is about $$2.64 \, w_{importance}$$. If all the experts had the same importance, the standard deviation would be zero, and so would the loss.

Equal importances do not guarantee equal loads, however. One expert may receive a few inputs with large gate values, and another many inputs with small gate values: both have the same importance, but the second one processes many more tokens. On distributed hardware, such an imbalance costs memory and time, because the device with the most tokens finishes last, and it also leaves some experts with too few examples to train. The **load loss** therefore balances the number of tokens per expert. This number cannot serve as a loss directly, because it is a discrete quantity, a step function of the logits with no useful gradient. Shazeer et al. use instead a smooth estimate of the load: the sum, over the batch, of the probability $$P(x, i)$$ that expert $$i$$ is among the top-$$k$$ of input $$x$$, with a hand-tuned weight $$w_{load}$$:

$$
L_{load} = w_{load} \cdot CV(\mathrm{Load}(X))^2, \quad \mathrm{Load}(X)_i = \sum_{x \in X} P(x, i)
$$

This is where the noise of the gate comes in. Expert $$i$$ is selected when its noisy logit $$h_i(x)$$ is larger than the $$k$$-th largest of the other noisy logits, written $$\mathrm{kth\_excluding}(h(x), k, i)$$. The probability $$P(x, i)$$ of this event is taken over a new draw of the noise of expert $$i$$, while the noise of the other experts stays fixed. Since this noise is Gaussian, the probability has a closed form, with $$\Phi$$ the cumulative distribution function of the [standard normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):

$$
P(x, i) = \Phi \left( \frac{(x W_g)_i - \mathrm{kth\_excluding}(h(x), k, i)}{\mathrm{Softplus}((x W_{noise})_i)} \right)
$$

### Batching and Parallelism

Sparsity saves compute, but it creates a problem of its own. If the gate picks $$k$$ of the $$N$$ experts for each token, an expert receives only about $$kT/N$$ of the $$T$$ tokens of a batch. With many experts, the batch of each expert becomes small, and the hardware, which is efficient only on large batches, is poorly used. The remedy for this *shrinking batch problem* is to make the batch as large as possible. However, the memory that stores the activations for the backward pass limits the batch of each device, so Shazeer et al. increase the batch of the experts in two other ways:
- **Mixing data and model parallelism.** The standard layers and the gating network are replicated on all devices with [data parallelism]({{ site.baseurl }}{% post_url 2023-08-18-data-parallelism %}), but each expert has a single copy, on one device. With $$W$$ devices of $$T$$ tokens each, an expert then receives tokens from all the devices, about $$kTW/N$$ of them, which is $$W$$ times more than on a single device. This placement of the experts is now called **expert parallelism**.
- **Convolutional application.** An expert processes every position of the text in the same way, independently of the other positions. The MoE therefore does not need to run step by step with the LSTM: it runs once on all the time steps of the previous LSTM layer, as one large batch.

With thousands of experts, the gate itself has a large output, one logit per expert. A two-level **hierarchical MoE** reduces this branching factor. A primary gating network picks a sparse weighted combination of experts, and each of these experts is itself an MoE, with its own gating network. In the paper, the primary gate uses data parallelism, and each secondary MoE resides on one device.

### Results

The experiments compare MoE models with LSTM baselines of the same compute per token, on several language tasks:
- On the 1-billion-word benchmark, the MoE models outperform the baselines. At 256 experts, the hierarchical MoE is not better than the flat one ([perplexity](https://en.wikipedia.org/wiki/Perplexity) 36.0 against 35.7), but it scales to more experts, and reaches 34.1 with 4096 experts.
- On 100 billion words of Google News, the perplexity keeps improving up to 65536 experts (68B parameters), where it is 39% lower than that of the compute-matched baseline, and it degrades at 131072 experts, possibly because of too much sparsity. The same models trained on 10 billion words gain less from more experts: a larger capacity helps more on a larger training set.
- On the Google Production translation dataset, the MoE model reaches a 1.01 higher test [BLEU](https://en.wikipedia.org/wiki/BLEU) score, even after training for only one sixth of the time.
- Overall, the authors claim "greater than 1000x improvements in model capacity with only minor losses in computational efficiency".

Finally, the experts specialize, in syntax as well as in semantics. The table below, Table 9 of the paper, lists the contexts of the inputs that three experts of a translation model receive with the largest gate values. Expert 381 sees words about research and innovation, expert 752 the article "a" after verbs such as "plays", and expert 2004 words about speed, such as "rapidly", "swift" and "fast":

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
Contexts corresponding to a few of the 2048 experts in the MoE layer in the encoder portion of the WMT’14 En→Fr translation model. For each expert $$i$$, the inputs of a training batch are sorted in decreasing order of $$G(x)_i$$, and the table shows the words around the corresponding positions in the input sentences. Source: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

A PyTorch implementation of this layer is available in [this repository](https://github.com/lucidrains/mixture-of-experts).

## GShard

Three years later, the Transformer had replaced the LSTM in most language models, and [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) (Lepikhin et al., 2020) brought sparse MoE layers to it. GShard scales a Transformer for machine translation, from 100 languages to English, to 600B parameters, and trains it on 2048 [TPU](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) v3 accelerators in 4 days. Thanks to the sparse MoE layers, the compute grows sub-linearly with the number of parameters.

GShard keeps two ideas of Shazeer et al.: position-wise sparse MoE layers, and a combination of data parallelism and sharding. The dense layers are replicated, as "the attention layer is parallelized by splitting along the batch dimension and replicating its weights to all devices". The experts, on the other hand, are too large to be replicated, and the only viable strategy is to shard them across the devices. What GShard adds is the way in which this parallelism is expressed.

### Compiler and SPMD Partitioning

Writing the communication of such a model by hand, for thousands of devices, is a large effort. Instead, GShard separates the description of the model from its parallel partitioning. It consists "of a set of simple APIs for annotations, and a compiler extension in [XLA](https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra) for automatic parallelization". The developer writes the model as for a single device, and annotates only "a few critical tensors in the model with partitioning policies", which say how each of these tensors is split across the devices. The compiler then partitions the whole computation graph accordingly, and inserts the communication that the partitioning requires. It generates a [single program for all devices](https://en.wikipedia.org/wiki/Single_program,_multiple_data) (SPMD), instead of one program per device, so the compilation time remains nearly constant as the number of devices grows.

### Model Architecture

The model is a Transformer encoder-decoder. Its blocks consist of self-attention, a feed-forward layer and, in the decoder, cross-attention, and GShard turns them into conditional computation modules by replacing the feed-forward layer of every second block with an MoE layer with top-2 gating, in both the encoder and the decoder. The figure shows the encoder, in the notation of the paper, with $$E$$ experts and $$N$$ layers. In panel (c), each device holds one expert, and two all-to-all collectives connect the devices: the first one sends each token to the devices of its experts (*dispatch*), and the second one returns the outputs of the experts to the devices of their tokens (*combine*):

{: .picture}
<img width="97%" height="97%" src="{{ site.assets }}/distributed-training/imgs/moe_gshard.svg"/>

{: .caption}
Illustration of scaling of Transformer Encoder with MoE Layers. The MoE layer replaces every other Transformer feed-forward layer. Decoder modification is similar. (a) The encoder of a standard Transformer model is a stack of self-attention and feed forward layers interleaved with residual connections and layer normalization. (b) Replacing every other feed-forward layer with an MoE layer gives the MoE Transformer Encoder. (c) When scaling to multiple devices, the MoE layer is sharded across devices, while all other layers are replicated. Adapted from: [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

### Expert Capacity and Random Routing

Even a sparse router can send too many tokens to one expert, and on parallel hardware, the most loaded expert sets the pace of all the others. For this reason, GShard introduces the **expert capacity**, the maximum number of tokens that an expert processes. The tokens of a batch are split into groups, which are dispatched independently, and with top-2 gating, each expert accepts at most $$2T/N$$ of the $$T$$ tokens of a group, the number it would receive if the $$2T$$ assignments of the group were spread evenly over the $$N$$ experts. An assignment to an expert that is already full is dropped, and a token whose experts are all full *overflows*: the MoE layer skips it, and the [residual connection](https://en.wikipedia.org/wiki/Residual_neural_network) passes it unchanged to the next layer.

GShard also routes the second expert at random. With this **random routing**, the first expert of a token always receives it, but the second expert receives it only with a probability proportional to its gate value, so each token is routed to one or two experts. The rationale is that a second expert with a small gate value contributes little to the output, so skipping it saves compute, and it leaves room in the capacity of the experts for other tokens.

### Auxiliary Loss

The capacity limits the cost of an unbalanced routing, but a router that keeps sending most tokens to the same few experts would still overflow many tokens. To prevent this, GShard adds an **auxiliary loss** $$L_{aux}$$ to the negative log-likelihood $$L_{nll}$$ of the model, which keeps a few experts from taking most of the tokens early in training. The total loss is $$L = L_{nll} + \alpha L_{aux}$$, with a constant weight $$\alpha$$, and:

$$
L_{aux} = \frac{1}{N} \sum_{i=1}^{N} \frac{c_i}{T} \, m_i
$$

Here $$c_i$$ is the number of tokens of the group whose first choice is expert $$i$$, so $$c_i / T$$ is the fraction of the tokens routed to it, and $$m_i$$ is the mean gate value of expert $$i$$ over the $$T$$ tokens. Why this product? The fractions $$c_i / T$$ sum to 1, and their mean square, $$\frac{1}{N} \sum_i (c_i / T)^2$$, is smallest when the tokens are spread evenly over the experts, so it would be a natural loss. However, $$c_i$$ comes from the top-2 selection and has no gradient, so GShard replaces one of the two factors $$c_i / T$$ of the square by the mean gate value $$m_i$$, a differentiable approximation of it.

## Switch Transformers

Despite these successes, the widespread adoption of MoEs was "hindered by complexity, communication costs, and training instability". [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021) addresses these three problems by simplifying the routing and improving the training recipe, which gives the first large MoE models that can be trained in [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format). Its starting point is a result of [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020): the loss of a language model follows power laws in the model size, the data size and the compute. The authors then explore a further direction:

> Heeding these results, we investigate a fourth axis: increase the parameter count while keeping the floating point operations (FLOPs) per example constant.

As in GShard, MoE layers replace the feed-forward layers, and the experts are sharded across the devices. The difference lies in the routing. **Switch routing** sends each token to a single expert (top-1). Shazeer et al. had conjectured that routing to more than one expert is necessary for the router to receive non-trivial gradients, since learning to route requires a comparison of at least two experts. Switch Transformers show that top-1 routing preserves the quality of the model, and that it has three benefits: the router computes less, the batch of each expert, its capacity, can be at least halved, and the communication is simpler and cheaper.

The training recipe addresses the instabilities. Low precision can cause trouble in the softmax of the router, so Switch Transformers compute the router in float32, a technique called *selective precision*, while the rest of the model runs in bfloat16. The weights also start from a smaller initialization, a tenth of the default scale. With these changes, large sparse models train in bfloat16 for the first time. The results show what sparsity brings: with the same compute per token, Switch-Base reaches the quality of the dense T5-Base up to 7 times faster, and the largest model, Switch-C, has 1.6 trillion parameters and 2048 experts.

{: .picture}
<img width="96.3%" height="96.3%" src="{{ site.assets }}/distributed-training/imgs/moe_switch_transformers.svg"/>

{: .caption}
Illustration of a Switch Transformer encoder block. The dense feed forward network (FFN) layer of the Transformer is replaced by a sparse Switch FFN layer (light blue). The layer operates independently on the tokens in the sequence. Two tokens (<i>x</i><sub>1</sub> = “More” and <i>x</i><sub>2</sub> = “Parameters” below) are routed (solid lines) across four FFN experts; the router routes each token independently. The switch FFN layer returns the output of the selected FFN multiplied by the router gate value (dotted-line). Adapted from: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

### Expert Capacity

Switch Transformers keep the expert capacity of GShard, and make it a tunable hyper-parameter through the **capacity factor**. The capacity of an expert is the mean number of tokens per expert, multiplied by this factor:

$$
\text{expert capacity} = \frac{\text{tokens per batch}}{\text{number of experts}} \times \text{capacity factor}
$$

A factor of 1 leaves no room for an uneven routing, and a factor above 1 adds a buffer for it. As in GShard, the tokens beyond the capacity of their expert skip the layer, through the residual connection. The figure shows the trade-off with 6 tokens and 3 experts, one per device. With a capacity factor of 1.0, each expert takes 2 tokens: expert 1 receives 3 tokens, so one of them overflows (dotted red line), while expert 3 receives a single token and leaves a slot empty. With a factor of 1.5, each expert takes 3 tokens, and no token overflows, but three slots stay empty. A larger factor thus drops fewer tokens, but it adds compute, memory and communication for the padding:

{: .picture}
<img width="96.9%" height="96.9%" src="{{ site.assets }}/distributed-training/imgs/moe_switch_transformers_2.svg"/>

{: .caption}
Illustration of token routing dynamics. Each expert processes a fixed batch-size of tokens modulated by the <i>capacity factor</i>. Each token is routed to the expert with the highest router probability, but each expert has a fixed batch size of (total\_tokens / num\_experts) × capacity\_factor. If the tokens are unevenly dispatched then certain experts will overflow (denoted by dotted red lines), resulting in these tokens not being processed by this layer. A larger capacity factor alleviates this overflow issue, but also increases computation and communication costs (depicted by the empty grey slots). Adapted from: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

### Load Balancing Loss

For the balance of the load, Switch Transformers replace the importance and load losses of [Shazeer et al.](#sparsely-gated-mixture-of-experts) by a single auxiliary loss, which is added to the total loss for every Switch layer. It has the form of the GShard loss. With $$c_i / T$$ the fraction of the $$T$$ tokens of the batch dispatched to expert $$i$$, and $$m_i$$ the fraction of the router probability allocated to expert $$i$$, i.e. its mean router probability, the loss is a scaled dot product of the two vectors:

$$
L_{balance} = \alpha N \sum_{i=1}^{N} \frac{c_i}{T} \, m_i
$$

Ideally, both vectors are uniform, with all their values equal to $$1/N$$. Under such a uniform routing, $$\sum_i \frac{c_i}{T} m_i = 1/N$$, so the factor $$N$$ keeps the loss at $$\alpha$$ for any number of experts. The paper uses $$\alpha = 10^{-2}$$, large enough to balance the load, and small enough not to overwhelm the main loss.

### Distillation

A downside of sparse models is their large number of parameters, which must all be stored, even though each token uses only a few of them. When a smaller model is needed, Switch Transformers show that the sparse model can be [distilled](https://en.wikipedia.org/wiki/Knowledge_distillation) into a small dense model, which learns to reproduce the outputs of the sparse teacher: "We reduce the model size by up to 99% while preserving 30% of the quality gains of the large sparse teacher." The paper has more details on the training techniques and the experiments, and a single-device implementation of Switch Transformers is available on [this page](https://nn.labml.ai/transformers/switch/index.html).

## Comparing the Routing Methods

All the routing methods above store the parameters of all $$N$$ experts, so their memory grows with $$N$$. What sets them apart is the number of experts that process each token, which sets the expert compute per token. With $$F$$ the compute of one expert for one token, the table compares them:

| Routing | Experts per token | Expert compute per token | Load balancing |
|:--|:-:|:-:|:--|
| Dense (Jacobs et al., Eigen et al.) | $$N$$ | $$N F$$ | none, or a gate constraint in a first training phase |
| Noisy top-$$k$$ (Shazeer et al.) | $$k$$ | $$k F$$ | importance and load losses |
| Top-2 with random routing (GShard) | 1 or 2 | at most $$2 F$$ | auxiliary loss, expert capacity |
| Top-1 (Switch Transformers) | 1 | $$F$$ | auxiliary loss, capacity factor |

The communication follows the compute: with expert parallelism, each expert that processes a token is a destination of the all-to-all, so the communication volume per token also grows with the number of experts per token. The expert capacity bounds the work of each expert, at the price of some compute and memory for the padding, and of dropped tokens when the factor is small. Which method fits which case? Dense MoEs suit only a few experts, since their compute grows with $$N$$, and sparse routing is what makes a large number of experts affordable. Among the sparse methods, top-1 routing has the lowest compute and communication per token, and for a fixed compute and training time, Switch Transformers reach a better quality than top-2 MoEs and dense Transformers. Top-2 routing costs more per token, but it remains a common choice: ST-MoE, in the further reading below, recommends it as a starting point, and Mixtral uses it.

## Distributed Implementation in PyTorch

The ideas of the previous sections come together in an implementation of an MoE layer with expert parallelism. Whatever the routing method, an MoE layer runs in four steps, which the [MegaBlocks paper](https://arxiv.org/abs/2211.15841), described at the end of the post, illustrates:

{: .picture}
<img width="94.8%" height="94.8%" src="{{ site.assets }}/distributed-training/imgs/moe_megablocks.svg"/>

{: .caption}
<b>A Mixture-of-Experts Layer.</b> Shown for <i>num\_experts=3</i>, <i>top\_k=1</i> and <i>capacity\_factor=1</i> with the prevalent, token dropping formulation. <b>First (1)</b>, tokens are mapped to experts by the router. Along with expert assignments, the router produces probabilities that reflect the confidence of the assignments. <b>Second (2)</b>, the feature vectors are permuted to group tokens by expert assignment. If the number of tokens assigned to an expert exceeds its capacity, extra tokens are dropped. <b>Third (3)</b>, the expert layers are computed for the set of tokens they were assigned as well as any padding needed for unused capacity. <b>Lastly (4)</b>, the results of the expert computation are un-permuted and weighted by the router probabilities. The outputs for dropped tokens are shown here set to zero. Adapted from: [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)

In the example of the figure, the router assigns each of six tokens to one of three experts, with a capacity factor of 1, so each expert takes two tokens. The router sends "quick", "fox" and "over" to expert 0, which has room for only two of them, so "over" is dropped, while expert 2 receives only "brown" and pads its second slot. After the computation, the un-permutation restores the order of the tokens, scales each output by its router probability, and gives the dropped token an output of zero.

With expert parallelism, the permutation and the un-permutation are communication steps: two [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) collectives send the tokens to the devices of their experts, and bring the outputs back. In the backward pass, the gradients travel the same path in reverse. `dist.all_to_all_single` has no backward pass, so it is wrapped in an autograd function whose backward pass runs the opposite all-to-all, with the send and receive sizes swapped:

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

The MoE block `MoE_dist` has a router, replicated with data parallelism, and one expert per GPU. The number of experts is thus the number of processes, and the rank of a process is the id of its expert. For simplicity, the tokens beyond the capacity of an expert are dropped:

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

The **routing step** computes the top-`k` experts of each token, their ids `topk_experts` and their probabilities `topk_probs`. It then groups the assignments by expert: for each expert, `ids_per_expert` holds the positions `(b, s)` of the tokens that chose it, together with the place of the expert in their top-`k`, and `probs_per_expert` holds the matching probabilities:

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

The **permutation step** sends each token to the process of its expert, in two stages. A process does not know in advance how many tokens it will receive, so a first `all_to_all` exchanges the counts: each process tells every other process how many tokens it will send to it. With these counts, every process can size its receive buffers, and two all-to-alls of variable size follow: an `all_to_all_single` exchanges the metadata of the tokens, and an `AllToAll` exchanges the tokens themselves. The metadata of a token are its position `(b, s)` and the place of the expert in its top-`k`. The row `b` is offset by the rank of the sender times the batch size, so that the rows of different processes do not collide on the receiver. This is the same pattern as the view swap of the [distributed sparse matrix transpose]({{ site.baseurl }}{% post_url 2015-04-21-matrix-transpose %}): a dense all-to-all of the counts, followed by all-to-alls of variable size for the elements and their indices.

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

The **computation step** turns the received tokens into a regular batch for the expert. It groups them by row, i.e. by the sequence they come from, into a 3D batch of shape `Rows x Capacity x Embedding`: the tokens of each row are cropped to the capacity of the expert, which drops the tokens beyond it, or padded up to it. The expert then processes the whole batch at once:

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

The **un-permutation step** writes the outputs of the expert back in the order in which the tokens arrived, with the padding value for the dropped tokens, and an `AllToAll` with the counts of the permutation step, swapped, sends them back to the processes of their tokens. The buffer `out_flat` is computed from `recv_toks`, so that it stays in the autograd graph even when the expert received no tokens: in the backward pass, every process must take part in both all-to-alls, or the others would wait forever:

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

Finally, the un-permutation step **scales** each output by the router probability of its expert, and sums the top-`k` outputs of each token. The outputs arrive grouped by expert, so `index_add` adds each one at the position `(b, s)` of its token, which the metadata of the permutation step provide:

```python
    x = x * torch.cat(probs_per_expert).view(-1,1) # multiply by router probabilities
    rows = (send_ids[:,0] - global_rank*B) * S + send_ids[:,1] # position b*S+s of each output, without the batch offset
    out = torch.zeros(B*S, E, dtype=x.dtype, device=device).index_add(0, rows, x) # sum the top-k outputs of each token
    return out.view(B,S,E)
```

As written, the implementation has no load balancing loss, so nothing keeps the router from sending most tokens to a few experts. The Switch loss can be added after the routing step, from `probs` and `topk_experts`: the fraction of the tokens whose first choice is each expert gives $$c_i / T$$, and the mean of `probs` over the tokens gives $$m_i$$.

### Applying the MoE to an Existing LLM

The `MoE_dist` block can replace the feed-forward block of an existing model. In the [GPT-2-based `GPTlite` model of a previous post]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}), all the modules are wrapped in Distributed Data Parallel, except the feed-forward module of each transformer block, which becomes the expert-parallel MoE:

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

The rest of the training loop is unchanged. Writing all the communication steps from scratch requires some background in HPC, and libraries such as [DeepSpeed Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/) implement them.

## Further Reading

The training, fine-tuning, scaling and variants of MoEs are active research topics, and the papers below are a selection of them:

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2022 [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)</summary>
MegaBlocks is "a system for efficient Mixture-of-Experts (MoE) training on GPUs". It starts from the trade-off of the expert capacity: with a fixed capacity, an MoE layer must either drop the tokens beyond it, which costs model quality, or pad the batches of its experts, which costs compute and memory. In the experiments of the paper, the loss decreases as the capacity grows, and it is lowest when no token is dropped, with the dynamic capacity factor of [Tutel](https://arxiv.org/abs/2206.03382). Dropping no tokens is therefore the goal. The difficulty is that the sparse computation of MoEs "does not map cleanly to the software primitives supported in major frameworks and libraries", which expect batches of a fixed size.

MegaBlocks solves this by writing the MoE layer as block-sparse matrix products, with new GPU kernels for a variable number of tokens per expert. Its *dropless* MoEs (dMoEs) never drop tokens, while regular MoEs use batched matrix products with a fixed batch per expert. The kernels store the sparse matrices in a blocked-CSR-COO format, with *transpose indices* for the products with transposed matrices. Training is up to 40% faster than MoE training with Tutel, and up to 2.4 times faster than dense training with Megatron-LM.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)</summary>
This paper studies, in theory, "how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model". It tests its analysis on synthetic data with 4 clusters, on CIFAR-10, and on rotated CIFAR-10 images. Two conditions turn out to be essential: the "cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE". With them, the router learns "the cluster-center features", which divides the complex problem into "simpler linear classification sub-problems that individual experts can conquer", the same division of a task into subtasks that the first MoEs showed in 1991.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)</summary>
GLaM (Generalist Language Model) is a family of dense and sparse decoder-only language models, which "uses a sparsely activated mixture-of-experts architecture to scale the model capacity while also incurring substantially less training cost compared to dense variants". As in GShard, an MoE layer with top-2 routing replaces every other feed-forward layer, here with 64 experts, so each token activates 96.6B of the 1.2 trillion parameters of the largest GLaM, about 8% of them. This largest model has about 7 times more parameters than GPT-3, but it needs a third of the energy of GPT-3 for training and half of its FLOPs for inference, and it reaches better average zero-shot and one-shot results on 29 NLP tasks. Dense and MoE variants trained on the same data (Table 4 of the paper) show that:
- "the quality of the pretrained data also plays a critical role, specifically, in the performance of downstream tasks" (section 6.2);
- MoE models perform better than dense models of similar compute per token, so sparse models scale better (section 6.3);
- MoE models "require significantly less data than dense models of comparable FLOPs" to reach similar zero-, one- and few-shot results (section 6.4).
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)</summary>
ST-MoE studies the stability of the training and the fine-tuning of large sparse MoEs. It introduces ST-MoE-32B, the Stable Transferable Mixture-of-Experts, a sparse model of 269B parameters with the compute of a 32B dense model, which reaches state-of-the-art results on several NLP benchmarks, such as SuperGLUE. It also introduces the **router z-loss**, which removes the instabilities and slightly improves the quality. The z-loss penalizes large logits $$h$$ at the input of the router, for $$T$$ tokens and $$N$$ experts:

$$
L_z = \frac{1}{T} \sum_{t=1}^{T} \left( \log \sum_{i=1}^{N} e^{h_i^{(t)}} \right)^2
$$

Sparse models are sensitive to [roundoff errors](https://en.wikipedia.org/wiki/Round-off_error), such as those of low-precision formats, because their routers add exponential functions. The z-loss keeps the logits small, which reduces these errors. The paper also analyzes the trade-off between quality and stability, and gives advice for fine-tuning:
- Removing operations with multiplicative interactions, the GELU Gated Linear Units (GEGLU) and the Root Mean Square scale parameters, improves the stability but costs quality. So do noise on the input (input jitter), dropout, and the clipping of the updates of the Adafactor optimizer.
- Sparse models overfit more easily: on the small CB task, the dense model is better on the validation set.
- In fine-tuning, a dropout of 0.1 works best, higher rates hurt the quality, and a higher dropout in the experts improves the quality only moderately.
- Fine-tuning only the non-MoE parameters works about as well as fine-tuning all parameters, and fine-tuning only the dense feed-forward layers works slightly better, while fine-tuning only the MoE parameters, about 80% of the model, strongly reduces the quality.
- Sparse models fine-tune best with small batches and high learning rates, and dense models with the opposite.
- Sparse models are robust to dropped tokens in fine-tuning: dropping up to 10-15% of the tokens does not hurt the quality.
- The recommended design is top-2 routing, a capacity factor of 1.25 and at most one expert per core. The capacity factor can change at evaluation time, to fit new memory and compute limits. Dense layers stacked with the sparse layers, and a multiplicative bias, improve the quality.

The paper also shows how the experts divide the work. Passing a batch of tokens through the model and inspecting by hand which tokens each expert receives, the authors find that the encoder experts specialize in simple categories of tokens. At each layer, at least one expert specializes in sentinel tokens, the mask tokens that mark the blanks to fill in, and other encoder experts operate mainly on punctuation, verbs, proper names or counting (Tables 13 and 15). In other words, the MoE divides the problem into smaller problems, which different experts solve. The decoder experts specialize far less. The authors relate this to the span corruption objective: the decoder routes fewer tokens together (groups of 456 tokens, against 2048 in the encoder), and a larger share of them are sentinel tokens, so the tokens of a group cover a smaller semantic space.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2022 [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382)</summary>
In an MoE, the router decides the load of each expert at runtime, and this load changes from one iteration to the next. However, existing MoE systems "suffer inefficient computation due to their static execution, namely static parallelism and pipelining, which does not adapt to the dynamic workload". Tutel adapts both at runtime: it switches the parallelism strategy at every iteration (*adaptive parallelism*), and it tunes the pipelining degree and the algorithm of the dispatch and combine all-to-alls (*adaptive pipelining*). The table compares the communication of data (DP), model (MP) and expert (EP) parallelism in an MoE layer, alone and combined. $$P$$ is the number of parameters of all experts, $$C_g$$ the token capacity per GPU, $$W$$ the number of GPUs, and $$E$$ the number of experts ($$N$$ above). In ⑦, each expert is split into $$r$$ slices with model parallelism, and each slice is replicated on $$(W/E)/r$$ GPUs with data parallelism.

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

The table shows why no single strategy is the best. Data parallelism communicates parameters, $$\mathcal{O}(P)$$, while expert parallelism communicates tokens, $$\mathcal{O}(C_g)$$, so the better choice depends on the ratio of parameters to tokens, which changes with the workload. The combination ⑦ contains ⑤ and ⑥ as special cases, and its parameter $$r$$ moves between them, which lets Tutel switch strategies at runtime.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2023 [Mixture-of-Experts Meets Instruction Tuning: a Winning Combination for Large Language Models](https://arxiv.org/abs/2305.14705)</summary>
This paper compares three ways to adapt an MoE to downstream tasks: fine-tuning on each task alone, instruction tuning on many tasks followed by zero-shot or few-shot use, and instruction tuning followed by fine-tuning on each task. Without instruction tuning, MoE models fine-tuned on single tasks are worse than dense models of the same compute. With instruction tuning, followed or not by fine-tuning on the task, MoE models are better than dense models: they benefit more from instruction tuning than dense models do, and their quality scales better with the number of instruction-tuning tasks than with the number of experts. The largest model, FLAN-MoE-32B, outperforms FLAN-PaLM-62B on four benchmarks with a third of the FLOPs.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2023 [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)</summary>

Soft MoE is a fully differentiable sparse Transformer. A sparse MoE makes a discrete choice: its router searches for a good hard assignment of tokens to experts. A Soft MoE makes a soft assignment instead, by mixing tokens. Each expert processes a few *slots*, and each slot is a weighted average of all the input tokens, with weights that depend on the tokens and on the slot. The compute then depends on the number of slots, not on the number of experts. This addresses four issues of sparse MoEs: training instability, dropped tokens, the limit on the number of experts, and poor fine-tuning.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/soft_moes.svg"/>

{: .caption}
<b>Sparse and Soft MoE layers.</b> While the router in Sparse MoE layers (left) learns to <i>assign</i> individual input tokens to each of the available slots, in Soft MoE layers (right) each slot is the result of a (different) <i>weighted average</i> of all the input tokens. Learning to make discrete assignments introduces several optimization and implementation issues that Soft MoE sidesteps. Adapted from: [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)

In image classification, Soft MoE outperforms dense [vision transformers](https://en.wikipedia.org/wiki/Vision_transformer) (ViTs) and the sparse Tokens Choice and Experts Choice MoEs by a large margin. With 128 experts, it has over 40 times the parameters of ViT-H/14, for about 2% more inference time. There is a limit, however: each slot mixes all the tokens of the input, so Soft MoE does not apply directly to autoregressive decoders, where a token must not depend on later tokens.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2024 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)</summary>
Mixtral 8x7B is a sparse MoE with the architecture of Mistral 7B (Table 1 of the paper), with two differences: it has a fully dense context of 32k tokens, and its feed-forward blocks are MoE layers of 8 feed-forward experts. Each token goes to 2 of the 8 experts, and since each token can use a different pair of experts, "each token has access to 47B parameters, but only uses 13B active parameters during inference". The gate is $$g(x) = \mathrm{softmax}(\mathrm{TopK}(x W_g))$$, where $$\mathrm{TopK}$$ sets the logits outside the top-$$k$$ to $$-\infty$$: this is the noisy top-$$k$$ gating of Shazeer et al., without the noise. It is also the formulation of GShard, with an MoE in every block instead of every second block, and a simpler choice of the second expert, without random routing. Each expert is a [SwiGLU](https://arxiv.org/abs/2002.05202v1) feed-forward network, as in Llama 2, so the output for an input $$x$$ is:

$$
z = \sum_{i=1}^{N} \mathrm{softmax}(\mathrm{Top2}(x W_g))_i \cdot \mathrm{SwiGLU}_i(x)
$$

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> 2024 [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)</summary>

MoEs divide the problem into smaller problems, each solved by different experts. Mixture-of-Depths (MoD) starts from a related observation: not all tokens need the same amount of compute. It uses conditional computation, which "tries to reduce total compute by expending it only when needed". Here, "the network must learn how to dynamically allocate the available compute by making decisions per-token, in each layer, about where to spend compute from the available budget", where the budget is a fixed number of FLOPs.

For each token and each block, a router decides whether the token goes through the block (self-attention and MLP), as in a standard Transformer, or around it, through the residual connection, which leaves the token unchanged and saves compute. Unlike in MoEs, the routing applies to both the MLP and the self-attention: a token routed around a block is not updated, and the other tokens cannot attend to it. "We refer to this strategy as Mixture-of-Depths (MoD) to emphasize how individual tokens pass through different numbers of layers, or blocks, through the depth of the transformer". The router gives each token a scalar weight, and the top-$$k$$ tokens of each sequence go through the block. Since $$k$$ is fixed, the computation graph and the tensor sizes are static, and $$k$$ sets the trade-off between quality and speed for a given compute budget.

{: .picture}
<img width="95.4%" height="95.4%" src="{{ site.assets }}/distributed-training/imgs/mixture_of_depths.svg"/>

{: .caption}
<b>Mixture-of-Depths Transformer.</b> As in mixture-of-experts (MoE) transformers, a router chooses among potential computational paths. But unlike in MoE transformers the possible choices are a standard block’s computation (i.e., self-attention and MLP) or a residual connection. Since some tokens take this second route, Mixture-of-Depths (MoD) transformers have a smaller total FLOP footprint compared to vanilla or MoE transformers. On the top right is depicted a trained model’s routing decisions for a short sequence truncated to 64 tokens for visualization purposes. When examining the choices one can find tokens processed by later blocks’ layers, despite passing through relatively few total blocks throughout the model’s depth. This is a unique feature of MoD compared to conventional halting-based, or "early-exit" conditional computation, which instead engage blocks serially, or vanilla transformers, which engage every block. Adapted from: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

The main design choice is the routing scheme. With *token-choice* routing, each token picks its paths (experts), as in the MoEs above, and a path that receives too many tokens must drop some of them. With *expert-choice* routing, each path picks its top-$$k$$ tokens instead, so every path receives exactly $$k$$ tokens. The authors use expert-choice routing, which needs no load balancing loss, and with which "routers can try to ensure that the most critical tokens are among the top-$$k$$ by setting their weight appropriately".

{: .picture}
<img width="99.5%" height="99.5%" src="{{ site.assets }}/distributed-training/imgs/mixture_of_depths_2.svg"/>

{: .caption}
<b>Routing schemes.</b> Tokens are funnelled to the computational path of their choice when using token-choice routing (left). If a given path exceeds its capacity (e.g., more than two tokens in this example) then surplus tokens must be dropped (purple token). The exact token that is ultimately dropped depends on the precise implementation in the underlying code. For example, priority is often given to those tokens that come earlier in the sequence or batch order. With expert-choice routing (middle), precisely <i>k</i> (in this case, two) tokens are chosen per path using a top-<i>k</i> mechanism across the tokens’ router weights. Here, tokens are dropped if they are not among the top-<i>k</i> with respect to any given path (orange token), and some tokens may even be funnelled to multiple paths (yellow token). Mixture-of-Depths uses expert-choice routing with a single path (right). Since tokens are dropped whenever <i>k</i> is less than the sequence length, the dropped tokens are routed around the self-attention and MLP computations, and a forward pass spends fewer FLOPs. Adapted from: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

The best models process 12.5% of the tokens in every second block. They match the quality of the baseline for the same training FLOPs and time, while they need a fraction of its FLOPs per forward pass, and they can be upwards of 50% faster per step in sampling. Expert-choice routing has one difficulty in autoregressive sampling, though: the top-$$k$$ of a sequence depends on later tokens, which are not known yet when a token is generated. An auxiliary loss on the router, or a small auxiliary predictor, decides instead whether a token is in the top-$$k$$.

</details>
{::options parse_block_html="false" /}
