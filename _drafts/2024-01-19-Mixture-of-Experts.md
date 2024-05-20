---
layout: post
title:  "Mixture-of-Experts: a historical overview, with distributed Pytorch and DeepSpeed implementations"
categories: [machine learning, Transformer, GPT, DeepSpeed, mixture-of-experts]
tags: [machinelearning]
---

(**Disclaimer:** this post is continuously being updated.)


Details of [GPT-4](https://en.wikipedia.org/wiki/GPT-4), the current [ChatBot benchmark](https://chat.lmsys.org/) reigning champion, were recently [leaked](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/). It mentions the usage of **Mixture-of-Experts (MoEs)**, with 16 experts of circa 110 billion parameters each, totalling 1.8 trillion parameters. 
<!-- Briefly after, [Mistral 7x8B](https://mistral.ai/news/mixtral-of-experts/) has been released as a MoE model of 8 experts with 7 billion parameters each, claiming its performance beats [GPT-3.5](https://en.wikipedia.org/wiki/GPT-3) with circa 175 billion parameters. -->

Behind this success is the fact that MoEs uses **sparse computation** and **conditional computation** to provide better training scaling (despite difficulties to control **overfitting**), faster inference and improved model accuracy compared to non-MoE systems, for the same compute budget or parameter count. OpenAI goes further and cliams that [Expert Parallelism is the 4th dimension of parallelism](https://openai.com/research/techniques-for-training-large-neural-networks), in addition to data, model and pipeline parallelism.

So in this post, we will discuss MoEs development from early days, and provide implementations of MoEs on a single node and large-scale distributed MoEs using PyTorch distributed and DeepSpeed implementations. We will also discuss loss functions, and finetuning. We will follows the following publications storyline (credit: partially inspired by the [MoE post from hugging face](https://huggingface.co/blog/moe)):
  
- [Early days: small MoEs as a combination of weighted expert outputs](#early-days-small-moes-as-a-combination-of-weighted-expert-outputs)
  - [1991 Adaptive Mixture of Local Experts](#1991-adaptive-mixture-of-local-experts)
  - [1991 Task Decomposition Through Competition in a Modular Connectionist Architecture](#1991-task-decomposition-through-competition-in-a-modular-connectionist-architecture)
  - [2014 Learning Factored Representations in a Deep Mixture of Experts](#2014-learning-factored-representations-in-a-deep-mixture-of-experts)
- [Large-scale MoEs via sparsing, routing, data- and model-parallelism](#large-scale-moes-via-sparsing-routing-data--and-model-parallelism)
  - [Distributed implementation on PyTorch](#distributed-implementation-on-pytorch)
  - [Distributed implementation on DeepSpeed](#distributed-implementation-on-deepspeed)
  - [2017 Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](#2017-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer)
  - [2020 GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](#2020-gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding)
  - [2021 Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](#2021-switch-transformers-scaling-to-trillion-parameter-models-with-simple-and-efficient-sparsity)
- [Towards the future: tweaking and improvements](#towards-the-future-tweaking-and-improvements)
  - [2022 MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](#2022-megablocks-efficient-sparse-training-with-mixture-of-experts)
  - [2022 Towards Understanding Mixture of Experts in Deep Learning](#2022-towards-understanding-mixture-of-experts-in-deep-learning)
  - [2022 GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](#2022-glam-efficient-scaling-of-language-models-with-mixture-of-experts)
  - [2022 ST-MoE: Designing Stable and Transferable Sparse Expert Models](#2022-st-moe-designing-stable-and-transferable-sparse-expert-models)
  - [2023 Mixture-of-Experts Meets Instruction Tuning: a Winning Combination for Large Language Models](#2023-mixture-of-experts-meets-instruction-tuning-a-winning-combination-for-large-language-models)
  - [2024 Mixtral of Experts](#2024-mixtral-of-experts)
  - [2024 Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](#2024-mixture-of-depths-dynamically-allocating-compute-in-transformer-based-language-models)

## Early days: small MoEs as a weighted sum of expert outputs

### 1991 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
### 1991 [Task Decomposition Through Competition in a Modular Connectionist Architecture](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1502_2) 


{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="/assets/Mixture-of-Experts/MoE_2014_Ilya.png" alt="MoE paper 2014"/>

{: style="text-align:center; font-size: small;"}
**Left:** a Mixture of Experts. **Right:** a Deep Mixture of Experts (with 2 layers). Source: [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

One of the earliest developments of architectures similar to MoEs were presented in [Adaptive Mixture of Local Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) and [Task Decomposition Through Competition in a Modular Connectionist Architecture (1991)](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1502_2). The MoE was defined as a set of independent **experts** (feed-forward networks, $$f_i(x)$$ in the picture) alongside a **gating network** (also a feed-forward network, $$g(x)$$). All the experts and the gating network receive the same input $$x$$. The gating network outputs the distribution of each expert relevance/importance for the given input and is defined as by $$g(x) = Softmax(x · W_g) $$ in its simplest form, where $$W_g$$ is a (optional) learnable transformation. Finally, the output of the system ($$z$$) is the sum of the outputs of all experts weighted by the output of the gating network.

The implementation of a MoE module is straighforward, and you can find the full code with running examples in the [`moe.py` file in the supporting repository](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/Mixture-of-Experts/moe.py):

```python
class MoE(nn.Module):

  def __init__(self, input_size, output_size, num_experts, dropout=0.1):
    super().__init__()
    self.router = Router(input_size, num_experts, dropout=dropout)
    self.experts = nn.ModuleList([
      FeedForward(input_size, output_size, dropout=dropout) for _ in range(num_experts) ])

  def forward(self, x):
    probs = self.router(x)
    outputs = torch.stack([ expert(x) for expert in self.experts ], dim=-2) # B*T*Experts*C
    weighted_outputs = outputs * probs.unsqueeze(-1) # B*T*E*C x B*T*1*C -> B*T*E*C
    weighted_sum = weighted_outputs.sum(dim=-2) # sum over experts: B*T*E*C -> B*T*C
    return weighted_sum, probs, outputs
```

The main motivation for using a mixture of experts instead of a single large feed-forward is described by the authors as:

> If backpropagation is used to train a single, multilayer network to perform different subtasks on different occasions, there will generally be strong interference effects that lead to slow learning and poor generalization.  

Which eventually lead to the following discovery on the behaviour of MoEs:

> the learning procedure divides up a vowel discrimination task into appropriate subtasks, each of which can be solved by a very simple expert network.

A major difference between the two 1991 approaches is the loss function picked. One picks an error function $$E = \| d - \sum_i p_i \, o_i \| ^2$$,  where $$o_i$$ is the output *vector* of expert $$i$$, $$p_i$$ is the router's probability for the expert $$i$$, and $$d$$ is the correct label for the input. The other claims that the previous form leads to **starvation** of experts where some experts are never assigned any importance (and one expert is **overpowered** i.e. assigned most of the gating importance). It suggestes instead the error function $$E = \sum_i p_i  \| d - \, o_i \| ^2$$ and its variant form based on the negative log-likelihood to overcome this.

```python
  def loss_per_token_Adaptive_Mixture_1991(probs, outputs, labels):
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=outputs.shape[-1])
    mse_expert_i = lambda i: (one_hot_labels-outputs[:,:,i,:]).square().mean()  # B*T*1 * B*T*classes
    loss_per_expert = (torch.stack([mse_expert_i(i) for i in range(8)])*probs) # B*T*experts
    return loss_per_expert.sum(-1) # B*T, 1 value per token
```

### 2014 [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

With the surge of Deep Neural Networks, [Learning Factored Representations in a Deep Mixture of Experts (2014)](https://arxiv.org/abs/1312.4314) extended "the Mixture of Experts work (MoE) to a stacked model, the **Deep Mixture of Experts**, with multiple sets of gating and experts".

This deep MoE model can be coded simply by stacking several of the previous MoEs (also available in [`moe.py`](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/Mixture-of-Experts/moe.py):):

```python
class MoE_deep(nn.Modue):
  def __init__(self, input_size, output_size, num_experts, depth=2, dropout=0.1):
    super().__init__()
    self.stack = nn.Sequential(
      [MoE(input_size, input_size,  num_experts, dropout) for i in range(depth-1)] +
      [MoE(input_size, output_size, num_experts, dropout)]
    )
  
  def forward(self,x):
    return self.stack(x)
```

During training, as before, the authors noticed that training led to a **degenerate local minimum**: "experts at each layer that perform best for the first few examples end up overpowering the remaining experts”, because as they are better initially, then end up being picked more by the gates. This was overcome by **capping (*max clipping*)** the assignments to a maximum value, and re-normalizing across all (post max-clipping) gating probabilities. This **cap value is then removed in a second training stage**.

The benchmark compared a single MoE, a two-level MoE, a two-layer MoE where the information between layers is the concatenation of all layer-1 experts (instead of the weighted sum of outputs), and a regular feed-forward network with a similar number of parameters. Results showed that, "In most cases, the deeply stacked experts performs
between the single and concatenated experts baselines on the training set, as expected. However, the deep models often suffer from **overfitting**: the mixture’s error on the test set is worse than that of the single expert for two of the four model sizes. Encouragingly, the DMoE performs almost as well as a fully-connected network (DNN) with the same number of parameters, even though this network imposes fewer constraints on its structure".

An important result emphasizes the relevance of stacking MoEs into a deep model:

> we find that the Deep Mixture of Experts automatically learns to develop location-dependent ("where") experts at the first layer, and class-specific ("what") experts at the second layer.

And this led to the follow up work that explored having the gating being used not only to weight the sum of output of experts, but as well as a router of the input to individual experts, as we will see in the following section.

## Large-scale MoEs via sparsing, routing, data- and model-parallelism


The previous MoE implementations had a big drawback: all experts process all inputs, leading to high training costs, even when there's low little relevance of some experts for some inputs. This led to the introduction of **Conditional Computation** or **Sparsity**, a method where only a subset of experts are picked per input. In practice, the distribution output by the gating mechanism is now used to pick only the highest-relevance experts for each task, and delegate the input only to those top-$$k$$ experts, effectively working as a **routing mechanism**. $$k$$ is a hyper-parameter that drives the trade-off between computation and *expertise*, and may ultimately result in under- or over-computation.
In practice, the MoE selects for each token in the text a possibly different combination of experts. When $$g_i(x)=0$$ for a given expert $$i$$, then $$f_i(x)$$ is not computed.

To scale the MoE on distributed systems they use **model parallelism for the MoE blocks and data parallelism for non-MoE blocks**: each processor has a data-parallel replica of the standard layers and the gating networks, and a model-parallel shard of the experts (ie each device holds a subset of experts, and there is only 1 copy of each expert across the distributed system). Batch is passed in a data-parallel fashion - each device receives a different micro-batch.
An important message is that, by picking only the top-$$k$$ experts per iteration, one can increase the number of experts without incurring an aditional computational overhead, leading to great MoE scaling properties.

Modern large-scale MoE efforts improved on the ideas of sparsity, conditional computing, data- and model- parallelism to deliver very large scale MoEs. Moreover, MoEs became a replacement for feed-forward networks that improves scaling and accuracy in pre-existent architectures, as we will see later.

### Distributed implementation on PyTorch

The MoE processing performs a four-step algorithm, best illustrated in the [MegaBlocks paper](https://arxiv.org/abs/2211.15841) (detailled later): 

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/Mixture-of-Experts/MoE_MegaBlocks.png"/>

We start by defining our distributed mixture-of-experts block as `MoE_dist`, with a sharded router and a local expert:

```python
class MoE_dist(nn.Module):
  def __init__(self, k=2, capacity_factor=1.25, padding_val=0, local_rank=local_rank):
    super().__init__()
    self.capacity_factor = capacity_factor
    self.padding_val = padding_val
    self.num_experts = dist.get_world_size() # 1 expert per GPU
    self.k = k

    self.router = DDP(Router(n_embd, self.num_experts).to(device), device_ids=[local_rank])
    self.expert = FeedForward(n_embd).to(device)
```

In here, the `Router` is a feed-forward network that takes as input the embedding of a token, and ouputs a distribution of allocations (probabilities) over the experts. Also, we will assume there is only one expert per process, thus the process id matches the expert id, and the expert count matches the distributed network size. 

The **routing step** computes the top-`k` expert assignments for each token, as expert id in `topk_experts` and corresponding probabilities `topk_probs`: 
```python
  def forward(self, x):
    probs = self.router(x) #get assignements from router, shape B * T * n_experts
    topk_probs, topk_experts = torch.topk(probs, k=self.k) # top-k experts per sentence
    ids_per_expert = [ (topk_experts==expert).nonzero() for expert in range(self.num_experts) ]
    probs_per_expert = [ topk_probs[topk_experts==expert] for expert in range(self.num_experts) ]
```

The following **permutation step** selectively sends the tokens to each expert's processor. We will use collective communication, and perform this in two steps: an `all_to_all` operation to exchange the amount of items to be sent receive, followed by an `all_to_all_single` to exchange the metadata of the tokens and the token themselves (based on those counts): 

```python
    # all-to-all to exchange the count of inputs to send/receive to/from each processor
    send_count = [torch.tensor([len(ids)], dtype=torch.int64, device=device) for ids in ids_per_expert]
    recv_count = [torch.tensor([0], dtype=torch.int64, device=device) for _ in ids_per_expert]
    dist.all_to_all(recv_count, send_count)
    fn_count = lambda tensor, scale=1: [x.item()*scale for x in tensor] 

    # send/receive the metadata row_id+token_id to/from the appropriate processors
    M = ids_per_expert[0].shape[-1] # number of columns in metadata
    send_ids = torch.cat(ids_per_expert, dim=0).to(device)
    send_ids[:,0] += global_rank*B # add processor's batch offset
    recv_ids = torch.zeros(sum(recv_count)*M, dtype=send_ids.dtype).to(device)
    dist.all_to_all_single(recv_ids, send_ids.flatten(), fn_count(recv_count,M), fn_count(send_count,M))
    recv_ids = recv_ids.view(-1, M) # reshape to M columns 

    # send/receive input tokens to/from the appropriate processors
    send_toks = torch.cat([ x[ids[:,:2].T.tolist()] for ids in ids_per_expert], dim=0).to(device)
    recv_toks = torch.zeros(sum(recv_count)*C, dtype=x.dtype).to(device)
    dist.all_to_all_single(recv_toks, send_toks.flatten(), fn_count(recv_count,C), fn_count(send_count,C))
    recv_toks = recv_toks.view(-1, C) # reshape to C columns 
```

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/Mixture-of-Experts/AlltoAll.png"/>

{: style="text-align:center; font-size: small;"}
An illustration of the MPI all-to-all collective operation. When sending only 1 element per row, it is equivalent to a distributed matrix transpose - note the red row before the all-to-all (left) as a transposed column after the all-to-all (right). We use a single-element all-to-all to exchange the count of items to be sent received among processors. We then use those counts to perform a new all-to-all with variable-sized elements in the permutation step. Finally, in the un-permutation step, we perform another all-to-all that performs the converse communication, by swapping the send and receive counts and buffers.

The **computation step** will reshuffle the received data into a 3D batch of size `Rows x Capacity x Embedding` that will either crap or pad sequences to the capacity of each export:

```python
    # group received metadata by row id 
    uniq_rows, recv_row_lens = recv_ids[:,0].unique(sorted=True, return_counts=True)
    recv_row_offsets = [0] + torch.cumsum(recv_row_lens, dim=0).tolist()
    recv_row_slice = lambda row: slice(recv_row_offsets[row], recv_row_offsets[row+1])

    # crop or pad received items PER SENTENCE to max capacity. Batch shape: Rows * Capacity * C
    capacity = int( T / self.num_experts *self.capacity_factor)
    pad_fn = lambda toks, value = self.padding_val: F.pad(toks, (0,0,0,capacity-toks.shape[0]), value=value) #pad or crop
    batch_toks = torch.stack([ pad_fn(recv_toks[recv_row_slice(i)]) for i in range(len(uniq_rows))], dim=0).to(device)

    batch_row_len = torch.tensor([ min(recv_row_lens[r], capacity) for r in range(len(uniq_rows))])
    batch_toks = self.expert(batch_toks) # Rows * Capacity * C
``` 

The ouput of the expert is stored in `batch_toks`. The following **Un-permutation step** will now perform an `all_to_all_single` that will perform the opposite communication of the permutation step. This will populate the all-to-all output buffer with the partial results all experts.

```python
    recv_toks = recv_toks.fill_(self.padding_val) # reset recv_toks, will be used to SEND results
    recv_tok_offsets  = np.concatenate([ range(recv_row_offsets[i], recv_row_offsets[i]+batch_row_len[i]) for i in range(len(uniq_rows)) ])
    batch_tok_offsets = np.concatenate([ [ [i]*batch_row_len[i], range(batch_row_len[i]) ] for i in range(len(uniq_rows)) ], axis=1)
    recv_toks[recv_tok_offsets] = batch_toks[batch_tok_offsets[0], batch_tok_offsets[1]] # fill recv_toks with results

    send_toks = send_toks.fill_(self.padding_val).flatten() # reset send_toks, will be used to RECEIVE results
    dist.all_to_all_single(send_toks, recv_toks.flatten(), fn_count(send_count,C), fn_count(recv_count,C))
    x = send_toks.view(-1,C)
```

Finally, the **scale step** performs a weighted sum of the top-k probabilities for each token:

```python
    x *= torch.concatenate(probs_per_expert).view(-1,1) # multiply by router probabilities
    if self.k>1: # sum over k probabilities for each input
      x = torch.stack( [ x[send_ids[:,-1]==k] for k in range(self.k)]).sum(dim=0)
    return x.view(B,T,C)
```

You can find the complete implementation in the [Mixture-of-Experts repo](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/Mixture-of-Experts/moe_dist.py).

### Distributed implementation on DeepSpeed

The previous code is a bit complex to understand, particularly if you do not come from an HPC background. An easier way to implement this is to use a library that handles all communication under the hood. Here we will use [DeepSpeed Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/).  

Our previous implementations was a GPU-only implementation of a data- and model-parallel MoE. DeepSpeed extends this by supporting CPU and GPU memory and to mix different parallelism techniques. Table from the [DeepSpeed MoE post](https://www.deepspeed.ai/tutorials/mixture-of-experts/#expert-groups-initialization):

|-|-|-|
| **Short Name** | **Flexible Parallelism Configurations** | **Benefit** |
|-|-|-|
| E | Expert | Scales the model size by increasing the number of experts |
| E+D | Expert + Data | Accelerates training throughput by scaling to multiple data parallel groups |
| E+Z | Expert + ZeRO-powered Data | Partitions the nonexpert parameters to support larger base models |
| E+D+M | Expert + Data + Model | Supports massive hidden sizes and even larger base models than E+Z |
| E + D + Z | Expert + Data + ZeRO-powered data | Supports massive hidden sizes and even larger base models than E+Z |
| E + Z-Off + M | Expert + ZeRO-Offload + Model | Leverages both GPU and CPU memory for large MoE models on limited # of GPUs |
|-|-|-|

```python
class MoE_ds(nn.Module):

  def __init__(self, input_size, output_size, num_experts, dropout=0.1):
    super().__init__()
    self.router = Router(input_size, num_experts, dropout=dropout)
    self.experts = nn.ModuleList([
      FeedForward(input_size, output_size, dropout=dropout) for _ in range(num_experts) ])

  def forward(self, x):
    probs = self.router(x)
    outputs = torch.stack([ expert(x) for expert in self.experts ], dim=-2) # B*T*Experts*C
    weighted_outputs = outputs * probs.unsqueeze(-1) # B*T*E*C x B*T*1*C -> B*T*E*C
    weighted_sum = weighted_outputs.sum(dim=-2) # sum over experts: B*T*E*C -> B*T*C
    return weighted_sum, probs, outputs
```

Remember that the previous code snippets cover the MoE module only, which is nothing more than a distributed representation of several feed-forward networks for experts and router. This is usually embedded as a single module of a larger module, as we will see next.

### 2017 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

The 2017 paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) introduces an text processing model, consisting of a recursive LSTM architecture with thousands of MoEs, totalling 137B parameters:

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2017_Dean.png"/>

Few changes were added to the typical MoE block:
- Instead of the typical softmax gating, the paper proposes a **noisy top-k gating**, where they add **tunable gaussian noise** $$H(X)$$, controlled by $$W_{noise}$$, to the input value of each experts. To handle sparsity, the output value of the non-picked experts to -∞ (which causes the corresponding gate values to equal 0)”:

$$
G(x) = softmax( \, KeepTopK ( \, H(x), k) )
$$

where

$$
H(x)_i = (x · W_g)_i + StandardNormal() · Softplus ( (x · W_{noise})_i )
$$


Note: the $$SoftPlus$$ is here used to constrain its output to be always positive.

- When applied to a language task, they take advantage of **Convolutionality** where each expert if applied to the same time step, recursively. This allows one to apply the MoE to all the time steps together as one big batch, increasing the size of the input batch by a factor of the number of unrolled time steps.
- To handle overpowering / starvation of experts, they adopted a soft constraing approach (as a side note, a hard approach would be fixing a maximum cap threshold). In practice, they add an additional **importance loss**, equal to the square of the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) of the set of importance values, as:

$$
L_{importance} = w_{importance} \cdot CV \, (Importance (X))^2
$$

where the coefficient of variation ($$CV$$, or relative standard deviation) represents the dispersion of the probability distribution and is defined as the ratio of the standard deviation $$\sigma$$ to the mean $$\mu$$ as $$CV= \frac {\sigma }{\mu }$$. The **Importance** of an expert is computed as the batchwise sum of the gate values for that expert, and $$w_{importance}$$ is a hand-tuned scaling factor. As an example, if you had a batch of 4 samples $$x_i$$ assigned across 5 experts $$E_j$$, with the following distribution:

|-|-|-|-|-|-|
| | $$E_1$$ | $$E_2$$ | $$E_3$$ | $$E_4$$ | $$E_5$$ |
|-|-|-|-|-|-|
| $$x_1$$ | 0.1 | 0.1 | 0 | 0.8 | 0 |
| $$x_2$$ | 0 | 0 | 0.2 | 0.7 | 0.1 |
| $$x_3$$ | 0.1 | 0 | 0 | 0.9 | 0 |
| $$x_4$$ | 0 | 0 | 0 | 1 | 0 |
| **Importance (sum)** | **0.2** | **0.1** | **0.2** | **2.4** | **0.1** |
|-|-|-|-|-|-|

In this example, the mean of the importance is 0.6, and standard deviation is 0.9, thus the CV is 0.667. If experts would be assigned a similar importance, then the variance would've been smaller and the CV also smaller, thus reducing the importance loss. 

This loss tries to balance overall importance across experts, but experts may receive different numbers of examples. This may lead memory and compute imbalance on distributed hardware, and to having experts that are undertrained. To solve this, a second **load loss** $$L_{load}$$ is introduced to encourages experts to receive a similar amount of training samples. However, note that the number of received tokens per expert is a constant and can not be backpropagated, so instead they use a smooth operator $$Load(X)$$ that can be back propagated, as:

$$
L_{load}(X)_i = \sum_{x \in X} P(x,i)
$$

where $$P(x,i) = Pr(h(x)_i) \gt kth\_excluding(H(x), k, i)$$ denotes  the probability that $$G(x)_i$$ is **non-zero,  given a new random choice of noise on element $$i$$, but keeping the already-sampled choices of noise on the other elements** (see Appendix A for details). Note that $$G(x)_i$$ is nonzero if and only if $$H(x)_i$$ is greater than the $$k$$th-greatest element of H(x).

The experiments fixed a compute budgets and compared a baseline model with several configurations of MoE, including a hierarchical MoE, on several language tasks. Note: a **hierarchical MoE** is a structure were a primary gating
network chooses a sparse weighted combination of experts, where each expert is by itself a MoE with its own gating network. Results show MoE models beating baseline LSTM models, with hierarchical MoE beating non-hierarchichal (with a higher parameter count, but same compute). The perplexity improved significantly by increasing data size (from 10B to 100B Word Google News Corpus). In the machine translation task, on the Google Production dataset, the model achieved 1.01 higher test BLEU score even after training for only one sixth of the time.

An important message in the results is that **different experts specialize on different tasks based on syntax and semantics**, as shown in Appendix E table 9:

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2017_Appendix_E_table_9.png"/>

Another important message in this paper is that it is suggested that learning to route does not work without the ability to compare at least two experts (due to instability purposes), i.e. we always need to backpropagate on least 2 experts on any MoE setup. In terms of scalability, the results claim to obtain "greater than 1000x improvements in model capacity with only minor losses in computational efficiency".

### 2020 [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

So far we covered deep MoE models based on recursive LSTM and stacks of MoEs. In 2020, GShard explored the scaling of Transformer-based Sparsely-Gated MoEs on a model of 600 billion parameters on a sequence-to-sequence (machine translation) task. They claim that the their methods yield sub-linear computation cost and graph $$O(1)$$ compilation time on 2048 TPU devices. This performance was achieved via:

- **Position-wise Sparsely-Gated MoEs**, just like the previous paper, that scales the capacity of RNN-based machine translation and language models and delivers sub-linear scaling;
- **GShard module that separates model description from the parallel partitioning**. It consists of "of a set of simple APIs for annotations, and a compiler extension in XLA for automatic parallelization." This allows developers to write the module and the runtime will automatically partition it accross compute unitrs;
- **compiler technique for SPMD (Single Program Multiple Data) transformation that generates a single program to run on all devices**, in order to keep the compilation time constant independent of the number of devices. 

The model used for the task presented is a Transformer-based language model, made of stacks of encoder and decoder layers. Each Transformer block (made of self-attention, feed-forward and on the decoder's case, attention mask) was converted into a conditional computation by **replacing the feed-forward layer in every second block with a MoE** layer with a variant of **top-2 gating** in both the encoder and the decoder. The encoder part can be represented as:

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Mixture-of-Experts/MoE_GShard.png"/>

Strangely enough, the paper does not mention why they introduced MoE in every second block, instead of every third or every single block instead. The parallelism follows the hybrib sharded-MoEs with data-parallel FFN and gating, as in the previous paper. As a side node, the "model-parallel MoE" in the picture refers to sharding and not to tensor/activation parallelism. The underlying formulation of the gating mechanism is the following: a MoE layer for Transformer consists of $$E$$ feed-forward networks $$FFN_1$$ ... $$FFN_E$$, formulated as:

$$
FFN_e(x_s) = wo_e · ReLU(wi_e · x_s)
$$

and

$$
y_x = \sum_{e=1}^{E} G_e(x_s) · FFN_e(x_s)
$$

where $$G_e(x_s)$$ is the probabilities (post-softmax) output by the gating mechanism for expert $$e$$ given input $$x_s$$, and $$y_x$$ is as before the weighted sum of products of the gating probabilities and the FFN outputs (FFN here refers to an expert). $$x_s$$ is the input token to the MoE layer, $$w_i$$ and $$w_o$$ are the input and output projection matrices for the feed-forward layer.

The total loss is described as $$L = l_{nll} + k ∗ l_{aux}$$, where $$l_{nll}$$ is the negative log-likelihood loss, and $$l_{aux}$$ is an **auxiliary loss** to help avoiding few experts *stealing* most of the attention at the early stages of training. This auxiliary loss is formulated as

$$
l_{aux} = \frac{1}{E} \sum^E_{e=1} \frac{c_e}{S} \, m_e
$$

where the term $$c_e/S$$ represents the fraction of input routed to each expert and $$m_e$$ is the mean gates per expert (note that $$m_e$$ is added because $$c_e/S$$ is a constant and is not differentiable).

**Load balancing** was regulated by guaranteeing that the total number of tokens assigned to each expert is not below a given threshold (this is in contrast to have a threshold in the number of samples). Finally, they introducted the concept of **random routing**, where in a top-2 setup, "if the weight for the 2nd expert is very small, we can simply ignore the 2nd expert to conserve the overall expert capacity". 

The parallel-partitioning is a combination of data-parallel and sharding: (1) "the attention layer is parallelized by splitting along the batch dimension and replicating its weights to all devices"; and (2) "experts in the MoE layer are infeasible to be replicated in all the devices due to its sheer size and the only viable strategy is to shard experts into many devices".

<!--
Finally, they mention **Mixing manual and automatic sharding**, where it allows user to manually specify the sharding by telling the runtime "on how operators are partitioned, and one example is that the user has more run-time knowledge beyond the operators’ semantics", more concretely the "user might know that a specific Gather operator shuffles data only within each partition".
However, they claim that "the automatic sharding assignment is not the focus of this paper and we leave it as future work".
-->

We start by taking the [GPT2-based `GPTlite` model we built on a previous post]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), and re-define all its modules as Distributed Data Parallel, except the feed-forward module in the transformer block, that we will replace by our mixture of experts:

```python
  model = GPTlite(vocab_size).to(device)
  model.token_embedding_table = DDP(model.token_embedding_table.to(device), device_ids=[local_rank])
  model.position_embedding_table = DDP(model.position_embedding_table.to(device), device_ids=[local_rank])
  model.ln = DDP(model.ln.to(device), device_ids=[local_rank])
  model.lm_head = DDP(model.lm_head.to(device), device_ids=[local_rank])
  for b, block in enumerate(model.blocks): 
    block.sa = DDP(block.sa.to(device), device_ids=[local_rank])
    block.ln1 = DDP(block.ln1.to(device), device_ids=[local_rank])
    block.ln2 = DDP(block.ln2.to(device), device_ids=[local_rank])
    if b%2==0: 
      block.ffwd = MoE().to(device) #replace FeedForward with Mixture of Experts
    else:
      block.ffwd = DDP(block.ffwd.to(device), device_ids=[local_rank])
```

### 2021 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

A big issue with large MoEs are training instability. With that in mind [Switch Transformers](https://arxiv.org/abs/2101.03961) tackle that problem by simplifying the routing algorithm and improving model specifications. There is also a strong motivational quote about the usage of MoEs, particularly to solve low-data problems:

> The benefit of scale was exhaustively studied in [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) which uncovered powerlaw scaling with model size, data set size and computational budget,  [and]
advocates training large models on relatively small amounts of data as the computationally optimal approach.
Heeding these results, we investigate a fourth axis: increase the parameter count while
keeping the floating point operations (FLOPs) per example constant.

In order to achieve scaling, the sparsely activated layers split *unique weights* on different devices. Just like in GShard, the authors replaced the Feed Forward Network with a MoE. However, in Switch transformers, every FFN is replaced (instead of every other), and each Switch Transformer layer received two inputs (tokens) on for experts.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_Switch_Transformers.png"/>

The regular top-$$k$$ routing mechanism used in common use cases was replaced by a **Switch Routing**.  Here, in contradiction to the paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), where the authors claim that you need to compute a loss of at least 2 experts in order to train an MoE successfully, in **Switch Transformers they route to only a single expert**. The claim is that "simplification preserves model quality, reduces routing computation and performs better. This $$k=1$$ routing strategy is later referred to as a Switch layer". This leads to several benefits: reduced batch size as only a single expert is used, simpler routing and reduced communication.

They introduce the concept of **expert capacity factor**, which is the number of tokens that each expert computes, computed as batch size divided by number of experts (as before), times a capacity factor. If this capacity factor is exceeded (ie if the router sends too many inputs to a given expert), then extra tokens do not have computation associeated with them and are instead passed to the next layer via a residual connection.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_Switch_Transformers_2.png"/>

> Increasing the capacity factor increases the quality but also increases the communication overheard and the memory in the activations. A recommended initial setup is to use an MoE with a top-2 routing with 1.25 capacity factor, with one expert per core. At evaluation time, the capacity factor can be changed to reduce compute.

One of the downsides of MoEs is the large number of parameters. If one wants to run it on a smaller network we can perform distilation: 

> Successful distillation of sparse pre-trained and specialized fine-tuned models into
small dense models. We reduce the model size by up to 99% while preserving 30% of
the quality gains of the large sparse teacher.

## Towards the future: tweaking, finetuning and improvements

### 2022 [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)

MegaBlocks is "a system for efficient Mixture-of-Experts (MoE) training on GPUs", that addresses the **model quality vs hardware efficiency tradeoff** on the dynamic routing of MoE layers. In detail, the load-imbalanced computation on MoEs forces one to either (1) drop tokens from the computation or (2) waste computation and memory on padding, a parameter that is usually improved via hyperparameter tuning. However, we know that "The loss reached by the MoE models decreases significantly as expert capacity is increased, but at the cost of additional computation" and "the lowest loss is achieved by the 'max expert capacity value', which avoids dropping tokens through the dynamic
capacity factor mechanism proposed by [Adaptive
mixture-of-experts at scale (2002)](#)". So not dropping tokens is ideal.

As a second motivation, "the sparse computation in MoEs does not map cleanly to the software primitives supported in major frameworks and libraries". 
 To overcome these limitations, MegaBlocks "reformulates MoE computation in terms of block-sparse operations and develop new block-sparse GPU kernels that efficiently handle the dynamism present in MoEs".
In practice, it proposes an approach of MoE routing based on **sparse primitives, that matches efficiently to GPUs and never drops tokens**, " enabling end-to-end training speedups of up to 40% and 2.4× over state of the art". This is achieved by performing block-sparse operations in the MoE layers to accommodate imbalanced assignment of tokens to experts. This is called **dropless-MoEs (dMoEs)** and contrasts with regular MoEs that use batched matrix multiplication instead. These dropless-MoEs kernels use blocked-CSR-COO encoding and transpose indices to enable efficient matrix products with sparse inputs.


### 2022 [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)

Formally studies "how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model". It experiments on a 4-cluster classification vision task and claims that: (1) "cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE"; (2) there needs to be non-linearity in the experts for an MoE system to work; and (3) "the router can learn the cluster-center features, which helps divide the input complex problem into simpler linear classification sub-problems that individual experts can conquer".

### 2022 [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)

Introduces GLaM (Generalist Language Model), a family of dense and sparse decoder-only language models, "which uses a sparsely activated mixture-of-experts architecture to scale the model capacity while also incurring substantially less training cost compared to dense variants". GLam has 1.2 trillion parameters (7x larger than ChatGPT-3) but requires 1/3 of the energy for training, and half of the computation flops for inference, while "achieving better overall zero, one and few-shot performance across 29 NLP tasks" (Table 1, Figure 1).

Results were collected by training several variants of GLaM to study the behavior of MoE and dense models on the same training data (Table 4). The study highlights the that:
- "the quality of the pretrained data also plays a critical role, specifically, in the performance of downstream tasks" (section 6.2);
- sparse models scale better (section 6.3);
- "MoE models require significantly less data than dense models of comparable FLOPs to achieve similar zero, one, and fewshot performance" (section 6.4).

### 2022 [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)

Presents a study with experiments, results and techniques related to the stability of training and fine-tuning of very large sparse MoEs. Also introduces the **router z-loss**, that resolves instability issues and improves slightly the model quality, and a 269B sparse model -- **the Stable Transferable Mixture-of-Experts** or ST-MoE-32B - that achieves state-of-the-art performance across several NLP benchmarks. The z-loss formulation is:

$$
L_z(x) = \frac{1}{B} \sum_{i=1}^B \left( \log \sum_{j=1}^N e^{x_j^{(i)}} \right)
$$

where $$B$$ is the number of tokens, $$N$$ is the number of experts, and $$x \in \mathbb{R}^{B×N}$$ are the logits going into the router. The rationale behind z-loss is that it penalizes large logits into the gating network.  The paper also contains a very extensive analysis of quality-stability trade-off techniques and results:

- "Sparse models often suffer from training instabilities (Figure 1) worse than those observed in standard densely-activated Transformers. [...] Some architectural improvements involve more multiplications than additions or do not sum many items at once": the paper studies GELU Gated Linear Units (GEGLU) and Root Mean Square Scale Parameters.
- Adding noise (input-jitter) and adding dropout improves stability, but leads to a significant loss of model quality (Section 3.2).
- Gradient clipping in updates (in Adafactor optimizer) improves stability, but at a high loss of quality (Section 3.3).
- Sparse expert models are sensitive to roundoff errors (eg due to lower precision representation) because they have more exponential functions due to the routers (Section 3.4).
- Sparse models are prone to overfit (Fig. 3).
- Adding dropout at 0.1 provides modest boosts while higher rates penalizes performance; and selectively increasing experts dropout increase accuracy (Fig. 4).
- Updating the non MoE parameters (while freezing others) yields an accuracy similar to updating all the parameters; and updating only the FFN parameters works a bit better (Fig 5).
- "Sparse models benefit from noisier hyperparameters including small batch sizes and high learning rates. Dense models behave nearly oppositely" (Fig. 6).
- "Sparse models are robust to dropped tokens when fine-tuning. [...] dropping 10-15% of tokens can perform approximately as well as models that drop < 1%" (Table 5).

The paper provides the following 3 recommendations when designing sparse systems:
> 
- top-2 routing with 1.25 capacity factor and at most one expert per core.
- the capacity factor can be changed during evaluation to adjust to new memory/compute requirements.
- dense layer stacking and a multiplicative bias can boost quality (Appendix C).

Related to the functioning of mixture of experts, the authors showed that **encoder experts learn very shallow tasks**, e.g. ponctuation, nouns, etc. I.e., **MoEs subdivide a problem into smaller problems that can solved by different experts combined**. The authors show this by "visualizing how tokens are routed among (encoder) experts, [...] by passing a batch of tokens to the model and manually inspecting token assignment at each layer". They observed that at each layer, at least one expert specializes in **sentinel tokens** (mask tokens that represent blanks to fill-in); and some encoder experts exhibit clear specialization, with some experts primarily operating on punctuation, verbs, proper names, counting, etc. These are detailed in table 13 and 15.

At the decoder level, "expert specialization is far less noticeable in the decoder", and meaningful specialization (semantics or syntax) are not visible in decoder experts:
  > We hypothesize that this lack of meaningful expert specialization is caused by the distribution of
target tokens induced by the span corruption objective. In particular, (a) a smaller number of tokens
are routed jointly in the decoder due to longer sequence lengths in the encoder (e.g. group size
is 2048 in the encoder vs 456 in the decoder in our setup) and (b) a higher proportion of tokens
are sentinel tokens in the decoder. As a result, target tokens in each group typically cover a smaller
semantic space (compared to the encoder), perhaps explaining the lack of expert specialization in the
decoder

### 2023 [Mixture-of-Experts Meets Instruction Tuning: a Winning Combination for Large Language Models](https://arxiv.org/abs/2305.14705)

This paper claims that **Instruction tuning supplemented by further finetuning on individual downstream tasks outperforms fine-tuning or instruction-tuning alone**. To show that they perform single-task fine-tuning, multi-task instruction-tuning and multi-task instruction-tuning followed by single-task fine-tuning. They showed that MoEs benefit more from instruction tuning than other models and benefit more from a higher number of tasks. In practice:
  > in the absence of instruction tuning, MoE models fall short in performance when compared to dense models on downstream tasks. [...] When supplemented with instruction tuning, MoE models exceed the performance of dense models on downstream tasks, as well as on held-out zero-shot and few-shot tasks.

### 2024 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

The current state-of-art MoE model is the Mixtral 8x7B, a sparse mixture of expert, with the same architecture as Mixtral 7B except that it supports a fully dense context length of 32k tokens and the feed-forward blocks are replaced by a Mixture of 8 feed-forward network experts. Model architecture is detailed in Table 1. It performs a top-2 routing, and because tokens can be allocated to different experts, "each token has access to 47B parameters, but only uses 13B active parameters during inference". 

The formulation of the model, in section 2.1, matches the GShard architecture (except that the MoE is in every layer instead of every second), with sparse top-$$k$$ MoE with gating $$G(x) = Softmax(TopK(x · Wg))$$ where $$TopK(ℓ)= −∞$$ for the non top-$$k$$ experts. As before, "in a Transformer model, the MoE layer is applied independently per token and replaces the feed-forward (FFN) sub-block of the transformer block". They use the [SwiGLU activation function](https://arxiv.org/abs/2002.05202v1) (same as Llama-2), and therefore for a given input $$x$$ the output is computed as:

$$
y = \sum_{i=0}^{n-1} Softmax(Top2(x · W_g))_i· SwiGLU_i(x).
$$

### 2024 [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

As background, remember that we mentioned before that MoEs work by subdividing the problem domain into smaller problems that are solved individually by different experts. Now think that, not all problems require the same amount of effort to train a model accurately. This is explored in this work, via **Conditional computation**, a technique that "tries to reduce total compute by expending it only when needed". Here, "the network must learn how to dynamically allocate the available compute by making decisions per-token, in each layer, about where to spend compute from the availablebudget" - where this compute budget is a total number of FLOPs pre-defined beforehand by the user and remains unchanged throught the execution.

In this work, for every token, the model decides to either apply a computation (as in the standard transformer), or skip the computation and passing it through a residual connection (remaining unchanged and saving compute). Moreover, contrarily to MoE, the **routing is applied to both the feedforward network and the multi-head attention**. In the multi-head routing, the router will not only decide on which tokens to update, but which tockens are made available to attend to. "We refer to this strategy as Mixture-of-Depths (MoD) to emphasize how individual tokens pass through different numbers of layers, or blocks, through the depth of the transformer". In practice: (1) the user picks a fixed compute budget beforehand; (2) during training, for every input, the router produces a scalar weight (importance) per token; and (3) we pick the top-$$k$$ tokens per sentence per block to participate in the transformer block computation. $$k$$ is a hyper-parameter that defined the max number of tokens passed to a block, thus the computation graph and tensor sizes remain static throughout the execution. As $$k$$ is set to be smaller than the sentence length, MoD allows one to trade off between performance and speed, while achieving a high level of accuracy for a given compute budget.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/Mixture_of_Depths.png"/>

The big challenge is the routing scheme. The authors considered (1) token-choice where  a router produces per-token probability distributions across computational paths (experts); and the orthogonal approach (2) expert-choice where instead of having tokens choose the expert they prefer, each expert instead picks the top-$$k$$ tokens based on the tokens’ preferences. They ultimately adopted the expert-choise because it does not require load balancing, and "routers can try to ensure that the most critical tokens are among the
top-$$k$$ by setting their weight appropriately".

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/Mixture_of_Depths_2.png"/>

At the time of writing of this post, this is still very recent work, so future will tell if MoDs become useful for the general use case.
