---
layout: post
title:  "Tensor parallelism like Megatron-LM"
categories: ["Distributed Training"]
tags: ["tensor parallelism", "Megatron-LM", "sequence parallelism", "spatial parallelism", "collectives", "all-reduce", "PyTorch"]
description: "Tensor parallelism as in Megatron-LM: the column-parallel and row-parallel split of the MLP and attention blocks, the f and g operators in PyTorch, sequence parallelism, and spatial parallelism for CNNs."
---

In 2012, the network AlexNet, from the paper [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), was too large for the 3 GB of memory of a single GPU. Its authors placed half of the neurons of each layer on each of two GPUs, which exchanged activations only at some layers: an early form of what is now called tensor parallelism.

This post discusses tensor (model) parallelism, particularly the [Megatron-LM](https://www.deepspeed.ai/tutorials/megatron/) implementation. Tensor parallelism is one more dimension of parallelism, in a universe of data, tensor, pipeline, context, sequence and expert parallelism.

**Tensor parallelism** aims at partitioning computation across the model’s tensor dimensions (e.g., sharding linear layer weights and corresponding activations). This is a hard problem: in practice, the dimension of tensor partitioning (row-wise, column-wise, or none) must be decided, and the communication and computation adapted accordingly. Therefore, it is a model- and data-specific implementation. The following picture illustrates this partitioning on a small fully-connected network:

{: .picture}
<img width="58.5%" height="58.5%" src="{{ site.assets }}/distributed-training/imgs/dnn_model_parallelism.svg"/>

{: .caption}
A fully-connected network (left) and the same network split by tensor parallelism over two processors $$P_0$$ and $$P_1$$ (right), each holding half of the neurons of every layer. Red lines are the connections between neurons on different processors, whose activations must be communicated.

The previous picture shows a major drawback of this method: during training, processors need to continuously communicate the activations that are needed across the processor network. In the picture, 16 of the 32 connections (the red ones) link neurons on different processors, as each neuron is connected to all the neurons of the next layer, half of which are on the other processor. With $$P$$ processors, a fraction $${(P-1)/P}$$ of the connections of every layer crosses processors. This communication is synchronous and can limit overlap with compute (depending on the implementation). And it can require a large amount of communication at every layer of the network and for every input batch.

There are several alternative ways to distribute multi-dimensional tensors across several compute nodes, which aim at reducing the number of communication steps or the communication volume. This post details and implements the Megatron-LM approach, and then looks at the sharding of its activations (sequence parallelism) and at tensor parallelism on convolutional networks (spatial parallelism).

## Megatron-LM tensor parallelism

A better understanding can be gained by replicating the partitioning suggested by Megatron-LM in the paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). The paper trains a language model of 8.3 billion parameters, similar to GPT-2, on 512 GPUs: 64 copies of the model (data parallelism), each split across 8 GPUs by tensor parallelism. The main rationale in Megatron-LM is that Transformer-based models have two main components, a feed-forward (MLP) and an attention block. A forward and a backward pass on each of these blocks can be done with a single [collective communication](https://en.wikipedia.org/wiki/Collective_operation) step each, i.e., two all-reduces in the forward pass and two in the backward pass for each transformer layer. The two blocks, split over two GPUs, are illustrated in the following picture (Figure 3 in the paper):

{: .picture}
<img width="76.4%" height="76.4%" src="{{ site.assets }}/distributed-training/imgs/megatron-lm.svg"/>

{: .caption}
(a) The MLP block and (b) the self-attention block of Megatron-LM, split over two GPUs: the first weight matrix $$A$$ is split by columns ($$Q$$, $$K$$ and $$V$$ by attention heads), and the second one, $$B$$, by rows. $$f$$ and $$g$$ are conjugate operators: $$f$$ is an identity operator in the forward pass and an all-reduce in the backward pass, while $$g$$ is an all-reduce in the forward pass and an identity in the backward pass. Adapted from: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

### The MLP block: column-parallel and row-parallel linear layers

The rationale is the following: the MLP is a sequence of (1) a matrix multiplication (MatMul) $$XA$$, (2) a GeLU (Gaussian Error Linear Unit) activation, whose output is $$H$$, (3) a MatMul $$HB$$ and (4) a dropout layer, where $$X$$ is the input, with one row per token, and $$A$$ and $$B$$ are the weight matrices of the two linear layers. A MatMul is also called a GEMM, after the general matrix multiplication of [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). Consider the first two operations:
1. "One option to parallelize the GEMM is to split the weight matrix $$A$$ along its rows and input $$X$$ along its columns. [...] This partitioning will result in $$Y = \operatorname{GeLU}(X_1 A_1 + X_2 A_2)$$. Since GeLU is a nonlinear function, $$\operatorname{GeLU}(X_1 A_1 + X_2 A_2) \neq \operatorname{GeLU}(X_1 A_1) + \operatorname{GeLU}(X_2 A_2)$$ and this approach will require a synchronization point [*sum-reduce*] before the GeLU function". This approach can be followed for each of the 2 MatMuls, yielding a total of two communication steps.

2. Another option is to not split $$X$$ and to split $$A$$ along its columns, which "allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM". The output of the GeLU $$H$$ is then $$[H_1, H_2] = [\operatorname{GeLU}(X A_1), \operatorname{GeLU}(X A_2)]$$, and this removes a synchronization point.
After the MatMul and GeLU are done, they are followed by another MatMul and a dropout. Note that with this approach (pictured below), the output $$H$$ of the first MatMul and GeLU (which is the input for the second MatMul) is split column-wise. Therefore, Megatron uses the previous approach for the second MatMul, which requires 1 collective communication step. In total, 1 communication step is performed for 2 MatMuls.

{: .picture}
<img width="92.5%" height="92.5%" src="{{ site.assets }}/distributed-training/imgs/tp_megatron_lm.svg"/>

{: .caption}
The MLP block of Megatron-LM on two GPUs. The first MatMul follows option 2: $$X$$ is replicated and $$A$$ is split by columns, so each GPU computes its own columns of $$XA$$ and applies the GeLU to them without any communication. The second MatMul follows option 1: $$H$$ is already split by columns and $$B$$ is split by rows, so each GPU computes a partial sum of $$HB$$, and a sum-reduce adds the partial sums on both GPUs. Inspired by: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

The two linear layers of the MLP are named after the split of their weights: the first one is **column-parallel**, as each GPU holds some of the columns of $$A$$, and the second one is **row-parallel**, as each GPU holds some of the rows of $$B$$. A column-parallel layer takes a replicated input and gives an output split by columns, and a row-parallel layer takes an input split by columns and gives partial sums that must be added.

As an example, consider the MLP block of GPT-3 Small, from the paper [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), on 2 GPUs. Its embedding size is $${E = 768}$$ ($$N$$ in the picture above), and its hidden layer has $${4E = 3072}$$ neurons ($$O$$). For one sequence of 2,048 tokens ($$M$$), $$X$$ is a $$2048 \times 768$$ matrix, held in full by both GPUs. The weights $$A$$ ($$768 \times 3072$$) are split into $$A_1$$ and $$A_2$$ of $$768 \times 1536$$, and $$B$$ ($$3072 \times 768$$) into $$B_1$$ and $$B_2$$ of $$1536 \times 768$$, so each GPU holds 2,359,296 of the 4,718,592 weights of the block. Each GPU computes its $$2048 \times 1536$$ part of $$H$$ and a $$2048 \times 768$$ partial sum of $$HB$$, and the sum-reduce (an all-reduce) adds the two partial sums: 1,572,864 values, or 3.1 MB in BF16.

The cost of this communication is best seen against the computation. In the forward pass of the block, each GPU computes 9.7 GFLOPs (2 FLOPs per multiply-add of its two GEMMs) and sends 3.1 MB, as a ring all-reduce over $$P$$ GPUs (a reduce-scatter followed by an all-gather) sends $${2(P-1)/P}$$ times the size of the data from each GPU. This is 3,072 FLOPs per byte sent. With 8 GPUs, each GPU computes 4 times less but sends 1.75 times more, and the ratio drops 7 times, to 439 FLOPs per byte. In general, the ratio is $${4E/(P-1)}$$ FLOPs per byte in BF16: tensor parallelism suits wide models split over a few GPUs with fast links between them.

### The operators f and g

Megatron-LM makes this implementation very simple, by adding only the $$f$$ and $$g$$ functions to the serial use case. Following the paper: "$$f$$ is an identity operator in the forward pass and all reduce in the backward pass while $$g$$ is an all reduce in the forward pass and identity in the backward pass".

The intuition is that $$f$$ marks where the split part of the block starts, and $$g$$ where it ends. In the forward pass, every GPU reads the same input $$X$$, so $$f$$ has nothing to do. In the backward pass, each GPU computes only the part of the gradient of $$X$$ that comes from its own columns of $$A$$, and the gradient of $$X$$ is the sum of these parts:

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial (X A_1)} \, A_1^T + \frac{\partial L}{\partial (X A_2)} \, A_2^T$$

where $$L$$ is the loss and GPU $$i$$ computes the term of $$A_i$$, so the all-reduce of $$f$$ adds them up. $$g$$ does the opposite: in the forward pass, it adds the partial sums $$H_1 B_1$$ and $$H_2 B_2$$; in the backward pass, both partial sums receive the gradient of their sum, which every GPU already holds, so $$g$$ has nothing to do. In brief, the backward pass of a copy is a sum, and the backward pass of a sum is a copy: this is why $$f$$ and $$g$$ are conjugate. In PyTorch, each of them is an autograd function, with its own forward and backward pass:

```python
class Megatron_f(torch.autograd.Function):
  """ The f function in Figure 3 in Megatron paper """

  @staticmethod
  def forward(ctx, x, mp_comm_group=None):
      ctx.mp_comm_group = mp_comm_group
      return x

  @staticmethod
  def backward(ctx, gradient):
      dist.all_reduce(gradient, dist.ReduceOp.SUM, group=ctx.mp_comm_group)
      return gradient, None
```

and

```python
class Megatron_g(torch.autograd.Function):
  """ The g function in Figure 3 in Megatron paper """

  @staticmethod
  def forward(ctx, x, mp_comm_group=None):
      dist.all_reduce(x, dist.ReduceOp.SUM, group=mp_comm_group)
      return x

  @staticmethod
  def backward(ctx, gradient):
      return gradient, None
```

Note that the code adds an extra argument `mp_comm_group` that refers to the model-parallel communication group, which allows model parallelism to be combined with other types of parallelism. As an example, 8 GPUs can be split into 2 model-parallel groups of 4 GPUs, each holding a copy of the model, and the GPUs that hold the same shard form 4 data-parallel groups of 2 GPUs. Tensor parallelism is now added to the MLP by inserting $$f$$ and $$g$$ in the forward pass, at the beginning and end of the block, as in the paper. In the code, `n_embd_mid` is the hidden size on each GPU (1,536 for GPT-3 Small on 2 GPUs):

```python
class Megatron_FeedForward(nn.Module):
  """ the feed forward network (FFN) in the paper, with tensor parallelism as in Megatron-LM MLP block"""

  def __init__(self, n_embd, mp_comm_group=None):
    super().__init__()
    self.mp_comm_group = mp_comm_group

    #Fig 3a. MLP: splits first GEMM across columns and second GEMM across rows
    n_embd_mid = n_embd * 4
    if self.mp_comm_group:
        n_embd_mid //= dist.get_world_size(group=self.mp_comm_group)

    self.fc1 = nn.Linear(n_embd, n_embd_mid)
    self.fc2 = nn.Linear(n_embd_mid, n_embd, bias=False)   # <-- no bias here
    self.fc2_bias = nn.Parameter(torch.zeros(n_embd))      # <-- bias added after all-reduce
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    if self.mp_comm_group:
        x = Megatron_f.apply(x, self.mp_comm_group)

    y = F.gelu(self.fc1(x))

    z = self.fc2(y)  # matmul only (partial)

    if self.mp_comm_group:
        z = Megatron_g.apply(z, self.mp_comm_group)

    z = z + self.fc2_bias  # <-- bias AFTER all-reduce
    z = self.dropout(z)
    return z
```

Note that, for correctness, the sum-reduce (the all-reduce of $$g$$) is added after the MatMul but before the bias of the second linear layer, as otherwise the bias would be added once per GPU. The bias of the first linear layer stays in it, split with the columns of $$A$$, as each GPU adds the bias of its own columns.

### The attention block

The algorithm of the attention mechanism is analogous, except that instead of a single MatMul $$XA$$ at the beginning of the workflow, there are three MatMuls $$XV$$, $$XQ$$ and $$XK$$ for the value, query and key matrices. Therefore, the attention block also requires one (not two) communication step, as the MLP block. Its parts are split as follows:

- The Q, K and V projections are column-parallel: their output features (the heads) are split, so each GPU (a rank, in the code) owns a subset of the heads, e.g., 6 of the 12 heads of GPT-3 Small on 2 GPUs.
- The attention softmax is computed locally on each GPU, because the heads are independent of each other.
- The output projection is row-parallel, and that is where the all-reduce of $$g$$ is done, to add up the partial outputs of the GPUs.

The multi-head attention block can then be written as follows, where `B`, `S` and `E` are the batch size, the sequence length and the embedding size (this `B` is not the weight matrix $$B$$ of the MLP):

```python
class Megatron_MHA(nn.Module):
  """
  Megatron-LM tensor-parallel Multi-Head Self-Attention (Fig. 3b style):

  - Q/K/V GEMM is column-parallel: each rank owns a subset of heads (local heads).
    Use Megatron_f on the input (identity forward, all-reduce backward) to match
    column-parallel input-gradient behavior.

  - Attention (softmax path) is fully local per rank (no communication).

  - Output projection is row-parallel: each rank projects its local concat heads
    to n_embd, then Megatron_g all-reduces the output (sum) across ranks.
  """

  def __init__(self, n_embd, n_head, block_size, dropout, mp_comm_group=None):
    super().__init__()
    self.mp_comm_group = mp_comm_group
    self.n_embd = n_embd
    self.n_head = n_head

    self.tp_size = dist.get_world_size(group=mp_comm_group) if mp_comm_group else 1
    if n_head % self.tp_size != 0:
      raise ValueError(f"n_head ({n_head}) must be divisible by tp_size ({self.tp_size}).")

    self.n_head_local = n_head // self.tp_size

    if n_embd % n_head != 0:
      raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head}).")
    self.head_dim = n_embd // n_head
    self.hidden_local = self.n_head_local * self.head_dim

    # Column-parallel QKV: each rank produces Q,K,V only for its local heads.
    self.qkv = nn.Linear(n_embd, 3 * self.hidden_local, bias=False)

    # Row-parallel output projection: consumes local heads and produces full n_embd,
    # then we all-reduce (Megatron_g) to combine ranks.
    self.proj = nn.Linear(self.hidden_local, n_embd, bias=False)

    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.attn_dropout = nn.Dropout(dropout)
    self.out_dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, S, E = x.shape
    if E != self.n_embd:
      raise ValueError(f"Expected x.shape[-1] == {self.n_embd}, got {E}.")

    # Column-parallel input handling: identity forward, all-reduce on backward.
    if self.mp_comm_group:
      x = Megatron_f.apply(x, self.mp_comm_group)

    # Local QKV for local heads
    qkv = self.qkv(x)  # (B, S, 3*hidden_local)
    q, k, v = qkv.split(self.hidden_local, dim=-1)

    # (B, S, hidden_local) -> (B, n_head_local, S, head_dim)
    q = q.view(B, S, self.n_head_local, self.head_dim).transpose(1, 2)
    k = k.view(B, S, self.n_head_local, self.head_dim).transpose(1, 2)
    v = v.view(B, S, self.n_head_local, self.head_dim).transpose(1, 2)

    # Attention scores: (B, n_head_local, S, S)  -- LOCAL, no comm
    wei = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

    # Causal mask (broadcasts over B and n_head_local)
    wei = wei.masked_fill(self.tril[:S, :S] == 0, float("-inf"))

    wei = F.softmax(wei, dim=-1)
    wei = self.attn_dropout(wei)

    # Context: (B, n_head_local, S, head_dim)
    out = wei @ v

    # Concat heads: (B, S, hidden_local)
    out = out.transpose(1, 2).contiguous().view(B, S, self.hidden_local)

    # Row-parallel output projection: local matmul then all-reduce sum on forward
    out = self.proj(out)  # (B, S, n_embd) partial contribution
    if self.mp_comm_group:
      out = Megatron_g.apply(out, self.mp_comm_group)

    out = self.out_dropout(out)
    return out
``` 

### Sequence parallelism: sharding the activations

The main caveat of this method is that the input `X` is replicated across all GPUs. This is not ideal, as large inputs may require a large memory chunk and may not fit in one GPU. This is particularly relevant during training, where activations during the forward pass need to be stored for the backward pass.

To reduce the memory consumption, one can hold only a shard of the input data and activations on each GPU, split along the sequence dimension. This is the **sequence parallelism** of [Korthikanti et al. (Reducing Activation Recomputation in Large Transformer Models, 2022)](https://arxiv.org/abs/2205.05198). It applies to the operations between two blocks (the layer normalizations, the dropouts and the residual connections), which treat each token independently, so each GPU can apply them to its own tokens.

To do this, the all-reduce is replaced with a reduce-scatter, so that the output of the reduction is sharded across GPUs. An additional all-gather is then added at the beginning of the block, to gather all shards of `X` into the full `X`, as required by the first MatMul. In the backward pass, the two collectives swap roles, as $$f$$ and $$g$$ do: the all-gather becomes a reduce-scatter, and the reduce-scatter an all-gather.

As for activations, only the shard of `X` produced between the reduce-scatter of the current block and the all-gather of the next one needs to be kept. With this, the number of collective communication steps increases from 1 to 2, but the activation memory is reduced linearly with the number of GPUs. The communication volume stays the same, as an all-reduce is a reduce-scatter followed by an all-gather.

The paper counts the memory that this saves. Outside the attention scores, a transformer layer stores $$34E$$ bytes of activations per token for the backward pass, in 16-bit precision. Tensor parallelism splits the $$24E$$ bytes stored inside the two blocks, and leaves $$10E$$ bytes replicated on every GPU: the inputs of the two blocks and of the two layer normalizations, and the masks of the two dropouts after the blocks. Sequence parallelism splits these too. With 2 GPUs, each GPU then stores $$17E$$ instead of $$22E$$ bytes per token, and, with 8 GPUs, $$4.25E$$ instead of $$13E$$: the replicated part dominates as the number of GPUs grows.

## Detour: spatial parallelism on Convolutional Neural Nets

The whole point of Megatron-LM was to reduce the number of communication steps, due to the high volume of data involved per step. However, it is worth mentioning that tensor parallelism has some use cases where it can be efficient and require little communication. An example is the parallelism of [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNNs).

In practice, due to the convolution filter (kernel) of CNNs, which has a short spatial span, the activations to be communicated are limited to the ones that neighboring activations need. This method, called **spatial parallelism**, has been detailed by [Dryden et al. (Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism, Proc. IPDPS 2019)](https://arxiv.org/pdf/1903.06681.pdf). The method is illustrated in the picture below and works as follows:
1. Input data and activations are split across the height and width dimensions among processors;
2. For a given convolutional layer, the convolution can be computed independently by each processor, with the exception of the activations at the split boundaries. These activations (the *halo region* in purple in the picture below) will need to be communicated at every forward/backward step.

{: .picture}
<img width="91.9%" height="91.9%" src="{{ site.assets }}/distributed-training/imgs/argonne_parallel_2.svg"/>

{: .caption}
Two ways to parallelize a convolutional neural network. (a) Spatial parallelism: the activations of a 2D input are split across four processors $$P_0$$ to $$P_3$$. The red box is where the 3x3 convolution filter is centered, the red arrows are the data that must be received from the other processors to compute it, and the purple region is the halo, formed by all the elements exchanged at every step. (b) Sample (data) parallelism on two processors $$P_0$$ and $$P_1$$, with a global mini-batch of 4 samples. Blue arrows are the dependencies of the forward pass, which computes the output $$y$$ from the input $$x$$ and the filters $$w$$; green arrows are those of the backward pass, which computes the gradients of the filters, $$dL/dw$$, and of the input, $$dL/dx$$, from $$dL/dy$$. The labels eq 1 to eq 3 refer to the equations of the paper. The only communication between $$P_0$$ and $$P_1$$ is the all-reduce of $$dL/dw$$. Adapted from: [Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism](https://arxiv.org/abs/1903.06681)

For the sake of comparison, the sample (data) parallelism shown in (b) is limited by the mini-batch size, and by the memory that a whole sample needs on one processor; spatial parallelism has neither limit, but adds the halo exchanges. These exchanges are small: with a 3x3 filter, the halo is one element wide, so a processor that holds a $$512 \times 512$$ quarter of a $$1024 \times 1024$$ input receives 1,025 values per channel, 0.4% of the values it holds. In practice, spatial parallelism suits datasets with very large samples, and the paper combines it with sample parallelism.

## Final remarks

PyTorch now has an [API for tensor parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html), still marked as experimental. It is built on distributed tensors (DTensor), and the user picks a parallel style for each layer: a column-wise split (`ColwiseParallel`), a row-wise split (`RowwiseParallel`), or the sequence parallelism of this post (`SequenceParallel`). Also, combining data, pipeline and tensor parallelism requires one to define the correct strategy for [custom model parallelism](https://www.deepspeed.ai/training/#model-parallelism).

To find more about the topic, [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) combines tensor, pipeline and data parallelism on up to 3072 GPUs. It recommends tensor parallelism inside a server, where the GPUs share fast links, and pipeline parallelism across servers.






