---
layout: post
title:  "Tensor parallelism"
categories: ["Distributed Training"]
tags: ["tensor parallelism", "Megatron-LM", "collectives", "PyTorch"]
---

This post discusses tensor (model) parallelism, particularly the [Megatron-LM](https://www.deepspeed.ai/tutorials/megatron/) implementation. Tensor parallelism is just one more dimension of parallelism, in a universe of data, tensor, pipeline, context, sequence and expert parallelism.

Tensor parallelism aims at partitioning computation across the model’s tensor dimensions (e.g., sharding linear layer weights and corresponding activations). This is a hard problem: in practice, the dimension of tensor partitioning (row-wise, column-wise, or none) must be decided, and the communication and computation adapted accordingly. Therefore, it is a model- and data-specific implementation.

{: .picture}
<img width="54.6%" height="54.6%" src="{{ site.assets }}/distributed-training/imgs/dnn_model_parallelism.svg"/>

{: .caption}
A fully-connected network (left) and the same network split by tensor parallelism over two processors $$P_0$$ and $$P_1$$ (right), each holding half of the neurons of every layer. Red lines are the connections between neurons on different processors, whose activations must be communicated.

The previous picture shows a major drawback of this method: during training, processors need to continuously communicate the activations that are needed across the processor network. This communication is synchronous and can limit overlap with compute (depending on the implementation). And it can require a large amount of communication at every layer of the network and for every input batch.

There are several alternative ways to distribute multi-dimensional tensors across several compute nodes, which aim at reducing the number of communication steps or the communication volume. This post details and implements the Megatron-LM approach.


## Megatron-LM tensor parallelism

A better understanding can be gained by replicating the partitioning suggested by Megatron-LM in the paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). The main rationale in Megatron-LM is that Transformer-based models have two main components, a feed-forward (MLP) and an attention block, and that a forward and a backward pass on each of these blocks can be done with a single [collective communication](https://en.wikipedia.org/wiki/Collective_operation) step each.

{: .picture}
<img width="76.4%" height="76.4%" src="{{ site.assets }}/distributed-training/imgs/megatron-lm.svg"/>

{: .caption}
(a) The MLP block and (b) the self-attention block of Megatron-LM, split over two GPUs: the first weight matrix $$A$$ is split by columns ($$Q$$, $$K$$ and $$V$$ by attention heads), and the second one, $$B$$, by rows. $$f$$ and $$g$$ are conjugate operators: $$f$$ is an identity operator in the forward pass and an all-reduce in the backward pass, while $$g$$ is an all-reduce in the forward pass and an identity in the backward pass. Adapted from: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

The rationale is the following: the MLP is a sequence of (1) a MatMul $$XA$$, (2) a GeLU $$Y$$, (3) a MatMul $$YB$$ and (4) a dropout layer. Consider the first two operations:
1. "One option to parallelize the GEMM is to split the weight matrix $$A$$ along its rows and input $$X$$ along its columns. [...] This partitioning will result in $$Y = GeLU(X_1 A_1 + X_2 A_2)$$. Since GeLU is a nonlinear function, $$GeLU (X_1 A_1 + X_2 A_2) \neq  GeLU( X_1 A_1 ) + GeLU (X_2 A_2)$$ and this approach will require a synchronization point [*sum-reduce*] before the GeLU function". This approach (pictured below) can be followed for each of the 2 MatMuls, yielding a total of two communication steps.

  {: .picture}
  <img width="84.6%" height="84.6%" src="{{ site.assets }}/distributed-training/imgs/megatron_model_parallel_1.svg">

  {: .caption}
  Option 1 on two GPUs: $$X$$ is split by columns and $$A$$ by rows. Each GPU computes a partial product of size $$M \times O$$, and a sum-reduce adds them into $$XA$$ on both GPUs before the GeLU, which then gives the same $$Y$$ on both GPUs. Inspired by: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)


2. Another option is to not split $$X$$ and to split $$A$$ along its columns, which "allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM". The output of the GeLU $$Y$$ is then $$[Y_1, Y_2] = [GeLU (X A_1), GeLU(X A_2)]$$, and this removes a synchronization point.
After the MatMul and GeLU are done, they are followed by another MatMul and a dropout. Note that with this approach (pictured below), the output $$Y$$ of the first MatMul and GeLU (which is the input for the second MatMul) is split column-wise. Therefore, Megatron uses the previous approach for the second MatMul, which requires 1 collective communication step. In total, 1 communication step is performed for 2 MatMuls.

  {: .picture}
  <img width="61%" height="61%" src="{{ site.assets }}/distributed-training/imgs/megatron_model_parallel_2.svg">

  {: .caption}
  Option 2 on two GPUs: $$X$$ is replicated and $$A$$ is split by columns, so each GPU computes its own columns of $$XA$$ and applies the GeLU to them without any communication. Inspired by: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)


The algorithm of the attention mechanism is analogous, except that instead of a single MatMul $$XA$$ at the beginning of the workflow, there are three MatMuls $$XV$$, $$XQ$$ and $$XK$$ for the value, query and key matrices of the attention mechanism. Therefore, this algorithm requires one (not two) communication step per MLP and per attention block.

Megatron-LM makes this implementation very simple, by adding only the $$f$$ and $$g$$ functions to the serial use case. Following the paper: "$$f$$ is an identity operator in the forward pass and all reduce in the backward pass while $$g$$ is an all reduce in the forward pass and identity in the backward pass". The all-reduce of $$f$$ in the backward pass is needed because each GPU computes only the part of the gradient of $$X$$ that comes from its own columns of $$A$$. So this can be implemented as:

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

Note that the code adds an extra argument `mp_comm_group` that refers to the model-parallel communication group, which allows model parallelism to be combined with other types of parallelism. As an example, 8 GPUs can be split into 2 model-parallel groups of 4 GPUs, each holding a copy of the model, and the GPUs that hold the same shard form 4 data-parallel groups of 2 GPUs. Tensor parallelism is now added to the MLP by inserting $$f$$ and $$g$$ in the forward pass, at the beginning and end of the block, just like in the paper:

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

Note that, for correctness, the sum-reduce is added after the matmul but before the bias of the second linear layer, as otherwise the bias would be added once per GPU. The attention block follows a similar approach:

- Q/K/V projections are column-parallel: the output features are split (i.e., the heads / hidden width), so each rank owns a subset of heads.
- The attention softmax is computed locally per rank (because each rank’s heads are independent).
- The output projection is row-parallel, and that is where the all-reduce is typically done (to combine partial output contributions).

The multi-head attention block can then be written as:

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

### Sharding activations to reduce memory (all-reduce = reduce-scatter + all-gather)

The main caveat of this method is that the input `X` is replicated across all GPUs. This is not ideal, as large inputs may require a large memory chunk and may not fit in one GPU. This is particularly relevant during training, where activations during the forward pass need to be stored for the backward pass. To reduce the memory consumption, one can hold only a shard/subset of the input data and activations on each GPU, split along the sequence dimension, as in the **sequence parallelism** of [Korthikanti et al. (Reducing Activation Recomputation in Large Transformer Models, 2022)](https://arxiv.org/abs/2205.05198). To do this, the all-reduce is replaced with a reduce-scatter (so that the output of the reduction is sharded across GPUs), and an additional all-gather is added at the beginning of the block (that will gather all shards of `X` into the full `X`, as required by the first matmul). As for activations, only the shard of `X` produced between the reduce-scatter of the current block and the all-gather of the next one needs to be kept. With this, the number of collective communication steps increases from 1 to 2, but the activation memory is reduced linearly with the number of GPUs. The communication volume stays the same, as an all-reduce is a reduce-scatter followed by an all-gather.

## Detour: tensor parallelism on Convolutional Neural Nets

The whole point of Megatron-LM was to reduce the number of communication steps, due to the high volume of data involved per step. However, it is worth mentioning that tensor parallelism has some use cases where it can be efficient and require little communication. An example is the parallelism of [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNNs). In practice, due to the kernel operator in CNNs (which has a short spatial span), the activations to be communicated are limited to the ones that neighboring activations need. This method, called **spatial parallelism**, has been detailed by [Dryden et al. (Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism, Proc. IPDPS 2019)](https://arxiv.org/pdf/1903.06681.pdf). The method is illustrated in the picture below and works as follows:
1. Input data and activations are split across the height and width dimensions among processors;
2. For a given convolutional layer, the convolution can be computed independently by each processor, with the exception of the activations at the split boundaries. These activations (the *halo region* in purple in the picture below) will need to be communicated at every forward/backward step.

{: .picture}
<img width="98.8%" height="98.8%" src="{{ site.assets }}/distributed-training/imgs/argonne_parallel_2.svg"/>

{: .caption}
Two ways to parallelize a convolutional neural network. (a) Spatial parallelism: the activations of a 2D input are split across four processors $$P_0$$ to $$P_3$$. The red box is where the 3x3 convolution filter is centered, the red arrows are the data that must be received from the other processors to compute it, and the purple region is the halo, formed by all the elements exchanged at every step. (b) Sample (data) parallelism on two processors $$P_0$$ and $$P_1$$, with a global mini-batch of 4 samples. Blue arrows are the dependencies of the forward pass, which computes the output $$y$$ from the input $$x$$ and the filters $$w$$; green arrows are those of the backward pass, which computes the gradients of the filters, $$dL/dw$$, and of the input, $$dL/dx$$, from $$dL/dy$$. The only communication between $$P_0$$ and $$P_1$$ is the all-reduce of $$dL/dw$$. Adapted from: [Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism](https://arxiv.org/abs/1903.06681)

For the sake of comparison, the sample (data) parallelism shown in (b) is limited by the mini-batch size, and by the memory that a whole sample needs on one processor; spatial parallelism has neither limit, but adds the halo exchanges.

## Final remarks and code

There is [ongoing work from PyTorch to support general model parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html), where the user can pick row-wise split, column-wise split and sharding of individual tensors. Also, combining data, pipeline and tensor parallelism requires one to define the correct strategy for [custom model parallelism](https://www.deepspeed.ai/training/#model-parallelism).

As an important remark, finding the best parallelism strategy is hard, due to the high number of hyper-parameters (ZeRO stages, offloading, activation checkpointing intervals, pipeline parallelism stages, data parallelism, tensor parallelism, etc.), and because it depends on the ML model, data and hardware. In practice, the DeepSpeed config files and parallelism settings of the distributed training posts are a manually-optimized ballpark figure of the default config file with some parameter grid search. On this topic, there is still plenty of work to be done to make it optimal, possibly by exploring the [autotuning](https://www.deepspeed.ai/tutorials/autotuning/) tools in DeepSpeed.

These posts only scratched the surface of DeepSpeed's capabilities. There are plenty of resources that should also be explored. To name a few: [**autotuning**](https://www.deepspeed.ai/tutorials/autotuning/) ([README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)) for parallelism hyper-parameter discovery; [**flops profiler**](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html) to measure the time, flops and parameters of individual layers; [**sparse attention kernels**](https://www.deepspeed.ai/2020/09/08/sparse-attention.html) ([API](https://www.deepspeed.ai/docs/config-json/#sparse-attention)) to support long sequences of model inputs, such as text, image, or sound; [**communication optimizers**](https://www.deepspeed.ai/tutorials/zero-one-adam/) that offer the same convergence as Adam/LAMB but incur up to 26x less communication on BERT-large pretraining; [**monitor**](https://www.deepspeed.ai/training/#monitor) to log live training metrics to TensorBoard, csv file or other backend; [**model compression**](https://www.deepspeed.ai/compression/) ([API](https://www.deepspeed.ai/docs/config-json/#compression)) via layer reduction, weight quantization, activation quantization, sparse pruning, row pruning, head pruning and channel pruning, to deliver faster speed and smaller model size.






