---
layout: post
title:  "Tensor parallelism"
categories: ["Distributed Training"]
tags: ["tensor parallelism", "Megatron-LM", "collectives", "PyTorch"]
---

Tensor parallelism, a form of model parallelism, splits the computation of each layer across processes along the dimensions of its tensors, for example by sharding the weights of the linear layers and the matching activations. It is one of several dimensions of parallelism in distributed training, alongside data, pipeline, context, sequence and expert parallelism. Splitting a model this way is a hard problem: each tensor needs a split dimension (rows, columns, or none), and the communication and the computation must be adapted to these choices, so the implementation is specific to the model and the data.

{: .picture}
<img width="54.5%" height="54.5%" src="{{ site.assets }}/distributed-training/imgs/dnn_model_parallelism.svg"/>

{: .caption}
A fully-connected network (left) and the same network split by tensor parallelism over two processors $$P_0$$ and $$P_1$$ (right), each holding half of the neurons of every layer. Red lines are the connections between neurons on different processors, whose activations must be communicated.

The figure shows the main drawback of this method. Every layer needs activations that the other processor holds (the red lines), so during training the processors must continuously exchange the activations that the others need. This communication is synchronous, since a layer waits for the activations of the previous layer from the other processor, and depending on the implementation it leaves little room to overlap communication and computation. A naive split may also communicate a large volume, at every layer, in the forward and the backward pass, and for every batch. Several methods split multi-dimensional tensors so as to reduce the number of communication steps or the communication volume.

## Megatron-LM Tensor Parallelism

[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019) starts from the structure of the transformer. A transformer layer has two main blocks, a feed-forward network (the MLP) and a self-attention block, and Megatron-LM splits each of them so that its forward pass needs a single [collective communication](https://en.wikipedia.org/wiki/Collective_operation) step, and its backward pass another one:

{: .picture}
<img width="74.5%" height="74.5%" src="{{ site.assets }}/distributed-training/imgs/megatron-lm.svg"/>

{: .caption}
(a) The MLP block and (b) the self-attention block of Megatron-LM, split over two GPUs: the first weight matrix $$A$$ is split by columns ($$Q$$, $$K$$ and $$V$$ by attention heads), and the second one, $$B$$, by rows. $$f$$ and $$g$$ are conjugate operators: $$f$$ is an identity operator in the forward pass and an all-reduce in the backward pass, while $$g$$ is an all-reduce in the forward pass and an identity in the backward pass. Adapted from: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

To see how, consider the MLP first. It is a sequence of four operations: (1) a MatMul (a GEMM) $$XA$$, (2) a GeLU, whose output is $$Y$$, (3) a MatMul $$YB$$ and (4) a dropout. The first MatMul and the GeLU can be split in two ways.

**Option 1: split by rows.** "One option to parallelize the GEMM is to split the weight matrix $$A$$ along its rows and input $$X$$ along its columns. [...] This partitioning will result in $$Y = GeLU(X_1 A_1 + X_2 A_2)$$. Since GeLU is a nonlinear function, $$GeLU (X_1 A_1 + X_2 A_2) \neq  GeLU( X_1 A_1 ) + GeLU (X_2 A_2)$$ and this approach will require a synchronization point before the GeLU function". This synchronization is an all-reduce that sums the partial products $$X_1 A_1$$ and $$X_2 A_2$$, the *sum-reduce* in the figure below. Each partial product has the full size of the output, so the all-reduce communicates the whole output, and splitting both MatMuls this way needs two all-reduces.

  {: .picture}
  <img width="84.4%" height="84.4%" src="{{ site.assets }}/distributed-training/imgs/megatron_model_parallel_1.svg">

  {: .caption}
  Option 1 on two GPUs: $$X$$ is split by columns and $$A$$ by rows. Each GPU computes a partial product of size $$M \times O$$, and a sum-reduce adds them into $$XA$$ on both GPUs before the GeLU, which then gives the same $$Y$$ on both GPUs. Inspired by: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

**Option 2: split by columns.** Another option is to keep $$X$$ whole and to split $$A$$ along its columns, which "allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM". The output of the GeLU is then $$[Y_1, Y_2] = [GeLU (X A_1), GeLU(X A_2)]$$, and the synchronization point disappears. This choice also fits the second MatMul: its input $$Y$$ arrives split by columns, which is exactly the input of a split by rows. Megatron-LM therefore splits $$B$$ by rows, and one all-reduce sums the partial products $$Y_1 B_1$$ and $$Y_2 B_2$$ before the dropout. The two MatMuls thus need one all-reduce in total, instead of two.

  {: .picture}
  <img width="60.8%" height="60.8%" src="{{ site.assets }}/distributed-training/imgs/megatron_model_parallel_2.svg">

  {: .caption}
  Option 2 on two GPUs: $$X$$ is replicated and $$A$$ is split by columns, so each GPU computes its own columns of $$XA$$ and applies the GeLU to them without any communication. Inspired by: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

The self-attention block follows the same pattern. The projections of the queries, keys and values, $$Q$$, $$K$$ and $$V$$, are split by columns, so that each GPU holds a subset of the attention heads, and each GPU computes the attention of its heads without communication, as the heads are independent. The output projection $$B$$ is split by rows, and one all-reduce sums its partial products. In the backward pass, the roles swap. Each GPU computes only the part of the gradient of $$X$$ that comes from its own columns of $$A$$ (or of $$Q$$, $$K$$ and $$V$$), so these parts must be summed with an all-reduce, while every GPU already holds the full gradient of the output of the block. Each block thus needs one all-reduce in the forward pass, in $$g$$, and one in the backward pass, in $$f$$.

Megatron-LM implements all of this by adding only the functions $$f$$ and $$g$$ to the serial code. $$f$$ is the identity in the forward pass and an all-reduce in the backward pass, and $$g$$ is the opposite:

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

The extra argument `mp_comm_group` is the communication group of the model-parallel GPUs, which allows tensor parallelism to be combined with other types of parallelism. For example, 8 GPUs can form 2 model-parallel groups of 4 GPUs, each group holding one copy of the model, while data parallelism runs across the 4 data-parallel groups of 2 GPUs that hold the same shard. Tensor parallelism is then added to the MLP by calling $$f$$ at the start of the block and $$g$$ at its end, as in the figure:

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

Each GPU holds `n_embd_mid` columns of the first weight matrix, the width of the hidden layer divided by the number of GPUs in the group, and the same number of rows of the second one. For correctness, the all-reduce sums the partial products of the second MatMul before its bias, so that the bias is added once and not once per GPU. The dropout after $$g$$ needs care too. It acts on activations that all GPUs of the group hold in full, so these GPUs must draw the same dropout mask, from the same random state. Megatron-LM keeps a second random state, different on each GPU of the group, for the dropout inside the split region, such as the attention dropout of the local heads. The self-attention block splits the heads in the same way: each GPU computes the queries, keys and values of its `n_head_local` heads, runs their attention locally, and $$g$$ sums the partial outputs of the row-parallel output projection:

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

### Sharding the Activations

The main caveat of this method is that the input $$X$$ of each block is replicated on all GPUs of the group, and so are the activations of the layer norms and dropouts between the blocks. Large inputs have a large memory footprint, and this matters most during training, where the activations of the forward pass are kept for the backward pass: their memory does not shrink with the number of GPUs. The *sequence parallelism* of Megatron-LM ([Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), Korthikanti et al., 2022) splits these activations along the sequence dimension. The all-reduce of $$g$$ becomes a reduce-scatter, which leaves each GPU with a shard of the output, and $$f$$ becomes an all-gather, which rebuilds the full $$X$$ for the first MatMul. Between the reduce-scatter of one block and the all-gather of the next, each GPU keeps only its shard, so the memory of these activations drops linearly with the number of GPUs. Since an all-reduce is a reduce-scatter followed by an all-gather, the change adds no communication volume, although each block now runs two collectives instead of one. The backward pass needs the full $$X$$ again, for the gradient of the weights, and rebuilds it with an extra all-gather, which overlaps with the computation of the input gradient.

## Spatial Parallelism in Convolutional Neural Networks

Megatron-LM reduces the number of communication steps because each step moves a large volume of data. In some models, however, a split of the activations needs little communication. One example is [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNNs): a convolution kernel covers a small spatial window, so a processor needs only a thin border of the activations of its neighbors. [Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism](https://arxiv.org/pdf/1903.06681.pdf) (Dryden et al., 2019) calls this *spatial parallelism*, and it works as follows:
1. The input and the activations are split along the height and the width across processors.
2. Each processor computes the convolution of its part independently, except near the split boundaries. The activations there, the *halo region* (purple in the figure below), are exchanged with the neighbors at every forward and backward pass.

{: .picture}
<img width="98.8%" height="98.8%" src="{{ site.assets }}/distributed-training/imgs/argonne_parallel_2.svg"/>

{: .caption}
Two ways to parallelize a convolutional neural network. (a) Spatial parallelism: the activations of a 2D input are split across four processors $$P_0$$ to $$P_3$$. The red box is where the 3x3 convolution filter is centered, the red arrows are the data that must be received from the other processors to compute it, and the purple region is the halo, formed by all the elements exchanged at every step. (b) Sample (data) parallelism on two processors $$P_0$$ and $$P_1$$, with a global mini-batch of 4 samples. Blue arrows are the dependencies of the forward pass, which computes the output $$y$$ from the input $$x$$ and the filters $$w$$; green arrows are those of the backward pass, which computes the gradients of the filters, $$dL/dw$$, and of the input, $$dL/dx$$, from $$dL/dy$$. The only communication between $$P_0$$ and $$P_1$$ is the all-reduce of $$dL/dw$$. Adapted from: [Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism](https://arxiv.org/abs/1903.06681)

The halo grows with the perimeter of each part, while the computation grows with its area, so for large inputs the communication stays small compared to the computation. The right side of the figure shows sample (data) parallelism for comparison: each processor convolves its own samples, and an all-reduce sums the weight gradients. Spatial parallelism suits datasets whose samples are large and need much memory, where data parallelism alone stops scaling, because the mini-batch cannot grow without limit.

## Combining Parallelism Dimensions

PyTorch implements tensor parallelism in [`torch.distributed.tensor.parallel`](https://pytorch.org/docs/stable/distributed.tensor.parallel.html), on top of distributed tensors (DTensor). The function `parallelize_module` splits each linear layer by columns (`ColwiseParallel`) or by rows (`RowwiseParallel`), as Megatron-LM does, and `SequenceParallel` shards the layer norms and dropouts along the sequence dimension. DeepSpeed, in turn, combines its data and pipeline parallelism with the tensor parallelism of a framework such as Megatron-LM, which provides the [model-parallel and data-parallel communication groups](https://www.deepspeed.ai/training/#model-parallelism).

Finding the best combination of parallelism dimensions is hard, because it depends on the model, the data and the hardware, and because it has many settings: the ZeRO stage, offloading, the activation checkpointing interval, the number of pipeline stages, and the data-parallel and tensor-parallel sizes. The configs of this series are a manually tuned approximation, obtained from the DeepSpeed defaults with a small grid search. [DeepSpeed autotuning](https://www.deepspeed.ai/tutorials/autotuning/) ([README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)) searches the ZeRO stage, the micro-batch size and the other ZeRO settings automatically.

## Further Reading

DeepSpeed provides many more tools than the ones used in this series:
- The [DeepSpeed tutorial on Megatron-LM](https://www.deepspeed.ai/tutorials/megatron/) trains a Megatron-LM GPT-2 model with DeepSpeed.
- The [flops profiler](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html) measures the time, FLOPs and parameters of each layer.
- [Sparse attention kernels](https://www.deepspeed.ai/2020/09/08/sparse-attention.html) ([API](https://www.deepspeed.ai/docs/config-json/#sparse-attention)) support long input sequences of text, images or sound.
- Communication-efficient optimizers such as [0/1 Adam](https://www.deepspeed.ai/tutorials/zero-one-adam/) cut the communication volume of BERT-Large pre-training by up to 26 times.
- The [monitor](https://www.deepspeed.ai/training/#monitor) logs live training metrics to TensorBoard, CSV files or other backends.
- [Model compression](https://www.deepspeed.ai/compression/) ([API](https://www.deepspeed.ai/docs/config-json/#compression)) reduces the model size and the runtime with layer reduction, weight and activation quantization, and sparse, row, head and channel pruning.

