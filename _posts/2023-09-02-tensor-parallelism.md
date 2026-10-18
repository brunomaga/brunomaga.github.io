---
layout: post
title:  "Tensor parallelism"
categories: ["Distributed Training"]
tags: ["tensor parallelism", "Megatron-LM", "collectives", "PyTorch"]
---

Tensor parallelism splits the tensors of each layer, such as the weights of the linear layers and the activations, across processes. It is one of the parallelism dimensions of distributed training, with data, pipeline, context, sequence and expert parallelism. Each tensor needs a split dimension (rows, columns, or none), and the communication follows from these choices, so the implementation depends on the model.

{: .picture}
<img width="48%" height="48%" src="{{ site.assets }}/distributed-training/imgs/dnn_model_parallelism.svg"/>

{: .caption}
Tensor parallelism on a fully-connected DNN, on two processors $$p0$$ and $$p1$$. The input sample and activations are distributed across processors. Red lines represent the activations that must be communicated to a different processor.

In the figure, every layer needs activations that the other processor holds (red lines). This communication is synchronous: a layer waits for the activations of the previous layer from the other processor. A naive split communicates at every layer, in the forward and the backward pass, for every batch. Methods that split multi-dimensional tensors reduce the number of communication steps or the communication volume.

## Megatron-LM Tensor Parallelism

[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019) splits the two blocks of a transformer layer, the MLP and the self-attention, so that each block needs one [collective communication](https://en.wikipedia.org/wiki/Collective_operation) in the forward pass and one in the backward pass:

{: .picture}
<img width="65%" height="65%" src="{{ site.assets }}/distributed-training/imgs/megatron-lm.svg"/>

{: .caption}
The MLP and self-attention blocks in the Megatron-LM implementation. $$f$$ and $$g$$ are conjugate identity and all-reduce operations: $$f$$ is an identity operator in the forward pass and all-reduce in the backward pass, while $$g$$ is an all-reduce in the forward pass and identity in the backward pass. Adapted from: [Megatron-LM paper](https://arxiv.org/abs/1909.08053)

The MLP computes a MatMul (a GEMM) $$XA$$, a GeLU, a MatMul $$YB$$ and a dropout, where $$Y$$ is the output of the GeLU. The first MatMul and the GeLU can be split in two ways.

**Split by rows.** "One option to parallelize the GEMM is to split the weight matrix $$A$$ along its rows and input $$X$$ along its columns. [...] This partitioning will result in $$Y = GeLU(X_1 A_1 + X_2 A_2)$$. Since GeLU is a nonlinear function, $$GeLU (X_1 A_1 + X_2 A_2) \neq  GeLU( X_1 A_1 ) + GeLU (X_2 A_2)$$ and this approach will require a synchronization point before the GeLU function". The synchronization is an all-reduce that sums the partial products $$X_1 A_1$$ and $$X_2 A_2$$ (the *sum-reduce* in the figure below). Splitting both MatMuls this way needs two all-reduces.

  {: .picture}
  <img width="93%" height="93%" src="{{ site.assets }}/distributed-training/imgs/megatron_model_parallel_1.svg">


**Split by columns.** Another option is to keep $$X$$ whole and to split $$A$$ along its columns, which "allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM". The output of the GeLU is then $$[Y_1, Y_2] = [GeLU (X A_1), GeLU(X A_2)]$$, with no synchronization point.
The second MatMul receives $$Y$$ split by columns, which is the input of a split by rows. Megatron-LM therefore splits $$B$$ by rows, and one all-reduce sums the partial products $$Y_1 B_1$$ and $$Y_2 B_2$$ before the dropout. The two MatMuls need one all-reduce in total.

  {: .picture}
  <img width="71%" height="71%" src="{{ site.assets }}/distributed-training/imgs/megatron_model_parallel_2.svg">


The self-attention block follows the same pattern. The projections of the queries, keys and values, $$Q$$, $$K$$ and $$V$$, are split by columns, so that each GPU holds a subset of the attention heads. Each GPU computes the attention of its heads without communication, as the heads are independent. The output projection $$B$$ is split by rows, and one all-reduce sums its partial products. Each block thus needs one all-reduce in the forward pass, in $$g$$, and one in the backward pass, in $$f$$.

Megatron-LM adds only the functions $$f$$ and $$g$$ to the serial code. $$f$$ is the identity in the forward pass and an all-reduce in the backward pass, and $$g$$ is the opposite:

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

The argument `mp_comm_group` is the communication group of the model-parallel GPUs, which lets tensor parallelism combine with other parallelism dimensions. For example, 8 GPUs can form 2 data-parallel groups of 4 model-parallel GPUs. The MLP calls $$f$$ at the start of the block and $$g$$ at the end, as in the figure:

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

The all-reduce sums the partial products of the second MatMul before its bias, so that the bias is added once and not once per GPU. The dropout after $$g$$ acts on activations that all GPUs of the group hold in full, so these GPUs must draw the same dropout mask, from the same random state. Megatron-LM keeps a second random state, different on each GPU of the group, for the dropout inside the split region, such as the attention dropout of the local heads. The self-attention block splits the heads in the same way:

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

The input $$X$$ of each block, and the activations of the layer norms and dropouts between the blocks, are replicated on all GPUs of the group. Training keeps these activations for the backward pass, so their memory does not shrink with the number of GPUs. The *sequence parallelism* of Megatron-LM ([Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), Korthikanti et al., 2022) splits them along the sequence dimension. The all-reduce of $$g$$ becomes a reduce-scatter, which leaves each GPU with a shard of the output, and $$f$$ becomes an all-gather, which rebuilds the full $$X$$ for the first MatMul. The forward pass keeps only the shard of $$X$$, and the backward pass all-gathers it again. An all-reduce is a reduce-scatter followed by an all-gather, so the communication volume stays the same, with two collectives per block instead of one. The memory of the replicated activations drops by the number of GPUs.

## Spatial Parallelism in Convolutional Neural Networks

Megatron-LM minimizes the number of communication steps, because each step moves a large volume of data. In [convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network), a split of the activations needs little communication. A convolution kernel covers a small spatial window, so a processor needs only a thin border of the activations of its neighbors. [Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism](https://arxiv.org/pdf/1903.06681.pdf) (Dryden et al., 2019) calls this *spatial parallelism*:
1. The input and the activations are split along the height and the width across processors.
2. Each processor computes the convolution of its part. The activations next to the split boundaries, the *halo region* (violet in the figure below), are exchanged with the neighbors at every forward and backward pass.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/argonne_parallel_2.svg"/>

{: .caption}
Tensor parallelism applied to a Convolutional Neural Network. **LEFT:** split of the activations of a 2D input across four processors $$p0-p3$$. <b><span style="color: #b85450;">red box</span></b>: center of the 3x3 convolution filter; <b><span style="color: #b85450;">red arrow</span></b>: data movement required for updating the neuron in the center of the filter; <b><span style="color: #9673a6;">violet region:</span></b> <i>halo region</i> formed by the elements communicated at every step. <b>RIGHT:</b> communication between processors $$p0$$ and $$p1$$. <b><span style="color: #b85450;">Red arrow</span></b>: forward pass dependencies; <b><span style="color: #6c8ebf;">blue arrow</span></b>: backward pass dependencies. Adapted from: [Dryden et al.](https://arxiv.org/pdf/1903.06681.pdf)

The right side of the figure shows sample (data) parallelism for comparison: each processor convolves its own samples, and an all-reduce sums the weight gradients.

## Combining Parallelism Dimensions

PyTorch implements tensor parallelism in [`torch.distributed.tensor.parallel`](https://pytorch.org/docs/stable/distributed.tensor.parallel.html), on top of distributed tensors (DTensor). The function `parallelize_module` splits each linear layer by columns (`ColwiseParallel`) or by rows (`RowwiseParallel`), and `SequenceParallel` shards the layer norms and dropouts along the sequence dimension. DeepSpeed combines its data and pipeline parallelism with the tensor parallelism of a framework such as Megatron-LM, which provides the [model-parallel and data-parallel communication groups](https://www.deepspeed.ai/training/#model-parallelism).

The best combination of parallelism dimensions depends on the model, the data and the hardware, and has many settings: the ZeRO stage, offloading, the activation checkpointing interval, the number of pipeline stages, and the data-parallel and tensor-parallel sizes. The configs of this series come from the DeepSpeed defaults and a small grid search. [DeepSpeed autotuning](https://www.deepspeed.ai/tutorials/autotuning/) ([README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)) searches the ZeRO stage, the micro-batch size and the ZeRO settings automatically.

## Further Reading

- The [DeepSpeed tutorial on Megatron-LM](https://www.deepspeed.ai/tutorials/megatron/) trains a Megatron-LM GPT-2 model with DeepSpeed.
- The [flops profiler](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html) measures the time, FLOPs and parameters of each layer.
- [Sparse attention kernels](https://www.deepspeed.ai/2020/09/08/sparse-attention.html) ([API](https://www.deepspeed.ai/docs/config-json/#sparse-attention)) support long input sequences of text, images or sound.
- Communication-efficient optimizers such as [0/1 Adam](https://www.deepspeed.ai/tutorials/zero-one-adam/) cut the communication volume of BERT-Large pre-training by up to 26 times.
- The [monitor](https://www.deepspeed.ai/training/#monitor) logs live training metrics to TensorBoard, CSV files or other backends.
- [Model compression](https://www.deepspeed.ai/compression/) ([API](https://www.deepspeed.ai/docs/config-json/#compression)) reduces the model size and the runtime with layer reduction, weight and activation quantization, and sparse, row, head and channel pruning.

