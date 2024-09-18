---
layout: post
title:  "Distributed GPT model (part 3): Megatron-LM model parallelism"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

This post follows from the previous posts [Distributed training of a GPT model using DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed-sharding %}) and [Distributed training of a GPT model using DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-30-GPT-lite-DeepSpeed-pipeline%}), where we implemented Data and Pipeline parallelism on a GPT model, 2 dimensions of parallelism on the **3D parallelism** of ML models, via Data, Pipeline and Tensors/Models parallelism. In this post, we will discuss model (tensor) parallelism, particularly the [Megatron-LM](https://www.deepspeed.ai/tutorials/megatron/) implementation.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPT-lite-distributed/GPT_3D_parallelism_2.png"/>

{: style="text-align:center; font-size: small;"}
The 3D parallelism aims and partitioning (color-coded) computer resources  across the 3D space of data, pipeline and tensor (model) dimensions. In this post of will focus on model/tensor parallelism. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

## Megatron-LM model/tensor parallelism

**Tensor parallelism**, **vertical parallelism**, **intra-layer parallelism**, **activation parallelism** or most commonly ad confusedly called **model parallelism**, is the third dimension of parallelism and aims at partitioning the computation of tensors (activations) in the forward and backward passes. This requires a modification of the workflow of the computation in order to work in a distributed manner, particularly on the matrix multiplications format: in practice we must decide for the dimension of tensor partitioning (row, wise, none) and adapt the communication and computation accordingly, leading to an all-gather, all-reduce, scatter-reduced distributed matrix multiplicatioon, etc. Therefore, it is a model-specific implementation, and is [supported but not provided](https://www.deepspeed.ai/training/#support-for-custom-model-parallelism) by DeepSpeed, except in some built-in implementations such as [Megatron-LM](https://www.deepspeed.ai/tutorials/megatron/), for BERT, T5, GPT2 and others.

We can have a better understanding if we try to replicate the partitioning suggested by Megatron-LM in the paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). The main rationale underlying the paper is that Transformer-based models have two main components - a feed-forward (MLP) and an attention head - and we can do a forward and a backward pass on each of these blocks with a single collective communication step.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPT-lite-distributed/Megatron-LM.png"/>

{: style="text-align:center; font-size: small;"}
The MLP and self-attention blocks in the Megatron-LM implementation. f and g
are conjugate identity and all-reduce operations that c. f is an identity operator in the forward pass and all
reduce in the backward pass while g is an all reduce in the forward
pass and identity in the backward pass. Source: [Megatron-LM paper](https://arxiv.org/abs/1909.08053)

The rationale is the following: the MLP is sequence of (1) a MatMul $$AX$$, (2) a ReLU $$Y$$; a (3) MatMul $$YB$$ and a (4) dropout layer. Let's focus on the first two operations: 
1. "One option to parallelize the GEMM is to split the weight matrix $$A$$ along its rows and input $$X$$ along its columns. [...] This partitioning will result in $$Y = GeLU(X_1 A_1 + X_2 A_2)$$. Since GeLU is a nonlinear function, $$GeLU (X_1 A_1 + X_2 A_ 2) \neq  GeLU( X_1 A_1 ) + GeLU (X_2 A_2)$$ and this approach will require a synchronization point (*sum-reduce*) before the GeLU function". We can follow this approach - pictured below - for each of the 2 MatMuls, yielding a total of two communication steps.

  {: style="text-align:center; font-size: small;"}
  <img width="100%" height="100%" src="/assets/GPT-lite-distributed/megatron_model_parallel_1.png">


2. Another option is to don't split $$X$$ and split $$A$$ along its columns, that "allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM". The output of the ReLU $$Y$$ is then $$[Y_1, Y_2] = [GeLU (X A_1), GeLU(X A_2)]$$ and this removes a synchrnization point.
After the MatMul and ReLU is done, we must follow with another MatMul and a dropout. Note that if we follow this approach (pictured below), the output $$Y$$ of the first MatMul (which is the input for the second MatMul) is split column-wise. Therefore, Megatron will then use the previous approach for the second MatMul, that requires 1 collective communication step. We therefore perform 1 communication step for 2 MatMuls.

  {: style="text-align:center; font-size: small;"}
  <img width="80%" height="80%" src="/assets/GPT-lite-distributed/megatron_model_parallel_2.png">


The algorithm of the attention mechanism is analogous, except that instead of a single MatMul $$AX$$ at the beginning of the workflow, we perform three MatMuls $$XV$$, $$XQ$$ and $$XK$$ for the value, query and key matrices of the attention mechanism. Therefore, this algorithm requires one (not two) communication steps per MLP and attention block.

Megatron-LM makes this implementation very simple, by adding only the $$f$$ and $$g$$ functions to the serial use case. Following the paper: "$$f$$ is an identity operator in the forward pass and all reduce in the backward pass while $$g$$ is an all reduce in the forward pass and identity in the backward pass". So we can implement this as:

```python
class Megatron_f(torch.autograd.Function):
  """ The f function in Figure 3 in Megratron paper """

  @staticmethod
  def forward(ctx, x, mp_comm_group=None):
      ctx.mp_comm_group = mp_comm_group #save for backward pass
      return x

  @staticmethod
  def backward(ctx, gradient):
      dist.all_reduce(gradient, dist.ReduceOp.SUM, group=ctx.mp_comm_group)
      return gradient
```

and

```python
class Megatron_g(torch.autograd.Function):
  """ The g function in Figure 3 in Megratron paper """

  @staticmethod
  def forward(ctx, x, mp_comm_group=None):
      dist.all_reduce(x, dist.ReduceOp.SUM, group=mp_comm_group)
      return x

  @staticmethod
  def backward(ctx, gradient):
      return gradient
```

Note that we added an extra argument `mp_comm_group` that refers to the model-parallel communication group. This refers to the communication group is to allow us to combine MP with other types of parallelism. As an example, if you have 8 GPUs, you can have 2 data parallel groups of 4 model parallel GPUs. We now add model parallelism to the MLP by inserting $$f$$ and $$g$$ in the forward pass, at the beginning and end of the block, just like in the paper:

```python
class Megatron_FeedForward(nn.Module):
  """ the feed forward network (FFN), with tensor parallelism as in Megatron-LM MLP block"""

  def __init__(self, n_embd, mp_comm_group=None):
    super().__init__()
    self.mp_comm_group = mp_comm_group

    #Fig 3a. MLP: splits first GEMM across colums and second GEMM across rows
    n_embd_mid = n_embd*4 #width of MLP middle layer, as before
    if self.mp_comm_group:
        n_embd_mid //= dist.get_world_size()

    self.fc1 = nn.Linear(n_embd, n_embd_mid)
    self.fc2 = nn.Linear(n_embd_mid, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    if self.mp_comm_group:
        x = Megatron_f.apply(x, self.mp_comm_group) #Fig 3a. apply f on input
        
    y = F.relu(self.fc1(x))
    z = self.fc2(y)

    if self.mp_comm_group:
        z = Megatron_g.apply(z, self.mp_comm_group) #Fig 3a. apply g before dropout
            
    z = self.dropout(z)
    return z
```

The attention head follows a similar approach, where we apply the tensor reduction to all the key, query and value tensors:

```python
class Megatron_Head(nn.Module):
  """ the attention block with tensor parallelism as in Megatron-LM paper"""

  def __init__(self, head_size, mp_comm_group=None):
    super().__init__()
  
    self.mp_comm_group = mp_comm_group
    if mp_comm_group:
        #Fig 3b. Self-attention: splits first GEMM across colums and second GEMM across rows
        head_size //= dist.get_world_size()
        
    self.key   = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape

    if self.mp_comm_group:
      x = Megatron_f.apply(x, self.mp_comm_group) #Fig 3b. apply f on input

    k = self.key(x) #shape (B,T, head_size)
    q = self.query(x) #shape (B,T, head_size)
    v = self.value(x) #shape (B,T, head_size)

    # compute self-attention scores
    # [...] as before

    if self.mp_comm_group:
      wei = Megatron_g.apply(wei, self.mp_comm_group) #Fig 3b. apply g after dropout

    #perform weighted aggregation of values
    out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
    return out
``` 

### Final remarks and code

There is [ongoing work from PyTorch to support general model parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html), where the user can pick row-wise split, column-wise split and sharding of individual tensor. Also, combining data, pipeline and model parallelism requires one to define the corect strategy for [custom model parallelism](https://www.deepspeed.ai/training/#model-parallelism).

As an important remark: finding the best parallelism strategy is hard, due to the high number of hyper-paramemers: ZeRO stages, offloading, activation checkpointing intervals, pipeline parallelism stages, data parallelism, model parallelism, etc, as it depends on the ML model, data and hardware. In practice, our config file and parallelism settings are a manually-optimized ballpark figure of the default config file with some parameter grid search. In this topic, there is still plenty of work to be done to make it optimal, possibly by exploring the [autotuning](https://www.deepspeed.ai/tutorials/autotuning/) tools in DeepSpeed.

We just scratched the surface of DeepSpeed capabilities. There are plenty of resources that should also be explored. To name a few: [**autotuning**](https://www.deepspeed.ai/tutorials/autotuning/) ([README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)) for parallelism hyper-parameters discovery; [**flops profiler**](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html) measures the time, flops and parameters of individual layers, [**sparse attention kernels**](https://www.deepspeed.ai/2020/09/08/sparse-attention.html) ([API](https://www.deepspeed.ai/docs/config-json/#sparse-attention)) to support long sequences of model inputs, such as text, image, or sound; [**communication optimizers**](https://www.deepspeed.ai/training/#1-bit-adam-01-adam-and-1-bit-lamb-optimizers-with-up-to-26x-less-communication) offer the same convergence as Adam/LAMB but incur 26x less communication and 6.6x higher throughput on large BERT pretraining, [**monitor**](https://www.deepspeed.ai/training/#monitor) to log live training metrics to TensorBoard, csv file or other backend; [**model compression**](https://www.deepspeed.ai/compression/) ([API](https://www.deepspeed.ai/docs/config-json/#compression)) via layer reduction, weight quantization, activation quantization, sparse pruning, row pruning, head pruning and channel pruning, to deliver faster speed and smaller model size.

Finally, this code has been added to the [GPT-lite-distributed repo](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-distributed), if you want to try it.
