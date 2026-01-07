---
layout: post
title:  "Distributed GPT model (part 3): model parallelism with Megatron-LM"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

This post follows from the previous posts [Distributed training of a GPT model using DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPTlite-data-parallelism %}) and [Distributed training of a GPT model using DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-30-GPTlite-DeepSpeed-pipeline%}), where we implemented Data and Pipeline parallelism on a GPT model. Data and pipeline parallelism are 2 dimensions of the **3D parallelism** of ML models, via Data, Pipeline and Tensors/Models parallelism. In this post, we will discuss model (tensor) parallelism, particularly the [Megatron-LM](https://www.deepspeed.ai/tutorials/megatron/) implementation.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPTlite-distributed/GPT_3D_parallelism_2.png"/>

{: style="text-align:center; font-size: small;"}
The 3D parallelism aims and partitioning (color-coded) computer resources  across the 3D space of data, pipeline and tensor (model) dimensions. In this post of will focus on model/tensor parallelism. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)


**Tensor parallelism**, **vertical parallelism**, **intra-layer parallelism**, **activation parallelism** or most commonly ad confusedly called **model parallelism**, is the third dimension of parallelism and aims at partitioning the computation on the activations dimension. This is a hard problem: in practice we must decide for the dimension of tensor partitioning (row, wise, none) and adapt the communication and computation accordingly. Therefore, it is a model- and data-specific implementation.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/AI-Supercomputing/DNN_model_parallelism.png"/>

{: style="text-align:center; font-size: small;"}
A representation of tensor parallelism on a fully-connected DNN, on two processors $$p0$$ and $$p1$$. Input sample and activations are distributed across different processors. Red lines represents the activations that have to be communicated to a different processor.

Looking at the previous picture, we notice a major drawback in this method. During training, processors need to continuously communicate activations that are needed across the processor network. This communication is synchronous and does not allow an overlap with compute. And it creates a major drawback on the execution as it requires a tremendous ammount of communication at every layer of the network and for every input batch. 

There are several alternative ways to distribute multi-dimensional tensors across several compute nodes, that aim at reducing number of comunication steps or ammount of communcation volue. In this post, we will detail and implement Megatron-LM paper.


## Megatron-LM model parallelism

We can have a better understanding if we try to replicate the partitioning suggested by Megatron-LM in the paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). The main rationale in Megatron-LM is that Transformer-based models have two main components - a feed-forward (MLP) and an attention head - and we can do a forward and a backward pass on each of these blocks with a single collective communication step.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPTlite-distributed/Megatron-LM.png"/>

{: style="text-align:center; font-size: small;"}
The MLP and self-attention blocks in the Megatron-LM implementation. f and g
are conjugate identity and all-reduce operations that c. f is an identity operator in the forward pass and all
reduce in the backward pass while g is an all reduce in the forward
pass and identity in the backward pass. Source: [Megatron-LM paper](https://arxiv.org/abs/1909.08053)

The rationale is the following: the MLP is sequence of (1) a MatMul $$AX$$, (2) a ReLU $$Y$$; a (3) MatMul $$YB$$ and a (4) dropout layer. Let's focus on the first two operations: 
1. "One option to parallelize the GEMM is to split the weight matrix $$A$$ along its rows and input $$X$$ along its columns. [...] This partitioning will result in $$Y = GeLU(X_1 A_1 + X_2 A_2)$$. Since GeLU is a nonlinear function, $$GeLU (X_1 A_1 + X_2 A_ 2) \neq  GeLU( X_1 A_1 ) + GeLU (X_2 A_2)$$ and this approach will require a synchronization point (*sum-reduce*) before the GeLU function". We can follow this approach - pictured below - for each of the 2 MatMuls, yielding a total of two communication steps.

  {: style="text-align:center; font-size: small;"}
  <img width="100%" height="100%" src="/assets/GPTlite-distributed/megatron_model_parallel_1.png">


2. Another option is to don't split $$X$$ and split $$A$$ along its columns, that "allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM". The output of the ReLU $$Y$$ is then $$[Y_1, Y_2] = [GeLU (X A_1), GeLU(X A_2)]$$ and this removes a synchrnization point.
After the MatMul and ReLU is done, we must follow with another MatMul and a dropout. Note that if we follow this approach (pictured below), the output $$Y$$ of the first MatMul (which is the input for the second MatMul) is split column-wise. Therefore, Megatron will then use the previous approach for the second MatMul, that requires 1 collective communication step. We therefore perform 1 communication step for 2 MatMuls.

  {: style="text-align:center; font-size: small;"}
  <img width="80%" height="80%" src="/assets/GPTlite-distributed/megatron_model_parallel_2.png">


The algorithm of the attention mechanism is analogous, except that instead of a single MatMul $$AX$$ at the beginning of the workflow, we perform three MatMuls $$XV$$, $$XQ$$ and $$XK$$ for the value, query and key matrices of the attention mechanism. Therefore, this algorithm requires one (not two) communication steps per MLP and attention block.

Megatron-LM makes this implementation very simple, by adding only the $$f$$ and $$g$$ functions to the serial use case. Following the paper: "$$f$$ is an identity operator in the forward pass and all reduce in the backward pass while $$g$$ is an all reduce in the forward pass and identity in the backward pass". So we can implement this as:

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

Note that we added an extra argument `mp_comm_group` that refers to the model-parallel communication group. This refers to the communication group is to allow us to combine MP with other types of parallelism. As an example, if you have 8 GPUs, you can have 2 data parallel groups of 4 model parallel GPUs. We now add model parallelism to the MLP by inserting $$f$$ and $$g$$ in the forward pass, at the beginning and end of the block, just like in the paper:

```python

class Megatron_FeedForward(nn.Module):
  """ the feed forward network (FFN) in the paper, with tensor parallelism as in Megatron-LM MLP block"""

  def __init__(self, n_embd, mp_comm_group=None):
    super().__init__()
    self.mp_comm_group = mp_comm_group

    #Fig 3a. MLP: splits first GEMM across colums and second GEMM across rows
    n_embd_mid = n_embd * 4
    if self.mp_comm_group:
        n_embd_mid //= dist.get_world_size()

    self.fc1 = nn.Linear(n_embd, n_embd_mid)
    self.fc2 = nn.Linear(n_embd_mid, n_embd, bias=False)   # <-- no bias here
    self.fc2_bias = nn.Parameter(torch.zeros(n_embd))      # <-- bias added after all-reduce
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    if self.mp_comm_group:
        x = Megatron_f.apply(x, self.mp_comm_group)

    y = F.relu(self.fc1(x))

    z = self.fc2(y)  # matmul only (partial)

    if self.mp_comm_group:
        z = Megatron_g.apply(z, self.mp_comm_group)

    z = z + self.fc2_bias  # <-- bias AFTER all-reduce
    z = self.dropout(z)
    return z
```

Note that the sum-reduce is added after the matmul but before the bias of the second linear layer, for correctness. The attention head follows a similar approach, where we apply the tensor reduction to all the key, query and value tensors:

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
    B, T, C = x.shape

    if self.mp_comm_group:
      x = Megatron_f.apply(x, self.mp_comm_group)

    k = self.key(x)
    q = self.query(x)
    v = self.value(x)

    # scores
    wei = q @ k.transpose(-2, -1)  # (B,T,T)

    if self.mp_comm_group:
      wei = Megatron_g.apply(wei, self.mp_comm_group)  # <-- reduce BEFORE softmax

    wei *= (q.size(-1) ** -0.5)  # <-- scale by head dim
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    #perform weighted aggregation of values
    out = wei @ v
    return out
``` 


## Detour: model parallelism on Convolutional Neural Nets

The whole point of Megatron-LM was to reduce the ammount of high volume communication steps due to the high volume of data involved per step. However, it is relevant to mention that model parallelism has some use cases where it can be efficient and of low volume communication. An example is on the parallelism of [Convolutional Neural Networks]({{ site.baseurl }}{% post_url 2018-03-27-Deep-Neural-Networks %}). In practice, due to the kernel operator in CNNs (that has a short spatial span), the amount of activations to be communicated is limited to the ones ones that neighboring activations need. This method has been detailed by [Dryden et al. (Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism, Proc. IPDPS 2019)](https://arxiv.org/pdf/1903.06681.pdf). The functioning is illustrated in the picture below and is as follows:
1. Input data and activations are split across the height and width dimensions among processors;
2. For a given convolutional layer, the convolution can be computed in independently be each processor, with the exception of the activation in the split boundaries. These activations (the *halo region* in purple in the picture blow) will need to be communicated at every forward/step.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/AI-Supercomputing/argonne_parallel_2.PNG"/>

{: style="text-align:center; font-size: small;"}
Illustration of model parallelism applied to Convolutional Neural network. **LEFT:** splitting of activations on a 2D input across four processors $$p0-p3$$. <b><span style="color: red;">red box</span></b>: center of the 3x3 convolution filter; <b><span style="color: red;">red arrow</span></b>: data movement required for updating neuron in center of filter; <b><span style="color: violet;">violet region:</span></b> <i>halo region</i> formed of the elements that need to be communicated at every step. <b>RIGHT:</b> communication between processors $$p0$$ and $$p1$$. <b><span style="color: red;">Red arrow</span></b>: forward pass dependencies; <b><span style="color: blue;">blue arrow</span></b>: backward pass dependencies;

## Final remarks and code

There is [ongoing work from PyTorch to support general model parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html), where the user can pick row-wise split, column-wise split and sharding of individual tensor. Also, combining data, pipeline and model parallelism requires one to define the corect strategy for [custom model parallelism](https://www.deepspeed.ai/training/#model-parallelism).

As an important remark: finding the best parallelism strategy is hard, due to the high number of hyper-paramemers: ZeRO stages, offloading, activation checkpointing intervals, pipeline parallelism stages, data parallelism, model parallelism, etc, as it depends on the ML model, data and hardware. In practice, our config file and parallelism settings are a manually-optimized ballpark figure of the default config file with some parameter grid search. In this topic, there is still plenty of work to be done to make it optimal, possibly by exploring the [autotuning](https://www.deepspeed.ai/tutorials/autotuning/) tools in DeepSpeed.

We just scratched the surface of DeepSpeed capabilities. There are plenty of resources that should also be explored. To name a few: [**autotuning**](https://www.deepspeed.ai/tutorials/autotuning/) ([README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)) for parallelism hyper-parameters discovery; [**flops profiler**](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html) measures the time, flops and parameters of individual layers, [**sparse attention kernels**](https://www.deepspeed.ai/2020/09/08/sparse-attention.html) ([API](https://www.deepspeed.ai/docs/config-json/#sparse-attention)) to support long sequences of model inputs, such as text, image, or sound; [**communication optimizers**](https://www.deepspeed.ai/training/#1-bit-adam-01-adam-and-1-bit-lamb-optimizers-with-up-to-26x-less-communication) offer the same convergence as Adam/LAMB but incur 26x less communication and 6.6x higher throughput on large BERT pretraining, [**monitor**](https://www.deepspeed.ai/training/#monitor) to log live training metrics to TensorBoard, csv file or other backend; [**model compression**](https://www.deepspeed.ai/compression/) ([API](https://www.deepspeed.ai/docs/config-json/#compression)) via layer reduction, weight quantization, activation quantization, sparse pruning, row pruning, head pruning and channel pruning, to deliver faster speed and smaller model size.

Finally, the Megatron-LM model parallelism code has been added to the [GPTlite-distributed repo](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPTlite-distributed), if you want to try it.

