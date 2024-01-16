---
layout: post
title:  "Distributed training of a GPT model (part 2): pipeline parallelism, Megatron-LM tensor parallelism and communication quantization"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

This post follows from the previous post [Distributed training of a GPT model using DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}). We discussed that an ML model allows for three dimensions of parallelism, on Data, Pipeline and Tensors/Models. We covered distributed data parallellism and sharded data parallelism in the previous post. Here we will discuss pipeline and model (tensor) parallelism.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPT-lite-DeepSpeed/GPT_3D_parallelism_2.png"/>

{: style="text-align:center; font-size: small;"}
The 3D parallelism aims and partitioning (color-coded) computer resources  across the 3D space of data, pipeline and tensor (model) dimensions. In this post of will focus on pipeline and tensor/model parallelism. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

 
## Pipeline parallelism on DeepSpeed

[Pipeline parallelism](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules) improves both the memory and compute efficiency during training by partitioning the layers of a model into stages that can be processed in parallel.
It adds an additional overhead for the inter-layer communication in the GPU boundaries. As we will see later, for an optimal design, it requires the user to manually define the content of each pieline block and stage.  Pipeline parallelism is based on the principal that the model *needs* to be split across GPUs. In practice, it's meant for use cases where the model is too big or the batch size is too big. Also, it's *orthogonal* and can be combined with other parallelism efforts such as data and model parallel.

The pipeline parallelism algorithm implemented in DeepSpeed is the [PipeDream-Flush implementation with default 1F1B scheduling](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/) (1 Forward pass followed by 1 Backward pass, Figure 4 top on the [Megatron LM paper](https://browse.arxiv.org/pdf/2104.04473.pdf) ), however it is possible to [extend pipeline parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule) to other algorithms.

{: style="text-align:center; font-size: small;"}
<img width="79%" height="80%" src="/assets/GPT-lite-DeepSpeed/GPT_pipelining.png"/>

{: style="text-align:center; font-size: small;"}
Two-way data parallel pipelines with four stages each. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

We will build pipeline parallelism taking the code from the previous post as base model, and enable it by passing the number of stages as the `---pipeline_num_stages` argument (default: 0, no pipelining) on the command line:

```python
## train.py

def get_cmd_line_args(description='GPT lite on DeepSpeed'):
  # ...
  parser.add_argument('--pipeline-parallel-size', type=int, default=0,
                      help='enable pipeline parallelism with N stages (0 means disabled)')
  # ...
```

The number of pipeline stages must divide the number of GPUs, so that DeepSpeed automatically creates several parallel pipelines with the same stage count, and distributes them across GPUs.

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="/assets/GPT-lite-DeepSpeed/pipeline_8_stages.png"/>
&nbsp;
<img width="30%" height="30%" src="/assets/GPT-lite-DeepSpeed/pipeline_4_stages.png"/>
&nbsp;
<img width="30%" height="30%" src="/assets/GPT-lite-DeepSpeed/pipeline_2_stages.png"/>

{: style="text-align:center; font-size: small;"}
An illustration of pipeline parallelism on a network of 8 workers for different number of pipeline stages. Left: 8 pipeline-parallel workers. Center: 2 data-parallel groups of 4 pipeline-parallel workers. Right: 4 data-parallel groups of 2 pipeline-parallel workers.

DeepSpeed supports pipeline parallelism on any sequence of network blocks in a `nn.Sequential` container or `list`. The can be then broken into pipeline states. So we expose the pipeline parallelism in our model by creating a method `to_layers()` in `GPTlite`, that returns the sequence of actions to be executed. Note that `to_layers()` follows the same order as the `forward` pass of `GPTlite`, and that `self.blocks` is of type `nn.Sequential`:

```python
## gptlite.py

class GPTlite(nn.Module):
  # ...
  def to_layers(self):  
      layers = [
          lambda idx:
            self.token_embedding_table(idx) +
            self.position_embedding_table(torch.arange(idx.shape[1]).to(idx.device)),
          *self.blocks,
          self.ln,
          self.lm_head,
      ]
      return layers
```

Note that the output of `layers` is of shape `B,T,C` which is incompatible with [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) module in PyTorch. A quick fix is to simple add `lambda logits: torch.swapaxes(logits,1,2)` to `to_layers()` to make it of shape `B,C,T`. However when you try to back-propragate, you will bump into the error `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`, as discussed in bugs [4279](https://github.com/microsoft/DeepSpeed/issues/4274) and [4479](https://github.com/microsoft/DeepSpeed/issues/4479), and you'd have to use `outputs = outputs.requires_grad_(True)` to fix it. Alternatively, you can adapt the loss function to do the `swapaxes` or the `view` change instead. It is a cleaner approach, and will be useful later for the pipeline parallelism use case:

```python
## train.py

class CrossEntropyLoss_FlatView(torch.nn.Module):
  def forward(self, logits, labels):
    B, T, C = logits.shape
    logits = logits.view(B*T,C)
    labels = labels.view(-1)
    return torch.nn.functional.cross_entropy(logits, labels)
  
def main_deepspeed(n_epochs=100, random_seed=42):
  # ...
  criterion = CrossEntropyLoss_TransposedLogits() #initialize loss function
```

As a next step, in our DeepSpeed initialization code, we must create a pipeline wrapper around our model. This wrapped model is the new `model` variable that will be passed to `deepspeed.initialize()`:

```python
## gptlite.py

def get_model(criterion, vocab_size, pipeline_num_stages=0):
  # ...
  if pipeline_num_stages:
    deepspeed.runtime.utils.set_random_seed(random_seed)
    pipe_kwargs={
      'num_stages': pipeline_num_stages,
      'loss_fn': criterion,
      }
    model = gptlite.GPTlite(vocab_size).to(device_str)
    model = deepspeed.pipe.PipelineModule(layers=model.to_layers(), **pipe_kwargs)
  else:
    # ... as before: model = gptlite.GPTlite(vocab_size)
```

Finally, the training iteration code in the pipelining use case is reduced to a call to `engine.train_batch()`, that is [equivalent to a forward pass, backward pass and gradient updates of an entire micro-batch](https://www.deepspeed.ai/tutorials/pipeline/#training-loops) of size `engine.gradient_accumulation_steps()`:

```python
## train.py

def main_deepspeed(n_epochs=100, random_seed=42):
  # ...
  for epoch in range(n_epochs):
    if pipeline_num_stages:
      step_count = len(train_dataset)//engine.gradient_accumulation_steps()
      for step in range(step_count):
        loss = engine.train_batch()
    else:
      # ... forward, backward, and update step as before
```

An important nuance: by default, pipeline parralelism expects all mini-batches of the dataset - i.e. in every call to `train_batch()` - to be of the same shape. If this is not the case, you can reset the shapes at the onset of every mini-batch by running `engine.reset_activation_shape()`, and this will infer an additional communication step to broadcast the shapes of the first micro-batch as the default for the remaining micro-batches. However, it is not possible to have different shapes across micro-batches, and the only work around is to trim or pad all micro-batches of a mini-batch to the same shape beforehand.

As a final remark, [pipeline parallelism is not compatible with ZeRO stages 2 or 3](https://deepspeed.readthedocs.io/en/latest/pipeline.html#pipeline-parallelism), as discussed [here](https://github.com/microsoft/DeepSpeed/issues/1110#issuecomment-850835817).

### Increasing compute and memory efficiency with LayerSpec (optional) 

The implementation of pipelining for the `GPTlite` model above is neither memory efficient nor scalable as each GPU replicates the whole model in memory. See [Memory-Efficient Model Construction](https://www.deepspeed.ai/tutorials/pipeline/#memory-efficient-model-construction) for details. So we will use the DeepSpeed class `LayerSpec` ([API](https://deepspeed.readthedocs.io/en/latest/pipeline.html#deepspeed.pipe.LayerSpec)) that delays the construction of modules until the model layers have been partitioned across workers, therefore having each worker allocate only the layers it’s assigned to. To do this, we will create a new model class `GPTlitePipeSpec` that inherits from `PipelineModule` with an `__init__` method that follows very closely the `forward()` pass in the original `GPTlite`.

The tricky bit here is that the `LayerSpec` constructor only works with the type `nn.Module` as argument, and some operations, specifically the sum of embeddings in `forward()` is not of type `nn.Module`. To overcome this, we create the classe `EmbeddingsSum` that encapsulate thoat logic into an `nn.Module`. We will also use `CrossEntropy_FlatView` as the loss function. The full implementation for the pipeline class is then:

```python
## gptlite.py

from deepspeed.pipe import PipelineModule, LayerSpec

class GPTlitePipeSpec(PipelineModule):

  class EmbeddingsSum(nn.Module):
    """ converts tok_emb + pos_emb into an nn.Module. Required for LayerSpec"""

    def __init__(self, vocab_size):
      super().__init__()
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
      B, T = idx.shape
      tok_emb = self.token_embedding_table(idx)
      pos_emb = self.position_embedding_table(torch.arange(T).to(idx.device))
      return tok_emb + pos_emb

  def __init__(self, vocab_size, pipe_kwargs):
    self.specs = \
      [ LayerSpec(GPTlitePipeSpec.EmbeddingsSum, vocab_size) ] + \
      [ LayerSpec(Block, n_embd, n_head) for _ in range(n_layer)] + \
      [ LayerSpec(nn.LayerNorm, n_embd),
        LayerSpec(nn.Linear, n_embd, vocab_size, bias=False) ]
    super().__init__(layers=self.specs, **pipe_kwargs)
```

then we add the flag `--pipeline_spec_layers` to the command line arguments, so that we can optionally enable this feature:

```python
## train.py

def get_cmd_line_args():
  # ...
  parser.add_argument("--pipeline_spec_layers", action="store_true",
                      help="enable LayerSpecs in pipeline parallelism")
```

and change the `get_model()` method to retrieve the efficient pipeline variant as:

```python
## gptlite.py

def get_model(criterion, vocab_size, pipeline_num_stages=0, pipeline_spec_layers=False):

  if pipeline_num_stages:
    if pipeline_spec_layers:
      model = GPTlitePipeSpec(vocab_size, pipe_kwargs=pipe_kwargs)
    else:
      # ... GPTlite model as before 
```

We will denominate the `LayerSpec`-based implementation of pipeline parallelism by *memory-efficient pipelining*.

For heterogeneous models, load balancing of the model across GPUs may be an issue. There are several metrics to load balance from: runtime, memory usage, parameter count, etc. Here, we will not tune the [load balancing method for pipeline modules](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules), and will instead use the default `partition_method=parameters`. This assigns layers to stages in a way to load-balance the parameters, i.e. stages may have different lengths. Finally, in the extreme case the 1F1B algorithm is not the pipeline algorithm we want, we can [extend pipeline parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule) with a different algorithm. 

### Adding activation checkpointing to pipeline parallelism

In case we are using pipelining, introducing checkpoint at a fixed layer interval is straightforward, we just need to specify it by the argument `activation_checkpoint_interval` in the `PipelineModule` constructor: 

```python
#gptlite.py

def get_model(criterion, vocab_size, pipeline_num_stages=0, \
  pipeline_spec_layers=False, activation_checkpoint_interval=0):

  if pipeline_num_stages:
    pipe_kwargs={ # ...
      'activation_checkpoint_interval': args.activation_checkpoint_interval, 
    }
  # ....
```

However, activation checkpointing is also tricky to configure when using pipelining, if the checkpoint layer falls in another GPU. The rationale is that if a checkpoint layer falls in a different GPU than the layer being back-propagatems, this requires extra communication. This is an use case that I believe DeepSpeed is not handling correctly, so make sure there's a checkpoint layer *at the beginning* of the first block on each GPU.

### Gradient accumulation and micro-batching in pipeline parallelism

We saw before that we could activate the number of micro-batches per GPU, batch size and gradient accumulation steps per GPU, by settings those flags in the [DeepSpeed config file](https://www.deepspeed.ai/docs/config-json/). In pipeline parallelism, the concept of micro-batching refers to the number of micro-batches processed sequentially per stage per GPU. There is an important difference in memory using when performing gradient accumulation in data vs pipeline parallelism:
- in data parallelism, we perform a sequence of forward and backward passes. Memory increase is the same for any number of micro-batches as activations are released from memory ate every backward pass.
- in pipeline parallelism, we pass inputs sequentially but we still need to accumulate in memory the activations for all forward passes in that mini-batch. This may lead to a substantial memory increase.

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="/assets/GPT-lite-DeepSpeed/pipeline_4_stages.png"/>
&nbsp;
<img width="60%" height="60%" src="/assets/GPT-lite-DeepSpeed/pipeline_4_stages_grad_accumulation.png"/>

{: style="text-align:center; font-size: small;"}
An illustration of a regular pipeline execution across 8 processes, 4 stages, with no gradient accumulation (left) and with gradient accumulation of 6 micro-batches per mini-batch (right) .

Finally, there are several algorithms for pipeline and micro-batch scheduling. The two most commonly used are:
- the **regular** algorithm performs a sequence of all micro-batch forward passes, waits for their completion, then performs a sequence of backward passes, and waits for their completion, before starting a new mini-batch.
- the **1F1B** implemented by DeepSpeed performs a sequence of forward passes, and asynchronously starts the backward pass for each micto-batch forward pass completed. It then wait for all forward and backward passes to complete before starting the new mini-batch.

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/GPT-lite-DeepSpeed/pipeline_algorithms.png"/>

{: style="text-align:center; font-size: small;"}
Regular and 1F1B pipeline algorithms diagram. Source: paper [Training and Serving System of Foundation Models: A Comprehensive Survey](https://arxiv.org/pdf/2401.02643.pdf)

With that in mind, you can define the micro-batching properties by setting the fields `train_micro_batch_size_per_gpu` (defaulted to `train_batch_size`) or `gradient_accumulation_steps` (defaulted to `1`) in the [DeepSpeed config file](https://www.deepspeed.ai/docs/config-json/). At runtime, the micro-batch size can be retrieved by `engine.gradient_accumulation_steps()`.

<!--
{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/GPT-lite-DeepSpeed/GPT_pipelining_2.png"/>

{: style="text-align:center; font-size: small;"}
"An illustration of how DeepSpeed will train a batch with eight micro-batches using hybrid two-way data parallelism and two-stage pipeline parallelism. GPUs 0 and 2 are arranged in a pipeline and will alternate forward (F) and backward (B) passes. They will then all-reduce (AR) gradients with their data parallel counterparts, GPUs 1 and 3, respectively. Finally, the two pipeline stages update their model weights". This is the 1F1B pipeline algorithm. Source: [DeepSpeed pipelining documentation](https://www.deepspeed.ai/tutorials/pipeline/)
-->

## Communication quantization with ZeRO++

We can **optimize/compress communication with [ZeRO++](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/)**. To understand ZeRO++'s gains, we should undertand the communication workflow first (from the [ZeRO++ paper](https://arxiv.org/abs/2306.10209)): "Assume the model size as 𝑀. During the forward pass, ZeRO conducts an all-gather operation to collect all the parameters (𝑀) needed to train for all model layers. In the backward pass, ZeRO re-collects parameters (𝑀) with all-gather first, then each GPU can compute local gradients. After that, ZeRO operates reducescatter function to aggregate and redistribute gradients (𝑀) across accelerators. In total, ZeRO has a total communication volume of 3𝑀, spreads evenly across 2 all-gather and 1 reduce-scatter."

ZeRO++ introduces three new communication improvements:
1. **Quantized Weight Communication for ZeRO (qwZ)**: perform block quantization of the forward all-gather, converting weights  from `FP16` (2 bytes) to `INT8` (1 byte). The main improvement is to replace the typical quantization algorithm (multiplying all parameters by a scalar), by a quantization per block (ie per parameter subset) that includes multiplication by a factor and shifting values by another factor;
2. **Hierarchical Weight Partition for ZeRO (hpZ)**: data remapping that trades-off communication for more memory and reduces communication overhead of all-gather on weights during backward. Instead of having weights distributed across GPUs, we maintain a full copy on each machine, allowing us to replace the expensive cross-machine all-gather on weights with a faster intra-machine all-gather.
3. **Quantized Gradient Communication for ZeRO (qgZ)**: replaces the gradients reduce-scatter collective, by doing (1) block-based quantization of gradients to `INT4` during communication to reduce the communication size, and recovering the full precision before the reduction operator to preserve training accuracy.

ZeRO++ is particularly relevant for clusters with a low-latency network where collective communications are responsible for a large fraction of the overall runtime. It is also important for executions with a small batch size per GPU, where the memory increase of **qgZ** has no impact on scaling.

To set the hierarchical Weight partition for ZeRO (hpZ), quantized weight communication for ZeRO (qwZ) and quantized gradient Communication for ZeRO (qgZ) in the config file, add the following :

```json
{
  "zero_hpz_partition_size": 8, 
  "zero_quantized_weights": true,
  "zero_quantized_gradients": true,
}
``` 

Note that the according to documentation, the ideal value for `zero_hpz_partition_size` is the number of ranks (GPUs) per node. As a good engineering practice, it should be dynamically set with the API at runtime - not with the config file - to allow for a variable GPU count.

## Model/Tensor parallelism

**Tensor parallelism**, **vertical parallelism**, **intra-layer parallelism**, **activation parallelism** or most commonly ad confusedly called **model parallelism**, is the third dimension of parallelism and aims at partitioning the computation of tensors (activations) in the forward and backward passes. This requires a modification of the workflow of the computation in order to work in a distributed manner, particularly on the matrix multiplications format: in practice we must decide for the dimension of tensor partitioning (row, wise, none) and adapt the communication and computation accordingly, leading to an all-gather, all-reduce, scatter-reduced distributed matrix multiplicatioon, etc. Therefore, it is a model-specific implementation, and is [supported but not provided](https://www.deepspeed.ai/training/#support-for-custom-model-parallelism) by DeepSpeed, except in some built-in implementations such as [Megatron-ML](https://www.deepspeed.ai/tutorials/megatron/), for BERT, T5, GPT2 and others.

We can have a better understanding if we try to replicate the partitioning suggested by Megatron-LM in the paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). The main rationale underlying the paper is that Transformer-based models have two main components - a feed-forward (MLP) and an attention head - and we can do a forward and a backward pass on each of these blocks with a single collective communication step.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPT-lite-DeepSpeed/Megatron-LM.png"/>

{: style="text-align:center; font-size: small;"}
The MLP and self-attention blocks in the Megatron-LM implementation. f and g
are conjugate identity and all-reduce operations that c. f is an identity operator in the forward pass and all
reduce in the backward pass while g is an all reduce in the forward
pass and identity in the backward pass. Source: [Megatron-LM paper](https://arxiv.org/abs/1909.08053)

The rationale is the following: the MLP is sequence of (1) a MatMul $$AX$$, (2) a ReLU $$Y$$; a (3) MatMul $$YB$$ and a (4) dropout layer. Let's focus on the first two operations: 
1. "One option to parallelize the GEMM is to split the weight matrix $$A$$ along its rows and input $$X$$ along its columns. [...] This partitioning will result in $$Y = GeLU(X_1 A_1 + X_2 A_2)$$. Since GeLU is a nonlinear function, $$GeLU (X_1 A_1 + X_2 A_ 2) \neq  GeLU( X_1 A_1 ) + GeLU (X_2 A_2)$$ and this approach will require a synchronization point (*sum-reduce*) before the GeLU function". We can follow this approach - pictured below - for each of the 2 MatMuls, yielding a total of two communication steps.

  {: style="text-align:center; font-size: small;"}
  <img width="100%" height="100%" src="/assets/GPT-lite-DeepSpeed/megatron_model_parallel_1.png">


2. Another option is to don't split $$X$$ and split $$A$$ along its columns, that "allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM". The output of the ReLU $$Y$$ is then $$[Y_1, Y_2] = [GeLU (X A_1), GeLU(X A_2)]$$ and this removes a synchrnization point.
After the MatMul and ReLU is done, we must follow with another MatMul and a dropout. Note that if we follow this approach (pictured below), the output $$Y$$ of the first MatMul (which is the input for the second MatMul) is split column-wise. Therefore, Megatron will then use the previous approach for the second MatMul, that requires 1 collective communication step. We therefore perform 1 communication step for 2 MatMuls.

  {: style="text-align:center; font-size: small;"}
  <img width="80%" height="80%" src="/assets/GPT-lite-DeepSpeed/megatron_model_parallel_2.png">


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

Finally, there is [ongoing work from PyTorch to support general model parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html), where the user can pick row-wise split, column-wise split and sharding of individual tensor. Also, combining data, pipeline and model parallelism requires one to define the corect strategy for [custom model parallelism](https://www.deepspeed.ai/training/#model-parallelism).

## Results

We changed our config to <a href="/assets/GPT-lite-DeepSpeed/ds_config.json">`ds_config.json`</a> to run ZeRO stage 1 and tested our execution with different stage count and the memory-efficient `SpecLayer` implementation of our GPT model  (with `--pipeline_num_stages <num_stages> --pipeline_spec_layers`). We did not use activation checkpointing due to an open bug [4279](https://github.com/microsoft/DeepSpeed/issues/4274). We tested 1, 2, 4 and 8 pipeline stages per run. We rely on the default DeepSpeed algorithm for load balancing of stages, based on the parameter count. As an example, for the partitioning of GPT-lite pipeline across 8 GPUs and 4 stages, it outputs:
   ```
RANK=0 STAGE=0 LAYERS=4 [0, 4)   STAGE_PARAMS=21256704 (21.257M)
RANK=2 STAGE=1 LAYERS=3 [4, 7)   STAGE_PARAMS=21256704 (21.257M)
RANK=4 STAGE=2 LAYERS=3 [7, 10)  STAGE_PARAMS=21256704 (21.257M)
RANK=6 STAGE=3 LAYERS=6 [10, 16) STAGE_PARAMS=21308160 (21.308M)
   ```

Model parallelism results will be added in the new future. The current results are the following:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-DeepSpeed/benchmark_gptlite.png"/>


**Pipelining performance for variable number of stages.** Increasing the number of stages strongly decreases the average memory consumption, as expected. This is due to the model being partitioned in smaller blocks. There was no substantial decrease in maximum memory consumption, and this is something I am yet to understand (any hints?). The throughput demonstrated a peculiar behaviour: in the deep benchmark model, the throughput increases with the increase of stage count, while the opposite happens on the GPT-lite model. I believe this is due to load imbalance across stages, or a lower ratio of computation vs communication as we increase the stage count on the GPT-lite use case.

**Pipelining with optimized vs non-optimized memory efficiency implementation.** Using the `SpecLayer`-based implementation of the `PipelineModule` in our pipeline runs, resulted in a reduction of about 40% in memory consumption for the GPT-lite and deep benchmark models, when running pipeline parallelism with the highest stage count (8).

**Memory usage**: on pipeline parallelism, I noticed that the first GPU seems to require a considerably higher ammount of memory when compared to the remaining GPUs. This should not be the case, particularly on the deep benchmark model where we can guarantee a quasi-ideal stage partitioning across GPUs. This disparity in memory usage on GPU 0 is the main indicator of the maximum memory required, and balancing this would bring that value down. I [opened a bug report](https://github.com/microsoft/DeepSpeed/issues/4477) with DeepSpeed and will wait for their feedback or fix to correct this analysis.

Finding the best parallelism strategy, and choosing between different ZeRO stages, offloading, activation checkpointing intervals, pipeline parallelism stages, data parallelism, etc, is a very complex problem, as it depends on the ML model, data and hardware. In practice, our config file and parallelism settings are a manually-optimized ballpark figure of the default config file with some parameter grid search. In this topic, there is still plenty of work to be done to make it optimal, possibly by exploring the [autotuning](https://www.deepspeed.ai/tutorials/autotuning/) tools in DeepSpeed.

### Further resources and code

There is a lot of food for thought here, and I will be updating this post as I find new insights.

We just scratched the surface of DeepSpeed capabilities. There are plenty of resources that should also be explored. To name a few: [**autotuning**](https://www.deepspeed.ai/tutorials/autotuning/) ([README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)) for parallelism hyper-parameters discovery; [**model checkpointing**](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) for saving and resuming execution state, a life saviour for long runs; [**flops profiler**](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html) measures the time, flops and parameters of individual layers, [**sparse attention kernels**](https://www.deepspeed.ai/2020/09/08/sparse-attention.html) ([API](https://www.deepspeed.ai/docs/config-json/#sparse-attention)) to support long sequences of model inputs, such as text, image, or sound; [**communication optimizers**](https://www.deepspeed.ai/training/#1-bit-adam-01-adam-and-1-bit-lamb-optimizers-with-up-to-26x-less-communication) offer the same convergence as Adam/LAMB but incur 26x less communication and 6.6x higher throughput on large BERT pretraining, [**monitor**](https://www.deepspeed.ai/training/#monitor) to log live training metrics to TensorBoard, csv file or other backend; [**model compression**](https://www.deepspeed.ai/compression/) ([API](https://www.deepspeed.ai/docs/config-json/#compression)) via layer reduction, weight quantization, activation quantization, sparse pruning, row pruning, head pruning and channel pruning, to deliver faster speed and smaller model size.

Finally, this code has been added to the [GPT-lite-DeepSpeed repo](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-DeepSpeed), if you want to try it.
