---
layout: post
title:  "Distributed GPT model: data parallelism, sharding and offloading via Torch DDP and DeepSpeed"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---


Distributed data parallelism (DDP) refers to the parallel execution of different input samples across processors. If you consider any data input to be of shape $$B \times T \times E$$ as in batch size, sequence/temporal size, and embedding/features size, then data parallelism refers to the use case where we split the data across $$P$$ processes, leading to a local input of shape $$B/P \times T \times E$$:

{: style="text-align:center; font-size: small;"}
<img width="35%" height="35%" src="/assets/GPT-lite-distributed/data_parallelism.png"/>

{: style="text-align:center; font-size: small;"}
An illustration of the DDP data layout, split across 4 processes colorcoded as blue, yellow, read and green.

In this post, we will perform distributed data parallelism on the training process of the [GPT-lite model we built in the previous post]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), on a network of 8 GPUs, using PyTorch's [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer, a lightweight wrapper on PyTorch).

There are two main data parallelism approaches:

**Distributed Data Parallelism** keeps a full copy of the model (weights, optimizer parameters and gradients) on each processor. All models are initialized equally. Each processor takes as input a different minibatch and performs a forward pass to compute the loss that relates to that batch. On the backward pass, at every layer of the model, all processes compute the gradients of that batch, and perform an all-reduce to get the mean gradients across all processors. This is then used to update the optimizer states. Because parameters are initialized equally, and the gradients are mean-reduced, and all parameters perform the same updates, the models across all processors are kept in an identical state throughout the execution. 

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/GPT-lite-distributed/workflow_data_parallelism.png"/>

{: style="text-align:center; font-size: small;"}
Workflow for distributed data parallelism (DDP) across 4 color-coded processes. Each process holds a copy of a 4-layer feed-forward model, initialized identically. Each process performs a forward pass of its own data (arrow pointing up on the left of the model). On the backward pass, all processes compute the local gradients and mean-reduce them across the network (arrow pointing down on the right of the model). The mean of the gradients is then used by the optimizer to update the parameter states. 

Looking at the previous, we can see that each processor holds a copy of the model and this leads to a superfluos memory usage. That's where sharding comes into play.

**(Fully-)Sharded Data Parallelism (FSDP)** a.k.a **sharding** is a distributed setup where processors dont hold a full copy of the model, but only the parameters, optimizer states and gradients of a subset of layers. As before, different processors input different mini-batches. In DeepSpeed lingo, sharding goes by the name of **ZeRO (Zero Redundancy Optimizer)**. ZeRO has several alternative execution modes (**stages**). Each stage represents a different level of memory redundancy, corresponding to different variable types being distributed or replicated across nodes:
- **ZeRO stage 0**: no sharding of any variables, being equivalent to Distributed Data Parallelism;
- **ZeRO stage 1 (ZeRO-1)**: the optimizer states (e.g., for Adam optimizer, the weights, and the first and second moment estimates) are partitioned across the processes. Affects only the backward pass.
- **ZeRO stage 2 (ZeRO-2)**: the gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states. Also relevant only for the backward pass.
- **ZeRO stage 3 (ZeRO-3)**: the model parameters are partitioned across the processes. Includes a communication overhead on both the forward and backward passes. 

An important remark is that activations are not sharded i.e. they are kept in *full shape* on each processor. And in modern models, **a huge chunk of memory is allocated to residual memory (activations, normalization layers, etc) which is not sharded by FSDP**. With that in mind, the following diagram illustrates the workflow of the stage 3 sharding (of parameters, gradients and optimizer states):

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/workflow_sharding.png"/>

{: style="text-align:center; font-size: small;"}
Workflow for stage 3 sharding. Each processor contains a non-overlapping subset of parameters, gradients and optimiser data, split by assigning different layers to different processes. Each processor loads a different data batch. During forward and backward passes (represented by arrows on the left and right of model, respectively), when computing is needed for a given layer, the process who is responsible for those layers will broadcasts and gather those values to/from the remaining processes. Example for processor 1 (yellow): **Data loading:** yellow process loads data sample 1.  **Forward pass:** yellow process receives the parameters from rank 0 (blue) and computes the activations for layer 1. Afterwards, yellow process broadcasts its parameters to ranks 0, 2 and 3, so that they compute their activations for layer 2. Activations for layer 3 and 4 are computed similarly to layer 1, led by the red and green processes, specifically. **Backward pass:** the green process (3) broadcasts parameters to all other processes. Each process can use its activations and the received parameters to compute the gradients for the top layer. All processes gather their local gradients in process 3 that will use it to update the parameters of the last layer. For the remaining layers, the same happens, where the red, yellow and blue processes will be the ones doing the broadcast of parameters and gather of gradients (iteratively).

The higher the stage, the more communication we require, but the less memory we consume. These memory improvements are summarized in the [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) as:

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/GPT-lite-distributed/DeepSpeed_stages.png"/>

{: style="text-align:center; font-size: small;"}
Memory consumption of the three different stages of ZeRO FSDP.  Source: [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

Additionaly, on top of stages 1 and 2, we can enable [**ZeRO-Offload**](https://www.deepspeed.ai/tutorials/zero-offload/), a system for offloading optimizer and gradient states to CPU memory. On top of stage 3, we can enable [**ZeRO-Infinity**](https://arxiv.org/abs/2104.07857), also an offloading engine that extends ZeRO-offload with support to NVMe memory. According to the [ZeRO-3 documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html#zero), "ZeRO-Infinity has all of the savings of ZeRO-Offload, plus is able to offload more the model weights and has more effective bandwidth utilization and overlapping of computation and communication".

## Model and dataset setup

We start out implementation by taking our previous *GPT-lite* with the specs matching the *GPT-2 Small* model in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Fig 2.1):

```python
n_layer = 12   # depth of the network as number of decoder blocks.
n_embd = 768   # size of the embeddings (d_model)
n_head = 12   # number of attention heads in the Multi-Attention mechanism
block_size = 2048   # block size ie max number of training sequence, the $n_{ctx}$ in the paper .
dropout = 0.1   # dropout rate (variable p) for dropout units
```

We then define the methods `get_model()` and `get_dataset()` that return our model and the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset:

```python
## gptlite.py

def get_dataset():
  
  class GPTliteDataset(torch.utils.data.Dataset):

      def __init__(self, train_data, block_size):
        self.train_data = train_data
        self.block_size = block_size

      def __len__(self):
        return len(self.train_data)

      def __getitem__(self, idx):
        # generate 1 random offset on the data
        ix = torch.randint(len(self.train_data)-self.block_size , size=())
        # input is a random subset of tokens
        x = self.train_data[ix   : ix+self.block_size]
        # target is just x shifted right (ie the next predicted word)
        y = self.train_data[ix+1 : ix+1+self.block_size]
        return x, y

  train_data, valid_dataset, vocab_size = load_tiny_shakespeare_data()
  train_dataset = GPTliteDataset(train_data, gptlite.block_size)
  valid_dataset = GPTliteDataset(valid_data, gptlite.block_size)
  return train_dataset, valid_dataset, vocab_size


def get_model(vocab_size):
  return GPTlite(vocab_size)
```

### Detour: using a pre-existing model {#torchvision-model}

Note that code this code is applicable to any model of type `torch.nn.Module` and any dataset of type `torch.utils.data.Dataset`. As an example. if you wanted to perform a multi-class classification using the `ResNet` network on the `CIFAR10` dataset available in `torchvision`, you could define the previous 2 methods as:

```python
import torchvision

def get_dataset():
  import torchvision.transforms as transforms
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
  return dataset

def get_model(num_classes):
  return torchvision.models.resnet18(num_classes=num_classes)
```

As a relevant remark, pre-existing models do not define activation checkpointing layers and pipelining layers that are required to activate these two features (discuss later). 


## PyTorch implementation

We will now impement data parallelism in PyTorch. Firstly, we collect the global variables that are set by the [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) launcher (detailed below), as they're important to uniquely identify processes in the network and GPU devices within a node:

```python
import os
rank = int(os.environ['RANK'])   # the unique id across all processes in all nodes
local_rank = int(os.environ['LOCAL_RANK'])   # the unique id across this node
world_size = int(os.environ['WORLD_SIZE'])   # the number of processes across all nodes
```

Now we define the [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) that tells each process how to iterate through the data:

```python
dataloader = torch.utils.data.DataLoader(
  dataset, batch_size=4, sampler=sampler)
```

Note the argument `sampler`, that is a [`DistributedSampler`](DistributedSampler) that will delegate different samples from the dataloader to different processes. Without this, all processes would load exactly the same datapoints in every iteration.

```python 
sampler = torch.utils.data.distributed.DistributedSampler(
  dataset, num_replicas=world_size, rank=rank)
```

We then place each model in a different GPU with the correct data type:

```python
device = f"cuda:{local_rank}"
dtype = torch.bfloat16
model = model.to(device=device, dtype=dtype)
```

and we finally wrap it with the [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for the DDP implementation:

```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
```

or with the [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html) wrapper for the sharded implemetation, as:

```python
model = torch.distributed.fsdp.FullyShardedDataParallel(
    model,
    device_id=self.device,
    sharding_strategy=torch.distributed.fsdp.api.ShardingStrategy.SHARD_GRAD_OP, # define the stage here
)
```

And that's it. Now you can write your training loop normally and torch will do all the communication and synchronization  under the hood:

```python
for step, data in enumerate(dataloader):
  inputs = data[0].to(engine.device)
  labels = data[1].to(engine.device)
  outputs = engine(inputs)  # fwd pass
  loss = torch.nn.functional.cross_entropy(outputs, labels) # loss
  loss.backward() # compute gradients
  optimizer.step() # update parameters
  optimizer.zero_grad(set_to_none=True)
```  

Because gradients are computed and mean-reduced from the top to the bottom layer of the model, we can overlap the computation of the gradients from the lower layers with communication of the upper layers, as we go along our backward pass. According to the [PyTorch DDP documentation](https://pytorch.org/docs/master/notes/ddp.html#internal-design):

> The backward pass iteratively computes gradients (from last to first layer) and collects blocks of gradients to be communicated. These blocks will be mean-reduced asynchronously during the backward pass, while the computation for the backward pass proceeds. Therefore it overlaps backward pass computation with gradients communication. At the end of the backward pass, all GPUs wait for all gradient all-reduces to finish, and then triggers the parameter updates.

For extra memory savings, offloading of tensors can also be achieved via PyTorch by using custom [hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).

Finally, you can launch the application by calling [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html). Torchrun is a network bootstrapper that spaws a python script across compute all compute nodes in a network and sets the previous environmental variables that allow for processes to be uniquely identifiable across the network. The usage is simple:

```shell
$ torch --standalone, --nproc_per_node=4, ./train.py
```

where `nproc_per_node` is the number of GPUs on each node.

## DeepSpeed implementation


Implementing an existing code in DeepSpeed is pretty simple. To start, DeepSpeed features can be activated via the deepspeed API or its [Configuration JSON](https://www.deepspeed.ai/docs/config-json/). The number of possible optimizations is large, as it can defines parallelism, floating point precision, logger, communication parameters, etc. In our implementation, we will start with a simple config file, where we configure the DeepSpeed logger to output memory and throughput info at every 10 epochs (`steps_per_print`), we set the batch size to `256` and (optionally) define the settings of the optimizer (`optimizer`) and learning rate scheduler (`scheduler`):

```json
{
  "train_batch_size": 256,
  "steps_per_print": 10,
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 0.001, "betas": [ 0.8, 0.999  ], "eps": 1e-8, "weight_decay": 3e-7 }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": { "warmup_min_lr": 0, "warmup_max_lr": 0.001, "warmup_num_steps": 1000 }
  }
}
```

Note that in DeepSpeed lingo, the `micro_batch_size_per_gpu` refers to the batch size loaded per dataloader (ie per node, per gradient accumulation step), while `train_batch_size` refers to the batch size across all gradient accumulation steps and processes in the network ie:

```
train_batch_size = micro_batch_size_per_gpu * num_gpus * num_nodes * gradient_accumulation_steps
```

**ZeRO Fully-Sharded Data Parallelism** can be activated by specifying the relevant stage in the config file. If omitted, or when passing the stage 0, DeepSpeed is disabled and the execution follows a regular distributed data paralllel workflow:

```json
{
  "zero_optimization": { "stage": 3 }
}
```

CPU-offloading is called [**ZeRO-Infinity**](https://arxiv.org/abs/2104.07857) and performs offloading of several variables in memory to CPU and VNMe, providing huge memory savings. It is only compatible with ZeRO-3 and can be enabled with: 

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true },
  }
}
```

There's also similar field called [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) for stage 2. 

We're almost done now. Once we have our config file properly calibrated, the implementation is straighforward. All boilerplate that PyTorch requires for parallelism and data loaders is managed internally by DeepSpeed. So we only need to setup DeepSpeed as:

```python
def main_deepspeed(n_epochs=100, random_seed=42):

  torch.manual_seed(random_seed)  #set random seed (used by DataLoader)
  deepspeed.runtime.utils.set_random_seed(random_seed) #set DeepSpeed seed
  deepspeed.init_distributed()  # initialize distributed DeepSpeed
  config = 'ds_config.json'  # load the DeepSpeed config
  criterion = torch.nn.CrossEntropyLoss()  # initialize loss function
  train_dataset, _, vocab_size = gptlite.get_dataset()  # initialize GPT-lite dataset
  model = gptlite.get_model(vocab_size)  # initialize GPT-lite model

  engine, optimizer, train_dataloader , _ = deepspeed.initialize(
    config=config, model=model, training_data=train_dataset,) # initialize deepspeed
```

We then write the training loop, with a structure very similar to the PyTorch implementation. The only exception is that we don't perform zeroing of gradients, as this is managed internally by DeepSpeed. Also, `initialize()` already returns a `train_dataloader` that assigns disjoint subsets of data to each process, so we dont need to fiddle with distributed dataloaders and samplers.

```python
  for epoch in range(n_epochs):
    for step, data in enumerate(train_dataloader):
      inputs = data[0].to(engine.device)
      labels = data[1].to(engine.device)
              
      outputs = engine(inputs)  # fwd pass
      loss = criterion(outputs, labels)
      engine.backward(loss)  # backprop
      engine.step()  # update weights, no need for zero-ing
```

Finally, we can launch our run with the `torchrun` launcher as before, or with the launcher included in DeepSpeed as:

```shell
$ deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
```

<!--
[**Mixed precision representation**](https://arxiv.org/abs/1710.03740) allows for calculus with value types (parameters, activations, accumulators) stored with different numerical representations, leading to a reduction of memory and compute time. It can be enabled by adding the `fp16` entry [in the config](https://www.deepspeed.ai/docs/config-json/#fp16-training-options). As a side note, the `amp` config entry also enables mixed precision training that follows the [NVIDIA Apex](https://nvidia.github.io/apex/) implementation i.e. with the `O0` to `O3` opimization levels. However, [it is not compatible with ZeRO](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options), therefore we won't use it. The [`fp16` is equivalent to APEX optimization level O2](https://www.deepspeed.ai/docs/config-json/#fp16-training-options), and according to the [documentation](https://www.deepspeed.ai/docs/config-json/#fp16-training-options), "if you want to use ZeRO (currently) you must use this mode". We can enable it with the entry `"fp16: { enabled: true }` that is equivalent to the following default values:

```json
{
  "fp16": {
    "enabled": true,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "consecutive_hysteresis": false,
    "min_loss_scale": 1
  }
}
```

However, if your hardware supports bfloat16 (brain floating point), this should be used in lieu of float16, as it has a longer integer (exponent) representation: 8 bits instead of the 5 in float16, ie the same 8 bits as in float32. This makes it more numerically stable and does not require loss scaling. bfloat16 can be activated by adding to the config `bf16 { "enabled": true }`.

**Limiting the size of communication buffers** is important when activating ZeRO. In practice, enabling ZeRO leads to the distribution of parameters across all processors. This in practice will add a communication overhead, that requires memory to be allocated for all buffers responsible for the data to be sent or received. This is an issue as these buffers may be large. To overcome this issue, we can decrease the maximum size of the communication buffers so that communication is performed in parcels of smaller buffers. We can also enable **communication overlap** that attempts to overlap the reduction of the gradients with backward computation. To enable these 2 optimizations, we add to the config:

```json
{
  "zero_optimization": {
    "reduce_bucket_size": 4e5,
    "allgather_bucket_size": 4e5,
    "stage3_prefetch_bucket_size": 4e5,
    "overlap_comm": true,
  }
}
```


**Gradient accumulation** - also known as **micro-batching** - is a technique that simulates a large mini-batch as an iteration across several micro-batches. This is particularly relevant when the whole mini-batch does not fit into memory, and using an accumulation of micro-batches will overcome that limitation. This method is enabled by setting `train_micro_batch_size_per_gpu` (defaulted to `train_batch_size`) or `gradient_accumulation_steps` (defaulted to `1`). At runtime, the micro-batch size can be retrieved by `engine.gradient_accumulation_steps()`. In our case, we will start with a micro-batch of 1 single input per GPU, that accummulate up to a batch size of 256 across all 8 GPUs, therefore resulting in 32 gradient accumulation steps: 

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 1
}
```

**[Communication quantization (ZeRO++)](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/)** allows for optimization and compression of communication tensors by reducing its floating point representation. To understand ZeRO++'s gains, we should understand the communication workflow first (from the [ZeRO++ paper](https://arxiv.org/abs/2306.10209)): "Assume the model size as 𝑀. During the forward pass, ZeRO conducts an all-gather operation to collect all the parameters (𝑀) needed to train for all model layers. In the backward pass, ZeRO re-collects parameters (𝑀) with all-gather first, then each GPU can compute local gradients. After that, ZeRO operates reducescatter function to aggregate and redistribute gradients (𝑀) across accelerators. In total, ZeRO has a total communication volume of 3𝑀, spreads evenly across 2 all-gather and 1 reduce-scatter."

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

## Activation Checkpointing

[**Activation Checkpointing**](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html) allows for a large reduction in memory requirements by not storing all the forward pass activations that are required for the backward propagation. The rationale is simply: instead of storing the output of every layer after the forward pass, only a small subset of (checkpoint) layer outputs are kept in memory, and the remaining are computed on-the-fly - during the backward pass - with a forward pass from the closest lower layer. Activation checkpointing is extremelly relevant for DeepSpeed, as activations are not sharded, therefore not storing all layer activations in memory reduces substantially the memory footprint.

In our use case, and for simplicity, we will store layer activations at a user-specified interval. For that, we create the command line argument `--activation_checkpoint_interval` that specifies how often to store layer checkpoints:

```python
## train.py

def get_cmd_line_args(description='GPT lite on DeepSpeed'):
  # ...
  parser.add_argument('--activation_checkpoint_interval', type=int, default=0,
                      help='activation checkpoint interval (0 means disabled)')
  # ...
```

We have to manually specify which layers to checkpoint, by calling [`deepspeed.checkpointing.checkpoint`](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html#using-activation-checkpointing) at the checkpoint layers. We will use the previous `lo_layers()` method to iterate over the layers of a model and assign the relevant checkpointing in the `forward()` pass of `GPTlite` as:

```python
## gptlite.py

class GPTlite(nn.Module):
  #...

  def forward(self, idx, targets=None):

    if self.activation_checkpoint_interval > 0:
      x=idx
      for l, layer in enumerate(self.to_layers()):
        is_checkpoint = l % self.activation_checkpoint_interval == 0 
        x = deepspeed.checkpointing.checkpoint(layer, x) if is_checkpoint else layer(x)
      return x
```

where `self.activation_checkpoint_interval` is a value set during initialization of the class. Finally, when doing model parallelism, we can reduce memory substantially by partitioning activations and offloading those checkpoints to the CPU instead of saving them in memory. DeepSpeed does not support model/tensor parallelism natively so we will skip this, but check the [json documentation](https://www.deepspeed.ai/docs/config-json/#activation-checkpointing) if you are interested.

Finally, **activation checkpoint currently has two implementations: a reentrant and non-reentrant**. The non-reentrant will be the future default in PyTorch and is implemented via PyTorch saved variable hooks (as detailed [here](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d)). Non-checkpointed activations are not stored in memory, and instead replaced by a reference. Thus, the computation graph is not altered. The non-reentrant checkpointing allows for nested checkpointing (calling one checkpoint from another checkpoint function), allowing for **higher memory savings**.
The non-reentrant equivalent in deepspeed in implemented by [`deepspeed.checkpointing.non_reentrant_checkpoint`](https://github.com/microsoft/DeepSpeed/blob/42a8eaa705ed30b4a656ac71bdb400772df2cb21/deepspeed/runtime/activation_checkpointing/checkpointing.py).

The reentrant does not use hooks but calls the [`forward` autograd function](https://github.com/pytorch/pytorch/blob/670c5cf96249db28cde757da5a6aa97569760102/torch/utils/checkpoint.py#L75) instead. The gradient calculations are not part of the main computational graph anymore, and every checkpoint creates a mini-computational graph during the backward pass. One of the downsides, is that the whole `forward` function is computed for every call, while the non-reentrant counterpart can stop when the relevant activations are computed. Moreover, the whole graph is not stored (contrarily to non-reentrant) thus not allowing the backward to be run in the whole computational graph. More details in the [torch checkpoint documentation](https://pytorch.org/docs/stable/checkpoint.html).


### About activation checkpointing with parameters sharding

Combining activation checkpointing with sharded model parameters (ZeRO stage-3) may lead to a substantial runtime overhead. The problem is that, if you need to perform a forward pass from the closest checkpoint layer to collect the parameters required for the back propagation, and if those parameters are distributed (stage 3), then there has to be an extra collective communication step at every layer (from checkpoint layer to current back-prop layer) to collect those weights. This adds an extra communication overhead.
-->

### Detour: measuring memory allocated to parameters

We can use the [DeepSpeed API to estimate the memory requirements of model parameters](https://deepspeed.readthedocs.io/en/latest/memory.html#api-to-estimate-memory-usage) for different ZeRO implementations, by calling the following method at the onset of execution:

```python
def measure_parameters_memory(model):
  param_size_GB = sum([p.nelement() * p.element_size() for p in model.parameters()])/1024**3
  print(f"Native model parameters size: {round(param_size_GB, 2)}GB.")

  from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
  estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)

  from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
  estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
```

The output tells us that:
- the base model requires about 0.323GB for the storage of parameters, per GPU;
- DeepSpeed ZeRO-2 requires 0.161GB and 0.484GB for the with and without offload optimizations;
- DeepSpeed ZeRO-3 requires 0.009GB and 0.190GB for the with and without offload optimizations; 

This metric is very useful as it gives a quick overview of scaling and is very fast to compute. However, it has many fallacies: it only measures the parameters overheard, it does not take activations or other residual buffers (e.g. normalization variables) into account, does not take the batch size and numerical precision (or any field in the config file) into account, it does not consider temporary (e.g. communication) buffers, etc.

## Results

To measure our performance, we used the deepspeed logger to extract the following metrics from different runs at every 10 steps: model throughput as average number of samples per second, the average allocated memory, and the maximum allocated memory. We used `pytorch==2.01`, CUDA `11.7` and `deepspeed==0.10.3`.

All implementations use the same mixed-precision representation, communication bucket sizes, and other config parameters. We benchmarked the following implementations (and configs):

1. The distributed data parallel (DDP) implementation, i.e. no DeepSpeed (<a href="/assets/GPT-lite-distributed/ds_config.json">`ds_config.json`</a> with `'stage': 0`);
2. The fully-sharded data parallel implementation with ZeRO 1, 2 and 3 (<a href="/assets/GPT-lite-distributed/ds_config.json">`ds_config.json`</a> with `'stage' :1`, `2` or `3`);
3. The ZeRO-3 implementation with ZeRO-Infinity for CPU offloading (<a href="/assets/GPT-lite-distributed/ds_config_offload.json">`ds_config_offload.json`</a>);

We tested our GPT-lite model, with a micro-batch size of 1 sample per GPU, and the results are:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/benchmark_gptlite.png"/>


**Memory overhead from communication buffers.** Looking at the max vs average memory, note that the max memory in theory should be much higher at high ZeRO stages compared to low ZeRO stages and DPP. This is due to more parameters being communicated requiring more communication buffers. However, setting the communication bucket sizes to a low value in the config file overcomes this effect. In fact, we also benchmarked several runs with the default communication bucket sizes (`5e9`) and it led to a higher memory usage as expected (of approximately double the amount in stages 2 and 3), that became prohibitive for some runs.

<!-- 
**Activation checkpointing.** On the GPT-lite model, the main memory driver is the activations memory on the attention matrix (grows quadratically with the sequence length and linearly with the batch size). Therefore, sharding alone does not yield a signification memory reduction. Adding activation checkpointing overcomes this memory bottleneck by keeping at most one attention matrix in memory (recomputed on the fly), throughout the whole backward pass. Moreover, **mixed precision** has an important effect on throughtput as lower precision yields faster communication and computation. As an example, the results for ZeRO-1 with activation checkpointing and mixed precision are:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/benchmark_gptlite_activation_ckpt_throughput.png"/>

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/benchmark_gptlite_activation_ckpt_memory_usage.png"/>

**Parameter vs residual memory.** Note the difference between average memory and maximum memory. That gap in memory consumption is due to temporary memory dedicated to activations, residual buffers, communication buffers, etc. 
-->

**Communication vs computation trade-off from different stages in ZeRO.** In ideal scenarios, as you move from DDP to ZeRO-1, ZeRO-2, ZeRO-3 and ZeRO-Infinity, the memory consumption and throughput are reduced. As expected, we swap data locality for communication of parameters, and pay a price in performance for the communication/offload of parameters.

**Offloaded vs in-memory parameters.** Offloading proved to be a consistent way to reduce memory usage with the drawback of a small reduction of throughput, as expected.

**Lack of superlinear speedup**: We observed a small improvement of memory efficiency, but still far from the claims of the original DeepSpeed team. One explanation is that we used a small network of 8 GPUs, compared to the 64 to 800 GPUs used by the authors in their benchmarks, therefore we achieved a much smaller memory reduction. A large network of GPUs also underlies their claim of superlinear speed up that we did not observe.

Finally, we did not use **communication quantization** as did not result in any improvement. This goes in line with the ZeRO++ paper where it claims this method is applicable for small batch sizes or low-latency / low-bandwidth networks.

## Resources and code

In this post we explored only the dimension of data parallelism.  If you'd like to know more about DeepSpeed, check the [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest), the [training features page](https://www.deepspeed.ai/training/#features), the [tutorials page](https://www.deepspeed.ai/tutorials/), the [HuggingFace page for DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed), and the examples at [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/).

There are a lot of results and food for thought here, so I will update this post as I find new insights. Meanwhile, if you want to try this on your own, see the [GPT-lite-distributed repo](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-distributed).



