---
layout: post
title:  "Data parallelism, sharding and CPU offloading"
categories: ["Distributed Training"]
tags: ["data parallelism", "ZeRO", "FSDP", "CPU offloading", "activation checkpointing", "DeepSpeed", "PyTorch"]
---

[Data parallelism](https://en.wikipedia.org/wiki/Data_parallelism) trains one model on different input samples in parallel. For an input of shape $$B \times S \times E$$ (batch size, sequence length and embedding size), it splits the batch dimension across $$P$$ processes, so each process holds a local input of shape $$B/P \times S \times E$$:

{: .picture}
<img width="26%" height="26%" src="{{ site.assets }}/distributed-training/imgs/data_parallelism.svg"/>

{: .caption}
An illustration of the DDP data layout, split across 4 processes color-coded as blue, yellow, red and green.

The examples below train the GPTlite model of the [GPT-lite post]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}) on one node with 8 GPUs, with [PyTorch DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (Distributed Data Parallel), PyTorch FSDP (Fully Sharded Data Parallel) and DeepSpeed ZeRO (Zero Redundancy Optimizer). DeepSpeed is a training library built on PyTorch.

## Distributed Data Parallelism

Distributed data parallelism (DDP) keeps a full copy of the model on each process: its parameters, gradients and optimizer states. All copies start with the same parameters. Each process runs the forward and backward passes on its own samples. During the backward pass, an [all-reduce](https://en.wikipedia.org/wiki/Collective_operation) averages the gradients across the processes. Each process then runs the same optimizer step with the same average gradients, so the copies stay identical:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/workflow_data_parallelism.svg"/>

{: .caption}
Workflow for distributed data parallelism (DDP) across 4 color-coded processes. Each process holds a copy of a 4-layer feed-forward model, initialized identically. Each process performs a forward pass on its own data (arrow pointing up on the left of the model). On the backward pass, all processes compute the local gradients and mean-reduce them across the network (arrow pointing down on the right of the model). The mean of the gradients is then used by the optimizer to update the parameter states. 

The $$P$$ processes thus hold $$P$$ identical copies of the model states. Sharding removes these redundant copies.

## Sharded Data Parallelism

Sharded data parallelism splits the model states across the processes, and each process still trains on its own samples. DeepSpeed implements it as ZeRO ([ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), Rajbhandari et al., 2020), and PyTorch as FSDP ([PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277), Zhao et al., 2023). The *stage* of ZeRO sets which model states are split:
- **Stage 0**: nothing is split, as in DDP.
- **Stage 1 (ZeRO-1)**: splits the optimizer states. With [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) and [mixed precision](https://en.wikipedia.org/wiki/Mixed-precision_arithmetic), these are an fp32 copy of the parameters, the momentum and the variance. A reduce-scatter replaces the all-reduce: it gives each process the average gradients of its own partition. Each process updates its partition of the parameters, and an all-gather sends the new parameters to all processes.
- **Stage 2 (ZeRO-2)**: also splits the gradients. Each process keeps only the gradients of its partition.
- **Stage 3 (ZeRO-3)**: also splits the parameters. An all-gather collects the parameters of each layer before its forward pass, and again before its backward pass. Each process frees them after use, and keeps only its partition.

ZeRO splits only the model states. Each process still holds temporary buffers and the activations of its $$B/P$$ samples, which grow with the batch size and the sequence length. The ZeRO paper calls this memory the *residual states*. Activation checkpointing reduces it: the forward pass keeps the activations of a few layers only, and the backward pass recomputes the others. For $$n$$ layers, it needs $$O(\sqrt{n})$$ memory, for the cost of one extra forward pass ([Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), Chen et al., 2016).

The figure shows ZeRO-3 as the ZeRO paper describes it: each process owns the states of one layer, broadcasts its parameters before the layer runs, and receives the reduced gradients after its backward pass:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/workflow_sharding.svg"/>

{: .caption}
Workflow for stage 3 sharding. Each processor contains a non-overlapping subset of parameters, gradients and optimizer data, split by assigning different layers to different processes. Each processor loads a different data batch. During the forward and backward passes (represented by arrows on the left and right of the model, respectively), when a given layer needs to be computed, the process responsible for that layer broadcasts and gathers those values to/from the remaining processes. Example for processor 1 (yellow): **Data loading:** the yellow process loads data sample 1. **Forward pass:** the yellow process receives the parameters from rank 0 (blue) and computes the activations for layer 1. Afterwards, the yellow process broadcasts its parameters to ranks 0, 2 and 3, so that they compute their activations for layer 2. Activations for layers 3 and 4 are computed similarly to layer 1, led by the red and green processes, respectively. **Backward pass:** the green process (3) broadcasts its parameters to all other processes. Each process can use its activations and the received parameters to compute the gradients for the top layer. All processes gather their local gradients in process 3, which uses them to update the parameters of the last layer. The same happens for the remaining layers, with the red, yellow and blue processes (iteratively) broadcasting the parameters and gathering the gradients.

ZeRO-3 in DeepSpeed and PyTorch FSDP split the tensors of every layer into $$P$$ equal shards instead, and each process owns one shard of every layer. An all-gather replaces each broadcast, and a reduce-scatter each reduction, with the same communication volume.

A reduce-scatter and an all-gather move as much data as an all-reduce, so stages 1 and 2 have the communication volume of DDP. Stage 3 adds the all-gather of the backward pass, and communicates 1.5 times as much. The [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) summarizes the memory per GPU of each stage:

{: .picture}
<img width="80%" height="80%" src="{{ site.assets }}/distributed-training/imgs/deepspeed_stages.svg"/>

{: .caption}
Memory consumption of the three stages of ZeRO FSDP, where $$\Psi$$ is the number of parameters, $$K$$ the memory multiplier of the optimizer states and $$N_d$$ the data-parallel degree. Adapted from: [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

With mixed precision and Adam, each parameter takes 2 bytes in fp16, its gradient 2 bytes, and its optimizer states $$K = 12$$ bytes. The data-parallel degree $$N_d$$ is the number of processes $$P$$. For 7.5 billion parameters on 64 GPUs, ZeRO-3 reduces the model states per GPU from 120 GB to 1.9 GB. DDP communicates the least, and suits models whose states fit in one GPU. ZeRO-1 and ZeRO-2 save memory for the same communication volume. ZeRO-3 trains models whose fp16 parameters alone do not fit in one GPU, for 1.5 times the communication.

## CPU Offloading

Some models do not fit in GPU memory even with sharding. *Offloading* moves the data that the GPU does not need at the moment to CPU memory, and copies it back before its next use. [vDNN](https://arxiv.org/pdf/1602.08124.pdf) (Rhu et al., 2016) offloads the activations of convolutional networks, which take an order of magnitude more memory than their weights, and [vDNN+](https://ieeexplore.ieee.org/document/8820980) (Shriram et al., 2019) extends it.

Offloading works layer by layer, because the network is a composition of layers. For the mean absolute error of a batch of $$B$$ samples:

$$
\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \left\vert y_b - f^{(L+1)} \circ \dots \circ f^{(2)} \circ f^{(1)} (x_b) \right\vert
$$

where $$f^{(l)}$$ is the function of layer $$l$$, $$L$$ is the number of hidden layers, and $$y_b$$ is the label of the input $$x_b$$. The forward pass of a layer needs only its weights and the output of the previous layer. Its backward pass needs only its weights, its input and the gradient from the next layer. vDNN copies the input of each layer to the CPU during the forward pass, and back to the GPU during the backward pass, before the layer needs it. Two CUDA streams overlap these transfers with the computation:

{: .picture}
<img width="55%" height="55%" src="{{ site.assets }}/distributed-training/imgs/vdnn2.svg"/>

{: .caption}
The forward pass of the vDNN(+) implementation on convolutional neural networks. Red arrows represent the data flow of variables $$x$$ and $$y$$ (layer input and output) during forward propagation. Green arrows represent weight variables. Yellow arrows represent the *variables workspace in cuDNN*, needed in certain convolutional algorithms. Data not associated with the current layer being processed (layer N) is marked with a black cross and can safely be removed from the GPU's memory. Adapted from: <a href="https://arxiv.org/pdf/1602.08124.pdf">vDNN (Rhu et al.)</a>

The backward pass also propagates the gradients down the network:

{: .picture}
<img width="55%" height="55%" src="{{ site.assets }}/distributed-training/imgs/vdnn3.svg"/>

{: .caption}
The back propagation phase of the vDNN(+) implementation on convolutional neural networks. Red arrows represent the data flow of variables $$x$$ and $$y$$ (layer input and output) during forward propagation. Blue arrows represent the data flow during backward propagation. Green arrows represent weight variables. Yellow arrows represent the *variables workspace in cuDNN*, needed in certain convolutional algorithms. Data not associated with the current layer being processed (layer 2) is marked with a black cross and can safely be removed from the GPU's memory. Adapted from: <a href="https://arxiv.org/pdf/1602.08124.pdf">vDNN (Rhu et al.)</a>

## Model and Dataset

The examples use GPTlite with the dimensions of GPT-3 Small in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Table 2.1):

```python
n_layer = 12   # depth of the network as number of decoder blocks.
n_embd = 768   # size of the embeddings (d_model)
n_head = 12   # number of attention heads in the Multi-Attention mechanism
block_size = 2048   # block size, i.e. the maximum sequence length, the n_ctx in the paper
dropout = 0.1   # dropout rate (variable p) for dropout units
```

`get_dataset()` returns the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset, whose samples are windows of `block_size` tokens at random offsets, with the same windows shifted by one token as targets. `get_model()` returns the model:

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

  train_data, valid_data, vocab_size = load_tiny_shakespeare_data()
  train_dataset = GPTliteDataset(train_data, block_size)
  valid_dataset = GPTliteDataset(valid_data, block_size)
  return train_dataset, valid_dataset, vocab_size


def get_model(vocab_size):
  return GPTlite(vocab_size)
```

With a vocabulary of 65 characters, GPTlite has 86.7 million parameters. GPT-3 Small has 125 million, because of its vocabulary of 50257 tokens.

### Using a torchvision Model {#torchvision-model}

The code runs any model of type `torch.nn.Module` on any dataset of type `torch.utils.data.Dataset`. For example, these two functions return the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) images and the [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network)-18 classifier of `torchvision`:

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

## PyTorch Implementation

The launcher [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html), described below, sets environment variables that identify each process in the network and its GPU in the node:

```python
import os
rank = int(os.environ['RANK'])   # the unique id across all processes in all nodes
local_rank = int(os.environ['LOCAL_RANK'])   # the unique id across this node
world_size = int(os.environ['WORLD_SIZE'])   # the number of processes across all nodes
```

Each process joins the process group with `torch.distributed.init_process_group()`. A [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) iterates over the samples of each process:

```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)
```

Its `sampler`, a `DistributedSampler`, assigns different samples to each process. Without it, all processes would load the same samples:

```python 
sampler = torch.utils.data.distributed.DistributedSampler(
  dataset, num_replicas=world_size, rank=rank)
```

Each process moves the model to its GPU, in [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format):

```python
device = f"cuda:{local_rank}"
dtype = torch.bfloat16
model = model.to(device=device, dtype=dtype)
```

and wraps it in [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for DDP:

```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

or in [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html) for sharding. The strategy `SHARD_GRAD_OP` splits the gradients and the optimizer states, as ZeRO-2, and the default strategy, `FULL_SHARD`, also splits the parameters, as ZeRO-3:

```python
model = torch.distributed.fsdp.FullyShardedDataParallel(
    model,
    device_id=local_rank,
    sharding_strategy=torch.distributed.fsdp.api.ShardingStrategy.SHARD_GRAD_OP,  # sharded grads + optimizer states (ZeRO-2-like)
)
```

PyTorch has deprecated this API in favor of `fully_shard` (FSDP2), which splits each parameter along its first dimension ([FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)).

The training loop is the same as on one GPU, since the wrappers run the communication inside the forward and backward passes:

```python
for step, data in enumerate(dataloader):
  inputs = data[0].to(device)
  labels = data[1].to(device)
  outputs = model(inputs)  # fwd pass
  loss = torch.nn.functional.cross_entropy(outputs, labels) # loss
  loss.backward() # compute gradients
  optimizer.step() # update parameters
  optimizer.zero_grad(set_to_none=True)
```  

DDP overlaps the communication with the backward pass. It groups the gradients into buckets, in about the reverse order of the parameters, and starts an asynchronous all-reduce of a bucket as soon as all its gradients are ready, while the backward pass computes the gradients of the lower layers. At the end of the backward pass, it waits for all the all-reduces ([DDP design notes](https://pytorch.org/docs/master/notes/ddp.html#internal-design); [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704), Li et al., 2020). PyTorch can also offload the saved activations to the CPU, with [hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).

`torchrun` starts `nproc_per_node` processes on the node where it runs, one per GPU, and sets the environment variables above. A run on several nodes starts `torchrun` on each node, with the same rendezvous arguments. On one node:

```shell
$ torchrun --standalone --nproc_per_node=4 train.py
```

## DeepSpeed Implementation

A JSON [configuration file](https://www.deepspeed.ai/docs/config-json/) enables the DeepSpeed features: parallelism, precision, logging, communication and more. This config sets a batch of 256 samples, a progress report every 10 steps (`steps_per_print`), and, optionally, the optimizer and the learning-rate scheduler:

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

`train_micro_batch_size_per_gpu` is the batch of one process in one forward and backward pass, and `train_batch_size` is the batch of one optimizer step, over all processes and gradient accumulation steps:

```
train_batch_size = train_micro_batch_size_per_gpu * num_gpus * num_nodes * gradient_accumulation_steps
```

Setting two of the three values sets the third. *Gradient accumulation* sums the gradients of several micro-batches before each optimizer step, to train with a large batch in the memory of a small one. The configs of the benchmark set micro-batches of 1 sample, so each of the 8 GPUs runs 32 micro-batches per optimizer step.

The `stage` field enables ZeRO. Without it, or with stage 0, DeepSpeed runs DDP:

```json
{
  "zero_optimization": { "stage": 3 }
}
```

[ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) ([ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840), Ren et al., 2021) offloads the model states instead of the activations: it keeps the optimizer states and the gradients in CPU memory, and runs the optimizer step on the CPU. [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) (Rajbhandari et al., 2021) requires stage 3, and also offloads the parameters, to CPU memory or to [NVMe](https://en.wikipedia.org/wiki/NVM_Express) storage. According to the [ZeRO-3 documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html#zero), it also uses the bandwidth better, and overlaps more of the communication with the computation. This config offloads the optimizer states and the parameters to the CPU:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true }
  }
}
```

DeepSpeed handles the process group, the data loading and the distributed sampler, so the setup is short:

```python
def main_deepspeed(n_epochs=100, random_seed=42):

  torch.manual_seed(random_seed)  #set random seed (used by DataLoader)
  deepspeed.runtime.utils.set_random_seed(random_seed) #set DeepSpeed seed
  deepspeed.init_distributed()  # initialize distributed DeepSpeed
  config = 'ds_config.json'  # load the DeepSpeed config
  criterion = torch.nn.CrossEntropyLoss()  # initialize loss function
  train_dataset, _, vocab_size = gptlite.get_dataset()  # initialize GPTlite dataset
  model = gptlite.get_model(vocab_size)  # initialize GPTlite model

  engine, optimizer, train_dataloader , _ = deepspeed.initialize(
    config=config, model=model, training_data=train_dataset,) # initialize deepspeed
```

`initialize()` returns a data loader that gives each process a different subset of the samples. The training loop is the same as in PyTorch, except that `engine.step()` also zeroes the gradients:

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

The DeepSpeed launcher, or `torchrun`, starts the run:

```shell
$ deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
```

<!--
[**Mixed precision representation**](https://arxiv.org/abs/1710.03740) allows for calculus with value types (parameters, activations, accumulators) stored with different numerical representations, leading to a reduction of memory and compute time. It can be enabledd by adding the `fp16` entry [in the config](https://www.deepspeed.ai/docs/config-json/#fp16-training-options). As a side note, the `amp` config entry also enables mixed precision training that follows the [NVIDIA Apex](https://nvidia.github.io/apex/) implementation i.e. with the `O0` to `O3` opimization levels. However, [it is not compatible with ZeRO](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options), therefore we won't use it. The [`fp16` is equivalent to APEX optimization level O2](https://www.deepspeed.ai/docs/config-json/#fp16-training-options), and according to the [documentation](https://www.deepspeed.ai/docs/config-json/#fp16-training-options), "if you want to use ZeRO (currently) you must use this mode". We can enable it with the entry `"fp16: { enabled: true }` that is equivalent to the following default values:

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

However, if your hardware supports bfloat16 (brain floating point), this should be used in lieu of float16, as it has a longer integer (exponent) representation: 8 bits instead of the 5 in float16, i.e. the same 8 bits as in float32. This makes it more numerically stable and does not require loss scaling. bfloat16 can be activated by adding to the config `bf16 { "enabled": true }`.

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

### Memory Estimates

The DeepSpeed API [estimates the memory of the model states](https://deepspeed.readthedocs.io/en/latest/memory.html#api-to-estimate-memory-usage) of ZeRO-2 and ZeRO-3, with and without offloading. This function prints the estimates at the start of the run:

```python
def measure_parameters_memory(model):
  param_size_GB = sum([p.nelement() * p.element_size() for p in model.parameters()])/1024**3
  print(f"Native model parameters size: {round(param_size_GB, 2)}GB.")

  from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
  estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)

  from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
  estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
```

For GPTlite:
- the parameters take 0.323 GB in fp32;
- ZeRO-2 needs 0.161 GB per GPU with offloading and 0.484 GB without;
- ZeRO-3 needs 0.009 GB per GPU with offloading and 0.190 GB without.

ZeRO-3 with offloading keeps only the largest layer on the GPU, here a linear layer of the FFN with 2.36 million parameters, at 4 bytes per parameter. The estimates are cheap, but they cover only the model states, in mixed precision: they ignore the activations, the temporary and communication buffers, the batch size and the precision of the config.

## Results

The DeepSpeed engine logs the throughput and the allocated GPU memory every `steps_per_print` steps. All runs use `pytorch==2.0.1`, CUDA `11.7` and `deepspeed==0.10.3`, in fp32 (`fp16` is disabled in the configs), with the same configuration except for the ZeRO stage and the offloading:

1. DDP: <a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> with stage 0;
2. ZeRO-1, ZeRO-2 and ZeRO-3: <a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> with stage 1, 2 or 3;
3. ZeRO-3 with CPU offloading of the parameters and the optimizer states: <a href="{{ site.assets }}/distributed-training/ds_config_offload.json">`ds_config_offload.json`</a>.

With micro-batches of 1 sample per GPU, GPTlite gives:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/benchmark_gptlite.svg"/>

**Throughput.** DeepSpeed reduces the gradients of DDP and ZeRO-1 once per optimizer step, after the last micro-batch. It reduces those of ZeRO-2 and ZeRO-3 after every micro-batch, since each process keeps only the gradients of its partition, and ZeRO-3 also gathers the parameters for every micro-batch. With 32 micro-batches per step, ZeRO-2 thus communicates about 16 times the volume of DDP per step, and ZeRO-3 at least 32 times. ZeRO-1 loses 3% of the throughput of DDP, and ZeRO-2, which overlaps the reductions with the backward pass (`overlap_comm`), loses 9%. ZeRO-3, which must gather the parameters of a layer before it computes the layer, loses 38%, and the CPU offloading another 13%.

**Memory.** From DDP to ZeRO-3, the average allocated memory drops from 3.24 GB to 2.55 GB, and the peak from 10.16 GB to 9.37 GB. Offloading lowers them to 2.35 GB and 9.16 GB. In fp32 with Adam, each parameter takes 16 bytes of model states (parameter, gradient, momentum and variance): 1.3 GB for GPTlite, of which ZeRO-3 on 8 GPUs removes at most 7/8, or 1.1 GB. Most of the 10 GB peak holds the activations of the 2048-token samples and temporary buffers, which ZeRO does not split. The ZeRO paper trains models of over 100 billion parameters, whose model states dominate the memory, and reports a superlinear speedup from 64 to 400 GPUs, as the freed memory allows larger batches per GPU.

**Communication buffers.** ZeRO communicates in buckets, the buffers of one collective call. The configs set buckets of 5e4 elements (`allgather_bucket_size`, `reduce_bucket_size` and `stage3_prefetch_bucket_size`). With the DeepSpeed default of 5e8 elements, the memory of stages 2 and 3 about doubled, and some runs ran out of memory. Smaller buckets need less memory, but more collective calls.

**Communication quantization.** ZeRO++ ([ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209), Wang et al., 2023) quantizes the communicated weights to INT8 and gradients to INT4, and keeps a copy of the weights in each node, which cuts the communication volume of ZeRO by 4 times. It targets clusters with a low bandwidth, or a scale that forces small batches per GPU. On this single node it gave no speedup, so the runs above do not use it.

## Further Reading

The [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest), the [training features](https://www.deepspeed.ai/training/#features), the [tutorials](https://www.deepspeed.ai/tutorials/), the [Hugging Face guide to DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) and [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/) cover DeepSpeed in more detail. The code of all experiments is in the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training).
