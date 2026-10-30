---
layout: post
title:  "Data parallelism, sharding and CPU offloading"
categories: ["Distributed Training"]
tags: ["data parallelism", "ZeRO", "FSDP", "CPU offloading", "mixed precision", "ZeRO++", "activation checkpointing", "DeepSpeed", "PyTorch"]
description: "Data parallelism (DDP), sharding (ZeRO and FSDP) and CPU offloading: what each GPU stores and sends, and how to train a GPT model with them in PyTorch and DeepSpeed."
---

Training a model of 7.5 billion parameters with the [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) optimizer in [mixed precision](https://en.wikipedia.org/wiki/Mixed-precision_arithmetic) takes 16 bytes per parameter for its weights, gradients and optimizer states. For the whole model, that is 120 GB before a single activation is stored, more than the 80 GB of memory of an A100 GPU. Training on many GPUs helps in two ways: each GPU processes a part of the data, so a training step takes less time, and, with sharding, each GPU stores only a part of the model.

In 2012, the paper [Large Scale Distributed Deep Networks](https://papers.nips.cc/paper/2012/hash/6aca97005c68f1206823815f66102863-Abstract.html) trained neural networks with billions of parameters on tens of thousands of CPU cores, by running copies (*replicas*) of the model on different subsets of the training data. The replicas sent their updates to a *parameter server* split across many machines. Modern frameworks such as PyTorch exchange the gradients directly between the processes instead, with a collective communication at every training step.

This post explains how data parallelism works, how sharding (ZeRO and FSDP) removes the redundant copies of the model that data parallelism keeps on every GPU, and how CPU offloading moves part of what is left to the CPU memory. It applies them to the training of the [GPTlite model built in the previous post]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}), on a network of 8 GPUs, with [PyTorch DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (Distributed Data Parallel) and [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) (the Zero Redundancy Optimizer of DeepSpeed, a library built on PyTorch). The DeepSpeed implementation also covers mixed precision, gradient accumulation, communication quantization (ZeRO++) and activation checkpointing.

## Data parallelism

[Data parallelism](https://en.wikipedia.org/wiki/Data_parallelism) refers to the parallel execution of different input samples across processors. Consider any data input of shape $$B \times S \times E$$ (batch size, sequence/temporal size and embedding/feature size): data parallelism refers to the use case where the data is split across $$P$$ processes along the batch dimension, leading to a local input of shape $$B/P \times S \times E$$:

{: .picture}
<img width="24.2%" height="24.2%" src="{{ site.assets }}/distributed-training/imgs/data_parallelism.svg"/>

{: .caption}
An illustration of the DDP data layout, split across 4 processes color-coded as blue, yellow, red and green.

There are two main data parallelism approaches, which differ in what each process stores: a full copy of the model, or only a part of it.

### Distributed Data Parallel (DDP)

**Distributed Data Parallel (DDP)** keeps a full copy of the model (weights, optimizer states and gradients) on each processor. All copies are initialized equally. Each processor takes a different mini-batch as input and performs a forward pass to compute the loss of that batch.

On the backward pass, at every layer of the model, all processes compute the gradients of their batch and perform an [all-reduce](https://en.wikipedia.org/wiki/Collective_operation) to get the mean gradients across all processors. These are then used to update the parameters and the optimizer states. Because parameters are initialized equally, gradients are mean-reduced and all processes perform the same updates, the models across all processors are kept in an identical state throughout the execution. The following diagram illustrates this workflow:

{: .picture}
<img width="89.5%" height="89.5%" src="{{ site.assets }}/distributed-training/imgs/workflow_data_parallelism.svg"/>

{: .caption}
Distributed data parallelism (DDP) on 4 processes. Each process holds a full copy of a 4-layer model (the parameters, optimizer states and gradients of every layer), initialized identically, and takes its own input. The forward pass (left arrow) computes the activations and the loss of each process. In the backward pass (right arrow), each process computes the local gradients of every layer, and an all-reduce averages them across the processes. The optimizer then applies the same mean gradients on every process, so all copies stay identical.

The backward pass **overlaps computation and communication**: it computes the gradients of some layers while the gradients of other layers are sent across the network, which *hides* the communication latency behind the computation. This is possible because the gradients of a layer are final as soon as the backward pass leaves that layer, while the layers before it are still being computed.

The diagram above shows that each processor holds a copy of the model, so the **model states** (the parameters, gradients and optimizer states) are stored $$P$$ times, which leads to superfluous memory usage. That is where sharding comes into play.

### Sharding with FSDP and ZeRO

**Fully Sharded Data Parallel (FSDP)**, a.k.a. **sharding**, is a distributed setup where processors do not hold a full copy of the model, but only a part (a *shard*) of its parameters, optimizer states and gradients, e.g. those of a subset of layers. As before, each processor takes a different mini-batch as input.

In DeepSpeed terminology, sharding goes by the name of **ZeRO (Zero Redundancy Optimizer)**, introduced in the paper [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054). ZeRO has several alternative execution modes (**stages**). Each stage represents a different level of memory redundancy, corresponding to different variable types being distributed or replicated across processes:
- **ZeRO stage 0**: no sharding of any variables, equivalent to DDP.
- **ZeRO stage 1 (ZeRO-1)**: the optimizer states (e.g., for Adam in mixed precision, an FP32 copy of the weights and the first and second moment estimates) are partitioned across the processes, so that each process updates only its part of the parameters. The forward pass does not change.
- **ZeRO stage 2 (ZeRO-2)**: the gradients used to update the model weights are also partitioned, so that each process retains only the gradients corresponding to its portion of the optimizer states. The forward pass does not change either.
- **ZeRO stage 3 (ZeRO-3)**: the model parameters are also partitioned across the processes. Adds a communication overhead to both the forward and backward passes.

An important remark is that activations are not sharded, i.e. they are kept in *full shape* on each processor. In modern models, **a huge chunk of memory is allocated to the residual states, as the ZeRO paper calls them (activations, temporary buffers, fragmented memory, buffers of normalization layers, etc.), which FSDP does not shard**. With that in mind, the following diagram illustrates the workflow of stage 3 sharding (of parameters, gradients and optimizer states):

{: .picture}
<img width="92.2%" height="92.2%" src="{{ site.assets }}/distributed-training/imgs/workflow_sharding.svg"/>

{: .caption}
Stage 3 sharding (ZeRO-3) on 4 processes. Process k holds the parameters, optimizer states and gradients of layer k only (solid boxes; the dashed boxes are layers held by other processes), takes its own input and keeps its own activations. Forward pass (left): before layer k is computed, process k broadcasts its parameters, and the other processes discard them after use. Backward pass (right): process k broadcasts the parameters of layer k again, every process computes its local gradients of layer k, and these gradients are summed (reduced) on process k, which updates the parameters of layer k. Each pass overlaps the computation of a layer with the communication of its neighbours: the all-gather of the parameters of the next layer to compute and, in the backward pass, the reduce-scatter of the gradients of the layer computed before. Inspired by: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

Instead of a single all-reduce of the gradients, sharding uses two collective operations. The **all-gather** temporarily brings together the shards of the parameters, so that each process rebuilds the full weights of the layer it computes, in the forward and in the backward pass. The **reduce-scatter**, in the backward pass, sums the gradients across all processes and scatters the result, so that each process only stores and updates the gradient shard of its optimizer states. Note that DeepSpeed ZeRO-3 and [PyTorch FSDP](https://arxiv.org/abs/2304.11277) split the parameters of every layer into $$P$$ shards, one per process, while the diagram assigns whole layers to processes: there, the all-gather of a layer is a broadcast from the process that holds it, and the reduce-scatter is a reduction on that process.

### Memory and communication of the ZeRO stages

The higher the stage, the less memory is consumed, and the memory of each stage follows from counting bytes. With mixed-precision Adam, each parameter takes 2 bytes of FP16 weights, 2 bytes of FP16 gradients and $$K = 12$$ bytes of optimizer states (an FP32 copy of the weight and the two moments of Adam), so a model of $$\Psi$$ parameters needs $$(2 + 2 + K)\Psi = 16\Psi$$ bytes on every GPU. Each ZeRO stage divides one more of these terms by the number of processes $$P$$, called the data-parallel degree $$N_d$$ in the picture below. For $$\Psi = 7.5$$ billion parameters and $$N_d = 64$$ GPUs, a GPU holds 120 GB with plain data parallelism, 31.4 GB with ZeRO-1, 16.6 GB with ZeRO-2 and 1.9 GB with ZeRO-3.

The savings are not free: stages 1 and 2 communicate as much as regular data parallelism, $$2\Psi$$ elements per process and training step. An all-reduce of the gradients is a reduce-scatter followed by an all-gather, as in a ring all-reduce, each of which moves about $$\Psi$$ elements per process; ZeRO-1 and ZeRO-2 run the same two collectives, on the gradients and then on the updated parameters (ZeRO paper, section 7). Stage 3 communicates 1.5 times as much, $$3\Psi$$, as it gathers the parameters in both the forward and the backward passes. These memory improvements are summarized in the [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) as:

{: .picture}
<img width="92.4%" height="92.4%" src="{{ site.assets }}/distributed-training/imgs/deepspeed_stages.svg"/>

{: .caption}
Per-GPU memory of the model states (parameters, gradients and optimizer states) with plain data parallelism (baseline) and with the three ZeRO stages: $$P_{os}$$ partitions the optimizer states across the GPUs, $$P_{os+g}$$ also the gradients, and $$P_{os+g+p}$$ also the parameters. $$\Psi$$ is the number of parameters, $$K$$ the memory multiplier of the optimizer states and $$N_d$$ the data-parallel degree; the example uses $$K=12$$ (mixed-precision Adam), $$\Psi=7.5B$$ and $$N_d=64$$. The last column is the communication volume relative to the baseline. Adapted from: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) and the [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

## CPU offloading

Sometimes the model is so big that, even with sharding, it does not fit in the memory of the GPUs. A common technique to handle such memory limitations is **CPU offloading**, also referred to as virtualized Deep Neural Networks by [vDNN (Rhu et al.)](https://arxiv.org/pdf/1602.08124.pdf) and [vDNN+ (Shriram et al.)](https://ieeexplore.ieee.org/document/8820980).

The main goal of this method is to iteratively move to the GPU only the portions of activations and model that are required for the current and next computation steps. Previously processed layers are moved from GPU to CPU, while upcoming layers are moved from CPU to GPU. vDNN offloads only the activations (the feature maps), as they take most of the GPU memory of convolutional neural networks.

This is possible because the loss (and its derivative) can be written as a composition of the functions of the layers, e.g. for the [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE):

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N | y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) |
$$

where $$f^{(l)}$$ is the function implemented by layer $$l$$, $$L$$ is the number of hidden layers, $$y$$ is the ground truth, $$x_n^{(0)}$$ is the input of sample $$n$$ and $$N$$ is the number of samples.

The important concept here is the **composition** of the functions $$f$$ across layers. The computation of a layer needs only its weights, its input and its output (and, in the backward pass, the gradient coming from the next layer), so the data of all other layers can be kept in CPU memory.

To reduce the time spent waiting to push (pull) a layer to (from) the GPU, the computation of the current layers can be overlapped with the communication of the upcoming ones. This overlap matters, as the link between the CPU and the GPU is much slower than the memory of the GPU: a [PCI Express](https://en.wikipedia.org/wiki/PCI_Express) 4.0 x16 link moves about 32 GB/s, while the memory of an A100 GPU of 80 GB moves about 2 TB/s. The forward pass can be illustrated as:

{: .picture}
<img width="88.9%" height="88.9%" src="{{ site.assets }}/distributed-training/imgs/vdnn2.svg"/>

{: .caption}
The forward pass of vDNN on a convolutional neural network, at layer N. Blue arrows are the input and output feature maps $$X$$ and $$Y$$ of the current layer, the purple arrow its weights $$W$$ and the yellow arrow its cuDNN workspace $$WS$$. White arrows are data that the current layer does not use. The input feature maps $$X$$ of the preceding layers, marked with a red cross, are not used by the current layer and can be released from GPU memory. Adapted from: [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124)


The backward propagation phase is trickier, as it also includes the gradients being propagated down the model:

{: .picture}
<img width="88.9%" height="88.9%" src="{{ site.assets }}/distributed-training/imgs/vdnn3.svg"/>

{: .caption}
The backward pass of vDNN at layer 2. Blue arrows are the feature maps $$X$$ and $$Y$$ of the current layer, green arrows its gradients $$dX$$ and $$dY$$, the purple arrow its weights $$W$$ and the yellow arrow its cuDNN workspace $$WS$$. White arrows are data that the current layer does not use. The data used only by the layers whose backward pass is complete (the layers after layer 2 and the loss function), marked with a red cross, is not reused for this batch and can be released from GPU memory. Adapted from: [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124)

## Model and dataset setup

The implementation starts by taking the previous *GPTlite* model, with specs matching the *GPT-3 Small* model in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Table 2.1):

```python
n_layer = 12   # depth of the network as number of decoder blocks.
n_embd = 768   # size of the embeddings (d_model)
n_head = 12   # number of attention heads in the Multi-Attention mechanism
block_size = 2048   # block size, i.e. the maximum sequence length, the n_ctx in the paper
dropout = 0.1   # dropout rate (variable p) for dropout units
```

It then defines the methods `get_model()` and `get_dataset()` that return the model and the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset:

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

### Detour: using a pre-existing model {#torchvision-model}

Note that this code is applicable to any model of type `torch.nn.Module` and any dataset of type `torch.utils.data.Dataset`. As an example, to perform multi-class classification using the [`ResNet`](https://en.wikipedia.org/wiki/Residual_neural_network) network on the [`CIFAR10`](https://en.wikipedia.org/wiki/CIFAR-10) dataset available in `torchvision`, the previous 2 methods could be defined as:

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


## PyTorch implementation

This section implements data parallelism in PyTorch. First, the code collects the environment variables that are set by the [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) launcher (detailed below), as they uniquely identify each process in the network (its **rank**, from 0 to the **world size**, the number of processes, minus 1) and its GPU device within a node (its local rank):

```python
import os
rank = int(os.environ['RANK'])   # the unique id across all processes in all nodes
local_rank = int(os.environ['LOCAL_RANK'])   # the unique id across this node
world_size = int(os.environ['WORLD_SIZE'])   # the number of processes across all nodes
```

Now the code defines the [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) that tells each process how to iterate through the data:

```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)
```

Note the argument `sampler`, a `torch.utils.data.distributed.DistributedSampler` that assigns different samples of the dataset to different processes. Without it, all processes would load exactly the same datapoints in every iteration. The sampler is created as:

```python 
sampler = torch.utils.data.distributed.DistributedSampler(
  dataset, num_replicas=world_size, rank=rank)
```

Each model is then placed on a different GPU, with the correct data type (here [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)):

```python
device = f"cuda:{local_rank}"
dtype = torch.bfloat16
model = model.to(device=device, dtype=dtype)
```

and finally wrapped with [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for the DDP implementation:

```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

or with the [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html) wrapper for the sharded implementation:

```python
model = torch.distributed.fsdp.FullyShardedDataParallel(
    model,
    device_id=local_rank,
    sharding_strategy=torch.distributed.fsdp.api.ShardingStrategy.SHARD_GRAD_OP,  # sharded grads + optimizer states (ZeRO-2-like)
)
```

The argument `sharding_strategy` plays the role of the ZeRO stage: `NO_SHARD` replicates all variables as DDP does, `SHARD_GRAD_OP` shards the gradients and optimizer states similarly to ZeRO-2, and `FULL_SHARD` (the default) also shards the parameters, as ZeRO-3.

And that is all. The training loop is then written as usual, and PyTorch does all the communication and synchronization under the hood:

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

The [PyTorch DDP documentation](https://pytorch.org/docs/master/notes/ddp.html#internal-design) and the [PyTorch DDP paper](https://arxiv.org/abs/2006.15704) describe how DDP overlaps the communication of the gradients with the backward pass. The backward pass computes the gradients from the last to the first layer, and DDP collects them into blocks (*buckets*), of 25 MiB by default (the argument `bucket_cap_mb`). Each bucket is mean-reduced with an asynchronous all-reduce as soon as all its gradients are computed, while the computation of the backward pass proceeds. At the end of the backward pass, all GPUs wait for all gradient all-reduces to finish, and then the optimizer updates the parameters.

For extra memory savings, tensors can also be offloaded in PyTorch by using custom [hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).

Finally, the application is launched by calling [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html). Torchrun is a network bootstrapper that, started on each compute node in a network, spawns one process of a Python script per GPU and sets the environment variables mentioned above, which uniquely identify processes across the network. The usage is simple:

```shell
$ torchrun --standalone --nproc_per_node=4 train.py
```

where `nproc_per_node` is the number of GPUs on each node.

## DeepSpeed implementation


Porting existing code to DeepSpeed is pretty simple. To start, DeepSpeed features can be activated via the DeepSpeed API or its [Configuration JSON](https://www.deepspeed.ai/docs/config-json/). The number of possible optimizations is large, as the config can define parallelism, floating-point precision, logging, communication parameters, etc.

This implementation starts with a simple config file. The file configures the DeepSpeed logger to output memory and throughput info every 10 steps (`steps_per_print`), sets the batch size to `256` and (optionally) defines the settings of the optimizer (`optimizer`) and learning rate scheduler (`scheduler`):

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

Note that, in DeepSpeed terminology, `train_micro_batch_size_per_gpu` refers to the batch size loaded per dataloader (i.e. per GPU, per gradient accumulation step), while `train_batch_size` refers to the batch size across all gradient accumulation steps and processes in the network, i.e.:

```
train_batch_size = train_micro_batch_size_per_gpu * num_gpus * num_nodes * gradient_accumulation_steps
```

Therefore, as `train_batch_size` is already set, **gradient accumulation** can be enabled by setting `train_micro_batch_size_per_gpu`, or `gradient_accumulation_steps` (1 by default). Gradient accumulation, also known as micro-batching, simulates a large mini-batch as an iteration across several micro-batches: each micro-batch runs a forward and a backward pass, its gradients are added to those of the previous ones, and the optimizer updates the parameters once, after the last micro-batch. It is particularly relevant when the whole mini-batch does not fit into memory.

The configs of this post use a micro-batch of 1 sample per GPU, which accumulates up to the batch of 256 samples across the 8 GPUs in $$256 / (1 \times 8) = 32$$ gradient accumulation steps, the value that `engine.gradient_accumulation_steps()` returns at runtime:

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 1
}
```

Sharding with **ZeRO** is enabled by setting its stage in the config file. If omitted, or when passing stage 0, ZeRO is disabled and the execution follows a regular distributed data parallel workflow:

```json
{
  "zero_optimization": { "stage": 3 }
}
```

CPU offloading in DeepSpeed is called [**ZeRO-Offload**](https://www.deepspeed.ai/tutorials/zero-offload/), introduced in the [ZeRO-Offload paper](https://arxiv.org/abs/2101.06840): it offloads the optimizer states and the gradients to the CPU memory, and runs the optimizer step on the CPU. On top of stage 3, [**ZeRO-Infinity**](https://arxiv.org/abs/2104.07857), also an offloading engine, extends ZeRO-Offload with the offloading of the parameters and with support for [NVMe](https://en.wikipedia.org/wiki/NVM_Express) memory. According to the [ZeRO-3 documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html#zero), "ZeRO-Infinity has all of the savings of ZeRO-Offload, plus is able to offload more the model weights and has more effective bandwidth utilization and overlapping of computation and communication". They are enabled with the entries `offload_optimizer` and `offload_param`, where `pin_memory` places the offloaded data in **pinned** (page-locked) CPU memory, from which the GPU copies data faster and asynchronously:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true }
  }
}
```

Almost everything is ready now. Once the config file is properly calibrated, the implementation is straightforward. All boilerplate that PyTorch requires for parallelism and data loaders is managed internally by DeepSpeed, so the code only needs to set up DeepSpeed as:

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

The training loop is then written with a structure very similar to the PyTorch implementation. The only exception is that the gradients are not zeroed, as this is managed internally by DeepSpeed. Also, `initialize()` already returns a `train_dataloader` that assigns disjoint subsets of data to each process, so there is no need to fiddle with distributed dataloaders and samplers.

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

Finally, the run is launched with the `torchrun` launcher as before, or with the launcher included in DeepSpeed as:

```shell
$ deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
```

The config file also sets the numerical precision, the communication buffers and the communication quantization of ZeRO, and DeepSpeed provides activation checkpointing. The following sections cover these options, which save memory or communication.

### Mixed precision

[**Mixed precision training**](https://arxiv.org/abs/1710.03740) stores the parameters, activations and gradients in a 16-bit floating-point format, and keeps an FP32 copy of the weights for the optimizer step, which is part of the $$K = 12$$ bytes of optimizer states counted above. It reduces the memory of the activations and the compute time. In DeepSpeed, it is enabled by the `fp16` entry of the [config](https://www.deepspeed.ai/docs/config-json/#fp16-training-options).

To keep small gradients from underflowing in FP16, the loss is multiplied by a scale factor before the backward pass (**loss scaling**), and the gradients are divided by it before the optimizer step. With `"loss_scale": 0`, DeepSpeed adjusts this factor dynamically, starting from $$2^{16}$$ (`initial_scale_power`). The entry `"fp16": {"enabled": true}` is equivalent to the following default values:

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

If the hardware supports bfloat16 (brain floating point, BF16), it should be used instead of FP16: it has 8 exponent bits, the same as FP32, instead of the 5 of FP16. This gives BF16 the range of FP32, which makes it more numerically stable and removes the need for loss scaling. BF16 is enabled with `"bf16": {"enabled": true}`, which cannot be combined with the `fp16` mode.

### Communication buffers and overlap

**Limiting the size of the communication buffers** is important when ZeRO is enabled. ZeRO spreads the gradients, and in stage 3 the parameters, across the processes, and each collective that gathers or reduces them needs buffers for the data that it sends and receives. By default, DeepSpeed communicates up to `5e8` elements at a time (`reduce_bucket_size`, `allgather_bucket_size` and `stage3_prefetch_bucket_size`), i.e. buffers of 1 GB for 16-bit values. Smaller buckets split each collective into several smaller ones, which need less memory but use the network less efficiently.

The config can also enable **communication overlap** (`overlap_comm`), which overlaps the reduction of the gradients with the computation of the backward pass. The configs of this post set buckets of `5e4` elements and enable the overlap:

```json
{
  "zero_optimization": {
    "reduce_bucket_size": 5e4,
    "allgather_bucket_size": 5e4,
    "stage3_prefetch_bucket_size": 5e4,
    "overlap_comm": true
  }
}
```

### Communication quantization with ZeRO++

**[Communication quantization (ZeRO++)](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/)** reduces the communication of ZeRO-3 by sending the communicated tensors in fewer bits. To understand the gains of ZeRO++, the communication workflow of ZeRO comes first. The authors of the [ZeRO++ paper](https://arxiv.org/abs/2306.10209) describe it as:

> Assume the model size as $$M$$. During the forward pass, ZeRO conducts an all-gather operation to collect all the parameters ($$M$$) needed to train for all model layers. In the backward pass, ZeRO re-collects parameters ($$M$$) with all-gather first, then each GPU can compute local gradients. After that, ZeRO operates reducescatter function to aggregate and redistribute gradients ($$M$$) across accelerators. In total, ZeRO has a total communication volume of $$3M$$, spreads evenly across 2 all-gather and 1 reduce-scatter.

This is the $$3\Psi$$ of ZeRO-3 above. ZeRO++ introduces three communication improvements, which together cut the communication between nodes from $$3\Psi$$ to $$0.75\Psi$$, i.e. by 4 times:
1. **Quantized Weight Communication for ZeRO (qwZ)**: the all-gather of the forward pass sends the weights in INT8 (1 byte) instead of FP16 (2 bytes), which halves its volume. Instead of one scale for the whole tensor, the weights are quantized by blocks (subsets of parameters), each with its own scale, which reduces the quantization error;
2. **Hierarchical Weight Partition for ZeRO (hpZ)**: each node keeps a full copy of the FP16 weights, partitioned across its GPUs, so that the all-gather of the weights in the backward pass runs inside the node, over its faster links, instead of across nodes. It trades memory for communication;
3. **Quantized Gradient Communication for ZeRO (qgZ)**: replaces the reduce-scatter of the gradients with an all-to-all of gradients quantized to INT4 by blocks, and recovers the full precision before the reduction, to preserve the training accuracy.

ZeRO++ is particularly relevant for clusters with a low-bandwidth network between the nodes, where collective communications are responsible for a large fraction of the overall runtime. It also helps with a small batch size per GPU, as when training on thousands of GPUs, where the computation of each GPU is too short to hide its communication. Its price is the memory of hpZ: in the paper's example of a model of 100 billion parameters on 1,024 GPUs, hpZ takes 8.9 times the memory of ZeRO-3 for the model states, which is still 114 times less than plain data parallelism.

The three methods are set inside the `zero_optimization` entry of the config, as in the [ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/):

```json
{
  "zero_optimization": {
    "zero_hpz_partition_size": 8,
    "zero_quantized_weights": true,
    "zero_quantized_gradients": true
  }
}
```

According to the documentation, the ideal value of `zero_hpz_partition_size` is the number of ranks (GPUs) per node, and its default, 1, disables hpZ. As a good engineering practice, it should be set at runtime, in a config passed to `deepspeed.initialize()` as a dictionary, and not in the config file, to allow for a variable GPU count.

### Activation checkpointing

[**Activation checkpointing**](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html) reduces the memory of the activations, which, unlike the model states, ZeRO does not shard. Instead of storing the output of every layer for the backward pass, only a few layers (the *checkpoints*) keep their inputs, and the activations of the others are computed again during the backward pass, with a forward pass from the closest checkpoint. The paper [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) showed that checkpointing an $$n$$-layer network every $$\sqrt{n}$$ layers stores $$O(\sqrt{n})$$ activations, at the cost of one extra forward pass per mini-batch.

In this implementation, and for simplicity, the layers are checkpointed at a user-specified interval, given by the command line argument `--activation_checkpoint_interval`:

```python
## train.py

def get_cmd_line_args(description='GPT lite on DeepSpeed'):
  # ...
  parser.add_argument('--activation_checkpoint_interval', type=int, default=0,
                      help='activation checkpoint interval (0 means disabled)')
  # ...
```

The layers to checkpoint are wrapped with [`deepspeed.checkpointing.checkpoint`](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html#using-activation-checkpointing), which keeps only the input of the layer and computes its activations again in the backward pass. The `forward()` pass of `GPTlite` iterates over the layers of the model with the `to_layers()` method and checkpoints one layer in every `activation_checkpoint_interval`:

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

where `self.activation_checkpoint_interval` is set when the class is initialized. Note that this scheme recomputes only the checkpointed layers, one in every interval, while the other layers keep all their activations, as without checkpointing. DeepSpeed can also offload the checkpointed inputs to the CPU (`cpu_checkpointing`) and, with model parallelism, partition them across the processes (`partition_activations`), as detailed in the [config documentation](https://www.deepspeed.ai/docs/config-json/#activation-checkpointing). This post skips these options, and its benchmarks run without activation checkpointing (`run.sh` sets the interval to 0).

Activation checkpointing has two implementations in PyTorch, **reentrant** and **non-reentrant**, and PyTorch recommends the non-reentrant one (`use_reentrant=False`). The non-reentrant version is implemented with saved tensor hooks, as detailed in [this PyTorch blog post](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d): the activations of a checkpointed function are not stored in memory, but replaced by a reference, and computed again when the backward pass needs them. Thus, the computation graph is not altered. It also allows for nested checkpointing (a checkpointed function that calls another one), which leads to **higher memory savings**, and its DeepSpeed equivalent is [`deepspeed.checkpointing.non_reentrant_checkpoint`](https://github.com/microsoft/DeepSpeed/blob/42a8eaa705ed30b4a656ac71bdb400772df2cb21/deepspeed/runtime/activation_checkpointing/checkpointing.py).

The reentrant version does not use hooks. It runs the checkpointed function inside a custom [autograd function](https://github.com/pytorch/pytorch/blob/670c5cf96249db28cde757da5a6aa97569760102/torch/utils/checkpoint.py#L75), without storing its activations, and runs it again in the backward pass, where every checkpoint creates a small computational graph of its own. One of its downsides is that the whole function is computed again at every call, while the non-reentrant version can stop once the activations it needs are computed.

Moreover, the graph of each checkpoint is not part of the main graph, so the backward pass cannot run over the whole computational graph (e.g. with `torch.autograd.grad`), and at least one input of the checkpointed function must require gradients, or that part of the model gets no gradients. `deepspeed.checkpointing.checkpoint`, used above, is a reentrant implementation. More details are in the [torch checkpoint documentation](https://pytorch.org/docs/stable/checkpoint.html).

Combining activation checkpointing with sharded parameters (ZeRO-3) may lead to a substantial runtime overhead. In the backward pass, the recomputation of a checkpointed layer needs the parameters of that layer, which ZeRO-3 spreads across the processes: they are all-gathered for the recomputation, and gathered again for the backward pass of the layer if they were released in between. DeepSpeed does not release a parameter that will be reused within `stage3_max_reuse_distance` parameters (`1e9` in the configs of this post): smaller values use less memory, but perform more communication.

### Detour: measuring memory allocated to parameters

The [DeepSpeed API to estimate the memory requirements of model parameters](https://deepspeed.readthedocs.io/en/latest/memory.html#api-to-estimate-memory-usage) for different ZeRO implementations can be used by calling the following method at the onset of execution:

```python
def measure_parameters_memory(model):
  param_size_GB = sum([p.nelement() * p.element_size() for p in model.parameters()])/1024**3
  print(f"Native model parameters size: {round(param_size_GB, 2)}GB.")

  from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
  estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)

  from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
  estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
```

The output shows that:
- the base model requires about 0.323GB per GPU to store its parameters, i.e. its 86.7 million parameters in FP32, at 4 bytes each;
- DeepSpeed ZeRO-2 requires 0.161GB and 0.484GB with and without offloading, respectively;
- DeepSpeed ZeRO-3 requires 0.009GB and 0.190GB with and without offloading, respectively.

These estimates follow the formulas of the [DeepSpeed memory documentation](https://deepspeed.readthedocs.io/en/latest/memory.html), which assume mixed-precision Adam on the 8 GPUs. Without offloading, ZeRO-2 keeps 4 bytes per parameter for the 16-bit parameters and gradients, plus one eighth of the 16 bytes per parameter of the FP32 parameters, gradients and moments of Adam; with offloading, it keeps only the 16-bit parameters, 2 bytes per parameter. ZeRO-3 divides 18 bytes per parameter (the same 16 bytes and the 16-bit parameters) by the 8 GPUs, and adds the 16-bit parameters and gradients of the largest layer, which a GPU gathers to compute it: 2.4 million parameters, in the first linear layer of the feed-forward network of a block. With offloading, only this largest layer stays on the GPU.

This metric is very useful as it gives a quick overview of scaling and is very fast to compute. However, it has many limitations: it only measures the memory of the model states, and it does not take the residual states (activations, temporary buffers, normalization variables) into account. The batch size and numerical precision (or any field in the config file) and the temporary (e.g. communication) buffers are also ignored.

## Results

To measure performance, the DeepSpeed logger was used to extract the following metrics from different runs every 10 steps: model throughput as the average number of samples per second, the average allocated memory, and the maximum allocated memory. The runs used `pytorch==2.0.1`, CUDA `11.7` and `deepspeed==0.10.3`.

All implementations use the same numerical precision, communication bucket sizes, and other config parameters, and none uses activation checkpointing. The following implementations (and configs) were benchmarked:

1. The distributed data parallel (DDP) implementation, i.e. no DeepSpeed (<a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> with `'stage': 0`);
2. The fully-sharded data parallel implementation with ZeRO 1, 2 and 3 (<a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> with `'stage' :1`, `2` or `3`);
3. The ZeRO-3 implementation with ZeRO-Infinity for CPU offloading (<a href="{{ site.assets }}/distributed-training/ds_config_offload.json">`ds_config_offload.json`</a>).

The GPTlite model was tested with a micro-batch size of 1 sample per GPU, and the results are:

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/distributed-training/imgs/benchmark_gptlite.svg"/>

{: .caption}
Maximum and average memory allocated, and throughput, of GPTlite (GPT-3 small) with distributed data parallelism, DeepSpeed ZeRO-1 to ZeRO-3, and ZeRO-3 with CPU offloading, at a micro-batch size of 1 sample per GPU.

**Memory overhead from communication buffers.** Looking at the max vs average memory, note that, in theory, the max memory should be much higher at high ZeRO stages than at low ZeRO stages and DDP. This is because more parameters are communicated, which requires more communication buffers. However, setting the communication bucket sizes (the number of elements communicated at a time) to a low value in the config file overcomes this effect: the max memory stays between 9.16 GB and 10.16 GB in all runs, and it is lowest with ZeRO-3 and offloading. In fact, several runs were also benchmarked with the default communication bucket sizes (`5e8`), which led to higher memory usage, as expected (approximately double in stages 2 and 3), and became prohibitive for some runs.

**Communication vs computation trade-off from different stages in ZeRO.** As expected, as the execution moves from DDP to ZeRO-1, ZeRO-2, ZeRO-3 and ZeRO-Infinity, both the memory consumption and the throughput are reduced. Compared to DDP, ZeRO-1 and ZeRO-2 reduce the average memory by 16% (from 3.24 GB to 2.71 GB) for 3% and 9% of the throughput, while ZeRO-3 reduces it by 21% (to 2.55 GB) for 38% of the throughput (from 11.45 to 7.1 samples per second). Data locality is traded for communication of parameters, and a price in performance is paid for the communication and offloading of parameters. In practice, the fastest choice is therefore the lowest stage that fits the model in memory.

**Offloaded vs in-memory parameters.** Offloading proved to be a consistent way to reduce memory usage, with the drawback of a reduction in throughput, as expected: compared to ZeRO-3 without offloading, it reduces the average memory by 8% (from 2.55 GB to 2.35 GB) and the throughput by 13% (from 7.1 to 6.2 samples per second).

**Lack of superlinear speedup.** A small improvement in memory efficiency was observed, but still far from the claims of the original DeepSpeed team. One explanation is that a small network of 8 GPUs was used, compared to the 64 to 400 GPUs used by the authors in their benchmarks, so the memory reduction is much smaller. Moreover, most of the memory of a model as small as GPTlite is taken by its activations, which are not sharded: its model states take about 1.3 GB per GPU with DDP (16 bytes for each of its 86.7 million parameters), out of a peak of 10.16 GB. A large network of GPUs also underlies their claim of [superlinear speedup](https://en.wikipedia.org/wiki/Speedup) (with more GPUs, each GPU holds fewer model states and fits a larger batch, at a higher throughput), which was not observed here.

Finally, **communication quantization** (ZeRO++) was not used, as it did not result in any improvement. This is in line with the [ZeRO++ paper](https://arxiv.org/abs/2306.10209), which claims this method is applicable to small batch sizes per GPU or low-bandwidth networks.

## Comparison of DDP, ZeRO and offloading

The following table sums up the memory of the model states and the communication of each method, per process and training step, for a model of $$\Psi$$ parameters trained with mixed-precision Adam on $$P$$ processes:

| Method | Model states per process (bytes) | Communication per process and step (elements) | When to use it |
|---|---|---|---|
| DDP (ZeRO-0) | $$16\Psi$$ | $$2\Psi$$: an all-reduce of the gradients | The model states fit on every GPU: the fastest method |
| ZeRO-1 | $$4\Psi + 12\Psi/P$$ | $$2\Psi$$: a reduce-scatter of the gradients and an all-gather of the parameters | DDP does not fit: most of the memory of the model states goes, with no extra communication |
| ZeRO-2 | $$2\Psi + 14\Psi/P$$ | $$2\Psi$$, as ZeRO-1 | ZeRO-1 does not fit: the gradients are sharded too, with no extra communication |
| ZeRO-3, FSDP | $$16\Psi/P$$ | $$3\Psi$$: also an all-gather of the parameters in each pass | ZeRO-2 does not fit: 1.5 times the communication of DDP |
| ZeRO-Offload, ZeRO-Infinity | Less: the offloaded states move to the CPU memory (or NVMe) | As above, plus the copies between the CPU and the GPU | ZeRO-3 does not fit: the slowest, limited by the link between the CPU and the GPU |

## Resources and code

This post explored only the dimension of data parallelism. More details about DeepSpeed can be found in the [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest), the [training features page](https://www.deepspeed.ai/training/#features), the [tutorials page](https://www.deepspeed.ai/tutorials/), the [HuggingFace page for DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed), and the examples at [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/).

As an important remark, finding the best parallelism strategy is hard, due to the high number of hyper-parameters (ZeRO stages, offloading, activation checkpointing intervals, pipeline parallelism stages, data parallelism, tensor parallelism, etc.). The best strategy also depends on the ML model, the data and the hardware. In practice, the DeepSpeed config files of this post are a manually-optimized ballpark figure of the default config file with some parameter grid search. On this topic, there is still plenty of work to be done to make it optimal, possibly by exploring the [autotuning](https://www.deepspeed.ai/tutorials/autotuning/) tools in DeepSpeed.

This post only scratched the surface of the capabilities of DeepSpeed. There are plenty of features that should also be explored. To name a few:
- [**autotuning**](https://www.deepspeed.ai/tutorials/autotuning/) ([README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)) for parallelism hyper-parameter discovery;
- [**flops profiler**](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html) to measure the time, flops and parameters of individual layers;
- [**sparse attention kernels**](https://www.deepspeed.ai/2020/09/08/sparse-attention.html) ([API](https://www.deepspeed.ai/docs/config-json/#sparse-attention)) to support long sequences of model inputs, such as text, image, or sound;
- [**communication optimizers**](https://www.deepspeed.ai/tutorials/zero-one-adam/) that offer the same convergence as Adam/LAMB but incur up to 26x less communication on BERT-large pretraining;
- [**monitor**](https://www.deepspeed.ai/training/#monitor) to log live training metrics to TensorBoard, csv file or other backend;
- [**model compression**](https://www.deepspeed.ai/compression/) ([API](https://www.deepspeed.ai/docs/config-json/#compression)) via layer reduction, weight quantization, activation quantization, sparse pruning, row pruning, head pruning and channel pruning, to deliver faster speed and smaller model size.

The code of all experiments is available in the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training).
