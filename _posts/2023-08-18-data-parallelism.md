---
layout: post
title:  "Data parallelism, sharding and CPU offloading"
categories: ["Distributed Training"]
tags: ["data parallelism", "ZeRO", "FSDP", "CPU offloading", "activation checkpointing", "DeepSpeed", "PyTorch"]
---

[Data parallelism](https://en.wikipedia.org/wiki/Data_parallelism) trains one model on several processors at once, each processor working on different input samples. Consider an input of shape $$B \times S \times E$$, where $$B$$ is the batch size, $$S$$ the sequence (or temporal) length and $$E$$ the embedding (or feature) size. Data parallelism splits this input along the batch dimension across $$P$$ processes, so that each process holds a local input of shape $$B/P \times S \times E$$:

{: .picture}
<img width="24.1%" height="24.1%" src="{{ site.assets }}/distributed-training/imgs/data_parallelism.svg"/>

{: .caption}
An illustration of the DDP data layout, split across 4 processes color-coded as blue, yellow, red and green.

The examples that follow train the GPTlite model of the [GPT-lite post]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}) on one node with 8 GPUs, with [PyTorch DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (Distributed Data Parallel), PyTorch FSDP (Fully Sharded Data Parallel) and DeepSpeed ZeRO (Zero Redundancy Optimizer), where DeepSpeed is a training library built on top of PyTorch.

## Distributed Data Parallelism

Distributed data parallelism (DDP) is the simplest approach: each process keeps a full copy of the model, with its parameters, gradients and optimizer states, which together form the *model states*. Each process takes a different mini-batch as input and runs the forward pass to compute the loss of that batch. During the backward pass, at every layer, each process computes the gradients of its local batch, and an [all-reduce](https://en.wikipedia.org/wiki/Collective_operation) computes the mean of these gradients across all processes. The optimizer then uses the mean gradients to update the parameters. Since all copies start from the same parameters and apply the same updates, they remain identical throughout the training:

{: .picture}
<img width="65.8%" height="65.8%" src="{{ site.assets }}/distributed-training/imgs/workflow_data_parallelism.svg"/>

{: .caption}
Distributed data parallelism (DDP) on 4 processes. Each process holds a full copy of a 4-layer model (the parameters, optimizer states and gradients of every layer), initialized identically, and takes its own input. The forward pass (left arrow) computes the activations and the loss of each process. In the backward pass (right arrow), each process computes the local gradients of every layer, and an all-reduce averages them across the processes. The optimizer then applies the same mean gradients on every process, so all copies stay identical.

As the diagram shows, the model states are stored $$P$$ times, once per process. Sharding removes these redundant copies.

## Sharded Data Parallelism

In sharded data parallelism, or *sharding*, each process holds only a partition (a shard) of the model states, and, as before, takes its own mini-batch as input. DeepSpeed implements it as ZeRO ([ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), Rajbhandari et al., 2020), and PyTorch as FSDP ([PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277), Zhao et al., 2023). ZeRO provides several execution modes, called *stages*, and each stage sets which model states are split across the processes and which are replicated:
- **Stage 0**: nothing is split, and the training runs as in DDP.
- **Stage 1 (ZeRO-1)**: the optimizer states are split. With [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) and [mixed precision](https://en.wikipedia.org/wiki/Mixed-precision_arithmetic), these are an fp32 copy of the parameters, the momentum and the variance. Since each process updates only its own partition, it needs only the mean gradients of that partition, so a reduce-scatter replaces the all-reduce. After the update, an all-gather sends the new parameters to all processes.
- **Stage 2 (ZeRO-2)**: the gradients are also split, so each process keeps only the gradients of its partition. Like stage 1, it changes only the backward pass and the optimizer step.
- **Stage 3 (ZeRO-3)**: the parameters are also split. An all-gather collects the parameters of each layer before its forward pass, and again before its backward pass, and each process frees them after use and keeps only its partition. This adds communication to both the forward and the backward passes.

ZeRO splits only the model states. Each process still keeps the activations of its $$B/P$$ samples in full, with temporary buffers and fragmented memory, which the ZeRO paper calls the *residual states*, and the activations grow with the batch size and the sequence length. Activation checkpointing reduces them: the forward pass keeps the activations of a few layers only, and the backward pass recomputes the others from the closest kept layer. For a network of $$n$$ layers, this needs $$O(\sqrt{n})$$ memory, for the cost of one extra forward pass ([Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), Chen et al., 2016).

The diagram below shows ZeRO-3 as the ZeRO paper describes it: each process owns the model states of one layer, broadcasts their parameters before the layer runs, and receives the sum of the gradients of all processes after its backward pass:

{: .picture}
<img width="95.3%" height="95.3%" src="{{ site.assets }}/distributed-training/imgs/workflow_sharding.svg"/>

{: .caption}
Stage 3 sharding (ZeRO-3) on 4 processes. Process k holds the parameters, optimizer states and gradients of layer k only (solid boxes; the dashed boxes are layers held by other processes), takes its own input and keeps its own activations. Forward pass (left): before layer k is computed, process k broadcasts its parameters, and the other processes discard them after use. Backward pass (right): process k broadcasts the parameters of layer k again, every process computes its local gradients of layer k, and these gradients are summed (reduced) on process k, which updates the parameters of layer k. Inspired by: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

DeepSpeed ZeRO-3 and PyTorch FSDP split the work in a different way: they cut the tensors of every layer into $$P$$ equal shards, and each process owns one shard of every layer. An all-gather then replaces each broadcast, and a reduce-scatter replaces each reduction, with the same communication volume.

How much more do the stages communicate than DDP? An all-reduce is commonly implemented as a reduce-scatter followed by an all-gather, so one reduce-scatter and one all-gather move as much data as one all-reduce. Stages 1 and 2 thus have the same communication volume as DDP. Stage 3 adds the all-gather of the parameters in the backward pass, and communicates 1.5 times as much. The [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) summarizes the memory per GPU of each stage:

{: .picture}
<img width="85.8%" height="85.8%" src="{{ site.assets }}/distributed-training/imgs/deepspeed_stages.svg"/>

{: .caption}
Per-GPU memory of the model states (parameters, gradients and optimizer states) with plain data parallelism (baseline) and with the three ZeRO stages: $$P_{os}$$ partitions the optimizer states across the GPUs, $$P_{os+g}$$ also the gradients, and $$P_{os+g+p}$$ also the parameters. $$\Psi$$ is the number of parameters, $$K$$ the memory multiplier of the optimizer states and $$N_d$$ the data-parallel degree; the example uses $$K=12$$ (mixed-precision Adam), $$\Psi=7.5B$$ and $$N_d=64$$. The last column is the communication volume relative to the baseline. Adapted from: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) and the [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

With mixed precision and Adam, each parameter takes 2 bytes in fp16, its gradient another 2 bytes, and its optimizer states $$K = 12$$ bytes. The data-parallel degree $$N_d$$ of the figure is the number of processes $$P$$. A model of $$\Psi = 7.5$$ billion parameters thus needs $$(2+2+K)\Psi = 120$$ GB of model states on every GPU with DDP. On 64 GPUs, ZeRO-1 reduces this to 31.4 GB, ZeRO-2 to 16.6 GB, and ZeRO-3, which divides all model states by $$N_d$$, to 1.9 GB. DDP suits models whose model states fit in one GPU. ZeRO-1 and ZeRO-2 save memory at the same communication volume, and ZeRO-3 trains models whose fp16 parameters alone do not fit in one GPU, at 1.5 times the communication.

## CPU Offloading

Some models do not fit in GPU memory, even with sharding. A common technique to overcome this limitation is *offloading*, which keeps on the GPU only the data of the current and the next computation steps: the data of the layers already processed moves from the GPU to the larger CPU memory, and the data of the upcoming layers moves from the CPU back to the GPU before it is needed. [vDNN](https://arxiv.org/pdf/1602.08124.pdf) (virtualized deep neural networks, Rhu et al., 2016) applies this idea to the activations of convolutional networks, which take an order of magnitude more memory than their weights, and [vDNN+](https://ieeexplore.ieee.org/document/8820980) (Shriram et al., 2019) extends it.

Why can the training be split into such steps? Because the network, and therefore its loss, is a composition of the functions of its layers. As an example, for the mean absolute error (MAE) of a batch of $$B$$ samples:

$$
\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \left\vert y_b - f^{(L+1)} \circ \dots \circ f^{(2)} \circ f^{(1)} (x_b) \right\vert
$$

where $$f^{(l)}$$ is the function of layer $$l$$, $$L$$ is the number of hidden layers, and $$y_b$$ is the label of the input $$x_b$$. Because of this composition, the forward pass of a layer needs only its weights and the output of the previous layer, and its backward pass needs only its weights, its input and the gradient that comes from the next layer. So vDNN copies the input of each layer to the CPU during the forward pass, and back to the GPU during the backward pass, before the layer needs it. A second CUDA stream runs these transfers, so that they overlap with the computation and hide their latency. The forward pass is illustrated below:

{: .picture}
<img width="88.3%" height="88.3%" src="{{ site.assets }}/distributed-training/imgs/vdnn2.svg"/>

{: .caption}
The forward pass of vDNN on a convolutional neural network, at layer N. Blue arrows are the input and output feature maps $$X$$ and $$Y$$ of the current layer, the purple arrow its weights $$W$$ and the yellow arrow its cuDNN workspace $$WS$$. White arrows are data that the current layer does not use. The input feature maps $$X$$ of the preceding layers, marked with a black cross, are not used by the current layer and can be released from GPU memory. Adapted from: [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124)

The backward pass is more involved, as the gradients also propagate down the network:

{: .picture}
<img width="88.3%" height="88.3%" src="{{ site.assets }}/distributed-training/imgs/vdnn3.svg"/>

{: .caption}
The backward pass of vDNN at layer 2. Blue arrows are the feature maps $$X$$ and $$Y$$ of the current layer, green arrows its gradients $$dX$$ and $$dY$$, the purple arrow its weights $$W$$ and the yellow arrow its cuDNN workspace $$WS$$. White arrows are data that the current layer does not use. The data used only by the layers whose backward pass is complete (the layers after layer 2 and the loss function), marked with a black cross, is not reused for this batch and can be released from GPU memory. Adapted from: [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124)

## Model and Dataset

The examples use GPTlite with the dimensions of the GPT-3 Small model of [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Table 2.1):

```python
n_layer = 12   # depth of the network as number of decoder blocks.
n_embd = 768   # size of the embeddings (d_model)
n_head = 12   # number of attention heads in the Multi-Attention mechanism
block_size = 2048   # block size, i.e. the maximum sequence length, the n_ctx in the paper
dropout = 0.1   # dropout rate (variable p) for dropout units
```

The functions `get_dataset()` and `get_model()` return the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset and the model. Each sample is a window of `block_size` tokens at a random offset of the text, and its target is the same window shifted by one token, so that the model learns to predict the next token at every position:

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

With a vocabulary of 65 characters, GPTlite has 86.7 million parameters. GPT-3 Small has 125 million, because its vocabulary of 50257 tokens needs a much larger embedding table.

### Using a torchvision Model {#torchvision-model}

Nothing in the code that follows is specific to GPTlite: it applies to any model of type `torch.nn.Module` and any dataset of type `torch.utils.data.Dataset`. As an example, multi-class classification of the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) images with the [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network)-18 network of `torchvision` requires only these two functions:

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

The PyTorch implementation starts by reading the environment variables set by the [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) launcher, described below. These variables identify each process in the network and each GPU within a node:

```python
import os
rank = int(os.environ['RANK'])   # the unique id across all processes in all nodes
local_rank = int(os.environ['LOCAL_RANK'])   # the unique id across this node
world_size = int(os.environ['WORLD_SIZE'])   # the number of processes across all nodes
```

Each process then joins the process group with `torch.distributed.init_process_group()`, and a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) defines how each process iterates through the data:

```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)
```

Its `sampler` argument is a `DistributedSampler`, which assigns different samples of the dataset to different processes. Without it, all processes would load the same samples, and all copies of the model would compute the same gradients:

```python 
sampler = torch.utils.data.distributed.DistributedSampler(
  dataset, num_replicas=world_size, rank=rank)
```

Each process places its copy of the model on its own GPU, here in [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format):

```python
device = f"cuda:{local_rank}"
dtype = torch.bfloat16
model = model.to(device=device, dtype=dtype)
```

and wraps it in [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for DDP:

```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

or in [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html) for sharding. Its `sharding_strategy` plays the role of the ZeRO stage: `SHARD_GRAD_OP` splits the gradients and the optimizer states, as ZeRO-2, and the default strategy, `FULL_SHARD`, also splits the parameters, as ZeRO-3:

```python
model = torch.distributed.fsdp.FullyShardedDataParallel(
    model,
    device_id=local_rank,
    sharding_strategy=torch.distributed.fsdp.api.ShardingStrategy.SHARD_GRAD_OP,  # sharded grads + optimizer states (ZeRO-2-like)
)
```

PyTorch has since deprecated this API in favor of `fully_shard` (FSDP2), which splits each parameter along its first dimension ([FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)).

The training loop is identical to the one on a single GPU, because the wrappers perform all the communication and synchronization inside the forward and backward passes:

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

Inside `loss.backward()`, the gradients are computed from the last layer to the first, and DDP uses this order to hide most of its communication. It groups the gradients into buckets, in about the reverse order of the parameters, and starts an asynchronous all-reduce of a bucket as soon as all its gradients are ready. The gradients of the upper layers are thus averaged while the backward pass computes those of the lower layers, and at the end of the backward pass each process waits for all the all-reduces to finish ([DDP design notes](https://pytorch.org/docs/master/notes/ddp.html#internal-design); [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704), Li et al., 2020). For additional memory savings, PyTorch can also offload the saved activations to the CPU, with custom [hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).

Finally, the run is launched with `torchrun`, which starts `nproc_per_node` processes on the node where it runs, one per GPU, and sets the environment variables read above. A run on several nodes starts `torchrun` on each node, with the same rendezvous arguments. On one node with 4 GPUs:

```shell
$ torchrun --standalone --nproc_per_node=4 train.py
```

## DeepSpeed Implementation

Porting the code to DeepSpeed takes few changes, because most DeepSpeed features are enabled through its API or its [configuration file](https://www.deepspeed.ai/docs/config-json/), a JSON file that covers parallelism, floating-point precision, logging, communication and more. The starting config below sets the batch size to 256 samples, makes the DeepSpeed logger report the memory and the throughput every 10 steps (`steps_per_print`), and, optionally, defines the optimizer and the learning-rate scheduler:

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

In DeepSpeed terminology, `train_micro_batch_size_per_gpu` is the number of samples that one process runs in one forward and backward pass, and `train_batch_size` is the number of samples in one optimizer step, over all processes and all gradient accumulation steps:

```
train_batch_size = train_micro_batch_size_per_gpu * num_gpus * num_nodes * gradient_accumulation_steps
```

For a given number of GPUs, setting two of the three values sets the third, so gradient accumulation is enabled by setting `train_micro_batch_size_per_gpu`. *Gradient accumulation* sums the gradients of several micro-batches before each optimizer step, which allows training with a large batch in the memory of a small one. The configs of the benchmark use micro-batches of 1 sample, so each of the 8 GPUs runs 32 micro-batches per optimizer step.

ZeRO is enabled by setting its stage in the config. If the stage is omitted or set to 0, ZeRO is disabled, and the training follows the regular distributed data parallel workflow:

```json
{
  "zero_optimization": { "stage": 3 }
}
```

CPU offloading in DeepSpeed is called [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) ([ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840), Ren et al., 2021). Unlike vDNN, it offloads the model states rather than the activations: it keeps the optimizer states and the gradients in CPU memory, and runs the optimizer step on the CPU. On top of stage 3, DeepSpeed provides [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) (Rajbhandari et al., 2021), which offloads the parameters as well, to CPU memory or to [NVMe](https://en.wikipedia.org/wiki/NVM_Express) storage. According to the [ZeRO-3 documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html#zero), it also makes better use of the bandwidth than ZeRO-Offload, and overlaps more of the communication with the computation. This config offloads the optimizer states and the parameters to the CPU:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true }
  }
}
```

DeepSpeed manages internally the boilerplate that PyTorch needs for the process group and the data loading, so the setup reduces to:

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

The training loop follows the same structure as in PyTorch, with two differences: `initialize()` returns a `train_dataloader` that already assigns a different subset of the samples to each process, so no distributed sampler is needed, and `engine.step()` also zeroes the gradients:

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

The run is launched with `torchrun`, as before, or with the DeepSpeed launcher:

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

The DeepSpeed API can [estimate the memory of the model states](https://deepspeed.readthedocs.io/en/latest/memory.html#api-to-estimate-memory-usage) of ZeRO-2 and ZeRO-3, with and without offloading, and this function prints the estimates at the start of the run:

```python
def measure_parameters_memory(model):
  param_size_GB = sum([p.nelement() * p.element_size() for p in model.parameters()])/1024**3
  print(f"Native model parameters size: {round(param_size_GB, 2)}GB.")

  from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
  estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)

  from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
  estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
```

For GPTlite, the output shows that:
- the parameters take 0.323 GB in fp32;
- ZeRO-2 needs 0.161 GB per GPU with offloading, and 0.484 GB without;
- ZeRO-3 needs 0.009 GB per GPU with offloading, and 0.190 GB without.

With offloading, ZeRO-3 keeps on the GPU only the largest layer, here a linear layer of the FFN with 2.36 million parameters, at 4 bytes per parameter. These estimates are cheap to compute and give a quick overview of how the memory scales. However, they cover only the model states, in mixed precision: they ignore the activations, the temporary and communication buffers, the batch size, and the precision and the other fields of the config.

## Results

The DeepSpeed logger reports three metrics every 10 steps (`steps_per_print`): the throughput, as the average number of samples per second, and the average and the maximum allocated GPU memory. All runs use `pytorch==2.0.1`, CUDA `11.7` and `deepspeed==0.10.3`, in fp32 (`fp16` is disabled in the configs), with the same config except for the ZeRO stage and the offloading:

1. DDP: <a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> with stage 0;
2. ZeRO-1, ZeRO-2 and ZeRO-3: <a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> with stage 1, 2 or 3;
3. ZeRO-3 with CPU offloading of the parameters and the optimizer states: <a href="{{ site.assets }}/distributed-training/ds_config_offload.json">`ds_config_offload.json`</a>.

With micro-batches of 1 sample per GPU, GPTlite gives:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/benchmark_gptlite.svg"/>

{: .caption}
Maximum and average memory allocated, and throughput, of GPTlite (GPT-3 small) with distributed data parallelism, DeepSpeed ZeRO-1 to ZeRO-3, and ZeRO-3 with CPU offloading, at a micro-batch size of 1 sample per GPU.

**Throughput.** Moving from DDP to ZeRO-1, ZeRO-2, ZeRO-3 and the offloading reduces both the memory and the throughput. This is the expected trade-off, as each method gives up some data locality for the communication, or the offloading, of the model states. Gradient accumulation widens the gap: DeepSpeed reduces the gradients of DDP and ZeRO-1 once per optimizer step, after the last micro-batch, because each process keeps the full gradients and accumulates them locally. It must reduce those of ZeRO-2 and ZeRO-3 after every micro-batch, since each process keeps only the gradients of its partition, and ZeRO-3 also gathers the parameters in every micro-batch. With 32 micro-batches per step, ZeRO-2 thus communicates about 16 times the volume of DDP per step, and ZeRO-3 at least 32 times. ZeRO-1 loses 3% of the throughput of DDP, and ZeRO-2, which overlaps its reductions with the backward pass (`overlap_comm`), loses 9%. ZeRO-3, which must gather the parameters of a layer before it can compute the layer, loses 38%, and the CPU offloading another 13%.

**Memory.** From DDP to ZeRO-3, the average allocated memory drops from 3.24 GB to 2.55 GB and the peak from 10.16 GB to 9.37 GB, while the offloading lowers them further, to 2.35 GB and 9.16 GB. These savings are small, far from the results reported by the DeepSpeed team, and the reason is the size of the model. In fp32 with Adam, each parameter takes 16 bytes of model states (the parameter, its gradient, the momentum and the variance), which is 1.3 GB for GPTlite, and ZeRO-3 on 8 GPUs removes at most 7/8 of it, or 1.1 GB. Most of the 10 GB peak holds the activations of the samples of 2048 tokens and temporary buffers, which ZeRO does not split. The ZeRO paper, in contrast, trains models of over 100 billion parameters, whose model states dominate the memory, and reports a superlinear speedup from 64 to 400 GPUs, because the freed memory allows larger batches per GPU.

**Communication buffers.** ZeRO communicates in buckets, the buffers of one collective call, and the higher stages, which communicate more, need more of these buffers. With large buckets, their peak memory would thus exceed that of DDP. The configs prevent this with buckets of 5e4 elements (`allgather_bucket_size`, `reduce_bucket_size` and `stage3_prefetch_bucket_size`). With the DeepSpeed default of 5e8 elements, the memory of stages 2 and 3 about doubled, and some runs ran out of memory. Smaller buckets need less memory, but more collective calls.

**Communication quantization.** ZeRO++ ([ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209), Wang et al., 2023) quantizes the communicated weights to INT8 and the gradients to INT4, and keeps a copy of the weights in each node, which together reduce the communication volume of ZeRO by 4 times. On this single node it gave no speedup, so the runs above do not use it. This is in line with the ZeRO++ paper, which targets clusters with a low bandwidth, or a scale that forces small batches per GPU.

## Further Reading

The [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest), the [training features](https://www.deepspeed.ai/training/#features), the [tutorials](https://www.deepspeed.ai/tutorials/), the [Hugging Face guide to DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) and the examples of [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/) cover DeepSpeed in more detail. The code of all experiments is available in the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training).
