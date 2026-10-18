---
layout: post
title:  "Data parallelism, sharding and CPU offloading"
categories: ["Distributed Training"]
tags: ["data parallelism", "ZeRO", "FSDP", "CPU offloading", "activation checkpointing", "DeepSpeed", "PyTorch"]
---


Distributed data parallelism (DDP) refers to the parallel execution of different input samples across processors. Consider an input of shape $$B \times S \times E$$ (batch size, sequence/temporal size and embedding/feature size). Data parallelism splits this input along the batch dimension across $$P$$ processes, resulting in a local input of shape $$B/P \times S \times E$$ per process:

{: .picture}
<img width="24.2%" height="24.2%" src="{{ site.assets }}/distributed-training/imgs/data_parallelism.svg"/>

{: .caption}
An illustration of the DDP data layout, split across 4 processes color-coded as blue, yellow, red and green.

This post applies distributed data parallelism to the training of the [GPTlite model introduced in the previous post]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}), on a network of 8 GPUs, using [PyTorch DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (Distributed Data Parallel) and [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer, a lightweight wrapper on PyTorch).

There are two main data parallelism approaches:

**Distributed Data Parallelism** keeps a full copy of the model (weights, optimizer states and gradients) on each processor. All replicas are initialized with the same parameters. Each processor takes a different mini-batch as input and performs a forward pass to compute the loss of that batch. During the backward pass, at every layer, each process computes the gradients of its local batch, and an all-reduce computes the mean gradients across all processors. The optimizer then uses these mean gradients to update the parameters. Since all replicas start from the same parameters and apply the same updates, they remain identical throughout training.

{: .picture}
<img width="65.7%" height="65.7%" src="{{ site.assets }}/distributed-training/imgs/workflow_data_parallelism.svg"/>

{: .caption}
Distributed data parallelism (DDP) on 4 processes. Each process holds a full copy of a 4-layer model (the parameters, optimizer states and gradients of every layer), initialized identically, and takes its own input. The forward pass (left arrow) computes the activations and the loss of each process. In the backward pass (right arrow), each process computes the local gradients of every layer, and an all-reduce averages them across the processes. The optimizer then applies the same mean gradients on every process, so all copies stay identical.

As shown in the diagram, every processor holds a full replica of the model, so the model states are stored redundantly $$P$$ times. Sharding removes this redundancy.

**(Fully-)Sharded Data Parallelism (FSDP)**, or simply **sharding**, is a setup where each processor holds only a partition of the model: the parameters, optimizer states and gradients of a subset of layers. As before, each processor takes a different mini-batch as input. In DeepSpeed, sharding is called **ZeRO (Zero Redundancy Optimizer)**. ZeRO provides several execution modes, called **stages**. Each stage defines which variable types are partitioned across processes and which are replicated:
- **ZeRO stage 0**: no sharding of any variables, equivalent to Distributed Data Parallelism.
- **ZeRO stage 1 (ZeRO-1)**: the optimizer states (e.g., for the Adam optimizer, the weights and the first and second moment estimates) are partitioned across the processes. Affects only the backward pass.
- **ZeRO stage 2 (ZeRO-2)**: the gradients used to update the model weights are also partitioned, so that each process retains only the gradients corresponding to its portion of the optimizer states. Also relevant only for the backward pass.
- **ZeRO stage 3 (ZeRO-3)**: the model parameters are also partitioned across the processes. Adds a communication overhead to both the forward and backward passes. 

Note that activations are not sharded, i.e. each processor keeps them in *full shape*. In modern models, **a large fraction of memory is allocated to residual memory (activations, normalization layers, etc.), which FSDP does not shard**. The following diagram illustrates the workflow of stage 3 sharding (of parameters, gradients and optimizer states):

{: .picture}
<img width="95.2%" height="95.2%" src="{{ site.assets }}/distributed-training/imgs/workflow_sharding.svg"/>

{: .caption}
Stage 3 sharding (ZeRO-3) on 4 processes. Process k holds the parameters, optimizer states and gradients of layer k only (solid boxes; the dashed boxes are layers held by other processes), takes its own input and keeps its own activations. Forward pass (left): before layer k is computed, process k broadcasts its parameters, and the other processes discard them after use. Backward pass (right): process k broadcasts the parameters of layer k again, every process computes its local gradients of layer k, and these gradients are summed (reduced) on process k, which updates the parameters of layer k. Inspired by: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

Higher stages lead to a lower memory footprint. Stages 1 and 2 have the same communication volume as regular data parallelism, while stage 3 communicates 1.5x as much. The memory savings of each stage are summarized in the [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) as:

{: .picture}
<img width="85.9%" height="85.9%" src="{{ site.assets }}/distributed-training/imgs/deepspeed_stages.svg"/>

{: .caption}
Per-GPU memory of the model states (parameters, gradients and optimizer states) with plain data parallelism (baseline) and with the three ZeRO stages: $$P_{os}$$ partitions the optimizer states across the GPUs, $$P_{os+g}$$ also the gradients, and $$P_{os+g+p}$$ also the parameters. $$\Psi$$ is the number of parameters, $$K$$ the memory multiplier of the optimizer states and $$N_d$$ the data-parallel degree; the example uses $$K=12$$ (mixed-precision Adam), $$\Psi=7.5B$$ and $$N_d=64$$. The last column is the communication volume relative to the baseline. Adapted from: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) and the [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

## CPU Offloading

Some models do not fit in GPU memory, even with sharding. A common technique to overcome this limitation is CPU offloading, also referred to as virtualized Deep Neural Networks by [vDNN (Rhu et al.)](https://arxiv.org/pdf/1602.08124.pdf) and [vDNN+ (Shriram et al.)](https://www.cse.iitb.ac.in/~shriramsb/submissions/GPU_mem_ML.pdf). The method keeps on the GPU only the activations and model states required for the current and next computation steps. Layers already processed are moved from GPU to CPU, while upcoming layers are moved from CPU to GPU.

This is possible because the loss (and its derivative) can be written as a composition of the activation functions of all layers. As an example, for the mean absolute error (MAE):

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N | y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) |
$$

where $$f^{(l)}$$ is the activation function of layer $$l$$, $$L$$ is the number of hidden layers and $$y$$ is the ground truth. 
The key concept is the **composition** of the functions $$f$$ across layers. To hide the latency of moving layers to and from the GPU, the computation of the current layers is overlapped with the transfer of the upcoming ones. The forward pass is illustrated below:

{: .picture}
<img width="88.3%" height="88.3%" src="{{ site.assets }}/distributed-training/imgs/vdnn2.svg"/>

{: .caption}
The forward pass of vDNN on a convolutional neural network, at layer N. Blue arrows are the input and output feature maps $$X$$ and $$Y$$ of the current layer, the purple arrow its weights $$W$$ and the yellow arrow its cuDNN workspace $$WS$$. White arrows are data that the current layer does not use. The input feature maps $$X$$ of the preceding layers, marked with a black cross, are not used by the current layer and can be released from GPU memory. Adapted from: [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124)


The backward pass is more involved, as gradients are also propagated down the model:

{: .picture}
<img width="88.3%" height="88.3%" src="{{ site.assets }}/distributed-training/imgs/vdnn3.svg"/>

{: .caption}
The backward pass of vDNN at layer 2. Blue arrows are the feature maps $$X$$ and $$Y$$ of the current layer, green arrows its gradients $$dX$$ and $$dY$$, the purple arrow its weights $$W$$ and the yellow arrow its cuDNN workspace $$WS$$. White arrows are data that the current layer does not use. The data used only by the layers whose backward pass is complete (the layers after layer 2 and the loss function), marked with a black cross, is not reused for this batch and can be released from GPU memory. Adapted from: [vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design](https://arxiv.org/abs/1602.08124)

## Model and dataset setup

The implementation uses the *GPTlite* model from the previous post, with specs matching the *GPT-3 Small* model in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Table 2.1):

```python
n_layer = 12   # depth of the network as number of decoder blocks.
n_embd = 768   # size of the embeddings (d_model)
n_head = 12   # number of attention heads in the Multi-Attention mechanism
block_size = 2048   # block size, i.e. the maximum sequence length, the n_ctx in the paper
dropout = 0.1   # dropout rate (variable p) for dropout units
```

The methods `get_model()` and `get_dataset()` return the model and the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset:

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

This code applies to any model of type `torch.nn.Module` and any dataset of type `torch.utils.data.Dataset`. As an example, multi-class classification with the `ResNet` network on the `CIFAR10` dataset from `torchvision` requires only the following two methods:

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

The PyTorch implementation starts by reading the environment variables set by the [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) launcher (detailed below). These variables uniquely identify each process in the network and each GPU device within a node:

```python
import os
rank = int(os.environ['RANK'])   # the unique id across all processes in all nodes
local_rank = int(os.environ['LOCAL_RANK'])   # the unique id across this node
world_size = int(os.environ['WORLD_SIZE'])   # the number of processes across all nodes
```

The [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) defines how each process iterates through the data:

```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler)
```

The `sampler` argument is a `torch.utils.data.distributed.DistributedSampler`, which assigns different samples of the dataset to different processes. Without it, all processes would load the same datapoints in every iteration.

```python 
sampler = torch.utils.data.distributed.DistributedSampler(
  dataset, num_replicas=world_size, rank=rank)
```

Each model replica is then placed on its own GPU, with the correct data type:

```python
device = f"cuda:{local_rank}"
dtype = torch.bfloat16
model = model.to(device=device, dtype=dtype)
```

and wrapped with [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for the DDP implementation:

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

The training loop is identical to the serial one, as PyTorch performs all communication and synchronization internally:

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

Gradients are computed and mean-reduced from the top to the bottom layer of the model. This allows the computation of the gradients of the lower layers to overlap with the communication of the gradients of the upper layers. According to the [PyTorch DDP documentation](https://pytorch.org/docs/master/notes/ddp.html#internal-design):

> The backward pass iteratively computes gradients (from last to first layer) and collects blocks of gradients to be communicated. These blocks will be mean-reduced asynchronously during the backward pass, while the computation for the backward pass proceeds. Therefore it overlaps backward pass computation with gradients communication. At the end of the backward pass, all GPUs wait for all gradient all-reduces to finish, and then triggers the parameter updates.

For additional memory savings, PyTorch can also offload tensors with custom [hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).

Finally, the application is launched with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html). Torchrun is a network bootstrapper that spawns a Python script on all compute nodes and sets the environment variables mentioned above. The usage is:

```shell
$ torchrun --standalone --nproc_per_node=4 train.py
```

where `nproc_per_node` is the number of GPUs on each node.

## DeepSpeed implementation


Porting existing code to DeepSpeed is simple. DeepSpeed features are activated via the DeepSpeed API or its [Configuration JSON](https://www.deepspeed.ai/docs/config-json/), which covers parallelism, floating-point precision, logging, communication and more. The starting config below sets the DeepSpeed logger to output memory and throughput info every 10 steps (`steps_per_print`), sets the batch size to `256`, and (optionally) defines the optimizer (`optimizer`) and learning rate scheduler (`scheduler`):

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

In DeepSpeed terminology, `train_micro_batch_size_per_gpu` is the batch size loaded per dataloader (i.e. per GPU, per gradient accumulation step), while `train_batch_size` is the batch size across all gradient accumulation steps and processes in the network:

```
train_batch_size = train_micro_batch_size_per_gpu * num_gpus * num_nodes * gradient_accumulation_steps
```

Therefore, **gradient accumulation** is enabled simply by setting `train_micro_batch_size_per_gpu`.
**ZeRO Fully-Sharded Data Parallelism** is activated by specifying the stage in the config file. If the stage is omitted or set to 0, ZeRO is disabled and the execution follows the regular distributed data parallel workflow:

```json
{
  "zero_optimization": { "stage": 3 }
}
```

CPU offloading in DeepSpeed is called [**ZeRO-Offload**](https://www.deepspeed.ai/tutorials/zero-offload/), a system for offloading optimizer and gradient states to CPU memory. On top of stage 3, DeepSpeed also provides [**ZeRO-Infinity**](https://arxiv.org/abs/2104.07857), also an offloading engine, which extends ZeRO-Offload with support for NVMe memory. According to the [ZeRO-3 documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html#zero), "ZeRO-Infinity has all of the savings of ZeRO-Offload, plus is able to offload more the model weights and has more effective bandwidth utilization and overlapping of computation and communication". They can be enabled via:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu", "pin_memory": true }
  }
}
```

Once the config file is calibrated, the implementation is straightforward. DeepSpeed manages internally the boilerplate that PyTorch requires for parallelism and data loading, so the setup reduces to:

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

The training loop follows the same structure as the PyTorch implementation. The only difference is that gradients are not zeroed explicitly, as DeepSpeed handles this internally. Also, `initialize()` returns a `train_dataloader` that already assigns disjoint subsets of data to each process, so no distributed sampler is required.

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

The run is launched with `torchrun` as before, or with the DeepSpeed launcher:

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

### Detour: measuring memory allocated to parameters

The [DeepSpeed API can estimate the memory requirements of model parameters](https://deepspeed.readthedocs.io/en/latest/memory.html#api-to-estimate-memory-usage) for each ZeRO stage. The estimates are printed by calling the following method at the start of the execution:

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
- the base model requires about 0.323GB per GPU to store its parameters;
- DeepSpeed ZeRO-2 requires 0.161GB and 0.484GB with and without offloading, respectively;
- DeepSpeed ZeRO-3 requires 0.009GB and 0.190GB with and without offloading, respectively.

This metric is cheap to compute and gives a quick overview of scaling. However, it only measures parameters. It ignores activations, residual buffers (e.g. normalization variables), temporary communication buffers, the batch size, the numerical precision and all other config fields.

## Results

Performance is measured with the DeepSpeed logger, which reports three metrics every 10 steps: model throughput (average number of samples per second), average allocated memory, and maximum allocated memory. All runs use `pytorch==2.0.1`, CUDA `11.7` and `deepspeed==0.10.3`.

All implementations use the same mixed-precision representation, communication bucket sizes and other config parameters. The benchmark covers the following implementations (and configs):

1. The distributed data parallel (DDP) implementation, i.e. no DeepSpeed (<a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> with `'stage': 0`);
2. The fully-sharded data parallel implementation with ZeRO 1, 2 and 3 (<a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> with `'stage' :1`, `2` or `3`);
3. The ZeRO-3 implementation with ZeRO-Infinity for CPU offloading (<a href="{{ site.assets }}/distributed-training/ds_config_offload.json">`ds_config_offload.json`</a>).

The results for the GPTlite model with a micro-batch size of 1 sample per GPU are:

{: .picture}
<img width="99.7%" height="99.7%" src="{{ site.assets }}/distributed-training/imgs/benchmark_gptlite.svg"/>

{: .caption}
Maximum and average memory allocated, and throughput, of GPTlite (GPT-3 small) with distributed data parallelism, DeepSpeed ZeRO-1 to ZeRO-3, and ZeRO-3 with CPU offloading, at a micro-batch size of 1 sample per GPU.


**Memory overhead from communication buffers.** In theory, high ZeRO stages have a much higher max memory than low stages and DDP, as they communicate more parameters and need more communication buffers. Small communication bucket sizes in the config file remove this effect. With the default bucket size (`5e9`), memory usage roughly doubles in stages 2 and 3, and some runs exceed the available memory.

**Communication vs computation trade-off across ZeRO stages.** Moving from DDP to ZeRO-1, ZeRO-2, ZeRO-3 and ZeRO-Infinity reduces both memory consumption and throughput. This is the expected trade-off: data locality is exchanged for the communication (or offloading) of parameters, at a cost in performance.

**Offloaded vs in-memory parameters.** Offloading consistently reduces memory usage, with a small reduction in throughput, as expected.

**Lack of superlinear speedup.** The gain in memory efficiency is small, and far from the results reported by the DeepSpeed team. The likely cause is the network size: 8 GPUs here, against 64 to 800 GPUs in the original benchmarks. Their superlinear speedup also relies on a large network, and is not observed here.

Finally, **communication quantization** is not used, as it did not yield any improvement. This is in line with the ZeRO++ paper, which states that the method targets small batch sizes or low-bandwidth networks.

## Resources and code

This post covers only data parallelism. For further reading on DeepSpeed, see the [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest), the [training features page](https://www.deepspeed.ai/training/#features), the [tutorials page](https://www.deepspeed.ai/tutorials/), the [HuggingFace page for DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed), and the examples at [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/).

The code for all experiments is available in the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training).



