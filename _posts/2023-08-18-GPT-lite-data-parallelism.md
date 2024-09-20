---
layout: post
title:  "Distributed GPT model: data parallelism, sharding, offloading, activation checkpointing, and communication quantization via DeepSpeed"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

Previously, in [Building a GPT model from scratch]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), we built GPT-lite, the small variant of the [GPT-2 model](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). In this post, we will perform distributed data parallelism on the training of the GPT model, on a network of 8 GPUs, using PyTorch's [DistributedDataParallel module](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer). DeepSpeed is a lightweight wrapper on PyTorch, and can be installed by the `deepspeed` package for `python`.


## About Data Parallelism

Data parallelism refers to the parallel execution of different input samples across processors. There are two main approches.

**Distributed Data Parallel** keeps a full copy of the model (weights, optimizer parameters and gradients) in all processors. All models are initialized equally. Each processor takes as input to its model a different minibatch and performs a forward pass to compute the loss. On the backward pass, at every layer of the model, each processor computes its own gradients for its batch, and mean-reduces across all processors. This leads to all processors having then the same weight updates, keeping the model in sync throughout execution. 

**Fully-Sharded Data Parallelism (FSDP)** a.k.a **sharding** where processors dont hold a full copy of the model, but only the parameters, optimizer states and gradients to different/distinct subsets of layers. Different processors input different mini-batches, and there is no sharding of activations i.e. they are kept fully on each processor (with activations that refer to the corresponding input). In DeepSpeed lingo, FSDP is called **ZeRO (Zero Redundancy Optimizer**. ZeRO has several alternative execution modes, called **stages**. Each stage represents a different level of memory redundancy, corresponding to different variables being communicated or not. These are enabled cumulatively. In practice, by increasing the stage we define the tradeoff between memory usage and communication:
- **ZeRO stage 0** is equivalent to no distributed model variables, and to Distributed Data Parallelism;
- **ZeRO stage 1 (ZeRO-1)**: the optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition. Affects backward pass runtime.
- **ZeRO stage 2 (ZeRO-2)**: the reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states. Also relevant only for the backward pass.
- **ZeRO stage 3 (ZeRO-3)**: the 16-bit model parameters are partitioned across the processes. Includes extra communication on **both** the forward and backward passes. 

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/GPT-lite-distributed/DeepSpeed_stages.png"/>

{: style="text-align:center; font-size: small;"}
Memory consumption of the three different stages of ZeRO FSDP. Residual memory (activations, normalization layers, etc) is not included as FSDP does not shard them. Source: [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

Additionaly, on top of stages 1 and 2, we can enable **[ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/), a system for offloading optimizer and gradient states to CPU memory**. On top of stage 3, we can enable [**ZeRO-Infinity**](https://arxiv.org/abs/2104.07857), also an offloading engine that extends ZeRO-offload with support to NVMe memory. According to the [ZeRO-3 documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html#zero), "ZeRO-Infinity has all of the savings of ZeRO-Offload, plus is able to offload more the model weights and has more effective bandwidth utilization and overlapping of computation and communication".

## Model and dataset setup

We start by taking our previous *GPT-lite* implementation and matching the architecture of the model to the *GPT-2 Small* model description in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Fig 2.1):

```python
## gptlite.py

# depth of the network as number of decoder blocks.
n_layer = 12

# size of the embeddings (d_model)
n_embd = 768

# number of attention heads in the Multi-Attention mechanism
n_head = 12

# block size ie max number of training sequence, the $n_{ctx}$ in the paper .
block_size = 2048

# dropout rate (variable p) for dropout units
dropout = 0.1
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

Note that code this code is applicable to any model of type `torch.nn.Module` and any dataset of type `torch.utils.data.Dataset`. As an example. jf you'd want to perform a multi-class classification using the `ResNet` network on the `CIFAR10` dataset available in `torchvision`, you'd define the previous 2 methods as:

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

## Main code

We start integrating DeepSpeed in our code by creating the `ArgumentParser` object that is required by the `initialize()` method in DeepSpeed. The `ArgumentParser` object must contain:
- the `--local_rank` parameter that is the local rank of each process in the network, and will be populated automatically by the `deepspeed` launcher when launching a script;
- optionally, we add the `--deepspeed_config` where we specify the path to the DeepSpeed config file. If you choose not to add it to the command line arguments, then it must be specified as the parameter `config` in the call to `deepspeed.initialize()`.

The most correct way to do this is to call `deepspeed.add_config_arguments()`, that adds the `--deepspeed_config` and other DeepSpeed-specific arguments:

```python
## train.py

import deepspeed

def get_cmd_line_args(description='GPT-lite on DeepSpeed'):
  import argparse
  parser = argparse.ArgumentParser(description=description)
  # Include DeepSpeed configuration arguments (--deepspeed, --deepspeed_config)
  parser = deepspeed.add_config_arguments(parser)
  return parser.parse_args()
```

The bulk of the code is pretty simple. In practice, all boilerplate code that PyTorch requires for optimizers, learning rates, parallelism, data loaders etc, are all managed by DeepSpeed and are defined in its config file. So the initialization of a DeepSpeed run is pretty straightforward:

```python
## train.py

def main_deepspeed(n_epochs=100, random_seed=42):

  torch.manual_seed(random_seed)  #set random seed (used by DataLoader)
  deepspeed.runtime.utils.set_random_seed(random_seed) #set DeepSpeed seed
  deepspeed.init_distributed()  # initialize distributed DeepSpeed
  args = get_cmd_line_args()  # initialize command line arguments parser
  criterion = torch.nn.CrossEntropyLoss()  # initialize loss function
  train_dataset, _, vocab_size = gptlite.get_dataset()  # initializer dataset
  model = gptlite.get_model(vocab_size)  # initialize model

  engine, optimizer, train_dataloader , _ = deepspeed.initialize(
    args=args, model=model, training_data=train_dataset,) # initialize deepspeed
```

We then write the training loop, with a structure very similar to a PyTorch implementation. The only exception is that we don't perform zeroing of gradients, as this is managed internally by DeepSpeed. Also, `initialize()` already returns a `train_dataloader` with an internal  `torch.utils.data.distributed.DistributedSampler` that assigns disjoiint subsets of data to each process.

```python
## train.py

def main_deepspeed(n_epochs=100, random_seed=42):
  # ...
  for epoch in range(n_epochs):
    for step, data in enumerate(train_dataloader):
      inputs = data[0].to(engine.device)
      labels = data[1].to(engine.device)
              
      outputs = engine(inputs)  # fwd pass
      loss = criterion(outputs, labels)
      engine.backward(loss)  # backprop
      engine.step()  # update weights, no need for zero-ing

  # print loss for epoch
  if engine.local_rank == 0: print(f"Epoch: {epoch}, Loss: {loss}")
```
 
## Config file

The *nuance* in using DeepSpeed is the `.json` config file. The number of possible optimizations is large, as it defines parallelism, floating point precision, logger, communication parameters, etc. These fields are detailed in the [DeepSpeed config documentation](https://www.deepspeed.ai/docs/config-json/). Here we start with a simple config, where the configure the DeepSpeed logger to output memory and throughput info at every 10 epochs (`steps_per_print`), and define the settings of the optimizer (`optimizer`) and learning rate scheduler (`scheduler`):

```json
{
  "train_batch_size": 256,
  "steps_per_print": 10,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 0.001,
      "warmup_num_steps": 1000
    }
  }
}
```


**ZeRO Fully-Sharded Data Parallel** can be activated by specifying the relevant stage in the config file. If omitted, or when passing the stage 0, DeepSpeed is disabled and the execution follows a regular distributed data paralllel workflow:

```json
{
  "zero_optimization": {
    "stage": 3
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

Note related to the implementation of communication and computation in the [pytorch DDP implementation](https://pytorch.org/docs/master/notes/ddp.html#internal-design): the backward pass iteratively computes gradients (from last to first layer) and collects blocks of gradients to be communicated. These blocks will be mean-reduced asynchronously during the backward pass, while the computation for the backward pass proceeds. Therefore it overlaps backward pass computation with gradients communication. At the end of the backward pass, all GPUs wait for all gradient all-reduces to finish, and then triggers the parameter updates.

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

[**ZeRO-Infinity**](https://arxiv.org/abs/2104.07857) performs offloading of several variables in memory to CPU and VNMe for huge memory savings. It is only compatible with ZeRO-3 and can be enabled with: 

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
  }
}
```

As a side note, offloading of tensors can also be achieved via pytorch by using custom [hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html).

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

As a final note, the configuration file can also be extended with custom fields, that are e.g. specific to application or hardware, but for brevity we'll omit those details here. 

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

Finally, **activation checkpoint currently has two implementations: a reentrant and non-reentrant**. The non-reentrant will be the future default in pytorch and is implemented via pytorch saved variable hooks (as detailed [here](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d)). Non-checkpointed activations are not stored in memory, and instead replaced by a reference. Thus, the computation graph is not altered. The non-reentrant checkpointing allows for nested checkpointing (calling one checkpoint from another checkpoint function), allowing for **higher memory savings**.
The non-reentrant equivalent in deepspeed in implemented by [`deepspeed.checkpointing.non_reentrant_checkpoint`](https://github.com/microsoft/DeepSpeed/blob/42a8eaa705ed30b4a656ac71bdb400772df2cb21/deepspeed/runtime/activation_checkpointing/checkpointing.py).

The reentrant does not use hooks but calls the [`forward` autograd function](https://github.com/pytorch/pytorch/blob/670c5cf96249db28cde757da5a6aa97569760102/torch/utils/checkpoint.py#L75) instead. The gradient calculations are not part of the main computational graph anymore, and every checkpoint creates a mini-computational graph during the backward pass. One of the downsides, is that the whole `forward` function is computed for every call, while the non-reentrant counterpart can stop when the relevant activations are computed. Moreover, the whole graph is not stored (contrarily to non-reentrant) thus not allowing the backward to be run in the whole computational graph. More details in the [torch checkpoint documentation](https://pytorch.org/docs/stable/checkpoint.html).


### About activation checkpointing with parameters sharding

Combining activation checkpointing with sharded model parameters (ZeRO stage-3) may lead to a substantial runtime overhead. The problem is that, if you need to perform a forward pass from the closest checkpoint layer to collect the parameters required for the back propagation, and if those parameters are distributed (stage 3), then there has to be an extra collective communication step at every layer (from checkpoint layer to current back-prop layer) to collect those weights. This adds an extra communication overhead.

## Launching a distributed execution

The installation of DeepSpeed includes the `deepspeed` launcher, a network bootstrapper that spaws a python script across compute nodes and GPUs, with different `--local_rank` argument and different environment variables for the *comm world*. In our example, to launch the script `train.py` on a compute node with 8 GPUs, with the DeepSpeed config file `ds_config.json`, we run on the shell:

```shell
$ deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
```

Run `deepspeed --help` for a brief summary of the launcher options. With `torchrun`, it can be launched with:
```shell
$ torchrun --standalone --nproc_per_node=8 train.py --deepspeed --deepspeed_config ds_config.json --no_local_rank
```

and on a slurm-cluster execution, with:
```shell
slurm-torchrun --torch-script-path="train.py"  \
  --torch-script-extra-args="--deepspeed --deepspeed_config ds_config.json --no_local_rank"
```

Now that in distributed runs, the batch size should take into consideration the number of compute nodes, the number of GPUs, and the number of gradient accumulation steps or micro-batch size (when applicable). In brief, each process needs at least 1 input sample and:

```
batch_size = micro_batch_size_per_gpu * num_gpus * num_nodes * gradient_accumulation_steps
```

## Detour: measuring memory allocated to parameters

We can use the [DeepSpeed API to estimate the memory requirements of model parameters](https://deepspeed.readthedocs.io/en/latest/memory.html#api-to-estimate-memory-usage) for different ZeRO implementations, by calling the following method at the onset of execution:

```python
## train.py

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

**Activation checkpointing.** On the GPT-lite model, the main memory driver is the activations memory on the attention matrix (grows quadratically with the sequence length and linearly with the batch size). Therefore, sharding alone does not yield a signification memory reduction. Adding activation checkpointing overcomes this memory bottleneck by keeping at most one attention matrix in memory (recomputed on the fly), throughout the whole backward pass. Moreover, **mixed precision** has an important effect on throughtput as lower precision yields faster communication and computation. As an example, the results for ZeRO-1 with activation checkpointing and mixed precision are:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/benchmark_gptlite_activation_ckpt_throughput.png"/>

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/benchmark_gptlite_activation_ckpt_memory_usage.png"/>

**Parameter vs residual memory.** Note the difference between average memory and maximum memory. That gap in memory consumption is due to temporary memory dedicated to activations, residual buffers, communication buffers, etc. 

**Communication vs computation trade-off from different stages in ZeRO.** In ideal scenarios, as you move from DDP to ZeRO-1, ZeRO-2, ZeRO-3 and ZeRO-Infinity, the memory consumption and throughput are reduced. As expected, we swap data locality for communication of parameters, and pay a price in performance for the communication/offload of parameters.

**Offloaded vs in-memory parameters.** Offloading proved to be a consistent way to reduce memory usage with the drawback of a small reduction of throughput, as expected.

**Lack of superlinear speedup**: We observed a small improvement of memory efficiency, but still far from the claims of the original DeepSpeed team. One explanation is that we used a small network of 8 GPUs, compared to the 64 to 800 GPUs used by the authors in their benchmarks, therefore we achieved a much smaller memory reduction. A large network of GPUs also underlies their claim of superlinear speed up that we did not observe.

Finally, we did not use **communication quantization** as did not result in any improvement. This goes in line with the ZeRO++ paper where it claims this method is applicable for small batch sizes or low-latency / low-bandwidth networks.

## Resources and code

In this post we explored only the dimension of data parallelism.  If you'd like to know more about DeepSpeed, check the [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest), the [training features page](https://www.deepspeed.ai/training/#features), the [tutorials page](https://www.deepspeed.ai/tutorials/), the [HuggingFace page for DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed), and the examples at [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/).

There are a lot of results and food for thought here, so I will update this post as I find new insights. Meanwhile, if you want to try this on your own, see the [GPT-lite-distributed repo](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-distributed).



