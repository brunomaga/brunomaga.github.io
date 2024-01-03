---
layout: post
title:  "Distributed training of a GPT model with DeepSpeed, FSDP, offloading, pipelining, activation checkpointing and communication quantization"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

Previously, in the [AI Supercomputing]({{ site.baseurl }}{% post_url 2020-05-12-AI-Supercomputing %}) and [AI Supercomputing (part 2)]({{ site.baseurl }}{% post_url 2020-05-28-AI-Supercomputing-2 %}) posts, we summarized existing Machine Learning (ML) parallelism techniques. Later, in [Building a GPT model from scratch]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), we built GPT-lite, the small variant of the [GPT-2 model](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). In this post, we will perform large-scale parallel training of a GPT model and a large DNN on a network of 8 GPUs, using [DeepSpeed and ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer). The DeepSpeed API is a lightweight wrapper on PyTorch, and can be installed by the `deepspeed` package for `python`.

## 3D Parallelism

An ML model allows for three types of parallelism, that can be combined into what we call **3D parallelism**:
1. **(Distributed) Data parallelism (DDP)**, by dividing the number of samples (batch size) across processors, and keeping copies of models across processors in sync, by using the average of the gradients across processors to perform the model updates.
2. **Pipeline parallelism**, by delegating different layers (or blocks of layers) of the model to different processors.
3. **Model parallelism**, by dividing the *tensors* on each layer across processors.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPT-lite-DeepSpeed/GPT_3D_parallelism_2.png"/>

{: style="text-align:center; font-size: small;"}
The 3D parallelism aims and partitioning (color-coded) computer resources  across the 3D space of data, layer and parameter dimensions. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

Model parallelism refers mainly to two approaches:
- **ZeRO (Zero-Redundancy Optimizer)** implements **Fully-Sharded Data Parallelism (FSDP)** and is similar to data parallelism but includes also distributed storage of the model parameters. While in data parallelism, all processors hold a syncronized clone of the full model, in ZeRO the tensors for the parameter, gradients and optimizer states are distributed/partitioned/**sharded**. This requires a communication overhead due to the scatter/gather operations required to communicate parameters across processors, at every layer. Therefore, ZeRO provides memory savings compared to data parallelism because of the partitioning of the model parameters. An important remark is that the activations on the forward and backward passes still happen in full form i.e. they are not distributed and they are kept on all processors for the backpropagation to work;
- **Tensor parallelism**, **vertical parallelism**, **intra-layer parallelism** or sometimes simply **model parallelism**, partitions  also the computation of activations in the forward and backward passes. This requires a modification of the workflow of the computation in order to work in a distributed manner, particularly on the matrix multiplications format, e.g. all-gather, all-reduce, scatter-reduced distributed matrix mult, etc. Therefore, it is model-specific, and is [supported but not provided by DeepSpeed](https://www.deepspeed.ai/training/#support-for-custom-model-parallelism), except in some built-in implementations such as [Megatron-ML](https://www.deepspeed.ai/tutorials/megatron/);

### ZeRO's FSDP

**ZeRO has three alternative execution modes, called stages**. Each stage represents a different level of memory redundancy, corresponding to the partitioning of optimizer states, gradients, and parameters, respectively. These are enabled cumulatively. In practice, by increasing the stage we define the tradeoff between memory usage and communication:
- **ZeRO stage 1 (ZeRO-1)**: the optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.
- **ZeRO stage 2 (ZeRO-2)**: the reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.
- **ZeRO stage 3 (ZeRO-3)**: the 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes. 

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/GPT-lite-DeepSpeed/DeepSpeed_stages.png"/>

{: style="text-align:center; font-size: small;"}
Memory consumption of the three different stages of ZeRO FSDP. Residual memory (activations, normalization layers, etc) is not included as FSDP does not shard them. Source: [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

### CPU/NVMe offloading

Additionaly, on top of stage 1 and 2, we can enable **[ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/), a system for offloading optimizer and gradient states to CPU memory**. On top of stage 3, we can enable [**ZeRO-Infinity**](https://arxiv.org/abs/2104.07857), also an offloading engine that extends ZeRO-offload with support to NVMe memory. According to the [ZeRO-3 documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html#zero), "ZeRO-Infinity has all of the savings of ZeRO-Offload, plus is able to offload more the model weights and has more effective bandwidth utilization and overlapping of computation and communication".

### Communication quantization

Finally, we can **optimize/compress communication with [ZeRO++](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/)**. To understand ZeRO++'s gains, we should undertand the communication workflow first (from the [ZeRO++ paper](https://arxiv.org/abs/2306.10209)): "Assume the model size as 𝑀. During the forward pass, ZeRO conducts an all-gather operation to collect all the parameters (𝑀) needed to train for all model layers. In the backward pass, ZeRO re-collects parameters (𝑀) with all-gather first, then each GPU can compute local gradients. After that, ZeRO operates reducescatter function to aggregate and redistribute gradients (𝑀) across accelerators. In total, ZeRO has a total communication volume of 3𝑀, spreads evenly across 2 all-gather and 1 reduce-scatter."

ZeRO++ introduces three new communication improvements:
1. **Quantized Weight Communication for ZeRO (qwZ)**: perform block quantization of the forward all-gather, converting weights  from `FP16` (2 bytes) to `INT8` (1 byte). The main improvement is to replace the typical quantization algorithm (multiplying all parameters by a scalar), by a quantization per block (ie per parameter subset) that includes multiplication by a factor and shifting values by another factor;
2. **Hierarchical Weight Partition for ZeRO (hpZ)**: data remapping that trades-off communication for more memory and reduces communication overhead of all-gather on weights during backward. Instead of having weights distributed across GPUs, we maintain a full copy on each machine, allowing us to replace the expensive cross-machine all-gather on weights with a faster intra-machine all-gather.
3. **Quantized Gradient Communication for ZeRO (qgZ)**: replaces the gradients reduce-scatter collective, by doing (1) block-based quantization of gradients to `INT4` during communication to reduce the communication size, and recovering the full precision before the reduction operator to preserve training accuracy.

ZeRO++ is particularly relevant for clusters with a low-latency network where collective communications are responsible for a large fraction of the overall runtime. It is also important for executions with a small batch size per GPU, where the memory increase of **qgZ** has no impact on scaling.


We will see in this post that finding the optimal parallelism hyperparameters is a hard problem. This is a resources allocation problem across the 3D volume in the data, parameters and layers (pipeline) space. It aims at allocating different partitions on that 3D space to different processors, in a way that best balances the compute time or memory across resources. In practice, balanced computation yields a low overall runtime, and balanced memory allows for an increase of the maximum model size.
 
## Model and dataset

The code that follows is applicable to any model of type `torch.nn.Module` and any dataset of type `torch.utils.data.Dataset`. So we will detail three use cases: an advanced use case, specific to a large language model (GPTlite), an out-of-the-box [pre-defined model from torchvision](#torchvision-model) and a [simple DNN model](#benchmark-model) of arbitrary width and depth used to simulate different ML workload conditions (we will call this our benchmark model).  

### GPTlite

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

### Using a torchvision model {#torchvision-model}

If you'd want to perform a multi-class classification using the `ResNet` network on the `CIFAR10` dataset available in `torchvision`, you'd define the previous 2 methods as:

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

### Benchmark model {#benchmark-model}

If we'd want instead to test the response of DeepSpeed scaling of a very simple model of varying width and depth, we could create a **benchmark model** which is simply a DNN of `L` layers of width `W`, for multi-label classification, whose objective is to compute the modulo of the sum of squares of a random input vector:

{: style="text-align:center; font-size: small;"}
<img width="22%" height="22%" src="/assets/GPT-lite/benchmark_model.png"/>

{: style="text-align:center; font-size: small;"}
The *benchmark model*, a DNN with L layers of dimensionality W (right)

The implementation of the benchmark model in `benchmark.py` is straightforward:

```python
## benchmark.py 

class BenchmarkModel(nn.Module):
  """" DNN with L layers and W neurons per layer """

  def __init__(self, W, L, in_size, out_size):
    super(BenchmarkModel, self).__init__()
    self.layers = [nn.Linear(in_size, W), nn.ReLU()]
    for _ in range(L-2):
      self.layers += [nn.Linear(W, W), nn.ReLU()]
    self.layers += [nn.Linear(W, out_size), nn.ReLU()]
    self.layers = nn.Sequential(*self.layers)

  def forward(self, x):
    return self.layers(x)


class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, in_size, out_size, len=2**16):
      self.in_size = in_size
      self.len = len
      self.out_size = out_size

    def __len__(self):
      return self.len

    def __getitem__(self, _):
      x = torch.Tensor(self.in_size).uniform_(-10, 10)
      y = int( x @ x % self.out_size)
      return x, torch.tensor(y, dtype=torch.long)


get_dataset = lambda W: BenchmarkDataset(W), BenchmarkDataset(W)
get_model = lambda W, L: BenchmarkModel(W, L)
```

We will call this the **Benchmark Model** and we will use it later in our benchmark section to test DeepSpeed's response to models of varying width and depth..

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
  # mandatory argument for calls with deepseed
  parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
  # Include DeepSpeed configuration arguments (--deepspeed, --deepspeed_config, ...)
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

We then write the training loop, with a structure very similar to a PyTorch implementation. The only exception is that we don't perform zeroing of gradients, as this is managed internally by DeepSpeed. Also, `train_dataloader` is of type `torch.utils.data.distributed.DistributedSampler` and created automatically by the `initialize()`, so multi-process runs will have each process automatically delegated to a different subset of data.

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

The real *nuance* and complexity in using DeepSpeed is the `.json` config file. The number of possible optimizations is large, as it defines parallelism, floating point precision, logger, communication parameters, etc. These fields are detailed in the [DeepSpeed config documentation](https://www.deepspeed.ai/docs/config-json/). Here we start with a simple config, where the configure the DeepSpeed logger to output memory and throughput info at every 10 epochs (`steps_per_print`), and define the settings of the optimizer (`optimizer`) and learning rate scheduler (`scheduler`):

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

**Gradient accumulation** based on **micro-batching** is a technique that simulates a large mini-batch as an iteration across several micro-batches. This is particularly relevant when the whole mini-batch does not fit into memory, and using an accumulation of micro-batches will overcome that limitation. This method is enabled by setting `train_micro_batch_size_per_gpu` (defaulted to `train_batch_size`) or `gradient_accumulation_steps` (defaulted to `1`). At runtime, the micro-batch size can be retrieved by `engine.gradient_accumulation_steps()`. In our case, we will start with a micro-batch of 1 single input per GPU, that accummulate up to a batch size of 256 across all 8 GPUs, therefore resulting in 32 gradient accumulation steps: 

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 1
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

[**ZeRO++**](https://arxiv.org/abs/2306.10209) was detailed above and allows for communication reduction via compression, quantization and memory tradeoff. It can be enabled by [3 independent components](https://www.deepspeed.ai/tutorials/zeropp/#three-components-of-zero) -- hierarchical Weight partition for ZeRO (hpZ), quantized weight communication for ZeRO (qwZ) and quantized gradient Communication for ZeRO (qgZ) -- enabled in the same order by:
```json
{
  "zero_hpz_partition_size": 8, 
  "zero_quantized_weights": true,
  "zero_quantized_gradients": true,
}
``` 
Note that the according to documentation, the ideal value for `zero_hpz_partition_size` is the number of ranks (GPUs) per node. 

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

As a final note, the configuration file can also be extended with custom fields, that are e.g. specific to application or hardware, but for brevity we'll omit those details here. 

## Pipeline parallelism

[Pipeline parallelism](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules) improves both the memory and compute efficiency during training by partitioning the layers of a model into stages that can be processed in parallel. The pipeline parallelims implemented in DeepSpeed is the [PipeDream-Flush implementation with default 1F1B scheduling](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/) (1 Forward pass followed by 1 Backward pass, Figure 4 top on the [Megatron LM paper](https://browse.arxiv.org/pdf/2104.04473.pdf) ), however it is possible to [extend pipeline parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule) to other algorithms.

{: style="text-align:center; font-size: small;"}
<img width="79%" height="80%" src="/assets/GPT-lite-DeepSpeed/GPT_pipelining.png"/>

{: style="text-align:center; font-size: small;"}
Two-way data parallel pipelines with four stages each. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

We will make pipeline parallelism optional in our use case, as in many cases (e.g. for small models) it is not benefitial. We will enable it by passing the number of stages as the `---pipeline_num_stages` argument (default: 0, no pipelining) on the command line:

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
<img width="80%" height="80%" src="/assets/GPT-lite-DeepSpeed/GPT_pipelining_2.png"/>

{: style="text-align:center; font-size: small;"}
"An illustration of how DeepSpeed will train a batch with eight micro-batches using hybrid two-way data parallelism and two-stage pipeline parallelism. GPUs 0 and 2 are arranged in a pipeline and will alternate forward (F) and backward (B) passes. They will then all-reduce (AR) gradients with their data parallel counterparts, GPUs 1 and 3, respectively. Finally, the two pipeline stages update their model weights". This is the 1F1B pipeline algorithm. Source: [DeepSpeed pipelining documentation](https://www.deepspeed.ai/tutorials/pipeline/)

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

As a side note, we will not tune the [load balancing method for pipeline modules](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules) and will use the default `partition_method=parameters`. This assigns layers to stages in a way to load-balance the parameters, i.e. stages may have different lengths.

Finally, in the extreme case the 1F1B algorithm is not the pipeline algorithm we want, we can [extend pipeline parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule) with a different algorithm. 

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

When not using pipelining, we have to manually specify which layers to checkpoint, by calling [`deepspeed.checkpointing.checkpoint`](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html#using-activation-checkpointing) at the checkpoint layers. We will use the previous `lo_layers()` method to iterate over the layers of a model and assign the relevant checkpointing in the `forward()` pass of `GPTlite` as:

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

## Launching a distributed execution

The installation of DeepSpeed includes the `deepspeed` launcher, a network bootstrapper that spaws a python script across compute nodes and GPUs, with different `--local_rank` argument and different environment variables for the *comm world*. In our example, to launch the script `train.py` on a compute node with 8 GPUs, with the DeepSpeed config file `ds_config.json`, we run on the shell:

```shell
$ deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
```

Few notes about distributed executions:
- `--num_gpus` is optional: if not provided, it will default to the available GPUs returned by the cuda toolkit;
- launching with `python` instead of `deepspeed` will perform a single-node single-GPU run;
- if we were required to run this on multiple compute nodes, we'd need to pass an extra parameter `--hostfile hostfile`, where `hostfile` is an MPI-style descriptor file of nodes and gpus per node;
- the batch size should take into consideration the number of compute nodes, the number of GPUs, and the number of gradient accumulation steps or micro-batch size (when applicable). In brief, `train_batch_size` must be equal to `train_micro_batch_size_per_gpu` * `gradient_accumulation_steps` * `--num_gpus`. Also, each process needs at least 1 input sample.

For more information on available flags, running `deepspeed --help` provides a brief summary of all options.

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

However, when activating pipelining, by launching the run with `--pipeline_num_stages 4 --pipeline_spec_layers`:
- the base model requires 0.053GB for the parameters; 
- ZeRO-2 requires 0.026GB and 0.079GB for the with and without offloading use cases;
- ZeRO-3 requires 0.009GB and 0.038GB of memory, with and without offloading, respectively; 

This metric is very useful as it gives a quick overview of scaling and is very fast to compute. However, it has many fallacies: it only measures the parameters overheard, it does not take activations or other residual buffers (e.g. normalization variables) into account, does not take the batch size and numerical precision (or any field in the config file) into account, it does not consider temporary (e.g. communication) buffers, etc. Also, the pipeline metrics are not accurate due to pipeline parallelism not being compatible with ZeRO stages 2 or 3.  


## Results

To measure our performance, we used the deepspeed logger to extract the following metrics from different runs at every 10 steps: model throughput as average number of samples per second, the average allocated memory, and the maximum allocated memory. We used `pytorch==2.01`, CUDA `11.7` and `deepspeed==0.10.3`.

All implementations use the same mixed-precision representation, communication bucket sizes, disabled communication quatization (ZeRO++) and other config parameters. We benchmarked the following implementations (and configs):

1. The distributed data parallel (DDP) implementation, i.e. no DeepSpeed (<a href="/assets/GPT-lite-DeepSpeed/ds_config.json">`ds_config.json`</a> with `'stage': 0`);
2. The fully-sharded data parallel implementation with ZeRO 1, 2 and 3 (<a href="/assets/GPT-lite-DeepSpeed/ds_config.json">`ds_config.json`</a> with `'stage' :1`, `2` or `3`);
3. The ZeRO-3 implementation with ZeRO-Infinity for CPU offloading (<a href="/assets/GPT-lite-DeepSpeed/ds_config_offload.json">`ds_config_offload.json`</a>);
4. The ZeRO-3 implementation without activation checkpointing and with activation checkpointing at every block. 
5. The memory-efficient pipeline implementation with ZeRO-1 (<a href="/assets/GPT-lite-DeepSpeed/ds_config.json">`ds_config.json`</a> with `'stage': 1` and launch with `--pipeline_num_stages <num_stages> --pipeline_spec_layers`) without activation checkpointing (due to bug [4279](https://github.com/microsoft/DeepSpeed/issues/4274)), where we tested 1, 2, 4 and 8 pipeline stages per run. We rely on the default DeepSpeed algorithm for load balancing of stages, based on the parameter count. As an example, for the partitioning of GPT-lite pipeline across 8 GPUs and 4 stages, it outputs:
   ```
RANK=0 STAGE=0 LAYERS=4 [0, 4)   STAGE_PARAMS=21256704 (21.257M)
RANK=2 STAGE=1 LAYERS=3 [4, 7)   STAGE_PARAMS=21256704 (21.257M)
RANK=4 STAGE=2 LAYERS=3 [7, 10)  STAGE_PARAMS=21256704 (21.257M)
RANK=6 STAGE=3 LAYERS=6 [10, 16) STAGE_PARAMS=21308160 (21.308M)
   ```

We tested three models. The first is a *wide* version of our benchmark model, with a high parametric space and a small layer count (`W=8192`, `L=3`), and input and output of size 8192. We used a batch size of $$2^{14}$$, and a micro-batch size of $$2^{11}$$ inputs per GPU, ie `'train_batch_size': 16384` and `'train_micro_batch_size_per_gpu': 2048`. The benchmark results are:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-DeepSpeed/benchmark_wide.png"/>
 
Then we tested a *deep benchmark model* with a small parameter space (`W=256`), a high layer count (`L=2048`), an input and output size of 256, and the same bath sizes as the previous wide model:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-DeepSpeed/benchmark_deep.png"/>

And finally our GPT-lite model, with a micro-batch size of 1 sample per GPU:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-DeepSpeed/benchmark_gptlite.png"/>

**Memory overhead from communication buffers.** Looking at the max vs average memory, note that the max memory in theory should be much higher at high ZeRO stages compared to low ZeRO stages and DPP. This is due to more parameters being communicated requiring more communication buffers. However, setting the communication bucket sizes to a low value in the config file overcomes this effect. In fact, we also benchmarked several runs with the default communication bucket sizes (`5e9`) and it led to a higher memory usage as expected (of approximately double the amount in stages 2 and 3), that became prohibitive for some runs.

**Parameter vs residual memory.** Note the difference between average memory and maximum memory. That gap in memory consumption is due to temporary memory dedicated to activations, residual buffers, communication buffers, etc. 

**Communication vs computation trade-off from different stages in ZeRO.** In ideal scenarios, as you move from DDP to ZeRO-1, ZeRO-2, ZeRO-3 and ZeRO-Infinity, the memory consumption and throughput are reduced. As expected, we swap data locality for communication of parameters, and pay a price in performance for the communication/offload of parameters. This is the pattern observed in the deep benchmark and GPT-lite models. However, the wide benchmark model does not respond similarly, as from stage 2 to stage 3 there is an increase in throughput. I believe this is due to ZeRO-3 distributing the fp16 model parameters, leading to a distributed parallelism of the very large sums of squares per layers.

**Offloaded vs in-memory parameters.** Offloading proved to be a consistent way to reduce memory usage with the drawback of a small reduction of throughput, as expected.

**Activation checkpointing** tested on GPTlite trades runtime for memory usage. It yielded a 4x memory reduction at the overhead of +20% in execution time.

**Pipelining performance for variable number of stages.** Increasing the number of stages strongly decreases the average memory consumption, as expected. This is due to the model being partitioned in smaller blocks. There was no substantial decrease in maximum memory consumption, and this is something I am yet to understand (any hints?). The throughput demonstrated a peculiar behaviour: in the deep benchmark model, the throughput increases with the increase of stage count, while the opposite happens on the GPT-lite model. I believe this is due to load imbalance across stages, or a lower ratio of computation vs communication as we increase the stage count on the GPT-lite use case.

**Pipelining with optimized vs non-optimized memory efficiency implementation.** Using the `SpecLayer`-based implementation of the `PipelineModule` in our pipeline runs, resulted in a reduction of about 40% in memory consumption for the GPT-lite and deep benchmark models, when running pipeline parallelism with the highest stage count (8).

## General disccusion

We observed a small improvement of memory efficiency, but still far from the claims of the original DeepSpeed team. One explanation is that we used a small network of 8 GPUs, compared to the 64 to 800 GPUs used by the authors in their benchmarks, therefore we achieved a much smaller memory reduction.

Also, maximum memory seems to be the main scaling bottleneck. In most runs, the maximum memory is much higher than the average memory. This is due to temporary buffers that drive the maximum memory up. On the GPTlite use case, most non-parameter memory is due to activations, and could have been improved with [Flash Attention](https://arxiv.org/abs/2307.08691) or [KV cache](https://arxiv.org/abs/2211.05102), that we did not include in this implementation.   

On pipeline parallelism, I noticed that the first GPU seems to require a considerably higher ammount of memory when compared to the remaining GPUs. This should not be the case, particularly on the deep benchmark model where we can guarantee a quasi-ideal stage partitioning across GPUs. This disparity in memory usage on GPU 0 is the main indicator of the maximum memory required, and balancing this would bring that value down. I [opened a bug report](https://github.com/microsoft/DeepSpeed/issues/4477) with DeepSpeed and will wait for their feedback or fix to correct this analysis.

Finally, finding the best parallelism strategy, and choosing between different ZeRO stages, offloading, activation checkpointing intervals, pipeline parallelism stages, etc, is very hard, as it depends heavily on the ML model and hardware. In practice, our config file is a manually-optimized ballpark figure of the default config file, and there is still plenty of work to be done to make it optimal, possibly by exploring the [autotuning](https://www.deepspeed.ai/tutorials/autotuning/) tools in DeepSpeed.

There is a lot of food for thought here, and I will be updating this post as I find new insights...

## Further resources and Code

We just scratched the surface of DeepSpeed capabilities. There are plenty of resources that should also be explored:
- [Autotuning](https://www.deepspeed.ai/tutorials/autotuning/) ([README.md](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/autotuning)) allows for the automatic finetuning of the allocation of computing (shards/layers) to processors, and is useful in very large models or large clusters; 
- [Model Parallelism](https://www.deepspeed.ai/training/#model-parallelism) for the implementation of custom tensor parallelism of models that are not implemented in DeepSpeed.
- [Model Checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) for saving and resuming execution state, applicable to large runs that are prune to failures and interrupts;
- [Flops profiler](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html) measures the time, flops and parameters of a PyTorch model and shows which modules or layers are the bottleneck;
- [Sparse attention kernels](https://www.deepspeed.ai/2020/09/08/sparse-attention.html) ([API](https://www.deepspeed.ai/docs/config-json/#sparse-attention)) to support long sequences of model inputs, such as text, image, or sound;
- [Communication optimizers](https://www.deepspeed.ai/training/#1-bit-adam-01-adam-and-1-bit-lamb-optimizers-with-up-to-26x-less-communication) offer the same convergence as Adam/LAMB but incur 26x less communication and 6.6x higher throughput on large BERT pretraining;
- [Monitor](https://www.deepspeed.ai/training/#monitor) logs live training metrics to one or more monitoring backends to TensorBoard, csv file or other resource;
- [Model Compression](https://www.deepspeed.ai/compression/) ([API](https://www.deepspeed.ai/docs/config-json/#compression)) via layer reduction, weight quantization, activation quantization, sparse pruning, row pruning, head pruning and channel pruning, to deliver faster speed and smaller model size.
- [Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/) ([API](https://deepspeed.readthedocs.io/en/latest/moe.html))  for sparsity during inference; 
- [Using pre-trained models for inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) for integrating pre-trained Hugging Face models into DeepSpeed;

For general documentation, I recommend the [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest), the [training features page](https://www.deepspeed.ai/training/#features), the [tutorials page](https://www.deepspeed.ai/tutorials/), the [HuggingFace page for DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed), and the examples at [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/).

All done! If you want to try this on your own, see the [GPT-lite-DeepSpeed repo](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-DeepSpeed).
