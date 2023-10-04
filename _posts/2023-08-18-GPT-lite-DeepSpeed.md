---
layout: post
title:  "Scaling a GPT model with ZeRO and DeepSpeed"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

Previously, in the [AI Supercomputing]({{ site.baseurl }}{% post_url 2020-05-12-AI-Supercomputing %}) and [AI Supercomputing (part 2)]({{ site.baseurl }}{% post_url 2020-05-28-AI-Supercomputing-2 %}) posts, we summarized existing parallelism techniques for ML models. Later, in the post [Building a GPT model from scratch]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), we built GPT-lite - a small version of the GPT model, trained on the "[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)" dataset. In this post, we will apply those parallelism techniques to that GPT model, and scale it on a network of GPUs using [DeepSpeed and ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer). We will use the `deepspeed` package for `python`.


### 3D Parallelism

A GPT model allows for three types of parallelism, that can be combined into what we call **3D parallelism**:
1. **Data parallelism**, by dividing the number of datapoints (batch size) across all processors, and using the average of the gradients across resources to perform the updates.
2. **Pipeline parallelism**, by delegating different layers (or blocks of layers) of the model to different processors.
3. **Model parallelism**, by dividing the *tensors* on each layer across the processors.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPT-lite-DeepSpeed/GPT_3D_parallelism_2.png"/>

{: style="text-align:center; font-size: small;"}
The resources allocation problem vizualized as a (color-coded) allocation of processors in the 3D space of data, layer and parameter dimensions. A GPT model allows for 3D parallelism as a combination of: pipelining across blocks/layers of the model, data parallelism across datapoints in the input batch, and sharded/model/tensor/vertical parallelism across the parameters of each layer. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

Model parallelism can be implemented by two distinct approaches:
- **ZeRO** implements **Fully-Sharded Data Parallelism (FSDP)** and is equivalent to data parallelism with distributed data storage. The main distinction is: in data parallelism, all processors hold a copy of the full model, and all models are kept in sync by averaging gradients at the end of every epoch and performing similar gradient updates. However, In ZeRO, the tensors for the parameter, gradients and optimizer states are distributed/partitioned/**sharded** across all the processors and scattered/gathered when needed. Thus the name: Zero-Redundancy *Optimizer*, but not data. ZeRO provides memory savings compared to data parallelism because of the partitioning of various tensors before and after the computations. An important remark is that the activations on the `forward` and `backward` still happen in full form i.e. they are not distributed and need to be kept on all processors for the backpropagation to work.
- **Tensor parallelism** is a more intense approach of model parallelism that divides not just the parameters, gradients and optimizer states, but also the computation. This requires the operations and the layer activations (the output of the forward pass) to be also sharded - horizontally or vertically - and distributed across processors. This approach requires a modification of the computations to work in a distributed manner. Therefore, it is a model-specific strategy. This is [supported but not provided by DeepSpeed](https://www.deepspeed.ai/training/#support-for-custom-model-parallelism), except in some implementations such as [Megatron-ML](https://www.deepspeed.ai/tutorials/megatron/).

Finally, **ZeRO has three alternative execution modes, called stages**. Each stage represents a different level of memory redundancy. In practice, picking a stage is equivalent to picking a tradeoff between memory usage and communication required to communicate distributed tensors:
- **ZeRO stage 1**: the optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.
- **ZeRO stage 2**: the reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.
- **ZeRO stage 3**: the 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes. 

Additionaly, on top of the previous stages, we can enable **ZeRO-Infinity**, an offload engine detailed in [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) that offloads parameters to both CPU and NVMe memory for huge memory savings.

Long story short, finding the optimal parallelism hyperparameters is a hard problem.
This is a resources allocation problem across the 3D volume in the data, parameters and layers space. It aims at finding the best partitioning across that 3D space, and allocating different partitions to different processors, in a way that best balances the compute time and/or memory across resources. In practice, balanced compute across resources allows for a low overall runtime, and balanced memory allows for an increase of the maximum model size.
 
### Main code

We start by matching the dimensions of our *GPT-lite* model architecture to the *GPT-3 Small* model description in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Fig 2.1), by changing the following variables in the original <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">python implementation</a>:

```python
# depth of the network as number of decoder blocks.
n_layer = 12

# size of the embeddings (d_model)
n_embd = 768

# number of attention heads in the Multi-Attention mechanism
n_head = 12

# number of heads, the $d_k$ in the paper formulation of Attn. Mech
head_size = 64

# block size ie max number of training sequence, the $n_{ctx}$ in the paper .
block_size = 2048

# dropout rate (variable p) for dropout units
dropout = 0.1
```

We then create the `ArgumentParser` object that is required by the `initialize()` method in DeepSpeed. The `ArgumentParser` object must contain:
- the `--local_rank` parameter that is the local rank of each process in the network, and will be populated automatically by the `deepspeed` launcher upon launch;
- optionally, we add the `--deepspeed_config` where we specify the path to the DeepSpeed config file. If you choose not to add it to the command line arguments, then it must be specified as the parameter `config` in the call to `deepspeed.initialize()`.

A simpler way to do this is to call `deepspeed.add_config_arguments()`, that adds the `--deepspeed` boolean flag that enables/disables deepspeed and the `--deepspeed_config` argument:

```python
def get_deepspeed_args(description='GPT lite on DeepSpeed'):
  import argparse
  parser = argparse.ArgumentParser(description=description)
  # mandatory argument for calls with deepseed
  parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
  # Include DeepSpeed configuration arguments (--deepspeed, --deepspeed_config, ...)
  parser = deepspeed.add_config_arguments(parser)
  return parser.parse_args()
```

We now create the function that returns a model of type `torch.nn.Module` and a dataset of type `torch.utils.data.Dataset`:

```python
import gptlite

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

  train_data, _, vocab_size = load_tiny_shakespeare_data() #load encoded data from text file
  dataset = GPTliteDataset(train_data, gptlite.block_size)
  return dataset, vocab_size

train_dataset, vocab_size = get_dataset()
model = gptlite.GPTlite(vocab_size)
```

As a side note, **any model and dataset can be used** in this code. As an example, if you'd want to perform a 10-class classification using the `ResNet` network on the `CIFAR10` dataset available in `torchvision`, you'd redefine the previous function as:

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

train_dataset = get_dataset()
model = torchvision.models.resnet18(num_classes=10)
```

Let's continue with our GPT-lite use case. The bulk of the code is pretty simple. In practice, all boilerplate code that PyTorch requires for optimizers, learning rates, parallelism, data loaders etc, are all managed by DeepSpeed and are defined in its config file. So the initialization of a DeepSpeed run is pretty straighforward:

```python
import deepspeed

def main_deepspeed():

  deepspeed.init_distributed()
  args = get_deepspeed_args() 
  train_dataset, vocab_size = get_dataset()
  model = gptlite.GPTlite(vocab_size)

  model_engine, optimizer, train_dataloader , _ = deepspeed.initialize(
    args=args, model=model, training_data=train_dataset,)
```

Then we do the initialization of the loss function, input datatype, and the local rank and device:

``` python
  criterion = torch.nn.CrossEntropyLoss()

  target_dtype = None # For float32, target_dtype is None so no datatype conversion needed
  if   model_engine.bfloat16_enabled(): target_dtype=torch.bfloat16
  elif model_engine.fp16_enabled():     target_dtype=torch.half

  local_rank = model_engine.local_rank
  local_device = deepspeed.accelerator.get_accelerator().device_name(local_rank)
  print(f"Starting training, I'm rank {local_rank} on {local_device}")
``` 

and finally the training loop, with a structure similar to the PyTorch implementation. The only exception is that we dont perform zeroing of gradients, this is managed internally by DeepSpeed. Also, `train_dataloader` a `DistributedSampler` created by DeepSpeed, so multi-process runs will have each process delegated to a different subset of data.

```python
  n_epochs=20
  for epoch in range(n_epochs):
    running_loss = 0.0
    for step, data in enumerate(train_dataloader):

      inputs = data[0].to(local_device)
      labels = data[1].to(local_device)

      if target_dtype != None:
        inputs = inputs.to(target_dtype)
            
      outputs = model_engine(inputs) #fwd pass
      loss = criterion(outputs, labels)
      running_loss += loss.item()

      model_engine.backward(loss) #backprop
      model_engine.step() #update weights, no zero-ing

    # print statistics
    if local_rank == 0:
        print("Epoch: {}, Step: {}, Loss: {}".format(epoch, step, running_loss / step))
```
 
### Config file

The real *nuance* and complexity in using DeepSpeed is the `.json` config file. The number of possible fields in the config is very large, as they define parallelism, floating point precision, logger, solver, communication, etc. These fields are detailed in the [DeepSpeed config documentation](https://www.deepspeed.ai/docs/config-json/). Here we start with a simple config, where the configure the DeepSpeed logger to output at every 100 epochs (`steps_per_print`), and define the settings of the optimizer (`optimizer`) and learning rate scheduler (`scheduler`):

```json
{
  "train_batch_size": 64,
  "steps_per_print": 100,
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
  },
},
```

**ZeRO** can be activated by specifying the relevant stage in the config file. If omitted, or when passing the stage 0, DeepSpeed is disabled and the execution follows the regular distributed data paralllel workflow:

```json
"zero_optimization": {
   "stage": 3
}
```

**Reducing the size of communication buffers** is relevant when activating ZeRO, as it will lead to the distribution of parameters across all processors. This in practice will add the overhead of reduce and broadcast operations, that require memory buffers to be allocated for all data to be sent of received. This may be an issue as these buffers may be large. To overcome this issue, we can reduce the maximum communication buffer size and perform the communication in parcels.
On top of that, we can enable  **communication overlap** that attempts to overlap the reduction of the gradients with backward computation. To enable these 2 optimizations, we add to the config:

```json
"zero_optimization": {
   "stage": 3,
   "reduce_bucket_size": 5e8,
   "all_reduce_bucket_size": 5e8,
   "overlap_comm": true,
}
```

[**ZeRO Infinity**](https://arxiv.org/abs/2104.07857) to perform offloading of optimizer coputation to CPU (with page-locked/pinned CPU memory), and parameter offloading (only compatible with stage 3), can be enabled with: 

```json
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
```

[**Mixed precision representation**](https://arxiv.org/abs/1710.03740) allows for calculus with value types (parameters, activations, accumulators) stored with different numerical representations, leading to a reduction of memory and compute time. It can be enabled by adding the `fp16` entry [in the config](https://www.deepspeed.ai/docs/config-json/#fp16-training-options). As a side note, the `amp` config entry also enables mixed precision training that follows the [NVIDIA Apex](https://nvidia.github.io/apex/) implementation i.e. with the `O0` to `O3` opimization levels. However, [it is not compatible with ZeRO](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options), therefore we won't use it. The [`fp16` is equivalent to APEX optimization level O2](https://www.deepspeed.ai/docs/config-json/#fp16-training-options), and according to the [documentation](https://www.deepspeed.ai/docs/config-json/#fp16-training-options), "if you want to use ZeRO (currently) you must use this mode". We will enable it with the entry `"fp16: { enabled: true }` that is equivalent to the following default values:

```json
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
```

**Gradient accumulation** based on **micro-batching** is a technique that simulates a large mini-batch as an iteration of several micro-batches. This is particularly relevant when the whole mini-batch does not fit into memory, and using an accumulation of micro-batches overcomes that limitation. This method is enabled by setting `train_micro_batch_size_per_gpu` or `gradient_accumulation_steps` in the config file.

[**Activation Checkpointing**](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html) allows for a large reduction in memory requirements by not storing all the forward-pass activations required for the backward propagation. The rationale is simply: instead of storing the output of every layer after the forward pass (required for the back propagation), only a small subset of - e.g. interleaved - layer outputs are kept in memory, and the remaining are computed on-the-fly with a forward pass from the closest lower layer. In our use case, we will use one activation checkpoint per decoder block (ie 12 in total) plus the 4 blocks that precede and follow the decoder blocks. This can be enabled by adding the following to the config file (see the [json documentation](https://www.deepspeed.ai/docs/config-json/#activation-checkpointing) for other options):

```json
"activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true,
    "num_checkpoints": 16,
    }
```

You may notice that hardcoding `num_checkpoints` in the config file is a bit cumbersome. To overcome this, it is possible to dynamically set and overwrite any config value by using the `deepspeed.checkpointing.configure` method. This allows config values to be computed on-the-fly or specified via command line arguments. In this particular example, we could set the activation checkpointing values as:

```python
def main_deepspeed():
  # ...
  deepspeed.checkpointing.configure(
        partition_activations=args.partition_activations,
        contiguous_checkpointing=args.contigious_checkpointing,
        num_checkpoints=len(model.to_layers()),
```

As a final note, the configuration file can be extended with e.g. problem or hardware specific parameters, but for brevity, I'll omit those details here. 

### Pipeline parallelism (optional, advanced)

[Pipeline parallelism](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules) improves both the memory and compute efficiency during training by partitioning the layers of a model into stages that can be processed in parallel. The pipeline parallelims implemented in DeepSpeed follows the [PipeDream-Flush implementation with 1F1B scheduling](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/).

DeepSpeed enables pipelining by assuming all modules in a `nn.Sequential` container or `list` to be the sequence of layers that can be broken into pipelining stages. A stage is a range of layers that define a block of computation that will be assigned to a section of the pipeline.

{: style="text-align:center; font-size: small;"}
<img width="79%" height="80%" src="/assets/GPT-lite-DeepSpeed/GPT_pipelining.png"/>

{: style="text-align:center; font-size: small;"}
A 4-stage pipeline. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

We will make pipeline parallelism optional in our use case, as in many cases (e.g. for small models) it is not benefitial. We will disable it by default and enable it by passing `--pipeline` on the command line:

```python
def get_deepspeed_args(description='GPT lite on DeepSpeed'):
  # ...
  parser.add_argument("--pipeline", action="store_true",
                      help="enable pipeline parallelism")
```

Now we expose the pipeline parallelism in our model by creating a method a new model `GPTlitePipe` that inherits from `GPTlite` and includes the method `to_layers()` that returns the sequence of actions to be executed. Note that `to_layers()` follows the same order as the `forward` pass in `GPTlite` and that `self.blocks` is of type `nn.Sequential`:

```python
class GPTlitePipe(GPTlite):
  
  def to_layers(self):  
      layers = [
          lambda idx:
            self.token_embedding_table(idx) +
            self.position_embedding_table(torch.arange(idx.shape[1]).to(idx.device)),
          *self.blocks,
          self.ln,
          self.lm_head,
          lambda logits: torch.swapaxes(logits,1,2)
      ]
      return layers
```

As a next step, in our DeepSpeed initialization code, we must create a pipeline wrapper around our model, with one stage per every 2 blocks of computation, and this will be fed later to the `deepspeed.initialize()` as the `model` parameter: 

```python
def main_deepspeed():
  # ...
  if args.pipeline:
    model = gptlite.GPTlitePipe(model.vocab_size)
    layers = model.to_layers()
    num_stages=int(os.environ["WORLD_SIZE"])
    model = deepspeed.pipe.PipelineModule(layers=layers, num_stages=num_stages)
```

As an important remark, when using pipeline parallelism, the micro-batch argument has a subtly different meaning: each batch of training data is divided into micro-batches that can be processed in parallel by the pipeline stages, as they are required for the gradient accumulation that follows. Therefore, it is important to set the micro-batch size (parameter `train_micro_batch_size_per_gpu` in config file) to a value greater than 1 to allow multi-stage parallelism.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/GPT-lite-DeepSpeed/GPT_pipelining_2.png"/>

{: style="text-align:center; font-size: small;"}
"An illustration of how DeepSpeed will train a batch with eight micro-batches using hybrid two-way data parallelism and two-stage pipeline parallelism. GPUs 0 and 2 are arranged in a pipeline and will alternate forward (F) and backward (B) passes. They will then all-reduce (AR) gradients with their data parallel counterparts, GPUs 1 and 3, respectively. Finally, the two pipeline stages update their model weights". Source: [DeepSpeed pipelining documentation](https://www.deepspeed.ai/tutorials/pipeline/)

Finally, [Pipeline parallelism is not compatible with ZeRO stages 2 or 3](https://deepspeed.readthedocs.io/en/latest/pipeline.html#pipeline-parallelism), as discussed [here](https://github.com/microsoft/DeepSpeed/issues/1110#issuecomment-850835817).

**Increasing compute and memory efficiency with LayerSpec:** 

The `GPTlitePipe` model above is neither memory efficient nor scalable as each GPU replicates the whole model in memory. See [Memory-Efficient Model Construction](https://www.deepspeed.ai/tutorials/pipeline/#memory-efficient-model-construction) for details. So we will use the DeepSpeed class `LayerSpec` (API [here](https://deepspeed.readthedocs.io/en/latest/pipeline.html#deepspeed.pipe.LayerSpec)) that delays the construction of modules until the model layers have been partitioned across workers, therefore having each worker will allocate only the layers it’s assigned to. To do this, we will create a new class `GPTlitePipeLayers` with an `__init__` method that follows very closely the method in the original `GPTlite`:

```python
from deepspeed.pipe import PipelineModule, LayerSpec
class GPTlitePipeLayers(PipelineModule):

  class Preprocess(nn.Module):
    """ converts preprocessing into an nn.Module. Required for LayerSpec"""
    def __init__(self, vocab_size):
      super().__init__()
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
      B, T = idx.shape
      tok_emb = self.token_embedding_table(idx)
      pos_emb = self.position_embedding_table(torch.arange(T).to(idx.device))
      return tok_emb + pos_emb

  def __init__(self, vocab_size, pipe_kwargs={}):
    self.specs = \
      [ LayerSpec(GPTlitePipeLayers.Preprocess, vocab_size) ] + \
      [ LayerSpec(Block, n_embd, n_head) for _ in range(n_layer)] + \
      [ LayerSpec(nn.LayerNorm, n_embd),
        LayerSpec(nn.Linear, n_embd, vocab_size, bias=False) ]
    super().__init__(layers=self.specs, loss_fn=nn.CrossEntropyLoss(), **pipe_kwargs)
```

and the main code altered to:

```python
def get_deepspeed_args():
  # ...
  parser.add_argument("--pipeline_spec_layers", action="store_true",
                      help="enable SpecLayers in pipeline parallelism")

def main_deepspeed():
  # ...
  if args.pipeline and args.pipeline_spec_layers:
    model = gptlite.GPTlitePipeLayers(vocab_size, pipe_kwargs={'num_stages':num_stages})
```

Finally, we will not tune the [load balancing method for pipeline modules](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules) and we will use the default `partition_method=parameters`, that "balances the number of trainable parameters on each pipeline stage". And in the extreme case the PipeDream algorithm is not the type of pipelining we want, then it is possible to [extend pipeline parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule). 

### Launching a distributed execution

The installation of DeepSpeed includes the `deepspeed` launcher, a network bootstrapper that detects all GPUs and nodes in a network and launches the main python script in all of them, with different `--local_rank` argument and different environment variables for the *comm world*. In our example, to launch the script `train.py` on a compute node with 8 GPUs, with the DeepSpeed config file `ds_config.json`, we run on the shell:

```shell
$ deepspeed --num_gpus=8 train.py --deepspeed_config ds_config.json
```

Few notes about parallel runs:
- Launching with `python` instead of `deepspeed` will perform a single-node single-GPU run.
- If we where required to run this on multiple compute nodes, we'd need to pass an extra parameter `--hostfile hostfile`, where `hostfile` is an MPI-style descriptor of nodes and gpus per node.
- In the config file we specified a batch size of 64, i.e. a batch of 8 for each GPU in our parallel runs.  We need to allocate at least 1 datapoint per process. Thus, the batch size in the config should take into consideration the number of compute nodes, the number of GPUs, and the number of gradient accumulation steps (when applicable). In brief, `train_batch_size` must be equal to `train_micro_batch_size_per_gpu` * `gradient_accumulation` * `--num_gpus`. Otherwise you'll get errors like `AssertionError: Micro batch size per gpu: 0 has to be greater than 0`.

For more information on available flags, running `deepspeed --help` provides a brief summary of all options.


### Benchmark of memory allocated to parameters (optional) 

We will use the [DeepSpeed API to estimate the memory requirements of model parameters](https://deepspeed.readthedocs.io/en/latest/memory.html#api-to-estimate-memory-usage) for different ZeRO implementations, by creatint the method `measure_parameters_memory`:

```python
def measure_parameters_memory(model):
  param_size_GB = sum([p.nelement() * p.element_size() for p in model.parameters()])/1024**3
  print(f"Native model parameters size: {round(param_size_GB, 2)}GB.")

  from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
  estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)

  from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
  estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
```

Calling the method, the output tells you that:
- the base model requires as parameters (not activations or buffers) about 0.323GB;
- DeepSpeed ZeRO-2 requires 0.161GB and 0.484GB for the with and without offload optimizations;
- DeepSpeed ZeRO-3 requires 0.009GB and 0.190GB for the with and without offload optimizations; 

However, when activating pipelining, ie by launching with `--pipeline --pipeline_spec_layers`:
- the base model requires 0.053GB for parameters; 
- ZeRO stage 2 requires 0.026GB and 0.079GB for the with and without offloading use cases;
- ZeRO stage 3 requires 0.009GB and 0.038GB of memory, with and without offloading, respectively; 

Now, this kind of metric has many fallacies: it only measures parameter overheard, and does not take activations or other residual buffers (e.g. normalization varibles) into account, does not look at the batch size, etc. Also, the pipeline metrics are not accurate as pipeline parallelism is not compatible with ZeRO stages 2 or 3. 

### Benchmark of memory and performance at runtime

To collect real performace metrics, we will use  `nvidia-smi` to quantify GPU memory usage and processor utilization. We'll also use the deepspeed logger to collect 4 metrics at a set interval: average samples per sec, average allocated memory, and max allocated memory at any given instant. Finally, we will test the largest model possible on each configuration. 

(Coming soon)

[//]: # {: style="text-align:center; font-size: small;"}
[//]: # <img width="100%" height="100%" src="/assets/GPT-lite-DeepSpeed/benchmark.png"/>


### Final remarks 

We just touched the surface of the capabilities of DeepSpeed, and there are plenty of resources that should also be taken into account:
- [Autotuning](https://www.deepspeed.ai/tutorials/autotuning/) allows for the automatic finetuning of the allocation of computing (shards/layers) to processors, and is useful in very large models or large clusters; 
- [Model Parallelism](https://www.deepspeed.ai/training/#model-parallelism) for implementation of custom tensor parallelism for the models that are not implemented in DeepSpeed;
- [Activation Partitioning](https://www.deepspeed.ai/training/#activation-checkpointing-api) reduces the memory consumed by activations during model parallel training, by storing these activations in a partitioned state after the forward pass, and doing an allgather right before they are needed again on the backward propagation; 
- [Model Checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html), applied on large runs that are prune to failures or interrupts;
- [Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/)  for sparsity during inference. See the [API here](https://deepspeed.readthedocs.io/en/latest/moe.html);
- [Using pre-trained models for inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) for integrating Hugging Face models into DeepSpeed;

... and many more covered by the [DeepSpeed API documentation](https://deepspeed.readthedocs.io/en/latest), the [training features page](https://www.deepspeed.ai/training/#features), the [tutorials page](https://www.deepspeed.ai/tutorials/) and the examples at [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/).


All done! If you want to download the files in this post and run it on your own, see <a href="/assets/GPT-lite-DeepSpeed/train.py">train.py</a> for the main python code, <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">gptlite.py</a> for the GPTlite model, <a href="/assets/GPT-lite-DeepSpeed/ds_config.json">ds_config.json</a> for the DeepSpeed config file, and <a href="/assets/GPT-lite-DeepSpeed/run.sh">run.sh</a> for the command line script to launch the execution.
