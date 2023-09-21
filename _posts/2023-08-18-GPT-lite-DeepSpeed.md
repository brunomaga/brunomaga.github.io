---
layout: post
title:  "Scaling a GPT model with ZeRO and DeepSpeed"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

In the [previous post]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), we built GPT-lite - a small version of the GPT model - and trained it on the "[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)" dataset. In this post, we will use the same model and scale it on a network of GPUs using [DeepSpeed and ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer), a multi-dimensional scaling technique detailed [in this post]({{ site.baseurl }}{% post_url  2020-05-28-AI-Supercomputing-2 %}). We'll use the `deepspeed` package for `python`.


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
- **ZeRO** implements **Fully-Sharded Data Parallelism** and is equivalent to data parallelism with distributed data storage. The main distinction is: in data parallelism, all processors hold a copy of the full model, and all models are kept in sync by averaging gradients at the end of every epoch and performing similar gradient updates. However, In ZeRO, the tensors for the parameter, gradients and optimizer states are distributed/partitioned/**sharded** across all the processors and scattered/gathered when needed. Thus the name: Zero-Redundancy *Optimizer*, but not data. ZeRO provides memory savings compared to data parallelism because of the partitioning of various tensors before and after the computations. An important remark is that the activations on the `forward` and `backward` still happen in full form i.e. they are not distributed and need to be kept on all processors for the backpropagation to work.
- **Tensor parallelism** is a more intense approach of model parallelism that divides not just the parameters, gradients and optimizer states, but also the computation. This requires the operations and the layer activations (the output of the forward pass) to be also sharded - horizontally or vertically - and distributed across processors. This approach requires a modification of the computations to work in a distributed manner. Therefore, it is a model-specific strategy. This is supported but not provided by DeepSpeed, except in some implementations such as [Megatron-ML](https://www.deepspeed.ai/tutorials/megatron/).

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
def get_model_and_dataset():
  import gptlite

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

  model = gptlite.GPTlite()
  dataset = GPTliteDataset(gptlite.train_data, gptlite.block_size)
  return model, dataset
```

As a side note, **any model and dataset can be used** in this code. As an example, if you'd want to perform a 10-class classification using the `ResNet` network on the `CIFAR10` dataset available in `torchvision`, you'd redefine the previous function as:

```python
def get_model_and_dataset():
    import torchvision
    import torchvision.transforms as transforms
    model = torchvision.models.resnet18(num_classes=10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    return model, dataset
```

Let's continue with our GPT-lite use case. The bulk of the code is pretty simple. In practice, all boilerplate code that PyTorch requires for optimizers, learning rates, parallelism, data loaders etc, are all managed by DeepSpeed and are defined in its config file. So the initialization of a DeepSpeed run is pretty straighforward:

```python
def main_deepspeed():

  import deepspeed
  deepspeed.init_distributed()
  args = get_deepspeed_args('CIFAR') 
  model, train_dataset = get_model_and_dataset()

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
 
### Enabling Pipelining

DeepSpeed uses all layers in a `nn.Sequential` container or `list` as the sequence of layers to be broken into pipelining stages. A stage is a range of layers that define a block of computation that will be assigned to a section of the pipeline.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/GPT-lite-DeepSpeed/GPT_pipelining.png"/>

{: style="text-align:center; font-size: small;"}
A 4-stage pipeline. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

In our use case, we expose the layers be creating a method `to_layers()` method inside the `GPTlite` class that returns the sequence of layers as functions to be executed. Note that `to_layers()` follows the same order as the `forward` pass in `GPTlite` and that `self.blocks` is of type `nn.Sequential`):

```python
class GPTlite(nn.Module):
    def to_layers(self):
        layers = [
            lambda tok_emb pos_emb: tok_emb + pos_emb,
            *self.blocks,
            self.ln,
            self.lm_head,
            lambda logits: torch.swapaxes(logits,1,2)
        ]
        return layers
```

Finally, we create a pipeline wrapper around `model`, that be fed later to the `deepspeed.initialize()` declaration: 

```python
model = deepspeed.pip.PipelineModule(layers=model.to_layers(), num_stages=2)
```


### DeepSpeed config file

The real *nuance* and complexity in using DeepSpeed is the config file (`json`). The number of possible fields in the config is very large, as they define parallelism, floating point precision, logger, solver, communication, etc. These fields are detailed in the [DeepSpeed config documentation](https://www.deepspeed.ai/docs/config-json/). Here we start with a simple config, where the configure the DeepSpeed logger to output at every 100 epochs (`steps_per_print`), and define the settings of the optimizer (`optimizer`) and learning rate scheduler (`scheduler`):

```json
{
  "train_batch_size": 64,
  "steps_per_print": 100,
  "optimizer": {
    "type": "Adam",
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
```

### Launching a distributed execution

The installation of DeepSpeed includes the `deepspeed` launcher, a network bootstrapper that detects all GPUs and nodes in a network and launches the main python script in all of them, with different `--local_rank` argument and different environment variables for the *comm world*. In our example, to launch the script `gptlit_deepspeed.py` on a compute node with 8 GPUs, with the DeepSpeed config file `gptlite_config_ds.json`, we run on the shell:

```shell
$ deepspeed --num_gpus=8 gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json
```

Few notes about parallel runs:
- Launching with `python` instead of `deepspeed` will perform a single-node single-GPU run.
- If we where required to run this on multiple compute nodes, we'd need to pass an extra parameter `--hostfile hostfile`, where `hostfile` is an MPI-style descriptor of nodes and gpus per node.
- In the config file we specified a batch size of 64, i.e. a batch of 8 for each GPU in our parallel runs.  We need to allocate at least 1 datapoint per process. Thus, the batch size in the config should take into consideration the number of compute nodes, the number of GPUs, and the number of gradient accumulation steps (when applicable). In brief, `train_batch_size` must be equal to `train_micro_batch_size_per_gpu` * `gradient_accumulation` * `--num_gpus`. Otherwise you'll get errors like `AssertionError: Micro batch size per gpu: 0 has to be greater than 0`.
- When using pipelining, each batch of training data is divided into micro-batches that can be processed in parallel by the pipeline stages, for a posterior gradient accumulation. Therefore, it is important to set `train_micro_batch_size_per_gpu` $$\gt 1$$ to allow multi-stage parallelism.
- To enable the [Autotuning of hyperparameters and parallelism](https://www.deepspeed.ai/tutorials/autotuning/), we'd pass the `--autotuning` flag.

For more information of available flags, running `deepspeed --help` provides a brief summary of all options.

### Benchmark 

We will run and benchmark several parallelism configurations. To collect our metrics, we will use  `nvidia-smi` to quantify GPU memory usage and processor utilization. We'll also use the deepspeed logger to collect 4 metrics at a set interval: running avg. samples per sec, average memory allocated and max memory allocated at any given instant. Finally, because ultimately the goal of DeepSpeed is scaling, we will test the largest model possible on each configuration. 

We will include the following analysis:
- the single-node single-GPU use case, i.e. no parallelism. 
- the Pipeline Parallel execution, enabled with `PipelineModule(..., num_stages=4)` and `train_micro_batch_size_per_gpu: 4` in the config.
- Distributed Data Parallelism by disabling DeepZero. Enabled by omitting `zero_optimization` in the config, of passing `"zero_optimization": { "stage": 0 }`.
- ZeRO stage 3. The ZeRO config will also include two new entries to define the number of elements reduced/allreduced at a time, to limit the memory required for the allgather for large model sizes:
     ```json
        "zero_optimization": {
          "stage": 3,
          "reduce_bucket_size": 5e8,
          "all_reduce_bucket_size": 5e8
      }
     ```
- DeepSpeed with ZeRO stage 3 Infinity. To perform CPU offloading, we add to the config:
    ```json
        "zero_optimization": {
            "offload_optimizer": { "device": "cpu" },
            "offload_param":     { "device": "cpu" }
        }
    ```
- DeepSpeed with 16-bit floating point (fp16) or automatic mixed precition (amp). Enables mixed precison with different optimization levels, based on [NVIDIA Apex](https://nvidia.github.io/apex/). For amp training the config adds `"amp":  { "enabled": true, "opt_level": "O1" } `. For fp16 trainig, we change it line with the [CIFAR10 example](https://github.com/microsoft/DeepSpeedExamples/blob/master/training/cifar/cifar10_deepspeed.py).
 
**NOTE:** I'm now working on autotuning the resource allocation problem and collecting performance numbers. Results will follow soon.

[//]: # {: style="text-align:center; font-size: small;"}
[//]: # <img width="100%" height="100%" src="/assets/GPT-lite-DeepSpeed/benchmark.png"/>


### Final remarks 

We just touched the surface of the capabilities of DeepSpeed. Other components of DeepSpeed that should be taken into account are:
- [Model Checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html), applied on large runs that are prune to failures or interrupts.
- [Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/)  for sparsity during inference. See the [API here](https://deepspeed.readthedocs.io/en/latest/moe.html).
- [Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/) for a simpler layer-level implementation of parallelism, without ZeRO.
- [Autotuning](https://www.deepspeed.ai/tutorials/autotuning/) of resources allocation.
- [Using pre-trained models for inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) for integrating Hugging Face models into DeepSpeed.
- [ZeRO API documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html) for the full list of config options for the `zero_optimization`.

And many others covered by the [tutorial page](https://www.deepspeed.ai/tutorials/) and the examples at [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/).

All done! If you want to download the files in this post and run it on your own, see:
- <a href="/assets/GPT-lite-DeepSpeed/gptlite_deepspeed.py">gptlite_deepspeed.py</a> for the main python code;
- <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">gptlite.py</a> for the GPTlite model (model only, no run/valid loop);
- <a href="/assets/GPT-lite-DeepSpeed/gptlite_config_ds.json">gptlite_config_ds.json</a> for the DeepSpeed config file for ZeRO stage 3 with mixed precision and CPU offloading; and
- <a href="/assets/GPT-lite-DeepSpeed/run.sh">run.sh</a> for the command line script to launch the execution.
