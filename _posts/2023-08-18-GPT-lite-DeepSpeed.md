---
layout: post
title:  "Scaling a GPT model with 3D parallelism via ZeRO and DeepSpeed"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

In the [previous post]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), we built GPT-lite - a small version of the GPT model - and trained it on the "[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)" dataset. In this post, we will use the same model and scale it on a network of GPUs using [DeepSpeed and ZeRO](https://arxiv.org/abs/1910.02054), a *zero-redundancy* scaling technique detailed [in this post]({{ site.baseurl }}{% post_url  2020-05-28-AI-Supercomputing-2 %}). We'll use the `deepspeed` package for `python`.

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/GPT-lite-DeepSpeed/GPT_3D_parallelism_2.png"/>

{: style="text-align:center; font-size: small;"}
The resources allocation problem vizualized a a (color-coded) allocation of compute resources in the 3D space with data, layers and parameter dimensions. A GPT model allows for 3D parallelism as a combination of: pipelining across blocks/layers of the network, data parallelism across datapoints in the input batch, and tensor/vertical parallelism across parameters on each layer. Source: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

We start by matching the dimensions of our *GPT-lite* model architecture to the *GPT-3 Small* model description in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Fig 2.1), by changing the following variables in the original <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">model implementation</a>:

```python
# depth of the network as number of decoder blocks.
n_layer = 12

# size of the embeddings (d_model)
n_embd = 768

# number of attention heads in the Multi-Attention mechanism
n_head = 12

# number of heads. this is the $d_k$ in the paper formulation of Attn. Mech
head_size = 64

# block size ie max number of training sequence, the $n_{ctx}$ in the paper .
block_size = 2048

# dropout rate (variable p) for dropout units
dropout = 0.1
```

We then create an `ArgumentParser` object that is required for the `initialize()` method in DeepSpeed. The `ArgumentParser` object must contain:
- the `--local_rank` parameter that is the local rank of each process in the network, and will be populated automatically by the `deepspeed` launcher;
- optionally, we add the `--deepspeed_config` where we specify the path to the DeepSpeed config file. If you choose not to add it to the command line arguments, then it must be specified as the parameter `config` in `deepspeed.initialize()`.

The easiest way to do this is to manually add `--local_rank` to the instantiation of `ArgumentParser` and then call `deepspeed.add_config_arguments()` to add the `--deepspeed` boolean flag that enables/disables deepspeed, and the `--deepspeed_config` argument:

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

We now create a function that returns a model of type `torch.nn.Module` and a dataset of type `torch.utils.data.Dataset`:

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

As a side not, **any model and dataset can be used** in this code. As an example, if you'd want to perform 10-class classification using the `ResNet` network on the `CIFAR10` dataset available in `torchvision`:

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

Let's continue with our GPT-lite model use case. The bulk of the code is pretty simple. In practice, all boilerplate code that PyTorch requires for optimizers, learning rates, parallelism, data loaders etc, are all managed by DeepSpeed and are defined by its config file. So the initialization is pretty straighforward:

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

  local_device = deepspeed.accelerator.get_accelerator().device_name(model_engine.local_rank)
  local_rank = model_engine.local_rank
  print(f"Starting training, I'm rank {local_rank} on {local_device}")
``` 

and finally the training loop, with a structure very similar to the PyTorch implementation, except that there's no zeroing of gradients (it is managed internally by DeepSpeed):

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
 
### The DeepSpeed config file

The bulk of the code is very simple as it follows the DeepSpeed *recipe*. The real *nuance* and complexity of using DeepSpeed is the config file (`.json`). The number of possible fields in the config is very large, as they define parallelism, precision, solvers, etc. These fields are detailed in the [DeepSpeed config documentation](https://www.deepspeed.ai/docs/config-json/). Here we start with a simple config, where the configure the DeepSpeed logger to output at every 100 epochs (`steps_per_print`), and define the optimizer (`optimizer`) and LR scheduler (`scheduler`) settings:

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

The installation of DeepSpeed includes the `deepspeed` app, a network bootstrapper that detects all GPUs and nodes in a network and launches the main python script in all of them, with different `--local_rank` argument and different environment variables for the *comm world*. In this blog, we will run our model in a single compute node with 8 GPUs. The python code above is in the file <a href="/assets/GPT-lite-DeepSpeed/gptlite_deepspeed.py">gptlite_deepspeed.py</a> and the config file is <a href="/assets/GPT-lite-DeepSpeed/gptlite_config_ds.json">gptlite_config_ds.json</a>. So the launch command is:

```
deepspeed --num_gpus=8 gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json
```

Note that in the config file we specified a batch size of 64, i.e. a batch of 8 for each GPU in our parallel runs. 
We need to allocate at least 1 input per process. Thus, next time you define the batch size, take into consideration the number of compute nodes, the number of GPUs, and the number of gradient accumulation steps (when applicable): `train_batch_size` must be equal to `train_micro_batch_size_per_gpu` * `gradient_accumulation` * number of GPUs. Otherwise you'll get errors like `AssertionError: Micro batch size per gpu: 0 has to be greater than 0`.

If we where required to run this in multiple compute nodes, we'd need to specify the DNS of each node and the number of GPUs per node. This is done by passing an extra parameter `--hostfile hostfile`, where `hostfile` is an MPI-style descriptor of nodes and gpus per node. Other relevant parameter is related to the [Autotuning of hyperparameters and parallelism](https://www.deepspeed.ai/tutorials/autotuning/), that requires the `--autotuning` flag. For more flags, running `deepspeed --help` provides a brief summary of all options.


### 3D Parallelism

A GPT model allows for three types of parallelism, that can be into what DeepSpeed calls **3D parallelism**:
1. **Data parallelism**, by dividing the number of datapoints (`train_batch_size`) across a subset of nodes.
2. **Pipeline parallelism**, by delegating different blocks of the GPT to different resources. Piepeline parallelism is possible with no modification by defining blocks of the model inside a `nn.Sequential` container, that DeepSpeed can then explore. This is declared in our model as:
  ```
  self.blocks = nn.Sequential( *[Block(n_embd, n_head=4) for _ in range(n_layer)])
  ```
3. **Tensor/vertical parallelism**, by dividing the parameters on each layer across compute resources, the specialization of the ZeRO operations.  

Finding the best combination of the individual levels of parallelism is a hard problem.
This is a resources allocation problem across the 3D volume in the space of data, parameters and layers space. It aims at finding the best *slicing* across the 3D volume, and allocating different volumetric regions to different resources, in a way that best balances the compute time and/or memory across resources:
 
### Benchmark Setup

We will run and test several parallelism configurations. To quantify our results, we will use the `nvidia-smi` to quantify GPU memory usage and processor utilization. We'll also use the deepspeed logger to collect 4 metrics at a set interval: running avg. samples per sec, average memory allocated and max memory allocated at any given instant. Finally, because ultimately the goal of DeepSpeed is scaling, we will test the largest model possible on each configuration. 

The following configurations will be benchmarked:
- the single GPU use case, i.e. no parallelism;
  - launched with `python gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json` (note: the config is still needed on serial runs, to define for optimizer, datatypes, etc);
- the pipeline parallel execution; 
- DeepSpeed with ZeRO stage 0 (ie ZeRO is disabled), performing distributed data parallelism;
  - launched with `deepspeed --num_gpus=8 gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json`
- DeepSpeed with ZeRO stage 1. The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.
  - launched as before, with the following field in the config file:
    ```
        "zero_optimization": {
            "stage": 1
        }
    ```
- DeepSpeed with ZeRO stage 2. The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.
  - These ZeRO config now includes two new entries to define the number of elements reduced/allreduced at a time. This limits the memory required for the allgather for large model sizes:
    ```
        "zero_optimization": {
            "stage": 2,
            "reduce_bucket_size": 5e8,
            "all_reduce_bucket_size": 5e8
        }
    ```
- DeepSpeed with ZeRO stage 3. The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.
  - The field in the config file needs to be changed to `"stage": 3`.
- DeepSpeed with ZeRO stage 3 Infinity. Infinity is an offload engine detailed in [ZeRO-Infinity](https://arxiv.org/abs/2104.07857), which can offload to both CPU and NVMe memory for huge memory savings.
  - To offload to the cpu, we add to our config:
    ```
        "zero_optimization": {
            "offload_optimizer": { "device": "cpu" },
            "offload_param":     { "device": "cpu" }
        }
    ```
- DeepSpeed with 16-bit floating point (fp16) or automatic mixed precition (amp). Enables mixed precison with different optimization levels, based on [NVIDIA Apex](https://nvidia.github.io/apex/).
  - for amp trainig:  `"amp":  { "enabled": true, "opt_level": "O1" } `
  - for fp16 trainig: `"fp16": { "enabled": true, ... }`, in line with the [CIFAR10 example](https://github.com/microsoft/DeepSpeedExamples/blob/master/training/cifar/cifar10_deepspeed.py).
 
### Benchmark Results 

I'm still working on autotuning the resource allocation problem. Results coming soon.

[//]: # {: style="text-align:center; font-size: small;"}
[//]: # <img width="100%" height="100%" src="/assets/GPT-lite-DeepSpeed/benchmark.png"/>


### Final remarks 

We just touched the surface of the capabilities of DeepSpeed. Other components of DeepSpeed that should be taken into account are:
- [Model Checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html), applied on large runs that are prune to failures or interrupts.
- [Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/)  for sparsity during inference. See the [API here](https://deepspeed.readthedocs.io/en/latest/moe.html).
- [Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/) for a simpler layer-level implementation of parallelism.
- [Using pre-trained models for inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) for integration Hugging Face transformers into deepspeed.
- The full list of options for the `zero_optimization` section in the config can be found in the [ZeRO API documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html).

And many others covered also in the [tutorial page](https://www.deepspeed.ai/tutorials/). The full list of options for the `zero_optimization` is in [the ZeRO API documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html). And a good set of tutorials and examples is available in the [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/) repo, particularly the [CIFAR10 tutorial](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/cifar).

All done! If you want to download the files in this post and run it on your own, see:
- <a href="/assets/GPT-lite-DeepSpeed/gptlite_deepspeed.py">gptlite_deepspeed.py</a> for the main python code;
- <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">gptlite.py</a> for the GPTlite model (model only, no run/valid loop);
- <a href="/assets/GPT-lite-DeepSpeed/gptlite_config_ds.json">gptlite_config_ds.json</a> for the DeepSpeed config file for ZeRO stage 3 with mixed precision and CPU offloading; and
- <a href="/assets/GPT-lite-DeepSpeed/run.sh">run.sh</a> for the command line script to launch the execution.
- <a href="/assets/GPT-lite-DeepSpeed/benchmark.xlsx">benchmark.xlsx</a> for the spreadsheet with the benchmark results and plot.
