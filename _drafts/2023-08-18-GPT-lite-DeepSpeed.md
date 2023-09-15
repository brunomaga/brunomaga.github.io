---
layout: post
title:  "Scaling a GPT model with DeepSpeed"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

In the [previous post]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), we built GPT-lite (a small version of the GPT model) and trained it on the "[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)" dataset. In this post, we will use the same model and scale it using [DeepSpeed and ZeRO](https://arxiv.org/abs/1910.02054), a *zero-redundancy* scaling technique detailed [in this post]({{ site.baseurl }}{% post_url  2020-05-28-AI-Supercomputing-2 %}). We'll use the `deepspeed` package for `python`.

We start by matching the dimensions of our *GPT-lite* model architecture to the *GPT-3 Small* model description in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Fig 2.1), by changing the following variables:

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

We then create the `ArgumentParser` required for the `initialize()` method in DeepSpeed. The `ArgumentParser` object must contain:
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

We now create a function that returns a model of type `torch.nn.Module` and dataset of type `torch.utils.data.Dataset`:

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

As a side not, **any model is possible** to be used in this code. E.g. if you'd want to perform 10-class classification on the `CIFAR10` dataset available in `torchvision`:

```python
def get_model_and_dataset():
    import torchvision
    import torchvision.transforms as transforms
    model = torchvision.models.resnet18(num_classes=10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return model, dataset
```

Let's continue with our GPT-lite model. The bulk of the code is pretty simple. In practice, all boilerplate code that pytorch requires for optimizers, learning rates, parallelism, data loaders etc, are all managed by DeepSpeed and defined in its config file. So the train loop is pretty simple:

```python
def main_deepspeed():


  import deepspeed
  deepspeed.init_distributed()
  args = get_deepspeed_args('CIFAR') 
  model, train_dataset = get_model_and_dataset()

  model_engine, optimizer, train_dataloader , _ = deepspeed.initialize(
    args=args, model=model, training_data=train_dataset,
    #config=args.deepspeed_config, #only needed when args.deepspeed_config does not exist
    )
```

Then we add the initialization of the loss function, input datatype, and local rank and device variables:

``` python
  criterion = torch.nn.CrossEntropyLoss()

  target_dtype = None # For float32, target_dtype is None so no datatype conversion needed
  if   model_engine.bfloat16_enabled(): target_dtype=torch.bfloat16
  elif model_engine.fp16_enabled():     target_dtype=torch.half

  local_device = deepspeed.accelerator.get_accelerator().device_name(model_engine.local_rank)
  local_rank = model_engine.local_rank
  print(f"Starting training, I'm rank {local_rank} on {local_device}")
``` 

and finally the training loop:

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
    if local_rank == 0: print("Epoch: {}, Step: {}, Loss: {}".format(epoch, step, running_loss / step))
```

### Config file

All code so far is pretty standard as it follows the DeepSpeed *recipe*. The real *nuance* and complexity of DeepSpeed is the config file. The number of possible fields in the DeepSpeed config is very large, and they are all detailed in the [DeepSpeed config documentation](https://www.deepspeed.ai/docs/config-json/). Here we start with a simple config, where the configure the logger to output every 100 epochs (`steps_per_print`), and define optimizer (`optimizer`) and LR scheduler (`scheduler`) settings:

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

### Launch

DeepSpeed installs with the `deepspeed` app, a network bootstrapper that detects all GPUs and nodes in a network and launches the python in all of them, with different `--local_rank` arguments. We will run this in a single compute node with 8 GPUs. Our  python python code is in the file `gptlite_deepspeed.py`  and the config file is `deepspeed_config_ds.json`. So our `deepspeed` call is:

```
deepspeed --num_gpus=8 gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json
```

Note that in the config file we specified a batch size of 64, i.e. a batch of 8 for each GPU in our parallel runs. 
We need to allocate at least 2 inputs per processor (it failed for me with 1 input per GPU), so take that into consideration the number of compute nodes, the number of GPUs, and the number of gradient accumulation steps (when applicable), when providing the batch size. In brief, `train_batch_size` must be equal to `train_micro_batch_size_per_gpu` * `gradient_accumulation` * number of GPUs. Otherwise you'll get errors like `AssertionError: Micro batch size per gpu: 0 has to be greater than 0`.

If we where required to run this in multiple compute nodes, we'd need to specify the DNS of each node and the number of GPUs per node in a file. This would requires the parameter `--hostfile hostfile`. Other relevant function of DeepSpeed is the [Autotuning of hyperparameters and parallelism](https://www.deepspeed.ai/tutorials/autotuning/), that requires the `--autotuning` flag. For more flags, running `deepspeed --help` provides a quick explanation of all options.

### no DeepSpeed (single GPU run):

launched with `python gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json` (Note: the config is needed for all optimizer, datatypes, etc variables).

`nvidia-smi` metrics: memory usage of 10.44 GB in a single GPU, with GPU utilization at 100%

output:
```
```


### Deepspeed without ZeRO, or ZeRO stage 0, or Data Parallelism:

Stage 0 disables deepzero, oly data parallelism

`nvidia-smi` metrics: memory usage : 6.79GB MB, GPU utilization at 100%

launched with `deepspeed --num_gpus=8 gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json`

Logger output in terminal (broken in several lines):

```
RunningAvgSamplesPerSec=67.20094094640265,
CurrSamplesPerSec=66.45660899684819,
MemAllocated=0.47GB,
MaxMemAllocated=6.07GB
```

### ZeRO stage 1

The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition. The confile file needs a new field:

```
   "zero_optimization": {
	"stage": 1
    }
```

`nvidia-smi` metrics: memory usage : 6.57GB, GPU utilization around 99%

Logger output in terminal:

```
RunningAvgSamplesPerSec=67.0053491112917,
CurrSamplesPerSec=63.59028627957837,
MemAllocated=0.33GB,
MaxMemAllocated=5.93GB
```

### ZeRO stage 2:

The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states. These fields were changed to the config file:

```
   "zero_optimization": {
	"stage": 2,
	"reduce_bucket_size": 5e8       # this MAY be necessary otherwise it will run out of memory
	"all_reduce_bucket_size": 5e8   # this MAY be necessary otherwise it will run out of memory
    }
```

The extra elements are the number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes

`nvidia-smi` metrics: memory usage : 6.5 GB, GPU utilization around 99%

Logger output in terminal:


```
RunningAvgSamplesPerSec=67.96643162283617,
CurrSamplesPerSec=63.47303343527717,
MemAllocated=0.33GB,
MaxMemAllocated=6.29GB
```

### ZeRO stage 3:

The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes. The field in the config file needs to be changed to `"stage": 2`. 

`nvidia-smi` metrics: memory usage : 7.06GB, GPU utilization in the range 100%

Logger output in terminal:

```
RunningAvgSamplesPerSec=68.19805706375571,
CurrSamplesPerSec=68.08978242810488,
MemAllocated=0.45GB,
MaxMemAllocated=6.11GB
```

### ZeRO stage 3 with offloading:

In addition, ZeRO-3 includes the infinity offload engine to form ZeRO-Infinity (paper), which can offload to both CPU and NVMe memory for huge memory savings. To offload to the cpu, we add to our config:

```
  "zero_optimization": {
      "offload_optimizer": {
          "device": "cpu"
      },
      "offload_param": {
          "device": "cpu"
      }
  }
```

`nvidia-smi` metrics: memory usage : 6.96GB, GPU utilization in the range 99%

Logger output in terminal:

```
RunningAvgSamplesPerSec=57.297071153914516,
CurrSamplesPerSec=56.074883086568896,
MemAllocated=0.39GB,
MaxMemAllocated=6.06GB
```



### Fp16 speed-up

A speed-up comes from using fp16. That requires a loss scaling:

```
  "zero_optimization": {
      "fp16": {
          "enabled": true,
          "fp16_master_weights_and_grads": false,
          "loss_scale": 0,
          "loss_scale_window": 500,
          "hysteresis": 2,
          "min_loss_scale": 1,
          "initial_scale_power": 15
      },
  }
```

As a comparison, for the stage 1 the new output of the logger is:

```
RunningAvgSamplesPerSec=447.665553440481,
CurrSamplesPerSec=453.0647979368359,
MemAllocated=0.06GB,
MaxMemAllocated=0.1GB
```

compared to the original

```
RunningAvgSamplesPerSec=357.6931784027813,
CurrSamplesPerSec=372.6779510418055,
MemAllocated=0.07GB,
MaxMemAllocated=0.16GB
```


### Final remarks 

We just touched the surface of DeepSpeed capabilities. Other tools that should be taking into account are:
- [Model Checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html), applied on large runs that are prune to failures or interrupts.
- [Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/)  for sparsity during inference. See the [API here](https://deepspeed.readthedocs.io/en/latest/moe.html).
- [Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/) for a simpler layer-level implementation of parallelism.
- [Using pre-trained models for inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) for integration Hugging Face transformers into deepspeed.
- The full list of options for the `zero_optimization` section in the config can be found in the [ZeRO API documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html).

And many others covered also in the [tutorial page](https://www.deepspeed.ai/tutorials/). The full list of options for the `zero_optimization` is in [the ZeRO API documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html). And a good set of tutorials and examples is available in the [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/) repo, particularly the [CIFAR10 tutorial](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/cifar).

All done! To download the files mentioned here, see <a href="/assets/GPT-lite-DeepSpeed/gptlite_deepspeed.py">gptlite_deepspeed.py</a> for the main python code, <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">gptlite.py</a> for the GPTlite model (model only, no run/valid loop), and <a href="/assets/GPT-lite-DeepSpeed/gptlite_config_ds.json">gptlite_config_ds.json</a> for the DeepSpeed config file.
