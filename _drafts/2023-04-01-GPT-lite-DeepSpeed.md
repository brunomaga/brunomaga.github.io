---
layout: post
title:  "Scaling a GPT model with DeepSpeed"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

Size matters. We configure the model to follow the $$GPT-3 Medium$$ model in Figure 2.1 in [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).

```
# depth of the network as number of decoder blocks.
n_layer = 24

# size of the embeddings (d_model)
n_embd = 1024

# number of attention heads in the Multi-Attention mechanism
n_head = 16

# number of heads. this is the $d_k$ in the paper formulation of Attn. Mech
head_size = 64

# block size ie max number of training sequence, the $n_{ctx}$ in the paper
block_size = 2048

# dropout rate (variable p) for dropout units
dropout = 0.1
```



How to (first deepspeed params, second app params, args is needed):
```
deepspeed --num_gpus 8  [--hostfile hostfile] [--autotuning run] --test_performance #TODO

deepspeed --num_gpus=8 gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json
```

To add to the code:
todo save model every X iterations

# Reminder:

train_batch_size = GPUs * train_micro_batch_size_per_gpu * gradient_accumulation_steps
if you have a `train_batch_size` too small, you'll get 
AssertionError: Micro batch size per gpu: 0 has to be greater than 0

if you dont pass --num_gpus, it will detect all gpus but launch the code in one. (default value= -1)





Note: we add to the config file `"steps_per_print": 100`so that DeepSpeed looger outputs a throughput and memory at every `100` steps.

The full list of options for the `zero_optimization` is in [the ZeRO API documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html).

### Hardware

8x NVIDIA GeForce GTX with 12GB RAM 

The batch size is set to `16384` i.e. 2048 per GPU on parallel runs, chosen as the largest batch that fitted in the single-GPU run of our benchmark.

### no DeepSpeed (single GPU run):

launched with `python gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json` (Note: the config is needed for all optimizer, datatypes, etc variables).

`nvidia-smi` metrics: memory usage of 10.44 GB in a single GPU, with GPU utilization at 100%

output:
```
RunningAvgSamplesPerSec=4954.027754429326, CurrSamplesPerSec=109021.60099884347, MemAllocated=0.15GB, MaxMemAllocated=9.89GB
```


### Deepspeed without ZeRO (Data Parallelism):

`nvidia-smi` metrics: memory usage : 2.23 MB, GPU utilization close to 100%

launched with `deepspeed --num_gpus=8 gptlite_deepspeed.py --deepspeed_config gptlite_config_ds.json`

Logger output in terminal (broken in several lines):

```
RunningAvgSamplesPerSec=33891.052271378714, CurrSamplesPerSec=378776.2256359377, MemAllocated=0.14GB, MaxMemAllocated=1.48GB
```

### ZeRO stage 1

The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition. The confile file needs a new field:

```
   "zero_optimization": {
	"stage": 1
    }
```

`nvidia-smi` metrics: memory usage : 2.28GB, GPU utilization close to 100%

Logger output in terminal:

```
RunningAvgSamplesPerSec=35155.892378540346,
CurrSamplesPerSec=311821.24019765767,
MemAllocated=0.07GB,
MaxMemAllocated=1.41GB
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

`nvidia-smi` metrics: memory usage : 4.19 GB, GPU utilization in the range of 95%-100%

Logger output in terminal:


```
RunningAvgSamplesPerSec=32593.177471945193,
CurrSamplesPerSec=239237.29211402152,
MemAllocated=0.07GB, MaxMemAllocated=3.13GB
```

### ZeRO stage 3:

The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes. The field in the config file needs to be changed to `"stage": 2`. 

`nvidia-smi` metrics: memory usage : 2424MB, GPU utilization in the range 15%-35%%
`nvidia-smi` metrics: memory usage : 2486MB, GPU utilization in the range 28%-57%%

Logger output in terminal:

```
[2023-09-14 23:57:11,583] [INFO] [timer.py:260:stop]
epoch=0/micro_step=2000/global_step=2000,
RunningAvgSamplesPerSec=140.71299830921302,
CurrSamplesPerSec=134.0175058462657,
MemAllocated=1.91GB,
MaxMemAllocated=2.01GB
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

`nvidia-smi` metrics: memory usage : 727MB, GPU utilization in the range 2%-38%%

Logger output in terminal:

```
[2023-09-15 00:41:08,912] [INFO] [timer.py:260:stop]
epoch=0/micro_step=2000/global_step=2000,
RunningAvgSamplesPerSec=53.90050911661717,
CurrSamplesPerSec=55.340898614186344,
MemAllocated=0.2GB,
MaxMemAllocated=0.31GB
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


### Files

For the full files mentioned here, see <a href="/assets/GPT-lite/gptlite_deepspeed.py">gptlite_deepspeed.py</a> for this code, <a href="/assets/GPT-lite/gptlite_model.py">gptlite_model.py</a> for the trimmed GPTlite model (model only, no run/valid loop), and <a href="/assets/GPT-lite/gptlite_config_ds.json">gptlite_config_ds.json</a> for the DeepSpeed config file.
