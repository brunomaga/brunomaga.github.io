---
layout: post
title:  "Scaling a GPT model with DeepSpeed"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

How to (first deepspeed params, second app params, args is needed):
```
deepspeed --num_gpus 8  [--hostfile hostfile] [--autotuning run]
```

test/inference-test.py --deepspeed_config ds_config.py<batch_size> --test_performance


To add to the code:
todo save model every X iterations

# Reminder:

train_batch_size = #GPUs * train_micro_batch_size_per_gpu * gradient_accumulation_steps
if you have a `train_batch_size` too small, you'll get 
AssertionError: Micro batch size per gpu: 0 has to be greater than 0

if you dont pass --num_gpus, it will detect all gpus but launch the code in one. (default value= -1)

if model is too big, use parameter `reduce_bucket_size` in config:
Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes




Note: we add to the config file `"steps_per_print": 2000`so that DeepSpeed looger outputs a throughput and memory at every `2000` steps.

The full list of options for the `zero_optimization` is in [the ZeRO API documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html).

### no ZeRO (default, only Data Parallelism):

`nvidia-smi` metrics: memory usage : 501MB, GPU utilization in the range 70%-82%

Logger output in terminal (broken in several lines):

```
[2023-09-15 00:09:45,317] [INFO] [timer.py:260:stop]
epoch=0/micro_step=2000/global_step=2000,
RunningAvgSamplesPerSec=502.1679292031866,
CurrSamplesPerSec=519.4063914924576,
MemAllocated=0.14GB,
MaxMemAllocated=0.23GB
```

### ZeRO stage 1:

The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition. The confile file needs a new field:

```
   "zero_optimization": {
	"stage": 1
    }
```

`nvidia-smi` metrics: memory usage : 485MB, GPU utilization in the range 69%-81%

Logger output in terminal:

```
[2023-09-15 00:00:22,209] [INFO] [timer.py:260:stop]
epoch=0/micro_step=2000/global_step=2000,
RunningAvgSamplesPerSec=357.6931784027813,
CurrSamplesPerSec=372.6779510418055,
MemAllocated=0.07GB,
MaxMemAllocated=0.16GB
```

### ZeRO stage 2:

The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states. These fields were changed to the config file:

```
   "zero_optimization": {
	"stage": 2,
	"reduce_bucket_size": 5e8   #this MAY be necessary otherwise it will run out of memory
	"all_reduce_bucket_size": 5e8   #this MAY be necessary otherwise it will run out of memory
    }
```

`nvidia-smi` metrics: memory usage : 2321MB, GPU utilization in the range 63%-80%

Logger output in terminal:


```
[2023-09-14 23:42:18,918] [INFO] [timer.py:260:stop]
epoch=0/micro_step=2000/global_step=2000,
RunningAvgSamplesPerSec=242.05217588341768,
CurrSamplesPerSec=248.45012772574137,
MemAllocated=0.07GB,
MaxMemAllocated=1.96GB
```

### ZeRO stage 3:

The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes. The field in the config file needs to be changed to `"stage": 2`. 

`nvidia-smi` metrics: memory usage : 2424MB, GPU utilization in the range 15%-35%%

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

For the full config file, see [cifar_config.json]().
