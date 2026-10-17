---
layout: post
title:  "Pipeline parallelism (1F1B, Zero Bubble, DualPipe)"
categories: ["Distributed Training"]
tags: ["pipeline parallelism", "1F1B", "zero bubble", "DualPipe", "DeepSpeed", "PyTorch"]
---

This post discusses and implements pipeline parallelism. Pipeline parallelism is one dimension of model parallelism, alongside data, tensor, context, sequence and expert parallelism.

Consider a model that is too large to fit in the local memory of a single process. A simple solution is to split the model along the layer dimension and assign a subset of layers to each process. The forward and backward passes are then performed by communicating activations and gradients between *connecting* processes. Each process holds a subset of layers and is called a **stage**. This type of parallelism is called **pipeline parallelism**. The following figure illustrates the process:

{: .picture}
<img width="53%" height="53%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline.svg"/>

{: .caption}
Left-to-right timeline of the serial training of a model divided across 4 compute units (GPUs) and 4 stages. Blue cells represent forward passes. Green cells represent backward passes, which take twice as long as forward passes. The number on each cell is the data sample index. Hatched cells represent idle time, i.e. a worker not performing any computation. Source: adapted from <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, arXiv)</a>

This method leads to a low utilization of compute resources: each process must wait for the connecting layers in other processes to be computed before it can perform its share of the forward/backward compute. For this reason, pipeline parallelism is usually combined with **micro-batching / gradient accumulation**. The mini-batch is split into several micro-batches, which are passed sequentially to the first process of the pipeline. When a process finishes the current micro-batch, it sends its activations to the following process and receives the activations of the next micro-batch from the previous process. This is mathematically equivalent to regular gradient accumulation. This approach is detailed in the paper [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Google, 2018, arXiv)](https://arxiv.org/abs/1811.06965) and is illustrated below:

{: .picture}
<img width="70%" height="70%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline_parallel.svg"/>

{: .caption}
A pipeline execution with gradient accumulation, computed as a sequence of 4 micro-batches. Source: adapted from <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, arXiv)</a>

A high micro-batching factor can lead to high memory usage, as activations must be stored until the backward pass of the current mini-batch completes. Two techniques reduce this cost: (1) **activation offloading**, which stores and loads activations to/from the CPU when needed; and (2) **activation checkpointing** at the beginning of every stage, which recomputes activations during the backward pass when needed, instead of keeping them in memory.

The other hyper-parameter to define is the number of stages. The examples so far used a number of stages equal to the number of workers, i.e. a single pipeline that spans all workers. However, several pipelines can run in parallel, and pipeline parallelism can be combined with other dimensions of parallelism. As an example, the following figure shows combinations of pipeline and data parallelism on an 8-GPU network:

{: .picture}
<img width="32%" height="32%" src="{{ site.assets }}/distributed-training/imgs/pipeline_8_stages.svg"/>
&nbsp;
<img width="32%" height="32%" src="{{ site.assets }}/distributed-training/imgs/pipeline_4_stages.svg"/>
&nbsp;
<img width="32%" height="32%" src="{{ site.assets }}/distributed-training/imgs/pipeline_2_stages.svg"/>

{: .caption}
An illustration of different combinations of pipeline and data parallelism. Left: 8 pipeline-parallel workers. Center: 2 data-parallel groups of 4 pipeline-parallel workers. Right: 4 data-parallel groups of 2 pipeline-parallel workers.

Every pipeline configuration has periods of idleness that cannot be removed. Much ongoing work aims to reduce them. As an example, [PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, arXiv)](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/) is a pipeline method that overlaps the forward and backward passes of different mini-batches. It keeps track of the version (micro-batch id) and stores several versions of the activations and parameters whose backward pass has not completed:

{: .picture}
<img width="64%" height="64%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline_parallel_microsoft.svg"/>

{: .caption}
The PipeDream scheduling algorithm. Several forward passes can be in flight, even if they belong to different micro-batches. Backward passes are prioritized over forward passes on each worker. Source: adapted from <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, arXiv)</a>

The caveat is that mixing forward and backward passes from different mini-batches can lead to wrong parameter updates. Therefore, the authors perform versioning of the parameters, i.e. the model keeps several versions of the same parameters. Forward passes use the latest version of the model layers, and the backward pass may use a previous version of the activations and parameters to compute the gradients. This leads to a substantial increase in memory requirements.

## Implementing pipeline parallelism with PyTorch

Before introducing DeepSpeed's 1F1B pipeline engine, it is useful to implement the simplest possible pipeline-parallel training loop in plain PyTorch. The goal of this section is *not* performance, but to show the mechanics of:

- partitioning a model into `num_stages` sequential stages,
- sending activations forward stage-by-stage,
- sending activation gradients backward stage-by-stage,
- and performing one optimizer step per mini-batch (optionally using micro-batches).

To match the scheduling in the first figure (no overlapping micro-batches between stages), the implementation uses a **non-overlapped schedule**: an entire micro-batch runs through all forward stages, then its backward pass runs all the way back, and only then does the next micro-batch start. This is the simplest schedule, but it leaves many bubbles (idle time) and is mainly useful for learning and debugging.

Below is a minimal example. It assumes:

- one pipeline group of size `num_stages` (often `num_stages == world_size` for a single pipeline),
- each rank is one stage (`stage_id = rank_in_pipeline`),
- each stage holds an `nn.Sequential` block of layers.

> **Note:** This is a toy implementation. Real systems also need shape/metadata communication, mixed precision, activation checkpointing, pipeline/data parallel composition, and better schedules (like 1F1B).

```python
import torch
import torch.distributed as dist
import torch.nn as nn

def split_layers(layers, num_stages):
    # naive equal split (by count). Real code should load-balance by params/runtime.
    idx = torch.arange(len(layers))
    chunks = torch.chunk(idx, num_stages)
    return [nn.Sequential(*[layers[i] for i in c.tolist()]) for c in chunks]

def pipeline_step_no_overlap(x_mb, y_mb, stage, stage_id, num_stages, criterion):
    """Run one micro-batch through the full pipeline forward, then full backward."""

    # ---- Forward ----
    if stage_id == 0:
        act = x_mb.detach().requires_grad_(True)
        act = stage(act)
        dist.send(act, dst=stage_id + 1)
    elif stage_id < num_stages - 1:
        # receive activation from previous stage
        act = torch.empty_like(x_mb)  # placeholder; in practice you must know the correct shape
        dist.recv(act, src=stage_id - 1)
        act = act.detach().requires_grad_(True)
        act = stage(act)
        dist.send(act, dst=stage_id + 1)
    else:
        # last stage: receive, compute loss
        act = torch.empty_like(x_mb)  # placeholder; in practice you must know the correct shape
        dist.recv(act, src=stage_id - 1)
        act = act.detach().requires_grad_(True)
        out = stage(act)
        loss = criterion(out, y_mb)
        loss.backward()
        # send grad of activation to previous stage
        dist.send(act.grad, dst=stage_id - 1)
        return loss.detach()

    # ---- Backward (non-last stages) ----
    # receive grad from next stage, backprop through local stage, send to previous
    grad_act = torch.empty_like(act)
    dist.recv(grad_act, src=stage_id + 1)

    act.backward(grad_act)

    if stage_id > 0:
        dist.send(act.grad, dst=stage_id - 1)

    return None

def train_one_minibatch(model_layers, num_stages, optimizer, criterion, x, y, micro_batches=1):
    """Each rank runs only its local stage."""
    rank = dist.get_rank()
    stage_id = rank  # single pipeline group example

    stages = split_layers(model_layers, num_stages)
    stage = stages[stage_id].to(x.device)

    optimizer.zero_grad(set_to_none=True)

    x_mbs = x.chunk(micro_batches, dim=0)
    y_mbs = y.chunk(micro_batches, dim=0)

    loss_out = None
    for x_mb, y_mb in zip(x_mbs, y_mbs):
        loss = pipeline_step_no_overlap(x_mb, y_mb, stage, stage_id, num_stages, criterion)
        if loss is not None:
            loss_out = loss_out + loss if loss_out is not None else loss

    # Each stage has its own optimizer step (weights are disjoint across stages)
    optimizer.step()

    return loss_out
```

This example shows the core idea: **activations flow forward, gradients flow backward**. The remaining sections use DeepSpeed to do the same efficiently, with better schedules (1F1B), load-balancing, activation checkpointing, and integration with other forms of parallelism.

## Implementing pipeline parallelism with DeepSpeed

The pipeline parallelism algorithm implemented in DeepSpeed is the [PipeDream-Flush implementation with default 1F1B scheduling](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/) (1 forward pass followed by 1 backward pass), but it is possible to [extend pipeline parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule) to other algorithms. The 1F1B algorithm performs a sequence of forward passes, and asynchronously starts the backward pass of each micro-batch whose forward pass has completed. It then waits for all forward and backward passes to complete before starting the next mini-batch.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/pipeline_gpipe.svg"/>

{: .caption}
GPipe on 4 GPUs with 8 micro-batches: all forward passes, then all backward passes. Each GPU keeps the activations of the 8 micro-batches until their backward passes.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/pipeline_1f1b.svg"/>

{: .caption}
1F1B on 4 GPUs with 8 micro-batches: after a warm-up of forward passes, each GPU alternates one forward and one backward pass. The iteration takes as long as with GPipe, but each GPU keeps the activations of at most 4 micro-batches instead of 8.

Pipeline parallelism is added to the `GPTlite` model implemented in the previous post, and is enabled by passing the number of stages as the `--pipeline_num_stages` argument (default: 0, no pipelining) on the command line:

```python
## train.py

def get_cmd_line_args(description='GPT lite on DeepSpeed'):
  # ...
  parser.add_argument('--pipeline_num_stages', type=int, default=0,
                      help='enable pipeline parallelism with N stages (0 means disabled)')
  # ...
```

DeepSpeed supports pipeline parallelism on any sequence of network blocks in an `nn.Sequential` container or `list`, which is split into pipeline stages. To expose pipeline parallelism, `GPTlite` gets a new method `to_layers()` that returns the sequence of operations to be executed. Note that `to_layers()` follows the same order as the `forward` pass of `GPTlite`, and that `self.blocks` is of type `nn.Sequential`:

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

The output of `layers` is of shape `B,S,C`, which is incompatible with [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in PyTorch. A quick fix is to add `lambda logits: torch.swapaxes(logits, 1, 2)` to `to_layers()` to change it to shape `B,C,S`. However, back-propagation may then fail with the error `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`, as discussed in bugs [4274](https://github.com/microsoft/DeepSpeed/issues/4274) and [4479](https://github.com/microsoft/DeepSpeed/issues/4479). This error is fixed with `outputs = outputs.requires_grad_(True)`. Alternatively, the loss function can perform the `view`/reshape change instead. This approach is cleaner, and is useful later for the pipeline parallelism use case:

```python
## train.py

class CrossEntropyLoss_FlatView(torch.nn.Module):
  def forward(self, logits, labels):
    B, S, C = logits.shape
    logits = logits.view(B*S, C)
    labels = labels.view(-1)
    return torch.nn.functional.cross_entropy(logits, labels)

def main_deepspeed(n_epochs=100, random_seed=42):
  # ...
  criterion = CrossEntropyLoss_FlatView()  # initialize loss function
```

Next, the DeepSpeed initialization code creates a pipeline wrapper around the model. This wrapped model is the new `model` variable passed to `deepspeed.initialize()`:

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

Finally, in the pipelining use case, the training iteration reduces to a call to `engine.train_batch()`, which is [equivalent to the forward pass, backward pass and gradient update of an entire mini-batch](https://www.deepspeed.ai/tutorials/pipeline/#training-loops) of `engine.gradient_accumulation_steps()` micro-batches:

```python
## train.py

def main_deepspeed(n_epochs=100, random_seed=42):
  # ...
  for epoch in range(n_epochs):
    if pipeline_num_stages:
      step_count = len(train_dataset) // engine.gradient_accumulation_steps()
      for step in range(step_count):
        loss = engine.train_batch()
    else:
      # ... forward, backward, and update step as before
```

An important nuance: by default, pipeline parallelism expects all mini-batches of the dataset (i.e. in every call to `train_batch()`) to have the same shape. If this is not the case, the shapes can be reset at the start of every mini-batch with `engine.reset_activation_shape()`. This adds a communication step that broadcasts the shape of the first micro-batch as the default for the remaining micro-batches. However, micro-batches of the same mini-batch cannot have different shapes. The only workaround is to trim or pad all micro-batches of a mini-batch to the same shape beforehand.

Finally, [pipeline parallelism is not compatible with ZeRO stages 2 or 3](https://deepspeed.readthedocs.io/en/latest/pipeline.html#pipeline-parallelism), as discussed [here](https://github.com/microsoft/DeepSpeed/issues/1110#issuecomment-850835817).

### Increasing compute and memory efficiency with LayerSpec (optional) 

The pipelining implementation of `GPTlite` above is neither memory efficient nor scalable, as each GPU replicates the whole model in memory (see [Memory-Efficient Model Construction](https://www.deepspeed.ai/tutorials/pipeline/#memory-efficient-model-construction) for details). The solution is the DeepSpeed class `LayerSpec` ([API](https://deepspeed.readthedocs.io/en/latest/pipeline.html#deepspeed.pipe.LayerSpec)), which delays the construction of modules until the model layers have been partitioned across workers. This way, each worker allocates only the layers assigned to it. To use it, a new model class `GPTlitePipeSpec` inherits from `PipelineModule`, with an `__init__` method that closely follows the `forward()` pass of the original `GPTlite`.

The `LayerSpec` constructor only accepts the type `nn.Module` as argument, but some operations, specifically the sum of embeddings in `forward()`, are not of type `nn.Module`. To overcome this, the class `EmbeddingsSum` encapsulates that logic in an `nn.Module`. The loss function is `CrossEntropyLoss_FlatView`. The full implementation of the pipeline class is:

```python
## gptlite.py

from deepspeed.pipe import PipelineModule, LayerSpec

class GPTlitePipeSpec(PipelineModule):

  class EmbeddingsSum(nn.Module):
    """Converts tok_emb + pos_emb into an nn.Module. Required for LayerSpec."""

    def __init__(self, vocab_size):
      super().__init__()
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
      B, S = idx.shape
      tok_emb = self.token_embedding_table(idx)
      pos_emb = self.position_embedding_table(torch.arange(S).to(idx.device))
      return tok_emb + pos_emb

  def __init__(self, vocab_size, pipe_kwargs):
    self.specs = \
      [ LayerSpec(GPTlitePipeSpec.EmbeddingsSum, vocab_size) ] + \
      [ LayerSpec(Block, n_embd, n_head) for _ in range(n_layer)] + \
      [ LayerSpec(nn.LayerNorm, n_embd),
        LayerSpec(nn.Linear, n_embd, vocab_size, bias=False) ]
    super().__init__(layers=self.specs, **pipe_kwargs)
```

Then, the flag `--pipeline_spec_layers` is added to the command line arguments to optionally enable this feature:

```python
## train.py

def get_cmd_line_args():
  # ...
  parser.add_argument("--pipeline_spec_layers", action="store_true",
                      help="enable LayerSpecs in pipeline parallelism")
```

and the `get_model()` method is changed to return the efficient pipeline variant:

```python
## gptlite.py

def get_model(criterion, vocab_size, pipeline_num_stages=0, pipeline_spec_layers=False):

  if pipeline_num_stages:
    if pipeline_spec_layers:
      model = GPTlitePipeSpec(vocab_size, pipe_kwargs=pipe_kwargs)
    else:
      # ... GPTlite model as before 
```

The rest of this post refers to the `LayerSpec`-based implementation of pipeline parallelism as *memory-efficient pipelining*.

For heterogeneous models, balancing the load across GPUs may be an issue. The load can be balanced by several metrics: runtime, memory usage, parameter count, etc. This post does not tune the [load balancing method for pipeline modules](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules), and uses the default `partition_method=parameters` instead. This method assigns layers to stages so that the number of parameters is balanced, i.e. stages may have different numbers of layers. Finally, in the extreme case where 1F1B is not the desired pipeline algorithm, it is possible to [extend pipeline parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule) with a different algorithm.

### Activation checkpointing

Activation checkpointing every X layers of the pipeline is enabled by setting that interval in the argument `activation_checkpoint_interval` of the `PipelineModule` constructor: 

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

However, activation checkpointing is hard to configure with pipelining when the checkpoint layer falls on another GPU. If a checkpoint layer is on a different GPU than the layer being back-propagated, extra communication is required. DeepSpeed does not appear to handle this case correctly, so there should be a checkpoint layer *at the beginning* of the first block on each GPU.

### Gradient accumulation and micro-batching

The micro-batching level is defined by the fields `train_micro_batch_size_per_gpu` (defaults to `train_batch_size`) or `gradient_accumulation_steps` (defaults to `1`) in the [DeepSpeed config file](https://www.deepspeed.ai/docs/config-json/). At runtime, `engine.gradient_accumulation_steps()` returns the number of micro-batches.

## Results

The config file <a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> was changed to run ZeRO stage 1. The experiments used different stage counts and the memory-efficient `LayerSpec` implementation of the GPT model (with `--pipeline_num_stages <num_stages> --pipeline_spec_layers`). Activation checkpointing was not used, due to an open bug [4274](https://github.com/microsoft/DeepSpeed/issues/4274). Each run used 1, 2, 4 or 8 pipeline stages. Stages are load-balanced with the default DeepSpeed algorithm, based on the parameter count. As an example, the partitioning of the GPTlite pipeline across 8 GPUs and 4 stages outputs:
   ```
RANK=0 STAGE=0 LAYERS=4 [0, 4)   STAGE_PARAMS=21256704 (21.257M)
RANK=2 STAGE=1 LAYERS=3 [4, 7)   STAGE_PARAMS=21256704 (21.257M)
RANK=4 STAGE=2 LAYERS=3 [7, 10)  STAGE_PARAMS=21256704 (21.257M)
RANK=6 STAGE=3 LAYERS=6 [10, 16) STAGE_PARAMS=21308160 (21.308M)
   ```

**Pipelining with optimized vs non-optimized memory efficiency implementation.** The `LayerSpec`-based implementation of the `PipelineModule` reduced memory consumption by about 40% for the GPTlite and deep benchmark models, when running pipeline parallelism with the highest stage count (8).

**Memory usage.** With pipeline parallelism, the first GPU appears to require more memory than the remaining GPUs. This should not be the case, particularly for the deep benchmark model, which has a quasi-ideal stage partitioning across GPUs. The higher memory usage on GPU 0 determines the maximum memory required, and balancing it would reduce that value. A [bug report](https://github.com/microsoft/DeepSpeed/issues/4477) was opened with DeepSpeed, and this analysis will be corrected after their feedback or fix.

The code is available in the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training). Detailed results for pipeline parallelism may be added in the future.


## Beyond 1F1B: reducing the pipeline bubble

Even with 1F1B, every stage idles for `(p-1)(F+B)` per iteration, where `p` is the number of stages and `F` and `B` are the times of the forward and backward passes of one micro-batch on one stage. Relative to the compute of `m` micro-batches, this is a bubble fraction of `(p-1)/m`. Therefore, 1F1B can only reduce the bubble with more micro-batches per iteration (i.e. a larger batch or smaller micro-batches) or with fewer stages. Newer schedules reduce the bubble directly. They are not part of DeepSpeed, where they would require a [custom schedule](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule), but they are available in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), in [PyTorch's `torch.distributed.pipelining`](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) and in [DeepSeek's DualPipe repository](https://github.com/deepseek-ai/DualPipe).

The diagrams in this section, like the GPipe and 1F1B diagrams above, were generated from the reference implementation of each schedule, for 4 GPUs and 8 micro-batches (6 GPUs and 12 micro-batches for DualPipe), and the diagrams on 4 GPUs share the same time scale. Time is normalized so that every GPU performs the same amount of compute in all diagrams on 4 GPUs: a forward pass takes 1 unit and a backward pass takes 2 units (1+1 when it is split into the gradients for inputs and for weights, as explained below). Schedules that place two model chunks on each GPU run each chunk in half the time. Communication time is not modeled. The number in each cell is the micro-batch id, hatched cells are idle time, pale cells with gray outlines are the forward passes of the next iteration, and the dashed line marks the end of the 1F1B schedule.

### Interleaved 1F1B

[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (NVIDIA, 2021, arXiv)](https://arxiv.org/abs/2104.04473) assigns several non-contiguous chunks of layers (*virtual stages*) to each GPU, instead of a single block of consecutive layers. With 4 GPUs and 2 chunks per GPU, the model is split into 8 chunks: GPU 0 holds chunks 0 and 4, GPU 1 holds chunks 1 and 5, and so on, so every micro-batch loops through the GPUs twice. Smaller chunks make the pipeline fill and drain faster, and the bubble shrinks by the number of chunks per GPU `v`, to `(p-1)(F+B)/v`. The cost is `v` times more point-to-point communication, and more activation memory, as the first GPU keeps more micro-batches in flight. The original implementation also requires the number of micro-batches to be a multiple of the number of GPUs. This schedule is available in Megatron-LM and as `ScheduleInterleaved1F1B` in PyTorch.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/pipeline_interleaved_1f1b.svg"/>

{: .caption}
Interleaved 1F1B with 2 model chunks per GPU. Light and dark cells refer to the first and second chunk on each GPU. Interleaving halves the bubble of 1F1B.

### Zero Bubble (ZB-H1 and ZB-H2)

[Zero Bubble Pipeline Parallelism (Sea AI Lab, ICLR 2024, arXiv)](https://arxiv.org/abs/2401.10241) builds on the observation that the backward pass consists of two independent computations. For a linear layer `y = xA`, the gradient with respect to the input, `dL/dx = dL/dy Aᵀ`, is needed right away by the previous stage, so it is on the critical path. The gradient with respect to the weights, `dL/dA = xᵀ dL/dy`, is only needed by the optimizer step, so it can be postponed. Zero Bubble therefore splits every backward pass in two: the *backward for inputs* is computed and sent upstream immediately, and the *backward for weights* is deferred to fill the bubbles. The paper proposes two handcrafted schedules:

- **ZB-H1** keeps the peak activation memory of 1F1B. Gradients travel back through the pipeline faster, and the deferred backward for weights passes fill the end of the iteration, reducing the bubble to `(p-1)(F+B-2W)`, where `W` is the time of the backward for weights. When the forward, backward for inputs and backward for weights take the same time, this is a third of the 1F1B bubble.
- **ZB-H2** allows up to `2p-1` micro-batches in flight on the first stage, about twice the activation memory of 1F1B. The extra forward passes fill the warm-up bubble, and the backward for weights passes are reordered at the end of the iteration. This turns the schedule into a parallelogram that interlocks with the previous and next iterations, so the bubble disappears.

However, the optimizer step usually synchronizes all stages, e.g. to compute the global gradient norm for gradient clipping, or to check for NaN/Inf values in mixed precision training, and this synchronization breaks the parallelogram. The authors replace it with an *optimizer post-validation*: each stage updates its weights right away and, in the rare case where gradient clipping or the NaN/Inf check is triggered, the update is rolled back and redone. They also provide an algorithm that searches for the best schedule, given the measured time of each pass and of the communication, and a memory limit. In their experiments, Zero Bubble achieved up to 23% higher throughput than 1F1B under a similar memory limit, and up to 31% with a relaxed memory limit. The implementation is available as a [fork of Megatron-LM](https://github.com/sail-sg/zero-bubble-pipeline-parallelism) and as `ScheduleInterleavedZeroBubble` in PyTorch.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h1.svg"/>

{: .caption}
ZB-H1, with the activation memory of 1F1B. Every backward pass is split into a backward for inputs and a backward for weights, and the latter fill the bubbles.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h2.svg"/>

{: .caption}
ZB-H2, with about twice the activation memory of 1F1B. When the optimizer step does not synchronize the stages, the next iteration starts on the first GPUs while the last GPUs finish the current one, and the previous iteration fills the idle time at the start in the same way.

### ZB-V

The Zero Bubble paper also introduces **ZB-V**, which places two model chunks on each GPU in a *V shape*: the model is split into `2p` chunks and GPU `i` holds chunks `i` and `2p-1-i`. A micro-batch goes down the GPUs through the first half of the model and comes back up through the second half. The first and last chunks (embeddings and loss) are both on GPU 0, and no communication is needed at the turn on the last GPU. When the forward, backward for inputs and backward for weights take the same time, ZB-V has zero bubble (again with the optimizer post-validation) with the same peak activation memory as 1F1B, balanced across all GPUs. The cost is twice the point-to-point communication of 1F1B. A follow-up paper, [Pipeline Parallelism with Controllable Memory (Sea AI Lab, 2024, arXiv)](https://arxiv.org/abs/2405.15362), generalizes V-shape schedules to a family that trades bubbles for memory, down to half (V-Half) and a third (V-Min) of the activation memory of 1F1B. ZB-V is available as `ScheduleZBVZeroBubble` in PyTorch.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_v.svg"/>

{: .caption}
ZB-V with 8 model chunks on 4 GPUs. Light cells refer to the first chunk on each GPU (chunks 0 to 3, going down) and dark cells to the second chunk (chunks 4 to 7, going back up). As in ZB-H2, the previous iteration fills the idle time at the start when the optimizer step does not synchronize the stages.

### DualPipe

[DeepSeek-V3 Technical Report (DeepSeek-AI, 2024, arXiv)](https://arxiv.org/abs/2412.19437) introduced DualPipe, later open-sourced in the [DualPipe repository](https://github.com/deepseek-ai/DualPipe). It was designed for Mixture-of-Experts models trained with expert parallelism across nodes, where the all-to-all communication that dispatches tokens to experts and combines their outputs takes roughly as long as the computation itself. The key idea is to overlap the forward pass of one micro-batch with the backward pass of another. Both passes are divided into attention, all-to-all dispatch, MLP and all-to-all combine components (with the backward further split into backward for inputs and backward for weights, as in Zero Bubble). These components are then rearranged, with a manually tuned split of the GPU's streaming multiprocessors between communication and computation, so that the communication of each pass is hidden behind the computation of the other.

To have enough of these forward-backward pairs, DualPipe uses a *bidirectional* pipeline: half of the micro-batches enter at the first GPU and flow forward, while the other half enter at the last GPU and flow in the opposite direction. GPU `i` therefore holds stages `i` and `p-1-i`, i.e. the pipeline holds two copies of the model parameters, whose gradients are summed across GPUs `i` and `p-1-i` before the optimizer step. DeepSeek notes that, in their setup, this duplication does not increase memory significantly, due to the large expert-parallel size. The bubble is `(p/2-1)(F&B+B-3W)`, where `F&B` is the time of an overlapped forward-backward pair, and each GPU stores the activations of `p+1` micro-batches. Compared to [Chimera (ETH Zurich, 2021, arXiv)](https://arxiv.org/abs/2107.06925), an earlier bidirectional schedule, DualPipe only requires the number of stages and micro-batches to be even (and not the micro-batches to be a multiple of the stages), and its bubble and activation memory do not grow with the number of micro-batches.

The diagrams in this section model only computation, so DualPipe's bubble looks comparable to ZB-H1's. The main advantage of DualPipe is that, within each black-bordered pair, the all-to-all communication of expert parallelism is hidden behind useful computation.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipe.svg"/>

{: .caption}
DualPipe on 6 GPUs: micro-batches 1 to 6 (light) enter at GPU 0, and micro-batches 7 to 12 (dark) enter at GPU 5. Each GPU holds two pipeline stages. Cells with a shared black border are the forward and backward passes of two different micro-batches executed together, so that the communication of one overlaps with the computation of the other.

### DualPipeV

Sea AI Lab later observed that DualPipe is made of two mirrored halves, as GPUs `i` and `p-1-i` hold the same stages and run the same schedule ([DualPipe could be better without the Dual (Sea AI Lab, 2025)](https://huggingface.co/blog/ufotalent/cut-in-half)). Keeping only the first half of the GPUs and placing the remaining stages on them in a V shape, as in ZB-V, gives a schedule with the same bubble and activation memory as DualPipe for the same number of stages, but on half the GPUs and without duplicated parameters. DeepSeek adopted this *cut-in-half* schedule as **DualPipeV** in the DualPipe repository, and recent PyTorch versions provide it as `ScheduleDualPipeV`. As in ZB-V, all micro-batches enter at GPU 0, go down the GPUs and come back up, and the loss is computed on GPU 0.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipev.svg"/>

{: .caption}
DualPipeV with 8 model chunks on 4 GPUs. Light cells refer to the first chunk on each GPU (going down) and dark cells to the second chunk (going back up). Cells with a shared black border are overlapped forward and backward passes of two different micro-batches.

### Summary

The table below compares the schedules above on 4 GPUs with 8 micro-batches. Idle time is measured in forward passes. Activation memory is measured in units of the activations of one micro-batch on one 1F1B stage, on the GPU that needs the most, assuming the backward for inputs and the backward for weights each release half of a micro-batch's activations.

| Schedule | Idle time per GPU | Peak activation memory | Parameters per GPU | Main cost |
|:--|:--:|:--:|:--:|:--|
| GPipe | 9 | 8 | 1× | activation memory of all micro-batches |
| 1F1B | 9 | 4 | 1× | - |
| Interleaved 1F1B (2 chunks) | 4.5 | 5.5 | 1× | 2× point-to-point communication |
| ZB-H1 | 3 | 4 | 1× | split backward pass |
| ZB-H2 | 0 (3) | 7 | 1× | 2× activation memory, optimizer post-validation |
| ZB-V | 0 (1.5) | 4 | 1× | 2× point-to-point communication, optimizer post-validation |
| DualPipe | 2 | 5 | 2× | duplicated parameters, overlapped forward-backward implementation |
| DualPipeV | 3 | 4.5 | 1× | overlapped forward-backward implementation |

For ZB-H2 and ZB-V, the value in parentheses is the idle time when the optimizer step synchronizes all stages. DualPipe and DualPipeV also hide the all-to-all communication of expert parallelism behind computation, which is not captured in this table.

