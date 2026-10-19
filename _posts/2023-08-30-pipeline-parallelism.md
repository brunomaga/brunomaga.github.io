---
layout: post
title:  "Pipeline parallelism (1F1B, Zero Bubble, DualPipe)"
categories: ["Distributed Training"]
tags: ["pipeline parallelism", "1F1B", "zero bubble", "DualPipe", "DeepSpeed", "PyTorch"]
---

In 2018, the papers [PipeDream](https://arxiv.org/abs/1806.03377) and [GPipe](https://arxiv.org/abs/1811.06965) placed consecutive layers of a neural network on different accelerators, and passed several inputs through them at the same time, as in an assembly line.

This post discusses and implements pipeline parallelism, first in plain PyTorch and then with DeepSpeed, and compares the newer schedules that reduce its idle time, from interleaved 1F1B to Zero Bubble and DualPipe. Pipeline parallelism is another dimension of model parallelism, in the universe of data, tensor, pipeline, context, sequence and expert parallelism.

Imagine a model that is too large to fit in the local memory of a single process. A simple way to overcome this is to split the model across the layer dimension and delegate a subset of layers to each process. Then the forward and backward passes can be done by communicating activations and gradients between *connecting* processes.

Each process is responsible for a subset of layers and is called a **stage**, as in the stages of a [pipeline](https://en.wikipedia.org/wiki/Pipeline_(computing)). This type of parallelism is called **pipeline parallelism**. The following picture gives a simple illustration of the process:

{: .picture}
<img width="36.4%" height="36.4%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline.svg"/>

{: .caption}
Training of a model split into 4 stages on 4 workers, without pipelining: the forward pass of a mini-batch goes from worker 1 to worker 4, and its backward pass, which takes twice as long, comes back, so only one worker computes at a time. The number in each cell is the mini-batch index. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

Now note that the above method would yield a low utilization of compute resources: processes would always have to wait for the connecting layers in other processes to be computed before doing their share of the forward/backward compute. So pipeline parallelism is usually combined with **micro-batching / gradient accumulation**.

In practice, the mini-batch is split into several micro-batches that are passed sequentially to the first process of the pipeline. When a process finishes the current micro-batch, it sends its activations to the following process and receives the activations of the next micro-batch from the previous process. This is mathematically equivalent to regular gradient accumulation. This approach is detailed in the paper [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Google, 2018, arXiv)](https://arxiv.org/abs/1811.06965) and can be illustrated as:

{: .picture}
<img width="59.4%" height="59.4%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline_parallel.svg"/>

{: .caption}
Pipeline parallelism with gradient accumulation (GPipe) on 4 workers: the mini-batch is split into 4 micro-batches, all forward passes run before the backward passes, and the gradients are added at the pipeline flush, before the next mini-batch starts. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

Setting a high micro-batching factor can lead to high memory usage, as activations have to be stored until the backward pass for the current mini-batch finishes. This can be improved by (1) **activation offloading**, which stores and loads activations to/from the CPU when needed; and (2) **activation checkpointing** at the beginning of every stage, which recomputes activations in the backward pass instead of storing them. In brief, offloading saves memory at the cost of data transfers between the GPU and the CPU, and checkpointing at the cost of extra computation.

The other hyper-parameter to define is the number of stages. So far, the examples above used a number of stages equal to the number of workers, i.e. a single pipeline that spans all workers. However, it is possible to have multiple pipelines in parallel and to combine pipeline parallelism with other dimensions of parallelism. As an example, here are some combinations of pipeline and data parallelism on an 8-GPU network:

{: .picture}
<img width="32.4%" height="32.4%" src="{{ site.assets }}/distributed-training/imgs/pipeline_8_stages.svg"/>
&nbsp;
<img width="32.4%" height="32.4%" src="{{ site.assets }}/distributed-training/imgs/pipeline_4_stages.svg"/>
&nbsp;
<img width="32.4%" height="32.4%" src="{{ site.assets }}/distributed-training/imgs/pipeline_2_stages.svg"/>

{: .caption}
Combinations of pipeline and data parallelism on 8 workers, for one micro-batch: forward passes in blue, backward passes in green, idle time hatched. Left: one pipeline of 8 stages. Center: 2 data-parallel groups of 4 pipeline stages. Right: 4 data-parallel groups of 2 pipeline stages.

In the pictures above, fewer stages lead to less idle time but to more layers per GPU, so the number of stages is usually the smallest one that fits the model in memory. However, no matter which pipeline configuration is used, there are periods of idleness that cannot be removed.

With this in mind, there is plenty of ongoing work to improve this. As an example, [PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, SOSP 2019)](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/) is a pipeline method that overlaps the forward and backward passes of different mini-batches. To do so, it keeps track of the version (mini-batch id) and stores several versions of the activations and parameters whose backward pass has not completed:

{: .picture}
<img width="56.3%" height="56.3%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline_parallel_microsoft.svg"/>

{: .caption}
The PipeDream schedule on 4 workers. In the startup state, worker 1 admits 4 mini-batches; in the steady state, each worker alternates the forward pass of one mini-batch and the backward pass of an earlier one, so several mini-batches are in flight at the same time. Backward passes are prioritized over forward passes on each worker. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

Where is the caveat? In practice, mixing forward and backward passes from different mini-batches can lead to wrong parameter updates. Therefore, the authors perform versioning of the parameters (*weight stashing*), effectively having several versions of the same parameters in the model.

Forward passes use the latest version of the model layers, and the backward pass may use a previous version of the activations and parameters to compute the gradients. This leads to a substantial increase in memory requirements.

## Implementing pipeline parallelism with PyTorch

Before jumping into DeepSpeed's 1F1B pipeline engine, it is helpful to implement the simplest possible pipeline-parallel training loop in plain PyTorch. The goal of this section is *not* performance, but only to show the mechanics of:

- partitioning a model into `num_stages` sequential stages,
- sending activations forward stage-by-stage,
- sending activation gradients backward stage-by-stage,
- and performing one optimizer step per mini-batch (optionally using micro-batches).

To match the scheduling shown in the first picture (no overlapping micro-batches between stages), the implementation uses a **non-overlapped schedule**. An entire micro-batch runs through all forward stages, then its backward pass runs all the way back, and only then does the next micro-batch start. This is the simplest schedule, but it leaves many bubbles (idle time) and is mainly useful for learning and debugging.

Below is a minimal example. It assumes:

- one pipeline group of size `num_stages` (often `num_stages == world_size` for a single pipeline),
- each rank is one stage (`stage_id = rank_in_pipeline`),
- each stage holds an `nn.Sequential` block of layers.

This is a toy implementation: real systems also need shape/metadata communication, [mixed precision](https://en.wikipedia.org/wiki/Mixed-precision_arithmetic), activation checkpointing, pipeline/data parallel composition, and better schedules (like 1F1B).

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

And that is all: **activations flow forward, gradients flow backward**. The remaining sections use DeepSpeed to do the same thing, but efficiently and with better schedules (1F1B), load-balancing, activation checkpointing, and integration with other forms of parallelism.

## Implementing pipeline parallelism with DeepSpeed

The pipeline parallelism algorithm implemented in DeepSpeed is the [PipeDream-Flush implementation with default 1F1B scheduling](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/) (1 forward pass followed by 1 backward pass), but it is possible to [extend pipeline parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule) to other algorithms. The **1F1B** algorithm performs a sequence of forward passes, and asynchronously starts the backward pass of each micro-batch whose forward pass has completed. It then waits for all forward and backward passes to complete before starting the next mini-batch.

PipeDream-Flush was introduced in [Memory-Efficient Pipeline-Parallel DNN Training (Microsoft, ICML 2021, arXiv)](https://arxiv.org/abs/2006.09503) and, unlike PipeDream, keeps a single version of the parameters. The regular (GPipe) and 1F1B algorithms are compared below:

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_gpipe.svg"/>

{: .caption}
GPipe on 4 GPUs with 8 micro-batches: all forward passes, then all backward passes. Each GPU keeps the activations of the 8 micro-batches until their backward passes. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_1f1b.svg"/>

{: .caption}
1F1B on 4 GPUs with 8 micro-batches: after a warm-up of forward passes, each GPU alternates one forward and one backward pass. The iteration takes as long as with GPipe, but each GPU keeps the activations of at most 4 micro-batches instead of 8. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

Pipeline parallelism is added to the `GPTlite` model implemented in the previous post, and enabled by passing the number of stages as the `--pipeline_num_stages` argument (default: 0, no pipelining) on the command line:

```python
## train.py

def get_cmd_line_args(description='GPT lite on DeepSpeed'):
  # ...
  parser.add_argument('--pipeline_num_stages', type=int, default=0,
                      help='enable pipeline parallelism with N stages (0 means disabled)')
  # ...
```

DeepSpeed supports pipeline parallelism on any sequence of network blocks in an `nn.Sequential` container or `list`, which will be broken into pipeline stages. Pipeline parallelism is exposed in the model by creating a method `to_layers()` in `GPTlite` that returns the sequence of actions to be executed. Note that `to_layers()` follows the same order as the `forward` pass of `GPTlite`, and that `self.blocks` is of type `nn.Sequential`, so that `*self.blocks` unpacks it into individual blocks that can be placed on different stages:

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

Note that the output of `layers` is of shape `B,S,C`, which is incompatible with [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in PyTorch, as it expects the classes in the second dimension. A quick fix is to add `lambda logits: torch.swapaxes(logits, 1, 2)` to `to_layers()` to make it of shape `B,C,S`. However, when back-propagating, one may bump into the error `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`, as discussed in bugs [4274](https://github.com/microsoft/DeepSpeed/issues/4274) and [4479](https://github.com/microsoft/DeepSpeed/issues/4479), and `outputs = outputs.requires_grad_(True)` is then needed to avoid it.

Alternatively, the loss function can be adapted to do the `view`/reshape change instead. This is a cleaner approach, and will be useful later for the pipeline parallelism use case:

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

As a next step, the DeepSpeed initialization code must create a pipeline wrapper around the model. This wrapped model is the new `model` variable that will be passed to `deepspeed.initialize()`:

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

Finally, the training iteration code in the pipelining use case is reduced to a call to `engine.train_batch()`, which is [equivalent to the forward pass, backward pass and gradient update of an entire mini-batch](https://www.deepspeed.ai/tutorials/pipeline/#training-loops) of `engine.gradient_accumulation_steps()` micro-batches:

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

An important nuance: by default, pipeline parallelism expects all mini-batches of the dataset (i.e. in every call to `train_batch()`) to be of the same shape. If this is not the case, the shapes can be reset at the onset of every mini-batch by running `engine.reset_activation_shape()`. This introduces an additional communication step to send the shape of the first micro-batch as the default for the remaining micro-batches.

However, up to DeepSpeed 0.14, it is not possible to have different shapes across micro-batches, and the only workaround is to trim or pad all micro-batches of a mini-batch to the same shape beforehand. Since version 0.15, with the argument `dynamic_shape=True` of `PipelineModule`, the shape of every micro-batch is sent instead, at some cost in performance.

As a final remark, [pipeline parallelism is not compatible with ZeRO stages 2 or 3](https://deepspeed.readthedocs.io/en/latest/pipeline.html#pipeline-parallelism), as discussed [here](https://github.com/microsoft/DeepSpeed/issues/1110#issuecomment-850835817).

### Increasing compute and memory efficiency with LayerSpec (optional) 

The implementation of pipelining for the `GPTlite` model above is neither memory efficient nor scalable, as each GPU replicates the whole model in memory. Details are given in [Memory-Efficient Model Construction](https://www.deepspeed.ai/tutorials/pipeline/#memory-efficient-model-construction). So this section uses the DeepSpeed class `LayerSpec` ([API](https://deepspeed.readthedocs.io/en/latest/pipeline.html#deepspeed.pipe.LayerSpec)), which delays the construction of modules until the model layers have been partitioned across workers, so that each worker allocates only the layers it is assigned to. To do this, a new model class `GPTlitePipeSpec` is created that inherits from `PipelineModule`, with an `__init__` method that closely follows the `forward()` pass of the original `GPTlite`.

The tricky bit here is that the `LayerSpec` constructor only works with the type `nn.Module` as argument, and some operations, specifically the sum of embeddings in `forward()`, are not of type `nn.Module`. To overcome this, the class `EmbeddingsSum` is created to encapsulate that logic into an `nn.Module`. Also, `CrossEntropyLoss_FlatView` is used as the loss function. The full implementation for the pipeline class is then:

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

then the flag `--pipeline_spec_layers` is added to the command line arguments, so that this feature can be optionally enabled:

```python
## train.py

def get_cmd_line_args():
  # ...
  parser.add_argument("--pipeline_spec_layers", action="store_true",
                      help="enable LayerSpecs in pipeline parallelism")
```

and the `get_model()` method is changed to retrieve the efficient pipeline variant as:

```python
## gptlite.py

def get_model(criterion, vocab_size, pipeline_num_stages=0, pipeline_spec_layers=False):

  if pipeline_num_stages:
    if pipeline_spec_layers:
      model = GPTlitePipeSpec(vocab_size, pipe_kwargs=pipe_kwargs)
    else:
      # ... GPTlite model as before 
```

From now on, the `LayerSpec`-based implementation of pipeline parallelism is referred to as *memory-efficient pipelining*.

For heterogeneous models, load balancing the model across GPUs may be an issue, as the slowest stage sets the pace of the whole pipeline. There are several metrics to balance the load by: runtime, memory usage, parameter count, etc. Here, the [load balancing method for pipeline modules](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules) is not tuned, and the default `partition_method=parameters` is used instead. This assigns layers to stages in a way that balances the number of parameters, i.e. stages may have different numbers of layers.

### Activation checkpointing

Introducing activation checkpointing every X layers in the pipeline is straightforward: the interval only needs to be specified in the argument `activation_checkpoint_interval` of the `PipelineModule` constructor:

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

However, activation checkpointing is also tricky to configure when using pipelining if the checkpoint layer falls on another GPU. The rationale is that if a checkpoint layer falls on a different GPU than the layer being back-propagated, extra communication is required. DeepSpeed avoids this use case, as it starts the checkpoint intervals at the first layer of each stage, so that there is always a checkpoint layer *at the beginning* of the first block on each GPU.

### Gradient accumulation and micro-batching

The micro-batching level can be defined by setting the fields `train_micro_batch_size_per_gpu` (defaults to `train_batch_size`) or `gradient_accumulation_steps` (defaults to `1`) in the [DeepSpeed config file](https://www.deepspeed.ai/docs/config-json/). At runtime, the number of micro-batches can be retrieved by `engine.gradient_accumulation_steps()`.

## Results

The config <a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a> was changed to run ZeRO stage 1, and the execution was tested with different stage counts and the memory-efficient `LayerSpec` implementation of the GPT model (with `--pipeline_num_stages <num_stages> --pipeline_spec_layers`). Activation checkpointing was not used, due to an open bug [4274](https://github.com/microsoft/DeepSpeed/issues/4274). Each run used 1, 2, 4 or 8 pipeline stages.

The load balancing of stages relies on the default DeepSpeed algorithm, based on the parameter count. As an example, for the partitioning of the GPTlite pipeline across 8 GPUs and 4 stages, DeepSpeed outputs:
   ```
RANK=0 STAGE=0 LAYERS=4 [0, 4)   STAGE_PARAMS=21256704 (21.257M)
RANK=2 STAGE=1 LAYERS=3 [4, 7)   STAGE_PARAMS=21256704 (21.257M)
RANK=4 STAGE=2 LAYERS=3 [7, 10)  STAGE_PARAMS=21256704 (21.257M)
RANK=6 STAGE=3 LAYERS=6 [10, 16) STAGE_PARAMS=21308160 (21.308M)
   ```

**Pipelining with optimized vs non-optimized memory efficiency implementation.** Using the `LayerSpec`-based implementation of the `PipelineModule` in the pipeline runs resulted in a reduction of about 40% in memory consumption, when running pipeline parallelism with the highest stage count (8). This was observed for both the GPTlite and the deep benchmark models (the latter with 2048 layers of 256 neurons).

**Memory usage.** With pipeline parallelism, the first GPU seems to require more memory than the remaining GPUs. This should not be the case, particularly for the deep benchmark model, where a quasi-ideal stage partitioning across GPUs can be guaranteed. 1F1B does keep more micro-batches in flight on the first stage, but their activations take only a few MB in the deep benchmark model.

This disparity in memory usage on GPU 0 determines the maximum memory required, and balancing it would bring that value down. A [bug report](https://github.com/microsoft/DeepSpeed/issues/4477) was opened with DeepSpeed, and this analysis will be corrected after their feedback or fix.

This code is available in the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training). Detailed results for pipeline parallelism may be added in the future, when time allows.


## Beyond 1F1B: reducing the pipeline bubble

Even with 1F1B, every stage idles for $$(p-1)(F+B)$$ per iteration (the **pipeline bubble**), where $$p$$ is the number of stages and $$F$$ and $$B$$ are the times of the forward and backward passes of one micro-batch on one stage. Relative to the compute of $$m$$ micro-batches, $$m(F+B)$$, this is a bubble fraction of $$(p-1)/m$$, so 1F1B can only reduce it with more micro-batches per iteration (i.e. a larger batch or smaller micro-batches) or with fewer stages. Newer schedules reduce the bubble directly, and matter most when there are few micro-batches per pipeline. They are not part of the DeepSpeed releases, where they would require a [custom schedule](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule), but they are available in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), in [PyTorch's `torch.distributed.pipelining`](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) and in [DeepSeek's DualPipe repository](https://github.com/deepseek-ai/DualPipe).

The diagrams in this section, like the GPipe and 1F1B diagrams above, were generated from the reference implementation of each schedule, for 4 GPUs and 8 micro-batches (8 GPUs and 16 micro-batches for DualPipe). Time is normalized so that every GPU performs the same amount of compute in all diagrams on 4 GPUs, drawn on the same time scale.

A forward pass takes 1 unit and a backward pass takes 2 units (1+1 when it is split into the gradients for inputs and for weights, as explained below). Schedules that place two model chunks on each GPU (interleaved 1F1B, ZB-V and DualPipeV) run each chunk in half the time. Communication time is not modeled. The number in each cell is the micro-batch id, and hatched cells are idle time.

### Interleaved 1F1B

[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (NVIDIA, 2021, arXiv)](https://arxiv.org/abs/2104.04473) introduced the **interleaved 1F1B** schedule, which assigns several non-contiguous chunks of layers (*virtual stages*) to each GPU, instead of a single block of consecutive layers. With 4 GPUs and 2 chunks per GPU, the model is split into 8 chunks: GPU 0 holds chunks 0 and 4, GPU 1 holds chunks 1 and 5, and so on, so every micro-batch loops through the GPUs twice.

Smaller chunks make the pipeline fill and drain faster, and the bubble shrinks by the number of chunks per GPU $$v$$, to $$(p-1)(F+B)/v$$. The price is $$v$$ times more point-to-point communication, and more activation memory, as the first GPU keeps more micro-batches in flight. The original implementation also requires the number of micro-batches to be a multiple of the number of GPUs. This schedule is available in Megatron-LM and as `ScheduleInterleaved1F1B` in PyTorch, and can be illustrated as:

{: .picture}
<img width="75.3%" height="75.3%" src="{{ site.assets }}/distributed-training/imgs/pipeline_interleaved_1f1b.svg"/>

{: .caption}
Interleaved 1F1B with 2 model chunks per GPU. Light and dark cells refer to the first and second chunk on each GPU. Interleaving halves the bubble of 1F1B. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

### Zero Bubble (ZB-H1 and ZB-H2)

[Zero Bubble Pipeline Parallelism (Sea AI Lab, ICLR 2024, arXiv)](https://arxiv.org/abs/2401.10241) builds on the observation that the backward pass consists of two independent computations. For a linear layer $$y = xA$$ and a loss $$L$$, the gradient with respect to the input, $$\partial L / \partial x = (\partial L / \partial y) A^T$$, is needed right away by the previous stage, so it is on the critical path. The gradient with respect to the weights, $$\partial L / \partial A = x^T (\partial L / \partial y)$$, is only needed by the optimizer step, so it can be postponed.

Zero Bubble therefore splits every backward pass in two: the *backward for inputs* is computed and sent upstream immediately, and the *backward for weights* is deferred to fill the bubbles. The paper proposes two handcrafted schedules:

- **ZB-H1** keeps the peak activation memory of 1F1B. Gradients travel back through the pipeline faster, and the deferred backward for weights passes fill the end of the iteration, reducing the bubble to $$(p-1)(F+B-2W)$$, where $$W$$ is the time of the backward for weights ($$B$$ is still the time of the whole backward pass). When the forward, backward for inputs and backward for weights take the same time, this is a third of the 1F1B bubble.
- **ZB-H2** allows up to $$2p-1$$ micro-batches in flight on the first stage, about twice the activation memory of 1F1B. The extra forward passes fill the warm-up bubble, and the backward for weights passes are reordered at the end of the iteration. This turns the schedule into a parallelogram that interlocks with the previous and next iterations, so the bubble disappears.

There is a catch: the optimizer step usually synchronizes all stages, e.g. to compute the global gradient norm for gradient clipping, or to check for NaN/Inf values in mixed precision training, and this synchronization breaks the parallelogram. The authors replace it with an *optimizer post-validation*: each stage updates its weights right away and, in the rare case where gradient clipping or the NaN/Inf check is triggered, the update is rolled back and redone.

They also provide an algorithm that searches for the best schedule, given the measured time of each pass and of the communication, and a memory limit. In their experiments, Zero Bubble outperformed the throughput of 1F1B by up to 23% under a similar memory limit, and by up to 31% with a relaxed memory limit. The implementation is available as a [fork of Megatron-LM](https://github.com/sail-sg/zero-bubble-pipeline-parallelism) and as `ScheduleInterleavedZeroBubble` in PyTorch. The ZB-H1 and ZB-H2 schedules are illustrated below:

{: .picture}
<img width="71.6%" height="71.6%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h1.svg"/>

{: .caption}
ZB-H1, with the activation memory of 1F1B. Every backward pass is split into a backward for inputs and a backward for weights, and the latter fill the bubbles. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

{: .picture}
<img width="71.6%" height="71.6%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h2.svg"/>

{: .caption}
ZB-H2, with about twice the activation memory of 1F1B. When the optimizer step does not synchronize the stages, the next iteration starts on the first GPUs while the last GPUs finish the current one; in the same way, the previous iteration fills the idle time at the start. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

### ZB-V

The Zero Bubble paper also introduces **ZB-V**, which places two model chunks on each GPU in a *V shape*: the model is split into $$2p$$ chunks and GPU $$i$$ holds chunks $$i$$ and $$2p-1-i$$. A micro-batch goes down the GPUs through the first half of the model and comes back up through the second half. The first and last chunks (embeddings and loss) are both on GPU 0, and no communication is needed at the turn on the last GPU.

When the forward, backward for inputs and backward for weights take the same time, ZB-V has zero bubble (again with the optimizer post-validation) with the same peak activation memory as 1F1B, balanced across all GPUs. The price is twice the point-to-point communication of 1F1B. A follow-up paper, [Pipeline Parallelism with Controllable Memory (Sea AI Lab, 2024, arXiv)](https://arxiv.org/abs/2405.15362), generalizes V-shape schedules to a family that trades bubbles for memory, down to half (V-Half) and a third (V-Min) of the activation memory of 1F1B. ZB-V is available as `ScheduleZBVZeroBubble` in PyTorch, and is illustrated below:

{: .picture}
<img width="67.9%" height="67.9%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_v.svg"/>

{: .caption}
ZB-V with 8 model chunks on 4 GPUs. Light cells refer to the first chunk on each GPU (chunks 0 to 3, going down) and dark cells to the second chunk (chunks 4 to 7, going back up). As in ZB-H2, the previous iteration fills the idle time at the start when the optimizer step does not synchronize the stages. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

### DualPipe

[DeepSeek-V3 Technical Report (DeepSeek-AI, 2024, arXiv)](https://arxiv.org/abs/2412.19437) introduced **DualPipe**, later open-sourced in the [DualPipe repository](https://github.com/deepseek-ai/DualPipe). It was designed for [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) models trained with expert parallelism across nodes, where the [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) communication that dispatches tokens to experts and combines their outputs takes roughly as long as the computation itself.

DualPipe overlaps the forward pass of one micro-batch with the backward pass of another. Both passes are divided into attention, all-to-all dispatch, MLP and all-to-all combine components (with the backward further split into backward for inputs and backward for weights, as in Zero Bubble). These components are then rearranged, with a manually tuned split of the GPU's streaming multiprocessors between communication and computation, so that the communication of each pass is hidden behind the computation of the other.

To have enough of these forward-backward pairs, DualPipe uses a *bidirectional* pipeline: half of the micro-batches enter at the first GPU and flow forward, while the other half enter at the last GPU and flow in the opposite direction. GPU $$i$$ therefore holds stages $$i$$ and $$p-1-i$$, i.e. the pipeline holds two copies of the model parameters, whose gradients are summed across GPUs $$i$$ and $$p-1-i$$ before the optimizer step. DeepSeek notes that, in their setup, this duplication does not increase memory significantly, due to the large expert-parallel size.

The bubble is given by $$(p/2-1)(F\&B+B-3W)$$, where $$F\&B$$ is the time of an overlapped forward-backward pair, and each GPU stores the activations of $$p+1$$ micro-batches. Compared to [Chimera (ETH Zurich, 2021, arXiv)](https://arxiv.org/abs/2107.06925), an earlier bidirectional schedule, DualPipe only requires the number of stages and micro-batches to be even (and not the micro-batches to be a multiple of the stages). Also, its bubble and activation memory do not grow with the number of micro-batches.

Note that the diagrams model only computation, so DualPipe's bubble looks comparable to ZB-H1's. Its main advantage is that, within each black-bordered pair, the all-to-all communication of expert parallelism is hidden behind useful computation. DualPipe on 8 GPUs is illustrated in the following picture:

{: .picture}
<img width="99.1%" height="99.1%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipe.svg"/>

{: .caption}
DualPipe on 8 GPUs: micro-batches 1 to 8 (light) enter at GPU 0, and micro-batches 9 to 16 (dark) enter at GPU 7. Each GPU holds two pipeline stages. Cells with a shared black border are the forward and backward passes of two different micro-batches executed together, so that the communication of one overlaps with the computation of the other. Adapted from: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

### DualPipeV

Sea AI Lab later noticed that DualPipe is made of two mirrored halves, as GPUs $$i$$ and $$p-1-i$$ hold the same stages and run the same schedule ([DualPipe could be better without the Dual (Sea AI Lab, 2025)](https://huggingface.co/blog/ufotalent/cut-in-half)). Keeping only the first half of the GPUs and placing the remaining stages on them in a V shape, as in ZB-V, gives a schedule with the same bubble and activation memory as DualPipe for the same number of stages. It runs on half the GPUs and holds no duplicated parameters.

As for ZB-V, the price is twice the point-to-point communication of 1F1B. DeepSeek adopted this *cut-in-half* schedule as **DualPipeV** in the DualPipe repository, and recent PyTorch versions provide it as `ScheduleDualPipeV`. As in ZB-V, all micro-batches enter at GPU 0, go down the GPUs and come back up, and the loss is computed on GPU 0, as illustrated below:

{: .picture}
<img width="71.6%" height="71.6%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipev.svg"/>

{: .caption}
DualPipeV with 8 model chunks on 4 GPUs. Light cells refer to the first chunk on each GPU (going down) and dark cells to the second chunk (going back up). Cells with a shared black border are overlapped forward and backward passes of two different micro-batches. Adapted from: [DualPipe](https://github.com/deepseek-ai/DualPipe)

### Summary

The table below compares the schedules in the diagrams above, on 4 GPUs and 8 micro-batches (also for DualPipe). Idle time is measured in forward passes. Activation memory is measured in units of the activations of one micro-batch on one 1F1B stage, on the GPU that needs the most, assuming the backward for inputs and the backward for weights each release half of a micro-batch's activations.

| Schedule | Idle time per GPU | Peak activation memory | Parameters per GPU | Main cost |
|:--|:--:|:--:|:--:|:--|
| GPipe | 9 | 8 | 1× | activation memory of all micro-batches |
| 1F1B | 9 | 4 | 1× | - |
| Interleaved 1F1B (2 chunks) | 4.5 | 5.5 | 1× | 2× point-to-point communication |
| ZB-H1 | 3 | 4 | 1× | split backward pass |
| ZB-H2 | 0 (3) | 7 | 1× | 2× activation memory, optimizer post-validation |
| ZB-V | 0 (1.5) | 4 | 1× | 2× point-to-point communication, optimizer post-validation |
| DualPipe | 2 | 5 | 2× | duplicated parameters, overlapped forward-backward implementation |
| DualPipeV | 3 | 4.5 | 1× | 2× point-to-point communication, overlapped forward-backward implementation |

For ZB-H2 and ZB-V, the value in parentheses is the idle time when the optimizer step synchronizes all stages. DualPipe and DualPipeV also hide the all-to-all communication of expert parallelism behind computation, which is not captured in this table.

In practice, ZB-H1 reduces the bubble of 1F1B without extra activation memory, while ZB-H2 and ZB-V remove it when there is spare memory or bandwidth, and when the optimizer step can be validated after the update. On the same GPUs, DualPipeV runs twice as many stages as DualPipe, each of half the size, and trades a larger bubble for half of the parameter memory.

To find more about the topic, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) of Hugging Face describes these schedules, and how to combine pipeline parallelism with data, tensor, context and expert parallelism.

