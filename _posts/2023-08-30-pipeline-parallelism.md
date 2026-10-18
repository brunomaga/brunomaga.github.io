---
layout: post
title:  "Pipeline parallelism (1F1B, Zero Bubble, DualPipe)"
categories: ["Distributed Training"]
tags: ["pipeline parallelism", "1F1B", "zero bubble", "DualPipe", "DeepSpeed", "PyTorch"]
---

Pipeline parallelism splits a model along the layer dimension, for models that do not fit in the memory of one GPU. Each process holds a block of consecutive layers, called a **stage**, and the stages form a [pipeline](https://en.wikipedia.org/wiki/Pipeline_(computing)). The forward pass sends the activations from each stage to the next one, and the backward pass sends the gradients back. Pipeline parallelism is one of the parallelism dimensions of distributed training, with data, tensor, context, sequence and expert parallelism.

{: .picture}
<img width="46.4%" height="46.4%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline.svg"/>

{: .caption}
Training of a model split into 4 stages on 4 workers, without pipelining: the forward pass of a mini-batch goes from worker 1 to worker 4, and its backward pass, which takes twice as long, comes back, so only one worker computes at a time. The number in each cell is the mini-batch index. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

With one input at a time, only one stage computes at any moment, and each GPU is busy $$1/p$$ of the time for $$p$$ stages. GPipe ([GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965), Huang et al., 2019) splits the mini-batch into $$m$$ micro-batches and feeds them to the first stage one after the other. When a stage finishes a micro-batch, it sends the activations to the next stage and starts the next micro-batch. This is gradient accumulation: the gradients of the $$m$$ micro-batches add up before one optimizer step.

{: .picture}
<img width="63.8%" height="63.8%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline_parallel.svg"/>

{: .caption}
Pipeline parallelism with gradient accumulation (GPipe) on 4 workers: the mini-batch is split into 4 micro-batches, all forward passes run before the backward passes, and the gradients are added at the pipeline flush, before the micro-batches of the next mini-batch (5 and 6) start. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

The GPUs still idle while the pipeline fills at the start of an iteration and drains at the end. This idle time is the **pipeline bubble**. Let $$F$$ and $$B$$ be the times of the forward and backward passes of one micro-batch on one stage. An iteration takes $$(m+p-1)(F+B)$$, so each GPU idles for $$(p-1)(F+B)$$, a fraction $$(p-1)/m$$ of its compute time $$m(F+B)$$.

GPipe keeps the activations of all $$m$$ micro-batches until their backward passes. Two techniques reduce this memory. **Activation offloading** moves the activations to CPU memory and back when they are needed. **Activation checkpointing** keeps only the input of each stage and recomputes the other activations in the backward pass, as GPipe does.

The number of stages can be smaller than the number of GPUs. Several pipelines then run in parallel, and pipeline parallelism combines with the other parallelism dimensions, for example with data parallelism on 8 GPUs:

{: .picture}
<img width="32.5%" height="32.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_8_stages.svg"/>
&nbsp;
<img width="32.5%" height="32.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_4_stages.svg"/>
&nbsp;
<img width="32.5%" height="32.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_2_stages.svg"/>

{: .caption}
Combinations of pipeline and data parallelism on 8 workers, for one micro-batch: forward passes in blue, backward passes in green, idle time hatched. Left: one pipeline of 8 stages. Center: 2 data-parallel groups of 4 pipeline stages. Right: 4 data-parallel groups of 2 pipeline stages.

[PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/) (Narayanan et al., 2019) does not wait for a mini-batch to finish before the next one starts. In the steady state, each stage alternates one forward pass and one backward pass of different mini-batches (*1F1B*), so no GPU idles:

{: .picture}
<img width="56.3%" height="56.3%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline_parallel_microsoft.svg"/>

{: .caption}
The PipeDream schedule on 4 workers. In the startup state, worker 1 admits 4 mini-batches; in the steady state, each worker alternates the forward pass of one mini-batch and the backward pass of an earlier one, so several mini-batches are in flight at the same time. Backward passes are prioritized over forward passes on each worker. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

Each stage updates its weights after every backward pass, while other mini-batches are still in flight. A backward pass must use the weights of its own forward pass. PipeDream therefore keeps one version of the weights per mini-batch in flight (*weight stashing*), up to $$p$$ versions on the first stage. Each forward pass uses the latest version. The extra versions increase the memory, and the updates are asynchronous: the first stage computes each gradient with weights that are $$p-1$$ updates old.

## PyTorch Implementation

A minimal pipeline in plain PyTorch shows the mechanics. The ranks form one pipeline, and each rank holds one stage, an `nn.Sequential` of consecutive layers. For each micro-batch, the forward pass sends the activations from stage to stage with `dist.send` and `dist.recv`. The last stage computes the loss and starts the backward pass, and each stage sends the gradient of its input to the previous stage. A micro-batch runs the whole forward and backward passes before the next one starts, as in the first figure. After the last micro-batch, each stage updates its own parameters:

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

This toy code leaves out what a real system needs: the communication of tensor shapes, mixed precision, activation checkpointing, the composition with data parallelism, load balancing and a better schedule. The DeepSpeed implementation below provides them.

## DeepSpeed Implementation

DeepSpeed implements the [1F1B schedule of PipeDream](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/) with a flush at the end of every mini-batch, a schedule called PipeDream-Flush ([Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2006.09503), Narayanan et al., 2021). Stage $$i$$, counted from 0, runs up to $$p-i$$ forward passes, then alternates one backward and one forward pass, and ends with its last backward passes. Each stage then updates its weights once per mini-batch. The updates are synchronous, as in GPipe, and one version of the weights is enough. The bubble is the same as with GPipe, but stage $$i$$ keeps the activations of at most $$p-i$$ micro-batches instead of $$m$$:

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/distributed-training/imgs/pipeline_gpipe.svg"/>

{: .caption}
GPipe on 4 GPUs with 8 micro-batches: all forward passes, then all backward passes. Each GPU keeps the activations of the 8 micro-batches until their backward passes. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/distributed-training/imgs/pipeline_1f1b.svg"/>

{: .caption}
1F1B on 4 GPUs with 8 micro-batches: after a warm-up of forward passes, each GPU alternates one forward and one backward pass. The iteration takes as long as with GPipe, but each GPU keeps the activations of at most 4 micro-batches instead of 8. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

For training, DeepSpeed releases implement only this schedule (`TrainSchedule`), and other schedules need a [custom schedule](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule).

This section adds pipeline parallelism to the DeepSpeed training of `GPTlite` from the previous post. The command-line argument `--pipeline_num_stages` sets the number of stages (default: 0, no pipelining):

```python
## train.py

def get_cmd_line_args(description='GPT lite on DeepSpeed'):
  # ...
  parser.add_argument('--pipeline_num_stages', type=int, default=0,
                      help='enable pipeline parallelism with N stages (0 means disabled)')
  # ...
```

DeepSpeed splits a list or an `nn.Sequential` of layers into stages. The new method `to_layers()` of `GPTlite` returns its layers in the order of `forward()`, and `*self.blocks` unpacks the `nn.Sequential` of transformer blocks:

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

DeepSpeed calls a layer that is not an `nn.Module`, such as the first lambda, in the forward pass, but does not register its parameters. The optimizer then does not update the embedding tables. The module `EmbeddingsSum` below avoids this.

The last layer outputs logits of shape (batch, sequence, vocabulary), but [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) expects the classes in the second dimension. A layer `lambda logits: torch.swapaxes(logits, 1, 2)` at the end of `to_layers()` fixes the shape. The backward pass can then fail with `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` (DeepSpeed issues [4274](https://github.com/microsoft/DeepSpeed/issues/4274) and [4479](https://github.com/microsoft/DeepSpeed/issues/4479)). Setting `outputs = outputs.requires_grad_(True)` avoids the error. A cleaner fix flattens the logits and the labels in the loss function, which DeepSpeed runs on the last stage:

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

`get_model()` wraps the layers in a `PipelineModule`, which is the model passed to `deepspeed.initialize()`:

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

The training loop calls `engine.train_batch()`, which runs the [forward passes, backward passes and weight update of one mini-batch](https://www.deepspeed.ai/tutorials/pipeline/#training-loops) of `engine.gradient_accumulation_steps()` micro-batches. The [DeepSpeed config file](https://www.deepspeed.ai/docs/config-json/) sets this number with `train_micro_batch_size_per_gpu` (default: `train_batch_size`) or `gradient_accumulation_steps` (default: 1):

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

By default, the pipeline engine expects all mini-batches to have the same shape. When the shape changes between mini-batches, `engine.reset_activation_shape()` at the start of a mini-batch adds a communication step. It sends the shape of the first micro-batch to the next stage, which uses it for all micro-batches of the mini-batch. The micro-batches of one mini-batch cannot differ in shape, so they are trimmed or padded to one shape beforehand.

The [pipeline engine](https://deepspeed.readthedocs.io/en/latest/pipeline.html#pipeline-parallelism) of DeepSpeed does not support ZeRO stages 2 and 3 ([discussion](https://github.com/microsoft/DeepSpeed/issues/1110#issuecomment-850835817)).

### Memory-Efficient Model Construction with LayerSpec

The code above builds the whole model on every GPU, which keeps the parameters of all layers in memory ([Memory-Efficient Model Construction](https://www.deepspeed.ai/tutorials/pipeline/#memory-efficient-model-construction)). [`LayerSpec`](https://deepspeed.readthedocs.io/en/latest/pipeline.html#deepspeed.pipe.LayerSpec) delays the construction of each layer until DeepSpeed has assigned the layers to the stages, so each GPU allocates only the layers of its stage. `LayerSpec` takes an `nn.Module` class and the arguments of its constructor, so the sum of the embeddings becomes the module `EmbeddingsSum`. The class `GPTlitePipeSpec` inherits from `PipelineModule` and lists the layers in the order of `GPTlite.forward()`. The loss function is still `CrossEntropyLoss_FlatView`, passed in `pipe_kwargs`:

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

The flag `--pipeline_spec_layers` enables this version:

```python
## train.py

def get_cmd_line_args():
  # ...
  parser.add_argument("--pipeline_spec_layers", action="store_true",
                      help="enable LayerSpecs in pipeline parallelism")
```

and `get_model()` returns it:

```python
## gptlite.py

def get_model(criterion, vocab_size, pipeline_num_stages=0, pipeline_spec_layers=False):

  if pipeline_num_stages:
    if pipeline_spec_layers:
      model = GPTlitePipeSpec(vocab_size, pipe_kwargs=pipe_kwargs)
    else:
      # ... GPTlite model as before 
```

DeepSpeed [balances the stages](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules) by the number of trainable parameters (`partition_method="parameters"`, the default), by the number of layers (`"uniform"`), or by the number of layers whose class name matches a regular expression (`"type:[regex]"`). The default suits models whose compute time per layer grows with its parameter count. The runs below use the default, so the stages can hold different numbers of layers.

### Activation Checkpointing

The argument `activation_checkpoint_interval` of the `PipelineModule` constructor checkpoints the activations every `activation_checkpoint_interval` layers:

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

DeepSpeed applies the interval within each stage. The first checkpoint starts at the first layer of the stage, and its input is the activation received from the previous stage, which the stage keeps for the backward pass. The recomputation therefore needs no data from another GPU.

## Results

The runs set ZeRO stage 1 in the config file <a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a>, the `LayerSpec` version (`--pipeline_num_stages <num_stages> --pipeline_spec_layers`), and 1, 2, 4 or 8 stages on 8 GPUs. They do not use activation checkpointing, due to the open DeepSpeed issue [4274](https://github.com/microsoft/DeepSpeed/issues/4274). The config sets 256 samples per mini-batch and micro-batches of 1 sample. With 8 stages, the one pipeline runs $$m = 256$$ micro-batches per iteration, and the bubble of 1F1B is $$7/256$$, under 3% of the compute.

The following partition comes from the `to_layers()` version with the swap lambda (16 layers), on 8 GPUs with 4 stages. Ranks 0, 2, 4 and 6 hold the stages of one of the 2 pipelines:
   ```
RANK=0 STAGE=0 LAYERS=4 [0, 4)   STAGE_PARAMS=21256704 (21.257M)
RANK=2 STAGE=1 LAYERS=3 [4, 7)   STAGE_PARAMS=21256704 (21.257M)
RANK=4 STAGE=2 LAYERS=3 [7, 10)  STAGE_PARAMS=21256704 (21.257M)
RANK=6 STAGE=3 LAYERS=6 [10, 16) STAGE_PARAMS=21308160 (21.308M)
   ```

Each transformer block holds 7.09M parameters, and stages 0 to 2 hold 3 blocks each. The embedding lambda on stage 0 counts 0 parameters, as its tables are not registered. Stage 3 holds the last 3 blocks, the final layer norm, the output layer and the swap lambda.

**LayerSpec.** With 8 stages, the `LayerSpec` version used about 40% less memory than the `to_layers()` version. This held for GPTlite and for the deep benchmark model of `benchmark.py`, a stack of 2048 linear layers of width 256 with ReLU activations.

**Memory per GPU.** The first GPU used more memory than the others ([DeepSpeed issue 4477](https://github.com/microsoft/DeepSpeed/issues/4477)), also with the deep benchmark model, whose stages hold the same number of parameters. 1F1B causes such an imbalance: stage $$i$$ keeps the activations of up to $$p-i$$ micro-batches, and DeepSpeed allocates $$\max(2, \min(p-i, m))$$ buffers on stage $$i$$ for the inputs and outputs of the micro-batches in flight. The first GPU therefore sets the memory needed per GPU. The V-shape schedules below balance it.

The code is in the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training).

## Beyond 1F1B: Reducing the Pipeline Bubble

1F1B reduces the bubble fraction $$(p-1)/m$$ only with more micro-batches per iteration (a larger mini-batch or smaller micro-batches) or with fewer stages. Newer schedules reduce the bubble directly. They are available in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), in [PyTorch's `torch.distributed.pipelining`](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) and in [DeepSeek's DualPipe repository](https://github.com/deepseek-ai/DualPipe).

The diagrams of this section, like the GPipe and 1F1B diagrams above, come from the reference implementation of each schedule, for 4 GPUs and 8 micro-batches (6 GPUs and 12 micro-batches for DualPipe). The diagrams on 4 GPUs share the same time scale and the same compute per GPU. A forward pass takes 1 unit, and a backward pass takes 2 units, or 1+1 when it is split into the gradients for inputs and for weights (see below). Schedules with two model chunks per GPU run each chunk in half the time. Communication time is not modeled. The number in each cell is the micro-batch id, and hatched cells are idle time. Pale cells with gray outlines are the forward passes of the next iteration, and the dashed line marks the end of the 1F1B schedule.

### Interleaved 1F1B

[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (Narayanan et al., 2021) assigns several non-contiguous chunks of layers (*virtual stages*) to each GPU, instead of one block of consecutive layers. With 4 GPUs and 2 chunks per GPU, the model is split into 8 chunks: GPU 0 holds chunks 0 and 4, GPU 1 holds chunks 1 and 5, and so on, so every micro-batch loops through the GPUs twice. Smaller chunks fill and drain the pipeline faster: with $$v$$ chunks per GPU, the bubble shrinks to $$(p-1)(F+B)/v$$. The cost is $$v$$ times more point-to-point communication, and more activation memory, as the first GPU keeps more micro-batches in flight. The original implementation also requires the number of micro-batches to be a multiple of the number of GPUs. The schedule is available in Megatron-LM and as `ScheduleInterleaved1F1B` in PyTorch.

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_interleaved_1f1b.svg"/>

{: .caption}
Interleaved 1F1B with 2 model chunks per GPU. Light and dark cells refer to the first and second chunk on each GPU. Interleaving halves the bubble of 1F1B. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

### Zero Bubble (ZB-H1 and ZB-H2)

[Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) (Qi et al., 2024) splits the backward pass into two independent computations. Take a linear layer $$y = xA$$. The previous stage needs the gradient with respect to the input, $$\partial L / \partial x = (\partial L / \partial y) A^T$$, right away, so it is on the critical path. Only the optimizer step needs the gradient with respect to the weights, $$\partial L / \partial A = x^T (\partial L / \partial y)$$, so it can wait. Zero Bubble computes the *backward for inputs* and sends it upstream at once, and runs the *backward for weights* later, in the bubbles. The paper proposes two handcrafted schedules:

- **ZB-H1** keeps the peak activation memory of 1F1B. The gradients travel back through the pipeline faster, and the deferred backward-for-weights passes fill the end of the iteration. The bubble drops to $$(p-1)(F+B-2W)$$, where $$W$$ is the time of the backward for weights. When the forward, the backward for inputs and the backward for weights take the same time, this is a third of the 1F1B bubble.
- **ZB-H2** allows up to $$2p-1$$ micro-batches in flight on the first stage, about twice the activation memory of 1F1B. The extra forward passes fill the warm-up bubble, and the backward-for-weights passes are reordered at the end of the iteration. The schedule becomes a parallelogram that interlocks with the previous and next iterations, and the bubble disappears.

The optimizer step usually synchronizes all stages, for example to compute the global gradient norm for gradient clipping, or to check for NaN and Inf values in mixed precision training. This synchronization breaks the parallelogram. The authors replace it with an *optimizer post-validation*. Each stage updates its weights at once. In the rare case where gradient clipping or the NaN/Inf check triggers, the stage rolls the update back and redoes it. They also give an algorithm that searches for the best schedule, given the measured time of each pass and of the communication, and a memory limit. In their experiments, Zero Bubble reached up to 23% higher throughput than 1F1B under a similar memory limit, and up to 31% with a relaxed memory limit. The code is available as a [fork of Megatron-LM](https://github.com/sail-sg/zero-bubble-pipeline-parallelism) and as `ScheduleInterleavedZeroBubble` in PyTorch.

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h1.svg"/>

{: .caption}
ZB-H1, with the activation memory of 1F1B. Every backward pass is split into a backward for inputs and a backward for weights, and the latter fill the bubbles. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h2.svg"/>

{: .caption}
ZB-H2, with about twice the activation memory of 1F1B. When the optimizer step does not synchronize the stages, the next iteration starts on the first GPUs while the last GPUs finish the current one; in the same way, the previous iteration fills the idle time at the start. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

### ZB-V

The Zero Bubble paper also introduces **ZB-V**, which places two model chunks on each GPU in a *V shape*: the model is split into $$2p$$ chunks, and GPU $$i$$ holds chunks $$i$$ and $$2p-1-i$$. A micro-batch goes down the GPUs through the first half of the model and comes back up through the second half. The first and last chunks (embeddings and loss) are both on GPU 0, and the turn on the last GPU needs no communication. When the forward, the backward for inputs and the backward for weights take the same time, ZB-V has zero bubble, again with the optimizer post-validation. Its peak activation memory equals that of 1F1B, but is balanced across the GPUs. The cost is twice the point-to-point communication of 1F1B. A follow-up paper, [Pipeline Parallelism with Controllable Memory](https://arxiv.org/abs/2405.15362) (Qi et al., 2024), generalizes V-shape schedules to a family that trades bubbles for memory, down to half (V-Half) and a third (V-Min) of the activation memory of 1F1B. ZB-V is available as `ScheduleZBVZeroBubble` in PyTorch.

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_v.svg"/>

{: .caption}
ZB-V with 8 model chunks on 4 GPUs. Light cells refer to the first chunk on each GPU (chunks 0 to 3, going down) and dark cells to the second chunk (chunks 4 to 7, going back up). As in ZB-H2, the previous iteration fills the idle time at the start when the optimizer step does not synchronize the stages. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

### DualPipe

The [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024) introduced DualPipe, later open-sourced in the [DualPipe repository](https://github.com/deepseek-ai/DualPipe). DualPipe targets [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) models trained with expert parallelism across nodes. There, the [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) communication that dispatches the tokens to the experts and combines their outputs takes about as long as the computation. DualPipe overlaps the forward pass of one micro-batch with the backward pass of another. It splits both passes into four components: attention, all-to-all dispatch, MLP and all-to-all combine. The backward pass further splits attention and MLP into backward for inputs and backward for weights, as in Zero Bubble. DualPipe then reorders the components so that the communication of each pass hides behind the computation of the other. A manually tuned split of the streaming multiprocessors of the GPU assigns some of them to communication and the rest to computation.

DualPipe needs many such forward-backward pairs, so it uses a *bidirectional* pipeline. Half of the micro-batches enter at the first GPU and flow forward, and the other half enter at the last GPU and flow the other way. GPU $$i$$ therefore holds stages $$i$$ and $$p-1-i$$. The pipeline holds two copies of the model parameters, and the gradients of each stage are summed across GPUs $$i$$ and $$p-1-i$$ before the optimizer step. DeepSeek notes that, with their large expert-parallel size, this duplication does not increase the memory significantly. The bubble is $$(p/2-1)(F\&B+B-3W)$$, where $$F\&B$$ is the time of an overlapped forward-backward pair, and each GPU stores the activations of $$p+1$$ micro-batches. [Chimera](https://arxiv.org/abs/2107.06925) (Li and Hoefler, 2021), an earlier bidirectional schedule, requires the number of micro-batches to be a multiple of the number of stages. DualPipe requires only even numbers of stages and micro-batches, and its bubble and activation memory do not grow with the number of micro-batches.

The diagrams model only the computation, so the bubble of DualPipe looks close to that of ZB-H1. The gain of DualPipe is that the all-to-all communication of expert parallelism hides behind computation within each black-bordered pair.

{: .picture}
<img width="86.8%" height="86.8%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipe.svg"/>

{: .caption}
DualPipe on 6 GPUs: micro-batches 1 to 6 (light) enter at GPU 0, and micro-batches 7 to 12 (dark) enter at GPU 5. Each GPU holds two pipeline stages. Cells with a shared black border are the forward and backward passes of two different micro-batches executed together, so that the communication of one overlaps with the computation of the other. Adapted from: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

### DualPipeV

Sea AI Lab later observed that DualPipe is made of two mirrored halves, as GPUs $$i$$ and $$p-1-i$$ hold the same stages and run the same schedule ([DualPipe could be better without the Dual](https://huggingface.co/blog/ufotalent/cut-in-half), Sea AI Lab, 2025). Keeping only the first half of the GPUs, and placing the remaining stages on them in a V shape as in ZB-V, gives a schedule with the bubble and activation memory of DualPipe for the same number of stages. This schedule runs on half the GPUs and has no duplicated parameters. DeepSeek adopted this *cut-in-half* schedule as **DualPipeV** in the DualPipe repository. Recent PyTorch versions provide it as `ScheduleDualPipeV`, and the development branch of DeepSpeed as `DualPipeVSchedule`. As in ZB-V, all micro-batches enter at GPU 0, go down the GPUs and come back up, and GPU 0 computes the loss.

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipev.svg"/>

{: .caption}
DualPipeV with 8 model chunks on 4 GPUs. Light cells refer to the first chunk on each GPU (going down) and dark cells to the second chunk (going back up). Cells with a shared black border are overlapped forward and backward passes of two different micro-batches. Adapted from: [DualPipe](https://github.com/deepseek-ai/DualPipe)

### Summary

The table compares the schedules on 4 GPUs with 8 micro-batches. Idle time is in units of forward passes. Activation memory is in units of the activations of one micro-batch on one 1F1B stage, on the GPU that needs the most. The backward for inputs and the backward for weights are assumed to release half of these activations each.

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

For ZB-H2 and ZB-V, the value in parentheses is the idle time when the optimizer step synchronizes all stages. DualPipe and DualPipeV also hide the all-to-all communication of expert parallelism behind computation, which the table does not capture.

With many more micro-batches than stages, the 1F1B bubble $$(p-1)/m$$ is small, and 1F1B is the simplest choice. The other schedules pay off with few micro-batches per pipeline, for example at a large data-parallel scale with a fixed global batch. Interleaving and Zero Bubble reduce the bubble for more communication or more memory. DualPipe and DualPipeV pay off for Mixture-of-Experts models with expert parallelism across nodes.

