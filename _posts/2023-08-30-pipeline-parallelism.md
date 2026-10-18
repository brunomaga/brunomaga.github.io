---
layout: post
title:  "Pipeline parallelism (1F1B, Zero Bubble, DualPipe)"
categories: ["Distributed Training"]
tags: ["pipeline parallelism", "1F1B", "zero bubble", "DualPipe", "DeepSpeed", "PyTorch"]
---

Consider a model that is too large to fit in the memory of a single GPU. A simple solution is to split it along the layer dimension, and to give each process a block of consecutive layers, called a **stage**. The stages then work like the stages of a [pipeline](https://en.wikipedia.org/wiki/Pipeline_(computing)): in the forward pass, each stage computes the activations of its layers and sends its output to the next stage, and in the backward pass, each stage computes the gradients of its layers and sends the gradient of its input back to the previous stage. This type of parallelism is called **pipeline parallelism**, and it is one of the dimensions of distributed training, alongside data, tensor, context, sequence and expert parallelism. The following figure illustrates the process for a model split into 4 stages:

{: .picture}
<img width="46.4%" height="46.4%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline.svg"/>

{: .caption}
Training of a model split into 4 stages on 4 workers, without pipelining: the forward pass of a mini-batch goes from worker 1 to worker 4, and its backward pass, which takes twice as long, comes back, so only one worker computes at a time. The number in each cell is the mini-batch index. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

The figure also shows the weakness of this simple approach. A stage must wait for the activations of the previous stage before it can start its forward pass, and for the gradients of the next stage before it can start its backward pass. With one input at a time, only one stage computes at any moment, so with $$p$$ stages, each GPU is busy only $$1/p$$ of the time.

The solution is to give the stages different inputs to work on at the same time. GPipe ([GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965), Huang et al., 2019) splits the mini-batch into $$m$$ micro-batches and feeds them to the first stage one after the other. When a stage finishes a micro-batch, it sends the activations to the next stage and starts working on the next micro-batch right away, so that several stages compute at the same time, each on a different micro-batch. The gradients of the $$m$$ micro-batches add up, and the optimizer updates the weights once, at the end of the mini-batch. This is mathematically equivalent to regular gradient accumulation, so the result is the same as if the whole mini-batch had gone through the model at once:

{: .picture}
<img width="63.8%" height="63.8%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline_parallel.svg"/>

{: .caption}
Pipeline parallelism with gradient accumulation (GPipe) on 4 workers: the mini-batch is split into 4 micro-batches, all forward passes run before the backward passes, and the gradients are added at the pipeline flush, before the micro-batches of the next mini-batch (5 and 6) start. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

The GPUs are still idle part of the time: at the start of an iteration, they wait for the pipeline to fill, and at the end, they wait for it to drain. This idle time is called the **pipeline bubble**, and its size follows from the figure. Let $$F$$ and $$B$$ be the times of the forward and the backward pass of one micro-batch on one stage. The forward passes take $$m+p-1$$ steps of length $$F$$: $$m$$ steps for the micro-batches to enter the pipeline, and $$p-1$$ more for the last one to reach the last stage. In the same way, the backward passes take $$m+p-1$$ steps of length $$B$$. An iteration therefore takes $$(m+p-1)(F+B)$$, of which each GPU computes during $$m(F+B)$$, so each GPU idles for $$(p-1)(F+B)$$, a fraction $$(p-1)/m$$ of its compute time. In other words, the bubble becomes small when there are many more micro-batches than stages.

Many micro-batches have a cost in memory, however, because GPipe keeps the activations of all $$m$$ micro-batches until their backward passes. Two techniques reduce this memory. **Activation offloading** moves the activations to the CPU memory after the forward pass, and loads them back when the backward pass needs them. **Activation checkpointing**, which GPipe itself uses, keeps only the input of each stage and recomputes the other activations during the backward pass, i.e. it trades extra computation for memory.

The other choice to make is the number of stages. The examples so far used as many stages as GPUs, i.e. a single pipeline that spans all of them. However, the number of stages can also be smaller than the number of GPUs. Several pipelines then run side by side, each on its own share of the data, which combines pipeline parallelism with data parallelism. The following figure shows the three options on 8 GPUs, for one micro-batch:

{: .picture}
<img width="32.5%" height="32.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_8_stages.svg"/>
&nbsp;
<img width="32.5%" height="32.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_4_stages.svg"/>
&nbsp;
<img width="32.5%" height="32.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_2_stages.svg"/>

{: .caption}
Combinations of pipeline and data parallelism on 8 workers, for one micro-batch: forward passes in blue, backward passes in green, idle time hatched. Left: one pipeline of 8 stages. Center: 2 data-parallel groups of 4 pipeline stages. Right: 4 data-parallel groups of 2 pipeline stages.

With 8 stages, each GPU holds an eighth of the model and computes during an eighth of the iteration. With 4 stages, every pass takes twice as long, each GPU is busy a quarter of the time, and two pipelines process different data. With 2 stages, four pipelines run side by side, and each GPU is busy half of the time. Fewer stages thus mean less idle time, but also a larger part of the model on each GPU, so the number of stages is usually the smallest one with which the model fits in the memory of the GPUs.

Every configuration above still has periods of idleness, and much work has gone into reducing them. [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/) (Narayanan et al., 2019) does not wait for a mini-batch to finish before the next one starts. After a startup phase that fills the pipeline, each stage alternates the forward pass of one mini-batch and the backward pass of an earlier one, a pattern called *one forward, one backward* (1F1B), so that in the steady state no GPU idles:

{: .picture}
<img width="56.3%" height="56.3%" src="{{ site.assets }}/distributed-training/imgs/pipedream_dnn_pipeline_parallel_microsoft.svg"/>

{: .caption}
The PipeDream schedule on 4 workers. In the startup state, worker 1 admits 4 mini-batches; in the steady state, each worker alternates the forward pass of one mini-batch and the backward pass of an earlier one, so several mini-batches are in flight at the same time. Backward passes are prioritized over forward passes on each worker. Adapted from: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

The caveat is that mixing the passes of different mini-batches can lead to wrong weight updates. Each stage updates its weights after every backward pass, while other mini-batches are still in flight, but the backward pass of a mini-batch must use the same weights as its forward pass, or its gradients would not match its activations. PipeDream therefore keeps several versions of the weights, one per mini-batch in flight, a technique called *weight stashing*: every forward pass uses the latest version, and every backward pass uses the version of its forward pass. The first stage has up to $$p$$ mini-batches in flight, so it keeps up to $$p$$ versions of its weights, which increases the memory substantially. The updates are also asynchronous: the first stage computes each gradient with weights that are $$p-1$$ updates old.

## PyTorch Implementation

Before turning to a library, it is useful to write the simplest possible pipeline-parallel training loop in plain PyTorch. The goal is not performance, but to show the mechanics that every implementation shares:
- partitioning a model into `num_stages` stages of consecutive layers;
- sending the activations forward from stage to stage, with `dist.send` and `dist.recv`;
- sending the gradients of the activations backward from stage to stage, in the same way;
- performing one optimizer step per mini-batch, after all its micro-batches.

The code assumes a single pipeline, in which the rank of each process is its stage id, and each stage is an `nn.Sequential` block of layers. It uses the simplest schedule, the one of the first figure: a micro-batch runs forward through all the stages, the last stage computes the loss, then the backward pass runs all the way back, and only then does the next micro-batch start. After the last micro-batch, each stage updates its own parameters, which no other stage holds:

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

This toy code leaves out what a real system needs: the communication of the tensor shapes (the receive buffers above are placeholders with the shape of the input), [mixed precision](https://en.wikipedia.org/wiki/Mixed-precision_arithmetic), activation checkpointing, the composition with data parallelism, load balancing and, most of all, a better schedule, since this one leaves every GPU idle most of the time. The DeepSpeed implementation below provides all of them.

## DeepSpeed Implementation

DeepSpeed combines the two ideas above. It uses the [1F1B schedule of PipeDream](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/), but with a flush at the end of every mini-batch, as in GPipe, a schedule called PipeDream-Flush ([Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2006.09503), Narayanan et al., 2021). Each stage first runs a few forward passes to fill the pipeline: stage $$i$$, counted from 0, runs up to $$p-i$$ of them. It then alternates one backward and one forward pass, so that the backward pass of each micro-batch starts as early as possible, and it ends with its remaining backward passes. Finally, each stage updates its weights once per mini-batch. The updates are therefore synchronous, as in GPipe, and one version of the weights is enough. The two figures below compare GPipe and 1F1B on 4 GPUs with 8 micro-batches:

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/distributed-training/imgs/pipeline_gpipe.svg"/>

{: .caption}
GPipe on 4 GPUs with 8 micro-batches: all forward passes, then all backward passes. Each GPU keeps the activations of the 8 micro-batches until their backward passes. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/distributed-training/imgs/pipeline_1f1b.svg"/>

{: .caption}
1F1B on 4 GPUs with 8 micro-batches: after a warm-up of forward passes, each GPU alternates one forward and one backward pass. The iteration takes as long as with GPipe, but each GPU keeps the activations of at most 4 micro-batches instead of 8. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

The two schedules take the same time, and their bubbles are the same, $$(p-1)(F+B)$$ per GPU. The difference lies in the memory. GPipe runs all the forward passes first, so every GPU keeps the activations of all 8 micro-batches. 1F1B starts the backward pass of each micro-batch as early as possible and frees its activations right after, so stage $$i$$ keeps the activations of at most $$p-i$$ micro-batches instead of $$m$$: 4 on GPU 0, and a single one on GPU 3. With many micro-batches, this is a large saving, and 1F1B is the default schedule of DeepSpeed and Megatron-LM. For training, the DeepSpeed releases implement only this schedule, in the class `TrainSchedule`, and any other schedule must be written as a [custom schedule](https://deepspeed.readthedocs.io/en/latest/pipeline.html#module-deepspeed.runtime.pipe.schedule).

The rest of this section adds pipeline parallelism to the DeepSpeed training of `GPTlite` from the [data parallelism post]({{ site.baseurl }}{% post_url 2023-08-18-data-parallelism %}). Pipelining is enabled by passing the number of stages in the command-line argument `--pipeline_num_stages` (default: 0, no pipelining):

```python
## train.py

def get_cmd_line_args(description='GPT lite on DeepSpeed'):
  # ...
  parser.add_argument('--pipeline_num_stages', type=int, default=0,
                      help='enable pipeline parallelism with N stages (0 means disabled)')
  # ...
```

DeepSpeed can pipeline any sequence of layers, given as a list or an `nn.Sequential` container, which it splits into stages. To expose the layers of `GPTlite`, a new method `to_layers()` returns them in the order in which `forward()` runs them. Since `self.blocks` is an `nn.Sequential`, `*self.blocks` unpacks it into the individual transformer blocks, which DeepSpeed can then place on different stages:

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

The first layer is a lambda that sums the token and the position embeddings. DeepSpeed calls it in the forward pass like any other layer, but since it is not an `nn.Module`, DeepSpeed does not register its parameters, and the optimizer never updates the embedding tables. The module `EmbeddingsSum` of the `LayerSpec` version below avoids this problem.

The last layer raises a second problem. It returns logits of shape (batch, sequence, vocabulary), but [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) expects the classes in the second dimension. A quick fix is to append the layer `lambda logits: torch.swapaxes(logits, 1, 2)` to `to_layers()`, which changes the shape to (batch, vocabulary, sequence). However, the backward pass may then fail with the error `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`, as discussed in the DeepSpeed issues [4274](https://github.com/microsoft/DeepSpeed/issues/4274) and [4479](https://github.com/microsoft/DeepSpeed/issues/4479). Setting `outputs = outputs.requires_grad_(True)` avoids the error. A cleaner fix leaves the model as it is, and flattens the logits and the labels in the loss function instead, which DeepSpeed runs on the last stage:

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

Next, `get_model()` wraps the layers in a `PipelineModule`, which assigns them to the stages. This wrapped model is the `model` passed to `deepspeed.initialize()`:

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

The training loop changes too. With pipelining, the engine runs the whole schedule of a mini-batch by itself, so a training iteration reduces to a call to `engine.train_batch()`, which runs the [forward passes, backward passes and weight update of one mini-batch](https://www.deepspeed.ai/tutorials/pipeline/#training-loops) of `engine.gradient_accumulation_steps()` micro-batches. The number of micro-batches is set in the [DeepSpeed config file](https://www.deepspeed.ai/docs/config-json/), through the fields `train_micro_batch_size_per_gpu` (default: `train_batch_size`) or `gradient_accumulation_steps` (default: 1):

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

One detail needs attention. Each stage learns the shape of the activations it receives from a message that comes with the first micro-batch, and it allocates its receive buffers once, with that shape. By default, the pipeline engine therefore expects all mini-batches, i.e. all calls to `train_batch()`, to have the same shape. When the shape changes from one mini-batch to the next, for example with [sequences of variable length]({{ site.baseurl }}{% post_url 2024-05-10-training-variable-length-curriculum %}), a call to `engine.reset_activation_shape()` at the start of the mini-batch adds a communication step that sends the new shape with its first micro-batch. The other micro-batches of the mini-batch must still have that shape, so they are trimmed or padded to one shape beforehand. Since version 0.15, DeepSpeed also accepts micro-batches of different shapes, with the argument `dynamic_shape=True` of `PipelineModule`, which sends the shape with every micro-batch, at some cost in performance.

Finally, the [pipeline engine of DeepSpeed does not support ZeRO stages 2 and 3](https://deepspeed.readthedocs.io/en/latest/pipeline.html#pipeline-parallelism), as discussed in [DeepSpeed issue 1110](https://github.com/microsoft/DeepSpeed/issues/1110#issuecomment-850835817).

### Memory-Efficient Model Construction with LayerSpec

The implementation above is neither memory efficient nor scalable: every GPU builds the whole model and keeps the parameters of all its layers in memory, although it runs only the layers of its own stage ([Memory-Efficient Model Construction](https://www.deepspeed.ai/tutorials/pipeline/#memory-efficient-model-construction)). The solution is the DeepSpeed class [`LayerSpec`](https://deepspeed.readthedocs.io/en/latest/pipeline.html#deepspeed.pipe.LayerSpec), which delays the construction of each layer until the layers have been assigned to the stages, so that each GPU allocates only the layers of its own stage. A `LayerSpec` takes the class of an `nn.Module` and the arguments of its constructor, so the sum of the embeddings, which is not a module, becomes the module `EmbeddingsSum`. The new class `GPTlitePipeSpec` inherits from `PipelineModule`, and its `__init__` lists the layers in the same order as `GPTlite.forward()`. The loss function is still `CrossEntropyLoss_FlatView`, passed in `pipe_kwargs`:

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

The command-line flag `--pipeline_spec_layers` enables this version:

```python
## train.py

def get_cmd_line_args():
  # ...
  parser.add_argument("--pipeline_spec_layers", action="store_true",
                      help="enable LayerSpecs in pipeline parallelism")
```

and `get_model()` returns it when the flag is set:

```python
## gptlite.py

def get_model(criterion, vocab_size, pipeline_num_stages=0, pipeline_spec_layers=False):

  if pipeline_num_stages:
    if pipeline_spec_layers:
      model = GPTlitePipeSpec(vocab_size, pipe_kwargs=pipe_kwargs)
    else:
      # ... GPTlite model as before 
```

How the layers are distributed across the stages matters too, because the slowest stage sets the pace of the whole pipeline. DeepSpeed can [balance the stages](https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules) by the number of trainable parameters (`partition_method="parameters"`, the default), by the number of layers (`"uniform"`), or by the number of layers whose class name matches a regular expression (`"type:[regex]"`). The default suits models in which the compute time of a layer grows with its number of parameters. The runs below keep the default, so the stages can hold different numbers of layers.

### Activation Checkpointing

The argument `activation_checkpoint_interval` of the `PipelineModule` constructor enables activation checkpointing, with one checkpoint every `activation_checkpoint_interval` layers:

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

At first sight, checkpointing seems hard to combine with pipelining: if a layer had to recompute its activations from a checkpoint stored on another GPU, the recomputation would need extra communication. DeepSpeed avoids this case, because it applies the interval within each stage. The first checkpoint of a stage starts at its first layer, and its input is the activation received from the previous stage, which the stage keeps anyway for the backward pass. The recomputation therefore never needs data from another GPU.

## Results

The runs used 8 GPUs, with the ZeRO stage set to 1 in the config file <a href="{{ site.assets }}/distributed-training/ds_config.json">`ds_config.json`</a>, and the `LayerSpec` version of the model (`--pipeline_num_stages <num_stages> --pipeline_spec_layers`), with 1, 2, 4 or 8 stages. They did not use activation checkpointing, because of the open DeepSpeed issue [4274](https://github.com/microsoft/DeepSpeed/issues/4274). The config sets 256 samples per mini-batch and micro-batches of 1 sample, so with 8 stages, the single pipeline runs $$m = 256$$ micro-batches per iteration, and the bubble of 1F1B is only $$(p-1)/m = 7/256$$, under 3% of the compute time.

As an example of the default partitioning, the following output shows the partition of the `to_layers()` version with the swap lambda, which has 16 layers, on 8 GPUs with 4 stages, i.e. 2 pipelines. Ranks 0, 2, 4 and 6 hold the stages of the first pipeline:
   ```
RANK=0 STAGE=0 LAYERS=4 [0, 4)   STAGE_PARAMS=21256704 (21.257M)
RANK=2 STAGE=1 LAYERS=3 [4, 7)   STAGE_PARAMS=21256704 (21.257M)
RANK=4 STAGE=2 LAYERS=3 [7, 10)  STAGE_PARAMS=21256704 (21.257M)
RANK=6 STAGE=3 LAYERS=6 [10, 16) STAGE_PARAMS=21308160 (21.308M)
   ```

Each transformer block holds 7.09M parameters, and stages 0 to 2 hold 3 blocks each. Stage 0 also holds the embedding lambda, whose parameters count as 0, since they are not registered. Stage 3 holds 6 layers: the last 3 blocks, the final layer norm, the output layer and the swap lambda. It has the most layers, but only slightly more parameters than the other stages, because the partition balances the parameters, not the layers.

**LayerSpec.** With 8 stages, the `LayerSpec` version used about 40% less memory than the `to_layers()` version. This held for GPTlite and for the deep benchmark model of `benchmark.py`, a stack of 2048 linear layers of width 256 with ReLU activations.

**Memory per GPU.** With both models, the first GPU used more memory than the others. Part of this imbalance is expected with 1F1B: stage $$i$$ keeps the activations of up to $$p-i$$ micro-batches, and DeepSpeed allocates $$\max(2, \min(p-i, m))$$ buffers on stage $$i$$ for the inputs and outputs of the micro-batches in flight. The first GPU therefore sets the memory needed per GPU, and the V-shape schedules below balance it. For the deep benchmark model, however, this explains little: its stages hold the same number of parameters, and its activations are small, a few MB per sample for the whole model, so 1F1B accounts for a few MB of difference at most. The cause of the imbalance for this model is not known, and a [bug report](https://github.com/microsoft/DeepSpeed/issues/4477) was opened with DeepSpeed.

The code is in the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training).

## Beyond 1F1B: Reducing the Pipeline Bubble

Even with 1F1B, every stage idles for $$(p-1)(F+B)$$ per iteration, a fraction $$(p-1)/m$$ of its compute time. 1F1B can reduce this bubble only with more micro-batches per iteration, i.e. a larger mini-batch or smaller micro-batches, or with fewer stages, and all three options have limits: the mini-batch size is limited by the convergence of training, small micro-batches use the GPU less efficiently, and fewer stages need more memory per GPU. Newer schedules therefore reduce the bubble directly. They are not part of the DeepSpeed releases, but they are available in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), in [PyTorch's `torch.distributed.pipelining`](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) and in [DeepSeek's DualPipe repository](https://github.com/deepseek-ai/DualPipe).

The diagrams of this section, like the GPipe and 1F1B diagrams above, were generated from the reference implementation of each schedule, for 4 GPUs and 8 micro-batches (6 GPUs and 12 micro-batches for DualPipe). The diagrams on 4 GPUs share the same time scale, and time is normalized so that every GPU performs the same amount of compute in all of them: a forward pass takes 1 unit, and a backward pass takes 2 units, or 1+1 units when it is split into the gradients for the inputs and for the weights, as explained below. Schedules that place two model chunks on each GPU run each chunk in half the time. Communication time is not modeled. As before, the number in each cell is the micro-batch id, and hatched cells are idle time. The pale cells with gray outlines are the forward passes of the next iteration, and the dashed line marks the time at which the 1F1B schedule ends.

### Interleaved 1F1B

[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (Narayanan et al., 2021) assigns several non-contiguous chunks of layers, or *virtual stages*, to each GPU, instead of one block of consecutive layers. With 4 GPUs and 2 chunks per GPU, the model is split into 8 chunks: GPU 0 holds chunks 0 and 4, GPU 1 holds chunks 1 and 5, and so on, so every micro-batch loops through the GPUs twice. Why does this help? The bubble comes from the steps in which the pipeline fills and drains, and each of these steps now takes the time of a smaller chunk. With $$v$$ chunks per GPU, the bubble therefore shrinks to $$(p-1)(F+B)/v$$. The cost is $$v$$ times more point-to-point communication, since every micro-batch crosses the GPUs $$v$$ times, and more activation memory, as the first GPU keeps more micro-batches in flight. The schedule is available in Megatron-LM and as `ScheduleInterleaved1F1B` in PyTorch.

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_interleaved_1f1b.svg"/>

{: .caption}
Interleaved 1F1B with 2 model chunks per GPU. Light and dark cells refer to the first and second chunk on each GPU. Interleaving halves the bubble of 1F1B. Adapted from: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

The figure shows how the micro-batches move in groups of $$p$$: each GPU first runs the forward passes of micro-batches 1 to 4 on its first chunk, and only then turns to its second chunk. This is why the original implementation requires the number of micro-batches to be a multiple of the number of GPUs.

### Zero Bubble (ZB-H1 and ZB-H2)

[Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) (Qi et al., 2024) builds on the observation that the backward pass consists of two independent computations. Take a linear layer $$y = xA$$. The previous stage needs the gradient with respect to the input, $$\partial L / \partial x = (\partial L / \partial y) A^T$$, right away to continue the backward pass, so this gradient is on the critical path. The gradient with respect to the weights, $$\partial L / \partial A = x^T (\partial L / \partial y)$$, is only needed by the optimizer step at the end of the iteration, so it can wait. Zero Bubble therefore splits every backward pass in two: it computes the *backward for inputs* and sends it upstream at once, and it runs the *backward for weights* later, to fill the bubbles. The paper proposes two handcrafted schedules:
- **ZB-H1** keeps the peak activation memory of 1F1B. Since only the backward for inputs is on the critical path, the gradients travel back through the pipeline faster, and the deferred backward-for-weights passes fill the gaps at the end of the iteration. The bubble drops to $$(p-1)(F+B-2W)$$, where $$W$$ is the time of the backward for weights. When the forward, the backward for inputs and the backward for weights take the same time, this is a third of the bubble of 1F1B.
- **ZB-H2** allows up to $$2p-1$$ micro-batches in flight on the first stage, about twice the activation memory of 1F1B. The extra forward passes fill the warm-up bubble, and the backward-for-weights passes are reordered at the end of the iteration. The schedule becomes a parallelogram that interlocks with the previous and the next iterations, and the bubble disappears.

There is a catch, however. The optimizer step usually synchronizes all stages, for example to compute the global norm of the gradients for gradient clipping, or to check for NaN and Inf values in mixed precision training, and this synchronization breaks the parallelogram. The authors replace it with an *optimizer post-validation*: each stage updates its weights right away, and in the rare case where gradient clipping or the NaN/Inf check is triggered, it rolls the update back and redoes it. They also give an algorithm that searches for the best schedule, given the measured time of each pass and of the communication, and a memory limit. In their experiments, Zero Bubble reached up to 23% higher throughput than 1F1B under a similar memory limit, and up to 31% with a relaxed memory limit. The code is available as a [fork of Megatron-LM](https://github.com/sail-sg/zero-bubble-pipeline-parallelism) and as `ScheduleInterleavedZeroBubble` in PyTorch.

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h1.svg"/>

{: .caption}
ZB-H1, with the activation memory of 1F1B. Every backward pass is split into a backward for inputs and a backward for weights, and the latter fill the bubbles. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h2.svg"/>

{: .caption}
ZB-H2, with about twice the activation memory of 1F1B. When the optimizer step does not synchronize the stages, the next iteration starts on the first GPUs while the last GPUs finish the current one; in the same way, the previous iteration fills the idle time at the start. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

In the ZB-H1 figure, GPU 0 waits 3 time units for the gradients of micro-batch 1 instead of 6, because only the backward for inputs travels back, and the backward-for-weights passes then fill the gaps that 1F1B leaves at the end of the iteration. In ZB-H2, the extra forward passes also fill the wait before the first backward pass, so that each GPU idles only until its first forward pass arrives.

### ZB-V

The Zero Bubble paper also introduces **ZB-V**, which places two model chunks on each GPU in a *V shape*: the model is split into $$2p$$ chunks, and GPU $$i$$ holds chunks $$i$$ and $$2p-1-i$$. A micro-batch goes down the GPUs through the first half of the model, and comes back up through the second half. The first and the last chunks, with the embeddings and the loss, are both on GPU 0, and the turn on the last GPU needs no communication, since chunks $$p-1$$ and $$p$$ are on the same GPU. When the forward, the backward for inputs and the backward for weights take the same time, ZB-V has zero bubble, again with the optimizer post-validation. Its peak activation memory equals that of 1F1B, but it is balanced across the GPUs: the GPU that keeps the most micro-batches in flight for its first chunk keeps the fewest for its second one. The cost is twice the point-to-point communication of 1F1B, as every micro-batch crosses the GPUs twice. A follow-up paper, [Pipeline Parallelism with Controllable Memory](https://arxiv.org/abs/2405.15362) (Qi et al., 2024), generalizes the V-shape schedules to a family that trades bubbles for memory, down to half (V-Half) and a third (V-Min) of the activation memory of 1F1B. ZB-V is available as `ScheduleZBVZeroBubble` in PyTorch.

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_v.svg"/>

{: .caption}
ZB-V with 8 model chunks on 4 GPUs. Light cells refer to the first chunk on each GPU (chunks 0 to 3, going down) and dark cells to the second chunk (chunks 4 to 7, going back up). As in ZB-H2, the previous iteration fills the idle time at the start when the optimizer step does not synchronize the stages. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

### DualPipe

The [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024) introduced DualPipe, later open-sourced in the [DualPipe repository](https://github.com/deepseek-ai/DualPipe). DualPipe was designed for [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) models trained with [expert parallelism]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) across nodes. In this setting, the [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) communication that dispatches the tokens to the experts, and combines the outputs of the experts, takes about as long as the computation itself. The key idea of DualPipe is to overlap the forward pass of one micro-batch with the backward pass of another. Both passes are split into four components, attention, all-to-all dispatch, MLP and all-to-all combine, and in the backward pass, attention and MLP are further split into a backward for inputs and a backward for weights, as in Zero Bubble. DualPipe then rearranges these components so that the communication of each pass hides behind the computation of the other, with a manually tuned split of the streaming multiprocessors of the GPU between communication and computation.

To have enough such pairs of forward and backward passes, DualPipe uses a *bidirectional* pipeline: half of the micro-batches enter at the first GPU and flow forward, while the other half enter at the last GPU and flow in the opposite direction. GPU $$i$$ therefore holds stages $$i$$ and $$p-1-i$$, so the pipeline holds two copies of the model parameters, and the gradients of each stage are summed across GPUs $$i$$ and $$p-1-i$$ before the optimizer step. DeepSeek notes that, with their large expert-parallel size, this duplication does not increase the memory significantly. The bubble is $$(p/2-1)(F\&B+B-3W)$$, where $$F\&B$$ is the time of an overlapped pair of forward and backward passes, and each GPU stores the activations of $$p+1$$ micro-batches. [Chimera](https://arxiv.org/abs/2107.06925) (Li and Hoefler, 2021), an earlier bidirectional schedule, requires the number of micro-batches to be a multiple of the number of stages, while DualPipe only requires even numbers of stages and micro-batches, and its bubble and activation memory do not grow with the number of micro-batches.

The diagrams model only the computation, so the bubble of DualPipe looks close to that of ZB-H1, and its main gain does not show in them: within each black-bordered pair, the all-to-all communication of expert parallelism hides behind the computation of the other pass.

{: .picture}
<img width="86.8%" height="86.8%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipe.svg"/>

{: .caption}
DualPipe on 6 GPUs: micro-batches 1 to 6 (light) enter at GPU 0, and micro-batches 7 to 12 (dark) enter at GPU 5. Each GPU holds two pipeline stages. Cells with a shared black border are the forward and backward passes of two different micro-batches executed together, so that the communication of one overlaps with the computation of the other. Adapted from: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

### DualPipeV

Sea AI Lab later observed that DualPipe is made of two mirrored halves, since GPUs $$i$$ and $$p-1-i$$ hold the same stages and run the same schedule ([DualPipe could be better without the Dual](https://huggingface.co/blog/ufotalent/cut-in-half), Sea AI Lab, 2025). Keeping only the first half of the GPUs, and placing the remaining stages on them in a V shape as in ZB-V, gives a schedule with the bubble and the activation memory of DualPipe for the same number of stages, but on half the GPUs and without duplicated parameters. Like ZB-V, it needs twice the point-to-point communication of 1F1B, which the authors consider small next to the all-to-all communication of expert parallelism. DeepSeek adopted this *cut-in-half* schedule as **DualPipeV** in the DualPipe repository. Recent PyTorch versions provide it as `ScheduleDualPipeV`, and the development branch of DeepSpeed as `DualPipeVSchedule`. As in ZB-V, all micro-batches enter at GPU 0, go down the GPUs and come back up, and GPU 0 computes the loss.

{: .picture}
<img width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipev.svg"/>

{: .caption}
DualPipeV with 8 model chunks on 4 GPUs. Light cells refer to the first chunk on each GPU (going down) and dark cells to the second chunk (going back up). Cells with a shared black border are overlapped forward and backward passes of two different micro-batches. Adapted from: [DualPipe](https://github.com/deepseek-ai/DualPipe)

### Summary

The table below compares the schedules on 4 GPUs with 8 micro-batches. Idle time is measured in forward passes, and activation memory in units of the activations of one micro-batch on one 1F1B stage, on the GPU that needs the most. The backward for inputs and the backward for weights are assumed to free half of these activations each.

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

Which schedule to choose depends mostly on the number of micro-batches per pipeline. With many more micro-batches than stages, the bubble of 1F1B, $$(p-1)/m$$, is already small, as in the GPTlite runs above, and 1F1B is the simplest choice. The other schedules pay off when each pipeline gets only a few micro-batches, for example when many data-parallel replicas share a fixed global batch. Interleaved 1F1B and the Zero Bubble schedules then reduce the bubble for more communication or more memory, and ZB-V removes almost all of it with the activation memory of 1F1B, for twice the point-to-point communication. DualPipe and DualPipeV pay off for Mixture-of-Experts models with expert parallelism across nodes, where hiding the all-to-all communication matters as much as the bubble.

