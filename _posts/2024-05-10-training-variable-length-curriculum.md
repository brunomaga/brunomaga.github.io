---
layout: post
title:  "Variable sequence lengths, curriculum learning, adaptive batch size and LR"
categories: ["Distributed Training"]
tags: ["curriculum learning", "variable sequence length", "adaptive batch size", "CUDA graphs", "DeepSpeed"]
---

Many datasets hold samples of variable length: audio tracks of different durations, sentences with different numbers of tokens, or videos with different numbers of frames. A common way to train a model on such data is to trim and pad all samples to one fixed length, so that every batch has the same shape. Training on the original lengths avoids the compute spent on padding, but adds complexity, especially in distributed (multi-node, multi-GPU) runs. Three techniques make such runs faster:

1. **Curriculum learning** presents the samples in order of increasing difficulty, for example by length, which makes training faster and more stable.
2. An **adaptive batch size and learning rate** pack many short samples or a few long ones into each batch, within a token budget, and scale the learning rate (LR) to the size of each batch.
3. **Static kernel compilation** compiles one version of the model per input shape, which reduces the runtime overhead.

## Curriculum Learning

[Curriculum learning](https://arxiv.org/abs/2101.10382) presents the samples to the model in order of increasing difficulty, such as increasing noise level, human score or length, much as students learn the easy material before the hard one. The idea goes back to [Bengio et al. (2009)](https://doi.org/10.1145/1553374.1553380), and it can improve the stability of training and the quality of the final model: very difficult samples early in training can produce large gradients and abrupt parameter updates, which may destabilize learning, while a gradual increase in difficulty leads to a smoother learning process.

On variable-length data, a curriculum by length also improves efficiency. A batch that mixes short and long sequences pads all of them to the length of the longest one, which adds substantial compute and memory overhead, while a batch of sorted samples holds sequences of similar lengths.

Curriculum learning has three steps: define a difficulty metric per sample, sort the samples by increasing difficulty, and iterate over the sorted dataset. In distributed runs with large datasets, however, no rank holds the whole dataset, and the sort becomes the hard step. The samples are typically assigned to the ranks in an interleaved order, for example by PyTorch's `DistributedSampler` (panels (a) and (b) below):

{: .picture}
<img width="80.4%" height="80.4%" src="{{ site.assets }}/distributed-training/imgs/curriculum_datasets.svg"/>

{: .caption}
The curriculum dataset setup problem, on a network of 4 ranks and a dataset of 16 samples. (a) Original dataset, with an interleaved assignment of the 16 samples (bars) of variable lengths (bar heights) to the 4 ranks (color-coded). (b) Dataset in (a) after distributed data loading. (c) Dataset in (b) after local sorting. (d) Dataset in (b) after distributed sorting. (e) Dataset in (d) with an interleaved assignment of the 16 samples to the 4 ranks (color-coded). (f) Dataset in (e) after distributed data loading.

There are two ways to sort such a dataset:

- **Local sort** (panel (c)): each rank sorts its own shard, with no communication. At a given step, however, the ranks process samples of different lengths, so the ranks with short samples wait for the ranks with long samples at every synchronization. The order of the samples, and thus the trained model, also depends on the number of ranks.
- **Distributed sort** (panel (d)), followed by an interleaved assignment of the sorted dataset to the ranks (panels (e) and (f)). It costs one distributed sort before training, but at every step, all ranks process samples of similar lengths, so the load is almost perfectly balanced, and the order of the samples does not depend on the number of ranks.

The rest of this section implements the second approach.

### Distributed Sorting

The distributed sort turns panel (b) into panel (d). The implementation uses [sample sort](https://en.wikipedia.org/wiki/Samplesort), described in [Distributed Sorting Algorithms]({{ site.baseurl }}{% post_url 2014-06-21-distributed-sort %}), because it scales well to many ranks: each element moves at most once, in a single [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) exchange. From a small sample of the data, sample sort picks *splitters*, key values that divide the keys into one interval per rank, and then sends every element to the rank of its interval:

{: .picture}
<img width="57%" height="57%" src="{{ site.assets }}/distributed-sort/sample_sort.svg"> 

{: .caption}
Sample sort of 21 elements on $$p = 3$$ ranks, colored by rank. Each rank sends $$p-1 = 2$$ samples to rank 0, which sorts them and picks every $$p$$-th one as a splitter (yellow): 4 and 10. The last interval is open: rank 2 receives every element from 10 upwards.

With only $$p-1 = 2$$ samples per rank for $$p = 3$$ ranks, the splitters of the figure are a rough estimate, and the ranks end with 9, 4 and 8 elements. The code below takes `n_samples=100` samples per rank, which gives splitters closer to the quantiles of the data, and buckets of closer sizes. It also replaces the gather and the broadcast of the figure with one all-gather, after which every rank picks the same splitters. The first column of the tensor is the key, and since no rank knows in advance how much data it will receive, the exchange takes two all-to-all calls, one for the counts and one for the elements:

```python
def sample_sort(tensor, comm_group, num_workers, n_samples=100):
    # Perform a distributed sample sort of a tensor and return this rank's sorted shard.
    device, dims = tensor.device, tensor.size()[1]

    # 1 - sort rows lexicographically (by col0, then col1, ...)
    tensor = torch.tensor(sorted(tensor.tolist()), dtype=tensor.dtype, device=tensor.device)

    # 2 - collect a few samples per rank (use the first column as key)
    idx = torch.round(torch.linspace(0, len(tensor) - 1, n_samples)).to(int)
    samples = tensor[idx][:, 0].contiguous().to(device)

    # 3 - all-gather samples
    all_samples = [torch.zeros(n_samples, dtype=samples.dtype, device=device) for _ in range(num_workers)]
    dist.all_gather(all_samples, samples, group=comm_group)
    all_samples = torch.cat(all_samples, dim=0).to(device)

    # 4 - sort all samples and choose range boundaries for each rank
    all_samples = all_samples.sort()[0]
    idx = torch.round(torch.linspace(0, len(all_samples) - 1, num_workers + 1)).to(int)
    ranges = all_samples[idx]  # rank r owns: ranges[r] <= x < ranges[r+1]
    ranges[-1] += 1  # ensure the last bucket includes the max value

    # 5 - bucket local rows by their key range
    send = []
    for rank in range(num_workers):
        mask = (tensor[:, 0] >= ranges[rank]) & (tensor[:, 0] < ranges[rank + 1])
        send.append(tensor[mask])

    # 6 - all-to-all to communicate send/recv sizes (in number of scalars)
    send_count = [torch.tensor([len(s) * dims], dtype=torch.int64, device=device) for s in send]
    recv_count = list(torch.empty([num_workers], dtype=torch.int64, device=device).chunk(num_workers))
    dist.all_to_all(recv_count, send_count, group=comm_group)

    # 7 - all-to-all-v to exchange the flattened payloads
    send = torch.cat(send, dim=0).flatten().to(device)
    recv = torch.zeros(sum(recv_count), dtype=send.dtype).to(device)
    send_count = [s.item() for s in send_count]
    recv_count = [r.item() for r in recv_count]
    dist.all_to_all_single(recv, send, recv_count, send_count, group=comm_group)
    del send

    # 8 - reshape and sort locally again
    recv = recv.view(-1, dims)
    recv = torch.tensor(sorted(recv.tolist()), dtype=recv.dtype, device=recv.device)
    return recv
```

[DeepSpeed PR 5129](https://github.com/microsoft/DeepSpeed/pull/5129) holds the rest of the code, such as `file_write_ordered`, which writes the sorted distributed tensor of panel (d) to one sequential file.

## Adaptive Batch Size and Learning Rate

How many samples should a batch hold when their lengths differ? A fixed number of samples gives batches with very different numbers of tokens, and thus different costs in compute and memory. On variable-length datasets, batches are therefore often built by *token count* instead: the samples are grouped so that their lengths add up to about a target number of tokens. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (section 5.1) does this:

> Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

A token budget balances the work across batches and across ranks, and it fits curriculum learning well. A batch has the shape $$B \times S \times E$$ (batch size × sequence length × embedding size). Early in the curriculum, a batch holds many short samples, with a large $$B$$ and a small $$S$$, and later, a few long samples, with a small $$B$$ and a large $$S$$.

A change of the batch size, however, usually requires a change of the learning rate, because the batch size sets the noise of the gradient estimate. When the batch grows from a reference size $$B_0$$ to $$B = k B_0$$, two common rules scale the reference LR $$\eta_0$$:

1. **Linear scaling**, $$\eta = k \, \eta_0$$. One step on a batch of $$k B_0$$ samples then moves the parameters about as far as $$k$$ steps on batches of $$B_0$$ samples, provided the gradients change little over these $$k$$ steps. [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) (Goyal et al., 2017) uses this rule, with a warmup phase, to train ResNet-50 with [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) on batches of up to 8192 images. Beyond that size, the accuracy drops.
2. **Square-root scaling**, $$\eta = \sqrt{k} \, \eta_0$$. The gradient $$g$$ of a batch is an average over its samples, so its variance falls by a factor $$k$$ when the batch grows by $$k$$, and the variance of the update $$\eta g$$, proportional to $$\eta^2 / B$$, stays constant under this rule. [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997) (Krizhevsky, 2014) states this rule, but finds linear scaling to work better for the batch sizes of that paper. For Adam and RMSprop, [Malladi et al. (2022)](https://arxiv.org/abs/2205.10287) derive square-root scaling, which also changes the hyperparameters $$\beta$$ and $$\epsilon$$.

In other words, linear scaling suits SGD, and square-root scaling suits adaptive optimizers such as Adam. The method thus has three hyperparameters: the token budget per batch, the reference LR $$\eta_0$$ and the reference batch size $$B_0$$.

### Illustrative Example

Consider a budget of $$30$$ tokens per batch, $$\eta_0 = 10^{-3}$$ and $$B_0 = 2$$. Samples are added to a batch as long as it stays within the budget, so early in the curriculum, when the sequences are short, a batch holds more samples than later:

{: .picture}
<img width="74.6%" height="74.6%" src="{{ site.assets }}/distributed-training/imgs/variable_batch_lr.svg"/>

{: .caption}
Batches of at most 30 tokens, of shape BxSxE (batch size x sequence length x embedding size), with a reference batch size of 2 and a reference learning rate (LR) of 1e-3. Short sentences (left) give a larger batch, and thus a larger LR, than long sentences (right). Grey cells are padding tokens.

Both batches hold 30 tokens, but the left one holds $$B = 10$$ short samples, and the right one only $$B = 4$$ long ones. Linear scaling, with $$k = 5$$ and $$k = 2$$, gives the LRs of the figure, $$5 \times 10^{-3}$$ and $$2 \times 10^{-3}$$, while square-root scaling would give $$2.24 \times 10^{-3}$$ and $$1.41 \times 10^{-3}$$.

### Pipeline Parallelism

[Pipeline parallelism]({{ site.baseurl }}{% post_url 2023-08-30-pipeline-parallelism %}) adds a constraint. Each stage of the DeepSpeed pipeline engine learns the shape of the activations it receives from the first micro-batch, and allocates its receive buffers once. A call to `engine.reset_activation_shape()` lets the shape change between batches, as a curriculum needs, but all micro-batches of a batch must have the same batch size and sequence length. The figure below compares [Distributed Data Parallel]({{ site.baseurl }}{% post_url 2023-08-18-data-parallelism %}) (DDP, left) with pipeline parallelism (right), for a batch of 4 micro-batches:

{: .picture}
<img width="94.9%" height="94.9%" src="{{ site.assets }}/distributed-training/imgs/variable_batch_lr_pipeline.svg"/>

With DDP, each micro-batch has its own shape and holds up to 30 tokens. With pipeline parallelism, all micro-batches have the same shape, $$3 \times 11 \times E$$, where 11 is the length of the longest sample. The 4 micro-batches then hold only 12 samples instead of 16, and 33% of their tokens are padding instead of 23%. Since version 0.15, the option `PipelineModule(dynamic_shape=True)` of DeepSpeed lifts this constraint, at some cost in performance, by sending the shape with every micro-batch.

### Attention Matrix

Attention needs particular care in variable-length training. With samples of one length, an input of shape $$B \times S \times E$$ uses one $$S \times S$$ mask, such as the causal mask, broadcast over the batch dimension. With padded samples of different lengths, each sample needs its own mask, which also hides its padding, so the mask becomes a $$B \times S \times S$$ tensor. For example, micro-batch 1 of the DDP panel above holds 4 samples of lengths 9, 7, 8 and 6, and its mask has one slice per sample (samples 0 to 3):

{: .picture}
<img width="98.9%" height="98.9%" src="{{ site.assets }}/distributed-training/imgs/variable_attn_matrix.svg"/>

{: .caption}
3D attention mask (BxSxS) of the non-pipeline micro-batch 1 above: its 4 samples of lengths 9, 7, 8 and 6 give a mask of size 4x9x9, with one causal mask per sample, padded to the length of the longest sample.

The cost of attention grows linearly with $$B$$ and quadratically with $$S$$, the length of the longest sequence of the batch. Letting $$S$$ vary from batch to batch, and adapting $$B$$ to it, therefore avoids padding the sequences to an unnecessarily large $$S$$. Variable-length attention kernels go one step further and remove the padding altogether: `flash_attn_varlen_func` of [FlashAttention](https://github.com/Dao-AILab/flash-attention), for example, takes the sequences of a batch concatenated along one dimension, with the offset of each sequence, and needs no mask tensor.

### Implementation

A scheduler wrapper implements the variable LR. At every step (an "epoch" in the code, as in the PyTorch scheduler interface), it restores the unscaled LR, advances the underlying scheduler, and scales the new LR to the size of the next batch:

```python
class VariableBatchSizeLR(LRScheduler):
    # An LR scheduler wrapper that scales LR based on the realized batch size.

    @property
    def optimizer(self):
        return self.base_lr_scheduler.optimizer

    def __init__(self, lr_scheduler, base_batch_size, batch_sizes, dataloader,
                 lr_scaling_method="linear", last_epoch=-1, verbose=False):
        self.batch_sizes = batch_sizes
        self.base_batch_size = base_batch_size
        self.lr_scaling_method = lr_scaling_method
        self.dataloader = dataloader
        self.base_lr_scheduler = lr_scheduler
        self.base_lrs = self.base_lr_scheduler.get_lr()
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.step(0)

    # [...]

    def step(self, epoch=None):
        # call the base scheduler's step method to get LR for next epoch
        # Note: optimizer.step() precedes lr_scheduler.step(), so:
        # init: lr_scheduler.step(0) --> set LR for epoch 0
        # epoch 0: optimizer.step(); lr_scheduler.step(1) --> set LR for epoch 1
        # epoch 1: optimizer.step(); lr_scheduler.step(2) --> set LR for epoch 2

        # reset unscaled LRs (to the base scheduler's values) for the current epoch
        # Note: epoch==0: reset LR scheduler; epoch==None: scale LR for next epoch;
        unscaled_lrs = self.base_lrs if epoch == 0 else self.get_last_lr()
        for group, lr in zip(self.base_lr_scheduler.optimizer.param_groups, unscaled_lrs):
            group['lr'] = lr

        self.base_lr_scheduler.step(epoch)

        # scale the learning rate for the next epoch for each parameter group
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        batch_size = self.batch_sizes[self.last_epoch % len(self.batch_sizes)]
        for group in self.base_lr_scheduler.optimizer.param_groups:
            group['lr'] = scale_lr(self.base_batch_size, batch_size, group['lr'], self.lr_scaling_method)
```

The restore step is needed because many schedulers compute the next LR from the current one, for example by multiplying it by a decay factor, so a scaled LR left in the optimizer would be scaled again at every step. The function `scale_lr` (not shown) multiplies the LR by $$k = B / B_0$$ for linear scaling, or by $$\sqrt{k}$$ for square-root scaling.

A data loader returns the micro-batches. The samples are packed beforehand within the token budget, and the input `microbatch_ids` holds the batch id and the sample ids of each micro-batch. A `DistributedSampler` assigns the micro-batches to the ranks in an interleaved order, and the collate function loads their samples. For pipeline parallelism (`required_microbatches_of_same_seqlen`), it also pads the samples to the longest sequence of their batch (`batch_max_seqlens`), so that all micro-batches of the batch have the same sequence length:

```python
def dataloader_for_variable_batch_size(dataset, microbatch_ids, batch_max_seqlens,
    dataloader_rank=0, dataloader_batch_size=1, dataloader_num_replicas=1,
    dataloader_collate_fn=None, dataloader_num_workers=2, dataloader_pin_memory=False,
    required_microbatches_of_same_seqlen=False, sample_padding_fn=None,
):

    # Equidistantly distribute the microbatches across replicas (interleaved).
    sampler = DistributedSampler(
        dataset=microbatch_ids,
        num_replicas=dataloader_num_replicas,
        rank=dataloader_rank,
        shuffle=False,
        drop_last=False,
    )

    # Wrap the user-defined collate function to handle variable-batch packing.
    def collate_fn_wrapper(list_microbatch_ids):
        # Each batch element is a list of sample ids that fills up to max tokens.
        # We return the collated batch over all dataset samples referenced in the microbatches.
        batch = []
        for batch_id, microbatch_ids in list_microbatch_ids:
            batch_data = [dataset[idx] for idx in microbatch_ids]
            if required_microbatches_of_same_seqlen:
                assert sample_padding_fn is not None,                     "sample_padding_fn must be provided if required_microbatches_of_same_seqlen is True"
                pad_len = batch_max_seqlens[batch_id]
                batch_data = [sample_padding_fn(sample, pad_len) for sample in batch_data]
            batch += batch_data
        return dataloader_collate_fn(batch) if dataloader_collate_fn else batch

    return DataLoader(
        dataset=microbatch_ids,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        num_workers=dataloader_num_workers,
        collate_fn=collate_fn_wrapper,
        pin_memory=dataloader_pin_memory,
    )
```

[DeepSpeed PR 7104](https://github.com/microsoft/DeepSpeed/pull/7104/) holds the complete code and an example.

## Kernel Compilation

[`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) compiles a model [just in time](https://en.wikipedia.org/wiki/Just-in-time_compilation), which can give substantial speedups: it captures the operations of the model in graphs, and its default backend, Inductor, fuses the operations of each graph into fewer kernels. Two aspects of compilation matter here: its interaction with the communication of distributed runs, and its handling of variable shapes. The tests of this section used `torch==2.4.0`.

### Compilation in Single- and Multi-Process Runs

The `mode` argument of `torch.compile` sets how aggressive the optimizations are. The modes `reduce-overhead` and `max-autotune` also use [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/). Every kernel launch costs CPU time, which adds up over many small kernels, and a CUDA graph removes most of this overhead: it records a sequence of kernels once, and then replays it with a single launch.

Consider a feed-forward network of 3 linear layers and 3 ReLU activations on one GPU. Without compilation, each layer runs its own kernels in the forward and in the backward pass:

{: .picture}
<img width="59.6%" height="59.6%" src="{{ site.assets }}/distributed-training/imgs/dnn_serial.svg"/> 

{: .caption}
Workflow of a single-GPU training iteration. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

`torch.compile` captures the forward pass in one graph and derives one graph for the backward pass. Each graph then runs as a few fused kernels, or as a single CUDA graph launch, which often improves the throughput:

{: .picture}
<img width="59.4%" height="59.4%" src="{{ site.assets }}/distributed-training/imgs/dnn_serial_compiled.svg"/>

{: .caption}
Workflow of a single-GPU training iteration with a compiled model. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

Now consider a DDP run on 2 GPUs (GPU 1 and GPU 2 in the figures). Each GPU computes the gradients of its own data, and an all-reduce averages them across the GPUs before the optimizer step. This communication can overlap with the backward pass, because `loss.backward()` only accumulates the gradients, and `optimizer.step()` applies them later. DDP groups the gradients in buckets, and starts the all-reduce of a bucket as soon as all its gradients are computed. The backward pass computes the gradients of the last layers first, so their all-reduce runs while the earlier layers are still being processed:

{: .picture}
<img width="61.2%" height="61.2%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc.svg"/>

{: .caption}
DDP execution workflow on 2 GPUs. The optimizer waits for all asynchronous `all_reduce`s of gradients to finish. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

Compilation breaks this overlap. When `torch.compile` compiles the whole model, the backward pass is one graph, and all its gradients become ready only when it ends. The all-reduces then run after the backward pass, and the optimizer waits longer:

{: .picture}
<img width="64.4%" height="64.4%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc_compiled.svg"/>

{: .caption}
DDP workflow on 2 GPUs with compiled models. The `all_reduce` runs only at the end, which increases the waiting time of the optimizer. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

The `DDPOptimizer` of TorchDynamo ([TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)) solves this problem by breaking the graph on purpose. It splits the graph into subgraphs whose parameters fill about one DDP bucket each, and compiles each subgraph on its own. The all-reduce of a bucket then starts when the backward pass of its subgraph ends, and overlaps with the backward pass of the next subgraph:

{: .picture}
<img width="61.2%" height="61.2%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc_optimized.svg"/>

{: .caption}
DDP workflow on 2 GPUs with compiled subgraphs and `DDPOptimizer`. Synchronization runs asynchronously between subgraphs. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

In the author's tests, a compiled model under DeepSpeed ZeRO stage 0, i.e. plain data parallelism, did not reach the speedup of PyTorch DDP, possibly because DeepSpeed had no equivalent of `DDPOptimizer`.

### Static Compilation of Variable-Shaped Inputs

How does `torch.compile` handle inputs whose shape changes from one batch to the next? Its `dynamic` argument has three settings. With `dynamic=False`, it compiles one static version of the model per input shape. With `dynamic=True`, it compiles kernels with symbolic shapes, which serve many input shapes. By default (`dynamic=None`), it compiles the first shape statically, and recompiles with symbolic shapes when a new shape appears. In the tests, however, the dynamic kernels were slower than the static ones, and dynamic compilation was not reliable when both the batch size and the sequence length varied.

Static compilation is therefore preferable, but the timing of its compilations matters: a rank that meets a new shape late in training recompiles, while the other ranks wait for it at the next collective. To avoid this, each rank runs, early in training, one forward and backward pass for every shape it will see, so that all its versions are compiled at the start (panel (b) below):

{: .picture}
<img width="61.2%" height="61.2%" src="{{ site.assets }}/distributed-training/imgs/torch_compile_dataset.svg"/>

{: .caption}
Dataset setup that allows static compilation for variable-length training on 3 processes. **Top (a):** original dataset of 31 samples (bars) of 4 different lengths (bar heights), with the default interleaved assignment to the processes (color-coded). The green and yellow ranks encounter a new shape later in training (red cross). This triggers a recompilation and, depending on the settings, possibly a runtime failure. **Bottom (b):** after reshuffling, each rank sees one sample of each shape (shapes 1 to 4) early on. One binary is compiled per shape, and no recompilation happens afterwards.

The environment variable `TORCH_LOGS=recompiles` logs each recompilation. For a dataset of lengths $$10$$, $$20$$, $$30$$ and $$40$$, each new length adds an entry such as:

```
Recompiling function forward in train.py:145 triggered by the following guard failure(s):
    - tensor 'L['timestep']' size mismatch at index 0. expected 10, actual 20
```

Each compiled version takes memory, so many shapes can cause out-of-memory (OOM) errors. TorchDynamo also compiles at most `torch._dynamo.config.cache_size_limit` versions of a function (8 by default) and runs every further shape without compilation, so more shapes need a higher limit. Two mitigations reduce the number of versions in memory:

1. **Pad the lengths to a few buckets**, which reduces the number of distinct shapes.
2. **Order the inputs by size** and call `torch.compiler.reset()` after a set of compilations, which frees the versions that are no longer needed. A curriculum by length already orders the inputs by size.

Finally, the model is compiled with static shapes, and with CUDA graphs only on a single GPU:

```python
world_size = torch.distributed.get_world_size()
mode = "max-autotune" if world_size == 1 else "max-autotune-no-cudagraphs"  # single vs multi-GPU
dynamic = False  # force static compilation
model = torch.compile(model, backend="inductor", mode=mode, dynamic=dynamic)
```

[`torch.compile`, the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/) covers compilation problems and their fixes in more detail.
