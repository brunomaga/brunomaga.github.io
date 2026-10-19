---
layout: post
title:  "Variable sequence lengths, curriculum learning, adaptive batch size and LR"
categories: ["Distributed Training"]
tags: ["curriculum learning", "variable sequence length", "adaptive batch size", "CUDA graphs", "DeepSpeed"]
---

Many datasets include samples of variable length: for example, audio tracks of different durations, text sentences with different numbers of tokens, and videos with different numbers of frames. To train a machine learning model on such data, it is common to trim and pad all samples to a fixed length so that batch shapes are consistent across training iterations. Alternatively, the model can be trained on the original sample sizes, which adds complexity, especially in distributed (multi-node, multi-GPU) environments.

This post introduces and implements three techniques that accelerate training with variable-length inputs in multi-process runs:

1. **Curriculum learning**, to make the model learn faster and more stably.
2. **Adaptive batch size and learning rate**, to better utilize the hardware by allowing large batches of short samples (and vice versa), with a corresponding learning-rate adjustment.
3. **Static kernel compilation**, to reduce runtime overhead.

## Curriculum Learning

[Curriculum learning](https://arxiv.org/abs/2101.10382) ([Bengio et al., 2009](https://doi.org/10.1145/1553374.1553380)) is a training method that presents samples to the model in order of increasing difficulty (e.g., increasing noise level, increasing human score, or increasing length). This can improve stability and final performance. The rationale is that presenting very difficult samples early in training can produce large gradients and abrupt parameter updates, which may destabilize learning. Presenting samples in order of increasing difficulty tends to yield a smoother and more stable learning process.

Curriculum learning can also improve efficiency for variable-length data. Mixing short and long sequences in the same batch forces padding up to the longest sequence, which adds substantial compute and memory overhead. A curriculum ordered by length instead puts samples of similar lengths in the same batch, with little padding.

At a high level, curriculum learning is simple: (1) define a difficulty metric per sample, (2) sort samples by increasing difficulty, and (3) iterate through the sorted dataset. In distributed runs with large datasets, however, this becomes non-trivial. Samples are typically assigned to ranks in an interleaved fashion (e.g., via PyTorch’s `DistributedSampler`; see (a) and (b) below), which can lead to [load imbalance](https://en.wikipedia.org/wiki/Load_balancing_(computing)) if each rank sorts only its local shard. The following picture illustrates this problem and the two approaches detailed next:

{: .picture}
<img width="80.6%" height="80.6%" src="{{ site.assets }}/distributed-training/imgs/curriculum_datasets.svg"/>

{: .caption}
The curriculum dataset setup problem, on a network of 4 ranks and a dataset of 16 samples. (a) Original dataset, with an interleaved assignment of the 16 samples (bars) of variable lengths (bar heights) to the 4 ranks (color-coded). (b) Dataset in (a) after distributed data loading. (c) Dataset in (b) after local sorting. (d) Dataset in (b) after distributed sorting. (e) Dataset in (d) with an interleaved assignment of the 16 samples to the 4 ranks (color-coded). (f) Dataset in (e) after distributed data loading.

There are two main approaches:

- **Simple (but potentially imbalanced and non-deterministic):** each rank loads its shard, sorts locally, and applies curriculum learning on its local order (diagram (c)). This can cause load/runtime imbalance and makes results dependent on the number of ranks.
- **Complex (but balanced and deterministic):** all ranks perform a distributed sort (diagram (d)), then reassign the globally sorted dataset in an interleaved fashion (diagram (e)). This yields an almost perfectly balanced distribution across ranks (diagram (f)).

In practice, the simple approach needs no communication, while the complex one needs a distributed sort of the whole dataset, which runs once, before training. The next sections detail the latter option.

### Distributed Sorting

The tricky part is the distributed sort that transforms (b) into (d). There are other distributed sorting algorithms one could use, but here the **Distributed [Sample Sort](https://en.wikipedia.org/wiki/Samplesort)** algorithm is implemented because it scales well to many processes (each element is sent only once, in a single [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) exchange). The workflow is:

{: .picture}
<img width="57%" height="57%" src="{{ site.assets }}/distributed-sort/sample_sort.svg"> 

{: .caption}
Sample sort of 21 elements on $$p = 3$$ ranks, colored by rank. Each rank sends $$p-1 = 2$$ samples to rank 0, which sorts them and picks every $$p$$-th one as a splitter (yellow): 4 and 10. The last interval is open: rank 2 receives every element from 10 upwards.

A Python implementation is provided below. Unlike the picture, it takes 100 samples per rank by default, and uses an all-gather instead of a gather and a broadcast, so that every rank computes the same range boundaries:

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

The remaining code can be found in the author's [DeepSpeed PR 5129](https://github.com/microsoft/DeepSpeed/pull/5129), which includes support utilities (e.g., writing the post-sorting distributed tensor in (d) to a sequential file via `file_write_ordered`).

To find more about curriculum learning, the [paper of Li et al.](https://arxiv.org/abs/2108.06084) studies a curriculum on the sequence length (*sequence length warmup*) that trains GPT models stably with larger batch sizes and learning rates. Also, [DeepSpeed Data Efficiency](https://arxiv.org/abs/2212.03597) provides a general curriculum learning library, used for the pre-training of GPT and BERT models.

## Adaptive batch size and learning rate

When training on variable-length datasets, it is common to batch by *token count* instead of by sample count, i.e., to group samples so that the sum of sequence lengths in the batch stays near a target. For example, in [Attention is all you need](https://arxiv.org/abs/1706.03762), section 5.1:

> Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

This helps equalize work across batches (and across ranks). It is also particularly relevant for curriculum learning. A batch with shape `B x S x E` (batch size × sequence length × embedding size) will often have high `B` and low `S` early in the curriculum (many short samples), and low `B` and high `S` later (fewer long samples).

However, varying the batch size `B` typically requires adjusting the [learning rate](https://en.wikipedia.org/wiki/Learning_rate) (LR). Two widely used scaling rules are:

1. **Linear scaling:** when the minibatch size is multiplied by `k`, multiply the LR by `k`, so that one step on the large batch gives about the same update as `k` steps on the small ones (e.g., [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Goyal et al.](https://arxiv.org/abs/1706.02677)).
2. **Square-root scaling:** when the minibatch size is multiplied by `k`, multiply the LR by `sqrt(k)` to keep gradient noise roughly constant (e.g., [One weird trick for parallelizing convolutional neural networks, Krizhevsky](https://arxiv.org/abs/1404.5997), although that paper finds linear scaling to work better with [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)). It is also the rule that [Malladi et al.](https://arxiv.org/abs/2205.10287) derive for adaptive optimizers such as [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam).

In brief, the works above favor linear scaling for SGD, and square-root scaling for adaptive optimizers such as Adam. In practice, the hyperparameters are: (1) a target token count per batch, (2) a reference learning rate, and (3) a reference batch size. During training, each batch is packed until it reaches the token budget, and the LR is scaled based on the realized batch size.

### Illustrative example

Assume a limit of $$30$$ tokens per batch, a reference LR of $$10^{-3}$$, and a reference batch size of $$2$$. Early in curriculum learning (short sequences), batches will contain more samples than later (long sequences), as shown below:

{: .picture}
<img width="74.6%" height="74.6%" src="{{ site.assets }}/distributed-training/imgs/variable_batch_lr.svg"/>

{: .caption}
Batches of at most 30 tokens, of shape BxSxE (batch size x sequence length x embedding size), with a reference batch size of 2 and a reference learning rate (LR) of 1e-3. Short sentences (left) give a larger batch, and thus a larger LR, than long sentences (right). Grey cells are padding tokens.

Here, samples are collected until the batch reaches (at most) 30 tokens. The batch sizes (number of samples) become $$10$$ and $$4$$ in the left and right examples, respectively. Using linear scaling, the corresponding LRs are $$5 \times 10^{-3}$$ and $$2 \times 10^{-3}$$.

### Pipeline parallelism

Pipeline parallelism requires the same batch size and sequence length across all micro-batches within a batch, because activation shapes must stay fixed during gradient accumulation. Enforcing a consistent `B x S x E` across micro-batches can lead to smaller micro-batches and additional padding. Since version 0.15, DeepSpeed lifts this constraint with the option `PipelineModule(dynamic_shape=True)`, at a possible cost in performance. The figure below contrasts standard Distributed Data Parallel (DDP, left) with pipeline parallelism (right) for 2 processes and 2 gradient accumulation steps (4 micro-batches total):

{: .picture}
<img width="94.9%" height="94.9%" src="{{ site.assets }}/distributed-training/imgs/variable_batch_lr_pipeline.svg"/>

In the pipeline case (right), all micro-batches in the batch have the same `B x S x E` shape. To satisfy this constraint, fewer samples are packed per micro-batch compared to the non-pipeline case (left), and padding is used when needed.

### Attention matrix

[Attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) is a particular challenge in variable-length training. With fixed shapes, an input of shape `B x S x E` often uses a 2D attention mask of shape `S x S`, broadcast over the batch dimension. With variable lengths, a *per-sample* mask is needed, which is naturally represented as a `B x S x S` tensor. For example, for the non-pipeline micro-batch 1 above (top-left; 4 sentences), the mask can be illustrated as:

{: .picture}
<img width="99%" height="99%" src="{{ site.assets }}/distributed-training/imgs/variable_attn_matrix.svg"/>

{: .caption}
3D attention mask (BxSxS) of the non-pipeline micro-batch 1 above: its 4 samples of lengths 9, 7, 8 and 6 give a mask of size 4x9x9, with one causal mask per sample, padded to the length of the longest sample.

A useful way to think about the cost is that attention scales linearly with `B` and quadratically with the maximum `S` in the batch. Allowing `S` to vary (and adapting `B` accordingly) can therefore improve utilization by avoiding padding to an unnecessarily large `S`.

### Implementation

First, the variable learning rate is implemented as a scheduler wrapper that scales the LR of an underlying scheduler at each step based on the realized batch size:

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

Then, the adaptive batch size is implemented as a data loader with a *special* collate function that packs samples into token-budgeted micro-batches, whose sample ids (`microbatch_ids`) are computed beforehand. When the micro-batches of a batch must have the same sequence length, as in pipeline parallelism, it also pads the samples to the longest sequence of their batch (`batch_max_seqlens`):

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

A complete example can be found in [DeepSpeed PR 7104](https://github.com/microsoft/DeepSpeed/pull/7104/).

## Kernels compilation

[Just-in-time compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation) of ML kernels via `torch.compile` can lead to substantial speedups. This section focuses on two aspects: (1) compilation on distributed runs and CUDA graphs, and (2) static vs dynamic compilation.

### CUDA graphs compilation on single- vs multi-process runs

Compilation via [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) uses a `mode` that controls optimization aggressiveness. One important optimization is compiling into a CUDA graph. [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/) can capture and replay a sequence of kernels entirely on the GPU, reducing CPU overhead from kernel launches.

Consider a [feed-forward network](https://en.wikipedia.org/wiki/Feedforward_neural_network) with 3 linear layers and 3 activations on a single GPU. The data loading, forward pass, backward pass, and optimizer step can be illustrated as:

{: .picture}
<img width="59.6%" height="59.6%" src="{{ site.assets }}/distributed-training/imgs/dnn_serial.svg"/> 

{: .caption}
Workflow of a single-GPU training iteration. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

In this example, the GPU executes many separate kernels in the forward and backward passes. If captured as a CUDA graph, those kernels can be replayed with fewer launches, which often improves throughput. The following picture illustrates the same iteration with a compiled model, where the forward and the backward passes each run as one compiled graph:

{: .picture}
<img width="59.7%" height="59.7%" src="{{ site.assets }}/distributed-training/imgs/dnn_serial_compiled.svg"/>

{: .caption}
Workflow of a single-GPU training iteration with a compiled model. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

Now consider a 2-GPU DDP run. During [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), DDP performs [`all_reduce`](https://en.wikipedia.org/wiki/Collective_operation#All-Reduce) operations to synchronize gradients across ranks. These collectives can overlap with gradient computation because `loss.backward()` accumulates into `w.grad` and `optimizer.step()` applies updates later. Thus, the gradients of the last layers, which are computed first, can be reduced while the gradients of the earlier layers are still being computed, as illustrated below:

{: .picture}
<img width="61.2%" height="61.2%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc.svg"/>

{: .caption}
DDP execution workflow on 2 GPUs. The optimizer waits for all asynchronous `all_reduce`s of gradients to finish. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

In a compiled setting, intermediate communication steps can introduce **graph breaks**, i.e., points where the compiler splits the model into several graphs. One naïve option is to compile a large graph and run the `all_reduce`s only at the end, but that reduces overlap and increases the time the optimizer waits:

{: .picture}
<img width="64.4%" height="64.4%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc_compiled.svg"/>

{: .caption}
DDP workflow on 2 GPUs with compiled models. The `all_reduce` runs only at the end, which increases the waiting time of the optimizer. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

PyTorch’s `DDPOptimizer` (explained [here](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)) addresses this by compiling around graph breaks into subgraphs and triggering asynchronous synchronization at subgraph boundaries:

{: .picture}
<img width="61.2%" height="61.2%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc_optimized.svg"/>

{: .caption}
DDP workflow on 2 GPUs with compiled subgraphs and `DDPOptimizer`. Synchronization runs asynchronously between subgraphs. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

Note that the subgraph boundaries follow the DDP gradient buckets, which are built from the parameter sizes, so they do not change with input shapes. Also, CUDA graphs can increase memory usage, and the additional graphs of `DDPOptimizer` add to it, so DDP runs often turn CUDA graphs off. As a side note, the author's experiments with DeepSpeed ZeRO-0 (i.e., with [ZeRO](https://arxiv.org/abs/1910.02054) sharding disabled) did not achieve the same distributed speedups as PyTorch DDP, possibly due to the lack of an equivalent optimization to `DDPOptimizer` in DeepSpeed.

### Static compilation of variable-shaped inputs

PyTorch provides limited support for compiling variable-shaped tensors via `torch.compile(dynamic=True)`. In practice (in the author's tests with `torch==2.4.0`), dynamic compilation was slower than static compilation and did not work reliably when both the batch size and the sequence length were dynamic.

Static compilation is ideal, but how to make it work with variable shapes? A practical approach is to ensure that each rank executes at least one forward+backward pass for every *shape* it will see later, near the beginning of training, so that the compiler can generate (and cache) a binary per shape. The idea is illustrated below:

{: .picture}
<img width="61.2%" height="61.2%" src="{{ site.assets }}/distributed-training/imgs/torch_compile_dataset.svg"/>

{: .caption}
Dataset setup that allows static compilation for variable-length training on 3 processes. **Top (a):** original dataset of 31 samples (bars) of 4 different lengths (bar heights), with the default interleaved assignment to the processes (color-coded). The green and yellow ranks encounter a new shape later in training (red cross). This triggers a recompilation and, depending on the settings, possibly a runtime failure. **Bottom (b):** after reshuffling, each rank sees one sample of each shape (shapes 1 to 4) early on. One binary is compiled per shape, and no recompilation happens afterwards.

Compilation behavior can be inspected by setting `TORCH_LOGS=recompiles`. For example, if the dataset contains lengths $$10$$, $$20$$, $$30$$, and $$40$$, recompiles are logged when a new length appears:

```
Recompiling function forward in train.py:145 triggered by the following guard failure(s):
    - tensor 'L['timestep']' size mismatch at index 0. expected 10, actual 20
```

and similarly for subsequent new lengths.

Note that each new shape can produce a new compiled variant that consumes memory. If the number of shapes is large, this can lead to out-of-memory (OOM) issues. Also, `torch.compile` keeps at most `torch._dynamo.config.cache_size_limit` compiled variants of a function (8 by default), and runs any further shape without compilation. Two practical mitigations are:

1. **Pad/bucket lengths** to reduce the number of distinct shapes.
2. **Order inputs by size** and call `torch.compiler.reset()` after a set of compilations to free cached variants that will not be needed again.

Finally, arguments like the following are passed to `torch.compile` to implement this logic, with CUDA graphs on single-GPU runs only:

```python
world_size = torch.distributed.get_world_size()
mode = "max-autotune" if world_size == 1 else "max-autotune-no-cudagraphs"  # single vs multi-GPU
dynamic = False  # force static compilation
model = torch.compile(model, backend="inductor", mode=mode, dynamic=dynamic)
```

As a final remark, this was tested on `torch==2.4.0`, and newer PyTorch releases continue to improve compilation for distributed and dynamic-shape workloads. For more details, see [“`torch.compile` the missing manual”](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/).

