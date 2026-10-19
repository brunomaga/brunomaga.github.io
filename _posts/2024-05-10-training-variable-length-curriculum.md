---
layout: post
title:  "Variable sequence lengths, curriculum learning, adaptive batch size and LR"
categories: ["Distributed Training"]
tags: ["curriculum learning", "variable sequence length", "adaptive batch size", "CUDA graphs", "DeepSpeed"]
---

Many datasets contain samples of variable length, such as audio tracks of different durations, sentences with different numbers of tokens, or videos with different numbers of frames. A common way to train a model on such data is to trim and pad all samples to a fixed length, so that batch shapes are the same in every training iteration. The alternative is to train on the original sample sizes. This adds complexity, especially in distributed (multi-node, multi-GPU) environments.

This post describes and implements three techniques that accelerate training with variable-length inputs in multi-process runs: (1) **curriculum learning**, to make learning faster and more stable; (2) **adaptive batch size and learning rate**, to improve hardware utilization with large batches of short samples (and vice versa) and a corresponding learning-rate adjustment; and (3) **static kernel compilation**, to reduce runtime overhead.

## Curriculum Learning

[Curriculum learning](https://arxiv.org/abs/2101.10382) is a training method that presents samples to the model in order of increasing difficulty (e.g., increasing noise level, human score or length). It can improve stability and final performance. Very difficult samples early in training can produce large gradients and abrupt parameter updates, which may destabilize learning. A gradual increase in difficulty leads to a smoother and more stable learning process.

Curriculum learning also improves efficiency on variable-length data. When short and long sequences are mixed in the same batch, all sequences are padded to the length of the longest one, which adds substantial compute and memory overhead.

Curriculum learning has three steps: (1) define a difficulty metric per sample, (2) sort samples by increasing difficulty, and (3) iterate through the sorted dataset. In distributed runs with large datasets, however, this is non-trivial. Samples are typically assigned to ranks in an interleaved fashion (e.g., by PyTorch’s `DistributedSampler`; see (a) and (b) below). If each rank sorts only its local shard, the result can be load imbalance.

{: .picture}
<img width="80.7%" height="80.7%" src="{{ site.assets }}/distributed-training/imgs/curriculum_datasets.svg"/>

{: .caption}
The curriculum dataset setup problem, on a network of 4 ranks and a dataset of 16 samples. (a) Original dataset, with an interleaved assignment of the 16 samples (bars) of variable lengths (bar heights) to the 4 ranks (color-coded). (b) Dataset in (a) after distributed data loading. (c) Dataset in (b) after local sorting. (d) Dataset in (b) after distributed sorting. (e) Dataset in (d) with an interleaved assignment of the 16 samples to the 4 ranks (color-coded). (f) Dataset in (e) after distributed data loading.

There are two main approaches:

- **Simple (but potentially imbalanced and non-deterministic):** each rank loads its shard, sorts it locally, and applies curriculum learning to its local order (diagram (c)). This can cause load and runtime imbalance, and the results depend on the number of ranks.
- **Complex (but balanced and deterministic):** a distributed sort across all ranks (diagram (d)) is followed by an interleaved reassignment of the globally sorted dataset (diagram (e)). The result is an almost perfectly balanced distribution across ranks (diagram (f)).

The following sections describe the second approach.

### Distributed Sorting

The difficult step is the distributed sort that transforms (b) into (d). There are other distributed sorting algorithms, but this implementation uses the **Distributed Sample Sort** algorithm, because it scales well to many processes. The workflow is:

{: .picture}
<img width="57%" height="57%" src="{{ site.assets }}/distributed-sort/sample_sort.svg"> 

{: .caption}
Sample sort of 21 elements on $$p = 3$$ ranks, colored by rank. Each rank sends $$p-1 = 2$$ samples to rank 0, which sorts them and picks every $$p$$-th one as a splitter (yellow): 4 and 10. The last interval is open: rank 2 receives every element from 10 upwards.

The Python implementation is:

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

The remaining code is available in [DeepSpeed PR 5129](https://github.com/microsoft/DeepSpeed/pull/5129). It includes support utilities, e.g., `file_write_ordered`, which writes the sorted distributed tensor in (d) to a sequential file.

## Adaptive batch size and learning rate

On variable-length datasets, batches are commonly built by *token count* instead of sample count: samples are grouped so that the sum of sequence lengths in each batch stays close to a target. For example, [Attention is all you need](https://arxiv.org/abs/1706.03762), section 5.1, states:

> Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

This balances the work across batches and across ranks. It is also relevant for curriculum learning: a batch of shape `B x S x E` (batch size × sequence length × embedding size) often has a high `B` and a low `S` early in the curriculum (many short samples), and a low `B` and a high `S` later (fewer long samples).

However, changing the batch size `B` usually requires adjusting the learning rate (LR). Two common scaling rules are:

1. **Linear scaling:** when the minibatch size is multiplied by `k`, the LR is multiplied by `k` (e.g., [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Goyal et al.](https://arxiv.org/abs/1706.02677)).
2. **Square-root scaling:** when the minibatch size is multiplied by `k`, the LR is multiplied by `sqrt(k)`, which keeps gradient noise roughly constant (e.g., [One weird trick for parallelizing convolutional neural networks, Krizhevsky](https://arxiv.org/abs/1404.5997)).

The hyperparameters are therefore: (1) a target token count per batch, (2) a reference learning rate, and (3) a reference batch size. During training, samples are packed into each batch until the token budget is reached, and the LR is scaled according to the resulting batch size.

### Illustrative example

Assume a limit of $$30$$ tokens per batch, a reference LR of $$10^{-3}$$, and a reference batch size of $$2$$. Early in curriculum learning (short sequences), batches contain more samples than later (long sequences):

{: .picture}
<img width="74.6%" height="74.6%" src="{{ site.assets }}/distributed-training/imgs/variable_batch_lr.svg"/>

{: .caption}
Batches of at most 30 tokens, of shape BxSxE (batch size x sequence length x embedding size), with a reference batch size of 2 and a reference learning rate (LR) of 1e-3. Short sentences (left) give a larger batch, and thus a larger LR, than long sentences (right). Grey cells are padding tokens.

Samples are added to a batch until it reaches at most 30 tokens. The resulting batch sizes (number of samples) are $$10$$ and $$4$$ in the left and right examples, respectively. With linear scaling, the corresponding LRs are $$5 \times 10^{-3}$$ and $$2 \times 10^{-3}$$.

### Pipeline parallelism

Pipeline parallelism
requires the same batch size and sequence length in all micro-batches of a batch, because activation shapes must remain fixed during gradient accumulation. A fixed `B x S x E` across micro-batches can lead to smaller micro-batches and more padding. The figure below compares standard Distributed Data Parallel (DDP, left) with pipeline parallelism (right), for 2 processes and 2 gradient accumulation steps (4 micro-batches in total):

{: .picture}
<img width="94.9%" height="94.9%" src="{{ site.assets }}/distributed-training/imgs/variable_batch_lr_pipeline.svg"/>

In the pipeline case (right), all micro-batches have the same `B x S x E` shape. Under this constraint, each micro-batch holds fewer samples than in the non-pipeline case (left), and padding is added where needed.

### Attention matrix

Attention is a particular challenge in variable-length training. With fixed shapes, an input of shape `B x S x E` usually uses a 2D attention mask of shape `S x S`, broadcast over the batch dimension. With variable lengths, a *per-sample* mask is required, represented as a `B x S x S` tensor. For example, the mask for the non-pipeline micro-batch 1 above (top left, 4 sentences) is:

{: .picture}
<img width="99%" height="99%" src="{{ site.assets }}/distributed-training/imgs/variable_attn_matrix.svg"/>

{: .caption}
3D attention mask (BxSxS) of the non-pipeline micro-batch 1 above: its 4 samples of lengths 9, 7, 8 and 6 give a mask of size 4x9x9, with one causal mask per sample, padded to the length of the longest sample.

The cost of attention scales linearly with `B` and quadratically with the maximum `S` in the batch. Letting `S` vary, and adapting `B` accordingly, therefore improves utilization, because sequences are not padded to an unnecessarily large `S`.

### Implementation

The variable learning rate is implemented as a scheduler wrapper. At each step, it scales the LR of an underlying scheduler based on the realized batch size:

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

The adaptive batch size is implemented as a data loader with a custom collate function that packs samples into micro-batches within the token budget:

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

A complete example is available in [DeepSpeed PR 7104](https://github.com/microsoft/DeepSpeed/pull/7104/).

## Kernels compilation

Just-in-time compilation of ML kernels with `torch.compile` can give substantial speedups. This section covers two aspects: (1) compilation in distributed runs and CUDA graphs, and (2) static vs dynamic compilation.

### CUDA graphs compilation on single- vs multi-process runs

Compilation via [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) takes a `mode` argument that controls how aggressive the optimizations are. One important optimization is compilation into a CUDA graph. [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/) capture and replay a sequence of kernels entirely on the GPU, which reduces the CPU overhead of kernel launches.

Consider a feed-forward network with 3 linear layers and 3 activations on a single GPU. Its data loading, forward pass, backward pass and optimizer step are illustrated below:

{: .picture}
<img width="59.6%" height="59.6%" src="{{ site.assets }}/distributed-training/imgs/dnn_serial.svg"/> 

{: .caption}
Workflow of a single-GPU training iteration. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

In this example, the GPU executes many separate kernels in the forward and backward passes. When captured as a CUDA graph, these kernels are replayed with fewer launches, which often improves throughput:

{: .picture}
<img width="59.7%" height="59.7%" src="{{ site.assets }}/distributed-training/imgs/dnn_serial_compiled.svg"/>

{: .caption}
Workflow of a single-GPU training iteration with a compiled model. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

Now consider a Distributed Data Parallel (DDP) run on 2 GPUs. During backpropagation, DDP performs `all_reduce` operations to synchronize gradients across ranks. These collectives can overlap with gradient computation, because `loss.backward()` accumulates gradients into `w.grad` and `optimizer.step()` applies the updates later. The gradients of the last layers, which are computed first, are therefore reduced while the gradients of earlier layers are still being computed:

{: .picture}
<img width="61.3%" height="61.3%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc.svg"/>

{: .caption}
DDP execution workflow on 2 GPUs. The optimizer waits for all asynchronous `all_reduce`s of gradients to finish. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

In a compiled model, intermediate communication steps can introduce **graph breaks**. A naïve option is to compile one large graph and run all `all_reduce`s at the end. However, this reduces overlap and increases the waiting time of the optimizer:

{: .picture}
<img width="64.5%" height="64.5%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc_compiled.svg"/>

{: .caption}
DDP workflow on 2 GPUs with compiled models. The `all_reduce` runs only at the end, which increases the waiting time of the optimizer. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

PyTorch’s `DDPOptimizer` (explained [here](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)) solves this problem. It compiles the model into subgraphs separated by the graph breaks, and starts an asynchronous synchronization at each subgraph boundary:

{: .picture}
<img width="61.3%" height="61.3%" src="{{ site.assets }}/distributed-training/imgs/dnn_multiproc_optimized.svg"/>

{: .caption}
DDP workflow on 2 GPUs with compiled subgraphs and `DDPOptimizer`. Synchronization runs asynchronously between subgraphs. Adapted from: [TorchDynamo Update 9: Making DDP Work with TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)

Note that subgraph boundaries may change with input shapes. In the author's experiments, DeepSpeed ZeRO-0 did not reach the distributed speedups of PyTorch DDP, possibly because DeepSpeed lacks an optimization equivalent to `DDPOptimizer`.

### Static compilation of variable-shaped inputs

PyTorch provides limited support for compiling variable-shaped tensors with `torch.compile(dynamic=True)`. In tests with `torch==2.4.0`, dynamic compilation was slower than static compilation, and it did not work reliably when both the batch size and the sequence length were dynamic.

Static compilation is preferable, but it must handle variable shapes. A practical solution is to make each rank execute, near the beginning of training, at least one forward and backward pass for every *shape* it will see later. The compiler then generates (and caches) one binary per shape. The idea is illustrated below:

{: .picture}
<img width="61.3%" height="61.3%" src="{{ site.assets }}/distributed-training/imgs/torch_compile_dataset.svg"/>

{: .caption}
Dataset setup that allows static compilation for variable-length training on 3 processes. **Top (a):** original dataset of 31 samples (bars) of 4 different lengths (bar heights), with the default interleaved assignment to the processes (color-coded). The green and yellow ranks encounter a new shape later in training (red cross). This triggers a recompilation and, depending on the settings, possibly a runtime failure. **Bottom (b):** after reshuffling, each rank sees one sample of each shape (shapes 1 to 4) early on. One binary is compiled per shape, and no recompilation happens afterwards.

Compilation behavior can be inspected by setting `TORCH_LOGS=recompiles`. For example, if the dataset contains lengths $$10$$, $$20$$, $$30$$, and $$40$$, a recompilation is logged whenever a new length appears:

```
Recompiling function forward in train.py:145 triggered by the following guard failure(s):
    - tensor 'L['timestep']' size mismatch at index 0. expected 10, actual 20
```

and similarly for each subsequent new length.

Each new shape can produce a new compiled variant, which consumes memory. With a large number of shapes, this can lead to out-of-memory (OOM) errors. Two practical mitigations are:

1. **Pad/bucket lengths** to reduce the number of distinct shapes.
2. **Order inputs by size** and call `torch.compiler.reset()` after a set of compilations, to free cached variants that are no longer needed.

Finally, arguments such as the following are passed to `torch.compile` to implement this logic:

```python
world_size = torch.distributed.get_world_size()
mode = "max-autotune" if world_size == 1 else "max-autotune-no-cudagraphs"  # single vs multi-GPU
dynamic = False  # force static compilation
model = torch.compile(model, backend="inductor", mode=mode, dynamic=dynamic)
```

These tests used `torch==2.4.0`. Newer PyTorch releases continue to improve compilation for distributed and dynamic-shape workloads. For more details, see [“`torch.compile` the missing manual”](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/).

