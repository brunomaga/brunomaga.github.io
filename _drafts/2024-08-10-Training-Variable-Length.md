---
layout: post
title:  "Distributed training of variable-length datasets: curriculum learning, variable batch size and learning rate, and static compilation"
categories: [machine learning, distributed computing]
tags: [machinelearning]
---

Many datasets include samples that are of variable length. To name a few, audio tracks have different durations, text sentences have a different number of words (tokens) and videos have a different number of frames. To train a machine learning model with such data, one faces two options: trim and pad all samples to a fixed length, or perform training with the original sample sizes. Here we foccus on the second approach, on a distributed (multi-node, multi-GPU) compute environment.

A major bottleneck of variable-length datasets is the heterogeneity across sample lengths leading to a high load imbalance across processes. So in this post, we will introduce and implement three features that accelerate such use cases on distributed memory: (1) **curriculum learning** to make the model learn better across all lengths, (2) **variable batch size and learning rate** that better utilize hardware by allowing large batchs for small smaples and vice-versa, and (3) **kernels compilation** to accelerate the execution.

<br/>

## Curriculum Learning

[Curriculum learning](https://arxiv.org/abs/2101.10382) is an ML training method that trains samples in the order of increasing difficulty (e.g. noise, human score, length), that has been shown to improve the model stability and performance. The underlying rationale is that if one presents difficult tasks to a model at the early stages of training, the model may have strong shifts of gradients (parameters) that may make learning hard or unstable. Showing samples ordered from an easy to a hard task overcomes this process as the model iteratively adapts and improves its performance as we increase the task difficulty. 

The workflow of implementing curriculum learning in a single process run is pretty straightforward: (1) collect the difficulty of each sample; (2) sort samples by increasing difficulty, and (3) process samples in their new order. In distributed runs with very large datasets, this is much harder. The main struggle is due to data samples being loaded in a distributed fashion across processes - defaulted to an interleaved assignment if you use torch's `DistributedSampler`, as pictured in a) and b) in the picture below. There are two ways to overcome this:
- the simple, slighly imbalanced, non-deterministic curriculum: load samples in a distributed fashion, sort locally the samples of each process, and use individual curriculum datasets - one per process - as in diagram c) below. The main issue here is potentially high load imbalance, runtime imbalance, and a run that is not deterministic for different process counts.
- the complex, load balanced, deterministic curriculum: perform a distributed sorting of samples across all processes - diagram d) below - and re-assign that dataset in an interleaved fashion - diagram e) - that leads to an almost perfectly-balanced distributed of samples across processes - diagram f). 

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Training-Variable-Length/curriculum_datasets.png"/>

{: style="text-align:center; font-size: small;"}
An illustration of the curricum dataset setup problem on a network of 4 ranks and a dataset of 16 samples, across 6 different steps explained in this post.

### Distributed Sorting

The tricky bit in the algorithm above is the distributed sorting that performs the transformation from b) to d) in the diagram above. In this post, we will user the Distributed Sample Sorting algorithm (pictured below) so that it scales well for a large number of processes, but there are [other distribubed sorting algorithms in this post]({{ site.baseurl }}{% post_url 2014-06-21-Distributed-Sort %}) that you could use. The workflow is the following:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/Distributed-Sort/sample_sort.png"> 

The python implementtion of this distributed sorting algorithm is provided below. If you are interested in the remaining code for e.g. sequential writing of a distributed tensor to a single file, check the [DeepSpeed PR 5129](https://github.com/microsoft/DeepSpeed/pull/5129) where I implemented all this logic:

```python
def sample_sort(tensor, comm_group, num_workers, n_samples=100):
    """ perform a distributed random sort of a tensor, and returns the sorted partial tensor"""
    device, dims = tensor.device, tensor.size()[1]

    # 1 - sort rows by first column, then second column, then third, etc...
    tensor = torch.tensor(sorted(tensor.tolist()), dtype=tensor.dtype, device=tensor.device)

    # 2 - collect few samples per rank
    idx = torch.round(torch.linspace(0, len(tensor) - 1, n_samples)).to(int)
    samples = tensor[idx][:, 0].contiguous().to(device)  #only first column, all but last row

    # 2 - Allgather samples
    all_samples = [torch.zeros(n_samples, dtype=samples.dtype, device=device) for _ in range(num_workers)]
    dist.all_gather(all_samples, samples, group=comm_group)
    all_samples = torch.cat(all_samples, dim=0).to(device)

    # 3 - Sort all samples and collect the ranges of each rank as equidistant
    all_samples = all_samples.sort()[0]
    idx = torch.round(torch.linspace(0, len(all_samples) - 1, num_workers + 1)).to(int)
    ranges = all_samples[idx]  # range of each rank r as ranges[r] <= x < ranges[r+1]
    ranges[-1] += 1  # increase upper limit of last rank so that x < ranges[r+1].

    # 4 - collect elements to send to each rank, based on the rank ranges
    send = []
    for rank in range(num_workers):
        mask = (tensor[:, 0] >= ranges[rank]) & (tensor[:, 0] < ranges[rank + 1])
        send.append(tensor[mask])

    # 5. all to all to communicate the sizes to be sent/recv
    send_count = [torch.tensor([len(s) * dims], dtype=torch.int64, device=device) for s in send]
    recv_count = list(torch.empty([num_workers], dtype=torch.int64, device=device).chunk(num_workers))
    dist.all_to_all(recv_count, send_count, group=comm_group)

    # 6. all-to-all-v to communicate the elements to be sent/recv as a single tensor
    send = torch.cat(send, dim=0).flatten().to(device)
    recv = torch.zeros(sum(recv_count), dtype=send.dtype).to(device)
    send_count = [s.item() for s in send_count]  # convert to list of ints
    recv_count = [r.item() for r in recv_count]
    dist.all_to_all_single(recv, send, recv_count, send_count, group=comm_group)
    del send

    # 7. the received tensor is the 1D disjoint subset of the distributed tensor.
    # We will recover the original dimensionality and sort it by columns again.
    recv = recv.view(-1, dims)
    recv = torch.tensor(sorted(recv.tolist()), dtype=recv.dtype, device=recv.device)
    return recv
```

<br/>

## Variable batch size and learning rate

When using variable length dataset, in order to best utilize the compute resources available, it is recommended to *pack* samples in batches that have a similar workload across processes, to avoid idleness or under-utilisation of memory. To do this, a common practice is to pack batches by token count (not by a fixed batch size), ie by putting together samples whose lengths  add up to an user-provided value. As an example in the text context, in [Attention is all you need](https://arxiv.org/abs/1706.03762), section 5.1:

> Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

This is also particularly relevant for curriculum learning where a `BxTxE` (Batch x Time x Embedding) -shaped input should ideally have high `B` and low `T` at the early curriculum steps (many short sentences packed together as a batch), and low `B` and high `T` at the late steps (few long samples in the batch). However, dynamic batch size `B` requires an adequate increase/decrease of learning rate. This technique has been applied previously, and the two most common LR scaling algorithms have been described as:
1. Linear Scaling Rule: "When the minibatch size is multiplied by k, multiply the learning rate by k", as in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Goyal et al.](https://arxiv.org/abs/1706.02677)
2.  Square Root scaling: "when multiplying the batch size by k, multiply the learning rate by √k, to keep the variance in the gradient expectation constant" by  [One weird trick for parallelizing convolutional neural networks, A. Krizhevsky et al.](https://arxiv.org/abs/1404.5997)

In practice, the user picks the total token count per batch as the metric that drives batching, instead of batching by sample count. During runtime, the variable batch size is computed and the LR is adjusted respectively, based on the LR and batch size provided by the config.

### Illustrative example

Imagine we picked a limit of `30` tokens per batch, and have set a reference `lr=1e-3` for a `train_batch_size=2` (in the deepspeed config). The batching algorithm for curriculum may pack the data into batches of short sentences (left) at the early stages, and batches of long sentences (right) as later stages, e.g.:

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Training-Variable-Length/variable_batch_lr.png"/>

Above, we collected samples until we filled up the batch with at most 30 tokens. The batch sizes (number of samples) became then `10` and `4` on the left and right examples, respectively. Using the linear scaling rule, the LR for those batches become `5e-3` and `2e-3`.    

### Pipeline parallelism

Pipeline parallelism requires the same batch size and same sequence length across all micro-batches in a batch, as the activation sizes must be fixed between gradient accumulation steps, so that the shapes of all pipeline steps is the same. Enforcing similar `BxTxE` between batches may lead to smaller micro-batches. As an example, below we can see an illustration of a 2-node 2-gradient-accumulation-step (ie 4 micro-batches) batching for the same dataset, when preparing data for the regular DDP (left) and for the pipeline parallelism use cases (right):

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Training-Variable-Length/variable_batch_lr_pipeline.png"/>

We can see that the pipeline use case (right) has the same `BxTxE` shape across all the 4 micro-batches in the same batch, and in order to respect that, it packs less samples in the batch, when compared to the standard use case (left hand size), and uses padding when needed to respect the expecte shapes. 

### Attention matrix

Attention is heavily used nowadays for any data format, and is a particular caveat in variable-length training. Usually, on fixed-size batching, an input of size `BxTxE` requires an attention mask of shape `TxT`. However, when samples have different sizes, we need a `BxTxT`  mask to allow a different mask per sample. This 3D attention matrix can be illustrated for the non-pipeline microbatch 1 (picture above, top-left, 4 sentences) as:
 
{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Training-Variable-Length/variable_attn_matrix.png"/>


Note the memory savings: the attention head has a size of `BxTxT`, i.e. a linear memory dependency on the batch size `B` and quadratic memory dependency on the largest sequence length `T` in the (micro-) batch. Thus, supporting a dynamic size `T` allows for an increase of `B`.

### Implementation

To be brief, the variable LR is implemented as a scheduler that wraps an user provided scheduler and adapts its learning rate at every iteration:

```python
class VariableBatchSizeLR(LRScheduler):
    """ an LR scheduler that scales the LR of a given scheduler's LR """

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
        # Note: optimizer.step precedes lr_scheduler.step(), so the stepping workflow is:
        # init: lr_scheduler.step(0) --> set LR for epoch 0
        # epoch 0: optimizer.step(); lr_scheduler.step(1) --> set LR for epoch 1
        # epoch 1: optimizer.step(); lr_scheduler.step(2) --> set LR for epoch 2

        # reset unscaled LRs (to the original scheduler's one) for the current epoch
        # Note: epoch==0: reset LR scheduler; epoch==None: scale LR for next epoch;
        unscaled_lrs = self.base_lrs if epoch == 0 else self.get_last_lr()
        for group, lr in zip(self.base_lr_scheduler.optimizer.param_groups, unscaled_lrs):
            group['lr'] = lr

        self.base_lr_scheduler.step(epoch)

        # scale the learning rate for next epoch for each parameter group.
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        batch_size = self.batch_sizes[self.last_epoch % len(self.batch_sizes)]
        for group in self.base_lr_scheduler.optimizer.param_groups:
            group['lr'] = scale_lr(self.base_batch_size, batch_size, group['lr'], self.lr_scaling_method)
```

The variable batch size is implemented by a data loader that uses a *special* collate function to pack samples into batches, given a token count per sample:

```python 
def dataloader_for_variable_batch_size( dataset, microbatch_ids, batch_max_seqlens,
    dataloader_rank=0, dataloader_batch_size=1, dataloader_num_replicas=1,
    dataloader_collate_fn=None, dataloader_num_workers=2, dataloader_pin_memory=False,
    required_microbatches_of_same_seqlen=False, sample_padding_fn=None,
):

    # equidistantly distribute the microbatches across the replicas in an interleaved fashion.
    sampler = DistributedSampler(
        dataset=microbatch_ids,
        num_replicas=dataloader_num_replicas,
        rank=dataloader_rank,
        shuffle=False,
        drop_last=False,
    )

    # collate function wraps user-defined collate function to the variable batch data
    def collate_fn_wrapper(list_microbatch_ids):
        # each batch is a list of sample ids that fill up to the max tokens per batch
        # we return the collated batch of all dataset samples of all input batches.
        batch = []
        for batch_id, microbatch_ids in list_microbatch_ids:
            batch_data = [dataset[idx] for idx in microbatch_ids]
            if required_microbatches_of_same_seqlen:
                assert sample_padding_fn is not None, \
                    "padding dataloader_padding_fn must be provided if required_microbatches_of_same_seqlen is True"
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

You can find the complete implementation in my [DeepSpeed PR 5237](https://github.com/microsoft/DeepSpeed/pull/5237/). 