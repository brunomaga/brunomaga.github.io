---
layout: post
title:  "Distributed training of variable-length datasets: curriculum learning, adaptive batch size and learning rate, and kernel compilation"
categories: [machine learning, distributed computing]
tags: [machinelearning]
---

Many datasets include samples of variable lengths. To name a few, audio tracks of variable durations, text sentences of variable number of words (tokens) and videos of variable number of frames. To train a machine learning model with such data, one usually trims and pads all samples to a fixed length, so batch shapes are consistent across training iterations. Alternatively, one can perform training with the original sample sizes, which adds some complexity, particularly on distributed (multi-node, multi-GPU) compute environment. 

So in this post, we will introduce and implement three techniques that accelerate training of variable-length inputs on multi-process runs: (1) **curriculum learning** to make the model learn better and faster, (2) **adaptive batch size and learning rate** that better utilize hardware by allowing large batches of short samples and vice-versa with an adequate learning rate, and (3) **kernels static compilation** to accelerate the execution.

## Curriculum Learning

[Curriculum learning](https://arxiv.org/abs/2101.10382) is an ML training method that presentes training samples to the model in the order of increasing difficulty, e.g. by increasing increasing noise, human score, length. This has been shown to improve the model stability and performance. The underlying rationale is that presenting very difficult tasks at the early stages of training may leads to high gradients and strong shifts in parameter values that may make learning unstable. However, showing samples in increasing difficulty will lead to a smoother and more stable learning process. In terms of efficiency, packing short and long sentences in one batch forces the batch to be padded to the longest sentence, adding a substantial computation memory overhead. 

The workflow for implementing curriculum learning is pretty straightforward: (1) collect the difficulty of each sample; (2) sort samples by increasing difficulty, and (3) process samples in their new order. In distributed runs with very large datasets, this is a hard process. The main struggle is due to data samples loaded in a distributed fashion across processes - defaulted to an interleaved assignment if you use torch's `DistributedSampler` - leading to a high load imbalance across processes, as pictured in a) and b) in the picture below.

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Training-Variable-Length/curriculum_datasets.png"/>

{: style="text-align:center; font-size: small;"}
An illustration of the curricum dataset setup problem on a network of 4 ranks and a dataset of 16 samples.

There are two ways of overcoming this:
- the simple, load-imbalanced, non-deterministic curriculum: load samples in a distributed fashion, sort locally the samples of each process, and have each process run curriculum on its local dataset, as in diagram c) below. The main issue here is potentially high load imbalance, runtime imbalance, and non-determinism for runs with different process counts;
- the complex, load balanced, deterministic curriculum: perform a distributed sorting of samples across all processes - diagram d) below - and re-assign that dataset in an interleaved fashion - diagram e) - that leads to an almost perfectly-balanced distributed of samples across processes - diagram f). 

In the next sections we will detail the latter option.

### Distributed Sorting

The tricky bit in the algorithm above is the distributed sorting that performs the transformation from steps b) to d). There are [other distributed sorting algorithms]({{ site.baseurl }}{% post_url 2014-06-21-Distributed-Sort %}) that one could use. But here we will implement the Distributed Sample Sorting algorithm, as it scales well for a large number of processes. The workflow is the following:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/Distributed-Sort/sample_sort.png"> 

The python implementtion of this distributed sorting algorithm is provided below.

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

 If you are interested in the remaining code, check my [DeepSpeed PR 5129](https://github.com/microsoft/DeepSpeed/pull/5129), where you can find the support code, including the method to write the post-sorting distributed tensor in d) to a sequential file (method `file_write_ordered`).


## Adaptive batch size and learning rate

When training variable-length datasets, batches of similar sizes may lead to inputs of very different lengths. Thus, a common practice is to pack batches by token count instead, by batching together samples whose sum of lengths add up to an user-provided value. As an example related to text datasets, in [Attention is all you need](https://arxiv.org/abs/1706.03762), section 5.1:

> Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

This attempts of providing similar workload across processes. Moreover, it is particularly relevant for curriculum learning where a batch of shape `BxTxE` (Batch size x Time x Embedding) will have high `B` and low `T` at the early curriculum steps (many short sentences packed together as a batch), and low `B` and high `T` at the late steps (few long samples in the batch). However, a dynamic batch size `B` requires an adequate increase/decrease of learning rate (LR). This technique has been applied previously, and the two most common LR scaling algorithms have been described as:
1. Linear Scaling Rule: "When the minibatch size is multiplied by k, multiply the learning rate by k", as in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Goyal et al.](https://arxiv.org/abs/1706.02677)
2.  Square Root scaling: "when multiplying the batch size by k, multiply the learning rate by √k, to keep the variance in the gradient expectation constant" by  [One weird trick for parallelizing convolutional neural networks, A. Krizhevsky et al.](https://arxiv.org/abs/1404.5997)

In practice, our hyper-parameters are (1) the total token count per batch, (2) a reference learning rate and (3) a reference batch size. During training, in every iteration, samples are packed in a batch until they reach the total token count, and the LR will be adjusted respectively to the new batch size (based on the reference batch size and LR).

### Illustrative example

Imagine we pick a limit of $$30$$ tokens per batch, and have set a reference learning rate of $$10^{-3}$$ and a reference batch size of $$2$$. The batching algorithm for curriculum will pack the data into batches of short sentences (left) at the early stages, and batches of long sentences (right) as later stages, as illustrated below:

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Training-Variable-Length/variable_batch_lr.png"/>

Above, we collected samples until we filled up the batch with at most 30 tokens. The batch sizes (number of samples) became then $$10$$ and $$4$$ on the left and right examples, respectively. Using the linear scaling rule, the LR for those batches become $$5*10^{-3}$$ and $$2* 10^{-3}$$.

### Pipeline parallelism

[Pipeline parallelism]({{ site.baseurl }}{% post_url 2023-08-30-GPT-lite-DeepSpeed-2%})
 requires the same batch size and same sequence length across all micro-batches in a batch, as the activation sizes must be fixed between gradient accumulation steps. Enforcing similar `BxTxE` across micro-batches may lead to smaller micro-batches. As an example, we can see below an illustration of a setup of 2 processes, training with 2 gradient accumulation steps, ie 4 micro-batches in totlal, applied to the regular Distributed Dara Parallel (DDP, left) and for the pipeline parallelism use cases (right):

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Training-Variable-Length/variable_batch_lr_pipeline.png"/>

We can see that the pipeline use case (right) has the same `BxTxE` shape across all the 4 micro-batches in the same batch, and in order to respect that, it packs less samples in the batch, when compared to the standard use case (left hand size), and that it also uses padding when it needs to enforce the expected shapes. 

### Attention matrix

Attention is heavily used nowadays for any data format, and is a particular caveat in variable-length training. Usually, on batches of fixed shapes, an input of size `BxTxE` requires the definition of a 2D attention mask of shape `TxT`, which is then broadcasted across the `B` dimension. However, when samples have different sizes, we need a `BxTxT` batch mask to enforce a different mask per sample. This 3D attention matrix can be illustrated for the non-pipeline microbatch 1 (picture above, top-left, 4 sentences) as:
 
{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Training-Variable-Length/variable_attn_matrix.png"/>


As a final remark, notice the memory savings and peformance increase: the attention head has a size of `BxTxT` leading a linear memory increase on the batch size `B` and quadratic memory increase on the largest sequence length `T` in the batch. Thus, instead of having a fixed `T` and `B` values, we now allow for a dynamic size `T` that allows for an better resource utilization by adjusting `B` .

### Implementation

The variable learning rate is implemented as a scheduler that wraps an-user provided optimizer of LR scheduler, and adapts its learning rate at every iteration:

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

The adaptive batch size is implemented as a data loader that uses a *special* collate function that pack samples into batches, given a token count per sample:

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

If you're looking for an example, you can find the complete implementation in my [DeepSpeed PR 5237](https://github.com/microsoft/DeepSpeed/pull/5237/). 

## Kernels compilation

We have seen in a [previous post]({{ site.baseurl }}{% post_url 2023-06-27-GPT-lite-cpp %}) that just-in-time compilation of ML kernels via `torch.compile` leads to a substantial training speedup. So let's look at two important aspects: (1) compilations on distributed runs and CUDA graphs, and (2) static vs dynamic compilation.


### CUDA graphs compilation on single- vs multi-process runs

Compilation via [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) requires the definition of a `mode` that defines the level of optimisations of the compiled binary. An important optimisation is the compilation of the model as a CUDA graph. [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/) allow for several GPU kernels to be compiled and executed entirely on the GPU as a workflow (graph), leading to a performance boost. However, they are not supported on multi-GPU runs. Why?

Imagine a feed-forward network composed of 3 linear layers and 3 activations, on a run with a single GPU. The data input loading, forward pass, backward pass and optimizer step can be illustrated as:

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/Training-Variable-Length/dnn_serial.png"/> 

{: style="text-align:center; font-size: small;"}
A single-GPU training iteration workflow (adapted from the original [pytorch dev discussion](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)).

In the example above, the GPU performs a total of 12 kernels in the forward and backward steps. If we compile it with CUDA graphs we will ideally have only 2 GPU kernels for the same 2 steps, which leads to a speedup: 

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/Training-Variable-Length/dnn_serial_compiled.png"/>

{: style="text-align:center; font-size: small;"}
A single-GPU training iteration workflow of a compiled model (adapted from the original [pytorch dev discussion](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)).

Now let's take the problem into scale, and imagine a 2-GPU distributed data parallel run. At every subset of layers/parameters, we perform an `all_reduce` communication step to communicate the gradients. But note that overlapping computation and communication of gradients is possible because in torch: `loss.backward()` does `w.grad += dL/dw` and `optimizer.step()` does `w += -lr * w.grad` , so we can send gradients of past layers asynchronously while computing the gradients of the following layers and have `optimizer.step()` wait for all gradients at the end:

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/Training-Variable-Length/dnn_multiproc.png"/>

{: style="text-align:center; font-size: small;"}
The Distributed Data Parallel execution workflow on 2 GPUs. The optimizer needs to wait for all asynchronous `allreduces` of gradients to finish (source: [pytorch dev discussion](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)).

Note that the previous execution has 2 **graph breaks** (an interruption in the computation graph due to an operation that is not differentiable) that are introduced by the intermediatte communication steps. Therefore, we'd need to disable CUDA graphs for this to be compiled. However, one could still compile the model into a graph and perform the `allreduce` synchronization at the final layer, adding an overhead to the time that the optimizer must wait for:

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/Training-Variable-Length/dnn_multiproc_compiled.png"/>

{: style="text-align:center; font-size: small;"}
The Distributed Data Parallel execution workflow on 2 GPUs, with compiled models. The `allreduce` runs only at the end, creating a long waiting time for the optimizer to start. (source: [pytorch dev discussion](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)).

This issue is solved by torch's `DDPOptimizer` (explained [here](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)) that will compile layers delimited by graph breaks as subgraphs, and then perform asynchronous synchronization at subgraph boundaries: 

{: style="text-align:center; font-size: small;"}
<img width="55%" height="55%" src="/assets/Training-Variable-Length/dnn_multiproc_optimized.png"/>

{: style="text-align:center; font-size: small;"}
The Distributed Data Parallel execution workflow on 2 GPUs, with compiled models and `DDPOptimizer`. Graph is compiled in subgraphs and synchronisation happens asynchronously between subgraphs. (source: [pytorch dev discussion](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860)).

Note that the subgraphs created may change depending on input sizes. As a personal note, I have not been able to achieve a similar speedup on distributed compiled models on DeepSpeed ZeRO-0 as I got on Torch's DDP, possibly due to the lack of an optimization like `DDPOptimizer` in DeepSpeed.  

### Static compilation of variable-shaped inputs

Torch provides *some* support for compilation of tensors of variable shapes, with `torch.compile(dynamic=True)`. This yields a binary that is slower than the ones provided with static compilation, with the advantage that the input dimensions are not hard-coded in the binary and can change throughout time. Dynamic compilation is slower than static, and as up to  `torch==2.4.0`, didn't work in my tests of variable batch and length dimensions. 

Static compilation is ideal, but how do it handle it with variable-shape inputs? The trick is to allow all processes to do a forward and a backward pass on every possible shape at the onset of the execution: 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/Training-Variable-Length/torch_compile_dataset.png"/>

{: style="text-align:center; font-size: small;"}
Setting up the dataset to allow static compilation of variable-length datasets on an environment with 3 processes. **Top, a)**: default dataset performing interleaved assignment of samples to processors (color-coded). The compiled model of the green and yellow processors are presented with a new shape (marked with a red cross) after few iterations, leading to a runtime error.  **Bottom, b)**: reshuffling the dataset in order to present at the onset of execution one sample of each shape to each of the 3 GPUs, allows the processes to compile one binary per shape and for the execution to run successfully.

We can analyze the behaviour of torch compile to check the shapes it's compiling by setting the environment variable `TORCH_LOGS=recompiles`. As an example, imagine the dataset pictures above has samples with the 4 lengths $$10$$, $$20$$, $$30$$ and $$40$$. In the first train iteration, torch will compile a model for an input of length $$10$$, as expected. In the second iteration, it will throw the a warning for the missing binary for the new shape, and recompile (and store in memory) another binary for the length $$20$$:

```
Recompiling function forward in train.py:145 triggered by the following guard failure(s):
    - tensor 'L['timestep']' size mismatch at index 0. expected 10, actual 20
```

In third iteration will again throw a similar warning and recompile another binary for the length $$30$$:

```
Recompiling function forward in train.py:145 triggered by the following guard failure(s):
    - tensor 'L['timestep']' size mismatch at index 0. expected 10, actual 30
    - tensor 'L['timestep']' size mismatch at index 0. expected 20, actual 30
```

and similary on the fourth iteration for the length $$40$$:
```
Recompiling function forward in train.py:145 triggered by the following guard failure(s):
    - tensor 'L['timestep']' size mismatch at index 0. expected 10, actual 40
    - tensor 'L['timestep']' size mismatch at index 0. expected 20, actual 40
    - tensor 'L['timestep']' size mismatch at index 0. expected 30, actual 40
```

Now keep in mind that every new shape will lead to a new compilation and will require a new model binary to be stored in memory. Thus, when the number of binaries, this will lead to an Out-Of-Memory (OOM) error. To overcome this, you have two options:
1. perform **padding** of samples to reduce the number of different lengths across the dataset;
2. order inputs by size, and call `torch.compiler.reset()` after few compilations to reset the torch compile status in order to free memory of previous shapes that won't be used again. 

Finally, pass the following arguments to `torch.compile` to implement this logic:
```python
world_size = torch.distributed.get_world_size()
torch_compile_kwargs={
    "backend": "inductor", # default
    "mode": "reduce-overhead" if world_size == 1 else "default", # single- vs multi-GPU runs
    "dynamic": False, # force static compilation
},
torch.compile(model, **torch_compile_kwargs)
```

As a final remark, note that I tested this on `torch==2.4.0` and may not work in earlier versions. 