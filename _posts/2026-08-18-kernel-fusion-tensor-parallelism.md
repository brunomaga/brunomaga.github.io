---
layout: post
title:  "Kernel Fusion for Megatron-LM Tensor Parallelism and MoE Expert Sharding"
categories: ["Efficient Inference"]
tags: ["LLM", "kernel fusion", "tensor parallelism", "Megatron-LM", "in-network computing", "collectives", "Mixture-of-Experts", "expert sharding"]
---

The method below fuses the computation and the communication of a [feedforward network](https://en.wikipedia.org/wiki/Feedforward_neural_network) (FFN), distributed with [Megatron-LM tensor parallelism]({{ site.baseurl }}{% post_url 2023-09-02-tensor-parallelism %}), into a single GPU kernel. It applies to the experts of a Mixture-of-Experts layer too, through expert sharding.

## The problem: kernels and graph breaks

An LLM runs as a long sequence of GPU kernels, usually one or more per layer. Each kernel loads its inputs from the global memory of the GPU ([HBM](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)) into the on-chip memory of the Streaming Multiprocessors (SMs), and writes its outputs back. *Kernel fusion* merges consecutive kernels into one, so that intermediate results stay in on-chip memory, where the next operation reads them. `torch.compile` does this automatically on a single GPU.

On several GPUs, fusion stops at every collective communication call, such as an [all-reduce](https://en.wikipedia.org/wiki/Collective_operation). The collective runs as a separate, synchronous kernel on full tensors. It acts as a *graph break*: the program is split into subgraphs, and each subgraph is fused on its own. The collectives need large communication buffers in global memory, and the GPUs wait for them to finish before they compute again.

## Megatron-LM tensor parallelism

The FFN has two linear layers. Each layer adds a bias and applies an activation, and the second activation is optional. With an input $$X \in \mathbb{R}^{M \times N}$$ of $$M$$ tokens, the FFN computes:

$$
H = \text{activation}(X W_1 + b_1), \quad Y = \text{activation}(H W_2 + b_2)
$$

where $$W_1 \in \mathbb{R}^{N \times O}$$ and $$W_2 \in \mathbb{R}^{O \times P}$$. On $$G$$ GPUs, Megatron-LM splits $$W_1$$ column-wise and $$W_2$$ row-wise. Each GPU $$g$$ computes a slice $$H^{(g)} \in \mathbb{R}^{M \times O/G}$$ of the hidden activations and a partial output $$H^{(g)} W_2^{(g)}$$. An all-reduce then sums the partial outputs of all GPUs, $$H W_2 = \sum_{g=1}^{G} H^{(g)} W_2^{(g)}$$, and the bias $$b_2$$ and the second activation follow. This all-reduce is the only communication step per pair of layers.

{: .picture}
<img width="96.7%" height="96.7%" src="{{ site.assets }}/inference/imgs/tp_megatron_lm.svg"/>

{: .caption}
Megatron-LM tensor parallelism of an FFN on 2 GPUs (A in green, B in blue). The bias and activation are omitted.

A regular implementation launches 7 kernels for this FFN: a matrix multiplication, a bias addition and an activation per layer, plus the all-reduce. Each kernel loads its inputs from global memory and stores its outputs there (top of the figure below). The fused kernel runs the whole FFN as one kernel (bottom). It loads the input and stores the output once, and its small all-reduce steps overlap with the computation. It is built in three steps, presented next.

{: .picture}
<img width="95.5%" height="95.5%" src="{{ site.assets }}/inference/imgs/tp_timeline.svg"/>

{: .caption}
Execution timeline of an FFN with Megatron-LM tensor parallelism. **Top:** the regular execution, with 7 kernels that load from (arrow up) and store to (arrow down) global memory. **Bottom:** the fused kernel, with several small, overlapping all-reduce steps performed in the network switch.

## Step 1: fusing the two linear layers

Two linear layers without a nonlinearity in between collapse into a single one. With an activation in between, the usual approach stores the full intermediate tensor $$H$$ in global memory, and a second kernel reads it back.

A matrix multiplication on a GPU is computed in *tiles*: each [thread block](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)) runs on one SM and computes one small submatrix of the output. The fused kernel keeps this tiling for the first layer. However, as soon as a tile $$H_{i,j}$$ is computed, with its bias and activation, it is multiplied by every tile in row $$j$$ of $$W_2$$. The partial products are added to the tiles in row $$i$$ of the output $$Y$$. When all tiles of $$H$$ are processed, $$Y$$ holds the partial output $$H W_2$$ of the GPU, and $$H$$ was never stored in global memory.

{: .picture}
<img width="97.8%" height="97.8%" src="{{ site.assets }}/inference/imgs/tp_matmuls.svg"/>

{: .caption}
**Left:** the regular execution computes all tiles of $$H$$, stores $$H$$ in global memory, and a second kernel computes $$Y$$. **Right:** the fused kernel multiplies each tile of $$H$$ by a row of tiles of $$W_2$$ and accumulates the results in a row of tiles of $$Y$$.

Many blocks add to the same tile of $$Y$$ at the same time, so the additions must be atomic. Blocking atomic additions quickly become a bottleneck. Instead, the kernel uses the *asynchronous* tile reductions of the GPU (`cp.reduce.async.bulk` in [PTX](https://en.wikipedia.org/wiki/Parallel_Thread_Execution)), in a *ping-pong* pattern: a block computes the product of the next tile while the reduction of the previous tile is still in flight. This hides most of the cost of the accumulation.

A per-tile counter tracks how many blocks have contributed to each output tile. The last contributor marks the tile as *ready*. A ready tile is final and can be sent to the other GPUs right away, without waiting for the rest of $$Y$$. Communication therefore happens per tile, with buffers of 32 KiB ($$128 \times 128$$ values in `bfloat16`) instead of buffers of several gigabytes.

## Step 2: overlapping computation and communication with block specialization

If every block computes and communicates, the faster blocks must [spin-wait](https://en.wikipedia.org/wiki/Busy_waiting) until all tiles of a region are ready, before they reduce it across GPUs. *Block specialization* splits the blocks of the kernel into two groups:

- **compute blocks** run the matrix multiplications of step 1 and mark tiles as ready;
- **communication blocks** each own a *band*, a contiguous range of output tiles. A communication block polls the ready flags of its band, all-reduces the band across GPUs as soon as it is complete, and applies the final bias and activation.

Compute blocks never wait for communication, and communication blocks never compute matrix multiplications. The matrix multiplications run on fewer SMs, which is only slightly slower on a GPU with many SMs, and the communication is hidden behind the computation.

The same idea appears at other levels. Inside a block, [FlashAttention-3](https://arxiv.org/abs/2407.08608) uses *warp specialization*: some warps load data from global memory while the others compute. Across the GPU, [DeepSeek-V3](https://arxiv.org/abs/2412.19437) dedicates 20 of the 132 SMs of an H800 GPU to the communication of its Mixture-of-Experts layers. Here, the specialized blocks run the all-reduce of the tensor-parallel FFN, band by band.

{: .picture}
<img width="64.4%" height="64.4%" src="{{ site.assets }}/inference/imgs/tp_block_specialization.svg"/>

{: .caption}
An example on a GPU with 6 SMs. **(a)** Without block specialization, all SMs compute and then spin-wait before each band is all-reduced. **(b)** With block specialization, SM1 and SM2 all-reduce each band as soon as SM3 to SM6 have computed it.

## Step 3: all-reduce with in-network computing

An all-reduce of a communication library such as [NCCL](https://developer.nvidia.com/nccl) is launched from the CPU, as a separate kernel, on full tensors. It cannot run from inside a running kernel, on single tiles. The fused kernel uses its own all-reduce, issued directly by the GPU threads and executed by the network switch that connects the GPUs. This is called *In-Network Computing* (INC): data is summed as it crosses the interconnect, not by the GPUs. On NVIDIA GPUs connected by an NVLink switch, the `multimem` instructions offer three operations on memory addresses that are mapped on all GPUs:

- a **load-reduce** (`multimem.ld_reduce`), which reads a value from all GPUs and returns their sum (all-to-one);
- a **multicast store** (`multimem.st`), which writes one value to all GPUs (one-to-all);
- an **in-place reduction** (`multimem.red`), which adds one value to the same address on all GPUs.

The all-reduce of a band is a reduce-scatter followed by an all-gather. Each GPU first load-reduces its $$1/G$$ share of the band, and then multicast-stores the result to all GPUs. Finally, each GPU adds 1 to a shared flag with an in-place reduction. When the flag reaches $$G$$, all GPUs know that the band is complete. This flag is a lightweight [barrier](https://en.wikipedia.org/wiki/Barrier_(computer_science)) that needs no CPU.

## Results

The fused kernel was compared with a tuned, non-fused implementation that uses the all-reduce of NCCL, on the FFNs of Llama, DeepSeek, GPT, Mistral and Qwen models, on 8 GPUs of one node connected by NVLink, in `bfloat16`. It reduces the latency by 22% to 62%, halves the memory of the activations, and reduces the communication buffer from gigabytes to 32 KiB. The chart below shows seven of these FFNs:

{: .picture}
<img width="55%" height="55%" src="{{ site.assets }}/inference/imgs/tp_results.svg"/>

{: .caption}
Latency of the regular (baseline) and fused FFNs, sorted by improvement.

The gains depend on the size of the FFN:

- **Small FFNs** (DeepSeek V3 experts, Llama 3 8B) gain the most: their latency drops by 46% to 62%. Their runtime is short and dominated by fixed costs: the extra kernels, the stores of $$H$$, and the all-reduce. The fused kernel removes or hides these costs.
- **Large FFNs** (Llama 3 70B, DeepSeek V3 dense, Llama 4 Scout shared): their latency drops by 29% to 38%. The two large matrix multiplications dominate their runtime. The absolute savings are larger, up to 122 ms, but the relative gain is smaller.

An ablation study shows that all three steps are needed. Without fusion, the in-network all-reduce runs on full tensors and is slower than the baseline. Without block specialization, the communication is not hidden, and the kernel is slower than the baseline in most cases. Without in-network computing, with [NVSHMEM](https://developer.nvidia.com/nvshmem) instead, the kernel is faster than the baseline but slower than the full method.

## Expert sharding

In a [Mixture-of-Experts (MoE)]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) layer, each expert is a small FFN, with an input matrix $$W_i$$ and an output matrix $$W_o$$ (the $$W_1$$ and $$W_2$$ above). With expert parallelism, each GPU holds a few whole experts and processes only the tokens routed to them. The router rarely spreads the tokens evenly, so the GPU with the most tokens finishes last, and the other GPUs wait for it.

[Expert sharding](https://arxiv.org/abs/2503.08467) splits every expert across all GPUs instead, as Megatron-LM splits the FFN. Each GPU holds a $$1/G$$ share of the columns of $$W_i$$ and the matching rows of $$W_o$$ of every expert. It processes all the tokens with these slices, and the partial outputs of all GPUs are summed. Each expert is then an FFN with Megatron-LM tensor parallelism, and runs with the fused kernel.

{: .picture}
<img width="59.5%" height="59.5%" src="{{ site.assets }}/inference/imgs/tp_expert_sharding.svg"/>

{: .caption}
One expert, with $$W_i$$ in green and $$W_o$$ in red. The activation is omitted. **(a)** Without parallelism, one GPU holds the whole expert. **(b)** With expert sharding, GPU A holds the first half of the columns of $$W_i$$ and of the rows of $$W_o$$, and GPU B the second half. The dashed slices are on the other GPU. Both GPUs process all the tokens $$x$$, and a sum-reduce adds their partial outputs $$y_A$$ and $$y_B$$. Inspired by Figure 2 of [Accelerating MoE Model Inference with Expert Sharding](https://arxiv.org/abs/2503.08467).

Every GPU then does the same amount of work, however the tokens are routed. No GPU waits for another, no tokens are dropped, and popular experts need no replicas. Both methods keep the same share of the expert weights on each GPU. Expert parallelism sends each token only to the GPUs of its experts, but its runtime follows the most loaded GPU. Expert sharding needs all the tokens on every GPU and an all-reduce of the partial outputs: it trades more communication for a perfect load balance. The fused kernel overlaps this all-reduce with the computation and runs it in the network switch. The results above include two experts split across 8 GPUs this way: the fused kernel reduces the latency by 55% on an expert of DeepSeek V3, and by 45% on an expert of Llama 4 Scout.

## Related work

Other systems also overlap the computation and the communication of tensor parallelism inside kernels:

- [CoCoNet](https://arxiv.org/abs/2105.05720) is a compiler that fuses and overlaps a matrix multiplication with the collective that follows it;
- [Flux](https://arxiv.org/abs/2406.06858) splits the Megatron-LM matrix multiplications and their collectives into tiles, and fuses them into one kernel per layer, so that the communication of each tile overlaps the computation of other tiles;
- [Triton-distributed](https://arxiv.org/abs/2504.19442) and [TileLink](https://arxiv.org/abs/2503.20313) add [SHMEM](https://en.wikipedia.org/wiki/SHMEM)-like communication primitives to Triton, so that the overlap of computation and communication can be programmed tile by tile.

These methods fuse one matrix multiplication at a time, keep the intermediate activations $$H$$ in global memory, and reduce on the GPUs. Megakernels fuse a whole forward pass into one kernel. The [low-latency megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) runs on a single GPU. [MPK](https://arxiv.org/abs/2512.22219), built on the [Mirage](https://arxiv.org/abs/2405.05751) superoptimizer, generates megakernels that also include the communication between GPUs. In-network all-reduces are available in NCCL, but only as separate collectives on full tensors. The fused kernel combines the three: fusion of both layers, per-tile communication inside the kernel, and in-network reduction.

## Future work

The fused kernel benefits most from network hardware with low latency for many small, asynchronous messages sent from inside GPU kernels, such as tiles of 32 KiB. Current network hardware is optimized for large, synchronous collectives. The same method could extend to attention, which also chains two matrix multiplications, and to expert parallelism in Mixture-of-Experts layers.

This post summarizes the paper *Towards Efficient LLM Inference via Full-Stack Fusion of Computation, Communication, and In-Network Computing*.
