---
layout: post
title:  "Kernel Fusion for Megatron-LM Tensor Parallelism and MoE Expert Sharding"
categories: ["Efficient Inference"]
tags: ["LLM", "kernel fusion", "tensor parallelism", "Megatron-LM", "in-network computing", "collectives", "Mixture-of-Experts", "expert sharding"]
---

This post covers a kernel that fuses the computation and the communication of a [feedforward network](https://en.wikipedia.org/wiki/Feedforward_neural_network) (FFN) distributed with [Megatron-LM tensor parallelism]({{ site.baseurl }}{% post_url 2023-09-02-tensor-parallelism %}) into a single GPU kernel. It then applies the same kernel to the experts of a Mixture-of-Experts layer, with expert sharding.

## The problem: kernels and graph breaks

A large language model (LLM) is executed as a long sequence of GPU kernels (usually one or more per layer). Each kernel loads its inputs from the global memory of the GPU ([HBM](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)) into the on-chip memory of the Streaming Multiprocessors (SMs), and writes its outputs back. **Kernel fusion** merges consecutive kernels into one, so that intermediate results stay in on-chip memory and are consumed directly by the next operation. In practice, `torch.compile` does this automatically on a single GPU.

However, on several GPUs, fusion stops at every [collective communication](https://en.wikipedia.org/wiki/Collective_operation) call, as a collective needs the results of all GPUs. The compiler then inserts a **graph break**: the program is split into subgraphs, separated by large, synchronous collectives (e.g. an all-reduce). These collectives need large communication buffers in global memory, and the GPUs wait for them to finish before computing again.

## Megatron-LM tensor parallelism

The FFN has two linear layers, with an activation in between. For an input $$X \in \mathbb{R}^{M \times N}$$ of $$M$$ tokens, its output is computed as:

$$
H = \text{activation}(X W_1 + b_1), \quad Y = \text{activation}(H W_2 + b_2)
$$

where $$W_1 \in \mathbb{R}^{N \times O}$$ and $$W_2 \in \mathbb{R}^{O \times P}$$ are the weights and $$b_1$$ and $$b_2$$ the biases of the two layers, $$O$$ is the hidden dimension, and the second activation is optional. On $$G$$ GPUs, Megatron-LM splits $$W_1$$ column-wise and $$W_2$$ row-wise. Therefore, each GPU $$g$$ computes a slice $$H^{(g)}$$ of the hidden activations and a partial output $$H^{(g)} W_2^{(g)}$$. An all-reduce sums the partial outputs of all GPUs, $$H W_2 = \sum_{g=1}^{G} H^{(g)} W_2^{(g)}$$, before the bias $$b_2$$ and the second activation are applied. This is the only communication step per pair of layers. The following picture illustrates this execution on 2 GPUs:

{: .picture}
<img width="96.7%" height="96.7%" src="{{ site.assets }}/inference/imgs/tp_megatron_lm.svg"/>

{: .caption}
Megatron-LM tensor parallelism of an FFN on 2 GPUs (A in green, B in blue). The bias and activation are omitted.

A regular implementation launches 7 kernels for this FFN: a matrix multiplication, a bias addition and an activation per layer, plus the all-reduce. Each kernel goes through global memory, as displayed at the top of the timeline below. The method in this post runs the whole FFN as one kernel, with a single load and a single store to global memory (bottom). It is built in three steps, presented next.

{: .picture}
<img width="95.5%" height="95.5%" src="{{ site.assets }}/inference/imgs/tp_timeline.svg"/>

{: .caption}
Execution timeline of an FFN with Megatron-LM tensor parallelism. **Top:** the regular execution, with 7 kernels that load from (arrow up) and store to (arrow down) global memory. **Bottom:** the fused kernel, with several small, overlapping all-reduce steps performed in the network switch.

## Step 1: fusing the two linear layers

Two linear layers without a nonlinearity in between collapse into a single one. However, with an activation in between, the usual approach materializes (i.e. stores) the full intermediate tensor $$H$$ in global memory, and a second kernel reads it back.

A matrix multiplication on a GPU is computed in *tiles*: each [thread block](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)) runs on one SM and computes one small submatrix of the output. On each GPU $$g$$, the fused kernel keeps this tiling for the first layer (below, $$H$$ and $$W_2$$ stand for the local slices $$H^{(g)}$$ and $$W_2^{(g)}$$). However, as soon as a tile $$H_{i,j}$$ is computed (with its bias and activation), it is multiplied by every tile in row $$j$$ of $$W_2$$, and the partial products are added to the tiles in row $$i$$ of the output $$Y$$. This follows from the product of [block matrices](https://en.wikipedia.org/wiki/Block_matrix), $$(H W_2)_{i,k} = \sum_j H_{i,j} (W_2)_{j,k}$$. When all tiles of $$H$$ are processed, $$Y$$ holds the full product $$H W_2$$, and $$H$$ was never stored in global memory, as displayed below:

{: .picture}
<img width="97.8%" height="97.8%" src="{{ site.assets }}/inference/imgs/tp_matmuls.svg"/>

{: .caption}
**Left:** the regular execution computes all tiles of $$H$$, stores $$H$$ in global memory, and a second kernel computes $$Y$$. **Right:** the fused kernel multiplies each tile of $$H$$ by a row of tiles of $$W_2$$ and accumulates the results in a row of tiles of $$Y$$.

Many blocks add to the same tile of $$Y$$ at the same time, so the additions must be atomic, and blocking atomic additions quickly become a bottleneck. Instead, the kernel uses the *asynchronous* tile reductions of the GPU, in a *ping-pong* pattern: a block computes the product of the next tile while the reduction of the previous tile is still in flight. This hides most of the cost of the accumulation.

A per-tile counter tracks how many blocks have contributed to each output tile; the last contributor marks the tile as *ready*. Note that a ready tile is final, and can be sent to the other GPUs right away, without waiting for the rest of $$Y$$. Communication can therefore happen per tile, with buffers of 32 KiB instead of buffers of several gigabytes.

## Step 2: overlapping computation and communication with block specialization

How to communicate the ready tiles? If every block computes and communicates, the faster blocks must [spin-wait](https://en.wikipedia.org/wiki/Busy_waiting) until all tiles of a region are ready before reducing it across GPUs. **Block specialization** splits the blocks of the kernel into two groups:

- **compute blocks** run the matrix multiplications of step 1 and mark tiles as ready;
- **communication blocks** each own a *band* (a contiguous range of output tiles). A communication block polls the ready flags of its band, all-reduces the band across GPUs as soon as it is complete, and applies the final bias and activation.

Compute blocks never wait for communication, and communication blocks never compute matrix multiplications. The matrix multiplications run on fewer SMs (which is only marginally slower on a GPU with many SMs), while the communication is hidden behind the computation.

As a side note, the same idea appears at other levels. Inside a block, [FlashAttention-3](https://arxiv.org/abs/2407.08608) uses *warp specialization* (some warps load data from global memory while the others compute). Across the GPU, [DeepSeek-V3](https://arxiv.org/abs/2412.19437) dedicates a fixed part of the SMs to the communication of its Mixture-of-Experts layers. Here, the specialized blocks run the all-reduce of the tensor-parallel FFN, one band at a time, as in the following example:

{: .picture}
<img width="64.4%" height="64.4%" src="{{ site.assets }}/inference/imgs/tp_block_specialization.svg"/>

{: .caption}
An example on a GPU with 6 SMs. **(a)** Without block specialization, all SMs compute and then spin-wait before each band is all-reduced. **(b)** With block specialization, SM1 and SM2 all-reduce each band as soon as SM3 to SM6 have computed it.

## Step 3: all-reduce with in-network computing

An all-reduce from the communication library is launched from the CPU, as a separate kernel, on full tensors, and cannot be called from inside a running kernel, on single tiles. The fused kernel uses its own all-reduce instead, issued directly by the GPU threads and executed by the network switch that connects the GPUs. This is called **In-Network Computing** (INC): data is summed as it traverses the interconnect, not by the GPUs. The switch offers three operations on memory addresses that are mapped on all GPUs:

- a **load-reduce**, which reads a value from all GPUs and returns their sum (all-to-one);
- a **multicast store**, which writes one value to all GPUs (one-to-all);
- an **in-place reduction**, which adds one value to the same address on all GPUs.

The all-reduce of a band is then a reduce-scatter followed by an all-gather: each GPU first load-reduces its $$1/G$$ share of the band, and then multicast-stores the result to all GPUs. However, a GPU cannot tell by itself when the other GPUs have finished their stores, so each GPU adds 1 to a shared flag with an in-place reduction. When the flag reaches $$G$$, all GPUs know that the band is complete. This acts as a lightweight [barrier](https://en.wikipedia.org/wiki/Barrier_(computer_science)) that runs entirely on the GPUs, without the CPU.

## Results

The fused kernel was compared to a tuned, non-fused implementation (the baseline) with the all-reduce of the communication library, on the FFNs of Llama, DeepSeek, GPT, Mistral and Qwen models, on 8 GPUs in one node, in [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format). It reduces the latency by 22% to 62%, halves the memory of the activations, and shrinks the communication buffer from gigabytes to 32 KiB. Seven of these FFNs are compared below:

{: .picture}
<img width="55%" height="55%" src="{{ site.assets }}/inference/imgs/tp_results.svg"/>

{: .caption}
Latency of the regular (baseline) and fused FFNs, sorted by improvement.

The gains follow the size of the FFN:

- **Small FFNs** (DeepSeek V3 experts, Llama 3 8B) gain the most, 46% to 62%. Their runtime is short and dominated by fixed costs, i.e. the extra kernels, the stores of $$H$$ and the all-reduce, which the fused kernel removes or hides.
- **Large FFNs** (Llama 3 70B, DeepSeek V3 dense, Llama 4 Scout shared) gain 29% to 38%. Their runtime is dominated by the two large matrix multiplications: the absolute savings are larger, up to 122 ms, but the relative gain is smaller.

An ablation study shows that all three steps are needed. Without fusion, the in-network all-reduce runs on full tensors, and the FFN is slower than the baseline; without block specialization, the communication is *exposed* (i.e. not hidden), and the kernel is slower than the baseline in most cases. Finally, when in-network computing is replaced by a GPU-side [shmem](https://en.wikipedia.org/wiki/SHMEM) library, the kernel is faster than the baseline but slower than the full method.

## Expert sharding

In a [Mixture-of-Experts (MoE)]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) layer, each expert is a small FFN, with an input matrix $$W_i$$ and an output matrix $$W_o$$ (the $$W_1$$ and $$W_2$$ above). With **expert parallelism**, each GPU holds a few whole experts and processes only the tokens routed to them. However, the router rarely spreads the tokens evenly, so the GPU with the most tokens finishes last, and the other GPUs wait for it.

[Expert sharding](https://arxiv.org/abs/2503.08467) splits every expert across all GPUs instead, exactly as Megatron-LM splits the FFN. Each GPU holds a $$1/G$$ share of the columns of $$W_i$$ and the matching rows of $$W_o$$ of every expert, and processes all the tokens with these slices. The partial outputs of all GPUs are then summed. Each expert is then an FFN with Megatron-LM tensor parallelism, and runs with the fused kernel of this post, as pictured below:

{: .picture}
<img width="59.5%" height="59.5%" src="{{ site.assets }}/inference/imgs/tp_expert_sharding.svg"/>

{: .caption}
One expert, with $$W_i$$ in green and $$W_o$$ in red. The activation is omitted. **(a)** Without parallelism, one GPU holds the whole expert. **(b)** With expert sharding, GPU A holds the first half of the columns of $$W_i$$ and of the rows of $$W_o$$, and GPU B the second half. The dashed slices are on the other GPU. Both GPUs process all the tokens $$x$$, and a sum-reduce adds their partial outputs $$y_A$$ and $$y_B$$. Inspired by Figure 2 of [Accelerating MoE Model Inference with Expert Sharding](https://arxiv.org/abs/2503.08467).

Compared to expert parallelism, expert sharding adds communication, as every GPU needs all the tokens, and the partial outputs must be all-reduced across GPUs. The fused kernel makes this communication efficient: the all-reduce runs one band at a time on dedicated blocks, overlaps with the computation, and is executed by the network switch. The results above include two experts split across 8 GPUs in this way, where the fused kernel reduces the latency by 55% for an expert of DeepSeek V3, and by 45% for an expert of Llama 4 Scout.

Most importantly, expert sharding fixes the load imbalance of the experts: every GPU holds the same share of every expert and processes all the tokens, so all GPUs do the same amount of work, however the tokens are routed. Therefore, no GPU waits for another, no tokens are dropped, and popular experts do not need to be replicated.

## Related work

Other systems also overlap the computation and the communication of tensor parallelism inside kernels:

- [CoCoNet](https://arxiv.org/abs/2105.05720) is a compiler that fuses and overlaps a matrix multiplication with the collective that follows it;
- [Flux](https://arxiv.org/abs/2406.06858) splits the Megatron-LM matrix multiplications and their collectives into tiles, and fuses them into one kernel per layer, where each tile is sent as soon as it is computed;
- [Triton-distributed](https://arxiv.org/abs/2504.19442) and [TileLink](https://arxiv.org/abs/2503.20313) add shmem-like communication primitives to Triton, so that the overlap of computation and communication can be programmed tile by tile.

These methods fuse one matrix multiplication at a time, and keep the intermediate activations $$H$$ in global memory. **Megakernels** fuse a whole forward pass, on a single GPU (e.g. the [low-latency megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)) or with the communication between GPUs (e.g. [MPK](https://arxiv.org/abs/2512.22219), built on the [Mirage](https://arxiv.org/abs/2405.05751) superoptimizer). Finally, in-network all-reduces are available in the communication library, but only as separate collectives on full tensors. The method in this post combines the three: fusion of both layers, per-tile communication inside the kernel, and in-network reduction.

## Final remarks

The main lesson is that computation, communication and the network switch should be optimized together: fusing the FFN into one kernel makes per-tile communication possible, and per-tile communication can be overlapped with the computation and executed in the switch. This also suggests a different design for network hardware, with support for many small, asynchronous messages sent from inside GPU kernels, instead of large, synchronous collectives. The same ideas could be applied to attention, which also chains two matrix multiplications, and to expert parallelism in Mixture-of-Experts.

This post summarizes the paper *Towards Efficient LLM Inference via Full-Stack Fusion of Computation, Communication, and In-Network Computing*.
