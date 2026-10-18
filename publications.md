---
layout: post
title: Publications bookmark
permalink: /publications/
---

{::options parse_block_html="true" /}

Summaries of publications on HPC and machine learning that the author came across and found interesting, newest first. The list is continuously updated, and each title opens its summary.

<table class="label-table pub-table">
<tr>
<td><span class="post-label">2026</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2606.19348" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek-V4 is a preview of the next MoE series of DeepSeek, with two models: **DeepSeek-V4-Pro**, with 1.6T parameters, 49B of them activated, pre-trained on 33T tokens, and **DeepSeek-V4-Flash**, with 284B parameters, 13B activated, pre-trained on 32T tokens. Both models have a native context of one million tokens, and most of the changes in the paper aim at making such long contexts affordable. The models keep the DeepSeekMoE framework and the MTP modules of DeepSeek-V3, with small changes to the MoE, among them: the affinity function is a Sqrt(Softplus) instead of a sigmoid, the number of nodes that a token can be routed to is no longer limited, and hash-routed MoE layers replace the first dense FFN layers. Three larger changes are new:

1. **Hybrid attention**, with CSA and HCA layers interleaved. Both kinds of layers compress the KV cache along the sequence, so that each query attends to far fewer entries. *Compressed Sparse Attention* (CSA) compresses the KV entries of every $$m$$ tokens into one entry, a learned softmax-weighted combination over overlapping windows. It then applies DeepSeek Sparse Attention to the compressed entries: the lightning indexer selects the top-k compressed entries for each query, and a small sliding window of recent uncompressed tokens adds back the local detail. *Heavily Compressed Attention* (HCA) compresses every $$m' \gg m$$ tokens into one entry, a much stronger compression, and attends densely to all the compressed entries. The attention also uses a shared-KV layout, as in MQA, with a grouped output projection, partial RoPE and attention sinks.
2. **mHC** (Manifold-Constrained Hyper-Connections, see its entry) instead of plain residual connections. The residual stream is widened into $$n_{hc}$$ parallel streams, and their mixing matrix is projected onto the [doubly stochastic matrices](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix) with [Sinkhorn-Knopp](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) iterations, which keeps the signal propagation stable.
3. The **Muon** optimizer, for faster convergence and more stable training.

These changes pay off at long contexts. At 1M tokens, V4-Pro needs 27% of the inference FLOPs per token of DeepSeek-V3.2 and 10% of its KV cache, and V4-Flash needs 10% and 7%. The weights of the routed experts are stored in FP4, and post-training includes FP4 quantization-aware training, so that the model learns to work with the 4-bit weights.

The paper also describes the infrastructure behind the models: a single fused MoE mega-kernel that overlaps computation, communication and memory access in expert parallelism; kernels written in TileLang; batch-invariant and deterministic kernels, whose results do not depend on how the requests are batched, so that training and inference are bitwise reproducible; two-stage context parallelism for the compressed attention; and a heterogeneous KV cache, with storage on disk to reuse shared prefixes. Post-training first trains one specialist per domain, with SFT and then RL with GRPO, and then merges the specialists into one model with on-policy distillation. In its maximum-reasoning mode, V4-Pro-Max, the model is reported as the best open model at its release: it is ahead of GPT-5.2 and Gemini-3.0-Pro on standard reasoning benchmarks, and slightly behind GPT-5.4 and Gemini-3.1-Pro.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2604.12374" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This technical report presents Nemotron 3 Super, a hybrid MoE model with 120B parameters, 12B of them active, which combines [Mamba-2](https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)) layers with attention layers. It is the first model of the Nemotron 3 family (see the Nemotron 3 entry) that is **pre-trained in NVFP4**, that uses **LatentMoE** experts, and that includes **MTP layers**, which predict several tokens at once and serve as the draft of a native speculative decoding. The model was pre-trained on 25T tokens, and then post-trained with SFT and RL, with a strong focus on agentic tasks.

The final model supports a context of up to 1M tokens. On common benchmarks, it is about as accurate as GPT-OSS-120B and Qwen3.5-122B, while its inference throughput is up to 2.2× and 7.5× higher, respectively. The datasets are open, and so are the base, post-trained and quantized checkpoints, in BF16, FP8 and NVFP4. Nemotron 3 Super is also the large-scale test of the LatentMoE and NVFP4 entries: both the latent experts and the 4-bit pre-training hold up on a 120B model trained on 25T tokens.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling</span><span class="pub-meta">, Zadouri, ..., Tri Dao (Princeton et al.) (MLSys 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2603.05451" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

FlashAttention-4 (FA4) redesigns the attention kernel for NVIDIA Blackwell GPUs (B200, GB200), to deal with what the authors call **asymmetric hardware scaling**. From Hopper to Blackwell, the dense BF16 throughput of the tensor cores grew from about 1 to about 2.25 petaFLOPS, but the units that do the rest of the work did not keep up: the exponential unit (MUFU.EX2), which computes the softmax, and the bandwidth of the shared memory. As a result, the bottleneck moves. In the earlier FlashAttention versions, the softmax was a small step between the two matrix multiplications of attention; on Blackwell, it becomes the bottleneck, and it has to be pipelined carefully. The aim of FA4 is therefore to keep the fast tensor cores busy, by overlapping the matrix multiplications with the softmax and the memory operations, which are now relatively slow.

The design has several parts. New software pipelines, for the forward and the backward pass, use the fully asynchronous MMA instructions and the larger tiles of Blackwell to overlap the tensor cores, the exponentials and the memory operations. To relieve the exponential unit, the forward pass computes a part of the exponentials in software, with a polynomial approximation on the FMA units. It also rescales the online softmax only when the running maximum grows by more than a threshold, which skips most of the rescaling steps and still gives the exact result. The backward pass is limited by the shared-memory traffic instead, so FA4 keeps intermediate results in the new tensor memory (TMEM) of Blackwell, and uses the new 2-CTA MMA mode, which reduces the shared-memory traffic further and halves the number of global atomic additions. Finally, the kernel is written entirely in **CuTe-DSL**, in Python. It compiles 20 to 30 times faster than C++ templates, with the same expressivity, which makes new attention variants easier to prototype.

FA4 reaches up to 1613 TFLOPs/s on a B200 in BF16, 71% of the peak (the blog post of the release quotes 1605). In the forward pass, it is up to 1.3× faster than cuDNN 9.13 and 2.7× faster than Triton, and in the backward pass, it is faster than all the baselines at long sequence lengths. FlashAttention-3 runs only on Hopper, so it is not a baseline. Several techniques of FA4 are also in cuDNN, from versions 9.13 and 9.14.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism</span><span class="pub-meta">, Arizona State University (ICML 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2602.04870" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This follow-up to LatentMoE (see its entry) addresses three limitations of expert parallelism (EP). First, the communication volume and the latency of the all-to-all grow linearly with the number of activated experts $$k$$, because each token is sent to each of its $$k$$ experts. Second, the load of the experts is imbalanced, so the all-to-all waits for the longest queue. Third, the traffic depends on the routing, so it is not known in advance (it is not deterministic), and an extra all-to-all has to exchange the metadata first. LatentMoE sends fewer bytes per token, but it still communicates *after* the routing, so the imbalance and the non-determinism remain.

**Multi-Head LatentMoE** multiplies each token by a learned $$d \times d$$ matrix and splits the result into $$N_h$$ sub-tokens. Each sub-token goes through an independent small MoE, with its own router and experts, and the outputs are concatenated and projected back. **Head Parallel (HP)** uses this structure to move the all-to-all *before* the routing: each GPU owns a subset of the heads, receives all the sub-tokens of these heads, and runs the routing and the experts locally. This removes the three limitations at once. Each token is sent exactly once, so the communication no longer grows with $$k$$. Every GPU sends and receives the same amount of data, so the traffic is balanced, no metadata has to be exchanged, and hot experts cannot cause out-of-memory errors. And HP combines with EP to scale beyond $$N_h$$ GPUs. So that the memory traffic does not grow $$N_h$$ times, the authors add exact IO-aware kernels: an online top-k router in Triton that keeps the scores in SRAM, and the expert computation written as block-sparse attention on top of FlexAttention.

At a small scale (0.2B active and up to 4.2B total parameters, trained on 10B tokens of FineWeb-Edu), Multi-Head LatentMoE with HP trains up to 1.61× faster than a standard MoE with EP, at the same quality. At $$k = 4$$, it sends a quarter of the inter-GPU traffic, since each token travels once instead of four times. With experts twice as fine-grained, it reaches a higher accuracy and is still 1.11× faster.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">In-Network Collective Operations: Game Changer or Challenge for AI Workloads?</span><span class="pub-meta">, Torsten Hoefler et al. (ETH Zürich, Microsoft, AMD, Broadcom, HPE) (IEEE Computer 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2601.19132" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Collective operations, such as the all-reduce of data-parallel training, can be offloaded to the network: to the network interfaces of the hosts (edge-INC), or to the switches (core-INC, as in NVIDIA SHARP). In the switches, the reduction can halve the traffic of an all-reduce. The paper names six obstacles, however, among them low-precision data types, sparse reductions, bitwise reproducible results, and encryption. [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) also limits the gain, since only the communication becomes faster. In the example of the paper, INC shortens an all-reduce of 8 GiB from 352 ms to 151 ms, but when the communication that is not overlapped takes 20% of the run time, the whole application is only about 11% faster. The authors therefore expect a gradual adoption over the next years.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2601.18089" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Which dimension of an MoE layer should shrink to make it cheaper? The paper answers with a [roofline](https://en.wikipedia.org/wiki/Roofline_model) analysis of the experts. At the small batch sizes of latency-critical serving, the experts are bound by the HBM bandwidth that loads their weights, and in throughput-oriented serving, they are bound by the all-to-all communication. Both costs scale with the hidden dimension $$d$$ of the model, while the number of active experts and the width of the expert FFNs set the nonlinear capacity of the model. So $$d$$ is the dimension to shrink.

LatentMoE therefore projects each token from the hidden dimension $$d$$ to a smaller latent dimension $$\ell$$ before it is dispatched to the routed experts (see the figure). The router still computes its gating scores from the full $$d$$-dimensional token: only the dispatch and combine traffic and the routed experts are in the latent space. After the combine all-to-all, the weighted sum of the outputs of the routed experts is projected back from $$\ell$$ to $$d$$, and added to the output of the shared experts, which work in the full hidden dimension. All the routed experts share a single down-projection and a single up-projection, while the weights of each expert shrink by $$d/\ell$$, and so do the bytes that each expert loads from memory and that each token sends over the network.

The savings pay for *more and finer experts*: LatentMoE multiplies both the number of experts and the number of active experts per token by about the same factor $$d/\ell$$, which keeps the cost about the same. A standard MoE with 128 experts and top-6 routing scores 48.30% on MMLU-Pro. With a 4× compression, a LatentMoE with 512 experts and top-22 routing, as reported in the Nemotron 3 white paper, scores 52.87% on MMLU-Pro and 55.14% on code (+3.19%), with similar parameter and FLOP counts (hybrid models with 8B active and 73B total parameters, trained on 1T tokens). Its measured throughput per GPU stays within about 6% of the baseline at higher concurrency. A design-space exploration up to 95B parameters, over 1T tokens, shows that LatentMoE beats a standard MoE in accuracy per FLOP and per parameter, and Nemotron 3 Super and Ultra use it.

{: .picture}
<img loading="lazy" width="85.9%" height="85.9%" src="{{ site.assets }}/publications/latentmoe.svg"/>

{: .caption}
<b>(a)</b> Standard MoE architecture. <b>(b)</b> LatentMoE architecture. In LatentMoE, tokens are projected from the model hidden dimension $$d$$ into a smaller latent dimension $$\ell$$ for expert routing and computation, which reduces routed parameter loads and all-to-all traffic by a factor of $$d/\ell$$ (typically about 4×). LatentMoE uses this efficiency to increase both the total number of experts and the top-$$K$$ active experts per token by the same factor $$d/\ell$$, which improves accuracy per byte while keeping overall inference cost approximately constant. The selected experts and their router scores are orange; the picture uses $$d/\ell = 2$$ (4 experts with top-2 become 8 experts with top-4). Adapted from: [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2025</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">mHC: Manifold-Constrained Hyper-Connections</span><span class="pub-meta">, DeepSeek-AI (ICML 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2512.24880" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Hyper-Connections (HC) widen the residual stream into $$n$$ parallel streams, mixed by learnable matrices. This makes the width of the residual stream independent of the width of the layers, at a small compute cost, and it improves the accuracy. The unconstrained mixing has a price, however: it breaks the identity mapping that makes deep residual networks trainable. In a 27B model, the signals grow from layer to layer and the training loss spikes. HC also adds a large memory-access overhead.

mHC restores the identity mapping by constraining the residual mixing matrix of each layer to the doubly stochastic matrices, which form the [Birkhoff polytope](https://en.wikipedia.org/wiki/Birkhoff_polytope): non-negative matrices whose rows and columns each sum to 1. The projection onto this set uses Sinkhorn-Knopp iterations, which normalize the rows and the columns in turn, and sigmoids keep the read-in and write-out mappings non-negative and bounded. Why does this constraint help? A doubly stochastic matrix has a spectral norm of 1, so the mixing cannot amplify the signal, and a product of doubly stochastic matrices is again doubly stochastic, so the signal stays stable at any depth. With $$n = 1$$, the only such matrix is the identity, as in a standard residual connection. On the systems side, kernel fusion, recomputation, and overlap with the communication of DualPipe reduce the cost of a 4× wider residual stream ($$n = 4$$) to about 6.7% more training time.

In pre-training experiments on MoE models from 3B to 27B parameters, mHC trains stably where HC spikes, and it beats the plain residual baseline at all the compute and token budgets tested. DeepSeek-V4 uses it (see its entry).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs</span><span class="pub-meta">, CMU et al. (OSDI 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2512.22219" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

MPK automates the megakernel of the Hazy Research Llama-1B post (see its entry), and comes from the group behind Mirage. It is the first compiler and runtime that turns multi-GPU model inference into a single persistent megakernel automatically, where earlier megakernels were written by hand. Its main abstraction is an **SM-level task graph**, which tracks the data dependencies per streaming multiprocessor (SM) instead of per kernel. This finer grain allows optimizations that the usual model of one kernel per operator cannot express, such as software pipelining across operators, where the next operator loads its data while the current one still computes, and a fine-grained overlap of computation with the communication between GPUs.

The compiler lowers a tensor program to an optimized SM-level task graph and generates CUDA code for each task. A parallel runtime inside the kernel then runs the tasks in one persistent kernel, with a decentralized scheduling across the SMs, so that no kernel launches or host round trips separate the operators. MPK is open source, in the Mirage repository, and its end-to-end inference latency is up to 1.7× lower than that of existing LLM serving systems that launch one kernel per operator.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2512.02556" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This is the full report behind the DeepSeek-V3.2-Exp model card (see its entry). The architecture is the same as that of V3.2-Exp: the only change from DeepSeek-V3.1-Terminus is **DeepSeek Sparse Attention (DSA)**, added through continued training. In DSA, a *lightning indexer* scores each preceding token for the current query. It has only a few heads, uses ReLU activations and can run in FP8, so it is cheap. A fine-grained selection step then keeps only the top-k key-value entries ($$k = 2048$$) for the main MLA attention. For a context of $$L$$ tokens, the cost of the core attention therefore drops from $$O(L^2)$$ to $$O(Lk)$$. The indexer itself is still quadratic, but much cheaper than the attention. The training has two stages: the indexer is first trained alone, with a KL loss that makes it match the dense attention distribution, while the main model is frozen, and then the whole model is trained with sparse attention.

The rest of the report is about post-training. It describes a scaled and stabilized RL recipe based on GRPO, with a post-training compute budget above 10% of the pre-training cost, and a large pipeline that synthesizes agentic tasks, with over 1,800 environments and 85,000 complex prompts. V3.2 is also the first DeepSeek model that integrates thinking directly into tool use. DeepSeek-V3.2 performs comparably to GPT-5, and its high-compute variant, **DeepSeek-V3.2-Speciale**, which reasons longer and does not support tool use, reaches gold-medal level at IMO 2025, CMO 2025, IOI 2025 and the ICPC World Finals 2025. It matches Gemini-3.0-Pro in reasoning, although it needs more tokens to get there.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Kimi Linear: An Expressive, Efficient Attention Architecture</span><span class="pub-meta">, Moonshot AI (Kimi Team)</span><a class="pub-link" href="https://arxiv.org/abs/2510.26692" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Kimi Linear is a hybrid linear-attention architecture. In fair comparisons, it *outperforms* full attention in short-context, long-context and RL scaling settings, which the authors present as a first. Its core is **Kimi Delta Attention (KDA)**, a linear attention with a fixed-size recurrent state. KDA extends Gated DeltaNet with a finer-grained gating: instead of one forget gate per head, each channel (feature dimension) has its own decay rate, so the limited memory of the state is used more selectively. For hardware efficiency, the state transition of KDA uses a specialized Diagonal-Plus-Low-Rank (DPLR) form, with a chunkwise-parallel algorithm that needs much less computation than a general DPLR form and stays consistent with the classical [delta rule](https://en.wikipedia.org/wiki/Delta_rule).

The model interleaves KDA layers and full-attention MLA layers in a fixed 3:1 ratio, with 48B total and 3B activated parameters. With the same training recipe, it beats a full-MLA baseline on all the tasks evaluated. Since only one layer in four keeps a KV cache that grows with the context, the KV cache is up to 75% smaller, and the decoding throughput is up to 6× higher at a context of 1M tokens. The KDA kernels, a vLLM implementation, and the pre-trained and instruction-tuned checkpoints are open.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Pretraining Large Language Models with NVFP4</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2509.25149" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper gives a recipe to pre-train LLMs in 4-bit floating point with **NVFP4**, the microscaling format of Blackwell GPUs. NVFP4 stores FP4 (E2M1) values in blocks of 16 elements, and each block has an FP8 (E4M3) scale, on top of an FP32 scale per tensor. MXFP4, in comparison, uses blocks of 32 elements with power-of-two (E8M0) scales, so the smaller blocks and the more precise scales of NVFP4 capture the local dynamic range much better. The motivation is speed: FP4 matrix multiplications run at 2 to 3 times the FP8 throughput on Blackwell (GB200, GB300). Naive 4-bit training, however, diverges or loses accuracy over long training runs.

The method combines four parts:

1. **random [Hadamard transforms](https://en.wikipedia.org/wiki/Hadamard_transform)** on the inputs of the weight-gradient matrix multiplications, which spread the outliers of a block into a distribution closer to a Gaussian;
2. **2D block scaling of the weights** (16×16 blocks), so that the forward and the backward pass see the same quantized weights: a square block stays the same block when the weight matrix is transposed for the backward pass;
3. **[stochastic rounding](https://en.wikipedia.org/wiki/Rounding#Stochastic_rounding)** of the gradients, which avoids the bias of round-to-nearest;
4. a small fraction of numerically sensitive linear layers kept in higher precision.

The recipe trains a 12B hybrid Mamba-Transformer on 10T tokens, the longest publicly documented 4-bit training run at the time. The loss stays within about 1% relative error of the FP8 baseline during the stable phase of the learning rate, and slightly above 1.5% late in its decay, and the downstream accuracies match those of FP8. On an 8B model, MXFP4 needs about 36% more tokens to reach the loss that NVFP4 reaches at 1T tokens. Transformer Engine supports the recipe, and Nemotron 3 Super was pre-trained with it (see its entry).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures</span><span class="pub-meta">, DeepSeek-AI (ISCA 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2505.09343" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This ISCA 2025 paper (industry track) looks back at the co-design of the hardware and the model in DeepSeek-V3 and R1. It complements the DeepSeek-V3 report (see its entry), and explains two things: how a 671B model could be trained on only 2,048 H800 GPUs, and what hardware should change for future models.

On the model side, the paper quantifies the design choices. MLA reduces the KV cache to about 70 KB per token, against 327 KB for Qwen-2.5 72B and 516 KB for LLaMA-3.1 405B, which both use GQA. The MoE design needs about 250 GFLOPs of training compute per token, against about 2,448 for the dense LLaMA-3.1 405B. The other choices are FP8 mixed-precision training, and MTP, which speeds up inference through speculative decoding. On the infrastructure side, the paper describes the **multi-plane two-layer [fat-tree](https://en.wikipedia.org/wiki/Fat_tree)** network, where the NIC of each GPU is on its own network plane, with eight planes per node. This network lowers the cost of the cluster network while it scales to thousands of GPUs. Expert parallelism and node-limited routing are shaped around the bandwidth asymmetry of the H800, whose NVLink is about four times faster than its InfiniBand: each token is routed to at most four nodes, and a token bound for several experts of the same node crosses InfiniBand only once, before NVLink forwards it to the other GPUs of the node.

The second half of the paper lists what future AI hardware needs, based on the bottlenecks that DeepSeek hit: more precise low-precision compute units (for example, a higher FP8 accumulation precision and native support for fine-grained block scaling), scale-up and scale-out networks that converge into one, and low-latency communication fabrics. It explains *why* the communication-overlap methods of other entries were needed on bandwidth-limited hardware.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</span><span class="pub-meta">, DeepSeek-AI, Peking University & University of Washington (ACL 2025 Best Paper)</span><a class="pub-link" href="https://arxiv.org/abs/2502.11089" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Why do most sparse-attention methods fail in practice? The NSA paper gives two reasons. First, their theoretical savings do not become wall-clock speedups: their memory accesses are scattered token by token, they fit poorly with GQA and MQA, where heads share the KV entries, or they speed up only the prefill or only the decode. Second, they are applied after training, to models trained with dense attention, so the sparsity pattern is never learned end to end. NSA is designed to avoid both problems. Each query attends through three branches, whose outputs are combined by learned gates:

1. **compressed** attention over coarse summaries of token blocks (blocks of 32 tokens with a stride of 16, compressed by a learned MLP);
2. **selected** attention over the most relevant fine-grained blocks (16 blocks of 64 tokens), chosen from the attention scores of the compression branch, so that the selection costs almost nothing;
3. a **sliding window** over the last 512 tokens, for the local context.

The kernel is hardware-aligned. All the heads of a GQA group share the same block selection, so each group loads contiguous KV blocks into SRAM once, which keeps the arithmetic intensity high on the Tensor Cores, in the forward pass, the backward pass and the decoding. A 27B-parameter MoE model (about 3B active), pre-trained from scratch with NSA on 260B tokens, matches or beats the full-attention baseline on general benchmarks, long-context retrieval and chain-of-thought reasoning. At a context of 64K tokens, it is up to 11.6× faster in decoding, 9.0× in the forward pass and 6.0× in the backward pass. Block compression and learned selection appear again in DeepSeek Sparse Attention (DeepSeek-V3.2) and in the CSA and HCA layers of DeepSeek-V4.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving</span><span class="pub-meta">, University of Washington et al. (MLSys 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2501.01005" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

FlashInfer is the attention kernel library under many serving systems, and it is integrated into SGLang, vLLM and MLC-Engine. It handles three properties of serving workloads that a single FlashAttention kernel does not cover. The first is the **heterogeneity of the KV cache**: paged, prefix-shared and sparse KV layouts are all stored in one **block-sparse format**, with composable formats for shared prefixes, so the same kernels serve all the layouts, and the KV entries of a shared prefix are not loaded again for each request. The second is **customization**: an attention template, with custom masks or logit transforms for example, is JIT-compiled into optimized block-sparse kernels, so new attention variants need no hand-written CUDA. The third is **dynamism**: a load-balanced scheduler, inspired by Stream-K, splits the work of variable-length requests across the SMs, and stays compatible with CUDA Graphs, which need a static launch configuration.

On an LLM serving benchmark, FlashInfer lowers the inter-token latency by 29% to 69% compared with compiler backends. It also lowers the latency of long-context inference by 28% to 30%, and speeds up LLM serving with parallel generation by 13% to 17%.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">NVIDIA Nemotron 3: Efficient and Open Intelligence</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2512.20856" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Nemotron 3 is a family of open models from NVIDIA, in three sizes, Nano, Super and Ultra, with a context of up to 1M tokens. The white paper combines four efficiency techniques:

1. **LatentMoE** (Super and Ultra), see its entry.
2. A **hybrid Mamba-Transformer MoE** architecture, for inference efficiency. During generation, a self-attention layer attends over a KV cache that grows linearly with the sequence, while a Mamba-2 layer only stores a state of constant size. So instead of interleaving the MoE layers with expensive self-attention layers, Nemotron 3 interleaves them mostly with cheaper Mamba-2 layers, and keeps only a few attention layers. These layers use no RoPE, since the Mamba layers already provide the positional information.
3. **Multi-token prediction** (Super and Ultra). Predicting several future tokens gives a richer training signal, which raises the average benchmark scores by about 2.4% in an ablation on an MoE with 8B active parameters. The MTP module also serves as the draft model of speculative decoding: a lightweight module reaches about 97% acceptance on the first two predicted tokens.
4. **NVFP4 training** (Super and Ultra): a stable and accurate pre-training of a hybrid Mamba-MoE on up to 25T tokens. The weights, activations and gradients are quantized to NVFP4, so that the matrix multiplications of the forward pass (fprop), of the input gradients (dgrad) and of the weight gradients (wgrad) all run in FP4, at 3× the peak throughput of FP8 on GB300. The recipe uses 2D block scaling for the weights, random Hadamard transforms on the inputs of wgrad, and stochastic rounding of the gradients, and it keeps the last 15% of the network in high precision. Sensitive layers keep a higher precision too. The QKV and attention projections stay in BF16 (the few GQA layers have only 2 KV heads), and the output projections of the Mamba layers stay in MXFP8, since NVFP4 flushes up to 40% of their values to zero. The paper reports a relative loss gap to BF16 below 1% on Nano, and below 0.6% on the MoE with 8B active parameters, so the gap shrinks as the model grows.

The architecture balances three layer types: MoE layers, which scale the parameters sparsely; a few attention layers, which route information precisely between all the tokens; and Mamba-2 layers, which model the sequence at a fixed cost. Nemotron 3 Nano 30B-A3B reaches 3.3× the throughput of Qwen3-30B-A3B (8K input and 16K output tokens), and stays competitive on reasoning and long-context benchmarks (RULER at 1M tokens). The first figure shows the layer pattern of Nano. The second one shows the loss gap of NVFP4 training at both model sizes, and how it widens when the sensitive layers are quantized to NVFP4 too.

{: .picture}
<img loading="lazy" width="69.5%" height="69.5%" src="{{ site.assets }}/publications/nemotron3.svg"/>

{: .caption}
<b>Nemotron-3-Nano-30B-A3B.</b> Nemotron 3 models (e.g., Nemotron 3 Nano) leverage a hybrid Mamba-Transformer MoE architecture consisting predominantly of interleaved Mamba-2 and MoE layers, with a select few self attention layers. Adapted from: [NVIDIA Nemotron 3: Efficient and Open Intelligence](https://arxiv.org/abs/2512.20856)

{: .picture}
<img loading="lazy" width="98.7%" height="98.7%" src="{{ site.assets }}/publications/nemotron3_2.svg"/>

{: .caption}
Relative difference in train loss (left) and validation loss (right) between models trained with NVFP4 and BF16 (higher is better), shown at two model scales: Nemotron 3 Nano (A3B) and the larger MoE model (A8B). Loss gaps decrease as model size increases (A3B → A8B). Recipe ablation on Nemotron 3 Nano started from Nemotron 3 NVFP4 checkpoint at 500B tokens, then quantizes sensitive layers (Mamba Output, QKV, and Attention projections) to NVFP4, highlighting the importance of keeping these layers in high precision. Adapted from: [NVIDIA Nemotron 3: Efficient and Open Intelligence](https://arxiv.org/abs/2512.20856)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts</span><span class="pub-meta">, ByteDance (MLSys 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2502.19811" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Distributed MoE training usually hides the communication of expert parallelism by overlapping it with computation, and this matters, since the communication can take about 47% of the total execution time with popular models and frameworks. According to Comet, the existing schemes overlap at a coarse grain: they split an MoE layer into a few large chunks, which either leaves part of the communication exposed, or shrinks the GEMM tiles so much that the kernels slow down. FLUX finds the same tension for tensor parallelism (see its entry).

Comet works in two parts. **Dependency resolving** finds which pieces of computation depend on which pieces of communication, and schedules them in a fine-grained pipeline, so that each piece starts as soon as its data is ready. **Adaptive workload assignment** splits the SMs of the GPU between communication and computation. The best split depends on the layer, the input shape and the degree of parallelism, so Comet tunes it for each configuration instead of fixing it. The figure below shows this for the layer1 kernel: its duration is shortest at a different number of communication thread blocks $$n_c$$ for each number of tokens $$M$$ and for each mix of tensor and expert parallelism.

{: .picture}
<img loading="lazy" width="90%" height="90%" src="{{ site.assets }}/publications/comet.svg"/>

{: .caption}
Duration of the MoE layer1 kernel with varying number of thread blocks assigned for communication ($$n_c$$). The total number of thread blocks is identical to the number of SMs on Hopper(132). The figure shows four cases with different parallelisms. Adapted from: [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811)

Comet ships several pre-compiled kernels, each with a different split of the SMs. Before deployment, the best kernel for each configuration is profiled and stored as metadata, and at runtime, Comet selects the kernel from this metadata. An MoE layer runs a dispatch, two matrix multiplications and a combine, and Comet fuses them into two kernels: the dispatch with the first matrix multiplication (layer0), and the second matrix multiplication with the combine (layer1). Both kernels interleave communication and computation tile by tile.

{: .picture}
<img loading="lazy" width="66.7%" height="66.7%" src="{{ site.assets }}/publications/comet_2.svg"/>

{: .caption}
<b>Top:</b> The producer-consumer modeling of layer0 (left) and layer1 (right) of an MoE layer. The global size of the shared tensor is $$(M \times topk, N)$$ for both layer0 and layer1. <b>Bottom:</b> Decompose and reschedule the shared tensor in MoE layer0. In this illustration, three experts are located on Rank 0, each requiring both local and remote data for computation. Adapted from: [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811)

On Hopper, the mechanism is thread-block specialization. Each fused kernel launches a fixed set of thread blocks, and each block either communicates or computes, never both. Since each thread block occupies one SM, this dedicates some SMs to communication and the rest to computation. The compute blocks call the device-side GEMM functions of CUTLASS inside the fused kernel, unchanged.

How does Comet find the dependencies? Between a communication operation and the computation that uses its data, there is a *shared tensor*: the buffer that the dispatch writes and the GEMM reads, or that the GEMM writes and the combine reads. Comet analyzes this buffer in two steps, shown at the bottom of the figure above. It first decomposes the shared tensor along one dimension. It then reschedules the computation to follow the order in which the data arrive: for each tile of the GEMM, it finds the slices of the shared tensor that the tile needs, and orders the tiles so that they consume the slices in the order in which the communication fills them.

On average, a single MoE layer runs 1.96× faster, and the end-to-end execution 1.71× faster. Comet runs on production clusters of about ten thousand GPUs, where it has saved millions of GPU hours.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2507.07120" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Helix Parallelism targets real-time decoding with a KV history of *millions of tokens*, under a tight token-to-token latency (TTL) budget. Two costs dominate in this regime: reading the **FFN weights** at every step, and streaming the large **KV cache** from DRAM. The difficulty is that the standard fix for one cost makes the other worse. Tensor parallelism (TP) spreads the reads of the FFN weights well, but it scales poorly for attention: once the TP width exceeds the number of KV heads, the KV cache has to be *duplicated* across GPUs, which consumes memory bandwidth, caps the parallelism and limits the batch size. And the KV reads from DRAM also grow linearly with the batch size. No single sharding strategy relieves both costs at once: each one trades the duplication of the KV cache against the load of the FFN weights.

Helix solves this with a hybrid execution: it changes the parallel layout between the attention and the FFN of each layer, within one decode step, on the same GPUs.

- **Attention** runs with KV parallelism (KVP), combined with TP over the KV heads (TPA, at most the number of KV heads), on $$N = KVP \times TPA$$ GPUs. The KV cache is sharded along the sequence dimension, so that no GPU holds a duplicate of it. A decode step then works as follows:
  - every KVP rank computes the full QKV projection of the new token locally, since duplicating this small projection is cheaper than gathering the query across GPUs;
  - each GPU computes the attention of this query over its local slice of the KV cache, and produces a partial output and a log-sum-exp (the softmax normalizer) per token, as in FlashDecoding;
  - a single all-to-all over the query heads exchanges the partial results, and each GPU rescales and sums them into the exact attention output. The traffic of this all-to-all grows with the batch size and the hidden size, but not with the KV length, since the KV cache never moves, so it stays constant as the context grows to millions of tokens.
- **FFN**: the same $$N$$ GPUs switch to TP, and the all-to-all has already left the attention output in the TP layout of the output projection. A dense FFN runs with TP, with each GPU holding a slice of the weight matrices, and an MoE runs with TP × expert parallelism (EP), with the experts distributed across the GPUs.

To hide the remaining communication, Helix adds HOP-B (batch-wise overlap): the all-to-all of one request overlaps with the attention computation of the next.

For DeepSeek-R1 on Blackwell, Helix reduces the TTL by up to 1.5× at a fixed batch size, and supports up to 32× larger batches under the same latency budget, compared with conventional parallelism. Attention-FFN disaggregation (for example, MegaScale-Infer) places the two phases on different GPUs, in *space*, whereas Helix alternates them on the same GPUs, in *time*.

{: .picture}
<img loading="lazy" width="88.5%" height="88.5%" src="{{ site.assets }}/publications/helix.svg"/>

{: .caption}
<b>Helix's layout per-GPU workflow in attention and FFN stages.</b> Helix reuses the same pool of $$N$$ GPUs by configuring $$N = KVP \times TPA \quad (TPA \le K) \quad \longrightarrow \quad N = TPF \times EP$$ on a per-layer basis. <b>(Top)</b> During attention, each of the $$KVP$$ GPUs independently projects the full batch $$[B, H]$$ into QKV, runs FlashAttention on its KV shard to produce partial outputs and log-sum-exp scalars, and then participates in a single All-to-All over the query-head axis; each GPU rescales and sums the received fragments into the exact softmax-normalized tensor, followed by a TP All-Reduce to form the final $$[B, H]$$ attention output. <b>(Bottom)</b> For FFNs, Helix follows two modes depending on model type: for <i>Dense</i> FFNs $$EP = 1$$, it retains all $$N$$ GPUs in tensor-parallel $$TPF = N$$ to compute $$[B, H] \rightarrow [B, F/N] \rightarrow [B, H]$$ followed by a TP All-Reduce; for <i>MoE</i> FFNs $$EP > 1$$, it repartitions the $$N$$ GPUs into a $$TPF \times EP$$ grid, routes tokens to the appropriate experts, applies TP to FC layers within each expert group, performs an intra-expert All-Reduce followed by an inter-expert All-Gather, and concludes with a local reduction to yield $$[B, H]$$. Helix switches between these configurations seamlessly, enabling zero-downtime pipelining and better GPU utilization. Adapted from: [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/abs/2507.07120)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference</span><span class="pub-meta">, Peking University & Tencent</span><a class="pub-link" href="https://arxiv.org/abs/2508.15881" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

TPLA resolves an incompatibility between Multi-Head Latent Attention (MLA) and tensor parallelism (TP). MLA, introduced in DeepSeek-V2, compresses the keys and values into a single low-rank latent vector, `cKV`, and caches only that vector, which shrinks the KV cache. Megatron-LM TP, however, splits the attention heads across devices, and each device still needs the *entire* `cKV` to compute its heads, so the cache is replicated on every GPU. This replication erodes the advantage of MLA: the memory per device ends up no better than with Grouped Query Attention (GQA), or even worse. An existing alternative, Grouped Latent Attention (GLA), shards the latent instead, but each head then sees only a part of it, which weakens its representational capacity.

TPLA **partitions both the latent vector and the input dimension of each head across the devices, runs the attention independently on each shard, and sums the outputs of the shards with an all-reduce**. The key difference from GLA is that every head still attends to the *full* latent space, only computed in pieces, so TPLA keeps more of the capacity of MLA while it cuts the KV cache per device. Since each shard applies its own softmax before the all-reduce, the result approximates MLA instead of reproducing it exactly. TPLA still works on checkpoints pre-trained with MLA, without retraining, and an orthogonal transform (Hadamard or PCA) applied before the slicing reduces the interference between the shards and keeps the accuracy loss small.

The split is asymmetric, with MLA for the prefill and TPLA for the decode, because the two phases have different bottlenecks. The **prefill** processes the whole prompt at once, so it is compute-bound and does not re-read a KV cache: it keeps the standard (reparameterized) MLA. The **decode** generates one token at a time and reloads the entire KV cache from HBM at every step, so it is memory-bound, and this is where the replicated `cKV` hurts. The decode therefore uses TPLA, with the first slice of `cKV` on device 0, the next one on device 1, and so on. This split explains the "disaggregated" of the title: a sharded latent for the memory-bound decode, and the standard MLA for the compute-bound prefill. In the decode of panel (c) of the figure, each device computes every head on its own slice of the latent cache, and the all-reduce sums the partial outputs.

Thanks to the smaller KV cache per device, TPLA speeds up DeepSeek-V3 by **1.79×** and Kimi-K2 by **1.93×** at a context of 32K tokens, and keeps the quality on commonsense and LongBench benchmarks. It can be implemented with FlashAttention-3, for end-to-end gains.

{: .picture}
<img loading="lazy" width="99.9%" height="99.9%" src="{{ site.assets }}/publications/tpla.svg"/>

{: .caption}
Comparison of MLA, GLA, and TPLA. In MLA, each device must load the entire KV cache. In GLA, each attention head only accesses the portion of the KV cache stored on its own device. In TPLA, the prefilling phase follows MLA for efficiency and accuracy, while during the decoding phase, attention heads are distributed across devices, each relying on the KV cache stored locally on its assigned device. Adapted from: [TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference](https://arxiv.org/abs/2508.15881)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">PipeFill: Using GPUs During Bubbles in Pipeline-parallel LLM Training</span><a class="pub-link" href="https://openreview.net/forum?id=650J0YFjnV" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

PipeFill raises the GPU utilization of pipeline-parallel LLM training by filling the idle *bubbles* with other pending jobs, instead of removing the bubbles. The bubbles often take 15% to 30% of the GPU time of a job, and sometimes more than 60%. PipeFill measures the duration of the bubbles and the free memory, fits the work of the fill jobs to them, adds explicit pipeline-bubble instructions, and runs the fill jobs inside the gaps. It raises the utilization by up to 63%, with less than 2% slowdown of the main job: on a run with 8K GPUs, this is the work of about 2.6K extra GPUs, and even smaller training runs gain 5% to 15%. PipeFill is a context-switching layer, not a pipeline schedule. It therefore complements the schedules that shrink or hide the bubbles of the training job itself, such as 1F1B, interleaving, Zero Bubble and DualPipe, through the overlap of forward and backward passes, or of computation and communication.

Not every job makes a good fill job. The bubbles come and go, so latency-sensitive jobs, such as online inference, are excluded, and a fill job must split into pieces that fit into a single bubble and into the memory that the main job leaves free. Batch (offline) inference fits best, as in the back-end jobs of recommendation and data analytics. The paper also uses the training of smaller models as fill jobs, since training is long-running and tolerates interruptions, but batch inference makes better use of the bubbles: its small memory needs allow larger batches in the free memory.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Speculative Decoding with Blockwise Sparse Attention</span><span class="pub-meta">, MatX</span><a class="pub-link" href="https://matx.com/research/sd_nsa" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This MatX article resolves a conflict between two methods that accelerate decoding. Both raise the operational intensity of decoding, the work done per byte loaded from HBM, which matters because decoding, especially with long contexts, otherwise leaves modern accelerators underused. **Speculative decoding (SD)** uses a small draft model to propose several tokens, which the large target model then verifies in parallel, and it typically gives about 2× gains in latency or throughput. **Blockwise sparse attention**, as in the Native Sparse Attention (NSA) of DeepSeek and the Mixture of Block Attention (MoBA) of Moonshot, divides the context into blocks, and each token attends to a dynamic subset of them, so that less KV data is loaded from HBM.

Combined naively, the two methods work against each other. In the verification step of SD, each draft token may select a different subset of blocks, so the step loads the union of these subsets, and the KV cache loses its sparsity. The article restores the sparsity by forcing all the draft tokens to attend to the same subset of the context: during training, a group of consecutive tokens, as many as a verification step holds, shares the KV blocks selected by its first token. NSA models trained this way match the language-modeling quality of the baseline NSA, and their verification step reaches up to **3.5× higher operational intensity** than with independent selections, in the limit of large batches. The models are small, however: 186M and 1.2B parameters, trained on sequences of 2K tokens.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention</span><span class="pub-meta">, Tsinghua & UC Berkeley</span><a class="pub-link" href="https://arxiv.org/abs/2509.24006" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

SLA targets the attention bottleneck of Diffusion Transformers (DiTs), in particular for video generation, where long token sequences make the quadratic cost of attention dominate the latency. Its key observation is that the attention weights split into two parts: a small fraction of large weights with a high rank, and the remaining weights, with a very low rank. This suggests a different acceleration for each part, sparse for the first one and low-rank (linear) for the second one, instead of a single approximation, pure sparsity or pure linear attention.

SLA therefore classifies the attention blocks as *critical*, *marginal* or *negligible*. For $$N$$ tokens, critical blocks get exact attention, at a cost of $$O(N^2)$$, marginal blocks get linear attention, at a cost of $$O(N)$$, and negligible blocks are skipped. A single GPU kernel fuses the three cases, for both the forward and the backward pass, so a pre-trained DiT needs only a few fine-tuning steps to adapt.

SLA reduces the attention computation by **95%** (about 20×) without degrading the end-to-end generation quality. On **Wan2.1-1.3B**, its kernel speeds up attention by **13.7×**, and the end-to-end video generation by **2.2×**.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSeek‑V3.2‑Exp (model card & notes)</span><span class="pub-meta">, DeepSeek‑AI</span><a class="pub-link" href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek-V3.2-Exp is an update of the DeepSeek-V3 family that aims at **serving efficiency**, especially with long contexts. It is DeepSeek-V3.1-Terminus with a single architectural change, **DeepSeek Sparse Attention (DSA)**, added through continued training (see the DeepSeek-V3.2 entry), and the core MoE transformer recipe stays the same. Compared with the original DeepSeek-V3 report (December 2024), the emphasis therefore moves from how the model is trained to how it can run at a lower cost. In DSA, a lightning indexer selects the top $$k = 2048$$ KV entries for each query, so the cost of the core attention drops from $$O(L^2)$$ to $$O(Lk)$$ for a context of $$L$$ tokens, in both the prefill and the decode. The KV cache itself does not shrink, however: the full latent cache is still stored, so that any token can be selected. What drops is the compute, and the number of KV entries read per step.

The release notes on Hugging Face also give practical serving details: how to run the model with SGLang and vLLM, the open kernels (in TileLang, and in the DeepGEMM and FlashMLA libraries), a fix for a mismatch of the RoPE layouts of the indexer and of the MLA in the demo code, and evaluation notes. The benchmarks stay on par with V3.1-Terminus, while the largest gains come at long contexts. The final DeepSeek-V3.2 keeps the same architecture ([technical report](https://arxiv.org/abs/2512.02556), December 2025).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepCompile: A Compiler-Driven Approach to Optimizing Distributed Deep Learning Training</span><span class="pub-meta">, Microsoft DeepSpeed</span><a class="pub-link" href="https://arxiv.org/abs/2504.09983" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepCompile extends deep learning compilers to distributed training. Compilers such as the PyTorch compiler speed up single-GPU training with kernel fusion and operator scheduling, but they fall short on distributed workloads. Distributed training frameworks, on the other hand, implement their optimizations at the PyTorch level, where compiler techniques such as dependency analysis and operator scheduling are hard to apply. The fully sharded approach of ZeRO-3 and FSDP, for example, relies on two runtime optimizations:

- *prefetching* starts the all-gather of the parameters of a layer before the layer needs them, so that the communication overlaps with the computation;
- *unsharding* keeps parameters in their full form when the memory allows, to reduce the communication.

DeepCompile closes this gap at the compiler level: it "takes a standard single-GPU model implementation and transforms it into an optimized multi-GPU training graph without requiring changes to the model code". The model becomes a computation graph, and a sequence of optimization passes, guided by profiled timings and memory use, adds, reorders or removes operations such as all-gathers and memory allocations. On top of this graph, the authors implement a fully sharded approach, as in ZeRO-3 and FSDP, with three optimization passes:

- **Proactive prefetching** starts each all-gather as early as possible, to maximize the overlap of communication and computation, and takes into account how the free memory changes during the forward and backward passes.
- **Selective unsharding** keeps as many parameters unsharded as possible, to avoid all-gathers, and chooses them from a memory profile of each operator.
- **Adaptive offloading** moves optimizer states, such as the momentum and variance of Adam, to CPU memory when the GPU memory is insufficient. It offloads only the data above the memory limit, and overlaps the transfers with computation.

DeepCompile implements ZeRO-3, ZeRO-1 and offloading automatically. On Llama-3 70B and Mixtral 8×7B, it is up to 1.28× faster than ZeRO-3 and up to 1.54× faster than FSDP, and with offloading, on limited GPU resources, its throughput is up to 7.01× higher. Future directions include automatic parallelization (sequence and tensor parallelism), better memory management, and dynamic adaptation at runtime.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler</span><span class="pub-meta">, ByteDance Seed</span><a class="pub-link" href="https://arxiv.org/abs/2504.19442" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Triton-distributed extends Triton with *distributed* primitives, so that a single kernel can **compute and communicate**, instead of a GEMM kernel followed by a separate NCCL call. In its programming model, a distributed kernel is a set of asynchronous tasks, which **signal** one another and move data directly between GPUs through **symmetric memory**, with [OpenSHMEM](https://en.wikipedia.org/wiki/SHMEM)-style primitives that are lowered to NVSHMEM or ROCSHMEM.

The difference from plain Triton with NCCL is that the communication becomes an explicit part of the kernel schedule. The compiler and the runtime can then overlap loads, arithmetic, and remote puts and gets, at a much finer grain than the overlap of independent kernels on separate streams. And while FLUX hand-optimizes specific communication patterns of LLMs, Triton-distributed is a **general programming model**, with primitives and compiler support that can be reused across operators and topologies.

On 8 H800 GPUs, its kernels are on average **1.42×** (AllGather+GEMM) and **1.28×** (GEMM+ReduceScatter) faster than PyTorch with NCCL. On two nodes (16 GPUs), these kernels reach about 96% of the performance of FLUX, while being written in Python. The MoE kernels show larger average speedups (**44.97×** for AllGather+MoE, **15.55×** for MoE+ReduceScatter), but against a weak baseline, with grouped GEMMs in Python loops. Triton-distributed scales to 64 GPUs, and also runs on AMD MI308X GPUs. Its re-implementation of the low-latency all-to-all kernels of DeepEP is faster than DeepEP from 8 to 64 GPUs, except for the dispatch on 64 GPUs, with average speedups of 1.18× in the dispatch and 1.44× in the combine; on 128 GPUs, DeepEP stays faster.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">TileLink: Generating Efficient Compute–Communication Overlapping Kernels Using Tile-Centric Primitives</span><span class="pub-meta">, ByteDance Seed (MLSys 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2503.20313" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

TileLink aims at the performance of FLUX without a hand-crafted fused kernel for each operation. It separates the design of the communication from that of the computation, so that each side can choose its own tile size, tile order and resource mapping, and it links the two through **tile-centric primitives**: producer-consumer signals (notify and wait) and tile data movement (push and pull). Each primitive maps to a tensor slice, a peer rank and a barrier channel. The system then schedules computation and communication *per tile*, instead of per tensor, and reaches an overlap close to that of highly tuned systems.

TileLink differs from two related systems in its level of abstraction:

- **FLUX** splits specific tensor-parallel operations of LLMs into fine slices and fuses them into larger kernels. It overlaps almost all of the communication, but each kernel is tailored to one operation.
- **Triton-distributed** offers low-level primitives inside kernels (symmetric memory, signals, tasks) and a general programming model.
- **TileLink** sits in between: it is higher-level than Triton-distributed, and more general and easier to program than FLUX, but it still models the overlap explicitly, at the level of the *tile schedule*.

On 8 H800 GPUs, its kernels are **1.17× to 20.76×** faster than non-overlapping baselines, and comparable to or faster than state-of-the-art overlapping libraries, such as FLUX and RingAttention. End to end, on eight language models, it is on average **1.32×** faster than PyTorch on 8 H800 GPUs, and **1.29×** faster on two nodes (16 GPUs).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production</span><span class="pub-meta">, ByteDance Seed & Peking University</span><a class="pub-link" href="https://arxiv.org/abs/2505.11432" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

MegaScale-MoE is a production system for **communication-efficient MoE training**. In MoE training, the hard part is not only to compute the experts, but also to route the tokens and to move the activations and gradients at scale: the training becomes dominated by **communication**, such as the all-to-all between the routers and the experts, and keeping the GPUs busy needs a strategy for the whole system. MegaScale-MoE has three parts:

1. it customizes the parallelism of each MoE layer to reduce the communication volume: sequence parallelism for the attention and expert parallelism for the FFNs, within a node, instead of the tensor parallelism of Megatron-LM;
2. it overlaps communication with computation, both between and within operators;
3. it compresses the communication to a lower precision, with adjusted communication patterns: in BF16 training, for example, the gradients of data parallelism are synchronized in BF16 instead of FP32, which halves their traffic.

Unlike FLUX, TileLink and Triton-distributed, which change how fused kernels are written, MegaScale-MoE is a system and a parallelization strategy for end-to-end training at a large scale: its gains come from the parallel layout, the scheduling and the communication precision of the whole training job.

Training a **352B MoE** model on **1,440 NVIDIA Hopper GPUs**, it reaches **1.41M tokens/s**, 1.88× the efficiency of Megatron-LM (1.65× to 1.88× across the strong-scaling settings). It runs the production MoE training of ByteDance, where it supports models with trillions of parameters and single training jobs on more than 10,000 GPUs.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism</span><span class="pub-meta">, ByteDance Seed & Peking University</span><a class="pub-link" href="https://arxiv.org/abs/2504.02263" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The sparse activation of an MoE makes its FFNs memory-bound during decoding: each expert receives only a few tokens of a batch, so its weights are loaded for little computation. MegaScale-Infer addresses this with **Disaggregated Expert Parallelism (DEP)**: the attention and the expert modules run on separate GPUs, so that each scales independently, with its own parallelism. The attention modules are replicated with data parallelism, and the FFN modules are scaled with expert parallelism. Each expert then gathers the tokens of many attention replicas, so its batch grows, and the FFNs become efficient, compute-bound work again. The disaggregation also allows a **heterogeneous deployment**, with a different GPU type for each module, to reduce the cost. The attention, which reads the KV cache, needs memory bandwidth and capacity, while the experts need compute, so the paper places the attention on H20 GPUs and the experts on L40S GPUs.

Two system components keep the communication from stalling the disaggregated modules:

- **ping-pong pipeline parallelism** splits a batch of requests into micro-batches and shuttles them between the attention and the FFN GPUs, so that both pools stay busy and the communication hides behind computation;
- a **high-performance M2N communication library** for the dispatch and the combine of the tokens removes unnecessary GPU-to-CPU copies, the overhead of group initialization, and GPU synchronization.

This is attention-FFN disaggregation for MoE serving: the two modules are split *in space*, across GPU pools, whereas Helix splits them *in time*, on the same GPUs. Prefill-decode disaggregation (as in TPLA) is a different, orthogonal split.

On homogeneous clusters, MegaScale-Infer reaches up to **1.90×** the decoding throughput per GPU of state-of-the-art LLM serving systems. On heterogeneous clusters, its decoding throughput per unit cost is up to **1.86×** that of TensorRT-LLM, and up to 3.24× that of vLLM, both running on H20 GPUs.

{: .picture}
<img loading="lazy" width="69.5%" height="69.5%" src="{{ site.assets }}/publications/megascale_infer.svg"/>

{: .caption}
MegaScale-Infer runtime instance architecture. The attention nodes are replicated ($$M$$ of them) and the expert nodes split the experts ($$N$$ of them, expert parallelism); inside each node the GPUs use tensor parallelism. Tokens go from the attention nodes to the expert nodes (M2N) and back (N2M) over InfiniBand or Ethernet, in a ping-pong pipeline. Adapted from: [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](https://arxiv.org/abs/2504.02263)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Accelerating MoE Model Inference with Expert Sharding (MoEShard)</span><span class="pub-meta">, EPFL & McGill (EuroMLSys 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2503.08467" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Balmau et al. target MoE inference with expert parallelism (EP), where each GPU holds a subset of the experts. The routing of tokens is skewed, so the GPUs that hold the popular experts are overloaded while the others sit idle. The usual fixes have a price: a capacity factor caps the number of tokens per expert and drops the rest, which can hurt accuracy, and replicas of the popular experts cost memory. **MoEShard** avoids both by sharding *every* expert across all GPUs, in the style of tensor parallelism. The first matrix of each expert is split by columns and the second by rows, so each GPU computes a slice of every selected expert without any synchronization between the two multiplications, and the sum of the partial outputs gives the output of the expert. Every GPU then does the same work, whatever the routing, and no token is dropped.

Sharding every expert across every GPU multiplies the number of small expert computations, so MoEShard fuses them to reduce the number of kernel launches: the tokens that all GPUs send to the same expert are concatenated into one tensor, which brings the number of expert computations down from one per expert and GPU to one per expert. On the encoder of an MoE model (Switch Transformer), on a single node with four GPUs, the time to first token is up to **6.4×** lower than with DeepSpeed. The work complements the MegaScale systems (see their entries): it is a small study, on one node, which shows that sharding the experts is an alternative to plain EP when the routing is imbalanced, as long as the fast links within the node carry the extra communication.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">TransMLA: Multi-Head Latent Attention Is All You Need</span><span class="pub-meta">, Peking University & Xiaomi</span><a class="pub-link" href="https://arxiv.org/abs/2502.07864" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Multi-head latent attention (MLA), introduced by DeepSeek-V2 (see its entry), caches one small latent vector per token, from which the keys and values of all heads are projected back. Most model providers, however, have invested heavily in models with Grouped-Query Attention (GQA), and they have little incentive to pre-train MLA models from scratch. TransMLA closes this adoption gap with a **post-training conversion**, which turns trained GQA models, such as LLaMA, Qwen and Mixtral, into MLA models. The theoretical core of the paper is a proof that **GQA can always be rewritten exactly as MLA with the same KV-cache size, but not the reverse**: for a given cache size, MLA is strictly more expressive than GQA. This asymmetry means that any GQA checkpoint can be rewritten in MLA form without loss, and then *upgraded* to use the extra expressiveness of MLA.

The conversion takes two steps:

1. It first **converts** the GQA attention into the MLA structure exactly, by rewriting the repeated KV heads of GQA as low-rank latent projections. The model then runs in the DeepSeek codebase and its optimized serving systems (vLLM, SGLang), and it can be combined with features such as FP8 quantization and multi-token prediction.
2. The exact conversion keeps the size of the KV cache, so on its own it does not make inference any faster: the speedup only comes when MLA uses a smaller cache. TransMLA therefore **compresses** the latent KV below the original cache size, and then **fine-tunes** the model briefly to recover its quality. About 6B tokens are enough to bring the performance back on par with the original model across benchmarks.

Compressing 93% of the KV cache of LLaMA-2-7B gives a 10.6× inference speedup at a context of 8K tokens, while preserving a meaningful output quality. TransMLA complements TPLA and Helix (see their entries): rather than designing new MLA models or new ways to shard them, it brings the many existing GQA models into the more efficient MLA family.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B</span><span class="pub-meta">, Stanford Hazy Research</span><a class="pub-link" href="https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This work targets an extreme regime of low latency: **single-sequence decoding** with a 1B-parameter LLM (Llama-3.2-1B). With a single sequence, every decoding step reads all the weights from global memory and does very little computation with them, so the performance depends on how continuously the GPU streams the weights. The authors argue that inference engines still suffer from *pipeline bubbles* in this regime, because a forward pass is split into about a hundred small kernels. Each kernel has its own launch and teardown costs, and the synchronization between kernels stalls the GPU, so the memory streaming stops at every kernel boundary.

Their solution fuses the *entire* forward pass into a single **megakernel**, which acts as an interpreter on the GPU: each SM executes a schedule of *instructions*, such as a fused RMSNorm, QKV projection and RoPE, the attention, the output projection, or a part of the MLP. Fusing about a hundred operations into one kernel raises three practical problems, and the paper answers each of them:

1. how to *program* such a megakernel: with an instruction abstraction, executed by the interpreter, whose schedules are generated ahead of time;
2. how to avoid contention for resources: with **paged shared memory**, in which an instruction releases its pages as soon as it no longer needs them, so that the next instruction can start loading its weights early;
3. how to synchronize without kernel boundaries: with a lightweight scheme of **counters** in global memory, which each instruction increments when it completes, and on which the instructions that depend on it wait.

On an **H100**, the megakernel uses about **78%** of the memory bandwidth, while vLLM and SGLang use at most about **50%** in this setting. End to end, with a prompt of 32 tokens, 128 generated tokens and no speculation, it is about **2.5× faster than vLLM** and more than **1.5× faster than SGLang** on the H100. On a **B200**, it is more than **3.5×** faster than vLLM, and still more than **1.5×** faster than SGLang. A forward pass of the 16-bit model takes **less than 1 ms** on the H100, and **less than 680 µs** on the B200.

The work sits in a different part of the design space than FLUX, TileLink and Triton-distributed (see their entries), which fuse computation with distributed communication: it removes the kernel boundaries *entirely*, for low-latency inference on a single GPU.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Mirage: A Multi-Level Superoptimizer for Tensor Programs</span><span class="pub-meta">, CMU et al. (OSDI 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2405.05751" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Mirage generates deeply fused tensor programs *automatically*. It frames kernel fusion as a [superoptimization](https://en.wikipedia.org/wiki/Superoptimization) problem, i.e. a search for the fastest program that computes the same function, over **µGraphs**: a uniform representation of a tensor program at the kernel, thread-block and thread levels of the GPU compute hierarchy. Because a µGraph describes all three levels at once, Mirage can search algebraic transformations, schedule transformations and the generation of new custom kernels jointly, and it rediscovers kernels like FlashAttention and FlashDecoding on its own. The search space is huge, so an abstraction-based pruning technique keeps it tractable, with an optimality guarantee, and a probabilistic equivalence verifier checks that the optimized µGraph computes the same function as the input program.

Mirage differs from three other lines of work on aggressive fusion:

- **hand-written megakernels**, such as "No Bubbles" (see its entry): Mirage *discovers* aggressive fusions automatically and generates their implementations;
- **FlashAttention-style** kernels: Mirage is broader, as it targets arbitrary tensor programs and not only attention, and it finds FlashAttention-like structures where they pay off;
- **compiler frameworks** such as TVM and XLA: Mirage superoptimizes small fused graphs, to reach a performance that can beat hand tuning.

Mirage outperforms existing approaches by **up to 3.3×** (the figure varies across versions of the paper), even on DNN components that are widely used and heavily optimized.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training</span><span class="pub-meta">, Tsinghua</span><a class="pub-link" href="https://arxiv.org/abs/2505.11594" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

SageAttention3 pushes the SageAttention line (see the SageAttention and SageAttention2 entries) toward even faster inference: it runs both matrix multiplications of attention on the new **FP4 Tensor Cores** of Blackwell GPUs. FP4 can represent very few values, so the kernel uses **microscaling** (block-scaled) FP4 quantization, in which each small block of values has its own scale, together with techniques specific to attention that keep the accuracy. Conceptually, it sits between attention kernels that assume FP16 or BF16 and models that are quantized end to end: it quantizes only the attention operator, and it is designed to plug into existing inference systems.

On an **RTX 5090**, it reaches **1038 TOPS**, 5× the speed of the fastest FlashAttention on that GPU, which is FlashAttention2, since FlashAttention3 runs only on Hopper. It speeds up various models end to end without changes, such as HunyuanVideo for video generation. The paper also explores low-bit attention for *training*, with SageBwd, an 8-bit attention for both the forward and the backward pass: it is lossless in fine-tuning, but it converges more slowly in pre-training.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Titans: Learning to Memorize at Test Time</span><span class="pub-meta">, Google Research (NeurIPS 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2501.00663" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Titans are a family of models that combine attention with a *neural* long-term memory. The memory is an MLP $$\mathcal{M}$$ whose weights keep being updated at test time. For each token $$x_t$$, with the key $$k_t = x_t W_K$$ and the value $$v_t = x_t W_V$$, the memory takes a gradient step on the associative-memory loss $$\ell(\mathcal{M}_{t-1}; x_t) = \Vert \mathcal{M}_{t-1}(k_t) - v_t \Vert^2$$. The gradient measures how *surprising* the token is to the memory, and the update adds two ingredients to it: a momentum, which carries the surprise of the previous tokens forward, and a forgetting gate $$\alpha_t \in [0, 1]$$, similar to weight decay, which removes old information:

$$
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t), \qquad \mathcal{M}_t = (1 - \alpha_t) \mathcal{M}_{t-1} + S_t
$$

where the data-dependent factors $$\eta_t$$ and $$\theta_t$$ set how fast the past surprise decays and how much of the new surprise enters. The memory lets attention focus on the current context and still use information from the distant past. It can be trained fast and in parallel, and it keeps inference fast. The paper reads the two parts as two kinds of memory: attention, with its limited context but precise modeling of dependencies, acts as a short-term memory, while the neural memory, which learns to memorize the data, acts as a long-term and more persistent memory.

The paper combines the two in three ways: the memory as a context (MAC), as a gate (MAG), or as a layer (MAL). Titans outperform Transformers and modern linear recurrent models, such as Mamba2 and DeltaNet, on language modeling, commonsense reasoning, genomics and time series. They scale to contexts of more than 2M tokens, with a higher needle-in-a-haystack accuracy than the baselines.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2024</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism</span><span class="pub-meta">, Peking University (SOSP 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2404.09526" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Long-context requests vary widely in length, and a single request changes its needs between its two phases: the prefill of a long prompt needs many GPUs to finish quickly, while the decode needs little computation but a large and growing KV cache. A fixed degree of parallelism per instance fits neither case well. LoongServe therefore introduces **Elastic Sequence Parallelism (ESP)**, which adapts the degree of sequence parallelism in real time, at each iteration and in each phase, by scaling up or down.

The prefill runs ring attention. While the KV tensors rotate across the GPUs, each GPU keeps a subset of them locally, so that they are already in place for the decode. This proactive scale-down avoids migrating the KV cache between the two phases. In the decode, however, the KV cache stays sharded across instances, so the decode follows the usual distributed decode algorithm, which is not very efficient and is often the bottleneck of context parallelism.

The prefill overlaps computation and communication with CUDA streams. For the decode, the paper claims that the communication of the partial attention results overlaps with computation, but this overlap is opportunistic: while a node sends its own results, it computes the matrix multiplications of the requests that it receives from other GPUs, only if such requests are there, which is not guaranteed. ESP also reduces the fragmentation of the KV cache across instances. Overall, LoongServe reaches up to **3.85×** the maximum throughput of chunked prefill, and up to **5.81×** that of prefill-decode disaggregation.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Mamba: Linear-Time Sequence Modeling with Selective State Spaces</span><span class="pub-meta">, CMU & Princeton (COLM 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2312.00752" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Mamba is the state-space model (SSM) that made a sequence model without attention competitive with Transformers on language. An SSM keeps a recurrent state of fixed size, so its cost grows only **linearly** with the sequence length. Earlier SSMs, such as S4, performed worse than Transformers on language, however, because their dynamics were fixed: they processed every token in the same way, whatever its content. Mamba fixes this with a **selection mechanism**, which makes the parameters of the SSM functions of the input, so that the model chooses, token by token, which information to propagate and which to forget. This is what allows content-based reasoning. The input dependence breaks the convolution form that made the earlier SSMs fast to train, so Mamba computes the recurrence with a **hardware-aware parallel scan**, which keeps the expanded state in SRAM, in the spirit of FlashAttention. The result is a model with 5× the inference throughput of Transformers and a linear scaling with the sequence length, and a Mamba-3B that matches Transformers twice its size. Its constant-size state also removes the growing KV cache, which is the property that hybrid models such as Jamba and Nemotron 3 exploit.

An SSM maps an input $$x_t$$ to an output $$y_t$$ through a hidden state $$h_t \in \mathbb{R}^N$$, independently for each channel of the input. Its parameters $$(\Delta, A, B, C)$$ describe a continuous-time system, so they are first **discretized** with the step size $$\Delta$$:

$$\bar A = \exp(\Delta A), \qquad \bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B$$

The discrete model then runs either as a **linear recurrence**, one token at a time during inference, or as a **global convolution** over the whole sequence during training:

$$h_t = \bar A\,h_{t-1} + \bar B\,x_t, \qquad y_t = C\,h_t$$

In S4, the parameters are **fixed across time**, as in a [linear time-invariant](https://en.wikipedia.org/wiki/Linear_time-invariant_system) (LTI) system, and this is what allows the convolution: unrolling the recurrence gives the same kernel $$(C \bar B, C \bar A \bar B, C \bar A^2 \bar B, \dots)$$ at every position of the sequence. In Mamba, $$B$$, $$C$$ and $$\Delta$$ are **functions of $$x_t$$** (the selective SSM, S6), so $$\bar A$$ and $$\bar B$$ change at every token. This is the selectivity, and it is also the reason why Mamba needs a scan instead of the convolution.

{: .picture}
<img loading="lazy" width="88.4%" height="88.4%" src="{{ site.assets }}/publications/mamba.svg"/>

{: .caption}
<b>Selective state space model with hardware-aware state expansion.</b> Structured SSMs independently map each channel (e.g. $$D = 5$$) of an input $$x$$ to output $$y$$ through a higher dimensional latent state $$h$$ (e.g. $$N = 4$$). Prior SSMs avoid materializing this large effective state ($$DN$$, times batch size $$B$$ and sequence length $$L$$) through clever alternate computation paths requiring time-invariance: the $$(\Delta, \boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C})$$ parameters are constant across time. Mamba's selection mechanism adds back input-dependent dynamics, which also requires a careful hardware-aware algorithm to only materialize the expanded states in more efficient levels of the GPU memory hierarchy. Adapted from: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Jamba: A Hybrid Transformer-Mamba Language Model</span><span class="pub-meta">, AI21 Labs</span><a class="pub-link" href="https://arxiv.org/abs/2403.19887" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Jamba (Lieber, Lenz et al., AI21) is the first **production-grade hybrid** that interleaves Transformer attention, Mamba SSM and MoE layers. It balances one central trade-off. The KV cache becomes the limiting factor when Transformers scale to long contexts, and replacing attention layers with Mamba layers shrinks it, because a Mamba layer carries only a constant-size state instead of a cache that grows with the context. Jamba therefore keeps attention sparingly, for precise recall, lets the Mamba layers do the cheap long-range work, and uses MoE to add capacity without adding active compute.

The released configuration shows the knobs of the design: 4 blocks of $$l = 8$$ layers, a ratio of attention to Mamba layers $$a : m = 1 : 7$$, an MoE in every other layer ($$e = 2$$), and 16 experts with top-2 routing. This gives 52B parameters in total but only 12B active ones, which fit on one 80 GB GPU. The 1:7 ratio comes from ablations: it performs as well as 1:3, and it costs less compute. The figure shows one such block: (a) its eight layers, one Transformer layer and seven Mamba layers, with an MoE in place of the MLP in every second layer, and (b) the four types of layers that a Jamba block can combine, each with an attention or a Mamba mixer, followed by an MLP or an MoE.

The headline benefit is the cache: at a context of 256K tokens, Jamba needs about 4 GB of KV cache, against 32 GB for Mixtral and 128 GB for Llama-2-7B. It gives strong results up to 256K tokens, with up to 3× the throughput of Mixtral at long contexts. Jamba 1.5 later scaled the same design to 94B active and 398B total parameters, and Nemotron 3 (see its entry) follows the same template, with Mamba-2 layers, and adds LatentMoE, MTP and NVFP4 in its larger models.

{: .picture}
<img loading="lazy" width="75.8%" height="75.8%" src="{{ site.assets }}/publications/jamba.svg"/>

{: .caption}
<b>(a)</b> A single Jamba block. <b>(b)</b> Different types of layers. The implementation shown here is with $$l = 8$$, $$a : m = 1 : 7$$ ratio of attention-to-Mamba layers, and MoE applied every $$e = 2$$ layers. Adapted from: [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Better & Faster Large Language Models via Multi-token Prediction</span><span class="pub-meta">, Meta (FAIR) (ICML 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2404.19737" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This paper (Gloeckle et al., Meta) establishes multi-token prediction (MTP) as a general training objective. It starts from a critique of the standard recipe: LLMs such as GPT and Llama are trained with a next-token prediction loss, and the authors argue that training a model to predict several future tokens at once makes it learn more from the same data, i.e. gives it a higher sample efficiency.

The model predicts the next $$n$$ tokens with $$n$$ independent output heads on top of a shared trunk, so multi-token prediction is an **auxiliary task** added to the usual objective, at no extra cost in training time. At position $$t$$, the main head predicts the next token $$x_{t+1}$$, as usual, while the auxiliary heads predict $$x_{t+2}$$, $$x_{t+3}$$, and so on. The extra heads also make inference faster, in two steps:

- *drafting*: during generation, the auxiliary heads produce a draft of the next few tokens, at the same time as the main head produces the next token;
- *parallel verification*: in the next step, the model checks all the drafted tokens with its main head, in a single forward pass.

The benefit grows with the model size. The 13B model solves 12% more HumanEval problems and 17% more MBPP problems than a comparable next-token model, and models trained to predict 4 tokens are up to 3× faster at inference with this self-speculative decoding, even at large batch sizes.

The scheme is a variant of standard speculative decoding, which splits the work between two models. A separate small draft model generates a few tokens, usually 3 to 5, one by one: it predicts a token, feeds it back as input, predicts the next one, and so on. These sequential steps are cheap because the draft model is small, so together they take less time than a single forward pass of the large model. The large target model then checks the whole draft in one forward pass with a causal mask. A Transformer processes all the positions of a sequence in parallel, so the target model computes the probability of the fourth drafted token at the same time as that of the first one, and it keeps the drafted tokens up to the first one it rejects. In about the time it takes to generate one token, it thus verifies the whole draft. With MTP, the heads of the model itself draft the tokens, so no second model is needed.

DeepSeek-V3 and Nemotron 3 (see their entries) both use MTP. DeepSeek-V3 refines it by replacing the independent parallel heads with *sequential, cascaded* modules, which keep the full causal chain, and Nemotron 3 uses a lightweight MTP module that reaches about 97% acceptance on its first two predicted tokens.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSeek LLM: Scaling Open-Source Language Models with Longtermism</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2401.02954" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek LLM, the first release of DeepSeek (also called DeepSeek-V1), has dense 7B and 67B models, trained on 2T tokens. It predates MoE and MLA, which came later with DeepSeekMoE and DeepSeek-V2. The architecture follows LLaMA: pre-norm with RMSNorm, SwiGLU FFNs and rotary position embeddings. The 7B model has 30 layers, and the 67B model has 95 layers and uses Grouped-Query Attention (GQA) to reduce the cost of inference.

The main contribution is a study of scaling laws. It first fits the optimal batch size and learning rate as power laws of the compute budget $$C$$: the optimal batch size grows with the budget, while the optimal learning rate shrinks. It then fits the optimal split of the compute between the model size and the data. For this split, the paper measures the model size in non-embedding FLOPs per token $$M$$, so that $$C = MD$$ for $$D$$ training tokens, instead of the usual estimate $$C = 6ND$$ from the number of parameters $$N$$: the parameter count ignores the cost of attention, and, when it includes the embeddings, it also counts the vocabulary computation, which adds little to the capacity of the model. The split also depends on the data: the higher the quality of the data, the larger the share of a growing budget that should go to the model rather than to the data. These fits set the configurations of both models. The "longtermism" of the title is the long-term perspective of the project on open-source models.

The chat models are aligned with supervised fine-tuning (SFT), then Direct Preference Optimization (DPO). DeepSeek LLM 67B outperforms LLaMA-2 70B, in particular on code, math and reasoning, and the 67B chat model outperforms GPT-3.5 in open-ended evaluations.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek-V2 introduces Multi-head Latent Attention (MLA) and combines it with [DeepSeekMoE](https://arxiv.org/abs/2401.06066), an MoE architecture that DeepSeek published a few months earlier. The two became the architectural foundations of V3 and R1. DeepSeek-V2 is an MoE model with 236B total parameters, of which 21B are activated per token, and a context of 128K tokens. Its two components target the two main bottlenecks of a Transformer: the KV cache of standard multi-head attention (MHA) during decoding, and the cost of dense FFNs during training.

**DeepSeekMoE** provides the FFN. It combines fine-grained experts with shared experts, so that sparse activation gives a large capacity at a low compute cost, and it rests on two ideas:

1. **Shared experts**: a small number $$K_s$$ of experts that every token uses, without routing. They are not copies of each other, but a fixed, always-active set that absorbs the common knowledge, so that the routed experts do not each need to learn it again.
2. **Fine-grained expert segmentation**: a regular MoE has $$N$$ experts, each a full-size FFN, and activates the top $$K$$ of them. DeepSeekMoE splits each expert into $$m$$ smaller ones, for $$mN$$ experts in total, and activates $$mK$$ of them. The total parameters and the FLOPs stay the same, but each token can now choose among many more combinations of experts, which lets the experts specialize more.

In DeepSeek-V2, each MoE layer has 2 shared experts and 160 routed experts, of which 6 are activated per token.

**MLA** attacks the KV cache with a low-rank joint compression of the keys and values. MHA caches the full keys and values of every head, and GQA and MQA shrink this cache by sharing keys and values across heads. MLA instead projects each token down to a single small latent vector, which is all that the cache stores, and projects it back up to distinct keys and values per head when the attention needs them. Let $$h_t$$ be the input hidden vector of the token at position $$t$$, $$n_h$$ the number of heads and $$d_h$$ the dimension per head. A learned down-projection $$W^{DKV}$$ compresses $$h_t$$ into the latent $$c_t^{KV}$$, of dimension $$d_c \ll n_h d_h$$, and two learned up-projections, $$W^{UK}$$ and $$W^{UV}$$, reconstruct the keys and values of all heads from it:

$$c_t^{KV} = W^{DKV}\, h_t \quad (\in \mathbb{R}^{d_c},\ \text{cached, together with the RoPE key } k_t^{R}); \qquad k_t^{C} = W^{UK} c_t^{KV},\quad v_t^{C} = W^{UV} c_t^{KV}$$

The letters $$D$$ and $$U$$ in the names of the matrices stand for down and up, and the superscript $$C$$ marks the *content* part of the keys and values, as opposed to the positional part below. The queries are compressed in the same way, which saves activation memory during training; they are not cached:

$$c_t^{Q} = W^{DQ}\, h_t, \qquad q_t^{C} = W^{UQ}\, c_t^{Q}$$

This compression lowers the rank of the attention on purpose. In MHA, the key projection of each head has a rank of at most $$d_h$$, but the key projection of all heads together can have full rank. In MLA, all heads share the latent $$c_t^{KV}$$, so the key projection of all heads, $$W^{UK} W^{DKV}$$, has a rank of at most $$d_c$$, and this latent is what the KV cache stores.

One difficulty remains: the positions. RoPE rotates each key by an angle that depends on its position, and this rotation would sit between the query and $$W^{UK}$$, which rules out the absorb trick described below. V2 therefore adds **decoupled RoPE**: the position follows a separate path, marked by the superscript $$R$$, with small rotary queries per head and one rotary key of dimension $$d_h^R$$, **shared** by all heads and cached as well:

$$q_t^{R} = \mathrm{RoPE}(W^{QR}\, c_t^{Q}), \qquad k_t^{R} = \mathrm{RoPE}(W^{KR}\, h_t)$$

The query and the key of each head $$i$$ concatenate the content and rotary parts:

$$q_{t,i} = [\,q_{t,i}^{C}\,;\,q_{t,i}^{R}\,], \qquad k_{t,i} = [\,k_{t,i}^{C}\,;\,k_t^{R}\,]$$

and the attention over the past tokens $$j \le t$$ is then standard, followed by the output projection $$W^O$$:

$$o_{t,i} = \sum_{j \le t} \mathrm{softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v_{j,i}^{C}, \qquad u_t = W^{O}\,[\,o_{t,1};\dots;o_{t,n_h}\,]$$

The KV cache per token and per layer shows the gain, with $$g$$ groups of KV heads in GQA:

| MHA | GQA | MQA | **MLA** |
|---|---|---|---|
| $$2\,n_h\,d_h$$ | $$2\,g\,d_h$$ | $$2\,d_h$$ | $$d_c + d_h^R$$ |

The *absorb* trick makes the decoding efficient: at decode time, $$W^{UK}$$ is folded into $$W^{UQ}$$, and $$W^{UV}$$ into $$W^{O}$$, so the attention runs directly on the cached latents $$c_j^{KV}$$, and the full keys and values per head are never materialized. DeepSeek-V2 uses $$d_c = 512$$ and $$d_h^R = 64$$, with $$d_h = 128$$, for a cache of $$576$$ elements per token and per layer. This is as much as GQA with only about 2.25 groups, and yet MLA performs better than MHA in the experiments of the paper.

Compared with the dense DeepSeek 67B (see the DeepSeek LLM entry), DeepSeek-V2 costs 42.5% less to train, has a **93.3% smaller KV cache**, and reaches up to **5.76× the maximum generation throughput**, with a higher quality.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization</span><a class="pub-link" href="https://arxiv.org/abs/2411.10958" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

SageAttention2 builds on SageAttention (see its entry) and quantizes more aggressively. Attention computes two products: $$QK^\top$$, the scores of the queries against the keys, and $$\widetilde{P}V$$, the product of the unnormalized attention probabilities and the values. SageAttention2 runs the expensive first product in **INT4**, with $$Q$$ and $$K$$ quantized *per thread*, a fine and hardware-friendly granularity that follows the layout of the matrix-multiply instructions, and the second product in **FP8**. Low-bit arithmetic is harder in attention than in linear layers, because attention suffers more from quantization problems, in particular from outliers, so the paper adds three techniques specific to attention:

1. per-thread INT4 quantization of $$Q$$ and $$K$$;
2. a smoothing of $$Q$$, which improves the accuracy of the INT4 product $$QK^\top$$;
3. a two-level accumulation, which improves the accuracy of the FP8 product $$\widetilde{P}V$$.

On an **RTX 4090**, SageAttention2 reaches about **3×** the OPS of FlashAttention2 and **4.5×** that of xFormers. On Hopper GPUs, it matches the speed of FlashAttention3 in FP8, with a higher accuracy.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</span><span class="pub-meta">, Tsinghua (ICLR 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2410.02367" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

SageAttention quantizes the attention kernel, not the whole model, and it is designed to plug into existing LLM inference. Its motivation is that attention can remain a major bottleneck even when the linear layers are well optimized, especially at long sequence lengths, and that it is hard to quantize: its mix of matrix multiplications, softmax and reductions amplifies numerical outliers.

SageAttention runs attention with **8-bit** arithmetic and keeps the accuracy. The paper finds two problems of naive INT8 attention, and it fixes each of them. First, $$K$$ has strong outliers in some channels, shared by all tokens. SageAttention *smooths* $$K$$ by subtracting its mean over the tokens, $$\bar{k}$$. This subtracts the same value $$q^\top \bar{k}$$ from all the scores of a query $$q$$, and the softmax does not change when the same constant is subtracted from all its inputs, so the result of the attention stays exact. Second, quantizing $$\widetilde{P}$$, the unnormalized attention probabilities, and $$V$$ to INT8 is not reliably accurate. SageAttention therefore computes only $$QK^\top$$ in INT8, and $$\widetilde{P}V$$ in FP16, with an FP16 accumulator. This first version is less aggressive than SageAttention2 and SageAttention3 (see their entries), and it serves as the baseline that shows that attention can be quantized with little integration work.

SageAttention reaches about **2.1×** the OPS of FlashAttention2 and **2.7×** that of xFormers. It is more accurate than the FP8 attention of FlashAttention3, with almost no loss in the end-to-end metrics of language, image and video generation models.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Centauri: Enabling Efficient Scheduling for Communication–Computation Overlap in Large Model Training via Communication Partitioning</span><span class="pub-meta">, Peking University et al. (ASPLOS 2024)</span><a class="pub-link" href="https://doi.org/10.1145/3620666.3651379" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Centauri overlaps communication with computation at the level of *operator scheduling*. Its core idea is to **decompose operators** into smaller pieces and to schedule these pieces across several CUDA streams, so that communication overlaps with useful computation more consistently than when whole operators overlap. The name *communication partitioning* says where the decomposition starts: a large collective, or any phase of data movement, is broken into smaller chunks, which can be scheduled earlier and more often, and so keep the pipeline busy. The partition space has three dimensions:

- primitive substitution, which rewrites a collective as other primitives;
- topology-aware group partitioning, for example into hierarchical collectives within and across nodes;
- workload partitioning, which splits the data into chunks.

The resulting pieces are then scheduled hierarchically, at the operation, layer and model levels of hybrid-parallel training.

Centauri is easiest to situate against FLUX, TileLink and Triton-distributed (see their entries):

- Centauri schedules decomposed operators on several streams, and improves the overlap between kernels that stay separate, so it needs no collectives inside kernels.
- FLUX, TileLink and Triton-distributed fuse computation and communication *inside* kernels. They can overlap at a finer grain, but they often need more specialized code generation or primitives.
- Centauri therefore fits workloads whose bottleneck is the *orchestration* (stream scheduling, dependencies, coarse kernels), while FLUX-like methods target the overheads that remain even with aggressive overlap across streams.

Across various hybrid-parallel training configurations, Centauri is up to **1.49×** faster than prevalent methods, and the paper received a Best Paper award at ASPLOS 2024. It also serves as a baseline for fused kernels: it shows the gain that better scheduling alone provides, without changing the kernels.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">FLUX: Fast Software-based Communication Overlap on GPUs Through Kernel Fusion</span><span class="pub-meta">, ByteDance & Peking University</span><a class="pub-link" href="https://arxiv.org/abs/2406.06858" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

FLUX is designed around the tensor-parallel patterns of Megatron-LM. A tensor-parallel layer usually runs one large GEMM and one large NCCL collective, one after the other. FLUX instead **decomposes** both operations into fine-grained pieces, such as tiles or chunks, and fuses the pieces into kernels that overlap communication with computation aggressively. This level of overlap is hard to reach when separate kernels overlap on several streams.

FLUX is easiest to understand in relation to three other works (see their entries):

- Centauri overlaps kernels that stay independent, while FLUX overlaps *within* the operator.
- Triton-distributed is a general programming model, while FLUX delivers peak performance for a *specific family* of LLM operations: the tensor-parallel attention and MLP of Megatron-LM.
- TileLink is motivated in part by the performance of FLUX, and it tries to keep this performance while raising the programming level.

FLUX overlaps up to about **96%** of the communication. In training, it is up to **1.24×** faster than Megatron-LM on a cluster of **128 GPUs**, across GPU generations and interconnects. In inference, on **8 GPUs**, it is up to **1.66×** faster than vLLM in the **prefill**, and **1.30×** faster in the **decoding**.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Optimizing Distributed ML Communication with Fused Computation–Collective Operations</span><span class="pub-meta">, AMD (SC 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2305.06942" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Punniyamurthy et al. identify common motifs of computation followed by a collective, which appear again and again in the graphs of distributed training and inference, and they implement each motif as a **single fused kernel** with **collectives inside the kernel**. The idea is to start communicating as early as possible: each workgroup sends its results to the remote GPUs as soon as its own computation finishes (GPU-initiated communication), while the other workgroups of the same kernel keep computing. Transfers within a node are [zero-copy](https://en.wikipedia.org/wiki/Zero-copy), as they write directly into the memory of the peer GPU. The fused kernels, written in Triton and HIP, thus remove the intermediate writes, the kernel boundaries and the separate communication steps. The contribution is partly a catalog of these motifs and partly a concrete engineering result.

The catalog helps to compare systems:

- FLUX focuses on the tensor-parallel patterns of LLMs, while this paper covers a **broader** set of motifs: embedding and all-to-all, GEMV and all-reduce, and GEMM and all-to-all.
- Triton-distributed and TileLink offer programming models to build such fused kernels, while this paper builds **instances by hand** and measures why the fusion matters.
- Unlike in-network computing (see the entry on in-network collective operations), these fused kernels still communicate through primitives on the GPUs, not in the network.

On average, the fused operators lower the execution time by 12% to 31%. Within a node of four GPUs, they lower it by 20% on average (up to 32%) for embedding and all-to-all, by 13% (up to 22%) for GEMV and all-reduce, and by 12% (up to 20%) for GEMM and all-to-all. Across nodes, embedding and all-to-all takes 31% less time on average (up to 58%). The gains are significant even for such simple two-stage pipelines, when the communication is tightly coupled to the computation.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSeek-V3 Technical Report (includes DualPipe)</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek-V3 is an MoE model with 671B parameters, of which 37B are active per token, trained on 14.8T tokens in 2.788M H800 GPU hours (about 5.6M USD). The report reads best as a reference on what training at this scale takes in practice, rather than as a single optimization. It covers the architecture (MLA and DeepSeekMoE from V2, auxiliary-loss-free load balancing, multi-token prediction), FP8 mixed-precision training, and the engineering trade-offs that papers on single algorithms often omit, which makes it particularly relevant to any discussion of communication-heavy structures such as MoE.

The training runs on 2,048 H800 GPUs, and its communication relies on custom all-to-all kernels across nodes, later open-sourced as DeepEP. These kernels use only 20 of the 132 SMs of each GPU, and they use InfiniBand and NVLink together by sending each token to at most 4 nodes. They overlap with computation at the level of the *schedule* (DualPipe, below), not inside the GEMM kernels or in the network. DeepSeek-V3 is therefore a strong end-to-end baseline for fused kernels and in-network computing: any gain they promise can be weighed against a system that already works at scale.

Three of its training techniques deserve more detail:

- **FP8 training.** DeepSeek-V3 is the first frontier-scale model trained in FP8. Earlier recipes use the format E4M3 in the forward pass and E5M2, which has a wider range, for the gradients, while DeepSeek-V3 uses E4M3 for all tensors. This works because of fine-grained scaling: activations are scaled per tile of 1×128 values and weights per block of 128×128 values, so an outlier only affects the scale of its own group. The scales are computed online from the current values, and the GEMMs accumulate in higher precision. The optimizer moments are kept in BF16, and a few sensitive operators stay in higher precision.
- **Sequential multi-token prediction.** Unlike the parallel heads of [Gloeckle et al.](https://arxiv.org/abs/2404.19737), each MTP module is a single Transformer layer that takes the hidden states of the previous depth and the embedding of the next token, so the causal chain is kept. The modules share the embedding and the output head with the main model, and each adds its own cross-entropy loss. DeepSeek-V3 uses one module, which can also serve as a draft model for speculative decoding: its second token is accepted 85% to 90% of the time, for about 1.8× more tokens per second.
- **Auxiliary-loss-free load balancing.** A bias per expert is added to the routing scores, but only to select the top-k experts, not to weight their outputs. After each step, the bias of the overloaded experts decreases and that of the underloaded experts increases. This balances the load without a loss term that competes with the language modeling objective. Only a very small sequence-wise balance loss remains, to avoid an extreme imbalance within a single sequence.

Post-training distills the reasoning ability of DeepSeek-R1, trained with large-scale reinforcement learning on top of DeepSeek-V3-Base, into DeepSeek-V3. R1 generates the reasoning data, and V3 is fine-tuned on it, which improves its math and code without taking on the length and style of a pure reasoning model.

**DualPipe** is a bidirectional pipeline-parallel schedule, introduced in the report. It addresses two problems at once: the pipeline bubbles, and the heavy communication of expert parallelism across nodes, which gives DeepSeek-V3 a computation-to-communication ratio of about 1:1. Its core idea is to overlap computation and communication within a pair of forward and backward chunks. Each chunk is split into four parts: attention, all-to-all dispatch, MLP and all-to-all combine, and, as in Zero Bubble (see its entry), the backward attention and MLP are further split into their input-gradient and weight-gradient parts. The parts are then rearranged so that one micro-batch runs forward while another runs backward, and the communication of each hides behind the computation of the other. Micro-batches also enter from both ends of the pipeline, which keeps the GPUs busier than sequential schedules such as 1F1B or Zero Bubble.

DualPipe thus has fewer bubbles, and it overlaps all the communication of the forward and backward passes. In exchange, it keeps two copies of the model parameters, one per direction, and slightly more activation memory. The copies are affordable because DeepSeek-V3 uses a large expert-parallel size, so each rank holds relatively few parameters. Further memory savings (recomputation, the EMA of the weights kept in CPU memory, and the embedding and output head shared with the MTP module) also let DeepSeek-V3 train without costly tensor parallelism. PipeFill (see its entry) accepts the bubbles and fills them with other jobs, while DualPipe removes them from the schedule of the main job, so the two approaches are complementary rather than competing. In the figure, the two directions start at opposite ends of the pipeline, so even the last GPU starts working right away, while in 1F1B it waits for the first micro-batch to cross all the stages.

{: .picture}
<img loading="lazy" width="86.8%" height="86.8%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipe.svg"/>

{: .caption}
DualPipe on 6 GPUs: micro-batches 1 to 6 (light) enter at GPU 0, and micro-batches 7 to 12 (dark) enter at GPU 5. Each GPU holds two pipeline stages. Cells with a shared black border are the forward and backward passes of two different micro-batches executed together, so that the communication of one overlaps with the computation of the other. Adapted from: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Zero Bubble Pipeline Parallelism</span><span class="pub-meta">, Sea AI Lab (ICLR 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2401.10241" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Zero Bubble is the first schedule that reaches zero pipeline bubbles while keeping synchronous training semantics. Its key insight is that the backward pass of a layer consists of two computations that do not depend on each other: the gradient with respect to the input of the layer (B), which the previous stage needs to continue the backward pass, and the gradient with respect to its weights (W), which only the optimizer step needs. Standard schedules run the two back to back, but nothing requires it. Decoupling B and W removes this sequential dependency, so the W passes can be delayed and used to fill the bubbles that a 1F1B schedule leaves at the end of each iteration. The paper gives two handcrafted schedules and an automatic scheduler, which formulates the schedule as an [integer linear program](https://en.wikipedia.org/wiki/Integer_programming), solved by an off-the-shelf solver, together with a heuristic for large numbers of micro-batches.

The two handcrafted schedules, shown in the figure, trade memory for bubbles. ZB-H1 keeps the peak activation memory of 1F1B and cuts its bubble to about a third, when the forward, B and W passes take the same time. ZB-H2 removes the bubbles entirely, at the cost of a higher peak memory, about twice the activations, and of an extra validation step in the optimizer. The optimizer step normally synchronizes all stages, for example to clip the gradients by their global norm or to check them for NaN values, and this synchronization would bring a bubble back. ZB-H2 therefore bypasses it: each stage updates its weights without waiting, and the update is validated afterwards, and rolled back if the check fails. The zero-bubble schedules reach up to 23% more throughput than 1F1B under a similar memory limit, and up to 31% more when the memory limit is relaxed. The method is orthogonal to data, tensor and ZeRO parallelism, and it replaces the pipeline-parallel component directly. Like DualPipe, it removes the bubbles from the schedule of the main job, while PipeFill fills them with other jobs (see their entries).

{: .picture}
<img loading="lazy" width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h1.svg"/>
<img loading="lazy" width="86.5%" height="86.5%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h2.svg"/>

{: .caption}
ZB-H1 (top) and ZB-H2 (bottom). Every backward pass is split into a backward for inputs and a backward for weights, and the latter fill the bubbles. ZB-H1 keeps the activation memory of 1F1B; ZB-H2 uses about twice as much, and when the optimizer step does not synchronize the stages, the next iteration starts on the first GPUs while the last GPUs finish the current one. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design</span><span class="pub-meta">, Microsoft & University of Sydney (USENIX ATC 2024, as Quant-LLM)</span><a class="pub-link" href="https://arxiv.org/abs/2401.14112" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

FP6-LLM makes six-bit floating-point (FP6) quantization practical for LLM inference on GPUs. FP6 is an appealing middle ground. It shrinks the model and the time to load its weights substantially, and it keeps the quality more reliably than 4-bit formats: 4-bit methods do well on zero-shot benchmarks, but they degrade on harder tasks, such as code generation and summarization, while 6-bit stays robust. The catch is that no system supported FP6 on Tensor Cores, for two reasons: the irregular 6-bit width causes misaligned memory accesses to the weights, and converting the weights back to a format that the Tensor Cores can multiply, the *dequantization*, adds a heavy runtime overhead. The paper solves both problems together with TC-FPx, the first full-stack GPU kernel design with unified Tensor Core support for floating-point weights of any bit width.

TC-FPx works as follows:

- The weights are FP6, with 1 sign bit, 3 exponent bits and 2 mantissa bits (**E3M2**), in a **W6A16** linear layer: 6-bit weights and FP16 activations. Only the weights are quantized, so the activations and the results of the matrix multiplications stay in FP16, and nothing is ever converted back to FP6.
- The weights are quantized from FP16 once, offline, and they stay in 6 bits in memory, which is the whole point: a smaller footprint and faster loads from DRAM. To keep the loads aligned, each 6-bit value is split into a 2-bit and a 4-bit fragment, and the fragments are packed ahead of time into an aligned layout.
- Just before the computation, the SIMT cores (CUDA cores) load a slice of the weights, join the fragments again with bit shifts and masks, and expand them to FP16 in registers. The Tensor Cores then multiply in FP16.
- The kernel is software-pipelined: the dequantization of slice $$N+1$$ on the CUDA cores overlaps with the matrix multiplication of slice $$N$$ on the Tensor Cores. The gain therefore comes from the memory bandwidth, as the kernel loads 6-bit weights instead of 16-bit ones, while the cost of the dequantization hides behind the Tensor Core math.

FP6-LLM integrates TC-FPx into an existing inference system, and this algorithm-system co-design is what turns the compression of FP6 into actual serving speedups. In FP6, the weights of LLaMA-70B take about 53 GB instead of 140 GB in FP16, so the model fits on a single 80 GB A100 GPU, with 1.69× to 2.65× the normalized inference throughput (tokens per GPU-second) of the FP16 baseline, which needs two such GPUs. At the kernel level, the 6-bit linear layer runs up to 2.6× faster than cuBLAS in FP16, and up to 1.9× faster than the 8-bit (W8A16) kernels of TensorRT-LLM. The source code is at [github.com/usyd-fsalab/fp6_llm](https://github.com/usyd-fsalab/fp6_llm).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">The Llama 3 Herd of Models</span><span class="pub-meta">, Meta</span><a class="pub-link" href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Llama 3 is a herd of language models with native support for multiple languages, coding, reasoning and tool use. The models have 8B, 70B and 405B parameters and a context of 128K tokens. The flagship, Llama 3 405B, has 126 layers, a model dimension of 16,384 and 128 attention heads, and it is trained on up to 16K H100 GPUs with 4D parallelism: tensor, pipeline, context and data parallelism.

How large should the flagship be for the available compute? The authors answer with scaling laws (Hoffmann et al., 2022; see the Chinchilla entry), in section 3.2.1. They train small models, from 40M to 16B parameters, with budgets between $$6 \times 10^{18}$$ and $$10^{22}$$ FLOPs. For each budget, the validation loss as a function of the number of training tokens traces an IsoFLOPs curve, and the minimum of the curve gives the compute-optimal model (left of the figure). The number of training tokens of these optimal models then follows a power law of the compute budget $$C$$, $$N^\star(C) = A C^{\alpha}$$, with $$\alpha = 0.537$$ and $$A = 0.299$$ (right of the figure). Extrapolated to the budget of the flagship, $$3.8 \times 10^{25}$$ FLOPs, the fit suggests a model of 402B parameters trained on 16.55T tokens, close to the final 405B model trained on 15.6T tokens. To predict the accuracy on downstream tasks, and not only the loss, the authors take two more steps: they first relate the negative log-likelihood of the compute-optimal model on the downstream tasks to the training FLOPs, and then fit a sigmoidal relation between that log-likelihood and the task accuracy (figure 4 of the paper).

{: .picture}
<img loading="lazy" width="74.9%" height="74.9%" src="{{ site.assets }}/publications/llama3_scaling_laws.svg"/>

{: .caption}
<b>Left:</b> <b>Scaling law IsoFLOPs curves</b> between $$6 \times 10^{18}$$ and $$10^{22}$$ FLOPs. The loss is the negative log-likelihood on a held-out validation set. Measurements at each compute scale are approximated by a second-degree polynomial. <b>Right:</b> <b>Number of training tokens in identified compute-optimal models as a function of pre-training compute budget.</b> The line is the fitted scaling-law prediction. The compute-optimal models correspond to the parabola minima of the left plot. Adapted from: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

The architecture follows Llama 2, with four changes:

1. grouped-query attention with 8 key-value heads, which speeds up inference and reduces the size of the KV cache in decoding;
2. an attention mask that prevents self-attention between different documents within the same sequence, which matters in continued pre-training on very long sequences;
3. a vocabulary of 128K tokens: 100K from `tiktoken` and 28K more for better non-English support;
4. a RoPE base frequency of 500,000, to better support long contexts.

Training has two stages: pre-training with next-token prediction, and post-training, where the model is "tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning)." Compared with Llama 2, the improvements come from three levers:

1. **data**: better quality, quantity, pre-processing and curation, with "15T multilingual tokens, compared to 1.8T tokens for Llama 2";
2. **scale**: almost $$50 \times$$ the pre-training compute of the largest Llama 2 model, with $$3.8 \times 10^{25}$$ FLOPs;
3. **managing complexity**: a standard Transformer with minor changes instead of a mixture of experts, and "a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO), as opposed to more complex reinforcement learning algorithms" (section 4).

The authors also experiment with image, video and speech capabilities, which they add in three more stages:

- **multi-modal encoder pre-training**: separate image and speech encoders (sections 7 and 8). The image encoder is trained on large amounts of image-text pairs, and the speech encoder with a self-supervised method that "masks out parts of the speech inputs and tries to reconstruct the masked out parts via a discrete-token representation";
- **vision adapter training**: an adapter, trained on text-image pairs, aligns the image representations with the language representations. A video adapter, trained on top of the image adapter on paired video-text data, then lets the model aggregate information across frames (section 7);
- **speech adapter training**: a third adapter converts the speech encodings into token representations.

The image encoder is a standard vision Transformer trained to align images and text, the ViT-H/14 variant. Cross-attention layers, which use grouped-query attention too, connect the visual tokens of the image encoder to the token representations of the language model, after every 4th self-attention layer. The figure shows the whole compositional design: the language model in the middle; the image encoder and the video aggregator on the left, whose representations enter the language model through the cross-attention layers; and the speech encoder on the right, whose adapter feeds the token embeddings directly.

{: .picture}
<img loading="lazy" width="99.4%" height="99.4%" src="{{ site.assets }}/publications/llama_3_multi_modal.svg"/>

{: .caption}
<b>Illustration of the compositional approach to adding multimodal capabilities to Llama 3 studied in the paper.</b> This approach leads to a multimodal model that is trained in five stages: <b>(1)</b> language model pre-training, <b>(2)</b> multi-modal encoder pre-training, <b>(3)</b> vision adapter training, <b>(4)</b> model finetuning, and <b>(5)</b> speech adapter training. Adapted from: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

Section 5 evaluates "(1) the pre-trained language model, (2) the post-trained language model, and (3) the safety characteristics of Llama 3". Section 6 makes the inference of Llama 3 405B efficient with two techniques. The first one is pipeline parallelism on 16 H100 GPUs in BF16: the weights alone take 810 GB in BF16, more than the 640 GB of a machine with 8 H100 GPUs. The second one is FP8 quantization, which applies to most parameters and activations of the feed-forward networks, but not to the parameters of the self-attention layers. As in Xiao et al. (2024b), the scaling factors are dynamic, for better accuracy, with an upper bound of 1200. The first and last Transformer layers are not quantized, and the quantization is row-wise: the scaling factors are computed per row of the parameter and activation matrices.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Universal Checkpointing: Efficient and Flexible Checkpointing for Large Scale Distributed Training</span><span class="pub-meta">, Microsoft DeepSpeed</span><a class="pub-link" href="https://arxiv.org/abs/2406.18820" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Large training runs save their state in distributed checkpoints, in which each rank writes its own shards of the parameters and of the optimizer states, cut according to the parallelism layout of the run. State-of-the-art distributed checkpointing therefore assumes a static allocation of GPU resources at the start of training, and it cannot resume with a different parallelism strategy or hardware configuration: when the hardware changes during training, the run cannot simply continue. Universal Checkpointing (UCP) creates checkpoints efficiently and resumes them on any parallelism strategy. This brings two benefits: a better resilience to hardware failures, since training can continue on the remaining healthy hardware, and a shorter training time, by using elastic capacity whenever it becomes available.

UCP relies on two components. The universal checkpoint format holds a consolidated representation of each model parameter, with metadata for "mapping parameter fragments into training ranks of arbitrary model-parallelism configuration". The universal checkpoint language is a simple specification language that converts distributed checkpoints into this format. In practice, the format consolidates the distributed shards into *atom checkpoints*, one per model parameter, whose files hold the FP32 weights and the optimizer states (for Adam, the first and second moments). An atom checkpoint no longer depends on the layout of the run that wrote it, so it can be re-sharded onto any new layout: ZeRO data parallelism, tensor, pipeline and sequence parallelism.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping</span><span class="pub-meta">, Microsoft DeepSpeed</span><a class="pub-link" href="https://arxiv.org/abs/2409.15241" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Domino hides the communication of tensor parallelism (TP) behind computation, in the training of large LLMs. Existing methods overlap the TP communication with computation only in part: when a collective takes much longer than a single GEMM, most of the communication time still stands out as the major overhead of training, and as the computation of the latest GPUs becomes faster, this overhead only grows. The answer of Domino is generic: it breaks the data dependencies of the training of a single batch into smaller independent pieces, and pipelines these pieces, so that communication and computation overlap at a fine grain. The overlap also covers a much wider scope than a single GEMM: an all-reduce can overlap with LayerNorm, dropout and other operations too.

In Megatron-LM, a TP block computes $$X \cdot A \cdot B$$, with the first weights $$A$$ split by columns and the second weights $$B$$ split by rows across the GPUs, followed by an all-reduce of the partial outputs. On every GPU, Domino adds a generic tensor partition in two dimensions on top of this one: a row-wise split of the inputs $$X$$, along the batch dimension, and a column-wise split of the weights $$B$$. Either split breaks $$X \cdot A \cdot B$$ into smaller compute units without data dependencies, which are then pipelined with the collectives, while $$A$$ is left untouched. Domino slices the input $$X$$ (section 3.2 of the paper), the weights $$B$$ (section 3.3), or both (section 3.4).

The figure shows the two splits. On the left, $$X$$ is split into $$X_1$$ and $$X_2$$: the all-reduce of the first output $$Y_1$$ runs while the GPU computes $$X_2 \cdot A \cdot B$$, an overlap within the layer, and the all-reduce of $$Y_2$$ runs while the GPU starts the next layer, an overlap across layers. On the right, $$B$$ is split into $$B_1$$ and $$B_2$$: the all-reduce of $$Y_3$$ overlaps with the computation of $$X \cdot A \cdot B_2$$, but the two outputs must be concatenated before the next layer starts, so only the overlap within the layer remains. Compared with Megatron-LM, Domino trains LLMs up to 1.3× faster on NVIDIA DGX-H100 GPUs.

{: .picture}
<img loading="lazy" width="99.6%" height="99.6%" src="{{ site.assets }}/publications/domino.svg"/>

{: .caption}
**Left:** Domino row-wise (batch-dim) split on input <i>X</i>. **Right:** Domino col-wise (last-dim) split on weights <i>B</i>. Adapted from: [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](https://arxiv.org/abs/2409.15241)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2023</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Simplifying Transformer Blocks</span><span class="pub-meta">, ETH Zurich (ICLR 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2311.01906" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The standard Transformer block has many parts, and this paper asks which of them are necessary. Guided by signal propagation theory, it simplifies the block step by step, and it removes the skip connections, the value and projection parameters, the sequential sub-blocks (attention and MLP run in parallel) and the normalization layers. Without its skip connection, attention tends to collapse the representations of all tokens into the same vector as the network gets deeper, so the simplified block uses the *shaped attention* of [Noci et al. (2023)](https://arxiv.org/abs/2306.17759), which adds a trainable multiple of the identity to the attention matrix and subtracts a centering term, so that the attention matrix starts as the identity and, at first, each token attends only to itself. The value and projection matrices, two consecutive linear maps without a nonlinearity between them, converge to the identity in most layers anyway, so the paper fixes them to the identity.

The figure compares the standard Pre-LN block (left) with the parallel block (bottom right), which already runs attention and MLP side by side, and with the most simplified block of the paper (top right), which keeps only the queries, the keys and the MLP. This block matches the training speed per update and the performance of standard autoregressive decoder-only models and BERT encoder-only models, with a 16% higher training throughput (the abstract on the arXiv page says 15%) and 15% fewer parameters. The experiments use models of 100M to 300M parameters.

{: .picture}
<img loading="lazy" width="63.9%" height="63.9%" src="{{ site.assets }}/publications/simplifying_transformer_blocks.svg"/>

{: .caption}
Comparison between different Transformer blocks. (Left) The standard Pre-LN block. (Top right) The most simplified block of the paper, “SAS-P, no norm”. (Bottom right) The parallel block (Zhao et al., 2019; Wang & Komatsuzaki, 2021). Like the parallel block, the simplified block eschews the need for sequential sub-blocks, but it additionally removes all skip connections and normalisation layers, as well as value and projection parameters. Here, ⊗ denotes a matrix multiplication, and ⊕ denotes a (potentially weighted) sum. Adapted from: [Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Training and inference of large language models using 8-bit floating point</span><span class="pub-meta">, Graphcore</span><a class="pub-link" href="https://arxiv.org/abs/2309.17224" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper "presents a methodology to select the scalings for FP8 linear layers, based on dynamically updating per-tensor scales for the weights, gradients and activations." Scaling is the central difficulty of FP8: the format has so few bits that a tensor keeps its accuracy only if it is first shifted into the representable range. The paper uses two FP8 formats, FP8-E4 and FP8-E5, with 4 and 5 exponent bits: E4 for the weights and the activations, and E5 for the gradients, which need a wider dynamic range. Despite the name, not all the arithmetic is 8-bit, as the partial results of the FP8 matrix multiplications accumulate in FP16 to prevent overflows. The scale is applied to the exponent, as an integer bias added to it, which is the same as a multiplication by a power of two.

The paper compares two scaling methods. AMAX derives the scale of each tensor from its absolute maximum, updated dynamically, while CSCALE keeps one constant scale for all tensors, found by a sweep. Both can match the accuracy of FP16. CSCALE, however, works only within a narrow range of scale values, and in fine-tuning this range shrinks as the models grow, until convergence is no longer guaranteed for the 6.7B and 13B models. AMAX matches FP16 at all sizes, so it is the more robust choice. The results compare FP8 with FP16, not with BF16, on GPT and Llama 2 models of 111M to 70B parameters. Figures 1 and 2 of the paper give the pseudocode for training and inference, and Figure 3 compares the forward pass of an FP16 and an FP8 linear layer.

The experiments ran on Graphcore IPUs without native FP8 support, so FP8 was simulated in software. On hardware that supports FP8, its matrix multiplications run at about twice the FP16 or BF16 throughput, and the per-tensor scaling adds only cheap element-wise work, so the end-to-end speedup is below 2×, but still significant.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSpeed ZeRO-Offload++: 6x Higher Training Throughput via Collaborative CPU/GPU Twin-Flow</span><a class="pub-link" href="https://github.com/microsoft/DeepSpeed/tree/offloadpp-news/blogs/deepspeed-offloadpp" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

ZeRO-Offload keeps the gradients and the optimizer states in CPU memory, and runs the optimizer step on the CPU. CPU memory is often orders of magnitude larger than GPU memory, so this lets models with billions of parameters train even on a single GPU. The DeepSpeed blog notes, however, that the "system efficiency is still far from optimal" in two cases: training with small batches, and models that do not fit in GPU memory but are not orders of magnitude larger either. In these cases, the offload adds a long end-to-end latency and leaves the GPU compute underused. ZeRO-Offload++ addresses this with three features:

- **Twin-Flow** replaces the all-or-nothing offload of the optimizer states. A fraction of the optimizer states stays in CPU memory and the rest in GPU memory, and during the optimizer step, the CPU and the GPU update their parameters at the same time, with the same optimizer. The user sets the fraction of the parameters that the CPU updates, the offload ratio. The smaller the ratio, the more GPU memory and cores are used and the shorter the iteration, so the best ratio fills the GPU memory almost to its limit.
- **MemCpy reduction** and an optimized **CPUAdam**, which the blog lists as ongoing work: fewer unnecessary copies between the device and the host, and an on-the-fly cast from FP16 to FP32 in CPUAdam.

The figure shows the split of Twin-Flow for a ratio of 0.4. With a tuned offload ratio, the blog reports training speedups over ZeRO-Offload of 6× for OPT-30B on a DGX-H100 with 80 GB GPUs, and of 3× for OPT-13B on a DGX-A100 with 40 GB GPUs.

{: .picture}
<img loading="lazy" width="62.1%" height="62.1%" src="{{ site.assets }}/publications/zerooffloadplusplus.svg"/>

{: .caption}
Twin-Flow with an offload ratio of 0.4: CPUAdam updates the first 40% of the parameters (weights, gradients and optimizer states) on the CPU, while FusedAdam updates the other 60% on the GPU. Adapted from: [DeepSpeed ZeRO-Offload++: 6x Higher Training Throughput via Collaborative CPU/GPU Twin-Flow](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-offloadpp)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DistFlashAttn: Distributed Memory-efficient Attention for Long-context LLMs Training (LightSeq)</span><span class="pub-meta">, UC Berkeley et al.</span><a class="pub-link" href="https://arxiv.org/abs/2310.03294" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DistFlashAttn extends FlashAttention (see its entry) from one GPU to the sequence-parallel **training** of long-context LLMs. FlashAttention computes attention tile by tile with an online softmax and never stores the full matrix of scores, so its memory grows linearly with the sequence length instead of quadratically. DistFlashAttn splits the sequence across the GPUs, not the heads or the hidden dimension: each GPU holds the Q, K and V of its own chunk of tokens, with all the heads and the full hidden dimension, only fewer rows. The challenge is to do this efficiently under *causal* attention, and the paper contributes three techniques:

1. **Token-level load balancing.** With a causal mask, a token attends only to the tokens before it, so the work of a chunk grows with its position in the sequence. In a naive sequence split, the GPUs that hold the early tokens therefore sit idle while the GPUs that hold the late tokens compute. In DistFlashAttn, an idle GPU fetches a block of queries from a busy GPU and computes their attention against its own keys and values. It then returns the partial output with its softmax statistics, and the owner of the queries merges them, in the same way as FlashAttention merges the results of two blocks.
2. **Overlap of the KV communication.** As in Ring Attention, the K and V chunks move between the GPUs, so that each GPU can attend to remote tokens. DistFlashAttn overlaps these transfers with the local attention computation, so that their latency hides behind useful work.
3. **Rematerialization-aware gradient checkpointing.** Standard gradient checkpointing places the checkpoints at the layer boundaries, and recomputes the forward pass of each layer during the backward pass. The backward pass of FlashAttention already recomputes the attention, however, so the attention forward pass would run twice. DistFlashAttn places the checkpoints at the output of the FlashAttention kernel instead, which saves one attention forward pass per layer.

The comparison with Ring Attention explains the design. Ring Attention is a general sequence-parallel attention, for training and inference, with or without a causal mask. DistFlashAttn targets causal long-context training only, the setting where the load imbalance and the cost of checkpointing matter most, so it is in essence Ring Attention with three fixes for that setting. Unlike Megatron-LM and DeepSpeed-Ulysses, which split the attention by heads, its degree of parallelism is not limited by the number of heads. The tests use Llama-7B and variants, with 32K to 512K tokens. DistFlashAttn trains on sequences 8× longer than Ring Self-Attention, at 4.45× to 5.64× its speed, and on sequences 2× to 8× longer than Megatron-LM with FlashAttention, at 1.24× to 2.01× its speed. It is also 1.67× faster than Ring Attention and 1.26× to 1.88× faster than DeepSpeed-Ulysses.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion</span><span class="pub-meta">, Peking University et al. (HPCA 2023)</span><a class="pub-link" href="https://ieeexplore.ieee.org/document/10071018" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Chimera (Zheng et al.) is a compiler framework that fuses *chains of compute-intensive operators*, such as two GEMMs in a row, into single kernels, to improve their [data locality](https://en.wikipedia.org/wiki/Locality_of_reference). The problem it addresses comes from the hardware. Compute throughput has grown faster than memory bandwidth, so even operators that are usually compute-bound, such as GEMMs and convolutions, become memory-bound when they run as a chain: each operator writes its full output to memory, only for the next operator to read it back. The ML compilers of the time had neither a precise analysis nor a good optimization of these chains on different accelerators, so much of the performance was left unused. The key idea of Chimera is to split each operator into a series of *computation blocks*, and to optimize at two levels:

- **inter-block**: an analytical model picks the execution order of the blocks that minimizes the data movement between them, so that the intermediate results stay in fast on-chip memory instead of making a round trip to DRAM;
- **intra-block**: a hardware-specific micro-kernel computes each block efficiently.

The paper applies it to three kinds of chains:

- **GEMM chain** $$C = AB$$, $$E = CD$$: each tile of $$C$$ is used to compute $$E$$ as soon as it is produced, so $$C$$ is never stored in full. A tile of $$E$$ needs a whole block of rows of $$C$$, so with one tile of $$E$$ per thread block, the thread blocks that work on the same rows would either compute the same tiles of $$C$$ several times, or share the work and add their partial products into $$E$$ with atomic operations, which causes contention. Instead, each thread block takes a full block of rows of $$A$$ and $$E$$: it produces the tiles of $$C$$ of that row block one at a time, multiplies each of them by the matching rows of $$D$$ right away, and in this way iterates over all the columns of $$B$$ and $$D$$.
- **Batch GEMM chain** (attention): $$QK^T$$, a softmax, then the product with $$V$$, in other words two batch GEMMs with a softmax between them.
- **Convolution chain** (CNNs): a 3×3 convolution, a ReLU, a 1×1 convolution and a ReLU, fused into one kernel. This chain is common in ResNet-style networks, and it is memory-bound for some input shapes.

Because Chimera relies on an analytical cost model rather than an exhaustive search, it avoids the long tuning times of search-based compilers such as [Ansor](https://arxiv.org/abs/2006.06762). On batch GEMM chains and convolution chains, it is up to 2.87×, 2.29× and 2.39× faster than hand-tuned libraries on CPU, GPU (A100) and NPU (Huawei Ascend), and up to 2.29×, 1.64× and 1.14× faster than state-of-the-art compilers on the same platforms. The same framework thus serves CPUs, GPUs with Tensor Cores and NPUs.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">ZeRO++: Extremely Efficient Collective Communication for Giant Model Training</span><span class="pub-meta">, Microsoft</span><a class="pub-link" href="https://arxiv.org/abs/2306.10209" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSpeed ZeRO shards the model states across the GPUs, so it must gather the weights in the forward and in the backward pass and average the gradients. This communication limits its throughput, most of all on clusters with low bandwidth and with small batches per GPU, where little computation is available to hide it. For a model of size $$M$$, ZeRO-3 communicates three times per step. In the forward pass, an all-gather collects the parameters of all the layers ($$M$$). In the backward pass, a second all-gather collects them again ($$M$$), so that each GPU can compute its local gradients, and a reduce-scatter then aggregates the gradients and redistributes them across the GPUs ($$M$$). The total volume is $$3M$$, spread evenly over two all-gathers and one reduce-scatter. ZeRO++ reduces each of the three with its own technique:

1. **Quantized weight communication (qwZ)**: the forward all-gather sends the weights in INT8 (1 byte) instead of FP16 (2 bytes), which halves its volume. The usual quantization multiplies all the parameters by a single scale, set by the largest value of the tensor, so the quantization steps are coarse for all the other values. qwZ quantizes each block of parameters on its own instead, with a scale and a shift (a zero point) per block, which reduces the quantization error, as the figure below shows.
2. **Hierarchical weight partition (hpZ)** trades memory for communication. Each machine keeps a full FP16 copy of the weights, partitioned across its own GPUs, so the all-gather of the backward pass runs over the fast links inside each machine instead of across machines.
3. **Quantized gradient communication (qgZ)** replaces the reduce-scatter of the gradients. The gradients are quantized to INT4 in blocks, to reduce the size of the messages, then sent with an all-to-all, and dequantized to full precision before the reduction, which preserves the training accuracy. The all-to-all runs in two steps, first inside each machine and then across machines, so that only the partly reduced gradients leave a machine. The paper also considered a simpler design, a quantized version of the usual ring reduce-scatter, but rejected it: every GPU of the ring would quantize and dequantize the data once, in sequence, and the reductions in low precision would accumulate and amplify the errors.

Together, the three techniques reduce the volume across machines from $$3M$$ to $$0.5M + 0 + 0.25M = 0.75M$$, a 4× reduction compared with ZeRO-3, which gives up to 2.16× higher throughput on 384 GPUs.

{: .picture}
<img loading="lazy" width="65.6%" height="65.6%" src="{{ site.assets }}/publications/zeroplusplus.svg"/>

{: .caption}
Basic quantization (top) and block quantization (bottom) of the same 3×3 weights in qwZ: a scale and a zero point per block (here, per row) keep the dequantized values (right) closer to the input (left); the digits that changed are orange. Adapted from: [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">QLoRA: Efficient Finetuning of Quantized LLMs</span><span class="pub-meta">, University of Washington</span><a class="pub-link" href="https://arxiv.org/abs/2305.14314" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

QLoRA reduces the memory of finetuning so much that a 65B-parameter model fits on a single 48 GB GPU, while it keeps the task performance of full 16-bit finetuning, which needs more than 780 GB of GPU memory for LLaMA-65B. The pretrained model stays frozen and quantized to 4 bits, and the gradients are backpropagated through it into Low-Rank Adapters (LoRA, see its entry), the only weights that are trained. Three techniques reduce the memory without losing performance:

1. **4-bit NormalFloat (NF4)**, a data type that the paper calls information-theoretically optimal for normally distributed data, and that gives better results than 4-bit integers and 4-bit floats. NF4 rounds each value to the nearest of 16 levels placed at the [quantiles](https://en.wikipedia.org/wiki/Quantile) of a standard normal distribution, so that each level receives about the same share of the values. In general, quantile quantization is expensive: it needs an approximate estimation of the quantiles of each tensor, such as SRAM quantiles, which also gives large errors for outliers. NF4 avoids this cost because pretrained weights roughly follow a zero-centred normal distribution, so the quantiles are computed once, and each block of 64 weights is only rescaled by its absolute maximum.
2. **Double quantization** quantizes the quantization constants themselves. With one FP32 constant per block of 64 weights, the constants cost $$32/64 = 0.5$$ bits per parameter. Stored in FP8 instead, with one FP32 constant for every block of 256 of them, they cost $$8/64 + 32/(64 \cdot 256) \approx 0.127$$ bits per parameter. This saves about 0.37 bits per parameter, or about 3 GB for a 65B model.
3. **Paged optimizers** use NVIDIA unified memory to avoid the memory spikes of gradient checkpointing on mini-batches with long sequences. The optimizer states are allocated in paged memory, which moves them to CPU RAM when the GPU runs out of memory, and back to the GPU when the optimizer update needs them.

The paper used QLoRA to finetune more than 1,000 models, and found that finetuning on a small, high-quality dataset gives state-of-the-art results, even with smaller models than the previous state of the art. Its best model family, Guanaco, reaches 99.3% of the performance of ChatGPT on the Vicuna benchmark after 24 hours of finetuning on a single GPU.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Llama 2: Open Foundation and Fine-Tuned Chat Models</span><span class="pub-meta">, Meta</span><a class="pub-link" href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Llama 2 is a family of pretrained and fine-tuned LLMs with 7B to 70B parameters, pretrained on 2T tokens, and Llama 2-Chat is its version fine-tuned for dialogue. The models outperform the open-source chat models on most benchmarks, and based on human evaluations of helpfulness and safety, the authors argue that they may be a suitable substitute for closed-source models (Figure 3 of the paper gives the results of the human safety evaluation of Llama 2-Chat). The training data come only from publicly available sources, without data from the products or services of Meta, and without sources that may include personal information of users. Table 2 of the paper gives the GPU hours, the power consumption and the carbon emissions of each model.

The pretraining setting and most of the architecture come from LLaMA (see its entry): [byte-pair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) (BPE), pre-normalization with RMSNorm, SwiGLU activations, rotary positional embeddings, the AdamW optimizer with $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$ and $$\epsilon = 10^{-5}$$, and a cosine learning rate schedule. The architecture differs in two main points: the context is longer, with 4K tokens instead of 2K, and the 34B and 70B models use **grouped-query attention (GQA)**, which shares each key and value head among a group of query heads and so makes inference cheaper.

The fine-tuning combines supervised fine-tuning (Section 3.1), initial and iterative reward modeling (Section 3.2.2), and [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) (Section 3.2.3). RLHF had a drawback, however: the initial RLHF models tended to forget the initial instruction after a few turns of dialogue, as in the left dialogue of the figure below. **Ghost Attention (GAtt)**, a simple method inspired by Context Distillation (Bai et al., 2022b), addresses this by changing the fine-tuning data, so that the attention stays focused on the instruction over several turns. A system instruction, such as "act as ...", is first concatenated to all the user messages of a multi-turn dialogue, and the responses are sampled with the latest RLHF model, which follows the instruction because it sees it in every turn. For fine-tuning, the instruction is then removed from all the turns but the first, and the loss is set to zero on the tokens of the previous turns. In this way, the model learns to keep following the instruction across turns, as in the right dialogue of the figure.

{: .picture}
<img loading="lazy" width="77%" height="77%" src="{{ site.assets }}/publications/llama2_gatt.svg"/>

{: .caption}
<b>Issues with multi-turn memory (<i>left</i>) can be improved with GAtt (<i>right</i>).</b> Both dialogues start with the instruction “Always answer with emojis”. Adapted from: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">LLaMA: Open and Efficient Foundation Language Models</span><span class="pub-meta">, Meta</span><a class="pub-link" href="https://arxiv.org/abs/2302.13971" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

LLaMA is a family of LLMs with 7B to 65B parameters, trained only on publicly available datasets. Even so, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B. The difference lies in the training recipe. The compute-optimal recipe of Chinchilla gives the best model for a fixed training budget, but LLaMA optimizes for the cost of inference instead: for a target performance, a smaller model trained on more tokens costs more to train, but less to run. Table 1 of the paper lists the pretraining datasets, and Table 2 the training hyperparameters. The text is tokenized with BPE into about 1.4T tokens, and the 7B and 13B models train on 1.0T of them, the 33B and 65B models on all 1.4T.

The architecture improves the original Transformer in three ways, each one taken from an earlier model:
- **Pre-normalization** (from GPT-3): RMSNorm normalizes the input of each transformer sub-layer instead of its output, which improves the training stability.
- **SwiGLU activation** (from PaLM) replaces ReLU and improves the performance, with a hidden dimension of $$\frac{2}{3} 4d$$ instead of the $$4d$$ of PaLM. The reason for this dimension is that SwiGLU has three weight matrices instead of two, so the smaller hidden dimension keeps the parameter count of a standard feed-forward layer with hidden dimension $$4d$$: $$3 \cdot d \cdot \frac{2}{3} 4d = 2 \cdot d \cdot 4d$$.
- **Rotary embeddings** (from GPT-Neo): rotary positional embeddings (RoPE) at each layer of the network replace the absolute positional embeddings.

The training uses the AdamW optimizer with $$\beta_1 = 0.9$$ and $$\beta_2 = 0.95$$, and a cosine learning rate schedule with 2,000 warmup steps and a final learning rate of 10% of the maximum, together with a weight decay of 0.1 and a gradient clipping of 1.0. Several optimizations make this training efficient:
- an efficient causal multi-head attention, which does not store the attention weights and does not compute the key/query scores that the causal mask of language modeling removes;
- selective activation checkpointing, which saves the activations that are expensive to compute, such as the outputs of the linear layers, and recomputes only the others in the backward pass. This needs a manual implementation of the backward pass of the transformer layers, instead of PyTorch autograd, and, to benefit fully from it, model and sequence parallelism (Korthikanti et al., 2022), which reduce the memory use of the model;
- an overlap of the computation of the activations with the communication between GPUs over the network (the all-reduce operations), which hides the latency of the communication.

With these optimizations, the 65B model processes about 380 tokens per second per GPU on 2,048 A100 GPUs with 80 GB of memory, so one pass over the 1.4T tokens takes about 21 days.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Retentive Network: A Successor to Transformer for Large Language Models</span><span class="pub-meta">, Microsoft and Tsinghua University</span><a class="pub-link" href="https://arxiv.org/abs/2307.08621" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

RetNet replaces the multi-head attention of the Transformer with a *multi-scale retention* mechanism. It aims at three properties that earlier architectures did not have together: parallel training, low-cost inference and good performance. Transformers train in parallel, but their cost per generated token grows with the sequence, while recurrent networks generate at a constant cost per token, but train sequentially. Retention combines both, with three forms of computation:
- a **parallel** form for training, which uses the GPUs fully;
- a **recurrent** form for inference, at a cost of $$O(1)$$ per token. Compared with a Transformer, it improves the decoding throughput by 8.4×, the latency by 15.6× and the GPU memory by 3.4×, without a loss of performance (Figure 1 of the paper);
- a **chunkwise recurrent** form, which models long sequences at linear complexity: each chunk is encoded in parallel, which is fast, while the chunks are summarized recurrently, which saves GPU memory.

RetNet is a stack of $$L$$ identical blocks, with the layout of a Transformer: residual connections and pre-LayerNorm. Each block has two modules, a multi-scale retention (MSR) module and a feed-forward network (FFN) module, and the model encodes the sequence in an autoregressive way. The input $$X_0$$ has the shape (sequence length × hidden size), and layer $$l$$ turns its input $$X_l$$ into the contextualized representations $$X_{l+1}$$. A retention head can be computed in three equivalent forms:
1. The **parallel representation**, $$\mathrm{Retention}(X) = (Q K^\intercal \odot D) V$$, is similar to attention, but without the softmax, and with an extra causal decay matrix $$D$$, where $$D_{nm} = \gamma^{n-m}$$ for $$n \ge m$$ and 0 otherwise (Eq. 5). The decay $$\gamma < 1$$ gives less weight to the tokens further in the past, and the zeros above the diagonal act as the causal mask. This form suits parallel training.
2. The **recurrent representation** writes the same computation as a [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN), which suits inference: $$S_n = \gamma S_{n-1} + K_n^\intercal V_n$$ and $$\mathrm{Retention}(X_n) = Q_n S_n$$, where the state $$S_n$$ depends on the previous state $$S_{n-1}$$. Unrolling the recurrence gives back the parallel form, since $$Q_n S_n = \sum_{m \le n} \gamma^{n-m} (Q_n K_m^\intercal) V_m$$ is row $$n$$ of $$(Q K^\intercal \odot D) V$$.
3. The **chunkwise recurrent representation** combines the two, to accelerate training on long sequences. The input sequence is split into chunks: inside each chunk, the computation uses the parallel representation, and the information passes from chunk to chunk with the recurrent representation.

The costs explain the use of each form. For a sequence of $$N$$ tokens, the parallel form costs $$O(N^2)$$, like attention. The recurrent form keeps a state $$S_n$$ of fixed size, $$d \times d$$ per head, so a decoding step costs the same at every position, while the KV cache of a Transformer grows with the sequence.

Each layer has $$h = d_{model}/d$$ retention heads, where $$d$$ is the head dimension. Each head has its own matrices $$W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d}$$ and its own fixed decay $$\gamma$$, so the heads retain the past over different time scales, which gives the mechanism its name. The outputs of the heads are normalized with GroupNorm. A layer $$l$$ then computes $$Y_l = \mathrm{MSR}(\mathrm{LN}(X_l)) + X_l$$ and $$X_{l+1} = \mathrm{FFN}(\mathrm{LN}(Y_l)) + Y_l$$, which is a Transformer layer with retention in place of attention.

{: .picture}
<img loading="lazy" width="63%" height="63%" src="{{ site.assets }}/publications/retnet.svg"/>

{: .caption}
Dual form of RetNet: <b>(a)</b> parallel representation and <b>(b)</b> recurrent representation. “GN” is short for GroupNorm. Adapted from: [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)

A simpler summary of RetNet is in this [video](https://www.youtube.com/watch?v=JaIL1VAEwZ8).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Operator Fusion in XLA: Analysis and Evaluation</span><span class="pub-meta">, UToronto</span><a class="pub-link" href="https://arxiv.org/abs/2301.13062" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Kernel fusion is one of the most important optimizations of [XLA](https://www.tensorflow.org/xla), the compiler behind TensorFlow and JAX. This paper, a course-project report, explains how XLA works and studies its key compiler passes in the source code. It then measures the speedup that kernel fusion gives, and its low-level effects on the hardware. A case study on the CartPole reinforcement-learning environment, written in JAX, shows how XLA makes its fusion decisions in practice, and the fusion strategies that the paper implements run up to 10.56× faster than its baseline implementation.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">LongNet: Scaling Transformers to 1,000,000,000 Tokens</span><span class="pub-meta">, Microsoft and Xi’an Jiaotong University</span><a class="pub-link" href="https://arxiv.org/abs/2307.02486" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

LongNet is a Transformer variant that scales the sequence length to more than 1 billion tokens, without a loss of performance on shorter sequences. Earlier methods for long sequences had to trade the computational complexity against the expressivity of the model, which limited the sequence length. LongNet's answer is **dilated attention**. It splits the sequence into segments of length $$w$$, keeps every $$r$$-th token of each segment, where $$r$$ is the dilation rate, and computes the attention among the kept tokens only. For $$N$$ tokens of dimension $$d$$, there are $$N/w$$ segments of $$w/r$$ kept tokens each, so one pair $$(w, r)$$ costs $$O(N w d / r^2)$$, instead of the $$O(N^2 d)$$ of dense attention. A single pair would either miss the long-range dependencies or skip most of the nearby tokens, so LongNet mixes several pairs $$(w, r)$$ that grow geometrically: short segments with a small dilation cover the nearby tokens, and long segments with a large dilation cover the distant ones. As a result, the attention given to a pair of tokens decreases exponentially with their distance, the cost is linear in the sequence length, and the dependency path between any two tokens has a logarithmic length. The sequence dimension can also be distributed across devices, to train on extremely long sequences.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning</span><a class="pub-link" href="https://arxiv.org/abs/2307.08691" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

FlashAttention (see its entry) computes exact attention without storing the attention matrix, but it reaches only 25–40% of the theoretical maximum FLOPs/s of an A100, far below optimized GEMM operations. The cause is a suboptimal partition of the work between the thread blocks and the warps of the GPU, which gives either a low occupancy or unnecessary reads and writes of shared memory. FlashAttention-2 improves the partition in three ways:
1. It changes the algorithm to reduce the non-matmul FLOPs, for example by rescaling the output once at the end of the loop instead of after every block. These FLOPs matter more than their count suggests: on an A100, FP16 matmuls run at up to 312 TFLOPs/s, but non-matmul FP32 operations at only 19.5 TFLOPs/s, so each non-matmul FLOP costs 16× more than a matmul FLOP.
2. It parallelizes the attention of a single head across thread blocks, over blocks of the sequence, and not only over the batch and the heads as FlashAttention does. This increases the occupancy for long sequences, where the batch size or the number of heads is often small.
3. Inside each thread block, it splits the queries across the warps, instead of the keys and values. Each warp then computes its own rows of the output, and the warps do not need to exchange partial results through shared memory.

Together, these changes give about 2× the speed of FlashAttention, and 50–73% of the theoretical maximum FLOPs/s on A100, close to the efficiency of GEMM. In the end-to-end training of GPT-style models, FlashAttention-2 reaches up to 225 TFLOPs/s per A100 GPU, a model FLOPs utilization of 72%.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Flow Matching for Generative Modeling</span><span class="pub-meta">, Meta AI & Weizmann Institute (ICLR 2023)</span><a class="pub-link" href="https://arxiv.org/abs/2210.02747" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Flow Matching is a simulation-free method to train [continuous normalizing flows](https://en.wikipedia.org/wiki/Flow-based_generative_model) (CNFs): training never solves the ODE of the flow. Instead, a network $$v_\theta$$ learns, by regression, a vector field that moves samples from noise to data. The field that moves the whole noise distribution to the whole data distribution is intractable, so the paper defines a target field per data sample $$x_1$$, which makes the loss tractable, and shows that regressing these conditional fields gives the same gradients. The method works with a general family of Gaussian probability paths between noise and data, which includes the paths of the reverse process of diffusion models. Furthermore, it opens the door to paths that do not come from diffusion. The paths of [optimal transport](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)) (OT) displacement interpolation are of particular interest: each one is a straight line from a noise sample to a data sample, and they give the loss

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, x_0, x_1} \left\lVert v_\theta(x_t, t) - \big(x_1 - (1 - \sigma_{min}) x_0\big) \right\rVert^2, \qquad x_t = \big(1 - (1 - \sigma_{min}) t\big) x_0 + t x_1,
$$

with $$t \sim \mathcal{U}[0, 1]$$, noise $$x_0 \sim \mathcal{N}(0, I)$$, data $$x_1$$, and a small $$\sigma_{min}$$. The target is the velocity of $$x_t$$ along its straight path, and it does not depend on $$t$$. The OT paths are more efficient than diffusion paths: training and sampling are faster, and the models generalize better. With $$\sigma_{min} = 0$$, the loss is the one of rectified flow (see its entry). A [post of the Cambridge Machine Learning Group](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) explains the method well.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2022</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow</span><span class="pub-meta">, UT Austin</span><a class="pub-link" href="https://arxiv.org/abs/2209.03003" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Rectified flow aims at reducing the number of steps needed to transport one distribution to another. This matters for diffusion models, for example, which sample in $$T$$ steps, so that fewer steps make the inference cheaper. A rectified flow is an [ordinary differential equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation) (ODE) model that transports a distribution $$\pi_0$$ to $$\pi_1$$ along paths that are as straight as possible, and it is learned with a simple least-squares regression of the velocity field onto the direction $$X_1 - X_0$$ between paired samples. Straight paths are preferred for two reasons: in theory, because a straight line is the shortest path between two end points, and in computation, because a straight path can be simulated exactly, without time discretization. Re-training the model on the couplings that it produces itself (*reflow*) straightens the paths further, so that a single [Euler step](https://en.wikipedia.org/wiki/Euler_method), optionally after a distillation, already gives good samples. The same formulation later became the training objective of Stable Diffusion 3.

**Rectified flow.** Given empirical observations of $$X_0 \sim \pi_0$$ and $$X_1 \sim \pi_1$$, the rectified flow induced from $$(X_0, X_1)$$ is an ODE on the time $$t \in [0, 1]$$,

$$
\mathrm{d}Z_t = v(Z_t, t)\mathrm{d}t,
$$

which converts $$Z_0$$ from $$\pi_0$$ into a $$Z_1$$ that follows $$\pi_1$$. The drift $$v \colon \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d$$ is chosen so that the flow follows the direction $$X_1 - X_0$$ of the linear path from $$X_0$$ to $$X_1$$ as closely as possible, which is a least-squares regression problem:

$$
\min_v \int_0^1 \mathbb{E}\left[\left\|(X_1 - X_0) - v\big(X_t,\, t\big)\right\|^2\right] \mathrm{d}t, \qquad \text{with} \quad X_t = tX_1 + (1 - t)X_0, \tag{1}
$$

where $$X_t$$ is the linear interpolation of $$X_0$$ and $$X_1$$. Why not follow the interpolation itself? The interpolation $$X_t$$ does follow the ODE $$\mathrm{d}X_t = (X_1 - X_0)\mathrm{d}t$$, but this ODE is non-causal, or anticipating: the update of $$X_t$$ needs the final point $$X_1$$, which is unknown when the sampling starts from $$X_0$$. Fitting the drift $$v$$ to $$X_1 - X_0$$ *causalizes* the paths of the linear interpolation, and gives an ODE flow that can be simulated without seeing the future. In practice, $$v$$ is a neural network or another nonlinear model, and (1) is solved with any off-the-shelf stochastic optimizer, such as stochastic gradient descent, on empirical draws of $$(X_0, X_1)$$. Algorithm 1 sums up the method, with its two optional steps, reflow and distillation:

$$
\begin{array}{l}
\hline
\textbf{Algorithm 1} \text{ Rectified Flow: Main Algorithm} \\
\hline
\quad \textbf{Procedure}\text{: } \boldsymbol{Z} = \texttt{RectFlow}((X_0, X_1))\text{:} \\
\qquad \textit{Inputs}\text{: Draws from a coupling } (X_0, X_1) \text{ of } \pi_0 \text{ and } \pi_1 \text{;} \\
\qquad \quad \text{velocity model } v_\theta \colon \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d \text{ with parameter } \theta. \\
\qquad \textit{Training}\text{: } \hat{\theta} = \mathop{\arg\min}\limits_{\theta} \mathbb{E}\left[\left\|X_1 - X_0 - v_\theta(tX_1 + (1 - t)X_0,\ t)\right\|^2\right] \text{,} \\
\qquad \quad \text{with } t \sim \text{Uniform}([0, 1]). \\
\qquad \textit{Sampling}\text{: Draw } (Z_0, Z_1) \text{ following } \mathrm{d}Z_t = v_{\hat{\theta}}(Z_t, t)\mathrm{d}t \text{ starting from } Z_0 \sim \pi_0 \\
\qquad \quad \text{(or backwardly } Z_1 \sim \pi_1 \text{).} \\
\qquad \textit{Return}\text{: } \boldsymbol{Z} = \{Z_t \colon t \in [0, 1]\}. \\
\quad \textbf{Reflow} \text{ (optional): } \boldsymbol{Z}^{k+1} = \texttt{RectFlow}((Z_0^k, Z_1^k)) \text{,} \\
\qquad \text{starting from } (Z_0^0, Z_1^0) = (X_0, X_1). \\
\quad \textbf{Distill} \text{ (optional): Learn a neural network } \hat{T} \text{ to distill the } k \text{-rectified flow,} \\
\qquad \text{such that } Z_1^k \approx \hat{T}(Z_0^k). \\
\hline
\end{array}
$$

For a given input coupling $$(X_0, X_1)$$, the exact minimum of (1) is reached at

$$
v^X(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x], \tag{2}
$$

the average of the directions $$X_1 - X_0$$ of all the lines that pass through $$x$$ at time $$t$$. Where two lines cross, the flow follows this average direction instead of either line, so the paths of the flow never cross, and the flow rewires the pairs of the coupling. This is why reflow, which retrains on the new pairs $$(Z_0, Z_1)$$, gives straighter paths.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($$\mu$$Transfer)</span><span class="pub-meta">, Microsoft</span><a class="pub-link" href="https://arxiv.org/abs/2203.03466" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Tuning the hyperparameters of a large model is expensive, because every trial is a full training run. µTransfer tunes a small model instead, and relies for this on the [Maximal Update Parametrization (µP)](https://arxiv.org/abs/2011.14522). µP sets the initialization variance, the learning rate and the output multiplier of each layer as functions of its width, so that the activations and their updates stay of order 1 as the width grows. Under µP, many optimal hyperparameters stay stable when the width of the model changes. The reason comes from the theory of Tensor Programs: when the width of a layer is large, every activation vector has roughly iid coordinates at any time during training, and Tensor Programs compute the distributions of these coordinates recursively, which describes how the function of the network evolves.

µTransfer, the tuning method of the paper, follows from this property: it parametrizes the target model in µP, tunes the hyperparameters on a smaller model, and transfers them zero-shot to the full-size model. Hyperparameters transferred from a 40M-parameter proxy outperform the published numbers of the 6.7B GPT-3, at a tuning cost of 7% of the total pretraining cost. Hyperparameters transferred from a 13M-parameter model outperform the published numbers of BERT-large (350M), at a total tuning cost of one pretraining of BERT-large. The figure below shows why the parametrization matters. In the standard parametrization, the optimal learning rate shifts as the width grows, and wider networks do not always reach a lower loss. In µP, the optimum stays at the same learning rate for all the widths, and the wider networks reach lower losses.

{: .picture}
<img loading="lazy" width="77.6%" height="77.6%" src="{{ site.assets }}/publications/mutransfer.svg"/>

{: .caption}
Training loss against learning rate on Transformers of varying $$d_{model}$$ trained with Adam, in the standard parametrization (left) and in µP (right). In the standard parametrization, unlike in µP, different widths do not share the same optimal hyperparameter; wider networks do not always perform better than narrower ones; in fact they underperform the same-width networks in µP even after tuning learning rate (see dotted line). Adapted from: [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)

Not every hyperparameter transfers, however. Table 1 of the paper lists the hyperparameters that µTransfer covers, with the caveats of Section 6.1. A * marks the ones validated only empirically, on Transformers, while the others also have a theoretical justification:
- µTransferable: the optimization hyperparameters (learning rate, momentum, Adam beta, learning rate schedule), the initialization (per-layer initialization variance) and the parameter multipliers (constants that multiply the weights or biases);
- not µTransferable: the regularization (dropout, weight decay);
- µTransferred across: width, depth\*, batch size\*, training time\*, sequence length\*.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale</span><span class="pub-meta">, Microsoft (ICML 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2201.05596" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSpeed-MoE addresses the bottlenecks that are specific to MoE models, namely the **token routing**, the **all-to-all communication**, the model size and the inference cost, from three sides:
- **model design**: the Pyramid-Residual MoE (PR-MoE) follows two observations of the paper. Experts help more in the later layers than in the first ones, so PR-MoE has more experts in the later layers (the pyramid). And a second expert per token improves the quality, but doubles the communication, so each layer sends every token through a fixed dense MLP and through one selected expert, which acts as a correction of the MLP (the residual). This reaches the quality of top-2 routing with the communication volume of top-1;
- **compression**: Mixture-of-Students (MoS), a distillation of the PR-MoE into a student PR-MoE with fewer layers;
- **inference system**: a combination of expert, tensor and data parallelism, with hierarchical all-to-all communication and fused sparse kernels.

DeepSpeed-MoE provides a general MoE framework with many practical optimizations, and it keeps the standard collectives (NCCL), outside the compute kernels. Later systems for production scale, such as MegaScale-MoE and MegaScale-Infer (see their entries), add more specialized parallelism strategies and communication libraries to handle extreme scale, and later kernel-level work moves the collectives into the kernels, for a fine-grained overlap with the computation.

For autoregressive language models, MoE reduces the training cost by about 5× compared with a dense model of equal quality. PR-MoE and MoS reduce the size of the MoE model by up to 3.7×, and the inference system gives 7.3× better latency and cost than existing MoE inference solutions. Compared with dense models of equal quality, the inference is up to 4.5× faster and 9× cheaper.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale</span><span class="pub-meta">, Microsoft DeepSpeed (SC 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2207.00032" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Inference kernels face a harder problem than training kernels. Inference typically runs at small batch sizes, where most of the time goes into reading the model weights, so its kernels must reach a high memory bandwidth utilization, and also a high compute utilization, at these small batch sizes. Training kernels only need a high compute utilization, at much larger batch sizes. DeepSpeed Inference addresses this problem with two components:
1. **DeepSpeed Transformer** scales a dense transformer automatically to several devices: it partitions the operators of the model across the devices, and adds the communication that the partition needs. Its single-GPU kernels maximize the memory bandwidth utilization of each GPU, and NCCL all-reduce collectives perform the communication across GPUs, which gives a high aggregate memory bandwidth utilization across the GPUs of a node. Tensor slicing does not scale efficiently beyond one node, however, because of its communication overhead, so DeepSpeed Inference uses pipeline parallelism to scale to several nodes. DeepSpeed Transformer has three parts:
   - single-GPU transformer kernels, which minimize the latency and maximize the throughput with fusion schedules and GEMM kernels designed around the memory bandwidth (Sec. III, described below);
   - a dense inference system for many GPUs, which combines tensor parallelism, to minimize the latency, with pipeline-parallel schedules optimized for inference and memory optimizations, to maximize the throughput (Sec. IV). Both kinds of parallelism run on top of the single-GPU kernels;
   - a sparse (MoE) inference system for massive numbers of GPUs, which serves models with trillions of parameters on hundreds of GPUs (Sec. V). It combines expert, data and tensor parallelism with new communication optimizations and sparse kernel optimizations. Because the data are replicated across the tensor-parallel ranks, the all-to-all of expert parallelism only needs to run within each subset of devices that share the same tensor-slicing rank. The sparse tensor representation of the gating function and the sparse einsum operators add a large latency, which kernel fusion reduces.
2. **ZeRO-Inference** uses CPU, NVMe and GPU memory together with GPU compute, so that massive models can run inference on limited resources (Sec. VI). Its key design decision is how to divide the GPU memory between the model weights, the inference inputs and the intermediate results. One approach pins as many weights as possible in GPU memory and fetches the rest from DRAM or NVMe when needed, but this leaves room only for small batches. ZeRO-Inference does the opposite: it keeps the weights in DRAM, when it is large enough, or in NVMe, and streams each layer into GPU memory when the computation needs it, which leaves the GPU memory free for large batches.

The single-GPU kernels rely on two techniques:
1. **Deep-Fusion** reduces the overhead of kernel launches and of data movement, by fusing more than element-wise operations. On a GPU, data that one thread block produces and another one consumes need a global memory synchronization, which means a new kernel launch. To avoid it, Deep-Fusion tiles the computation along the dimensions of the iteration space that have no data dependencies across tiles, and runs the tiles in parallel on different thread blocks. The dimensions with data dependencies are not tiled, so that one thread block processes them whole. After this tiling, two operators can be fused if each tile of the second operator depends on exactly one output tile of the first. In this way, Deep-Fusion can fuse not only element-wise operations, but also reductions, data transpositions and GEMMs, as long as they have no dependencies across tiles.
2. A **custom GEMM**, designed to be fusable with Deep-Fusion and to reach the maximum memory bandwidth utilization. The computation is tiled along the output dimension, which keeps the reduction inside each tile, so that a single kernel computes the GEMM. With this tiling, each warp of a thread block produces a partially reduced result for a tile of outputs, and a final reduction across the warps of the thread block is needed. Instead of a reduction in shared memory, a single transpose in shared memory makes the partial results of each output element contiguous, so that one warp reduces them with cooperative-group collectives, directly in registers. Finally, the weight matrix is transposed once at initialization, so that $$M$$ rows of each column are contiguous in memory, and each thread reads $$M$$ elements along the input dimension in one access ($$M = 2$$ in FP16 and 4 in INT8). The loads then use whole cache lines, while the number of parallel tiles stays the same.

The operations inside a transformer layer are fused in four regions: (1) the QKV GEMM and the input layer norm, (2) the transposition and the attention, (3) the post-attention layer norm and the intermediate GEMM, and (4) the bias and the residual addition.

Unlike later work that moves the communication into the kernels, DeepSpeed Inference fuses only the computation (Deep-Fusion, custom GEMMs), and keeps the communication as NCCL collectives outside the kernels. It therefore shows what kernel engineering and runtime scheduling achieve without a change of the communication model.

In the results, DeepSpeed Inference reduces the latency by up to 7.3× compared with the state of the art in latency-oriented scenarios, and increases the throughput by more than 1.5× in throughput-oriented ones. It serves trillion-parameter models in real time on hundreds of GPUs. With ZeRO-Inference, it runs models 25× larger than GPU-only solutions, at 84 TFLOPS, more than 50% of the peak of an A6000.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation</span><span class="pub-meta"> (EACL 2023)</span><a class="pub-link" href="https://arxiv.org/abs/2210.07558" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

LoRA blocks (see the LoRA entry) have two major problems. First, their rank is fixed after training, so a different rank means a new training from scratch. Second, finding the best rank needs an exhaustive search, with one training per rank. Dynamic LoRA (DyLoRA) solves both problems by training LoRA blocks for a range of ranks instead of a single rank. Inspired by [nested dropout](https://arxiv.org/abs/1402.0915), it sorts the representation that the adapter learns, so that the first ranks hold the most important information. It works as follows:
- Each LoRA module adds to the frozen pretrained weight $$W_0 \in \mathbb{R}^{m \times d}$$ the product of an up-projection $$W_{up} \in \mathbb{R}^{m \times r}$$ and a down-projection $$W_{dw} \in \mathbb{R}^{r \times d}$$. The module is trained to work for any rank in a range $$[r_{min}, r_{max}]$$, where $$r_{min}$$ and $$r_{max}$$ are hyperparameters.
- At each training step, a rank $$b$$ between $$r_{min}$$ and $$r_{max}$$ is sampled, usually from a uniform distribution, and $$W_{up}$$ and $$W_{dw}$$ are truncated to their first $$b$$ columns and rows, $$W_{up\downarrow b}$$ and $$W_{dw\downarrow b}$$. The step then computes $$h = W_0 x + \frac{\alpha}{b} W_{up\downarrow b} W_{dw\downarrow b} x$$, with the scale $$\alpha / b$$ of the sampled rank instead of the $$\alpha / r$$ of LoRA.

Since the loss is computed on random truncations, the adapter must work with its first $$b$$ ranks alone, for every $$b$$ of the range, and after training it can be truncated to any of these ranks without retraining. In the variant of the figure below, a step updates only the $$b$$-th column and row, and the others stay frozen. One DyLoRA training costs about as much as one LoRA training, but it replaces the training of one LoRA model per rank, so DyLoRA obtains these dynamic, search-free models 4 to 7 times faster than LoRA, depending on the task, without a significant loss of performance.

{: .picture}
<img loading="lazy" width="99.3%" height="99.3%" src="{{ site.assets }}/publications/dylora.svg"/>

{: .caption}
The overall diagram of DyLoRA. In each iteration, a rank $$b$$ is sampled from a pre-defined random distribution $$P_B$$, which truncates the up-projection and down-projection matrices in the LoRA (Hu et al., 2021) objective: the forward pass uses only the first $$b$$ columns of $$W_{up}$$ and the first $$b$$ rows of $$W_{dw}$$, and the parameter update changes only the $$b$$-th column and row, while the others stay frozen (snowflakes). Adapted from: [DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low Rank Adaptation](https://arxiv.org/abs/2210.07558)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Efficiently Scaling Transformer Inference</span><span class="pub-meta">, Google (MLSys 2023)</span><a class="pub-link" href="https://arxiv.org/abs/2211.05102" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper studies the generative inference of large Transformer models, with tight latency targets and long sequences, and how to partition it efficiently across many chips. It focuses on TPU v4 and its 3D torus network, and its methods are more efficient than [NVIDIA's FasterTransformer](https://github.com/NVIDIA/FasterTransformer).

Section 3.1 analyzes the collective communication, and Section 3.2 the parallelism of the feed-forward layer. The paper uses a compact notation for the partitions: $$BLE_{xyz}$$ means that the last dimension $$E$$ of a tensor of logical shape $$BLE$$ is split into $$X \times Y \times Z$$ partitions, so that the tensor on each chip has the shape $$[B, L, E/(X \times Y \times Z)]$$, while the axes without a subscript are replicated. $$F$$ is the hidden (intermediate) size of the feed-forward layer. Three layouts are compared:
- **1D weight-stationary** (Section 3.2.1), as in Megatron: the weights are partitioned as $$EF_{xyz}$$ and $$F_{xyz}E$$, over $$X \times Y \times Z = n_{chips}$$ chips. The weights stay in place and the activations move between the chips, but the communication time stays roughly constant as the number of chips grows, so it eventually limits the scaling.
- **2D weight-stationary** (Section 3.2.2): the weights are split along both $$E$$ and $$F$$, into square shards. The compute cost is the same, but the communication time shrinks as $$1/\sqrt{n_{chips}}$$, so this layout is more efficient beyond about 16 chips.
- **XYZ-weight-gathered** (Section 3.2.3): with a weight-stationary layout, the output of each per-chip matrix multiplication must be aggregated between chips before the next operation can use it. For very large batches, the activations become larger than the weights, and it is better to keep the activations fully stationary between sequential matrix multiplications, and to transfer the weights between all the chips instead.

The weight-stationary layouts therefore suit small batches, as in decoding, and the weight-gathered layout suits very large batches, as in the processing of long prompts.

Multi-head attention can be parallelized in essentially the same ways as a feed-forward layer (Section 3.3). Its keys and values, the KV cache, take much memory capacity and bandwidth, however. [Multi-query attention](https://arxiv.org/abs/1911.02150) shares one key head and one value head among all the query heads, which reduces the size of the KV cache by a factor of $$n_{heads}$$, and with it the time to load the cache from memory. But it also removes the head axis, which the other layouts use for parallelism, so the KV cache and its computations need another partition, one that minimizes the time of the repeated loads of the KV cache, which dominate the inference cost. The layout closest to multi-head attention (panel (b) below, Figure 4(b) of the paper) partitions the KV cache in the same way, over the heads. Since the keys and values are shared by all the heads, they are then replicated on each chip, and the memory savings of multi-query attention are lost. The paper instead partitions $$Q$$, $$K$$ and $$V$$ over the batch dimension $$B$$, into $$n_{chips}$$ partitions (panel (c)). This divides the cost of loading the KV cache per chip by $$n_{chips}$$, at the extra communication cost of resharding the input activations. With this layout, multi-query attention allows larger batches and longer sequences, which increases the throughput, in addition to the lower latency from the reduced memory time.

{: .picture}
<img loading="lazy" width="65.2%" height="65.2%" src="{{ site.assets }}/publications/efficiently_scaling_transformer_inference_multiquery.svg"/>

{: .caption}
Multiquery attention has lower memory cost to load the KV cache when sharded over batch. Top: the shapes of Q and K/V in multi-head and multi-query attention. <b>(a)</b> Multi-head attention, sharded over heads: it can be sharded across heads without replication. <b>(b)</b> Multi-query attention, sharded over heads: it requires full replication of the single head for K, increasing memory access cost. <b>(c)</b> Multi-query attention, sharded over batch: only a slice of K is needed for einsum, reducing memory access cost. Adapted from: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

Section 3.4 measures the gains of computing the attention and the feed-forward layer in parallel, as in [GPT-J](https://en.wikipedia.org/wiki/GPT-J#cite_note-Model_Card-2) and [PaLM](https://arxiv.org/abs/2204.02311). The standard transformer block applies them one after the other, $$y = x + \mathrm{Attention}(\mathrm{LayerNorm}(x)) + \mathrm{MLP}(\mathrm{LayerNorm}(x + \mathrm{Attention}(\mathrm{LayerNorm}(x))))$$, while the parallel block computes both from the same normalized input, $$y = x + \mathrm{MLP}(\mathrm{LayerNorm}(x)) + \mathrm{Attention}(\mathrm{LayerNorm}(x))$$. This has three benefits. The parallel block has one layer norm per layer instead of two, which reduces the latency at small batch sizes. Some of its matrices can be fused into larger matrix multiplications, which run more efficiently on accelerators. And it removes one of the two all-reduce operations of each layer, because the partial outputs of the attention and of the feed-forward layer can be summed before a single all-reduce.

Section 3.5 covers low-level optimizations, such as the overlap of communication and computation, and Section 3.6 the `int8` quantization of the weights. Combined, these techniques reach, on PaLM 540B with a context of 2048 tokens, a latency of 29 ms per generated token at small batch sizes, with `int8` weights, and a model FLOPs utilization (MFU) of 76% in the large-batch processing of the input tokens.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Random-LTD: Random and Layerwise Token Dropping Brings Efficient Training for Large-scale Transformers</span><span class="pub-meta">, Microsoft</span><a class="pub-link" href="https://arxiv.org/abs/2211.11586" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Random-LTD (random and layerwise token dropping) reduces the training cost of large transformer models by skipping the computation of a subset of the input tokens at all the middle layers, that is, at all the layers but the first and the last, which keep the full sequence. The tokens are dropped in a purely random way, so Random-LTD needs no scoring of the tokens and no manual design, and it takes only a few lines of code (see the figure below). This is its main difference from the other token-dropping methods, which choose the tokens with a metric:
- metrics based on the attention scores cost too much for layerwise dropping, since the metric must be computed at every layer;
- metrics based on the loss or on the frequency of the tokens are accumulated over the training, and do not change within an iteration (forward pass). The same tokens are then dropped at all the layers, and the multi-head attention of the middle layers cannot capture the dependencies of the dropped tokens.

Random dropping at every layer avoids both problems, but it adds variance to the gradients. To reduce it, the number of kept tokens grows linearly during the training, a schedule called **Monotonic Sequence Length Growth (MSLG)**.

The learning rate needs care too. Random-LTD reduces the effective batch size of the middle layers during the warmup, and MSLG reaches the full sequence length only after more than 2/3 of the training iterations, which is where the large compute saving comes from. As a result, the small learning rate of the warmup trains the model inefficiently, and in practice the warmup must be longer and the learning rate decay slower. To reduce this tuning effort, the paper introduces the **LayerToken learning rate** (Appendix C), which schedules the learning rate by the sum of the tokens that each layer has consumed, instead of by the number of iterations.

The results compare Random-LTD with the baseline on GPT-3 models with 350M and 1.3B parameters, trained on 300B tokens. Random-LTD reaches evaluation losses similar to those of the baseline with 1/3 less LayerToken consumption. The paper stresses, however, that this saving does not transfer directly to the wall-clock time, because of the implementation and the hardware: on GPT-3 1.3B, the 33.3% theoretical compute saving gives a 25.6% wall-clock saving, with a similar zero-shot accuracy. BERT and ViT models show similar results. Random-LTD also reaches a better perplexity than TokenBypass, which skips tokens in the middle layers as well, and MSLG outperforms a constant drop rate with a similar compute saving (Section 5.4).

{: .picture}
<img loading="lazy" width="81.4%" height="81.4%" src="{{ site.assets }}/publications/random_ltd.svg"/>

{: .caption}
**Left:** Transformer layers for (a) baseline and (b) random-LTD training. The dotted box is repeated <i>l</i> − 2 times (Fig. B.1 of the paper). In random-LTD, the randomly assigned tokens go through the layer and the other tokens bypass it, which saves compute. **Right:** random-LTD only requires a few lines of code. Here hs, k<sub>hs</sub> and d<sub>hs</sub> are the full input, the kept input and the dropped input; “gather”, “Layer” and “combine” are the functions for random selection, transformer layer and token combination. Adapted from: [Random-LTD: Random and Layerwise Token Dropping Brings Efficient Training for Large-scale Transformers](https://arxiv.org/abs/2211.11586)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models</span><span class="pub-meta">, Microsoft (NeurIPS 2022)</span><a class="pub-link" href="https://openreview.net/forum?id=JpZ5du_Kdh" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper studies the role of the sequence length in the training of GPT models, with experiments on GPT-2 models and public datasets. Its starting point is the **stability-efficiency dilemma**: "increasing the batch sizes and learning rates (commonly used for faster training) leads to better training efficiency but can also result in training instability, leading to poor generalization accuracy or failed runs". The experiments show where the instability comes from: it correlates strongly with extreme values of the gradient variance, and long sequences contribute to these extreme values, in particular at the beginning of the training.

The method of the paper follows from this finding. **Sequence Length Warmup (SLW)** starts the training with short sequences and increases their length gradually, which avoids the extreme gradient variance and so stabilizes the training. It is a form of [curriculum learning](https://en.wikipedia.org/wiki/Curriculum_learning) (CL), which presents easier examples first and increases their difficulty gradually. Its aim differs from that of traditional CL, however: traditional CL keeps the hyperparameters constant, while SLW uses the stability that it gains to allow more aggressive hyperparameters, for both an efficient convergence and a stable training.

On GPT-2 (117M and 1.5B), SLW trains stably with an 8× larger batch and a 4× larger learning rate, and for the same zero-shot quality, it reduces the training tokens by up to 2.2× and the wall-clock time by up to 3.7×. On GPT-3 (125M), it allows an 8× larger batch and a 40× larger learning rate, and keeps 99% of the zero-shot accuracy on 11 tasks with 10× less data and 17× less time than the original GPT-3 recipe.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</span><span class="pub-meta">, Stanford (NeurIPS 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Self-attention costs time and memory quadratic in the sequence length, so Transformers are slow and memory-hungry on long sequences. The paper argues that attention algorithms miss a principle, *IO-awareness*: accounting for the reads and writes between the levels of GPU memory. FlashAttention computes exact attention with tiling, which reduces the reads and writes between the high-bandwidth memory (HBM) of the GPU and its on-chip SRAM. In practice, it splits the attention matrix into blocks that are computed in the on-chip SRAM (shared memory), instead of in the global memory (HBM). For a sequence of length $$N$$, a head dimension $$d$$ and an SRAM of size $$M$$, FlashAttention needs $$O(N^2 d^2 / M)$$ HBM accesses, while standard attention needs $$O(N d + N^2)$$. For typical values, a head dimension of 64 to 128 and an SRAM of about 100 KB, $$d^2$$ is many times smaller than $$M$$, so FlashAttention needs many times fewer HBM accesses. As a result, FlashAttention trains Transformers faster than the baselines: 15% faster end-to-end on BERT-large than the MLPerf 1.1 record, 3× faster on GPT-2 and 2.4× faster on long-range arena. The figure below shows the memory hierarchy of the GPU, the two loops of the tiling, and the speedup of the attention on GPT-2.

{: .picture}
<img loading="lazy" width="99.5%" height="99.5%" src="{{ site.assets }}/publications/flashattention.svg"/>

{: .caption}
<b>Left:</b> FlashAttention uses tiling to prevent materialization of the large <i>N</i> × <i>N</i> attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the <b>K</b> and <b>V</b> matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of <b>Q</b> matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. <b>Right:</b> Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large <i>N</i> × <i>N</i> attention matrix to HBM, resulting in a 7.6× speedup on the attention computation. Adapted from: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

The tiling relies on two techniques. The first one is a numerically stable streaming (online) softmax. The softmax of a row needs the maximum and the sum over all the columns, so FlashAttention keeps a running maximum and a running sum for each row, and rescales its partial output whenever a new block of keys changes them. In this way, it stores neither the full matrix $$QK^\top$$ nor the large intermediate buffers of the softmax. The second one is recomputation: the backward pass computes the blocks of the attention matrix again from $$Q$$, $$K$$ and the softmax statistics of the forward pass, instead of reading a stored matrix. So instead of computing attention as a GEMM, a softmax and a GEMM, with large intermediate tensors in HBM, FlashAttention fuses the three stages and tiles them, so that the data is reused from the SRAM. This fusion changes the memory complexity of the algorithm, and not only the kernel launch overhead. Unlike the work on fused FFNs with in-kernel communication, FlashAttention is about IO-aware tiling, not about the overlap of distributed communication.

The memory becomes linear in the sequence length, and the longer contexts that this allows give better models: 0.7 better perplexity on GPT-2, 6.4 more points on long-document classification, and the first Transformers better than chance on Path-X (16K tokens) and Path-256 (64K tokens). Much of the later work on fusion follows the same template: it fuses the stages, tiles them, and streams the intermediate results instead of writing them to HBM.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepNet: Scaling Transformers to 1,000 Layers</span><span class="pub-meta">, Microsoft Research</span><a class="pub-link" href="https://arxiv.org/abs/2203.00555" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Very deep Transformers are hard to train. With Post-LN, which normalizes after the residual addition, the models perform well but the training becomes unstable as the depth grows, while Pre-LN, which normalizes the input of each sub-layer, trains stably but performs worse. Earlier work had shown that better initialization methods improve the stability of Transformer training. DeepNet goes further, with **DeepNorm**, a normalization function that changes the residual connection of the Transformer to $$x_{l+1} = LN(\alpha x_l + G_l(x_l, \theta_l))$$, where $$G_l$$ is the attention or feed-forward sub-layer $$l$$ with parameters $$\theta_l$$, and $$\alpha > 1$$ is a constant. A theoretically derived initialization also scales some weights down by a constant $$\beta$$. Together, the larger weight of the residual and the smaller initialization bound the model update by a constant, which is the theoretical justification of DeepNorm, and which stabilizes the training of extremely deep Transformers. In the words of the paper, "the proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DeepNorm a preferred alternative."

Figure 2 of the paper gives the pseudocode of `deepnorm` (the normalization layer) and `deepnorm_init` (the weight initialization), and the constants. For an encoder-only or a decoder-only model with $$N$$ layers, $$\alpha = (2N)^{1/4}$$ and $$\beta = (8N)^{-1/4}$$, so the weight of the residual grows and the initialization shrinks with the depth. $$\beta$$ scales the initialization of the feed-forward, value and output projections, while the query and key projections keep a gain of 1.

With DeepNorm, DeepNet trains Transformers with up to 1,000 layers (2,500 attention and feed-forward sub-layers). On a multilingual benchmark with 7,482 translation directions, a 200-layer model with 3.2B parameters outperforms the 48-layer, 12B-parameter state of the art by 5 BLEU points.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Contrastive Deep Supervision</span><span class="pub-meta">, Tsinghua University, Intel Corporation, and Xi’an Jiaotong (ECCV 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2207.05306" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Traditional training supervises a network only at its last layer, and the supervision propagates back layer by layer, which makes the intermediate layers hard to optimize. Deep supervision (see the Deeply-supervised Nets entry) addresses this by adding auxiliary classifiers to the intermediate layers and training them with the task loss, so that the supervision reaches the shallow layers directly. However, this conflicts with a well-known observation: the shallow layers learn low-level features, not the task-biased high-level semantic features that a task loss asks for. Contrastive Deep Supervision therefore supervises the intermediate layers with augmentation-based contrastive learning instead, which gives them a better supervision than the task loss. Contrastive learning treats two augmentations of the same image as a positive pair, and two different images as a negative pair, and it trains the network to minimize the distance of the positive pairs and to maximize the distance of the negative pairs. As a result, the network learns to be invariant to data augmentations such as Color Jitter and Random Gray Scale, a general property that does not bias the shallow layers toward the task.

The figure below compares the four approaches. Traditional supervised learning (a) puts the task loss only after the last layer, and contrastive learning (b) puts a contrastive loss there instead, after a projection head. Deep supervision (c) adds a classifier with the task loss to every intermediate layer, which avoids vanishing gradients but biases all the layers toward the task. Contrastive deep supervision (d) adds a projection head with the contrastive loss to the intermediate layers, and keeps the classifier and the task loss only after the last layer.

{: .picture}
<img loading="lazy" width="100%" height="100%" src="{{ site.assets }}/publications/contrastive_deep_supervision.svg"/>

{: .caption}
The overview of the four methods: (a) traditional supervised learning, (b) contrastive learning, (c) deep supervision and (d) contrastive deep supervision. “→” and “<span style="color:#6c8ebf">→</span>” indicate the path of forward computation and gradients backward computation. “`proj`” and “`fc`” indicate the projection heads and the fully connected classifiers, respectively. The gray dash line indicates whether the feature is task-irrelevant or task-biased. (a) Traditional supervised learning only applies supervision to the last layer and propagates it to the previous layers, leading to gradient vanishing. (c) Deep supervision trains both the last layer and the intermediate layers directly, which addresses gradient vanishing but makes all the layers be biased to the task. (d) Contrastive deep supervision introduces contrastive learning to supervise the intermediate layer and thus avoids these problems. Adapted from: [Contrastive Deep Supervision](https://arxiv.org/abs/2207.05306)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Training Compute-Optimal Large Language Models (Chinchilla)</span><span class="pub-meta">, DeepMind (NeurIPS 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper revisits a question of performance modeling, as in HPC, applied to the training of large language models: "Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens?" It answers with three approaches: (1) fix the model sizes and vary the number of training tokens; (2) vary the model size for 9 fixed FLOP counts; (3) fit a parametric loss function to the final losses of the first two approaches. The estimates come from more than 400 models, with 70M to 16B parameters, trained on 5B to 500B tokens. The parametric loss of approach 3 makes the trade-off explicit. For a model with $$N$$ parameters trained on $$D$$ tokens, it is

$$
\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},
$$

with $$E = 1.69$$, $$A = 406.4$$, $$B = 410.7$$, $$\alpha = 0.34$$ and $$\beta = 0.28$$. The first term is the loss of an ideal model, the entropy of natural text. The second term is the extra loss of a model with only $$N$$ parameters, and the third term the extra loss of a training on only $$D$$ tokens.

The main conclusion is that the large language models of the time were significantly undertrained, because their model size grew but their data did not. For compute-optimal training, the model size and the number of training tokens should grow in equal proportions: the optimal $$N$$ and $$D$$ both grow about as $$C^{0.5}$$ with the compute $$C$$. The paper confirms this with a compute-optimal model, Chinchilla, with 70B parameters. It uses the compute budget of Gopher (280B), with 4× more data, 1.4T tokens, or about 20 tokens per parameter. Chinchilla outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B) and Megatron-Turing NLG (530B) on several evaluation tasks, with an average accuracy of 67.5% on MMLU, more than 7% above Gopher.

This conclusion differs from the one of Kaplan et al. (2020) (see the scaling laws entry). They state that a compute-optimal model, with the lowest loss for a given compute budget, should not be trained to its lowest possible loss, and that for a 10× larger compute budget, the model should grow by 5.5× and the training tokens by only 1.8×. The first figure below shows the difference. The three approaches of the paper predict similar optimal model sizes, much smaller at large budgets than the projection of Kaplan et al. (dashed line). Chinchilla lies on these predictions, while the other large models lie close to the projection of Kaplan et al.

{: .picture}
<img loading="lazy" width="60.2%" height="60.2%" src="{{ site.assets }}/publications/training_compute_optimal_large_language_models.svg"/>

{: .caption}
<b>Overlaid predictions.</b> Predictions of the paper's three approaches, overlaid with the projections of Kaplan et al. (2020). All three approaches predict that current large models should be substantially smaller and therefore trained much longer than is currently done. <b><i>Chinchilla</i> outperforms <i>Gopher</i> and the other large models (Section 4.2 of the paper).</b> Adapted from: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

The second figure shows approach 1. Each curve on the left is one training run, and the envelope of these curves, the lowest loss reached for each number of FLOPs, gives the optimal model size (center) and number of training tokens (right) for each compute budget. For the compute budget of Gopher, it predicts about 67B parameters and 1.5T tokens, close to the size and the data of Chinchilla.

{: .picture}
<img loading="lazy" width="77.4%" height="77.4%" src="{{ site.assets }}/publications/training_compute_optimal_large_language_models_2.svg"/>

{: .caption}
<b>Training curve envelope.</b> The <b>left</b> panel shows all the training runs: model sizes from 70M to 10B parameters, each trained with four different cosine cycle lengths. The envelope of minimal loss per FLOP, taken from these curves, gives the optimal model size (<b>center</b>) and the optimal number of training tokens (<b>right</b>) for a given compute budget. Green: the projected optimal model size and number of training tokens for the FLOPs used to train <i>Gopher</i> ($$5.76 \times 10^{23}$$). Adapted from: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2021</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">RoFormer: Enhanced Transformer with Rotary Position Embedding (Rotary Embedding, RoPE)</span><span class="pub-meta">, Zhuiyi Technology</span><a class="pub-link" href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Absolute positional encodings, sinusoidal or learned, generalize poorly to long sequences, for two reasons: (1) absolute positions are fixed, and do not model the relations between positions well; (2) they have no mechanism to extrapolate beyond the sequence lengths seen in training. Rotary position embeddings (RoPE) address the first issue: they encode the **relative position** directly in the attention, with a dependency between tokens that decays as their relative distance grows. In the words of the paper, "RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation". The second issue remains, however: RoPE alone still extrapolates poorly beyond the training length, which later motivated [position interpolation](https://arxiv.org/abs/2306.15595), NTK-aware scaling and [YaRN](https://arxiv.org/abs/2309.00071).

The idea is to rotate the queries ($$Q$$) and the keys ($$K$$) of self-attention by angles that depend on the positions of the tokens. In two dimensions, a pair of coordinates $$(x_1, x_2)$$ of an input $$x$$ is rotated by an angle $$\theta$$:

$$
R(x) = 
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\cdot
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

For a token at position $$p$$, the $$i$$-th pair of dimensions is rotated by the angle $$\theta = p \, w_i$$, where $$w_i = 10000^{-2(i-1)/d}$$ is the frequency of the pair, so the first pairs rotate fast and the last ones slowly. A $$d$$-dimensional input is split into the pairs of dimensions $$[ (x_1, x_2), \, (x_3, x_4), \, \dots, \, (x_{d-1}, x_d)]$$, each pair is rotated, and the results are concatenated into the rotated vector. When $$d$$ is odd, the extra dimension is often left unrotated. Why does this encode the relative position? Rotations compose, and the transpose of a rotation is the rotation by the opposite angle. So with $$R_p$$ the rotation of all the pairs at position $$p$$, the attention score between a query at position $$m$$ and a key at position $$n$$ is $$\langle R_m q, R_n k \rangle = q^\top R_m^\top R_n k = \langle q, R_{n-m} k \rangle$$, which depends only on the relative position $$n-m$$.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Self-Attention Does Not Need $$O(n^2)$$ Memory</span><span class="pub-meta">, Google Research</span><a class="pub-link" href="https://arxiv.org/abs/2112.05682" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Self-attention is usually computed by building the full $$n \times n$$ matrix of attention scores, so its memory grows quadratically with the sequence length $$n$$, and on modern accelerators memory, rather than compute, often limits the length of the sequences. Rabe and Staats show that the matrix is not needed: the softmax-normalized attention of a query can be computed in a streaming (online) fashion, which reads the keys and values one at a time and keeps only two running sums, of the exponentiated scores and of the values weighted by them, plus a running maximum of the scores that keeps the exponentials from overflowing. One query then needs $$O(1)$$ memory, and self-attention $$O(\log n)$$, while the runtime stays quadratic. A practical implementation for accelerators processes the keys and values in chunks of about $$\sqrt{n}$$, so it needs $$O(\sqrt{n})$$ memory, runs within a few percent of the runtime of standard attention, and recomputes the chunks in the backward pass (gradient checkpointing). At a sequence length of 16,384, it reduces the memory overhead of self-attention by 59× for inference and by 32× for differentiation.

Not storing $$QK^\top$$ is therefore exact, not an approximation, and this gives the mathematical basis of many blockwise, numerically stable attention kernels for long contexts. FlashAttention (see its entry) adds IO-aware tiling, which also makes attention faster, while this algorithm keeps the runtime of standard attention.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Rethinking Attention with Performers</span><span class="pub-meta">, Google, Cambridge, DeepMind and Alan Turing Institute (ICLR 2021)</span><a class="pub-link" href="https://arxiv.org/abs/2009.14794" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Standard attention builds the $$L \times L$$ attention matrix $$A$$ of a sequence of $$L$$ tokens, so its time and memory grow quadratically with $$L$$. Performers approximate regular (softmax) full-rank attention with provable accuracy, in linear space and time, and without priors such as sparsity or low rank. Their method, *Fast Attention Via positive Orthogonal Random features* (FAVOR+), maps each query and each key with a random feature map $$\phi$$ to $$r$$ random features, $$Q' = \phi(Q)$$ and $$K' = \phi(K)$$ ($$r$$ is $$m$$ in the first figure), such that $$A \approx Q' K'^\top$$. Positive features keep the estimated attention weights nonnegative and the training stable, and orthogonal random directions reduce the variance of the estimate. The [Google Research post](https://blog.research.google/2020/10/rethinking-attention-with-performers.html) explains the two cases below.

In **bidirectional attention**, without a notion of past and future, the products run in the order of the dashed boxes, $$Q' (K'^\top V)$$ instead of $$(Q' K'^\top) V$$, so neither $$A$$ nor its approximation is ever built, and for tokens of dimension $$d$$ the cost drops from $$O(L^2 d)$$ to $$O(L r d)$$.

{: .picture}
<img loading="lazy" width="70.6%" height="70.6%" src="{{ site.assets }}/publications/performers.svg"/> 

{: .caption}
**Left:** Standard attention module computation, where the final desired result is computed by performing a matrix multiplication with the attention matrix $$A$$ and value tensor $$V$$. **Right:** Decoupling matrices $$Q′$$ and $$K′$$ used in the lower rank decomposition of $$A$$ and conducting the matrix multiplications in the order indicated by the dashed boxes gives a linear attention mechanism, which never explicitly constructs $$A$$ or its approximation. Adapted from: [Rethinking Attention with Performers](https://research.google/blog/rethinking-attention-with-performers/)

In **unidirectional (causal) attention**, a token must not attend to later tokens, so the attention matrix is masked to its lower-triangular part, and one product $$K'^\top V$$ can no longer serve all the queries. [Prefix sums](https://en.wikipedia.org/wiki/Prefix_sum) replace it: a running sum of the outer products of the key features and the values is built token by token, and multiplied by the features of the current query to give the next row of the result, so only running totals are stored, never a lower-triangular attention matrix.

{: .picture}
<img loading="lazy" width="75%" height="75%" src="{{ site.assets }}/publications/performers2.gif"/> 

{: .caption}
**Left:** Standard unidirectional attention requires masking the attention matrix to obtain its lower-triangular part. **Right:** Unbiased approximation on the LHS can be obtained via a prefix-sum mechanism, where the prefix-sum of the outer-products of random feature maps for keys and value vectors is built on the fly and left-multiplied by query random feature vector to obtain the new row in the resulting matrix.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads (CoCoNet)</span><span class="pub-meta">, Microsoft Research et al. (ASPLOS 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2105.05720" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Deep learning frameworks treat computation and communication as separate layers, which hides the optimizations that span both. CoCoNet breaks this abstraction barrier with a [domain-specific language](https://en.wikipedia.org/wiki/Domain-specific_language) (DSL) that expresses a program with both computation and communication, ML-aware transformations that optimize the program, and a compiler that generates high-performance kernels. With both as first-class constructs, a program written at a high level of abstraction still gets optimizations such as the fusion or the overlap of communication and computation, in data-, model- and pipeline-parallel workloads of large language models, with a few lines of code. The generated kernels speed up the data-parallel training of BERT by up to 1.68×, model-parallel inference by up to 1.51×, and pipeline-parallel inference by up to 1.77×.

The figure shows the workflow on the self-attention layer of a model-parallel Transformer: a MatMul, an AllReduce, a Dropout and a residual addition. The best schedule found by the autotuner splits the AllReduce into a ReduceScatter and an AllGather, computes the Dropout and the addition between the two on the slice of each rank, fuses the three steps into a single FusedAllReduce, and overlaps it with the MatMul.

{: .picture}
<img loading="lazy" width="75.6%" height="75.6%" src="{{ site.assets }}/publications/coconet.svg"/>

{: .caption}
Overview of CoCoNet’s workflow. First, a user expresses a machine learning algorithm in the DSL that contains both computation (MatMul) and communication (AllReduce). Then, the autotuner applies transformations to optimize the program while keeping the algorithm unchanged, such as fusing AllReduce and Dropout into FusedAllReduce and overlapping this with MatMul. Finally, CoCoNet generates custom communication and computation code, which is available through PyTorch. Adapted from: [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](https://arxiv.org/abs/2105.05720)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">LoRA: Low-Rank Adaptation of Large Language Models</span><span class="pub-meta">, Microsoft</span><a class="pub-link" href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Fine-tuning a large pretrained model on a new task or domain retrains the whole model, which is expensive in compute, and stores a full copy of it for every task. Low-Rank Adaptation (LoRA) fine-tunes large models efficiently instead: it freezes the pretrained weights and adds trainable low-rank updates to the layers. Decomposing the updates into low-rank matrices reduces the number of trainable parameters by orders of magnitude, and keeps the expressive power of the model.

For a frozen weight matrix $$W \in \mathbb{R}^{d \times k}$$, the layer uses $$W' = W + \Delta W$$. Instead of learning $$\Delta W$$ directly, LoRA factorizes it as $$\Delta W = BA$$, with $$B \in \mathbb{R}^{d \times r}$$, $$A \in \mathbb{R}^{r \times k}$$ and a rank $$r \ll \min(d, k)$$, so the layer trains $$r(d + k)$$ parameters instead of $$dk$$. The figure below shows the square case, $$k = d$$. The matrix $$A$$ starts from a random Gaussian and $$B$$ from zero, so that $$\Delta W$$ is zero and training starts from the pretrained model. The update is scaled by $$\alpha / r$$, and the paper applies it to the attention projections, for example $$W_q$$ and $$W_v$$. After training, $$\Delta W$$ can be merged into $$W$$, so inference has no extra latency, unlike with adapter layers.

Compared with the full fine-tuning of GPT-3 175B with Adam, LoRA reduces the number of trainable parameters by 10,000×, and the GPU memory during training by 3×, from 1.2 TB to 350 GB. With $$r = 4$$ on the query and value projections, the checkpoint of a task shrinks from 350 GB to 35 MB. The quality matches or exceeds that of full fine-tuning on RoBERTa, DeBERTa, GPT-2 and GPT-3.

{: .picture}
<img loading="lazy" width="24.5%" height="24.5%" src="{{ site.assets }}/publications/lora.svg"/>

{: .caption}
LoRA reparametrization: only <i>A</i> and <i>B</i> are trained. Adapted from: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Learning Transferable Visual Models From Natural Language Supervision (CLIP)</span><span class="pub-meta">, OpenAI</span><a class="pub-link" href="https://arxiv.org/abs/2103.00020" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

State-of-the-art vision systems are trained to predict a fixed, predetermined set of object classes, so every other visual concept needs more labeled data. In NLP, however, "the development of text-to-text as a standardized input-output interface has enabled task-agnostic architectures to zero-shot transfer to downstream datasets, removing the need for specialized output heads or dataset specific customization". The insight of the paper is that natural language can serve as such a flexible prediction space for vision too, and enable generalization and transfer. Through its generality, language can express, and therefore supervise, a much wider set of visual concepts than a list of labels. Learning from language also has an important advantage over most unsupervised or self-supervised methods: it does not only learn a representation, it also connects this representation to language, which enables flexible zero-shot transfer.

With this in mind, the paper introduces CLIP (Contrastive Language–Image Pre-training), a network that learns visual concepts from natural language supervision. It can be instructed in natural language to perform a large variety of classification benchmarks, without being optimized for any of them, similar to the zero-shot capabilities of GPT-2 and GPT-3, and it applies to nearly arbitrary visual classification tasks. For example, it matches the ImageNet accuracy of the original ResNet-50 without using any of its 1.28 million training examples. Not optimizing the model for a benchmark is the key change: its score then becomes much more representative of its performance on other data, as the robustness results below show.

CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in its dataset. Such pairs are an abundant source of supervision on the internet: the dataset holds 400 million pairs of images and texts, collected with 500,000 queries. The image encoders are 5 ResNets and 3 Vision Transformers. The ResNets are a ResNet-50, a ResNet-101, and three EfficientNet-style ResNet-50s scaled to 4×, 16× and 64× its compute, with changes such as an attention pooling, which works like QKV attention, in place of the global average pooling. The text encoder is a Transformer with masked self-attention, with a BPE vocabulary of 49,152 tokens and a maximum sequence length of 76 (Section 2.4).

{: .picture}
<img loading="lazy" width="67.5%" height="67.5%" src="{{ site.assets }}/publications/clip.svg"/> 

{: .caption}
Summary of CLIP: (1) contrastive pre-training, (2) creating a dataset classifier from label text and (3) using it for zero-shot prediction. While standard image models jointly train an image feature extractor and a linear classifier to predict some label, CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes. Adapted from: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

The contrastive pre-training (1) works on a batch of $$N$$ image–text pairs:
- The image encoder (ViT or ResNet) encodes each image $$i$$ into an embedding $$I_i$$, and the text encoder (Transformer) encodes each text $$i$$ into $$T_i$$.
- The matrix of the products $$I_i \cdot T_j$$ compares every image with every text of the batch, and its diagonal (in blue) holds the correct pairs.
- Contrastive learning pulls the embeddings of matching images and texts together, and pushes the others apart: the training maximizes the products of the $$N$$ matching pairs, on the diagonal, and minimizes those of the $$N^2 - N$$ other pairs. In the words of the paper, "the cosine similarity (i.e. cosine of the angle) of these embeddings is then calculated, scaled by a temperature parameter τ, and normalized into a probability distribution via a softmax". The scaled similarities are thus logits, and a softmax cross-entropy over the rows and another over the columns give a loss that is symmetric between images and texts. Figure 3 of the paper gives the pseudocode.
- Each image, for example $$I_3$$, thus gets a distribution over the texts of the batch, from $$I_3 \cdot T_1$$ to $$I_3 \cdot T_N$$.

The batch size $$N$$ is critical. Each image is contrasted with the $$N-1$$ other texts of the batch, so a larger batch approximates the whole dataset better and gives more detailed representations, but the similarity matrix costs $$O(N^2)$$. CLIP uses a batch of 32,768 pairs.

The trained encoders then classify images without any further training. To create a dataset classifier from label text (2), a prompt is added to each label of a set of labels, for example `A photo of a {object}.`, and the text encoder embeds the prompts into the target set. The zero-shot prediction (3) passes an image through the image encoder, computes its distribution over the prompted labels, and picks the top label, in the picture `A photo of a dog.` No training on the target task is needed, and the classes at test time can differ from any training label. This is the fundamental difference from standard image classification, which learns to tell apart a fixed set of classes: CLIP learns the connection between images and the structure of language.

The main results are:
- CLIP is much more efficient at zero-shot transfer than an image-captioning baseline (Figure 2). Although highly expressive, a Transformer language model that predicts the captions learns zero-shot ImageNet classification 3× slower than a baseline that predicts a bag-of-words encoding of the text, and the contrastive objective of CLIP improves the efficiency by another 4×.
- Prompt engineering and ensembling (Figure 4) improve the zero-shot accuracy by almost 5 points on average across 36 datasets. The gain is similar to that of 4× more compute with the baseline zero-shot method, but free at inference.
- Zero-shot CLIP is competitive with a fully supervised baseline (Figure 5), a *linear probe*: a linear classifier trained on the frozen features of a pretrained model, here a ResNet-50 trained on ImageNet. The name comes from the idea that the frozen features already hold a good representation, which a linear layer probes for the new task. Zero-shot CLIP wins on 16 of 27 datasets, including ImageNet itself (+1.9). On STL10, where each class has few labeled examples and supervised learning is hard, zero-shot CLIP reaches 99.3%, a new state of the art without any training example. The largest gain over the linear probe is on Stanford Cars (+28.9). On specialized tasks, such as satellite images or tumor detection, and on MNIST, the linear probe wins.
- Zero-shot CLIP matches the average performance of a 4-shot linear classifier trained on its own feature space, and nearly matches the best 16-shot linear classifier across publicly available models (Figure 6).
- To match zero-shot CLIP, a linear classifier on the CLIP features needs from less than 1 to 184 labeled examples per class, with a median of 5.4 (Figure 7).
- The error decreases smoothly as the compute and the model size grow (Figure 9). The results of individual evaluations are noisy, so the trend comes from the average over all of them.
- With linear probes, the best CLIP models beat the state-of-the-art computer vision models, on average across 12 and across 27 datasets (Figure 10).
- Models trained on ImageNet lose much accuracy when the dataset changes but the labels do not, and the loss grows as the distribution moves further from ImageNet (ImageNet sketches, adversarial images, etc.). Zero-shot CLIP reduces this robustness gap by up to 75% (Figure 13).
- A linear probe on top of CLIP, fitted to one dataset, increases the accuracy on that dataset, by 9.2% on ImageNet, but slightly reduces the accuracy on the distribution-shifted datasets (Figure 14).
- The label set matters: adding the class `child` greatly reduces the share of images of people under 20 that are classified into crime-related or non-human categories (Table 7).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">GSPMD: General and Scalable Parallelization for ML Computation Graphs</span><span class="pub-meta">, Google</span><a class="pub-link" href="https://arxiv.org/pdf/2105.04663.pdf" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

GSPMD (General and Scalable Parallelization for ML Computation Graphs) is an open-source, automatic, compiler-based parallelization system, built on the [XLA compiler](https://www.tensorflow.org/xla). The user writes the model as for a single device and annotates a few of its tensors with how they are distributed, and GSPMD partitions the whole computation from these hints. Different model architectures suit different parallelization strategies, so GSPMD supports a large variety of them, for example data parallelism for small models, pipeline parallelism for larger models, or a combination of both. The system is also described in a [Google blog post](https://blog.research.google/2021/12/general-and-scalable-parallelization.html).

Each tensor gets a sharding property, which specifies how its data is distributed across the devices, either from the user, as an initial annotation, or from the sharding completion pass of the compiler. GSPMD defines three types of sharding:
- replicated: all devices hold the same full data;
- tiled: the tensor is split into tiles, one per device, without duplication;
- partially tiled, an extension of [GShard](https://arxiv.org/abs/2006.16668): the devices are divided into subgroups, and the tensor is tiled across the subgroups and replicated within each subgroup.

The user sets the shardings with `mesh_split(tensor, device_mesh, dims_mapping)`, which splits a tensor across a mesh of devices, with a mapping from each dimension $$i$$ of the tensor to an optional dimension of the mesh. This simple API is general enough to express all types of sharding, across the batch, feature, channel or any other dimensions. The automatic partitioner is a set of compiler passes in XLA (Section 3.5). The sharding completion pass propagates the annotations to the other tensors, with information about the operators, for example that a dot product is a generalized matrix multiplication, and with iterations over the graph that refine the shardings.

GSPMD reaches 50% to 62% compute utilization on up to 2048 Cloud TPUv3 cores, for models with up to one trillion parameters. It produces a single program for all the devices ([SPMD](https://en.wikipedia.org/wiki/Single_program,_multiple_data)), so, as in GShard, its compilation time stays constant as the number of devices grows. The figure shows a simplified Transformer feedforward layer with its tensors annotated for a 2×2 mesh of 4 devices (left), and the program that each device runs (right): AllGathers (AG) combine tiles of the inputs across devices before the matrix multiplications, and a ReduceScatter (RS) accumulates the partial results across devices and redistributes them.

{: .picture}
<img loading="lazy" width="88.1%" height="88.1%" src="{{ site.assets }}/publications/gspmd.svg"/>

{: .caption}
**Left:** Computation where tensors are annotated to be split over 4 devices. A simplified feedforward layer of a Transformer model. Blue rectangles represent tensors with dashed red & blue lines overlaid representing the desired partitioning across a 2x2 mesh of devices. **Right:** Partitioned execution graph generated by GSPMD. A single partition, after GSPMD has been applied. Adapted from: [General and Scalable Parallelization for Neural Networks](https://research.google/blog/general-and-scalable-parallelization-for-neural-networks/)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">MLP-Mixer: An all-MLP Architecture for Vision</span><span class="pub-meta">, Google (NeurIPS 2021)</span><a class="pub-link" href="https://arxiv.org/abs/2105.01601" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Convolutions (CNNs) and attention (Transformers) are the building blocks of modern computer vision, and the paper argues that neither is necessary. It presents MLP-Mixer, an architecture based only on multi-layer perceptrons (MLPs), which cuts the image into patches and projects each patch linearly. "MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information)." In the figure, the first type is the channel-mixing MLP 2, and the second the token-mixing MLP 1. The token-mixing MLPs are sensitive to the order of the patches, so Mixer needs no position embeddings, and its cost is linear in the number of patches, while the self-attention of ViT is quadratic.

Pre-trained on large datasets, or trained with modern regularization, MLP-Mixer is competitive with existing methods at a comparable pre-training and inference cost, for example with 87.9% top-1 accuracy on ImageNet after pre-training on JFT-300M.

{: .picture}
<img loading="lazy" width="100%" height="100%" src="{{ site.assets }}/publications/mlp_mixer.svg"/>

{: .caption}
MLP-Mixer consists of per-patch linear embeddings, Mixer layers, and a classifier head. Mixer layers contain one token-mixing MLP and one channel-mixing MLP, each consisting of two fully-connected layers and a GELU nonlinearity. Other components include: skip-connections, dropout, and layer norm on the channels. Adapted from: [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Pay Attention to MLPs</span><span class="pub-meta">, Google (NeurIPS 2021)</span><a class="pub-link" href="https://arxiv.org/abs/2105.08050" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

How much of the success of Transformers comes from self-attention? The paper introduces gMLP (gated MLPs), an architecture without attention, and shows that it can perform as well as Transformers in language and vision applications. In vision, it claims that "self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy". On BERT, gMLP matches Transformers in pretraining perplexity and is better on some downstream tasks, and on the finetuning tasks where it is worse, a substantially larger gMLP closes the gap.

gMLP relies on channel projections and spatial projections with static parameters. It is a stack of $$L$$ blocks of identical size and structure, and each block computes, without its shortcut and normalizations,

$$
Z = \sigma(XU), \,\,\,\,\,\,\,\, \tilde{Z} = s(Z), \,\,\,\,\,\,\,\, Y = \tilde{Z} V
$$

where $$X$$ holds the representations of the $$n$$ tokens, $$\sigma$$ is an activation function such as GeLU, $$U$$ and $$V$$ are linear projections along the channel dimension, and $$s(\cdot)$$ is a layer that captures spatial interactions, across the tokens. When $$s$$ is the identity, the block is a regular FFN, without any communication across tokens. In gMLP, $$s(\cdot)$$ is the Spatial Gating Unit (Section 2.1), which splits $$Z$$ along the channels into $$(Z_1, Z_2)$$ and computes $$s(Z) = Z_1 \odot (W Z_2 + b)$$: one half, projected across the $$n$$ tokens by $$W \in \mathbb{R}^{n \times n}$$, gates the other half element-wise. The weights $$W$$ start close to 0 and the bias $$b$$ at 1, so each block starts as an FFN and learns the spatial interactions during training. The positions of the tokens are captured in $$W$$, so gMLP needs no position embeddings, but $$W$$ is static and of size $$n \times n$$: the model fixes the number of tokens, and the spatial projection costs $$O(n^2)$$ per channel, as attention does.

{: .picture}
<img loading="lazy" width="93.5%" height="93.5%" src="{{ site.assets }}/publications/pay_attention_to_mlps.svg"/>

{: .caption}
Overview of the gMLP architecture with Spatial Gating Unit (SGU). The model consists of a stack of <i>L</i> blocks with identical structure and size. All projection operations are linear and “⊙” refers to element-wise multiplication (linear gating). The input and output protocols follow BERT for NLP and ViT for vision. Unlike Transformers, gMLPs do not require positional encodings, nor is it necessary to mask out the paddings during NLP finetuning. Adapted from: [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</span><span class="pub-meta">, Google (ICLR 2021)</span><a class="pub-link" href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Vision Transformers (ViTs) apply the Transformer architecture to images with as few changes as possible. Self-attention over the pixels would have a cost quadratic in their number, so ViT splits the image into fixed-size patches, for example of 16×16 pixels, and feeds the Transformer encoder with their linear embeddings, plus position embeddings: a 224×224 image gives $$224^2/16^2 = 196$$ tokens instead of 50,176 pixels. For classification, a learnable class token is added to the sequence, and an MLP head on its output predicts the class.

Pre-trained on large datasets (ImageNet-21k, JFT-300M), ViT matches or beats state-of-the-art CNNs (ResNets) in image classification, with substantially fewer computational resources for training. On mid-sized datasets, such as ImageNet without strong regularization, it is a few percentage points less accurate than comparable ResNets, because it lacks the [inductive biases](https://en.wikipedia.org/wiki/Inductive_bias) of convolutions, locality and translation equivariance. Large-scale pre-training makes up for them.

{: .picture}
<img loading="lazy" width="76.2%" height="76.2%" src="{{ site.assets }}/publications/visual_transformer.svg"/>

{: .caption}
Model overview. The image is split into fixed-size patches; each patch is linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. For classification, an extra learnable “classification token” is added to the sequence, the standard approach. The illustration of the Transformer encoder was inspired by Vaswani et al. (2017). Adapted from: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Finetuned Language Models Are Zero-Shot Learners (FLAN)</span><span class="pub-meta">, Google (ICLR 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2109.01652" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Large language models such as GPT-3 are good few-shot learners, but much worse zero-shot learners. The paper improves their zero-shot performance with *instruction tuning*: the finetuning of a language model on more than 60 NLP datasets described with natural language instructions, so that the model learns to follow instructions, also for tasks it has not seen. The authors write ten unique templates for each dataset, as the figure shows for a natural language inference task. To evaluate the model on unseen tasks, the datasets are grouped into clusters by task type, and each cluster is held out of the instruction tuning of the model that is evaluated on it.

The resulting 137B model, FLAN, surpasses zero-shot 175B GPT-3 on 20 of the 25 evaluated datasets, and even beats few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA and StoryCloze. The number of finetuning datasets, the model scale and the natural language instructions are all key to this success: for models of 8B parameters and fewer, instruction tuning even hurts the performance on held-out tasks.

{: .picture}
<img loading="lazy" width="88.1%" height="88.1%" src="{{ site.assets }}/publications/finetune_language_models.svg"/>

{: .caption}
Multiple instruction templates describing a natural language inference task. Adapted from: [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2020</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Scaling Laws for Neural Language Models</span><span class="pub-meta">, Johns Hopkins, OpenAI</span><a class="pub-link" href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

How does the performance of a language model depend on its size, on the amount of training data, and on the compute spent on training? The paper measures empirical scaling laws for the cross-entropy loss of language models, and finds that the loss follows a power law in each of these three quantities, with some trends spanning more than seven orders of magnitude. Simple equations also give the dependence of overfitting on the sizes of the model and of the dataset, and the dependence of the training speed on the model size, and together these relations give the optimal allocation of a fixed compute budget.

{: .picture}
<img loading="lazy" width="74%" height="74%" src="{{ site.assets }}/publications/scaling_laws.svg"/>

{: .caption}
Language modeling performance improves smoothly as the model size, dataset size, and amount of compute used for training increase. For optimal performance all three factors must be scaled up in tandem. Empirical performance has a power-law relationship with each individual factor when not bottlenecked by the other two. Adapted from: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

**Keypoints:**
- Performance depends most strongly on scale, which has three factors: the number of model parameters $$N$$ (without the embeddings), the size of the dataset $$D$$, and the compute $$C$$ used for training. When the other two factors are not a bottleneck, the loss has a **power-law** relation with each of them (Figure 1): $$L(N) = (N_c/N)^{\alpha_N}$$ with $$\alpha_N \approx 0.076$$, $$L(D) = (D_c/D)^{\alpha_D}$$ with $$\alpha_D \approx 0.095$$, and $$L(C) = (C_c/C)^{\alpha_C}$$ with $$\alpha_C \approx 0.050$$, for compute-efficient training.
- Within reasonable limits, performance depends only weakly on other architectural hyperparameters, such as depth vs. width.
- Performance improves predictably as long as $$N$$ and $$D$$ grow together, but it has diminishing returns if one of them is fixed while the other grows. The penalty depends on the ratio $$N^{0.74}/D$$, so every 8× increase of the model size needs only about 5× more data to avoid it.
- On text with a different distribution than the training data, the results correlate strongly with those on the validation set of the training data, with a roughly constant offset in the loss. In other words, the transfer to another distribution costs a constant penalty, and otherwise improves in line with the performance on the training data.
- Larger models are significantly more sample-efficient: they reach the same loss with fewer optimization steps and fewer data points, as Figure 2 (below) shows. With a fixed compute budget $$C$$ and no other limit on $$N$$ or $$D$$, the optimal performance therefore comes from training very large models on a relatively modest amount of data, and stopping significantly before convergence. The optimal model size grows as $$N \propto C^{0.73}$$, while the data grow only as $$D \propto C^{0.27}$$. Chinchilla (see its entry) later revised this allocation: the parameters and the training tokens should grow in equal proportions.
- The ideal batch size for training these models is roughly a power of the loss only.

{: .picture}
<img loading="lazy" width="76.3%" height="76.3%" src="{{ site.assets }}/publications/scaling_laws_2.svg"/>

{: .caption}
Test loss of language model training runs, for models from 10<sup>3</sup> to 10<sup>9</sup> parameters (excluding embeddings). Left: against tokens processed; larger models require fewer samples to reach the same performance. Right: against compute; the optimal model size grows smoothly with the loss target and compute budget. Adapted from: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Language Models are Few-Shot Learners (GPT-3)</span><span class="pub-meta">, OpenAI</span><a class="pub-link" href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Before GPT-3, the large gains on many NLP tasks came from pre-training on a large corpus of text, followed by fine-tuning on a specific task. Fine-tuning, however, still needs a task-specific dataset of thousands or tens of thousands of examples, while humans can perform a new language task from a few examples or from simple instructions. The paper shows that scaling up language models greatly improves their task-agnostic few-shot performance, sometimes up to the level of earlier state-of-the-art fine-tuning approaches. It presents GPT-3, an autoregressive LLM with 175 billion parameters, 10× more than any earlier non-sparse language model, and tests it in the few-shot setting, without any gradient update.

**GPT-3 architecture.** GPT-3 uses the model and architecture of [GPT-2](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf), including its modified initialization, pre-normalization and reversible tokenization. The only difference is that the layers of the Transformer alternate dense and locally banded sparse attention patterns, as in the [Sparse Transformer](https://arxiv.org/abs/1904.10509).
- Table 2.1 lists the 8 GPT-3 models, from 125M to 175B parameters, with their sizes and hyperparameters. All of them have a context window of 2,048 tokens.
- Fig. 2.2 shows the total compute used during training. Following the analysis of [Scaling Laws For Neural Language Models](https://arxiv.org/abs/2001.08361), GPT-3 trains much larger models on many fewer tokens than was typical: all the models train on 300B tokens.
- Fig. 3.1 shows that the performance scales smoothly with compute: the cross-entropy loss follows a power law of the training compute.

**Background (Fig. 2.1).** The paper compares four ways for a language model to perform a task:
- Fine-Tuning (FT), the most common approach in recent years, updates the weights of a pre-trained model by training it on a supervised dataset specific to the task.
- Few-Shot (FS) gives the model a few demonstrations of the task at inference time, as conditioning, but allows no weight updates. The number of demonstrations is typically 10 to 100, as many as fit in the context window.
- One-Shot (1S) is the same as few-shot, with a single demonstration, in addition to a natural language description of the task.
- Zero-Shot (0S) is the same as one-shot, without any demonstration: the model gets only a natural language instruction that describes the task.

{: .picture}
<img loading="lazy" width="86.7%" height="86.7%" src="{{ site.assets }}/publications/gpt3_fig21.svg"/> 

{: .caption}
<b>Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning</b>. The panels above show four methods for performing a task with a language model – fine-tuning is the traditional method, whereas zero-, one-, and few-shot, the settings studied in the paper, require the model to perform the task with only forward passes at test time. In the few-shot setting the model is typically given a few dozen examples. Exact phrasings for all task descriptions, examples and prompts are in Appendix G of the paper. Adapted from: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

Why can a model learn a task from a few examples in its input, without any weight update? The second figure gives the intuition. During unsupervised pre-training, the outer loop, the language model develops a broad set of skills and pattern recognition abilities. At inference time, it uses these abilities to adapt quickly to the task described in its context, an inner loop that the paper calls *in-context learning*.

{: .picture}
<img loading="lazy" width="97.3%" height="97.3%" src="{{ site.assets }}/publications/gpt3_fig11.svg"/> 

{: .caption}
<b>Language model meta-learning.</b> During unsupervised pre-training, a language model develops a broad set of skills and pattern recognition abilities. It then uses these abilities at inference time to rapidly adapt to or recognize the desired task. The term “in-context learning” describes the inner loop of this process, which occurs within the forward-pass upon each sequence. The sequences in this diagram are not intended to be representative of the data a model would see during pre-training, but are intended to show that there are sometimes repeated sub-tasks embedded within a single sequence. Adapted from: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

**Tasks and performance:**
- On NLP tasks, GPT-3 gives promising results in the zero-shot and one-shot settings, and in the few-shot setting it is sometimes competitive with the state of the art, or even surpasses it.
- It also shows one-shot and few-shot proficiency at tasks that test rapid adaptation or on-the-fly reasoning, such as unscrambling words, performing arithmetic, and using novel words in a sentence after seeing them defined only once.
- Figures 3.3 to 3.12 show that the performance of GPT-3 grows with the model size, which suggests that language models keep absorbing knowledge as their capacity increases. The results cover TriviaQA, translation, [WinoGrande](https://arxiv.org/abs/1907.10641) (an adversarial version of the Winograd Schema Challenge), PIQA, reading comprehension, SuperGLUE, ANLI Round 3, arithmetic, word scrambling and SAT analogies, in the zero-, one- and few-shot settings.
- Fig. 3.13 shows that the ability of people to tell whether a news article was written by the model (the ratio of correct assignments to non-neutral assignments) decreases as the model size increases.
- Fig. 4.2 gives the analysis of benchmark contamination. Data contamination has a minimal effect on the performance of GPT-3 on most datasets, but a few datasets may have inflated results: PIQA and Winograd, whose results are marked with an asterisk.
- Even at this scale, few-shot learning still struggles on some tasks, such as natural language inference (the ANLI dataset) and some reading comprehension datasets (RACE and QuAC). Chapter 5 details the limitations of the model.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2019</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Graph Transformer Networks</span><span class="pub-meta">, Korea University (NeurIPS 2019)</span><a class="pub-link" href="https://arxiv.org/abs/1911.06455" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Most [graph neural networks](https://en.wikipedia.org/wiki/Graph_neural_network) (GNNs) assume a fixed and homogeneous graph, with a single type of nodes and of edges, while many graphs are heterogeneous: a citation graph, for example, connects papers, authors and conferences. Graph Transformer Networks (GTNs) generate new graph structures from such a graph, with useful connections between nodes that the original graph does not connect, and learn node representations on them, end to end. Their core layer, the Graph Transformer (GT) layer, learns a soft selection of edge types and composite relations, which gives useful multi-hop connections called meta-paths.

A meta-path of length $$k$$ follows edges of the types $$t_1, \ldots, t_k$$ from a node $$v_1$$ to a node $$v_{k+1}$$, and defines the composite relation $$R = t_1 \circ t_2 \circ \cdots \circ t_k$$ between them, such as author–paper–conference. In the figure, the first GT layer selects two adjacency matrices, $$Q_1$$ and $$Q_2$$, as softmax-weighted sums of the adjacency matrices of the edge types (the 1×1 convolutions), and their product, normalized by its degree matrix, is the adjacency matrix of a meta-path of length 2. Each further GT layer multiplies the result by a new selection, and since the identity matrix is among the matrices to select from, a stack of $$l$$ GT layers learns meta-paths of any length up to $$l+1$$. Several meta-paths are learned in parallel, one per channel, and the node representations that the same GCN computes on each new graph are concatenated for node classification. Without predefined, domain-specific meta-paths, GTNs reach state-of-the-art node classification on three heterogeneous graph benchmarks (DBLP, ACM, IMDB).

{: .picture}
<img loading="lazy" width="92.2%" height="92.2%" src="{{ site.assets }}/publications/graph_transformer_networks.svg"/>

{: .caption}
Graph Transformer Networks (GTNs) learn to generate a set of new meta-path adjacency matrices 𝔸<sup>(<i>l</i>)</sup> using GT layers and perform graph convolution as in GCNs on the new graph structures. Multiple node representations from the same GCNs on multiple meta-path graphs are integrated by concatenation and improve the performance of node classification. ℚ<sub>1</sub><sup>(<i>l</i>)</sup> and ℚ<sub>2</sub><sup>(<i>l</i>)</sup> ∈ <b>R</b><sup><i>N</i>×<i>N</i>×<i>C</i></sup> are intermediate adjacency tensors to compute meta-paths at the <i>l</i>th layer. Adapted from: [Graph Transformer Networks](https://arxiv.org/abs/1911.06455)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Root Mean Square Layer Normalization (RMSNorm)</span><span class="pub-meta">, University of Edinburgh & University of Zurich (NeurIPS 2019)</span><a class="pub-link" href="https://arxiv.org/abs/1910.07467" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Layer normalization (LayerNorm) stabilizes the training of deep networks, but its computational overhead slows them down, in particular recurrent networks. RMSNorm (Root Mean Square Normalization) is a simpler and computationally cheaper alternative, which only scales the input by the root mean square of its features, instead of normalizing it with their mean and variance:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}
$$

where $$x$$ is the input vector of dimension $$d$$, and

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma
$$

where $$\gamma \in \mathbb{R}^d$$ is a learnable gain, as in LayerNorm. LayerNorm computes $$\frac{x - \mu}{\sigma} \odot \gamma + \beta$$, with the mean $$\mu$$ and the standard deviation $$\sigma$$ of $$x$$, so RMSNorm computes one statistic instead of two, and no longer re-centers the input. The paper keeps a bias $$\beta$$ after the normalization, while large models that use RMSNorm, such as LLaMA, keep only the gain $$\gamma$$.

Why can the mean be dropped? The paper hypothesizes that the re-centering invariance of LayerNorm is dispensable, and that its re-scaling invariance, which RMSNorm keeps, is what matters. In the experiments, RMSNorm reaches an accuracy comparable to LayerNorm and reduces the running time by 7% to 64% across models.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention)</span><span class="pub-meta">, Noam Shazeer, Google</span><a class="pub-link" href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Training a Transformer is efficient because it parallelizes across the length of the sequence. Incremental decoding cannot use this parallelism, and it is slow, because each step loads the large key and value tensors of all the previous tokens from memory to do little arithmetic with them: for multi-head attention (MHA), the paper estimates the ratio of memory accesses to arithmetic operations as $$\Theta(n/d + 1/b)$$, for $$n$$ tokens, a model dimension $$d$$ and a batch size $$b$$, so the memory bandwidth becomes the bottleneck when $$n$$ approaches $$d$$ or $$b$$ approaches 1.

Multi-query attention (MQA) speeds up decoding. MHA runs $$h$$ attention heads in parallel, with different linear projections of the queries, keys, values and outputs, and MQA is identical, except that all the heads share a single set of keys and values. This divides the size of these tensors by $$h$$, and the ratio becomes $$\Theta(1/d + n/(dh) + 1/b)$$. On WMT 2014 English-to-German translation, with $$h = 8$$, the decoder time per token drops from 46 to 3.8 μs in greedy decoding, with a minor loss of quality (27.5 BLEU instead of 27.7). Grouped-query attention ([GQA](https://arxiv.org/abs/2305.13245), Ainslie et al., 2023), used in the larger Llama 2 models (see its entry), shares one set of keys and values per group of heads, and sits between MHA and MQA in memory and in quality.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Generating Long Sequences with Sparse Transformers</span><span class="pub-meta">, OpenAI</span><a class="pub-link" href="https://arxiv.org/abs/1904.10509" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The memory and the runtime of attention grow quadratically with the sequence length $$n$$, which limits Transformers to short sequences. The paper introduces several **sparse factorizations of the attention matrix**, which reduce this cost to $$O(n \sqrt{n})$$: they split the full attention into several faster attention operations that, combined, **approximate the dense attention**. Why should this work? The authors claim that sparsity is a natural pattern of attention: by visual inspection of a network trained with full attention, most layers have sparse attention patterns across most data points, so sparse attention should not hurt the performance much. Other layers, however, show global patterns and data-dependent sparsity, where a fixed sparse pattern could hurt the performance. One architecture models images, audio and raw bytes, and sets new state-of-the-art density-modeling results on Enwik8, CIFAR-10 and ImageNet-64, with hundreds of layers, on sequences tens of thousands of timesteps long.

**Factorized self-attention** has $$p$$ separate attention heads, and each head attends to a subset $$A$$ of the indices. The hard problem is to choose small subsets such that every output position still reaches every earlier input position through a path of at most $$p+1$$ steps. Section 4.3 details two 2D factorizations (figure below). The *strided* pattern, with a stride $$l$$ close to $$\sqrt{n}$$, has one head attend to the previous $$l$$ positions and the other to every $$l$$-th position, which suits data with a periodic structure, such as images. The *fixed* pattern has one head attend within blocks of $$l$$ positions and the other to a few summary positions at the end of each block, which suits data without a periodic structure, such as text.

{: .picture}
<img loading="lazy" width="87.9%" height="87.9%" src="{{ site.assets }}/publications/sparse_transformers.svg"/>

{: .caption}
Two 2D factorized attention schemes of the Sparse Transformer, (b) strided and (c) fixed, compared with the full attention of a standard Transformer (a). The top row indicates, for an example 6x6 image, which positions two attention heads receive as input when computing a given output. The bottom row shows the connectivity matrix (not to scale) between all such outputs (rows) and inputs (columns). Sparsity in the connectivity matrix can lead to significantly faster computation. In (b) and (c), full connectivity between elements is preserved when the two heads are computed sequentially. The paper tests whether such factorizations can match the performance of the rich connectivity patterns learned by full attention (Figure 2 of the paper). Adapted from: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">ALBERT: A Lite BERT for Self-supervised Learning of Language Representations</span><span class="pub-meta">, Google and Toyota (ICLR 2020)</span><a class="pub-link" href="https://arxiv.org/abs/1909.11942" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Larger pretrained language models often perform better on downstream tasks, but their growth soon hits the memory limits of the accelerators and long training times. ALBERT ("A Lite BERT") lowers the memory consumption and increases the training speed of BERT, so it scales better. It sets new records on several benchmarks (GLUE, RACE, SQuAD) with fewer parameters than BERT-large, and an ALBERT configuration similar to BERT-large has 18× fewer parameters and trains about 1.7× faster. It introduces three techniques:
1. **Factorized embedding parameterization**, for parameter reduction. In BERT, the size of the vocabulary embeddings equals the hidden size $$H$$, so the embedding matrix grows with the hidden layers. ALBERT instead projects the one-hot vectors of a vocabulary of size $$V$$ into a lower-dimensional embedding space of size $$E$$, and then into the hidden space. This reduces the embedding parameters from $$O(V \times H)$$ to $$O(V \times E + E \times H)$$: with $$V = 30{,}000$$, $$E = 128$$ and $$H = 4096$$, from about 123M to about 4.4M. Separating the hidden size from the size of the vocabulary embeddings makes it easier to grow the hidden size, without growing the vocabulary embeddings.
2. **Cross-layer parameter sharing**, for parameter reduction. A BERT model with 12 layers learns unique parameters for each of its 12 Transformer encoder blocks. ALBERT learns the parameters of a single block and reuses them in all 12 layers, so the number of parameters does not grow with the depth of the network. The compute per token stays the same: only the parameter count shrinks. The authors report that the parameter reduction also acts as a regularization, which stabilizes the training and helps generalization. It does not improve the accuracy, however, and the [Google Research blog post](https://research.google/blog/albert-a-lite-bert-for-self-supervised-learning-of-language-representations/) on ALBERT sums up the trade-off: "This approach slightly diminishes the accuracy, but the more compact size is well worth the tradeoff".
3. **Self-supervised loss for sentence-order prediction (SOP)**, for better performance. BERT uses an additional loss, next-sentence prediction (NSP), a binary classification of whether two segments appear consecutively in the original text. The authors conjecture that NSP is ineffective because it is too easy: its negative examples pair segments from different documents, so the model can solve it by predicting the topic, which is easier to learn than coherence. SOP replaces it and focuses on inter-sentence coherence. Its positive examples are as in BERT, two consecutive segments of the same document, and its negative examples are the same two segments with their order swapped. Both segments then share their topic, which forces the model to learn finer-grained distinctions about discourse-level coherence.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2018</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Group Normalization</span><span class="pub-meta">, Facebook AI Research (ECCV 2018)</span><a class="pub-link" href="https://arxiv.org/abs/1803.08494" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Batch Normalization (BN) normalizes the features with the statistics of a mini-batch, so its error grows quickly as the batch gets smaller and the statistics become inaccurate, which limits BN in detection, segmentation and video, where the memory allows only small batches. Group Normalization (GN) divides the channels into $$G$$ groups and normalizes each sample within each group, over $$C/G$$ channels and the spatial axes $$(H, W)$$, so its computation does not depend on the batch size. With $$G = 1$$ it becomes Layer Normalization, and with $$G = C$$ Instance Normalization (the figure compares the four methods); the paper uses $$G = 32$$.

On ResNet-50 trained on ImageNet with a batch size of 2, GN has a 10.6% lower error than BN, and at typical batch sizes the two are comparable. GN also outperforms BN in object detection and segmentation on COCO, and in video classification on Kinetics. Layer and Instance Normalization also avoid normalizing along the batch dimension, and they are effective for training sequential models (RNN/LSTM) and generative models (GANs), but both have limited success in visual recognition, where GN gives better results.

{: .picture}
<img loading="lazy" width="79%" height="79%" src="{{ site.assets }}/publications/group_normalization.svg"/>

{: .caption}
<b>Normalization methods</b>. Each subplot shows a feature map tensor, with <i>N</i> as the batch axis, <i>C</i> as the channel axis, and (<i>H</i>, <i>W</i>) as the spatial axes. The pixels in blue are normalized by the same mean and variance, computed by aggregating the values of these pixels. Adapted from: [Group Normalization](https://arxiv.org/abs/1803.08494)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2017</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Neural Discrete Representation Learning (VQ-VAE)</span><span class="pub-meta">, DeepMind (NeurIPS 2017)</span><a class="pub-link" href="https://arxiv.org/abs/1711.00937" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The Vector Quantised Variational AutoEncoder (VQ-VAE) learns **discrete (not continuous) latent representations**, without supervision. It differs from VAEs in two ways: the encoder outputs discrete codes instead of continuous ones, and the prior is learned instead of static. Discrete codes also avoid the *posterior collapse* of VAEs, where a powerful autoregressive decoder ignores the latents.

In the figure, the encoder maps the input $$x$$ to a grid of $$D$$-dimensional vectors $$z_e(x)$$, and each vector is replaced by its nearest neighbor in a codebook of $$K$$ embeddings $$e_1, \ldots, e_K$$ (Eq. 2), which gives the input $$z_q(x)$$ of the decoder. The lookup has no gradient, so the backward pass copies the gradient $$\nabla_z L$$ of the decoder input unchanged to the encoder (straight-through): the encoder output and the decoder input share the same $$D$$-dimensional space, so this gradient still tells the encoder how to change its output to lower the reconstruction loss. The loss (Eq. 3) adds to the reconstruction loss a codebook loss, which moves the embeddings $$e$$ towards the encoder outputs, and a commitment loss, which keeps the encoder committed to an embedding:

$$
L = -\log p(x \mid z_q(x)) + \lVert \mathrm{sg}[z_e(x)] - e \rVert_2^2 + \beta \lVert z_e(x) - \mathrm{sg}[e] \rVert_2^2,
$$

where $$\mathrm{sg}$$ is the stop-gradient operator, and $$\beta = 0.25$$ in the experiments. The prior over the codes is uniform during this training. Afterwards, an autoregressive prior, a PixelCNN for images or a WaveNet for audio, is trained on the codes, and sampling from it generates new data. On ImageNet, a 128×128×3 image becomes a 32×32 grid of codes from $$K = 512$$ embeddings.

{: .picture}
<img loading="lazy" width="100%" height="100%" src="{{ site.assets }}/publications/rqvae.svg"/>

{: .caption}
Left: A figure describing the VQ-VAE. Right: Visualisation of the embedding space. The output of the encoder <i>z</i><sub>e</sub>(<i>x</i>) is mapped to the nearest point <i>e</i><sub>2</sub>. The gradient ∇<sub><i>z</i></sub><i>L</i> (in red) will push the encoder to change its output, which could alter the configuration in the next forward pass. Adapted from: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Mixed Precision Training</span><span class="pub-meta">, Baidu and NVIDIA (ICLR 2018)</span><a class="pub-link" href="https://arxiv.org/abs/1710.03740" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Mixed precision training trains with half-precision (FP16) floating-point numbers, without a loss of model accuracy and without changes to the hyperparameters. FP16, however, represents magnitudes only from about $$6 \times 10^{-8}$$ ($$2^{-24}$$, as a subnormal number) to 65,504, a much narrower range than FP32, so three techniques prevent the loss of critical information to underflows and overflows:
1. An FP32 master copy of the weights accumulates the gradients at each optimizer step, and is rounded to FP16 for the forward and backward passes. The updates, the gradients times the learning rate, can be too small for FP16 and underflow to zero, and adding an update to a weight $$2^{11}$$ or more times larger can round the update away.
2. Loss scaling preserves the gradients of small magnitude: scaling the loss before backpropagation shifts, by the chain rule, all the gradients into the range of FP16, and the weight gradients are unscaled before the weight update. The paper uses factors from 8 to 32K, as long as the scaled gradients stay below 65,504.
3. FP16 arithmetic accumulates into FP32 outputs, which are converted to FP16 before they are stored to memory: dot products accumulate their partial products in FP32, large reductions (batch normalization statistics, softmax) run in FP32, and point-wise operations, which are limited by the memory bandwidth, can use either precision.

The method matches the FP32 accuracy across a wide range of tasks (image classification and detection, speech recognition, machine translation, language modeling, GANs), with models of more than 100 million parameters, and nearly halves the memory requirements. Formats with the 8-bit exponent of FP32, such as [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), keep the range of FP32 and need no loss scaling.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2016</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Semi-Supervised Classification with Graph Convolutional Networks</span><span class="pub-meta">, University of Amsterdam (ICLR 2017)</span><a class="pub-link" href="https://arxiv.org/abs/1609.02907" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper introduces the Graph Convolutional Network (GCN) for semi-supervised node classification, where only a few nodes of a graph carry a label, such as some documents of a citation network. Its layers follow a simple propagation rule,

$$
H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}),
$$

where $$H^{(l)}$$ holds the node features at layer $$l$$, $$W^{(l)}$$ is a trainable weight matrix, $$\tilde{A} = A + I$$ is the adjacency matrix with self-loops, and $$\tilde{D}$$ its degree matrix: each layer combines the features of every node with those of its neighbors, normalized by their degrees. The rule is a first-order approximation of spectral graph convolutions. Like CNNs, GCNs learn features by aggregating information from neighboring nodes, but they generalize this from regular Euclidean structures, such as images, to arbitrary graphs. Because $$\tilde{A}$$ is sparse, a layer costs $$O(\vert E \vert C F)$$ for $$\vert E \vert$$ edges, $$C$$ input channels and $$F$$ output features.

The two-layer model of the paper is trained with the cross-entropy over the labeled nodes only, the nodes labeled $$Y_1$$ and $$Y_4$$ in the figure. On citation networks (Citeseer, Cora, Pubmed), with 0.3% to 5.2% of the nodes labeled, and on a knowledge graph (NELL), GCN outperforms related methods by a significant margin.

{: .picture}
<img loading="lazy" width="75.3%" height="75.3%" src="{{ site.assets }}/publications/graphconvnets.svg"/>

{: .caption}
(a) Schematic depiction of multi-layer Graph Convolutional Network (GCN) for semi-supervised learning with <i>C</i> input channels and <i>F</i> feature maps in the output layer. The graph structure (edges shown as black lines) is shared over layers, labels are denoted by <i>Y<sub>i</sub></i>. (b) t-SNE (Maaten & Hinton, 2008) visualization of hidden layer activations of a two-layer GCN trained on the Cora dataset (Sen et al., 2008) using 5% of labels. Colors denote document class. Adapted from: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2015</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</span><span class="pub-meta">, Google (ICML 2015)</span><a class="pub-link" href="https://arxiv.org/abs/1502.03167" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Batch Normalization (BatchNorm, BN) improves the training of neural networks by normalizing the inputs of each layer over a mini-batch. The paper motivates it with the *internal covariate shift*. *Covariate shift* means that the distribution of the features differs between the training and the test data, which breaks the i.i.d. assumption of most of ML, and *internal* covariate shift is the same drift inside the network: as the weights are updated, the distribution of the outputs of each layer changes, so the higher layers must keep adapting to it, which slows down learning. Networks train faster on [whitened](https://en.wikipedia.org/wiki/Whitening_transformation) inputs, with zero means, unit variances and no correlations, and BN makes the data that flows between the layers look like whitened data, at a lower cost. It normalizes each feature $$x$$ to zero mean and unit variance over the mini-batch, $$\hat{x} = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$$, and then scales and shifts it with learned parameters, $$y = \gamma \hat{x} + \beta$$, so that the layer can still represent the identity. At inference, moving averages of the mean and the variance replace the statistics of the batch.

BN allows much higher learning rates and a less careful initialization, and it also acts as a regularizer, which in some cases removes the need for dropout. It reaches the same accuracy with 14 times fewer training steps and beats the original model by a significant margin, and an ensemble of BN networks reaches a top-5 validation error of 4.9% on ImageNet, better than human raters. Later work, such as [Santurkar et al. (2018)](https://arxiv.org/abs/1805.11604), argues that BN helps mostly by smoothing the optimization landscape, not by reducing the internal covariate shift.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Siamese neural networks for one-shot image recognition</span><span class="pub-meta">, University of Toronto (ICML 2015 Deep Learning Workshop)</span><a class="pub-link" href="https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

One-shot learning must recognize a new class from a single example. The paper uses **siamese neural networks** (described in the contrastive loss entry) for **one-shot learning**, in two steps: (1) train a model to discriminate between pairs of images of the same class and of different classes, a verification task; (2) evaluate new categories with the learned feature mappings, without retraining. Each twin of the siamese network is a convolutional neural network, followed by a flattening and a feed-forward network. The two twins share their weights, and the prediction is a sigmoid of a weighted L1 distance between their feature vectors $$h_1$$ and $$h_2$$, as in the figure: $$p = \sigma(\sum_j \alpha_j d_j)$$, with $$d_j = \vert h_{1,j} - h_{2,j} \vert$$. The loss function is a **binary cross-entropy** with a regularizer. In 20-way one-shot classification, a test image is compared with the single example of each of 20 new classes, and the most similar example gives the class: on Omniglot, the network reaches 92% accuracy, where humans reach 95.5%.

{: .picture}
<img loading="lazy" width="51%" height="51%" src="{{ site.assets }}/publications/siamese_networks_2.svg"/>

{: .caption}
A simple 2 hidden layer siamese network for binary classification with logistic prediction <i>p</i>. The structure of the network is replicated across the top and bottom sections to form twin networks, with shared weight matrices at each layer; the output weights <i>α<sub>j</sub></i> weight the component-wise distances <i>d<sub>j</sub></i>. Adapted from: [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Neural Machine Translation by Jointly Learning to Align and Translate (and Attention Mechanism)</span><span class="pub-meta">, D. Bahdanau, K. Cho, Y. Bengio (ICLR 2015)</span><a class="pub-link" href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The RNN encoder–decoder for translation ([Cho et al., 2014](https://arxiv.org/abs/1406.1078); concurrent with [Sequence to Sequence Learning with Neural Networks (Google, NeurIPS 2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)) compresses the whole source sentence into a single fixed-length vector, a bottleneck that makes the translation worse as the sentences get longer. The paper extends the encoder–decoder with attention: at each decoding step, the decoder learns to (soft-)align to the hidden states of every encoder step, not only the last one.

In the figure, the encoder is a bidirectional RNN, so its state $$h_j$$ concatenates a forward and a backward state, and summarizes the words around the source word $$x_j$$. The context of the target word $$y_t$$ is $$c_t = \sum_{j=1}^{T} \alpha_{tj} h_j$$, a weighted sum of the encoder states of the $$T$$ source words. The weights $$\alpha_{tj}$$ are the softmax over $$j$$ of the scores $$e_{tj} = v_a^\top \tanh(W_a s_{t-1} + U_a h_j)$$, where $$s_{t-1}$$ is the previous decoder state, and this alignment model, learned jointly with the rest of the model, measures how well the source words around position $$j$$ match the target word at position $$t$$. Each target word thus costs $$O(T)$$ instead of $$O(1)$$. The model reaches a translation performance comparable to the existing state-of-the-art phrase-based system on English-to-French, and degrades much less on long sentences.

{: .picture}
<img loading="lazy" width="25.1%" height="25.1%" src="{{ site.assets }}/publications/attention_mech.svg"/>

{: .caption}
The graphical illustration of the proposed model trying to generate the <i>t</i>-th target word <i>y<sub>t</sub></i> given a source sentence (<i>x</i><sub>1</sub>, <i>x</i><sub>2</sub>, . . . , <i>x<sub>T</sub></i>). Adapted from: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Spatial Transformer Networks</span><span class="pub-meta">, DeepMind (NeurIPS 2015)</span><a class="pub-link" href="https://arxiv.org/abs/1506.02025" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The feature maps of a CNN are not invariant to large transformations of the input, such as large translations, rotations or changes of scale. A spatial transformer is a differentiable module that can be inserted anywhere in a CNN to transform feature maps spatially, conditioned on the input, and learned without extra supervision. A localisation network regresses the parameters of the transformation (for example affine, projective or [thin-plate spline](https://en.wikipedia.org/wiki/Thin_plate_spline)), a grid generator produces the sampling grid, and a differentiable sampler, for example bilinear, warps the input. For an affine transformation, the 6 parameters form a matrix $$A_\theta \in \mathbb{R}^{2 \times 3}$$, and each output pixel at $$(x^t, y^t)$$ samples the input at $$(x^s, y^s)^\top = A_\theta (x^t, y^t, 1)^\top$$. The sampler is differentiable with respect to both the input and the grid, so the localisation network learns from the loss of the task alone.

The network thus learns invariance to translation, scale, rotation and more generic warping: in the figure, a spatial transformer crops, scales and straightens distorted MNIST digits for a fully connected classifier. Spatial transformers reduce the error of CNNs on distorted MNIST, and gave state-of-the-art results at the time on Street View House Numbers and CUB-200-2011 birds.

{: .picture}
<img loading="lazy" width="37%" height="37%" src="{{ site.assets }}/publications/stn.svg"/>

{: .caption}
The result of using a spatial transformer as the first layer of a fully-connected network trained for distorted MNIST digit classification. (a) The input to the spatial transformer network is an image of an MNIST digit that is distorted with random translation, scale, rotation, and clutter. (b) The localisation network of the spatial transformer predicts a transformation to apply to the input image. (c) The output of the spatial transformer, after applying the transformation. (d) The classification prediction produced by the subsequent fully-connected network on the output of the spatial transformer. The spatial transformer network (a CNN including a spatial transformer module) is trained end-to-end with only class labels – no knowledge of the groundtruth transformations is given to the system. Adapted from: [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2014</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Deeply-supervised Nets</span><span class="pub-meta">, UCSD and Microsoft (AISTATS 2015)</span><a class="pub-link" href="https://arxiv.org/abs/1409.5185" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

A deeply supervised model supervises its intermediate layers explicitly during training, in addition to its final output layer. Without this supervision, the hidden layers learn only from the gradients propagated back from the final layer, which can vanish or explode in a deep network, and nothing forces their features to be useful for the task. With it, the earlier layers learn features that are useful for the task directly. Deep supervision thus addresses vanishing gradients and poor feature learning in the intermediate layers, and it is common in image segmentation, object detection and biomedical image analysis.

The objective of each intermediate layer is a *companion objective*, which acts as an additional constraint, or a new regularization, of the learning. Each hidden layer $$m$$ feeds a classifier of its own, an SVM with weights $$w^{(m)}$$, and the [squared hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) $$\ell$$ of this classifier is the companion loss of the layer. The overall objective adds, to the objective of the output layer, a term $$\alpha_m [\Vert w^{(m)} \Vert^2 + \ell(W, w^{(m)}) - \gamma]_+$$ for each hidden layer, where $$W$$ holds the weights of the network, $$\alpha_m$$ balances the two objectives, and the threshold $$\gamma$$ is a hyperparameter. Because of the hinge $$[\cdot]_+$$, once the companion objective of a hidden layer reaches $$\gamma$$ or less, it vanishes and no longer plays a role in the learning. Empirically, the companion objective (1) acts as a kind of **feature regularization**, although an unusual one, which reduces the test error significantly but not necessarily the training error; and (2) gives a faster convergence, in particular with little training data. Deeply-supervised Nets gave state-of-the-art results at the time on MNIST, CIFAR-10, CIFAR-100 and SVHN.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Dropout: a simple way to prevent neural networks from overfitting</span><span class="pub-meta">, Univ. Toronto (Journal of ML Research 2014)</span><a class="pub-link" href="https://jmlr.org/papers/v15/srivastava14a.html" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Large neural networks overfit easily, and the usual remedy, averaging the predictions of many separately trained networks, is too expensive for them. Dropout approximates this average at a low cost. It **randomly drops neurons, in different layers, during training: each one with probability $$p$$** (the paper uses $$p$$ for the probability of keeping a neuron), together with its connections. Each training minibatch thus samples a new "thinned" network, as in the figure, and all the thinned networks share their weights. **At test time, all neurons are used, with their outgoing weights multiplied by $$1 - p$$**, so that the input of every neuron equals its expected input during training, which approximates the average of the exponentially many thinned networks. Max-norm regularization, a decaying learning rate and a high momentum improve dropout further.

Dropout **reduces overfitting**: the network can never rely on any given activation, so it learns redundant ways of solving the task with several neurons, and its hidden activations also become sparser. Dropping 20% of the input units and 50% of the hidden units ($$p = 0.2$$ and $$p = 0.5$$) was often close to optimal in the paper. A network with dropout, however, takes 2 to 3 times longer to train, because the parameter updates are very noisy. Fully connected layers hold most of the parameters, so they are the most prone to overfitting, and dropout in these layers **improves the generalization** of the model.

{: .picture}
<img loading="lazy" width="54.8%" height="54.8%" src="{{ site.assets }}/deep-neural-networks/dropout.svg"/>

{: .caption}
Dropout. Left: a standard network with two hidden layers ($$p = 0$$). Right: a thinned network produced by dropout with $$p = 0.5$$, where $$p$$ is the probability of dropping a unit: the crossed units are dropped, with their connections. Adapted from: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2006</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">Dimensionality Reduction by Learning an Invariant Mapping (contrastive loss)</span><span class="pub-meta">, Hadsell, Chopra & LeCun, New York Uni (CVPR 2006)</span><a class="pub-link" href="http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Dimensionality reduction maps high-dimensional inputs to a low-dimensional space. Most methods, however, need a meaningful distance measure in the input space, and they do not give a function that maps new samples, whose relation to the training data is unknown. The paper presents Dimensionality Reduction by Learning an Invariant Mapping (DrLIM): it learns a globally coherent, non-linear function that maps high-dimensional inputs to low-dimensional outputs, from the neighborhood relations between the samples in the input space alone. It introduces the **contrastive loss function**, which trains a **siamese network** to learn a representation space where similar samples are close together and dissimilar samples are far apart.

A siamese network compares two inputs by learning their similarity. It has two identical subnetworks (hence the name "siamese"), two copies of the same function $$G_W$$ with the same parameters $$W$$, and a cost module. Each subnetwork processes one of the two inputs, and the cost module combines their outputs into a similarity score or a distance. Since the subnetworks share their parameters, the total gradient is the sum of the contributions of the two.

The input of the system is a pair of images, one for each subnetwork, and a similarity label $$y$$: 1 for similar images and 0 for dissimilar ones (the paper uses the opposite convention). The subnetworks map the images to $$G_W(X_1)$$ and $$G_W(X_2)$$, and the cost module computes their Euclidean distance $$D_W = \lVert G_W(X_1) - G_W(X_2) \rVert_2$$. The loss is

$$
L = \frac{1}{2} \, y \, D_W^2 + \frac{1}{2} \, (1 - y) \max(0, m - D_W)^2,
$$

where the margin $$m$$ is a hyperparameter that sets the minimum distance of dissimilar pairs. The first term pulls similar pairs together. The second term pushes dissimilar pairs apart, but only while their distance is within the margin: pairs that are already further apart than $$m$$ do not contribute to the loss.

</div>
</details>
</td>
</tr>
</table>

{::options parse_block_html="false" /}
