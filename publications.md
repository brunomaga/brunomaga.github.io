---
layout: post
title: Publications bookmark
permalink: /publications/
---

{::options parse_block_html="true" /}

Summaries of publications on HPC and ML, continuously updated.

<table class="label-table pub-table">
<tr>
<td><span class="post-label">2026</span></td>
<td markdown="block">
<details>
<summary markdown="span"><span class="pub-title">DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2606.19348" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek-V4 is a preview of the next MoE series of DeepSeek, with two models: **DeepSeek-V4-Pro** (1.6T parameters, 49B activated, pre-trained on 33T tokens) and **DeepSeek-V4-Flash** (284B parameters, 13B activated, 32T tokens). Both have a native context of one million tokens. The models keep the DeepSeekMoE framework and the MTP modules of DeepSeek-V3, with three small changes to the MoE: a Sqrt(Softplus) affinity function instead of a sigmoid, no limit on the number of nodes that a token is routed to, and hash-routed MoE layers in place of the first dense FFN layers. Three larger changes are new:

1. **Hybrid attention**, with CSA and HCA layers interleaved. *Compressed Sparse Attention* (CSA) compresses the KV entries of every $$m$$ tokens into one entry, a learned softmax-weighted combination over overlapping windows. It then applies DeepSeek Sparse Attention to the compressed entries: the lightning indexer selects the top-k compressed entries for each query, and a small sliding window of recent uncompressed tokens adds the local detail. *Heavily Compressed Attention* (HCA) compresses every $$m' \gg m$$ tokens into one entry and attends to all of them. The attention also uses a shared-KV layout, as in MQA, with a grouped output projection, partial RoPE and attention sinks.
2. **mHC** (Manifold-Constrained Hyper-Connections, see its entry) instead of plain residual connections. The residual stream is widened to $$n_{hc}$$ parallel streams. Their mixing matrix is projected onto the [doubly stochastic matrices](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix) with [Sinkhorn-Knopp](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) iterations, which keeps the signal propagation stable.
3. The **Muon** optimizer, for faster convergence and more stable training.

At a context of 1M tokens, V4-Pro needs 27% of the inference FLOPs per token and 10% of the KV cache of DeepSeek-V3.2. V4-Flash needs 10% and 7%. The weights of the routed experts are stored in FP4, with FP4 quantization-aware training during post-training.

The paper also describes the infrastructure: a single fused MoE mega-kernel that overlaps computation, communication and memory access in expert parallelism; kernels written in TileLang; batch-invariant, deterministic kernels for bitwise reproducibility; two-stage context parallelism for the compressed attention; and a heterogeneous KV cache, with storage on disk to reuse shared prefixes. Post-training trains one specialist per domain (SFT, then RL with GRPO) and merges them into one model with on-policy distillation. The maximum-reasoning mode, V4-Pro-Max, is reported as the best open model at release. It is ahead of GPT-5.2 and Gemini-3.0-Pro on standard reasoning benchmarks, and slightly behind GPT-5.4 and Gemini-3.1-Pro.

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

Nemotron 3 Super is a hybrid [Mamba-2](https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)) and attention MoE model with 120B parameters, 12B of them active. It is the first model of the Nemotron 3 family (see the Nemotron 3 entry) that is **pre-trained in NVFP4**, uses **LatentMoE** experts, and includes **MTP layers** for native speculative decoding. It was pre-trained on 25T tokens, then post-trained with SFT and RL, with a focus on agentic tasks.

The model supports a context of up to 1M tokens. On common benchmarks, its accuracy is comparable to that of GPT-OSS-120B and Qwen3.5-122B, with up to 2.2× and 7.5× higher inference throughput, respectively. The datasets are open, and so are the base, post-trained and quantized checkpoints (BF16, FP8 and NVFP4). The model tests the LatentMoE and NVFP4 entries at scale: the LatentMoE experts and the 4-bit pre-training both hold up on a 120B model trained on 25T tokens.

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

FlashAttention-4 (FA4) redesigns the attention kernel for NVIDIA Blackwell GPUs (B200, GB200). It addresses what the authors call **asymmetric hardware scaling**. From Hopper to Blackwell, the dense BF16 throughput of the tensor cores grew from about 1 to about 2.25 petaFLOPS. The other units did not grow as much: the exponential unit (MUFU.EX2) that computes the softmax, and the bandwidth of the shared memory. On Blackwell, the softmax is therefore no longer a small step between the two matrix multiplications: it is the bottleneck. FA4 keeps the tensor cores busy by overlapping the matrix multiplications with the softmax and the memory operations.

The kernel has four main parts:

- New forward and backward software pipelines use the fully asynchronous MMA instructions and the larger tiles of Blackwell, to overlap the tensor cores, the exponentials and the memory operations.
- The forward pass also computes exponentials in software, with a polynomial approximation on the FMA units, which offloads the exponential unit. It rescales the online softmax only when the running maximum grows by more than a threshold, which skips most of the rescaling and keeps the result exact.
- The backward pass is limited by the shared-memory traffic. FA4 stores intermediate results in the new tensor memory (TMEM) of Blackwell. It also uses the new 2-CTA MMA mode, which reduces the shared-memory traffic further and halves the number of global atomic additions.
- The kernel is written in **CuTe-DSL**, in Python. It compiles 20 to 30 times faster than C++ templates, with the same expressivity, so new attention variants are faster to prototype.

FA4 reaches up to 1613 TFLOPs/s on a B200 in BF16, 71% of the peak (the blog post of the release quotes 1605). In the forward pass, it is up to 1.3× faster than cuDNN 9.13 and 2.7× faster than Triton. In the backward pass, it is faster than all baselines at long sequence lengths. FlashAttention-3 runs only on Hopper, so it is not a baseline. Several techniques of FA4 are also in cuDNN, from versions 9.13 and 9.14.

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

This follow-up to LatentMoE addresses three limitations of expert parallelism (EP). First, the communication volume and the latency of the all-to-all grow linearly with the number of activated experts $$k$$, because each token is sent to each of its $$k$$ experts. Second, the load of the experts is imbalanced, so the all-to-all waits for the longest queue. Third, the traffic depends on the routing, so an extra all-to-all exchanges the metadata. LatentMoE sends fewer bytes per token, but it still communicates *after* the routing, so the imbalance and the non-determinism remain.

**Multi-Head LatentMoE** multiplies each token by a learned $$d \times d$$ matrix and splits the result into $$N_h$$ sub-tokens. Each sub-token goes through an independent small MoE, with its own router and experts. The outputs are concatenated and projected back. **Head Parallel (HP)** uses this structure to move the all-to-all *before* the routing. Each GPU owns a subset of the heads, receives all the sub-tokens of these heads, and runs the routing and the experts locally. As a result:

- each token is sent exactly once, so the communication does not grow with $$k$$;
- every GPU sends and receives the same amount of data, so the traffic is balanced, no metadata is exchanged, and hot experts cannot cause out-of-memory errors;
- HP combines with EP to scale beyond $$N_h$$ GPUs.

So that the memory traffic does not grow $$N_h$$ times, the authors add exact IO-aware kernels: an online top-k router in Triton that keeps the scores in SRAM, and the expert computation written as block-sparse attention with FlexAttention.

At small scale (0.2B active and up to 4.2B total parameters, trained on 10B tokens of FineWeb-Edu), Multi-Head LatentMoE with HP trains up to 1.61× faster than a standard MoE with EP, at the same quality, and sends 25% of the inter-GPU traffic at $$k = 4$$. With experts twice as fine-grained, it reaches a higher accuracy and is still 1.11× faster.

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

Collectives can be offloaded to the network: to the NICs (edge-INC) or to the switches (core-INC, as in NVIDIA SHARP). This can halve the traffic of an all-reduce. Low-precision types, sparse reductions, reproducibility and encryption make the offload harder. [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) also limits the speedup, so the authors expect a slow adoption.

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

LatentMoE projects each token from the hidden dimension $$d$$ of the model to a smaller latent dimension $$\ell$$ before it is dispatched to the routed experts (see the figure). The router still computes its gating scores from the full $$d$$-dimensional token: only the dispatch and combine traffic and the routed experts are in the latent space. After the combine all-to-all, the weighted sum of the outputs of the routed experts is projected back from $$\ell$$ to $$d$$. It is then added to the output of the shared experts, which work in the full hidden dimension. All the routed experts share a single down-projection and a single up-projection, and the weights of each expert shrink by $$d/\ell$$, so fewer bytes are loaded from memory.

The [roofline](https://en.wikipedia.org/wiki/Roofline_model) analysis of the paper shows that the experts are bound by the HBM bandwidth, which loads their weights, at the small batch sizes of latency-critical serving, and by the all-to-all in throughput-oriented serving. Both costs scale with $$d$$. The number of active experts and the width of the expert FFNs set the nonlinear capacity of the model, so $$d$$ is the dimension to shrink.

The saved budget pays for *more and finer experts* at the same cost. With a 4× compression, a standard MoE with 128 experts and top-6 routing scores 48.30% on MMLU-Pro. A LatentMoE with 512 experts and top-22 routing, as reported in the Nemotron 3 white paper, scores 52.87% on MMLU-Pro and 55.14% on code (+3.19%), with similar parameter and FLOP counts (hybrid models with 8B active and 73B total parameters, trained on 1T tokens). Its measured throughput per GPU is within about 6% of the baseline at higher concurrency. A design-space exploration up to 95B parameters, over 1T tokens, shows that LatentMoE beats a standard MoE in accuracy per FLOP and per parameter. Nemotron 3 Super and Ultra use it.

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

Hyper-Connections (HC) widen the residual stream into $$n$$ parallel streams, mixed by learnable matrices. The width of the residual stream is then independent of the width of the layers, at a small compute cost. HC improves accuracy, but the unconstrained mixing breaks the identity mapping that makes deep residual networks trainable. In a 27B model, the signals grow across the layers and the training loss spikes. HC also adds a large memory-access overhead.

mHC constrains the residual mixing matrix of each layer to the doubly stochastic matrices, the [Birkhoff polytope](https://en.wikipedia.org/wiki/Birkhoff_polytope): non-negative matrices whose rows and columns each sum to 1. The projection uses Sinkhorn-Knopp iterations, which normalize the rows and the columns in turn. Sigmoids keep the read-in and write-out mappings non-negative and bounded. A doubly stochastic matrix has a spectral norm of 1, so the mixing does not amplify the signal, and a product of doubly stochastic matrices is doubly stochastic. The signal therefore stays stable at any depth. With $$n = 1$$, the mixing is the identity, as in a standard residual connection. Kernel fusion, recomputation, and overlap with the communication of DualPipe reduce the cost of a 4× wider residual stream ($$n = 4$$) to about 6.7% more training time.

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

MPK automates the megakernel of the Hazy Research Llama-1B post (see its entry), and comes from the group behind Mirage. It is the first compiler and runtime that turns multi-GPU inference into a single persistent megakernel automatically, instead of by hand. Its main abstraction is an **SM-level task graph**: the data dependencies are tracked per streaming multiprocessor (SM), not per kernel. This allows optimizations that one kernel per operator cannot express, such as software pipelining across operators (the next operator loads its data while the current one computes), and a fine-grained overlap of computation with the communication between GPUs.

The compiler lowers a tensor program to an optimized SM-level task graph and generates CUDA code for each task. A parallel runtime inside the kernel runs the tasks in one persistent kernel, with decentralized scheduling across the SMs, so there are no kernel launches or host round trips between operators. MPK is open source, in the Mirage repository. Its end-to-end inference latency is up to 1.7× lower than that of existing LLM serving systems that launch one kernel per operator.

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

This is the full report behind the DeepSeek-V3.2-Exp model card (see its entry). The architecture is the same as V3.2-Exp: the only change from DeepSeek-V3.1-Terminus is **DeepSeek Sparse Attention (DSA)**, added through continued training. A *lightning indexer* (a few heads with ReLU activations, which can run in FP8) scores each preceding token for the current query. A fine-grained selection step keeps only the top-k key-value entries ($$k = 2048$$) for the main MLA attention. The cost of the core attention drops from $$O(L^2)$$ to $$O(Lk)$$, for a context of $$L$$ tokens. The indexer itself is still quadratic, but much cheaper. The indexer is first trained alone, with a KL loss to match the dense attention distribution, while the main model is frozen. The whole model is then trained with sparse attention.

The rest of the report is about post-training. It describes a scaled and stabilized RL recipe based on GRPO, with a post-training compute budget above 10% of the pre-training cost. A large pipeline synthesizes agentic tasks (over 1,800 environments and 85,000 complex instructions), which makes V3.2 the first DeepSeek model that thinks inside tool use. DeepSeek-V3.2 performs comparably to GPT-5. The high-compute variant **DeepSeek-V3.2-Speciale** (longer reasoning, no tool use) reaches gold-medal level at IMO 2025, CMO 2025, IOI 2025 and the ICPC World Finals 2025. It matches Gemini-3.0-Pro in reasoning, but needs more tokens for the same results.

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

Kimi Linear is a hybrid linear-attention architecture. With the same training recipe, it *outperforms* full attention on short-context, long-context and RL post-training tasks, which the authors present as a first. Its core is **Kimi Delta Attention (KDA)**, a linear attention with a fixed-size recurrent state. KDA extends Gated DeltaNet with finer-grained gating: each channel (feature dimension) has its own decay rate, instead of one forget gate per head, so the limited state memory is used more selectively. For hardware efficiency, the state transition of KDA uses a specialized Diagonal-Plus-Low-Rank (DPLR) form, with a chunkwise-parallel algorithm. This algorithm needs much less computation than a general DPLR form, and stays consistent with the classical [delta rule](https://en.wikipedia.org/wiki/Delta_rule).

The model interleaves KDA layers and full-attention MLA layers in a fixed 3:1 ratio, with 48B total and 3B activated parameters. With the same training recipe, it beats a full-MLA baseline on all the tasks evaluated. It reduces the KV cache by up to 75%, since only one layer in four keeps a growing cache, and reaches up to 6× the decoding throughput at a context of 1M tokens. The KDA kernels, a vLLM implementation, and the pre-trained and instruction-tuned checkpoints are open.

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

The paper gives a recipe to pre-train LLMs in 4-bit floating point with **NVFP4**, the microscaling format of Blackwell GPUs. NVFP4 stores FP4 (E2M1) values in blocks of 16 elements. Each block has an FP8 (E4M3) scale, and each tensor an FP32 scale. MXFP4 uses blocks of 32 elements with power-of-two (E8M0) scales. The smaller blocks and more precise scales of NVFP4 capture the local dynamic range much better. FP4 matrix multiplications run at 2 to 3 times the FP8 throughput on Blackwell (GB200, GB300), but naive 4-bit training diverges, or loses accuracy over long training runs.

The method has four parts:

1. **random [Hadamard transforms](https://en.wikipedia.org/wiki/Hadamard_transform)** on the inputs of the weight-gradient matrix multiplications, which spread the outliers of a block into a distribution closer to a Gaussian;
2. **2D block scaling of the weights** (16×16 blocks), so that the forward and backward passes see the same quantized weights;
3. **[stochastic rounding](https://en.wikipedia.org/wiki/Rounding#Stochastic_rounding)** of the gradients, which avoids the bias of round-to-nearest;
4. a small fraction of numerically sensitive linear layers kept in higher precision.

The recipe trains a 12B hybrid Mamba-Transformer on 10T tokens, the longest publicly documented 4-bit training run at the time. The loss stays within about 1% relative error of the FP8 baseline during the stable phase, and slightly above 1.5% late in the learning-rate decay. The downstream accuracies match those of FP8. On an 8B model, MXFP4 needs about 36% more tokens to reach the loss that NVFP4 reaches at 1T tokens. Transformer Engine supports the recipe, and Nemotron 3 Super was pre-trained with it (see its entry).

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

This ISCA 2025 paper (industry track) reviews the hardware and model co-design of DeepSeek-V3 and R1. It complements the DeepSeek-V3 report (see its entry): it explains how a 671B model was trained on only 2,048 H800 GPUs, and what hardware should change.

On the model side, the paper quantifies the design choices. MLA reduces the KV cache to about 70 KB per token, against 327 KB for Qwen-2.5 72B and 516 KB for LLaMA-3.1 405B, which both use GQA. The MoE design needs about 250 GFLOPs of training compute per token, against about 2,448 for the dense LLaMA-3.1 405B. The other choices are FP8 mixed-precision training, and MTP for faster speculative decoding. On the infrastructure side, the paper describes the **multi-plane two-layer [fat-tree](https://en.wikipedia.org/wiki/Fat_tree)** network: the NIC of each GPU is on its own network plane, with eight planes per node. This network lowers the cost of the cluster network and scales to thousands of GPUs. Expert parallelism and node-limited routing are designed around the bandwidth asymmetry of the H800, whose NVLink is much faster than its InfiniBand.

The second half lists what future AI hardware needs, from the bottlenecks that DeepSeek hit: more precise low-precision compute units (for example, a higher FP8 accumulation precision and native support for fine-grained block scaling), scale-up and scale-out networks that converge, and low-latency communication fabrics. It explains *why* the communication-overlap methods of other entries were needed on bandwidth-limited hardware.

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

According to the NSA paper, most sparse-attention methods fail in practice for two reasons. First, their theoretical savings do not become wall-clock speedups: their memory accesses are scattered token by token, they fit poorly with GQA and MQA, where heads share the KV entries, or they speed up only the prefill or only the decode. Second, they are applied after training, to models trained with dense attention, so the sparsity pattern is never trained end to end. NSA addresses both. Each query attends through three branches, whose outputs are combined by learned gates:

1. **compressed** attention over coarse summaries of token blocks (blocks of 32 tokens with a stride of 16, compressed by a learned MLP);
2. **selected** attention over the top-n most relevant fine-grained blocks, chosen from the attention scores of the compression branch, so the selection costs almost nothing;
3. a **sliding window** over the last 512 tokens, for the local context.

The kernel is hardware-aligned. All the heads of a GQA group share the same block selection, so each group loads contiguous KV blocks into SRAM once. This keeps the arithmetic intensity high on the Tensor Cores, in the forward pass, the backward pass and the decoding. A 27B-parameter MoE model (about 3B active), pre-trained from scratch with NSA on 260B tokens, matches or beats the full-attention baseline on general benchmarks, long-context retrieval and chain-of-thought reasoning. At a context of 64K tokens, it is up to 11.6× faster in decoding, 9.0× in the forward pass and 6.0× in the backward pass. Block compression and learned selection appear again in DeepSeek Sparse Attention (DeepSeek-V3.2) and in the CSA and HCA layers of DeepSeek-V4.

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

FlashInfer is the attention kernel library under many serving systems, including SGLang, vLLM and MLC-Engine. It handles three properties of serving workloads that a single FlashAttention kernel does not cover:

- **KV-cache heterogeneity**: paged, prefix-shared and sparse KV layouts are all stored in one **block-sparse format**, with composable formats for shared prefixes. The same kernels serve all the layouts, and shared KV entries are loaded once.
- **Customization**: an attention template, with custom masks or logit transforms for example, is JIT-compiled into optimized block-sparse kernels, so new attention variants need no hand-written CUDA.
- **Dynamism**: a load-balanced scheduler, inspired by Stream-K, splits the work of variable-length requests across the SMs. It stays compatible with CUDA Graphs, which need a static launch configuration.

FlashInfer lowers the inter-token latency by 29% to 69% against compiler backends, on an LLM serving benchmark. It also lowers the latency of long-context inference by 28% to 30%, and speeds up LLM serving with parallel generation by 13% to 17%.

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

Nemotron 3 is a family of open models from NVIDIA, in three sizes (Nano, Super and Ultra), with a context of up to 1M tokens. The white paper combines four efficiency techniques:

1. **LatentMoE** (Super and Ultra), see its entry.
2. A **hybrid Mamba-Transformer MoE** architecture, for inference efficiency. A self-attention layer attends over a KV cache that grows linearly during generation, while a Mamba-2 layer stores only a constant-size state. Nemotron 3 therefore interleaves the MoE layers mostly with Mamba-2 layers, and keeps only a few attention layers. These layers use no RoPE, since the Mamba layers already provide the positional information.
3. **Multi-token prediction** (Super and Ultra). Predicting several future tokens gives a richer training signal: about 2.4% higher average benchmark scores in an ablation on an MoE with 8B active parameters. The MTP module also serves as the draft model of speculative decoding: a lightweight module reaches about 97% acceptance on the first two predicted tokens.
4. **NVFP4 training** (Super and Ultra): stable and accurate pre-training of a hybrid Mamba-MoE on up to 25T tokens. The weights, activations and gradients are quantized to NVFP4, so the matrix multiplications of the forward pass (fprop), of the input gradients (dgrad) and of the weight gradients (wgrad) all run in FP4, at 3× the peak throughput of FP8 on GB300. The recipe uses 2D block scaling for the weights, random Hadamard transforms on the inputs of wgrad, stochastic rounding of the gradients, and keeps the last 15% of the network in high precision. Sensitive layers keep a higher precision too. The QKV and attention projections stay in BF16 (the few GQA layers have only 2 KV heads). The output projections of the Mamba layers stay in MXFP8, since NVFP4 flushes up to 40% of their values to zero. The relative loss gap to BF16 is below 1% on Nano, and below 0.6% on the MoE with 8B active parameters: the gap shrinks as the model grows.

The architecture balances three layer types: MoE layers to scale the parameters sparsely, a few attention layers to route information precisely between all tokens, and Mamba-2 layers to model sequences at a fixed cost. Nemotron 3 Nano 30B-A3B reaches 3.3× the throughput of Qwen3-30B-A3B (8K input and 16K output tokens), and stays competitive on reasoning and long-context benchmarks (RULER at 1M tokens).

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

Distributed MoE training usually overlaps the communication of expert parallelism with computation. This communication can take about 47% of the total execution time with popular models and frameworks. According to Comet, existing schemes overlap at a coarse grain: they split an MoE layer into a few large chunks, which either leaves communication exposed or shrinks the GEMM tiles enough to slow the kernels down. FLUX finds the same tension for tensor parallelism (see its entry).

Comet has two parts:

1. **Dependency resolving**: Comet finds which pieces of computation depend on which pieces of communication, and schedules them in a fine-grained pipeline, so that each piece starts as soon as its data is ready.
2. **Adaptive workload assignment**: Comet splits the SMs of the GPU between communication and computation. The best split changes with the sequence length and the parallelization strategy, so Comet tunes it per configuration instead of fixing it.

The best number of SMs for communication depends on the layer, the input shape and the degree of parallelism:

{: .picture}
<img loading="lazy" width="90%" height="90%" src="{{ site.assets }}/publications/comet.svg"/>

{: .caption}
Duration of the MoE layer1 kernel with varying number of thread blocks assigned for communication ($$n_c$$). The total number of thread blocks is identical to the number of SMs on Hopper(132). The figure shows four cases with different parallelisms. Adapted from: [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811)

Comet ships several pre-compiled kernels, each with a different split of the SMs. Before deployment, the best kernel for each configuration is profiled and stored as metadata, and at runtime Comet selects the kernel from this metadata. An MoE layer runs a dispatch, two matrix multiplications and a combine. Comet fuses the dispatch with the first matrix multiplication into one kernel (layer0), and the second matrix multiplication with the combine into another kernel (layer1). Both kernels interleave communication and computation tile by tile.

{: .picture}
<img loading="lazy" width="66.7%" height="66.7%" src="{{ site.assets }}/publications/comet_2.svg"/>

{: .caption}
<b>Top:</b> The producer-consumer modeling of layer0 (left) and layer1 (right) of an MoE layer. The global size of the shared tensor is $$(M \times topk, N)$$ for both layer0 and layer1. <b>Bottom:</b> Decompose and reschedule the shared tensor in MoE layer0. In this illustration, three experts are located on Rank 0, each requiring both local and remote data for computation. Adapted from: [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811)

On Hopper, the mechanism is thread-block specialization. Each fused kernel launches a fixed set of thread blocks, and each block either communicates or computes, never both. Each thread block occupies one SM, so this dedicates some SMs to communication and the rest to computation. The compute blocks call the device-side GEMM functions of CUTLASS, unchanged.

Dependency resolving works on the *shared tensor* between a communication operation and the computation that uses its data: the buffer that the dispatch writes and the GEMM reads, or that the GEMM writes and the combine reads. Comet resolves the dependencies in two steps:

1. it decomposes the shared tensor along one dimension;
2. it reschedules the computation to follow the arrival order of the data: for each tile of the GEMM, it finds the slices of the shared tensor that the tile needs, and orders the tiles so that they consume the slices in the order in which the communication fills them.

On average, a single MoE layer runs 1.96× faster, and the end-to-end execution 1.71× faster. Comet runs on production clusters of about ten thousand GPUs, where it saved millions of GPU hours.

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

Helix Parallelism targets real-time decoding with a KV history of *millions of tokens*, under a tight token-to-token latency (TTL) budget. Two costs dominate in this regime: reading the **FFN weights** at every step, and streaming the large **KV cache** from DRAM. The standard fix for one cost makes the other worse. Tensor parallelism (TP) spreads the reads of the FFN weights well, but it scales poorly for attention: once the TP width exceeds the number of KV heads, the KV cache is *duplicated* across GPUs. This duplication consumes memory bandwidth, caps the parallelism and limits the batch size. The KV reads from DRAM also grow linearly with the batch size. No single sharding strategy relieves both costs.

Helix changes the parallel layout between the attention and the FFN of each layer, within one decode step, on the same GPUs:

- **Attention** runs with KV parallelism (KVP), combined with TP over the KV heads (TPA, at most the number of KV heads), on $$N = KVP \times TPA$$ GPUs. The KV cache is sharded along the sequence dimension, and no GPU holds a duplicate. Each decode step runs as follows:
  - every KVP rank computes the full QKV projection of the new token locally, since duplicating this small projection is cheaper than gathering the query across GPUs;
  - each GPU computes the attention of that query over its local slice of the KV cache, and produces a partial output and a log-sum-exp (the softmax normalizer) per token, as in FlashDecoding;
  - a single all-to-all over the query heads exchanges the partial results, and each GPU rescales and sums them into the exact attention output. The traffic grows with the batch size and the hidden size, but not with the KV length, since the KV cache never moves. It therefore stays constant as the context grows to millions of tokens.
- **FFN**: the same $$N$$ GPUs switch to TP. The all-to-all already left the attention output in the TP layout of the output projection. A dense FFN runs with TP, each GPU holding a slice of the weight matrices. An MoE runs with TP × expert parallelism (EP), with the experts distributed across the GPUs.

To hide the communication, Helix adds HOP-B (batch-wise overlap): the all-to-all of one request overlaps with the attention computation of the next.

For DeepSeek-R1 on Blackwell, Helix reduces the TTL by up to 1.5× at a fixed batch size, and supports up to 32× larger batches under the same latency budget, compared with conventional parallelism. Attention-FFN disaggregation (for example, MegaScale-Infer) places the two phases on different GPUs, in *space*. Helix alternates them on the same GPUs, in *time*.

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

TPLA resolves an incompatibility between Multi-Head Latent Attention (MLA) and tensor parallelism (TP). MLA, introduced in DeepSeek-V2, compresses the keys and values into a single low-rank latent vector, `cKV`, and caches only that vector, which shrinks the KV cache. Megatron-LM TP splits the attention heads across devices, but each device still needs the *entire* `cKV` for its heads, so the cache is replicated on every GPU. The memory per device is then no better than with Grouped Query Attention (GQA), or worse. Grouped Latent Attention (GLA) shards the latent instead, but each head then sees only part of it, which weakens its representational capacity.

TPLA **partitions the latent vector and the input dimension of each head across the devices, runs the attention independently on each shard, and sums the outputs of the shards with an all-reduce**. Unlike in GLA, every head attends to the *full* latent space, in pieces, so TPLA keeps more of the capacity of MLA while it cuts the KV cache per device. Each shard applies its own softmax before the all-reduce, so the result approximates MLA instead of reproducing it exactly. TPLA still works on checkpoints pre-trained with MLA, without retraining. An orthogonal transform (Hadamard or PCA) before the slicing reduces the interference between the shards and keeps the accuracy loss small.

The two phases of inference use different layouts, because they have different bottlenecks:

- the **prefill** processes the whole prompt at once, so it is compute-bound and does not re-read the KV cache: it keeps the standard (reparameterized) MLA;
- the **decode** generates one token at a time and reloads the entire KV cache from HBM at every step, so it is memory-bound: it uses TPLA, with the first slice of `cKV` on device 0, the next one on device 1, and so on.

At a context of 32K tokens, TPLA speeds up DeepSeek-V3 by **1.79×** and Kimi-K2 by **1.93×**, and keeps the quality on commonsense and LongBench benchmarks. It can be implemented with FlashAttention-3 for end-to-end gains.

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

PipeFill raises the GPU utilization of pipeline-parallel LLM training by filling the idle *bubbles* with other pending jobs, instead of removing the bubbles. The bubbles often take 15% to 30% of the GPU time of a job, and sometimes more than 60%. PipeFill fits the fill jobs to the measured bubble durations and to the free memory, adds explicit pipeline-bubble instructions, and runs the fill jobs inside the gaps. It raises the utilization by up to 63%, with less than 2% slowdown of the main job: on a run with 8K GPUs, this is the work of about 2.6K extra GPUs. PipeFill is a context-switching layer, not a pipeline schedule. It therefore complements the schedules that shrink or hide the bubbles of the training job itself, such as 1F1B, interleaving, Zero Bubble and DualPipe, through the overlap of forward and backward passes, or of computation and communication.

Good fill jobs tolerate interruptions. Batch (offline) inference fits best: computing the embeddings of a corpus, or running a smaller model over a backlog of prompts. Such work splits into chunks that fit short bubbles, is insensitive to latency, so pausing it is harmless, and needs little memory. The paper also uses training jobs as fill jobs, since training is long-running and tolerates interruptions, but inference is easier to slice into bubble-sized pieces.

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

This MatX article resolves a conflict between two methods that accelerate decoding. **Speculative decoding (SD)** uses a small draft model to propose several tokens, which the large target model verifies in parallel. This raises the operational intensity (the work per byte loaded from HBM), and gives about 2× gains in latency and throughput. **Blockwise sparse attention**, as in the Native Sparse Attention (NSA) of DeepSeek and the Mixture of Block Attention (MoBA) of Moonshot, divides the context into blocks, and each token attends to a dynamic subset of them. This reduces the KV data loaded from HBM and also raises the operational intensity, which matters because long contexts otherwise leave accelerators underused.

In the words of the article: "Speculative decoding (SD) and blockwise sparse attention both accelerate LLM decoding, but when combined naively, the KV cache may lose sparsity during the verification step of SD. We show that forcing all draft tokens to attend to the same subset of the context restores sparsity while preserving model quality."

NSA models trained this way match the language-modeling quality of the baseline NSA, with up to **3.5× higher operational intensity** in the verification step of speculative decoding. The models are small: 186M and 1.2B parameters, trained on sequences of 2K tokens.

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

SLA targets the attention bottleneck of Diffusion Transformers (DiTs), in particular for video generation, where long token sequences make the quadratic cost of attention dominate the latency. Its key observation is that the attention weights split into two parts: a small fraction of large weights with a high rank, and the remaining weights, with a very low rank. SLA therefore applies sparse acceleration to the first part and low-rank (linear) acceleration to the second, instead of a single approximation.

SLA classifies the attention blocks as *critical*, *marginal* or *negligible*. Critical blocks get exact attention, marginal blocks get linear attention, and negligible blocks are skipped. For $$N$$ tokens, exact attention costs $$O(N^2)$$ and linear attention $$O(N)$$. A single GPU kernel fuses the three cases, for both the forward and backward passes, so a pre-trained DiT needs only a few fine-tuning steps to adapt.

SLA reduces the attention computation by **95%** (about 20×) without degrading the end-to-end generation quality. Its kernel speeds up attention by **13.7×**, and the end-to-end video generation by **2.2×**, on **Wan2.1-1.3B**.

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

DeepSeek-V3.2-Exp is an update of the DeepSeek-V3 family for **serving efficiency**, especially with long contexts. It is DeepSeek-V3.1-Terminus with a single architectural change, **DeepSeek Sparse Attention (DSA)**, added through continued training (see the DeepSeek-V3.2 entry). A lightning indexer selects the top $$k = 2048$$ KV entries for each query, so the cost of the core attention drops from $$O(L^2)$$ to $$O(Lk)$$ for a context of $$L$$ tokens, in both the prefill and the decode. The KV cache itself does not shrink: the full latent cache is still stored, so that any token can be selected. What drops is the compute, and the number of KV entries read per step.

The release notes on Hugging Face also give serving details: recommended runtime settings, known issues and fixes, and evaluation notes. The benchmarks stay on par with V3.1-Terminus, and the largest gains come at long contexts. The final DeepSeek-V3.2 keeps the same architecture ([technical report](https://arxiv.org/abs/2512.02556), December 2025).

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

DeepCompile extends deep learning compilers to distributed training. Compilers such as the PyTorch compiler speed up single-GPU training with kernel fusion and operator scheduling, but they do not optimize distributed workloads. Distributed frameworks implement their optimizations at the PyTorch level, where compiler techniques such as dependency analysis and operator scheduling are hard to apply. The fully sharded approach of ZeRO-3 and FSDP relies on two runtime optimizations:

- *prefetching* starts the all-gather of the parameters of a layer before the layer needs them, to overlap communication with computation;
- *unsharding* keeps parameters in full form when memory allows, to reduce communication.

DeepCompile "takes a standard single-GPU model implementation and transforms it into an optimized multi-GPU training graph without requiring changes to the model code". The authors implement a fully sharded approach, as in ZeRO-3 and FSDP, on top of DeepCompile, with three optimization passes:

- **Proactive prefetching** starts each all-gather as early as possible, to maximize the overlap of communication and computation, and tracks how the free memory changes during the forward and backward passes.
- **Selective unsharding** keeps as many parameters unsharded as possible, to avoid all-gathers, and chooses them from a memory profile of each operator.
- **Adaptive offloading** moves optimizer states, such as the momentum and variance of Adam, to CPU memory when the GPU memory is insufficient. It offloads only the data above the memory limit, and overlaps the transfers with computation.

DeepCompile implements ZeRO-3, ZeRO-1 and offloading automatically. On Llama-3 70B and Mixtral 8×7B, it is up to 1.28× faster than ZeRO-3 and up to 1.54× faster than FSDP. With offloading, on limited GPU resources, its throughput is up to 7.01× higher. Future directions include automatic parallelization (sequence and tensor parallelism), better memory management, and dynamic adaptation at runtime.

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

Triton-distributed extends Triton with *distributed* primitives, so that one kernel can **compute and communicate**, instead of a GEMM kernel followed by a separate NCCL call. In its programming model, a distributed kernel is a set of asynchronous tasks. The tasks **signal** one another, and move data directly between GPUs through **symmetric memory**, with [OpenSHMEM](https://en.wikipedia.org/wiki/SHMEM)-style primitives that are lowered to NVSHMEM or ROCSHMEM.

Communication thus becomes an explicit part of the kernel schedule. The compiler and the runtime can overlap loads, arithmetic, and remote puts and gets, at a much finer grain than the overlap of independent kernels on separate streams. FLUX hand-optimizes specific communication patterns of LLMs, while Triton-distributed is a **general programming model**, with primitives and compiler support that apply to any operator and topology.

On 8 H800 GPUs, its kernels are on average **1.42×** (AllGather+GEMM) and **1.28×** (GEMM+ReduceScatter) faster than PyTorch with NCCL. On two nodes (16 GPUs), these kernels reach about 96% of the performance of FLUX, while being written in Python. The MoE kernels show larger average speedups (**44.97×** for AllGather+MoE, **15.55×** for MoE+ReduceScatter), but against a weak baseline, with grouped GEMMs in Python loops. Triton-distributed scales to 64 GPUs, re-implements the low-latency all-to-all kernels of DeepEP with comparable performance, and also runs on AMD MI308X GPUs.

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

TileLink aims at the performance of FLUX without a hand-crafted fused kernel for each operation. It separates the design of the communication from that of the computation, and links the two through **tile-centric primitives**: producer-consumer signals (notify and wait) and tile data movement (push and pull). Each primitive maps to a tensor slice, a peer rank and a barrier channel. The system then schedules computation and communication *per tile*, instead of per tensor, and reaches an overlap close to that of highly tuned systems.

The three systems differ in their level of abstraction:

- **FLUX** splits specific tensor-parallel operations of LLMs into fine slices and fuses them into larger kernels. It overlaps almost all of the communication, but each kernel is tailored to one operation.
- **Triton-distributed** offers low-level primitives inside kernels (symmetric memory, signals, tasks) and a general programming model.
- **TileLink** is higher-level than Triton-distributed, and more general and easier to program than FLUX, but it still models the overlap at the level of the *tile schedule*.

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

MegaScale-MoE is a production system for **communication-efficient MoE training**. At scale, MoE training is dominated by **communication**, such as the all-to-all between the routers and the experts, and keeping the GPUs busy needs a strategy for the whole system. MegaScale-MoE has three parts:

1. it customizes the parallelism of each MoE layer to reduce the communication volume: sequence parallelism for the attention and expert parallelism for the FFNs, within a node, instead of the tensor parallelism of Megatron-LM;
2. it overlaps communication with computation, both between and within operators;
3. it compresses the communication to a lower precision, with adjusted communication patterns.

FLUX, TileLink and Triton-distributed fuse kernels. MegaScale-MoE is instead a system and a parallelization strategy for end-to-end training at large scale: its gains come from the parallel layout, the scheduling and the communication precision of the whole training job.

Training a **352B MoE** model on **1,440 NVIDIA Hopper GPUs**, it reaches **1.41M tokens/s**, 1.88× the efficiency of Megatron-LM (1.65× to 1.88× across the strong-scaling settings). It runs the production MoE training of ByteDance, with models of trillions of parameters, and single jobs on more than 10,000 GPUs.

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

MegaScale-Infer addresses the MoE FFNs of the decode phase, which are memory-bound, with **Disaggregated Expert Parallelism (DEP)**. The attention and the expert modules run on separate GPUs, so that each scales independently, with its own parallelism. The attention modules are replicated with data parallelism, and the FFN modules are scaled with expert parallelism. Each expert receives the tokens of many attention replicas, so its batch grows, and the FFNs become compute-bound again. The disaggregation also allows a **heterogeneous deployment**, with cheaper or different GPUs for the attention and for the experts, to reduce the cost.

Two system components keep the communication from stalling the disaggregated modules:

- **ping-pong pipeline parallelism** splits a batch of requests into micro-batches and shuttles them between the attention and the FFN GPUs, so that both pools stay busy and the communication hides behind computation;
- a **high-performance M2N communication library** for the dispatch and combine of the tokens removes unnecessary GPU-to-CPU copies, the overhead of group initialization, and GPU synchronization.

This is attention-FFN disaggregation for MoE serving: the two modules are split *in space*, across GPU pools, whereas Helix splits them *in time*, on the same GPUs. Prefill-decode disaggregation (as in TPLA) is a different, orthogonal split.

On homogeneous clusters, MegaScale-Infer reaches up to **1.90×** the decoding throughput per GPU of state-of-the-art LLM serving systems. On heterogeneous clusters, with attention and experts on different GPU types, its decoding throughput per unit cost is up to **1.86×** that of TensorRT-LLM, and up to 3.24× that of vLLM, on H20 GPUs.

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

This paper targets MoE inference with expert parallelism (EP). Skewed routing overloads some GPUs while others are idle. The usual fixes either drop tokens (capacity factors) or cost memory (replicas of popular experts). **MoEShard** shards *every* expert across all GPUs instead, as in tensor parallelism: the first matrix of each expert is split by columns and the second by rows. Each GPU computes a slice of every selected expert, and the partial outputs are summed. The load is balanced whatever the routing, and no token is dropped.

MoEShard fuses the decomposed expert computations to reduce the number of kernel launches. On encoder-based MoE models (Switch Transformer), on a single multi-GPU node, its time to first token is up to **6.4×** lower than that of DeepSpeed. This small study, on one node, shows that sharding the experts is an alternative to plain EP when the routing is imbalanced. It relies on the fast links within the node for the extra communication.

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

TransMLA **converts** trained models with Grouped-Query Attention (GQA), such as LLaMA, Qwen and Mixtral, into models with MLA, without pre-training from scratch. Most model providers have invested in GQA models and have little reason to pre-train MLA models. The theoretical core of the paper is a proof that **GQA can always be rewritten exactly as MLA with the same KV-cache size, but not the reverse**: for a given cache size, MLA is strictly more expressive than GQA. Any GQA checkpoint can therefore be rewritten in MLA form without loss, and then *upgraded* to use the extra expressiveness of MLA.

TransMLA has two steps:

1. It **converts** the GQA attention into the MLA structure exactly, by rewriting the repeated KV heads of GQA as low-rank latent projections. The model is then compatible with the DeepSeek codebase and its optimized serving systems (vLLM, SGLang), and with features such as FP8 quantization and multi-token prediction.
2. The conversion alone does not speed up inference, so TransMLA **compresses** the latent KV below the original cache size and **fine-tunes** the model briefly to recover the quality. About 6B tokens restore a performance on par with the original model across benchmarks.

Compressing 93% of the KV cache of LLaMA-2-7B gives a 10.6× inference speedup at a context of 8K tokens, with meaningful output quality. TPLA and Helix change how models are sharded. TransMLA instead converts the many existing GQA models to MLA.

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

This work targets single-sequence decoding of a 1B-parameter LLM (Llama-3.2-1B), a regime of extreme low latency, where the performance depends on how fast the GPU streams the weights from global memory. According to the authors, inference engines suffer from *pipeline bubbles*: a forward pass is split into dozens to about a hundred small kernels, each with launch and teardown costs and synchronization stalls, so the memory streaming stops between kernels.

The solution fuses the *entire* forward pass into a single **megakernel** that acts as an interpreter on the GPU. Each SM executes a schedule of *instructions*: a fused RMSNorm, QKV projection and RoPE, the attention, the projections, parts of the MLP, and so on. Fusing about a hundred operations raises three problems:

1. how to *program* such a megakernel: with an instruction abstraction and an interpreter;
2. how to avoid contention for resources: with **paged shared memory**, which pipelines the weight loads;
3. how to synchronize without kernel boundaries: with a lightweight scheme of **counters**.

On an **H100**, the megakernel uses about **78%** of the memory bandwidth, while vLLM and SGLang use at most about **50%** in this setting. End to end (a prompt of 32 tokens, 128 generated tokens, no speculation), it is about **2.5× faster than vLLM** and more than **1.5× faster than SGLang** on the H100. On a **B200**, it is more than **3.5×** faster than vLLM, and still more than **1.5×** faster than SGLang. A forward pass of the 16-bit model takes **less than 1 ms** on the H100, and **less than 680 µs** on the B200.

FLUX, TileLink and Triton-distributed fuse computation with distributed communication. This work instead removes *all kernel boundaries*, for low-latency inference on a single GPU.

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

Mirage generates deeply fused tensor programs *automatically*. It treats kernel fusion as a [superoptimization](https://en.wikipedia.org/wiki/Superoptimization) problem over **µGraphs**, a uniform representation of a tensor program at the kernel, thread-block and thread levels of the GPU. With µGraphs, Mirage searches algebraic transformations, schedule transformations and new custom kernels jointly, and it rediscovers kernels like FlashAttention and FlashDecoding by itself. An abstraction-based pruning keeps the large search space tractable, with an optimality guarantee. A probabilistic equivalence verifier checks that the optimized µGraph computes the same function as the input program.

Mirage differs from three other lines of work:

- **hand-written megakernels** (for example "No Bubbles"): Mirage *discovers* aggressive fusions automatically and generates their implementations;
- **FlashAttention-style** kernels: Mirage targets arbitrary tensor programs, not only attention, and finds FlashAttention-like structures when they pay off;
- **compiler frameworks** such as TVM and XLA: Mirage superoptimizes small fused graphs, which can beat hand tuning.

Mirage outperforms existing approaches by **up to 3.3×** (the figure varies across versions of the paper), even on widely used and heavily optimized DNN components.

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

SageAttention3 runs both matrix multiplications of attention on the new **FP4 Tensor Cores** of Blackwell GPUs. It uses **microscaling** (block-scaled) FP4 quantization, with techniques specific to attention that keep the accuracy. It quantizes only the attention operator, not the whole model, and plugs into existing inference systems.

On an **RTX 5090**, it reaches **1038 TOPS**, 5× the speed of the fastest FlashAttention on that GPU: FlashAttention2, since FlashAttention3 runs only on Hopper. It speeds up various models end to end without changes, such as HunyuanVideo for video generation. The paper also explores low-bit attention for *training*, with SageBwd (8-bit forward and backward passes): it is lossless in fine-tuning, but converges more slowly in pre-training.

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

Titans are a family of models that combine attention with a *neural* long-term memory: an MLP whose weights are updated at test time, by gradient steps on an associative-memory loss. A "surprise" signal with momentum drives the updates, and a forgetting gate, similar to weight decay, removes old information. The memory lets attention focus on the current context while it uses information from the distant past. It trains fast and in parallel, and keeps inference fast. In the view of the paper, attention, with its limited context but precise dependencies, acts as a short-term memory, and the neural memory as a long-term, more persistent memory.

The paper combines the two in three ways: the memory as context (MAC), as a gate (MAG), or as a layer (MAL). Titans outperform Transformers and modern linear recurrent models (such as Mamba2 and DeltaNet) on language modeling, commonsense reasoning, genomics and time series. They scale to contexts of more than 2M tokens, with a higher needle-in-a-haystack accuracy than the baselines.

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

LoongServe introduces **Elastic Sequence Parallelism (ESP)**. Instead of a fixed degree of parallelism per instance, it adapts the degree of sequence parallelism at each iteration and in each phase, scaling up or down. The prefill runs ring attention. While the KV tensors rotate across the GPUs, each GPU keeps a subset of them locally, so that they are already in place for the decode. This proactive scale-down avoids migrating the KV cache. In the decode, the KV cache stays sharded across instances, so the decode runs the usual distributed decode algorithm, which is often the bottleneck of context parallelism.

The prefill overlaps computation and communication with CUDA streams. In the decode, the paper overlaps the communication of the partial attention results with computation, but the overlap is opportunistic: a node computes the matrix multiplications of the requests that it receives from other GPUs while it sends its own results, only when such requests are there. ESP also reduces the fragmentation of the KV cache across instances. LoongServe reaches up to **3.85×** the maximum throughput of chunked prefill, and up to **5.81×** that of prefill-decode disaggregation.

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

Mamba is the state-space model (SSM) that made a sequence model without attention competitive with Transformers on language. SSMs keep a fixed-size recurrent state and scale **linearly** with the sequence length. Earlier SSMs, such as S4, performed worse, because their dynamics were fixed, independent of the input. Mamba adds a **selection mechanism**: the SSM parameters become functions of the input, so the model propagates or forgets information depending on the current token, which allows content-based reasoning. This input dependence breaks the convolution form, so Mamba uses a **hardware-aware parallel scan**, kept in SRAM as in FlashAttention, to stay fast. Mamba has 5× the inference throughput of Transformers and scales linearly, and Mamba-3B matches Transformers twice its size. Its constant-size state removes the growing KV cache, which hybrid models such as Jamba and Nemotron 3 exploit.

An SSM maps an input $$x_t$$ to an output $$y_t$$ through a hidden state $$h_t \in \mathbb{R}^N$$. The continuous parameters $$(\Delta, A, B, C)$$ are first **discretized**:

$$\bar A = \exp(\Delta A), \qquad \bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B$$

The model then runs as a **linear recurrence** (inference) or a **global convolution** (training):

$$h_t = \bar A\,h_{t-1} + \bar B\,x_t, \qquad y_t = C\,h_t$$

In S4, the parameters are **fixed across time**, as in a [linear time-invariant](https://en.wikipedia.org/wiki/Linear_time-invariant_system) (LTI) system. In Mamba, $$B$$, $$C$$ and $$\Delta$$ are **functions of $$x_t$$** (the selective SSM, S6), so $$\bar A$$ and $$\bar B$$ change at every token. This is the selectivity, and the reason for the scan instead of the convolution.

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

Jamba (Lieber, Lenz et al., AI21) is the first **production-scale hybrid** of Transformer attention, Mamba SSM and MoE layers. It balances one trade-off: the KV cache limits Transformers at long contexts, and replacing attention layers with Mamba layers shrinks it, since a Mamba layer keeps only a constant-size state. A few attention layers remain, for precise recall. The Mamba layers do the cheap long-range work, and MoE adds capacity without adding active compute.

The released configuration has 4 blocks of 8 layers, a 1:7 ratio of attention to Mamba layers, an MoE every other layer, and 16 experts with top-2 routing: 52B total but only 12B active parameters, which fit on one 80GB GPU. The 1:7 ratio is empirical: it keeps most of the quality of attention and most of the memory savings of Mamba. At a context of 256K tokens, Jamba needs about 4GB of KV cache, against 32GB for Mixtral and 128GB for Llama-2-7B. It gives strong results up to 256K tokens, with up to 3× the throughput of Mixtral at long contexts. Jamba 1.5 later scaled the same design to 94B active and 398B total parameters. Nemotron 3 follows the same template, with Mamba-2 layers, LatentMoE, MTP and NVFP4.

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

This paper (Gloeckle et al., Meta) establishes multi-token prediction (MTP) as a general training objective. LLMs such as GPT and Llama are trained with a next-token prediction loss. According to the authors, training a model to predict several future tokens at once gives a higher sample efficiency.

The model predicts the next $$n$$ tokens with $$n$$ independent output heads on top of a shared trunk, so multi-token prediction is an **auxiliary task** added to the usual objective. At position $$t$$, the main head predicts the next token $$x_{t+1}$$, and the auxiliary heads predict $$x_{t+2}$$, $$x_{t+3}$$, and so on. The heads also speed up inference:

- *drafting*: during generation, the auxiliary heads produce a draft of the next few tokens, at the same time as the main head produces the next token;
- *parallel verification*: in the next step, the model checks the drafted tokens with its main head, in a single forward pass.

The benefit grows with the model size. The 13B model solves 12% more HumanEval problems and 17% more MBPP problems than a comparable next-token model. Models trained to predict 4 tokens are up to 3× faster at inference with this self-speculative decoding, even at large batch sizes.

In standard speculative decoding, a separate small draft model generates a few tokens (usually 3 to 5) one by one, and the large target model checks all of them in one forward pass with a causal mask: it computes the probabilities of all the drafted tokens at once, in about the time it takes to generate one token. With MTP, the heads of the model itself draft the tokens, so no second model is needed.

DeepSeek-V3 and Nemotron 3 use MTP. DeepSeek-V3 replaces the independent parallel heads with *sequential, cascaded* modules that keep the full causal chain. Nemotron 3 uses a lightweight MTP module that reaches about 97% acceptance on its first two predicted tokens.

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

DeepSeek LLM, the first release of DeepSeek (also called DeepSeek-V1), has dense 7B and 67B models, trained on 2T tokens. It predates MoE and MLA, which came with DeepSeekMoE and DeepSeek-V2. The architecture follows LLaMA: pre-norm with RMSNorm, SwiGLU FFNs and rotary position embeddings. The 7B model has 30 layers. The 67B model has 95 layers and uses Grouped-Query Attention (GQA) to reduce the cost of inference.

The main contribution is a study of scaling laws. It fits the optimal batch size and learning rate as functions of the compute budget, and the optimal split of the compute between model size and data. It measures the model size in non-embedding FLOPs per token instead of parameters. These fits set the configurations of both models. The "longtermism" of the title is the long-term perspective of the project on open-source models. The chat models are aligned with supervised fine-tuning (SFT), then Direct Preference Optimization (DPO). DeepSeek LLM 67B outperforms LLaMA-2 70B, in particular on code, math and reasoning, and the 67B chat model outperforms GPT-3.5 in open-ended evaluations.

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

DeepSeek-V2 introduces Multi-head Latent Attention (MLA) and DeepSeekMoE, the two architectural foundations of V3 and R1. It is an MoE model with 236B total parameters, of which 21B are activated per token, and a context of 128K tokens. The two innovations target two bottlenecks: the KV cache of standard multi-head attention (MHA) in decoding, and the cost of dense FFNs in training.

**DeepSeekMoE** provides the FFN, with fine-grained experts and shared experts, so that sparse activation gives a large capacity at a low compute cost. It has two main ideas:

1. **Shared experts**: a small number $$K_s$$ of experts that every token uses, without routing. They form a fixed, always-active set that absorbs the common knowledge.
2. **Fine-grained expert segmentation**: a regular MoE has $$N$$ experts, each a full-size FFN, and activates the top $$K$$. DeepSeekMoE splits each expert into $$m$$ smaller ones, for $$mN$$ experts in total, and activates $$mK$$ of them. The total parameters and the FLOPs stay the same, but each token can use many more combinations of experts.

**MLA** compresses the keys and values jointly with a low-rank projection. Instead of caching the full keys and values of each head, or sharing them across heads as in GQA and MQA, it projects them down to a single small latent vector `cKV`. The cache stores this vector, and the computation projects it back up to distinct keys and values per head. MLA keeps the expressiveness of full MHA and shrinks the cache. RoPE cannot be absorbed through the up-projection, so V2 adds **decoupled RoPE**: a separate set of small query dimensions and one shared key, also cached (a small extra $$d_h^R$$ per token), carry the positional rotation, apart from the compressed *content* path.

Here, $$h_t$$ is the input hidden vector of the token at position $$t$$, $$n_h$$ the number of heads, $$d_h$$ the dimension per head, $$d_c \ll n_h d_h$$ the dimension of the cached latent, and $$d_h^R$$ the small dimension of the RoPE part. $$W^{D\ast}$$ are down-projections (compression) and $$W^{U\ast}$$ up-projections (reconstruction): $$W^{DKV}$$ is the down-projection of the keys and values, and $$W^{UK}$$ and $$W^{UV}$$ are the up-projections of the keys and of the values. The superscripts $$C$$ and $$R$$ mark the content and RoPE parts, and $$j \le t$$ indexes the past tokens. MLA compresses $$h_t$$ into the cached latent, and reconstructs distinct keys and values per head from it:

$$c_t^{KV} = W^{DKV}\, h_t \quad (\in \mathbb{R}^{d_c},\ \text{cached, together with the RoPE key } k_t^{R}); \qquad k_t^{C} = W^{UK} c_t^{KV},\quad v_t^{C} = W^{UV} c_t^{KV}$$

The learned matrices $$W^{DKV}$$, $$W^{UK}$$ and $$W^{UV}$$ form low-rank transformations. The queries are also low-rank, which saves activation memory; they are not cached:

$$c_t^{Q} = W^{DQ}\, h_t, \qquad q_t^{C} = W^{UQ}\, c_t^{Q}$$

In MHA, the key projection of each head has a rank of at most $$d_h$$, but the key projection of all heads together can have full rank. In MLA, all heads share the latent $$c_t^{KV}$$, so the key projection of all heads, $$W^{UK} W^{DKV}$$, has a rank of at most $$d_c$$, by design. The KV cache stores this latent.

With decoupled RoPE, the position follows a separate path, with rotary queries per head and one **shared** rotary key:

$$q_t^{R} = \mathrm{RoPE}(W^{QR}\, c_t^{Q}), \qquad k_t^{R} = \mathrm{RoPE}(W^{KR}\, h_t)$$

The query and the key of each head $$i$$ concatenate the content and rotary parts:

$$q_{t,i} = [\,q_{t,i}^{C}\,;\,q_{t,i}^{R}\,], \qquad k_{t,i} = [\,k_{t,i}^{C}\,;\,k_t^{R}\,]$$

The attention is then standard:

$$o_{t,i} = \sum_{j \le t} \mathrm{softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v_{j,i}^{C}, \qquad u_t = W^{O}\,[\,o_{t,1};\dots;o_{t,n_h}\,]$$

The KV cache per token and per layer is, with $$g$$ groups of KV heads in GQA:

| MHA | GQA | MQA | **MLA** |
|---|---|---|---|
| $$2\,n_h\,d_h$$ | $$2\,g\,d_h$$ | $$2\,d_h$$ | $$d_c + d_h^R$$ |

The *absorb* trick folds $$W^{UK}$$ into $$W^{UQ}$$, and $$W^{UV}$$ into $$W^{O}$$, at decode time, so the attention runs directly on `cKV`, and the full keys and values per head are never materialized. DeepSeek-V2 uses $$d_c = 512$$ and $$d_h^R = 64$$ ($$d_h = 128$$), for a cache of $$576$$ elements per token and per layer: as much as GQA with about 2.25 groups, but with the expressiveness of full MHA.

Compared with the dense DeepSeek 67B, DeepSeek-V2 costs 42.5% less to train, has a **93.3% smaller KV cache**, and reaches up to **5.76× the maximum generation throughput**, with a higher quality.

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

SageAttention2 extends SageAttention. It quantizes $$Q$$ and $$K$$ to **INT4**, per thread, for the expensive product $$QK^\top$$, and uses **FP8** for the product $$\widetilde{P}V$$ of the unnormalized attention probabilities and the values. Attention has quantization problems that are weaker in linear layers, in particular outliers, so the paper adds three techniques specific to attention:

1. per-thread INT4 quantization of $$Q$$ and $$K$$;
2. a smoothing of $$Q$$, which improves the accuracy of the INT4 product $$QK^\top$$;
3. a two-level accumulation, which improves the accuracy of the FP8 product $$\widetilde{P}V$$.

On an **RTX 4090**, SageAttention2 reaches about **3×** the OPS of FlashAttention2 and **4.5×** that of xFormers. On Hopper, it matches the speed of FlashAttention3 in FP8, with a higher accuracy.

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

SageAttention quantizes the attention kernel, not the whole model, and plugs into existing LLM inference. Even when the linear layers are well optimized, attention can stay a major bottleneck, especially at long sequence lengths. Its mix of matrix multiplications, softmax and reductions also amplifies numerical outliers, which makes it hard to quantize.

SageAttention runs attention with **8-bit** arithmetic and keeps the accuracy. The paper finds two problems of naive INT8 attention. First, $$K$$ has strong outliers per channel. SageAttention *smooths* $$K$$: it subtracts the mean of $$K$$ over the tokens, which leaves the softmax unchanged. Second, quantizing $$\widetilde{P}$$ and $$V$$ to INT8 is not reliably accurate. SageAttention therefore computes $$QK^\top$$ in INT8, and $$\widetilde{P}V$$ in FP16, with an FP16 accumulator. This first version is less aggressive than SageAttention2 and SageAttention3, and shows that attention can be quantized with little integration work.

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

Centauri overlaps communication with computation at the level of *operator scheduling*. It **decomposes operators** and schedules the pieces across several CUDA streams, so that communication overlaps with useful computation more consistently than with the overlap of whole operators. *Communication partitioning* breaks a large collective into smaller chunks, which are scheduled earlier and more often. The partition space has three dimensions:

- primitive substitution: rewriting a collective as other primitives;
- topology-aware group partitioning: for example, hierarchical collectives within and across nodes;
- workload partitioning: chunking the data.

The pieces are scheduled hierarchically, at the operation, layer and model levels of hybrid-parallel training.

Centauri differs from FLUX, TileLink and Triton-distributed:

- Centauri schedules decomposed operators on several streams, and improves the overlap between kernels that stay separate. It needs no collectives inside kernels.
- FLUX, TileLink and Triton-distributed fuse computation and communication *inside* kernels. They overlap at a finer grain, but need more specialized code generation or primitives.
- Centauri fits workloads limited by *orchestration* (stream scheduling, dependencies, coarse kernels). FLUX-like methods target the overheads that remain even with aggressive overlap across streams.

Centauri is up to **1.49×** faster than prevalent methods, across various hybrid-parallel training configurations; the paper received an ASPLOS 2024 Best Paper award. It is also a baseline for fused kernels: it shows the gain that better scheduling alone provides.

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

FLUX targets the tensor-parallel patterns of Megatron-LM. A tensor-parallel layer usually runs one large GEMM and one large NCCL collective. FLUX **decomposes** both into fine-grained pieces (tiles or chunks), and fuses them into kernels that overlap communication with computation aggressively. This overlap is hard to reach when separate kernels overlap on several streams.

Compared with related work:

- Centauri overlaps independent kernels, while FLUX overlaps *within* the operator.
- Triton-distributed is a general programming model, while FLUX delivers peak performance for a *specific family* of LLM operations: the tensor-parallel attention and MLP of Megatron-LM.
- TileLink tries to keep the performance of FLUX while raising the programming level.

FLUX overlaps up to about **96%** of the communication. In training, it is up to **1.24×** faster than Megatron-LM on a cluster of **128 GPUs**, across GPU generations and interconnects. In inference, it is up to **1.66×** faster than vLLM in the **prefill**, and **1.30×** faster in the **decoding**, on **8 GPUs**.

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

Punniyamurthy et al. identify common motifs of computation followed by a collective, which appear again and again in distributed training and inference. They implement each motif as a **single fused kernel** with **collectives inside the kernel**. Each workgroup sends its results to remote GPUs as soon as its computation finishes (GPU-initiated communication), while the other workgroups of the same kernel keep computing. Transfers within a node are zero-copy: they write directly into the memory of the peer GPU. The fused kernels, written in Triton and HIP, remove the intermediate writes, the kernel boundaries and the separate communication steps.

Compared with related work:

- FLUX focuses on the tensor-parallel patterns of LLMs, while this paper covers a **broader** set: embedding and all-to-all, GEMV and all-reduce, GEMM and all-to-all.
- Triton-distributed and TileLink offer programming models to build such fused kernels, while this paper builds **instances by hand** and measures the gains of the fusion.
- Unlike in-network computing, these fused kernels communicate through primitives on the GPUs, not in the network.

On average, the fused operators lower the execution time by 12% to 31%. Within a node of four GPUs, they lower it by 20% on average (up to 32%) for embedding and all-to-all, by 13% (up to 22%) for GEMV and all-reduce, and by 12% (up to 20%) for GEMM and all-to-all. Across nodes, embedding and all-to-all takes 31% less time on average (up to 58%). The gains are significant even for simple two-stage pipelines, when the communication is tightly coupled to the computation.

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

DeepSeek-V3 is an MoE model with 671B parameters (37B active per token), trained on 14.8T tokens in 2.788M H800 GPU hours (about 5.6M USD). The report covers the architecture (MLA and DeepSeekMoE from V2, auxiliary-loss-free load balancing, multi-token prediction), FP8 mixed-precision training, and the engineering trade-offs of training at scale, which papers on single algorithms often omit.

The training runs on 2,048 H800 GPUs. Its communication relies on custom all-to-all kernels across nodes, later open-sourced as DeepEP. These kernels use only 20 of the 132 SMs, and use InfiniBand and NVLink together by sending each token to at most 4 nodes. They overlap with computation at the level of the *schedule* (DualPipe), not inside the GEMM kernels or in the network. DeepSeek-V3 is therefore a strong end-to-end baseline for fused kernels and in-network computing.

Three training techniques stand out:

- **FP8 training.** DeepSeek-V3 is the first frontier-scale model trained in FP8. Earlier recipes use E4M3 in the forward pass and E5M2 for the gradients, while DeepSeek-V3 uses E4M3 for all tensors. This works because of fine-grained scaling: activations are scaled per tile of 1×128 values and weights per block of 128×128, so an outlier only affects its own group. The scales are computed online from the current values, and the GEMMs accumulate in higher precision. The optimizer moments are kept in BF16, and a few sensitive operators stay in higher precision.
- **Sequential multi-token prediction.** Unlike the parallel heads of [Gloeckle et al.](https://arxiv.org/abs/2404.19737), each MTP module is a single Transformer layer that takes the hidden states of the previous depth and the embedding of the next token, so the causal chain is kept. The modules share the embedding and the output head with the main model, and each adds its own cross-entropy loss. DeepSeek-V3 uses one module, which can also serve as a draft model for speculative decoding: its second token is accepted 85% to 90% of the time, for about 1.8× more tokens per second.
- **Auxiliary-loss-free load balancing.** A bias per expert is added to the routing scores, but only to select the top-k experts, not to weight their outputs. After each step, the bias of overloaded experts decreases and that of underloaded experts increases. This balances the load without a loss term that competes with the language modeling objective. Only a very small sequence-wise balance loss remains, to avoid an extreme imbalance within a sequence.

Post-training distills the reasoning ability of DeepSeek-R1, trained with large-scale reinforcement learning on top of DeepSeek-V3-Base, into DeepSeek-V3. R1 generates the reasoning data, and V3 is fine-tuned on it, which improves its math and code without the length and style of a pure reasoning model.

**DualPipe** is a bidirectional pipeline-parallel schedule, introduced in the report. It reduces the pipeline bubbles and hides the heavy communication of expert parallelism across nodes, which gives DeepSeek-V3 a computation-to-communication ratio of about 1:1. It overlaps computation and communication within a pair of forward and backward chunks. Each chunk is split into four parts: attention, all-to-all dispatch, MLP and all-to-all combine. As in Zero Bubble, the backward attention and MLP are further split into input-gradient and weight-gradient parts. The parts are rearranged so that one set of micro-batches runs forward while another runs backward, and the communication hides behind computation. Micro-batches enter from both ends of the pipeline, which keeps the GPUs busier than sequential schedules such as 1F1B or Zero Bubble.

DualPipe has fewer bubbles and overlaps all the communication of the forward and backward passes. It costs two copies of the model parameters, one per direction, and slightly more activation memory. The copies are affordable because DeepSeek-V3 uses a large expert-parallel size, so each rank holds relatively few parameters. Memory savings (recomputation, the EMA of the weights kept in CPU memory, and an output head shared with the MTP module) also let DeepSeek-V3 train without tensor parallelism. PipeFill accepts the bubbles and fills them with other jobs, while DualPipe removes them from the schedule of the main job, so the two approaches are complementary.

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

Zero Bubble is the first schedule that reaches zero pipeline bubbles with synchronous training semantics. It splits the backward pass into two independent computations: the input gradient (B) and the weight gradient (W). Running them back to back is unnecessary. Decoupling B and W removes sequential dependencies, so the W passes, which can start later, fill the bubbles that a 1F1B schedule leaves at the end. The paper gives two handcrafted schedules and an automatic scheduler. The scheduler formulates the schedule as an [integer linear program](https://en.wikipedia.org/wiki/Integer_programming), solved by an off-the-shelf solver, with a heuristic for large numbers of micro-batches.

The two handcrafted schedules trade memory for bubbles. ZB-H1 keeps the peak activation memory of 1F1B and cuts its bubble to about a third. ZB-H2 removes the bubbles entirely, at the cost of a higher peak memory (about twice the activations) and an extra validation and rollback step of the optimizer: it bypasses the synchronization of the optimizer, and corrects the step afterwards. The zero-bubble schedules have up to 23% more throughput than 1F1B under a similar memory limit, and up to 31% more when the memory limit is relaxed. The method is orthogonal to data, tensor and ZeRO parallelism, and replaces the pipeline-parallel component directly. Like DualPipe, it removes the bubbles from the schedule of the main job, while PipeFill fills them with other jobs.

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

FP6-LLM makes six-bit floating-point (FP6) quantization practical for LLM inference on GPUs. FP6 shrinks the model and the time to load its weights, and keeps the quality more reliably than 4-bit formats: 4-bit methods do well on zero-shot benchmarks but degrade on harder tasks, such as code generation and summarization, while 6-bit stays robust. No system supported FP6 on Tensor Cores, for two reasons: the irregular 6-bit width causes misaligned memory accesses to the weights, and converting the weights back to a format that the Tensor Cores multiply is expensive at runtime. The paper introduces TC-FPx, the first full-stack GPU kernel design with unified Tensor Core support for floating-point weights of any bit width. It solves the memory layout and the on-the-fly dequantization together.

TC-FPx works as follows:

- The weights are FP6, with 1 sign bit, 3 exponent bits and 2 mantissa bits (**E3M2**), in a **W6A16** linear layer: 6-bit weights and FP16 activations. Only the weights are quantized: the activations and the results of the matrix multiplications stay in FP16.
- The weights stay in 6 bits in memory, which reduces their footprint and their loading time from DRAM. They are quantized from FP16 once, offline, and packed into an aligned layout: each 6-bit value is split into a 2-bit and a 4-bit fragment, so that the loads are aligned.
- Just before the computation, the SIMT cores (CUDA cores) load a slice of the weights, join the fragments with bit shifts and masks, and expand them to FP16 in registers. The Tensor Cores then multiply in FP16.
- The kernel is software-pipelined: the dequantization of slice $$N+1$$ on the CUDA cores overlaps with the matrix multiplication of slice $$N$$ on the Tensor Cores. The gain comes from the memory bandwidth: the kernel loads 6-bit weights instead of 16-bit ones, and hides the cost of the dequantization behind the Tensor Core math.

FP6-LLM integrates TC-FPx into an existing inference system. It serves LLaMA-70B on a single GPU, with 1.69× to 2.65× the normalized inference throughput of the FP16 baseline. The 6-bit linear layer runs up to 2.6× faster than cuBLAS in FP16, and up to 1.9× faster than the 8-bit (W8A16) kernels of TensorRT-LLM. The source code is at [github.com/usyd-fsalab/fp6_llm](https://github.com/usyd-fsalab/fp6_llm).

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

Llama 3 is a herd of language models with native support for multiple languages, coding, reasoning and tool use, with 8B, 70B and 405B parameters and a context of 128K tokens. Llama 3 405B has 126 layers, a model dimension of 16,384 and 128 attention heads. It is trained on up to 16K H100 GPUs, with 4D parallelism (tensor, pipeline, context and data). The authors use scaling laws (Hoffmann et al., 2022) to choose the optimal size of the flagship model for their pre-training compute budget (section 3.2.1). They first relate the negative log-likelihood of the compute-optimal model on downstream tasks to the training FLOPs, and then fit a sigmoidal relation between that log-likelihood and the task accuracy (figure 4):

{: .picture}
<img loading="lazy" width="74.9%" height="74.9%" src="{{ site.assets }}/publications/llama3_scaling_laws.svg"/>

{: .caption}
<b>Left:</b> <b>Scaling law IsoFLOPs curves</b> between $$6 \times 10^{18}$$ and $$10^{22}$$ FLOPs. The loss is the negative log-likelihood on a held-out validation set. Measurements at each compute scale are approximated by a second-degree polynomial. <b>Right:</b> <b>Number of training tokens in identified compute-optimal models as a function of pre-training compute budget.</b> The line is the fitted scaling-law prediction. The compute-optimal models correspond to the parabola minima of the left plot. Adapted from: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

The architecture follows Llama 2, with four changes:

1. grouped-query attention with 8 key-value heads, which speeds up inference and reduces the size of the KV cache in decoding;
2. an attention mask that prevents self-attention between different documents within the same sequence, which matters in continued pre-training on very long sequences;
3. a vocabulary of 128K tokens: 100K from `tiktoken` and 28K more for better non-English support;
4. a RoPE base frequency of 500,000, to better support long contexts.

Training has two stages: pre-training with next-token prediction, and post-training, where the model is "tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning)." The improvements are at three levels:

1. **data**: better quality, quantity, pre-processing and curation, with "15T multilingual tokens, compared to 1.8T tokens for Llama 2";
2. **scale**: almost $$50 \times$$ the pre-training compute of the largest Llama 2 model, with $$3.8 \times 10^{25}$$ FLOPs;
3. **complexity**: a standard Transformer with minor changes instead of a mixture of experts, and "a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO), as opposed to more complex reinforcement learning algorithms" (section 4).

The authors also add image, video and speech capabilities, with three more stages:

- **multi-modal encoder pre-training**: separate image and speech encoders (sections 7 and 8). The image encoder is trained on large amounts of image-text pairs. The speech encoder is trained with a self-supervised method that "masks out parts of the speech inputs and tries to reconstruct the masked out parts via a discrete-token representation";
- **vision adapter training**: an adapter, trained on text-image pairs, aligns the image representations with the language representations. A video adapter, trained on top of the image adapter on paired video-text data, aggregates information across frames (section 7);
- **speech adapter training**: a third adapter converts the speech encodings into token representations.

The image encoder is a standard vision Transformer trained to align images and text, the ViT-H/14 variant. Cross-attention layers, with grouped-query attention, connect the visual tokens of the image encoder to the token representations of the language model, after every 4th self-attention layer.

{: .picture}
<img loading="lazy" width="99.4%" height="99.4%" src="{{ site.assets }}/publications/llama_3_multi_modal.svg"/>

{: .caption}
<b>Illustration of the compositional approach to adding multimodal capabilities to Llama 3 studied in the paper.</b> This approach leads to a multimodal model that is trained in five stages: <b>(1)</b> language model pre-training, <b>(2)</b> multi-modal encoder pre-training, <b>(3)</b> vision adapter training, <b>(4)</b> model finetuning, and <b>(5)</b> speech adapter training. Adapted from: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

Section 5 evaluates "(1) the pre-trained language model, (2) the post-trained language model, and (3) the safety characteristics of Llama 3". Section 6 makes inference of Llama 3 405B efficient with two techniques: pipeline parallelism on 16 H100 GPUs in BF16, and FP8 quantization. FP8 applies to most parameters and activations of the feed-forward networks, but not to the parameters of the self-attention layers. As in Xiao et al. (2024b), the scaling factors are dynamic, for better accuracy, with an upper bound of 1200. The first and last Transformer layers are not quantized, and the quantization is row-wise: the scaling factors are computed per row of the parameter and activation matrices.

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

State-of-the-art distributed checkpointing (saving and resuming a model) needs a static allocation of GPU resources at the start of training, and cannot resume with a different parallelism strategy or hardware configuration. Training therefore cannot resume when the hardware changes during training. Universal Checkpointing creates checkpoints efficiently and resumes them on any parallelism strategy. This improves the resilience to hardware failures, since training continues on the remaining healthy hardware, and reduces the training time, by using elastic capacity when it becomes available.

Universal Checkpointing relies on a universal checkpoint format, which allows "mapping parameter fragments into training ranks of arbitrary model-parallelism configuration", and on a universal checkpoint language, which converts distributed checkpoints into this format. The format consolidates the distributed shards into atomic checkpoints, one set of files per model parameter (FP32 weights and optimizer states). These can be re-sharded onto a different layout: data, tensor, pipeline and sequence parallelism, and ZeRO stages.

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

Domino hides the communication of tensor parallelism (TP) behind computation, in the training of large LLMs. Existing methods overlap the TP communication with computation only in part: when a collective takes much longer than a single GEMM, most of the communication time remains as overhead. As GPUs compute faster, the communication overhead grows. Domino breaks the data dependencies of the training of a single batch into smaller independent pieces, and pipelines them, so that communication and computation overlap at a fine grain. The overlap covers more than a single GEMM: an all-reduce also overlaps with LayerNorm, dropout and other operations.

On every GPU, Domino adds a generic tensor partition in two dimensions, on top of the TP partition of the model: a row-wise split of the inputs $$X$$ and a column-wise split of the weights $$B$$. It breaks the TP computation $$X \cdot A \cdot B$$ into smaller compute units without data dependencies, and pipelines them with the collectives. It does not partition $$A$$. Domino slices the input $$X$$ (section 3.2), the weights $$B$$ of the second linear layer (section 3.3), or both (section 3.4). Compared with Megatron-LM, Domino trains LLMs up to 1.3× faster on NVIDIA DGX-H100 GPUs.

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

The paper simplifies the Transformer block, guided by signal propagation theory. It removes the skip connections, the value and projection parameters, the sequential sub-blocks (attention and MLP run in parallel) and the normalization layers. The simplified block matches the training speed per update and the performance of standard autoregressive decoder-only models and BERT encoder-only models, with a 16% higher training throughput (15% in the arXiv version) and 15% fewer parameters. The experiments use models of 100M to 300M parameters.

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

The paper "presents a methodology to select the scalings for FP8 linear layers, based on dynamically updating per-tensor scales for the weights, gradients and activations." It tests two FP8 formats, FP8-E4 and FP8-E5, with 4 and 5 exponent bits. The accumulation runs in 16 bits. The scale is applied to the exponent, as a power-of-two shift of the exponent bias, not as a multiplication of the value. Two scaling methods are compared: AMAX, which derives the scale from the absolute maximum of the tensor, and SCALE, which keeps the scale constant. Neither shows a major degradation. The results compare FP8 with FP16 on GPT and Llama 2 models of 111M to 70B parameters, but not with `bfloat16`, because the hardware was not available at the time. Figure 3 gives the algorithm. FP8 matrix multiplications run at about twice the FP16 or BF16 throughput on hardware that supports them, and the per-tensor scaling adds only cheap element-wise work, so the end-to-end speedup is below 2×.

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

ZeRO-Offload keeps the gradients and the optimizer states in CPU memory, and runs the optimizer step on the CPU. According to the DeepSpeed blog, this is far from optimal in two cases: small batches, and models that do not fit in GPU memory but are not orders of magnitude larger. The offload then adds end-to-end latency and leaves the GPU compute underused. ZeRO-Offload++ adds three features:

- **Twin-Flow** replaces the all-or-nothing offload of the optimizer states. A fraction of the optimizer states stays in CPU memory and the rest in GPU memory. During the optimizer step, the CPU and the GPU update their parameters at the same time. The user sets the ratio of parameters updated on the CPU. The GPU memory and compute stay in use, and the iteration time drops.
- **MemCpy reduction** and an optimized **CPUAdam**, which the blog does not describe.

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

DistFlashAttn extends FlashAttention (see its entry) from one GPU to the sequence-parallel **training** of long-context LLMs. FlashAttention computes attention tile by tile with an online softmax and never stores the full matrix of scores, so its memory grows linearly with the sequence length instead of quadratically. DistFlashAttn splits the sequence across the GPUs, not the heads or the hidden dimension. Each GPU holds the Q, K and V of its own chunk of tokens, with all the heads and the full hidden dimension. Three techniques make this efficient under causal attention:

1. **Token-level load balancing.** With a causal mask, a token attends only to the tokens before it, so the work of a chunk grows with its position in the sequence. In a naive sequence split, the GPUs that hold early tokens sit idle while the GPUs that hold late tokens compute. In DistFlashAttn, an idle GPU fetches a block of queries from a busy GPU and computes its attention against its own keys and values. It returns the partial output with its softmax statistics, and the owner of the queries merges them.
2. **Overlap of the KV communication.** As in Ring Attention, the K and V chunks move between the GPUs, so that each GPU attends to remote tokens. DistFlashAttn overlaps these transfers with the local attention computation.
3. **Rematerialization-aware gradient checkpointing.** Standard gradient checkpointing places the checkpoints at the layer boundaries and recomputes the forward pass of each layer during the backward pass. The backward pass of FlashAttention already recomputes the attention, so the attention forward pass runs twice. DistFlashAttn places the checkpoints at the output of the FlashAttention kernel, which saves one attention forward pass per layer.

Ring Attention is a general sequence-parallel attention, for training and inference, with or without a causal mask. DistFlashAttn targets causal long-context training, where the load imbalance and the cost of checkpointing matter. Its parallel degree is not limited by the number of heads, unlike Megatron-LM and DeepSpeed-Ulysses, which split the attention by heads. The tests use Llama-7B and variants, with 32K to 512K tokens. DistFlashAttn trains on sequences 8× longer than Ring Self-Attention, at 4.45× to 5.64× its speed. It trains on sequences 2× to 8× longer than Megatron-LM with FlashAttention, at 1.24× to 2.01× its speed. It is 1.67× faster than Ring Attention and 1.26× to 1.88× faster than DeepSpeed-Ulysses.

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

Chimera is a compiler framework that fuses chains of compute-intensive operators, such as two GEMMs in a row, into single kernels, to improve the [data locality](https://en.wikipedia.org/wiki/Locality_of_reference). Compute throughput has grown faster than memory bandwidth, so even GEMMs and convolutions become memory-bound when they run as a chain. Each operator writes its full output to memory, and the next one reads it back. The ML compilers of the time had neither a precise analysis nor a good optimization of these chains on different accelerators. Chimera splits each operator into *computation blocks* and optimizes at two levels:

- **inter-block**: an analytical model picks the execution order of the blocks that minimizes the data movement between them, so that the intermediate results stay in on-chip memory instead of going to DRAM;
- **intra-block**: a hardware-specific micro-kernel computes each block.

Examples from the paper:

- **GEMM chain** $$C = AB$$, $$E = CD$$: each tile of $$C$$ is used to compute $$E$$ as soon as it is produced, and $$C$$ is never stored in full. Assigning one tile of $$E$$ to each thread block would cause contention and need atomic operations. Instead, each thread block takes a full block of rows of $$A$$ and $$E$$. It produces the tiles of $$C$$ of that row block one at a time, and iterates over all the columns of $$B$$ and $$D$$.
- **Batch GEMM chain** (attention): $$QK^T$$, a softmax, then the product with $$V$$, which is two batch GEMMs with a softmax between them.
- **Convolution chain** (CNNs): a 3×3 convolution, a ReLU, a 1×1 convolution and a ReLU, fused into one kernel. This chain is common in ResNet-style networks, and is memory-bound for some input shapes.

Chimera uses an analytical cost model instead of an exhaustive search, so it avoids the long tuning times of search-based compilers such as Ansor. On batch GEMM chains and convolution chains, it is up to 2.87×, 2.29× and 2.39× faster than hand-tuned libraries on CPU, GPU (A100) and NPU (Huawei Ascend). On the same platforms, it is up to 2.29×, 1.64× and 1.14× faster than state-of-the-art compilers.

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

The throughput of DeepSpeed ZeRO is limited by its communication: it gathers the weights in the forward and in the backward pass, and averages the gradients. The cost is highest on clusters with low bandwidth, and with small batches per GPU. For a model of size $$M$$, ZeRO all-gathers the parameters ($$M$$) in the forward pass, all-gathers them again ($$M$$) in the backward pass, and then reduce-scatters the gradients ($$M$$). The total volume is $$3M$$, in two all-gathers and one reduce-scatter. ZeRO++ reduces it with three techniques:

1. **Quantized weight communication (qwZ)**: the forward all-gather sends the weights in INT8 (1 byte) instead of FP16 (2 bytes). The usual quantization scales all the parameters by a single factor. qwZ quantizes per block of parameters instead, with a scale and a zero point per block, which reduces the quantization error.
2. **Hierarchical weight partition (hpZ)**: each machine keeps a full FP16 copy of the weights, partitioned across its GPUs. The backward all-gather then runs inside each machine instead of across machines. This trades memory for communication.
3. **Quantized gradient communication (qgZ)** replaces the reduce-scatter of the gradients. The gradients are quantized to INT4 in blocks, sent with an all-to-all, and dequantized to full precision before the reduction, which preserves the training accuracy. A quantized ring reduce-scatter would quantize and dequantize once per GPU, in sequence, and the reductions in low precision would accumulate and amplify the errors.

The volume across machines drops from $$3M$$ to $$0.5M + 0 + 0.25M = 0.75M$$, a 4× reduction compared with ZeRO-3, which gives up to 2.16× higher throughput on 384 GPUs.

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

QLoRA finetunes a 65B-parameter model on a single 48 GB GPU, with the task performance of full 16-bit finetuning, which needs more than 780 GB of GPU memory for LLaMA-65B. It backpropagates the gradients through a frozen pretrained model, quantized to 4 bits, into Low-Rank Adapters (LoRA, see its entry). Three techniques reduce the memory:

1. **4-bit NormalFloat (NF4)**, a data type that the paper calls information-theoretically optimal for normally distributed data, and that gives better results than 4-bit integers and 4-bit floats. NF4 rounds each value to the nearest of 16 levels placed at the [quantiles](https://en.wikipedia.org/wiki/Quantile) of a standard normal distribution. Quantile quantization is expensive in general: it needs an approximate estimation of the quantiles, such as SRAM quantiles, which also gives large errors for outliers. NF4 avoids this cost: pretrained weights follow a zero-centred normal distribution, so the quantiles are computed once, and each block of 64 weights is rescaled by its absolute maximum.
2. **Double quantization** quantizes the quantization constants. With one FP32 constant per block of 64 weights, the constants cost $$32/64 = 0.5$$ bits per parameter. Stored in FP8, with one FP32 constant per block of 256 of them, they cost $$8/64 + 32/(64 \cdot 256) \approx 0.127$$ bits. This saves about 0.37 bits per parameter, or about 3 GB for a 65B model.
3. **Paged optimizers** use NVIDIA unified memory to avoid the memory spikes of gradient checkpointing on mini-batches with long sequences.

QLoRA was used to finetune more than 1,000 models. Finetuning on a small high-quality dataset gives state-of-the-art results, even with smaller models than the previous state of the art. The best model family, Guanaco, reaches 99.3% of the performance of ChatGPT on the Vicuna benchmark, after 24 hours of finetuning on a single GPU.

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

Llama 2 is a family of pretrained and fine-tuned LLMs with 7B to 70B parameters, pretrained on 2T tokens. Llama 2-Chat is the version fine-tuned for dialogue. The models outperform open-source chat models on most benchmarks, and based on human evaluations of helpfulness and safety, the authors argue that they may substitute closed-source models. Figure 3 gives the results of the human safety evaluation of Llama 2-Chat. The training data comes only from publicly available sources, without data from the products or services of Meta, and without sources that may include personal information of users. Table 2 gives the GPU hours, the power consumption and the carbon emissions of each model.

The pretraining setting and the architecture come from LLaMA (see its entry): [byte-pair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) (BPE), pre-normalization with RMSNorm, SwiGLU activations, rotary positional embeddings, the AdamW optimizer with $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$ and $$\epsilon = 10^{-5}$$, and a cosine learning rate schedule. The two main differences are a longer context, of 4K tokens instead of 2K, and **grouped-query attention (GQA)** in the 34B and 70B models.

The fine-tuning combines supervised fine-tuning (Section 3.1), initial and iterative reward modeling (Section 3.2.2), and [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) (Section 3.2.3). The initial RLHF models forgot the initial instruction after a few turns of dialogue (Figure 9, below, left). **Ghost Attention (GAtt)**, inspired by Context Distillation (Bai et al., 2022b), fixes this by changing the fine-tuning data. A system instruction, such as "act as ...", is concatenated to all the user messages of a multi-turn dialogue, and the responses are sampled with the latest RLHF model. For fine-tuning, the instruction is then removed from all the turns but the first, and the loss is set to zero on the tokens of the previous turns. The model learns to follow the instruction across turns (Figure 9, below, right).

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

LLaMA is a family of LLMs with 7B to 65B parameters, trained only on publicly available datasets. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B. Instead of the compute-optimal recipe of Chinchilla, LLaMA trains smaller models on more tokens, which lowers the cost of inference. Table 1 lists the pretraining datasets and Table 2 the training hyperparameters. The text is tokenized with BPE into about 1.4T tokens. The 7B and 13B models train on 1.0T tokens, and the 33B and 65B models on 1.4T.

The architecture changes the original Transformer in three ways:
- **Pre-normalization** (from GPT-3): RMSNorm normalizes the input of each transformer sub-layer instead of its output, which improves the training stability.
- **SwiGLU activation** (from PaLM) replaces ReLU, with a hidden dimension of $$\frac{2}{3} 4d$$ instead of the $$4d$$ of PaLM. SwiGLU has three weight matrices instead of two, so this keeps the parameter count of a standard feed-forward layer with hidden dimension $$4d$$: $$3 \cdot d \cdot \frac{2}{3} 4d = 2 \cdot d \cdot 4d$$.
- **Rotary embeddings** (from GPT-Neo): rotary positional embeddings (RoPE) at each layer replace the absolute positional embeddings.

The training uses:
- the AdamW optimizer, with $$\beta_1 = 0.9$$ and $$\beta_2 = 0.95$$;
- a cosine learning rate schedule with 2000 warmup steps and a final learning rate of 10% of the maximum, a weight decay of 0.1 and a gradient clipping of 1.0;
- an efficient causal multi-head attention, which does not store the attention weights and does not compute the key/query scores that the causal mask removes;
- selective activation checkpointing: the activations that are expensive to compute, such as the outputs of the linear layers, are saved, and only the others are recomputed in the backward pass. This needs a manual implementation of the backward pass of the transformer layers, instead of PyTorch autograd, and model and sequence parallelism (Korthikanti et al., 2022) to reduce the memory;
- an overlap of the computation of the activations with the communication between GPUs (the all-reduce operations).

With these, the 65B model processes about 380 tokens per second per GPU on 2048 A100 GPUs with 80 GB, so one pass over the 1.4T tokens takes about 21 days.

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

A video summary of RetNet is on [YouTube](https://www.youtube.com/watch?v=JaIL1VAEwZ8).

RetNet replaces the multi-head attention of the Transformer with a *multi-scale retention* mechanism, which has three forms of computation:
- a **parallel** form for training, which uses the GPUs fully;
- a **recurrent** form for inference, at $$O(1)$$ cost per token. Compared with a Transformer, it improves the decoding throughput by 8.4×, the latency by 15.6× and the GPU memory by 3.4×, without a loss of performance (Figure 1);
- a **chunkwise recurrent** form for long sequences at linear complexity: each chunk is encoded in parallel, which is fast, and the chunks are summarized recurrently, which saves GPU memory.

RetNet is a stack of $$L$$ identical blocks, with the layout of a Transformer (residual connections and pre-LayerNorm). Each block has a multi-scale retention (MSR) module and a feed-forward network (FFN) module. The input $$X_0$$ has the shape (sequence length × hidden size), and each layer $$l$$ computes the contextualized representations $$X_l$$. A retention head has three equivalent forms:
1. The **parallel representation**, $$\mathrm{Retention}(X) = (Q K^\intercal \odot D) V$$, is attention without the softmax, with an extra causal decay matrix $$D$$, where $$D_{nm} = \gamma^{n-m}$$ for $$n \ge m$$ and 0 otherwise (Eq. 5). It suits parallel training.
2. The **recurrent representation** is a [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN), which suits inference: $$S_n = \gamma S_{n-1} + K_n^\intercal V_n$$ and $$\mathrm{Retention}(X_n) = Q_n S_n$$, where the state $$S_n$$ depends on the previous state $$S_{n-1}$$.
3. The **chunkwise recurrent representation** combines the two, to accelerate training on long sequences. The input sequence is split into chunks. Inside each chunk, the computation uses the parallel representation, and the information passes across chunks with the recurrent representation.

For a sequence of $$N$$ tokens, the parallel form costs $$O(N^2)$$, like attention. The recurrent form keeps a state $$S_n$$ of fixed size ($$d \times d$$ per head), so a decoding step costs the same at every position. The KV cache of a Transformer, instead, grows with the sequence.

Each layer has $$h = d_{model}/d$$ retention heads, where $$d$$ is the head dimension. Each head has its own matrices $$W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d}$$ and its own scalar $$\gamma$$. A layer $$l$$ computes $$Y_l = \mathrm{MSR}(\mathrm{LN}(X_l)) + X_l$$ and $$X_{l+1} = \mathrm{FFN}(\mathrm{LN}(Y_l)) + Y_l$$: a Transformer layer with retention instead of attention.

{: .picture}
<img loading="lazy" width="63%" height="63%" src="{{ site.assets }}/publications/retnet.svg"/>

{: .caption}
Dual form of RetNet: <b>(a)</b> parallel representation and <b>(b)</b> recurrent representation. “GN” is short for GroupNorm. Adapted from: [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)

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

Kernel fusion is one of the most important optimizations of [XLA](https://www.tensorflow.org/xla). This paper, a course-project report, studies the key compiler passes in the source code of XLA. It measures the speedup that kernel fusion gives and its low-level effects on the hardware. A case study on the CartPole reinforcement-learning environment in JAX shows how XLA makes fusion decisions in practice. The fusion strategies that the paper implements run up to 10.56× faster than the baseline implementation.

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

LongNet is a Transformer variant that scales the sequence length to more than 1 billion tokens, without a loss of performance on shorter sequences. Earlier methods for long sequences are limited either by their computational complexity or by the expressivity of the model. LongNet uses **dilated attention**: it splits the sequence into segments of length $$w$$, keeps every $$r$$-th token of each segment ($$r$$ is the dilation rate), and computes the attention among the kept tokens. For $$N$$ tokens of dimension $$d$$, one pair $$(w, r)$$ costs $$O(N w d / r^2)$$, instead of the $$O(N^2 d)$$ of dense attention. LongNet mixes several pairs $$(w, r)$$ that grow geometrically, so that the attention given to a pair of tokens decreases exponentially with their distance. The cost is linear in the sequence length, and the dependency path between any two tokens has a logarithmic length. The sequence dimension can be distributed across devices, to train on extremely long sequences.

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

FlashAttention (see its entry) reaches only 25–40% of the theoretical maximum FLOPs/s of an A100, far below optimized GEMM operations. The cause is a suboptimal partition of the work between the thread blocks and the warps of the GPU, which gives either a low occupancy or unnecessary reads and writes of shared memory. FlashAttention-2 changes the partition in three ways:
1. It reduces the non-matmul FLOPs, for example by rescaling the output once at the end instead of after every block. On an A100, FP16 matmuls run at up to 312 TFLOPs/s but non-matmul FP32 operations at only 19.5 TFLOPs/s, so each non-matmul FLOP costs 16× more than a matmul FLOP.
2. It parallelizes the attention of a single head across thread blocks, over blocks of the sequence, which increases the occupancy.
3. Inside each thread block, it splits the queries across the warps, instead of the keys and values, which reduces the communication through shared memory.

This gives about 2× the speed of FlashAttention, and 50–73% of the theoretical maximum FLOPs/s on A100, close to the efficiency of GEMM. In the end-to-end training of GPT-style models, FlashAttention-2 reaches up to 225 TFLOPs/s per A100 GPU, a model FLOPs utilization of 72%.

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

Flow Matching is a simulation-free method to train [continuous normalizing flows](https://en.wikipedia.org/wiki/Flow-based_generative_model) (CNFs). A network $$v_\theta$$ regresses a vector field that moves samples from noise to data. The target field is defined per data sample $$x_1$$, which makes the loss tractable. The method works with a general family of Gaussian probability paths between noise and data, which includes the paths of the reverse process of diffusion models. It also allows paths that do not come from diffusion. The paths of [optimal transport](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)) (OT) displacement interpolation are straight lines, and give the loss

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, x_0, x_1} \left\lVert v_\theta(x_t, t) - \big(x_1 - (1 - \sigma_{min}) x_0\big) \right\rVert^2, \qquad x_t = \big(1 - (1 - \sigma_{min}) t\big) x_0 + t x_1,
$$

with $$t \sim \mathcal{U}[0, 1]$$, noise $$x_0 \sim \mathcal{N}(0, I)$$, data $$x_1$$, and a small $$\sigma_{min}$$. The OT paths are more efficient than diffusion paths: training and sampling are faster, and the models generalize better. With $$\sigma_{min} = 0$$, the loss is the one of rectified flow (see its entry). A [post of the Cambridge Machine Learning Group](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) explains the method.

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

Rectified flow reduces the number of steps needed to transport one distribution to another. Diffusion models, for example, sample with $$T$$ steps, and fewer steps make the inference cheaper. A rectified flow is an [ordinary differential equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation) (ODE) model that transports a distribution $$\pi_0$$ to $$\pi_1$$ along paths that are as straight as possible. It is learned with a least-squares regression of the velocity field onto the direction $$X_1 - X_0$$ between paired samples. Straight paths are preferred in theory, because a straight line is the shortest path between two end points, and in computation, because a straight path can be simulated exactly without time discretization. Re-training on the couplings that the model produces (*reflow*) straightens the paths further, so that a single [Euler step](https://en.wikipedia.org/wiki/Euler_method), optionally after distillation, already gives good samples. The same formulation later became the training objective of Stable Diffusion 3.

**Rectified flow.** Given empirical observations of $$X_0 \sim \pi_0$$ and $$X_1 \sim \pi_1$$, the rectified flow induced from $$(X_0, X_1)$$ is an ODE on time $$t \in [0, 1]$$,

$$
\mathrm{d}Z_t = v(Z_t, t)\mathrm{d}t,
$$

which converts $$Z_0$$ from $$\pi_0$$ into a $$Z_1$$ that follows $$\pi_1$$. The drift $$v \colon \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d$$ drives the flow to follow the direction $$X_1 - X_0$$ of the linear path from $$X_0$$ to $$X_1$$ as closely as possible. It is the solution of a least-squares regression problem:

$$
\min_v \int_0^1 \mathbb{E}\left[\left\|(X_1 - X_0) - v\big(X_t,\, t\big)\right\|^2\right] \mathrm{d}t, \qquad \text{with} \quad X_t = tX_1 + (1 - t)X_0, \tag{1}
$$

where $$X_t$$ is the linear interpolation of $$X_0$$ and $$X_1$$. The interpolation $$X_t$$ itself follows the ODE $$\mathrm{d}X_t = (X_1 - X_0)\mathrm{d}t$$, which is non-causal (anticipating): the update of $$X_t$$ needs the final point $$X_1$$. Fitting the drift $$v$$ to $$X_1 - X_0$$ *causalizes* the paths of the linear interpolation, and gives an ODE flow that can be simulated without seeing the future. In practice, $$v$$ is a neural network or another nonlinear model, and (1) is solved with any off-the-shelf stochastic optimizer, such as stochastic gradient descent, on empirical draws of $$(X_0, X_1)$$.

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

For a given input coupling $$(X_0, X_1)$$, the exact minimum of (1) is

$$
v^X(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x], \tag{2}
$$

the expectation of the directions $$X_1 - X_0$$ of the lines that pass through $$x$$ at time $$t$$.

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

The [Maximal Update Parametrization (µP)](https://arxiv.org/abs/2011.14522) sets the initialization variance, the learning rate and the output multiplier of each layer as functions of its width. The activations and their updates then stay of order 1 as the width grows. Under µP, many optimal hyperparameters stay stable when the width of the model changes. When the width of a layer is large, every activation vector has roughly iid coordinates at any time during training. Tensor Programs compute the distributions of these coordinates recursively, which describes how the function of the network evolves.

µTransfer, the tuning method of the paper, parametrizes the target model in µP, tunes the hyperparameters on a smaller model, and transfers them zero-shot to the full-size model. Hyperparameters transferred from a 40M-parameter proxy outperform the published numbers of the 6.7B GPT-3, at a tuning cost of 7% of the total pretraining cost. Hyperparameters transferred from a 13M-parameter model outperform the published numbers of BERT-large (350M), at a total tuning cost of one pretraining of BERT-large.

{: .picture}
<img loading="lazy" width="77.6%" height="77.6%" src="{{ site.assets }}/publications/mutransfer.svg"/>

{: .caption}
Training loss against learning rate on Transformers of varying $$d_{model}$$ trained with Adam, in the standard parametrization (left) and in µP (right). In the standard parametrization, unlike in µP, different widths do not share the same optimal hyperparameter; wider networks do not always perform better than narrower ones; in fact they underperform the same-width networks in µP even after tuning learning rate (see dotted line). Adapted from: [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)

Table 1 of the paper lists the hyperparameters that µTransfer covers, with caveats in Section 6.1. A * marks the ones validated only empirically, on Transformers; the others also have a theoretical justification.
- µTransferable: the optimization hyperparameters (learning rate, momentum, Adam beta, learning rate schedule), the initialization (per-layer initialization variance) and the parameter multipliers (constants that multiply the weights or biases);
- not µTransferable: the regularization (dropout, weight decay);
- µTransferred across: width, depth*, batch size*, training time*, sequence length*.

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

DeepSpeed-MoE addresses the bottlenecks specific to MoE models (**token routing**, **all-to-all communication**, model size and inference cost) in three ways:
- **model design**: the Pyramid-Residual MoE (PR-MoE) has more experts in the later layers, and in each layer a fixed dense MLP plus one residual expert;
- **compression**: Mixture-of-Students (MoS) distillation;
- **inference system**: a combination of expert, tensor and data parallelism, with hierarchical all-to-all communication and fused sparse kernels.

DeepSpeed-MoE keeps standard collectives (NCCL), outside the compute kernels. Later systems for production scale, such as MegaScale-MoE and MegaScale-Infer (see their entries), add more specialized parallelism strategies and communication libraries. Later kernel-level work moves the collectives into the kernels, for a fine-grained overlap with the computation.

For autoregressive language models, MoE reduces the training cost by about 5× compared with a dense model of equal quality. PR-MoE and MoS reduce the size of the MoE model by up to 3.7×. The inference system gives 7.3× better latency and cost than existing MoE inference solutions, and inference up to 4.5× faster and 9× cheaper than dense models of equal quality.

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

Inference runs at small batch sizes, so its kernels must reach a high memory bandwidth utilization and a high compute utilization at small batch sizes. Training kernels only need a high compute utilization, at much larger batch sizes. DeepSpeed Inference has two components:
1. **DeepSpeed Transformer** partitions the operators of a dense transformer across several devices, and adds the communication that the partition needs. Its single-GPU kernels maximize the memory bandwidth utilization of each GPU, and NCCL all-reduce collectives perform the communication across GPUs. This gives a high aggregate memory bandwidth utilization across the GPUs of a node. Tensor slicing does not scale efficiently beyond one node, because of its communication overhead, so DeepSpeed Inference uses pipeline parallelism across nodes. DeepSpeed Transformer has three parts:
   - single-GPU transformer kernels that minimize the latency and maximize the throughput, with fusion schedules and GEMM kernels designed around the memory bandwidth (Sec. III, described below);
   - a dense inference system for many GPUs, which combines tensor parallelism, to minimize the latency, with pipeline-parallel schedules optimized for inference and memory optimizations, to maximize the throughput (Sec. IV). The tensor and pipeline parallelism run on top of the single-GPU kernels;
   - a sparse (MoE) inference system for massive numbers of GPUs, which serves trillions of parameters on hundreds of GPUs (Sec. V). It combines expert, data and tensor parallelism with new communication optimizations and sparse kernel optimizations. The data is replicated across the tensor-parallel ranks, so the all-to-all of expert parallelism runs only within the subset of devices that share the same tensor-slicing rank. The sparse tensor representation of the gating function and the sparse einsum operators add a large latency, which kernel fusion reduces.
2. **ZeRO-Inference** uses CPU, NVMe and GPU memory together with GPU compute, to run the inference of massive models on limited resources (Sec. VI). The key design decision is how to divide the GPU memory between the model weights, the inference inputs and the intermediate results. One approach pins as many weights as possible in GPU memory and fetches the rest from DRAM or NVMe when needed, which leaves room only for small batches. ZeRO-Inference instead pins the weights in DRAM, when it is large enough, or in NVMe, and streams each layer into GPU memory when it is needed.

The single-GPU transformer kernels use two techniques:
1. **Deep-Fusion** reduces the overhead of kernel launches and data movement by fusing more than element-wise operations. On a GPU, data that one thread block produces and another one consumes needs a global memory synchronization, which means a new kernel launch. To avoid it, Deep-Fusion tiles the computation along the dimensions of the iteration space that have no data dependencies across tiles, and runs the tiles in parallel on different thread blocks. The dimensions with data dependencies are not tiled, and one thread block processes them. After this tiling, two operators can be fused if each tile of the second operator depends on exactly one output tile of the first. Deep-Fusion can then fuse element-wise operations, reductions, data transpositions and GEMMs, as long as they have no dependencies across tiles.
2. A **custom GEMM** that Deep-Fusion can fuse, and that reaches the maximum memory bandwidth utilization. The computation is tiled along the output dimension, so that one kernel computes the GEMM, with the reduction inside each tile. Each warp of a thread block produces a partial reduction for a tile of outputs, and a final reduction across the warps of the thread block is needed. Instead of a reduction in shared memory, a single transpose in shared memory makes the partial results of each output element contiguous. One warp then reduces them with cooperative-group collectives, directly in registers. The weight matrix is also transposed once at initialization, so that $$M$$ rows of each column are contiguous in memory. Each thread then reads $$M$$ elements along the input dimension in one access ($$M = 2$$ in FP16 and 4 in INT8). The loads then use whole cache lines, and the number of parallel tiles stays the same.

The operations inside a transformer layer are fused in four regions: (1) the QKV GEMM and the input layer norm, (2) the transposition and the attention, (3) the post-attention layer norm and the intermediate GEMM, and (4) the bias and residual addition.

DeepSpeed Inference fuses the computation aggressively (Deep-Fusion, custom GEMMs) but keeps the communication as NCCL collectives outside the kernels. It shows what kernel engineering and runtime scheduling achieve without a change of the communication model.

DeepSpeed Inference reduces the latency by up to 7.3× compared with the state of the art in latency-oriented scenarios, and increases the throughput by more than 1.5× in throughput-oriented ones. It serves trillion-parameter models in real time on hundreds of GPUs. With ZeRO-Inference, it runs models 25× larger than GPU-only solutions, at 84 TFLOPS, more than 50% of the peak of an A6000.

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

LoRA blocks (see the LoRA entry) have two problems. Their rank is fixed after training, so a new rank needs a new training from scratch. Finding the best rank also needs an exhaustive search. Dynamic LoRA (DyLoRA) trains LoRA blocks for a range of ranks instead of a single rank, by sorting the representation that the adapter learns at different ranks during training. It works as follows:
- Each LoRA module has an up-projection $$W_{up} \in \mathbb{R}^{m \times r}$$ and a down-projection $$W_{dw} \in \mathbb{R}^{r \times d}$$. The module is trained to work for any rank $$r$$ in a range $$[r_{min}, r_{max}]$$, where $$r_{min}$$ and $$r_{max}$$ are hyperparameters.
- At each training step, a rank $$b$$ between $$r_{min}$$ and $$r_{max}$$ is sampled, and $$W_{up}$$ and $$W_{dw}$$ are truncated to their first $$b$$ columns and rows, $$W_{up\downarrow b}$$ and $$W_{dw\downarrow b}$$. The step computes $$h = W_0 x + \frac{\alpha}{b} W_{up\downarrow b} W_{dw\downarrow b} x$$.

DyLoRA trains these dynamic, search-free models 4–7× faster than LoRA, depending on the task, without a significant loss of performance.

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

The paper studies the efficient generative inference of large Transformer models, with tight latency targets and long sequences, on TPU v4 and its 3D torus network. Its methods are more efficient than [NVIDIA's FasterTransformer](https://github.com/NVIDIA/FasterTransformer).

Section 3.1 analyzes the collective communication, and Section 3.2 the parallelism of the feed-forward layer. In the notation, $$BLE_{xyz}$$ means that the last dimension $$E$$ of a tensor of logical shape $$BLE$$ is split into $$X \times Y \times Z$$ partitions. The tensor on each chip then has the shape $$[B, L, E/(X \times Y \times Z)]$$, and the axes without a subscript are replicated. $$F$$ is the hidden (intermediate) size of the feed-forward layer. Three layouts are compared:
- **1D weight-stationary** (Section 3.2.1), as in Megatron: the weights are partitioned as $$EF_{xyz}$$ and $$F_{xyz}E$$, over $$X \times Y \times Z = n_{chips}$$ chips.
- **2D weight-stationary** (Section 3.2.2): the weights are split along both $$E$$ and $$F$$, into square shards. The compute cost is the same, but the communication is more efficient and scales better, in particular beyond 16 chips.
- **XYZ-weight-gathered** (Section 3.2.3): with a weight-stationary layout, the output of each per-chip matrix multiplication must be aggregated between chips before the next operation. For very large batches, it is best to keep the activations fully stationary between sequential matrix multiplications, and to transfer the weights fully between all chips.

The weight-stationary layouts suit small batches, such as in decoding, and the weight-gathered layout suits very large batches, such as in the processing of long prompts.

Multi-head attention can be parallelized in the same ways as a feed-forward layer (Section 3.3). The keys and values (the KV cache) take much memory capacity and bandwidth. [Multi-query attention](https://arxiv.org/abs/1911.02150) reduces the size of the KV cache by a factor of $$n_{heads}$$, and the time to load it from memory. It also removes the head axis, which the other layouts use for parallelism. The KV cache and its computations then need another partition, one that minimizes the time of the repeated loads of the KV cache, which dominate the inference cost. The layout closest to multi-head attention (Figure 4(b)) partitions the KV cache in the same way. The keys and values, shared by all heads, are then replicated on each chip, and the memory savings of multi-query attention are lost. The paper instead partitions $$Q$$, $$K$$ and $$V$$ over the batch dimension $$B$$, into $$n_{chips}$$ partitions. This divides the cost of loading the KV cache per chip by $$n_{chips}$$, at the extra communication cost of resharding the input activations. With this layout, multi-query attention allows larger batches and longer sequences, which increases the throughput, in addition to the lower latency from the reduced memory time.

{: .picture}
<img loading="lazy" width="65.2%" height="65.2%" src="{{ site.assets }}/publications/efficiently_scaling_transformer_inference_multiquery.svg"/>

{: .caption}
Multiquery attention has lower memory cost to load the KV cache when sharded over batch. Top: the shapes of Q and K/V in multi-head and multi-query attention. <b>(a)</b> Multi-head attention, sharded over heads: it can be sharded across heads without replication. <b>(b)</b> Multi-query attention, sharded over heads: it requires full replication of the single head for K, increasing memory access cost. <b>(c)</b> Multi-query attention, sharded over batch: only a slice of K is needed for einsum, reducing memory access cost. Adapted from: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

Section 3.4 measures the gains of computing the attention and the feed-forward layer in parallel, as in [GPT-J](https://en.wikipedia.org/wiki/GPT-J#cite_note-Model_Card-2) and [PaLM](https://arxiv.org/abs/2204.02311). The standard transformer block computes $$y = x + \mathrm{Attention}(\mathrm{LayerNorm}(x)) + \mathrm{MLP}(\mathrm{LayerNorm}(x + \mathrm{Attention}(\mathrm{LayerNorm}(x))))$$, while the parallel block computes $$y = x + \mathrm{MLP}(\mathrm{LayerNorm}(x)) + \mathrm{Attention}(\mathrm{LayerNorm}(x))$$. The parallel block has one layer norm per layer instead of two, which reduces the latency at small batch sizes. Some of its matrices can be fused into larger matrix multiplications, which run more efficiently on accelerators. It also removes one of the two all-reduce operations of each layer.

Section 3.5 covers low-level optimizations, such as the overlap of communication and computation, and Section 3.6 the `int8` quantization. Together, on PaLM 540B, these reach a latency of 29 ms per generated token at small batch sizes, with `int8` weights. In the large-batch processing of 2048-token inputs, they reach a model FLOPs utilization (MFU) of 76%.

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

Random-LTD (random and layerwise token dropping) reduces the training cost of large transformer models. At all the middle layers, excluding the first and the last, it skips the computation of a subset of the input tokens. The tokens are dropped at random, so Random-LTD needs no scoring and no manual design.

Other token-dropping methods use:
- metrics based on the attention scores, which cost too much for layerwise dropping, since the metric must be computed at every layer;
- metrics based on the loss or the frequency of the tokens, accumulated over training. Such a metric does not change within an iteration (forward pass), so the same tokens are dropped at all the layers. The multi-head attention of the middle layers then does not capture the dependencies of the dropped tokens.

Random-LTD drops tokens at random at every layer. To reduce the gradient variance that this adds, the number of kept tokens grows linearly during training: the **Monotonic Sequence Length Growth (MSLG)**.

Random-LTD reduces the effective batch size of the middle layers during the warmup, and MSLG reaches the full length only after more than 2/3 of the training iterations, for a large compute saving. The small learning rate of the warmup then trains Random-LTD inefficiently, and in practice the warmup must be longer and the learning rate decay slower. The paper therefore introduces the **LayerToken learning rate** (Appendix C), which scales the learning rate with the sum of the tokens that each layer consumes, to reduce the tuning effort.

The results compare Random-LTD with the baseline on GPT-3 models with 350M and 1.3B parameters, on up to 300B tokens. Random-LTD reaches evaluation losses similar to the baseline with 1/3 less LayerToken consumption. The paper stresses that this saving does not transfer directly to wall-clock time, because of the implementation and the hardware. On GPT-3 1.3B, the 33.3% theoretical compute saving gives a 25.6% wall-clock saving, with a similar zero-shot accuracy. BERT and ViT models show similar results. Random-LTD gives a better training and validation perplexity than TokenBypass, which also skips tokens in the middle layers, and MSLG gives a better perplexity than a constant drop rate (Table 6.4).

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

The paper studies the role of the sequence length in the training of GPT models. Experiments with GPT-2 models on public datasets explore the **stability-efficiency dilemma**: "increasing the batch sizes and learning rates (commonly used for faster training) leads to better training efficiency but can also result in training instability, leading to poor generalization accuracy or failed runs". The training instability correlates strongly with extreme values of the gradient variance, and long sequences contribute to these extreme values, in particular at the beginning of training.

**Sequence Length Warmup (SLW)** starts the training with short sequences and increases their length gradually, which avoids the extreme gradient variance. It is a form of [curriculum learning](https://en.wikipedia.org/wiki/Curriculum_learning) (CL), which presents easier examples first and increases their difficulty gradually. Traditional CL keeps the hyperparameters constant. SLW instead uses the stable training to allow more aggressive hyperparameters, for both efficient convergence and better stability.

On GPT-2 (117M and 1.5B), SLW trains stably with an 8× larger batch and a 4× larger learning rate. For the same zero-shot quality, it reduces the training tokens by up to 2.2× and the wall-clock time by up to 3.7×. On GPT-3 (125M), it allows an 8× larger batch and a 40× larger learning rate, and keeps 99% of the zero-shot accuracy on 11 tasks with 10× less data and 17× less time.

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

Self-attention costs time and memory quadratic in the sequence length, so Transformers are slow and memory-hungry on long sequences. The paper argues that attention algorithms miss *IO-awareness*: accounting for the reads and writes between the levels of GPU memory. FlashAttention computes exact attention with tiling, which reduces the reads and writes between the high-bandwidth memory (HBM) of the GPU and its on-chip SRAM. It splits the attention matrix into blocks that are computed in the on-chip SRAM (shared memory) instead of the global memory (HBM). For a sequence of length $$N$$, a head dimension $$d$$ and an SRAM of size $$M$$, FlashAttention needs $$O(N^2 d^2 / M)$$ HBM accesses, while standard attention needs $$O(N d + N^2)$$. It trains Transformers faster than the baselines: 15% faster end-to-end on BERT-large than the MLPerf 1.1 record, 3× faster on GPT-2 and 2.4× faster on long-range arena.

{: .picture}
<img loading="lazy" width="99.5%" height="99.5%" src="{{ site.assets }}/publications/flashattention.svg"/>

{: .caption}
<b>Left:</b> FlashAttention uses tiling to prevent materialization of the large <i>N</i> × <i>N</i> attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the <b>K</b> and <b>V</b> matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of <b>Q</b> matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. <b>Right:</b> Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large <i>N</i> × <i>N</i> attention matrix to HBM, resulting in a 7.6× speedup on the attention computation. Adapted from: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

FlashAttention stores neither the full matrix $$QK^\top$$ nor the large intermediate buffers of the softmax: it uses a numerically stable streaming (online) softmax. Instead of computing attention as a GEMM, a softmax and a GEMM, with large intermediate tensors in HBM, it tiles the computation so that the data is reused from the SRAM. The fusion changes the memory complexity of the algorithm, not only the kernel launch overhead. FlashAttention is about IO-aware tiling, not about the overlap of distributed communication.

The memory becomes linear in the sequence length. The longer contexts that this allows give better models: 0.7 better perplexity on GPT-2, 6.4 more points on long-document classification, and the first Transformers better than chance on Path-X (16K tokens) and Path-256 (64K tokens).

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

DeepNet introduces **DeepNorm**, a normalization function that changes the residual connection of the Transformer to $$x_{l+1} = LN(\alpha x_l + G_l(x_l, \theta_l))$$, with a constant $$\alpha > 1$$. A theoretically derived initialization scales some weights down by a constant $$\beta$$. It stabilizes the training of extremely deep Transformers.
- Earlier work showed that better initialization methods improve the stability of Transformer training.
- DeepNorm has a theoretical justification: it bounds the model update by a constant.
- "The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DeepNorm a preferred alternative."
- Figure 2 gives the pseudocode of `deepnorm` (the normalization layer) and `deepnorm_init` (the weight initialization), and the constants. For an encoder-only or a decoder-only model with $$N$$ layers, $$\alpha = (2N)^{1/4}$$ and $$\beta = (8N)^{-1/4}$$. $$\beta$$ scales the initialization of the feed-forward, value and output projections, and the query and key projections keep a gain of 1.
- DeepNet trains Transformers with up to 1,000 layers (2,500 attention and feed-forward sub-layers). On a multilingual benchmark with 7,482 translation directions, a 200-layer model with 3.2B parameters outperforms the 48-layer, 12B-parameter state of the art by 5 BLEU points.

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

Traditional training supervises a network only at its last layer, and the supervision propagates layer by layer, which makes the intermediate layers hard to optimize. Deep supervision (see the Deeply-supervised Nets entry) adds auxiliary classifiers to the intermediate layers and trains them with the task loss, so that the supervision reaches the shallow layers directly. The shallow layers, however, learn low-level features, not task-biased high-level semantic features, which conflicts with a task loss at those layers. Contrastive Deep Supervision supervises the intermediate layers with augmentation-based contrastive learning instead, which gives them a better supervision than the task loss. Contrastive learning treats two augmentations of the same image as a positive pair, and two different images as a negative pair. The network learns to minimize the distance of the positive pairs and to maximize the distance of the negative pairs. It thus learns the invariance to data augmentations such as Color Jitter and Random Gray Scale.

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

The paper asks: "Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens?" It answers with three approaches: (1) fix the model sizes and vary the number of training tokens; (2) vary the model size for 9 fixed FLOP counts; (3) fit a parametric loss function to the final losses of the first two approaches. The estimates come from more than 400 models, with 70M to 16B parameters, trained on 5B to 500B tokens. Approach 3 fits $$\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$ for $$N$$ parameters and $$D$$ training tokens, with $$E = 1.69$$, $$A = 406.4$$, $$B = 410.7$$, $$\alpha = 0.34$$ and $$\beta = 0.28$$.

The main conclusion is that the large language models of the time are significantly undertrained, because their model size grew but their data did not. For compute-optimal training, the model size and the number of training tokens should grow in equal proportions: the optimal $$N$$ and $$D$$ both grow about as $$C^{0.5}$$ with the compute $$C$$. A compute-optimal model, Chinchilla (70B parameters), confirms this. It uses the compute budget of Gopher (280B) with 4× more data (1.4T tokens), about 20 tokens per parameter. Chinchilla outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B) and Megatron-Turing NLG (530B) on several evaluation tasks, with an average accuracy of 67.5% on MMLU, more than 7% above Gopher.

This differs from Kaplan et al. (2020) (see the scaling laws entry). They state that a compute-optimal model, with the lowest loss for a given compute budget, should not be trained to its lowest possible loss. For a 10× larger compute budget, the model should then grow by 5.5× and the training tokens by 1.8×.

{: .picture}
<img loading="lazy" width="60.2%" height="60.2%" src="{{ site.assets }}/publications/training_compute_optimal_large_language_models.svg"/>

{: .caption}
<b>Overlaid predictions.</b> Predictions of the paper's three approaches, overlaid with the projections of Kaplan et al. (2020). All three approaches predict that current large models should be substantially smaller and therefore trained much longer than is currently done. <b><i>Chinchilla</i> outperforms <i>Gopher</i> and the other large models (Section 4.2 of the paper).</b> Adapted from: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

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

Absolute positional encodings, sinusoidal or learned, generalize poorly to long sequences, for two reasons: (1) absolute positions are fixed and do not model the relations between positions; (2) they have no mechanism to extrapolate beyond the sequence lengths seen in training. Rotary position embeddings (RoPE) address the first issue: they encode the **relative position** directly in the attention, with a dependency between tokens that decays as their relative distance grows. RoPE alone still extrapolates poorly beyond the training length, which later motivated position interpolation, NTK-aware scaling and YaRN. In the words of the paper, "RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation".

RoPE rotates the queries ($$Q$$) and the keys ($$K$$) of self-attention, by angles that depend on the positions of the tokens. A pair of coordinates $$(x_1, x_2)$$ of an input $$x$$ is rotated by an angle $$\theta$$:

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

For a token at position $$p$$, the $$i$$-th pair of dimensions is rotated by $$\theta = p \, w_i$$, where $$w_i = 10000^{-2(i-1)/d}$$ is the frequency of the pair. A $$d$$-dimensional input is split into the pairs of dimensions $$[ (x_1, x_2), \, (x_3, x_4), \, \dots, \, (x_{d-1}, x_d)]$$, each pair is rotated, and the results are concatenated into the rotated vector. When $$d$$ is odd, the extra dimension is often left unrotated. Rotations compose, so with $$R_p$$ the rotation of all the pairs at position $$p$$, $$\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$$: the attention score depends only on the relative position $$n-m$$.

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

Rabe and Staats show that self-attention can be computed exactly without storing the full $$n \times n$$ attention matrix, with a streaming (online) computation of the softmax-normalized attention. The paper gives the mathematical basis for computing attention in blocks with numerical stability and correctness, which later attention kernels use. Its algorithm needs $$O(1)$$ memory for single-query attention and $$O(\log n)$$ memory for self-attention. A practical implementation for accelerators needs $$O(\sqrt{n})$$ memory, and runs within a few percent of the runtime of standard attention. At a sequence length of 16384, it reduces the memory overhead of self-attention by 59× for inference and by 32× for differentiation.

Not storing $$QK^\top$$ is therefore exact, not an approximation. Many attention kernels for long contexts build on this: they recompute values instead of storing them. FlashAttention (see its entry) adds IO-aware tiling, which also makes attention faster, while this algorithm keeps the runtime of standard attention.

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

Performers approximate regular (softmax) full-rank attention with provable accuracy, in linear instead of quadratic space and time, without priors such as sparsity or low rank. They approximate the softmax attention kernel with Fast Attention Via positive Orthogonal Random features (FAVOR+). A random feature map $$\phi$$ gives $$A \approx Q' K'^\top$$, with $$Q' = \phi(Q)$$ and $$K' = \phi(K)$$, which have $$r$$ columns (random features). The features are positive, which keeps the estimated attention weights nonnegative, and the random directions are orthogonal, which reduces the variance of the estimate. For $$L$$ tokens of dimension $$d$$, computing $$Q' (K'^\top V)$$ instead of $$(Q' K'^\top) V$$ costs $$O(L r d)$$ instead of $$O(L^2 d)$$. The [Google Research post](https://blog.research.google/2020/10/rethinking-attention-with-performers.html) explains the two cases below.

**Bidirectional attention**, without a notion of past and future: the matrix multiplications run in the order of the dashed boxes, so that neither $$A$$ nor its approximation is ever built.

{: .picture}
<img loading="lazy" width="70.6%" height="70.6%" src="{{ site.assets }}/publications/performers.svg"/> 

{: .caption}
**Left:** Standard attention module computation, where the final desired result is computed by performing a matrix multiplication with the attention matrix $$A$$ and value tensor $$V$$. **Right:** Decoupling matrices $$Q′$$ and $$K′$$ used in the lower rank decomposition of $$A$$ and conducting the matrix multiplications in the order indicated by the dashed boxes gives a linear attention mechanism, which never explicitly constructs $$A$$ or its approximation. Adapted from: [Rethinking Attention with Performers](https://research.google/blog/rethinking-attention-with-performers/)

**Unidirectional (causal) attention**, where tokens do not attend to later tokens: the computation uses prefix sums, which store only running totals of the matrix products instead of an explicit lower-triangular attention matrix.

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

CoCoNet provides a [domain-specific language](https://en.wikipedia.org/wiki/Domain-specific_language) (DSL) that expresses a program with both computation and communication, ML-aware transformations that optimize the program, and a compiler that generates high-performance kernels. With computation and communication as first-class constructs, a program is written at a high level of abstraction and gets optimizations such as the fusion or the overlap of communication and computation. CoCoNet optimizes data-, model- and pipeline-parallel workloads of large language models with a few lines of code. For example, it fuses a ReduceScatter, the sliced computation that follows it, and an AllGather into a single FusedAllReduce operation. Each rank then computes only its slice of the element-wise operations between the two collectives, instead of the full tensor.

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

Low-Rank Adaptation (LoRA) fine-tunes large models efficiently by adding trainable low-rank updates to their layers. It adapts pretrained models to new tasks or domains without retraining the whole model, which is expensive in compute and in storage. LoRA decomposes the parameter updates into low-rank matrices, which reduces the number of trainable parameters and keeps the expressive power of the model.

A trainable low-rank matrix $$\Delta W$$ is added to the frozen weight matrix $$W \in \mathbb{R}^{d \times k}$$, and the layer uses $$W' = W + \Delta W$$. Instead of learning $$\Delta W$$ directly, LoRA factorizes it as $$\Delta W = BA$$, with $$B \in \mathbb{R}^{d \times r}$$ and $$A \in \mathbb{R}^{r \times k}$$, and a rank $$r \ll \min(d, k)$$: the layer trains $$r(d + k)$$ parameters instead of $$dk$$. $$A$$ starts from a random Gaussian and $$B$$ from zero, so training starts from the pretrained model. The update is scaled by $$\alpha / r$$, and the paper applies it to the attention projections (e.g., $$W_q$$ and $$W_v$$). $$\Delta W$$ can be merged into $$W$$ after training, so inference has no extra latency. Compared with the full fine-tuning of GPT-3 175B with Adam, LoRA reduces the number of trainable parameters by 10,000×, and the GPU memory during training by 3×, from 1.2 TB to 350 GB. It matches or exceeds the quality of fine-tuning on RoBERTa, DeBERTa, GPT-2 and GPT-3.

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

State-of-the-art vision systems are trained on a fixed, predetermined set of object classes, and every other visual concept needs more labeled data. In NLP, however, "the development of text-to-text as a standardized input-output interface has enabled task-agnostic architectures to zero-shot transfer to downstream datasets, removing the need for specialized output heads or dataset specific customization". Natural language can serve as a flexible prediction space for vision too, which enables generalization and transfer. Through its generality, it can express, and therefore supervise, a much wider set of visual concepts. Unlike most unsupervised or self-supervised methods, learning from natural language does not only learn a representation: it also connects the representation to language, which enables flexible zero-shot transfer.

CLIP (Contrastive Language–Image Pre-training) learns visual concepts from natural language supervision. It can be instructed in natural language to perform a large variety of classification benchmarks, without optimizing for any of them, similar to the zero-shot capabilities of GPT-2 and GPT-3. CLIP models apply to nearly arbitrary visual classification tasks.

CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in its dataset. The dataset holds 400 million pairs of images and texts, collected from the internet with 500K queries. The image encoders are 5 ResNets and 3 Vision Transformers. The ResNets are ResNet-50, ResNet-101, and three EfficientNet-style ResNet-50s scaled to 4×, 16× and 64× the compute, with changes such as an attention pooling that works like QKV attention. The text encoder is a Transformer with masked self-attention, with a BPE vocabulary of 49,152 tokens and a maximum sequence length of 76 (Section 2.4).

{: .picture}
<img loading="lazy" width="67.5%" height="67.5%" src="{{ site.assets }}/publications/clip.svg"/> 

{: .caption}
Summary of CLIP: (1) contrastive pre-training, (2) creating a dataset classifier from label text and (3) using it for zero-shot prediction. While standard image models jointly train an image feature extractor and a linear classifier to predict some label, CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes. Adapted from: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

The contrastive pre-training (1) works as follows:
- The image encoder (ViT or ResNet) encodes each image $$i$$ into $$I_i$$, and the text encoder (Transformer) encodes each text $$i$$ into $$T_i$$.
- The diagonal of the matrix $$I_i T_j$$ holds the correct image–text pairs (in blue).
- Contrastive learning pulls the embeddings of matching texts and images together, and pushes the others apart. The training maximizes the inner product of the $$N$$ matching pairs (the diagonal), and minimizes the inner product of the $$N^2-N$$ other pairs (off the diagonal). The products are logits, and a softmax in both directions gives a loss that is symmetric between texts and images. In the words of the paper, "the cosine similarity (i.e. cosine of the angle) of these embeddings is then calculated, scaled by a temperature parameter τ, and normalized into a probability distribution via a softmax". Figure 3 gives the pseudocode.
- Each image, for example $$I_3$$, thus gets a distribution over the texts $$I_3 T_1$$, $$I_3 T_2$$, …, $$I_3 T_N$$.

The batch size $$N$$ matters: each image is contrasted with the $$N-1$$ other texts of the batch, and the similarity matrix costs $$O(N^2)$$. CLIP uses a batch of 32,768 pairs.

At inference (2), a prompt is added to each label of a set of labels, for example `A photo of a {label}.` The text encoder embeds the prompts into the target set. The zero-shot prediction (3) passes an image through the image encoder, computes its distribution over the prompted labels, and picks the top label, in the picture `A photo of a dog.` No training on the target task is needed, and the classes at test time can differ from any training label, unlike standard image classification with a fixed set of classes.

The main results are:
- CLIP is much more efficient at zero-shot transfer than an image-captioning baseline (Figure 2). Although highly expressive, a Transformer language model learns zero-shot ImageNet classification 3× slower than a baseline that predicts a bag-of-words encoding of the text. The contrastive objective of CLIP improves the efficiency by another 4×.
- Prompt engineering and ensembling (Figure 4) improve the zero-shot accuracy by almost 5 points on average across 36 datasets. The gain is similar to 4× more compute with the baseline zero-shot method, but free at inference.
- Zero-shot CLIP is competitive with a fully supervised baseline (Figure 5): a *linear probe*, which trains a linear classifier on the frozen features of a pretrained model, here a ResNet-50 trained on ImageNet. Zero-shot CLIP wins on 16 of 27 datasets, including ImageNet itself (+1.9). On STL10, where each class has few labeled examples and supervised learning is hard, zero-shot CLIP reaches 99.3%, a new state of the art without any training example. The largest gain over the linear probe is on Stanford Cars (+28.9). On specialized tasks and on MNIST, the linear probe wins.
- Zero-shot CLIP matches the average performance of a 4-shot linear classifier trained on its own feature space, and nearly matches the best 16-shot linear classifier across publicly available models (Figure 6).
- To match zero-shot CLIP, a linear classifier on the CLIP features needs from less than 1 to 184 labeled examples per class, with a median of 5.4 (Figure 7).
- The error decreases as the compute and the model size grow (Figure 9). The individual results are noisy, so the conclusions come from the average of all the experiments.
- With linear probes, the best CLIP models beat the state-of-the-art computer vision models, on average across 12 and 27 datasets (Figure 10).
- Models trained on ImageNet lose much accuracy when the dataset changes but the labels do not, and the loss grows as the distribution moves further from ImageNet (ImageNet sketches, adversarial images, etc.). Zero-shot CLIP reduces this robustness gap by up to 75% (Figure 13).
- A linear probe on top of CLIP for one dataset increases its accuracy by 9.2% on ImageNet, but slightly reduces the accuracy on the distribution-shifted datasets (Figure 14).
- The label set matters: adding the class `child` greatly reduces the share of images of people under 20 classified into crime-related or non-human categories (Table 7).

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

The method is also described in a [Google blog post](https://blog.research.google/2021/12/general-and-scalable-parallelization.html).

GSPMD (General and Scalable Parallelization for ML Computation Graphs) is an open-source, automatic, compiler-based parallelization system, built on the [XLA compiler](https://www.tensorflow.org/xla). Different model architectures suit different parallelization strategies, so GSPMD supports a large variety of parallelism algorithms, for example data parallelism for small models, pipeline parallelism for larger models, or a combination of both.

Each tensor gets a sharding property, either from the user, as initial annotations, or from the sharding completion pass. The sharding property specifies how the data is distributed across the devices. GSPMD defines three types of sharding:
- replicated: all devices hold the same full data;
- tiled: the tensor is split into tiles, without duplication;
- partially tiled, an extension of [GShard](https://arxiv.org/abs/2006.16668): the devices are divided into subgroups, and the tensor is tiled across the subgroups and replicated within each subgroup.

The user sets the shardings with `mesh_split(tensor, device_mesh, dims_mapping)`, which splits a tensor across the device mesh, with a mapping from each dimension $$i$$ of the tensor to an optional dimension of the device mesh. This API expresses all types of sharding, across the batch, feature, channel or other dimensions. The automatic partitioner of GSPMD is a set of compiler passes in XLA (Section 3.5). It uses information about the operators (for example, that a dot product is a generalized matrix multiplication), and iterative methods that refine the shardings over the iterations. GSPMD reaches 50% to 62% compute utilization on 128 to 2048 Cloud TPUv3 cores, for models with up to one trillion parameters. It produces a single program for all the devices ([SPMD](https://en.wikipedia.org/wiki/Single_program,_multiple_data)), so its compilation time stays constant as the number of devices grows.

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

The paper argues that neither convolutions (CNNs) nor attention (Transformers) are necessary for computer vision. It presents MLP-Mixer, an architecture based only on multi-layer perceptrons (MLPs). "MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information)." Its cost is linear in the number of patches, while the self-attention of ViT is quadratic. Pre-trained on large datasets, or trained with modern regularization, MLP-Mixer is competitive with existing methods, at a comparable pre-training and inference cost. For example, it reaches 87.9% top-1 accuracy on ImageNet after pre-training on JFT-300M.

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

The paper introduces gMLP (gated MLPs) and shows that they can perform as well as Transformers in language and vision applications. It claims that "self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy". On BERT, gMLP matches Transformers in pretraining perplexity and is better on some downstream tasks. On the finetuning tasks where it is worse, a substantially larger gMLP model closes the gap.

gMLP has no self-attention. It relies instead on channel projections and spatial projections with static parameters. It is a stack of $$L$$ blocks of identical size and structure, and each block computes, without the shortcuts and normalizations,

$$
Z = \sigma(XU), \,\,\,\,\,\,\,\, \tilde{Z} = s(Z), \,\,\,\,\,\,\,\, Y = \tilde{Z} V
$$

where $$\sigma$$ is an activation function, $$U$$ and $$V$$ are linear projections along the channel dimension, and $$s(\cdot)$$ is a layer that captures spatial interactions. When $$s$$ is the identity, the block is a regular FFN, without communication across tokens. In gMLP, $$s(\cdot)$$ is the Spatial Gating Unit (Section 2.1). It splits $$Z$$ along the channels into $$(Z_1, Z_2)$$ and computes $$s(Z) = Z_1 \odot (W Z_2 + b)$$, where $$W \in \mathbb{R}^{n \times n}$$ is a linear projection across the $$n$$ tokens (the spatial dimension). $$W$$ starts close to 0 and $$b$$ at 1, so that each block starts close to an FFN. The positions are captured in $$s(\cdot)$$, so, unlike Transformers, gMLP needs no position embeddings. The weights $$W$$ are static and of size $$n \times n$$, so the number of tokens is fixed by the model, and the spatial projection costs $$O(n^2)$$ per channel, as attention does.

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

Vision Transformers (ViTs) apply the transformer architecture to images. The input of the transformer is a sequence of linear embeddings of image patches, for example of 16×16 pixels. This also avoids the quadratic cost of letting every pixel attend to every other pixel: a 224×224 image gives $$224^2/16^2 = 196$$ tokens instead of 50,176 pixels. Pre-trained on large datasets (ImageNet-21k, JFT-300M), ViT matches or beats state-of-the-art CNNs (ResNets) in image classification, with substantially fewer computational resources for training. On mid-sized datasets, such as ImageNet alone, it underperforms comparable ResNets, because it lacks the inductive biases of convolutions (locality, translation equivariance).

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

The paper presents a simple method that improves the zero-shot learning of language models. *Instruction tuning*, the finetuning of a language model on a collection of more than 60 NLP datasets described with natural language instructions, substantially improves the zero-shot performance on unseen tasks. For each dataset, the authors write ten unique templates that describe its task with natural language instructions. The resulting 137B model, FLAN, surpasses zero-shot 175B GPT-3 on 20 of the 25 evaluated datasets, and even beats few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA and StoryCloze.

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

The paper measures empirical scaling laws for the cross-entropy loss of language models. Simple equations give the dependence of overfitting on the model and dataset sizes, and the dependence of the training speed on the model size. These relations give the optimal allocation of a fixed compute budget.

{: .picture}
<img loading="lazy" width="74%" height="74%" src="{{ site.assets }}/publications/scaling_laws.svg"/>

{: .caption}
Language modeling performance improves smoothly as the model size, dataset size, and amount of compute used for training increase. For optimal performance all three factors must be scaled up in tandem. Empirical performance has a power-law relationship with each individual factor when not bottlenecked by the other two. Adapted from: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

**Keypoints:**
- Performance depends most strongly on scale, which has three factors: the number of model parameters $$N$$ (without the embeddings), the size of the dataset $$D$$, and the compute $$C$$ used for training. The loss has a **power-law** relation with each of the three factors, with some trends spanning more than seven orders of magnitude (Figure 1): $$L(N) = (N_c/N)^{\alpha_N}$$ with $$\alpha_N \approx 0.076$$, $$L(D) = (D_c/D)^{\alpha_D}$$ with $$\alpha_D \approx 0.095$$, and $$L(C) = (C_c/C)^{\alpha_C}$$ with $$\alpha_C \approx 0.050$$, for compute-efficient training.
- Within reasonable limits, performance depends very weakly on other architectural hyperparameters, such as depth vs. width.
- Performance improves predictably as long as $$N$$ and $$D$$ grow together, but has diminishing returns if one of them is fixed while the other grows.
- On text with a different distribution than the training data, the results correlate strongly with those on the validation set of the training data, with a roughly constant offset in the loss. The transfer costs a constant penalty, and improves in line with the training performance.
- Larger models are significantly more sample-efficient. With a fixed compute budget $$C$$ and no other limit on $$N$$ or $$D$$, the optimal performance comes from training very large models on a relatively modest amount of data, and stopping significantly before convergence. The optimal model size grows as $$N \propto C^{0.73}$$, and the data only as $$D \propto C^{0.27}$$. Chinchilla (see its entry) later revised this allocation: the parameters and the training tokens should grow in equal proportions.
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

Before GPT-3, the large gains on many NLP tasks came from pre-training on a large corpus of text, followed by fine-tuning on a specific task. Fine-tuning still needs a task-specific dataset of thousands or tens of thousands of examples. The paper shows that scaling up language models greatly improves the task-agnostic few-shot performance, sometimes up to the level of earlier state-of-the-art fine-tuning approaches. It tests GPT-3, an autoregressive LLM with 175 billion parameters, 10× more than any earlier non-sparse language model, in the few-shot setting.

**GPT-3 architecture.** GPT-3 uses the model and architecture of [GPT-2](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf), including its modified initialization, pre-normalization and reversible tokenization. The only difference is that the layers of the transformer alternate dense and locally banded sparse attention patterns, as in the [Sparse Transformer](https://arxiv.org/abs/1904.10509).
- Table 2.1 lists the 8 GPT-3 models and their sizes and hyperparameters.
- Fig. 2.2 shows the total compute used during training. Following the analysis of [Scaling Laws For Neural Language Models](https://arxiv.org/abs/2001.08361), GPT-3 trains much larger models on many fewer tokens than was typical: all the models train on 300B tokens.
- Fig. 3.1 shows that the performance scales smoothly with compute: the cross-entropy loss follows a power law of the training compute.

**Background (Fig. 2.1):**
- Fine-Tuning (FT), the most common approach, updates the weights of a pre-trained model by training it on a supervised dataset specific to the task.
- Few-Shot (FS) gives the model a few demonstrations of the task at inference time, as conditioning, but allows no weight updates.
- One-Shot (1S) is the same as few-shot, with only one demonstration, in addition to a natural language description of the task.
- Zero-Shot (0S) is the same as one-shot, with no demonstration: the model gets only a natural language instruction that describes the task.

{: .picture}
<img loading="lazy" width="86.7%" height="86.7%" src="{{ site.assets }}/publications/gpt3_fig21.svg"/> 

{: .caption}
<b>Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning</b>. The panels above show four methods for performing a task with a language model – fine-tuning is the traditional method, whereas zero-, one-, and few-shot, the settings studied in the paper, require the model to perform the task with only forward passes at test time. In the few-shot setting the model is typically given a few dozen examples. Exact phrasings for all task descriptions, examples and prompts are in Appendix G of the paper. Adapted from: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

{: .picture}
<img loading="lazy" width="97.3%" height="97.3%" src="{{ site.assets }}/publications/gpt3_fig11.svg"/> 

{: .caption}
<b>Language model meta-learning.</b> During unsupervised pre-training, a language model develops a broad set of skills and pattern recognition abilities. It then uses these abilities at inference time to rapidly adapt to or recognize the desired task. The term “in-context learning” describes the inner loop of this process, which occurs within the forward-pass upon each sequence. The sequences in this diagram are not intended to be representative of the data a model would see during pre-training, but are intended to show that there are sometimes repeated sub-tasks embedded within a single sequence. Adapted from: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

**Tasks and performance:**
- On NLP tasks, GPT-3 gives promising results in the zero-shot and one-shot settings, and in the few-shot setting it is sometimes competitive with the state of the art, or even surpasses it.
- It also shows one-shot and few-shot proficiency at tasks that test rapid adaptation or on-the-fly reasoning, such as unscrambling words, performing arithmetic, and using novel words in a sentence after seeing them defined only once.
- Figures 3.3 to 3.12 show that the performance of GPT-3 grows with the model size, which suggests that language models keep absorbing knowledge as their capacity increases. The results cover TriviaQA, translation, the [Winograd Schema Challenge](https://arxiv.org/abs/1907.10641), PIQA, reading comprehension, SuperGLUE, ANLI Round 3, arithmetic, word scrambling and SAT tasks, in the zero-, one- and few-shot settings.
- Fig. 3.13 shows that the ability of people to identify model-generated news articles (the ratio of correct assignments to non-neutral assignments) decreases as the model size increases.
- Fig. 4.2 gives the analysis of benchmark contamination. Data contamination has a minimal effect on the performance of GPT-3 on most datasets, but a few datasets may have inflated results (PIQA and Winograd, marked with an asterisk).
- Chapter 5 details the limitations. GPT-3 struggles with natural language inference tasks, such as the ANLI dataset, and with some reading comprehension datasets, such as RACE or QuAC.

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

Most [graph neural networks](https://en.wikipedia.org/wiki/Graph_neural_network) (GNNs) assume a fixed and homogeneous graph structure, with similar types of nodes and edges. Graph Transformer Networks (GTNs) generate new graph structures: they find useful connections between nodes that are not connected in the original graph, and learn node representations on the new graphs, end to end. The Graph Transformer (GT) layer, the core layer of GTNs, learns a soft selection of edge types and composite relations, which generates useful multi-hop connections.
- GTNs generate meta-paths. A meta-path is a composite relation over node and edge types, a sequence of relations: $$R = t_1 \circ t_2 \circ \cdots \circ t_l$$ defines a composite relation between the nodes $$v_1$$ and $$v_{l+1}$$. Each GT layer computes two soft selections $$Q_1$$ and $$Q_2$$ of the adjacency matrices of the edge types, with softmax weights. Their product $$Q_1 Q_2$$ is the adjacency matrix of a meta-path of length 2. Stacking $$l$$ GT layers gives meta-paths of length $$l+1$$.
- Without any predefined, domain-specific meta-paths, GTNs reach state-of-the-art node classification on three heterogeneous graph benchmarks (DBLP, ACM, IMDB).

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

RMSNorm (Root Mean Square Normalization) is a simpler and computationally cheaper alternative to LayerNorm. Instead of normalizing with the mean and the variance of the input features, it only scales the input by the root mean square of the features. It has two benefits. (1) Efficiency: RMSNorm costs less than LayerNorm, and large models such as LLaMA use it. (2) Per the hypothesis of the paper, the re-centering invariance of LayerNorm is dispensable, and its re-scaling invariance is what matters for training stability. In the experiments, RMSNorm reaches an accuracy comparable to LayerNorm and reduces the running time by 7%–64% across models. It computes

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}
$$

where $$x$$ is the input vector of dimension $$d$$, and

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma
$$

where $$\gamma \in \mathbb{R}^d$$ is a learnable gain, as in LayerNorm. LayerNorm computes $$\frac{x - \mu}{\sigma} \odot \gamma + \beta$$, with the mean $$\mu$$ and the standard deviation $$\sigma$$ of $$x$$. RMSNorm drops $$\mu$$ and the bias $$\beta$$, and computes one statistic instead of two.

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

Training a Transformer is efficient because it parallelizes across the length dimension. Decoding, where this parallelism is impossible, is slow, because it loads the large key and value tensors from memory again at every step. Multi-query attention (MQA) is a variant of multi-head attention (MHA) that speeds up decoding. MHA has several attention heads in parallel, with different linear projections of the queries, keys, values and outputs. MQA is identical, except that all the heads share a single set of keys and values. This reduces the size of these tensors by the number of heads, and hence the memory bandwidth of incremental decoding. Decoding becomes much faster, with a minor loss of quality compared with the baseline. Grouped-query attention (GQA), used in Llama 2 (see its entry), shares one set of keys and values per group of heads, and sits between MHA and MQA in memory and in quality.

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

The paper introduces several **sparse factorizations of the attention matrix**, which reduce the quadratic cost of Transformers in memory and runtime to $$O(n \sqrt{n})$$, and allow longer sequences. They split the full attention into several faster attention operations that, combined, **approximate the dense attention**. The authors claim that sparsity in attention is a natural pattern. By visual inspection, most layers have sparse attention patterns across most data points, so sparse attention should not hurt the performance much. Other layers, however, show global patterns and data-dependent sparsity, where sparse attention could hurt the performance. With the same architecture, the paper sets new state-of-the-art density-modeling results on Enwik8, CIFAR-10 and ImageNet-64, with hundreds of layers, on sequences tens of thousands of timesteps long.

**Factorized self-attention** has $$p$$ separate attention heads, and each head attends to a subset $$A$$ of the indices. The hard problem is to find efficient choices for the subsets. Section 4.3 details two 2D factorizations (figure below). With a stride $$l \approx \sqrt{n}$$, the *strided* pattern has one head attend to the previous $$l$$ positions and the other to every $$l$$-th position. The *fixed* pattern has one head attend within blocks of $$l$$ positions, and the other to a few summary positions at the end of each block.

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

ALBERT ("A Lite BERT") lowers the memory consumption and increases the training speed of BERT. It scales better, and sets new records on several benchmarks (GLUE, RACE, SQuAD) with fewer parameters than BERT-large. An ALBERT configuration similar to BERT-large has 18× fewer parameters and trains about 1.7× faster. It introduces three techniques:
1. **Factorized embedding parameterization**, for parameter reduction. Instead of projecting the one-hot vectors of a vocabulary of size $$V$$ directly into the hidden space of size $$H$$, ALBERT projects them into a lower-dimensional embedding space of size $$E$$, and then into the hidden space. This reduces the embedding parameters from $$O(V \times H)$$ to $$O(V \times E + E \times H)$$: with $$V = 30{,}000$$, $$E = 128$$ and $$H = 4096$$, from about 123M to about 4.4M. The hidden size can then grow without growing the vocabulary embeddings.
2. **Cross-layer parameter sharing**, for parameter reduction. A BERT model with 12 layers learns unique parameters for each of its 12 Transformer encoder blocks. ALBERT learns the parameters of a single block and reuses them in all 12 layers, so the number of parameters does not grow with the depth of the network. The compute per token stays the same: only the parameter count shrinks. The authors report that the parameter reduction also acts as a regularization, which stabilizes the training and helps generalization. It does not improve the accuracy, however: "This approach slightly diminishes the accuracy, but the more compact size is well worth the tradeoff".
3. **Self-supervised loss for sentence-order prediction (SOP)**, for better performance. BERT uses an additional loss, next-sentence prediction (NSP), a binary classification of whether two segments appear consecutively in the original text. SOP replaces it, to address the ineffectiveness of NSP, and focuses on inter-sentence coherence. Its positive examples are as in BERT, two consecutive segments of the same document, and its negative examples are the same two segments with their order swapped. This forces the model to learn finer-grained distinctions about discourse-level coherence.

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

Group Normalization (GN) divides the channels into groups and normalizes the features within each group, so its computation does not depend on the batch size. With $$G$$ groups, GN normalizes each sample over $$C/G$$ channels and the spatial axes $$(H, W)$$. With $$G = 1$$ it is Layer Normalization, and with $$G = C$$ it is Instance Normalization; the paper uses $$G = 32$$. GN outperforms Batch Normalization (BN) in particular at small batch sizes, where the error of BN grows quickly. On ResNet-50 trained on ImageNet with a batch size of 2, GN has a 10.6% lower error than BN. At typical batch sizes, the two are comparable. Layer Normalization and Instance Normalization also avoid normalizing along the batch dimension. They are effective for training sequential models (RNN/LSTM) and generative models (GANs), but both have limited success in visual recognition, where GN gives better results.

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

The Vector Quantised Variational AutoEncoder (VQ-VAE) learns **discrete (not continuous) latent representations**, without supervision. It differs from VAEs in two ways: the encoder outputs discrete codes instead of continuous ones, and the prior is learned instead of static. In the forward pass, the decoder gets the nearest codebook embedding $$z_q(x)$$ of the encoder output $$z_e(x)$$ (Eq. 2). In the backward pass, the gradient $$\nabla_z L$$ passes unchanged to the encoder (straight-through), across the non-differentiable mapping from $$z_e(x)$$ to $$z_q(x)$$. The output of the encoder and the input of the decoder share the same $$D$$-dimensional space, so this gradient tells the encoder how to change its output to lower the reconstruction loss. The loss (Eq. 3) has three terms: the reconstruction loss, a codebook loss that moves the embeddings $$e$$ towards the encoder outputs, and a commitment loss that keeps the encoder committed to an embedding:

$$
L = -\log p(x \mid z_q(x)) + \lVert \mathrm{sg}[z_e(x)] - e \rVert_2^2 + \beta \lVert z_e(x) - \mathrm{sg}[e] \rVert_2^2,
$$

where $$\mathrm{sg}$$ is the stop-gradient operator, and $$\beta = 0.25$$ in the experiments.

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

Mixed precision training trains with half-precision (FP16) floating-point numbers, without a loss of model accuracy and without changes to the hyperparameters. FP16 represents magnitudes only from about $$6 \times 10^{-8}$$ ($$2^{-24}$$, subnormal) to 65,504, a much narrower range than FP32, so three techniques prevent the loss of critical information (underflows and overflows):
1. An FP32 master copy of the weights accumulates the gradients at each optimizer step. It is rounded to FP16 for the forward and backward passes.
2. Loss scaling preserves the gradients of small magnitude. The loss of the forward pass is scaled before backpropagation, which, by the chain rule, shifts the gradients into the range of FP16. The weight gradients are unscaled before the weight update, so that the updates keep the magnitudes of FP32 training.
3. FP16 arithmetic accumulates into FP32 outputs, which are converted to FP16 before they are stored to memory. Vector dot products, reductions and point-wise operations need different treatments.

The method matches the FP32 accuracy across a wide range of tasks (image classification and detection, speech recognition, machine translation, language modeling, GANs), and nearly halves the memory requirements. Formats with the 8-bit exponent of FP32, such as bfloat16, keep the range of FP32 and need no loss scaling.

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

The paper introduces the Graph Convolutional Network (GCN), with a simple layer-wise propagation rule, $$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$, where $$\tilde{A} = A + I$$ is the adjacency matrix with self-loops and $$\tilde{D}$$ its degree matrix. The rule is a first-order approximation of spectral graph convolutions. As CNNs do, GCNs learn features by aggregating information from neighboring nodes. CNNs operate on regular Euclidean structures, such as images, and GCNs generalize this to arbitrary graphs. $$\tilde{A}$$ is sparse, so a layer costs $$O(\vert E \vert C F)$$ for $$\vert E \vert$$ edges, $$C$$ input channels and $$F$$ output features. In semi-supervised node classification, on citation networks (Citeseer, Cora, Pubmed) and on a knowledge graph (NELL), GCN outperforms related methods by a significant margin.

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

Batch Normalization (BatchNorm, BN) improves the training of neural networks by normalizing the inputs of each layer over a mini-batch. It "reduces the internal covariate shift". *Covariate shift* means that the distribution of the features differs between the training and the test data, which breaks the i.i.d. assumption of most of ML. *Internal* covariate shift is the same drift inside the network: as the weights are updated, the distribution of the outputs of each layer changes. The higher layers must adapt to this drift, which slows down learning. BN normalizes each feature $$x$$ to zero mean and unit variance over the mini-batch, $$\hat{x} = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$$, and then scales and shifts it with learned parameters, $$y = \gamma \hat{x} + \beta$$. This allows higher learning rates. At inference, BN uses moving averages of the mean and the variance instead of the statistics of the batch. In the results, BN reaches the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. An ensemble of BN networks reaches a top-5 validation error of 4.9% on ImageNet, better than human raters. Later work, such as Santurkar et al. (2018), argues that BN helps mostly by smoothing the optimization landscape, not by reducing the internal covariate shift.

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

The paper uses **siamese neural networks** (described in the contrastive loss entry) for efficient **one-shot learning**. The strategy has two steps: (1) train a model to discriminate between pairs of same and different classes; (2) evaluate new categories with the learned feature mappings, for verification. Each twin of the siamese network is a convolutional neural network, followed by a flattening and a feed-forward network. The two twins share their weights, and the prediction is a sigmoid of a weighted L1 distance between their feature vectors $$h_1$$ and $$h_2$$: $$p = \sigma(\sum_j \alpha_j d_j)$$, with $$d_j = \vert h_{1,j} - h_{2,j} \vert$$. The loss function is a **binary cross-entropy** with a regularizer. On Omniglot 20-way one-shot classification, the network reaches 92% accuracy.

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

The paper improves the RNN encoder–decoder for translation (Cho et al., 2014; concurrent with [Sequence to Sequence Learning with Neural Networks (Google, NeurIPS 2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)), which compresses the whole source sentence into a single fixed-length vector. It introduces attention: at each decoding step, the decoder learns to (soft-)align to the hidden states of every encoder step, not only the last one, which removes the bottleneck of the fixed-length vector. For the target word $$y_t$$, the context is $$c_t = \sum_{j=1}^{T} \alpha_{tj} h_j$$, a weighted sum of the encoder states $$h_j$$ of the $$T$$ source words. The weights $$\alpha_{tj}$$ are the softmax over $$j$$ of the scores $$e_{tj} = v_a^\top \tanh(W_a s_{t-1} + U_a h_j)$$, where $$s_{t-1}$$ is the previous decoder state. Each target word thus costs $$O(T)$$ instead of $$O(1)$$. The model reaches a translation performance comparable to the existing phrase-based system on English-to-French, and degrades much less on long sentences.

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

A spatial transformer is a differentiable module that can be inserted anywhere in a CNN, to transform feature maps spatially, conditioned on the input, and learned without extra supervision. A localisation network regresses the parameters of the transformation (for example affine, projective or thin-plate spline), a grid generator produces the sampling grid, and a differentiable (for example bilinear) sampler warps the input. For an affine transformation, the localisation network outputs the 6 parameters of a matrix $$A_\theta \in \mathbb{R}^{2 \times 3}$$, and each output pixel at $$(x^t, y^t)$$ samples the input at $$(x^s, y^s)^\top = A_\theta (x^t, y^t, 1)^\top$$. The network learns invariance to translation, scale, rotation and more generic warping, which gave state-of-the-art results at the time on distorted MNIST, Street View House Numbers and CUB-200-2011 birds.

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

A deeply supervised model supervises its intermediate layers explicitly during training, in addition to its final output layer. The earlier layers then learn features that are useful for the task directly, instead of relying only on the gradients propagated from the final layer. Deep supervision addresses vanishing gradients and poor feature learning in the intermediate layers. It is common in image segmentation, object detection and biomedical image analysis.

The objective of each intermediate layer is a *companion objective*, an additional constraint, or a new regularization, of the learning. It is a hinge loss with a threshold $$\gamma$$ (a hyperparameter). Once the companion loss of a hidden layer reaches $$\gamma$$ or less, it vanishes and no longer plays a role in the learning. Empirically, the companion objective (1) acts as a kind of **feature regularization**, which reduces the test error significantly but not necessarily the training error; and (2) gives a faster convergence, in particular with little training data. Deeply-supervised Nets gave state-of-the-art results at the time on MNIST, CIFAR-10, CIFAR-100 and SVHN.

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

Dropout **randomly drops neurons, in different layers, during training: each one with probability $$p$$** (the paper uses $$p$$ for the probability of keeping a neuron). Each training minibatch samples a new "thinned" network. Max-norm regularization, a decaying learning rate and a high momentum improve dropout further. **At test time, all neurons are used, with their outgoing weights multiplied by $$1 - p$$**, which approximates the average of the exponentially many thinned networks. Dropout **reduces overfitting**: the network can never rely on any given activation, so it learns redundant ways of solving the task with several neurons, and its hidden activations also become sparser. Dropping 20% of the input units and 50% of the hidden units ($$p = 0.2$$ and $$p = 0.5$$) was often optimal in the paper. Dropout costs less than the averaging of several trained DNNs. A network with dropout, however, takes 2 to 3 times longer to train than the same network without it, because the parameter updates are very noisy. Fully connected layers hold most of the parameters and are the most prone to overfitting, so dropout **improves the generalization** of the model there.

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

The paper presents Dimensionality Reduction by Learning an Invariant Mapping (DrLIM): it learns a function that maps high-dimensional inputs to low-dimensional outputs, given the neighborhood relations between the samples in the input space. It introduces the **contrastive loss function**, which trains a **siamese network** to learn a representation space where similar samples are close together and dissimilar samples are far apart. A siamese network compares two inputs by learning their similarity. It has two identical subnetworks (hence the name "siamese"), with the same architecture and the same weights. Each subnetwork processes one of the two inputs, and a cost module combines their outputs into a similarity score or a distance. The total gradient is the sum of the contributions of the two subnetworks.

The input of the system is a pair of images, one for each subnetwork, and a similarity label $$y$$: 1 for similar images and 0 for dissimilar ones (the paper uses the opposite convention). The subnetworks map the images to $$G_W(X_1)$$ and $$G_W(X_2)$$, and the cost module computes their Euclidean distance $$D_W = \lVert G_W(X_1) - G_W(X_2) \rVert_2$$. The loss is

$$
L = \frac{1}{2} \, y \, D_W^2 + \frac{1}{2} \, (1 - y) \max(0, m - D_W)^2,
$$

where the margin $$m$$ is a hyperparameter that sets the minimum distance of dissimilar pairs.

</div>
</details>
</td>
</tr>
</table>

{::options parse_block_html="false" /}
