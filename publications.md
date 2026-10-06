---
layout: post
title: Publications bookmark
permalink: /publications/
---

<style>
details { margin-bottom: 0.5em; }
</style>

A summary of some interesting publications I came across. Continuously updated. Click $$\small{\blacktriangleright}$$ to expand.

{::options parse_block_html="true" /}

<details> <summary markdown="span">2026 [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, DeepSeek-AI](https://arxiv.org/abs/2606.19348)</summary>

DeepSeek-V4 is a preview of DeepSeek's next MoE series: **DeepSeek-V4-Pro** (1.6T parameters, 49B activated, pre-trained on 33T tokens) and **DeepSeek-V4-Flash** (284B parameters, 13B activated, 32T tokens), both with a native one-million-token context. It keeps the DeepSeekMoE framework and the MTP modules of DeepSeek-V3, with minor MoE changes (a Sqrt(Softplus) affinity function instead of Sigmoid, no limit on the number of nodes a token is routed to, and hash-routed MoE layers replacing the first dense FFN layers), and introduces three upgrades:

1. **Hybrid attention with CSA and HCA**, interleaved across layers. *Compressed Sparse Attention* (CSA) compresses the KV entries of every $$m$$ tokens into one entry (a learned, softmax-weighted combination over overlapping windows) and then applies DeepSeek Sparse Attention over the compressed entries: the lightning indexer picks the top-k compressed entries for each query, and a small sliding window of recent uncompressed tokens is added for local detail. *Heavily Compressed Attention* (HCA) compresses every $$m' \gg m$$ tokens into one entry and keeps dense attention over them. Other details include a shared-KV (MQA-style) layout with grouped output projection, partial RoPE and attention sinks.
2. **mHC** (Manifold-Constrained Hyper-Connections, see its entry) instead of plain residual connections: the residual stream is widened to $$n_{hc}$$ parallel streams whose mixing matrix is projected onto the doubly stochastic matrices with Sinkhorn-Knopp iterations, which keeps signal propagation stable.
3. The **Muon** optimizer, for faster convergence and more stable training.

The headline is efficiency: at 1M tokens of context, V4-Pro needs only 27% of the single-token inference FLOPs and 10% of the KV cache of DeepSeek-V3.2 (V4-Flash: 10% and 7%), and the routed-expert weights are stored in FP4 (with FP4 quantization-aware training during post-training). The infrastructure is equally relevant to this page: a single fused MoE mega-kernel that overlaps computation, communication and memory access in expert parallelism, kernels written in TileLang, batch-invariant and deterministic kernels for bitwise reproducibility, two-stage context parallelism for the compressed attention, and a heterogeneous KV cache with on-disk storage for shared-prefix reuse. Post-training trains per-domain specialists (SFT, then GRPO RL) and merges them into one model with on-policy distillation. The maximum-reasoning mode, V4-Pro-Max, is reported as the best open model at release: ahead of GPT-5.2 and Gemini-3.0-Pro on standard reasoning benchmarks, slightly behind GPT-5.4 and Gemini-3.1-Pro.

</details>


<details> <summary markdown="span">2026 [Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning, NVIDIA](https://arxiv.org/abs/2604.12374)</summary>

The technical report of Nemotron 3 Super, a 120B-parameter (12B active) hybrid Mamba-2/attention MoE model, and the first model of the Nemotron 3 family (see the Nemotron 3 white-paper entry) to be **pre-trained in NVFP4**, to use **LatentMoE** experts, and to include **MTP layers** for native speculative decoding. It was pre-trained on 25T tokens and then post-trained with SFT and RL, with a strong emphasis on agentic capabilities.

The final model supports up to 1M tokens of context and reaches accuracy comparable to GPT-OSS-120B and Qwen3.5-122B on common benchmarks, while delivering up to 2.2× and 7.5× higher inference throughput than them, respectively. The datasets and the base, post-trained and quantized (NVFP4, FP8, BF16) checkpoints are open. For this page, it is the large-scale validation of the LatentMoE and NVFP4 entries: the accuracy-per-byte expert design and 4-bit pre-training both hold up on a 120B model trained on 25T tokens.

</details>


<details> <summary markdown="span">2026 [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling, Zadouri, ..., Tri Dao (Princeton et al.), MLSys 2026](https://arxiv.org/abs/2603.05451)</summary>

FlashAttention-4 (FA4) re-designs the attention kernel for NVIDIA Blackwell (B200/GB200) to confront what the authors call **asymmetric hardware scaling**: from Hopper to Blackwell, dense BF16 tensor-core (matmul) throughput jumped from ~1 to ~2.25 petaFLOPS, but the units that do everything else—the exponential unit for softmax, shared-memory bandwidth—did not speed up proportionally.  The consequence is a bottleneck flip from earlier FlashAttention versions: on Blackwell the tensor cores got much faster but the exponential unit (MUFU.EX2) did not, so softmax is no longer "just the thing between the two matmuls"—it becomes the bottleneck that must be carefully pipelined.  FA4's whole job is to keep those over-provisioned tensor cores busy by overlapping the matmuls with the now-relatively-slow softmax and memory work.

The co-design has a few moving parts. New forward and backward software pipelines exploit Blackwell's fully asynchronous MMA and larger tile sizes to overlap tensor cores, the softmax exponential, and memory operations.  To beat the exponential bottleneck specifically, the forward pass emulates the exponential in software via a polynomial approximation on the FMA units, plus conditional (selective) online-softmax rescaling that only rescales when a new max actually shifts the result enough to matter.  On the backward pass, where shared-memory traffic dominates, FA4 stores intermediate results in Blackwell's new tensor memory (TMEM) to relieve shared-memory traffic, and uses the new 2-CTA MMA mode to cut shared-memory traffic further and halve the number of global atomic adds.  It's also written entirely in **CuTe-DSL** (Python), giving 20–30× faster compile times than C++ template approaches while keeping full expressivity,  which lowers the barrier to prototyping new attention variants.

Results: up to 1613 TFLOPs/s on B200 BF16 (71% utilization; the accompanying blog post quotes 1605), up to 1.3× faster than cuDNN 9.13 and 2.7× faster than Triton in the forward pass, and faster than all baselines in the backward pass at long sequence lengths. FlashAttention-3 is Hopper-only, so it is not a baseline here; several FA4 techniques were also upstreamed into cuDNN (9.13/9.14 onwards). This is the kernel-level companion to the systems papers below: where they reshape *what* runs where, FA4 squeezes the attention primitive itself to the hardware's asymmetric limits.

</details>


<details> <summary markdown="span">2026 [Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism, Arizona State University (ICML 2026)](https://arxiv.org/abs/2602.04870)</summary>

A follow-up to LatentMoE that targets the three limitations of Expert Parallelism (EP): communication volume and all-to-all latency grow linearly with the number of activated experts $$k$$ (each token is replicated $$k$$ times), expert load imbalance makes the all-to-all wait for the longest queue, and the data-dependent traffic needs an extra all-to-all to exchange metadata. LatentMoE reduces the bytes per token, but it still communicates *after* routing, so imbalance and non-determinism remain.

**Multi-Head LatentMoE** projects each token with a learned $$d \times d$$ matrix and splits it into $$N_h$$ sub-tokens, each processed by an independent small MoE with its own router and experts; the outputs are concatenated and projected back. **Head Parallel (HP)** exploits this structure by moving the all-to-all *before* routing: each GPU owns a subset of the heads, receives all sub-tokens for those heads, and does all routing and expert computation locally. Each token is sent exactly once, so communication is $$O(1)$$ in $$k$$; every GPU sends and receives the same amount of data, so traffic is perfectly balanced and needs no metadata exchange (and no out-of-memory risk from hot experts); and HP composes with EP to scale beyond $$N_h$$ GPUs. To avoid multiplying memory traffic by $$N_h$$, they add exact IO-aware kernels: an online top-k router that keeps the scores in SRAM (Triton), and expert computation expressed as block-sparse attention on top of FlexAttention.

At small scale (0.2B active, up to 4.2B total parameters, 10B FineWebEdu tokens), Multi-Head LatentMoE with HP trains up to 1.61× faster than a standard MoE with EP at identical quality and cuts inter-GPU traffic to 25% at $$k = 4$$; with doubled expert granularity it reaches higher accuracy while still being 1.11× faster.

</details>


<details> <summary markdown="span">2026 [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts, NVIDIA](https://arxiv.org/abs/2601.18089)</summary>

In NVIDIA's LatentMoE, "tokens are projected from the model hidden dimension d into a smaller latent dimension ℓ for expert routing and computation, which reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ. We use this efficiency to increase the total number of experts and the top-k active experts per token by the same factor d/ℓ". Here "routing" means token dispatch: the router's gating scores are still computed from the full d-dimensional token, and only the dispatch/combine traffic and the routed experts live in the latent space. The weighted sum of outputs (from the combine all-to-all) from all routed experts goes through the converse projection from the latent dimension back to the model hidden dimension, and is then summed with the output of the shared experts, which operate in the original hidden space (no dimensionality reduction). There's only a single down- and up-projection shared by all routed experts, while each expert's weights shrink by d/ℓ, reducing memory bandwidth. Their roofline analysis shows that expert computation is bound by HBM bandwidth (weight loading) at latency-critical batch sizes, and by all-to-all communication in throughput-oriented deployments. Both costs scale with d, whereas the number of active experts and the expert FFN width set the model's nonlinear budget, so d is the dimension to shrink.

The payoff is that the freed budget can be spent on *more, finer experts* at the same cost: with a 4× compression, against a standard MoE (128 experts, top-6) scoring 48.30% on MMLU-Pro, a LatentMoE configuration with (N', K') = (512, 22) (as reported in the Nemotron 3 white paper) reaches 52.87% on MMLU-Pro and 55.14% on code (+3.19%) at a similar parameter and FLOP count (8B-active / 73B-total hybrid models trained on 1T tokens), with measured per-GPU throughput within ~6% of the baseline at higher concurrency.  Validated by design-space exploration up to 95B parameters over a 1T-token horizon, LatentMoE consistently beats standard MoE on accuracy per FLOP and per parameter, and has been adopted by the flagship Nemotron-3 Super and Ultra models. 

<img loading="lazy" width="1758" height="1453" alt="image" src="https://github.com/user-attachments/assets/f6470954-b902-46d4-9cd9-dea0a60d040c" />

</details>




<details> <summary markdown="span">2025 [mHC: Manifold-Constrained Hyper-Connections, DeepSeek-AI (ICML 2026)](https://arxiv.org/abs/2512.24880)</summary>

Hyper-Connections (HC) widen the residual stream into $$n$$ parallel streams mixed by learnable matrices, decoupling the residual width from the layer width at little compute cost. The gains are real, but the unconstrained mixing breaks the identity-mapping property that makes deep residual networks trainable: in a 27B model, signals get amplified across layers and training shows loss spikes. HC also adds significant memory-access overhead.

mHC constrains the stream-mixing (residual) matrix of each layer to the manifold of **doubly stochastic matrices** (the Birkhoff polytope: non-negative, with rows and columns summing to 1) using iterative Sinkhorn-Knopp row/column normalization, and makes the read-in and write-out mappings non-negative and bounded with sigmoids. A doubly stochastic matrix has spectral norm at most 1 (the residual mixing is non-expansive) and products of such matrices stay doubly stochastic, so signal propagation stays stable at any depth; with $$n = 1$$ it reduces to the ordinary identity residual. On the systems side, kernel fusion, recomputation and overlapping with DualPipe communication bring the cost of a 4× wider residual stream ($$n = 4$$) down to about 6.7% extra training time.

In pre-training experiments on MoE models from 3B to 27B parameters, mHC trains stably where HC spikes and improves over the plain residual baseline, with the advantage holding across compute and token budgets. It was then adopted in DeepSeek-V4 (see its entry).

</details>


<details> <summary markdown="span">2025 [Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs, CMU et al. (OSDI 2026)](https://arxiv.org/abs/2512.22219)</summary>

MPK automates the megakernel idea from the Hazy Research Llama-1B post (see its entry) and comes from the same group as Mirage: it is the first compiler and runtime that automatically turns multi-GPU model inference into a single persistent mega-kernel, instead of writing one by hand. The key abstraction is an **SM-level task graph** that captures data dependencies at the granularity of individual streaming multiprocessors rather than whole kernels, which enables optimizations that the kernel-per-operator model cannot express, such as cross-operator software pipelining (starting the next operator's loads while the current one computes) and fine-grained overlap of computation with inter-GPU communication.

The MPK compiler lowers a tensor program into an optimized SM-level task graph and generates CUDA code for each task; an in-kernel parallel runtime then executes the tasks inside one persistent kernel with decentralized scheduling across SMs, so there are no kernel launches or host round-trips between operators. It is open source (in the Mirage repository) and reduces end-to-end inference latency by up to 1.7× compared to existing kernel-per-operator LLM serving systems, pushing LLM inference close to the limits of the hardware.

</details>


<details> <summary markdown="span">2025 [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models, DeepSeek-AI](https://arxiv.org/abs/2512.02556)</summary>

The formal report behind the DeepSeek-V3.2-Exp model card (see its entry). The architecture is the same as V3.2-Exp: the only change from DeepSeek-V3.1-Terminus is **DeepSeek Sparse Attention (DSA)**, added through continued training. A *lightning indexer* (a few heads, ReLU activation, runnable in FP8) scores each preceding token for the current query, and a fine-grained selection step keeps only the top-k key-value entries (k = 2048) for the main MLA attention, which reduces the core attention cost from $$O(L^2)$$ to $$O(Lk)$$ (the indexer itself is still quadratic, but much cheaper). The indexer is first trained alone to match the dense attention distribution (KL loss) while the main model is frozen, and then the whole model is trained with sparse attention.

The rest of the report is about post-training: a scaled and stabilized GRPO-based RL recipe with a post-training compute budget above 10% of the pre-training cost, and a large-scale agentic task-synthesis pipeline (over 1,800 environments and 85,000 complex instructions) that makes V3.2 the first DeepSeek model to integrate thinking directly into tool use. DeepSeek-V3.2 performs comparably to GPT-5, and the high-compute variant **DeepSeek-V3.2-Speciale** (longer reasoning, no tool use) reaches gold-medal level at IMO 2025, CMO 2025, IOI 2025 and the ICPC World Finals 2025, on par with Gemini-3.0-Pro in reasoning, though with clearly worse token efficiency.

</details>


<details> <summary markdown="span">2025 [Kimi Linear: An Expressive, Efficient Attention Architecture, Moonshot AI (Kimi Team)](https://arxiv.org/abs/2510.26692)</summary>

Kimi Linear is a hybrid linear-attention architecture that, under matched training recipes, *outperforms* full attention across short-context, long-context and RL-style post-training tasks, which the authors claim is a first. Its core is **Kimi Delta Attention (KDA)**, a linear attention with a fixed-size RNN state that extends Gated DeltaNet with finer-grained gating: instead of one forget gate per head, each channel (feature dimension) gets its own decay rate, so the limited state memory is used more selectively. For hardware efficiency, KDA's transition uses a specialized Diagonal-Plus-Low-Rank (DPLR) form with a bespoke chunkwise-parallel algorithm that needs substantially less computation than general DPLR while staying consistent with the classical delta rule.

The model interleaves KDA with periodic full-attention MLA layers in a uniform 3:1 ratio (48B total, 3B activated parameters). With an identical training recipe it beats a full-MLA baseline on all evaluated tasks, while cutting the KV cache by up to 75% (only one layer in four keeps a growing cache) and reaching up to 6× decoding throughput at 1M tokens of context. The KDA kernels, a vLLM implementation, and the pre-trained and instruction-tuned checkpoints are open.

</details>


<details> <summary markdown="span">2025 [Pretraining Large Language Models with NVFP4, NVIDIA](https://arxiv.org/abs/2509.25149)</summary>

The recipe for pre-training LLMs in 4-bit floating point with **NVFP4**, the Blackwell microscaling format: FP4 (E2M1) values in blocks of 16 elements, each block with an FP8 (E4M3) scale, plus a per-tensor FP32 scale. Compared to MXFP4 (blocks of 32 with power-of-two E8M0 scales), the smaller blocks and more precise scales capture the local dynamic range much better. FP4 GEMMs run at 2–3× the FP8 math throughput on Blackwell (GB200/GB300), but naive 4-bit training diverges or loses accuracy over long token horizons.

The method combines four ingredients: (1) **Random Hadamard Transforms** on the inputs of the weight-gradient GEMMs, which spread block-level outliers into a more Gaussian-like distribution; (2) **2D (16×16) block scaling for weights**, so the forward and backward passes see the same quantized weights; (3) **stochastic rounding** for gradients, to avoid the bias of round-to-nearest; and (4) keeping a small fraction of numerically sensitive linear layers in higher precision.

They validate it by training a 12B hybrid Mamba-Transformer on 10T tokens, the longest publicly documented 4-bit training run at the time: the loss tracks the FP8 baseline within about 1% relative error during the stable phase (slightly above 1.5% late in the learning-rate decay), and downstream accuracies match FP8. In a comparison on an 8B model, MXFP4 needed about 36% more tokens to reach the loss NVFP4 reached at 1T tokens. Support is in Transformer Engine, and NVFP4 pre-training was later used for Nemotron 3 Super (see its entry).

</details>


<details> <summary markdown="span">2025 [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures, DeepSeek-AI (ISCA 2025)](https://arxiv.org/abs/2505.09343)</summary>

A hardware/model co-design retrospective of DeepSeek-V3/R1 (ISCA 2025 industry track), complementary to the DeepSeek-V3 report on this page: how a 671B model was trained on only 2,048 H800 GPUs, and what hardware should change. On the model side it quantifies the choices: MLA cuts the KV cache to about 70 KB per token, versus 327 KB for Qwen-2.5 72B and 516 KB for LLaMA-3.1 405B (both GQA); the MoE design needs about 250 GFLOPs of training compute per token, versus about 2,448 for the dense LLaMA-3.1 405B; plus FP8 mixed-precision training and MTP for faster speculative decoding. On the infrastructure side it describes the **multi-plane two-layer fat-tree** network (each GPU's NIC sits on its own network plane, eight planes per node), which cuts cluster network cost while scaling to thousands of GPUs, and how expert parallelism and node-limited routing are shaped around the H800's bandwidth asymmetry (much faster NVLink than InfiniBand).

The second half is a wish list for future AI hardware, drawn from the bottlenecks they hit: more precise low-precision compute units (e.g., higher FP8 accumulation precision and native support for fine-grained block scaling), convergence of scale-up and scale-out networks, and low-latency communication fabrics. It is a good companion to the communication-overlap papers on this page because it explains *why* those optimizations were needed on bandwidth-limited hardware.

</details>


<details> <summary markdown="span">2025 [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention, DeepSeek-AI, Peking University & University of Washington (ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089)</summary>

NSA argues that most sparse-attention methods fail in practice for two reasons: their theoretical savings don't turn into wall-clock speedups (scattered token-level memory access, poor fit with GQA/MQA where heads share KV, or speedups limited to only prefill or only decode), and they are applied post hoc to models trained with dense attention, so the sparsity pattern is never trained end to end. NSA is designed for both: each query attends through three branches whose outputs are combined by learned gates: (1) **compressed** attention over coarse summaries of token blocks (blocks of 32 tokens with stride 16, compressed by a learned MLP), (2) **selected** attention over the top-n most relevant fine-grained blocks, chosen by reusing the compression branch's attention scores so that selection is almost free, and (3) a **sliding window** over the most recent 512 tokens for local context.

The kernel is hardware-aligned: block selection is shared across all heads of a GQA group, so each group loads contiguous KV blocks once into SRAM and keeps the arithmetic intensity balanced on Tensor Cores in the forward pass, the backward pass and decoding. A 27B-parameter MoE model (about 3B active) pre-trained from scratch with NSA on 260B tokens matches or beats the full-attention baseline on general benchmarks, long-context retrieval and chain-of-thought reasoning, while running up to 11.6× faster for decoding, 9.0× for the forward pass and 6.0× for the backward pass at 64K context. Its ideas (block compression plus learned selection) resurface in DeepSeek Sparse Attention (DeepSeek-V3.2) and in the CSA/HCA hybrid of DeepSeek-V4.

</details>


<details> <summary markdown="span">2025 [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving, University of Washington et al. (MLSys 2025)](https://arxiv.org/abs/2501.01005)</summary>

FlashInfer is the attention kernel library underneath many serving stacks (it is integrated into SGLang, vLLM and MLC-Engine). It addresses three properties of serving workloads that a single FlashAttention kernel doesn't cover. First, **KV-cache heterogeneity**: paged, prefix-shared and sparse KV layouts are all represented in one **block-sparse format**, with composable formats for shared prefixes, so the same kernels serve all of them and redundant loads of shared KV are avoided. Second, **customization**: a customizable attention template (e.g., custom masks or logits transforms) is JIT-compiled into optimized block-sparse kernels, so new attention variants don't need hand-written CUDA. Third, **dynamism**: a load-balanced scheduler splits the work of variable-length requests across SMs (inspired by stream-K) while staying compatible with CUDA Graphs, which require a static launch configuration.

Compared to state-of-the-art serving solutions, it achieves 29–69% lower inter-token latency than compiler backends on an LLM serving benchmark, 28–30% lower latency for long-context inference, and a 13–17% speedup for LLM serving with parallel generation.

</details>


<details> <summary markdown="span">2025 [NVIDIA Nemotron 3: Efficient and Open Intelligence, NVIDIA](https://arxiv.org/abs/2512.20856)</summary>

Nemotron 3 (NVIDIA) is a family of open models: Nano, Super and Ultra with a context of up to 1M tokens. 

The report stacks four efficiency techniques:
1. **LatentMoE** (Super/Ultra): see the LatentMoE entry.
2. **hybrid Mamba-Transformer MoE**, for inference efficiency: rather than interleaving MoE layers with expensive self-attention—which must attend over a linearly growing KV cache during generation—Nemotron 3 predominantly interleaves MoE layers with cheaper Mamba-2 layers, which require storing only a constant state during generation, keeping just a select few attention layers (without RoPE, since the Mamba layers already provide positional information).
3. **Multi-Token Prediction** (Super/Ultra): predicting several future tokens adds richer training signal (~2.4% average benchmark gain in an 8B-active MoE ablation) and doubles as a draft for speculative decoding—a lightweight MTP module reaches ~97% acceptance on the first two predicted tokens.  
4. **NVFP4 4-bit training** (Super/Ultra): stable, accurate pretraining on a hybrid Mamba-MoE for up to 25T tokens, with weights, activations, and gradients quantized to NVFP4 so the fprop, dgrad, and wgrad GEMMs all run natively in FP4 (3× peak throughput vs FP8 on GB300).  The recipe leans on 2D block scaling for weights, Random Hadamard Transforms on wgrad inputs, stochastic rounding on gradients, and keeping the last 15% of the network in high precision;  sensitive layers are protected too—QKV/attention projections (the few GQA layers use only 2 KV heads) stay in BF16, and Mamba output projections, which flush up to 40% of values to zero in NVFP4, are kept in MXFP8.  The relative loss gap stays under 1% vs BF16 on Nano and shrinks to under 0.6% on an 8B-active MoE, with the gap narrowing as model size grows.

The design philosophy is a balance of three layer types: MoE for sparse parameter scaling, a few attention layers for high-fidelity all-to-all information routing between tokens, and Mamba-2 for fixed-cost sequence modeling. Nemotron 3 Nano 30B-A3B achieves 3.3× higher throughput than Qwen3-30B-A3B (8K input / 16K output), while staying competitive on reasoning and long-context (RULER @ 1M) benchmarks. 

<img loading="lazy" width="1744" height="547" alt="image" src="https://github.com/user-attachments/assets/229ddc64-b770-4ccf-b82d-003722cae9ca" />

<img loading="lazy" width="2182" height="1350" alt="image" src="https://github.com/user-attachments/assets/8f3fae96-31fa-45c4-bcd6-0759bf265d8e" />

</details>

<details> <summary markdown="span">2025 [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts, ByteDance (MLSys 2025)](https://arxiv.org/abs/2502.19811)</summary>

The standard technique for **distributed MoE** efficiency is to pipeline the expert-parallel communication (which can take ~47% of the total execution time with popular models and frameworks) with computation. Comet argues existing schemes do it too coarsely: splitting an MoE layer into a few big chunks to expose overlap either leaves communication exposed or shrinks the GEMM tiles enough to hurt kernel efficiency (the same tension FLUX identifies for tensor parallelism).

Comet works in two parts. (1) Dependency resolving: it maps out which pieces of computation depend on which pieces of communication, then schedules them into a fine-grained pipeline so each can start the moment its data is ready. (2) Adaptive workload assignment: it splits GPU resources between comm and compute, and because the best split shifts with sequence length and parallelization strategy, it tunes that split per-configuration instead of fixing it.

Authors note that the division point of how many computation vs communication SMs to be allocated depends on the model layer, input shape and level of parallelism:

<img loading="lazy" width="1066" height="1024" alt="image" src="https://github.com/user-attachments/assets/98139ecb-ae00-439c-b598-4f431e47760e" />

"Comet’s library comprises multiple pre-compiled kernels, each with a distinct division point (of computation vs communication SM split). Prior to deployment, the optimal configuration for each setup is profiled and stored as metadata. During runtime, Comet utilizes this metadata to select the optimal kernel for execution." As an example, if you take the MoE workflow dispatch→MatMul→MatMul→combine, it fuses dispatch→MatMul into one kernel and MatMul→combine into another, and executes communication and computation at a tile level.

<img loading="lazy" width="1872" height="636" alt="image" src="https://github.com/user-attachments/assets/bd3a944f-fd78-421e-952f-c149d556d422" />
 
The mechanism on Hopper is thread-block specialization. Comm and compute run in separate thread blocks. One fused kernel launches a fixed set of thread blocks. Each block is specialized — it's either a comm block or a compute block, not both. Because thread blocks occupy SMs, this effectively dedicates some SM capacity to comm and the rest to compute. When you fuse comm and compute into one kernel with specialized thread blocks, the compute blocks call device-side GEMM functions inside the fused megakernel. The GEMM implementation (CUTLASS) is reused unchanged.

How does it "map out" pieces of computation to communication? The idea: between a communication op and the computation op that consumes (or produces) its data, there is a shared tensor — the buffer that comm writes into and compute reads from (dispatch), or that compute writes and comm reads (combine). Comet analyzes that buffer to figure out the fine-grained dependencies. Two steps:
- Decompose the shared tensor along a specific dimension.  
- Reschedule computation to match data arrival order. Now the key move, "Mapping out the dependencies" means: for each tile of the GEMM, determine which slice(s) of the shared tensor it needs, then order the GEMM's tile traversal to consume slices in the order communication fills them.

Results: a single MoE layer runs 1.96× faster and end-to-end execution 1.71× faster on average; Comet is deployed in production clusters of ten-thousand-GPU scale, saving millions of GPU hours.
</details>



<details> <summary markdown="span">2025 [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding, NVIDIA](https://arxiv.org/abs/2507.07120)</summary>

Helix Parallelism targets real-time decoding when the KV history reaches *multi-million tokens* under a tight token-to-token latency (TTL) budget. Two bottlenecks dominate this regime: reading the **FFN weights** every step, and streaming the enormous **KV cache** from DRAM. The problem is that the standard fix for one hurts the other. Tensor Parallelism (TP) handles FFN weight reads well, but it scales poorly for attention: once the TP width exceeds the number of KV heads, the KV cache has to be *duplicated* across GPUs, which wastes memory bandwidth, caps parallelism, and limits batch size. Meanwhile KV DRAM reads grow linearly with batch size. No single uniform sharding strategy relieves both pressures at once—you're forced to trade KV duplication against FFN load.

Helix Parallelism overcomes this by introducing a hybrid execution strategy that applies KV parallelism during attention to shard KV caches across GPUs, then reuses the same GPUs for TP in dense LLMs or TP×Expert Parallel (EP) in MoEs during FFN computation.

Helix changes the parallelism layout between the attention phase and the FFN phase, within a single decode step, reusing the same GPUs.

- Phase 1 — Attention, laid out as KV Parallelism (KVP), combined with tensor parallelism over the KV heads (TPA ≤ number of KV heads), so the pool has N = KVP × TPA GPUs. The KV cache is sharded along the sequence/token dimension. No GPU stores a duplicate of the KV cache. The algorithm is:
  - Every KVP rank computes the full QKV projection for the new token locally (duplicating this small projection is cheaper than gathering the query across GPUs).
  - Each GPU computes attention between that query and only its local slice of the KV cache — producing a partial attention output plus a log-sum-exp (softmax normalizer) per token, FlashDecoding-style.
  - A single all-to-all over the query-head dimension exchanges these partial results, and each GPU rescales and sums them into the exact full attention output. The traffic scales with batch size and hidden size but not with the KV length (the huge KV cache never moves), so it stays constant as the context grows to millions of tokens. It's exact, not approximate.
- Phase 2 — FFN/MoE, re-laid-out as Tensor Parallel. Now the same N GPUs switch view (the all-to-all above already leaves the attention output in a TP layout for the output projection). The FFN runs as:
  - TP for a dense model (each GPU holds a slice of the FFN weight matrices), or
  - TP × Expert Parallel for an MoE (experts distributed across GPUs).

To minimize the exposed communication cost, Helix introduces HOP-B (batch-wise overlap): the all-to-all of one request overlaps with the attention computation of the next.

Compared to conventional parallelism approaches, Helix reduces TTL by up to 1.5x at fixed batch sizes and supports up to 32× larger batches under the same latency budget for DeepSeek-R1 on Blackwell. Unlike attention–FFN disaggregation (e.g., MegaScale-Infer), which splits the two phases *spatially* onto different GPUs, Helix splits them *temporally* on the same GPUs.

<img loading="lazy" width="503" height="596" alt="image" src="https://github.com/user-attachments/assets/2e699dd5-4cb9-4414-a8f4-b074f975b782" />

</details>



<details> <summary markdown="span">2025 [TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference, Peking University & Tencent](https://arxiv.org/abs/2508.15881)</summary>

TPLA  fixes an incompatibility between Multi-Head Latent Attention (MLA) and tensor parallelism:
- **MLA**, introduced in DeepSeek-V2, compresses the key–value state into a single low-rank latent vector `cKV` and caches only that, slashing KV-cache memory.
- But under Megatron-LM **tensor parallelism (TP)**—where attention heads are split across devices—every device still needs the *entire* `cKV` to compute its heads, so the cache is replicated on each GPU. That replication erodes MLA's whole advantage: per-device memory ends up no better than (or worse than) plain Grouped Query Attention. An existing alternative, Grouped Latent Attention (GLA), shards the latent but then each head only sees part of it, weakening representational capacity.

TPLA's scheme **partitions both the latent representation and each head's input dimension across devices, runs attention independently on each shard, and combines the shard outputs with an all-reduce**. The key difference from GLA: every head in TPLA still attends to the *full* latent space (the partial results are summed back together), so it keeps more of MLA's representational power while cutting the per-device KV cache. Since each shard applies its own softmax before the all-reduce, the result approximates rather than exactly reproduces MLA; still, TPLA works on MLA-pretrained checkpoints without retraining, and applying an orthogonal transform (Hadamard or PCA) before slicing mitigates the cross-shard interference, keeping the accuracy loss minimal.

The split is asymmetric — TPLA uses MLA for prefill and TPLA for decode — because the two phases have different bottlenecks.

- Prefill: keep standard (reparameterized) MLA. Prefill processes the whole prompt at once, so it's compute-bound, not memory-bound, and doesn't repeatedly re-read a KV cache.

- Decode: switch to TPLA — partition the latent across devices. Decode generates one token at a time, so it's memory-bound: every step you must reload the entire KV cache from HBM, and that's where MLA's full-cKV-on-every-device replication hurts. Here TPLA changes the layout: The latent vector cKV (and each head's input dimension) is sliced across the TP devices. Device 0 holds the first chunk of the latent, device 1 the next, etc; then each device runs attention independently on its local shard; then they sum-reduce individual contributions. Because each head's contribution is split and then summed, every head still effectively attends to the full latent space (just computed in pieces).

On DeepSeek-V3 and Kimi-K2 at 32K context it delivers **1.79× and 1.93×** speedups while holding quality on commonsense and LongBench benchmarks, and it can be implemented with FlashAttention-3 for real end-to-end gains. The "disaggregated" framing fits the prefill/decode split: sharded latent for memory-bound decode, standard MLA for compute-bound prefill.

<img loading="lazy" width="736" height="335" alt="image" src="https://github.com/user-attachments/assets/6310c491-9f82-4233-9ec5-97ca17fe9662" />

</details>




<details> <summary markdown="span">2025 [PipeFill: Using GPUs During Bubbles in Pipeline-parallel LLM Training](https://openreview.net/forum?id=650J0YFjnV)</summary>

PipeFill improves GPU utilization in pipeline-parallel LLM training by filling idle "bubble" time—often 15–30% and sometimes over 60% of a job's allocation—with unrelated pending jobs rather than eliminating the bubbles themselves. It fits fill-job work to measured bubble durations and available memory, adds explicit pipeline-bubble instructions, and orchestrates execution inside the gaps, yielding up to 63% higher utilization with under 2% slowdown of the main job (about 2.6K extra GPUs' worth of work on an 8K-GPU run). This makes it complementary to schedule-optimization approaches like DualPipe, 1F1B, interleaving, and ZeroBubble, which instead shrink or hide bubbles by densifying the training job's own schedule through forward/backward and compute/communication overlap—PipeFill is a context-switching layer, not a pipeline schedule.

Good fill jobs are workloads that tolerate interruption, with batch (offline) inference being ideal: generating embeddings for a corpus or running a smaller model over a prompt backlog. Such work is chunkable to match short bubble windows, latency-insensitive so pausing is fine, and small enough in memory to fit the spare capacity PipeFill frees up. The paper also uses other DL training jobs as fill work since training is long-running and interruption-tolerant, though inference is easier to slice into bubble-sized pieces.

</details>




<details> <summary markdown="span">2025 [Speculative Decoding with Blockwise Sparse Attention, MatX](https://matx.com/research/sd_nsa)</summary>

This MatX article tackles a conflict that arises when you combine two popular decoding accelerators. **Speculative decoding (SD)** uses a small draft model to propose several tokens that the big target model then verifies in parallel, raising operational intensity (work done per byte loaded from HBM) and giving ~2× latency/throughput gains. **Blockwise sparse attention**—as in DeepSeek's Native Sparse Attention (NSA) and Moonshot's Mixture of Block Attention (MoBA)—divides the context into blocks and lets each token dynamically attend to only a subset of them, cutting the KV data moved from HBM and again raising operational intensity (important because long contexts otherwise leave modern accelerators underutilized).

"Speculative decoding (SD) and blockwise sparse attention both accelerate LLM decoding, but when combined naively, the KV cache may lose sparsity during the verification step of SD. We show that forcing all draft tokens to attend to the same subset of the context restores sparsity while preserving model quality."

The authors show NSA models trained this way match the language-modeling quality of baseline NSA, while achieving up to **3.5× higher operational intensity** during the verification step of speculative decoding (validated on small 186M and 1.2B-parameter models trained with 2K-token sequences). Conceptually it's a co-design point: rather than treating SD and sparse attention as independent layers, it reshapes the sparsity pattern so the two compose efficiently.

</details>



<details> <summary markdown="span">2025 [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention, Tsinghua & UC Berkeley](https://arxiv.org/abs/2509.24006)</summary>

SLA targets the attention bottleneck in Diffusion Transformers (DiTs), especially for video generation, where very long token sequences make attention's quadratic cost dominate latency. Its key observation is that attention weights split into two parts: a small fraction of large weights with high rank, and the remaining weights with very low rank. This suggests applying sparse acceleration to the first part and low-rank (linear) acceleration to the second, instead of committing to a single approximation (pure sparsity or pure linear attention).

SLA classifies attention blocks as *critical*, *marginal* or *negligible*: critical ones get exact O(N²) attention, marginal ones get an O(N) linear-attention approximation, and negligible ones are skipped—all fused into a single GPU kernel that supports both the forward and backward passes, so a pretrained DiT only needs a few fine-tuning steps to adapt. This makes SLA behave like a “knob” between dense and approximate attention, which differentiates it from static sparse masks or fixed-kernel linear attention.

In their evaluation, SLA reduces attention computation by **95%** (≈20×) without degrading end-to-end generation quality, and its kernel gives a **13.7×** attention speedup and a **2.2× end-to-end** video generation speedup on **Wan2.1-1.3B**.
</details>


<details> <summary markdown="span">2025 [DeepSeek‑V3.2‑Exp (model card & notes), DeepSeek‑AI](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)</summary>

DeepSeek‑V3.2‑Exp is an incremental update in the DeepSeek V3 family aimed primarily at **serving efficiency**, especially for long-context workloads. It is DeepSeek‑V3.1‑Terminus with a single architectural change introduced through continued training; compared to the original DeepSeek‑V3 technical report (Dec 2024), the emphasis shifts from “how we trained it” to “how we run it cheaper”, without changing the core MoE transformer recipe.

The headline change is **DeepSeek Sparse Attention (DSA)**: a lightweight “lightning indexer” (a few heads, ReLU scoring, runnable in FP8) scores every preceding token for the current query, and a fine-grained top-k selection (k = 2048 tokens) keeps only those key/value entries for the main MLA attention. This cuts the core attention cost from O(L²) to O(L·k), for both prefill and decode at long context lengths. Note that the KV cache itself does not shrink (the full latent cache is still stored so that any token can be selected); what drops is the compute and the number of KV entries read per step. DSA is introduced by continued training: a short dense warm-up that trains the indexer to mimic the dense attention distribution (KL loss), followed by sparse training of the whole model.

The Hugging Face release notes also call out practical serving details (recommended runtime settings, known issues/fixes, and evaluation notes). If you already have DeepSeek‑V3 deployed, treat V3.2‑Exp as a **serving-oriented refresh**: benchmarks stay on par with V3.1‑Terminus, while the biggest wins come at large context lengths. The same architecture was kept for the final DeepSeek‑V3.2 ([tech report](https://arxiv.org/abs/2512.02556), Dec 2025).

</details>



<details> <summary markdown="span">2025 [DeepCompile: A Compiler-Driven Approach to Optimizing Distributed Deep Learning Training, Microsoft DeepSpeed](https://arxiv.org/abs/2504.09983)</summary>

DeepCompile extends the capabilities of deep learning compilers to support distributed training. "Distributed training has become essential for scaling today’s massive deep learning models. While deep learning compilers like PyTorch compiler dramatically improved single-GPU training performance through optimizations like kernel fusion and operator scheduling, they fall short when it comes to distributed workloads."

Existing distributed training frameworks require distributed optimizations to be implemented at the PyTorch level, which limits the ability to apply compiler-style techniques like dependency analysis or operator scheduling.
"The fully sharded approach, as implemented in systems like ZeRO-3 and FSDP, employs runtime optimizations such as prefetching and unsharding":
- Prefetching aims to reduce communication overhead by initiating all-gather operations earlier than the layer where the parameters are actually needed, thereby overlapping communication with computation;
- unsharding keeps parameters in their full form to reduce communication when memory permits;

DeepCompile addresses this gap by enabling compiler-level optimizations for distributed training. "It takes a standard single-GPU model implementation and transforms it into an optimized multi-GPU training graph without requiring changes to the model code".
The authors implement a fully sharded approach like ZeRO-3 and FSDP on top of DeepCompile, along with three optimizations: proactive prefetching, selective unsharding, and adaptive offloading.
- Proactive prefetching. To maximize overlap between communication and computation, this optimization pass initiates all-gather as early as possible, considering how available memory changes as the forward and backward passes progress.
- Selective unsharding. This pass keeps as many parameters unsharded as possible to reduce communication overhead caused by all-gather communication, and decides which parameters to unshard based on operator-level memory profiling.
- Adaptive offloading. DeepCompile offloads optimizer states such as momentum and variance used by Adam to CPU memory when GPU memory is insufficient. To reduce data transfer overhead, it offloads only the amount of data that exceeds the memory limit and schedules transfers to overlap with computation.

It automatically implements distributed ZeRO-3, ZeRO-1, and offloading. On Llama-3 70B and Mixtral 8×7B, it achieves up to 1.28× and 1.54× speedups over the ZeRO-3 and FSDP baselines, respectively, and up to 7.01× higher throughput with offloading when GPU resources are limited. Future directions include automated parallelization (sequence/tensor parallelisms), smarter memory management, and dynamic adaptation to runtime behavior.
</details>


<details> <summary markdown="span">2025 [Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler, ByteDance Seed](https://arxiv.org/abs/2504.19442)</summary>

Triton-distributed extends Triton with *distributed* primitives so you can write kernels that **compute and communicate inside the same program**, rather than launching a GEMM kernel and then calling NCCL as a separate step. The paper’s mental model is “a distributed kernel is a set of asynchronous tasks” that can **signal** one another and use **symmetric memory** (OpenSHMEM-style primitives, lowered to NVSHMEM/ROCSHMEM) to move data directly between GPUs.

The key difference vs “plain Triton + NCCL” is that communication becomes an explicit part of the kernel schedule: the compiler/runtime can overlap loads, math, and remote puts/gets at a much finer granularity than stream-level overlap of independent kernels. Compared to systems like Flux that hand-optimize specific LLM communication patterns, Triton-distributed tries to be a **general programming model** (primitives + compiler support) that you can reuse across operators and topologies.

**Results / takeaways:**
- On 8 H800 GPUs it averages **1.42×** (AllGather+GEMM) and **1.28×** (GEMM+ReduceScatter) over PyTorch+NCCL; across two nodes (16 GPUs) these kernels reach ~96% of FLUX’s performance while being written in Python. MoE kernels show much larger average speedups (**44.97×** for AllGather+MoE, **15.55×** for MoE+ReduceScatter), but against a weak baseline (group GEMMs in Python loops).
- It scales to 64 GPUs, re-implements DeepEP-style low-latency AllToAll kernels with comparable performance, and also runs on AMD MI308X GPUs.
- In practice, it’s a “make the compiler responsible for overlap” approach: you write a single kernel that *contains* both compute and the collective-ish data movement, and the system handles the async orchestration.

</details>


<details> <summary markdown="span">2025 [TileLink: Generating Efficient Compute–Communication Overlapping Kernels Using Tile-Centric Primitives, ByteDance Seed (MLSys 2025)](https://arxiv.org/abs/2503.20313)</summary>

TileLink is a step toward **Flux-like performance** without requiring you to hand-craft a very specialized fused kernel for each operation. It decouples the design space of communication and computation and links the two through **tile-centric primitives** (producer/consumer notify–wait signals and tile push/pull data movement), each mapped to a tensor slice, a peer rank and a barrier channel; the system then schedules computation and communication *per tile* (rather than per whole tensor), enabling overlap that approximates what highly tuned systems achieve.

A useful way to distinguish TileLink from nearby work:
- **Flux**: decomposes specific tensor-parallel LLM ops into finely sliced pieces, then fuses them into larger kernels to overlap up to “almost all” communication; highly effective, but quite tailored.
- **Triton-distributed**: offers low-level in-kernel primitives (symmetric memory, signals, tasks) and a general programming model.
- **TileLink**: sits in between—**higher-level than Triton-distributed**, more general and easier to program than Flux, but still explicitly models overlap at the *tile schedule* level.

**Results / takeaways:**
- On 8×H800, the paper reports **~1.17× to ~20.76×** speedups over non-overlapping baselines, with performance comparable to (or better than) state-of-the-art overlapping libraries such as FLUX and RingAttention.
- End to end, across eight language models it averages **1.32×** over PyTorch on 8 H800 GPUs and **1.29×** on two nodes (16 GPUs).

</details>


<details> <summary markdown="span">2025 [MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production, ByteDance Seed & Peking University](https://arxiv.org/abs/2505.11432)</summary>

MegaScale-MoE is a production-oriented system for **communication-efficient MoE training**, where the “hard part” is not just computing experts but routing tokens and moving activations/gradients at scale. The paper’s central point is that MoE training becomes dominated by **communication** (e.g., all-to-all between token routers and experts), and that you need a holistic strategy to keep GPUs busy. Concretely, it (1) customizes the parallelism of each MoE layer—sequence parallelism for attention and expert parallelism for the FFNs within a node, instead of Megatron-LM’s tensor parallelism—to reduce communication volume; (2) overlaps communication with computation at both the inter-operator and intra-operator levels; and (3) compresses communication to lower precision, with adjusted communication patterns.

How it differs from “kernel-level fusion” work (Flux / TileLink / Triton-distributed):
- MegaScale-MoE is primarily a **system/runtime + parallelization strategy** for end-to-end training at very large scale, not a new kernel programming model.
- Its gains come from choosing the right parallelism layout, scheduling and communication precision for the whole training job, rather than from a new way of writing fused kernels.

**Results / takeaways (as reported):**
- Training a **352B MoE** model on **1,440 NVIDIA Hopper GPUs**, it reaches **1.41M tokens/s**, a **1.88×** efficiency improvement over Megatron-LM (1.65–1.88× across the strong-scaling settings).
- It powers ByteDance’s production MoE training, scaling to trillions of parameters and thousands of GPUs; the headline is that communication efficiency is not a rounding error at this scale.

</details>


<details> <summary markdown="span">2025 [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism, ByteDance Seed & Peking University](https://arxiv.org/abs/2504.02263)</summary>

MegaScale-Infer's answer to the memory-bound performance issue of MoE FFNs during decode is **Disaggregated Expert Parallelism (DEP)**: disaggregate the attention and expert modules, assigning them to separate GPUs,  so each can scale independently with its own parallelism. Attention modules are replicated using data parallelism, while FFN modules are scaled with expert parallelism; by consolidating requests from multiple attention replicas, the GPU utilization of each expert increases significantly as the batch size per attention replica grows.  In other words, many attention replicas funnel their tokens into the expert pool, restoring large per-expert batches and turning the FFNs back into efficient compute-bound work; it also allows **heterogeneous deployment**—cheaper/different GPUs for attention vs. experts—to cut cost. Two systems pieces make disaggregation pay off rather than stall on communication: **ping-pong pipeline parallelism**, which partitions a request batch into micro-batches and shuttles them between attention and FFNs  so the two GPU pools stay busy and communication hides behind compute, and a **high-performance M2N communication library** that eliminates unnecessary GPU-to-CPU data copies, group initialization overhead, and GPU synchronization  for the token dispatch/combine traffic. This is the MoE-serving flavor of attention–FFN disaggregation: the two modules are split *spatially* across GPU pools, whereas Helix splits them *temporally* by re-laying-out the same GPUs; prefill/decode disaggregation (as in TPLA) is a different, orthogonal split.

**Results / takeaways (as reported):**
- Up to **1.90×** higher per-GPU decoding throughput than state-of-the-art LLM serving systems on homogeneous clusters.
- Up to **1.86×** higher throughput per cost on heterogeneous clusters, by placing attention and experts on different GPU types.

<img loading="lazy" width="360" height="224" alt="image" src="https://github.com/user-attachments/assets/ba7fd9f2-2827-4ede-8755-356735e699ec" />

</details>


<details> <summary markdown="span">2025 [Accelerating MoE Model Inference with Expert Sharding (MoEShard), EPFL & McGill (EuroMLSys 2025)](https://arxiv.org/abs/2503.08467)</summary>

Balmau et al. target MoE inference with expert parallelism (EP), where skewed token routing leaves some GPUs overloaded while others sit idle, and the usual fixes (capacity factors that drop tokens, or replicating popular experts) either lose tokens or cost memory. Their system, **MoEShard**, instead shards *every* expert across all GPUs, tensor-parallel style: the first expert matrix is split column-wise and the second row-wise, so each GPU computes a slice of every selected expert and the partial outputs are summed. Load is perfectly balanced regardless of routing skew, and no token is dropped.

To keep this efficient, MoEShard fuses the decomposed expert computations to minimize kernel launches. Evaluated on encoder-based MoE models (Switch Transformer) on a single multi-GPU node, it reports up to **6.4×** lower time-to-first-token than DeepSpeed.

This work is a useful complement to the MegaScale-* systems: it is a small-scale (single-node, workshop) study showing that tensor sharding of experts is a viable alternative to plain EP when routing is imbalanced, relying on fast intra-node links for the extra communication.

</details>



<details> <summary markdown="span">2025 [TransMLA: Multi-Head Latent Attention Is All You Need, Peking University & Xiaomi](https://arxiv.org/abs/2502.07864)</summary>

TransMLA is a **post-training conversion** method that turns existing Grouped-Query Attention (GQA)-based models (LLaMA, Qwen, Mixtral) into MLA-based ones, without retraining from scratch. The motivation is an adoption gap: most model providers have invested heavily in GQA models and have little incentive to pretrain MLA models from scratch. The paper's theoretical core is a proof that **GQA can always be rewritten exactly as MLA at the same KV-cache budget, but not vice versa**—i.e., MLA is strictly more expressive than GQA for a given cache size. That asymmetry means any GQA checkpoint can be losslessly re-expressed in MLA form and then *upgraded* to use the spare expressiveness MLA affords.

TransMLA exploits this in two steps. First, it **equivalently converts** the model's GQA attention into MLA structure—reformulating the repeated KV heads of GQA as low-rank latent projections—producing a model directly compatible with DeepSeek's codebase and its optimized serving stacks (vLLM, SGLang), plus features like FP8 quantization and Multi-Token Prediction. Second, since the conversion alone does not speed anything up, it **compresses** the latent KV below the original cache size and then **fine-tunes** briefly to recover quality—only ~6B tokens are needed to regain performance on par with the original model across benchmarks. The payoff: compressing 93% of LLaMA-2-7B's KV cache yields a 10.6× inference speedup at 8K context while preserving meaningful output quality. It's the migration path that complements TPLA and Helix: rather than designing new MLA models or new sharding, TransMLA *retrofits* the large existing fleet of GQA models into the more efficient MLA family.

Two observations:
- For a fixed KV-cache budget, MLA is strictly more expressive than GQA.
- Inference acceleration occurs only when MLA uses a smaller KV cache.
</details>



<details> <summary markdown="span">2025 [Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B, Stanford Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)</summary>

This work targets an extreme low-latency regime: **single-sequence decoding** for a ~1B-parameter LLM (Llama-3.2-1B), where performance is dominated by how efficiently the GPU can stream weights from global memory. The authors argue that modern inference engines still suffer from “pipeline bubbles” because a forward pass is split into **dozens to ~100 small kernels**, each incurring launch/teardown costs and synchronization stalls that prevent continuous memory streaming.

Their solution is to fuse the *entire* forward pass into a single **megakernel** that effectively acts like an on-GPU interpreter: each SM executes a schedule of “instructions” (fused RMSNorm+QKV+RoPE, attention, projections, MLP pieces, etc.). They then focus on three practical problems that show up when you fuse “about a hundred” operations:
1. How to *program* such a megakernel (an instruction abstraction + interpreter),
2. How to avoid resource contention (e.g., **paged shared memory** to pipeline weight loads),
3. How to synchronize without kernel boundaries (a lightweight **counter-based** scheme).

**Results / takeaways (as reported):**
- On an **H100**, they report using **~78%** of available memory bandwidth and outperforming popular engines by **>1.5×**; they also report that vLLM and SGLang use at most **~50%** of the bandwidth in this setting.
- In their end-to-end comparison (32-token prompt, 128 generated tokens, no speculation), they report the megakernel is **~2.5× faster than vLLM** and **>1.5× faster than SGLang** on **H100**.
- On **B200**, they report **>3.5×** speedup over vLLM and still **>1.5×** over SGLang.
- They highlight achieving a **<1 ms** forward pass for a 16-bit 1B+ model on H100, and **<680 µs** per forward pass on B200.

This sits in a different part of the design space than Flux/TileLink/Triton-distributed: it’s about eliminating *kernel boundaries entirely* for single-GPU low-latency inference, rather than fusing compute with distributed communication.

</details>


<details> <summary markdown="span">2025 [Mirage: A Multi-Level Superoptimizer for Tensor Programs, CMU et al. (OSDI 2025)](https://arxiv.org/abs/2405.05751)</summary>

Mirage is an *automatic* approach to generating deeply fused tensor programs, framing kernel fusion as a superoptimization problem over **µGraphs**: a uniform representation of a tensor program at the kernel, thread-block, and thread levels of the GPU compute hierarchy. µGraphs let Mirage jointly search algebraic transformations, schedule transformations, and the generation of new custom kernels—it rediscovers FlashAttention- and FlashDecoding-like kernels on its own. An abstraction-based pruning technique keeps the huge search space tractable (with an optimality guarantee), and a probabilistic equivalence verifier checks that the optimized µGraph computes the same function as the input program.

How it differs from other “big fusion” lines:
- Compared to **manual megakernels** (e.g., “No Bubbles”), Mirage aims to *discover* aggressive fusions automatically and generate optimized implementations.
- Compared to **FlashAttention-style** work, Mirage is broader: it targets arbitrary tensor programs (not just attention), and can potentially find FlashAttention-like structures when they are profitable.
- Compared to **compiler frameworks** like TVM/XLA, Mirage emphasizes superoptimization over small fused graphs to get performance that can beat hand tuning.

**Results / takeaways (as reported in the paper):**
- Mirage outperforms existing approaches by **up to 3.3×** (the exact figure varies across paper versions), even for DNN components that are widely used and heavily optimized.

</details>


<details> <summary markdown="span">2025 [SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training, Tsinghua](https://arxiv.org/abs/2505.11594)</summary>

SageAttention3 pushes the SageAttention line toward even more aggressive inference acceleration by running both attention matmuls on the new **FP4 Tensor Cores** of Blackwell GPUs, with **microscaling** (block-scaled) FP4 quantization plus attention-specific tricks to keep accuracy. Conceptually, it sits between “attention kernels that assume FP16/BF16” and “end-to-end quantized models”: it focuses on the attention operator itself and is designed to be dropped into existing inference stacks.

On **RTX 5090**, the paper reports **1038 TOPS**, a **5×** speedup over the fastest FlashAttention on that GPU (FlashAttention2, since FlashAttention3 only runs on Hopper), with plug-and-play end-to-end speedups on various models (e.g., video generation with HunyuanVideo). It also pioneers low-bit attention for *training* (SageBwd, 8-bit forward and backward): lossless in fine-tuning, but with slower convergence in pretraining.
</details>


<details> <summary markdown="span">2025 [Titans: Learning to Memorize at Test Time, Google Research (NeurIPS 2025)](https://arxiv.org/abs/2501.00663)</summary>
A family of models that combine attention with a *neural* long-term memory: an MLP whose weights keep being updated at test time by gradient steps on an associative-memory loss, driven by a "surprise" signal with momentum and a weight-decay-style forgetting gate. From the abstract: "We present a new
neural long-term memory module that learns to memorize historical context and helps an attention to attend to the
current context while utilizing long past information. We show that this neural memory has the advantage of a fast
parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its
limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its
ability to memorize the data, acts as a long-term, more persistent, memory."

The paper proposes three ways to combine the two: memory as context (MAC), as a gate (MAG), or as a layer (MAL). Titans outperform Transformers and modern linear recurrent models (e.g., Mamba2, DeltaNet) on language modeling, commonsense reasoning, genomics and time series, and scale to context windows beyond 2M tokens with higher needle-in-a-haystack accuracy than the baselines.
</details>




<details> <summary markdown="span">2024 [LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism, Peking University (SOSP 2024)](https://arxiv.org/abs/2404.09526)</summary>
LoongServe introduces **Elastic Sequence Parallelism (ESP)**: instead of a fixed degree of parallelism per instance, the degree of sequence parallelism is adapted in real time per iteration and per phase (scale-up/scale-down). It does ring attention in the prefill phase and while rotating KV tensors across GPUs, each GPU keeps a subset of KV locally so that it is automatically available during decode (a proactive scale-down that avoids migrating the KV cache). When it comes to the decode step, the KV cache stays sharded across instances, so they follow the regular (not so efficient) distributed decode algorithm, which is usually the execution bottleneck of context parallelism. They overlap comp/comm with cuda streams during prefill (ring attn); for decode, the paper claims the partial-attention communication is overlapped with computation, but in practice this is opportunistic: while one node is sending its requests, it MAY be doing matmuls for requests it receives from other GPUs, which is not guaranteed. ESP also reduces KV-cache fragmentation across instances. They report up to **3.85×** higher maximum throughput than chunked prefill and **5.81×** than prefill–decode disaggregation.
</details>



<details> <summary markdown="span">2024 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces, CMU & Princeton (COLM 2024)](https://arxiv.org/abs/2312.00752)</summary>

Mamba is the state-space model (SSM) that made a non-attention sequence model competitive with Transformers on language. SSMs keep a fixed-size recurrent state and scale **linearly** in sequence length, but earlier SSMs (S4) underperformed because their dynamics were fixed regardless of input. The fix is the **selection mechanism**: making the SSM parameters functions of the input lets the model selectively propagate or forget information depending on the current token—giving  content-based reasoning. Input-dependence breaks the convolution trick, so Mamba adds a **hardware-aware parallel scan** (kept in SRAM, FlashAttention-style) to stay fast. Payoff: 5× higher inference throughput than Transformers, linear scaling, and Mamba-3B matching Transformers twice its size.  Its constant-size state is what eliminates the growing KV cache—the property hybrids (Jamba, Nemotron-3) exploit.

**Formula.** An SSM maps input $x_t$ → output $y_t$ through a hidden state $h_t \in \mathbb{R}^N$. Continuous params $(\Delta, A, B, C)$ are first **discretized**:

$$\bar A = \exp(\Delta A), \qquad \bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B$$

Then it runs as a **linear recurrence** (inference) or a **global convolution** (training):

$$h_t = \bar A\,h_{t-1} + \bar B\,x_t, \qquad y_t = C\,h_t$$

In S4 these params are **fixed across time** (LTI). Mamba's change: $B$, $C$, and $\Delta$ become **functions of $x_t$** (the S6 selective SSM), so $(\bar A, \bar B)$ vary per token — that's the selectivity, and what forces the scan-based instead of convolution-based compute.

<img loading="lazy" width="623" height="308" alt="image" src="https://github.com/user-attachments/assets/073ff296-7eea-431d-b7de-88facaa8fad8" />

</details>



<details> <summary markdown="span">2024 [Jamba: A Hybrid Transformer-Mamba Language Model, AI21 Labs](https://arxiv.org/abs/2403.19887)</summary>

Jamba (Lieber, Lenz et al., AI21) is the first **production-scale hybrid** interleaving Transformer attention, Mamba SSM, and MoE layers. The core trade-off it balances: the KV cache becomes a limiting factor when scaling Transformers to long contexts, and trading attention layers for Mamba layers shrinks it—since  Mamba carries only a constant-size state, not a growing cache. Attention is kept sparingly (for high-fidelity recall), Mamba does the cheap long-range work, and MoE adds capacity without adding active compute.

**Released config** (tunable knobs): 4 blocks of 8 layers, 1:7 attention-to-Mamba ratio, MoE every other layer, 16 experts with top-2 per token  → 52B total but only 12B active, fits on one 80GB GPU.  The 1:7 ratio is empirical (most of attention's quality, most of Mamba's memory benefit). Headline win is the cache: at 256K context Jamba needs ~4GB of KV cache vs 32GB for Mixtral and 128GB for Llama-2-7B, with strong results up to 256K tokens and up to 3× the throughput of Mixtral at long contexts. Jamba 1.5 later scaled the same design to 94B active / 398B total. It's a template Nemotron-3 follows—with Mamba-2 layers, and adding LatentMoE, MTP and NVFP4.

<img loading="lazy" width="401" height="424" alt="image" src="https://github.com/user-attachments/assets/67898fc1-c338-494c-92b4-3c6ef8a42f9e" />

</details>


<details> <summary markdown="span">2024 [Better & Faster Large Language Models via Multi-token Prediction, Meta (FAIR), ICML 2024](https://arxiv.org/abs/2404.19737)</summary>

This is the paper that establishes Multi-Token Prediction (MTP) as a general training objective (Gloeckle et al., Meta). The premise is a critique of the standard recipe: LLMs like GPT and Llama are trained with a next-token-prediction loss, and the authors argue that training to predict multiple future tokens at once yields higher sample efficiency. 

The model is trained with multiple "heads."
The model predicts the following n tokens using n independent output heads sitting on top of a shared model trunk,  treating multi-token prediction as an **auxiliary task** layered on the usual objective. 
While the main head predicts the next token $x_{n+1}$, auxiliary heads predict $x_{n+2}, x_{n+3}$, and so on.
- Inference Trick: During generation, the model uses these auxiliary heads to produce a "draft" of the next few tokens simultaneously with the main token.
- Parallel Verification: in the next step, the model takes those guesses and runs them through its main, high-fidelity blocks in one single forward pass.

Results: the benefit grows with model size; the 13B model solves 12% more HumanEval and 17% more MBPP problems than a comparable next-token model, and 4-token models are up to 3× faster at inference with this self-speculative decoding, even at large batch sizes.

In a standard speculative decoding setup, the roles are split strictly between iterative guessing and parallel verification:
1. The Draft Model (Iterative). The small model acts just like a standard LLM. It generates tokens one by one, sequentially. Process: It predicts token 1, feeds it back into itself, predicts token 2, feeds it back, and so on, until it has a "look-ahead" sequence (usually 3–5 tokens). Why: It is small enough that these multiple sequential passes are still much faster than a single pass of the large model.
2. The Larger/Target Model (All at once). The large model performs a single parallel forward pass to check the draft model's work. Process: It takes the entire sequence of drafted tokens and processes them simultaneously using a causal mask. The "Magic": Because of how Transformers work, the large model can calculate the probability of "Token 4" being correct at the exact same time it calculates the probability for "Token 1." Outcome: In the time it would normally take the large model to generate one token, it verifies all the drafted tokens.

MTP is used on DeepSeek-V3 and Nemotron 3: DeepSeek-V3 refined MTP into *sequential, cascaded* modules that preserve the full causal chain (rather than independent parallel heads), and Nemotron 3 adopts a lightweight MTP module reaching ~97% acceptance on its first two predicted tokens. 

</details>



<details> <summary markdown="span">2024 [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, DeepSeek-AI](https://arxiv.org/abs/2405.04434)</summary>

DeepSeek-V2 introduces Multi-head Latent Attention (MLA) and DeepSeekMoE—the two architectural pillars later carried into V3 and R1. It's a 236B-total-parameter MoE model with only 21B activated per token and a 128K context. These innovations target the two big performance bottlenecks: the KV cache of vanilla Multi-Head Attention on decoding, and dense FFNs on training.

**DeepSeekMoE** provides the FFN: fine-grained experts plus shared experts, so strong capacity comes from sparse activation at low compute. Two main ideas:
1. **Shared experts**: a small number ($K_s$) that every token always uses, no routing. Used for common knowledge. These aren't clones of each other; they're a fixed always-on set that absorbs common/general knowledge.
2. **Fine-grained expert segmentation.** A regular MoE has $N$ experts, each a full-size FFN, and activates top-$K$. DeepSeekMoE slices each expert into $m$ smaller ones → $mN$ total experts, and activates $mK$ of them. Same total parameters, same FLOPs — but far more *combinations* of experts per token. Finely segmenting the experts into $mN$ ones and activating $mK$ from them allows for a more flexible combination of activated experts.

**MLA** uses low-rank joint compression of keys and values: rather than caching full per-head K/V (or sharing heads as in GQA/MQA), it down-projects them into a single small latent vector `cKV` that is cached instead of the full K/V, then up-projects back to distinct per-head K/V at compute time—keeping full-MHA expressiveness while shrinking the cache. Because RoPE can't be absorbed through that up-projection, V2 adds the **decoupled-RoPE** trick: a separate set of small query dims plus one shared key (also cached, a small extra $d_h^R$ per token) carry the positional rotation, kept apart from the compressed "content" path.

**Symbols:** $h_t$ = token's input hidden vector, where $t$ is the token's position index, $n_h$ = #heads, $d_h$ = per-head dim · $d_c$ = latent (cache) dim, $d_c \ll n_h d_h$ · $d_h^R$ = small RoPE dim · $W^{D\ast}$ = down-projections (compress), $W^{U\ast}$ = up-projections (reconstruct) · superscript $C$ = content part, $R$ = RoPE part · $j \le t$ = past tokens.
Compress $h_t$ into the cached latent, then reconstruct distinct per-head K/V from it at compute time:

$$c_t^{KV} = W^{DKV}\, h_t \quad (\in \mathbb{R}^{d_c},\ \text{cached, together with the RoPE key } k_t^{R}); \qquad k_t^{C} = W^{UK} c_t^{KV},\quad v_t^{C} = W^{UV} c_t^{KV}$$

where $W^{DKV}$, $W^{UK}$ and $W^{UV}$ are learnt parameters that give lower-rank transformations. Queries are also low-rank (saves activation memory, not cached):

$$c_t^{Q} = W^{DQ}\, h_t, \qquad q_t^{C} = W^{UQ}\, c_t^{Q}$$

- As a side note, when comparing with the regular attention, where the $W$ projections also give a lower-rank matrix: regular attention factors $W^K$ *by head* (and stays full-rank overall); MLA factors $W^K$ *through a shared low-dim latent* (and goes deliberately rank-deficient) — and that latent is the thing the KV cache stores.
- Note on notation: $W^{DKV}$: **D**own projection on KV.  $W^{UK}$ and $W^{UV}$: **U**p projection on K and V.

Decoupled RoPE — position rides a separate path: per-head rotary queries and one **shared** rotary key:

$$q_t^{R} = \mathrm{RoPE}(W^{QR}\, c_t^{Q}), \qquad k_t^{R} = \mathrm{RoPE}(W^{KR}\, h_t)$$

Final per-head query/key concatenate content + rotary parts:

$$q_{t,i} = [\,q_{t,i}^{C}\,;\,q_{t,i}^{R}\,], \qquad k_{t,i} = [\,k_{t,i}^{C}\,;\,k_t^{R}\,]$$

Then attention is standard:

$$o_{t,i} = \sum_{j \le t} \mathrm{softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v_{j,i}^{C}, \qquad u_t = W^{O}\,[\,o_{t,1};\dots;o_{t,n_h}\,]$$

**KV cache per token (per layer)**

| MHA | GQA | MQA | **MLA** |
|---|---|---|---|
| $2\,n_h\,d_h$ | $2\,g\,d_h$ | $2\,d_h$ | $d_c + d_h^R$ |

The "absorb" trick folds $W^{UK}$ into $W^{UQ}$ and $W^{UV}$ into $W^{O}$ at decode time, so attention runs directly against `cKV` and the full per-head K/V are never materialized. DeepSeek-V2 settings: $d_c = 512$, $d_h^R = 64$ ($d_h = 128$) → cache $= 576$ elements per token per layer, comparable to GQA with ~2.25 groups but with full-MHA expressiveness.

Improvements over the earlier dense DeepSeek 67B are concrete: 42.5% lower training cost, a **93.3% smaller KV cache**, and up to **5.76× higher maximum generation throughput**, while improving quality.

</details>


<details> <summary markdown="span">2024 [SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](https://arxiv.org/abs/2411.10958)</summary>

SageAttention2 builds on SageAttention by moving to **INT4** quantization for the expensive \(QK^\top\) path (at a *thread-level* granularity) while using **FP8** for the \,\(\widetilde{P}V\)\, product. The key idea is that attention has quantization pathologies that don’t show up as strongly in linear layers—especially outliers—so the paper adds attention-specific techniques to make low-bit arithmetic viable. 

Concretely, it proposes (1) per-thread INT4 quantization for \(Q,K\), (2) a smoothing method for \(Q\) to improve INT4 \(QK^\top\) accuracy, and (3) a two-level accumulation strategy to improve FP8 \(\widetilde{P}V\) accuracy. The paper reports that on **RTX 4090**, SageAttention2’s OPS surpasses FlashAttention2 and xFormers by about **3×** and **4.5×**, respectively, and that it can match FlashAttention3(fp8) speed on Hopper while achieving higher accuracy.
</details>


<details> <summary markdown="span">2024 [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration, Tsinghua (ICLR 2025)](https://arxiv.org/abs/2410.02367)</summary>

SageAttention is an “attention-first” quantization paper: rather than quantizing an entire model, it targets the attention kernel and aims to make it plug-and-play for existing LLM inference. The motivation is that, even when linear layers are heavily optimized, attention can remain a major bottleneck—especially at long sequence lengths—because it mixes GEMMs, softmax, and reductions in ways that amplify numerical outliers.

The core idea is to run attention with **8-bit** arithmetic while preserving accuracy. The paper identifies two pathologies of naïve INT8 attention: \(K\) has strong channel-wise outliers (fixed by *smoothing* \(K\), i.e. subtracting its mean over tokens, which leaves the softmax unchanged), and quantizing \(P, V\) to INT8 is not reliably accurate—so it computes \(QK^\top\) in INT8 and keeps \(\widetilde{P}V\) in FP16 with an FP16 accumulator. Compared to later SageAttention2/3, this first version is less aggressive and functions as the baseline that demonstrates the feasibility of attention quantization with minimal integration work.

For performance, the paper reports about **2.1×** the OPS of FlashAttention2 and **2.7×** that of xFormers, with better accuracy than FlashAttention3's FP8 attention and almost no end-to-end metric loss across language, image and video generation models.
</details>


<details> <summary markdown="span">2024 [Centauri: Enabling Efficient Scheduling for Communication–Computation Overlap in Large Model Training via Communication Partitioning, Peking University et al. (ASPLOS 2024)](https://doi.org/10.1145/3620666.3651379)</summary>

Centauri tackles communication–computation overlap at the *operator scheduling* level. The core idea is to **decompose operators** and then schedule the decomposed pieces across multiple CUDA streams so that communication can be overlapped with useful compute more consistently than “whole-operator overlap.” The “communication partitioning” framing highlights that a large collective (or data movement phase) can be broken into smaller chunks that are scheduled earlier and more frequently, improving pipeline utilization. Concretely, the partition space has three dimensions—primitive substitution (rewriting a collective as other primitives), topology-aware group partitioning (e.g., hierarchical intra-/inter-node collectives), and workload partitioning (chunking the data)—and the resulting pieces are scheduled hierarchically at the operation, layer, and model levels of hybrid-parallel training.

How to situate Centauri relative to Flux / TileLink / Triton-distributed:
- Centauri is mainly about **multi-stream scheduling of decomposed operators** and improving overlap among (still separate) kernels; it does not require in-kernel collectives.
- Flux/TileLink/Triton-distributed pursue **in-kernel** fusion of compute and communication, which can overlap at a finer granularity, but often requires more specialized codegen/primitives.
- Centauri is therefore a good fit when your bottleneck is *orchestration* (stream scheduling, dependencies, coarse kernel granularity), whereas Flux-like approaches target cases where overheads persist even with aggressive stream overlap.

**Results / takeaways:** up to **1.49×** speedup over prevalent methods across various hybrid-parallel training configurations (the paper received an ASPLOS 2024 Best Paper award). (If you end up writing your own fused kernels later, Centauri is still useful as a baseline: it shows what you can get “for free” from better scheduling alone.)

</details>


<details> <summary markdown="span">2024 [FLUX: Fast Software-based Communication Overlap on GPUs Through Kernel Fusion, ByteDance & Peking University](https://arxiv.org/abs/2406.06858)</summary>

Flux is explicitly designed around Megatron-LM-style tensor-parallel patterns. Instead of treating tensor-parallel layers as “one big GEMM + one big NCCL collective,” Flux **decomposes** those operations into fine-grained pieces (e.g., tiles/chunks) and then fuses them into kernels that can overlap communication with computation aggressively—claiming overlap levels that are hard to reach with standard multi-stream overlap of separate kernels.

The most useful way to think about Flux in the related-work landscape:
- Compared to Centauri, Flux aims for **overlap *within* the operator**, not just between independent kernels.
- Compared to Triton-distributed, Flux is less about a general programming model and more about delivering peak performance for a *specific family* of LLM ops (Megatron tensor-parallel attention/MLP patterns).
- TileLink is motivated partly by Flux’s effectiveness: it tries to preserve Flux-like performance while raising the programming level.

**Results / takeaways (as reported):**
- Flux reports up to **~96%** communication overlap.
- For training, up to **1.24×** speedup over Megatron-LM on a cluster of **128 GPUs** (various GPU generations and interconnects).
- For inference, up to **1.66×** speedup for **prefill** and **1.30×** for **decoding** over vLLM on **8 GPUs**.

</details>


<details> <summary markdown="span">2024 [Optimizing Distributed ML Communication with Fused Computation–Collective Operations, AMD (SC 2024)](https://arxiv.org/abs/2305.06942)</summary>

Punniyamurthy et al. focus on identifying common “compute + collective” motifs that appear repeatedly in distributed training/inference graphs and then implementing them as **single fused kernels** with **in-kernel collectives**: workgroups send their results to remote GPUs (GPU-initiated communication) as soon as their computation finishes, while other workgroups of the same kernel keep computing, and scale-up transfers are zero-copy (written directly into peer GPU memory). The contribution is partly a taxonomy (“these patterns show up everywhere”) and partly a concrete engineering result: implementing fused kernels in Triton/HIP that reduce overheads from intermediate writes, kernel boundaries, and separated communication steps.

The paper’s patterns are especially useful to keep in mind when comparing systems:
- Flux is very focused on tensor-parallel LLM patterns; this paper’s catalog is **broader** (e.g., embedding + all-to-all, GEMV + all-reduce, GEMM + all-to-all).
- Triton-distributed/TileLink offer programming models to build such fused kernels; this paper provides **hand-built instances** and measurements that show why the fusion matters.
- Unlike in-network-compute (INC)/offload approaches, these fused kernels still assume the communication happens through GPU-side primitives (no in-network execution).

**Results / takeaways (as reported):**
- The paper reports **12%–31%** lower execution time across the three fused operators (e.g., intra-node on four GPUs: embedding + all-to-all 20% lower on average and up to 32%, GEMV + all-reduce 13% and up to 22%, GEMM + all-to-all 12% and up to 20%), showing that the win can be material even for “simple” two-stage pipelines when communication is tightly coupled to the compute.

</details>


<details> <summary markdown="span">2024 [DeepSeek-V3 Technical Report (includes DualPipe), DeepSeek-AI](https://arxiv.org/abs/2412.19437)</summary>

DeepSeek-V3 is a 671B-parameter MoE (37B active per token) trained on 14.8T tokens in 2.788M H800 GPU hours (~5.6M USD), and it is useful to read as a “what it takes in practice” reference: model architecture decisions (MLA and DeepSeekMoE from V2, auxiliary-loss-free load balancing, multi-token prediction), FP8 mixed-precision training, and the kinds of engineering trade-offs that are often omitted from purely algorithmic papers. The details are particularly relevant in a related-work section that discusses communication-heavy structures (like MoE) and the real limits of standard communication substrates.

Where it fits in the landscape here:
- DeepSeek-V3 demonstrates that you can hit strong scaling and quality with careful MoE architecture/routing and training practices on 2,048 H800 GPUs. Its communication relies on custom cross-node all-to-all kernels (later open-sourced as DeepEP) that use only 20 of the 132 SMs, exploit IB + NVLink by limiting each token to at most 4 nodes, and are overlapped with computation at the *schedule* level (DualPipe), rather than fused into the GEMM kernels or offloaded to the network.
- This makes it a good “control point” when discussing what incremental improvements from fused kernels or in-network compute might buy: you can compare against a strong end-to-end system that already works at scale.

**Results / takeaways:** the report is a systems-and-training reference more than a single “one trick” optimization; it’s most valuable for its end-to-end design and empirical scaling observations, which help ground discussions about where communication becomes dominant and what kinds of optimizations remain on the table.

DualPipe is a bidirectional pipeline-parallelism algorithm introduced in the DeepSeek-V3 Technical Report, designed to reduce pipeline bubbles while addressing the heavy communication overhead of cross-node expert parallelism (which gives DeepSeek-V3 a roughly 1:1 compute-to-communication ratio). Its core idea is to overlap computation and communication within a paired forward and backward chunk: each chunk is split into four components—attention, all-to-all dispatch, MLP, and all-to-all combine (with the backward attention/MLP further split into input- and weight-gradient parts, as in ZeroBubble)—which are rearranged so that one set of micro-batches runs forward while another runs backward, hiding communication behind compute. Micro-batches are fed symmetrically from both ends of the pipeline, keeping hardware more consistently busy than sequential schedules like 1F1B or ZeroBubble.

The result is fewer bubbles and full forward/backward computation-communication overlap, at the cost of holding two copies of model parameters (one per direction) and slightly higher activation memory. The parameter duplication is affordable because DeepSeek-V3 uses a large expert-parallel size, so each rank holds relatively few parameters; separately, careful memory savings (recomputation, CPU-resident EMA, an output head shared with the MTP module) let them train without costly tensor parallelism. Unlike PipeFill, which accepts bubbles and fills them with unrelated jobs, DualPipe attacks the bubbles directly by densifying the main job's own schedule—making the two approaches complementary rather than competing.

<img loading="lazy" width="1391" height="418" alt="image" src="https://github.com/user-attachments/assets/5a580ef6-139f-4ee9-891c-2b4f6832d790" />

</details>




<details> <summary markdown="span">2024 [Zero Bubble Pipeline Parallelism, Sea AI Lab (ICLR 2024)](https://arxiv.org/abs/2401.10241)</summary>

Zero Bubble is the first scheduling strategy to achieve zero pipeline bubbles under synchronous training semantics. The key insight is to split the backward pass into two independent computations—one that produces the input gradient (B) and one that produces the parameter/weight gradient (W)—since grouping them sequentially is unnecessary. Decoupling B and W reduces sequential dependencies, letting the later-starting W passes fill the tail-end bubbles that a standard 1F1B schedule leaves behind. The paper offers two handcrafted schedules plus an automatic scheduler: it formulates the problem as integer linear programming (solvable by an off-the-shelf ILP solver) and pairs it with a heuristic for large microbatch counts.

The two variants trade memory for bubble reduction: ZB-H1 keeps the same peak activation memory as 1F1B but cuts the bubble to about a third of 1F1B's, while ZB-H2 eliminates bubbles entirely at the cost of higher peak memory (roughly 2× activations) and an extra optimizer validation/rollback step (it bypasses optimizer synchronization, then corrects after the step). In experiments, the zero-bubble schedules outperform 1F1B by up to 23% in throughput under a similar memory limit, and by up to 31% when the memory constraint is relaxed. The method is orthogonal to data, tensor, and ZeRO parallelism and can drop in as a replacement for the PP component. Like DualPipe, it attacks bubbles by reshaping the main job's schedule—contrasting with PipeFill, which leaves bubbles in place and fills them with other jobs.

<img loading="lazy" width="899" height="597" alt="image" src="https://github.com/user-attachments/assets/3deb5a12-5d63-4c32-898f-c966f18271a3" />

</details>




<details> <summary markdown="span">2024 [FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design, Microsoft & University of Sydney (USENIX ATC 2024, as Quant-LLM)](https://arxiv.org/abs/2401.14112)</summary>

FP6-LLM makes six-bit floating-point (FP6) quantization practical for LLM inference on GPUs. FP6 is an appealing sweet spot: it shrinks model size and weight-loading time substantially while preserving quality more reliably than 4-bit—4-bit methods look fine on zero-shot benchmarks but degrade on harder tasks like code generation and summarization, whereas 6-bit stays robust. The catch is that no existing system gave FP6 real Tensor Core support, because an irregular 6-bit width causes unfriendly, misaligned memory accesses to model weights and incurs heavy runtime overhead from de-quantizing weights back to a format the Tensor Cores can multiply. The paper's answer is TC-FPx, the first full-stack GPU kernel design with unified Tensor Core support for floating-point weights at arbitrary quantization bit-widths, handling the memory-layout and on-the-fly de-quantization problems together.

How TC-FPx works, short version:

- Weights are FP6 (1 sign + 3 exponent + 2 mantissa bits = 6 bits, denoted **E3M2**) feeding a **W6A16** linear layer (6-bit weights × FP16 activations).
- Weights live in memory as 6-bit the whole time — that's the point, smaller footprint and faster loads from DRAM. They're pre-packed offline into an aligned layout (the odd 6-bit width is split into 2-bit + 4-bit fragments so loads aren't misaligned).
- On the fly, FP6 → FP16 just before compute: SIMT cores load a slice, stitch the fragments back together with bit shifts/masks, and expand to FP16 in registers. Tensor Cores multiply in FP16, overlapped with the next slice's de-quant.
- Outputs are never converted back to FP6. Only the weights are quantized (W6A16 = 6-bit weights, 16-bit activations). Activations and the matmul results stay FP16 — there's no "store back as FP6" step.
- The win is memory bandwidth: you move 6-bit weights instead of 16-bit, and the de-quant cost is hidden behind the Tensor Core math
  - in practice, the dequant work is overlapped on cuda cores while the tensor core does matmuls. The kernel is software-pipelined so the dequant of slice N+1 on SIMT core overlaps with matmul of slice N on tensor core.
  - Quantize = FP16 → FP6 (compress down). Done once, offline, before serving.
  - De-quant = FP6 → FP16 (expanding the compressed 6-bit weight back up to 16-bit so the Tensor Cores can use it). Done on the fly at runtime, on the CUDA cores, every time the weights are loaded.

Integrating TC-FPx into an existing inference stack yields the end-to-end FP6-LLM system (W6A16: 6-bit weights, 16-bit activations). Results: LLaMA-70B can be served on a single GPU, with 1.69×–2.65× higher normalized inference throughput than the FP16 baseline; at the kernel level, the 6-bit linear layer runs up to 2.6× faster than cuBLAS FP16 and up to 1.9× faster than TensorRT-LLM's 8-bit (W8A16) kernels. It's an algorithm-system co-design—the kernel-level work is what turns FP6's theoretical compression into actual serving speedups and better cost/quality trade-offs. Source code is available at [github.com/usyd-fsalab/fp6_llm](https://github.com/usyd-fsalab/fp6_llm).
</details>



<details> <summary markdown="span">2024 [The Llama 3 Herd of Models, Meta](https://arxiv.org/abs/2407.21783)</summary>

Llama 3 is "a herd of language models that natively support multi-linguality, coding, reasoning, and tool usage." The models are made of 8B, 70B and 405B parameters and a context window of 128K tokens. Llama 3 405B uses an architecture with 126 layers, a token representation dimension of 16,384, and 128 attention heads. 
Llama 3 405B is trained on up to 16K H100 GPUs, via 4D parallelism (tensor, pipeline, context and data).
The authors used scaling laws (Hoffmann et al., 2022) to determine the optimal model size for their flagship model given their pre-training compute budget (section 3.2.1): they first correlate the compute-optimal model's negative log-likelihood on downstream tasks with the training FLOPs, and then establish a sigmoidal relation between that log-likelihood and task accuracy (figure 4):

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/llama3_scaling_laws.png"/>

The model architecture does not deviate from Llama 2, except that they:

1. use grouped query attention with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding, and
2. "use an attention mask that prevents self-attention between different documents within the same sequence as is important in continued pre-training on very long sequences".
3. vocabulary with 128K tokens: 100K from `tiktoken` and 28k for better non-English support.
4. increase the RoPE base frequency hyperparameter to 500,000 to better support longer contexts.

Training is performed in two stages: pre-training via next-token prediction, and post-training where the model is "tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning)." The improvements were performed at 3 levels:
1. at the data level, the authors improved quality, quantity, pre-processing and curation. The dataset includes "15T multilingual tokens, compared to 1.8T tokens for Llama 2."
2. At the scale level, the pre-training compute grew almost $$50 \times$$ over the largest Llama 2 model, reaching $$3.8 \times 10^{25}$$ FLOPs; and 
3. managing complexity, where they used a regular transformer with minor adaptations instead of a mixture of experts, and "a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO), as opposed to more complex reinforcement learning algorithms." (section 4)

The authors also experiment adding image, video, and speech capabilities, by adding three additional stages:
- multi-modal encoder pre-training, where image and speech encoders are trained separately (sections 7 and 8). The image encoder is trained on large amounts of image-text pairs, while the speech encoder is trained with a self-supervised method that "masks out parts of the speech inputs and tries to reconstruct the masked out parts via a discrete-token representation".
- vision-adapter training, where the authors train an adapter on text-image pairs to align image representations with language representations. Then they train a video adapter on top of the image adapter on paired video-text data, to enable model to aggregate information across frames (section 7).
- Speech adapter training: a third adapter converts speech encodings into token representations.

The image encoder is a standard vision transformer trained to align images and text, the ViT-H/14 variant. They introduce cross-attention layers (using grouped-query attention) between the visual token representations produced by the image encoder and the token representations produced by the language model, after every 4th self-attention layer.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="100%" height="100%" src="/assets/publications/llama_3_multi_modal.png"/>

Results (section 5) investigate the "performance of: (1) the pre-trained language model, (2) the post-trained language model, and (3) the safety characteristics of Llama 3".

In section 6, they investigated two main techniques to make inference with the Llama 3 405B model efficient: (1) pipeline parallelism on 16 H100s with BF16 and (2) FP8 quantization. FP8 quantization is applied to most parameters and activations in feed-forward network but not to parameters of self-attention layers of the model. Similarly to Xiao et al 2024b they use dynamic scaling factors for better accuracy (with upper bound of 1200), and do not perform quantization in the first and last Transformer layers, and use row-wise quantization, computing scaling factors across rows for parameter and activation matrices.
</details>


<details> <summary markdown="span">2024 [Universal Checkpointing: Efficient and Flexible Checkpointing for Large Scale Distributed Training, Microsoft DeepSpeed](https://arxiv.org/abs/2406.18820)</summary>

According to the paper, the issue with state-of-the-art distributed checkpointing (model save/resume) is that it requires "static allocation of GPU resources at the beginning of training and lacks the capability
to resume training with a different parallelism strategy and hardware configuration" and usually it is not possible to resume when hardware changes during the training process. To this extent, the paper proposes "Universal Checkpointing, a technique that enables efficient checkpoint
creation while providing the flexibility of resuming on arbitrary parallelism strategy" and "improved resilience to hardware failures through continued training on remaining healthy hardware, and reduced training time through opportunistic exploitation of elastic capacity". This is achieved by writing in the universal checkpoint format, which allows "mapping parameter
fragments into training ranks of arbitrary model-parallelism configuration", and universal checkpoint language that allows for "converting distributed checkpoints into the universal checkpoint format". The UCP format consolidates the distributed shards into atomic checkpoints—one consolidated set of files per model parameter (fp32 weights and optimizer states)—which can then be re-sharded onto a different layout (data, tensor, pipeline and sequence parallelism, ZeRO stages).
</details>


<details> <summary markdown="span">2024 [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping, Microsoft DeepSpeed](https://arxiv.org/abs/2409.15241)</summary>

Domino "provides a generic scheme to hide communication behind computation" when training large LLMs where tensor parallelism (TP) is applied. "By breaking data dependency of a single batch training into smaller independent pieces, Domino pipelines these independent pieces training and provides generic strategy of fine-grained communication and computation overlapping. … comparing with Megatron-LM, Domino achieves up to 1.3x speedup for LLM training on Nvidia DGX-H100 GPUs". The rationale for the paper is: current efforts to overlap computation and communication during TP are not enough, especially "in the cases where collective communication takes much longer than a single GeMM computation, most of the communication time still stands out as the major training overhead". And "given that computation on the latest GPUs  is becoming faster,
communication overhead is more pronounced". The paper proposes "Domino, a generic approach that breaks data dependency of transformer model training into pieces, and then pipelines these pieces training to overlap communication with computation ….  Domino provides a much wider scope of computation and communication overlapping (e.g., AllReduce not only overlaps with a single GeMM, but also LayerNorm, DropOut, etc). … To hide TP communication behind computation, Domino provides extra and generic tensor partition in two dimensions on every GPU: row-wise split on inputs X and column-wise split on weights B on top of original TP model partitions. At high level, Domino generically breaks TP’s $$X \cdot A \cdot B$$ into smaller compute units without data dependency. Then it pipelines these independent compute units with collective communication to achieve fine-grained computation and communication overlapping … we keep $$A$$ untouched and do not conduct any tensor partitioning on $$A$$. Therefore, we only conduct tensor slicing on input tensor $$X$$ (section 3.2) and the second group of linear weights as $$B$$ (section 3.3). We also provide a hybrid tensor partition strategy of both $$X$$ and $$B$$ (section 3.4). After these tensor slicing, Domino breaks $$X \cdot A \cdot B$$ into pieces and removes data dependency. Then we enable computation-communication overlapping on these independent pieces to reduce communication overhead in TP."

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="100%" height="100%" src="/assets/publications/domino.png"/>

</details>


<details> <summary markdown="span"> 2024 [The Road Less Scheduled, Meta (NeurIPS 2024)](https://arxiv.org/abs/2405.15682)</summary>

"Existing learning rate schedules that do not require specification of the optimization stopping step T are greatly out-performed by learning rate schedules that depend on T." The Schedule-Free approach is an optimization method that does not need the specification of T by removing the need of schedulers entirely. It requires no new hyper-parameters.

Background: take the typical SGD optimization with step size $$γ$$ in the form $$z_{t+1} = z_t − γ g_t$$, where $$g_t$$ is the gradient at step $$t$$. "Classical convergence theory suggests that the expected loss of this $$z$$ sequence is suboptimal, and that the Polyak-Ruppert (PR) average $$x$$ of the sequence should be returned instead" as $$x_{t+1} = (1 − c_{t+1}) x_t + c_{t+1} z_{t+1}$$. If we use $$c_{t+1} = 1/(t+1)$$, then $$x_T = \frac{1}{T} \sum_{t=1}^T z_t$$. As an example, after 4 steps we have:

$$
  \begin{align*}
x_1 = & z_1\\
x_2 = & \frac{1}{2} x_1 + \frac{1}{2} z_2, \\
x_3 = & \frac{2}{3} x_2 + \frac{1}{3} z_3, \\  
x_4 = & \frac{3}{4} x_3 + \frac{1}{4} z_4, \\
x_5 = & \frac{4}{5} x_4 + \frac{1}{5} z_5, \\
  \end{align*}
$$

However, "despite their theoretical optimality, PR averages give much worse results in practice than using the last-iterate of SGD":

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="50%" height="50%" src="/assets/publications/schedule_free.png"/>

Recently, Zamani and Glineur (2023) and Defazio et al. (2023) showed that the exact worst-case optimal rates can be achieved via carefully chosen learning rate schedules alone, without the use of averaging. However, LR schedulers require the definition of the stopping time T in advance. So the question of the paper is:

> Do there exist iterate averaging approaches that match the empirical performance of learning rate schedules, without sacrificing theoretical guarantees?

This paper shows that it exists by introducing "a new approach to averaging that maintains the worst-case convergence rate theory of PR averaging, while matching and often exceeding the performance of schedule-based approaches", demonstrated on 28 problems (Schedule-Free AdamW was also the core of the winning entry of the MLCommons 2024 AlgoPerf Algorithmic Efficiency Challenge, self-tuning track).  Schedule-Free methods show strong performance, matching or out-performing heavily-tuned cosine schedules. The formulation of this **Schedule-Free SGD** is:

$$
  \begin{align*}
y_t = \, & (1 − β) z_t + β x_t, \\
z_{t+1} = \, & z_t − γ∇f(y_t, ζ_t), \\
x_{t+1} = \, & (1 − c_{t+1}) x_t + c_{t+1} z_{t+1}, \\
  \end{align*}
$$

where $$f(y_t, ζ_t)$$ is the loss between model output and random variable $$ζ$$, $$c_{t+1}$$ is defined as before and $$z_1 = x_1 $$. "Note that with this weighting, the $$x$$ sequence is just an online equal-weighted average of the $$z$$ sequence." This method has a momentum parameter $$β$$ that interpolates between Polyak-Ruppert averaging ($$β = 0$$) and Primal averaging ($$β = 1$$). Primal averaging is the same as PR except that gradient is evaluated at the averaged point $$x$$, instead of $$z$$ (see paper for definition), and "maintains the worst-case optimality of PR averaging but is generally considered to
converge too slowly to be practical (Figure 2)."

The main point is: "The advantage of our interpolation is that we get the
best of both worlds. We can achieve the fast convergence of Polyak-Ruppert averaging (since the
$$z$$ sequence moves much quicker than the $$x$$ sequence), while still keeping some coupling between
the returned sequence $$x$$ and the gradient-evaluation locations $$y$$, which increases stability (Figure 2). Values of β similar to standard momentum values $$β ≈ 0.9$$ appear to work well in practice."
</details>


<details> <summary markdown="span">2023 [Simplifying Transformer Blocks, ETH Zurich (ICLR 2024)](https://arxiv.org/abs/2311.01906)</summary>

A simpler transformer block, motivated by signal propagation theory, that removes skip connections, value and projection parameters, sequential sub-blocks (attention and MLP run in parallel) and normalisation layers. It claims to match the per-update training speed and performance of standard autoregressive decoder-only and BERT encoder-only models, with 16% faster training throughput (15% in the arXiv version) while using 15% fewer parameters. The experiments are at the 100–300M-parameter scale.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/simplifying_transformer_blocks.png"/>
</details>


<details> <summary markdown="span"> 2023 [Training and inference of large language models using 8-bit floating point, Graphcore](https://arxiv.org/abs/2309.17224)</summary>

The paper "presents a methodology to select the scalings for FP8 linear layers, based on dynamically updating per-tensor scales for the weights, gradients and activations." The FP8 representations tested are FP8-E4 and FP8-E5, with 4 and 5 bits of exponent, respectively. Despite the naming, intermediate computation (accumulation) is performed in 16 bits. The scaling is applied to the exponent (a power-of-two shift of the exponent bias), not by rescaling the final value. They tested two scaling techniques, AMAX (scale derived from the tensor's absolute maximum) and SCALE (keeping the scale constant), and noticed there isn't a major degradation. Results compare FP8 to FP16 on GPT and Llama 2 models from 111M to 70B parameters, but not to `bfloat16` because hardware was not available at the time. Algorithm in Figure 3. Note: FP8 matmuls run at ~2× the FP16/BF16 throughput on hardware that supports them, and the per-tensor scaling only adds cheap element-wise work, so the end-to-end gain is below 2× but still significant.
</details>


<details> <summary markdown="span"> 2023 [DeepSpeed ZeRO-Offload++: 6x Higher Training Throughput via Collaborative CPU/GPU Twin-Flow](https://github.com/microsoft/DeepSpeed/tree/offloadpp-news/blogs/deepspeed-offloadpp)</summary>

"System efficiency is still far from optimal when adopting ZeRO-Offload in some scenarios. Especially in the cases like small batch training, model that could not fit into GPU memory but not orders-of-magnitude bigger than GPU memory capacity, CPU offload not only introduce long end-to-end latency, but also underutilize GPU computation resources." With that in mind, Zero-Offload++ introduces 3 features:
- Twin-Flow: instead of having an all-or-nothing policy (ie offload all or none of) in the values to be offloaded, "Twin-Flow allows a portion of optimizer states to be held in CPU memory and the other portion of optimizer states remaining in GPU memory. When optimization step is triggered, both CPU and GPU can do parameter updates simultaneously." The user can choose the percentage of ratio of parameters in CPU and GPU. "Therefore, with Twin-Flow, we can achieve decent GPU memory and core utilization rate, at the same time reduce training iteration time in optimizer offloading cases." 
- MemCpy reduction: details not available yet;
- CPUAdam optimization: details not available yet;

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="50%" height="50%" src="/assets/publications/ZeroOffloadPlusPlus.png"/>
</details>



<details> <summary markdown="span">2023 [DistFlashAttn: Distributed Memory-efficient Attention for Long-context LLMs Training (LightSeq), UC Berkeley et al.](https://arxiv.org/abs/2310.03294)</summary>

DistFlashAttn extends FlashAttention from a single GPU to a distributed, sequence-parallel setting for **training** long-context LLMs. FlashAttention already turns attention's quadratic peak memory into linear by computing it tile-by-tile with online softmax and never materializing the full score matrix. The split is by sequence (tokens), not by head or by feature dimension. Each GPU stores the Q, K, and V for its own token chunk only — full hidden dimension, all heads, just fewer rows. The challenge is doing this efficiently under *causal* attention, and the paper contributes three techniques:

1. **Token-level workload balancing:** causal masking means a token only attends to its prefix, so a GPU holding *early* tokens has little work while one holding *late* tokens does much more (its work grows with the chunk's position in the sequence)—a severe imbalance in naive sequence parallelism. DistFlashAttn rebalances this work across devices so no worker sits idle waiting for the heavily loaded ones.

2. **Overlapping KV communication:** like Ring Attention, K/V chunks are passed between GPUs so each device can attend to remote tokens, but DistFlashAttn overlaps that communication with the local attention computation, hiding the transfer latency behind useful work.

3. **Rematerialization-aware gradient checkpointing:** standard checkpointing recomputes a layer's forward pass during the backward pass to save memory, but applied naively to FlashAttention it redundantly recomputes the attention—DistFlashAttn makes checkpointing aware of what FlashAttention already recomputes, avoiding the double work.

One framing difference to Ring Attention: Ring Attention is a fairly general sequence-parallel attention mechanism (works for inference and training, causal or not). DistFlashAttn is specifically tuned for causal, long-context LLM training — the load imbalance and the checkpointing interaction are both problems that only really bite in that setting.

Together these let it train up to **8× longer sequences** (2–8× longer than Megatron-LM with FlashAttention) and run faster than the alternatives of the time: 4.45–5.64× over Ring Self-Attention, 1.24–2.01× over Megatron-LM with FlashAttention, 1.67× over Ring Attention and 1.26–1.88× over DeepSpeed-Ulysses, tested on Llama-7B at 32K–512K tokens. It's applied to training only; there's no inference use case. DistFlashAttn is essentially Ring Attention plus three fixes for causal training: (1) causal load balancing (the biggest one); (2) communication/computation overlap; and (3) rematerialization-aware gradient checkpointing.

</details>




<details> <summary markdown="span">2023 [Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion, Peking University et al. (HPCA 2023)](https://ieeexplore.ieee.org/document/10071018)</summary>

Chimera (Zheng et al.) is a compiler framework that fuses *chains of compute-intensive operators*—like back-to-back GEMMs, or GEMM-then-softmax in attention—into single kernels to improve data locality. The motivation: as hardware compute throughput has outpaced memory bandwidth, even compute-heavy operators like GEMM and convolution end up memory-bound when run as a chain, because each operator writes its full output to memory only for the next to read it back. Existing ML compilers lacked both precise analysis and good optimization for these chains on different accelerators, so they left performance on the table. Chimera's key idea is to break each operator into a series of *computation blocks* and then optimize at two levels. For *inter-block* optimization it uses an analytical model to pick the block execution order that minimizes data movement between blocks (so intermediate results stay in fast on-chip memory instead of round-tripping to DRAM); for *intra-block* optimization it generates an efficient hardware-specific "micro-kernel."

 Examples from the paper:

- The GEMM→GEMM chain `C = A×B` and `E = C×D` where a tile of `C` will be immediately used to compute `E`, without fully instantiating `C`.
  - if we had assigned to each GPU block a tile of `E`, there would be a lot of contention and the need to introduce atomics. Instead, here, they assign to each GPU block: a full row (block) of `A` and `E`, it produces and immediately iterates tiles of `C`, and iterates all columns of `B` and `D`.
- Batch GEMM chain (attention). `Q×Kᵀ → softmax → ×V` — two batch GEMMs with a softmax in the middle.
- Convolution chain (CNNs). A `3×3 conv → ReLU → 1×1 conv → ReLU` fused into one kernel — common in ResNet-style nets, memory-bound at certain input shapes.

Because it relies on an analytical cost model rather than exhaustive auto-tuning, Chimera avoids the long tuning times of search-based compilers like Ansor. Evaluated on batch-GEMM chains and convolution chains, it achieves up to 2.87×, 2.29×, and 2.39× speedups over hand-tuned libraries on CPU, GPU (A100), and NPU (Huawei Ascend), respectively, and up to 2.29×, 1.64×, and 1.14× over state-of-the-art compilers on the same platforms—so the work is hardware-portable across CPUs, Tensor Core GPUs and NPUs.
 
</details>




<details> <summary markdown="span"> 2023 [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training, Microsoft](https://arxiv.org/abs/2306.10209)</summary>

DeepSpeed ZeRO's compute throughput is limited by the high communication cost from gathering weights in forward pass, backward pass, and averaging gradients. This is mostly prominent on clusters with low-bandwidth, and at very small batch sizes per GPU.

**Background, communication pipeline:** "Assume the model size as 𝑀. During the forward pass, ZeRO conducts an all-gather operation to collect all the parameters (𝑀) needed to train for all model layers. In the backward pass, ZeRO re-collects parameters (𝑀) with all-gather first, then each GPU can compute local gradients. After that, ZeRO operates reducescatter function to aggregate and redistribute gradients (𝑀) across accelerators. In total, ZeRO has a total communication volume of 3𝑀, spreads evenly across 2 all-gather and 1 reduce-scatter."

The paper introduces three communication reduction techniques, packed as ZeRO++:
1. **Quantized Weight Communication for ZeRO (qwZ):** perform block quantization of the forward all-gather, converting weights  from FP16 (2 bytes) to INT8 (1 byte). The main improvement is to replace the typical quantization algorithm (multiplying all parameters by a scalar), by a quantization per block (ie per parameter subset) that includes multiplication by a factor and shifting values by another factor;
2. **Hierarchical Weight Partition for ZeRO (hpZ):** data remapping that trades-off communication for more memory and reduces communication overhead of all-gather on weights during backward. Instead of having weights distributed across GPUs, we maintain a full copy on each machine, allowing us to replace the expensive cross-machine all-gather on weights with a faster intra-machine all-gather.
3. **Quantized Gradient Communication for ZeRO (qgZ):** replaces the gradients reduce-scatter collective, by doing (1) block-based quantization of gradients to `INT4` during communication to reduce the communication size, and recovering the full precision before the reduction operator to preserve training accuracy. Having a fully block-based quantization approach like in (1) was also considered but led to high precision loss and a high error propagation across layers during backpropagation. 

The results section claims that  ZeRO++ yields a communication reduction of 4x compared to ZeRO-3, leading to up to 2.16x higher compute throughput on 384 GPUs.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/ZeROplusplus.png"/>
</details>


<details> <summary markdown="span"> 2023 [QLoRA: Efficient Finetuning of Quantized LLMs, University of Washington](https://arxiv.org/abs/2305.14314)</summary>

"An efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while
preserving full 16-bit finetuning task performance. QLORA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). QLORA introduces multiple innovations designed to reduce memory use without sacrificing performance: (1) 4-bit NormalFloat, an information theoretically optimal quantization data type for
normally distributed data that yields better empirical results than 4-bit Integers and 4-bit Floats.
(2) Double Quantization, a method that quantizes the quantization constants, saving an average
of about 0.37 bits per parameter (approximately 3 GB for a 65B model). (3) Paged Optimizers,
using NVIDIA unified memory to avoid the gradient checkpointing memory spikes that occur when
processing a mini-batch with a long sequence length.  We use QLORA
to finetune more than 1,000 models, [and] results show that QLoRA
finetuning on a small high-quality dataset leads to state-of-the-art results, even
when using smaller models than the previous SoTA". Their best model family, Guanaco, reaches 99.3% of ChatGPT's performance on the Vicuna benchmark after 24 hours of finetuning on a single GPU. Notes to self:
- 4-bit NormalFloat (NF4) rounds each value to the nearest of 16 levels placed at the quantiles of a standard normal distribution. General quantile quantization is expensive (it needs approximate quantile estimation, e.g. "SRAM quantiles", which also gives large errors for outliers); NF4 sidesteps this because pretrained weights are roughly zero-centred normal, so the quantiles are precomputed once and each block of weights is simply rescaled by its absmax.
</details>


<details> <summary markdown="span"> 2023 [Better speech synthesis through scaling (TorToise), James Betker](https://arxiv.org/abs/2305.07243)</summary>

The paper describes a way to apply the techniques used for image generation to speech
synthesis. This result is TorToise, an expressive, multi-voice text-to-speech system. So far, TTS models were hard to train efficiently due to high sampling rate, unavailability of large datasets, or encoder-decoder challenges.

Background: most modern text-to-speech systems operate on speech data that is encoded as a MEL spectrogram. Because of this, most efforts focus on the high-quality decoding of MEL spectrograms back into audio waveforms, a.k.a. a vocoder or a MEL inverter. The author reviews the state-of-the-art autoregressive transformers and DDPMs:
- **DALL-E**, a transformer model with a (quadratic complexity) full-sequence self-attention, that showed how an autoregressive decoder can be applied to text-to-image generation. The author believes that the "VQVAE decoder used by DALL-E is principally responsible for the blurry incoherence exhibited by most of it’s samples".
  - DALL-E also introduced the process of **re-ranking**, that samples from the autoregressive model and picks the best output for downstream use. Re-ranking requires a strong discriminator to tell good from bad text/image pairings. CLIP was used for this purpose.
- Denoising diffusion probabilistic models (**DDPMs**) generate crisp high quality images, and are effective on using low-quality signals to reconstruct the high-dimensional space where those signals derived from. However, DDPMs rely on fixed output shapes, known beforehand. Thus, they "cannot learn to convert text into audio signals because they cannot solve the implicit alignment problem between text and audio". Also, DDPMs must be sampled from over multiple iterations, leading to high compute cost and latency. 

With that in mind: **TorToise works by joining autoregressive decoders and DDPMs**: "the autoregressive model will be used to convert a sequence of text tokens to a sequence of tokens representing the output space (in our case, speech tokens). The DDPM will then be used to decode these tokens into a high quality representation of speech." In practice, for Text-To-Speech, we train the following neural networks:
- An auto-regressive model on text tokens that yields the probability of each audio token;
- A contrastive model that ranks outputs of the autoregressive decoder. DALL-E uses CLIP (for images), but TorToise uses Contrastive Language-Voice Pretrained Transformer (CLVP, for TTS). 
- A DDPM to convert speech tokens back into speech spectrograms;
- A vocoder (UnivNet) to convert the MEL spectrograms into waveforms.

The inputs of the auto-regressive and DDPM models include (or are conditioned to) an additional speech conditioning input, which is one or more audio clips (MEL spectrograms) of the same speaker as the target. This allows the model to "infer vocal characteristics like tone and prosody" desired in the target output audio. Finally, they apply the **TorToise trick**: the DDPM is first trained on converting discrete speech codes into MEL spectrograms,  and then **fine-tuned** on the latent space of the AR model outputs instead of the speech codes. "The logic here is that the AR latent space is far more semantically rich than discrete tokens. By fine-tuning on this latent space, we improve the efficiency of the downstream diffusion model"

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/TorToise-v2.png"/>
</details>


<details> <summary markdown="span"> 2023 [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E), Microsoft](https://arxiv.org/abs/2301.02111)</summary>

The paper introduces a pipeline for text-to-speech synthesis (TTS), based on a neural codec language model (VALL-E) using discrete codes (encode/decode embeddings) derived from an off-the-shelf neural audio codec model (EnCodec, Défossez et al., 2022).
This model treats TTS as a conditional language modeling task rather than continuous signal regression as in previous work.
In practice, contrarily to e.g. AudioLM, a generative audio-to-audio / speech-to-speech model that predicts future audio from input audio, VALL-E is a TTS model that takes as input the phoneme sequence of the text and a 3-second enrolled recording of an unseen target speaker (the acoustic prompt), and generates the speech for that text in the prompt speaker's voice (a continuation mode, where the prompt is the first 3 seconds of the utterance itself, is also evaluated).
VALL-E uses an audio codec code as intermediate representation and language model as objective, contrary to previous models using mel spectrogram as intermediate representation and continuous signal regression as objective.
VALL-E is trained with the LibriLight dataset, consisting of 60K hours of English speech with over 7000 unique speakers. This dataset is audio-only, so the authors employ a speech recognition model to generate the (text) transcriptions. VALL-E significantly outperforms the state-of-the-art zero-shot TTS system (YourTTS) in speech naturalness and speaker similarity, and can preserve the speaker's emotion and the acoustic environment of the prompt.

**Background, quantization, tokenizer and encoding**: audio is typically stored as a sequence of 16-bit integer values, therefore a generative model is required to output $$2^{16}$$ = 65536 probabilities per timestep to synthesize the raw audio. Added to the high output size, its long sequence length makes it more intractable for audio synthesis. Therefore, speech **quantization** is required to compress integer values and sequence length. Common methods are $$\mu$$-law, vector quantization (HuBERT, vq-wav2vec), k-means/self-supervised method, etc.  As **audio tokenizer**, VALL-E uses a pre-trained neural audio codec model, EnCodec, a convolutional encoder-decoder model, whose input and output are both 24 kHz audio across variable bitrates. The encoder produces embeddings at 75 Hz for input waveforms at 24 kHz, which is a 320-fold reduction in the sampling rate.  Each embedding is modeled by residual vector quantization (RVQ), with eight hierarchy quantizers with 1024 entries each as shown in Figure 2.

**Model architecture:** formally speaking, $$Encodec(y) = C^{T \times 8}$$, where $$C$$ represents the two-dimensional acoustic code matrix (the 8-channel audio embeddings), and $$T$$ is the downsampled utterance length. Each row in $$C$$ represents the eight codes for a given time frame. After quantization, the neural codec decoder is able to reconstruct the waveform, i.e. $$Decodec(C) ≈ \hat{y}$$. Given an acoustic prompt matrix $$\hat{C}^{T \times 8}$$, the optimization objective of the TTS model is $$max\, p(C \mid x, \hat{C})$$, where $$x$$ is the corresponding phoneme transcription. I.e. the model learns to extract the content and speaker information from the phoneme sequence and the acoustic prompt, respectively.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/VALLE.png"/>

There are two models, that refer to the two inference steps:
1. an auto-regressive (AR) model, a transformer decoder-only architecture, conditioned on the phoneme (text) and acoustic prompt (3-second audio), that gives the discrete tokens of the audio from the first quantizer (Formula 1).
2. a non auto-regressive (NAR) model, a transformer decoder with full (non-causal) attention, that predicts the remaining 7 quantizers one level at a time, each conditioned on the text, the prompt and the previously predicted levels, for all time steps in parallel (Formula 2).

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/VALLE2.png"/>

Note to self: for the use case of synthesizing audio in a different language, i.e. that differs from the 3-sec input language and text, see [VALL-E X](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-x/).
</details>



<details> <summary markdown="span"> 2023 [High-Fidelity Audio Compression with Improved RVQGAN (DAC), Descript Inc.](https://arxiv.org/abs/2306.06546)</summary>

An audio encoder-decoder (the Descript Audio Codec, DAC) that compresses 44.1 kHz audio from all domains (speech, environment, music) with a single universal model into discrete tokens at just 8 kbps (~90× compression), outperforming competing codecs such as Meta's EnCodec. Achieved by combining advances in high-fidelity audio generation with better vector quantization techniques from the image domain, along with improved adversarial and reconstruction losses. Methods:
- to account for periodicity in audio inputs, they adopted the snake activation function for frequency $$\alpha$$ as $$snake(x) = x + \frac{1}{α} sin^2 (αx)$$.
- vanilla VQ-VAEs struggle from low codebook usage due to poor initialization, leading to a significant portion of the codebook being unused. This leads to poor reconstruction quality. To address this issue, they use two techniques: (1) factorized codes that decouple code lookup and code embedding, by performing code lookup in a low-dimensional space (section 3.2) and (2) L2-normalization of the encoded and codebook vectors converts Euclidean distance to cosine similarity, which is helpful for stability and quality.
- state-of-the-art applying quantizer dropout degrades the audio reconstruction quality at full bandwidth. To overcome it, they instead apply quantizer dropout to each input example with some probability $$p=0.5$$.
- an improved STFT discriminator  at multiple time-scales, that works better in practice and leads to improved phase modeling, compared to Encodec and Soundstream.
-  for **frequency domain reconstruction loss**, they use a mel-reconstruction loss to improve stability, fidelity and convergence speed; and multi-scale spectral losses to encourage modeling of frequencies in multiple time-scales. For **adversarial loss**, they use HingeGAN.  For **codebook learning**, they use commitment losses with stop-gradients from the original VQ-VAE formulation. All these losses are weighted to sum up to the final loss.
</details>


<details> <summary markdown="span"> 2023 [Llama 2: Open Foundation and Fine-Tuned Chat Models, Meta](https://arxiv.org/abs/2307.09288)</summary>

Llama 2 is a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters, pretrained on 2T tokens. Llama 2-Chat is a finetuned LLM optimized for dialogue use cases. The models outperform open-source chat models on most benchmarks, and based on
human evaluations for helpfulness and safety, the authors argue they may be a suitable substitute for closed-source models. Results on safety human evaluation for Llama 2-Chat are presented in Figure 3. The train dataset is only publicly available sources, which does not include data from Meta’s products or services, or sources that may include users' personal information. Table 2 presents the GPU compute hours, power consumption and carbon emissions of each model.

The pretraining setting and model architecture are adopted from Llama 1, i.e. bytepair encoding (BPE), pre-normalization via RMSNorm, SwiGLU activations, rotary positional embeddings, AdamW optimizer, cosine learning rate scheduler. However, the primary architectural differences from Llama 1 include **increased context length** (4K tokens) and **grouped-query attention (GQA)** (used in the 34B and 70B models).

The finetuning was performed with supervised fine-tuning (Section 3.1), initial and iterative reward modeling (Section 3.2.2) and RLHF (Section 3.2.3). As drawback of RLHF, "initial RLHF models tended to forget the initial instruction after a few turns of dialogue (Figure 9, below, left). To address these limitations, we propose **Ghost Attention (GAtt)**, a very simple method inspired by Context Distillation (Bai et al., 2022b) that hacks the fine-tuning data to help the attention focus in a multi-stage process". In GAtt, a system instruction (e.g., "act as ...") is synthetically concatenated to all user messages of a multi-turn dialogue and responses are sampled with the latest RLHF model; for fine-tuning, the instruction is then dropped from all but the first turn and the loss is set to zero on the tokens of previous turns, so the model learns to keep following the instruction across turns (Figure 9, below, right).

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="65%" height="65%" src="/assets/publications/llama2_gatt.png"/>
</details>


<details> <summary markdown="span"> 2023 [LLaMA: Open and Efficient Foundation Language Models, Meta](https://arxiv.org/abs/2302.13971)</summary>

LLaMA is a collection of Large Language Models (LLM) with 7B to 65B parameters trained on publicly available datasets only: LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B. Rather than following the Chinchilla compute-optimal recipe, it trains smaller models on more tokens, optimizing for inference cost. The datasets used for the pre-training data are presented in Table 1, with training hyperparameters in Table 2. Strings are tokenized using the bytepair encoding (BPE) algorithm, with circa 1.4T tokens after tokenization.

The models architecture is made of several improvements over the original Transformer:
- **Pre-normalization [GPT3]:** training stability is improved with RMSNorm normalization at the input of each transformer sub-layer, instead of output.
- **SwiGLU activation function [PaLM]:** ReLU activation is replaced with SwiGLU to improve performance, with a dimension of $$\frac{2}{3} 4d$$ instead of $$4d$$ as in PaLM.
- **Rotary Embeddings [GPTNeo]:** positional embeddings are replaced by rotary positional embeddings (RoPE) at each layer of the network. 
- **Optimization** performed with AdamW optimizer with $$β_1 = 0.9$$, $$β_2 = 0.95$$ and $$eps = 10^{−5}$$.
- **Cosine learning rate schedule** with a warmup of $$2000$$ steps, a weight decay of $$0.1$$, a gradient clipping of $$1.0$$ and a final learning rate of 10% of the maximum learning rate.
- **Efficient causal multi-Head attention** achieved by not storing the attention weights and not computing the key/query scores that are masked due to
the causal nature of the language modeling task.
- **Activation checkpointing** was implemented to reduce memory. Yet it required manually implementing the Pytorch backward propagation function for the Transformer (instead of PyTorch autograd). This also required model and sequence parallelism (following Korthikanti et al., 2022, to reduce memory enough that only the expensive activations need to be saved).
- **Overlap of the computation of activations and the communication between GPUs** over the network, to reduce latency.   

With these, the 65B model processes ~380 tokens/sec/GPU on 2048 A100-80GB GPUs, i.e. about 21 days of training for the 1.4T-token dataset.
</details>


<details> <summary markdown="span"> 2023 [Sparks of Artificial General Intelligence: Experiments with an early version of GPT-4, Microsoft](https://arxiv.org/abs/2303.12712)</summary>

A summary paper reporting early results of the experiments with GPT-4 when it was still in active development by OpenAI. The authors "demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4’s performance is strikingly close to human-level performance". The bulk of the paper contains dozens of examples that compare GPT-4 and ChatGPT and demonstrate that GPT-4 surpasses ChatGPT in performance, in code generation, music generation (output as ABC notation), drawings (SVG, TIKZ), and mathematical resolutions (LaTeX). As weaknesses, besides the regular hallucinations it was also observed:
- Incapacity of planning correctly, when planning is not a linear path.
- Occasional arithmetic mistakes on long expressions (without tool use), highlighting limits in exact computation.
- Trained on past information only, without real-time/temporal awareness.
- lack of rigorous algorithms e.g. `What is the 11th letter of "abacadab"?  …  the 11th letter is "b."`
- Analogical reasoning can reproduce social stereotypes present in training data.

Some of these can be mitigated by letting the model call external tools (e.g., a search engine, a calculator or other APIs) from within the prompt.
</details>


<details> <summary markdown="span"> 2023 [Retentive Network: A Successor to Transformer for Large Language Models, Microsoft and Tsinghua University](https://arxiv.org/abs/2307.08621)</summary>

(note: a simpler summary video of RetNet can be found [here](https://www.youtube.com/watch?v=JaIL1VAEwZ8))

RetNet is a multi-scale retention mechanism to substitute multi-head attention in Transformers, which has three computation paradigms:
- parallel framework, for training parallelism that utilizes GPU devices fully.
- recurrent framework for low-cost $$O(1)$$ inference, which improves decoding throughput (8.4x improvement over Transformer), latency (15.6x), and GPU memory (3.4x) without sacrificing performance, on Figure 1.
- a chunkwise recurrent representation that can perform efficient long-sequence modeling with linear complexity,  where each chunk is encoded parallelly while recurrently summarizing the chunks. It allows encoding each local block for computation speed while recurrently encoding the global blocks to save GPU memory.

Retentive network (RetNet) is a stack of $$L$$ identical blocks, which follows a similar layout (i.e.,
residual connection, and pre-LayerNorm) as in Transformer. Each RetNet block contains
two modules: a multi-scale retention (MSR) module, and a feed-forward network (FFN) module.  The MSR module calls the tokens in a sequence in an auto-regressive manner. The input vector is first created as $$X_0$$ in the shape of sequence length by hidden domain size. Then we calculate contextualized vector representations $$X_n$$ for each layer of the RetNet. Retention heads can be computed in **three equivalent forms**:
1. in the **parallel representation**, where $$Retention(X) = (Q K^\intercal \odot D) V$$, similar to the transformer but with an extra causal-decay matrix $$D$$, with $$D_{nm} = γ^{n-m}$$ for $$n ≥ m$$ and 0 otherwise (Eq. 5). This is beneficial for parallel training.
2. in the **recurrent representation**, it is written as a recurrent neural net (RNN) which is beneficial for inference: $$S_n = γ S_{n-1} + K_n^\intercal V_n$$ and $$Retention(X_n)=Q_n S_n$$, where the state $$S_n$$ depends on the previous term $$S_{n-1}$$.
3. a **hybrid form** combining the previous two representations is also possible to accelerate training on large sequences. Input sequence is divided into chunks. Within each chunk, the computation is performed in the parallel representation. Cross-chunk information is passed in the recurrent representation.

Finally, the model uses $$h = d_{model}/d$$ retention heads in each layer, where $$d$$ is the head dimension. The heads use different parameter matrices $$W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$$ and scalar $$γ$$ per head. The overall architecture for a given layer $$l$$ of the RetNet is then $$Y_l = MSR(LayerNorm(X_l)) + X_l$$ and $$X_{l+1} = FFN(LN(Y_l)) + Y_l$$, ie similar to a regular transformer but replacing the attention by a retention head.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="60%" height="60%" src="/assets/publications/RetNet.png"/>
</details>


<details> <summary markdown="span"> 2023 [Operator Fusion in XLA: Analysis and Evaluation, UToronto](https://arxiv.org/abs/2301.13062)</summary>

Kernel fusion is one of the most important optimizations performed by [XLA](https://www.tensorflow.org/xla). This paper (a course-project report) details XLA and key compiler passes of XLA's source code. It also presents the speedup that kernel fusion can deliver, and what low-level effects it has on hardware: using a JAX CartPole reinforcement-learning environment as a case study, they show how XLA makes fusion decisions in practice and implement fusion strategies that reach up to 10.56× speedup over their baseline implementation.
</details>


<details> <summary markdown="span"> 2023 [LongNet: Scaling Transformers to 1,000,000,000 Tokens, Microsoft and Xi’an Jiaotong University](https://arxiv.org/abs/2307.02486)</summary>

LongNet is a Transformer variant that can scale the sequence length up to 1B tokens, and without sacrificing the performance on shorter sequences. This overcomes current limitations of attention size in regular transformers, that requires a tradeoff between computational complexity and the model expressivity. The main trick is based on the **dilated attention**, which splits the sequence into segments and sparsifies each segment with a dilation rate, mixing several (segment length, dilation rate) pairs that grow geometrically, so that the attention allocation decreases exponentially as the distance between tokens grows. This gives linear computational complexity and a logarithmic dependency path between any two tokens, and the sequence dimension can be distributed across devices to train on extremely long sequences.
</details>


<details> <summary markdown="span"> 2023 [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)</summary>

"FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. We propose FlashAttention-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. These yield around 2× speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations. We empirically validate that when used end-to-end to train GPT-style models, FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization)."
</details>


<details> <summary markdown="span"> 2023 [Flow Matching for Generative Modeling, Meta AI & Weizmann Institute (ICLR 2023)](https://arxiv.org/abs/2210.02747)</summary>

Flow Matching is a simulation-free approach for training CNFs (Continuous Normalizing Flows) that is compatible with a general family of Gaussian
probability paths for transforming between noise and data samples, as required by the reverse process in diffusion models. "Furthermore, Flow Matching opens
the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement
interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization". See a good explanation in this [Cambridge ML group post](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html).
</details>


<details> <summary markdown="span"> 2022 [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, UT Austin](https://arxiv.org/abs/2209.03003)</summary>

Rectified flows aim at reducing the number of steps when transitioning between two distributions. This is important for e.g. diffusion models where we perform inference by performing $$T$$ sampling steps and we want to do it in fewer steps.  The rectified flow is an ODE model that transports distribution $$π_0$$ to $$π_1$$ by following straight line paths as much as possible, learned with a simple least-squares regression of the velocity field onto the direction $$X_1 − X_0$$ between paired samples. 
The straight paths are preferred both theoretically because it is the shortest path between two end points, and computationally because it can be exactly simulated without time discretization. Recursively re-training on the model's own couplings ("reflow") straightens the paths further, so that a single Euler step (optionally after distillation) already gives good samples; the same formulation later became the training objective of Stable Diffusion 3.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/rectified_flow_1.png"/>

 
{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/rectified_flow_2.png"/>
</details>


<details> <summary markdown="span"> 2022 [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($$\mu$$Transfer), Microsoft](https://arxiv.org/abs/2203.03466)</summary>

Under the [Maximal Update Parametrization (muP)](https://arxiv.org/abs/2011.14522), many optimal hyper-parameters remain stable as the model width changes: "When (layer) width is large, every activation vector has roughly iid coordinates, at any time
during training. Using Tensor Programs, we can recursively calculate such coordinate distributions, and consequently understand how the neural network function evolves".

With that in mind, here they propose a hyper-parameter tuning paradigm called muTransfer: "parametrize the target model in muP, tune the HP indirectly on a smaller model, and zero-shot transfer them to the full-sized model". By transferring from a 40M-parameter proxy, they outperform the published numbers of the 6.7B GPT-3 with a tuning cost of only 7% of the total pretraining cost; transferring from a 13M-parameter model beats the published BERT-large (350M) numbers.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/muTransfer.png"/>

{: style="text-align:center; font-size: small;"}
Figure 1: Training loss against learning rate on Transformers of varying $$d_{model}$$ trained with Adam. Conventionally and in contrast with our technique, different widths do not share the same optimal hyperparameter; wider networks do not always perform better than narrower ones; in fact they underperform the same-width networks in our technique even after tuning learning rate (see dashed line).

Hyperparameters that can be µTransferred, that cannot, and the dimensions µTransfer works across, with a few caveats discussed in Section 6.1. * means empirically validated only on Transformers, while all others additionally have theoretical justification.
- µTransferable: optimization related (learning rate, momentum, Adam beta, LR schedule, etc), init (per-layer init variance), parameter multipliers (multiplicative constants after weight/biases, etc), etc
- Not µTransferable: regularization  (dropout, weight decay, etc)
- µTransferred Across: width, depth*, batch size*,  training time*, seq length*
</details>


<details> <summary markdown="span">2022 [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale, Microsoft (ICML 2022)](https://arxiv.org/abs/2201.05596)</summary>

DeepSpeed-MoE is an influential MoE system that attacks the MoE-specific bottlenecks (**token routing**, **all-to-all communication**, model size and inference cost) from three sides: model design (Pyramid-Residual MoE, PR-MoE, with more experts in the later layers and a fixed dense MLP plus a residual expert per layer), compression (Mixture-of-Students distillation), and an inference system that combines expert, tensor and data parallelism with hierarchical all-to-all communication and fused sparse kernels.

How it compares to later MoE system lines (MegaScale-*):
- DeepSpeed-MoE provides a general MoE framework and many practical optimizations, but later production-scale systems often introduce more specialized parallelism strategies and communication libraries to handle extreme scale.
- DeepSpeed-MoE is also representative of approaches that keep collectives “standard” (NCCL-style), whereas newer kernel-level work explores in-kernel collectives and fine-grained overlap.

**Results / takeaways:** MoE gives a ~5× training-cost saving over a quality-equivalent dense model for autoregressive LMs; PR-MoE plus Mixture-of-Students reduce the MoE model size by up to 3.7×; and the inference system gives 7.3× better latency and cost than existing MoE inference solutions—up to 4.5× faster and 9× cheaper inference than quality-equivalent dense models. It helped popularize MoE as a practical path to scaling model capacity without proportional compute.

</details>



<details> <summary markdown="span">2022 [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale, Microsoft DeepSpeed (SC 2022)](https://arxiv.org/abs/2207.00032)</summary>

Because inference typically runs at small batch sizes, inference kernels must achieve high memory bandwidth utilization and high compute utilization at small batch sizes, whereas training kernels simply need to achieve high compute utilization at much larger batch sizes. This makes developing inference kernels quite challenging.
DeepSpeed Inference consists of two components:
1. DeepSpeed Transformer:  DeepSpeed Inference can automatically scale a dense transformer model to multiple devices by partitioning transformer operators across multiple devices while also adding appropriate communication operations needed across GPUs. Under the hood, it leverages the single GPU kernels to maximize per GPU memory bandwidth utilization, while using NCCL all-reduce collectives to perform the necessary across GPU communication. This allows DeepSpeed Inference to achieve **excellent aggregate memory bandwidth utilization across several GPUs within a node**. However, **tensor slicing can not be scaled efficiently beyond a single node due to significant communication overhead. Thus to further scale to multi-node systems, DeepSpeed Inference uses pipeline parallelism**. DeepSpeed Transformer includes three transformer modules:
  - single-GPU transformer kernels for minimizing latency and maximizing throughput via memory-bandwidth-centric fusion schedules and GeMM kernels (Sec. III). This is described below.
  - A many-GPU dense transformer inference system that combines tensor-parallelism to minimize latency with inference optimized pipeline parallelism schedules and memory optimizations to maximize throughput (Sec. IV). The model and pipeline parallelism techniques are then applied on top of the single GPU kernels.
  - A massive-GPU sparse (MoE) model inference system that combines: i) expert, data, and tensor parallelism, ii) novel communication optimizations and iii) sparse kernel optimizations to scale sparse inference on trillions of parameters across hundreds of GPUs (Sec. V). Expert parallelism is also introduced, where all-to-all can happen within just the subset of devices that share the same tensor-slicing rank, since the data across tensor-parallel ranks are replicated. The sparse tensor representation in the gating function and sparse einsum operators introduce a significant latency overhead, optimized with kernel fusion.
2. ZeRO-Inference that leverages CPU, NVMe and GPU memory along with GPU compute to make massive model inference accessible with limited resources (Sec. VI). An important design decision is how to apportion GPU memory among model weights, inference inputs, and intermediate results. One approach is to pin as much of the model weights as possible into GPU memory, and fetch the remainder (from DRAM or NVMe) when needed for computation. The big downside is that this leaves room only for small batch sizes. ZeRO-Inference adopts a different approach that pins the model weights either in DRAM (if large enough) or NVMe, and streams each layer into GPU memory for computation when needed.

The single GPU transformer kernels introduce two techniques: 
1. **Deep-Fusion** to reduce kernel-invocation and data-movement overheads by fusing multiple kernels beyond element-wise operation. The rationale is: on GPU, if data produced by a thread-block is consumed by a different one, a global memory synchronization is needed which invokes a new kernel. To avoid the need for a global synchronization, Deep-Fusion tiles the computation-space along dimensions of the iteration space which incur no cross-tile data-dependencies and executes them in parallel across different thread-blocks. The dimensions of the computation-space which does contain data dependencies are not tiled, and instead processed by the same thread-block. After this tiling, two operators can be fused using DeepFusion if each tile of the second operator depends on exactly one output tile of the first operator.  Deep-Fusion can fuse not only element-wise operations but also reductions, data transpositions, and GeMMs as long as there are no cross-tile dependencies.
2. a **Custom GeMM implementation**  designed to be fusable with Deep-Fusion while achieving maximum memory bandwidth utilization. "We first tile the computation along the output dimension. That allows us to implement GeMM using a single kernel by keeping the reduction within a tile" Then, with the aforementioned tiling strategy, each warp in a thread block is responsible for producing a partially reduced result for a tile of outputs and a final reduction is needed across all the warps within the thread block. To avoid having to reduce the partial results in shared memory, we perform a single data-layout transpose in
shared memory such that partial results of the same output element are contiguous in memory, and can be reduced by a single warp using cooperative-group collectives directly in registers. Finally, we also transpose the weight matrix during initialization such that M rows for each column are contiguous in memory. We fuse the operations inside a transformer layer at four main regions: 1) the QKV GeMM and input layer-norm, 2) transposition plus attention, 3) post-attention layer-norm and intermediate GeMM, and 4) bias and residual addition.

In the context of fusion-heavy related work, DeepSpeed-Inference aggressively fuses *compute* (Deep-Fusion, custom GeMMs) but keeps **communication as NCCL collectives outside** of those kernels—a useful baseline for what strong kernel engineering and runtime scheduling achieve *without* changing the underlying communication model.

**Results / takeaways:** up to 7.3× lower latency than the state of the art for latency-oriented scenarios and over 1.5× higher throughput for throughput-oriented ones; real-time trillion-parameter inference on hundreds of GPUs; and, with ZeRO-Inference, 25× larger models than GPU-only solutions at 84 TFLOPS (over 50% of A6000 peak).

</details>


<details> <summary markdown="span"> 2022 [DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation (EACL 2023)](https://arxiv.org/abs/2210.07558)</summary>

LoRA blocks "suffer from two major problems: first, the size of these blocks is fixed and cannot be modified after training (for example, if we need to change the rank of LoRA blocks, then we need to re-train them from scratch); second, optimizing their rank requires an exhaustive search and effort". Dynamic LoRA (DyLoRA) addresses these two problems. "DyLoRA method trains LoRA blocks for a range of ranks instead of a single rank by sorting the representation learned by the adapter module at different ranks during training". How does it work:
- In each LoRA module, we have an up-projection ($$W_{up} ∈ R^{m×r}$$) and a down-projection matrix ($$W_{dw} ∈ R^{r×d}$$). Let’s assume that we would like to train the LoRA module to operate in the range of $$r ∈$$ Range $$[r_{min}, r_{max}]$$ where $$r_{min}$$ and $$r_{max}$$ are hyper-parameters.
- At each training step, we sample $$b$$ (a value between ranks $$r_{min}$$ and $$r_{max}$$), and truncate $$W_{dw}$$ and $$W_{up}$$ to include only $$b$$ columns/rows, accordingly. The truncated matrices are represented as $$W_{dw↓b}$$ and $$W_{up↓b}$$, and they're the ones used in this training step: $$h = W_0x + \frac{α}{b} W_{up↓b} W_{dw↓b} x$$.

They report training such dynamic, search-free models 4–7× faster than LoRA (depending on the task) without significantly compromising performance.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="100%" height="100%" src="/assets/publications/DyLoRA.png"/>
</details>


<details> <summary markdown="span"> 2022 [Efficiently Scaling Transformer Inference, Google (MLSys 2023)](https://arxiv.org/abs/2211.05102)</summary>

The paper focuses on efficient generative inference of Transformer models with large deep models, with tight latency targets and long sequence lengths. It claims its methods surpass the efficiency of [NVIDIA's FasterTransformer](https://github.com/NVIDIA/FasterTransformer). It focuses on TPU v4 and its 3D-torus network layout. 

Section 3.1 provides an analysis of collective communication. Section 3.2 analyzes parallelism in the FeedForward module.  Notation used: $$BLE_{xyz}$$ means that the last dimension $$E$$ of a tensor of logical shape $$BLE$$ is split into $$X × Y × Z$$, ie the per-chip tensor is of shape $$[B, L, E/(X × Y × Z)]$$ (omitted axis are replicated). $$F$$ is the hidden (intermediate) size of the feed-forward layer. It compares data splitting *a la Megatron* (1D weight-stationary layout, section 3.2.1) where the partition layout for weights is $$EF_{xyz}$$ and $$F_{xyz}E$$, i.e. partitioned in to $$X × Y × Z = n_{chips}$$; with a 2D weight-stationary layout along both the E and F axes (section 3.2.2), where shards are square, compute cost is the same but communication is more efficient and scalable (particularly on more than 16 chips). Section 3.2.3 describes the *XYZ-weight-gathered* approach, where "the output of each per-chip matrix multiplication must then be aggregated between chips to be used as input to the subsequent operations", however "for very large batch sizes, it is best to keep the activations fully stationary between sequential matrix multiplications, requiring that we fully transfer the weights between all chips".

Related to attention layers (section 3.3), "Multihead attention can be parallelized in essentially the same ways as a feedforward layer". The attention Keys and Values (aka the "KV cache")  incur significant memory capacity and bandwidth costs. Improving with [multi-query attention](https://arxiv.org/abs/1911.02150) reduces the size of the KV cache tensors by a factor of $$n_{heads}$$ and the time spent loading them in memory, but "removes an axis otherwise used for parallelism, so the KV cache and related computations need to be partitioned differently" in order to "minimize the memory time of repeatedly loading the KV cache that dominates the inference cost. …
The most similar partitioning layout for multiquery attention (shown in Figure 4(b)) treats the KV cache the same as in multihead attention. Even though the key and value tensors are shared across all heads, they must be replicated on each chip and the memory cost savings of multiquery attention are lost". Instead the paper proposes "a partitioning strategy for the multiquery attention where the Q, K, and V matrices are partitioned over the batch $$B$$ dimension into $$n_{chips}$$ partitions". This reduces the cost of loading the KV cache per chip by a factor of $$n_{chips}$$ but incurs additional communication cost of resharding the input activation tensors. "With the proposed partitioning layout, multiquery attention enables using larger batch sizes and sequence lengths, thereby increasing throughput in addition to the latency reduction from reduced memory time".

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="50%" height="50%" src="/assets/publications/Efficiently_Scaling_Transformer_Inference_MultiQuery.png"/>

Section 3.4 details the gains of using [GPT-J](https://en.wikipedia.org/wiki/GPT-J#cite_note-Model_Card-2)'s approach to **compute attention heads and feed forward in parallel**, also applied to [PaLM](https://arxiv.org/abs/2204.02311). For comparison, the standard formulation of the transformer block is $$ y = x + MLP(LayerNorm(x + Attention(LayerNorm(x)))) $$, whereas the parallel formulation is:
$$y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x))$$. Using the parallel formulation has only one layernorm per layer instead of two,
which reduces latency at small batch sizes. Also, some matrices can be fused which results in larger matrix multiplications that run more efficiently on accelerators. Finally, it removes  one of the two all-reduce operations in each layer.

Finally, section 3.5 discusses low-level optimization (overlapping communication and computation, etc), and section 3.6 discusses `int8` quantization. Combined, these reach a 29 ms per-token generation latency at small batch size (with int8 weights) and 76% MFU during large-batch processing of 2048-token inputs on the PaLM 540B model.
</details>


<details> <summary markdown="span"> 2022 [Random-LTD: Random and Layerwise Token Dropping Brings Efficient Training for Large-scale Transformers, Microsoft](https://arxiv.org/abs/2211.11586)</summary>

Random-LTD (random and layerwise token dropping method) is a method to reduce the training costs of very large transformer models, which skips the computation of a subset of the input tokens at all middle layers (excluding the first and last layers). Tokens are dropped in a purely random manner, thus Random-LTD requires no scoring or manual design. 

Other alternatives of token dropping are:
- attention score related metrics, where compute cost for LTD is too high since the metric has to be calculated for every layer; and
- loss-/frequency-based metrics, where  accumulated loss or frequency is used and this accumulated metric would not be changed within the same iteration (forward pass), and this makes the dropped token to be the same for different layers, making the token dependency not be captured by the MHA of middle layers.

To that extent, Random-LTD uses purely random dropping of tokens at every layer. To reduce the gradient variance introduced by random-LTD, for better training, the authors monotonically increase the kept sentence length throughout training, with a linear schedule. This method is called the **Monotonic Sequence Length Growth (MSLG)**.

Related to the learning rate, note that: random-LTD reduces the effective batch size of middle layers at the initial warmup phase, and MSLG does not reach the full length until > 2/3 of training iterations for large compute saving. Therefore, the small learning rate during warmup cannot provide efficient training "dynamics" for Random-LTD. In practice, it is necessary to increase the warmup iterations and slow down the LR decay. So the paper also introduces the **LayerToken learning rate** (appendix C), which scales the learning rate  based on the sum of consumed tokens of each layer. The point of `LayerToken` is to reduce the tuning effort for Random-LTD. 

The results compare Random-LTD with the baseline, on GPT3 models with 350M and 1.3B parameters, and a dataset of up to 300B tokens. Here, Random-LTD shows similar evaluation losses as the baseline with 1/3 less LayerToken consumption. However, an important claim is that "reiterate that the LayerToken consumption saving ratio cannot directly transfer to GPU wall-clock training time saving ratio due to the implementation/hardware"; still, on GPT-3 1.3B they report 33.3% theoretical compute savings and 25.6% wall-clock savings with similar zero-shot accuracy. On BERT and ViT models, similar results are shown. When compared with TokenBypass (a different technique that skips middle layers tokens), Random-LTD shows better train and validation perplexity. Also, MSLG shows better perplexity than a constant-drop rate (table 6.4).

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="60%" height="60%" src="/assets/publications/Random_LTD.png"/>
</details>


<details> <summary markdown="span"> 2022 [The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models, Microsoft (NeurIPS 2022)](https://openreview.net/forum?id=JpZ5du_Kdh)</summary>

This paper investigates and demonstrates the importance of sequence length in GPT models. The paper presents a set of experiments on a GPT2 model on public datasets to study the **stability-efficiency dilemma**: "increasing the batch sizes and learning rates (commonly used for faster training) leads to better training efficiency but can also result in training instability, leading to poor generalization accuracy or failed runs". The paper finds that **there is a strong correlation between training instability and extreme values of gradient variance** and **long sequence lengths contribute to these extreme gradient variance values, especially at the beginning of the training** (and this could be a source of training instability).

The paper presents the **Sequence Length Warmup (SLW)** method (which starts
training with short sequences and gradually increases the length)  that aims to solve the training stability-efficiency dilemma by avoiding extreme gradient variance values. This method can be understood as a type of curriculum learning (CL), which presents easier/simpler examples earlier during training and gradually increases the sample difficulties. However here, the work aims to achieve both efficient convergence and better stability by enabling stable training with more aggressive hyperparameters, instead of keeping them constant as in the traditional CL.

Results on GPT-2 (117M and 1.5B) show stable training with 8x larger batch size and 4x larger learning rate, reducing the training tokens and wall-clock time needed for the same zero-shot quality by up to 2.2x and 3.7x. On a GPT-3 (125M) model, it enables 8x larger batch size and 40x larger learning rate, retaining 99% of the zero-shot accuracy on 11 tasks using 10x less data and 17x less time.
</details>



<details> <summary markdown="span"> 2022 [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, Stanford (NeurIPS 2022)](https://arxiv.org/abs/2205.14135)</summary>

Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. The authors "argue that a missing principle is making attention algorithms IO-aware—accounting for reads and writes between levels of GPU memory". To overcome it, Flash Attention improves the attention mechanism with one that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. In practice, it tiles the square attention matrix into partial computations that can be computed in the on-chip SRAM (shared memory) instead of the GPU's global memory (HBM). It also trains Transformers faster than existing baselines (15% end-to-end on BERT-large vs. the MLPerf 1.1 record, 3× on GPT-2 and 2.4× on long-range arena).

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/FlashAttention.png"/>

FlashAttention is the canonical example of *algorithmic* fusion in attention: it avoids materializing the full \(QK^\top\) (and often avoids materializing large intermediate softmax buffers) by using a numerically stable streaming formulation. Instead of computing attention as “GEMM → softmax → GEMM” with large intermediate tensors in HBM, it tiles the computation so that data is reused from SRAM/shared memory.

How it differs from “FFN fusion + in-kernel communication” work:
- FlashAttention is primarily about **IO-aware tiling** of attention and a streaming softmax trick, not about overlapping distributed communication.
- It shows that fusion can change the *algorithmic memory complexity* (and not just launch overhead), which is why it became such a cornerstone kernel.

**Results / takeaways:** besides the speedups above, memory becomes linear in sequence length, which enables longer contexts and better models (0.7 better perplexity on GPT-2, 6.4 points on long-document classification) and the first better-than-chance Transformers on Path-X (16K) and Path-256 (64K). It established the template that later fusion work often emulates: fuse stages, tile aggressively, and never write big intermediates to HBM if you can stream them instead.

</details>


<details> <summary markdown="span"> 2022 [High Fidelity Neural Audio Compression (Encodec), Meta AI](https://arxiv.org/abs/2210.13438)</summary>

EnCodec is a neural network model for a real-time, high-fidelity, audio codec. It consists of a streaming encoder-decoder architecture with quantized latent space trained in an end-to-end fashion. For faster and simpler training, they use a single multiscale spectrogram adversary that efficiently reduces artifacts and produces high-quality samples.
Two main problems arise in lossy neural compression of audio. The first one is overfitting to a subset of audio samples, and it was overcome by using (1) a large and diverse dataset and (2) discriminator networks that serve as perceptual loss. The second problem is compressing efficiently, both in compute time and in size, solved by using residual vector quantization of the neural encoder floating-point output. The authors claim that "designing end-to-end neural compression models is a set of intertwined choices, among which at least the encoder-decoder architecture, the quantization method, and the perceptual loss play key parts". To that extent, audio quality evaluations (MUSHRA) consist of having humans listen to, compare, and rate excerpts of speech or music compressed with competitive codecs; there, EnCodec outperforms the baselines (e.g., Opus, EVS, Lyra-v2) across all evaluated bandwidths and audio domains, for both 24 kHz mono and 48 kHz stereo audio.

**Background and model:** An audio signal of duration d can be represented by a sequence $$x ∈ [−1, 1]^{C_a × T}$$ with $$C_a$$ the number of audio channels, $$T = d · f_{sr}$$ the number of audio samples at a given sample rate $$f_{sr}$$. The EnCodec model is composed of three main components:
1. an encoder network $$E$$ that inputs an audio extract and outputs a latent representation $$z$$. It's a 1D convolution followed by convolution blocks (each a residual unit plus a strided convolution for downsampling), followed by a two-layer LSTM for sequence modelling, and a final 1D convolution with $$D$$ output channels. 
2. a quantization layer $$Q$$ produces a compressed representation $$z_q$$, using vector quantization. They use Residual Vector Quantization (RVQ, Zeghidour et al. 2021) to quantize the output of the encoder. As background, general Vector Quantization consists in projecting an input vector onto the closest entry in a codebook of a given size. In this case, RVQ refines this process by computing the residual after quantization, and further quantizing it using a second codebook, and so forth.
3. a decoder network $$G$$ that reconstructs the time-domain signal, $$\hat{x}$$, from the compressed latent representation $$z_q$$. The decoder's architecture is the inverse of the encoder, using transposed convolutions instead of strided convolutions.

There are two variants of the model, targeted for the low-latency streamable setup, or a high fidelity non-streamable usage.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/Encodec.png"/>

The training objective minimizes a linear combination of the following losses:
- **reconstruction loss**, comprised of a time and a frequency domain loss term, to minimize the L1 distance between the target and compressed audio over the time domain, i.e. $$l_t(x, \hat{x}) = \| x − \hat{x} \|_1$$. For the frequency domain loss $$l_f$$ (Note: typo here, this is $$l_s$$ in the picture above, as in *spectrogram loss*), they use an averaged sum of the L1 and L2 losses between the elements of the input and output mel-spectrograms. 
- **discriminative loss**,  a perceptual loss term based on a multi-scale STFT-based (MS-STFT) discriminator, as in Figure 2.
In practice, the decoder acts as a generator in an adversarial network and Encodec includes a discriminator module (in orange), with an adversarial loss for the generator as  $$l_g(\hat{x}) = \frac{1}{K} \sum_k max (0, 1 − D_k(\hat{x}))$$, where $$K$$ is the number of discriminators. They additionally include a relative feature-matching loss for the generator, $$l_{feat} (x, \hat{x})$$, in Formula 2 (PS: where is this in Figure 1?).
The discriminators are trained with the adversarial loss $$l_d(x, \hat{x}) = \frac{1}{K} \sum_{k=1}^K max (0, 1 − D_k(x)) + max(0, 1 + D_k(\hat{x}))$$, where $$K$$ is the number of discriminators.
- **VQ commitment loss**.  To support Multi-bandwidth learning, at 24 kHz, the model is trained to support the bandwidths 1.5, 3, 6, 12, and 24 kbps by selecting the appropriate number of codebooks to keep in the RVQ step (section 3.2). At 48 kHz, it's trained to support 3, 6, 12 and 24 kbps. They add a commitment loss $$l_w$$ between the output of the encoder, and its quantized value, with no gradient being computed for the quantized value. For each residual step $$c$$, where $$C$$ is the number of residual steps (set by the target bandwidth): $$l_w = \sum_{c=1}^C \| z_c - q_c (z_c) \|_2^2$$, where $$q_c(z_c)$$ is the nearest entry in the corresponding codebook.

The authors also claim that "We introduce a loss balancer in order to stabilize training, in particular the varying scale of the gradients coming from the discriminators" and "We additionally train a small Transformer based language model (Vaswani et al., 2017) with the objective of keeping faster than real time end-to-end compression/decompression on a single CPU core." (Section 3.3) that I have skipped.
</details>


<details> <summary markdown="span"> 2022 [DeepNet: Scaling Transformers to 1,000 Layers, Microsoft Research](https://arxiv.org/abs/2203.00555)</summary>

This paper introduces a normalization function (**DeepNorm**) to modify the residual connection in Transformer, $$x_{l+1} = LN(\alpha x_l + G_l(x_l, \theta_l))$$ with a constant $$\alpha > 1$$, accompanied with a theoretically derived initialization (some weights scaled down by a constant $$\beta$$), in order to stabilize extremely deep Transformers.
- Background: previous work had shown that better initialization methods improve the stability of the training of Transformer.
- DeepNorm works by introducing a new normalization function at residual connections, which has theoretical justification of bounding the model update by a constant.
- "The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DeepNorm a preferred alternative.". 
- Figure 2 shows the `deepnorm` (the normalization layer function), `deepnorm_init` (the weights initialization) and constants.
- Results: they train Transformers with up to 1,000 layers (2,500 attention and feed-forward sub-layers); on a multilingual benchmark with 7,482 translation directions, a 200-layer, 3.2B-parameter model outperforms the 48-layer, 12B-parameter state of the art by 5 BLEU points.
</details>


<details> <summary markdown="span"> 2022 [Contrastive Deep Supervision, Tsinghua University, Intel Corporation, and Xi’an Jiaotong (ECCV 2022)](https://arxiv.org/abs/2207.05306)</summary>

From the abstract: "the traditional training method only supervises the neural network at its last layer and propagates the supervision layer-by-layer, which leads to hardship in optimizing the intermediate layers. Recently, deep supervision has been proposed to add auxiliary classifiers to the intermediate layers of deep neural networks. By optimizing these auxiliary classifiers with the supervised task loss, the supervision can be applied to the shallow layers directly. However, deep supervision conflicts with the well-known observation that the shallow layers learn low-level features instead of task-biased high-level semantic features. To address this issue, this paper proposes a novel training framework named Contrastive Deep Supervision, which supervises the intermediate layers with augmentation-based contrastive learning".  The rationale is that contrastive learning can provide better supervision for intermediate layers than the supervised task loss. Contrastive learning "regards two augmentations from the same image as a positive pair and different images as negative pairs. During training, the neural network is trained to minimize the distance of a positive pair while maximizing the distance of a negative pair. As a result, the network can learn the invariance to various data augmentation, such as Color Jitter and Random Gray Scale". Contrastive Deep Supervision starts from those advancements, and optimizes the intermediate layers with contrastive learning instead of traditional supervised learning. 

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/contrastive_deep_supervision.png"/>
</details>


<details> <summary markdown="span"> 2022 [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper), OpenAI](https://arxiv.org/abs/2212.04356)</summary>

Whisper is an automatic speech recognition (ASR) system, that performs several tasks on a Speech-to-text setup. It is trained on 680K hours of multilingual and multitask supervised data collected from the web. I.e. it's a weakly supervised dataset. The authors "show that the use of such a large and diverse dataset leads to improved robustness to accents, background noise and technical language.

The Whisper architecture (section 2.2) is an encoder-decoder Transformer. Input audio is split into 30-second chunks, converted into a log-Mel spectrogram, and then passed into an encoder. A decoder is trained to predict the corresponding text caption, intermixed with special tokens that direct the single model to perform tasks such as language identification, phrase-level timestamps, multilingual speech transcription, and to-English speech translation."
Audio is re-sampled to 16,000 Hz, and an 80-channel log-magnitude Mel spectrogram representation is computed on 25-millisecond windows with a stride of 10 milliseconds. Whisper uses the same Byte-Pair Encoding text tokenizer as in GPT2 for English-only models, and a refit vocabulary for the multilingual models. 

Evaluated zero-shot, without any fine-tuning, the models generalize well to standard benchmarks, are often competitive with prior fully supervised results, and approach human accuracy and robustness; across many diverse datasets they make far fewer errors (roughly 50% fewer, per OpenAI) than supervised models that match them on LibriSpeech.
</details>


<details> <summary markdown="span"> 2022 [Emergent Abilities of Large Language Models, Google Research & Stanford (TMLR 2022)](https://openreview.net/forum?id=yzkSU5zdwD)</summary>

The paper discusses the phenomenon of **emergent abilities** of large language models. An ability is emergent if it is not present in smaller models but is present in larger models, and not extrapolated from scaling laws. *Phase transition* is the scale at which such abilities are exposed. Scale in this context may represent different compute budgets, data quality or other factors - the paper focuses not on ideal training but on the discussion of such phenomena. As a disclaimer, "model scale is not the singular factor for unlocking an emergent ability" and "as the science of training large language models progresses, certain abilities may be unlocked for smaller models with new architectures, higher-quality data, or improved training procedures".

The first analysis of emergent abilities focuses on the few-shot prompting paradigm, where outcome is emergent when a model has random performance until a certain scale, after which performance increases to well-above random. This was analysed on eight tasks across five model families (LaMDA, GPT-3, Gopher, Chinchilla, PaLM):

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/Emergent_Abilities_1.png"/>

A similar analysis with augmented prompting exposes the emergent property as related to when the model output starts having a positive effect (e.g. being able to do arithmetic only after a certain scale). A multi-step reasoning by providing a chain-of-thoughts as a sequence of intermediate steps was also analysed, and claimed to be exposed only after $$10^{23}$$ training FLOPs or approx. 100B parameters. Such scale is also required for instruction following tasks (ie new tasks without prior few-shots exemplars, and only by reading a set of instructions). Program execution tasks (a scratchpad for 8-digit addition) require $$9 \times 10^{19}$$ FLOPs or 40M parameters or larger. For model calibration (the ability of a model responding as True or False (or the correctness probability) to which questions they'll be able to predict correctly) requires $$3 \times 10^{23}$$ FLOPs or 52B parameters. It is summarized as:

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/Emergent_Abilities_2.png"/>
</details>


<details> <summary markdown="span"> 2022 [Training Compute-Optimal Large Language Models (Chinchilla), DeepMind (NeurIPS 2022)](https://arxiv.org/abs/2203.15556)</summary>

Heavily related to HPC's performance modelling applied to large language models. The authors revisit the question "Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens?" to which they present three approaches: (1) fix model sizes and vary number of training tokens; (2) vary model sizes for 9 different FLOP counts; (3) fit a parametric loss function to the final losses retrieved from the two previous approaches. Estimates come from over 400 models (70M to 16B parameters, trained on 5B to 500B tokens). 

 The main conclusion is that current large language models are significantly undertrained, as they only scaled the model size and not the data size. For compute-optimal training, the model size and number of training tokens should be scaled equally. This hypothesis is demonstrated with a "compute-optimal" model, Chinchilla (70B parameters), trained with the same compute budget as Gopher (280B) on 4× more data (1.4T tokens). Chinchilla outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on several evaluation tasks, reaching 67.5% average accuracy on MMLU (more than 7% above Gopher). 

To be compute optimal (i.e., lowest loss for a given compute budget), Kaplan et al. (2020) claims that models should not be trained to their lowest possible loss, and for a 10× increase in computational budget, the model should increase by 5.5× and the training tokens by 1.8x. In this paper, the authors defend that model size and training tokens should be scaled in equal proportions. 

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/Training_Compute_Optimal_Large_Language_Models.png"/> 

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/Training_Compute_Optimal_Large_Language_Models_2.png"/> 
</details>


<details> <summary markdown="span"> 2021 [RoFormer: Enhanced Transformer with Rotary Position Embedding (Rotary Embedding, RoPE), Zhuiyi Technology](https://arxiv.org/abs/2104.09864)</summary>

Traditional positional encoding methods, like sinusoidal or learned embeddings, struggle to generalize well to long sequences because they either: (1) Use absolute positions that are fixed and cannot model relative relationships effectively. (2) Lack a mechanism to extrapolate beyond the sequence lengths seen during training. Rotary embeddings address the first issue by encoding **relative positional information** directly into the attention mechanism, with an inter-token dependency that decays as the relative distance grows (on its own, RoPE still extrapolates poorly beyond the training length, which later motivated position interpolation, NTK-aware scaling and YaRN). "RoPE encodes the absolute position with a rotation matrix
and meanwhile incorporates the explicit relative position dependency in self-attention formulation".

Rotary embeddings modify the query (𝑄) and key (K) embeddings in self-attention. They do this by applying rotations to the embeddings based on the positions of tokens in the sequence.

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

The rotation angle $$𝜃_𝑝$$ for position $$p$$ is determined as $$𝜃_𝑝 = p \, w_i$$, where $$w_i = 10000^{-2(i-1)/d}$$ is the frequency of the $$i$$-th pair of dimensions. Important bit: $$x_1$$ and $$x_2$$ are the input $$x$$ expressed in the 2D coordinates. To handle a $$d$$-dimensional input, we split $$x$$ into pairs of dimensions $$[ (x_1, x_2), \, (x_3, x_4), \, \dots, \, (x_{d-1}, x_d)]$$, apply the 2D rotation matrix to each pair, and then concatenate the results to reconstruct the rotated vector. If $$x$$ has an odd dimensionality $$d$$, the extra dimension is often left unrotated. Because rotations compose, $$\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$$: the attention score depends only on the relative position $$n-m$$.
</details>


<details> <summary markdown="span">2021 [Self-Attention Does Not Need \(O(n^2)\) Memory, Google Research](https://arxiv.org/abs/2112.05682)</summary>

Rabe & Staats formalize the idea that self-attention can be computed without storing the full \(n \times n\) attention matrix by using a streaming / online computation of the softmax-normalized attention. This is a conceptual ancestor of several practical kernels: it gives the mathematical justification for computing attention in blocks while maintaining numerical stability and correctness. They give an algorithm that needs O(1) memory for single-query attention and O(log n) for self-attention, plus a practical accelerator implementation with O(√n) memory that runs within a few percent of standard attention; at sequence length 16384 it reduces the memory overhead of self-attention by 59× for inference and 32× for differentiation.

Why it still matters in modern systems papers:
- It provides a clean argument for why “don’t materialize \(QK^\top\)” is not just an approximation—it can be exact.
- It underpins many attention kernels that trade memory for compute and enable long-context attention to be feasible.

</details>


<details> <summary markdown="span"> 2021 [Rethinking Attention with Performers, Google, Cambridge, DeepMind and Alan Turing Institute (ICLR 2021)](https://arxiv.org/abs/2009.14794)</summary>

From the abstract: Performers are "Transformer architectures which **can estimate regular
(softmax) full-rank-attention Transformers with provable accuracy**, but using only
linear (as opposed to quadratic) space and time complexity, without relying on
any priors such as sparsity or low-rankness. To approximate softmax attention kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+)".

A clearer explanation can be found on this [google research post](https://blog.research.google/2020/10/rethinking-attention-with-performers.html):

**Bidirectional attention**, where there's no notion of past and future: by decoupling matrices $$Q′$$ and $$K′$$ used in lower rank decomposition of $$A$$ and conducting matrix multiplications in the order indicated by dashed-boxes, we obtain a linear attention mechanism, never explicitly constructing $$A$$ or its approximation:

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="75%" height="75%" src="/assets/publications/performers.jpg"/> 

{: style="text-align:center; font-size: small;"}
**Left:** Standard attention module computation, where the final desired result is computed by performing a matrix multiplication with the attention matrix $$A$$ and value tensor $$V$$. **Right:** By decoupling matrices $$Q′$$ and $$K′$$ used in lower rank decomposition of $$A$$ and conducting matrix multiplications in the order indicated by dashed-boxes, we obtain a linear attention mechanism, never explicitly constructing $$A$$ or its approximation.

**Unidirectional (causal) attention**, where tokens do not attend to other tokens appearing later in the sequence: the previous approach is modified to use prefix-sum computations, which only store running totals of matrix computations rather than storing an explicit lower-triangular regular attention matrix.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="75%" height="75%" src="/assets/publications/performers2.gif"/> 

{: style="text-align:center; font-size: small;"}
**Left:** Standard unidirectional attention requires masking the attention matrix to obtain its lower-triangular part. **Right:** Unbiased approximation on the LHS can be obtained via a prefix-sum mechanism, where the prefix-sum of the outer-products of random feature maps for keys and value vectors is built on the fly and left-multiplied by query random feature vector to obtain the new row in the resulting matrix.
</details>


<details> <summary markdown="span"> 2021 [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads (CoCoNet), Microsoft Research et al. (ASPLOS 2022)](https://arxiv.org/abs/2105.05720)</summary>

Abstract: the paper presents CoCoNet "with a DSL (Domain Specific Language) to express a program with both computation and communication. CoCoNeT contains several machine learning aware transformations to optimize a program and a compiler to generate high performance kernels. Providing both computation and communication as first class constructs allows users to work on a high-level abstraction and apply powerful optimizations, such as fusion or overlapping of communication and computation. CoCoNeT enables us to optimize data-, model-and pipeline-parallel workloads in large language models with only a few lines of code. " For example, it can fuse a ReduceScatter, the sliced computation that follows it, and an AllGather into a single FusedAllReduce operation.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="100%" height="100%" src="/assets/publications/coconet.png"/> 
</details>


<details> <summary markdown="span"> 2021 [LoRA: Low-Rank Adaptation of Large Language Models, Microsoft](https://arxiv.org/abs/2106.09685)</summary>

Low-Rank Adaptation (LoRA) is a technique used to efficiently fine-tune large machine learning models by injecting trainable low-rank parameter updates into the model's layers. It is particularly useful for adapting pre-trained models to new tasks or domains without retraining the entire model, which is computationally expensive and requires large storage resources.

LoRA focuses on decomposing parameter updates into low-rank matrices, drastically reducing the number of trainable parameters while maintaining the model's expressive power.

A small, trainable low-rank matrix $$Δ𝑊$$ is added to the original (frozen) weight matrix $$𝑊$$. The final output is $$W′=W+ΔW$$. Instead of directly learning $$Δ𝑊$$, it is factorized as $$ΔW=AB^⊤$$ where $$A$$ and $$B$$ are thin matrices of rank $$r$$ much smaller than the layer dimensions. One factor is initialised to zero (so training starts from the pretrained model), the update is scaled by $$α/r$$, and the paper applies it to the attention projections (e.g., $$W_q$$, $$W_v$$). Because $$ΔW$$ can be merged into $$W$$ after training, there is no extra inference latency. Compared to full fine-tuning of GPT-3 175B with Adam, LoRA reduces the number of trainable parameters by 10,000× and the GPU memory requirement by 3×, while matching or exceeding fine-tuning quality on RoBERTa, DeBERTa, GPT-2 and GPT-3.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="25%" height="25%" src="/assets/publications/LoRA.png"/>
</details>


<details> <summary markdown="span"> 2021 [Learning Transferable Visual Models From Natural Language Supervision (CLIP), OpenAI](https://arxiv.org/abs/2103.00020)</summary>

Motivation: state-of-the-art vision systems are trained on a fixed predetermined set of objects (labels). Additional labeled data is needed to specify any other visual concept. However, "the development of text-to-text as a standardized input-output interface has enabled task-agnostic architectures to zero-shot transfer to downstream datasets, removing the need for specialized output heads or dataset specific customization". A critical insight is that it is possible to leverage natural language as a flexible prediction space to enable generalization and transfer -- ie train a text model, and then specialize it on a non-textual task.
Natural language is able to express, and therefore supervise, a much wider set of visual concepts through its generality.
Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches in that it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot transfer. 

With that in mind, this paper introduces a neural network called CLIP (Contrastive Language–Image Pre-training) which efficiently learns visual concepts from natural language supervision. By design, the network can be instructed in natural language to perform a great variety of classification benchmarks, without directly optimizing for the benchmark’s performance, similar to the “zero-shot” capabilities.
CLIP models can then be applied to nearly arbitrary visual classification tasks.
Thus, the main keypoint is: by not directly optimizing the model for the benchmark, we show that it becomes much more representative.

In practice, CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset.
The dataset is an abundantly available source of supervision: 400 million pairs of text and respective images found across 500K queries from the internet.
For the image encoder, the authors consider 5 ResNets (ResNet-50, ResNet-101 and three EfficientNet-style scaled-up ResNet-50s with 4×, 16× and 64× the compute), with improvements such as attention pooling similar to a QKV attention, and 3 Vision Transformers. The text encoder is a transformer with masked self-attention, with Byte-Pair encoding with a 49152 vocab size. The max sequence length was capped at 76 (section 2.4).

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="100%" height="100%" src="/assets/publications/CLIP.png"/> 

The pipeline for contrastive pre-training (1) is the following:
- we pass an image through the image encoder (ViT or ResNet). Each image $$i$$ is pictured as $$I_i$$ above.
- we pass the text through the text encoder (transformer). Each text $$i$$ is pictured as $$T_i$$, analogously.
- the diagonal of the $$I_i T_j$$ matrix is the correct image/text labels (in blue).
- We use contrastive learning and train related text and images to be maximally close, and unrelated to be minimally close. In practice, we maximize the inner product of the `N` pairs that go together (the diagonal cells) and minimize the inner product of the `N^2-N` pairs that don't go together (the off-diagonal cells). We then interpret the product as a logit and then use the softmax on both directions to get the loss (ie it's a symmetric loss from text and image perspective).
  - in practice, in contrastive learning, "the cosine similarity (i.e. cosine of the angle) of these embeddings is then calculated,  scaled by a temperature parameter τ , and normalized into a probability distribution via a softmax". See figure 3 for source code.
- In practice, for each image input e.g. $$I_3$$, we get the classification distributions $$I_3 T_1$$, $$I_3 T_2$$, …, $$I_3 T_N$$.  

As you can tell from section (1) in the picture, minibatch size `N` is critical. As the minibatch approximates the entire dataset, the representations are more detailed. And the computation of the matrix $$I_i T_j$$ increases quadratically with `N`. 

During inference, in (2), we create a dataset by taking a set of labels and adding a prompt to all labels e.g. `A photo of a {label}`. We then put them through the text encoder, and that is our target set.
Then, to perform zero-shot prediction (3), we take an image, pass it through the image encoder, and get the classification distribution of that image over the prompted labels, from where we pick the top label - in the picture `A photo of a dog`.
The main point is: there was zero training needed on the entire task, the image and test datasets can be entirely different, which is a fundamental difference to regular image qualification tasks that have fixed input/output datasets. In CLIP, the model learns the fundamental structure of a language, not just how to difference classes.

Summary of results:
- Figure 2 shows that " CLIP is much more efficient at zero-shot transfer than our image caption baseline" and "although highly expressive, they found that transformer-based language models are relatively weak at zero-shot ImageNet classification."
- Prompt engineering and ensembling (Figure 4) boost zero-shot accuracy by almost 5 points on average across 36 datasets—a gain similar to using 4× more compute with the baseline zero-shot method, but "free" at inference.
- Figure 5 shows that zero-shot CLIP is competitive with fully supervised  baselines: in practice, we perform supervised learning of a ResNet model on the ImageNet dataset, then replace the last ResNet layer with a linear layer to allow it to perform a new task. This technique is called **Linear Probing** method as is based on the fact that the remaining ResNet includes a good representation of the basis. Surprisingly, even on ImageNet where ResNet was trained, the CLIP beats ResNet-50 by +1.9. On STL10, where there are only a few labeled examples per class and supervised learning is therefore very hard, zero-shot CLIP reaches 99.3%, a new state of the art without using any training examples; the largest gains over the ResNet-50 linear probe are on datasets like Stanford Cars (+28.9). Similarly, on e.g. MNIST where number of labels is reduced and there are many samples per label, ResNet beats CLIP.
- In Figure 6, they compare CLIP to few-shot linear probes: zero-shot CLIP matches the average performance of a 4-shot linear classifier trained on its own feature space, and nearly matches the best 16-shot linear classifier across publicly available models.
- Following the class count vs accuracy trade-off after linear probing from figure 5, in figure 7 they show the number of labeled examples per class a linear classifier on the same CLIP feature space requires to match the performance of the zero-shot classifier.  
- Figure 9 shows that error goes down as we increase compute and model size. They observed a lot of noise in the results so the conclusions are drawn from the average of all experiments.
- Figure 10 shows that the best CLIP models with linear probing beat state-of-the-art computer vision models in computer vision tasks, averaged across 12 and 27 datasets.
- Figure 13 shows the resiliency of the model to distribution shift: a model trained on a dataset loses a lot of performance as soon as we change the dataset (but not the labels). The accuracy gap between CLIP and ResNet increases as we shift the distribution further away from ImageNet (ImageNet sketches, adversarial images, etc.); zero-shot CLIP reduces this robustness gap by up to 75%.
- Figure 14 shows that doing linear probe on top of CLIP for a given dataset (e.g., +9.2% on ImageNet) improves accuracy massively for that dataset, but degrades mildly the accuracy on the other (distribution-shifted) datasets.
- Table 7 shows that the prompting matters, by showing that adding the label `child` to the dataset improves accuracy, dropping the percentage of non-human or crime-related label assignments dramatically. 
</details>



<details> <summary markdown="span"> 2021 [GSPMD: General and Scalable Parallelization for ML Computation Graphs, Google](https://arxiv.org/pdf/2105.04663.pdf)</summary>

( also covered on a [google blog post](https://blog.research.google/2021/12/general-and-scalable-parallelization.html) )

GSPMD (General and Scalable Parallelization for ML Computation Graphs) is an open-source, automatic, compiler-based parallelization system based on the [XLA compiler](https://www.tensorflow.org/xla). Because different model architectures may be better suited to different parallelization strategies, GSPMD is designed to support a large variety of parallelism algorithms appropriate for different use cases (e.g. data parallelism for small models, pipelining parallelism for larger models, or a combination of both).

In GSPMD, each tensor will be assigned a sharding property, either explicitly by the user as initial annotations, or by the sharding completion pass. The sharding property specifies how the data is distributed across devices. GSPMD defines three types of sharding: replicated (all devices have the same full data), tiled (a tiled sharding of the tensor, without data duplication), and partially tiled (an extension to [GShard](https://arxiv.org/abs/2006.16668), where the devices are divided into subgroups, and the tensor is tiled across subgroups but replicated within each subgroup). 

The sharding properties are user-defined with `mesh_split(tensor, device_mesh, dims_mapping)` that allows a tensor to be split across the device mesh and a mapping from each data tensor dimension (i) to an optional device mesh dimension. This simple API is general enough to express
all types of sharding, across the dimension(s) of batch, features, channels and/or others. The automatic partitioner in GSPMD is implemented as transformation/compiler passes in the XLA compiler (Section 3.5), using information about the operator (e.g. dot product is a generalized matrix multiply) or using iterative methods where  shardings assigned by the pass are refined incrementally over the iterations. GSPMD achieves 50% to 62% compute utilization on 128 to 2048 Cloud TPUv3 cores for models with up to one trillion parameters, and since it produces a single program for all devices, its compilation time stays constant as the number of devices grows.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="60%" height="60%" src="/assets/publications/GSPMD.png"/>

{: style="text-align:center; font-size: small;"}
**Left:** A simplified feedforward layer of a Transformer model. Blue rectangles represent tensors with dashed red & blue lines overlaid representing the desired partitioning across a 2x2 mesh of devices. **Right:** A single partition, after GSPMD has been applied. **Source**: <a href="https://blog.research.google/2021/12/general-and-scalable-parallelization.html">google research post</a>.
</details>


<details> <summary markdown="span"> 2021 [Skilful precipitation nowcasting using deep generative models of radar, DeepMind (Nature 2021)](https://www.nature.com/articles/s41586-021-03854-z)</summary>

Operational precipitation nowcasting (high-resolution forecasts up to ~2 hours ahead) typically advects radar precipitation fields using radar-based wind estimates, while numerical weather prediction, which solves the physical equations of the atmosphere, is too coarse and slow for such short lead times. These methods struggle to capture non-linear events such as convective initiation, and earlier deep-learning nowcasts become blurry at longer lead times and perform poorly on rarer medium-to-heavy rain events.
This paper demonstrates improvements in the skill of probabilistic precipitation nowcasting with DGMR, a conditional deep generative model (GAN) of radar that produces realistic, spatio-temporally consistent predictions 5–90 minutes ahead. In a systematic evaluation by more than fifty expert meteorologists, it ranked first for accuracy and usefulness in 89% of cases against two competitive methods.
</details>


<details> <summary markdown="span"> 2021 [Reduced, Reused and Recycled: The Life of a Dataset in Machine Learning Research, Google and Univ. California, NeurIPS 2021](https://arxiv.org/abs/2112.01716)</summary>

Winner of the "Datasets and Benchmarks Best Paper Award" at NeurIPS 2021. Abstract: "We study how dataset usage patterns differ across machine learning subcommunities and across time from 2015-2020. We find increasing concentration on fewer and fewer datasets within task communities, significant adoption of datasets from other tasks, and concentration across the field on datasets that have been introduced by researchers situated within a small number of elite institutions." 

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="75%" height="75%" src="/assets/publications/reduced_recycled_datasets.png"/> 
</details>


<details> <summary markdown="span"> 2021 [MLP-Mixer: An all-MLP Architecture for Vision, Google, NeurIPS 2021](https://arxiv.org/abs/2105.01601)</summary>

The paper argues that neither convolutions (CNNs) nor attention (Transformers) are necessary for computer vision setups. To that extent, it presents MLP-mixers, a Multi-Layer Perceptron only architecture. "MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information)." When pre-trained on large datasets (or with modern regularization), results are competitive with existing methods (e.g., 87.9% ImageNet top-1 when pre-trained on JFT-300M), at comparable pre-training and inference cost.
 
{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/mlp_mixer.png"/> 
</details>


<details> <summary markdown="span"> 2021 [Pay Attention to MLPs, Google, NeurIPS 2021](https://arxiv.org/abs/2105.08050)</summary>

The paper introduces gMLP (gated MLPs) and shows that they can perform as well as Transformers in language and vision applications. It claims that "self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy". In some BERT tasks it performed better than Transformers, and on finetuning tasks, it performed worse (but this can be overcome by making the gMLP model substantially larger).

The gMLPs have no self-attention, and instead rely on channel projections and spatial projections with static parameterization. It consists of a stack of $$L$$ blocks with identical size and structure. Each block is defined as:

$$
Z = σ(XU), \,\,\,\,\,\,\,\, \tilde{Z} = s(Z), \,\,\,\,\,\,\,\, Y = \tilde{Z} V
$$

where $$σ$$ is an activation function, $$U$$ and $$V$$ are linear projections along the channel dimension, and $$s(·)$$ is a layer which captures spatial interactions. When $$s$$ is an identity mapping, the above transformation degenerates to a regular FFN, ie no cross-token communication. Here, $$s(·)$$ is the Spatial Gating Unit (Section 2.1): $$Z$$ is split along channels into $$(Z_1, Z_2)$$ and $$s(Z) = Z_1 \odot (W Z_2 + b)$$, a linear projection across the token (spatial) dimension, initialised close to identity ($$W \approx 0$$, $$b = 1$$). Unlike Transformers, it does not require position embeddings because that is captured in $$s(·)$$.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/pay_attention_to_mlps.png"/> 
</details>


<details> <summary markdown="span"> 2021 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Google, ICLR 2021](https://arxiv.org/abs/2010.11929)</summary>

Introduces Vision Transformers (ViTs), an extension of the transformer architecture to images. Works by passing as input to the transformer a sequence of linear embeddings of image patches (e.g., 16×16 pixels), which also avoids the quadratic cost of letting every pixel attend to every other pixel. When pre-trained on large datasets (ImageNet-21k, JFT-300M), ViT matches or beats state-of-the-art CNNs (ResNets) on image classification while requiring substantially fewer computational resources to train; on mid-sized datasets like ImageNet alone it underperforms comparable ResNets, since it lacks the convolutional inductive biases (locality, translation equivariance).

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="70%" height="70%" src="/assets/publications/visual_transformer.png"/> 
</details>


<details> <summary markdown="span"> 2021 [Finetuned Language Models Are Zero-Shot Learners (FLAN), Google, ICLR 2022](https://arxiv.org/abs/2109.01652)</summary>

The paper presents a simple method for improving the zero-shot learning abilities of language models. It shows that instruction tuning -- finetuning language models on a collection of over 60 NLP datasets described via instructions -- substantially improves zero-shot performance on unseen tasks.
The intuition is that performing instruction tuning—finetuning of the model with datasets expressed via natural language instructions, substantially improves the zero-shot performance of the model.
For each dataset, the authors manually compose ten unique templates that use natural language instructions to describe the task for that dataset. The resulting 137B model (FLAN) surpasses zero-shot 175B GPT-3 on 20 of the 25 datasets evaluated, and even beats few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="67%" height="67%" src="/assets/publications/finetune_language_models.png"/> 
</details>


<details> <summary markdown="span"> 2020 [Scaling Laws for Neural Language Models, Johns Hopkins, OpenAI](https://arxiv.org/abs/2001.08361)</summary>

Abstract: We study empirical scaling laws for language model performance on the cross-entropy loss.
The loss scales as a power-law with model size, dataset size, and the amount of compute
used for training, with some trends spanning more than seven orders of magnitude. Other
architectural details such as network width or depth have minimal effects within a wide
range. Simple equations govern the dependence of overfitting on model/dataset size and the
dependence of training speed on model size. These relationships allow us to determine the
optimal allocation of a fixed compute budget. Larger models are significantly more sample efficient, such that optimally compute-efficient training involves training very large models
on a relatively modest amount of data and stopping significantly before convergence.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="75%" height="75%" src="/assets/publications/scaling_laws.png"/>

**Keypoints:**
- Model performance depends most strongly on scale, which consists of three factors: the number of model parameters N, the size of the dataset D, and the amount of compute C used for training. Performance has a **power-law** relationship with each of the three scale factors (Fig.1).
- Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width.
- Performance improves predictably as long as we scale up N and D in tandem,
but enters a regime of diminishing returns if either N or D is held fixed while the other increases.
- When we evaluate models on text with a different distribution
than they were trained on, the results are strongly correlated to those on the training validation set with
a roughly constant offset in the loss, i.e. incurs a constant
penalty but improves in line with the performance of the training set.
- When working within a fixed compute budget C but without any other restrictions on the model size N or available data D, we attain optimal performance by training very large models
and stopping significantly short of convergence (Chinchilla later revised this allocation: parameters and training tokens should grow in equal proportion).
- The ideal batch size for training these models is roughly a power of the loss only

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="75%" height="75%" src="/assets/publications/scaling_laws_2.png"/>
</details>


<details> <summary markdown="span"> 2020 [Language Models are Few-Shot Learners (GPT-3), OpenAI](https://arxiv.org/abs/2005.14165)</summary>

Up until now, substantial gains on many NLP tasks were achieved by pre-training on a large corpus of text followed by fine-tuning on a specific task. This method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. This paper shows that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art finetuning approaches. This paper presents and tests GPT-3 (an autoregressive LLM with 175 billion parameters, 10x more than any previous non-sparse language model) in the few-shot setup. 

**GPT-3 architecture** uses the same model and architecture as [GPT-2](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf), including the modified initialization, pre-normalization, and reversible tokenization described therein, with the exception that we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the [Sparse Transformer](https://arxiv.org/abs/1904.10509). 
- Table 2.1 includes the 8 GPT-3 models built and their sizes/hyperparameters.
- Fig. 2.2 shows the total compute used during training. Based on the analysis in [Scaling Laws For Neural Language Models](https://arxiv.org/abs/2001.08361) we train much larger models on many fewer tokens than is typical. 
- Fig 3.1 shows the pattern of smooth scaling of performance with compute. Performance (cross-entropy loss) follows a power-law trend with the amount of compute used for training. 

**Background (Fig 2.1):**
- Fine-Tuning (FT) has been the most common approach in recent years, and involves updating the weights of
a pre-trained model by training on a supervised dataset specific to the desired task.
- Few-Shot (FS) is the term we will use in this work to refer to the setting where the model is given a few
demonstrations of the task at inference time as conditioning [RWC+19], but no weight updates are allowed.
- One-Shot (1S) is the same as few-shot except that only one demonstration is allowed, in addition to a natural
language description of the task
- Zero-Shot (0S) is the same as one-shot except that no demonstrations are allowed,  and the model is only given
a natural language instruction describing the task.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/gpt3_fig21.png"/> 

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/gpt3_fig11.png"/> 

**Tasks tested and performance**:
- On NLP tasks it achieves promising results in the zero-shot and one-shot settings, and in the few-shot setting is sometimes competitive with or even occasionally surpasses state-of-the-art.
- It also displays one-shot and few-shot proficiency at tasks designed to test rapid adaption or on-the-fly reasoning,
which include unscrambling words, performing arithmetic, and using novel words in a sentence after seeing them defined only once.
- Fig 3.3 to 3.12 show that  GPT3’s performance grows with model size, suggesting that language models continue to absorb knowledge as their capacity increases. Results plotted for the TriviaQA, translation, [Winograd Schema Challenge](https://arxiv.org/abs/1907.10641), PIQA, comprehension, SuperGLUE, ANLI Round 3, arithmetic, word scrambling, and SAT tasks; in the zero-, one- and few-shot settings, respectively.
- Fig 3.13 shows that people’s ability to identify whether news articles are model-generated (measured by the ratio of correct
assignments to non-neutral assignments) decreases as model size increases.
- Fig 4.2 plots the benchmark contamination analysis. Data contamination has a minimal effect on GPT-3’s performance on most datasets, but the authors identify a few datasets where it could be inflating results (PIQA and Winograd, whose results are marked with an asterisk).
- Chapter 5 details the limitations. GPT-3 struggles with natural language inference tasks like the ANLI dataset, and some reading comprehension datasets like RACE or QuAC.
</details>


<details> <summary markdown="span"> 2019 [Graph Transformer Networks, Korea University (NeurIPS 2019)](https://arxiv.org/abs/1911.06455)</summary>

One limitation of most GNNs is that they assume the graph structure to be fixed and homogeneous, ie similar types of nodes and edges. From the abstract: "Graph Transformer Networks (GTNs) are capable of
generating new graph structures, which involve identifying useful connections
between unconnected nodes on the original graph, while learning effective node
representation on the new graphs in an end-to-end fashion. Graph Transformer layer,
a core layer of GTNs, learns a soft selection of edge types and composite relations
for generating useful multi-hop connections". 
- GTNs perform Meta-Path Generation: a meta-path defines a compositional relation over node/edge types (a sequence of relations). In the notation used in the paper, a meta-path can be written as a composed relation, e.g. $$R = t_1 \circ t_2 \circ \cdots \circ t_l$$, which defines a composite relation between nodes $$v_1$$ and $$v_{l+1}$$.
- Without any predefined, domain-specific meta-paths, GTNs achieve state-of-the-art node classification on three heterogeneous graph benchmarks (DBLP, ACM, IMDB).

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="75%" height="75%" src="/assets/publications/graph_transformer_networks.png"/> 
</details>


<details> <summary markdown="span"> 2019 [Root Mean Square Layer Normalization (RMSNorm), University of Edinburgh & University of Zurich (NeurIPS 2019)](https://arxiv.org/abs/1910.07467)</summary>
 RMSNorm (Root Mean Square Normalization) is a simpler and computationally efficient alternative to LayerNorm. Instead of normalizing based on the mean and variance of the input features, it only scales the input using the root mean square of the features. The benefits are (1) Efficiency: RMSNorm is computationally lighter than LayerNorm, making it ideal for large-scale models like LLaMA; and (2) the paper's hypothesis that LayerNorm's re-centering invariance is dispensable, and that its re-scaling invariance is what matters for training stability. In their experiments RMSNorm reaches accuracy comparable to LayerNorm while reducing running time by 7%–64% across models. Formulated as:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}
$$

where $$x$$ is the input vector with dimensionality $$d$$, and

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

where $$γ$$ is a learnable scaling parameter (similar to LayerNorm).

</details>

<details> <summary markdown="span"> 2019 [Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention), Noam Shazeer, Google](https://arxiv.org/abs/1911.02150)</summary>

Efficient training in transformers model is possible due to parallelism across the length dimension. However, decoding (where such parallelization is impossible) is slow due to continuously loading large keys and values tensors into memory. Thus, this introduces a variant of the multi-head attention that improves inference (decoding), called multi-query attention. While MHA consists of multiple attention layers (heads) in parallel with different linear transformations on the queries, keys, values and outputs,  MQA is identical except that the different heads share a single set of keys and values. This greatly reduces the size of these tensors and hence the memory bandwidth requirements of incremental decoding. This leads to a much faster decoding, with minor degradation of quality from the baseline. 
</details>


<details> <summary markdown="span"> 2019 [Generating Long Sequences with Sparse Transformers, OpenAI](https://arxiv.org/abs/1904.10509)</summary>

The paper introduces several **sparse factorizations of the attention matrix** that reduce the quadratic complexity on memory and runtime Transformers to $$O(n \sqrt{n})$$. It also allows for larger sequences. These work by separating the full attention computation into several faster attention operations which, when combined, can **approximate the dense attention** operation.  The authors claim that sparsity in attention is a natural pattern and show (by visual inspection) various examples where  most layers had sparse attention patterns across most data points, suggesting that adding sparsity to the attention would not significantly affect performance. In other layers, however, they noticed global patterns and data-dependent sparsity, whose performance could be affected by sparsity in the attention matrix. With the same architecture they set new state-of-the-art density-modeling results on Enwik8, CIFAR-10 and ImageNet-64, using hundreds of layers on sequences tens of thousands of timesteps long.

**Factorized self-attention** proposes $$p$$ separate attention heads, where each head handles a subset of the indices. The hard problem here is to find efficient choices for the subset $$A$$. Section 4.3 details 2D factorization methods via strided attention, or fixed patterns (figure below).  

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/sparse_transformers.png"/> 
</details>


<details> <summary markdown="span"> 2019 [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, Google and Toyota (ICLR 2020)](https://arxiv.org/abs/1909.11942)</summary>

ALBERT ("A Lite BERT") lowers memory consumption and increases the training speed of BERT. It allows for better scaling, establishes new record performance in several benchmarks (GLUE, RACE, SQuAD), and with fewer parameters than BERT-large. An ALBERT configuration similar to BERT-large has 18x fewer parameters and can be trained about 1.7x faster. The techniques introduced are:
1. **Factorized embedding parameterization**, for parameter reduction.
"Instead of projecting the one-hot vectors directly into the hidden space of size $$H$$, we first project them into a lower dimensional embedding space of size $$E$$, and then project it to the hidden space. By using this decomposition, we reduce the embedding parameters from $$O(V × H)$$ to $$O(V × E + E × H)$$". This separation of the size of the hidden layers from the size of vocabulary embedding, makes it easier to grow the hidden size without significantly increasing the parameter size of the vocabulary embeddings.
2. **Cross-layer parameter sharing**, for parameter reduction. The authors mention that the parameter reduction also acts as regularisation/generalisation (reduces overfitting as the model learns a representation that generalizes well for all tasks). It does not improve the performance of the model though: "This approach slightly diminishes the accuracy, but the more compact size is well worth the tradeoff". This technique prevents the number of parameters from growing with the depth of the network. As a practical example, take a BERT model with 12 layers ie 12 Transformer encoder blocks: instead of learning unique parameters for each layer, ALBERT learns the parameters of a single block and reuses it in all 12 layers. 
3. **Self-supervised loss for sentence-order prediction (SOP)**, for performance improvement. Instead of BERT's additional loss called next-sentence prediction (NSP, a binary classification loss for predicting whether two segments appear consecutively in the original text), the authors propose SOP, focused on inter-sentence coherence which is designed to address the ineffectiveness of the NSP in BERT. The SOP loss uses as positive examples the same technique as BERT (two consecutive segments from the same document), and as negative examples the same two consecutive segments but with their order swapped. This forces the model to learn finer-grained distinctions about discourse-level coherence properties.
</details>


<details> <summary markdown="span"> 2018 [Averaging Weights Leads to Wider Optima and Better Generalization (Stochastic Weight Averaging), Cornell & Samsung AI (UAI 2018)](https://arxiv.org/abs/1803.05407)</summary>

The authors present SWA, a "simple averaging of multiple points along the trajectory of SGD, with a cyclical or constant learning rate, that leads to better generalization than conventional training" and provides "much flatter solutions than SGD". The rationale is: (1) SGD with constant or cyclical LR traverses regions of weight space that correspond to high-performing networks, never reaching their central points. (2) Fast Geometric Ensembling (FGE) of $$k$$ models requires $$k$$ times more computation at test time. SWA is an approximation of FGE with the efficiency of a single model, with a better solution than SGD. The algorithm is the following: Starting from $$\hat{w}$$ we continue training, using a cyclical or constant learning rate schedule: 

- When using a cyclical learning rate we capture the models $$w_i$$ that correspond to the minimum values of the learning rate, i.e. the values at the end of each cycle (at the lowest learning rate value); 

- For *high constant* learning rates we capture models at each epoch. 

Next, we average the weights of all the captured networks $$w_i$$ to get our final model $$w_{SWA}$$. For cyclical learning rate schedule, the SWA algorithm is related to FGE, except that instead of averaging the predictions of the models, we average their weights, and we use a different type of learning rate cycle. SWA improves test accuracy over conventional SGD training on CIFAR-10, CIFAR-100 and ImageNet, with almost no computational overhead.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="90%" height="90%" src="/assets/publications/SWA.png"/> 
</details>


<details> <summary markdown="span">2018 [Group Normalization, Facebook AI Research (ECCV 2018)](https://arxiv.org/abs/1803.08494)</summary>

This paper presents Group Normalization (GN), which divides the channels into groups and normalizes the features within each group, so its computation is independent of the batch size. GN surpasses Batch Normalization particularly on small batch sizes, where BN's error increases rapidly: on ResNet-50 trained on ImageNet with a batch size of 2, GN has 10.6% lower error than BN, and with typical batch sizes it is comparably good. Layer Normalization and Instance Normalization also avoid normalizing along the batch dimension. These methods are effective for training sequential models (RNN/LSTM) or generative models (GANs), but both have limited success in visual recognition, for which GN presented better results. 

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="100%" height="100%" src="/assets/publications/group_normalization.png"/>
</details>


<details> <summary markdown="span"> 2017 [Neural Discrete Representation Learning (VQ-VAE), DeepMind (NeurIPS 2017)](https://arxiv.org/abs/1711.00937)</summary>

The Vector Quantised Variational AutoEncoder (VQ-VAE)  aims at learning **discrete (not continuous) latent space** representations without supervision.
It differs from VAEs in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static.
"During forward computation the nearest embedding $$z_q(x)$$ (equation 2) is passed to the decoder, and
during the backwards pass the gradient $$∇_z$$L is passed unaltered to the encoder. Since the output
representation of the encoder and the input to the decoder share the same $$D$$ dimensional space,
the gradients contain useful information for how the encoder has to change its output to lower the
reconstruction loss."
Equation 3 specifies the overall loss: the reconstruction loss, a codebook loss $$\| sg[z_e(x)] - e \|_2^2$$ that moves the embeddings towards the encoder outputs, and a commitment loss $$\beta \| z_e(x) - sg[e] \|_2^2$$ that keeps the encoder committed to an embedding ($$sg$$ is the stop-gradient operator), with the gradient copied straight-through across the non-differentiable mapping from $$z_e(x)$$ to $$z_q(x)$$.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="75%" height="75%" src="/assets/publications/RQVAE.png"/> 
</details>


<details> <summary markdown="span"> 2017 [Mixed Precision Training, Baidu and NVIDIA (ICLR 2018)](https://arxiv.org/abs/1710.03740)</summary>

A method for training using half-precision floating point numbers, without losing model accuracy or having to modify hyperparameters. Due to the reduced range of 16- vs 32-bit representation, three techniques are proposed to prevent the loss of critical information (numerical underflows/overflows):
1. Maintaining a single-precision copy of weights that accumulates the gradients after each optimizer step. This copy must then be rounded to half-precision for the forward- and back-propagation.
2. performing loss-scaling to preserve gradient values with small magnitudes. To implement scaling, scale the loss value computed in the forward pass prior to back-propagation, which (by the chain rule) shifts the gradient values into the FP16-representable range. Weight gradients must be unscaled before weight update to maintain the update magnitudes as in FP32 training.
3. Using half-precision arithmetic that accumulates into single-precision outputs, which are converted to half precision before storing to memory.  Different arithmetics (vector dot-products, reductions, and point-wise operations) require different treatment.

It matches FP32 accuracy across a wide range of tasks (image classification and detection, speech recognition, machine translation, language modeling, GANs) while nearly halving memory requirements.
</details>


<details> <summary markdown="span"> 2016 [Semi-Supervised Classification with Graph Convolutional Networks, University of Amsterdam (ICLR 2017)](https://arxiv.org/abs/1609.02907)</summary>

Introduces the Graph Convolutional Network (GCN), a simple layer-wise propagation rule, $$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$, with $$\tilde{A} = A + I$$ (the adjacency matrix with self-loops) and $$\tilde{D}$$ its degree matrix, derived as a first-order approximation of spectral graph convolutions. Similarly to CNNs, GCNs learn the features by aggregating information from neighboring nodes. The main difference is that CNNs are meant to operate on regular Euclidean structures (e.g. images), while GCNs generalize this to arbitrary graph structures. On semi-supervised node classification in citation networks (Citeseer, Cora, Pubmed) and a knowledge graph (NELL), it outperforms related methods by a significant margin.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="65%" height="65%" src="/assets/publications/GraphConvNets.png"/> 
</details>

<details> <summary markdown="span"> 2016 [Neural Architecture Search with Reinforcement Learning, Google, ICLR 2017](https://arxiv.org/abs/1611.01578)</summary>

Neural Architecture Search (NAS) is a subfield of machine learning that focuses on automating the design of neural network architectures. Instead of manually designing the structure of a neural network (e.g., number of layers, number of neurons per layer, type of activation functions), NAS uses algorithms to search for optimal architectures within a defined search space.

The authors propose "a recurrent network to generate the model descriptions of neural networks and train this RNN with reinforcement learning to maximize the expected accuracy of the generated architectures on a validation set." Basically, a DNN that defines the structure of another DNN using RL. The structure and connectivity of the model being designed (the **child network**) is represented as a variable-length string. This string is generated by the **controller** network - a recurrent neural network - that uses the child network's accuracy on the validation set as a reward signal. On CIFAR-10, the discovered architecture reaches 3.65% test error (0.09% better and 1.05× faster than the previous state of the art), and on Penn Treebank a newly discovered recurrent cell reaches 62.4 test perplexity (3.6 better than the previous state of the art).

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="80%" height="80%" src="/assets/publications/NAS.png"/> 
</details>


<details> <summary markdown="span"> 2015 [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, Google (ICML 2015)](https://arxiv.org/abs/1502.03167) </summary>

Batch Normalization (BatchNorm) is a technique used in deep learning to improve the training process of neural networks by normalizing the inputs to each layer within a mini-batch. It "reduces the internal covariate shift". Covariate shift means the distribution of the features differs between training and test data, breaking the i.i.d assumption used across most of ML; *internal* covariate shift is the analogous drift inside the network: as the network learns and the weights are updated, the distribution of outputs of a specific layer in the network changes. This forces the higher layers to adapt to that drift, which slows down learning. BN helps by making the data flowing between intermediate layers of the network look like whitened data, this means you can use a higher learning rate. In the results, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin; an ensemble of BN networks reaches 4.9% top-5 validation error on ImageNet, exceeding the accuracy of human raters. (Later work, e.g. Santurkar et al. 2018, argues that BN helps mostly by smoothing the optimization landscape rather than by reducing internal covariate shift.)
</details>


<details> <summary markdown="span"> 2015 [Siamese neural networks for one-shot image recognition, University of Toronto, ICML 2015 Deep Learning Workshop](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)</summary>

The paper describes **siamese neural networks** (see below for details) for efficient **one-shot learning**. General strategy. 1) Train a model to discriminate between a collection of same/different pairs; 2) Generalize to evaluate new categories based on learned feature mappings for verification.  The architecture of each siamese network is a convolutional neural network, with a flattening and a feed-forward network in the head; the two twins share weights and the prediction is a sigmoid over a weighted L1 distance between their feature vectors. The loss function is a **binary cross-entropy** with a regularizer. On Omniglot 20-way one-shot classification, it reaches 92% accuracy.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="47%" height="47%" src="/assets/publications/siamese_networks.png"/> $$\, \, \,$$ <img loading="lazy" width="47%" height="47%" src="/assets/publications/siamese_networks_2.png"/> 
</details>


<details> <summary markdown="span"> 2015 [Neural Machine Translation by Jointly Learning to Align and Translate (and Attention Mechanism), D. Bahdanau, K. Cho, Y. Bengio (ICLR 2015)](https://arxiv.org/abs/1409.0473)</summary>

An improvement over the RNN encoder–decoder for translation (Cho et al., 2014; concurrent with [Sequence to Sequence Learning with Neural Networks (Google, NeurIPS 2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)), which compresses the whole source sentence into a single fixed-length vector. Introduces the concept of attention: at each decoding step, the decoder learns to (soft-)align to and use the latent states of every encoder step (not just the last), which increases the model capabilities. It reaches translation performance comparable to the existing phrase-based system on English-to-French, and degrades much less on long sentences.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="60%" height="60%" src="/assets/publications/attention_mech.png"/> 
</details>


<details> <summary markdown="span"> 2015 [Spatial Transformer Networks, DeepMind, NeurIPS 2015](https://arxiv.org/abs/1506.02025) </summary>

A differentiable module that can be inserted anywhere in a CNN to spatially transform feature maps, conditioned on the input and learned without extra supervision. A localisation network regresses the transformation parameters (e.g., affine, projective or thin-plate spline), a grid generator produces the sampling grid, and a differentiable (e.g., bilinear) sampler warps the input. This gives the network learned invariance to translation, scale, rotation and more generic warping, yielding state-of-the-art results at the time on distorted MNIST, Street View House Numbers and CUB-200-2011 birds.

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="85%" height="85%" src="/assets/publications/STN.png"/> 
</details>


<details> <summary markdown="span"> 2015 [Cyclical Learning Rates for Training Neural Networks, US Naval Research Lab (WACV 2017)](https://arxiv.org/abs/1506.01186)</summary>

The author claims that cyclic learning rates improve time to convergence and increase the accuracy of most models (e.g., on CIFAR-10 the same 81.4% accuracy is reached in 25,000 instead of 70,000 iterations). It suggests the triangular scheduler as an efficient method with similar results to other non-triangular cyclic schedulers. The paper also provides a method to find good learning-rate bounds (the "LR range test"): run the model for a few epochs while increasing the learning rate linearly between low and high values, and pick as min and max bounds the values where accuracy starts to increase and where it plateaus or drops. Finally, it provides "rule of thumb" parameters for the triangular scheduler, e.g. a step size of 2–10× the number of iterations per epoch. 
</details>


<details> <summary markdown="span"> 2014 [Deeply-supervised Nets, UCSD and Microsoft (AISTATS 2015)](https://arxiv.org/abs/1409.5185) </summary>

A deeply supervised model in machine learning refers to a model architecture where intermediate layers are explicitly supervised during training, in addition to the supervision applied to the final output layer. This technique encourages better learning throughout the model by enforcing that earlier layers learn features useful for solving the task directly, rather than solely relying on gradients propagated from the final layer.

Deep supervision was introduced to address challenges such as vanishing gradients, poor feature learning in intermediate layers, and inefficiency in deep networks. It is particularly common in tasks like image segmentation, object detection, and biomedical image analysis.

The objective of the intermediate layers is called the "companion objective", which is used as an additional constraint (or
a new regularization) to the learning process. Example: Adding a parameter "$$γ$$ as a threshold (a hyper parameter) … with a hinge loss: once the overall value of the hidden layer reaches or is below $$γ$$, it
vanishes and no longer plays role in the learning process. … The empirical result suggests the following main properties of the
companion objective: (1) it acts as a kind of **feature regularization** (although an unusual one), which
leads to significant reduction to the testing error but not necessarily to the train error; (2) it results
in faster convergence, especially in presence of small training data". It gave state-of-the-art results at the time on MNIST, CIFAR-10, CIFAR-100 and SVHN.
</details>


<details> <summary markdown="span"> 2014 [Dropout: a simple way to prevent neural networks from overfitting, Univ. Toronto, Journal of ML Research 2014](https://jmlr.org/papers/v15/srivastava14a.html)</summary>

A method that **randomly drops neurons (in different layers) during train time, retaining each one with probability $$p$$**. For each training minibatch, a new "thinned" network is sampled. Dropout can be improved by adding max-norm regularization, decaying learning rate and high momentum. **At test time, all neurons are used, with outgoing weights multiplied by $$p$$**, which approximates averaging the exponentially many thinned networks. Dropout helps **reducing overfitting**, as the network learns to never rely on any given activations, so it learns "redundant" ways of solving the task with multiple neurons; as a side effect, hidden activations also become sparser. Dropping 20% of input units and 50% of hidden units (i.e. $$p = 0.8$$ and $$p = 0.5$$) was often found to be optimal in the original publication. It's computationally less expensive than regular model averaging of multiple trained DNNs. However, it takes 2-3 times longer to train than single fully-connected DNNs because it requires way more epochs, as parameter updates are very noisy. Because a fully connected layer occupies most of the parameters, it is prone to overfitting. Therefore, dropout **increases model generalization**. 

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="50%" height="50%" src="/assets/publications/dropout.png"/> 
</details>


<details> <summary markdown="span"> 2006 [Dimensionality Reduction by Learning an Invariant Mapping (contrastive loss), Hadsell, Chopra & LeCun, New York Uni, CVPR 2006](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)</summary>

 The paper presents Dimensionality Reduction by Learning an Invariant Mapping (DrLIM). The problem is to find a function that maps high dimensional input patterns to lower dimensional outputs, given neighborhood relationships between samples in input space. It presents the **Contrastive Loss Function**. The contrastive loss trains 2 **siamese networks**, and encourages the model to learn a representation space where similar samples are close together, and dissimilar samples are far apart.  A Siamese Network is a type of neural network architecture designed to compare two inputs by learning their similarity or relationship. It consists of two identical subnetworks (hence the name "Siamese") that share the same architecture and weights. Each subnetwork processes one of the two inputs independently, and the outputs are then combined to compute a similarity score or distance metric.  The input to system is a pair of images (one to each of the siamese networks) and the similarity label (0 for dissimilar images or 1 for similar images). The images are passed through the functions, yielding two outputs $$G(X_1)$$ and $$G(X_2)$$. The cost module then computes the Euclidean distance between both outputs as $$D_W(G_W(X_1), G_W(X_2))$$. The objective is formulated in terms of the similarity label $$y$$ (1 for similar, 0 for dissimilar; the paper itself uses the opposite convention) and the Euclidean distance $$D$$ between the two images as:

$$
L = \frac{1}{2} ⋅y⋅D^2 + \frac{1}{2} ⋅(1−y)⋅max(0,m−D)^2
$$

where $$m$$ is a margin hyperparameter that sets the minimum distance for dissimilar pairs. The architecture is a **siamese architecture**, two copies of the same network which share the same set of parameters, and a cost module.  The total gradient is the sum of the contributions from the two instances.
</details>


<details> <summary markdown="span"> 1999 [Popular Ensemble Methods: An Empirical Study, Opitz & Maclin, JAIR 1999](https://arxiv.org/abs/1106.0257)</summary>

A summary of results and conclusions on ensemble methods (bagging, boosting) on neural networks and decision trees. Bagging ensemble generally produces a classifier that is more accurate than a standard classifier. About Boosting: for a few data sets Boosting produced dramatic reductions in error (even compared to Bagging), but for other data sets it actually increases the error over a single classifier (particularly with neural networks). Alternatively, an **ensemble of similar neural networks initialized with different random seeds is surprisingly effective**, often producing results as good as Bagging. An ideal ensemble consists of highly correct classifiers that disagree as much as possible.

**Bagging trains several different models with different datapoints** randomly sampled (**with replacement**, ie same samples can be redrawn) from the same dataset.  Bagging is effective on “unstable” learning algorithms (such as neural networks) where small changes in the training set result in large changes in predictions.  

**Boosting produces a series of classifiers**. The training set used for each member of the series is **chosen based on the performance of the earlier classifier(s) in the series**. Examples that are incorrectly predicted by previous classifiers in the series are chosen more often than those correctly predicted. Thus Boosting attempts to produce new classifiers that are better able to predict examples for which the current ensemble’s performance is poor. Ada-Boosting can use the approach of (1) selecting a set of examples based on the probabilities of the examples, or (2) simply using all of the examples and weight the error of each example by the probability for that example (i.e., examples with higher probabilities have more effect on the error) -- easier as these probabilities are incorporated in the dataset. 

{: style="text-align:center; font-size: small;"}
<img loading="lazy" width="45%" height="45%" src="/assets/publications/ensemble_methods.png"/> 
</details>
 
{::options parse_block_html="false" /}
