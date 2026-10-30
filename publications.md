---
layout: post
title: Publication Summaries
permalink: /publications/
date_text: updated %b %Y
---

{::options parse_block_html="true" /}

A summary of interesting publications on machine learning models and the systems that train and serve them, newest first. Each summary gives the problem that the paper solves, its key idea, how it works and its results, and links to the related entries on this page.

<table class="label-table pub-table">
<tr>
<td><span class="post-label">2026</span></td>
<td markdown="block">
<details id="deepseek-v4">
<summary markdown="span"><span class="pub-title">DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2606.19348" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek-V4 is a preview of DeepSeek's next [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) series: **DeepSeek-V4-Pro** (1.6T parameters, 49B activated, pre-trained on 33T tokens) and **DeepSeek-V4-Flash** (284B parameters, 13B activated, 32T tokens), both with a native one-million-token context. At this length, attention and its KV cache dominate the cost of inference: the Multi-Head Latent Attention (MLA) of DeepSeek-V3 caches about 70 KB per token (see [the Insights into DeepSeek-V3 entry](#insights-deepseek-v3)), i.e. about 70 GB for a single sequence of one million tokens. The model keeps the DeepSeekMoE framework and the MTP modules of DeepSeek-V3. The MoE changes are minor: a Sqrt(Softplus) affinity function instead of Sigmoid, no limit on the number of nodes a token is routed to, and hash-routed MoE layers replacing the first dense FFN layers. DeepSeek-V4 introduces three upgrades:

1. **Hybrid attention with CSA and HCA**, interleaved across layers. *Compressed Sparse Attention* (CSA) compresses the KV entries of every $$m = 4$$ tokens into one entry (a learned, softmax-weighted combination over overlapping windows) and then applies DeepSeek Sparse Attention over the compressed entries. Its lightning indexer picks the top-k compressed entries for each query (512 in V4-Flash, 1,024 in V4-Pro). *Heavily Compressed Attention* (HCA) compresses every $$m' = 128$$ tokens into one entry and keeps dense attention over them. Both add a sliding window over the last 128 uncompressed tokens for local detail. Other details include a shared-KV layout ([MQA](#multi-query-attention)-style) with grouped output projection, partial RoPE and attention sinks.
2. **mHC** (Manifold-Constrained Hyper-Connections, see [the mHC entry](#mhc)) instead of plain residual connections: the residual stream is widened to $$n_{hc}$$ parallel streams whose mixing matrix is projected onto the [doubly stochastic matrices](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix) with [Sinkhorn-Knopp](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) iterations, which keeps signal propagation stable.
3. The **Muon** optimizer, for faster convergence and more stable training.

The headline is efficiency: at 1M tokens of context, V4-Pro needs only 27% of the single-token inference FLOPs and 10% of the KV cache of DeepSeek-V3.2 (V4-Flash: 10% and 7%). The routed-expert weights are stored in FP4, with FP4 quantization-aware training during post-training. The infrastructure is equally relevant to this page: a single fused MoE mega-kernel that overlaps computation, communication and memory access in expert parallelism, kernels written in TileLang, and batch-invariant and deterministic kernels for bitwise reproducibility. The infrastructure also includes two-stage context parallelism for the compressed attention, and a heterogeneous KV cache with on-disk storage for shared-prefix reuse.

Post-training trains per-domain specialists (supervised fine-tuning, then [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) with GRPO) and merges them into one model with on-policy [distillation](https://en.wikipedia.org/wiki/Knowledge_distillation). The maximum-reasoning mode, V4-Pro-Max, is reported as the best open model at release: ahead of GPT-5.2 and Gemini-3.0-Pro on standard reasoning benchmarks, slightly behind GPT-5.4 and Gemini-3.1-Pro. CSA thus combines the block compression of [the Native Sparse Attention entry](#native-sparse-attention) with the lightning indexer of [the DeepSeek-V3.2 entry](#deepseek-v3-2), which here selects compressed entries instead of single tokens.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="nemotron-3-super">
<summary markdown="span"><span class="pub-title">Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2604.12374" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The technical report of Nemotron 3 Super, a 120B-parameter (12B active) hybrid [Mamba-2](https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture))/attention MoE model. Its design targets the inference throughput of long-context and agentic reasoning, where the KV cache of the attention layers grows with every token. It is the first model of the Nemotron 3 family (see [the Nemotron 3 entry](#nemotron-3)) to be **pre-trained in NVFP4**, to use **LatentMoE** experts, and to include **MTP layers** for native [speculative decoding](https://en.wikipedia.org/wiki/Speculative_decoding). It was pre-trained on 25T tokens and then post-trained with SFT and RL, with a strong emphasis on agentic capabilities.

The final model supports up to 1M tokens of context and reaches accuracy comparable to GPT-OSS-120B and Qwen3.5-122B on common benchmarks, while delivering up to 2.2× and 7.5× higher inference throughput than them, respectively, with 8K input and 64K output tokens. Its MTP layers accept on average 3.45 tokens per verification step, with drafts of 7 tokens (SPEED-Bench, Table 2 in the report). The datasets and the checkpoints (base BF16, and post-trained BF16, FP8 and NVFP4) are open. For this page, it is the large-scale validation of [the LatentMoE entry](#latentmoe) and [the NVFP4 entry](#nvfp4): the accuracy-per-byte expert design and 4-bit pre-training both hold up on a 120B model trained on 25T tokens.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="flashattention-4">
<summary markdown="span"><span class="pub-title">FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling</span><span class="pub-meta">, Zadouri, ..., Tri Dao (Princeton et al.) (MLSys 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2603.05451" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024) sped up attention on Hopper (H100) GPUs by 1.5–2.0× over FlashAttention-2 (see [the FlashAttention-2 entry](#flashattention-2)). It uses the asynchronous tensor cores and memory copies (TMA) of Hopper, with warp specialization, to overlap the softmax and the data movement with the matmuls.

FlashAttention-4 (FA4) re-designs the attention kernel for NVIDIA Blackwell (B200/GB200) to confront what the authors call **asymmetric hardware scaling**. From Hopper to Blackwell, BF16 tensor-core (matmul) throughput doubled, from 4,096 to 8,192 FLOPs per clock and SM (from about 1 to 2.25 petaFLOPS per GPU). The units that do everything else did not speed up: the exponential unit (MUFU) still computes 16 exponentials per clock and SM, and the shared-memory bandwidth stayed the same. As an example, for a 128 × 128 tile with head dimension 128, the two matmuls and the 16,384 exponentials of the softmax both take 1,024 cycles on Blackwell (Table 1 in the paper). On Hopper, the matmuls took 2,048 cycles, twice as long as the exponentials. So softmax is no longer "just the thing between the two matmuls" but a bottleneck that must be carefully pipelined. FA4's whole job is to **keep the tensor cores busy by overlapping the matmuls with the now relatively slow softmax and memory work**.

The co-design has a few moving parts. As in FlashAttention-3, new forward and backward [software pipelines](https://en.wikipedia.org/wiki/Software_pipelining) overlap the tensor cores, the softmax exponential and the memory operations, but they exploit Blackwell's fully asynchronous MMA, which writes its results to the new tensor memory (TMEM) instead of registers, and its larger tile sizes. To beat the exponential bottleneck specifically, the forward pass emulates part of the exponentials (10–25% of the entries of each softmax row) in software via a polynomial approximation on the [FMA](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation) units. It also uses conditional (selective) online-softmax rescaling, which only rescales when a new max shifts the result enough to matter.

On the backward pass, where shared-memory traffic dominates, FA4 stores intermediate results in TMEM, and uses the new 2-CTA MMA mode to cut shared-memory traffic further and halve the number of global atomic adds. It is also written entirely in **CuTe-DSL** (Python), giving 20–30× faster compile times than C++ template approaches while keeping full expressivity, which lowers the barrier to prototyping new attention variants.

Results: up to 1613 TFLOPs/s on B200 in BF16, 71% of the peak (1605 in the accompanying blog post). In the forward pass, FA4 is up to 1.3× faster than cuDNN 9.13 and 2.7× faster than Triton, and in the backward pass it is faster than all baselines at long sequence lengths. FlashAttention-3 runs only on Hopper, so it is not a baseline here; some FA4 techniques were also integrated into cuDNN, from versions 9.13 and 9.14. This is the kernel-level companion to the systems papers below: where they reshape *what* runs where, FA4 squeezes the attention primitive itself to the hardware's asymmetric limits.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="multi-head-latentmoe">
<summary markdown="span"><span class="pub-title">Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism</span><span class="pub-meta">, Arizona State University (ICML 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2602.04870" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

A follow-up to [the LatentMoE entry](#latentmoe) that targets three limitations of expert parallelism (EP):
- communication volume and [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) latency grow linearly with the number of activated experts $$k$$ (each token is replicated $$k$$ times);
- expert load imbalance makes the all-to-all wait for the longest queue;
- the data-dependent traffic needs an extra all-to-all to exchange metadata.

LatentMoE reduces the bytes per token, but it still communicates *after* routing, so imbalance and non-determinism remain.

**Multi-Head LatentMoE** projects each token with a learned $$d \times d$$ matrix and splits it into $$N_h$$ sub-tokens, each processed by an independent small MoE with its own router and experts; the outputs are concatenated and projected back. **Head Parallel (HP)** exploits this structure by moving the all-to-all *before* routing: each GPU owns a subset of the heads, receives all sub-tokens for those heads, and does all routing and expert computation locally.

Each token is sent exactly once, so communication is $$O(1)$$ in $$k$$. Every GPU sends and receives the same amount of data, so traffic is perfectly balanced and needs no metadata exchange (and no out-of-memory risk from hot experts). HP also composes with EP to scale beyond $$N_h$$ GPUs. To avoid multiplying memory traffic by $$N_h$$, they add exact IO-aware kernels: an online top-k router that keeps the scores in SRAM (Triton), and expert computation expressed as block-sparse attention on top of FlexAttention.

At small scale (0.2B active, up to 4.2B total parameters, 10B FineWeb-Edu tokens), Multi-Head LatentMoE with HP trains up to 1.61× faster than a standard MoE with EP at about the same quality: the average accuracy differs by 0.13 points and the [perplexity](https://en.wikipedia.org/wiki/Perplexity) by 0.01. It also cuts inter-GPU traffic to 25% at $$k = 4$$. With doubled expert granularity, it reaches higher accuracy while still being 1.11× faster.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="in-network-collectives">
<summary markdown="span"><span class="pub-title">In-Network Collective Operations: Game Changer or Challenge for AI Workloads?</span><span class="pub-meta">, Torsten Hoefler et al. (ETH Zürich, Microsoft, AMD, Broadcom, HPE) (IEEE Computer 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2601.19132" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Data, tensor, pipeline and expert parallelism all depend on [collective operations](https://en.wikipedia.org/wiki/Collective_operation) (all-reduce, all-gather, reduce-scatter, all-to-all), and this article asks whether the network itself should compute them. **In-network computing (INC)** offloads the collectives to the [NICs](https://en.wikipedia.org/wiki/Network_interface_controller) (**edge-INC**), which forward the data without a round trip through the host or accelerator memory, or to the switches (**core-INC**, as in NVIDIA SHARP), which sum the data on its way through the network. Core-INC can halve the traffic of an all-reduce: a ring all-reduce sends and receives each segment twice, once to reduce it and once to broadcast the result, while with core-INC each node sends its data once and receives the result once. Core-INC cannot easily speed up an all-to-all, such as the MoE dispatch of [the Comet entry](#comet), as it reduces no data.

Six obstacles stand in the way: low-precision data types (the switches must forward wider partial sums, which can cancel the 2× gain), block-scaled (MX) formats, sparse reductions (partial sums get denser up the tree), bitwise reproducibility, the coordination between endpoints and switches, and encryption. Moreover, [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) limits the speedup of a whole training iteration. In the article's model of an 8 GiB all-reduce in data-parallel training, INC makes the iteration at most 34% faster, when the communication that is not overlapped takes half of the iteration, and 11% faster when it takes 20%. So the authors expect a gradual adoption, starting with deployments inside a single switch.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="latentmoe">
<summary markdown="span"><span class="pub-title">LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2601.18089" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

LatentMoE starts from a [roofline](https://en.wikipedia.org/wiki/Roofline_model) analysis of MoE inference: expert computation is bound by [HBM](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) bandwidth (weight loading) at latency-critical batch sizes, and by all-to-all communication in throughput-oriented deployments. Both costs scale with the hidden dimension $$d$$ of the model, whereas the number of active experts and the expert FFN width set the model's nonlinear budget, so $$d$$ is the dimension to shrink.

The key idea is to run the routed experts in a **latent space**, as the paper puts it:

> In LatentMoE, tokens are projected from the model hidden dimension d into a smaller latent dimension ℓ for expert routing and computation, which reduces routed parameter loads and all-to-all traffic by a factor of d/ℓ. We use this efficiency to increase the total number of experts and the top-k active experts per token by the same factor d/ℓ.

Here "routing" means token dispatch: the router's gating scores are still computed from the full token, of dimension $$d$$, and only the dispatch and combine traffic and the routed experts live in the latent space.

The weighted sum of the outputs of the routed experts (after the combine all-to-all) goes through the converse projection, from the latent dimension back to $$d$$, and is then summed with the output of the shared experts, which operate in the original hidden space. A single down-projection and a single up-projection serve all routed experts, while the weights of each expert shrink by $$d/\ell$$, which reduces the memory traffic.

The payoff is that the freed budget can be spent on *more, finer experts* at the same cost. A standard MoE (128 experts, top-6) scores 48.30% on MMLU-Pro. With a 4× compression, a LatentMoE configuration with $$(N', K') = (512, 22)$$ (as reported in the Nemotron 3 white paper, see [the Nemotron 3 entry](#nemotron-3)) reaches 52.87% on MMLU-Pro (+4.57 points) and 55.14% on code (+3.19 points). Both are 8B-active / 73B-total hybrid models trained on 1T tokens, with a similar parameter and FLOP count, and the measured per-GPU throughput of LatentMoE is within about 6% of the baseline at higher concurrency.

Validated by design-space exploration up to 95B parameters and up to 1T training tokens, LatentMoE consistently beats standard MoE on accuracy per FLOP and per parameter, and has been adopted by the flagship Nemotron 3 Super and Ultra models (see [the Nemotron 3 Super entry](#nemotron-3-super)). [The Multi-Head LatentMoE entry](#multi-head-latentmoe) builds on it to also remove the load imbalance of the all-to-all. The following picture compares a standard MoE layer (a) with a LatentMoE layer (b), in which the router and the shared expert (SE) still read the full token:

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
<details id="mhc">
<summary markdown="span"><span class="pub-title">mHC: Manifold-Constrained Hyper-Connections</span><span class="pub-meta">, DeepSeek-AI (ICML 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2512.24880" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

**[Hyper-Connections](https://arxiv.org/abs/2409.19606)** (HC) widen the residual stream into $$n$$ parallel streams mixed by learnable matrices, decoupling the residual width from the layer width at little compute cost. The gains are real, but the unconstrained mixing breaks the identity-mapping property that makes deep [residual networks](https://en.wikipedia.org/wiki/Residual_neural_network) trainable: in a 27B model, the composite mapping of the layers amplifies signals by up to about 3,000 (section 3.1 in the paper), and training shows loss spikes. HC also adds memory traffic that grows with $$n$$.

mHC constrains the stream-mixing (residual) matrix of each layer to the manifold of **[doubly stochastic matrices](https://en.wikipedia.org/wiki/Doubly_stochastic_matrix)** (the [Birkhoff polytope](https://en.wikipedia.org/wiki/Birkhoff_polytope): non-negative, with rows and columns summing to 1), using 20 iterations of [Sinkhorn-Knopp](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) row and column normalization. It also makes the read-in and write-out mappings non-negative and bounded with sigmoids. A doubly stochastic matrix has spectral norm at most 1 (the residual mixing is non-expansive) and products of such matrices stay doubly stochastic, so signal propagation stays stable at any depth: the maximum gain drops to about 1.6. With $$n = 1$$, mHC reduces to the ordinary identity residual. On the systems side, kernel fusion, recomputation and overlapping with the DualPipe communication (see [the DeepSeek-V3 entry](#deepseek-v3)) bring the cost of a 4× wider residual stream ($$n = 4$$) down to about 6.7% extra training time.

In pre-training experiments on MoE models from 3B to 27B parameters, mHC trains stably where HC spikes and improves over the plain residual baseline, with the advantage holding across compute and token budgets. On the 27B model, it scores 51.0 on BBH and 53.9 on DROP, against 43.8 and 47.0 for the baseline, and 2.1 and 2.3 points more than HC. Similarly to DeepNorm (see [the DeepNet entry](#deepnet)), mHC keeps the residual stream of a deep model stable, but by constraining the mixing matrices instead of scaling the residual. It was then adopted in DeepSeek-V4 (see [the DeepSeek-V4 entry](#deepseek-v4)).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="mirage-persistent-kernel">
<summary markdown="span"><span class="pub-title">Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs</span><span class="pub-meta">, CMU et al. (OSDI 2026)</span><a class="pub-link" href="https://arxiv.org/abs/2512.22219" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Inference with one kernel per operator pays a kernel launch and a synchronization at every operator boundary. For Qwen3-8B on a B200, the 293 kernel launches per token cost about 1.1 ms per token in eager mode, and 0.2 ms with CUDA Graphs (section 6.6 in the paper). MPK automates the megakernel idea of the Hazy Research Llama-1B post (see [the Look Ma, No Bubbles entry](#look-ma-no-bubbles)) and comes from the same group as Mirage (see [the Mirage entry](#mirage)). It is the first compiler and runtime that automatically turns multi-GPU model inference into a single persistent mega-kernel, instead of writing one by hand. The key abstraction is an **SM-level task graph** that captures data dependencies at the granularity of individual streaming multiprocessors rather than whole kernels. It enables optimizations that the kernel-per-operator model cannot express, such as cross-operator software pipelining (starting the next operator's loads while the current one computes) and fine-grained overlap of computation with inter-GPU communication.

The MPK compiler lowers a tensor program into an optimized SM-level task graph and generates CUDA code for each task. An in-kernel parallel runtime then executes the tasks inside one persistent kernel with decentralized scheduling across SMs, so there are no kernel launches or host round-trips between operators. It is open source (in the Mirage repository) and reduces end-to-end inference latency by up to 1.7× compared to existing kernel-per-operator LLM serving systems, pushing LLM inference close to the limits of the hardware.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepseek-v3-2">
<summary markdown="span"><span class="pub-title">DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2512.02556" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The formal report behind the DeepSeek-V3.2-Exp model card (see [the DeepSeek-V3.2-Exp entry](#deepseek-v3-2-exp)). Dense attention costs $$O(L^2)$$ for a context of $$L$$ tokens, which dominates long-context training and inference. The architecture is the same as V3.2-Exp: the only change from DeepSeek-V3.1-Terminus is **DeepSeek Sparse Attention (DSA)**, added through continued training.

A *lightning indexer* (a few heads, ReLU activation, runnable in FP8) scores each preceding token for the current query, and a fine-grained selection step keeps only the top-$$k$$ key-value entries ($$k = 2048$$) for the main MLA attention. This reduces the core attention cost from $$O(L^2)$$ to $$O(Lk)$$ (the indexer itself is still quadratic, but much cheaper). The indexer is first trained alone to match the dense attention distribution ([KL](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) loss) while the main model is frozen, for 1,000 steps (about 2.1B tokens), and then the whole model is trained with sparse attention, for 15,000 steps (about 944B tokens). Where [the Native Sparse Attention entry](#native-sparse-attention) selects blocks of tokens, DSA selects single tokens.

The rest of the report is about post-training: a scaled and stabilized GRPO-based RL recipe with a post-training compute budget above 10% of the pre-training cost, and a large-scale agentic task-synthesis pipeline (over 1,800 environments and 85,000 complex prompts). According to the [release note](https://api-docs.deepseek.com/news/news251201), V3.2 is the first DeepSeek model to integrate thinking directly into tool use, while the high-compute variant **DeepSeek-V3.2-Speciale** (longer reasoning) has no tool use. DeepSeek-V3.2 performs comparably to GPT-5, and Speciale reaches gold-medal level at IMO 2025, CMO 2025, IOI 2025 and the ICPC World Finals 2025, on par with Gemini-3.0-Pro in reasoning, though with clearly worse token efficiency.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="kimi-linear">
<summary markdown="span"><span class="pub-title">Kimi Linear: An Expressive, Efficient Attention Architecture</span><span class="pub-meta">, Moonshot AI (Kimi Team)</span><a class="pub-link" href="https://arxiv.org/abs/2510.26692" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Full attention keeps a KV cache that grows with every token, while linear attention keeps a fixed-size [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) state (here 128 × 128 values per head) but has trailed full attention in accuracy. Kimi Linear is a hybrid linear-attention architecture that, under matched training recipes, *outperforms* full attention across short-context, long-context and RL-style post-training tasks, which the authors claim is a first. Its core is **Kimi Delta Attention (KDA)**, a linear attention that extends Gated DeltaNet with finer-grained gating. Instead of one forget gate per head, each channel (feature dimension) gets its own decay rate, so the limited state memory is used more selectively. For hardware efficiency, KDA's transition uses a specialized Diagonal-Plus-Low-Rank (DPLR) form with a custom chunkwise-parallel algorithm that needs substantially less computation than general DPLR while staying consistent with the classical [delta rule](https://en.wikipedia.org/wiki/Delta_rule).

The model interleaves KDA with periodic full-attention MLA layers in a uniform 3:1 ratio (48B total, 3B activated parameters). With an identical training recipe (1.4T tokens), it beats a full-MLA baseline on 28 of the 34 benchmarks of Tables 3 to 5 in the paper, ties on one (MATH) and trails on five (LiveBench, EvalPlus of the instruction-tuned model, LongBench V2, Frames and Long Code Arena Commit). It also cuts the KV cache by up to 75% (only one layer in four keeps a growing cache) and reaches up to 6× decoding throughput at 1M tokens of context. The KDA kernels, a vLLM implementation, and the pre-trained and instruction-tuned checkpoints are open. Like the Mamba layers of [the Jamba entry](#jamba) and [the Nemotron 3 entry](#nemotron-3), the KDA layers replace most attention layers with a fixed-size state.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="nvfp4">
<summary markdown="span"><span class="pub-title">Pretraining Large Language Models with NVFP4</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2509.25149" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The recipe for pre-training LLMs in 4-bit floating point ([FP4](https://en.wikipedia.org/wiki/Minifloat)) with **NVFP4**, the Blackwell microscaling format: FP4 (E2M1) values in blocks of 16 elements, each block with an FP8 (E4M3) scale, plus a per-tensor FP32 scale. Compared to MXFP4 (blocks of 32 with power-of-two E8M0 scales), the smaller blocks and more precise scales capture the local dynamic range much better, at 4.5 instead of 4.25 bits per value. FP4 GEMMs run at 2× (GB200) and 3× (GB300) the math throughput of FP8 on Blackwell, but naive 4-bit training diverges or loses accuracy over long token horizons.

The method combines four ingredients:
1. **Random [Hadamard Transforms](https://en.wikipedia.org/wiki/Hadamard_transform)** (16 × 16) on the inputs of the weight-gradient GEMMs, which spread block-level outliers into a more Gaussian-like distribution;
2. **2D (16×16) block scaling for weights**, so the forward and backward passes see the same quantized weights;
3. **[stochastic rounding](https://en.wikipedia.org/wiki/Rounding#Stochastic_rounding)** for gradients, to avoid the bias of round-to-nearest;
4. keeping a small fraction of numerically sensitive linear layers in higher precision (under 15%, mostly at the end of the network).

They validate it by training a 12B hybrid Mamba-Transformer on 10T tokens, the longest publicly documented 4-bit training run at the time. The loss tracks the FP8 baseline within about 1% relative error during the stable phase (slightly above 1.5% late in the learning-rate decay), and downstream accuracies match FP8 (62.58% against 62.62% on MMLU-Pro), although some coding benchmarks lag. In a comparison on an 8B model, MXFP4 needed about 36% more tokens to reach the loss NVFP4 reached at 1T tokens. Support is in Transformer Engine, and NVFP4 pre-training was later used for Nemotron 3 Super (see [the Nemotron 3 Super entry](#nemotron-3-super)). [The FP8 training entry](#fp8-training) covers the previous step, training in 8 bits.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="insights-deepseek-v3">
<summary markdown="span"><span class="pub-title">Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures</span><span class="pub-meta">, DeepSeek-AI (ISCA 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2505.09343" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

A hardware/model co-design retrospective of DeepSeek-V3/R1 (ISCA 2025 industry track), complementary to [the DeepSeek-V3 entry](#deepseek-v3): how a 671B model (37B activated per token) was trained on only 2,048 H800 GPUs, and what hardware should change. On the model side it quantifies the choices: MLA cuts the KV cache to about 70 KB per token, versus 328 KB for Qwen-2.5 72B and 516 KB for LLaMA-3.1 405B (both GQA). The MoE design needs about 250 GFLOPs of training compute per token, versus about 2,448 for the dense LLaMA-3.1 405B. The paper also covers FP8 mixed-precision training and MTP, whose module accepts the second predicted token 80% to 90% of the time, for 1.8× more generated tokens per second.

On the infrastructure side it describes the **multi-plane two-layer [fat-tree](https://en.wikipedia.org/wiki/Fat_tree)** network (each GPU's NIC sits on its own network plane, eight planes per node), which can connect up to 16,384 GPUs with two layers of switches, at a cost per endpoint 41% below that of a three-layer fat-tree. It also explains how expert parallelism and node-limited routing (each token is sent to at most four nodes) are shaped around the H800's bandwidth asymmetry: [NVLink](https://en.wikipedia.org/wiki/NVLink) delivers about 160 GB/s in practice, about four times the 40 GB/s of [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand).

The second half is a wish list for future AI hardware, drawn from the bottlenecks they hit. It asks for more precise low-precision compute units (e.g., higher FP8 accumulation precision and native support for fine-grained block scaling), convergence of scale-up and scale-out networks, and low-latency communication fabrics. It is a good companion to the communication-overlap papers on this page, such as [the Comet entry](#comet) and [the FLUX entry](#flux), because it explains *why* those optimizations were needed on bandwidth-limited hardware.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="native-sparse-attention">
<summary markdown="span"><span class="pub-title">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</span><span class="pub-meta">, DeepSeek-AI, Peking University & University of Washington (ACL 2025 Best Paper)</span><a class="pub-link" href="https://arxiv.org/abs/2502.11089" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

NSA argues that most sparse-attention methods fail in practice for two reasons. First, their theoretical savings do not turn into wall-clock speedups (scattered token-level memory access, poor fit with GQA/MQA where heads share KV, or speedups limited to only prefill or only decode). Second, they are applied post hoc to models trained with dense attention, so the sparsity pattern is never trained end to end. NSA is designed for both: each query attends through three branches whose outputs are combined by learned gates:
1. **compressed** attention over coarse summaries of token blocks (blocks of 32 tokens with stride 16, compressed by a learned MLP);
2. **selected** attention over the top-n most relevant fine-grained blocks (16 blocks of 64 tokens), chosen by reusing the compression branch's attention scores so that selection is almost free;
3. a **sliding window** over the most recent 512 tokens for local context.

The kernel is hardware-aligned. Block selection is shared across all heads of a GQA group, so each group loads contiguous KV blocks once into SRAM and keeps the arithmetic intensity balanced on Tensor Cores in the forward pass, the backward pass and decoding.

A 27B-parameter MoE model (about 3B active) pre-trained from scratch with NSA on 260B tokens matches or beats the full-attention baseline on general benchmarks, long-context retrieval and chain-of-thought reasoning. At 64K context, a decoding step loads the KV cache of 5,632 tokens instead of 65,536 (4,096 compressed tokens, 16 selected blocks of 64 tokens and the window of 512 tokens), for an expected decoding speedup of 11.6× (Table 4 in the paper). The kernels measure 9.0× for the forward pass and 6.0× for the backward pass. Its ideas resurface in DeepSeek's later models: the learned token selection of DeepSeek Sparse Attention (see [the DeepSeek-V3.2 entry](#deepseek-v3-2)), and both the block compression and the selection in the CSA/HCA hybrid of DeepSeek-V4 (see [the DeepSeek-V4 entry](#deepseek-v4)).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="flashinfer">
<summary markdown="span"><span class="pub-title">FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving</span><span class="pub-meta">, University of Washington et al. (MLSys 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2501.01005" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

FlashInfer is the attention kernel library underneath many serving stacks (it is integrated into SGLang, vLLM and MLC-Engine). It addresses three properties of serving workloads that a single FlashAttention kernel (see [the FlashAttention-2 entry](#flashattention-2)) does not cover.

First, **KV-cache heterogeneity**: paged, prefix-shared and sparse KV layouts are all represented in one **block-sparse format**, with composable formats for shared prefixes, so the same kernels serve all of them and redundant loads of shared KV are avoided. Second, **customization**: a customizable attention template (e.g., custom masks or logits transforms) is [JIT-compiled](https://en.wikipedia.org/wiki/Just-in-time_compilation) into optimized block-sparse kernels, so new attention variants do not need hand-written CUDA. Third, **dynamism**: a load-balanced scheduler splits the work of variable-length requests across SMs (inspired by Stream-K) while staying compatible with CUDA Graphs, which require a static launch configuration.

Compared to state-of-the-art serving solutions, it achieves 29–69% lower inter-token latency than compiler backends (the Triton backend) on an LLM serving benchmark, 28–30% lower latency for long-context inference (StreamingLLM, with fused RoPE kernels), and a 13–17% speedup for LLM serving with parallel generation (MLC-Engine, at a parallel degree of 4).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="nemotron-3">
<summary markdown="span"><span class="pub-title">NVIDIA Nemotron 3: Efficient and Open Intelligence</span><span class="pub-meta">, NVIDIA</span><a class="pub-link" href="https://arxiv.org/abs/2512.20856" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Nemotron 3 (NVIDIA) is a family of open models: Nano, Super and Ultra with a context of up to 1M tokens.

The report stacks four efficiency techniques:
1. **LatentMoE** (Super/Ultra): see [the LatentMoE entry](#latentmoe).
2. **hybrid Mamba-Transformer MoE**, for inference efficiency: rather than interleaving MoE layers with expensive self-attention (which must attend over a linearly growing KV cache during generation), Nemotron 3 predominantly interleaves MoE layers with cheaper [Mamba-2](https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)) layers, which require storing only a constant state during generation. It keeps only a select few attention layers (without RoPE, since the Mamba layers already provide positional information).
3. **Multi-Token Prediction** (Super/Ultra, see [the Multi-token Prediction entry](#multi-token-prediction)): predicting several future tokens adds a richer training signal (a 2.4% relative gain of the average benchmark score, about 1.6 points, in an 8B-active MoE ablation). It also doubles as a draft for speculative decoding: a lightweight MTP module reaches ~97% acceptance on the first two predicted tokens.
4. **NVFP4 4-bit training** (Super/Ultra): stable, accurate pretraining on a hybrid Mamba-MoE for up to 25T tokens. Weights, activations, and gradients are quantized to NVFP4, so the fprop, dgrad, and wgrad GEMMs all run natively in FP4 (3× peak throughput vs FP8 on GB300). On top of the recipe of [the NVFP4 entry](#nvfp4) (2D block scaling for weights, Random Hadamard Transforms on wgrad inputs, stochastic rounding on gradients, and the last 15% of the network in high precision), sensitive layers are protected too. The QKV and attention projections (the few GQA layers use only 2 KV heads) stay in BF16, and the Mamba output projections, which flush up to 40% of values to zero in NVFP4, are kept in MXFP8. The paper reports that the relative loss gap stays under 1% vs BF16 on Nano and shrinks to under 0.6% on an 8B-active MoE, with the gap narrowing as model size grows.

The design philosophy is a balance of three layer types: MoE for sparse parameter scaling, a few attention layers for high-fidelity information routing between all tokens, and Mamba-2 for fixed-cost sequence modeling, the same balance as in [the Jamba entry](#jamba). Nemotron 3 Nano 30B-A3B achieves 3.3× higher throughput than Qwen3-30B-A3B (8K input / 16K output), while staying competitive on reasoning and long-context (RULER @ 1M) benchmarks. The following pictures illustrate the layer pattern of Nemotron 3 Nano, where a few attention layers sit among the interleaved Mamba-2 and MoE layers, and the loss gap between NVFP4 and BF16 training at two model scales:

{: .picture}
<img loading="lazy" width="82.4%" height="82.4%" src="{{ site.assets }}/publications/nemotron3.svg"/>

{: .caption}
<b>Nemotron-3-Nano-30B-A3B.</b> Nemotron 3 models (e.g., Nemotron 3 Nano) leverage a hybrid Mamba-Transformer MoE architecture consisting predominantly of interleaved Mamba-2 and MoE layers, with a select few self attention layers. Adapted from: [NVIDIA Nemotron 3: Efficient and Open Intelligence](https://arxiv.org/abs/2512.20856)

{: .picture}
<img loading="lazy" width="92.8%" height="92.8%" src="{{ site.assets }}/publications/nemotron3_2.svg"/>

{: .caption}
Relative difference in train loss (left) and validation loss (right) between models trained with NVFP4 and BF16 (higher is better), shown at two model scales: Nemotron 3 Nano (A3B) and the larger MoE model (A8B). Loss gaps decrease as model size increases (A3B → A8B). Recipe ablation on Nemotron 3 Nano started from Nemotron 3 NVFP4 checkpoint at 500B tokens, then quantizes sensitive layers (Mamba Output, QKV, and Attention projections) to NVFP4, highlighting the importance of keeping these layers in high precision. Adapted from: [NVIDIA Nemotron 3: Efficient and Open Intelligence](https://arxiv.org/abs/2512.20856)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="comet">
<summary markdown="span"><span class="pub-title">Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts</span><span class="pub-meta">, ByteDance (MLSys 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2502.19811" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Expert-parallel communication takes 47% of the forward-pass time on average for popular [MoE](https://en.wikipedia.org/wiki/Mixture_of_experts) models in Megatron-LM (on 8 H800 GPUs), so **distributed MoE** systems pipeline it with computation. Comet argues that existing schemes do it too coarsely: splitting an MoE layer into a few big chunks either leaves part of the communication exposed or shrinks the GEMM tiles enough to hurt kernel efficiency (the same tension that [the FLUX entry](#flux) identifies for tensor parallelism).

Comet works in two parts. (1) **Dependency resolving** maps out which pieces of computation depend on which pieces of communication, and schedules them into a fine-grained pipeline, so that each piece starts as soon as its data is ready. (2) **Adaptive workload assignment** splits the resources of the GPU between communication and computation, per configuration instead of with a fixed split.

On Hopper GPUs, both parts run inside one fused kernel, with **thread-block specialization**: each [thread block](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)) of the kernel either communicates or computes, not both. Because thread blocks occupy SMs, this dedicates part of the SMs to communication and the rest to computation. The computation blocks call device-side GEMM functions inside the fused kernel and, since the two kinds of blocks are isolated, reuse the GEMM implementation of the unfused kernel (built from CUTLASS templates) unchanged.

The best division point between computation and communication SMs depends on the model layer, the input shape and the level of parallelism. As the following picture illustrates, the number of thread blocks for communication that gives the shortest kernel duration (the dots) changes with the input size and with the mix of tensor and expert parallelism:

{: .picture}
<img loading="lazy" width="92.8%" height="92.8%" src="{{ site.assets }}/publications/comet.svg"/>

{: .caption}
Duration of the MoE layer1 kernel with varying number of thread blocks assigned for communication ($$n_c$$). The total number of thread blocks is identical to the number of SMs on Hopper(132). The figure shows four cases with different parallelisms. Adapted from: [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811)

As section 3.2.2 of the paper explains, "Comet’s library comprises multiple pre-compiled kernels, each with a distinct division point [of the SMs between computation and communication]. Prior to deployment, the optimal configuration for each setup is profiled and stored as metadata. During runtime, Comet utilizes this metadata to select the optimal kernel for execution."

How does it "map out" pieces of computation to communication? Between a communication operation and the computation that consumes (or produces) its data, there is a **shared tensor**: the buffer that communication writes and computation reads (dispatch), or that computation writes and communication reads (combine). Comet decomposes the shared tensor along one dimension, finds which slices of it each GEMM tile needs, and orders the traversal of the tiles so that they consume the slices in the order in which communication fills them.

As an example, for the MoE workflow dispatch→MatMul→MatMul→combine, Comet fuses dispatch→MatMul into one kernel (layer0 in the paper) and MatMul→combine into another (layer1), and executes communication and computation at the level of tiles. The following picture illustrates the two fused kernels, each a producer and a consumer around a shared tensor (top), and the rescheduling of the shared tensor of the first kernel to match the data arrival order (bottom):

{: .picture}
<img loading="lazy" width="66.8%" height="66.8%" src="{{ site.assets }}/publications/comet_2.svg"/>

{: .caption}
<b>Top:</b> The producer-consumer modeling of layer0 (left) and layer1 (right) of an MoE layer. The global size of the shared tensor is $$(M \times topk, N)$$ for both layer0 and layer1. <b>Bottom:</b> Decompose and reschedule the shared tensor in MoE layer0. In this illustration, three experts are located on Rank 0, each requiring both local and remote data for computation. Adapted from: [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811)

Results: on 8 H800 GPUs, against Megatron-LM, FasterMoE and Tutel, a single MoE layer runs 1.96× faster and end-to-end execution 1.71× faster on average. Comet is deployed in production clusters of over ten thousand GPUs, saving millions of GPU hours.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="tpla">
<summary markdown="span"><span class="pub-title">TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference</span><span class="pub-meta">, Peking University & Tencent</span><a class="pub-link" href="https://arxiv.org/abs/2508.15881" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

TPLA fixes an incompatibility between Multi-Head Latent Attention (MLA) and tensor parallelism:
- **MLA**, introduced in DeepSeek-V2 (see [the DeepSeek-V2 entry](#deepseek-v2)), compresses the key–value state into a single low-rank latent vector `cKV` and caches only that, which reduces the memory of the KV cache.
- But under Megatron-LM **tensor parallelism (TP)**, where attention heads are split across devices, every device still needs the *entire* `cKV` to compute its heads, so the cache is replicated on each GPU. That replication erodes MLA's advantage. With TP = 4, each GPU running DeepSeek-V3 caches 576 values per token and layer (a latent of 512 and a RoPE key of 64), while the Grouped Query Attention (GQA) of LLaMA-3-70B caches 512 (2 of its 8 KV heads of 128 dimensions, for keys and values). An existing alternative, Grouped Latent Attention (GLA), shards the latent but then each head only sees part of it, weakening representational capacity.

TPLA's scheme **partitions both the latent representation and each head's input dimension across devices, runs attention independently on each shard, and combines the shard outputs with an all-reduce**. The key difference from GLA: every head in TPLA still attends to the *full* latent space (the partial results are summed back together), so it keeps more of MLA's representational power while cutting the per-device KV cache. Since each shard applies its own softmax before the all-reduce, the result approximates rather than exactly reproduces MLA. Still, TPLA works on MLA-pretrained checkpoints without retraining, and applying an orthogonal transform (Hadamard or [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)) before slicing mitigates the cross-shard interference, keeping the accuracy loss minimal.

The split is asymmetric (TPLA uses MLA for prefill and TPLA for decode) because the two phases have different bottlenecks.

- Prefill: keep standard (reparameterized) MLA. Prefill processes the whole prompt at once, so it is compute-bound, not memory-bound, and does not repeatedly re-read a KV cache.

- Decode: switch to TPLA and partition the latent across devices. Decode generates one token at a time, so it is memory-bound: every step reloads the entire KV cache from HBM, and that is where MLA's full-cKV-on-every-device replication hurts. On two devices, each device caches 320 values per token and layer (half of the 512 latent values, plus the 64 RoPE values, which every device keeps) instead of 576, i.e. 1.8× less.

On DeepSeek-V3 and Kimi-K2 at 32K context, it delivers **1.79× and 1.93×** speedups while holding quality on commonsense and LongBench benchmarks, and it can be implemented with FlashAttention-3 for real end-to-end gains. Where [the TransMLA entry](#transmla) turns GQA models into MLA models, TPLA makes MLA models fit tensor parallelism. The following picture compares MLA, GLA and TPLA on two devices (orange and green), from the latent KV cache at the bottom to the attention output at the top:

{: .picture}
<img loading="lazy" width="92.4%" height="92.4%" src="{{ site.assets }}/publications/tpla.svg"/>

{: .caption}
Comparison of MLA, GLA, and TPLA. In MLA, each device must load the entire KV cache. In GLA, each attention head only accesses the portion of the KV cache stored on its own device. In TPLA, the prefilling phase follows MLA for efficiency and accuracy, while during the decoding phase, attention heads are distributed across devices, each relying on the KV cache stored locally on its assigned device. Adapted from: [TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference](https://arxiv.org/abs/2508.15881)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="pipefill">
<summary markdown="span"><span class="pub-title">PipeFill: Using GPUs During Bubbles in Pipeline-parallel LLM Training</span><a class="pub-link" href="https://openreview.net/forum?id=650J0YFjnV" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

In pipeline-parallel training, the GPUs idle in the pipeline bubbles: with $$p$$ stages and $$m$$ micro-batches, a GPipe or 1F1B schedule leaves each GPU idle for a fraction $$(p-1)/(m+p-1)$$ of the time. This is 32% for 16 stages and 32 micro-batches, and 65% when a cluster four times larger splits the same global batch into 8 micro-batches per pipeline. PipeFill improves GPU utilization by **filling this idle "bubble" time with unrelated pending jobs** (the bubbles often take 15–30%, and sometimes over 60%, of a job's allocation), rather than eliminating the bubbles themselves. It fits fill-job work to measured bubble durations and available memory, adds explicit pipeline-bubble instructions, and orchestrates execution inside the gaps. This yields up to 63% higher utilization with under 2% slowdown of the main job (about 2.6K extra GPUs' worth of work on an 8K-GPU run, and 5–15% even for smaller runs).

This makes PipeFill complementary to schedule optimizations such as 1F1B, interleaving, Zero Bubble (see [the Zero Bubble entry](#zero-bubble)) and DualPipe (see [the DeepSeek-V3 entry](#deepseek-v3)), which instead shrink or hide bubbles by densifying the training job's own schedule through forward/backward and compute/communication overlap. PipeFill is a [context-switching](https://en.wikipedia.org/wiki/Context_switch) layer, not a pipeline schedule.

Good fill jobs are workloads that tolerate interruption, with batch (offline) inference being ideal, as in the back-end jobs of content recommendation and data analytics. Such work is chunkable to match short bubble windows, latency-insensitive so pausing is fine, and small enough in memory to fit the spare capacity PipeFill frees up. The paper also uses other DL training jobs as fill work since training is long-running and interruption-tolerant, though batch inference makes better use of the bubbles, as it needs less memory and runs larger batches in the free memory.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="speculative-decoding-sparse-attention">
<summary markdown="span"><span class="pub-title">Speculative Decoding with Blockwise Sparse Attention</span><span class="pub-meta">, MatX</span><a class="pub-link" href="https://matx.com/research/sd_nsa" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This MatX article tackles a conflict that arises when two popular decoding accelerators are combined. **[Speculative decoding](https://en.wikipedia.org/wiki/Speculative_decoding) (SD)** uses a small draft model to propose several tokens that the big target model then verifies in parallel, raising [operational intensity](https://en.wikipedia.org/wiki/Roofline_model) (work done per byte loaded from HBM) and giving ~2× latency/throughput gains. **Blockwise sparse attention**, as in DeepSeek's Native Sparse Attention (NSA, see [the Native Sparse Attention entry](#native-sparse-attention)) and Moonshot's Mixture of Block Attention (MoBA), divides the context into blocks and lets each token dynamically attend to only a subset of them. This cuts the KV data moved from HBM and again raises operational intensity (important because long contexts otherwise leave modern accelerators underutilized).

The key idea is that **all draft tokens share the same selected blocks**, as the article sums up:

> Speculative decoding (SD) and blockwise sparse attention both accelerate LLM decoding, but when combined naively, the KV cache may lose sparsity during the verification step of SD. We show that forcing all draft tokens to attend to the same subset of the context restores sparsity while preserving model quality.

The authors show that NSA models trained this way match the language-modeling quality of baseline NSA. The gain is in the memory traffic of the verification step: with 3 draft tokens and a context of 8,192 tokens, the verification loads the KV cache of 2,048 tokens when the drafts share their blocks, against 5,120 tokens when each draft selects its own (2.5× more). In the limit of large batches, the article estimates that the operational intensity may be over **3.5× higher** in the shared case. The results come from small 186M and 1.2B-parameter models trained with 2K-token sequences. Conceptually, it is a co-design point: rather than treating SD and sparse attention as independent layers, it reshapes the sparsity pattern so the two compose efficiently.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="sla">
<summary markdown="span"><span class="pub-title">SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention</span><span class="pub-meta">, Tsinghua & UC Berkeley</span><a class="pub-link" href="https://arxiv.org/abs/2509.24006" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

SLA targets the attention bottleneck in Diffusion Transformers (DiTs), the backbone of recent [diffusion models](https://en.wikipedia.org/wiki/Diffusion_model) for video generation, where sequences of 10K to 100K tokens make attention's quadratic cost dominate latency. Its key observation is that attention weights split into two parts: a small fraction of large weights with high rank (under 10%), and the remaining weights with very low rank. This suggests applying **sparse acceleration to the first part and [low-rank](https://en.wikipedia.org/wiki/Low-rank_approximation) (linear) acceleration to the second**, instead of committing to a single approximation (pure sparsity or pure linear attention).

SLA splits the attention map into blocks of 64 × 64 and classifies them, from a compressed attention map, as *critical* (the top 5%), *marginal* (the next 85%) or *negligible* (the bottom 10%). For a sequence of $$N$$ tokens, critical blocks get exact $$O(N^2)$$ attention, marginal ones an $$O(N)$$ linear-attention approximation, and negligible ones are skipped. These computations are fused into a single GPU kernel that supports both the forward and backward passes, so a pretrained DiT only needs a short fine-tuning to adapt (2,000 steps, under 0.1% of the pretraining cost). This makes SLA behave like a "knob" between dense and approximate attention, which differentiates it from static sparse masks or fixed-kernel linear attention.

On Wan2.1-1.3B (about 30K tokens per video), SLA reduces attention computation by **95%** (from 52.75T to 2.74T FLOPs, 19.3× less) without degrading end-to-end generation quality. Its kernel gives a **13.7×** attention speedup over FlashAttention2, for a **2.2× end-to-end** video generation speedup (the attention time drops from 97 s to 11 s). Similarly to [the Sparse Transformers entry](#sparse-transformers) and [the Performers entry](#performers), it trades exact attention for sparse and linear approximations, but it chooses between them block by block.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepseek-v3-2-exp">
<summary markdown="span"><span class="pub-title">DeepSeek-V3.2-Exp (model card & notes)</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek-V3.2-Exp is an incremental update in the DeepSeek V3 family aimed primarily at **serving efficiency**, especially for long-context workloads. It is DeepSeek-V3.1-Terminus with a single architectural change introduced through continued training. Compared to the original DeepSeek-V3 technical report (Dec 2024, see [the DeepSeek-V3 entry](#deepseek-v3)), the emphasis shifts from "how it was trained" to "how to run it cheaper", without changing the core MoE transformer recipe.

The headline change is **DeepSeek Sparse Attention (DSA)**, described in [the DeepSeek-V3.2 entry](#deepseek-v3-2). It cuts the core attention cost for both prefill and decode at long context lengths. Note that the KV cache itself does not shrink (the full latent cache is still stored so that any token can be selected); what drops is the compute and the number of KV entries read per decoding step.

The Hugging Face release notes also call out practical serving details: recommended runtime settings, evaluation notes, and known issues, such as a mismatch of the RoPE layout in the indexer of the earlier demo code, fixed on 2025-11-17. For an existing DeepSeek-V3 deployment, V3.2-Exp is a **serving-oriented refresh**: benchmarks stay on par with V3.1-Terminus (85.0 on MMLU-Pro for both, 89.3 against 88.4 on AIME 2025), while the biggest wins come at large context lengths. The same architecture was kept for the final DeepSeek-V3.2 ([tech report](https://arxiv.org/abs/2512.02556), Dec 2025).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepcompile">
<summary markdown="span"><span class="pub-title">DeepCompile: A Compiler-Driven Approach to Optimizing Distributed Deep Learning Training</span><span class="pub-meta">, Microsoft DeepSpeed</span><a class="pub-link" href="https://arxiv.org/abs/2504.09983" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepCompile extends the capabilities of deep learning compilers to support distributed training. As the [DeepSpeed blog post](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepcompile/README.md) that introduced it puts it, "Distributed training has become essential for scaling today’s massive deep learning models. While deep learning compilers like PyTorch compiler dramatically improved single-GPU training performance through optimizations like kernel fusion and operator scheduling, they fall short when it comes to distributed workloads."

Existing distributed training frameworks require distributed optimizations to be implemented at the PyTorch level, which limits the ability to apply compiler techniques such as [dependency analysis](https://en.wikipedia.org/wiki/Dependence_analysis) or operator scheduling. As the paper puts it, "the fully sharded approach, as implemented in systems like ZeRO-3 and FSDP, employs runtime optimizations such as prefetching and unsharding":
- prefetching reduces the communication overhead by starting the all-gather of the parameters before the layer that needs them, which overlaps communication with computation;
- unsharding keeps parameters in their full form, to reduce communication when memory permits.

DeepCompile addresses this gap with **compiler-level optimizations for distributed training**. As the blog post says, "It takes a standard single-GPU model implementation and transforms it into an optimized multi-GPU training graph without requiring changes to the model code." On top of it, the authors implement a fully sharded approach like ZeRO-3 and FSDP, with three optimization passes:
- **Proactive prefetching** starts each all-gather as early as possible, to maximize the overlap of communication and computation, while following how the available memory changes during the forward and backward passes.
- **Selective unsharding** keeps as many parameters unsharded as memory allows, chosen from operator-level memory profiling, to reduce the all-gather communication.
- **Adaptive offloading** moves optimizer states (e.g., the momentum and variance of Adam) to CPU memory when GPU memory is insufficient: only the part that exceeds the memory limit, with transfers that overlap with computation.

It automatically implements distributed ZeRO-3, ZeRO-1 and offloading. On 32 GPUs, with 16 gradient accumulation steps, it trains Llama-3 70B up to 1.28× and Mixtral 8×7B up to 1.54× faster than ZeRO-3 (section 5.2 in the paper); FSDP fails to run with more than one accumulation step, and without accumulation DeepCompile is up to 1.26× and 1.14× faster than FSDP. With offloading, it reaches up to 7.01× higher throughput when GPU resources are limited. Future directions, as listed in the blog post, include automated parallelization (sequence and tensor parallelism), smarter memory management, and dynamic adaptation to runtime behavior. [The GSPMD entry](#gspmd) is an earlier compiler-driven approach, which partitions a computation graph in XLA from a few sharding annotations.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="triton-distributed">
<summary markdown="span"><span class="pub-title">Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler</span><span class="pub-meta">, ByteDance Seed</span><a class="pub-link" href="https://arxiv.org/abs/2504.19442" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Triton-distributed extends Triton with *distributed* primitives so that kernels can **compute and communicate inside the same program**, rather than launching a GEMM kernel and then calling NCCL as a separate step. The paper’s mental model is “a distributed kernel is a set of asynchronous tasks” that can **signal** one another and use **symmetric memory** ([OpenSHMEM](https://en.wikipedia.org/wiki/SHMEM)-style primitives, lowered to NVSHMEM/ROCSHMEM) to move data directly between GPUs.

The key difference vs "plain Triton + NCCL" is that communication becomes an explicit part of the kernel schedule: the compiler/runtime can overlap loads, math, and remote puts/gets at a much finer granularity than stream-level overlap of independent kernels. Compared to systems like Flux (see [the FLUX entry](#flux)) that hand-optimize specific LLM communication patterns, Triton-distributed tries to be a **general programming model** (primitives + compiler support) that can be reused across operators and topologies.

Results: on 8 H800 GPUs, it averages **1.42×** for all-gather + GEMM and **1.28×** for GEMM + reduce-scatter over PyTorch+NCCL, and across two nodes (16 GPUs) these kernels reach about 96% of FLUX's performance while being written in Python. MoE kernels show much larger average speedups (**44.97×** for all-gather + MoE, **15.55×** for MoE + reduce-scatter), but against a weak baseline (group GEMMs in Python loops). Its low-latency all-to-all kernels, a re-implementation of the DeepEP ones, are faster than DeepEP from 8 to 64 GPUs (except the dispatch on 64 GPUs; on 128 GPUs DeepEP stays faster), and it also runs on 8 AMD MI308X GPUs. [The TileLink entry](#tilelink) builds a higher-level, tile-centric layer on the same idea.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="tilelink">
<summary markdown="span"><span class="pub-title">TileLink: Generating Efficient Compute–Communication Overlapping Kernels Using Tile-Centric Primitives</span><span class="pub-meta">, ByteDance Seed (MLSys 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2503.20313" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

TileLink is a step toward **Flux-like performance** (see [the FLUX entry](#flux)) without requiring the programmer to hand-craft a very specialized fused kernel for each operation. It decouples the design space of communication and computation and links the two through **tile-centric primitives** (producer/consumer notify–wait signals and tile push/pull data movement), each mapped to a tensor slice, a peer rank and a [barrier](https://en.wikipedia.org/wiki/Barrier_(computer_science)) channel. The system then schedules computation and communication *per tile* (rather than per whole tensor), enabling overlap that approximates what highly tuned systems achieve.

A useful way to distinguish TileLink from nearby work:
- **Flux**: decomposes specific tensor-parallel LLM ops into finely sliced pieces, then fuses them into larger kernels to overlap up to “almost all” communication; highly effective, but quite tailored.
- **Triton-distributed**: offers low-level in-kernel primitives (symmetric memory, signals, tasks) and a general programming model.
- **TileLink**: sits in between: **higher-level than Triton-distributed**, more general and easier to program than Flux, but still explicitly models overlap at the *tile schedule* level.

Results: on 8 H800 GPUs, the paper reports speedups of 1.17× to 20.76× over non-overlapping baselines (the largest on MoE layers), with performance comparable to (or better than) state-of-the-art overlapping libraries such as FLUX and RingAttention. End to end, across eight language models, it averages **1.32×** over PyTorch on 8 H800 GPUs and **1.29×** on two nodes (16 GPUs). Its kernels take about 200 lines of Python, against about 2,000 lines of CUDA for FLUX (Table 2 in the paper).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="megascale-moe">
<summary markdown="span"><span class="pub-title">MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production</span><span class="pub-meta">, ByteDance Seed & Peking University</span><a class="pub-link" href="https://arxiv.org/abs/2505.11432" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

MegaScale-MoE is a production-oriented system for **communication-efficient MoE training**, where the hard part is not only computing experts but routing tokens and moving activations and gradients at scale. Communication (e.g., the [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) between token routers and experts) takes 43.6% of the forward pass and 32% of the whole training time of an internal MoE model on Hopper GPUs, and a holistic strategy is needed to keep the GPUs busy. Concretely, it:
1. customizes the parallelism of each MoE layer (sequence parallelism for attention and expert parallelism for the FFNs within a node, instead of Megatron-LM's tensor parallelism) to reduce communication volume;
2. overlaps communication with computation at both the inter-operator and intra-operator levels;
3. compresses communication to lower precision, with adjusted communication patterns: in BF16 training, the gradient synchronization between nodes runs in BF16 instead of FP32, with an all-to-all and an FP32 sum instead of a ring reduction.

Unlike the kernel-level fusion of [the FLUX entry](#flux), [the TileLink entry](#tilelink) and [the Triton-distributed entry](#triton-distributed), MegaScale-MoE is primarily a **system and parallelization strategy** for end-to-end training at very large scale. Its gains come from choosing the parallelism layout, the scheduling and the communication precision of the whole training job, not from a new kernel programming model.

Results: training a **352B MoE** model on 1,440 Hopper GPUs, it reaches **1.41M tokens/s**, 1.88× the throughput of Megatron-LM (1.65–1.88× across the strong-scaling settings). It powers ByteDance's production MoE training, with models of trillions of parameters and single jobs of more than 10,000 GPUs: at this scale, communication efficiency is not a rounding error.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="megascale-infer">
<summary markdown="span"><span class="pub-title">MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism</span><span class="pub-meta">, ByteDance Seed & Peking University</span><a class="pub-link" href="https://arxiv.org/abs/2504.02263" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

During decoding, sparse activation leaves each expert with few tokens, so the MoE FFNs are memory-bound. As an example, Mixtral 8×22B on A100 GPUs (312 TFLOP/s in BF16, 2 TB/s of HBM bandwidth) needs a batch of at least 156 tokens to be compute-bound. With top-2 routing over 8 experts, a batch of 156 tokens gives each expert only 39 of them, for a theoretical FFN utilization of 25%. MegaScale-Infer's answer is **Disaggregated Expert Parallelism (DEP)**: disaggregate the attention and expert modules, assigning them to separate GPUs, so each can scale independently with its own parallelism. Attention modules are replicated using [data parallelism](https://en.wikipedia.org/wiki/Data_parallelism), while FFN modules are scaled with expert parallelism; many attention replicas funnel their tokens into the expert pool, which restores large per-expert batches and turns the FFNs back into efficient compute-bound work. Disaggregation also allows **heterogeneous deployment** (cheaper or different GPUs for attention and for experts) to cut cost.

Two systems pieces make disaggregation pay off rather than stall on communication. The first is **ping-pong pipeline parallelism**, which partitions a request batch into micro-batches and shuttles them between attention and FFNs so the two GPU pools stay busy and communication hides behind compute. The second is a **high-performance M2N communication library** that eliminates unnecessary GPU-to-CPU data copies, group initialization overhead, and GPU synchronization for the token dispatch/combine traffic: at 256 KB, its median latency is 68.2% lower than that of NCCL, for 4.2× the throughput. This is the MoE-serving flavor of attention–FFN disaggregation: the two modules are split *spatially* across GPU pools; prefill/decode disaggregation (as in [the TPLA entry](#tpla)) is a different, orthogonal split.

Results: up to **1.90×** higher per-GPU decoding throughput than state-of-the-art LLM serving systems on homogeneous clusters, and up to **1.86×** higher throughput per cost on heterogeneous clusters, with the attention on H20 GPUs and the experts on L40S GPUs.

The following picture illustrates a MegaScale-Infer instance, with the attention nodes on the left and the expert nodes on the right:

{: .picture}
<img loading="lazy" width="69.7%" height="69.7%" src="{{ site.assets }}/publications/megascale_infer.svg"/>

{: .caption}
MegaScale-Infer runtime instance architecture. The attention nodes are replicated ($$M$$ of them) and the expert nodes split the experts ($$N$$ of them, expert parallelism); inside each node the GPUs use tensor parallelism. Tokens go from the attention nodes to the expert nodes (M2N) and back (N2M) over InfiniBand or Ethernet, in a ping-pong pipeline. Adapted from: [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](https://arxiv.org/abs/2504.02263)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="moeshard">
<summary markdown="span"><span class="pub-title">Accelerating MoE Model Inference with Expert Sharding (MoEShard)</span><span class="pub-meta">, EPFL & McGill (EuroMLSys 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2503.08467" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Balmau et al. target MoE inference with expert parallelism (EP), where skewed token routing leaves some GPUs overloaded while others sit idle. The usual fixes (capacity factors that drop tokens, or replicating popular experts) either lose tokens or cost memory. Their system, **MoEShard**, instead shards *every* expert across all GPUs, tensor-parallel style: the first expert matrix is split column-wise and the second row-wise, so each GPU computes a slice of every selected expert and the partial outputs are summed. Load is perfectly balanced regardless of routing skew, and no token is dropped.

To keep this efficient, MoEShard concatenates the tokens of each expert from all GPUs, so that each GPU computes one shard per expert instead of one per expert and source GPU. It also uses the block-sparse matrix multiplication of MegaBlocks, which makes the number of kernel launches independent of the number of experts. Evaluated on the encoder of Switch-Base (a Switch Transformer MoE model) on one node of 4 A100 GPUs, it reports up to **6.4×** lower time-to-first-token than DeepSpeed-MoE (see [the DeepSpeed-MoE entry](#deepspeed-moe)).

This work is a useful complement to [the MegaScale-MoE entry](#megascale-moe) and [the MegaScale-Infer entry](#megascale-infer). It is a small-scale (single-node, workshop) study showing that tensor sharding of experts is a viable alternative to plain EP when routing is imbalanced, relying on fast intra-node links for the extra communication. Like the Head Parallel of [the Multi-Head LatentMoE entry](#multi-head-latentmoe), it balances the load by construction, but it keeps the model unchanged.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="transmla">
<summary markdown="span"><span class="pub-title">TransMLA: Multi-Head Latent Attention Is All You Need</span><span class="pub-meta">, Peking University & Xiaomi</span><a class="pub-link" href="https://arxiv.org/abs/2502.07864" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

TransMLA is a **post-training conversion** method that turns existing Grouped-Query Attention (GQA)-based models (LLaMA, Qwen, Mixtral) into MLA-based ones (see [the DeepSeek-V2 entry](#deepseek-v2)), without retraining from scratch. The motivation is an adoption gap: most model providers have invested heavily in GQA models and have little incentive to pretrain MLA models from scratch. The paper's theoretical core is a proof that **GQA can always be rewritten exactly as MLA with a KV cache of the same size, but not vice versa**, i.e., MLA is strictly more expressive than GQA for a given cache size. That asymmetry means any GQA checkpoint can be losslessly re-expressed in MLA form and then *upgraded* to use the spare expressiveness MLA affords.

TransMLA exploits this in two steps. First, it **equivalently converts** the model's GQA attention into MLA structure (reformulating the repeated KV heads of GQA as low-rank latent projections). This produces a model directly compatible with DeepSeek's codebase and its optimized serving stacks (vLLM, SGLang), which could then use features such as FP8 quantization and Multi-Token Prediction (the paper does not test them).

Second, since the conversion alone does not speed anything up, it **compresses** the latent KV below the original cache size and then **fine-tunes** briefly to recover quality. Only ~6B tokens are needed to come close to the original model (an average score of 58.68 against 59.85 for LLaMA-2-7B). Compressing 93% of the KV cache of LLaMA-2-7B gives an inference speedup of up to 10.6× at 8K context on one of the three GPUs tested (6.5× and 5.1× on the other two), while preserving meaningful output quality. It is the migration path that complements [the TPLA entry](#tpla): rather than designing new MLA models or new sharding, TransMLA *retrofits* the large existing fleet of GQA models into the more efficient MLA family.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="look-ma-no-bubbles">
<summary markdown="span"><span class="pub-title">Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B</span><span class="pub-meta">, Stanford Hazy Research</span><a class="pub-link" href="https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This work targets an extreme low-latency regime: **single-sequence decoding** for a ~1B-parameter LLM (Llama-3.2-1B), where performance is dominated by how efficiently the GPU can stream weights from global memory. The authors argue that modern inference engines still suffer from "pipeline bubbles" because a forward pass is split into **around a hundred small kernels**, each with launch and teardown costs and synchronization stalls that prevent continuous memory streaming. In the post's estimate for an H100, the memory bandwidth (3.35 TB/s) could stream the 2.48 GB of weights about 1,350 times per second. However, 7 kernels per layer over 16 layers, with an optimistic 5 µs of stall each, add 0.56 ms to each forward pass and bring the rate down to about 770 forward passes per second.

Their solution is to fuse the *entire* forward pass into a single **megakernel** that acts like an on-GPU [interpreter](https://en.wikipedia.org/wiki/Interpreter_(computing)): each SM executes a schedule of "instructions" (fused RMSNorm+QKV+RoPE, attention, projections, MLP pieces, etc.). They then focus on three practical problems that show up when "about a hundred" operations are fused:
1. how to *program* such a megakernel (an instruction abstraction and an interpreter);
2. how to avoid resource contention (e.g., **paged shared memory** to pipeline weight loads);
3. how to synchronize without kernel boundaries (a lightweight **counter-based** scheme).

Results: on an **H100**, the megakernel uses about 78% of the memory bandwidth, while vLLM and SGLang use at most about 50% in this setting. In an end-to-end comparison (32-token prompt, 128 generated tokens, no speculation), it is about 2.5× faster than vLLM and over 1.5× faster than SGLang on H100, and over 3.5× and 1.5× faster than them on **B200**. A forward pass of this 16-bit model of over 1B parameters takes under 1 ms on H100, and under 680 µs on B200.

This sits in a different part of the design space than [the FLUX entry](#flux), [the TileLink entry](#tilelink) and [the Triton-distributed entry](#triton-distributed): it is about eliminating *kernel boundaries entirely* for single-GPU low-latency inference, rather than fusing compute with distributed communication. [The Mirage Persistent Kernel entry](#mirage-persistent-kernel) later automated this approach in a compiler.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="mirage">
<summary markdown="span"><span class="pub-title">Mirage: A Multi-Level Superoptimizer for Tensor Programs</span><span class="pub-meta">, CMU et al. (OSDI 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2405.05751" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Fusing the operators of a tensor program into fast GPU kernels takes expert work for each new pattern, as FlashAttention did for attention. Mirage is an *automatic* approach to generating deeply fused tensor programs, framing kernel fusion as a [superoptimization](https://en.wikipedia.org/wiki/Superoptimization) problem over **µGraphs**: a uniform representation of a tensor program at the kernel, thread-block, and thread levels of the GPU compute hierarchy. µGraphs let Mirage jointly search algebraic transformations, schedule transformations, and the generation of new custom kernels; it rediscovers FlashAttention- and FlashDecoding-like kernels on its own. An abstraction-based pruning technique keeps the huge search space tractable (with an optimality guarantee under the paper's conditions), and a probabilistic equivalence verifier checks that the optimized µGraph computes the same function as the input program.

How it differs from other "big fusion" lines:
- Compared to **manual megakernels** (e.g., [the Look Ma, No Bubbles entry](#look-ma-no-bubbles)), Mirage aims to *discover* aggressive fusions automatically and generate optimized implementations.
- Compared to **FlashAttention-style** work (see [the FlashAttention entry](#flashattention)), Mirage is broader: it targets arbitrary tensor programs (not only attention), and can potentially find FlashAttention-like structures when they are profitable.
- Compared to **compiler frameworks** like TVM/[XLA](https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra) (see [the Operator Fusion in XLA entry](#operator-fusion-in-xla)), Mirage emphasizes superoptimization over small fused graphs to get performance that can beat hand tuning.

Results: on six benchmarks of DNN components (GQA, QKNorm, RMSNorm, LoRA, GatedMLP and nTrans) on A100 and H100 GPUs, Mirage outperforms existing approaches by **up to 3.3×** (the exact figure varies across paper versions), even for components that are widely used and heavily optimized. [The Mirage Persistent Kernel entry](#mirage-persistent-kernel) extends it from single kernels to whole models in one megakernel.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="sageattention3">
<summary markdown="span"><span class="pub-title">SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training</span><span class="pub-meta">, Tsinghua</span><a class="pub-link" href="https://arxiv.org/abs/2505.11594" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

SageAttention3 pushes the SageAttention line (see [the SageAttention2 entry](#sageattention2)) toward even more aggressive inference acceleration by running both attention matmuls on the new **FP4 Tensor Cores** of Blackwell GPUs, with **microscaling** (block-scaled) FP4 quantization plus attention-specific tricks to keep accuracy. It uses NVFP4 (see [the NVFP4 entry](#nvfp4)), with blocks of 1 × 16 values and E4M3 scales, which keeps a cosine similarity of 99.52% to full-precision attention, against 98.37% with MXFP4. The attention probabilities $$P$$, which lie between 0 and 1, get a two-level scaling: each row is first stretched by an FP32 factor, so that the FP8 scales of its blocks use their full range, and is then quantized to FP4 per block. This raises the cosine similarity from 93.32% with direct quantization to 99.52%. Conceptually, it sits between "attention kernels that assume FP16/BF16" and "end-to-end quantized models": it focuses on the attention operator itself and is designed to be dropped into existing inference stacks.

On **RTX 5090**, the paper reports **1038 TOPS**, about **5×** the fastest FlashAttention on that GPU (FlashAttention2, at 212 TOPS, since FlashAttention3 only runs on Hopper). Used plug-and-play in various models, it speeds them up end to end: video generation takes 164 s instead of 489 s with HunyuanVideo (3.0×), and 27 s instead of 64 s with CogVideoX (2.4×). It also pioneers low-bit attention for *training* (SageBwd, 8-bit forward and backward): lossless in fine-tuning, but with slower convergence in pretraining.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="titans">
<summary markdown="span"><span class="pub-title">Titans: Learning to Memorize at Test Time</span><span class="pub-meta">, Google Research (NeurIPS 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2501.00663" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Titans are a family of models that combine attention with a **neural long-term memory**. The memory is an MLP whose weights keep being updated at test time by [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) steps on an associative-memory loss, driven by a "surprise" signal with momentum and a weight-decay-style forgetting gate.

In the words of the abstract:

> We present a new neural long-term memory module that learns to memorize historical context and helps an attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of a fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory.

In practice, for an input $$x_t$$ with key $$k_t = x_t W_K$$ and value $$v_t = x_t W_V$$, the memory $$\mathcal{M}$$ learns to map keys to values, with the loss given by (Eq. 12 in the paper):

$$\ell(\mathcal{M}_{t-1}; x_t) = \lVert \mathcal{M}_{t-1}(k_t) - v_t \rVert_2^2$$

The gradient of this loss is the *surprise* of the input: it is large when the memory does not predict $$v_t$$ well. The memory is then updated with momentum and a forgetting gate (Eqs. 13 and 14 in the paper):

$$S_t = \eta_t \, S_{t-1} - \theta_t \, \nabla \ell(\mathcal{M}_{t-1}; x_t)$$

$$\mathcal{M}_t = (1 - \alpha_t) \, \mathcal{M}_{t-1} + S_t$$

where $$S_t$$ is the surprise with momentum, $$\eta_t$$ the decay of the past surprise, $$\theta_t$$ the weight of the momentary surprise (a learning rate), and $$\alpha_t$$ the forgetting gate, which acts as a weight decay. All three depend on the input.

The paper proposes three ways to combine the memory with attention: memory as a context (MAC), as a gate (MAG), or as a layer (MAL). Titans outperform Transformers and modern linear recurrent models (e.g., Mamba2, DeltaNet) on language modeling and commonsense reasoning: at 760M parameters, Titans (MAC) reaches an average accuracy of 52.51%, against 48.69% for Transformer++. The memory is also competitive on genomics, and best on most time-series metrics. Titans scale to context windows beyond 2M tokens, with higher needle-in-a-haystack accuracy than the baselines (on the pass-key retrieval task at 16K tokens, 98.4% against 88.4% for TTT and 5.4% for Mamba2). With a linear memory instead of an MLP, the gradient step of the update is a delta-rule update with a forgetting gate, the rule that Gated DeltaNet and the KDA of [the Kimi Linear entry](#kimi-linear) build on.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2024</span></td>
<td markdown="block">
<details id="loongserve">
<summary markdown="span"><span class="pub-title">LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism</span><span class="pub-meta">, Peking University (SOSP 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2404.09526" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

In long-context serving, the resources that a request needs vary widely: across requests, from a few tokens to about 500K tokens in the datasets of the paper, and across the two phases of a request, as the prefill processes the whole prompt at once and the decode adds one token per step. A static parallelism strategy cannot use the GPUs efficiently for all of them. LoongServe introduces **Elastic Sequence Parallelism (ESP)**: instead of a fixed degree of parallelism per instance, the degree of sequence parallelism is adapted in real time, per iteration and per phase (scale-up and scale-down). The prefill runs ring attention, and while the KV tensors rotate across the GPUs, each GPU keeps a subset of the KV cache locally, already in place for the decode (a proactive scale-down that avoids migrating the KV cache).

In the decode phase, the KV cache stays sharded across the instances, so the decode follows the regular distributed decode algorithm, which is not so efficient and is usually the execution bottleneck of context parallelism. During the prefill, LoongServe overlaps the communication of the ring attention with computation on CUDA streams. For the decode, the paper claims that the communication of the partial attention results is overlapped with computation. In practice, however, this overlap is opportunistic: while one instance sends its requests, it *may* be computing the matmuls of the requests that it receives from other instances.

ESP also reduces the [fragmentation](https://en.wikipedia.org/wiki/Fragmentation_(computing)) of the KV cache across instances. With 8 A800 GPUs per server and a 7B model with a context of 1M tokens, LoongServe reports up to **3.85×** higher maximum throughput than chunked prefill, and **5.81×** than prefill–decode disaggregation. Like [DistFlashAttn](#distflashattn), it builds on ring attention, for serving instead of training.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="mamba">
<summary markdown="span"><span class="pub-title">Mamba: Linear-Time Sequence Modeling with Selective State Spaces</span><span class="pub-meta">, CMU & Princeton (COLM 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2312.00752" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Attention compares each token with all the tokens before it, so its compute grows quadratically with the sequence length, and its KV cache linearly. [State-space models](https://en.wikipedia.org/wiki/State-space_representation) (SSMs) keep a fixed-size recurrent state instead, and scale **linearly** in sequence length, but earlier SSMs (S4) underperformed on language, because their dynamics were fixed regardless of the input. [Mamba](https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)) is the SSM that made a non-attention sequence model competitive with Transformers on language. Its fix is the **selection mechanism**: making the SSM parameters functions of the input lets the model selectively propagate or forget information depending on the current token, which gives it content-based reasoning.

Input-dependence breaks the convolution trick of S4 (below), so Mamba adds a **hardware-aware parallel [scan](https://en.wikipedia.org/wiki/Prefix_sum)**, which keeps the expanded state in the [SRAM](https://en.wikipedia.org/wiki/Static_random-access_memory) of the GPU, as [FlashAttention](#flashattention) does, to stay fast. The result: 5× higher inference throughput than Transformers, linear scaling in sequence length, and a Mamba-3B model that matches Transformers twice its size. Its constant-size state is what removes the growing KV cache, the property that the hybrids [Jamba](#jamba) and [Nemotron 3](#nemotron-3) exploit. [RetNet](#retentive-network) also has a constant-size state, but its decay is fixed per head instead of depending on the input.

**Formula.** An SSM maps an input $$x_t$$ to an output $$y_t$$ through a hidden state $$h_t \in \mathbb{R}^N$$. Its continuous parameters $$(\Delta, A, B, C)$$ are first **[discretized](https://en.wikipedia.org/wiki/Discretization)**, with the zero-order hold of the paper, as:

$$\bar A = \exp(\Delta A), \qquad \bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B$$

Then it runs as a **linear recurrence** (inference) or a **global convolution** (training):

$$h_t = \bar A\,h_{t-1} + \bar B\,x_t, \qquad y_t = C\,h_t$$

In S4, these parameters are **fixed across time** ([LTI](https://en.wikipedia.org/wiki/Linear_time-invariant_system)), so the recurrence unrolls into a convolution of the input with a fixed kernel. Mamba makes $$B$$, $$C$$ and $$\Delta$$ **functions of $$x_t$$** (the S6 selective SSM), so that $$(\bar A, \bar B)$$ vary per token: that is the selectivity, and it is what forces a scan instead of a convolution. The following picture illustrates one time step of the selective SSM, in which the expanded states (orange) stay in the SRAM of the GPU, while the inputs, outputs and parameters (green) are in its [HBM](https://en.wikipedia.org/wiki/High_Bandwidth_Memory):

{: .picture}
<img loading="lazy" width="88.5%" height="88.5%" src="{{ site.assets }}/publications/mamba.svg"/>

{: .caption}
<b>Selective state space model with hardware-aware state expansion.</b> Structured SSMs independently map each channel (e.g. $$D = 5$$) of an input $$x$$ to output $$y$$ through a higher dimensional latent state $$h$$ (e.g. $$N = 4$$). Prior SSMs avoid materializing this large effective state ($$DN$$, times batch size $$B$$ and sequence length $$L$$) through clever alternate computation paths requiring time-invariance: the $$(\Delta, \boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C})$$ parameters are constant across time. Mamba's selection mechanism adds back input-dependent dynamics, which also requires a careful hardware-aware algorithm to only materialize the expanded states in more efficient levels of the GPU memory hierarchy. Adapted from: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="jamba">
<summary markdown="span"><span class="pub-title">Jamba: A Hybrid Transformer-Mamba Language Model</span><span class="pub-meta">, AI21 Labs</span><a class="pub-link" href="https://arxiv.org/abs/2403.19887" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The KV cache becomes a limiting factor when Transformers are scaled to long contexts. Trading attention layers for [Mamba](#mamba) layers shrinks it, since Mamba carries only a constant-size state, not a growing cache. Jamba (Lieber, Lenz et al., AI21) is the first production-grade **Attention-SSM hybrid**: it interleaves Transformer attention, Mamba and [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) layers. Attention is kept sparingly, for high-fidelity recall, Mamba does the cheap long-range work, and MoE adds capacity without adding active compute.

The released model has 4 Jamba blocks of 8 layers, with a 1:7 ratio of attention to Mamba layers and an MoE layer every other layer, of 16 experts with the top 2 used per token. This gives 52B parameters in total but only 12B active ones, and the model fits on one 80 GB GPU. These settings are the tunable knobs of the design: the 1:7 ratio is empirical, as it performs as well as 1:3 with less compute and keeps most of the memory benefit of Mamba.

The main gain is the cache: at a context of 256K tokens, Jamba needs about 4 GB of KV cache, against 32 GB for Mixtral and 128 GB for Llama-2 7B. It also gives strong results up to 256K tokens, and 3× the throughput of Mixtral at 128K tokens. Jamba 1.5 later scaled the same design to 94B active and 398B total parameters.

Jamba is a template that [Nemotron 3](#nemotron-3) follows, with Mamba-2 layers instead of Mamba ones, and with [LatentMoE](#latentmoe), [multi-token prediction](#multi-token-prediction) (MTP) and [NVFP4](#nvfp4) added. The following picture illustrates a Jamba block of eight layers (a), and the four types of layers of the architecture (b):

{: .picture}
<img loading="lazy" width="58.8%" height="58.8%" src="{{ site.assets }}/publications/jamba.svg"/>

{: .caption}
<b>(a)</b> A single Jamba block. <b>(b)</b> Different types of layers. The implementation shown here is with $$l = 8$$, $$a : m = 1 : 7$$ ratio of attention-to-Mamba layers, and MoE applied every $$e = 2$$ layers. Adapted from: [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="multi-token-prediction">
<summary markdown="span"><span class="pub-title">Better & Faster Large Language Models via Multi-token Prediction</span><span class="pub-meta">, Meta (FAIR) (ICML 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2404.19737" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Large language models such as GPT and Llama are trained with a next-token prediction loss. This paper (Gloeckle et al., Meta) argues that training to predict several future tokens at once gives a higher sample efficiency, and establishes **Multi-Token Prediction (MTP)** as a general training objective.

The model predicts the following $$n$$ tokens with $$n$$ independent output heads on top of a shared trunk, which makes multi-token prediction an **auxiliary task** added to the usual objective, with no overhead in training time. At position $$t$$, while the main head predicts the next token $$x_{t+1}$$, the auxiliary heads predict $$x_{t+2}, x_{t+3}$$, and so on. At inference, the heads give a self-speculative decoding:
- draft: the auxiliary heads produce a draft of the next few tokens at the same time as the main token;
- parallel verification: in the next step, the model runs these guesses through its main blocks in one single forward pass.

Standard [speculative decoding](https://en.wikipedia.org/wiki/Speculative_decoding) splits the roles strictly between two models:
1. The **draft model**, a small LLM, generates a look-ahead sequence (usually 3 to 5 tokens) one token at a time, feeding each token back into itself. It is small enough that these sequential passes are still much faster than a single pass of the large model.
2. The **target model**, the large one, checks the whole draft in a single forward pass with a causal mask: it computes the probability of the fourth drafted token at the same time as that of the first one, so that it verifies all the drafted tokens in the time it would take to generate one.

MTP needs no separate draft model: the auxiliary heads of the model itself write the draft.

Results: the benefit grows with model size. The 13B model solves 12% more HumanEval and 17% more MBPP problems than a comparable next-token model, and 4-token models are up to 3× faster at inference with this self-speculative decoding, even at large batch sizes.

[DeepSeek-V3](#deepseek-v3) refined MTP into *sequential* modules that keep the full causal chain, instead of independent parallel heads, and [Nemotron 3](#nemotron-3) adopts a lightweight MTP module, which reaches about 97% acceptance on its first two predicted tokens in an ablation.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepseek-llm">
<summary markdown="span"><span class="pub-title">DeepSeek LLM: Scaling Open-Source Language Models with Longtermism</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2401.02954" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek LLM is the first general-purpose language model of [DeepSeek](https://en.wikipedia.org/wiki/DeepSeek), released a few weeks after DeepSeek Coder (November 2023), with dense 7B and 67B models trained on 2T tokens. It is the pre-MoE, pre-MLA baseline: [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) and Multi-head Latent Attention (MLA) came with DeepSeekMoE and [DeepSeek-V2](#deepseek-v2). The architecture follows the [LLaMA](#llama) design: pre-norm with RMSNorm, SwiGLU FFNs and rotary position embeddings. The 7B model has 30 layers, and the 67B model has 95 layers and uses Grouped-Query Attention (GQA) to cut the cost of inference.

The central contribution of the paper is an empirical study of [scaling laws](https://en.wikipedia.org/wiki/Neural_scaling_law), as earlier studies reached different conclusions on how to trade off model size and data for a given compute budget. The model size $$M$$ is measured in non-embedding FLOPs per token (counting the attention but not the vocabulary computation) instead of parameters, so that the compute budget is $$C = M D$$ for $$D$$ training tokens. The paper fits the optimal batch size and learning rate as power laws of the compute budget, and the optimal split of the compute between model size and data, $$M_{\text{opt}} \propto C^{0.5243}$$ and $$D_{\text{opt}} \propto C^{0.4757}$$, close to the even split of [Chinchilla](#chinchilla).

These laws set the batch size and learning rate of both models and predicted their performance; *longtermism* in the title refers to the long-term perspective of the project, which scales the models under the guidance of these laws. The study also finds that the higher the quality of the data, the larger the share of a growing compute budget that should go to the model size rather than to the data.

The base models are aligned with supervised [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) (SFT) on about 1.5M instances, then with Direct Preference Optimization (DPO), which gives the DeepSeek Chat models. DeepSeek LLM 67B surpasses [LLaMA-2 70B](#llama-2), especially on code, math and reasoning, and the 67B chat model beats GPT-3.5 in open-ended evaluations.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepseek-v2">
<summary markdown="span"><span class="pub-title">DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2405.04434" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The two big performance bottlenecks of a Transformer are the KV cache of vanilla Multi-Head Attention (MHA) during decoding, and the dense FFNs during training. DeepSeek-V2 targets both: it introduces **Multi-head Latent Attention (MLA)** and adopts [DeepSeekMoE](https://arxiv.org/abs/2401.06066), the two architectural pillars later carried into [DeepSeek-V3](#deepseek-v3) and R1. It is a [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) model with 236B parameters in total, of which only 21B are activated per token, and a context of 128K tokens.

**DeepSeekMoE** provides the FFN, with a large capacity from a sparse activation at a low compute, through two main ideas:
1. **Shared experts**: a small number ($$K_s$$) of experts that every token always uses, with no routing. They are not clones of each other, but an always-on set that absorbs the common knowledge.
2. **Fine-grained expert segmentation**: where a regular MoE has $$N$$ experts, each a full-size FFN, and activates the top $$K$$, DeepSeekMoE slices each expert into $$m$$ smaller ones and activates $$mK$$ of the $$mN$$. The parameters and the FLOPs stay the same, but each token has far more *combinations* of experts: $$\binom{16}{2} = 120$$ for $$N = 16$$ and top-2 routing, against $$\binom{64}{8} = 4{,}426{,}165{,}368$$ with $$m = 4$$.

In DeepSeek-V2, each MoE layer has 2 shared and 160 routed experts, of which 6 are activated per token.

**MLA** uses a [low-rank](https://en.wikipedia.org/wiki/Low-rank_approximation) joint compression of the keys and values: rather than caching the keys and values of each head, or sharing heads as GQA and [MQA](#multi-query-attention) do, it down-projects them into a single small latent vector $$c_t^{KV}$$ per token, which is cached, and up-projects this vector back to distinct per-head keys and values at compute time. This keeps the expressiveness of full MHA while shrinking the cache. RoPE on the keys would sit between the queries and that up-projection, which could then no longer be absorbed (the *absorb* trick below). Therefore, DeepSeek-V2 adds the **decoupled RoPE**: a separate set of small query dimensions and one shared key, also cached (a small extra $$d_h^R$$ per token), carry the positional rotation, apart from the compressed *content* path.

In the formulas below, $$h_t$$ is the input hidden vector of the token at position $$t$$, $$n_h$$ the number of heads, $$d_h$$ the dimension per head, $$d_c \ll n_h d_h$$ the dimension of the cached latent vector, and $$d_h^R$$ that of the RoPE part. The matrices $$W^{D\ast}$$ are down-projections (D), which compress, and $$W^{U\ast}$$ are up-projections (U), which reconstruct; the superscripts $$C$$ and $$R$$ mark the content and RoPE parts, and $$j \le t$$ indexes the past tokens. The cached latent vector and the content keys and values of all heads are computed as:

$$c_t^{KV} = W^{DKV}\, h_t \quad (\in \mathbb{R}^{d_c},\ \text{cached, together with the RoPE key } k_t^{R}); \qquad k_t^{C} = W^{UK} c_t^{KV},\quad v_t^{C} = W^{UV} c_t^{KV}$$

where $$W^{DKV}$$, $$W^{UK}$$ and $$W^{UV}$$ are learned projections of lower rank. The queries are compressed too, which saves activation memory during training, although they are not cached:

$$c_t^{Q} = W^{DQ}\, h_t, \qquad q_t^{C} = W^{UQ}\, c_t^{Q}$$

As a side note, the projections of regular attention also have a lower rank per head: regular attention factors $$W^K$$ *by head*, and stays full-rank overall, while MLA factors it *through a shared low-dimensional latent*, which is deliberately rank-deficient, and that latent is what the KV cache stores.

With the decoupled RoPE, the position follows a separate path, with rotary queries per head and one **shared** rotary key:

$$q_t^{R} = \mathrm{RoPE}(W^{QR}\, c_t^{Q}), \qquad k_t^{R} = \mathrm{RoPE}(W^{KR}\, h_t)$$

The final query and key of each head $$i$$ concatenate the content and rotary parts:

$$q_{t,i} = [\,q_{t,i}^{C}\,;\,q_{t,i}^{R}\,], \qquad k_{t,i} = [\,k_{t,i}^{C}\,;\,k_t^{R}\,]$$

Then attention is standard:

$$o_{t,i} = \sum_{j \le t} \mathrm{softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v_{j,i}^{C}, \qquad u_t = W^{O}\,[\,o_{t,1};\dots;o_{t,n_h}\,]$$

The KV cache per token and per layer, in elements, is given by the following table, where $$g$$ is the number of key-value groups of GQA:

| MHA | GQA | MQA | **MLA** |
|---|---|---|---|
| $$2\,n_h\,d_h$$ | $$2\,g\,d_h$$ | $$2\,d_h$$ | $$d_c + d_h^R$$ |

The **absorb** trick folds $$W^{UK}$$ into $$W^{UQ}$$ and $$W^{UV}$$ into $$W^{O}$$ at decode time, so that attention runs directly against $$c_t^{KV}$$ and the full per-head keys and values are never materialized. DeepSeek-V2 has $$n_h = 128$$ heads of dimension $$d_h = 128$$, with $$d_c = 512$$ and $$d_h^R = 64$$, so its cache holds $$512 + 64 = 576$$ elements per token and per layer, about 57 times fewer than the $$2 n_h d_h = 32{,}768$$ of MHA: as much as GQA with only 2.25 groups, but with the expressiveness of full MHA.

Compared with the earlier dense [DeepSeek 67B](#deepseek-llm), DeepSeek-V2 has a 42.5% lower training cost, a **93.3% smaller KV cache** and up to **5.76× higher maximum generation throughput**, with a better quality. [TransMLA](#transmla) later converts GQA models to MLA, and [TPLA](#tpla) splits the latent KV cache of MLA across GPUs with tensor parallelism.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="sageattention2">
<summary markdown="span"><span class="pub-title">SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization</span><a class="pub-link" href="https://arxiv.org/abs/2411.10958" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Attention has quantization pathologies that do not show up as strongly in linear layers, especially [outliers](https://en.wikipedia.org/wiki/Outlier), which make low-bit arithmetic inaccurate. SageAttention2 builds on [SageAttention](#sageattention) with attention-specific techniques that make lower bits viable: it moves from INT8 to **INT4** quantization for the expensive $$QK^\top$$ product, at a *per-thread* granularity, and from FP16 to **[FP8](https://en.wikipedia.org/wiki/Minifloat)** for the $$\widetilde{P}V$$ product (the unnormalized attention probabilities times the values).

Concretely, it proposes (1) per-thread INT4 quantization for $$Q$$ and $$K$$, (2) a smoothing method for $$Q$$ to improve the accuracy of the INT4 $$QK^\top$$, and (3) a two-level accumulation strategy to improve the accuracy of the FP8 $$\widetilde{P}V$$. On an **RTX 4090**, the OPS (operations per second) of SageAttention2 surpass those of FlashAttention2 and xFormers by about **3×** and **4.5×**, respectively; on Hopper GPUs, it matches the speed of FlashAttention3 (FP8) with a higher accuracy, and the end-to-end metrics show a negligible loss. [SageAttention3](#sageattention3) then goes down to FP4.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="sageattention">
<summary markdown="span"><span class="pub-title">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</span><span class="pub-meta">, Tsinghua (ICLR 2025)</span><a class="pub-link" href="https://arxiv.org/abs/2410.02367" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[Quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) is widely used for the linear layers of a model, while attention usually stays in 16 bits. Yet attention can remain a major bottleneck, especially at long sequence lengths, as its cost grows as $$O(N^2)$$ in the sequence length $$N$$, against $$O(N)$$ for the linear layers. SageAttention is an "attention-first" quantization paper: rather than quantizing an entire model, it targets the attention kernel and makes it plug-and-play for existing inference. This is harder than for the linear layers, as attention mixes GEMMs, softmax and reductions in ways that amplify numerical outliers.

The core idea is to run attention with **8-bit** arithmetic while preserving accuracy. The paper identifies two pathologies of naïve INT8 attention. First, $$K$$ has strong channel-wise outliers, fixed by *smoothing* $$K$$, i.e. subtracting its mean over tokens, which shifts all the scores of a query by the same value and so leaves the softmax unchanged. Second, quantizing $$\widetilde{P}, V$$ to INT8 is not reliably accurate, so it computes $$QK^\top$$ in INT8 and keeps $$\widetilde{P}V$$ in FP16 with an FP16 accumulator.

The paper reports about **2.1×** the OPS of FlashAttention2 and **2.7×** that of xFormers, with a better accuracy than the FP8 attention of FlashAttention3, and almost no end-to-end metric loss across language, image and video generation models. Compared with the later [SageAttention2](#sageattention2) (INT4 and FP8) and [SageAttention3](#sageattention3) (FP4), this first version is less aggressive, and serves as the baseline that shows that attention quantization is feasible with minimal integration work.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="centauri">
<summary markdown="span"><span class="pub-title">Centauri: Enabling Efficient Scheduling for Communication–Computation Overlap in Large Model Training via Communication Partitioning</span><span class="pub-meta">, Peking University et al. (ASPLOS 2024)</span><a class="pub-link" href="https://doi.org/10.1145/3620666.3651379" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Large models are trained with hybrid parallelism, whose communication is a large overhead, and existing overlap methods "lean towards either fine-grained kernel fusion or limited operation scheduling", in the words of the paper. Centauri tackles communication–computation overlap at the *operator scheduling* level. The key idea is to **decompose operators**, and then to schedule the pieces across multiple CUDA streams, so that communication overlaps with useful compute more consistently than with a *whole-operator overlap*. The *communication partitioning* of the title means that a large [collective](https://en.wikipedia.org/wiki/Collective_operation) (or data movement phase) is broken into smaller chunks, which are scheduled earlier and more frequently, and keep the pipeline busy.

Concretely, the partition space has three dimensions: primitive substitution (rewriting a collective as other primitives), topology-aware group partitioning (e.g., hierarchical intra-/inter-node collectives), and workload partitioning (chunking the data). The resulting pieces are scheduled hierarchically at the operation, layer, and model levels of hybrid-parallel training.

Centauri reaches up to **1.49×** speedup over prevalent methods across various hybrid-parallel training configurations, and the paper received an ASPLOS 2024 Best Paper award. Even where fused kernels are written later, it remains a useful baseline: it shows what better scheduling alone gains, with no change to the kernels.

Compared with [Flux](#flux), [TileLink](#tilelink) and [Triton-distributed](#triton-distributed):
- Centauri is mainly about the **multi-stream scheduling of decomposed operators**, and improves the overlap among kernels that stay separate; it does not require in-kernel collectives.
- Flux, TileLink and Triton-distributed pursue the **in-kernel** fusion of compute and communication, which overlaps them at a finer granularity, but often requires more specialized code generation and primitives.
- Centauri is therefore a good fit when the bottleneck is the *orchestration* (stream scheduling, dependencies, coarse kernel granularity), whereas approaches like Flux target the cases where overheads persist even with an aggressive overlap of streams.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="flux">
<summary markdown="span"><span class="pub-title">FLUX: Fast Software-based Communication Overlap on GPUs Through Kernel Fusion</span><span class="pub-meta">, ByteDance & Peking University</span><a class="pub-link" href="https://arxiv.org/abs/2406.06858" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

In Megatron-LM-style tensor parallelism, the communication of each layer takes a significant portion of the overall runtime. Flux is explicitly designed around these tensor-parallel patterns. Instead of treating a tensor-parallel layer as "one big GEMM + one big NCCL collective", Flux **decomposes** these operations into fine-grained pieces (e.g., tiles or chunks), and fuses them into kernels that overlap communication with computation aggressively. It claims overlap levels that are hard to reach with the standard multi-stream overlap of separate kernels.

Flux reports up to about **96%** of the communication overlapped. For training, it reaches up to **1.24×** speedup over Megatron-LM on a cluster of **128 GPUs** (of various GPU generations and interconnects), and for inference, up to **1.66×** for the **prefill** and **1.30×** for the **decoding** over vLLM on **8 GPUs**.

Flux compares with the related work as follows:
- Compared with [Centauri](#centauri), Flux aims for **overlap *within* the operator**, not only between independent kernels.
- Compared with [Triton-distributed](#triton-distributed), Flux is less a general programming model and more a way to deliver peak performance for a *specific family* of LLM operations (the Megatron tensor-parallel attention and MLP patterns).
- [TileLink](#tilelink) is motivated partly by the effectiveness of Flux: it tries to keep a performance like that of Flux while raising the programming level.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="fused-computation-collective">
<summary markdown="span"><span class="pub-title">Optimizing Distributed ML Communication with Fused Computation–Collective Operations</span><span class="pub-meta">, AMD (SC 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2305.06942" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

In distributed training and inference, the collectives sit on the critical path: a kernel writes its whole output before the collective that sends it can start. Punniyamurthy et al. identify common "compute + collective" motifs that appear repeatedly in distributed training and inference graphs, and implement each one as a **single fused kernel** with **in-kernel collectives**. Workgroups send their results to remote GPUs (GPU-initiated communication) as soon as their computation finishes, while other workgroups of the same kernel keep computing, and the scale-up transfers are [zero-copy](https://en.wikipedia.org/wiki/Zero-copy) (written directly into the memory of the peer GPU). The contribution is partly a taxonomy (these patterns show up everywhere) and partly a concrete engineering result: fused kernels in Triton and HIP that remove the overheads of intermediate writes, kernel boundaries and separate communication steps.

The three fused operators are an embedding with an [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)), a GEMV with an all-reduce, and a GEMM with an all-to-all. Intra-node, on four GPUs, the execution time is 20% lower on average (up to 32%) for embedding + all-to-all, 13% (up to 22%) for GEMV + all-reduce, and 12% (up to 20%) for GEMM + all-to-all. Inter-node, it is 31% lower on average (up to 58%) for embedding + all-to-all, and a simulation with ASTRA-Sim shows 21% lower execution time for DLRM on 128 nodes. So the gain is material even for *simple* two-stage pipelines, when the communication is tightly coupled to the compute.

These patterns are useful to keep in mind when comparing systems:
- [Flux](#flux) focuses on tensor-parallel LLM patterns, while the catalog of this paper is **broader**.
- [Triton-distributed](#triton-distributed) and [TileLink](#tilelink) offer programming models to build such fused kernels, while this paper provides **hand-built instances** and measurements that show why the fusion matters.
- Unlike the [in-network computing](#in-network-collectives) (INC) and offload approaches, these fused kernels still communicate through GPU-side primitives, with no execution in the network.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepseek-v3">
<summary markdown="span"><span class="pub-title">DeepSeek-V3 Technical Report (includes DualPipe)</span><span class="pub-meta">, DeepSeek-AI</span><a class="pub-link" href="https://arxiv.org/abs/2412.19437" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSeek-V3 is a 671B-parameter [MoE](https://en.wikipedia.org/wiki/Mixture_of_experts) model, with 37B parameters active per token, trained on 14.8T tokens in 2.788M H800 GPU hours (about 5.6M USD at 2 USD per GPU hour). It keeps MLA and DeepSeekMoE from [DeepSeek-V2](#deepseek-v2), and adds an auxiliary-loss-free load balancing, multi-token prediction, FP8 mixed-precision training and the DualPipe schedule. The report is a systems-and-training reference more than a single *one-trick* optimization: it shows the engineering trade-offs that purely algorithmic papers often omit, and its scaling observations show where communication becomes dominant and which optimizations remain on the table.

Its communication relies on custom cross-node all-to-all kernels (later open-sourced as DeepEP), which use only 20 of the 132 SMs of each GPU and exploit both InfiniBand (IB) and NVLink, as each token is sent to at most 4 nodes. These kernels overlap with computation at the *schedule* level (DualPipe, below), rather than being fused into the GEMM kernels, as in [Comet](#comet), or offloaded to the network, as with [in-network collectives](#in-network-collectives). This makes DeepSeek-V3 a good *control point* for these approaches: a strong end-to-end system that already works at scale, on 2,048 H800 GPUs.

Three of the training techniques deserve more detail:
- **FP8 training.** The report validates FP8 training on an extremely large-scale model for the first time. Unlike earlier recipes, which use E5M2, with its wider range, for the gradients, it uses E4M3 for all tensors, for a higher precision. This works because of a fine-grained scaling, computed online: per tile of 1×128 activations and per block of 128×128 weights, instead of once per tensor, so that an outlier only affects its own group. The GEMMs accumulate in higher precision, and the optimizer moments are kept in [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).
- **Sequential multi-token prediction (MTP).** Unlike the parallel independent heads of [Gloeckle et al.](https://arxiv.org/abs/2404.19737), each MTP module is a Transformer layer that takes the hidden states of the previous depth and the embedding of the next token, which keeps the full causal chain. As the draft model of speculative decoding, its second token is accepted 85% to 90% of the time, for about 1.8× more tokens per second.
- **Auxiliary-loss-free load balancing.** An auxiliary balance loss competes with the language modeling objective and can hurt quality. Instead, a bias per expert is added to the routing scores, only to select the top-k experts, and after each step the bias of the overloaded experts is decreased and that of the underloaded ones increased.

Post-training [distills](https://en.wikipedia.org/wiki/Knowledge_distillation) the reasoning ability of DeepSeek-R1 (trained with large-scale reinforcement learning on top of DeepSeek-V3-Base) into DeepSeek-V3: R1 generates the reasoning data, and V3 is fine-tuned on it. This improves its math and code while keeping its output style and controllable length, without making it a pure reasoning model.

The cross-node expert parallelism of DeepSeek-V3 has a compute-to-communication ratio of about 1:1. **DualPipe**, a bidirectional pipeline-parallelism algorithm introduced in the report, reduces the pipeline [bubbles](https://en.wikipedia.org/wiki/Pipeline_stall) and hides this communication, by overlapping computation and communication within a pair of forward and backward chunks.

Each chunk is split into four components (attention, all-to-all dispatch, MLP and all-to-all combine), and the backward attention and MLP are further split into their input-gradient and weight-gradient parts, as in [Zero Bubble](#zero-bubble). These components are rearranged so that one micro-batch runs its forward pass while another runs its backward pass, and the communication of each hides behind the compute of the other. The micro-batches are fed from both ends of the pipeline, which keeps the GPUs busier than sequential schedules such as 1F1B or Zero Bubble.

The result is fewer bubbles and a full overlap of computation and communication, at the cost of two copies of the model parameters (one per direction) and a slightly higher activation memory. The duplication is affordable because the large expert-parallel size of DeepSeek-V3 leaves each rank with few parameters, and memory savings (recomputation, a CPU-resident [EMA](https://en.wikipedia.org/wiki/Exponential_smoothing) of the parameters, an embedding and an output head shared with the MTP module) allow training without costly tensor parallelism.

Unlike [PipeFill](#pipefill), which accepts the bubbles and fills them with unrelated jobs, DualPipe removes them by densifying the schedule of the main job itself, so the two approaches are complementary rather than competing. The following picture illustrates the DualPipe schedule, with the overlapped forward and backward pairs in the middle of each row:

{: .picture}
<img loading="lazy" width="92.8%" height="92.8%" src="{{ site.assets }}/distributed-training/imgs/pipeline_dualpipe.svg"/>

{: .caption}
DualPipe on 8 GPUs: micro-batches 1 to 8 (light) enter at GPU 0, and micro-batches 9 to 16 (dark) enter at GPU 7. Each GPU holds two pipeline stages. Cells with a shared black border are the forward and backward passes of two different micro-batches executed together, so that the communication of one overlaps with the computation of the other. Adapted from: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="zero-bubble">
<summary markdown="span"><span class="pub-title">Zero Bubble Pipeline Parallelism</span><span class="pub-meta">, Sea AI Lab (ICLR 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2401.10241" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Pipeline parallelism leaves [bubbles](https://en.wikipedia.org/wiki/Pipeline_stall): idle time on each GPU while it waits for the other stages. Zero Bubble is the first scheduling strategy to achieve zero pipeline bubbles under synchronous training semantics. The key insight is to **split the backward pass** into two independent computations: the input gradient (B), which the previous stage needs to continue the backward pass, and the weight gradient (W), which only the optimizer step needs. The later-starting W passes can then fill the tail-end bubbles that a standard 1F1B schedule leaves behind.

The paper offers two handcrafted schedules and an automatic scheduler, which formulates the problem as [integer linear programming](https://en.wikipedia.org/wiki/Integer_programming) and pairs it with a heuristic for large numbers of micro-batches. **ZB-H1** keeps the peak activation memory of 1F1B, $$p M_B$$ for $$p$$ stages and an activation memory $$M_B$$ per micro-batch, and cuts the bubble to about a third. **ZB-H2** removes the bubbles entirely, at the cost of a peak memory of $$(2p-1) M_B$$, about twice as much, and of an optimizer step that bypasses the synchronization before it and validates the update after it, rolling it back if needed.

With $$T_F$$, $$T_B$$ and $$T_W$$ the times of the forward (F), B and W passes of one micro-batch, the bubble of each GPU is $$(p-1)(T_F + T_B + T_W)$$ for 1F1B, $$(p-1)(T_F + T_B - T_W)$$ for ZB-H1 and $$(p-1)(T_F + T_B - 2T_W)$$ for ZB-H2 (Table 2 of the paper). With equal pass times and 4 GPUs, as in the pictures below, that is 9, 3 and 0 time units per iteration (for ZB-H2, once the next iteration overlaps with the current one, as the caption explains). The zero-bubble schedules outperform 1F1B by up to 23% in throughput under a similar memory limit, and by up to 31% when the memory constraint is relaxed.

The method is orthogonal to data, tensor and ZeRO parallelism, and can replace the pipeline-parallel component as it is. Like the DualPipe schedule of [DeepSeek-V3](#deepseek-v3), it removes bubbles by reshaping the schedule of the main job, while [PipeFill](#pipefill) fills them with other jobs. The following picture illustrates both schedules on 4 GPUs, where ZB-H2 fills the warm-up bubbles of ZB-H1 with extra forward passes:

{: .picture}
<img loading="lazy" width="71.6%" height="71.6%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h1.svg"/>
<img loading="lazy" width="71.6%" height="71.6%" src="{{ site.assets }}/distributed-training/imgs/pipeline_zb_h2.svg"/>

{: .caption}
ZB-H1 (top) and ZB-H2 (bottom). Every backward pass is split into a backward for inputs and a backward for weights, and the latter fill the bubbles. ZB-H1 keeps the activation memory of 1F1B; ZB-H2 uses about twice as much, and when the optimizer step does not synchronize the stages, the next iteration starts on the first GPUs while the last GPUs finish the current one. Adapted from: [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="fp6-llm">
<summary markdown="span"><span class="pub-title">FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design</span><span class="pub-meta">, Microsoft & University of Sydney (USENIX ATC 2024, as Quant-LLM)</span><a class="pub-link" href="https://arxiv.org/abs/2401.14112" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

In FP16, the weights of a 70B model take 140 GB, more than the 80 GB of one A100 GPU, while in six-bit floating point (FP6) they take 52.5 GB. FP6 is an appealing sweet spot: it preserves the quality more reliably than 4-bit [quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)), which looks fine on zero-shot benchmarks but degrades on harder tasks such as code generation and summarization. FP6-LLM makes FP6 quantization practical for LLM inference on GPUs.

The catch is that no existing system gave FP6 real Tensor Core support: the irregular 6-bit width causes misaligned memory accesses to the weights, and their de-quantization at runtime, back to a format that the Tensor Cores can multiply, is a heavy overhead. The answer of the paper is **TC-FPx**, the first full-stack GPU kernel design with unified Tensor Core support for floating-point weights of arbitrary bit-widths, which handles the memory layout and the on-the-fly de-quantization together.

TC-FPx works as follows:
- The weights are FP6 (1 sign, 3 exponent and 2 mantissa bits, denoted **E3M2**), in a **W6A16** linear layer (6-bit weights, FP16 activations): only the weights are quantized, and the results of the matmuls stay in FP16.
- The weights are quantized from FP16 to FP6 once, offline, and stay in 6 bits in memory, which is the point: a smaller footprint and faster loads from DRAM. They are pre-packed into an aligned layout, where the odd 6-bit width is split into 2-bit and 4-bit fragments, so that the loads are not misaligned.
- At runtime, every time the weights are loaded, the [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) cores (the CUDA cores) stitch the fragments of a slice back together with bit shifts and masks, and de-quantize them to FP16 in registers, for the Tensor Cores to multiply.
- The cost of the de-quantization is hidden: the kernel is [software-pipelined](https://en.wikipedia.org/wiki/Software_pipelining), so that the de-quantization of slice N+1 on the SIMT cores overlaps with the matmul of slice N on the Tensor Cores.

Integrating TC-FPx into an existing inference stack yields the end-to-end FP6-LLM system. Results: LLaMA-70B can be served on a single GPU, with 1.69×–2.65× higher normalized inference throughput (tokens per GPU-second) than the FP16 baseline on two GPUs. At the kernel level, the 6-bit linear layer runs up to 2.6× faster than cuBLAS FP16 and up to 1.9× faster than TensorRT-LLM's 8-bit (W8A16) kernels.

It is an algorithm-system co-design: the kernels turn the theoretical compression of FP6 into actual serving speedups. The source code is available at [github.com/usyd-fsalab/fp6_llm](https://github.com/usyd-fsalab/fp6_llm). [QLoRA](#qlora) also keeps low-bit weights in memory and de-quantizes them for each matmul, but to fine-tune a model in less memory.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="llama-3">
<summary markdown="span"><span class="pub-title">The Llama 3 Herd of Models</span><span class="pub-meta">, Meta</span><a class="pub-link" href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[Llama 3](https://en.wikipedia.org/wiki/Llama_(language_model)) is "a herd of language models that natively support multilinguality, coding, reasoning, and tool usage", with 8B, 70B and 405B parameters and a context window of 128K tokens. The flagship model, Llama 3 405B, has 126 layers, a token representation dimension of 16,384 and 128 attention heads, and is trained on up to 16K H100 GPUs with 4D parallelism (tensor, pipeline, context and data).

The authors used [scaling laws](https://en.wikipedia.org/wiki/Neural_scaling_law) ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556), the [Chinchilla](#chinchilla) paper) to determine the optimal model size for their flagship model, given their pre-training compute budget (section 3.2.1). They first correlate the negative log-likelihood of the compute-optimal models on downstream tasks with the training FLOPs, and then establish a sigmoidal relation between that log-likelihood and the task accuracy (figure 4 of the paper).

The compute-optimal models are the minima of the IsoFLOPs curves (the validation loss against the number of training tokens, for a fixed compute budget), and their number of training tokens follows a power law of the compute budget. Extrapolated to the budget of the flagship model, this law suggests a 402B-parameter model trained on 16.55T tokens. The following picture illustrates the IsoFLOPs curves and the power law (figures 2 and 3 of the paper):

{: .picture}
<img loading="lazy" width="92.8%" height="92.8%" src="{{ site.assets }}/publications/llama3_scaling_laws.svg"/>

{: .caption}
<b>Left:</b> <b>Scaling law IsoFLOPs curves</b> between $$6 \times 10^{18}$$ and $$10^{22}$$ FLOPs. The loss is the negative log-likelihood on a held-out validation set. Measurements at each compute scale are approximated by a second-degree polynomial. <b>Right:</b> <b>Number of training tokens in identified compute-optimal models as a function of pre-training compute budget.</b> The line is the fitted scaling-law prediction. The compute-optimal models correspond to the parabola minima of the left plot. Adapted from: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

The model architecture does not deviate from that of [Llama 2](#llama-2), except that the authors:

1. use grouped query attention with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding;
2. "use an attention mask that prevents self-attention between different documents within the same sequence as is important in continued pre-training on very long sequences";
3. use a vocabulary with 128K tokens: 100K from `tiktoken` and 28K for better non-English support; and
4. increase the RoPE base frequency hyperparameter to 500,000 to better support longer contexts.

Training has two stages: pre-training via next-token prediction, and post-training, where the model is "tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning)." The improvements were performed at 3 levels:
1. data: better quality, quantity, pre-processing and curation, with "15T multilingual tokens, compared to 1.8T tokens for Llama 2";
2. scale: almost $$50 \times$$ the pre-training compute of the largest Llama 2 model, $$3.8 \times 10^{25}$$ FLOPs, about $$6 N D$$ (6 FLOPs per parameter and token: 2 in the forward pass, 4 in the backward pass) for $$N$$ = 405B parameters and $$D$$ = 15.6T training tokens; and
3. managing complexity: a regular transformer with minor adaptations instead of a Mixture-of-Experts (MoE) model, and "a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO), as opposed to more complex reinforcement learning algorithms." (section 4)

The authors also experiment with adding image, video and speech capabilities, in three more stages: the pre-training of the image and speech encoders (sections 7 and 8); the training of a vision adapter, which aligns the image representations with the language representations, with a video adapter on top of it that aggregates information across frames (section 7); and the training of a speech adapter, which converts the speech encodings into token representations.

The image encoder is a standard [vision transformer](https://en.wikipedia.org/wiki/Vision_transformer) trained to align images and text, the ViT-H/14 variant. They introduce cross-attention layers (using grouped-query attention) between the visual token representations produced by the image encoder and the token representations produced by the language model, after every 4th self-attention layer. The following picture illustrates the resulting multimodal model, where the color of each module marks its training stage:

{: .picture}
<img loading="lazy" width="92.8%" height="92.8%" src="{{ site.assets }}/publications/llama_3_multi_modal.svg"/>

{: .caption}
<b>Illustration of the compositional approach to adding multimodal capabilities to Llama 3 studied in the paper.</b> This approach leads to a multimodal model that is trained in five stages: <b>(1)</b> language model pre-training, <b>(2)</b> multi-modal encoder pre-training, <b>(3)</b> vision adapter training, <b>(4)</b> model finetuning, and <b>(5)</b> speech adapter training. Adapted from: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

Section 5 evaluates the pre-trained and the post-trained models and their safety: Llama 3 delivers a quality comparable to leading language models such as GPT-4 on a plethora of tasks, and the release includes Llama Guard 3, a model for the safety of inputs and outputs.

Section 6 makes the inference of Llama 3 405B efficient in two ways. First, its BF16 weights, 810 GB, do not fit in the 640 GB of one machine of 8 H100 GPUs of 80 GB, so it runs with pipeline parallelism on 16 GPUs. Second, FP8 quantization is applied to most parameters and activations of the feed-forward layers, but not to the self-attention layers, nor to the first and last Transformer layers. Similarly to Xiao et al. (2024b), it uses dynamic scaling factors, with an upper bound of 1200, and row-wise quantization, which computes the scaling factors across the rows of the parameter and activation matrices.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="universal-checkpointing">
<summary markdown="span"><span class="pub-title">Universal Checkpointing: Efficient and Flexible Checkpointing for Large Scale Distributed Training</span><span class="pub-meta">, Microsoft DeepSpeed</span><a class="pub-link" href="https://arxiv.org/abs/2406.18820" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

According to the paper, state-of-the-art distributed [checkpointing](https://en.wikipedia.org/wiki/Application_checkpointing) (model save/resume) requires "static allocation of GPU resources at the beginning of training and lacks the capability
to resume training with a different parallelism strategy and hardware configuration". Usually, it is not possible to resume when hardware changes during the training process. To this end, the paper proposes **Universal Checkpointing** (UCP), "a technique that enables efficient checkpoint
creation while providing the flexibility of resuming on arbitrary parallelism strategy". The technique also offers "improved resilience to hardware failures through continued training on remaining healthy hardware, and reduced training time through opportunistic exploitation of elastic capacity".

This is achieved by writing in the universal checkpoint format, which allows "mapping parameter fragments into training ranks of arbitrary model-parallelism configuration", and with a universal checkpoint language that allows for "converting distributed checkpoints into the universal checkpoint format". The UCP format consolidates the distributed shards into **atom checkpoints**, one consolidated set of files per model parameter (its FP32 weights and optimizer states), which can then be re-sharded onto a different layout (data, tensor, pipeline and sequence parallelism, and ZeRO stages).

Saving a checkpoint costs the same as before, and loading a universal checkpoint takes 1.14× to 1.37× the time of loading a standard one. Training resumed on another parallelism configuration follows the loss curve of the original run, for models from GPT-3 medium (350M parameters) and LLaMA 7B to BLOOM (176B) and a Mixtral-style MoE (42B). Universal Checkpointing comes from the DeepSpeed team, as do [ZeRO++](#zero-pp), [ZeRO-Offload++](#zero-offload-pp) and [Domino](#domino).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="domino">
<summary markdown="span"><span class="pub-title">Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping</span><span class="pub-meta">, Microsoft DeepSpeed</span><a class="pub-link" href="https://arxiv.org/abs/2409.15241" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

In tensor parallelism (TP), each layer ends with an all-reduce that the next layer waits for, and this communication takes 17% to 43% of the end-to-end training iteration time of GPT-3 13B. The current efforts to overlap computation and communication during TP are not enough: "in the cases where collective communication takes much longer than a single GeMM computation, most of the communication time still stands out as the major training overhead", and this overhead grows as the GPUs compute faster.

Domino "provides a generic scheme to hide communication behind computation" when large LLMs are trained with TP. The key idea is to **break the data dependency** of a single batch into smaller independent pieces, and to pipeline their training, so that the communication of one piece overlaps with the computation of another, with "a much wider scope of computation and communication overlapping (e.g., AllReduce not only overlaps with a single GeMM, but also LayerNorm, DropOut, etc)".

A TP layer computes $$X \cdot A \cdot B$$, followed by an all-reduce, where $$X$$ is the input, and $$A$$ and $$B$$ are the first and second groups of linear weights of the attention or of the MLP. "Domino provides extra and generic tensor partition in two dimensions on every GPU: row-wise split on inputs X and column-wise split on weights B on top of original TP model partitions". This breaks $$X \cdot A \cdot B$$ into smaller compute units without data dependency, which are pipelined with the collective communication.

As the attention computation is modularized and highly optimized (e.g. [FlashAttention](#flashattention)), $$A$$ stays untouched: Domino slices only the input $$X$$ (section 3.2 of the paper), the second group of weights $$B$$ (section 3.3), or both (section 3.4). The following picture illustrates how both splits break $$X \cdot A \cdot B$$ into independent pieces, so that the all-reduce of one piece overlaps with the computation of another:

{: .picture}
<img loading="lazy" width="92.4%" height="92.4%" src="{{ site.assets }}/publications/domino.svg"/>

{: .caption}
**Left:** Domino row-wise (batch-dim) split on input <i>X</i>. The AllReduce of <i>Y</i><sub>1</sub> overlaps with the computation of <i>X</i><sub>2</sub> ⊗ <i>A</i> ⊗ <i>B</i> (intra-layer overlap), and the AllReduce of <i>Y</i><sub>2</sub> with the computation of <i>X</i><sub>1</sub> ⊗ <i>A</i> ⊗ <i>B</i> of the next layer (inter-layer overlap). **Right:** Domino col-wise (last-dim) split on weights <i>B</i>. The AllReduce of <i>Y</i><sub>3</sub> overlaps with the computation of <i>X</i> ⊗ <i>A</i> ⊗ <i>B</i><sub>2</sub>, but the next layer needs the concatenation of <i>Y</i><sub>3</sub> and <i>Y</i><sub>4</sub>, so the overlap stays within the layer. Here ⊗ is a matrix multiplication (GeMM). Adapted from: [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](https://arxiv.org/abs/2409.15241)

Compared with Megatron-LM, "Domino achieves up to 1.3x speedup for LLM training on Nvidia DGX-H100 GPUs", on GPT-3 models (less on Llama-2, whose rotary embeddings add a data dependency). [Flux](#flux) hides the same TP communication by fusing it into the GEMM kernels, while Domino, like [Centauri](#centauri), changes the schedule of the kernels.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2023</span></td>
<td markdown="block">
<details id="simplifying-transformer-blocks">
<summary markdown="span"><span class="pub-title">Simplifying Transformer Blocks</span><span class="pub-meta">, ETH Zurich (ICLR 2024)</span><a class="pub-link" href="https://arxiv.org/abs/2311.01906" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The standard Transformer block interweaves attention and MLP sub-blocks with skip connections and normalization layers in precise arrangements, and "seemingly minor changes can significantly reduce training speed, or render models untrainable", as the abstract of the paper says. The paper asks to what extent the block can be simplified, and proposes a simpler block, motivated by signal propagation theory, that removes the [skip connections](https://en.wikipedia.org/wiki/Residual_neural_network), the value and projection parameters, the sequential sub-blocks (attention and MLP run in parallel) and the [normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)) layers. [DeepNet](#deepnet) and [mHC](#mhc) go the other way: they keep the residual connections, and change how they are scaled or widened.

The paper claims that the simplified block matches the per-update training speed and performance of standard autoregressive decoder-only and BERT encoder-only models, with 16% faster training throughput (15% in the arXiv version) while using 15% fewer parameters. The experiments are at the 100–300M-parameter scale. The following picture compares the simplified block with two standard Transformer blocks:

{: .picture}
<img loading="lazy" width="64.1%" height="64.1%" src="{{ site.assets }}/publications/simplifying_transformer_blocks.svg"/>

{: .caption}
Comparison between different Transformer blocks. (Left) The standard Pre-LN block. (Top right) The most simplified block of the paper, “SAS-P, no norm”. (Bottom right) The parallel block (Zhao et al., 2019; Wang & Komatsuzaki, 2021). Like the parallel block, the simplified block eschews the need for sequential sub-blocks, but it additionally removes all skip connections and normalisation layers, as well as value and projection parameters. Here, ⊗ denotes a matrix multiplication, and ⊕ denotes a (potentially weighted) sum. Adapted from: [Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="fp8-training">
<summary markdown="span"><span class="pub-title">Training and inference of large language models using 8-bit floating point</span><span class="pub-meta">, Graphcore</span><a class="pub-link" href="https://arxiv.org/abs/2309.17224" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

FP8 has a narrower range of representable values than the 16-bit formats, so the scaling of each tensor must be chosen carefully, and the scaling methods that are well studied for integer formats are less so for FP8. The paper "presents a methodology to select the scalings for FP8 linear layers, based on dynamically updating per-tensor scales for the weights, gradients and activations." It uses FP8-E4 for the weights and activations and FP8-E5 for the gradients, with 4 and 5 bits of exponent, respectively, while the accumulation of the matmuls is performed in higher precision (FP16, BF16 or FP32). The scaling is applied to the exponent (a power-of-two shift of the [exponent bias](https://en.wikipedia.org/wiki/Exponent_bias)), not by rescaling the final value.

Two scaling techniques are tested: **AMAX**, where the scale of each tensor is derived from its absolute maximum, and **CSCALE**, which keeps the same constant scale for all tensors. AMAX shows no major degradation, while CSCALE converges only within a narrow range of scales. Results compare FP8 to FP16 on GPT and Llama 2 models from 111M to 70B parameters, but not to BF16. The experiments run on [Graphcore](https://en.wikipedia.org/wiki/Graphcore) IPUs, which support FP8 in software only.

The algorithm is given in Figures 1 and 2 of the paper. Note: FP8 matmuls run at about 2× the FP16 and BF16 throughput on hardware that supports them, and the per-tensor scaling only adds cheap element-wise work, so the end-to-end gain is below 2×, but still significant. The per-tensor scales extend the loss scaling of [mixed precision training](#mixed-precision-training); [DeepSeek-V3](#deepseek-v3) later scales FP8 training up with finer scales, per tile and per block, and [NVFP4](#nvfp4) goes down to 4 bits.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="zero-offload-pp">
<summary markdown="span"><span class="pub-title">DeepSpeed ZeRO-Offload++: 6x Higher Training Throughput via Collaborative CPU/GPU Twin-Flow</span><a class="pub-link" href="https://github.com/microsoft/DeepSpeed/tree/offloadpp-news/blogs/deepspeed-offloadpp" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

"System efficiency is still far from optimal when adopting ZeRO-Offload in some scenarios. Especially in the cases like small batch training, model that could not fit into GPU memory but not orders-of-magnitude bigger than GPU memory capacity, CPU offload not only introduce long end-to-end latency, but also underutilized GPU computation resources." With that in mind, ZeRO-Offload++ introduces 3 features:
- Twin-Flow: instead of an all-or-nothing policy (i.e. offload all or none of the optimizer states), "Twin-Flow allows a portion of optimizer states to be held in CPU memory and the other portion of optimizer states remaining in GPU memory. When optimization step is triggered, both CPU and GPU can do parameter updates simultaneously." The user can choose the ratio of parameters in CPU and GPU (the offload ratio). "Therefore, with Twin-Flow, we can achieve decent GPU memory and core utilization rate, at the same time reduce training iteration time in optimizer offloading cases."
- MemCpy reduction: an on-going optimization that removes unnecessary device-to-host and host-to-device memory copies (no details yet);
- CPUAdam optimization: an on-going optimization that casts FP16 to FP32 on the fly for CPUAdam (no details yet).

A smaller offload ratio uses more GPU memory and cores, and shortens the training iteration. With Twin-Flow, the blog reports a 6× training speedup for Meta OPT models on a single DGX-H100 (80 GB), and 3× on a single DGX-A100 (40 GB). Where ZeRO-Offload++ speeds up the CPU offloading of ZeRO, [ZeRO++](#zero-pp) reduces its communication. The following picture illustrates how Twin-Flow splits the parameter update between the CPU and the GPU:

{: .picture}
<img loading="lazy" width="62.3%" height="62.3%" src="{{ site.assets }}/publications/zerooffloadplusplus.svg"/>

{: .caption}
Twin-Flow with an offload ratio of 0.4: CPUAdam updates the first 40% of the parameters (weights, gradients and optimizer states) on the CPU, while FusedAdam updates the other 60% on the GPU. Adapted from: [DeepSpeed ZeRO-Offload++: 6x Higher Training Throughput via Collaborative CPU/GPU Twin-Flow](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-offloadpp)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="distflashattn">
<summary markdown="span"><span class="pub-title">DistFlashAttn: Distributed Memory-efficient Attention for Long-context LLMs Training (LightSeq)</span><span class="pub-meta">, UC Berkeley et al.</span><a class="pub-link" href="https://arxiv.org/abs/2310.03294" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[FlashAttention](#flashattention) turns the quadratic peak memory of attention into a linear one, by computing it tile by tile with an online softmax and never materializing the full score matrix. Yet the activations of a long-context LLM still outgrow the memory of one GPU. DistFlashAttn extends FlashAttention from a single GPU to a distributed, **sequence-parallel** setting for the **training** of long-context LLMs.

The split is by sequence (tokens), not by head or by feature dimension: each GPU stores the Q, K and V of its own chunk of tokens only, with the full hidden dimension and all heads, but fewer rows. The challenge is to do this efficiently under *causal* attention, and the paper contributes three techniques:

1. **Token-level workload balancing:** with causal masking, a token only attends to its prefix, so a GPU holding *early* tokens has little work, while one holding *late* tokens does much more: with 8 GPUs, the last one computes 15 times the attention of the first one, a severe imbalance in naive sequence parallelism. DistFlashAttn rebalances this work across devices ([load balancing](https://en.wikipedia.org/wiki/Load_balancing_(computing))), so that no worker sits idle waiting for the heavily loaded ones.

2. **Overlapping KV communication:** like Ring Attention, K/V chunks are passed between GPUs so each device can attend to remote tokens, and that communication is overlapped with the local attention computation, hiding the transfer latency behind useful work.

3. **[Rematerialization](https://en.wikipedia.org/wiki/Rematerialization)-aware gradient checkpointing:** standard checkpointing recomputes the forward pass of a layer during the backward pass to save memory, but the backward pass of FlashAttention already recomputes the attention, so that a naive checkpointing computes it one extra time. DistFlashAttn places the checkpoints at the output of the FlashAttention kernel instead of at the layer boundaries, which avoids the double work.

Together, these techniques train **8× longer sequences** than Ring Self-Attention (2× to 8× longer than Megatron-LM with FlashAttention), and run faster than the alternatives of the time: 4.45× to 5.64× faster than Ring Self-Attention, 1.24× to 2.01× than Megatron-LM with FlashAttention, 1.67× than Ring Attention and 1.26× to 1.88× than DeepSpeed-Ulysses, on Llama-7B and its variants, with sequences of 32K to 512K tokens.

DistFlashAttn is essentially Ring Attention plus two fixes for causal, long-context LLM training, where the load imbalance and the checkpointing interaction bite: the load balancing and the rematerialization-aware checkpointing. Ring Attention itself is a general sequence-parallel attention, for inference and training, causal or not. [LoongServe](#loongserve) applies ring attention to serving instead.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="chimera">
<summary markdown="span"><span class="pub-title">Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion</span><span class="pub-meta">, Peking University et al. (HPCA 2023)</span><a class="pub-link" href="https://ieeexplore.ieee.org/document/10071018" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

As hardware compute throughput has outpaced memory bandwidth, even compute-heavy operators such as GEMM and convolution end up [memory-bound](https://en.wikipedia.org/wiki/Roofline_model) when run as a chain: each operator writes its full output to memory, only for the next one to read it back. Existing ML compilers lacked both a precise analysis and a good optimization of these chains. Chimera (Zheng et al.) is a compiler framework that fuses such *chains of compute-intensive operators* (back-to-back GEMMs, or GEMM-then-softmax in attention) into single kernels, to improve [data locality](https://en.wikipedia.org/wiki/Locality_of_reference).

The key idea is to break each operator into a series of **computation blocks**, and to optimize at two levels. For the *inter-block* optimization, an analytical model picks the execution order of the blocks that minimizes the data movement between them, so that intermediate results stay in fast on-chip memory instead of round-tripping to DRAM. For the *intra-block* optimization, Chimera generates an efficient hardware-specific *micro-kernel*.

Examples from the paper:
- The GEMM chain $$C = A \times B$$ and $$E = C \times D$$, where each tile of $$C$$ is used to compute $$E$$ as soon as it is produced, without materializing the whole $$C$$. A [thread block](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)) that computed one tile of $$C$$ would add its partial products into a block of rows of $$E$$ that other thread blocks also update, which needs atomics, and one that computed one tile of $$E$$ would need a whole block of rows of $$C$$, recomputed by every thread block of these rows. Instead, each thread block is assigned a block of rows of $$A$$ and $$E$$: it produces the tiles of $$C$$ of these rows one at a time and uses each one immediately, iterating over all columns of $$B$$ and $$D$$.
- A batch GEMM chain (attention): $$Q K^\top$$, a softmax, then a product with $$V$$.
- A convolution chain (CNNs): a 3×3 convolution, a ReLU, a 1×1 convolution and a ReLU, common in ResNet-style networks and memory-bound at certain input shapes.

Because it relies on an analytical cost model rather than exhaustive auto-tuning, Chimera avoids the long tuning times of search-based compilers such as [Ansor](https://arxiv.org/abs/2006.06762). On batch-GEMM chains and convolution chains, it achieves up to 2.87×, 2.29× and 2.39× speedups over hand-tuned libraries on CPU, GPU (A100) and NPU (Huawei Ascend), respectively, and up to 2.29×, 1.64× and 1.14× over state-of-the-art compilers on the same platforms. XLA leaves these compute-intensive operators to library calls that it does not fuse ([Operator Fusion in XLA](#operator-fusion-in-xla)), and [FlashAttention](#flashattention) is a hand-written instance of the same fusion, for attention.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="zero-pp">
<summary markdown="span"><span class="pub-title">ZeRO++: Extremely Efficient Collective Communication for Giant Model Training</span><span class="pub-meta">, Microsoft</span><a class="pub-link" href="https://arxiv.org/abs/2306.10209" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

DeepSpeed ZeRO's compute throughput is limited by the high communication cost from gathering weights in forward pass, backward pass, and averaging gradients. This is most prominent on clusters with low bandwidth, and at very small batch sizes per GPU.

**Background, communication pipeline:** "Assume the model size as 𝑀. During the forward pass, ZeRO conducts an all-gather operation to collect all the parameters (𝑀) needed to train for all model layers."

"In the backward pass, ZeRO re-collects parameters (𝑀) with all-gather first, then each GPU can compute local gradients. After that, ZeRO operates reducescatter function to aggregate and redistribute gradients (𝑀) across accelerators. In total, ZeRO has a total communication volume of 3𝑀, spreads evenly across 2 all-gather and 1 reduce-scatter."

The paper introduces three communication reduction techniques, packed as ZeRO++:
1. **Quantized Weight Communication for ZeRO (qwZ):** a block [quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) of the forward all-gather, from FP16 (2 bytes) to INT8 (1 byte), where each block of parameters has its own scale, instead of one scale for all the parameters. The text of the paper calls it a symmetric quantization, while its Figure 2(a), redrawn below, also shifts each block by a zero point.
2. **Hierarchical Weight Partition for ZeRO (hpZ):** a data remapping that trades memory for communication: each machine keeps a full copy of the weights, partitioned across its GPUs, so that the all-gather of the weights in the backward pass stays inside the machine, instead of crossing machines.
3. **Quantized Gradient Communication for ZeRO (qgZ):** the reduce-scatter of the gradients is replaced by a block quantization of the gradients to INT4, as in (1), and a hierarchical all-to-all, first inside each machine and then across machines. The gradients are de-quantized to full precision before the reduction, which preserves the training accuracy. A ring-based reduce-scatter that quantizes and de-quantizes at every step was also considered, but its errors accumulate along the ring, and its quantization kernels run one after another, which adds latency.

Together, the three techniques reduce the cross-node communication from $$3M$$ to $$0.75M$$: $$0.5M$$ for the INT8 weights of the forward pass, none for the weights of the backward pass, and $$0.25M$$ for the INT4 gradients. This is the 4× communication reduction of the results section, which leads to up to 2.16× higher compute throughput on 384 GPUs. The following picture compares the basic and the block quantization of qwZ (item 1) on a small example:

{: .picture}
<img loading="lazy" width="65.7%" height="65.7%" src="{{ site.assets }}/publications/zeroplusplus.svg"/>

{: .caption}
Basic quantization (top) and block quantization (bottom) of the same 3×3 weights in qwZ: a scale and a zero point per block (here, per row) keep the dequantized values (right) closer to the input (left); the digits that changed are orange. Adapted from: [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209)

Block quantization also stores the 4-bit weights of [QLoRA](#qlora), with one scale per block of 64 weights, there to save memory rather than communication.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="qlora">
<summary markdown="span"><span class="pub-title">QLoRA: Efficient Finetuning of Quantized LLMs</span><span class="pub-meta">, University of Washington</span><a class="pub-link" href="https://arxiv.org/abs/2305.14314" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Regular 16-bit [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) of a 65B-parameter model takes more than 780 GB of GPU memory. QLoRA is, in the words of the paper, "an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLORA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA)" (see [the LoRA entry](#lora)).

QLoRA "introduces multiple innovations designed to reduce memory use without sacrificing performance: (1) 4-bit NormalFloat, an information theoretically optimal quantization data type for
normally distributed data that yields better empirical results than 4-bit Integers and 4-bit Floats.
(2) Double Quantization, a method that quantizes the quantization constants, saving an average
of about 0.37 bits per parameter (approximately 3 GB for a 65B model). (3) Paged Optimizers,
using NVIDIA unified memory to avoid the gradient checkpointing memory spikes that occur when
processing a mini-batch with a long sequence length."

**4-bit NormalFloat** (NF4) rounds each value to the nearest of 16 levels placed at the [quantiles](https://en.wikipedia.org/wiki/Quantile) of a standard normal distribution. General quantile quantization needs an expensive quantile estimation (e.g. "SRAM quantiles", which also gives large errors for outliers), but pretrained weights are roughly zero-centered normal, so that the quantiles are precomputed once, and each block of 64 weights is rescaled by its absmax. Double quantization then quantizes these absmax constants to 8 bits, with one 32-bit constant per block of 256 of them, so that their cost falls from $$32/64 = 0.5$$ to $$8/64 + 32/(64 \cdot 256) \approx 0.127$$ bits per parameter, the saving of about 0.37 bits quoted above.

The authors "use QLORA to finetune more than 1,000 models, [and] results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA". Their best model family, Guanaco, reaches 99.3% of the performance of ChatGPT on the Vicuna benchmark after 24 hours of finetuning on a single GPU. [FP6-LLM](#fp6-llm) also keeps low-bit weights in memory and de-quantizes them for each matmul, to serve a model faster rather than to fine-tune it.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="llama-2">
<summary markdown="span"><span class="pub-title">Llama 2: Open Foundation and Fine-Tuned Chat Models</span><span class="pub-meta">, Meta</span><a class="pub-link" href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Public pretrained LLMs such as [LLaMA](#llama) "match the performance of closed pretrained competitors like GPT-3", but closed product LLMs such as ChatGPT are "heavily fine-tuned to align with human preferences", a step that costs compute and human annotation and is "often not transparent or easily reproducible". [Llama 2](https://en.wikipedia.org/wiki/Llama_(language_model)) is a collection of pretrained and fine-tuned LLMs from 7B to 70B parameters, pretrained on 2T tokens, and Llama 2-Chat is fine-tuned for dialogue. The models outperform open-source chat models on most benchmarks, and, based on human evaluations for helpfulness and for safety (Figure 3 of the paper), the authors argue that they may be a suitable substitute for closed-source models.

The training dataset is made only of publicly available sources, which do not include data from Meta’s products or services, and the authors made an effort to remove the data of sites known to hold a lot of personal information. Table 2 presents the GPU compute hours, power consumption and carbon emissions of each model.

The pretraining setting and model architecture are adopted from [Llama 1](#llama): [bytepair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) (BPE), pre-normalization via RMSNorm, SwiGLU activations, rotary positional embeddings, the AdamW optimizer and a cosine learning rate schedule. The primary architectural differences are an **increased context length** (4K tokens instead of 2K) and **grouped-query attention (GQA)** in the 34B and 70B models, where each key-value head is shared by a group of query heads, between multi-head and [multi-query attention](#multi-query-attention). [Llama 3](#llama-3) keeps this architecture, with a few changes.

The fine-tuning was performed with supervised fine-tuning (Section 3.1), initial and iterative reward modeling on more than 1.4M binary comparisons of human preferences collected by Meta (Section 3.2.2), and [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) (Section 3.2.3). As a drawback of RLHF, "initial RLHF models tended to forget the initial instruction after a few turns of dialogue, as illustrated in Figure 9 (left). To address these limitations, we propose **Ghost Attention (GAtt)**, a very simple method inspired by Context Distillation (Bai et al., 2022b) that hacks the fine-tuning data to help the attention focus in a multi-stage process".

In GAtt, a system instruction (e.g., "act as ...") is synthetically concatenated to all user messages of a multi-turn dialogue, and responses are sampled with the latest RLHF model. For fine-tuning, the instruction is kept only in the first turn, and the loss of the tokens of the previous turns is set to zero, so the model learns to keep following the instruction across turns. The following picture, from Figure 9 of the paper, shows a dialogue without GAtt (left) and with GAtt (right):

{: .picture}
<img loading="lazy" width="79.4%" height="79.4%" src="{{ site.assets }}/publications/llama2_gatt.svg"/>

{: .caption}
<b>Issues with multi-turn memory (<i>left</i>) can be improved with GAtt (<i>right</i>).</b> Both dialogues start with the instruction “Always answer with emojis”. Adapted from: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="llama">
<summary markdown="span"><span class="pub-title">LLaMA: Open and Efficient Foundation Language Models</span><span class="pub-meta">, Meta</span><a class="pub-link" href="https://arxiv.org/abs/2302.13971" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The compute-optimal recipe of [Chinchilla](#chinchilla) gives the best model size for a training budget, but disregards the inference budget: it recommends training a 10B model on 200B tokens, while the performance of a 7B model continues to improve even after 1T tokens. [LLaMA](https://en.wikipedia.org/wiki/Llama_(language_model)) trains smaller models on more tokens instead, optimizing for the inference cost. It is a collection of Large Language Models (LLM) with 7B to 65B parameters, trained on publicly available datasets only: LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, despite being 10× smaller, and LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B.

The datasets of the pre-training are listed in Table 1 of the paper, and the training hyperparameters in Table 2; the [bytepair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) (BPE) tokenizer gives about 1.4T tokens.

The architecture of the models makes several improvements over the original Transformer:
- **Pre-normalization [GPT3]:** [RMSNorm](#rmsnorm) at the input of each Transformer sub-layer, instead of at its output, improves the training stability.
- **SwiGLU activation function [PaLM]:** SwiGLU replaces ReLU, with a dimension of $$\frac{2}{3} 4d$$ instead of $$4d$$ as in PaLM (SwiGLU has three weight matrices instead of two, so this keeps the parameter count of the feed-forward layer).
- **Rotary Embeddings [GPTNeo]:** rotary positional embeddings ([RoPE](#roformer)) at each layer replace the absolute positional embeddings.
- **Optimization** performed with the AdamW optimizer, with $$β_1 = 0.9$$ and $$β_2 = 0.95$$.
- **Cosine learning rate schedule** with a warmup of $$2000$$ steps, a weight decay of $$0.1$$, a gradient clipping of $$1.0$$ and a final learning rate of 10% of the maximum learning rate.
- **Efficient causal multi-head attention**, which neither stores the attention weights nor computes the key/query scores that the causal mask removes.
- **Activation checkpointing** saves the activations that are expensive to compute, such as the outputs of the linear layers, with a manual implementation of the backward pass of the Transformer instead of PyTorch autograd, and model and sequence parallelism (following Korthikanti et al., 2022) to reduce the memory.
- **Overlap of the computation of activations and the communication between GPUs** over the network, to reduce latency.

With these, the 65B model processes about 380 tokens per second per GPU on 2048 A100 GPUs of 80 GB, i.e. about 21 days of training for the 1.4T-token dataset. [Llama 2](#llama-2) and [Llama 3](#llama-3) keep this design, with longer contexts and more data.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="retentive-network">
<summary markdown="span"><span class="pub-title">Retentive Network: A Successor to Transformer for Large Language Models</span><span class="pub-meta">, Microsoft and Tsinghua University</span><a class="pub-link" href="https://arxiv.org/abs/2307.08621" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Transformers train in parallel, but their decoding cost grows with the sequence, while [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) decode at a constant cost but train sequentially: no architecture had training parallelism, low-cost inference and good performance at once, the *impossible triangle* of the paper. RetNet replaces the multi-head attention of Transformers with a **multi-scale retention** mechanism, which has a parallel form for training and a recurrent form for inference, at a cost of $$O(1)$$ per token. The recurrent form improves the decoding throughput 8.4× over the Transformer and reduces the latency 15.6× and the GPU memory 3.4×, without sacrificing performance (Figure 1 of the paper, for a 7B model and an input of 8K tokens).

RetNet is a stack of $$L$$ identical blocks, with a layout similar to that of the Transformer (residual connections and pre-LayerNorm). Each RetNet block contains two modules: a multi-scale retention (MSR) module and a feed-forward network (FFN) module. The MSR module encodes the tokens of a sequence in an auto-regressive manner.

The input $$X_0$$ is the sequence of embeddings, of shape sequence length by hidden size, and each layer $$l$$ computes the contextualized representations $$X_l$$. A retention head can be computed in **three equivalent forms**:
1. the **parallel representation**, $$\mathrm{Retention}(X) = (Q K^\intercal \odot D) V$$, similar to the transformer but without the softmax, and with an extra causal-decay matrix $$D$$, where $$D_{nm} = γ^{n-m}$$ for $$n ≥ m$$ and 0 otherwise (Eq. 5 in the paper), which suits parallel training;
2. the **recurrent representation**, a recurrent neural network that suits inference: $$S_n = γ S_{n-1} + K_n^\intercal V_n$$ and $$\mathrm{Retention}(X_n) = Q_n S_n$$, where the state $$S_n$$ depends on the previous state $$S_{n-1}$$;
3. the **chunkwise recurrent representation**, for long sequences, with a linear complexity: each chunk is encoded in parallel with the parallel representation, for speed, and the information passes across chunks through the recurrent representation, to save GPU memory.

Each layer has $$h = d_{model}/d$$ retention heads, where $$d$$ is the head dimension, and each head has its own matrices $$W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d}$$ and its own decay, $$γ = 1 - 2^{-5-i}$$ for head $$i$$ from 0 to $$h-1$$, so that the heads decay at different rates: the multi-scale of the name. A layer $$l$$ then computes $$Y_l = \mathrm{MSR}(\mathrm{LN}(X_l)) + X_l$$ and $$X_{l+1} = \mathrm{FFN}(\mathrm{LN}(Y_l)) + Y_l$$, where LN is the LayerNorm, i.e. a regular Transformer layer with the attention replaced by retention. The following picture illustrates the parallel form of a retention head, used for training, and its recurrent form, used for inference:

{: .picture}
<img loading="lazy" width="63%" height="63%" src="{{ site.assets }}/publications/retnet.svg"/>

{: .caption}
Dual form of RetNet: <b>(a)</b> parallel representation and <b>(b)</b> recurrent representation. “GN” is short for GroupNorm. Adapted from: [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)

A simpler summary of RetNet can be found in [this video](https://www.youtube.com/watch?v=JaIL1VAEwZ8). [Mamba](#mamba) and [Kimi Linear](#kimi-linear) also keep a constant-size recurrent state, with a decay that depends on the input instead of a fixed one per head.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="operator-fusion-in-xla">
<summary markdown="span"><span class="pub-title">Operator Fusion in XLA: Analysis and Evaluation</span><span class="pub-meta">, UToronto</span><a class="pub-link" href="https://arxiv.org/abs/2301.13062" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Each kernel of a model writes its output to the GPU memory, and the next kernel reads it back. **Kernel fusion** merges such kernels into one, so that the intermediate values stay on chip, and it is one of the most important optimizations performed by [XLA](https://openxla.org/xla). This paper (a course-project report) details XLA and the key compiler passes of its source code: instruction fusion, fusion merger, multi-output (sibling) fusion and horizontal fusion. XLA does not fuse the library calls to cuBLAS and cuDNN, nor operations that it marks as expensive, such as convolutions, sorts and all-reduces, and it does not fuse a concatenation that has several users, which would duplicate code.

The paper also presents the speedup that kernel fusion can deliver, and what low-level effects it has on hardware. With a [JAX](https://en.wikipedia.org/wiki/JAX_(software)) CartPole reinforcement-learning environment (an [inverted pendulum](https://en.wikipedia.org/wiki/Inverted_pendulum) on a cart) as a case study, the authors show how XLA makes fusion decisions in practice, and implement fusion strategies that reach up to 10.56× speedup over their baseline implementation. A handwritten CUDA implementation is still 2.7× faster than their best XLA version, and the paper concludes that "Automatic kernel fusion is far from a solved problem." [Chimera](#chimera) fuses the compute-intensive operators that XLA leaves to libraries.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="longnet">
<summary markdown="span"><span class="pub-title">LongNet: Scaling Transformers to 1,000,000,000 Tokens</span><span class="pub-meta">, Microsoft and Xi’an Jiaotong University</span><a class="pub-link" href="https://arxiv.org/abs/2307.02486" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The [self-attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) of a regular Transformer costs $$O(N^2 d)$$ for a sequence of $$N$$ tokens of dimension $$d$$, so a longer sequence requires a trade-off between computational complexity and model expressivity. LongNet is a Transformer variant that scales the sequence length to more than 1B tokens, without sacrificing the performance on shorter sequences.

The main trick is the **dilated attention**, which splits the sequence into segments of length $$w$$ and sparsifies each segment with a dilation rate $$r$$, keeping one row in $$r$$. It mixes several pairs $$(w, r)$$ that grow geometrically, so that the attention allocation decreases exponentially as the distance between tokens grows. This gives a linear computational complexity, $$O(N d)$$, and a logarithmic dependency path between any two tokens.

The sequence dimension can also be distributed across devices, to train on extremely long sequences: each device computes the dilated attention of its part of the sequence, and the devices all-gather the sparsified keys and values of the segments that span several of them. As these sparsified segments have a fixed size, the communication does not grow with the sequence length, and the runtime stays almost constant as the sequence length scales up to 1B tokens. In language modeling, trained on sequences of 32K tokens, LongNet reaches a perplexity of 3.01 at 32K, against 3.64 for the [Sparse Transformer](#sparse-transformers).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="flashattention-2">
<summary markdown="span"><span class="pub-title">FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning</span><a class="pub-link" href="https://arxiv.org/abs/2307.08691" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

"FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different [thread blocks](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)) and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. We propose FlashAttention-2, with better work partitioning to address these issues."

FlashAttention-2 makes three changes. First, it swaps the two loops of [FlashAttention](#flashattention): the outer loop goes over the blocks of $$Q$$ and the inner loop over the blocks of $$K$$ and $$V$$, so that each block of the output stays on chip until it is complete, and is written to HBM once. The blocks of $$Q$$ are independent and run on thread blocks that do not communicate, so the work is parallelized over the sequence length, and not only over the batch and the heads as in FlashAttention. This keeps the GPU busy for long sequences, which usually come with small batches or few heads.

Then, inside a thread block, the warps split $$Q$$ instead of $$K$$ and $$V$$ (the *split-K* scheme of FlashAttention), so they no longer write partial results to shared memory to add them up. Finally, the output is divided by the sum of the exponentials only once, at the end of the inner loop, instead of at every step, which removes non-matmul FLOPs: an A100 runs 312 TFLOP/s of FP16 matmul but only 19.5 TFLOP/s of FP32 non-matmul operations, so a non-matmul FLOP costs 16 times more than a matmul FLOP.

These changes "yield around 2× speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations". When used end-to-end to train GPT-style models, "FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization)". [DistFlashAttn](#distflashattn) distributes this kernel across GPUs, for long sequences.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="flow-matching">
<summary markdown="span"><span class="pub-title">Flow Matching for Generative Modeling</span><span class="pub-meta">, Meta AI & Weizmann Institute (ICLR 2023)</span><a class="pub-link" href="https://arxiv.org/abs/2210.02747" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[Continuous Normalizing Flows](https://en.wikipedia.org/wiki/Flow-based_generative_model) (CNFs) transform noise into data by integrating a learned vector field, an ODE, and their maximum-likelihood training simulates this ODE, which is expensive. Flow Matching is a simulation-free approach for training CNFs, compatible with a general family of Gaussian probability paths for transforming between noise and data samples, such as the reverse process of [diffusion models](https://en.wikipedia.org/wiki/Diffusion_model). "Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization".

The key idea is to regress a neural vector field $$v_t(x)$$ onto the target vector field that generates the probability path from noise to data. That target is intractable, but regressing onto the *conditional* field $$u_t(x \vert x_1)$$, which generates the path to a single data sample $$x_1$$, gives the same gradients. The **Conditional Flow Matching** (CFM) loss is defined as:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, q(x_1),\, p_t(x \vert x_1)} \left\lVert v_t(x) - u_t(x \vert x_1) \right\rVert^2$$

where $$\theta$$ are the parameters of $$v_t$$, the time $$t$$ is uniform in $$[0, 1]$$, $$q(x_1)$$ is the data distribution and $$p_t(x \vert x_1)$$ is the conditional probability path. The OT path moves each noise sample $$x_0$$ along a straight line to $$x_1$$, as $$\psi_t(x_0) = (1 - (1 - \sigma_{\min}) t)\, x_0 + t\, x_1$$, at a constant velocity, so that the loss becomes (Eq. 23 in the paper):

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, q(x_1),\, p(x_0)} \left\lVert v_t(\psi_t(x_0)) - \left(x_1 - (1 - \sigma_{\min})\, x_0\right) \right\rVert^2$$

where $$p(x_0)$$ is a standard normal distribution and $$\sigma_{\min}$$ is the small standard deviation of the path at $$t = 1$$.

On ImageNet 64×64, Flow Matching with OT paths reaches a negative log-likelihood of 3.31 bits per dimension and an FID of 14.45, with 138 function evaluations of the ODE solver, against 3.32, 17.36 and 264 for DDPM (Table 1 of the paper). [Flow Straight and Fast](#flow-straight-and-fast) (rectified flow) proposed straight paths between noise and data at about the same time.

A good explanation can be found in this [Cambridge ML group post](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html).

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2022</span></td>
<td markdown="block">
<details id="flow-straight-and-fast">
<summary markdown="span"><span class="pub-title">Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow</span><span class="pub-meta">, UT Austin</span><a class="pub-link" href="https://arxiv.org/abs/2209.03003" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Rectified flows aim at reducing the number of steps when transitioning between two distributions. This is important for e.g. [diffusion models](https://en.wikipedia.org/wiki/Diffusion_model), where inference is performed in $$T$$ sampling steps, each a full pass through the network, and the goal is to do it in fewer steps.

The rectified flow is an [ordinary differential equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation) (ODE) model that transports distribution $$π_0$$ to $$π_1$$ by following straight line paths as much as possible. It is learned with a simple least-squares regression of the velocity field onto the direction $$X_1 − X_0$$ between paired samples.
The straight paths are preferred both theoretically because they are the shortest paths between two end points, and computationally because they can be exactly simulated without time discretization. Recursively re-training on the model's own couplings (**reflow**) straightens the paths further, so that a single [Euler step](https://en.wikipedia.org/wiki/Euler_method) (optionally after distillation) already gives good samples; the same formulation later became the training objective of [Stable Diffusion 3](https://arxiv.org/abs/2403.03206).

**Rectified flow.** Given empirical observations of $$X_0 \sim \pi_0, X_1 \sim \pi_1$$, the rectified flow induced from $$(X_0, X_1)$$ is an ODE on time $$t \in [0, 1]$$,

$$
\mathrm{d}Z_t = v(Z_t, t)\mathrm{d}t,
$$

which converts $$Z_0$$ from $$\pi_0$$ to a $$Z_1$$ following $$\pi_1$$. The drift force $$v \colon \mathbb{R}^d \to \mathbb{R}^d$$ is set to drive the flow to follow the direction $$(X_1 - X_0)$$ of the linear path pointing from $$X_0$$ to $$X_1$$ as much as possible, by solving a simple least squares regression problem:

$$
\min_v \int_0^1 \mathbb{E}\left[\left\|(X_1 - X_0) - v\big(X_t,\, t\big)\right\|^2\right] \mathrm{d}t, \qquad \text{with} \quad X_t = tX_1 + (1 - t)X_0, \tag{1}
$$

where $$X_t$$ is the linear interpolation of $$X_0$$ and $$X_1$$. Naively, $$X_t$$ follows the ODE of $$\mathrm{d}X_t = (X_1 - X_0)\mathrm{d}t$$, which is non-causal (or anticipating) as the update of $$X_t$$ requires the information of the final point $$X_1$$. By fitting the drift $$v$$ with $$X_1 - X_0$$, the rectified flow *causalizes* the paths of linear interpolation $$X_t$$, yielding an ODE flow that can be simulated without seeing the future.

In practice, $$v$$ is parameterized with a neural network or other nonlinear models, and (1) is solved with any off-the-shelf stochastic optimizer, such as stochastic gradient descent, with empirical draws of $$(X_0, X_1)$$.

 
$$
\begin{array}{l}
\hline
\textbf{Algorithm 1} \text{ Rectified Flow: Main Algorithm} \\
\hline
\quad \textbf{Procedure}\text{: } \boldsymbol{Z} = \texttt{RectFlow}((X_0, X_1))\text{:} \\
\qquad \textit{Inputs}\text{: Draws from a coupling } (X_0, X_1) \text{ of } \pi_0 \text{ and } \pi_1 \text{;} \\
\qquad \quad \text{velocity model } v_\theta \colon \mathbb{R}^d \to \mathbb{R}^d \text{ with parameter } \theta. \\
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

First, for a given input coupling $$(X_0, X_1)$$, it is easy to see that the exact minimum of (1) is achieved if

$$
v^X(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x], \tag{2}
$$

which is the expectation of the line directions $$X_1 - X_0$$ that pass through $$x$$ at time $$t$$. Since $$v^X$$ gives a single, averaged direction where the lines cross, the paths of the rectified flow do not cross. The paper then shows that the flow keeps the marginal distribution of $$X_t$$ at every time $$t$$, does not increase any convex transport cost, and gets straighter with reflow (Theorems 3.3, 3.5 and 3.7).

On CIFAR-10, the 1-rectified flow reaches an [FID](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) (lower is better) of 2.58 with an adaptive ODE solver of 127 steps. With a single Euler step, its FID is 378, but 12.21 after one reflow, and 4.85 after one reflow and a distillation. [Flow Matching](#flow-matching), a concurrent work, also regresses a velocity field along straight paths between noise and data, without the reflow.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="tensor-programs-v">
<summary markdown="span"><span class="pub-title">Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($$\mu$$Transfer)</span><span class="pub-meta">, Microsoft</span><a class="pub-link" href="https://arxiv.org/abs/2203.03466" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Tuning the [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) (HPs) of a neural network, such as its learning rate, takes many training runs, "prohibitively so for neural networks (NNs) with billions of parameters", and in the standard parametrization the optimal values found on a smaller model shift as the model grows wider.

Under the [Maximal Update Parametrization (µP)](https://arxiv.org/abs/2011.14522), which scales the initialization and the learning rate of each layer with its width, many optimal hyperparameters remain stable as the model width changes, as the µP paper explains: "When (layer) width is large, every activation vector has roughly iid coordinates, at any time during training. Using Tensor Programs, we can recursively calculate such coordinate distributions, and consequently understand how the neural network function evolves".

With that in mind, the authors propose a hyperparameter tuning paradigm called **µTransfer**: "parametrize the target model in µP, tune the HP indirectly on a smaller model, and zero-shot transfer them to the full-sized model". By transferring from a 40M-parameter proxy, they outperform the published numbers of the 6.7B [GPT-3](#gpt-3) model, with a tuning cost of only 7% of the total pretraining cost; transferring from a 13M-parameter model beats the published numbers of BERT-large (350M). The following picture shows how the optimal learning rate depends on the width:

{: .picture}
<img loading="lazy" width="88.5%" height="88.5%" src="{{ site.assets }}/publications/mutransfer.svg"/>

{: .caption}
Training loss against learning rate on Transformers of varying $$d_{model}$$ trained with Adam, in the standard parametrization (left) and in µP (right). In the standard parametrization, unlike in µP, different widths do not share the same optimal hyperparameter; wider networks do not always perform better than narrower ones; in fact they underperform the same-width networks in µP even after tuning learning rate (see dashed line). Adapted from: [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)

Table 1 of the paper lists the hyperparameters that can be µTransferred, those that cannot, and the dimensions that µTransfer works across, with a few caveats discussed in Section 6.1 (\* marks the dimensions validated only empirically, on Transformers; the others also have a theoretical justification):
- µTransferable: the optimization hyperparameters (learning rate, momentum, Adam beta, learning rate schedule), the initialization (per-layer init variance) and the parameter multipliers (multiplicative constants after weights and biases).
- Not µTransferable: the regularization (dropout, weight decay).
- µTransferred across: width, depth\*, batch size\*, training time\*, sequence length\*.

µTransfer complements [the scaling laws](#scaling-laws) and [Chinchilla](#chinchilla), which give the model size and the number of training tokens for a compute budget, with the hyperparameters to train that model.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepspeed-moe">
<summary markdown="span"><span class="pub-title">DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale</span><span class="pub-meta">, Microsoft (ICML 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2201.05596" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) models reach the quality of a dense model at a lower training cost, but their many parameters make their inference slow and expensive. DeepSpeed-MoE is an influential MoE system that attacks the MoE-specific bottlenecks (**token routing**, [**all-to-all communication**](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)), model size and inference cost) from three sides: model design, compression and the inference system. The model design is the **Pyramid-Residual MoE** (PR-MoE), with more experts in the later layers and, in each MoE layer, a fixed dense MLP plus one expert selected by the router, and the compression is **Mixture-of-Students** (MoS), a [distillation](https://en.wikipedia.org/wiki/Knowledge_distillation) into a student with fewer layers. The inference system combines expert, tensor and data parallelism with hierarchical all-to-all communication and fused sparse kernels.

The hierarchical all-to-all first exchanges the data among the $$G$$ GPUs of each node, then among the nodes, which reduces the communication hops of an all-to-all over $$p$$ GPUs from $$O(p)$$ to $$O(G + p/G)$$: from 256 to 40 hops for 256 GPUs in nodes of 8.

MoE gives a ~5× training-cost saving over a quality-equivalent dense model for autoregressive language models (a 1.3B-parameter base with 128 experts matches a 6.7B dense model), and PR-MoE plus Mixture-of-Students reduce the MoE model size by up to 3.7×. The inference system gives 7.3× better latency and cost than existing MoE inference solutions, up to 4.5× faster and 9× cheaper inference than quality-equivalent dense models, and serves trillion-parameter MoE models in under 25 ms. It helped popularize MoE as a practical path to scaling model capacity without proportional compute.

DeepSpeed-MoE keeps the collectives "standard" (NCCL-style). Later production-scale systems, such as [MegaScale-MoE](#megascale-moe) and [MegaScale-Infer](#megascale-infer), introduce more specialized parallelism strategies and communication libraries to handle extreme scale, and [Comet](#comet) overlaps the all-to-all with the expert computation at a fine granularity.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepspeed-inference">
<summary markdown="span"><span class="pub-title">DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale</span><span class="pub-meta">, Microsoft DeepSpeed (SC 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2207.00032" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Because inference typically runs at small batch sizes, inference kernels must achieve high memory bandwidth utilization and high compute utilization at small batch sizes, whereas training kernels only need to achieve high compute utilization at much larger batch sizes. This makes developing inference kernels quite challenging. DeepSpeed Inference consists of two components:
1. **DeepSpeed Transformer**, a GPU-only system with three parts: single-GPU transformer kernels with memory-bandwidth-centric fusion schedules and GeMM kernels (Sec. III, described below); a many-GPU dense transformer inference system (Sec. IV); and a massive-GPU sparse (MoE) inference system (Sec. V). The dense system partitions the transformer operators across the GPUs of a node (tensor slicing), with NCCL [all-reduce](https://en.wikipedia.org/wiki/Collective_operation) collectives between them, and uses pipeline parallelism across nodes, as "tensor slicing can not be scaled efficiently beyond a single node due to significant communication overhead". The sparse system combines expert, data and tensor parallelism: as the data is replicated across the tensor-parallel ranks, the [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) of the experts runs only among the GPUs that share the same tensor-slicing rank, and kernel fusion removes the latency overhead of the sparse operators of the gating function.
2. **ZeRO-Inference** (Sec. VI) uses CPU, [NVMe](https://en.wikipedia.org/wiki/NVM_Express) and GPU memory along with GPU compute, to run massive models with limited resources. Pinning as many weights as possible in GPU memory leaves room only for small batch sizes, so ZeRO-Inference instead pins the model weights in DRAM (if large enough) or NVMe, and streams each layer into GPU memory when it is needed.

The single-GPU transformer kernels introduce two techniques:
1. **Deep-Fusion** fuses kernels beyond element-wise operations, to reduce the kernel-invocation and data-movement overheads. On a GPU, data produced by one [thread block](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)) and consumed by another needs a global memory synchronization, i.e. a new kernel. Deep-Fusion therefore tiles the computation along the dimensions of the iteration space that have no cross-tile data dependencies, processes each tile in one thread block, and fuses two operators when each tile of the second depends on exactly one output tile of the first. This way, it fuses not only element-wise operations, but also reductions, data transpositions and GeMMs.
2. A **custom GeMM** for small batch sizes, fusable with Deep-Fusion, that reaches the maximum memory bandwidth: "We first tile the computation along the output dimension. That allows us to implement GeMM using a single kernel by keeping the reduction within a tile". The partial results of the warps of a thread block are transposed in shared memory, so that a single warp reduces them in registers, and the weight matrix is transposed at initialization, so that each thread reads $$M$$ contiguous elements along the input dimension: $$M$$ is 2 in FP16 and 4 in INT8, i.e. 4 bytes per thread and one 128-byte cache line per warp of 32 threads.

The operations of a transformer layer are fused into four regions: 1) the QKV GeMM and the input layer-norm, 2) the transposition and the attention, 3) the post-attention layer-norm and the intermediate GeMM, and 4) the bias and residual addition.

DeepSpeed Inference reduces the latency by up to 7.3× over the state of the art for latency-oriented scenarios (with MoE models), and increases the throughput by over 1.5× for throughput-oriented ones. It enables real-time trillion-parameter inference on hundreds of GPUs and, with ZeRO-Inference, inference on 25× larger models than GPU-only solutions (530B against 20B parameters), at 84 TFLOPS, over 50% of the peak of an A6000 GPU.

In the context of fusion-heavy related work, DeepSpeed Inference aggressively fuses *compute* (Deep-Fusion, custom GeMMs), but keeps the **communication as NCCL collectives outside** of those kernels. This makes it a useful baseline for what strong kernel engineering and runtime scheduling achieve *without* changing the underlying communication model, as [FLUX](#flux), [TileLink](#tilelink) and [Triton-distributed](#triton-distributed) do by fusing the communication into the compute kernels.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="dylora">
<summary markdown="span"><span class="pub-title">DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation</span><span class="pub-meta"> (EACL 2023)</span><a class="pub-link" href="https://arxiv.org/abs/2210.07558" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[LoRA](#lora) blocks "suffer from two major problems: first, the size of these blocks is fixed and cannot be modified after training (for example, if we need to change the rank of LoRA blocks, then we need to re-train them from scratch); second, optimizing their rank requires an exhaustive search and effort". **Dynamic LoRA** (DyLoRA) addresses these two problems: "DyLoRA method trains LoRA blocks for a range of ranks instead of a single rank by sorting the representation learned by the adapter module at different ranks during training". It works as follows:
- Each LoRA module has an up-projection matrix $$W_{up} \in \mathbb{R}^{m \times r}$$ and a down-projection matrix $$W_{dw} \in \mathbb{R}^{r \times d}$$, and is trained to operate in a range of ranks $$r \in [r_{min}, r_{max}]$$, where $$r_{min}$$ and $$r_{max}$$ are hyperparameters.
- At each training step, a rank $$b$$ between $$r_{min}$$ and $$r_{max}$$ is sampled from a distribution $$P_B$$ (uniform in most experiments), and $$W_{dw}$$ and $$W_{up}$$ are truncated to their first $$b$$ rows and columns, $$W_{dw \downarrow b}$$ and $$W_{up \downarrow b}$$. The forward pass of this step uses them, together with the frozen pretrained weights $$W_0$$ ($$W$$ in the picture below), as $$h = W_0 x + \frac{\alpha}{b} W_{up \downarrow b} W_{dw \downarrow b} \, x$$, where $$\alpha$$ is a scaling constant.

DyLoRA trains dynamic, search-free models at least 4 to 7 times faster than LoRA (depending on the task), without significantly compromising performance: on GLUE with RoBERTa-base, one DyLoRA model averages 84.22 at rank 1 and 85.44 at rank 8 (82.86 and 84.35 in the frozen mode of the picture), against 85.27 for a LoRA model of rank 8, which falls to 54.90 when truncated to rank 1. The following picture illustrates the forward pass and the parameter updates of DyLoRA for two sampled ranks:

{: .picture}
<img loading="lazy" width="92.4%" height="92.4%" src="{{ site.assets }}/publications/dylora.svg"/>

{: .caption}
The overall diagram of DyLoRA. In each iteration, a rank $$b$$ is sampled from a pre-defined random distribution $$P_B$$, which truncates the up-projection and down-projection matrices in the LoRA (Hu et al., 2021) objective: the forward pass uses only the first $$b$$ columns of $$W_{up}$$ and the first $$b$$ rows of $$W_{dw}$$, and the parameter update changes only the $$b$$-th column and row, while the others stay frozen (snowflakes). Adapted from: [DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low Rank Adaptation](https://arxiv.org/abs/2210.07558)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="efficiently-scaling-transformer-inference">
<summary markdown="span"><span class="pub-title">Efficiently Scaling Transformer Inference</span><span class="pub-meta">, Google (MLSys 2023)</span><a class="pub-link" href="https://arxiv.org/abs/2211.05102" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper focuses on efficient generative inference of Transformer models in "one of its most challenging settings: large deep models, with tight latency targets and long sequence lengths". It develops an **analytical model of the inference cost**, to select the best multi-dimensional partitioning for [TPU v4](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) chips and their [3D-torus](https://en.wikipedia.org/wiki/Torus_interconnect) network, depending on the latency and throughput targets of the application. Its methods surpass the efficiency of [NVIDIA's FasterTransformer](https://github.com/NVIDIA/FasterTransformer).

Section 3.1 analyzes the [collective communication](https://en.wikipedia.org/wiki/Collective_operation), and Section 3.2 the partitioning of the feed-forward layer, in the following notation: $$BLE_{xyz}$$ means that the last dimension $$E$$ of a tensor of logical shape $$BLE$$ (batch, sequence length and model size) is split into $$X \times Y \times Z$$ parts, i.e. the per-chip tensor is of shape $$[B, L, E/(X \times Y \times Z)]$$ (omitted axes are replicated), and $$F$$ is the hidden (intermediate) size of the feed-forward layer. The paper compares three layouts:
- **1D weight-stationary** (Section 3.2.1), *a la Megatron*: the weights are partitioned as $$EF_{xyz}$$ and $$F_{xyz}E$$, i.e. along $$F$$ only, into $$X \times Y \times Z = n_{chips}$$ parts.
- **2D weight-stationary** (Section 3.2.2): the weights are split along both $$E$$ and $$F$$, in roughly square shards. The compute cost is the same, but the communication time falls as $$1/\sqrt{n_{chips}}$$ instead of staying constant, which makes it lower than in the 1D layout beyond 16 chips.
- **Weight-gathered** (Section 3.2.3): in the weight-stationary layouts, each chip keeps its shard of the weights, and "the output of each per-chip matrix multiplication must then be aggregated between chips". However, "for very large batch sizes, it is best to keep the activations fully stationary between sequential matrix multiplications, requiring that we fully transfer the weights between all chips".

For the attention (Section 3.3), "Multihead attention can be parallelized in essentially the same ways as a feedforward layer", but its keys and values (the **KV cache**) incur significant memory capacity and bandwidth costs. [Multi-query attention](https://arxiv.org/abs/1911.02150) reduces the size of the KV cache tensors by a factor of $$n_{heads}$$, and the time spent loading them from memory. However, when it is partitioned over the heads ((b) in the picture below), its single head of keys and values "must be replicated on each chip and the memory cost savings of multiquery attention are lost". The paper instead partitions the $$Q$$, $$K$$ and $$V$$ matrices of multi-query attention over the batch dimension $$B$$, into $$n_{chips}$$ partitions. This reduces the cost of loading the KV cache per chip by a factor of $$n_{chips}$$, at the cost of resharding the input activations, and enables larger batch sizes and up to 32× longer contexts. The following picture compares the three partitioning layouts of the attention, on two devices:

{: .picture}
<img loading="lazy" width="65.1%" height="65.1%" src="{{ site.assets }}/publications/efficiently_scaling_transformer_inference_multiquery.svg"/>

{: .caption}
Multiquery attention has lower memory cost to load the KV cache when sharded over batch. Top: the shapes of Q and K/V in multi-head and multi-query attention. <b>(a)</b> Multi-head attention, sharded over heads: it can be sharded across heads without replication. <b>(b)</b> Multi-query attention, sharded over heads: it requires full replication of the single head for K, increasing memory access cost. <b>(c)</b> Multi-query attention, sharded over batch: only a slice of K is needed for einsum, reducing memory access cost. Adapted from: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

Section 3.4 details the gains of the formulation of [GPT-J](https://en.wikipedia.org/wiki/GPT-J), also applied in [PaLM](https://arxiv.org/abs/2204.02311), which **computes the attention and the feed-forward layer in parallel**. The standard formulation of the transformer block is:

$$
y = x + \text{Attention}(\text{LayerNorm}(x)) + \text{MLP}(\text{LayerNorm}(x + \text{Attention}(\text{LayerNorm}(x))))
$$

whereas the parallel formulation is:

$$
y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x))
$$

The parallel formulation has only one LayerNorm per layer instead of two, which reduces latency at small batch sizes. Also, some matrices can be fused, which results in larger matrix multiplications that run more efficiently on accelerators. Moreover, it removes one of the two all-reduce operations of each layer, since the output projections of the attention and of the feed-forward layer are fused and need a single all-reduce.

Finally, Section 3.5 overlaps the communication with the computation (Looped CollectiveEinsum), and Section 3.6 quantizes the weights to INT8. Combined, these reach a 29 ms per-token generation latency at small batch size and a 76% model FLOPs utilization (MFU) during the large-batch processing of 2,048-token inputs, on the PaLM 540B model.

Partitioning multi-query attention over the batch is one answer to the cost of the KV cache; others are the grouped-query attention of [Llama 2](#llama-2) and the latent attention of [DeepSeek-V2](#deepseek-v2), which [TPLA](#tpla) partitions across devices.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="stability-efficiency-dilemma">
<summary markdown="span"><span class="pub-title">The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models</span><span class="pub-meta">, Microsoft (NeurIPS 2022)</span><a class="pub-link" href="https://openreview.net/forum?id=JpZ5du_Kdh" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

This paper investigates and demonstrates the importance of sequence length in GPT models. The paper presents a set of experiments on a GPT-2 model on public datasets to study the **stability-efficiency dilemma**. The dilemma is that "increasing the batch sizes and learning rates (commonly used for faster training) leads to better training efficiency but can also result in training instability, leading to poor generalization accuracy or failed runs".

The paper finds that **there is a strong correlation between training instability and extreme values of gradient variance**. It also finds that **long sequence lengths contribute to these extreme gradient variance values, especially at the beginning of the training** (and this could be a source of training instability).

The paper presents the **Sequence Length Warmup (SLW)** method, which starts training with short sequences and gradually increases their length (linearly, in multiples of 8, from 8 or 64 tokens to the full length), and aims to solve the training stability-efficiency dilemma by avoiding extreme gradient variance values. This method can be understood as a type of [curriculum learning](https://en.wikipedia.org/wiki/Curriculum_learning) (CL), which presents easier/simpler examples earlier during training and gradually increases the sample difficulties. However, here the work aims to achieve both efficient convergence and better stability by enabling stable training with more aggressive hyperparameters, instead of keeping them constant as in the traditional CL.

Results on GPT-2 (117M and 1.5B) show stable training with 8x larger batch size and 4x larger learning rate, reducing the training tokens and wall-clock time needed for the same zero-shot quality by up to 2.2x and 3.7x. On a GPT-3 (125M) model, it enables 8x larger batch size and 40x larger learning rate, retaining 99% of the zero-shot accuracy on 11 tasks using 10x less data and 17x less time.

Like [DeepNet](#deepnet), SLW targets the training instability of large Transformers, but through the length of the sequences rather than through the architecture.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="flashattention">
<summary markdown="span"><span class="pub-title">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</span><span class="pub-meta">, Stanford (NeurIPS 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. The authors "argue that a missing principle is making attention algorithms IO-aware—accounting for reads and writes between levels of GPU memory". FlashAttention is such an **IO-aware** exact attention algorithm: it uses [tiling](https://en.wikipedia.org/wiki/Loop_nest_optimization) to reduce the number of memory reads and writes between the GPU [high bandwidth memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) (HBM) and the GPU on-chip [SRAM](https://en.wikipedia.org/wiki/Static_random-access_memory).

In practice, it tiles the square attention matrix into blocks that are computed in the on-chip SRAM (shared memory) instead of the GPU's global memory (HBM). The tiling runs as two nested loops: the outer loop loads blocks of $$K$$ and $$V$$ from HBM into SRAM, and the inner loop loads the blocks of $$Q$$, computes their attention with these blocks and writes the updated blocks of the output back to HBM. The backward pass recomputes the $$N \times N$$ attention matrix ($$N$$ being the sequence length) block by block, from the blocks of $$Q$$, $$K$$ and $$V$$ and the softmax statistics (the maximum and the sum of the exponentials of each row) stored by the forward pass, instead of storing it. As a result, FlashAttention needs $$O(N^2 d^2 / M)$$ HBM accesses, for a head dimension $$d$$ and an SRAM of size $$M$$, against $$\Omega(Nd + N^2)$$ for standard attention.

FlashAttention trains Transformers faster than existing baselines (15% end-to-end on BERT-large against the MLPerf 1.1 record, 3× on GPT-2 and 2.4× on the long-range arena), and its attention computation runs 7.6× faster than the PyTorch implementation on GPT-2. Memory becomes linear in sequence length, which enables longer contexts and better models (0.7 better perplexity on GPT-2, 6.4 points on long-document classification), and the first better-than-chance Transformers on Path-X (16K tokens, 61.4% accuracy) and Path-256 (64K tokens, 63.1%, with block-sparse FlashAttention).

FlashAttention is the canonical example of *algorithmic* fusion in attention: it avoids materializing the full $$QK^\top$$ and the large intermediate softmax buffers, by using a numerically stable streaming formulation (the [online softmax](https://arxiv.org/abs/1805.02867)). Instead of computing attention as "GEMM → softmax → GEMM" with large intermediate tensors in HBM, it tiles the computation so that data is reused from SRAM. It is about **IO-aware tiling**, not about overlapping distributed communication, and it shows that fusion can change the *algorithmic memory complexity*, not only the launch overhead, which is why it became a cornerstone kernel. It established the template that later fusion work often follows: fuse stages, tile aggressively, and never write big intermediates to HBM if they can be streamed instead. [Rabe and Staats](#self-attention-memory) also compute attention in blocks, to reduce its memory, and [FlashAttention-2](#flashattention-2) and [FlashAttention-4](#flashattention-4) later improve its work partitioning and adapt it to newer GPUs.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="deepnet">
<summary markdown="span"><span class="pub-title">DeepNet: Scaling Transformers to 1,000 Layers</span><span class="pub-meta">, Microsoft Research</span><a class="pub-link" href="https://arxiv.org/abs/2203.00555" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Very deep Transformers are unstable to train. Previous work had shown that better initialization methods improve the stability of the training of Transformers. The [layer normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization) (LN) is placed in one of two ways: **Post-LN** applies it after the [residual](https://en.wikipedia.org/wiki/Residual_neural_network) addition, as in the original Transformer, while **Pre-LN** applies it to the input of each sub-layer. This paper introduces a normalization function, **DeepNorm**, that modifies the residual connection of a Post-LN Transformer as $$x_{l+1} = LN(\alpha x_l + G_l(x_l, \theta_l))$$, where $$G_l$$ is an attention or feed-forward sub-layer with parameters $$\theta_l$$, and $$\alpha > 1$$ is a constant. It comes with a theoretically derived initialization, in which the weights of the feed-forward layers and the value and output projections of the attention are scaled down by a constant $$\beta$$, so that the model update is bounded by a constant (Figure 2 of the paper gives the code). For an encoder-only or decoder-only model of $$N$$ layers, $$\alpha = (2N)^{1/4}$$ and $$\beta = (8N)^{-1/4}$$, e.g. $$\alpha = 3.76$$ and $$\beta = 0.188$$ for 100 layers. "The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DeepNorm a preferred alternative."

DeepNet trains Transformers with up to 1,000 layers (500 encoder and 500 decoder layers, i.e. 2,500 attention and feed-forward sub-layers). On a multilingual benchmark with 7,482 translation directions, a 200-layer, 3.2B-parameter model outperforms the 48-layer, 12B-parameter state of the art (M2M-100) by 5 [BLEU](https://en.wikipedia.org/wiki/BLEU) points. Later LLMs such as [LLaMA](#llama) use Pre-LN with [RMSNorm](#rmsnorm) instead.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="contrastive-deep-supervision">
<summary markdown="span"><span class="pub-title">Contrastive Deep Supervision</span><span class="pub-meta">, Tsinghua University, Intel Corporation, and Xi’an Jiaotong (ECCV 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2207.05306" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

From the abstract: "the traditional training method only supervises the neural network at its last layer and propagates the supervision layer-by-layer, which leads to hardship in optimizing the intermediate layers. Recently, deep supervision has been proposed to add auxiliary classifiers to the intermediate layers of deep neural networks. By optimizing these auxiliary classifiers with the supervised task loss, the supervision can be applied to the shallow layers directly".

However, "deep supervision conflicts with the well-known observation that the shallow layers learn low-level features instead of task-biased high-level semantic features. To address this issue, this paper proposes a novel training framework named Contrastive Deep Supervision, which supervises the intermediate layers with augmentation-based contrastive learning". The rationale is that contrastive learning can provide better supervision for intermediate layers than the supervised task loss.

Contrastive learning "regards two augmentations from the same image as a positive pair and different images as negative pairs. During training, the neural network is trained to minimize the distance of a positive pair while maximizing the distance of a negative pair. As a result, the network can learn the invariance to various data augmentation, such as Color Jitter and Random Gray Scale".

**Contrastive Deep Supervision** starts from those advancements, and optimizes the intermediate layers with contrastive learning instead of traditional supervised learning: projection heads attached to the intermediate layers are trained with a contrastive loss, while the last layer keeps the cross-entropy loss of the task, and the heads are discarded at inference. The paper evaluates it on nine datasets with eleven models, in image classification, fine-grained classification and object detection: it raises the top-1 accuracy of ResNet-50 on ImageNet from 75.30% to 78.25%. It extends the deep supervision of [Deeply-Supervised Nets](#deeply-supervised-nets). The following picture compares it with traditional supervised learning, contrastive learning and deep supervision:

{: .picture}
<img loading="lazy" width="92.6%" height="92.6%" src="{{ site.assets }}/publications/contrastive_deep_supervision.svg"/>

{: .caption}
The overview of the four methods: (a) traditional supervised learning, (b) contrastive learning, (c) deep supervision and (d) contrastive deep supervision. “→” and “<span style="color:#6c8ebf">→</span>” indicate the path of forward computation and gradients backward computation. “`proj`” and “`fc`” indicate the projection heads and the fully connected classifiers, respectively. The gray dash line indicates whether the feature is task-irrelevant or task-biased. (a) Traditional supervised learning only applies supervision to the last layer and propagates it to the previous layers, leading to gradient vanishing. (c) Deep supervision trains both the last layer and the intermediate layers directly, which addresses gradient vanishing but makes all the layers be biased to the task. (d) Contrastive deep supervision introduces contrastive learning to supervise the intermediate layer and thus avoids these problems. Adapted from: [Contrastive Deep Supervision](https://arxiv.org/abs/2207.05306)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="chinchilla">
<summary markdown="span"><span class="pub-title">Training Compute-Optimal Large Language Models (Chinchilla)</span><span class="pub-meta">, DeepMind (NeurIPS 2022)</span><a class="pub-link" href="https://arxiv.org/abs/2203.15556" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper is heavily related to HPC's performance modeling, applied to large language models. It revisits the question of the [scaling laws](https://en.wikipedia.org/wiki/Neural_scaling_law): "Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens?" The authors present three approaches: (1) fix the model sizes and vary the number of training tokens; (2) vary the model sizes for 9 different FLOP counts; (3) fit a parametric loss function to the final losses of the two previous approaches. The estimates come from over 400 models (70M to 16B parameters, trained on 5B to 500B tokens).

To be compute optimal (i.e. the lowest loss for a given compute budget), [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) claim that models should not be trained to their lowest possible loss: for a 10× increase in computational budget, the model should grow by 5.5× and the training tokens by 1.8×. In this paper, the authors defend that model size and training tokens should be scaled in equal proportions: the optimal model size grows as $$N_{opt} \propto C^a$$ and the optimal number of tokens as $$D_{opt} \propto C^b$$ for a compute budget $$C$$, with $$a$$ and $$b$$ between 0.46 and 0.54 in the three approaches (0.73 and 0.27 for Kaplan et al.), so that a 10× larger budget grows both by about 3.2×. The third approach fits the loss as:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

where $$N$$ is the number of parameters and $$D$$ the number of training tokens, with $$E = 1.69$$ (the entropy of natural text), $$A = 406.4$$, $$B = 410.7$$, $$\alpha = 0.34$$ and $$\beta = 0.28$$. The following pictures illustrate the predictions of the three approaches, compared with Kaplan et al. and with current large models, and the training curves of the first approach:

{: .picture}
<img loading="lazy" width="60.6%" height="60.6%" src="{{ site.assets }}/publications/training_compute_optimal_large_language_models.svg"/>

{: .caption}
<b>Overlaid predictions.</b> Predictions of the paper's three approaches, overlaid with the projections of Kaplan et al. (2020). All three approaches predict that current large models should be substantially smaller and therefore trained much longer than is currently done. <b><i>Chinchilla</i> outperforms <i>Gopher</i> and the other large models (Section 4.2 of the paper).</b> Adapted from: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

{: .picture}
<img loading="lazy" width="80.1%" height="80.1%" src="{{ site.assets }}/publications/training_compute_optimal_large_language_models_2.svg"/>

{: .caption}
<b>Training curve envelope.</b> The <b>left</b> panel shows all the training runs: model sizes from 70M to 10B parameters, each trained with four different cosine cycle lengths. The envelope of minimal loss per FLOP, taken from these curves, gives the optimal model size (<b>center</b>) and the optimal number of training tokens (<b>right</b>) for a given compute budget. Green: the projected optimal model size and number of training tokens for the FLOPs used to train <i>Gopher</i> ($$5.76 \times 10^{23}$$). Adapted from: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

The main conclusion is that current large language models are significantly undertrained, as their model size was scaled while the amount of training data was kept constant. This is demonstrated with a compute-optimal model, **Chinchilla** (70B parameters), trained with the same compute budget as Gopher (280B) on 4× more data (1.4T tokens, i.e. 20 tokens per parameter). Chinchilla outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B) and Megatron-Turing NLG (530B) on several evaluation tasks, reaching 67.5% average accuracy on [MMLU](https://en.wikipedia.org/wiki/MMLU), more than 7 points above Gopher.

Chinchilla revises the allocation of [the scaling laws](#scaling-laws) that [GPT-3](#gpt-3) followed, and [LLaMA](#llama) later trained smaller models on more tokens than Chinchilla recommends, to lower the cost of inference.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2021</span></td>
<td markdown="block">
<details id="roformer">
<summary markdown="span"><span class="pub-title">RoFormer: Enhanced Transformer with Rotary Position Embedding (Rotary Embedding, RoPE)</span><span class="pub-meta">, Zhuiyi Technology</span><a class="pub-link" href="https://arxiv.org/abs/2104.09864" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Traditional positional encoding methods, like sinusoidal or learned embeddings, struggle to generalize well to long sequences because they either: (1) Use absolute positions that are fixed and cannot model relative relationships effectively. (2) Lack a mechanism to extrapolate beyond the sequence lengths seen during training.

**Rotary Position Embedding** (RoPE) addresses the first issue by encoding **relative positional information** directly into the attention mechanism, with an inter-token dependency that decays as the relative distance grows: "RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation". On its own, RoPE still extrapolates poorly beyond the training length, which later motivated [position interpolation](https://arxiv.org/abs/2306.15595), NTK-aware scaling and [YaRN](https://arxiv.org/abs/2309.00071).

Rotary embeddings modify the query (Q) and key (K) embeddings in self-attention. They do this by applying [rotations](https://en.wikipedia.org/wiki/Rotation_matrix) to the embeddings based on the positions of tokens in the sequence, as:

$$
\begin{bmatrix}
(R_p x)_{2i-1} \\
(R_p x)_{2i}
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_{p,i} & -\sin\theta_{p,i} \\
\sin\theta_{p,i} & \cos\theta_{p,i}
\end{bmatrix}
\cdot
\begin{bmatrix}
x_{2i-1} \\
x_{2i}
\end{bmatrix}
$$

The rotation angle of the $$i$$-th pair of dimensions at position $$p$$ is $$\theta_{p,i} = p \, w_i$$, where $$w_i = 10000^{-2(i-1)/d}$$ is the frequency of the pair. Important bit: each pair $$(x_{2i-1}, x_{2i})$$ of the input $$x$$ is rotated as a vector of 2D coordinates. To handle a $$d$$-dimensional input, $$x$$ is split into $$d/2$$ pairs of dimensions $$[ (x_1, x_2), \, (x_3, x_4), \, \dots, \, (x_{d-1}, x_d)]$$, each pair is rotated by its own angle, and the results are then concatenated into the rotated vector $$R_p x$$, so that $$R_p$$ is a block-diagonal matrix of $$d/2$$ rotations. The pairs thus encode the position at different scales: with $$d = 128$$, the first pair turns once every 6.3 tokens, and the last one once every 54,410 tokens. If $$x$$ has an odd dimensionality $$d$$, the extra dimension is often left unrotated.

For a query $$q$$ at position $$m$$ and a key $$k$$ at position $$n$$, since rotations compose and the transpose of a rotation is the rotation by the opposite angle, $$\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$$: the attention score depends only on the relative position $$n-m$$.

RoFormer reaches 27.5 BLEU on WMT 2014 English-to-German translation, against 27.3 for the Transformer-base, and converges faster than BERT in pre-training. RoPE then became the position encoding of many LLMs, e.g. [LLaMA](#llama).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="self-attention-memory">
<summary markdown="span"><span class="pub-title">Self-Attention Does Not Need $$O(n^2)$$ Memory</span><span class="pub-meta">, Google Research</span><a class="pub-link" href="https://arxiv.org/abs/2112.05682" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Self-attention is commonly believed to need $$O(n^2)$$ memory for a sequence of length $$n$$, as it computes an $$n \times n$$ matrix of attention scores: at 16,384 tokens, this matrix alone takes 1 GiB in FP32, for each head and each sequence. Rabe & Staats formalize the idea that self-attention can be computed without storing the full matrix, by a streaming (online) computation of the softmax-normalized attention: the keys and values are processed in chunks, and the softmax normalization is accumulated along with the weighted sum of the values. This is a conceptual ancestor of several practical kernels: it gives the mathematical justification for computing attention in blocks while maintaining numerical stability and correctness.

They give an algorithm that needs $$O(1)$$ memory for single-query attention and $$O(\log n)$$ for self-attention, plus a practical accelerator implementation with $$O(\sqrt{n})$$ memory that runs within a few percent of standard attention. At sequence length 16,384, it reduces the memory overhead of self-attention by 59× for inference and 32× for differentiation.

It still matters in modern systems papers, as it gives a clean argument that not materializing $$QK^\top$$ is not an approximation: the result is exact. It underpins many attention kernels that trade memory for compute and make long-context attention feasible, such as [FlashAttention](#flashattention), which also minimizes the reads and writes of the GPU memory to make the same blockwise computation faster than standard attention.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="performers">
<summary markdown="span"><span class="pub-title">Rethinking Attention with Performers</span><span class="pub-meta">, Google, Cambridge, DeepMind and Alan Turing Institute (ICLR 2021)</span><a class="pub-link" href="https://arxiv.org/abs/2009.14794" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Regular attention stores the $$L \times L$$ attention matrix of a sequence of length $$L$$, which costs $$O(L^2 d)$$ time and $$O(L^2 + Ld)$$ memory for a head dimension $$d$$.

From the abstract: Performers are "Transformer architectures which **can estimate regular
(softmax) full-rank-attention Transformers with provable accuracy**, but using only
linear (as opposed to quadratic) space and time complexity, without relying on
any priors such as sparsity or low-rankness. To approximate softmax attention kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+)".

FAVOR+ approximates the softmax kernel $$\exp(q^\top k)$$ of a query $$q$$ and a key $$k$$ with $$m$$ [random features](https://en.wikipedia.org/wiki/Random_feature) of each, $$\phi(x) = \frac{1}{\sqrt{m}} \exp(-\lVert x \rVert^2/2) \, [\exp(\omega_1^\top x), \dots, \exp(\omega_m^\top x)]$$, where the $$\omega_i$$ are random Gaussian vectors, drawn orthogonal to each other to lower the variance of the estimate. These features are positive, unlike the sine and cosine features of earlier work, which make the estimate unstable where the attention scores are close to 0.

A clearer explanation can be found on this [Google Research post](https://research.google/blog/rethinking-attention-with-performers/):

**Bidirectional attention**, where there is no notion of past and future: the matrices $$Q'$$ and $$K'$$ (of size $$L \times m$$) of the random features of the queries and the keys are used in a [lower rank decomposition](https://en.wikipedia.org/wiki/Low-rank_approximation) of the attention matrix $$A$$. Decoupling them, and conducting the matrix multiplications in the order indicated by the dashed boxes of the following picture, gives a linear attention mechanism in $$O(Lmd)$$ time, which never explicitly constructs $$A$$ or its approximation:

{: .picture}
<img loading="lazy" width="70.6%" height="70.6%" src="{{ site.assets }}/publications/performers.svg"/> 

{: .caption}
**Left:** Standard attention module computation, where the final desired result is computed by performing a matrix multiplication with the attention matrix $$A$$ and value tensor $$V$$. **Right:** Decoupling matrices $$Q′$$ and $$K′$$ used in the lower rank decomposition of $$A$$ and conducting the matrix multiplications in the order indicated by the dashed boxes gives a linear attention mechanism, which never explicitly constructs $$A$$ or its approximation. Adapted from: [Rethinking Attention with Performers](https://research.google/blog/rethinking-attention-with-performers/)

**Unidirectional (causal) attention**, where tokens do not attend to other tokens appearing later in the sequence: the previous approach is modified to use [prefix-sum](https://en.wikipedia.org/wiki/Prefix_sum) computations. These only store running totals of matrix computations, rather than an explicit lower-triangular regular attention matrix, as illustrated in the following animation:

{: .picture}
<img loading="lazy" width="75%" height="75%" src="{{ site.assets }}/publications/performers2.gif"/> 

{: .caption}
**Left:** Standard unidirectional attention requires masking the attention matrix to obtain its lower-triangular part. **Right:** Unbiased approximation on the LHS can be obtained via a prefix-sum mechanism, where the prefix-sum of the outer-products of random feature maps for keys and value vectors is built on the fly and left-multiplied by query random feature vector to obtain the new row in the resulting matrix. Source: [Rethinking Attention with Performers](https://research.google/blog/rethinking-attention-with-performers/)

On protein sequences of 8,192 tokens, the Performer trains at a batch size of 8 per chip, where a regular Transformer runs out of memory even at a batch size of 1. Like [Sparse Transformers](#sparse-transformers), Performers lower the quadratic cost of attention, but by approximating the whole attention matrix instead of computing a sparse subset of it.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="lora">
<summary markdown="span"><span class="pub-title">LoRA: Low-Rank Adaptation of Large Language Models</span><span class="pub-meta">, Microsoft</span><a class="pub-link" href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Full [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) of a large pre-trained model updates all its weights, so each fine-tuned model is a full copy of the model. **Low-Rank Adaptation** (LoRA) instead injects trainable low-rank parameter updates into the layers of the frozen model, which drastically reduces the number of trainable parameters while maintaining the model's expressive power. It is particularly useful for adapting pre-trained models to new tasks or domains without retraining the entire model, which is computationally expensive and requires large storage resources.

A small, trainable low-rank matrix $$\Delta W$$ is added to the original (frozen) weight matrix $$W \in \mathbb{R}^{d \times k}$$, so that the final weight matrix is $$W' = W + \Delta W$$. Instead of directly learning $$\Delta W$$, it is factorized as $$\Delta W = BA$$, with $$B \in \mathbb{R}^{d \times r}$$ and $$A \in \mathbb{R}^{r \times k}$$ of a rank $$r$$ much smaller than $$d$$ and $$k$$ (the picture below shows the square case $$k = d$$). $$A$$ is initialized with random Gaussian values and $$B$$ with zeros, so that training starts from the pretrained model, the update is scaled by $$\alpha / r$$, and the paper applies it to the attention projections (e.g. $$W_q$$, $$W_v$$).

Because $$\Delta W$$ can be merged into $$W$$ after training, there is no extra inference latency. Compared to full fine-tuning of GPT-3 175B with Adam, LoRA reduces the number of trainable parameters by 10,000× and the GPU memory requirement by 3× (from 1.2 TB to 350 GB), and shrinks the checkpoint of each task from 350 GB to 35 MB, while matching or exceeding the fine-tuning quality on RoBERTa, DeBERTa, GPT-2 and GPT-3. The following picture illustrates the reparametrization, where the input goes through the frozen pretrained weights (left) and through the two low-rank factors (right), and the two outputs are summed:

{: .picture}
<img loading="lazy" width="24.5%" height="24.5%" src="{{ site.assets }}/publications/lora.svg"/>

{: .caption}
LoRA reparametrization: only <i>A</i> and <i>B</i> are trained. Adapted from: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

[DyLoRA](#dylora) trains the LoRA blocks for a range of ranks at once, and [QLoRA](#qlora) fine-tunes them on top of a 4-bit quantized model, which fits a 65B-parameter model on one 48 GB GPU.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="clip">
<summary markdown="span"><span class="pub-title">Learning Transferable Visual Models From Natural Language Supervision (CLIP)</span><span class="pub-meta">, OpenAI</span><a class="pub-link" href="https://arxiv.org/abs/2103.00020" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

State-of-the-art vision systems are trained to predict a fixed, predetermined set of objects (labels), so additional labeled data is needed to specify any other visual concept. In natural language processing, however, "the development of text-to-text as a standardized input-output interface has enabled task-agnostic architectures to zero-shot transfer to downstream datasets, removing the need for specialized output heads or dataset specific customization".

A critical insight is that natural language can be used as a flexible prediction space, to enable generalization and transfer: natural language is able to express, and therefore supervise, a much wider set of visual concepts through its generality. Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches: it does not only learn a representation, but also connects that representation to language, which enables flexible [zero-shot](https://en.wikipedia.org/wiki/Zero-shot_learning) transfer. With that in mind, this paper introduces **CLIP** (Contrastive Language–Image Pre-training), a neural network that efficiently learns visual concepts from natural language supervision, and can be instructed in natural language to perform a great variety of classification benchmarks, without directly optimizing for the benchmark's performance, similar to the zero-shot capabilities of GPT-2 and [GPT-3](#gpt-3). The key point is that a model that is not optimized directly for the benchmark becomes much more representative.

In practice, CLIP trains an image encoder and a text encoder jointly, from scratch, to predict which images were paired with which texts in its dataset: 400 million pairs of images and texts from the internet, collected with 500,000 queries. For the image encoder, the authors consider 5 [ResNets](https://en.wikipedia.org/wiki/Residual_neural_network) (ResNet-50, ResNet-101 and three EfficientNet-style scaled-up ResNet-50s with 4×, 16× and 64× the compute), with improvements such as an attention pooling similar to a QKV attention, and 3 [Vision Transformers](https://en.wikipedia.org/wiki/Vision_transformer). The text encoder is a transformer with masked self-attention, with a [byte-pair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) of 49,152 tokens and sequences capped at 76 tokens (Section 2.4). The following picture illustrates the pre-training of the two encoders (1), and their use for zero-shot prediction (2 and 3):

{: .picture}
<img loading="lazy" width="67.6%" height="67.6%" src="{{ site.assets }}/publications/clip.svg"/> 

{: .caption}
Summary of CLIP: (1) contrastive pre-training, (2) creating a dataset classifier from label text and (3) using it for zero-shot prediction. While standard image models jointly train an image feature extractor and a linear classifier to predict some label, CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset’s classes. Adapted from: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

The contrastive pre-training (1) works on a mini-batch of $$N$$ pairs. Each image $$i$$ goes through the image encoder, which gives its embedding $$I_i$$, and each text $$i$$ through the text encoder, which gives $$T_i$$. The matrix of the products $$I_i \cdot T_j$$ holds the $$N$$ correct pairs on its diagonal (in blue) and the $$N^2 - N$$ wrong ones off the diagonal, and training maximizes the first and minimizes the second. In practice, "the cosine similarity of these embeddings is then calculated, scaled by a temperature parameter τ, and normalized into a probability distribution via a softmax" (Section 3.1.2 of the paper), where the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of two embeddings is the cosine of the angle between them. Each row of the matrix (e.g. $$I_3 \cdot T_1, \dots, I_3 \cdot T_N$$) thus gives the classification of an image over the texts, each column that of a text over the images, and the loss is the symmetric [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) of the two directions (Figure 3 of the paper gives the pseudocode). The mini-batch size $$N$$ is therefore critical: the larger it is, the better it approximates the entire dataset and the more detailed the representations, but the matrix grows quadratically with $$N$$. CLIP uses mini-batches of 32,768 pairs, i.e. about 1.07 billion products per step.

At inference, in (2), a classifier is created from a set of labels by adding a prompt to each label, e.g. `A photo of a {object}.`, and passing the prompts through the text encoder: their embeddings are the target set. Then, to perform zero-shot prediction (3), an image is passed through the image encoder, which gives its classification distribution over the prompted labels, and the top label is picked: in the picture, `A photo of a dog.` No training is needed on the target task, whose images and labels can be entirely different from the training ones. This is a fundamental difference from regular image classification, which has a fixed set of classes: CLIP learns the structure of the language, not only how to differentiate classes.

The main results of the paper:
- Zero-shot accuracy: the best CLIP model reaches 76.2% on ImageNet, the accuracy of the original ResNet-50. Predicting a bag-of-words of the caption learns 3× faster than predicting its exact words, and the contrastive objective another 4× faster (Figure 2). Prompt engineering and ensembling add almost 5 points on average across 36 datasets (Figure 4), a gain similar to using 4× more compute, but "free" at inference.
- Against supervised models (Figure 5): zero-shot CLIP beats a **linear probe**, a logistic regression classifier on the features of a ResNet-50 trained on ImageNet, on 16 of 27 datasets, even on ImageNet (+1.9 points), and by 28.9 points on Stanford Cars; on STL10, it reaches 99.3%, a new state of the art without any training examples. On [MNIST](https://en.wikipedia.org/wiki/MNIST_database), however, it reaches only 88%, below a logistic regression on the raw pixels, as almost no images of its pre-training data resemble MNIST digits (Section 6).
- Few-shot (Figure 6): zero-shot CLIP matches the average performance of a 4-shot linear classifier trained on its own feature space.
- Robustness (Figure 13): a model trained on ImageNet loses much of its accuracy when the images change but the labels do not (ImageNet sketches, adversarial images, etc.), and zero-shot CLIP reduces this robustness gap by up to 75%.
- Bias (Table 7): the class design matters, as adding the label `child` to the label set dramatically drops the percentage of images of people under 20 that are assigned non-human or crime-related labels.

The objective of CLIP descends from [the contrastive loss](#contrastive-loss), with a softmax over the texts of the mini-batch instead of a margin on pairs, and its best image encoders are [Vision Transformers](#vit).

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="gspmd">
<summary markdown="span"><span class="pub-title">GSPMD: General and Scalable Parallelization for ML Computation Graphs</span><span class="pub-meta">, Google</span><a class="pub-link" href="https://arxiv.org/pdf/2105.04663.pdf" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

(also covered in a [Google blog post](https://research.google/blog/general-and-scalable-parallelization-for-neural-networks/))

GSPMD (General and Scalable Parallelization for ML Computation Graphs) is an open-source, automatic, compiler-based parallelization system based on the [XLA compiler](https://openxla.org/xla). Different model architectures may be better suited to different parallelization strategies. Therefore, GSPMD is designed to support a large variety of parallelism algorithms appropriate for different use cases (e.g. data parallelism for small models, pipeline parallelism for larger models, or a combination of both).

In GSPMD, each tensor is assigned a **sharding property**, either explicitly by the user as initial annotations, or by the sharding completion pass. The sharding property specifies how the data is distributed across devices. GSPMD defines three types of sharding: **replicated** (all devices have the same full data), **tiled** (a tiled sharding of the tensor, without data duplication), and **partially tiled**. The partially tiled sharding is an extension to [GShard](https://arxiv.org/abs/2006.16668), where the devices are divided into subgroups, and the tensor is tiled across subgroups but replicated within each subgroup. 

The sharding properties are user-defined with `mesh_split(tensor, device_mesh, dims_mapping)` that allows a tensor to be split across the device mesh, given a mapping from each data tensor dimension (i) to an optional device mesh dimension. This simple API is general enough to express
all types of sharding, across the dimension(s) of batch, features, channels and/or others.

The sharding completion pass is implemented in the XLA compiler (Section 3.5). It uses information about the operator (e.g. dot product is a generalized matrix multiply), and iterative methods where the shardings assigned by the pass are refined incrementally over the iterations.

GSPMD achieves 50% to 62% compute utilization on up to 2048 Cloud TPUv3 cores for models with up to one trillion parameters. It produces a single program for all devices ([SPMD](https://en.wikipedia.org/wiki/Single_program,_multiple_data)), which is crucial for scaling to thousands of devices, as one program per device would increase the compilation time significantly. The following picture illustrates the sharding annotations of the two matrix multiplications of a feed-forward layer (left), and the program of one device after partitioning, where GSPMD inserts all-gather (AG) and reduce-scatter (RS) collectives (right):

{: .picture}
<img loading="lazy" width="88.1%" height="88.1%" src="{{ site.assets }}/publications/gspmd.svg"/>

{: .caption}
**Left:** Computation where tensors are annotated to be split over 4 devices. A simplified feedforward layer of a Transformer model. Blue rectangles represent tensors with dashed red & blue lines overlaid representing the desired partitioning across a 2x2 mesh of devices. **Right:** Partitioned execution graph generated by GSPMD. A single partition, after GSPMD has been applied. Adapted from: [General and Scalable Parallelization for Neural Networks](https://research.google/blog/general-and-scalable-parallelization-for-neural-networks/)

GSPMD is the partitioner of XLA, whose fusion passes [Operator Fusion in XLA](#operator-fusion-in-xla) analyzes, and [DeepCompile](#deepcompile) brings the same compiler-driven approach to PyTorch.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="mlp-mixer">
<summary markdown="span"><span class="pub-title">MLP-Mixer: An all-MLP Architecture for Vision</span><span class="pub-meta">, Google (NeurIPS 2021)</span><a class="pub-link" href="https://arxiv.org/abs/2105.01601" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Computer vision models are built on convolutions ([CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network)) or attention (Transformers). The paper argues that neither is necessary, and presents **MLP-Mixer**, an architecture based only on [multi-layer perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLPs): "MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. 'mixing' the per-location features), and one with MLPs applied across patches (i.e. 'mixing' spatial information)." The input is a table of $$S$$ patches (rows) by $$C$$ channels (columns): the **token-mixing** MLP acts on each column, shared across the channels, and the **channel-mixing** MLP on each row, shared across the patches. As the hidden width of the token-mixing MLP is chosen independently of the number of patches, the computational complexity of the network is linear in the number of patches, while it is quadratic in a Vision Transformer.

When pre-trained on large datasets (or with modern regularization), the results are competitive with existing methods, at a comparable pre-training and inference cost: pre-trained on JFT-300M, Mixer-H/14 reaches 87.94% top-1 accuracy on ImageNet, at 40 images per second per core, against 88.55% for ViT-H/14 at 15 images per second. Trained on ImageNet only, Mixer-B/16 is about 3 points below ViT-B/16 (76.44% against 79.67%). [gMLP](#pay-attention-to-mlps) later adds a gating unit across the tokens. The following picture illustrates the architecture of MLP-Mixer (bottom) and one of its Mixer layers (top), where the table of patches and channels is transposed (T) before and after the token-mixing MLP:

{: .picture}
<img loading="lazy" width="92.5%" height="92.5%" src="{{ site.assets }}/publications/mlp_mixer.svg"/>

{: .caption}
MLP-Mixer consists of per-patch linear embeddings, Mixer layers, and a classifier head. Mixer layers contain one token-mixing MLP and one channel-mixing MLP, each consisting of two fully-connected layers and a GELU nonlinearity. Other components include: skip-connections, dropout, and layer norm on the channels. Adapted from: [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="pay-attention-to-mlps">
<summary markdown="span"><span class="pub-title">Pay Attention to MLPs</span><span class="pub-meta">, Google (NeurIPS 2021)</span><a class="pub-link" href="https://arxiv.org/abs/2105.08050" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Transformers mix the information of the tokens with self-attention. This paper introduces **gMLP** (gated MLPs), which mixes the tokens with a gated linear projection instead, building on [MLP-Mixer](#mlp-mixer), and shows that it can perform as well as Transformers in language and vision applications. It claims that "self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy": on ImageNet, gMLP-B reaches 81.6% top-1 accuracy with 73M parameters, against 81.8% for the [Vision Transformer](https://en.wikipedia.org/wiki/Vision_transformer) DeiT-B with 86M. In some [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) tasks it performed better than Transformers, and on other fine-tuning tasks it performed worse, such as SQuAD v2.0 (an F1 score of 78.3 against 81.0 for BERT-large), but this can be overcome by making the gMLP model substantially larger, or by adding a tiny single-head attention (aMLP, with an F1 score of 85.4).

A gMLP has no self-attention, and instead relies on channel projections and spatial projections with static parameterization. It consists of a stack of $$L$$ blocks with identical size and structure. Each block is defined as:

$$
Z = σ(XU), \,\,\,\,\,\,\,\, \tilde{Z} = s(Z), \,\,\,\,\,\,\,\, Y = \tilde{Z} V
$$

where $$σ$$ is an activation function, $$U$$ and $$V$$ are linear projections along the channel dimension, and $$s(·)$$ is a layer which captures spatial interactions. When $$s$$ is an identity mapping, the above transformation degenerates to a regular FFN, i.e. no cross-token communication.

Here, $$s(·)$$ is the **Spatial Gating Unit** (Section 2.1): $$Z$$ is split along channels into $$(Z_1, Z_2)$$ and $$s(Z) = Z_1 \odot (W Z_2 + b)$$, where $$\odot$$ is the element-wise product and $$W$$ is an $$n \times n$$ matrix, for a sequence of $$n$$ tokens, that projects across the token (spatial) dimension. The paper finds it "critical to initialize W as near-zero values and b as ones", so that the unit starts close to an identity and the block close to a regular FFN. Unlike Transformers, gMLP does not require position embeddings, because the positional information is captured in $$s(·)$$. The following picture illustrates a gMLP block (left), its Spatial Gating Unit (center) and their pseudo-code (right):

{: .picture}
<img loading="lazy" width="92.7%" height="92.7%" src="{{ site.assets }}/publications/pay_attention_to_mlps.svg"/>

{: .caption}
Overview of the gMLP architecture with Spatial Gating Unit (SGU). The model consists of a stack of <i>L</i> blocks with identical structure and size. All projection operations are linear and “⊙” refers to element-wise multiplication (linear gating). The input and output protocols follow BERT for NLP and ViT for vision. Unlike Transformers, gMLPs do not require positional encodings, nor is it necessary to mask out the paddings during NLP finetuning. Adapted from: [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="vit">
<summary markdown="span"><span class="pub-title">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</span><span class="pub-meta">, Google (ICLR 2021)</span><a class="pub-link" href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The Transformer had become the standard model of natural language processing, while computer vision still relied on [convolutional networks](https://en.wikipedia.org/wiki/Convolutional_neural_network), as letting every pixel of an image attend to every other pixel is too expensive. The paper introduces the **[Vision Transformer](https://en.wikipedia.org/wiki/Vision_transformer)** (ViT), an extension of the transformer architecture to images: an image of $$H \times W$$ pixels is split into $$N = HW/P^2$$ patches of $$P \times P$$ pixels (e.g. 16 × 16), and the transformer takes as input the sequence of linear embeddings of these patches, as the tokens of a sentence. A 224 × 224 image becomes 196 tokens instead of 50,176 pixels, with 65,536× fewer pairs in the attention.

When pre-trained on large datasets (ImageNet-21k, JFT-300M), ViT matches or beats state-of-the-art CNNs (ResNets) on image classification, while requiring substantially fewer computational resources to train: ViT-H/14 reaches 88.55% top-1 accuracy on ImageNet, 94.55% on CIFAR-100 and 77.63% on the VTAB suite, with 2.5k TPUv3 core-days of pre-training, against 9.9k for BiT-L and 12.3k for Noisy Student. However, on mid-sized datasets like ImageNet alone, it is a few percentage points below comparable ResNets, since it lacks the convolutional [inductive biases](https://en.wikipedia.org/wiki/Inductive_bias) (locality, translation equivariance). [MLP-Mixer](#mlp-mixer) replaces its attention with MLPs, and [CLIP](#clip) uses it as an image encoder. The following picture illustrates the model (left) and its Transformer encoder (right):

{: .picture}
<img loading="lazy" width="76.2%" height="76.2%" src="{{ site.assets }}/publications/visual_transformer.svg"/>

{: .caption}
Model overview. The image is split into fixed-size patches; each patch is linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. For classification, an extra learnable “classification token” is added to the sequence, the standard approach. The illustration of the Transformer encoder was inspired by Vaswani et al. (2017). Adapted from: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2020</span></td>
<td markdown="block">
<details id="scaling-laws">
<summary markdown="span"><span class="pub-title">Scaling Laws for Neural Language Models</span><span class="pub-meta">, Johns Hopkins, OpenAI</span><a class="pub-link" href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Abstract: "We study empirical scaling laws for language model performance on the cross-entropy loss.
The loss scales as a power-law with model size, dataset size, and the amount of compute
used for training, with some trends spanning more than seven orders of magnitude. Other
architectural details such as network width or depth have minimal effects within a wide
range. Simple equations govern the dependence of overfitting on model/dataset size and the
dependence of training speed on model size. These relationships allow us to determine the
optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models
on a relatively modest amount of data and stopping significantly before convergence."

The following picture illustrates the test loss as a function of the compute, the dataset size and the number of parameters:

{: .picture}
<img loading="lazy" width="92.8%" height="92.8%" src="{{ site.assets }}/publications/scaling_laws.svg"/>

{: .caption}
Language modeling performance improves smoothly as the model size, dataset size, and amount of compute used for training increase. For optimal performance all three factors must be scaled up in tandem. Empirical performance has a power-law relationship with each individual factor when not bottlenecked by the other two. Adapted from: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

**Key points:**
- Model performance depends most strongly on scale, which consists of three factors: the number of model parameters $$N$$ (excluding embeddings), the size of the dataset $$D$$, and the amount of compute $$C$$ used for training. Performance has a **[power-law](https://en.wikipedia.org/wiki/Power_law)** relationship with each of the three scale factors when not bottlenecked by the other two (Fig. 1), e.g. $$L(N) = (N_c/N)^{\alpha_N}$$ with $$\alpha_N \approx 0.076$$.
- Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width: what matters is the number of non-embedding parameters, $$N \approx 12 \, n_{layer} \, d_{model}^2$$.
- Performance improves predictably as long as $$N$$ and $$D$$ are scaled up in tandem, but enters a regime of diminishing returns if either $$N$$ or $$D$$ is held fixed while the other increases: to avoid overfitting, $$D$$ should grow as $$N^{0.74}$$.
- When models are evaluated on text with a different distribution than the one they were trained on, the results are strongly correlated to those on the training validation set, with a roughly constant offset in the loss: the transfer to a different distribution incurs a constant penalty, but improves in line with the performance on the training set.
- When working within a fixed compute budget $$C$$ but without any other restrictions on the model size $$N$$ or available data $$D$$, optimal performance is attained by training very large models and stopping significantly short of convergence: the optimal model size grows as $$C^{0.73}$$, and the data only as $$C^{0.27}$$. [GPT-3](#gpt-3) followed this allocation, and [Chinchilla](#chinchilla) later revised it: parameters and training tokens should grow in equal proportion.
- The ideal batch size for training these models is roughly a power of the loss only (Eq. 1.4).

The following picture illustrates the test loss of all training runs, against the tokens processed (left) and against the compute (right):

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
<details id="gpt-3">
<summary markdown="span"><span class="pub-title">Language Models are Few-Shot Learners (GPT-3)</span><span class="pub-meta">, OpenAI</span><a class="pub-link" href="https://arxiv.org/abs/2005.14165" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Before GPT-3, substantial gains on many NLP tasks were achieved by pre-training on a large corpus of text followed by [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) on a specific task, which still requires task-specific datasets of thousands or tens of thousands of examples. This paper shows that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. It presents **GPT-3**, an autoregressive language model with 175 billion parameters, 10× more than any previous non-sparse language model, and tests it in the few-shot setting.

The paper compares four settings (Fig. 2.1):
- Fine-Tuning (FT), the most common approach in recent years, updates the weights of a pre-trained model by training on a supervised dataset specific to the desired task.
- Few-Shot (FS) gives the model a few demonstrations of the task at inference time as conditioning (as in GPT-2), typically 10 to 100, but allows no weight updates.
- One-Shot (1S) is the same as few-shot, except that only one demonstration is allowed, in addition to a natural language description of the task.
- Zero-Shot (0S) is the same as one-shot, except that no demonstrations are allowed, and the model is only given a natural language instruction describing the task.

The few-shot settings rely on **in-context learning**: the model adapts to the task from the examples in its input, during the forward pass, with no weight update. The following picture illustrates the outer and inner loops of the meta-learning of a language model:

{: .picture}
<img loading="lazy" width="92.4%" height="92.4%" src="{{ site.assets }}/publications/gpt3_fig11.svg"/> 

{: .caption}
<b>Language model meta-learning.</b> During unsupervised pre-training, a language model develops a broad set of skills and pattern recognition abilities. It then uses these abilities at inference time to rapidly adapt to or recognize the desired task. The term “in-context learning” describes the inner loop of this process, which occurs within the forward-pass upon each sequence. The sequences in this diagram are not intended to be representative of the data a model would see during pre-training, but are intended to show that there are sometimes repeated sub-tasks embedded within a single sequence. Adapted from: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

GPT-3 uses the same model and architecture as [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), including the modified initialization, pre-normalization, and reversible tokenization described therein, except that it alternates dense and locally banded sparse attention patterns in the layers of the transformer, similar to the [Sparse Transformer](https://arxiv.org/abs/1904.10509). The largest of the 8 models of Table 2.1 has 96 layers of size 12,288 and 96 attention heads. Based on the analysis of [Scaling Laws For Neural Language Models](https://arxiv.org/abs/2001.08361), the authors train much larger models on many fewer tokens than is typical (Fig. 2.2): 300 billion tokens for the 175B model, i.e. about $$6ND \approx 3.15 \times 10^{23}$$ FLOPs for $$N$$ parameters and $$D$$ tokens, and the cross-entropy loss follows a power-law trend with the compute used for training (Fig. 3.1).

**Tasks tested and performance**:
- On NLP tasks, GPT-3 achieves promising results in the zero-shot and one-shot settings, and in the few-shot setting it is sometimes competitive with or even occasionally surpasses the state of the art (86.4% few-shot accuracy on LAMBADA, against 68.0%), also on tasks that test on-the-fly reasoning, such as unscrambling words, performing arithmetic, or using a novel word in a sentence after seeing it defined only once.
- Figs. 3.3 to 3.12 show that its performance grows with model size, on TriviaQA, translation, the adversarial [Winograd Schema Challenge](https://arxiv.org/abs/1907.10641) (WinoGrande), SuperGLUE, arithmetic and other tasks, suggesting that language models continue to absorb knowledge as their capacity increases.
- Fig. 3.13 shows that people's ability to identify whether news articles are model-generated decreases as model size increases, to about 52% for the 175B model, barely above chance.
- Data contamination has a minimal effect on most datasets, but could inflate the results of a few, such as PIQA and Winograd (Fig. 4.2). GPT-3 struggles with natural language inference (ANLI) and with some reading comprehension datasets, such as RACE or QuAC (Section 5).

[Chinchilla](#chinchilla) later revised this allocation of [the scaling laws](#scaling-laws): a compute-optimal model of 175B parameters would need 3.7T tokens, not 300B.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2019</span></td>
<td markdown="block">
<details id="graph-transformer-networks">
<summary markdown="span"><span class="pub-title">Graph Transformer Networks</span><span class="pub-meta">, Korea University (NeurIPS 2019)</span><a class="pub-link" href="https://arxiv.org/abs/1911.06455" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

One limitation of most [graph neural networks](https://en.wikipedia.org/wiki/Graph_neural_network) (GNNs) is that they assume the graph structure to be fixed and homogeneous, i.e. with similar types of nodes and edges. From the abstract: "Graph Transformer Networks (GTNs) are capable of generating new graph structures, which involve identifying useful connections between unconnected nodes on the original graph, while learning effective node representation on the new graphs in an end-to-end fashion. Graph Transformer layer, a core layer of GTNs, learns a soft selection of edge types and composite relations for generating useful multi-hop connections".
- GTNs perform **meta-path** generation: a meta-path defines a compositional relation over node and edge types (a sequence of relations). Following Section 3.1 of the paper, a meta-path of length $$k$$ is the composite relation $$R = t_1 \circ t_2 \circ \cdots \circ t_k$$ between the nodes $$v_1$$ and $$v_{k+1}$$, and its [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) is the product of the adjacency matrices of its edge types. A stack of $$l$$ Graph Transformer (GT) layers learns meta-paths of length up to $$l + 1$$ (Section 3.2), and the node representations are then learned with a [GCN](#gcn) on the new graphs.
- Without any predefined, domain-specific meta-paths, GTNs achieve state-of-the-art node classification on three heterogeneous graph benchmarks, with F1 scores of 94.18 on DBLP, 92.68 on ACM and 60.92 on IMDB, against 93.71, 92.33 and 58.14 for GAT.

The following picture illustrates a stack of GT layers, where each layer selects edge types with a 1x1 convolution, and the selected adjacency matrices are multiplied into meta-path graphs:

{: .picture}
<img loading="lazy" width="92.3%" height="92.3%" src="{{ site.assets }}/publications/graph_transformer_networks.svg"/>

{: .caption}
Graph Transformer Networks (GTNs) learn to generate a set of new meta-path adjacency matrices 𝔸<sup>(<i>l</i>)</sup> using GT layers and perform graph convolution as in GCNs on the new graph structures. Multiple node representations from the same GCNs on multiple meta-path graphs are integrated by concatenation and improve the performance of node classification. ℚ<sub>1</sub><sup>(<i>l</i>)</sup> and ℚ<sub>2</sub><sup>(<i>l</i>)</sup> ∈ <b>R</b><sup><i>N</i>×<i>N</i>×<i>C</i></sup> are intermediate adjacency tensors to compute meta-paths at the <i>l</i>th layer. Adapted from: [Graph Transformer Networks](https://arxiv.org/abs/1911.06455)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="rmsnorm">
<summary markdown="span"><span class="pub-title">Root Mean Square Layer Normalization (RMSNorm)</span><span class="pub-meta">, University of Edinburgh & University of Zurich (NeurIPS 2019)</span><a class="pub-link" href="https://arxiv.org/abs/1910.07467" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[LayerNorm](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization) stabilizes training and speeds up convergence by re-centering and re-scaling the inputs of a layer, but its computation slows down the network, especially recurrent ones. **RMSNorm** (Root Mean Square Normalization) is a simpler and computationally lighter alternative: instead of normalizing based on the mean and variance of the input features, it only scales the input using the root mean square of the features. It rests on the paper's hypothesis that "re-centering invariance in LayerNorm is dispensable", and that the re-scaling invariance is what matters for training stability. It is formulated as:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}
$$

where $$x$$ is the input vector with dimensionality $$d$$, and

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma
$$

where $$γ$$ is a learnable scaling parameter for each feature (similar to LayerNorm).

For example, $$x = (1, 2, 3, 4)$$ has $$\text{RMS}(x) = \sqrt{30/4} \approx 2.739$$, so RMSNorm gives $$x/\text{RMS}(x) \approx (0.365, 0.730, 1.095, 1.461)$$ (with $$\gamma = 1$$), while LayerNorm, which first subtracts the mean 2.5, gives $$(-1.342, -0.447, 0.447, 1.342)$$. In the experiments of the paper, RMSNorm reaches an accuracy comparable to LayerNorm while reducing the running time by 7% to 64% across models, which makes it the normalization of large-scale models such as [LLaMA](#llama). The paper also proposes pRMSNorm, which estimates the RMS from only $$p$$% of the inputs.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="multi-query-attention">
<summary markdown="span"><span class="pub-title">Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention)</span><span class="pub-meta">, Noam Shazeer, Google</span><a class="pub-link" href="https://arxiv.org/abs/1911.02150" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Efficient training in Transformer models is possible due to parallelism across the length dimension. However, decoding (where such parallelization is impossible) is slow due to continuously loading large keys and values tensors from memory. Thus, the paper introduces a variant of the [multi-head attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) (MHA) that improves inference (decoding), called **multi-query attention (MQA)**.

While MHA consists of multiple attention layers (heads) in parallel with different linear transformations on the queries, keys, values and outputs, MQA is identical except that the different heads share a single set of keys and values. This divides the size of these tensors by the number of heads, and hence the memory bandwidth requirements of incremental decoding. This leads to a much faster decoding, with a minor degradation of quality from the baseline: on WMT14 English-to-German translation, the decoder time per token falls from 46 µs to 3.8 µs with greedy decoding (12× faster), while the log-perplexity grows from 1.424 to 1.439, with a similar BLEU score. The grouped-query attention of [Llama 2](#llama-2), which shares keys and values within groups of heads, sits between MHA and MQA.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="sparse-transformers">
<summary markdown="span"><span class="pub-title">Generating Long Sequences with Sparse Transformers</span><span class="pub-meta">, OpenAI</span><a class="pub-link" href="https://arxiv.org/abs/1904.10509" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Self-attention compares every position of a sequence of length $$n$$ with every other one, so its memory and runtime grow quadratically with $$n$$. The paper introduces several **sparse factorizations of the attention matrix** that reduce this complexity to $$O(n \sqrt{n})$$, and allow for longer sequences. They separate the full attention computation into several faster attention operations which, when combined, can **approximate the dense attention** operation.

The authors claim that sparsity in attention is a natural pattern, and show (by visual inspection) various examples where most layers had sparse attention patterns across most data points. This suggests that adding sparsity to the attention would not significantly affect performance. In other layers, however, they noticed global patterns and data-dependent sparsity, whose performance could be affected by sparsity in the attention matrix.

**Factorized self-attention** has $$p$$ separate attention heads, where the $$m$$-th head attends only to a subset $$A^{(m)}_i$$ of the positions before position $$i$$. The hard problem here is to find efficient choices for these subsets. Section 4.3 details two 2D factorizations ($$p = 2$$), with a stride $$l$$ close to $$\sqrt{n}$$: in **strided attention**, one head attends to the previous $$l$$ positions and the other to every $$l$$-th position; in **fixed attention**, one head attends to the positions of the same block of $$l$$ positions, and the other to the last positions of each block, which summarize it. For the last of 4,096 tokens and $$l = 64$$, strided attention reads 127 positions instead of 4,096.

Strided attention is convenient when the data has a structure that aligns with the stride, like images or some types of music. For data without a periodic structure, like text, the strided pattern can fail to route information properly, and a fixed pattern is used instead. Both patterns are compared with the full attention in the following picture:

{: .picture}
<img loading="lazy" width="88%" height="88%" src="{{ site.assets }}/publications/sparse_transformers.svg"/>

{: .caption}
Two 2D factorized attention schemes of the Sparse Transformer, (b) strided and (c) fixed, compared with the full attention of a standard Transformer (a). The top row indicates, for an example 6x6 image, which positions two attention heads receive as input when computing a given output. The bottom row shows the connectivity matrix (not to scale) between all such outputs (rows) and inputs (columns). Sparsity in the connectivity matrix can lead to significantly faster computation. In (b) and (c), full connectivity between elements is preserved when the two heads are computed sequentially. The paper tests whether such factorizations can match the performance of the rich connectivity patterns learned by full attention (Figure 2 of the paper). Adapted from: [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)

With the same architecture, the paper sets new state-of-the-art density-modeling results on Enwik8 (0.99 bits per byte), CIFAR-10 and ImageNet 64 × 64 (2.80 and 3.44 bits per dimension), using hundreds of layers on sequences tens of thousands of timesteps long. [GPT-3](#gpt-3) alternates dense and sparse attention patterns of this kind in its layers, and [LongNet](#longnet) and [Native Sparse Attention](#native-sparse-attention) follow the same idea.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="albert">
<summary markdown="span"><span class="pub-title">ALBERT: A Lite BERT for Self-supervised Learning of Language Representations</span><span class="pub-meta">, Google and Toyota (ICLR 2020)</span><a class="pub-link" href="https://arxiv.org/abs/1909.11942" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Larger language models perform better, but at some point their growth is limited by the memory of the accelerators and by longer training times: [BERT](https://en.wikipedia.org/wiki/BERT_(language_model))-large already has 334M parameters. ALBERT ("A Lite BERT") lowers memory consumption and increases the training speed of BERT, which allows for better scaling: an ALBERT configuration similar to BERT-large has 18× fewer parameters and can be trained about 1.7× faster. The techniques introduced are:
1. **Factorized embedding parameterization**, for parameter reduction. "Instead of projecting the one-hot vectors directly into the hidden space of size $$H$$, we first project them into a lower dimensional embedding space of size $$E$$, and then project it to the hidden space. By using this decomposition, we reduce the embedding parameters from $$O(V × H)$$ to $$O(V × E + E × H)$$", where $$V$$ is the size of the vocabulary: with $$V = 30{,}000$$, $$E = 128$$ and $$H = 4{,}096$$, from 122.9M to 4.4M. This separation of the size of the hidden layers from the size of the vocabulary embedding makes it easier to grow the hidden size.
2. **Cross-layer parameter sharing**, for parameter reduction: ALBERT learns the parameters of a single Transformer encoder block and reuses it in all layers (e.g. 12), so that the number of parameters does not grow with the depth of the network. The authors mention that the parameter reduction also acts as a regularization, but the sharing lowers the accuracy slightly (by 1.5 points on average with $$E = 128$$), as the [Google blog post](https://research.google/blog/albert-a-lite-bert-for-self-supervised-learning-of-language-representations/) on ALBERT says: "This approach slightly diminishes the accuracy, but the more compact size is well worth the tradeoff".
3. **Self-supervised loss for sentence-order prediction (SOP)**, for performance improvement. BERT's additional loss, the next-sentence prediction (NSP, a binary classification loss for predicting whether two segments appear consecutively in the original text), is too easy, as the topic of the two segments is often enough to solve it. The SOP loss uses as positive examples two consecutive segments from the same document, as BERT does, and as negative examples the same two segments with their order swapped. This forces the model to learn finer-grained distinctions about discourse-level coherence properties: a model trained with NSP scores 52.0% on SOP, close to random, while a model trained with SOP scores 78.9% on NSP.

The best model, ALBERT-xxlarge (235M parameters), has fewer parameters than BERT-large, and its ensembles established new records: a score of 89.4 on GLUE, an F1 score of 92.2 on SQuAD 2.0 and an accuracy of 89.4% on RACE. Its factorized embedding is a low-rank factorization of the embedding matrix, the idea that [LoRA](#lora) later applies to the weight updates of fine-tuning.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2018</span></td>
<td markdown="block">
<details id="group-normalization">
<summary markdown="span"><span class="pub-title">Group Normalization</span><span class="pub-meta">, Facebook AI Research (ECCV 2018)</span><a class="pub-link" href="https://arxiv.org/abs/1803.08494" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[Batch Normalization](https://en.wikipedia.org/wiki/Batch_normalization) (BN) normalizes the features along the batch dimension, so its error increases rapidly when the batch is small, as in object detection, segmentation or video models, where the memory limits the batch to a few images per GPU. This paper presents **Group Normalization** (GN), which divides the channels into groups (32 by default) and normalizes the features within each group, so its computation is independent of the batch size. On ResNet-50 trained on ImageNet with a batch size of 2, GN has an error 10.6 points lower than BN (24.1% against 34.7%), and with a typical batch size of 32 it is comparably good (24.1% against 23.6%).

[Layer Normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization) and Instance Normalization (GN with one group, and with one channel per group) also avoid normalizing along the batch dimension. These methods are effective for training sequential models (RNN/LSTM) or generative models (GANs), but both have limited success in visual recognition, for which GN presented better results: on COCO, for example, a Mask R-CNN with GN reaches 40.8 box AP, 2.2 points more than with frozen BN. [RMSNorm](#rmsnorm) later simplified the layer normalization of Transformers. The following picture illustrates which values of a feature map tensor each method normalizes together:

{: .picture}
<img loading="lazy" width="78.9%" height="78.9%" src="{{ site.assets }}/publications/group_normalization.svg"/>

{: .caption}
<b>Normalization methods</b>. Each subplot shows a feature map tensor, with <i>N</i> as the batch axis, <i>C</i> as the channel axis, and (<i>H</i>, <i>W</i>) as the spatial axes. The pixels in blue are normalized by the same mean and variance, computed by aggregating the values of these pixels. Adapted from: [Group Normalization](https://arxiv.org/abs/1803.08494)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2017</span></td>
<td markdown="block">
<details id="vq-vae">
<summary markdown="span"><span class="pub-title">Neural Discrete Representation Learning (VQ-VAE)</span><span class="pub-meta">, DeepMind (NeurIPS 2017)</span><a class="pub-link" href="https://arxiv.org/abs/1711.00937" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The [Vector Quantised](https://en.wikipedia.org/wiki/Vector_quantization) Variational AutoEncoder (VQ-VAE) aims at learning **discrete (not continuous) latent space** representations without supervision. It differs from [VAEs](https://en.wikipedia.org/wiki/Variational_autoencoder) in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learned rather than static. The encoder output $$z_e(x)$$ is replaced by the nearest of the $$K$$ embedding vectors $$e_1, \dots, e_K$$ of a learned codebook, $$z_q(x) = e_k$$ with $$k = \arg\min_j \lVert z_e(x) - e_j \rVert_2$$ (equation 2), and the discrete latents avoid the "posterior collapse" of VAEs, where the latents are ignored when the decoder is a powerful autoregressive model.

"During forward computation the nearest embedding $$z_q(x)$$ (equation 2) is passed to the decoder, and
during the backwards pass the gradient $$∇_z L$$ is passed unaltered to the encoder. Since the output
representation of the encoder and the input to the decoder share the same $$D$$ dimensional space,
the gradients contain useful information for how the encoder has to change its output to lower the
reconstruction loss."

Equation 3 specifies the overall loss: the reconstruction loss, a codebook loss $$\lVert sg[z_e(x)] - e \rVert_2^2$$ that moves the embeddings towards the encoder outputs, and a commitment loss $$\beta \lVert z_e(x) - sg[e] \rVert_2^2$$ that keeps the encoder committed to an embedding, where $$sg$$ is the stop-gradient operator and $$\beta = 0.25$$ in all experiments. The gradient of the reconstruction loss is copied straight-through across the non-differentiable mapping from $$z_e(x)$$ to $$z_q(x)$$. The following picture illustrates the nearest-embedding lookup between the encoder and the decoder, and the straight-through gradient (in red):

{: .picture}
<img loading="lazy" width="92.4%" height="92.4%" src="{{ site.assets }}/publications/rqvae.svg"/>

{: .caption}
Left: A figure describing the VQ-VAE. Right: Visualisation of the embedding space. The output of the encoder <i>z</i><sub>e</sub>(<i>x</i>) is mapped to the nearest point <i>e</i><sub>2</sub>. The gradient ∇<sub><i>z</i></sub><i>L</i> (in red) will push the encoder to change its output, which could alter the configuration in the next forward pass. Adapted from: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

While the VQ-VAE is trained, the prior over the codes is kept uniform; after training, an autoregressive prior (a PixelCNN for images, a WaveNet for raw audio) is fitted over the discrete latents, and new samples are generated from it. On ImageNet, the VQ-VAE compresses 128 × 128 × 3 images of 8-bit values into 32 × 32 latents of $$K = 512$$ codes (9 bits each), about 43 times fewer bits. The autoregressive prior thus generates 1,024 latents instead of the 49,152 values of an image, which an autoregressive model of the pixels, such as [Sparse Transformers](#sparse-transformers), generates one by one.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="mixed-precision-training">
<summary markdown="span"><span class="pub-title">Mixed Precision Training</span><span class="pub-meta">, Baidu and NVIDIA (ICLR 2018)</span><a class="pub-link" href="https://arxiv.org/abs/1710.03740" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Training in [half precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) (FP16) halves the memory of the activations and the weights and speeds up the arithmetic, but FP16 represents a much smaller range of values than [single precision](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) (FP32): its largest value is 65,504, and magnitudes below $$2^{-24} \approx 6 \times 10^{-8}$$ become zero. The paper presents a method for training in FP16 without losing model accuracy or having to modify hyperparameters, with three techniques that prevent the loss of critical information (numerical underflows and overflows):
1. An **FP32 master copy of the weights**, which accumulates the gradients after each optimizer step, as the updates are often too small for FP16, and is rounded to FP16 for the forward pass and the backpropagation. In FP16, an update can be lost when the weight is at least $$2^{11}$$ (2,048) times larger than it, and about 5% of the weight gradients of a speech model are below $$2^{-24}$$: without the master copy, this model loses 80% of its accuracy (relative).
2. **Loss scaling**, to preserve gradient values with small magnitudes. The loss value computed in the forward pass is scaled prior to backpropagation, which (by the chain rule) shifts the gradient values into the FP16-representable range (a factor of 8 is enough for the SSD detector). The weight gradients must be unscaled before the weight update, to maintain the update magnitudes of FP32 training.
3. **FP16 arithmetic with FP32 accumulation**: the dot products accumulate their partial products into FP32 values, which are converted to FP16 before storing to memory, and the large reductions (as in batch normalization and softmax) run in FP32.

It matches FP32 accuracy across a wide range of tasks (image classification and detection, speech recognition, machine translation, language modeling, GANs) while nearly halving memory requirements: ResNet-50, for example, reaches 76.04% top-1 accuracy on ImageNet in mixed precision, against 75.92% in FP32. Later work lowers the precision further, to [FP8](#fp8-training) and to the 4-bit [NVFP4](#nvfp4) format.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2016</span></td>
<td markdown="block">
<details id="gcn">
<summary markdown="span"><span class="pub-title">Semi-Supervised Classification with Graph Convolutional Networks</span><span class="pub-meta">, University of Amsterdam (ICLR 2017)</span><a class="pub-link" href="https://arxiv.org/abs/1609.02907" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

In semi-supervised node classification, only a few nodes of a graph have a label, and the labels of the others must be inferred from the graph structure and the node features. The paper introduces the **Graph Convolutional Network** (GCN), based on a simple layer-wise propagation rule, $$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$, with node features $$H^{(l)}$$, layer weights $$W^{(l)}$$, an activation function $$\sigma$$, $$\tilde{A} = A + I$$ (the [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) $$A$$ with self-loops) and $$\tilde{D}$$ its [degree matrix](https://en.wikipedia.org/wiki/Degree_matrix). This rule is derived as a first-order approximation of [spectral](https://en.wikipedia.org/wiki/Spectral_graph_theory) graph convolutions. The model of the paper has two such layers, with 16 hidden units on the citation networks, and each layer costs $$O(\vert E \vert F C)$$, linear in the number of edges $$\vert E \vert$$, for $$C$$ input channels and $$F$$ output features.

Similarly to CNNs, GCNs learn the features by aggregating information from neighboring nodes. The main difference is that CNNs are meant to operate on regular Euclidean structures (e.g. images), while GCNs generalize this to arbitrary graph structures. On semi-supervised node classification in citation networks (Citeseer, Cora, Pubmed) and a knowledge graph (NELL), it outperforms related methods by a significant margin: on Cora, it reaches 81.5% accuracy, against 75.7% for Planetoid. [Graph Transformer Networks](#graph-transformer-networks) later run a GCN on the graphs that they learn. The following picture illustrates the GCN (a) and the hidden activations that it learns on the Cora dataset (b):

{: .picture}
<img loading="lazy" width="75.4%" height="75.4%" src="{{ site.assets }}/publications/graphconvnets.svg"/>

{: .caption}
(a) Schematic depiction of multi-layer Graph Convolutional Network (GCN) for semi-supervised learning with <i>C</i> input channels and <i>F</i> feature maps in the output layer. The graph structure (edges shown as black lines) is shared over layers, labels are denoted by <i>Y<sub>i</sub></i>. (b) t-SNE (Maaten & Hinton, 2008) visualization of hidden layer activations of a two-layer GCN trained on the Cora dataset (Sen et al., 2008) using 5% of labels. Colors denote document class. Adapted from: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2015</span></td>
<td markdown="block">
<details id="batch-normalization">
<summary markdown="span"><span class="pub-title">Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</span><span class="pub-meta">, Google (ICML 2015)</span><a class="pub-link" href="https://arxiv.org/abs/1502.03167" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[Batch Normalization](https://en.wikipedia.org/wiki/Batch_normalization) (BatchNorm, BN) is a technique used in deep learning to improve the training process of neural networks by normalizing the inputs to each layer within a mini-batch. It "reduces the internal covariate shift".

**Covariate shift** means the distribution of the features differs between training and test data, breaking the [i.i.d.](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) assumption used across most of ML. *Internal* covariate shift is the analogous drift inside the network: as the network learns and the weights are updated, the distribution of outputs of a specific layer in the network changes. This forces the higher layers to adapt to that drift, which slows down learning. BN helps by making the data flowing between intermediate layers of the network look like [whitened](https://en.wikipedia.org/wiki/Whitening_transformation) data, so a higher learning rate can be used.

For each feature, BN computes the mean $$\mu_B$$ and the variance $$\sigma_B^2$$ of its values over the mini-batch, normalizes them as $$\hat{x} = (x - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$$, and then scales and shifts them as $$y = \gamma \hat{x} + \beta$$, with learned $$\gamma$$ and $$\beta$$, so that the layer can still represent the identity. At inference, the statistics of the mini-batch are replaced by averages over the training data.

In the results, Batch Normalization applied to an Inception network achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. An ensemble of BN networks reaches 4.9% top-5 validation error on ImageNet, exceeding the accuracy of human raters. (Later work, e.g. [Santurkar et al. 2018](https://arxiv.org/abs/1805.11604), argues that BN helps mostly by smoothing the optimization landscape rather than by reducing internal covariate shift.) As BN depends on the batch, [Group Normalization](#group-normalization) normalizes within groups of channels instead, for small batches.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="siamese-networks">
<summary markdown="span"><span class="pub-title">Siamese Neural Networks for One-shot Image Recognition</span><span class="pub-meta">, University of Toronto (ICML 2015 Deep Learning Workshop)</span><a class="pub-link" href="https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

In **[one-shot learning](https://en.wikipedia.org/wiki/One-shot_learning_(computer_vision))**, a model must recognize new categories from a single example of each, where a regular classifier needs many. The paper describes **[siamese neural networks](https://en.wikipedia.org/wiki/Siamese_neural_network)** (defined in [the contrastive loss entry](#contrastive-loss)) for efficient one-shot learning. The general strategy is to 1) train a model to discriminate between a collection of same/different pairs; and 2) generalize to evaluate new categories based on learned feature mappings for verification. The architecture of each twin of the siamese network is a convolutional neural network, with a flattening and a feed-forward network in the head. The two twins share weights, and the prediction is a sigmoid over a weighted L1 distance between their feature vectors, $$p = \sigma(\sum_j \alpha_j d_j)$$ with $$d_j = \vert h_{1,j} - h_{2,j} \vert$$, where $$h_{1,j}$$ and $$h_{2,j}$$ are the $$j$$-th features of the two twins and the weights $$\alpha_j$$ are learned.

The loss function is a **binary cross-entropy** on the same/different label, with a regularizer. On Omniglot 20-way one-shot classification, the network reaches 92.0% accuracy, close to the 95.2% of Hierarchical Bayesian Program Learning and the 95.5% of humans. The following picture illustrates the principle on a simpler siamese network, with fully connected layers instead of convolutions:

{: .picture}
<img loading="lazy" width="51.1%" height="51.1%" src="{{ site.assets }}/publications/siamese_networks_2.svg"/>

{: .caption}
A simple 2 hidden layer siamese network for binary classification with logistic prediction <i>p</i>. The structure of the network is replicated across the top and bottom sections to form twin networks, with shared weight matrices at each layer; the output weights <i>α<sub>j</sub></i> weight the component-wise distances <i>d<sub>j</sub></i>. Adapted from: [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="bahdanau-attention">
<summary markdown="span"><span class="pub-title">Neural Machine Translation by Jointly Learning to Align and Translate (and Attention Mechanism)</span><span class="pub-meta">, D. Bahdanau, K. Cho, Y. Bengio (ICLR 2015)</span><a class="pub-link" href="https://arxiv.org/abs/1409.0473" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper presents an improvement over the RNN encoder–decoder for translation ([Cho et al., 2014](https://arxiv.org/abs/1406.1078); concurrent with [Sequence to Sequence Learning with Neural Networks (Google, NeurIPS 2014)](https://proceedings.neurips.cc/paper_files/paper/2014/file/5a18e133cbf9f257297f410bb7eca942-Paper.pdf)), which compresses the whole source sentence into a single fixed-length vector, a bottleneck for long sentences. It introduces the concept of **[attention](https://en.wikipedia.org/wiki/Attention_(machine_learning))**: at each decoding step, the decoder learns to (soft-)align to and use the latent states of every encoder step, not only the last. To generate the target word $$y_t$$, an alignment model scores each encoder state $$h_j$$ against the previous decoder state $$s_{t-1}$$, a softmax turns the scores into the attention weights $$\alpha_{t,j}$$, and the context vector $$c_t = \sum_j \alpha_{t,j} h_j$$ enters the decoder. The following picture illustrates how the decoder combines the states of a [bidirectional RNN](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks) encoder, weighted by the attention weights, to generate a target word:

{: .picture}
<img loading="lazy" width="25.2%" height="25.2%" src="{{ site.assets }}/publications/attention_mech.svg"/>

{: .caption}
The graphical illustration of the proposed model trying to generate the <i>t</i>-th target word <i>y<sub>t</sub></i> given a source sentence (<i>x</i><sub>1</sub>, <i>x</i><sub>2</sub>, . . . , <i>x<sub>T</sub></i>). Adapted from: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

On English-to-French translation, on the sentences without unknown words, the model with attention (RNNsearch-50) reaches a BLEU score of 34.16, against 26.71 for the RNN encoder–decoder, and 36.15 when trained longer, above the 35.63 of the phrase-based system Moses; it also degrades much less on long sentences. This attention is the ancestor of the attention of Transformers, whose cost [Sparse Transformers](#sparse-transformers) and [FlashAttention](#flashattention) later reduce.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="spatial-transformer-networks">
<summary markdown="span"><span class="pub-title">Spatial Transformer Networks</span><span class="pub-meta">, DeepMind (NeurIPS 2015)</span><a class="pub-link" href="https://arxiv.org/abs/1506.02025" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

[Convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNNs) are only invariant to small translations of their input, through their local max-pooling layers. The spatial transformer is a differentiable module that can be inserted anywhere in a CNN to spatially transform feature maps, conditioned on the input and learned without extra supervision. A **localisation network** regresses the transformation parameters $$\theta$$ (e.g. [affine](https://en.wikipedia.org/wiki/Affine_transformation), projective or [thin-plate spline](https://en.wikipedia.org/wiki/Thin_plate_spline)), a **grid generator** maps each point of the output grid to the point of the input to sample (with the 2 × 3 matrix of $$\theta$$ for an affine transformation), and a differentiable (e.g. [bilinear](https://en.wikipedia.org/wiki/Bilinear_interpolation)) **sampler** warps the input. This gives the network a learned invariance to translation, scale, rotation and more generic warping. The following picture illustrates how a spatial transformer crops and straightens distorted MNIST digits before their classification:

{: .picture}
<img loading="lazy" width="37%" height="37%" src="{{ site.assets }}/publications/stn.svg"/>

{: .caption}
The result of using a spatial transformer as the first layer of a fully-connected network trained for distorted MNIST digit classification. (a) The input to the spatial transformer network is an image of an MNIST digit that is distorted with random translation, scale, rotation, and clutter. (b) The localisation network of the spatial transformer predicts a transformation to apply to the input image. (c) The output of the spatial transformer, after applying the transformation. (d) The classification prediction produced by the subsequent fully-connected network on the output of the spatial transformer. The spatial transformer network (a CNN including a spatial transformer module) is trained end-to-end with only class labels – no knowledge of the groundtruth transformations is given to the system. Adapted from: [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)

The spatial transformer lowers the error of a CNN on MNIST distorted by rotation, translation and scale from 0.8% to 0.5%, and reaches state-of-the-art results at the time on Street View House Numbers (3.6% error, against 3.9%) and on the CUB-200-2011 birds (84.1% accuracy, against 81.0%). Its learned crops act as a spatial attention over the image, as the attention of [Bahdanau et al.](#bahdanau-attention) does over the words of a sentence.

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2014</span></td>
<td markdown="block">
<details id="deeply-supervised-nets">
<summary markdown="span"><span class="pub-title">Deeply-Supervised Nets</span><span class="pub-meta">, UCSD and Microsoft (AISTATS 2015)</span><a class="pub-link" href="https://arxiv.org/abs/1409.5185" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

A deeply supervised model in machine learning refers to a model architecture where intermediate layers are explicitly supervised during training, in addition to the supervision applied to the final output layer. This technique encourages better learning throughout the model by enforcing that earlier layers learn features useful for solving the task directly, rather than solely relying on gradients propagated from the final layer.

Deep supervision was introduced to address challenges such as [vanishing gradients](https://en.wikipedia.org/wiki/Vanishing_gradient_problem), poor feature learning in intermediate layers, and inefficiency in deep networks. It is particularly common in tasks like image segmentation, object detection, and biomedical image analysis.

This paper adds a classifier, trained with a squared [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss), to each hidden layer. The objective of these intermediate classifiers is called the **companion objective**, which is used as an additional constraint (or a new regularization) to the learning process: the overall objective is the loss of the output layer plus the companion losses, each weighted by a constant $$\alpha_m$$ and reduced by a threshold $$\gamma$$ (a hyperparameter): "once the overall value of the hidden layer reaches or is below $$γ$$, it vanishes and no longer plays role in the learning process". From the paper: "The empirical result suggests the following main properties of the companion objective: (1) it acts as a kind of **feature regularization** (although an unusual one), which leads to significant reduction to the testing error but not necessarily to the train error; (2) it results in faster convergence, especially in presence of small training data". It gave state-of-the-art results at the time, with test errors of 0.39% on MNIST, 9.78% on CIFAR-10 (without data augmentation), 34.57% on CIFAR-100 and 1.92% on SVHN. [Contrastive Deep Supervision](#contrastive-deep-supervision) later supervises the intermediate layers with a contrastive loss instead.

</div>
</details>
</td>
</tr>
<tr>
<td></td>
<td markdown="block">
<details id="dropout">
<summary markdown="span"><span class="pub-title">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</span><span class="pub-meta">, Univ. Toronto (Journal of ML Research 2014)</span><a class="pub-link" href="https://jmlr.org/papers/v15/srivastava14a.html" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

Large neural networks [overfit](https://en.wikipedia.org/wiki/Overfitting) easily, and combining the predictions of many separately trained networks, which reduces overfitting, is too slow with large networks. **Dropout** is a method that **randomly drops neurons (in different layers) during train time, each one with probability $$p$$** (the paper itself uses the probability of retaining a neuron instead), together with their connections. For each training mini-batch, a new "thinned" network is sampled. **At test time, all neurons are used, with outgoing weights multiplied by $$1 - p$$**, which approximates averaging the exponentially many thinned networks. Dropout can be improved by adding max-norm regularization, decaying learning rate and high momentum.

Dropout helps in **reducing overfitting**, as the network learns to never rely on any given activations ("dropout prevents co-adaptation by making the presence of other hidden units unreliable"), so it learns "redundant" ways of solving the task with multiple neurons; as a side effect, hidden activations also become sparser. Dropping 20% of input units and 50% of hidden units (i.e. $$p = 0.2$$ and $$p = 0.5$$) was often found to be optimal in the original publication. It is computationally less expensive than regular model averaging of multiple trained DNNs. However, "a dropout network typically takes 2-3 times longer to train than a standard neural network of the same architecture", because it requires many more epochs, as parameter updates are very noisy.

Because a fully connected layer occupies most of the parameters, it is prone to overfitting. Therefore, dropout **increases model generalization**: on MNIST, for example, a dropout network with max-norm regularization reaches a test error of 0.95%, against 1.60% for a standard network. [Batch Normalization](#batch-normalization) later removes the need for dropout in some cases. The following picture illustrates a standard network (left) and one of the thinned networks that dropout samples from it (right):

{: .picture}
<img loading="lazy" width="68.5%" height="68.5%" src="{{ site.assets }}/deep-neural-networks/dropout.svg"/>

{: .caption}
Dropout. Left: a standard network with two hidden layers ($$p = 0$$). Right: a thinned network produced by dropout with $$p = 0.5$$, where $$p$$ is the probability of dropping a unit: the crossed units are dropped, with their connections. Adapted from: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)

</div>
</details>
</td>
</tr>
<tr>
<td><span class="post-label">2006</span></td>
<td markdown="block">
<details id="contrastive-loss">
<summary markdown="span"><span class="pub-title">Dimensionality Reduction by Learning an Invariant Mapping (contrastive loss)</span><span class="pub-meta">, Hadsell, Chopra & LeCun, New York Uni (CVPR 2006)</span><a class="pub-link" href="http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf" target="_blank" rel="noopener" title="Open the paper"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a></summary>
<div class="pub-notes" markdown="block">

The paper presents Dimensionality Reduction by Learning an Invariant Mapping (DrLIM). The problem is to find a function that maps high dimensional input patterns to lower dimensional outputs, given neighborhood relationships between samples in input space: earlier [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) methods, such as LLE and Isomap, embed the training samples only, and "None of these methods attempt to compute a function that could map a new, unknown data point". It presents the **Contrastive Loss Function**, which trains a **siamese network**, and encourages the model to learn a representation space where similar samples are close together, and dissimilar samples are far apart.

A Siamese Network is a type of neural network architecture designed to compare two inputs by learning their similarity or relationship. It consists of two identical subnetworks (hence the name "Siamese") that share the same architecture and weights. Each subnetwork processes one of the two inputs independently, and the outputs are then combined to compute a similarity score or distance metric.

The input to the system is a pair of images (one to each of the subnetworks) and their similarity label. The images are passed through the functions, yielding two outputs $$G(X_1)$$ and $$G(X_2)$$. The cost module then computes the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between both outputs as $$D = \Vert G(X_1) - G(X_2) \Vert_2$$. The objective is formulated in terms of the similarity label $$y$$ (1 for similar, 0 for dissimilar; the paper itself uses the opposite convention) and the Euclidean distance $$D$$ between the two outputs as:

$$
L = \frac{1}{2} ⋅y⋅D^2 + \frac{1}{2} ⋅(1−y)⋅\max(0,m−D)^2
$$

where $$m$$ is a margin hyperparameter that sets the minimum distance for dissimilar pairs: "Dissimilar pairs contribute to the loss function only if their distance is within this radius". Since the two copies of the network share their parameters, the total gradient is the sum of the contributions from the two instances. In the experiments, DrLIM maps images of an airplane of the NORB dataset, under different poses and lightings, to a 3D cylinder that follows the pose of the camera and is invariant to the lighting, and it maps new samples that were not seen during training. The same idea, pulling similar pairs together and pushing dissimilar ones apart, drives the training of [CLIP](#clip).

</div>
</details>
</td>
</tr>
</table>

{::options parse_block_html="false" /}
