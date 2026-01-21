<details> <summary markdown="span">2025 [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention](https://arxiv.org/abs/2509.24006)</summary>

SLA proposes an attention design for **Diffusion Transformers (DiTs)** that aims to hit better quality–efficiency trade‑offs than “pure sparsity” or “pure linear attention” baselines. The core idea is to combine *structured sparsity* with a *linear‑attention-style* component, and make that mixture **fine‑tunable**, so you can adjust compute at inference time without changing the overall model.

Why it matters specifically for diffusion: DiTs typically run attention many times across denoising steps, so attention cost compounds. Fixed sparse patterns can harm global consistency or details; SLA’s tunable blend is intended to soften that brittleness.

Use this as a related‑work reference when you want an attention method that is **approximate** (unlike FlashAttention) and **budget‑controllable** (unlike many fixed sparse patterns) for diffusion‑model backbones.
</details>


<details> <summary markdown="span">2025 [DeepSeek‑V3.2‑Exp (model card & notes), DeepSeek‑AI](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)</summary>

DeepSeek‑V3.2‑Exp is a **model release / technical note** positioned as an iteration on DeepSeek‑V3, with the emphasis shifting from “scale + MoE efficiency” to **long‑context practicality**. Relative to V3, the advertised differences focus on **improving long‑context efficiency** (prefill/throughput for long sequences) and on better behavior in long-document settings.

Treat it as an engineering iteration on top of the V3 design: if you cite DeepSeek‑V3 for the base architecture/training recipe, cite V3.2‑Exp for the **what changed to make long context cheaper / more usable in practice** story.
</details>


<details> <summary markdown="span">2025 [DeepCompile: A Compiler-Driven Approach to Optimizing Distributed Deep Learning Training, Microsoft DeepSpeed](https://arxiv.org/abs/2504.09983)</summary>

DeepCompile pushes distributed-training optimizations into a **compiler-driven** workflow: rather than implementing sharding/communication overlap primarily in the framework runtime, it uses compiler analysis and graph-level scheduling to orchestrate distributed execution. The emphasis is on systematically improving overlap, memory management, and communication scheduling for multi‑GPU training without requiring users to rewrite model code into bespoke distributed kernels.
</details>


<details> <summary markdown="span">2025 [Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler](https://arxiv.org/abs/2504.19442)</summary>

Triton‑distributed extends Triton with primitives for writing **compute–communication overlapping kernels**. It uses NVSHMEM-style ideas (symmetric memory, signals, asynchronous tasks) so that developers can express in‑kernel communication and overlap it with compute, instead of relying only on coarse-grained NCCL collectives outside kernels.

This is best cited as an enabling substrate for “fused distributed kernels” rather than a single-workload optimization.
</details>


<details> <summary markdown="span">2025 [TileLink: Generating Efficient Compute–Communication Overlapping Kernels Using Tile-Centric Primitives](https://arxiv.org/abs/2503.20313)</summary>

TileLink generates compute–communication overlapping kernels using **tile-centric** compiler primitives. By structuring tensors into 2D tiles and scheduling at tile granularity, it becomes easier to overlap communication with ongoing computation while giving the compiler more regular structure to optimize.

The paper reports end‑to‑end improvements over strong baselines and highlights substantial gains for MoE overlap kernels, positioning TileLink as a higher-level, more general way to express Flux‑style overlap.
</details>


<details> <summary markdown="span">2025 [MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production](https://arxiv.org/abs/2505.11432)</summary>

MegaScale‑MoE is a production-oriented system for **communication-efficient MoE training**. It targets MoE bottlenecks like dispatch/combination and expert-parallel traffic, and proposes a stack of techniques (parallelism strategy + communication optimizations + scheduling) aimed at improving end‑to‑end throughput at scale in production settings.
</details>


<details> <summary markdown="span">2025 [MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism](https://arxiv.org/abs/2504.02263)</summary>

MegaScale‑Infer focuses on **serving** MoE models using **disaggregated expert parallelism**, separating attention and expert computation across node groups to scale experts independently and improve utilization. It is best cited when discussing MoE serving constraints (latency/throughput, routing skew, many‑to‑many communication) and system-level pipeline design.
</details>


<details> <summary markdown="span">2025 [Accelerating MoE Model Inference with Expert Sharding](https://proceedings.mlsys.org/paper_files/paper/2025/hash/4ebbaae5e3a4e2ff8244ea69bbbd5210-Abstract-Conference.html)</summary>

This work studies MoE inference where experts are **sharded across devices** to fit larger experts and reduce per-device memory pressure. Instead of replicating experts or requiring a monolithic expert-parallel deployment, it explores partitioning experts and scheduling inference so that routing, communication, and expert compute remain efficient.

It is most relevant for the “how do we serve MoEs when experts don’t fit cleanly on a device” question, and complements disaggregated-serving systems (which separate attention vs expert nodes) by focusing on **within-expert partitioning**.
</details>


<details> <summary markdown="span">2025 [Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)</summary>

This work explores an extreme “megakernel” direction for inference: fusing large portions of the **Llama‑1B forward pass** into a single (or very small number of) kernels to minimize launch overhead and reduce end‑to‑end latency. It combines heavy fusion with careful scheduling/synchronization inside the kernel to keep execution smooth and avoid idle “bubbles”.

Use this as a related‑work reference for the *hand‑engineered end‑to‑end fusion* approach, complementary to compiler/superoptimizer efforts that aim to automate deep fusion.
</details>


<details> <summary markdown="span">2025 [Mirage: A {Multi-Level} Superoptimizer for Tensor Programs (OSDI 2025)](https://arxiv.org/abs/2506.11295)</summary>

Mirage is a superoptimizer for tensor programs that searches for deeply fused implementations across multiple abstraction levels. It is relevant as an example of **automatic deep fusion**: rather than hand-designing a single fused kernel (e.g., an attention kernel), Mirage aims to automatically discover profitable fusion opportunities and generate optimized code.
</details>


<details> <summary markdown="span">2025 [SageAttention3: Microscaling FP8 Attention for Inference](https://arxiv.org/abs/2505.11594)</summary>

SageAttention3 targets attention efficiency using **FP8 attention** with a “microscaling” approach designed to keep attention numerically stable. Relative to earlier SageAttention work, it is most relevant if your stack is moving toward **end‑to‑end FP8** inference/training on newer GPUs and you want an attention method that is explicitly FP8‑friendly.
</details>


<details> <summary markdown="span">2024 [SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](https://arxiv.org/abs/2411.10958)</summary>

SageAttention2 extends the original SageAttention approach with **outlier smoothing** and **per‑thread INT4 quantization** for attention. The practical goal is to go below 8‑bit precision (INT4) while keeping attention accuracy acceptable, by mitigating the outlier-driven error that often breaks aggressive quantization.
</details>


<details> <summary markdown="span">2024 [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367)</summary>

SageAttention is a plug‑and‑play attention acceleration method that targets **accurate 8‑bit attention** for inference. It is best read as “attention‑specific quantization done carefully enough to be a drop‑in speedup,” contrasting with exact IO‑optimized kernels (FlashAttention) and with approximate attention variants (sparse/linear attention).
</details>


<details> <summary markdown="span">2024 [Centauri: Enabling Efficient Scheduling for Communication–Computation Overlap in Large Model Training via Communication Partitioning](https://doi.org/10.1145/3620666.3651379)</summary>

Centauri addresses communication/computation overlap in large-model training via **communication partitioning** and scheduling. It is best read as a runtime/scheduling approach: partition collective communications into smaller chunks aligned with computation so that communication can start earlier and overlap more effectively, reducing pipeline bubbles without requiring model-code changes.
</details>


<details> <summary markdown="span">2024 [FLUX: Fast Software-based Communication Overlap on GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858)</summary>

Flux achieves fast software-based communication overlap on GPUs by **kernel fusion** for Megatron‑LM-style tensor-parallel operations. It decomposes tensor-parallel ops into fine-grained pieces and fuses them into kernels that overlap communication with computation to hide collective overhead.

The paper reports high overlap (up to ~96% communication hidden) and speedups up to **1.66×** for prefill and **1.3×** for decoding on representative LLM inference workloads, making it a concrete quantitative reference for overlap in tensor-parallel stacks.
</details>


<details> <summary markdown="span">2024 [Optimizing Distributed ML Communication with Fused Computation–Collective Operations](https://arxiv.org/abs/2305.06942)</summary>

Punniyamurthy et al. identify and implement **fused computation–collective** patterns (e.g., GEMM/GEMV fused with all‑reduce/all‑to‑all), showing that bringing communication into the compute kernel can reduce launch overhead and improve overlap. It is a useful reference when motivating in‑kernel collectives as a reusable optimization pattern beyond a single LLM stack.
</details>


<details> <summary markdown="span">2024 [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)</summary>

DeepSeek‑V3 is a MoE LLM technical report describing the model architecture, training pipeline, and efficiency strategies used to reach strong quality at scale. It is mainly relevant here as a modern “MoE at scale” reference that connects architectural choices to the distributed systems work required for routing, dispatch/combines, and stable training.
</details>


<details> <summary markdown="span">2024 [The Llama 3 Herd of Models, Meta](https://arxiv.org/abs/2407.21783)</summary>

The Llama 3 technical report documents Meta’s dense model family and the training/post‑training pipeline. It is often cited for its scale and engineering choices (data mixture, architecture details, post‑training/alignment, and long-context settings) and serves as a canonical reference for how large dense LLMs are trained and tuned.
</details>


<details> <summary markdown="span">2024 [Universal Checkpointing: Efficient and Flexible Checkpointing for Large Scale Distributed Training, Microsoft DeepSpeed](https://arxiv.org/abs/2406.18820)</summary>

Universal Checkpointing proposes a checkpoint format and workflow that decouples saved state from a specific parallelism configuration, enabling resuming training under different numbers of ranks or different parallelism decompositions. This is particularly relevant for elastic training, fault recovery, and moving runs across clusters with different GPU counts.
</details>


<details> <summary markdown="span">2024 [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping, Microsoft DeepSpeed](https://arxiv.org/abs/2409.15241)</summary>

Domino targets distributed LLM training efficiency by reducing communication overhead using a combination of **generic tensor slicing** and **overlap-friendly scheduling**. The paper frames this as a way to systematically restructure tensors and communication so that collective costs are reduced and more overlap is achieved in practice.

It reports speedups on modern GPU systems (e.g., DGX‑H100-class nodes) relative to strong baselines, making it a useful related‑work reference for “end‑to‑end distributed training efficiency” that is more general than single-operator kernel fusion.
</details>


<details> <summary markdown="span">2023 [Simplifying Transformer Blocks, ETH Zurich](https://arxiv.org/abs/2311.01906)</summary>

This paper simplifies Transformer blocks by exploiting redundant components across layers and tightening the compute path, with the goal of **improving inference efficiency** without sacrificing accuracy. It reports throughput improvements and parameter reductions while maintaining model quality, making it a good reference for “architectural simplification for efficiency” (distinct from kernel-level or communication-level optimizations).
</details>


<details> <summary markdown="span">2025 [Titans: Learning to Memorize at Test Time, Google Research](https://arxiv.org/abs/2501.00663)</summary>

Titans introduces mechanisms for **test‑time memory**, enabling models to store and retrieve information during inference rather than relying only on a fixed context window. It is best cited for the “learned memory at test time” direction, which is conceptually different from long-context *efficiency* work (making attention cheaper) even though both address long-horizon use cases.
</details>


<details> <summary markdown="span">2022 [Self-Attention Does Not Need \(O(n^2)\) Memory](https://arxiv.org/abs/2112.05682)</summary>

Rabe & Staats show that attention can be computed with sub‑quadratic memory using streaming formulations that avoid materializing the full attention matrix. This is a conceptual precursor to later streaming/flash attention implementations.
</details>


<details> <summary markdown="span">2022 [DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale (SC 2022)](https://arxiv.org/abs/2207.00032)</summary>

DeepSpeed‑Inference is a systems paper focused on scaling Transformer inference via kernel optimizations, parallelism orchestration, and memory/communication optimizations. It is often cited as a reference for production-grade inference at scale and how multiple optimizations (tensor/pipeline parallelism, kernel fusion, quantization support) interact end-to-end.
</details>


<details> <summary markdown="span">2022 [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale (ICML 2022)](https://arxiv.org/abs/2201.05596)</summary>

DeepSpeed‑MoE presents MoE training and inference optimizations in DeepSpeed, including expert parallelism and efficient token dispatch/combines. It is best cited as a practical systems implementation reference that complements MoE algorithm papers (e.g., GShard/Switch).
</details>
