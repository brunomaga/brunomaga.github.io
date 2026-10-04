---
layout: post
title:  "GPT model inference optimization: caching, batching, fast attention, speculative decoding, quantization, pruning and distillation"
categories: [machine learning, Transformer, GPT, inference, quantization, pruning, distillation]
tags: [machinelearning]
---

Previously, in [Distributed training of a large GPT model with DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPTlite-data-parallelism %}), we focused on training a very large model on a distributed network of GPUs. The aim was to reduce training runtime via increased parallelism, and to increase model accuracy by increasing model size. In this post, we look at the other end of the model lifecycle: **inference**. The goal is to generate text as fast and as cheaply as possible, with a small memory footprint, and with little or no loss of quality. This matters for real-time and embedded systems, and for any service where the cost per generated token adds up.

Just like in the previous posts, we use the **GPTlite model**, a small variant of the [GPT-2 model](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) that generates text by predicting the next character in a sequence. Each section explains one idea, points to the code that implements it on top of GPTlite, and discusses when it helps. GPTlite is tiny, so some techniques designed for billion-parameter models give small or no speedups at our scale; when that happens, we explain why.

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="/assets/GPTlite/gpt_lite_compact.png"/>

The post is organized as follows:

1. **Background**: how a GPT generates text, and where the time goes.
2. **Caching**: the KV cache, prefix caching and semantic caching.
3. **Batching and scheduling**: continuous batching, PagedAttention, chunked prefill, disaggregation and parallelism.
4. **Faster attention**: FlashAttention, smaller KV caches, and quantized, sparse and linear attention.
5. **Speculative decoding**: generating several tokens per forward pass.
6. **Quantization**: fewer bits per weight, activation and cached value.
7. **Distillation and pruning**: smaller models that behave like larger ones.
8. **Compilers and kernels**: kernel fusion, `torch.compile` and CUDA graphs.
9. **Architectures built for fast inference**: mixture of experts, hybrid and diffusion models.

## Background: how a GPT generates text, and where the time goes

### Prefill and decode

A GPT model generates text *autoregressively*: it reads the prompt, predicts the next token, appends it to the input, and repeats. Inference therefore has two phases that behave very differently:

- **Prefill** processes all the prompt tokens in a single forward pass. The tokens are processed in parallel, so the GPU performs large matrix-matrix multiplications and is **compute-bound**. Prefill ends with the first generated token, so its duration is the **time to first token (TTFT)**.
- **Decode** generates the remaining tokens, one forward pass per token. Each pass processes a single new token per sequence, so the GPU performs matrix-vector multiplications: it reads every weight of the model from memory to do very little arithmetic with it. Decode is **memory-bandwidth-bound**, and the duration of each step is the **time per output token (TPOT)**, also called inter-token latency.

A quick calculation shows why decode is memory-bound. In a matrix-vector product, each weight (2 bytes in a 16-bit format) is used for one multiplication and one addition per sequence in the batch. With a batch of $$B$$ sequences, the GPU performs about $$B$$ floating point operations (FLOPs) per byte read from memory. An NVIDIA H100 delivers about 1,000 TFLOP/s of 16-bit compute and 3.35 TB/s of memory bandwidth, so it needs about 300 FLOPs per byte read to keep its compute units busy. With any batch smaller than a few hundred sequences, the GPU spends most of the decode time waiting for memory.

This observation explains most of this post. Almost every decode optimization does one of three things: it **reads fewer bytes** per token (quantization, smaller KV caches, smaller models), it **does more useful work per byte read** (batching, speculative decoding), or it **removes overheads** around the math (kernel fusion, CUDA graphs, better scheduling).

Throughout the post, we look at:

- **TTFT** and **TPOT**, the latencies a user perceives;
- **throughput**, the total number of tokens generated per second across all requests;
- **peak memory**;
- **quality**, measured as the validation loss (or perplexity) of the model.

Latency and throughput pull in opposite directions: larger batches increase throughput, but make each step slower, so every user waits a little longer for each token.

### Where the compute goes: MLP versus attention

For a model of width $$d$$ (the embedding size) and a context of $$n$$ tokens, each layer spends, for every token:

- about $$8d^2$$ FLOPs in the query, key, value and output projections of the attention block;
- about $$16d^2$$ FLOPs in the MLP (the feed-forward network), whose hidden layer has size $$4d$$;
- about $$4nd$$ FLOPs computing the attention scores $$QK^T$$ and the weighted sum of the values.

The first two terms do not depend on the context length; the third grows with every token in the context. Attention becomes the dominant cost when $$4nd > 24d^2$$, i.e., when the context is longer than about $$6d$$ tokens. For a typical 8B-parameter model ($$d = 4096$$), that is around 25,000 tokens; smaller models reach this point sooner. For short prompts the MLP dominates; for long documents, long conversations and long reasoning traces, attention does.

Memory follows the same pattern. The model weights are read once per decode step and shared by all the sequences of the batch, while each sequence keeps its own cache of attention keys and values (next section) that grows with every token. For long contexts and large batches, reading these caches dominates the memory traffic of each decode step.

## Caching

There are three kinds of inference caching, each operating at a different layer of the stack (this three-layer view is adapted from [this guide](https://machinelearningmastery.com/the-complete-guide-to-inference-caching-in-llms/)):

- **KV caching** stores the attention keys and values computed during a single request, so that the model does not recompute them at every decode step. Every serving engine uses it.
- **Prefix caching** extends KV caching across requests. When different requests share the same leading tokens, such as a system prompt, a reference document, few-shot examples or the previous turns of a conversation, the keys and values of that shared prefix are computed once and reused. It is also called prompt caching or context caching.
- **Semantic caching** is an application-level cache that stores complete input/output pairs and retrieves them by meaning. Unlike prefix caching, which reuses internal attention states, semantic caching skips the model call entirely when a sufficiently similar query was answered before.

These are complementary layers, not alternatives. KV caching is always on, prefix caching is the highest-leverage optimization for most production applications, and semantic caching pays off when many queries are similar. In short: the KV cache reuses work within one generation, prefix caching reuses work across requests, and semantic caching reuses whole answers.

### KV cache

In the attention layer, the query of the new token attends to the keys and values of all previous tokens. In a causal model, the keys and values of past tokens never change, since token $$i$$ only depends on the tokens before it. Without a cache, each decode step recomputes the keys and values of the whole sequence, so generating $$n$$ tokens costs $$O(n^2)$$ work in the projections and MLPs and $$O(n^3)$$ in attention. With a cache, each step only computes the query, key and value of the new token, appends the new key and value to the cache, and attends over all cached ones: $$O(n)$$ and $$O(n^2)$$ in total. Queries do not need to be cached, because the query of a past token was only needed to compute the output at that past position.

The price is memory. For every token, every layer stores a key and a value vector per key-value head:

$$
\text{KV cache size} = 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times \text{bytes per value} \times \text{sequence length} \times \text{batch size}
$$

For example, Llama-3-8B (32 layers, 8 key-value heads of size 128, 16-bit values) needs 128 KB per token, or 1 GB for a single sequence of 8K tokens. This memory, not compute, often limits how many sequences can be batched together, which is why many techniques in this post shrink the KV cache.

Two implementation details matter in practice:

- A **dynamic cache** grows by concatenating the new keys and values to the cache at every step. This is simple, but every concatenation allocates a new tensor and copies the whole cache. A **static cache** preallocates a buffer for the maximum sequence length and writes each new entry in place. Static shapes also allow CUDA graphs to remove launch overheads (see the compilers section).
- GPTlite uses **learned absolute positional embeddings** and a context window of `seqlen` tokens. When a sequence grows longer than `seqlen`, the model without a cache slides its window and re-encodes the last `seqlen` tokens at positions 0 to `seqlen-1`, at every step. A KV cache cannot reproduce this, because the cached keys and values were computed at their original positions and with the context available back then. The model with a cache therefore matches the original model exactly for the first `seqlen` tokens only. Models with relative positional encodings, such as RoPE, handle a sliding window more gracefully, but keeping only the most recent tokens still changes the output (see sparse attention below).

The following code implements a KV cache in GPTlite:

```python

```

### Prefix caching

Many requests start with the same tokens: a system prompt, tool definitions, a long document that users ask several questions about, or the history of a conversation. Prefix caching keeps the keys and values of these tokens after a request finishes, so that the next request starting with the same tokens skips their prefill and only processes its new part. The first token arrives much sooner, and the GPU is free for other work. Hosted LLM APIs offer this as *prompt caching* or *context caching*, and bill cached input tokens at a discount.

**Why the prefix must match exactly.** It is tempting to look for *similar* prompts, e.g., with a vector database, instead of identical ones. This does not work for attention states. The key and value of token $$i$$ depend on all the tokens before it, through every layer of the model. Two prompts that differ in a single early token, even if they mean the same, have different keys and values for every token after that difference. Reusing the cache of a similar prompt would silently feed the model a different context than the one the user sent. Exact matching makes prefix caching *lossless*: the output is identical to running the model without the cache. It is also cheap: a hash or tree lookup, instead of computing an embedding and searching a vector index. Similarity search does make sense one level up, where whole answers are reused: that is semantic caching, below.

**Prefixes, not substrings.** For the same reason, only a prefix can be reused, i.e., the first $$N$$ tokens of the prompt. A paragraph in the middle of a prompt has different keys and values depending on what comes before it. Matching is done on token IDs, not on raw text, so the same text tokenized differently (e.g., with an extra space) is a miss. Research systems reuse non-prefix chunks approximately: [CacheBlend](https://arxiv.org/abs/2405.16444) precomputes the cache of each retrieved document independently and recomputes a small fraction of tokens to restore the attention between documents, and [Prompt Cache](https://arxiv.org/abs/2311.04934) reuses precomputed modules of prompts written in a structured template.

**How the lookup works.** [vLLM](https://arxiv.org/abs/2309.06180) splits the cache into fixed-size blocks (e.g., 16 tokens) and identifies each full block by a hash of its tokens combined with the hash of the previous block. The chain of hashes identifies the whole prefix, so a lookup walks the blocks from the start of the prompt and stops at the first miss. [SGLang](https://arxiv.org/abs/2312.07104)'s RadixAttention stores all cached sequences in a radix tree (a compressed prefix tree), finds the longest prefix shared with each new request, and evicts the least recently used leaves when memory runs out.

**What is kept.** The KV cache holds the keys and values of every token processed so far, in every layer: the prompt and all the generated tokens, not only the last decode step, since every new token attends to all of them. When a request finishes, its blocks are not freed right away: they stay in memory, marked as reusable, until the memory is needed for something else. Generated tokens are cached too, which is what makes multi-turn chat fast: the prompt of the next turn is the previous prompt, plus the previous answer, plus the new message.

**Practical consequences.** Put the static content first (system prompt, tools, documents) and the variable content last (the user's question, timestamps): anything that changes early in the prompt invalidates the cache for everything after it. At scale, engines offload cached blocks to CPU memory, SSDs or a shared store to keep more prefixes available (e.g., [LMCache](https://github.com/LMCache/LMCache) and [Mooncake](https://arxiv.org/abs/2407.00079)), and *cache-aware routers* send each request to the replica that already holds its prefix.

The following code implements a simple prefix cache on top of the KV cache, where the keys and values of a shared prompt are computed once and copied into every new request:

```python

```

### Semantic caching

A semantic cache sits in front of the model, in the application. For every incoming query, it:

1. computes an *embedding*, a vector that represents the meaning of the query;
2. searches a vector index for the most similar past query;
3. if the similarity is above a threshold, returns the stored answer without calling the model;
4. otherwise, calls the model and stores the new (embedding, answer) pair.

**How embeddings are created.** Embeddings come from a separate and much smaller *embedding model*, typically a transformer encoder such as the [Sentence-BERT](https://arxiv.org/abs/1908.10084) family (e.g., all-MiniLM-L6-v2), E5, BGE or GTE, or from an embedding API. The text is tokenized and processed by the encoder, whose attention is bidirectional: every token sees the whole text. The resulting token vectors are pooled into a single vector, by averaging them (mean pooling) or by taking the vector of a special first token, and normalized to unit length. Typical embeddings have between 384 and a few thousand dimensions. The encoder is trained with *contrastive learning*: pairs of texts with the same meaning (paraphrases, or a question and its answer) are pulled together, and the other texts of the batch are pushed apart. After training, the cosine similarity between two embeddings measures how close their meanings are. The raw hidden states of a generative LLM make poor embeddings unless the model is fine-tuned in the same contrastive way, which is how some of the best recent embedding models are built.

**Where embeddings are stored.** Any vector index works. [FAISS](https://github.com/facebookresearch/faiss) is a popular choice: an in-process library, for CPUs and GPUs, with exact indexes (e.g., `IndexFlatIP`, an inner product, which equals the cosine similarity for normalized vectors) and approximate ones for millions of vectors (IVF, HNSW, product quantization). Since FAISS is a library and not a database, persistence, metadata, deletions and expiration are up to us; vector databases such as Milvus, Qdrant, Weaviate, pgvector or Redis provide them. Tools such as [GPTCache](https://github.com/zilliztech/GPTCache) implement the whole loop and support FAISS as a backend.

**Risks.** A semantic cache is approximate, so it can return a wrong answer: "how do I delete a file in Python?" and "how do I delete a folder in Python?" are very similar sentences with different answers. Answers also go stale, and queries that depend on the conversation history or on the user should not share answers. In practice, we tune the threshold on real traffic, scope the cache per user or tenant, add an expiration time, and optionally confirm hits with a more precise re-ranking model. The strict version of a semantic cache is the **exact-match cache**, which maps a hash of the full request (prompt, model and sampling parameters) to the response. It never returns a wrong answer, but only hits identical requests.

GPTlite is a character-level model trained on Shakespeare, so semantic caching is not meaningful for it. The following code implements a minimal, model-agnostic semantic cache with FAISS:

```python

```

### Is RAG an inference optimization?

Not really. Retrieval-augmented generation ([RAG](https://arxiv.org/abs/2005.11401)) uses the same embedding and vector search machinery, but for another purpose: it retrieves documents and adds them to the prompt, so that the model answers with up-to-date or private knowledge. This makes the prompts longer, so each request usually does *more* prefill work, not less. Retrieval does speed up inference in a few specific ways:

- semantic caching is retrieval over past answers;
- retrieval can propose the draft tokens of speculative decoding, as in [REST](https://arxiv.org/abs/2311.08252) (see the speculative decoding section);
- knowledge stored in a retrieval index does not have to be stored in the model's parameters, so a smaller model can do the job: [RETRO](https://arxiv.org/abs/2112.04426) performed comparably to GPT-3 on the Pile with 25 times fewer parameters;
- the keys and values of frequently retrieved documents can be precomputed and reused, as in CacheBlend.

## Batching and scheduling

Batching is the most effective way to increase throughput. Because decode is memory-bound, a decode step for a batch of 32 sequences takes almost the same time as for a single sequence: the weights are read once and used 32 times. Throughput grows almost linearly with the batch size, until the GPU becomes compute-bound or runs out of memory for the KV caches. The difficulty is that requests arrive at different times and generate different numbers of tokens.

### Static batching

The simplest approach groups $$B$$ requests and runs them together until all of them finish. Sequences that finish early keep their slot and generate tokens that are thrown away, and new requests wait until the whole batch is done. Prompts of different lengths are padded to the same length, and a padding mask prevents the tokens from attending to the padding.

The following code implements static batching:

```python

```

### Continuous batching

Continuous batching, introduced by [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) as *iteration-level scheduling*, takes the scheduling decision at every decode step instead of once per batch: as soon as a sequence finishes, it leaves the batch, and a waiting request takes its slot. The GPU stays busy, and new requests do not wait for the longest sequence of the previous batch.

The implementation must track each sequence separately: where it starts in its slot, its own positions, its own KV cache, and a mask that prevents it from attending to the tokens of the previous occupant of the slot. A new request also needs a prefill while the other sequences are decoding, which is the subject of chunked prefill, below. All modern serving engines, such as vLLM, SGLang, TensorRT-LLM and DeepSpeed-FastGen, use continuous batching.

The following code implements continuous batching in GPTlite, with per-slot positions and padding masks:

```python

```

### PagedAttention: memory management for the KV cache

Continuous batching makes memory management hard, because the final length of each sequence is unknown. A naive engine reserves a contiguous KV buffer of the maximum length for every request, and most of that memory stays empty: the authors of [vLLM](https://arxiv.org/abs/2309.06180) measured that only 20% to 40% of the KV memory reserved by existing systems held actual tokens. PagedAttention borrows the idea of virtual memory from operating systems. The KV cache is split into fixed-size *blocks* (pages) that are allocated on demand, and a *block table* maps the logical blocks of each sequence to physical blocks anywhere in GPU memory. The attention kernel reads the blocks through this table. Almost no memory is wasted, so more sequences fit in a batch: vLLM reported 2-4 times higher throughput than previous systems. Blocks can also be shared by several sequences and copied only when one of them writes to it (copy-on-write), which is how prefix caching, parallel sampling and beam search share their common prefix.

### Chunked prefill

A long prompt can take hundreds of milliseconds to prefill. If the engine processes it in one go, all the other sequences of the batch stop decoding meanwhile, and their users see the text stall. [Sarathi-Serve](https://arxiv.org/abs/2403.02310) splits long prefills into chunks, and runs each chunk together with the decode tokens of the other sequences, under a fixed budget of tokens per step. Decode-only steps leave most of the GPU's compute unused, and the prefill chunks fill that gap, so the time per output token stays stable and the utilization goes up.

### Prefill-decode disaggregation

Prefill is compute-bound and decode is memory-bound, so they interfere when they share GPUs, and each would prefer a different batch size and parallelism. Disaggregated serving runs them on separate pools of GPUs: prefill workers compute the KV cache of each prompt and send it over a fast interconnect to decode workers, which generate the answer. Each pool is sized and tuned for its own latency target: TTFT for prefill, TPOT for decode. This is the design of [DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677) and [Mooncake](https://arxiv.org/abs/2407.00079), the serving platform of Moonshot AI's Kimi, and it is supported by open-source stacks such as NVIDIA Dynamo and llm-d.

### Scaling out: parallelism and offloading

When a model does not fit in a single GPU, or is too slow on one, the work is split across GPUs:

- **Tensor parallelism** ([Megatron-LM](https://arxiv.org/abs/1909.08053)) splits every weight matrix across GPUs. Each GPU reads only its share of the weights, which reduces the latency of each decode step, at the cost of an all-reduce per layer that requires a fast interconnect such as NVLink.
- **Pipeline parallelism** places different layers on different GPUs. It increases the throughput of very large models, but does not reduce the latency of a single request.
- **Expert parallelism** places the experts of a mixture-of-experts model (see the architectures section) on different GPUs, and routes the tokens to them with all-to-all communication.
- **Sequence (or context) parallelism**, e.g., [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) or [Ring Attention](https://arxiv.org/abs/2310.01889), splits a very long prompt across GPUs to speed up its prefill.
- **Offloading**, e.g., [ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html), keeps the weights in CPU memory or on NVMe drives and streams them to the GPU layer by layer. It runs models that do not fit in GPU memory, with good throughput on large batches but high latency.

### Other serving techniques

- **Multi-LoRA serving** (e.g., S-LoRA and Punica) serves hundreds of fine-tuned LoRA adapters on top of a single base model, and batches requests for different adapters together with specialized kernels.
- **Structured (constrained) decoding** (e.g., XGrammar) forces the output to follow a grammar or a JSON schema, by masking the invalid tokens at every step. When the grammar allows a single continuation, some engines append the forced tokens without calling the model, as in SGLang's *jump-forward* decoding.

## Faster attention

### Why attention matters

Attention is the only part of a transformer whose cost grows with the context length:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_\text{head}}} \right) V
$$

During prefill, $$QK^T$$ compares every token with every previous token, so the compute grows quadratically with the prompt length, and a naive implementation stores an $$n \times n$$ matrix of scores per head. During decode, every new token reads the keys and values of the whole context, so the cost of each step grows linearly with the context. As shown in the background section, attention dominates once the context is longer than about six times the model width: the authors of Native Sparse Attention (below) estimated that attention accounts for 70-80% of the latency when decoding with a 64K-token context. Long documents, long conversations, agents and reasoning models with long chains of thought all live in this regime.

There are three families of solutions, which can be combined:

- **compute exact attention faster**, with better kernels (FlashAttention, FlashDecoding) and lower precision (SageAttention);
- **store fewer keys and values**, by sharing or compressing them (MQA, GQA and MLA);
- **compute less attention**, by attending only to the most relevant tokens (sparse attention), or by replacing softmax attention with a linear-time alternative (linear attention and hybrid models).

### FlashAttention

GPUs have a small amount of very fast on-chip memory (SRAM, around 200 KB per streaming multiprocessor) and a large amount of slower high-bandwidth memory (HBM, tens of GB). A standard attention implementation writes the $$n \times n$$ score matrix $$S = QK^T$$ to HBM, reads it back to compute the softmax $$P$$, writes $$P$$, and reads it again to compute $$PV$$. These memory round-trips, not the math, dominate its runtime.

[FlashAttention](https://arxiv.org/abs/2205.14135) computes exactly the same result without ever writing $$S$$ or $$P$$ to HBM. It splits $$Q$$, $$K$$ and $$V$$ into tiles that fit in SRAM and, for each tile of queries, loops over the tiles of keys and values while accumulating the output on chip. The challenge is the softmax, which needs the maximum and the sum over *all* the keys of a row before it can normalize. FlashAttention uses an **online softmax**: it keeps, for each query, a running maximum $$m$$, a running sum $$\ell$$ and an unnormalized output $$O$$, and corrects them whenever a new tile of scores $$s_j$$ and values $$v_j$$ arrives:

$$
\begin{align*}
m' & = \max\left(m, \max_j s_j\right) \\
\ell' & = e^{m - m'} \, \ell + \sum_j e^{s_j - m'} \\
O' & = e^{m - m'} \, O + \sum_j e^{s_j - m'} \, v_j
\end{align*}
$$

After the last tile, the output is $$O / \ell$$. The memory used by attention drops from $$O(n^2)$$ to $$O(n)$$, and the traffic to HBM drops by a large factor. With a causal mask, the tiles that are entirely above the diagonal are skipped, which saves about half of the work. During training, the backward pass recomputes $$S$$ and $$P$$ tile by tile instead of storing them.

Each new version of FlashAttention adapted the algorithm to newer hardware:

- [FlashAttention-2](https://arxiv.org/abs/2307.08691) improved the parallelism and the partitioning of work across the GPU, and reduced the non-matmul operations, roughly doubling the speed of the first version.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) targets NVIDIA Hopper GPUs (H100). It overlaps data movement, matrix multiplications and softmax using Hopper's asynchronous hardware, and supports FP8.
- [FlashAttention-4](https://arxiv.org/abs/2603.05451) (2026) targets NVIDIA Blackwell GPUs (B200), whose tensor cores became much faster while the units computing exponentials and the shared memory did not. It emulates part of the exponentials in software, skips most of the rescaling steps of the online softmax, and keeps intermediate results in Blackwell's new tensor memory. On a B200, it reaches up to 1,613 TFLOP/s (71% utilization), up to 1.3 times faster than cuDNN 9.13 and 2.7 times faster than Triton.

In PyTorch, `torch.nn.functional.scaled_dot_product_attention` (SDPA) dispatches to a FlashAttention or memory-efficient kernel when the inputs allow it, and FlexAttention generates fused kernels for custom masks. For GPTlite, using it is a one-line change.

The following code replaces the attention of GPTlite with PyTorch's fused scaled dot product attention:

```python

```

### FlashDecoding

FlashAttention parallelizes over the batch, the heads, and blocks of queries. During decode there is a single query per sequence, so with a small batch most of the GPU sits idle while a few thread blocks walk through a long KV cache. [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) also splits the keys and values along the sequence: the chunks are processed in parallel, each producing a partial output and its softmax statistics (maximum and sum), and a final reduction combines them with the same rescaling as the online softmax. This makes decoding with very long contexts up to 8 times faster. Kernel libraries such as [FlashInfer](https://github.com/flashinfer-ai/flashinfer) implement this split-KV strategy, together with attention over paged KV caches.

### Smaller KV caches: MQA, GQA and MLA

Decode reads the whole KV cache at every step, so a smaller cache means faster decoding, and room for larger batches and longer contexts.

**Multi-Query Attention** ([MQA](https://arxiv.org/abs/1911.02150)) keeps one query per head, but all heads share a single key head and a single value head. The KV cache shrinks by a factor equal to the number of heads, at some cost in quality.

**Grouped-Query Attention** ([GQA](https://arxiv.org/abs/2305.13245)) is the middle ground, used by most current models: the heads are split into $$G$$ groups, and the heads of a group share one key head and one value head. The cache shrinks by a factor $$n_\text{heads} / G$$. GQA becomes multi-head attention when $$G = n_\text{heads}$$, and MQA when $$G = 1$$. A trained multi-head model can be converted to GQA by averaging (mean pooling) the key and value heads of each group, followed by a short *uptraining* with about 5% of the original pre-training compute.

These savings only appear when there is a KV cache to read during decode. Without a cache, GQA only saves a little compute in the key and value projections, so it must be combined with the KV cache to show its benefit.

The following code implements GQA with a KV cache that stores only the grouped keys and values:

```python

```

The following code converts a trained multi-head GPTlite into a GQA model, by mean pooling its key and value heads:

```python

```

**Multi-head Latent Attention** (MLA), introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434), compresses the key and value of each token into a single small *latent* vector, and caches only that vector. The keys and values of each head are recovered with up-projection matrices, and these matrices can be merged ("absorbed") into the query and output projections, so that attention runs directly on the cached latent vectors. Rotary positional embeddings (RoPE) prevent this merge, so MLA adds a small separate key component that carries the position. DeepSeek-V2 reduced the KV cache by 93.3% compared with DeepSeek 67B, and increased its maximum generation throughput 5.76 times. GPTlite adds learned positional embeddings to its input instead of using RoPE, so the absorption works without the extra positional component.

The following code implements MLA in GPTlite:

```python

```

### Quantized attention: SageAttention

Attention can also run in lower precision. The [SageAttention](https://github.com/thu-ml/SageAttention) family, from Tsinghua University, quantizes the inputs of the two matrix multiplications of attention, in a plug-and-play way that needs no retraining:

- [SageAttention](https://arxiv.org/abs/2410.02367) computes $$QK^T$$ in INT8 and $$PV$$ in 16 bits. In some channels, all keys share a large common value that would waste the INT8 range, so it first *smooths* $$K$$ by subtracting its mean over the tokens. This does not change the result: subtracting the same vector $$\bar{k}$$ from all keys subtracts the same value $$q \cdot \bar{k}$$ from every score of a row, and the softmax is not affected by that.
- [SageAttention2](https://arxiv.org/abs/2411.10958) quantizes $$Q$$ and $$K$$ to INT4 with fine-grained (per-thread) scales, and computes $$PV$$ in FP8.
- [SageAttention3](https://arxiv.org/abs/2505.11594) uses the FP4 tensor cores of NVIDIA Blackwell GPUs with micro-scaling (see the quantization section), and reaches 1,038 TOPS on an RTX 5090, about 5 times faster than the fastest FlashAttention on that GPU.

FlashAttention-3 also has an FP8 mode. The quantization section explains the number formats and scaling tricks these methods rely on.

### Sparse attention

In practice, each query puts most of its attention weight on a small subset of the tokens. Sparse attention computes attention only over the tokens that matter. The difficulty is to find them cheaply, and to read them from memory efficiently: GPUs are fast on contiguous blocks of memory, and slow on scattered individual tokens.

**Fixed patterns.** The simplest patterns are known in advance. In *sliding window* attention, each token attends only to the last $$w$$ tokens, which bounds the size of the KV cache; it is used, for example, in [Mistral 7B](https://arxiv.org/abs/2310.06825), and stacking layers still lets information travel further than $$w$$ tokens. [Longformer](https://arxiv.org/abs/2004.05150) and [BigBird](https://arxiv.org/abs/2007.14062) combine local windows with a few global and random tokens. [StreamingLLM](https://arxiv.org/abs/2309.17453) found that models put a lot of attention on the first few tokens, regardless of their content, and called them *attention sinks*. Keeping these few tokens plus a window of recent tokens lets a model generate stably over millions of tokens without retraining, where a plain sliding window collapses.

**KV eviction and selection without retraining.** [H2O](https://arxiv.org/abs/2306.14048) keeps only the recent tokens and the "heavy hitters" that received the most attention so far, and evicts the others; SnapKV selects the important tokens of each head from the attention of the last prompt tokens. Eviction saves memory, but an evicted token is lost even if it becomes relevant later. [Quest](https://arxiv.org/abs/2406.10774) keeps the whole cache, but stores the element-wise minimum and maximum of the keys of each page. These bound the attention score that any query can give to the page, so each decode step loads only the most promising pages for the current query.

**Trainable sparse attention.** Applying sparsity only at inference creates a mismatch with how the model was trained, and selecting individual tokens is hard to make fast. Recent methods train the model with sparse attention from the start:

- **Native Sparse Attention** ([NSA](https://arxiv.org/abs/2502.11089); DeepSeek-AI, Peking University and University of Washington; ACL 2025 Best Paper) runs three attention branches in parallel and mixes their outputs with learned gates: a *compressed* branch attends to coarse summaries of blocks of tokens, a cheap global view; a *selected* branch uses the scores of the compressed branch to pick the most important blocks and attends to their tokens in full detail; and a *sliding window* branch covers the local context. Selection works on contiguous blocks, and all heads of a GQA group share the same selected blocks so they are loaded only once, which keeps the kernels fast. NSA matches or beats full attention on general, long-context and reasoning benchmarks, and on 64K-token sequences it is up to 9 times faster in the forward pass, 6 times faster in the backward pass and 11.6 times faster in decoding.
- **Mixture of Block Attention** ([MoBA](https://arxiv.org/abs/2502.13189), Moonshot AI) applies the idea of mixture of experts to attention: for each query, a gate picks the few blocks of keys and values to attend to.
- **DeepSeek Sparse Attention** (DSA), introduced in [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp), adds a small and fast *lightning indexer*, running in FP8, that scores all previous tokens for each query; the main attention then attends only to the top 2,048 tokens. The cost of the core attention drops from $$O(L^2)$$ to $$O(L k)$$, for a context of $$L$$ tokens and $$k$$ selected ones. The indexer is first trained to imitate the dense attention, and then the whole model is trained with sparse attention.

Sparse attention reduces the compute and the amount of KV cache read at each step; except for the eviction methods, the cache itself still stores every token.

### Linear attention and hybrid models

Linear attention removes the softmax, so that the matrix products can be reordered. Replacing $$\text{softmax}(QK^T)V$$ with $$\phi(Q)\left(\phi(K)^T V\right)$$, for some feature map $$\phi$$, makes $$\phi(K)^T V$$ a small $$d_\text{head} \times d_\text{head}$$ matrix, and the cost becomes linear in the sequence length ([Katharopoulos et al.](https://arxiv.org/abs/2006.16236)). During decode, linear attention behaves like a recurrent neural network: instead of a KV cache that grows with every token, each layer keeps a fixed-size *state* matrix $$S$$ that maps keys to values (we drop $$\phi$$ for readability):

$$
S_t = S_{t-1} + v_t k_t^T, \quad\quad o_t = S_t \, q_t
$$

The memory and the time per token are constant, whatever the context length. The weakness is that a fixed-size state is a lossy memory that cannot hold all the details of a long context, so pure linear attention models recall exact information worse than softmax attention. Two ideas improved this in recent years:

- **Gating** lets the model forget: $$S_t = \alpha_t S_{t-1} + v_t k_t^T$$, with a data-dependent decay $$\alpha_t \in (0, 1)$$. State space models such as [Mamba](https://arxiv.org/abs/2312.00752) and [Mamba-2](https://arxiv.org/abs/2405.21060) are closely related.
- The **delta rule** lets the model overwrite: $$S_t = S_{t-1} - \beta_t \left(S_{t-1} k_t - v_t\right) k_t^T$$. Instead of adding $$v_t$$ on top of what is already stored for the key $$k_t$$, it moves the stored value towards $$v_t$$ by a learned step $$\beta_t$$. [Gated DeltaNet](https://arxiv.org/abs/2412.06464) combines both ideas.

**Kimi Linear** ([code and models](https://github.com/MoonshotAI/Kimi-Linear), Moonshot AI, 2025) introduces Kimi Delta Attention (KDA), a Gated DeltaNet with a finer-grained gate: one decay per channel instead of one per head. It is a *hybrid* model: for every three KDA layers, there is one full attention layer (MLA), which keeps the exact recall that linear attention lacks. With the same training recipe, the 48B-parameter model (a mixture of experts with 3B active parameters) outperformed full MLA attention, while reducing the KV cache by up to 75% and increasing the decoding throughput up to 6 times at a 1M-token context. Other hybrid models follow the same recipe with different linear layers, such as Jamba (Mamba layers), MiniMax-01 (lightning attention) and Qwen3-Next (Gated DeltaNet).

Linear attention and hybrid models change the architecture, so they require (pre-)training: they are not drop-in replacements for the attention of an existing model.

### Sparse plus linear attention: SLA

Diffusion transformers (DiTs), the models behind modern image and video generators, use bidirectional attention over very long sequences: a short video has tens of thousands of tokens, and attention runs at every denoising step, so it dominates the generation time. [SLA](https://arxiv.org/abs/2509.24006) (Sparse-Linear Attention; Tsinghua University; ICLR 2026) starts from an observation: the attention weights split into a small fraction of large weights, which have a high rank, and a large majority of small weights, which have a very low rank. Sparse attention alone must keep too many blocks to stay accurate, and linear attention alone loses too much quality. SLA splits the attention matrix into blocks and classifies each block as:

- **critical**, computed exactly with FlashAttention (quadratic cost, but only for a few blocks);
- **marginal**, computed with linear attention (cheap);
- **negligible**, skipped.

The three run in a single fused GPU kernel, for both the forward and backward passes, and a few fine-tuning steps adapt the model to it. On the Wan2.1-1.3B video model, SLA reduces the attention computation by 95% without degrading the generation quality, with a 13.7 times faster attention kernel and 2.2 times faster end-to-end video generation. SLA targets diffusion transformers; it is not designed for the token-by-token decoding of language models.

### Summary of attention methods

| Method | Exact? | Needs training? | What it reduces | Best for |
|---|---|---|---|---|
| FlashAttention 1-4 | yes | no | memory traffic, quadratic memory | prefill |
| FlashDecoding | yes | no | idle GPU during decode | long-context decode |
| SageAttention | almost (quantized) | no | compute, memory traffic | prefill, diffusion models |
| MQA, GQA, MLA | changes the model | uptraining or pre-training | KV cache size and reads | decode, large batches |
| Sliding window with attention sinks | no | no | KV cache size | streaming generation |
| H2O, SnapKV, Quest | no | no | KV cache size or reads | long-context decode |
| NSA, MoBA, DSA | no (learned sparsity) | yes | compute, KV cache reads | long contexts |
| Linear and hybrid attention (Kimi Linear) | changes the model | pre-training | KV cache, quadratic cost | very long contexts |
| SLA | no (learned) | a few fine-tuning steps | compute | image and video diffusion |

## Speculative decoding: more than one token per forward pass

### The idea

Decode is memory-bound: a forward pass over one token costs about the same as a forward pass over a handful of tokens, because the time goes into reading the weights. Speculative decoding exploits this. A cheap method *drafts* $$K$$ tokens, and the large *target* model checks all of them in a single forward pass. Every draft token the target agrees with is a token generated almost for free. When the target disagrees, we keep the tokens before the disagreement, take the target's own token at the disagreement, and draft again. With the right acceptance rule, the output is exactly the same as with the target model alone: speculative decoding is lossless.

If each draft token is accepted with probability $$\alpha$$, the expected number of tokens produced per forward pass of the target model is:

$$
\frac{1 - \alpha^{K+1}}{1 - \alpha}
$$

The speedup therefore depends on how often the drafts are right and on how cheap they are. Speculative decoding helps most at small batch sizes, where decode is most memory-bound; at large batch sizes, the GPU is already busy, and the extra verification work competes with useful work. The methods below differ in where the drafts come from.

### Draft-model speculative decoding

The original method ([Leviathan et al.](https://arxiv.org/abs/2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318)) drafts with a small language model that uses the same tokenizer as the target, ideally a distilled version of the target, such as the student we train in the distillation section. Each iteration:

1. the draft model generates $$K$$ tokens autoregressively;
2. the target model runs one forward pass over the context and the $$K$$ draft tokens, which gives its distribution at each of the $$K$$ draft positions, plus one more after the last draft;
3. the draft tokens are checked from left to right.

The acceptance rule depends on how we decode:

- With **greedy decoding**, draft tokens are accepted while they are equal to the target's most likely token. At the first mismatch, the draft token is replaced by the target's most likely token, and the remaining drafts are discarded. If all $$K$$ drafts are accepted, the target's prediction after the last draft is appended as a *bonus* token. The output is identical to greedy decoding with the target alone.
- With **sampling** (*speculative sampling*), the draft model samples each token $$x$$ from its distribution $$p$$, and the target accepts it with probability $$\min\left(1, q(x) / p(x)\right)$$, where $$q$$ is the target's distribution. On rejection, a replacement token is sampled from the normalized residual distribution $$\max(0, q - p)$$, and the remaining drafts are discarded. This modified rejection sampling produces tokens distributed exactly as $$q$$. It requires the drafts to be *sampled* from $$p$$: mixing greedy drafts with this probabilistic rule recovers neither greedy decoding nor sampling.

In a batch, each sequence accepts a different number of tokens, so the sequences progress at different speeds, and the implementation tracks their lengths separately, as in continuous batching.

The following code implements greedy speculative decoding in GPTlite, with the distilled model as the draft:

```python

```

### Prompt lookup decoding

[Prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) needs no draft model at all. It takes the last few generated tokens (an n-gram), searches for the same n-gram earlier in the prompt or in the generated text, and proposes the tokens that followed it as drafts; the verification is the same as above. It costs almost nothing, and gives large speedups when the output copies from the input, as in summarization, code editing, question answering over documents, or rewriting a previous answer. Related methods draft from other sources: [lookahead decoding](https://arxiv.org/abs/2402.02057) collects n-grams from Jacobi iterations of the model itself, and [REST](https://arxiv.org/abs/2311.08252) retrieves continuations from a datastore of text.

The following code implements prompt lookup decoding in GPTlite:

```python

```

### Medusa

[Medusa](https://arxiv.org/abs/2401.10774) adds a few extra *decoding heads* on top of the last hidden state of the target model: the original output head predicts the next token, and head $$i$$ predicts the token $$i+1$$ positions ahead. The top candidates of each head are combined into a tree of possible continuations, and all of them are verified in a single forward pass with *tree attention*, a mask that lets each candidate attend only to its own ancestors in the tree. The longest accepted path is kept. Training only the heads, with the model frozen (Medusa-1), gives speedups above 2.2 times; training the heads together with the model (Medusa-2) reaches 2.3 to 3.6 times. Medusa needs no separate draft model, but each head guesses its token independently, without seeing the tokens guessed by the other heads, so the accuracy drops quickly for later positions. For sampling, Medusa also proposes a faster *typical acceptance* rule, which is not exactly lossless.

### EAGLE

[EAGLE](https://arxiv.org/abs/2401.15077) drafts at the level of *features* rather than tokens. Its draft model is a single lightweight transformer layer that takes the top-layer hidden states (features) of the target, together with the embeddings of the tokens sampled so far, predicts the next feature, and turns it into a token with the target's own output layer. Feature sequences are more regular than token sequences, so these drafts are much more accurate than Medusa's independent heads. [EAGLE-2](https://arxiv.org/abs/2406.16858) shapes the draft tree dynamically, expanding the branches where the draft model is confident. [EAGLE-3](https://arxiv.org/abs/2503.01840) (NeurIPS 2025) noticed that EAGLE barely improved with more training data, and traced this to its feature prediction objective. It drops feature prediction and predicts tokens directly, fuses low-, mid- and high-level features of the target instead of the top layer only, and uses *training-time test*: during training, the draft model is fed its own predictions, to simulate multi-step drafting. EAGLE-3 reaches speedups of up to 6.5 times, about 1.4 times more than EAGLE-2, and 1.38 times higher throughput at a batch size of 64 in SGLang. EAGLE-style drafting is supported by vLLM, SGLang and TensorRT-LLM.

### Self-speculative decoding: LayerSkip

[LayerSkip](https://arxiv.org/abs/2404.16710) uses the early layers of the target model as the draft. The model is trained with *layer dropout*, which skips the later layers more often, and with an *early exit loss*, which trains every layer to make good predictions through the shared output head. At inference, the draft exits after the first $$E$$ layers, and the verification runs only the remaining layers, reusing the computation and the cache of the first $$E$$. There is no second model to store, and the reported speedups reach up to about 2 times on summarization and coding tasks.

### Multi-token prediction

Multi-token prediction (MTP) trains the model itself, during pre-training, to predict several future tokens:

- [Gloeckle et al.](https://arxiv.org/abs/2404.19737) (Meta, 2024) add $$n$$ independent output heads on a shared trunk, each predicting one of the next $$n$$ tokens. Besides improving sample efficiency, notably on code, the extra heads serve as drafts for self-speculative decoding, making the inference of a model trained to predict 4 tokens up to 3 times faster.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) uses *sequential* MTP modules: each module combines the representation of the previous depth with the embedding of the next token, and runs one transformer block, so every prediction stays conditioned on the previous ones. MTP is an extra training objective, and at inference the modules can either be discarded or used as drafts: the second predicted token is accepted 85-90% of the time, giving 1.8 times more tokens per second.

Medusa and EAGLE add drafting to an existing model, while MTP builds it in during pre-training.

### Summary of speculative decoding methods

| Method | Source of the drafts | Extra training | Extra memory | Reported speedup |
|---|---|---|---|---|
| Draft model | a separate small model | none, or distilling a draft | a second model | about 2-3x |
| Prompt lookup | n-grams of the context | none | none | large on copy-heavy tasks |
| Medusa | extra heads | the heads (or joint fine-tuning) | small | 2.2-3.6x |
| EAGLE-3 | a feature-level draft layer | the draft layer | small | up to 6.5x |
| LayerSkip | early layers of the model | special fine-tuning | none | up to about 2x |
| Multi-token prediction | MTP heads or modules | pre-training | small | 1.8x (DeepSeek-V3) to 3x |

The reported speedups come from the respective papers, with different models, tasks and hardware, and they shrink as the batch size grows.

## Quantization

Quantization stores numbers with fewer bits. Weights take less memory, fewer bytes are read at every decode step (which is what limits decode), and low-precision tensor cores perform more operations per second. The cost is a loss of precision, which must stay small enough not to hurt the quality of the model.

### Number formats

A floating point number has a sign bit, exponent bits that set its *range*, and mantissa (fraction) bits that set its *precision*. The picture below compares the three most common formats: FP32, FP16 and BF16. BF16 (*brain floating point*) keeps the 8 exponent bits of FP32 and only 7 mantissa bits, so it covers the same range as FP32 with the memory and speed of a 16-bit format. FP16 has more precision but a much smaller range (5 exponent bits), so large values can overflow. This is why BF16 is the default format of LLMs today.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/AI-Supercomputing/floating_point_representation.png"/>

Recent GPUs added smaller formats:

| Format | Bits | Sign / exponent / mantissa bits | Largest value | Typical use |
|---|---|---|---|---|
| FP32 | 32 | 1 / 8 / 23 | about 3.4 × 10³⁸ | reference precision |
| FP16 | 16 | 1 / 5 / 10 | 65,504 | weights and activations |
| BF16 | 16 | 1 / 8 / 7 | about 3.4 × 10³⁸ | default for LLM weights and activations |
| FP8 E4M3 | 8 | 1 / 4 / 3 | 448 | weights, activations and KV cache (Hopper, Ada and newer GPUs) |
| FP8 E5M2 | 8 | 1 / 5 / 2 | 57,344 | values that need more rangrecision |
| FP4 E2M1 | 4 | 1 / 2 / 1 | 6 | weights and activations with micro-scaling (Blackwell GPUs) |
| INT8 | 8 | integers from -128 to 127 | 127 | weights, activations and KV cache |
| INT4 | 4 | integers from -8 to 7 | 7 | weights, with group-wise scales |

Four bits represent only 16 values: FP4 E2M1 can only represent $$\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$$. Such small formats only work with fine-grained scaling. In the **micro-scaling** ([MX](https://arxiv.org/abs/2310.10537)) formats, every block of 32 consecutive values shares an 8-bit power-of-two scale (MXFP8, MXFP6 and MXFP4), and NVIDIA's **NVFP4** uses blocks of 16 values with an FP8 scale, plus one FP32 scale per tensor. Blackwell tensor cores apply these scales in hardware. OpenAI's gpt-oss models, for example, ship their mixture-of-experts weights in MXFP4.

### How quantization works

To quantize a tensor $$x$$ to $$b$$-bit integers, we choose a scale $$s$$ (and optionally a zero-point $$z$$), then round and clip:

$$
x_q = \text{clip}\left(\text{round}\left(\frac{x}{s}\right) + z, \; q_\text{min}, \; q_\text{max} \right), \quad\quad \hat{x} = s \, (x_q - z)
$$

where $$\hat{x}$$ is the dequantized approximation of $$x$$. In **symmetric** (*absmax*) quantization, $$z = 0$$ and $$s = \max \lvert x \rvert / q_\text{max}$$, e.g., with $$q_\text{max} = 127$$ for INT8. **Asymmetric** quantization maps the range $$[\min x, \max x]$$ onto the full integer range, which fits skewed distributions better.

The **granularity** of the scales matters as much as the number of bits. A single scale per tensor is cheap but fragile. One scale per output channel (a row of the weight matrix), per *group* of, e.g., 128 consecutive weights, or per token for the activations, follows the data much more closely, at the cost of storing more scales. Group-wise scales are the standard for 4-bit weights.

The main enemy is **outliers**: a single large value stretches the scale, and all the small values collapse onto a few quantization levels. Weights are well-behaved and easy to quantize. Activations are not: [LLM.int8()](https://arxiv.org/abs/2208.07339) showed that large transformers develop a few *outlier feature dimensions*, with values much larger than the rest, that break naive 8-bit quantization of the activations.

Quantization can be applied after training with a small calibration dataset (**post-training quantization**, PTQ), which is fast and is what we do here, or simulated during training so that the model learns to tolerate it (**quantization-aware training**, QAT), which gives the best results at 4 bits and below. Finally, there are four things to quantize: the weights, the activations, the KV cache and the attention computation.

### Weight-only quantization

Storing the weights in 4 or 8 bits while computing in 16 bits (e.g., W4A16: 4-bit weights, 16-bit activations) speeds up decode almost in proportion to the bytes saved, because decode is memory-bound: 4-bit weights are read four times faster than BF16 ones. The kernel must dequantize the weights on the fly, in registers, right before the multiplication; dequantizing the whole matrix to memory first would cancel the gain. Prefill, which is compute-bound, barely benefits.

Rounding each weight to the nearest level (*round-to-nearest*) works well at 8 bits, but loses too much accuracy at 4 bits. Two methods made 4-bit weights practical:

- [GPTQ](https://arxiv.org/abs/2210.17323) quantizes a weight matrix one column at a time and, after each column, updates the columns not yet quantized to compensate for the error just introduced. The update uses second-order information: the inverse of the Hessian $$H = 2XX^T$$ of the layer's reconstruction error, computed from calibration inputs $$X$$. GPTQ quantizes 175B-parameter models to 3-4 bits in about four GPU hours.
- [AWQ](https://arxiv.org/abs/2306.00978) (Activation-aware Weight Quantization) observes that a small fraction of the weights matters much more than the rest: those that multiply large activations. Instead of keeping them in high precision, it multiplies the salient input channels of the weights by a factor before quantization (and divides the corresponding activations by the same factor, which can be folded into the previous operation), so that they are rounded more accurately. The factors are searched on calibration data.

Other common formats are NF4, from [QLoRA](https://arxiv.org/abs/2305.14314), whose levels are placed for normally-distributed weights, and the GGUF formats of llama.cpp, for CPUs and edge devices.

To see how it works, we write our own quantizer: symmetric per-channel INT8 and group-wise INT4 quantization of all linear layers of GPTlite, and we measure the memory and the validation loss. Without a fused kernel, dequantizing in plain PyTorch saves memory but usually makes the model slower; actual speedups require fused kernels, such as those in torchao or Marlin, which we compare against.

The following code implements our own weight-only INT8 and INT4 quantization of GPTlite:

```python

```

### Weight and activation quantization

To make the matrix multiplication itself faster, both of its inputs must be in low precision, so that the GPU can use its INT8, FP8 or FP4 tensor cores, which are 2 to 4 times faster than the BF16 ones. This also speeds up the compute-bound prefill and large-batch decode. The difficulty is the outliers of the activations:

- **LLM.int8()** keeps the few outlier feature dimensions in 16 bits and runs everything else in INT8.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) moves the difficulty from the activations to the weights. Dividing the $$j$$-th channel of the activations by a factor $$s_j$$, and multiplying the $$j$$-th input channel of the weights by the same factor, does not change the output of the layer but tames the outliers. With $$s_j = \max \lvert X_j \rvert^\alpha / \max \lvert W_j \rvert^{1-\alpha}$$ and typically $$\alpha = 0.5$$, both the weights and the activations become easy to quantize to INT8 (W8A8).
- **FP8** weights and activations are the most common choice on Hopper and Blackwell GPUs: with per-channel or per-token scales, they are nearly lossless for most LLMs.
- **Rotations** ([QuaRot](https://arxiv.org/abs/2404.00456), [SpinQuant](https://arxiv.org/abs/2405.16406)) multiply the weights and activations by orthogonal matrices, such as Hadamard matrices. Since $$RR^T = I$$, the output does not change, but the rotation spreads each outlier across all channels. This makes 4-bit weights, activations and KV cache possible (W4A4KV4).
- **FP4** (NVFP4 and MXFP4) on Blackwell GPUs uses the micro-scaled formats described above, for weights and activations.

### KV cache quantization

At long contexts and large batches, the KV cache can be larger than the model weights. Storing it in FP8 or INT8 halves its size and the bytes read at each decode step, with almost no loss of quality; most serving engines enable it with a single configuration flag. More aggressive methods go down to 2 bits: [KIVI](https://arxiv.org/abs/2402.02750) quantizes the keys per channel, since their outliers live in fixed channels, and the values per token, and keeps the most recent tokens in full precision. Quantized attention kernels, such as SageAttention, apply the same ideas to the attention computation itself (see the attention section).

### Which one to use

A reasonable order: start in BF16; quantize the KV cache to FP8 for long contexts; use FP8 weights and activations on Hopper or newer GPUs; use 4-bit weights (AWQ or GPTQ) when memory is the constraint or the batch size is small; and use FP4 on Blackwell GPUs, ideally with a model trained or fine-tuned for it (QAT). Always measure the quality as well as the speed: the perplexity on held-out data, and the tasks you care about.

## Distillation and pruning

The most effective way to make a model faster is to make it smaller. Distillation trains a small model to behave like a large one; pruning removes parts of a large model. The two are usually combined.

### Knowledge distillation

Knowledge Distillation (KD) trains a *student* model from a *teacher* model. Information flows from a larger or pre-trained teacher to a smaller or untrained student, to make the student smaller and/or better than it would be if trained alone. The main rationale is that the *soft labels* produced by a trained network, i.e., its full output distribution, are a richer training signal than the user-provided *hard labels*.

As a quick example, take a two-label (dog, cat) classification task. An image of a cat that looks like a dog has the ground-truth label distribution `[0,1]`. A trained model, queried with the same image, outputs something like `[0.4, 0.6]`: it believes it is a cat, but it could also be a dog. The soft label `[0.4, 0.6]` carries more information than the hard label `[0,1]`, and training a second model on such labels lets it use its capacity better, spending less of it on learning noise. In language modeling, the teacher's distribution over the next token tells the student not only which token is right, but also which alternatives are plausible.

There are several categories of KD methods. The loss can match the soft labels of the student and the teacher, as in the example above, or intermediate representations such as feature maps. The student can be a scaled-down version of the teacher's architecture, or a different one. In **offline distillation**, the teacher is trained first and then frozen while the student learns from it; in **online distillation**, both are trained simultaneously. We can use a single teacher or an ensemble of teachers.

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/GPTlite-Compression/model_distillation_offline_online.png"/>

For details on the different methods, see [Distilling the Knowledge in a Neural Network, Google](https://arxiv.org/abs/1503.02531), [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/), and [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525).

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPTlite-Compression/kd_methods.jpg"/>

{: style="text-align:center; font-size: small;"}
An illustration of the different categories of knowledge distillation methods, and of the branches within each category. In this section, we implement offline distillation using soft labels, underlined in red in the picture. Adapted from [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/).

#### Implementing offline distillation with soft labels

Our teacher is the pre-trained GPTlite, and our student is a smaller GPTlite with fewer layers and a smaller embedding. The teacher is frozen: it runs in evaluation mode (no dropout), and its forward pass runs inside `torch.no_grad()`, so that no computation graph or gradients are created for it. At every training step, both models process the same batch, and the student learns to match the teacher's output distribution at every position of the sequence. We compute the teacher's soft labels on the fly instead of storing them on disk: storing them takes batch size × sequence length × vocabulary size values per batch, and only pays off when the teacher is too large to run next to the student (in that case, one usually stores only the top-k logits, together with the inputs they belong to).

The loss is the Kullback-Leibler (KL) divergence between the teacher's distribution $$p$$ and the student's distribution $$q$$:

$$
\begin{equation}
\begin{split}
D_{KL}(p \parallel q) & = H(p,q) - H(p) \\
 & = - \sum_i p_i \log (q_i) + \sum_i p_i \lo(p_i) \\
 & = \sum_i p_i \log \frac{p_i}{q_i}
\end{split}
\end{equation}
$$

where $$H(p,q)$$ is the cross entropy and $$H(p)$$ the entropy of the teacher's distribution. Since $$H(p)$$ does not depend on the student, minimizing the KL divergence is equivalent to minimizing the cross entropy. The loss *values* differ, though: the KL divergence is zero when both distributions match, while the cross entropy equals the entropy of the target. This is why the cross entropy is the usual loss for hard labels, whose entropy is zero, and the KL divergence is used to compare two distributions. In PyTorch, `F.kl_div` expects the student's log-probabilities as input. We also pass the teacher's distribution as log-probabilities (`log_target=True`), which the documentation recommends to avoid numerical issues, and use `reduction='batchmean'` on tensors of shape (batch × sequence length, vocabulary size), so that the loss is the mean KL divergence per token.

The **temperature** $$t$$ controls how soft the distributions ar. For logits $$z$$, the softened output is:

$$
y_i (x \mid t) = \frac{ \exp\frac{z_i(x)}{t} }{ \sum_j \, \exp\frac{z_j(x)}{t} }
$$

A temperature above 1 flattens the distribution and reveals the relative probabilities of the unlikely tokens, which carry most of the extra information. Since the gradients of the softened loss scale with $$1/t^2$$, we multiply the loss by $$t^2$$, as proposed by Hinton et al., so that the size of the gradients does not depend on the temperature. A common variant adds the regular cross entropy with the ground-truth labels, weighted by a factor $$\alpha$$. There are also claims that the mean squared error between logits works better than the KL divergence ([Kim et al.](https://arxiv.org/abs/2105.08919)). For LLMs, recent methods also train the student on sequences it generated itself, scored by the teacher (*on-policy* distillation), which removes the mismatch between the sequences seen in training and in generation.

The following code implements the distillation of GPTlite into a smaller student:

```python

```

The student is useful on its own, as a faster model, and as the draft model for speculative decoding.

### Pruning

Pruning removes parts of a trained model. It is a hard problem, for three reasons: we must decide *what* to remove without trying every option; the parts of a network are coupled, so removing one forces changes elsewhere; and removing anything hurts accuracy, which must then be recovered. A fourth difficulty is turning the removal into actual speed.

**What can be removed.** Unstructured pruning removes individual weights by setting them to zero. Structured pruning removes whole structures, which shrinks the weight matrices. In GPTlite, a linear layer `nn.Linear(d_in, d_out)` stores a weight matrix of shape `d_out × d_in`, so:

- removing **MLP neuron** $$j$$ removes row $$j$$ (and bias $$j$$) of the first MLP layer, and column $$j$$ of the second;
- removing **attention head** $$h$$ removes its $$d_\text{head}$$ rows from the query, key and value projectios, and the corresponding $$d_\text{head}$$ columns of the output projection;
- removing **embedding channel** $$c$$ removes column $$c$$ of the token and position embeddings, entry $$c$$ of every LayerNorm, column $$c$$ of every matrix that reads from the residual stream (query, key and value projections, first MLP layer and final output layer), and row $$c$$ (and bias $$c$$) of every matrix that writes to it (attention output projection and second MLP layer). Every layer changes, which makes this the hardest and most impactful dimension;
- removing a **transformer block** deletes it entirely; thanks to the residual connections, the shapes of the other blocks stay valid.

**Unstructured sparsity needs special hardware.** A weight matrix with 50% of zeros scattered at random is still multiplied as a dense matrix by GPUs, so it saves no time, and saves memory only with a sparse storage format. The exception is NVIDIA's *2:4 semi-structured sparsity* ([Mishra et al.](https://arxiv.org/abs/2104.08378)): in every group of 4 consecutive weights, 2 are zero, and the sparse tensor cores of Ampere and newer GPUs skip them, doubling the peak throughput of those matrix multiplications. The end-to-end speedups are smaller, since attention, memory traffic and the other operations are not accelerated.

**How to decide what to remove.** Pruning methods estimate the *importance* of each weight or structure, usually on a small calibration dataset:

- *magnitude*: small weights matter less (simple, but crude);
- *activations*: a neuron, head or channel whose activations are small on average contributes little (used by Minitron, below);
- *weights times activations*: [Wanda](https://arxiv.org/abs/2306.11695) scores each weight by $$\lvert W_{ij} \rvert \cdot \lVert X_j \rVert_2$$, its magnitude times the norm of its input feature, and prunes without any retraining;
- *gradients*: the first-order Taylor expansion $$\lvert w \cdot \partial L / \partial w \rvert$$ estimates how much the loss increases when $$w$$ is removed ([LLM-Pruner](https://arxiv.org/abs/2305.11627), which also groups coupled structures and removes them together);
- *second-order information*: [SparseGPT](https://arxiv.org/abs/2301.00774) prunes and updates the remaining weights to compensate, one column at a time, using the inverse Hessian (the same idea as GPTQ). It prunes 175B-parameter models to 50-60% unstructured sparsity, or to 2:4 sparsity, in one shot;
- *layer redundancy*: [ShortGPT](https://arxiv.org/abs/2403.03853) removes the blocks whose output is most similar to their input (by cosine similarity), since they change the hidden state the least;
- *learned masks*: [Sheared-LLaMA](https://arxiv.org/abs/2310.06694) learns which heads, neurons, channels and layers to keep to reach a target architecture.

**The Minitron recipe.** NVIDIA's [Minitron](https://arxiv.org/abs/2407.14679) combines structured pruning with distillation:

1. compute the importance of heads, MLP neurons and embedding channels from their activations, and the importance of each layer from how much removing it hurts, on a small calibration set of about a thousand samples;
2. prune the model to the target architecture;
3. retrain the pruned model by distilling from the original one (the KL divergence on the logits, from the previous section), using a few percent of the original training tokens;
4. repeat for smaller sizes.

Deriving 8B and 4B models from a 15B one this way required up to 40 times fewer training tokens than training them from scratch, and gave better accuracy. A [follow-up](https://arxiv.org/abs/2408.11796) on Llama 3.1 8B found that width pruning (heads, neurons and channels) preserves more accuracy, while depth pruning (whole blocks) gives larger speedups. The key insight is that a pruned teacher is a much better starting point for the student than a random initialization.

The following code implements structured pruning of the attention heads and MLP neurons of GPTlite, based on their average activations, followed by distillation from the original model:

```python

```

## Compilers and kernels

### Why kernels matter

Every PyTorch operation launches one or more GPU kernels. Each launch costs a few microseconds of CPU and driver time, and each kernel reads its inputs from GPU memory and writes its outputs back. For large models and batches, these costs hide behind the math. For small models and batches, like GPTlite, they dominate: a decode step launches hundreds of small kernels, and the GPU spends much of its time idle, waiting for the CPU to launch the next kernel, or for the memory round-trips between kernels.

### Kernel fusion and torch.compile

*Kernel fusion* merges consecutive operations into a single kernel, so that intermediate results stay in registers or on-chip memory instead of going through GPU memory. Typical candidates are the chains of element-wise operations after a matrix multiplication (bias, activation function, dropout, residual addition) and the normalization layers. FlashAttention is an extreme example of fusion.

`torch.compile` fuses kernels automatically: TorchDynamo captures the graph of the model from the Python bytecode, and TorchInductor generates fused kernels, written in Triton for GPUs. It is a one-line change: `model = torch.compile(model)`. A change in the shapes of the inputs triggers a recompilation, which is one more reason to use a static KV cache during generation.

### CUDA graphs

Even after fusion, a decode step launches many kernels from Python, one after the other. A *CUDA graph* records the whole sequence of kernel launches once, and replays it with a single launch, removing almost all the CPU overhead. The recording fixes the shapes and memory addresses of all tensors, so the decode step must use static shapes: a preallocated KV cache of maximum length, with attention over the full buffer and a mask for the positions not yet filled. `torch.compile(model, mode="reduce-overhead")` uses CUDA graphs automatically. Combining these techniques with quantization and speculative decoding, the PyTorch team's [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2/) made the decoding of Llama-7B almost 10 times faster, in plain PyTorch. Going one step further, recent work fuses a whole forward pass into a single persistent *megakernel*, removing the gaps between kernels altogether for low-latency decoding.

The following code implements a static KV cache for GPTlite, and compiles its decode step with CUDA graphs:

```python

```

### Inference engines

Production deployments rarely hand-write these optimizations: inference engines combine most of the techniques in this post.

- [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) are open-source serving engines with continuous batching, PagedAttention or RadixAttention, prefix caching, speculative decoding and quantization.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) is NVIDIA's engine, with graph optimizations, kernel fusion, in-flight batching, paged KV caches, and FP8 and FP4 quantization.
- [DeepSpeed-FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen/2024-01-19) and [DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) are DeepSpeed's serving and inference engines.
- llama.cpp runs models on CPUs and edge devices, with its own quantized formats (GGUF).
- ONNX Runtime runs models exported to the portable ONNX format on many kinds of hardware; it is covered in a separate post.

## Architectures built for fast inference

Some of the largest gains come from the architecture of the model itself. These choices are made before pre-training, so we describe them without implementing them in GPTlite.

### Mixture of experts

A mixture-of-experts (MoE) layer replaces the MLP of a block with many smaller MLPs (*experts*) and a *router* that sends each token to the top few experts. Only a fraction of the parameters is used per token: [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) has 47B parameters but uses 13B per token, and [DeepSeek-V3](https://arxiv.org/abs/2412.19437) uses 37B of its 671B. The compute per token follows the *active* parameters, but the memory follows the *total* parameters, since all experts must be loaded. At small batch sizes, each decode step reads only the weights of the selected experts, which makes MoE models fast; at large batch sizes, most experts are used at every step. Serving large MoE models relies on expert parallelism and on balancing the load between experts.

### Hybrid attention models

Models that mix a few full attention layers with many linear attention or state space layers, such as Kimi Linear, Qwen3-Next, Jamba and MiniMax-01, have a much smaller KV cache and a lower cost per token at long contexts (see the linear attention section).

### Diffusion language models

Diffusion language models do not generate text from left to right. They start from a fully masked sequence and *denoise* it over a number of steps, predicting many tokens in parallel at every step, as in [LLaDA](https://arxiv.org/abs/2502.09992). Since the number of steps can be much smaller than the number of tokens, they can generate faster than autoregressive models, and commercial diffusion LMs such as Inception Labs' Mercury and Google's Gemini Diffusion advertise very high generation speeds. They are still an active research area: their bidirectional attention makes KV caching harder, and their quality and controllability are still catching up with autoregressive models.

## Summary

| Technique | What it speeds up | Lossless? | Needs training? | When it helps most |
|---|---|---|---|---|
| KV cache | decode | yes | no | always |
| Prefix caching | prefill (time to first token) | yes | no | shared prompts, multi-turn chat |
| Semantic caching | whole requests | no | no | many similar queries |
| Continuous batching, PagedAttention | throughput | yes | no | serving many users |
| Chunked prefill, disaggregation | latency stability, throughput | yes | no | large deployments |
| FlashAttention, FlashDecoding | attention | yes | no | long contexts |
| MQA, GQA, MLA | KV cache size and reads | no (changes the model) | yes | long contexts, large batches |
| Sparse attention (NSA, DSA) | attention | no | yes | very long contexts |
| Linear and hybrid attention | attention, KV cache | no (changes the model) | yes (pre-training) | very long contexts |
| Speculative decoding | decode latency | yes (with exact acceptance) | depends on the method | small batches, latency-sensitive applications |
| Quantization | memory, bandwidth, compute | no (small loss) | no (PTQ) or yes (QAT) | memory-bound decode, large models |
| Distillation and pruning | everything (smaller model) | no | yes | when a smaller model is good enough |
| Kernel fusion, torch.compile, CUDA graphs | overheads | yes | no | small models, small batches |
| Mixture of experts | compute per token | no (changes the model) | yes (pre-training) | large-scale serving |

The following code benchmarks the methods above on GPTlite, measuring the time to first token, the time per output token, the throughput, the peak memory and the validation loss:

```python

```

## Further reading

- [Lilian Weng: Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (the vLLM paper)
- [DeepSpeed model compression](https://www.deepspeed.ai/tutorials/model-compression/)
- [Achieving top inference performance with the NVIDIA H100 Tensor Core GPU and NVIDIA TensorRT-LLM](https://developer.nvidia.com/blog/achieving-top-inference-performance-with-the-nvidia-h100-tensor-core-gpu-and-nvidia-tensorrt-llm/)
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
- [Pruning and Quantization for Deep Neural Network Acceleration: A Survey](https://arxiv.org/abs/2101.09671)
