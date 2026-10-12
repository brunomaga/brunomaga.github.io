---
layout: post
title:  "Efficient inference: faster attention"
categories: [machine learning, Transformer, GPT, inference, attention]
tags: [machinelearning]
---

This post focuses on attention, the only part of a transformer whose cost grows with the context.  Attention is a very important component of LLMs  because the only part of a transformer whose cost grows with the context length:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_\text{head}}} \right) V
$$

During prefill, $$QK^T$$ compares every token with every previous token, so the compute grows quadratically with the prompt length, and a naive implementation stores an $$n \times n$$ matrix of scores per head. During decode, every new token reads the keys and values of the whole context, so the cost of each step grows linearly with the context. Attention dominates at long contexts, roughly once the context is longer than six times the model width: the authors of Native Sparse Attention (below) estimated that attention accounts for 70-80% of the latency when decoding with a 64K-token context. Long documents, long conversations, agents and reasoning models with long chains of thought all live in this regime.

There are three families of solutions, which can be combined:

- **compute exact attention faster**, with better kernels (FlashAttention, FlashDecoding) and lower precision (SageAttention);
- **store fewer keys and values**, by sharing or compressing them (MQA, GQA and MLA);
- **compute less attention**, by attending only to the most relevant tokens (sparse attention), or by replacing softmax attention with a linear-time alternative (linear attention and hybrid models).

## Multi-Head (MHA), Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

Decode reads the whole KV cache at every step, so a smaller KV cache means faster decoding, and room for larger batches and longer contexts: 
- **Multi-Head Attention** keeps one query, one key and one value per head.
- **Multi-Query Attention** ([MQA](https://arxiv.org/abs/1911.02150)) keeps one query per head, but all heads share a single key head and a single value head. The KV cache shrinks by a factor equal to the number of heads, at some cost in quality.
- **Grouped-Query Attention** ([GQA](https://arxiv.org/abs/2305.13245)) is the middle ground, used by most current models: the heads are split into $$G$$ groups, and the heads of a group share one key head and one value head. The cache shrinks by a factor $$n_\text{heads} / G$$. GQA becomes multi-head attention when $$G = n_\text{heads}$$, and MQA when $$G = 1$$. A trained multi-head model can be converted to GQA by averaging (mean pooling) the key and value heads of each group, followed by a short *uptraining* with about 5% of the original pre-training compute.

GQA (Grouped-Query Attention) and MQA (Multi-Query Attention) primarily accelerate the decode phase, though they also provide a massive secondary benefit to overall system throughput by freeing up memory. These savings *only appear when there is a KV cache to read during decode*. Without a cache, GQA only saves a little compute in the key and value projections. 

The following code implements GQA. Its keys and values, which are what a KV cache stores, have $$G$$ heads instead of $$n_\text{heads}$$:

<details markdown="1">
<summary>Show code</summary>

```python
class GroupedQueryAttention(nn.Module):
    """ - Grouped-Query Attention (GQA).
        - GQA becomes Multi-Head Attention (MHA) when n_groups == n_heads
        - GQA becomes Multi-Query Attention (MQA) when n_groups==1
    """

    def __init__(self, d_model, n_heads, d_head, dropout_p, n_groups):
        super().__init__()
        assert n_heads % n_groups == 0, "the number of heads must be a multiple of the number of groups"
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_groups = n_groups

        # one query per head, but one key and one value per group
        self.query_proj = nn.Linear(d_model, n_heads * d_head)
        self.key_proj = nn.Linear(d_model, n_groups * d_head)
        self.value_proj = nn.Linear(d_model, n_groups * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, causal_mask=True):
        (B, S, _), H, G, D = x.shape, self.n_heads, self.n_groups, self.d_head

        q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # BxSxE -> BxSxHxD -> BxHxSxD
        k = self.key_proj(x).view(B, S, G, D).transpose(1, 2)    # BxSxE -> BxSxGxD -> BxGxSxD
        v = self.value_proj(x).view(B, S, G, D).transpose(1, 2)  # BxSxE -> BxSxGxD -> BxGxSxD

        # enable_gqa: each group of H/G consecutive heads uses the key and value of its group, the same as
        # repeating them with k.repeat_interleave(H // G, dim=1). E.g. with 12 heads and 4 groups, heads 0-2
        # use group 0, heads 3-5 use group 1, etc.: BxHxSxD (queries), BxGxSxD (keys, values) -> BxHxSxD
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal_mask, enable_gqa=True,
                                             dropout_p=self.dropout.p if self.training else 0.0)

        out = out.transpose(1, 2).reshape(B, S, H * D)  # BxHxSxD -> BxSxHxD -> BxSx(H*D)
        out = self.out_proj(out)                        # BxSx(H*D) -> BxSxE
        return self.dropout(out)
```

</details>

## FlashAttention

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

Thus, flash attention provides a **speedup primarily from better memory usage**, not fewer FLOPs. FlashAttention computes the same exact attention result with essentially the same asymptotic arithmetic — $$O(N^2d)$$ — but tiles the computation to keep intermediate values on-chip and reduce expensive reads and writes to GPU high-bandwidth memory. That cuts memory traffic and often makes attention faster; it doesn’t fundamentally reduce the number of attention operations.

Each new version of FlashAttention adapted the algorithm to newer hardware:

- [FlashAttention-2](https://arxiv.org/abs/2307.08691) improved the parallelism and the partitioning of work across the GPU, and reduced the non-matmul operations, roughly doubling the speed of the first version.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) targets NVIDIA Hopper GPUs (H100). It overlaps data movement, matrix multiplications and softmax using Hopper's asynchronous hardware, and supports FP8.
- [FlashAttention-4](https://arxiv.org/abs/2603.05451) (2026) targets NVIDIA Blackwell GPUs (B200), whose tensor cores became much faster while the units computing exponentials and the shared memory did not. It emulates part of the exponentials in software, skips most of the rescaling steps of the online softmax, and keeps intermediate results in Blackwell's new tensor memory. On a B200, it reaches up to 1,613 TFLOP/s (71% utilization), up to 1.3 times faster than cuDNN 9.13 and 2.7 times faster than Triton.

In PyTorch, `torch.nn.functional.scaled_dot_product_attention` (SDPA) dispatches to a FlashAttention or memory-efficient kernel when the inputs allow it, and FlexAttention generates fused kernels for custom masks. In a multi-head attention module, a single call to SDPA replaces the computation of the scores, the softmax and the weighted sum of the values:

## FlashDecoding

FlashAttention parallelizes over the batch, the heads, and blocks of queries. During decode there is a single query per sequence, so with a small batch most of the GPU sits idle while a few thread blocks walk through a long KV cache. [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) also splits the keys and values along the sequence: the chunks are processed in parallel, each producing a partial output and its softmax statistics (maximum and sum), and a final reduction combines them with the same rescaling as the online softmax. This makes decoding with very long contexts up to 8 times faster. Kernel libraries such as [FlashInfer](https://github.com/flashinfer-ai/flashinfer) implement this split-KV strategy, together with attention over paged KV caches.


## Quantized attention: SageAttention

Attention can also run in lower precision. The [SageAttention](https://github.com/thu-ml/SageAttention) family, from Tsinghua University, quantizes the inputs of the two matrix multiplications of attention, in a plug-and-play way that needs no retraining:

- [SageAttention](https://arxiv.org/abs/2410.02367) computes $$QK^T$$ in INT8 and $$PV$$ in 16 bits. In some channels, all keys share a large common value that would waste the INT8 range, so it first *smooths* $$K$$ by subtracting its mean over the tokens. This does not change the result: subtracting the same vector $$\bar{k}$$ from all keys subtracts the same value $$q \cdot \bar{k}$$ from every score of a row, and the softmax is not affected by that.
- [SageAttention2](https://arxiv.org/abs/2411.10958) quantizes $$Q$$ and $$K$$ to INT4 with fine-grained (per-thread) scales, and computes $$PV$$ in FP8.
- [SageAttention3](https://arxiv.org/abs/2505.11594) uses the FP4 tensor cores of NVIDIA Blackwell GPUs with micro-scaling, and reaches 1,038 TOPS on an RTX 5090, about 5 times faster than the fastest FlashAttention on that GPU. FlashAttention-3 also has an FP8 mode. 

## Sparse attention

In practice, each query puts most of its attention weight on a small subset of the tokens. Sparse attention computes attention only over the tokens that matter. The difficulty is to find them cheaply, and to read them from memory efficiently: GPUs are fast on contiguous blocks of memory, and slow on scattered individual tokens. **Sparse attention kernels are faster primarily because they avoid loading unnecessary key-value (KV) blocks or tiles from the GPU's high-bandwidth memory (HBM)** into fast on-chip shared memory (SRAM), by skipping unwanted GPU tiles.

**Fixed patterns.** The simplest patterns are known in advance. In *sliding window* attention, each token attends only to the last $$w$$ tokens, which bounds the size of the KV cache; it is used, for example, in [Mistral 7B](https://arxiv.org/abs/2310.06825), and stacking layers still lets information travel further than $$w$$ tokens. [Longformer](https://arxiv.org/abs/2004.05150) and [BigBird](https://arxiv.org/abs/2007.14062) combine local windows with a few global and random tokens. [StreamingLLM](https://arxiv.org/abs/2309.17453) found that models put a lot of attention on the first few tokens, regardless of their content, and called them *attention sinks*. Keeping these few tokens plus a window of recent tokens lets a model generate stably over millions of tokens without retraining, where a plain sliding window collapses.

**KV eviction and selection without retraining.** [H2O](https://arxiv.org/abs/2306.14048) keeps only the recent tokens and the "heavy hitters" that received the most attention so far, and evicts the others; SnapKV selects the important tokens of each head from the attention of the last prompt tokens. Eviction saves memory, but an evicted token is lost even if it becomes relevant later. [Quest](https://arxiv.org/abs/2406.10774) keeps the whole cache, but stores the element-wise minimum and maximum of the keys of each page. These bound the attention score that any query can give to the page, so each decode step loads only the most promising pages for the current query.

**Trainable sparse attention.** Applying sparsity only at inference creates a mismatch with how the model was trained, and selecting individual tokens is hard to make fast. Recent methods train the model with sparse attention from the start:

- **Native Sparse Attention** ([NSA](https://arxiv.org/abs/2502.11089); DeepSeek-AI, Peking University and University of Washington; ACL 2025 Best Paper) runs three attention branches in parallel and mixes their outputs with learned gates: a *compressed* branch attends to coarse summaries of blocks of tokens, a cheap global view; a *selected* branch uses the scores of the compressed branch to pick the most important blocks and attends to their tokens in full detail; and a *sliding window* branch covers the local context. Selection works on contiguous blocks, and all heads of a GQA group share the same selected blocks so they are loaded only once, which keeps the kernels fast. NSA matches or beats full attention on general, long-context and reasoning benchmarks, and on 64K-token sequences it is up to 9 times faster in the forward pass, 6 times faster in the backward pass and 11.6 times faster in decoding.
- **Mixture of Block Attention** ([MoBA](https://arxiv.org/abs/2502.13189), Moonshot AI) applies the idea of mixture of experts to attention: for each query, a gate picks the few blocks of keys and values to attend to.
- **DeepSeek Sparse Attention** (DSA), introduced in DeepSeek-V3.2, selects the tokens with a small *lightning indexer*; we describe and implement it in the section on the DeepSeek models, below.

Sparse attention reduces the compute and the amount of KV cache read at each step; except for the eviction methods, the cache itself still stores every token.

## Linear attention and hybrid models

Linear attention is a completely different architectural approach to solving the transformer's $$O(N^2)$$ bottleneck, and it mathematically rewrites the attention formula so it scales linearly with sequence length $$O(N)$$.

Linear attention removes the softmax, so that the matrix products can be reordered. Replacing $$\text{softmax}(QK^T)V$$ with $$\phi(Q)\left(\phi(K)^T V\right)$$, for some feature map $$\phi$$, makes $$\phi(K)^T V$$ a small $$d_\text{head} \times d_\text{head}$$ matrix, and the cost becomes linear in the sequence length ([Katharopoulos et al.](https://arxiv.org/abs/2006.16236)). 

During decode, linear attention behaves like a recurrent neural network: instead of a KV cache that grows with every token, each layer keeps a fixed-size state matrix $S$ that maps keys to values using a feature mapping function $\phi$:

$$
S_t = S_{t-1} + \phi(k_t) v_t^T, \quad\quad o_t = S_t \phi(q_t)
$$

The matrix $S$ has a **fixed-size** of $d_k \times d_v$ (where $d_k$ is the feature dimension of the keys/queries after applying $\phi$, and $d_v$ is the dimension of the values). This allows for **sequence Independence** where the sequence length ($T$ or $N$) is completely absent from its dimensions. Unlike a standard KV cache that grows with every token generated, the memory and the time per token are constant, whatever the context length. The weakness is that a **fixed-size state is a lossy memory** that cannot hold all the details of a long context, so pure linear attention models recall exact information worse than softmax attention. A few improved this in recent years:

- **Gating** lets the model forget: $$S_t = \alpha_t S_{t-1} + v_t k_t^T$$, with a data-dependent decay $$\alpha_t \in (0, 1)$$. State space models such as [Mamba](https://arxiv.org/abs/2312.00752) and [Mamba-2](https://arxiv.org/abs/2405.21060) are closely related.
- The **delta rule** lets the model overwrite: $$S_t = S_{t-1} - \beta_t \left(S_{t-1} k_t - v_t\right) k_t^T$$. Instead of adding $$v_t$$ on top of what is already stored for the key $$k_t$$, it moves the stored value towards $$v_t$$ by a learned step $$\beta_t$$. [Gated DeltaNet](https://arxiv.org/abs/2412.06464) combines both ideas.
-  [**Kimi Linear**](https://github.com/MoonshotAI/Kimi-Linear) introduces Kimi Delta Attention (KDA), a Gated DeltaNet with a finer-grained gate: one decay per channel instead of one per head. It is a *hybrid* model: for every three KDA layers, there is one full attention layer (MLA), which keeps the exact recall that linear attention lacks. With the same training recipe, the 48B-parameter model (a mixture of experts with 3B active parameters) outperformed full MLA attention, while reducing the KV cache by up to 75% and increasing the decoding throughput up to 6 times at a 1M-token context. Other hybrid models follow the same recipe with different linear layers, such as Jamba (Mamba layers), MiniMax-01 (lightning attention) and Qwen3-Next (Gated DeltaNet).

## Sparse plus linear attention: SLA

Diffusion transformers (DiTs), the models behind modern image and video generators, use bidirectional attention over very long sequences: a short video has tens of thousands of tokens, and attention runs at every denoising step, so it dominates the generation time. [SLA](https://arxiv.org/abs/2509.24006) (Sparse-Linear Attention; Tsinghua University; ICLR 2026) starts from an observation: the attention weights split into a small fraction of large weights, which have a high rank, and a large majority of small weights, which have a very low rank. Sparse attention alone must keep too many blocks to stay accurate, and linear attention alone loses too much quality. SLA splits the attention matrix into blocks and classifies each block as:

- **critical**, computed exactly with FlashAttention (quadratic cost, but only for a few blocks);
- **marginal**, computed with linear attention (cheap);
- **negligible**, skipped.

The three run in a single fused GPU kernel, for both the forward and backward passes, and a few fine-tuning steps adapt the model to it. On the Wan2.1-1.3B video model, SLA reduces the attention computation by 95% without degrading the generation quality, with a 13.7 times faster attention kernel and 2.2 times faster end-to-end video generation. SLA targets diffusion transformers; it is not designed for the token-by-token decoding of language models.

## DeepSeek Multi-head Latent Attention

**Multi-head Latent Attention** (MLA), introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434), compresses the key and value of each token into a single small *latent* vector, and caches only that vector. The keys and values of each head are recovered with up-projection matrices, and these matrices can be merged ("absorbed") into the query and output projections, so that attention runs directly on the cached latent vectors. Rotary positional embeddings (RoPE) prevent this merge, so MLA adds a small separate key component that carries the position. With absorption, attention runs directly over the cached latent vectors, which act as the keys and the values: MLA behaves like multi-query attention with a single, larger key-value head shared by all heads. DeepSeek-V2 reduced the KV cache by 93.3% compared with DeepSeek 67B, and increased its maximum generation throughput 5.76 times. 

MLA accelerates both the decode and prefill phases, though through very different mechanisms. During decode, this is because MLA slashes the KV cache memory footprint to a fraction, and just like MQA and GQA is allows for memory efficiency. During prefill, where the prompt is processed in parallel, the KV down-projection matrices can be mathematically absorbed into the Query projection weights ($$W^Q$$ and $$W^K$$) - this avoids the need to explicitly up-project the full KV cache for every token during the attention computation, making the prefill phase efficient and avoiding the overhead penalties seen in other compressed-attention designs.

The following code implements MLA. With absorption (`absorb = True`), attention runs directly over the latent vectors, and gives the same output as recomputing the keys and values of every head (`absorb = False`):

<details markdown="1">
<summary>Show code</summary>

```python
class MultiHeadLatentAttention(nn.Module):
    """ Multi-head Latent Attention (MLA): the keys and values of all heads are computed from a single latent
        vector of size d_latent (C) per token, and a KV cache stores only that vector. With absorption, the
        latent vectors behave like a single key-value head shared by all heads (multi-query attention) """

    def __init__(self, d_model, n_heads, d_head, dropout_p, d_latent):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.query_proj = nn.Linear(d_model, n_heads * d_head)
        self.kv_down_proj = nn.Linear(d_model, d_latent, bias=False)  # compresses each token into a latent vector
        self.key_up_proj = nn.Linear(d_latent, n_heads * d_head)      # recovers the keys of all heads
        self.value_up_proj = nn.Linear(d_latent, n_heads * d_head)    # recovers the values of all heads
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.absorb = True  # attention directly over the latent vectors, without recomputing keys and values

    def forward(self, x, causal_mask=True):
        (B, S, _), H, D = x.shape, self.n_heads, self.d_head
        dropout_p = self.dropout.p if self.training else 0.0

        q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # BxSxE -> BxSxHxD -> BxHxSxD
        c = self.kv_down_proj(x)                                 # BxSxE -> BxSxC: the latent vectors

        if self.absorb:
            # absorb the key up-projection into the query: q.(W_uk c + b_uk) = (W_uk^T q).c + q.b_uk, where the last
            # term adds the same value to all the scores of a query, so the softmax is not affected by it
            W_uk = self.key_up_proj.weight.view(H, D, -1)       # (H*D)xC -> HxDxC
            q_latent = torch.einsum('bhsd,hdc->bhsc', q, W_uk)  # BxHxSxD -> BxHxSxC
            # multi-query attention with the latent vectors as keys and values, scaled by 1/sqrt(d_head) as in
            # the original attention: BxHxSxC (queries), Bx1xSxC (keys, values) -> BxHxSxC
            kv = c.unsqueeze(1)                                 # BxSxC -> Bx1xSxC
            out_latent = F.scaled_dot_product_attention(q_latent, kv, kv, is_causal=causal_mask, dropout_p=dropout_p,
                                                        scale=D ** -0.5, enable_gqa=True)
            # absorb the value up-projection: sum_t w_t (W_uv c_t + b_uv) = W_uv (sum_t w_t c_t) + b_uv,
            # as the attention weights w_t sum to 1
            W_uv = self.value_up_proj.weight.view(H, D, -1)           # (H*D)xC -> HxDxC
            out = torch.einsum('bhsc,hdc->bhsd', out_latent, W_uv)    # BxHxSxC -> BxHxSxD
            out = out + self.value_up_proj.bias.view(H, 1, D)         # BxHxSxD -> BxHxSxD
        else:
            # recompute the keys and values of all heads from the latent vectors
            k = self.key_up_proj(c).view(B, S, H, D).transpose(1, 2)    # BxSxC -> BxSxHxD -> BxHxSxD
            v = self.value_up_proj(c).view(B, S, H, D).transpose(1, 2)  # BxSxC -> BxSxHxD -> BxHxSxD
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal_mask, dropout_p=dropout_p)  # BxHxSxD -> BxHxSxD

        out = out.transpose(1, 2).reshape(B, S, H * D)  # BxHxSxD -> BxSxHxD -> BxSx(H*D)
        out = self.out_proj(out)                        # BxSx(H*D) -> BxSxE
        return self.dropout(out)
```

</details>

## DeepSeek-V3.2: DeepSeek Sparse Attention

[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) keeps MLA and adds DeepSeek Sparse Attention (DSA). For each query token $$t$$, a small *lightning indexer* scores every past token $$s$$:

$$
I_{t,s} = \sum_{h} w_{t,h} \cdot \text{ReLU}\left( q_{t,h} \cdot k_s \right)
$$

where $$q_{t,h}$$ are the queries of the indexer heads and $$w_{t,h}$$ their weights, both computed from the query token, and $$k_s$$ is a single indexer key per token. The indexer runs in FP8 and is much cheaper than the main attention. The main attention then reads only the 2,048 tokens with the highest index scores, the same tokens for all heads, so its cost per query stops growing with the context.

DSA was added to an already trained model, in two stages. During a short *dense warm-up*, the model keeps its dense attention and only the indexer is trained, to imitate the attention distribution of its layer (the attention weights summed over the heads and normalized) with a KL-divergence loss. Then the whole model is trained with sparse attention, and the input of the indexer is detached from the rest of the model, so the indexer learns only from its own loss.

With a KV cache, the indexer key of every token is cached too. Index scores are often tied, since the ReLU outputs many zeros, so the selection below uses a stable sort, which breaks ties the same way whether the tokens are processed together or one at a time: decoding with a KV cache then gives the same result as processing the whole sequence.

The following code adds DeepSeek Sparse Attention to multi-head attention. It keeps the projections of `MultiHeadAttention`, so it can load the weights of a trained model, and adds an indexer. During the dense warm-up (`dense = True`), the queries attend to all past tokens and the module computes `indexer_loss`, which is backpropagated alone to train the indexer; afterwards, each query attends only to the `top_k` past tokens with the highest index scores:

<details markdown="1">
<summary>Show code</summary>

```python
def lightning_index_scores(q, w, k):
    """ Index scores I[t, s] = sum_h w[t, h] * ReLU(q[t, h] . k[s]) of the lightning indexer, for the queries
        q (BxSqxHIxDI), their head weights w (BxSqxHI) and a single key per token or entry k (BxSkxDI), where
        HI is the number of indexer heads and DI their size """
    dots = F.relu(torch.einsum('bqhd,bkd->bqhk', q, k))  # BxSqxHIxDI, BxSkxDI -> BxSqxHIxSk
    return torch.einsum('bqhk,bqh->bqk', dots, w)        # BxSqxHIxSk, BxSqxHI -> BxSqxSk


def top_k_indices(scores, k):
    """ Indices of the k highest scores of each row: BxSqxSk -> BxSqxk. A stable sort breaks ties by keeping the
        earliest entries, so the selection is the same whether a sequence is processed at once or one token at a time """
    return scores.sort(dim=-1, descending=True, stable=True).indices[..., :k]


def indexer_loss(index_scores, attention_scores, visible):
    """ KL divergence between the attention distribution (the target: attention weights summed over heads and
        normalized) and the softmax of the index scores, over the entries each query can see.
        index_scores: BxSqxSk, attention_scores: BxHxSqxSk logits, visible: SqxSk or BxSqxSk """
    with torch.no_grad():
        weights = F.softmax(attention_scores.masked_fill(~visible.unsqueeze(-3), -1e9), dim=-1)  # BxHxSqxSk
        target = weights.sum(dim=1) * visible                                                    # BxHxSqxSk -> BxSqxSk
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    log_index = F.log_softmax(index_scores.masked_fill(~visible, -1e9), dim=-1).masked_fill(~visible, 0.0)  # BxSqxSk
    return F.kl_div(log_index.flatten(0, -2), target.flatten(0, -2), reduction='batchmean')


class DeepSeekSparseAttention(nn.Module):
    """ Causal multi-head attention with DeepSeek Sparse Attention (DSA): a small lightning indexer scores the
        past tokens of each query, and all heads attend only to the top_k tokens with the highest scores. The
        selection is applied as a mask over the scores of all tokens, which shows its effect on quality; the
        speedup needs a kernel that reads only the keys and values of the selected tokens """

    def __init__(self, d_model, n_heads, d_head, dropout_p, top_k, n_index_heads=4, d_index=16):
        super().__init__()
        self.n_heads, self.d_head, self.top_k = n_heads, d_head, top_k
        self.n_index_heads, self.d_index = n_index_heads, d_index
        self.query_proj = nn.Linear(d_model, n_heads * d_head)
        self.key_proj = nn.Linear(d_model, n_heads * d_head)
        self.value_proj = nn.Linear(d_model, n_heads * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_model)
        self.dropout = nn.Dropout(dropout_p)
        # lightning indexer: a few query heads, a single key per token, and a weight per head
        self.index_query_proj = nn.Linear(d_model, n_index_heads * d_index)
        self.index_key_proj = nn.Linear(d_model, d_index)
        self.index_weight_proj = nn.Linear(d_model, n_index_heads)
        self.dense = False        # dense warm-up: attend to all past tokens, and train the indexer to imitate them
        self.indexer_loss = None  # KL divergence between the attention and the indexer, during the dense warm-up

    def forward(self, x, causal_mask=True):
        assert causal_mask, "DSA is always causal"
        (B, S, _), H, D = x.shape, self.n_heads, self.d_head
        HI, DI = self.n_index_heads, self.d_index

        q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # BxSxE -> BxSxHxD -> BxHxSxD
        k = self.key_proj(x).view(B, S, H, D).transpose(1, 2)    # BxSxE -> BxSxHxD -> BxHxSxD
        v = self.value_proj(x).view(B, S, H, D).transpose(1, 2)  # BxSxE -> BxSxHxD -> BxHxSxD
        scores = q @ k.transpose(-2, -1) / D ** 0.5              # BxHxSxD @ BxHxDxS -> BxHxSxS

        # lightning indexer: it is trained on its own, without gradients into the rest of the model
        x_index = x.detach()
        q_index = self.index_query_proj(x_index).view(B, S, HI, DI)       # BxSxE -> BxSxHIxDI
        w_index = self.index_weight_proj(x_index)                         # BxSxE -> BxSxHI
        k_index = self.index_key_proj(x_index)                            # BxSxE -> BxSxDI
        index_scores = lightning_index_scores(q_index, w_index, k_index)  # BxSxS

        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()  # SxS
        if self.dense:  # dense attention; the indexer learns to imitate it
            visible = causal  # SxS
            self.indexer_loss = indexer_loss(index_scores, scores, causal)
        else:  # the top_k past tokens with the highest index scores
            top = top_k_indices(index_scores.masked_fill(~causal, float('-inf')), min(self.top_k, S))  # BxSxS -> BxSxtop_k
            visible = torch.zeros(B, S, S, dtype=torch.bool, device=x.device).scatter_(-1, top, True) & causal  # BxSxS
            self.indexer_loss = None

        # attention over the visible tokens, the same for all heads: BxHxSxS -> BxHxSxS
        weights = F.softmax(scores.masked_fill(~visible.unsqueeze(-3), float('-inf')), dim=-1)
        out = self.dropout(weights) @ v                 # BxHxSxS @ BxHxSxD -> BxHxSxD
        out = out.transpose(1, 2).reshape(B, S, H * D)  # BxHxSxD -> BxSxHxD -> BxSx(H*D)
        out = self.out_proj(out)                        # BxSx(H*D) -> BxSxE
        return self.dropout(out)
```

</details>

## DeepSeek-V4: Compressed Sparse Attention and Heavily Compressed Attention

[DeepSeek-V4](https://arxiv.org/abs/2606.19348) (2026) supports a context of one million tokens. It replaces MLA with two types of attention that compress the KV cache along the sequence, interleaved across the layers ([Hugging Face's implementation](https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4) documents the details):

- **Compressed Sparse Attention (CSA)** compresses the keys and values of every $$m = 4$$ tokens into a single entry. Each entry is a sum of per-token vectors, weighted by a softmax over the tokens, computed separately for every channel. Each entry also covers the previous block, so the windows overlap. A lightning indexer, as in DSA but over the compressed entries, then selects the top entries of each query (512 in the default configuration).
- **Heavily Compressed Attention (HCA)** compresses every $$m' = 128$$ tokens into a single entry, without overlap, and attends to all the entries: the compressed sequence is short enough for dense attention.

Both use multi-query attention with a single key-value vector per entry, shared by all the query heads and used both as the key and as the value. A query only sees the entries of blocks that ended before its own block, so every layer also attends to the uncompressed vectors of the last 128 tokens: a sliding window that keeps the local details. A few other details complete the design: low-rank queries whose latent vector is shared with the indexer, RMSNorm on the queries and the entries to keep the attention logits from exploding, an *attention sink* per head (a learned logit that lets a head attend to almost nothing), and a grouped output projection. Serving engines must manage a different cache in every layer: the compressed entries, the indexer keys, the sliding window, and the tokens of the last, incomplete block. With a context of one million tokens, DeepSeek-V4-Pro needs only 27% of the per-token inference FLOPs and 10% of the KV cache of DeepSeek-V3.2.

| Model | Attention | KV cache per layer | Each query attends to |
|---|---|---|---|
| DeepSeek-V2 and V3 | MLA | 576 values per token | all the past tokens |
| DeepSeek-V3.2 | MLA and DSA | 576 values and an indexer key per token | the top 2,048 past tokens |
| DeepSeek-V4, CSA layers | compression by 4, then top-k | an entry and an indexer key every 4 tokens, plus the last 128 tokens | the top-k entries and the last 128 tokens |
| DeepSeek-V4, HCA layers | compression by 128 | an entry every 128 tokens, plus the last 128 tokens | all the entries and the last 128 tokens |

The following code implements both attention types in a single module, CSA when `top_k` is given and HCA otherwise, with the overlapping compression, the lightning indexer, the shared key-value multi-query attention, the sliding window, the low-rank queries, RMSNorm, the attention sink and the grouped output projection. A model alternates CSA and HCA across its blocks. The module expects positional information in its input, such as learned positional embeddings, so it omits rotary embeddings. As in DSA, the indexer learns to imitate the attention over the compressed entries with the KL-divergence loss `indexer_loss`, which is added to the training loss, from an input detached from the rest of the model. These attention types change the architecture, so a model that uses them is trained from scratch:

<details markdown="1">
<summary>Show code</summary>

```python
# lightning_index_scores, top_k_indices and indexer_loss are defined in the DSA code above

class Compressor(nn.Module):
    """ Compresses the hidden states of every m tokens into one entry: a sum of per-token entries, weighted per
        channel by a softmax over the tokens of the block, with learned positional biases. With overlap (CSA),
        each entry also covers the m tokens of the previous block, through a second series of entries and weights.
        The tokens of a trailing incomplete block are not compressed (the sliding window covers them) """

    def __init__(self, d_model, d_entry, m, overlap):
        super().__init__()
        self.m, self.overlap = m, overlap
        n_series = 2 if overlap else 1
        self.entry_proj = nn.Linear(d_model, n_series * d_entry, bias=False)   # C^a (and C^b)
        self.weight_proj = nn.Linear(d_model, n_series * d_entry, bias=False)  # Z^a (and Z^b)
        self.position_bias = nn.Parameter(torch.zeros(n_series, m, d_entry))  # B^a (and B^b)

    def forward(self, h):
        """ h: BxSxE -> compressed entries: BxNxDE, with N = S // m blocks and DE = d_entry """
        B, N = h.size(0), h.size(1) // self.m
        h = h[:, :N * self.m]                           # BxSxE -> Bx(N*m)xE
        C = self.entry_proj(h).unflatten(1, (N, self.m))   # Bx(N*m)xE -> BxNxmx(n_series*DE)
        Z = self.weight_proj(h).unflatten(1, (N, self.m))  # Bx(N*m)xE -> BxNxmx(n_series*DE)
        if self.overlap:
            (Ca, Cb), (Za, Zb) = C.chunk(2, dim=-1), Z.chunk(2, dim=-1)    # BxNxmxDE each
            Za, Zb = Za + self.position_bias[0], Zb + self.position_bias[1]
            Cb = F.pad(Cb, (0, 0, 0, 0, 1, 0))[:, :N]  # series b of the previous block (none for block 0)
            Zb = F.pad(Zb, (0, 0, 0, 0, 1, 0), value=float('-inf'))[:, :N]
            C, Z = torch.cat([Ca, Cb], dim=2), torch.cat([Za, Zb], dim=2)  # BxNx(2m)xDE
        else:
            Z = Z + self.position_bias[0]
        return (F.softmax(Z, dim=2) * C).sum(dim=2)  # BxNx(m or 2m)xDE -> BxNxDE


class CompressedAttention(nn.Module):
    """ DeepSeek-V4 attention, always causal. CSA (top_k given) compresses every m tokens with overlap, and a
        lightning indexer selects the top_k compressed entries of each query. HCA (top_k None) compresses every
        m tokens without overlap, and attends to all of them. Both use shared key-value multi-query attention:
        each entry is both the key and the value of all query heads, together with uncompressed entries of the
        last n_win tokens. The indexer is trained, as in DeepSeek Sparse Attention, to imitate the attention over
        the compressed entries: in training, add indexer_loss to the loss. The input is expected to carry
        positional information (e.g. learned positional embeddings), so DeepSeek-V4's partial RoPE is omitted """

    def __init__(self, d_model, n_heads, d_head, dropout_p, m, n_win, top_k=None,
                 n_groups=2, n_index_heads=4, d_index=16):
        super().__init__()
        self.n_heads, self.d_head, self.m, self.n_win, self.top_k = n_heads, d_head, m, n_win, top_k
        self.n_index_heads, self.d_index = n_index_heads, d_index
        d_q_latent = d_model // 2
        self.q_down_proj = nn.Linear(d_model, d_q_latent)  # low-rank queries; the latent is shared with the indexer
        self.q_up_proj = nn.Linear(d_q_latent, n_heads * d_head)
        self.compressor = Compressor(d_model, d_head, m, overlap=top_k is not None)
        self.window_proj = nn.Linear(d_model, d_head)  # uncompressed entries, for the sliding window
        self.q_norm, self.kv_norm = nn.RMSNorm(d_head), nn.RMSNorm(d_head)
        self.sink = nn.Parameter(torch.zeros(n_heads))  # attention sink: a learned logit per head, with no value
        if top_k is not None:  # lightning indexer, with keys compressed like the entries
            self.index_compressor = Compressor(d_model, d_index, m, overlap=True)
            self.index_query_proj = nn.Linear(d_q_latent, n_index_heads * d_index)
            self.index_weight_proj = nn.Linear(d_model, n_index_heads)
        # grouped output projection: each group of heads is projected separately, then all groups together
        self.n_groups, d_group_in, d_group = n_groups, n_heads // n_groups * d_head, d_model // n_groups
        self.group_proj = nn.Parameter(torch.randn(n_groups, d_group_in, d_group) * d_group_in ** -0.5)
        self.out_proj = nn.Linear(n_groups * d_group, d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.indexer_loss = None  # KL divergence between the attention over the entries and the indexer, in training

    def forward(self, x, causal_mask=True):
        assert causal_mask, "CSA and HCA are always causal"
        (B, S, _), H, D, G = x.shape, self.n_heads, self.d_head, self.n_groups
        c_q = self.q_down_proj(x)                                               # BxSxE -> BxSx(E/2)
        q = self.q_norm(self.q_up_proj(c_q).view(B, S, H, D)).transpose(1, 2)  # BxSx(E/2) -> BxSxHxD -> BxHxSxD
        entries = self.kv_norm(self.compressor(x))                             # BxSxE -> BxNxD, with N = S // m blocks
        window = self.kv_norm(self.window_proj(x))                             # BxSxE -> BxSxD
        N, t = entries.size(1), torch.arange(S, device=x.device)

        # each query sees the compressed blocks before its own block, and the uncompressed last n_win tokens
        block_visible = torch.arange(N, device=x.device)[None, :] < (t // self.m)[:, None]   # SxN
        window_visible = (t[None, :] <= t[:, None]) & (t[None, :] > t[:, None] - self.n_win)  # SxS
        block_visible = block_visible.expand(B, S, N)                                        # SxN -> BxSxN
        self.indexer_loss = None
        if self.top_k is not None and N > 0:  # CSA: keep the top_k visible blocks with the highest index scores
            HI, DI = self.n_index_heads, self.d_index
            x_index, c_q_index = x.detach(), c_q.detach()  # the indexer is trained only by its own loss
            q_index = self.index_query_proj(c_q_index).view(B, S, HI, DI)     # BxSx(E/2) -> BxSxHIxDI
            w_index = self.index_weight_proj(x_index)                         # BxSxE -> BxSxHI
            k_index = self.index_compressor(x_index)                          # BxSxE -> BxNxDI
            index_scores = lightning_index_scores(q_index, w_index, k_index)  # BxSxN
            if self.training:
                entry_scores = q @ entries.transpose(-2, -1).unsqueeze(1) / D ** 0.5  # BxHxSxD @ Bx1xDxN -> BxHxSxN
                self.indexer_loss = indexer_loss(index_scores, entry_scores, block_visible)
            index_scores = index_scores.masked_fill(~block_visible, float('-inf'))            # BxSxN
            top = top_k_indices(index_scores, min(self.top_k, N))                             # BxSxN -> BxSxtop_k
            block_visible = torch.zeros_like(block_visible).scatter_(-1, top, True) & block_visible  # BxSxN

        # multi-query attention over the visible entries and window tokens; the sink takes part of the weight
        kv = torch.cat([entries, window], dim=1)[:, None]                             # BxNxD, BxSxD -> Bx1x(N+S)xD
        visible = torch.cat([block_visible, window_visible.expand(B, S, S)], dim=-1)  # BxSxN, BxSxS -> BxSx(N+S)
        scores = (q @ kv.transpose(-2, -1) / D ** 0.5).masked_fill(~visible[:, None], float('-inf'))  # BxHxSx(N+S)
        sink = self.sink.view(1, H, 1, 1).expand(B, H, S, 1)                          # H -> BxHxSx1
        weights = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]      # BxHxSx(N+S+1) -> BxHxSx(N+S)
        out = self.dropout(weights) @ kv                                              # BxHxSx(N+S) @ Bx1x(N+S)xD -> BxHxSxD

        # grouped output projection, with G = n_groups
        out = out.transpose(1, 2).reshape(B, S, G, -1)              # BxHxSxD -> BxSxHxD -> BxSxGx(H/G*D)
        out = torch.einsum('bsgi,gio->bsgo', out, self.group_proj)  # BxSxGx(H/G*D) -> BxSxGx(E/G)
        out = self.out_proj(out.reshape(B, S, -1))                  # BxSxGx(E/G) -> BxSxE -> BxSxE
        return self.dropout(out)
```

</details>

## Summary of attention methods

| Method | Exact? | Needs training? | What it reduces | Best for |
|---|---|---|---|---|
| FlashAttention 1-4 | yes | no | memory traffic, quadratic memory | prefill |
| FlashDecoding | yes | no | idle GPU during decode | long-context decode |
| SageAttention | almost (quantized) | no | compute, memory traffic | prefill, diffusion models |
| MQA, GQA, MLA | changes the model | uptraining or pre-training | KV cache size and reads | decode, large batches |
| Sliding window with attention sinks | no | no | KV cache size | streaming generation |
| H2O, SnapKV, Quest | no | no | KV cache size or reads | long-context decode |
| NSA, MoBA | no (learned sparsity) | yes | compute, KV cache reads | long contexts |
| DSA (DeepSeek-V3.2) | no (learned sparsity) | continued training | compute, KV cache reads | long contexts |
| Linear and hybrid attention (Kimi Linear) | changes the model | pre-training | KV cache, quadratic cost | very long contexts |
| CSA and HCA (DeepSeek-V4) | changes the model | pre-training | KV cache, compute | very long contexts |
| SLA | no (learned) | a few fine-tuning steps | compute | image and video diffusion |
