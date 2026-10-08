---
layout: post
title:  "Inference optimization (3): faster attention"
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

In PyTorch, `torch.nn.functional.scaled_dot_product_attention` (SDPA) dispatches to a FlashAttention or memory-efficient kernel when the inputs allow it, and FlexAttention generates fused kernels for custom masks. For GPTlite, it only changes a few lines of the attention module.

The following code replaces the attention of GPTlite with PyTorch's fused scaled dot product attention, by changing the class of the attention module of every block, which keeps its weights.
<details markdown="1">
<summary>Show code</summary>

```python
class MultiHeadAttention_SDPA(MultiHeadAttention):
  """ GPTlite's multi-head attention, computed by PyTorch's fused scaled_dot_product_attention, which
      runs a FlashAttention (or memory-efficient) kernel when the GPU and the inputs allow it """

  def forward(self, x, causal_mask=True):
    (B, S, _), H, D = x.shape, self.n_heads, self.d_head
    q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    k = self.key_proj(x).view(B, S, H, D).transpose(1, 2)
    v = self.value_proj(x).view(B, S, H, D).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal_mask,
                                         dropout_p=self.dropout.p if self.training else 0.0)
    out = out.transpose(1, 2).reshape(B, S, H * D)
    return self.dropout(self.out_proj(out))

def use_sdpa(model):
  """ Replaces the attention of every block by the fused one, keeping the weights """
  for block in model.blocks:
    block.mha.__class__ = MultiHeadAttention_SDPA
  return model
```

</details>

## FlashDecoding

FlashAttention parallelizes over the batch, the heads, and blocks of queries. During decode there is a single query per sequence, so with a small batch most of the GPU sits idle while a few thread blocks walk through a long KV cache. [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) also splits the keys and values along the sequence: the chunks are processed in parallel, each producing a partial output and its softmax statistics (maximum and sum), and a final reduction combines them with the same rescaling as the online softmax. This makes decoding with very long contexts up to 8 times faster. Kernel libraries such as [FlashInfer](https://github.com/flashinfer-ai/flashinfer) implement this split-KV strategy, together with attention over paged KV caches.

## Multi Query Attention (MQA) and Grouped Query Attention (GQA)

Decode reads the whole KV cache at every step, so a smaller KV cache means faster decoding, and room for larger batches and longer contexts: 
- **Multi-Head Attention** keeps one query, one key and one value per head.
- **Multi-Query Attention** ([MQA](https://arxiv.org/abs/1911.02150)) keeps one query per head, but all heads share a single key head and a single value head. The KV cache shrinks by a factor equal to the number of heads, at some cost in quality.
- **Grouped-Query Attention** ([GQA](https://arxiv.org/abs/2305.13245)) is the middle ground, used by most current models: the heads are split into $$G$$ groups, and the heads of a group share one key head and one value head. The cache shrinks by a factor $$n_\text{heads} / G$$. GQA becomes multi-head attention when $$G = n_\text{heads}$$, and MQA when $$G = 1$$. A trained multi-head model can be converted to GQA by averaging (mean pooling) the key and value heads of each group, followed by a short *uptraining* with about 5% of the original pre-training compute.

GQA (Grouped-Query Attention) and MQA (Multi-Query Attention) primarily accelerate the decode phase, though they also provide a massive secondary benefit to overall system throughput by freeing up memory. These savings *only appear when there is a KV cache to read during decode*. Without a cache, GQA only saves a little compute in the key and value projections. 

The following code implements GQA with a KV cache that stores only the grouped keys and values. The full code is in [`gptlite_gqa.py`]({{ site.assets }}/Inference/gptlite_gqa.py):

<details markdown="1">
<summary>Show code</summary>

```python
class MultiHeadAttention_GQA(nn.Module):
    """ Multi Head Attention with Grouped Query Attention (GQA) and an optional KV cache.
        Every head has its own query, but the heads are split in n_groups groups, and the heads of a
        group share the same key and value. GQA becomes Multi-Head Attention (MHA) when the number of
        groups equals the number of heads, and Multi-Query Attention (MQA) when there is 1 group.
        The cache stores the keys and values of the n_groups groups only.
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

    def forward(self, x, kv_cache=None, causal_mask=True, max_seqlen=None):
        (B, S, _), H, G, D = x.shape, self.n_heads, self.n_groups, self.d_head

        q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, n_heads, S, d_head]
        k = self.key_proj(x).view(B, S, G, D).transpose(1, 2)    # [B, n_groups, S, d_head]
        v = self.value_proj(x).view(B, S, G, D).transpose(1, 2)

        # If a cache is provided, prepend the past keys and values (of the groups only)
        if kv_cache is not None:
            past_keys, past_values = kv_cache
            k = torch.cat([past_keys, k], dim=2)  # [B, n_groups, S_past + S, d_head]
            v = torch.cat([past_values, v], dim=2)
            if max_seqlen is not None and k.size(2) > max_seqlen:
                k = k[:, :, -max_seqlen:, :]
                v = v[:, :, -max_seqlen:, :]

        # Each group of n_heads/n_groups consecutive heads uses the key and value of its group,
        # e.g. with 12 heads and 4 groups, heads 0-2 use group 0, heads 3-5 use group 1, etc.
        k_heads = k.repeat_interleave(H // G, dim=1)  # [B, n_heads, S_past + S, d_head]
        v_heads = v.repeat_interleave(H // G, dim=1)
        out = scaled_dot_product_attention_kv_cache(q, k_heads, v_heads, causal_mask=causal_mask,
                                                    dropout=self.dropout if self.training else None)

        out = out.transpose(1, 2).reshape(B, S, H * D)  # [B, S, n_heads*d_head]
        out = self.dropout(self.out_proj(out))
        return out, (k, v)

class GPTlite_GQA(GPTlite_KVCache):
    """ GPTlite with a KV cache, whose blocks use Grouped Query Attention """

    def __init__(self, vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, n_groups):
        super().__init__(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen)
        for block in self.blocks:
            block.mha = MultiHeadAttention_GQA(d_model, n_heads, d_head, dropout_p, n_groups)
```

</details>

<br/>
The following code converts the trained multi-head GPTlite into GQA by mean pooling its key and value heads, and uptrains it for a few steps. The full code is in [`main_gqa.py`]({{ site.assets }}/Inference/main_gqa.py), which compares the size of the KV cache, the time per token and the validation loss of multi-head attention, GQA and MQA:

<details markdown="1">
<summary>Show code</summary>

```python
def convert_mha_to_gqa(state_dict, n_heads, d_head, n_groups):
    """ Converts the weights of a multi-head GPTlite into GQA: the key and value projections of the
        heads of each group are replaced by their average (mean pooling) """
    gqa_state_dict = dict(state_dict)
    for name, param in state_dict.items():
        if '.mha.key_proj.' in name or '.mha.value_proj.' in name:
            heads = param.view(n_groups, n_heads // n_groups, d_head, -1)  # [n_groups, heads per group, d_head, d_model or 1]
            gqa_state_dict[name] = heads.mean(dim=1).reshape(n_groups * d_head, *param.shape[1:])
    return gqa_state_dict

# main_gqa.py: convert the trained model and uptrain it for a few steps
model = GPTlite_GQA(vocab_size, d_model, n_heads, d_head, n_layers, dropout_p, seqlen, n_groups).to(device).eval()
model.load_state_dict(convert_mha_to_gqa(model_mha.state_dict(), n_heads, d_head, n_groups))
train_steps(model, train_data, uptrain_iters, lr, batch_size, seqlen)
```

</details>


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

The following code implements MLA in GPTlite, with absorbed attention and the conversion of a trained multi-head model. The full code is in [`main_mla.py`]({{ site.assets }}/Inference/main_mla.py), which also checks that absorbed attention gives the same output as recomputing the keys and values:

<details markdown="1">
<summary>Show code</summary>

```python
class MultiHeadAttention_MLA(nn.Module):
  """ Multi-head Latent Attention (MLA): the keys and values of all heads are computed from a single latent
      vector per token, and only that vector is cached. The cache of a layer is a tuple with a tensor of
      shape [B, 1, S, d_latent]: the latent vector behaves like a single key-value head shared by all heads """

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

  def forward(self, x, kv_cache=None, causal_mask=True, max_seqlen=None):
    (B, S, _), H, D = x.shape, self.n_heads, self.d_head
    dropout = self.dropout if self.training else None
    q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    c = self.kv_down_proj(x).unsqueeze(1)                    # [B, 1, S, d_latent]

    # If a cache is provided, prepend the past latent vectors
    if kv_cache is not None:
      c = torch.cat([kv_cache[0], c], dim=2)  # [B, 1, S_past + S, d_latent]
      if max_seqlen is not None and c.size(2) > max_seqlen:
        c = c[:, :, -max_seqlen:]

    if self.absorb:
      # absorb the key up-projection into the query: q.(W_uk c + b_uk) = (W_uk^T q).c + q.b_uk, where the last
      # term adds the same value to all the scores of a query, so the softmax is not affected by it
      W_uk = self.key_up_proj.weight.view(H, D, -1)            # [H, D, d_latent]
      q_latent = torch.einsum('bhsd,hdc->bhsc', q, W_uk)       # [B, H, S, d_latent]
      # attention with the latent vectors as keys and values; the attention function divides the scores
      # by sqrt(d_latent), so we rescale the queries to divide by sqrt(d_head) as in the original attention
      q_latent = q_latent * (q_latent.size(-1) / D) ** 0.5
      out_latent = scaled_dot_product_attention_kv_cache(q_latent, c, c, causal_mask=causal_mask, dropout=dropout)
      # absorb the value up-projection: sum_t w_t (W_uv c_t + b_uv) = W_uv (sum_t w_t c_t) + b_uv
      W_uv = self.value_up_proj.weight.view(H, D, -1)          # [H, D, d_latent]
      out = torch.einsum('bhsc,hdc->bhsd', out_latent, W_uv) + self.value_up_proj.bias.view(H, 1, D)
    else:
      # recompute the keys and values of all heads from the latent vectors
      k = self.key_up_proj(c[:, 0]).view(B, -1, H, D).transpose(1, 2)    # [B, H, S_past + S, D]
      v = self.value_up_proj(c[:, 0]).view(B, -1, H, D).transpose(1, 2)
      out = scaled_dot_product_attention_kv_cache(q, k, v, causal_mask=causal_mask, dropout=dropout)

    out = out.transpose(1, 2).reshape(B, S, H * D)
    out = self.dropout(self.out_proj(out))
    return out, (c,)

def convert_mha_to_mla(state_dict, n_heads, d_head, d_latent):
  """ Converts the weights of a multi-head GPTlite into MLA: the key and value projections are stacked
      in a single matrix [W_k; W_v], and factorized with a truncated singular value decomposition (SVD)
      into an up-projection (U sqrt(S)) times a down-projection (sqrt(S) V^T) of rank d_latent """
  mla_state_dict = {name: param for name, param in state_dict.items()
                    if '.mha.key_proj.' not in name and '.mha.value_proj.' not in name}
  for prefix in {name[:name.index('mha.') + 4] for name in state_dict if '.mha.' in name}:
    W = torch.cat([state_dict[prefix + 'key_proj.weight'], state_dict[prefix + 'value_proj.weight']])  # [2*H*D, d_model]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    sqrt_S = S[:d_latent].sqrt()
    W_up = U[:, :d_latent] * sqrt_S  # [2*H*D, d_latent]
    mla_state_dict[prefix + 'kv_down_proj.weight'] = sqrt_S[:, None] * Vh[:d_latent]  # [d_latent, d_model]
    mla_state_dict[prefix + 'key_up_proj.weight'] = W_up[:n_heads * d_head]
    mla_state_dict[prefix + 'value_up_proj.weight'] = W_up[n_heads * d_head:]
    mla_state_dict[prefix + 'key_up_proj.bias'] = state_dict[prefix + 'key_proj.bias']
    mla_state_dict[prefix + 'value_up_proj.bias'] = state_dict[prefix + 'value_proj.bias']
  return mla_state_dict
```

</details>

## DeepSeek-V3.2: DeepSeek Sparse Attention

[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) keeps MLA and adds DeepSeek Sparse Attention (DSA). For each query token $$t$$, a small *lightning indexer* scores every past token $$s$$:

$$
I_{t,s} = \sum_{h} w_{t,h} \cdot \text{ReLU}\left( q_{t,h} \cdot k_s \right)
$$

where $$q_{t,h}$$ are the queries of the indexer heads and $$w_{t,h}$$ their weights, both computed from the query token, and $$k_s$$ is a single indexer key per token. The indexer runs in FP8 and is much cheaper than the main attention. The main attention then reads only the 2,048 tokens with the highest index scores, the same tokens for all heads, so its cost per query stops growing with the context.

DSA was added to an already trained model, in two stages. During a short *dense warm-up*, the model keeps its dense attention and only the indexer is trained, to imitate the attention distribution of its layer (the attention weights summed over the heads and normalized) with a KL-divergence loss. Then the whole model is trained with sparse attention, and the input of the indexer is detached from the rest of the model, so the indexer learns only from its own loss.

We do the same with GPTlite: we keep the weights of the trained model, add an indexer to every layer, train the indexers with the dense warm-up, and evaluate the model when each query attends to a quarter of the context window. We compare the indexer's selection with a simpler one of the same size, the most recent tokens. The KV cache also stores the indexer key of every token. Index scores are often tied, since the ReLU outputs many zeros, so the selection uses a stable sort, which breaks ties the same way whether the tokens are processed together or one at a time: decoding with the KV cache then gives the same result as processing the whole sequence.

The following code implements DeepSeek Sparse Attention in GPTlite and the dense warm-up of its indexers. The full code is in [`main_dsa.py`]({{ site.assets }}/Inference/main_dsa.py):

<details markdown="1">
<summary>Show code</summary>

```python
def lightning_index_scores(q, w, k):
  """ Index scores I[t, s] = sum_h w[t, h] * ReLU(q[t, h] . k[s]) of the lightning indexer, for the queries
      q [B, Sq, n_heads, d], the head weights w [B, Sq, n_heads] and a single key per entry k [B, Sk, d] """
  dots = F.relu(torch.einsum('bqhd,bkd->bqhk', q, k))  # [B, Sq, n_heads, Sk]
  return torch.einsum('bqhk,bqh->bqk', dots, w)        # [B, Sq, Sk]

def top_k_indices(scores, k):
  """ Indices of the k highest scores of each row. A stable sort breaks ties by keeping the earliest entries, so the
      selection is the same whether a sequence is processed at once or one token at a time """
  return scores.sort(dim=-1, descending=True, stable=True).indices[..., :k]

def indexer_loss(index_scores, attention_scores, visible):
  """ KL divergence between the attention distribution (the target: attention weights summed over heads and
      normalized) and the softmax of the index scores, over the entries each query can see.
      index_scores: [B, Sq, Sk], attention_scores: [B, H, Sq, Sk] logits, visible: [Sq, Sk] or [B, Sq, Sk] """
  with torch.no_grad():
    target = F.softmax(attention_scores.masked_fill(~visible.unsqueeze(-3), -1e9), dim=-1).sum(dim=1) * visible
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-9)
  log_index = F.log_softmax(index_scores.masked_fill(~visible, -1e9), dim=-1).masked_fill(~visible, 0.0)
  return F.kl_div(log_index.flatten(0, -2), target.flatten(0, -2), reduction='batchmean')

class MultiHeadAttention_DSA(nn.Module):
  # ...

  def forward(self, x, kv_cache=None, causal_mask=True, max_seqlen=None):
    (B, S, _), H, D = x.shape, self.n_heads, self.d_head
    q = self.query_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    k = self.key_proj(x).view(B, S, H, D).transpose(1, 2)
    v = self.value_proj(x).view(B, S, H, D).transpose(1, 2)
    x_index = x.detach()  # the indexer is trained on its own, without gradients into the rest of the model
    k_index = self.index_key_proj(x_index).unsqueeze(1)      # [B, 1, S, d_index]

    # If a cache is provided, prepend the past keys, values and indexer keys
    if kv_cache is not None:
      past_k, past_v, past_k_index = kv_cache
      k, v = torch.cat([past_k, k], dim=2), torch.cat([past_v, v], dim=2)
      k_index = torch.cat([past_k_index, k_index], dim=2)
      if max_seqlen is not None and k.size(2) > max_seqlen:
        k, v, k_index = k[:, :, -max_seqlen:], v[:, :, -max_seqlen:], k_index[:, :, -max_seqlen:]

    Sk = k.size(2)
    causal = torch.ones(S, Sk, dtype=torch.bool, device=x.device).tril(diagonal=Sk - S)  # [S, Sk]
    scores = q @ k.transpose(-2, -1) / D ** 0.5  # [B, H, S, Sk]
    self.indexer_loss = None
    if self.selection == "local":  # the top_k most recent tokens of each query
      n = min(self.top_k, Sk)
      position = torch.arange(Sk - S, Sk, device=x.device)  # position of each query in the keys
      top = (position[:, None] - torch.arange(n, device=x.device)).clamp(min=0).expand(B, S, n)
      visible = torch.zeros(B, S, Sk, dtype=torch.bool, device=x.device).scatter_(-1, top, True) & causal
    else:
      q_index = self.index_query_proj(x_index).view(B, S, self.n_index_heads, self.d_index)
      index_scores = lightning_index_scores(q_index, self.index_weight_proj(x_index), k_index[:, 0])  # [B, S, Sk]
      if self.selection == "dense":  # dense attention; the indexer learns to imitate it
        visible = causal
        self.indexer_loss = indexer_loss(index_scores, scores, causal)
      else:  # the top_k past tokens with the highest index scores
        top = top_k_indices(index_scores.masked_fill(~causal, float('-inf')), min(self.top_k, Sk))
        visible = torch.zeros(B, S, Sk, dtype=torch.bool, device=x.device).scatter_(-1, top, True) & causal

    weights = F.softmax(scores.masked_fill(~visible.unsqueeze(-3), float('-inf')), dim=-1)
    out = (self.dropout(weights) if self.training else weights) @ v  # [B, H, S, D]
    out = self.out_proj(out.transpose(1, 2).reshape(B, S, H * D))
    out = self.dropout(out) if self.training else out
    return out, (k, v, k_index)

def warmup_indexer(model, data, n_steps, lr, batch_size, seqlen):
  """ Dense warm-up of DeepSeek-V3.2: the model runs dense attention and only the indexers are trained, to
      imitate the attention distribution of their layer """
  indexer_params = [p for name, p in model.named_parameters() if '.index_' in name]
  optimizer = torch.optim.Adam(indexer_params, lr=lr)
  set_selection(model, "dense")
  for _ in range(n_steps):
    x, _ = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    model(x.to(device))
    loss = sum(block.mha.indexer_loss for block in model.blocks)
    loss.backward()  # only the indexers receive gradients: their input and target are detached
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
  set_selection(model, "indexer")
  return model
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

We implement both attention types in GPTlite, alternating CSA and HCA across the blocks, with the overlapping compression, the lightning indexer, the shared key-value multi-query attention, the sliding window, the low-rank queries, RMSNorm, the attention sink and the grouped output projection. GPTlite adds learned positional embeddings to its input, so we don't need rotary embeddings, and we scale the compression rates, the window and the number of selected entries down to the short context of GPTlite. As in DSA, the indexer learns to imitate the attention over the compressed entries with the KL-divergence loss, from an input detached from the rest of the model. These attention types change the architecture, so we train the model from scratch, and compare it with the dense GPTlite trained with the same number of steps.

The following code implements CSA and HCA in GPTlite. The full code is in [`main_csa_hca.py`]({{ site.assets }}/Inference/main_csa_hca.py), which compares the validation loss and the size of the KV cache with those of dense attention:

<details markdown="1">
<summary>Show code</summary>

```python
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
    """ h: [B, S, d_model] -> compressed entries [B, S // m, d_entry] """
    B, n_blocks = h.size(0), h.size(1) // self.m
    h = h[:, :n_blocks * self.m]
    C = self.entry_proj(h).view(B, n_blocks, self.m, -1)  # [B, n_blocks, m, n_series * d_entry]
    Z = self.weight_proj(h).view(B, n_blocks, self.m, -1)
    if self.overlap:
      (Ca, Cb), (Za, Zb) = C.chunk(2, dim=-1), Z.chunk(2, dim=-1)
      Za, Zb = Za + self.position_bias[0], Zb + self.position_bias[1]
      Cb = F.pad(Cb, (0, 0, 0, 0, 1, 0))[:, :n_blocks]  # series b of the previous block (none for block 0)
      Zb = F.pad(Zb, (0, 0, 0, 0, 1, 0), value=float('-inf'))[:, :n_blocks]
      C, Z = torch.cat([Ca, Cb], dim=2), torch.cat([Za, Zb], dim=2)  # [B, n_blocks, 2m, d_entry]
    else:
      Z = Z + self.position_bias[0]
    return (F.softmax(Z, dim=2) * C).sum(dim=2)

class CompressedAttention(nn.Module):
  # ...

  def forward(self, x, causal_mask=True):
    (B, S, _), H, D = x.shape, self.n_heads, self.d_head
    c_q = self.q_down_proj(x)
    q = self.q_norm(self.q_up_proj(c_q).view(B, S, H, D)).transpose(1, 2)  # [B, H, S, D]
    entries = self.kv_norm(self.compressor(x))                             # [B, n_blocks, D]
    window = self.kv_norm(self.window_proj(x))                             # [B, S, D]
    n_blocks, t = entries.size(1), torch.arange(S, device=x.device)

    # each query sees the compressed blocks before its own block, and the uncompressed last n_win tokens
    block_visible = torch.arange(n_blocks, device=x.device)[None, :] < (t // self.m)[:, None]   # [S, n_blocks]
    window_visible = (t[None, :] <= t[:, None]) & (t[None, :] > t[:, None] - self.n_win)        # [S, S]
    block_visible = block_visible.expand(B, S, n_blocks)
    self.indexer_loss = None
    if self.top_k is not None and n_blocks > 0:  # CSA: keep the top_k visible blocks with the highest index scores
      x_index, c_q_index = x.detach(), c_q.detach()  # the indexer is trained only by its own loss
      q_index = self.index_query_proj(c_q_index).view(B, S, self.n_index_heads, self.d_index)
      index_scores = lightning_index_scores(q_index, self.index_weight_proj(x_index), self.index_compressor(x_index))
      if self.training:
        self.indexer_loss = indexer_loss(index_scores, q @ entries.transpose(-2, -1).unsqueeze(1) / D ** 0.5, block_visible)
      top = top_k_indices(index_scores.masked_fill(~block_visible, float('-inf')), min(self.top_k, n_blocks))
      block_visible = torch.zeros_like(block_visible).scatter_(-1, top, True) & block_visible

    kv = torch.cat([entries, window], dim=1)[:, None]  # [B, 1, n_blocks + S, D]: keys and values of all heads
    visible = torch.cat([block_visible, window_visible.expand(B, S, S)], dim=-1)  # [B, S, n_blocks + S]
    scores = (q @ kv.transpose(-2, -1) / D ** 0.5).masked_fill(~visible[:, None], float('-inf'))
    sink = self.sink.view(1, H, 1, 1).expand(B, H, S, 1)
    weights = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]  # the sink takes part of the weight
    out = (self.dropout(weights) if self.training else weights) @ kv        # [B, H, S, D]
    out = out.transpose(1, 2).reshape(B, S, self.n_groups, -1)
    out = torch.einsum('bsgi,gio->bsgo', out, self.group_proj).reshape(B, S, -1)
    return self.dropout(self.out_proj(out))

def use_csa_hca(model, m, m_heavy, n_win, top_k):
  """ Replaces the attention of every block: CSA in even blocks, HCA in odd blocks """
  for i, block in enumerate(model.blocks):
    d_model, n_heads, d_head, dropout_p = block.ln1.normalized_shape[0], block.mha.n_heads, block.mha.d_head, block.mha.dropout.p
    if i % 2 == 0:
      block.mha = CompressedAttention(d_model, n_heads, d_head, dropout_p, m, n_win, top_k=top_k)
    else:
      block.mha = CompressedAttention(d_model, n_heads, d_head, dropout_p, m_heavy, n_win)
  return model.to(device)

def train(model, data, n_steps, lr, batch_size, seqlen):
  """ Trains a model with the cross entropy loss, plus the loss of the indexers of its CSA layers """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model.train()
  for _ in range(n_steps):
    x, y = get_batch(data, batch_size=batch_size, seqlen=seqlen)
    logits = model(x.to(device))
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1))
    loss = loss + sum(block.mha.indexer_loss for block in model.blocks if getattr(block.mha, 'indexer_loss', None) is not None)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
  model.eval()
  return model
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

As before, all the code is in the [repository of this post](https://github.com/{{ site.repository }}/tree/master/assets/Inference).
