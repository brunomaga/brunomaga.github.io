---
layout: post
title:  "Prefill vs decode, roofline, KV cache, continuous/ragged batching, disaggregation"
categories: ["Efficient Inference"]
tags: ["LLM", "prefill", "decode", "roofline", "KV cache", "batching", "disaggregation"]
---

The goal of inference optimization is to generate text as fast and as cheaply as possible, with a small memory footprint, and with little or no loss of quality. This matters for real-time and embedded systems, and for any service where the cost per generated token adds up. We will focus on performance. Most methods come with a short PyTorch snippet, which assumes `import torch`, `import torch.nn as nn` and `import torch.nn.functional as F`. The snippets that run a model use the small GPT-like model defined in the section on the KV cache.

## Prefill and decode phases

A GPT-like model generates text *autoregressively*: it reads the prompt, predicts the next token, appends it to the input, and repeats. Inference therefore has two phases that behave very differently:
- **Prefill** processes all the prompt tokens in a single forward pass. The tokens are processed in parallel, so the GPU performs large matrix-matrix multiplications and is **compute-bound**. Prefill ends with the first generated token, so its duration is the **time to first token (TTFT)**.
- **Decode** generates the remaining tokens, one forward pass per token. Because of the Key-Value (KV) cache, which we will detail later, each pass processes a single new token per sequence, so the GPU performs matrix-vector multiplications: it reads every weight of the model from memory to do very little arithmetic with it. Decode is **memory-bandwidth-bound**, and the duration of each step is the **time per output token (TPOT)**, also called **inter-token latency (ITL)**.
- Thus, the total runtime, or **time to last token (TTLT)**, is $$\text{TTLT} = \text{TTFT} + \text{TPOT} \times ( n - 1)$$, where $$n$$ is the number of output tokens.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/prefill_decode.svg"/>

This observation motivates most of this post. Almost every decode optimization does one of three things:
- it **reads fewer bytes** per token (quantization, smaller KV caches, smaller models), or
- it **does more useful work per byte read** (batching, speculative decoding), or
- it **removes overheads** around the math (kernel fusion, CUDA graphs, better scheduling).


### Where does the compute go: MLP versus attention

For a model of width $$d$$ (the embedding size) and a context of $$n$$ tokens, the compute of each layer has, for every token:
- a term in the order of $$O(d^2)$$ for the projections and the MLP;
- a term in the order of $$O(nd)$$ for the attention scores $$QK^T$$ and the weighted sum of the values.

The first term does not depend on the context length; the second grows with every token in the context. So attention becomes the dominant cost as the sequence length $$n$$ increases. **Memory follows the same pattern as computation**. The model weights are read once per decode step and shared by all the sequences of the batch, while each sequence keeps its own cache of attention keys and values (KV cache) that grows with every token. For long contexts and large batches, reading the KV cache dominates the memory traffic and the runtime of each decode step.

## Roofline model

The [roofline model](https://en.wikipedia.org/wiki/Roofline_model) turns this analysis into a picture. It bounds the performance $$P$$ of a program by its *arithmetic intensity* $$I$$, the number of FLOPs it performs per byte that moves between memory and the processor: $$P \le \min(P_\text{peak}, B_\text{peak} \times I)$$, where $$P_\text{peak}$$ is the peak compute performance and $$B_\text{peak}$$ the peak memory bandwidth of the hardware. A roofline chart is plotted on a **log-log scale**, where:
* **X-Axis (Arithmetic Intensity):** How many math operations you perform per byte of data moved (FLOPs/byte).
* **Y-Axis (Attainable Performance):** How fast your code actually runs (GFLOPs/s).

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/roofline.svg"/>

{: style="text-align:center; font-size: small;"}
A roofline plot. App<sub>1</sub> is memory-bound. App<sub>2</sub> is compute-bound, running below the peak performance. Adapted from [Wikipedia's Roofline Model](https://en.wikipedia.org/wiki/Roofline_model).

On a **log-log plot**, the two terms of the minimum form a "roof": a slanted line for the bandwidth bound and a flat line for the peak compute, which meet at the **ridge point**:
- Computation to the left of the ridge point is **memory-bound**: the processor waits for data, and the only ways to speed it up are to move fewer bytes or to do more work per byte. This is typically where the **decode** step falls.
- Computation to the right is **compute-bound**: only faster math, or less of it, helps. The gap between a point and the roof shows how much is lost to other overheads, such as kernel launches, memory latencies or poor memory access patterns. This is typically where the **prefill** step falls, close to the roof.

**Double buffering** can help a program approach the ideal of the roofline model, in which data movement and computation overlap. With one buffer, memory loads and computation alternate in a loop, so transfer and compute times add up. With two buffers, the program computes one tile while loading the next tile into the other buffer. The roofline bound assumes this overlap: it is based on the *slower* of compute and memory movement.

We can move a point **right**, i.e., do more FLOPs per byte, via:
- A larger batch or more tokens per pass, so each weight byte is reused across more rows. For decode, this means batching and speculative decoding, which verifies several draft tokens in one pass.
- Better data reuse through tiling and blocking in shared memory or SRAM.
- Operator fusion, which avoids writing intermediates out to DRAM and reading them back.
- Fewer bytes per FLOP: quantized weights and KV cache, and attention variants that shrink the KV cache (GQA, MLA). These matter most for decode, which is dominated by weight and KV-cache reads.

We can move a point **up**, i.e., closer to the roof at the same intensity, by:
- Using tensor cores or matrix units and vectorized instructions.
- Good occupancy and overlapping compute with memory transfers.
- Coalesced, aligned memory access, which brings achieved bandwidth closer to peak.
- Fewer stalls: synchronization, bank conflicts, warp divergence, and kernel launch overhead (reduced with CUDA graphs, which helps decode's many small kernels).

For **compute-bound prefill**, which is already near the flat roof, the remaining gains come from:
- **Raising the roof**: lower-precision tensor-core math such as FP8 or FP4.
- **Doing less work**: sparse attention or smaller models. These cut runtime without necessarily moving the point on the plot.


### How to calculate the roofline point for an application

To find the $$X$$ coordinate of our application (❌ in the roofline plot above), we count the floating-point operations and the bytes moved, e.g., by looking at our code or with hardware counters:

$$\text{Arithmetic Intensity } (X) = \frac{\text{Total FLOPs executed}}{\text{Total Bytes read and written}}$$

To find the $$Y$$ coordinate of ❌, we measure the wall-clock execution time of the kernel:

$$\text{Performance } (Y) = \frac{\text{Total FLOPs executed}}{\text{Execution Time (Seconds)} \times 10^9} \quad \text{[GFLOPs/s]}$$

### Vector Sum (real use case)

Consider a vector loop over $$N = 1{,}000{,}000$$ elements that computes $$C[i] = A[i] + B[i]$$ with 32-bit single-precision floats ($$1 \text{ float} = 4 \text{ bytes}$$).
  * **Total FLOPs:** 1 addition per iteration $$\times 1{,}000{,}000 = 1{,}000{,}000 \text{ FLOPs}$$.
  * **Total Bytes:** Read array $$A$$ ($$4\text{MB}$$), read array $$B$$ ($$4\text{MB}$$), write array $$C$$ ($$4\text{MB}$$) = $$12{,}000{,}000 \text{ bytes}$$.
  * **$$X$$ Coordinate:** $$1{,}000{,}000 / 12{,}000{,}000 = \mathbf{0.083 \text{ FLOPs/byte}}$$.

If the GPU runs the vector loop above in exactly $$0.0001 \text{ seconds}$$:
  * **$$Y$$ Coordinate:** $$1{,}000{,}000 / (0.0001 \times 10^9) = \mathbf{10 \text{ GFLOPs/s}}$$.

Therefore, the roofline point ❌ of this example is $$\mathbf{(0.083, 10)}$$.

### Matrix Multiplication (general formulation)

For the multiplication of two matrices of shapes **$$A \times B$$** and **$$B \times C$$**, which gives an output matrix of shape **$$A \times C$$**, here are the FLOPs and the memory traffic, for a data type of **$$dt$$** bytes per element.

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/inference/imgs/matmul.svg"/>

**Step 1: Floating Point Operations (FLOPs).** To compute each of the $$AC$$ elements of the output matrix, you perform a dot product of $$B$$ multiplications and $$B$$ additions. Each multiply-accumulate counts as $$2$$ FLOPs. Therefore:

$$
\text{Total FLOPs} = 2ABC
$$

**Step 2, naive: Memory Traffic (Bytes) for an implementation without cache.** The number of bytes loaded from memory depends heavily on the implementation (naive vs. tiled/optimized). If every element is fetched directly from main memory, without any cache reuse:
* Matrix 1 ($$AB$$) is loaded $$C$$ times (once for every column of Matrix 2).
  $$\text{Bytes} = ABC\text{ }dt$$
* Matrix 2 ($$BC$$) is loaded $$A$$ times (once for every row of Matrix 1).
  $$\text{Bytes} = ABC\text{ }dt$$
* Total Bytes loaded: $$2ABC\text{ }dt$$ bytes.

**Step 2, cache-optimized: Memory Traffic (Bytes) for an implementation with cache.** In hardware-accelerated environments (like GPUs using shared memory/tiling or Tensor Cores), data is loaded into fast on-chip SRAM/cache and reused across multiple calculations:
* Matrix 1: $$AB\text{ }dt$$ bytes loaded (each element loaded once).
* Matrix 2: $$BC\text{ }dt$$ bytes loaded (each element loaded once).
* Output written: $$AC\text{ }dt$$ bytes.
* Total Memory Traffic (loaded + written): $$(AB + BC + AC)\text{ }dt \text{ bytes}$$

**Step 3: Arithmetic Intensity (Operational Intensity).** Dividing the total FLOPs by the minimum memory traffic gives the arithmetic intensity (FLOPs per byte transferred):

$$
\text{Arithmetic Intensity} = \frac{2ABC}{(AB + BC + AC)\text{ }dt}
$$

For large square matrices where $$A = B = C = N$$, this simplifies to roughly $$\frac{2N}{3\text{ }dt}$$ FLOPs per byte, *making matrix multiplication compute-bound for large sizes and memory-bound for very small sizes*.

**Step 4: Achieved Performance and Roofline Coordinate.** To plot the point ❌ on the roofline model, we combine the arithmetic intensity (X-axis) with the achieved performance in GFLOPs per second (Y-axis). The Y-axis value (Performance) is given by:

$$
\text{Performance (GFLOPs/s)} = \frac{\text{Total FLOPs}}{\text{Execution Time (seconds)} \times 10^9} = \frac{2ABC}{\text{Time}_{\text{sec}} \times 10^9}
$$

leading to the final roofline coordinate ❌:

$$
(X, Y) = \left( \frac{2ABC}{(AB + BC + AC)\text{ }dt}, \; \frac{2ABC}{\text{Time}_{\text{sec}} \times 10^9} \right)
$$

#### Tensor multiplication

When we extend matrix multiplication to higher-dimensional tensors, e.g., with shapes $$A \times B \times D \times E \times F$$ and $$B \times C \times D \times E \times F$$, the extra dimensions ($$D \times E \times F$$) act as batch or parallel execution axes. Both the total FLOPs and the memory traffic scale up by this batch size factor, so it cancels out in the arithmetic intensity, and the X-axis coordinate is identical to that of the standard 2D case.

However, the total FLOPs in the Y-axis performance are multiplied by the batch size $$DEF$$, resulting in the final roofline coordinate:

$$
(X, Y) = \left( \frac{2ABC}{(AB + BC + AC) \text{ } dt}, \; \frac{2ABC \text{ } DEF}{\text{Time}_{\text{sec}} \times 10^9} \right)
$$

#### Side note: FLOPs on forward and backward passes

(heavily based on [the scaling book](https://jax-ml.github.io/scaling-book/transformers/))

In model training, we compute not only the output of the forward pass, but also the gradients in the backward pass. Consider a layer where the input $$X$$ (dimensions $$A \times B$$) is multiplied by the weights $$W$$ (dimensions $$B \times C$$) to give $$Y = XW$$ (dimensions $$A \times C$$). By the chain rule, the gradient of the loss $$L$$ with respect to the weight matrix $$W$$ is:

$$\frac{\partial L}{\partial W} = X^T \left(\frac{\partial L}{\partial Y}\right)$$

At its core, computing this gradient is a standard matrix multiplication. Even though it is written in calculus notation (∂), the operation your computer actually performs is the multiplication of two matrices: in this example, $$X^T$$ times the incoming gradient. So it costs as many FLOPs as any other matrix multiplication of the same dimensions: here, about $$2ABC$$ FLOPs, contracting over the $$A$$ dimension. Similarly, the gradient with respect to the input $$X$$ is:

$$\frac{\partial L}{\partial X} = \left(\frac{\partial L}{\partial Y}\right) W^T$$

This also costs about $$2ABC$$ FLOPs, since $$\frac{\partial L}{\partial Y}$$ has dimensions $$A \times C$$. Therefore, **when both gradients are needed, the backward pass requires about twice the FLOPs of the forward pass: it computes gradients with respect to the weights and the input.** The incoming gradient $$\frac{\partial L}{\partial Y}$$ is supplied by the next layer (or computed from the loss for the final layer).

In brief, this layer requires approximately $$2ABC$$ FLOPs for the forward pass (and inference) and $$4ABC$$ FLOPs for the backward pass, for a total of $$6ABC$$ FLOPs for a full training step. For a transformer linear layer, if $$BC$$ is the number of weights and $$A$$ is the number of tokens processed, this is approximately $$6 \times \text{number of weights} \times \text{number of tokens}$$ FLOPs, or $$6 \times \text{number of weights}$$ FLOPs per token. This estimate covers the matrix multiplications and omits other operations.

Finally, for a weight matrix $$W$$ with $$B \times C$$ parameters, the optimizer step of stochastic gradient descent, which updates the model parameters, computes:

$$
W \leftarrow W-\eta g
$$

for a learning rate $$\eta$$ and gradients $$g$$. It requires $$2BC$$ FLOPs: one multiplication and one subtraction per parameter.

### How to draw the Roofline for a GPU Architecture

To sketch the limits of a hardware architecture, we need two values from the manufacturer's specification sheet, for the chosen precision (e.g., FP32):
1. **Peak Compute Performance ($$P_{\text{peak}}$$):** Measured in GFLOPs/s.
2. **Peak Memory Bandwidth ($$B_{\text{peak}}$$):** Measured in GB/s.

The **Ridge Point** is the arithmetic intensity at which the hardware shifts from being purely memory-bound to purely compute-bound.

$$\text{Ridge Point} = \frac{\text{Peak Compute Performance } (P_{\text{peak}})}{\text{Peak Memory Bandwidth } (B_{\text{peak}})}$$

* **Example (NVIDIA A100 PCIe FP32):** $$P_{\text{peak}} = 19{,}500 \text{ GFLOPs/s}$$ and $$B_{\text{peak}} = 1{,}555 \text{ GB/s}$$.
  * $$\text{Ridge Point} = 19{,}500 / 1{,}555 \approx \mathbf{12.54 \text{ FLOPs/byte}}$$.

To draw the roofline, we use a **logarithmic scale** on both axes:
- The sloped line (left side), where $$x < \text{Ridge Point}$$, is the 45-degree diagonal $$y = B_\text{peak} \times x$$: this is the memory ceiling.
- The horizontal line (right side), where $$x \ge \text{Ridge Point}$$, is flat at $$y = P_\text{peak}$$: this is the compute ceiling.

The following code plots the roofline of a given GPU, and the point of a given operation:

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. HARDWARE SPECS (e.g., NVIDIA A100 GPU)
P_peak = 19500  # Peak Compute: 19,500 GFLOPs/s
B_peak = 1555   # Peak Bandwidth: 1,555 GB/s
ridge_point = P_peak / B_peak  # ~12.54 FLOPs/byte

# 2. CODE SPECS (Vector Addition Example)
total_flops = 1_000_000                  # 1M additions
total_bytes = 1_000_000 * 3 * 4          # 3 arrays (A,B,C) of 4-byte floats
code_x = total_flops / total_bytes       # Arithmetic Intensity = FLOPs / Bytes = 0.083

# Performance = FLOPs / (Time * 1e9)
execution_time = 0.0001                  # 0.1 milliseconds
code_y = total_flops / (execution_time * 1e9)  # 10 GFLOPs/s

# 3. GENERATE ROOFLINE BOUNDARIES
x_line = np.logspace(-2, 3, 500)
y_line = np.minimum(P_peak, B_peak * x_line)

# 4. PLOT THE CHART
plt.figure(figsize=(8, 5))
plt.loglog(x_line, y_line, label="GPU Roofline Boundary", color="red", lw=2)
plt.scatter([code_x], [code_y], color="blue", s=100, zorder=5, label="Your Code")
plt.text(ridge_point * 1.2, P_peak * 0.7, f"Ridge Point: {ridge_point:.2f} FLOPs/B", color="red")
plt.text(code_x * 1.2, code_y, f"Your Code ({code_x:.3f}, {code_y:.1f})", color="blue")
plt.xlabel("Arithmetic Intensity (FLOPs/byte)")
plt.ylabel("Attainable Performance (GFLOPs/s)")
plt.title("Simple GPU Roofline Model Analysis")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("simple_roofline.png") # Save the plot
plt.show()
```


## Caching

There are three kinds of inference caching, each operating at a different layer of the stack (this three-layer view is adapted from [this guide](https://machinelearningmastery.com/the-complete-guide-to-inference-caching-in-llms/)):

- **KV caching** stores the attention keys and values computed during a single request, so that the model does not recompute them at every decode step. Every serving engine uses it.
- **Prefix caching** extends KV caching across requests. When different requests share the same leading tokens, such as a system prompt, a reference document, few-shot examples or the previous turns of a conversation, the keys and values of that shared prefix are computed once and reused. It is also called **prompt caching** or **context caching**.
- **Semantic caching** is an application-level cache that stores complete input/output pairs and retrieves them by meaning. Unlike prefix caching, which reuses internal attention states, semantic caching skips the model call entirely when a sufficiently similar query was answered before.

These are complementary layers, not alternatives. KV caching is always on, prefix caching is the highest-leverage optimization for most production applications, and semantic caching pays off when many queries are similar. In short: the KV cache reuses work within one generation, prefix caching reuses work across requests, and semantic caching reuses whole answers.

### KV cache

In the attention layer, the query of the new token attends to the keys and values of all previous tokens. In a causal model, the keys and values of past tokens never change, since token $$i$$ only depends on itself and the tokens before it. Without a cache, each decode step recomputes the keys and values of the whole sequence, so generating $$n$$ tokens costs $$O(n^2)$$ work in the projections and MLPs and $$O(n^3)$$ in attention. With a cache, each step only computes the query, key and value of the new token, appends the new key and value to the cache, and attends over all cached ones: $$O(n)$$ and $$O(n^2)$$ in total. Queries do not need to be cached, because the query of a past token was only needed to compute the output at that past position.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/kv_cache.svg"/>

The price is memory. For every token, every layer stores a key and a value vector per key-value head:

$$
\text{KV cache size} = 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times \text{bytes per value} \times \text{sequence length} \times \text{batch size}
$$

This memory, not compute, often limits how many sequences can be batched together, which is why many techniques in this post shrink the KV cache.

Two implementation details matter in practice:

- A **dynamic cache** grows by concatenating the new keys and values to the cache at every step. This is simple, but every concatenation allocates a new tensor and copies the whole cache. A **static cache** preallocates a buffer for the maximum sequence length and writes each new entry in place.
- Many models use **learned absolute positional embeddings** and a context window of `seqlen` tokens. When a sequence grows longer than `seqlen`, the model without a cache slides its window and re-encodes the last `seqlen` tokens at positions 0 to `seqlen-1`, at every step. A KV cache cannot reproduce this, because the cached keys and values were computed at their original positions and with the context available back then. The model with a cache therefore matches the model without a cache exactly for the first `seqlen` tokens only.

The following code implements a KV cache in a small GPT-like model, and generates text with and without the cache:
- It keeps the last `seqlen` keys and values and, once the window is full, gives every new token the last position, `seqlen-1`: generation can continue past the context window, but its output no longer matches that of the model without a cache. Models with relative positional encodings, such as RoPE, handle a sliding window more gracefully, but keeping only the most recent tokens still changes the output.
- It also accepts several new tokens on top of a cache, as needed by prefix caching and chunked prefill (both covered below) and by speculative decoding: the causal mask is shifted, so that each new token attends to all the cached tokens and to the new ones up to itself.


```python
def causal_mask(Sq, Sk, device=None):
    """ Causal mask of the last Sq of Sk tokens: [Sq, Sk], True where query i can attend to a key, i.e. up to its own
        position Sk - Sq + i. Without a KV cache, Sq == Sk. With a cache of P past tokens, Sk = P + Sq, and the Sq new
        tokens see all the cached tokens and the new ones up to themselves """
    return torch.ones(Sq, Sk, dtype=torch.bool, device=device).tril(diagonal=Sk - Sq)


class MHA(nn.Module):
    """ Multi-Head Attention with a KV cache: the keys and values of the new tokens are appended to those of the past
        tokens, of which at most the last max_seqlen are kept. x: [B, S, E], with H = num_heads and h = head_dim """

    def __init__(self, input_dim, head_dim, num_heads):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.wq = nn.Linear(input_dim, head_dim * num_heads)
        self.wk = nn.Linear(input_dim, head_dim * num_heads)
        self.wv = nn.Linear(input_dim, head_dim * num_heads)
        self.out = nn.Linear(head_dim * num_heads, input_dim)

    def forward(self, x, kv_cache=None, max_seqlen=None):
        B, S, _ = x.shape
        H, h = self.num_heads, self.head_dim

        # queries, keys and values of the new tokens: [B, S, E] -> [B, S, H*h] -> [B, H, S, h]
        q = self.wq(x).view(B, S, H, h).permute(0, 2, 1, 3)
        k = self.wk(x).view(B, S, H, h).permute(0, 2, 1, 3)
        v = self.wv(x).view(B, S, H, h).permute(0, 2, 1, 3)

        # KV cache: append the keys and values of the new tokens to those of the P past tokens, and keep at most the
        # last max_seqlen of them: [B, H, P, h], [B, H, S, h] -> [B, H, P+S, h]
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            if max_seqlen is not None:
                k, v = k[:, :, -max_seqlen:], v[:, :, -max_seqlen:]
        kv_cache = (k, v)

        # the new tokens attend to the cached tokens, and to the new ones up to themselves: [B, H, S, P+S]
        scores = q @ k.transpose(-1, -2) / h ** 0.5
        scores = scores.masked_fill(~causal_mask(S, k.size(2), x.device), float('-inf'))
        attn = F.softmax(scores, dim=-1) @ v                             # @ [B, H, P+S, h] -> [B, H, S, h]
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, S, H * h)  # [B, H, S, h] -> [B, S, H*h]
        return self.out(attn), kv_cache                                  # [B, S, H*h] -> [B, S, E]


class Block(nn.Module):
    """ Transformer block with pre-layer normalization, passing the KV cache of its layer to the attention """

    def __init__(self, input_dim, head_dim, num_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(input_dim), nn.LayerNorm(input_dim)
        self.attn = MHA(input_dim, head_dim, num_heads)
        self.mlp = nn.Sequential(nn.Linear(input_dim, 4 * input_dim), nn.ReLU(), nn.Linear(4 * input_dim, input_dim))

    def forward(self, x, kv_cache=None, max_seqlen=None):
        attn, kv_cache = self.attn(self.ln1(x), kv_cache, max_seqlen)
        x = x + attn
        return x + self.mlp(self.ln2(x)), kv_cache


class GPT(nn.Module):
    """ GPT-like model with a KV cache: a list with the (keys, values) of every layer. The learned positional embeddings
        cover a context window of seqlen tokens: the new tokens follow the P cached ones, at positions P, P+1, ..., and
        at most seqlen-1 """

    def __init__(self, vocab_size, input_dim, head_dim, num_heads, num_layers, seqlen):
        super().__init__()
        self.seqlen = seqlen
        self.token_embedding = nn.Embedding(vocab_size, input_dim)
        self.position_embedding = nn.Embedding(seqlen, input_dim)
        self.blocks = nn.ModuleList(Block(input_dim, head_dim, num_heads) for _ in range(num_layers))
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, vocab_size)

    def forward(self, tokens, kv_cache=None, max_seqlen=None):
        """ tokens: [B, S], the new tokens -> logits: [B, S, V], and the KV cache with the new tokens appended """
        B, S = tokens.shape
        P = 0 if kv_cache is None else kv_cache[0][0].size(2)  # number of cached tokens
        positions = torch.arange(P, P + S, device=tokens.device).clamp(max=self.seqlen - 1)
        x = self.token_embedding(tokens) + self.position_embedding(positions)  # [B, S, E]
        new_cache = []
        for block, layer_cache in zip(self.blocks, kv_cache or [None] * len(self.blocks)):
            x, layer_cache = block(x, layer_cache, max_seqlen)
            new_cache.append(layer_cache)
        return self.head(self.norm(x)), new_cache


@torch.no_grad()
def generate(model, prompt, n_tokens, seqlen):
    """ Greedy generation without a KV cache: every step processes the last seqlen tokens again. prompt: [B, S] """
    tokens = prompt
    for _ in range(n_tokens):
        logits, _ = model(tokens[:, -seqlen:])
        tokens = torch.cat([tokens, logits[:, -1:].argmax(dim=-1)], dim=1)
    return tokens


@torch.no_grad()
def generate_kvcache(model, prompt, n_tokens, seqlen, kv_cache=None):
    """ Greedy generation with a KV cache: the prompt is processed once (prefill), on top of kv_cache if given, and
        every decode step processes only the newest token """
    logits, kv_cache = model(prompt, kv_cache, max_seqlen=seqlen)  # prefill
    tokens = prompt
    for i in range(n_tokens):
        next_token = logits[:, -1:].argmax(dim=-1)  # [B, 1]
        tokens = torch.cat([tokens, next_token], dim=1)
        if i < n_tokens - 1:
            logits, kv_cache = model(next_token, kv_cache, max_seqlen=seqlen)  # decode
    return tokens
```

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="{{ site.assets }}/inference/imgs/static_cache.svg"/>

The code above uses a dynamic cache. The following code implements a static cache of one layer, where the keys and values of the new tokens are written in place:

```python
class StaticKVCache:
    """ KV cache of one layer, preallocated for max_len tokens: the keys and values of the new tokens are written in
        place, instead of being concatenated into new tensors. In MHA, k, v = cache.update(k, v) replaces torch.cat """

    def __init__(self, batch_size, num_heads, max_len, head_dim, device=None, dtype=None):
        self.k = torch.zeros(batch_size, num_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0  # number of cached tokens

    def update(self, k, v):
        """ k, v: [B, H, S, h], of the new tokens -> the keys and values of all the cached tokens: [B, H, P+S, h] """
        S = k.size(2)
        self.k[:, :, self.length:self.length + S] = k
        self.v[:, :, self.length:self.length + S] = v
        self.length += S
        return self.k[:, :, :self.length], self.v[:, :, :self.length]
```


#### PagedAttention: memory management for the KV cache

Serving many requests at once makes memory management hard, because the final length of each sequence is unknown. A naive engine reserves a contiguous KV buffer of the maximum length for every request, and most of that memory stays empty: the authors of [vLLM](https://arxiv.org/abs/2309.06180) measured that only 20% to 40% of the KV memory reserved by existing systems held actual tokens. PagedAttention borrows the idea of virtual memory from operating systems. The KV cache is split into fixed-size *blocks* (pages) that are allocated on demand, and a *block table* maps the logical blocks of each sequence to physical blocks anywhere in GPU memory. The attention kernel reads the blocks through this table. Almost no memory is wasted, so more sequences fit in a batch: vLLM reported 2-4 times higher throughput than previous systems. A block can also be shared by several sequences, and copied only when one of them writes to it (copy-on-write), which is how prefix caching, parallel sampling and beam search share their common prefix.

{: style="text-align:center; font-size: small;"}
<img width="99%" height="99%" src="{{ site.assets }}/inference/imgs/paged_attention.svg"/>

The following code implements a paged KV cache, and the attention of one decode step over it:

```python
class PagedKVCache:
    """ KV cache of one attention layer, split into blocks of block_size tokens, allocated on demand from a shared
        pool. Each sequence has a block table: the list of the physical blocks that hold its tokens """

    def __init__(self, num_blocks, block_size, num_heads, head_dim, device=None, dtype=None):
        self.block_size = block_size
        self.k_pool = torch.zeros(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free_blocks = list(range(num_blocks))
        self.block_tables, self.lengths = {}, {}

    def append(self, seq_id, k, v):
        """ Writes the key and the value [H, h] of a new token, allocating a new block when the last one is full """
        table = self.block_tables.setdefault(seq_id, [])
        position = self.lengths.get(seq_id, 0)
        if position % self.block_size == 0:
            if not self.free_blocks:
                raise RuntimeError("out of KV cache blocks: the scheduler must preempt a sequence")
            table.append(self.free_blocks.pop())
        block, offset = table[position // self.block_size], position % self.block_size
        self.k_pool[block, :, offset] = k
        self.v_pool[block, :, offset] = v
        self.lengths[seq_id] = position + 1

    def free(self, seq_id):
        """ Returns the blocks of a finished sequence to the pool """
        self.free_blocks += self.block_tables.pop(seq_id)
        del self.lengths[seq_id]


def paged_decode_attention(q, cache, seq_ids):
    """ Attention of one new token per sequence, q: [B, H, h], over the cached tokens of its sequence. The blocks of
        each sequence are gathered into a padded tensor; a GPU kernel reads them in place instead """
    B, H, h = q.shape
    tables = [cache.block_tables[s] for s in seq_ids]
    n_blocks = max(len(t) for t in tables)
    L = n_blocks * cache.block_size
    table = torch.tensor([t + [0] * (n_blocks - len(t)) for t in tables], device=q.device)  # [B, n_blocks]

    # the blocks of every sequence: [B, n_blocks, H, block_size, h] -> [B, H, L, h]
    k = cache.k_pool[table].permute(0, 2, 1, 3, 4).reshape(B, H, L, h)
    v = cache.v_pool[table].permute(0, 2, 1, 3, 4).reshape(B, H, L, h)
    lengths = torch.tensor([cache.lengths[s] for s in seq_ids], device=q.device)
    mask = torch.arange(L, device=q.device)[None, :] < lengths[:, None]  # [B, L]: the real tokens

    scores = (q[:, :, None] @ k.transpose(-1, -2))[:, :, 0] / h ** 0.5  # [B, H, 1, h] @ [B, H, h, L] -> [B, H, L]
    scores = scores.masked_fill(~mask[:, None], float('-inf'))
    return (F.softmax(scores, dim=-1)[:, :, None] @ v)[:, :, 0]          # [B, H, 1, L] @ [B, H, L, h] -> [B, H, h]
```


### Prefix caching, prompt caching or context caching

Many requests start with the same tokens: a system prompt, tool definitions, a long document that users ask several questions about, or the history of a conversation. Prefix caching keeps the keys and values of these tokens after a request finishes, so that the next request starting with the same tokens skips their prefill and only processes its new part. The first token arrives much sooner, and the GPU is free for other work.

**How the lookup works.** [vLLM](https://arxiv.org/abs/2309.06180) splits the KV cache into fixed-size blocks (e.g., 16 tokens) and identifies each block by a hash of its tokens combined with the hash of the previous block. The chain of hashes identifies the whole prefix, so a lookup walks the blocks from the start of the prompt and stops at the first miss.
- During the request: prefill computes the K/V of all $$P$$ prompt tokens in one pass, and each decode step appends the K/V of one new token. At the end, the cache holds the K/V of all the prompt and generated tokens.
- Labeling: each full block gets a hash of its 16 token IDs combined with the previous block's hash. So a block's hash identifies the whole prefix up to the end of that block.
- Next request: the engine hashes the new prompt block by block from the start, reuses blocks until the first mismatch, and only prefills the rest.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/prefix_caching.svg"/>

Automatic Prefix Caching does not require extra memory. It works almost entirely by reusing existing memory through pointers. Because vLLM is built natively on PagedAttention, its memory is already divided into discrete, fixed-size physical pages (blocks of usually 16 tokens). Prefix caching simply piggybacks on this virtual memory architecture.

[SGLang](https://arxiv.org/abs/2312.07104)'s RadixAttention stores all cached sequences in a radix tree (a compressed prefix tree), finds the longest prefix shared with each new request, and evicts the least recently used leaves when memory runs out.

**Why the prefix must match exactly.** It is tempting to look for *similar* prompts, e.g., with a vector database, instead of identical ones. This does not work for attention states. The key and value of token $$i$$ depend on all the tokens before it, through every layer of the model. Two prompts that differ in a single early token, even if they mean the same thing, have different keys and values for every token after that difference. Reusing the cache of a similar prompt would silently feed the model a different context than the one the user sent. Exact matching makes prefix caching *lossless*: the output is identical to running the model without the cache. It is also cheap: a hash or tree lookup, instead of computing an embedding and searching a vector index. Similarity search does make sense one level up, where whole answers are reused: that is semantic caching, described below.

**Prefixes, not substrings.** For the same reason, only a prefix can be reused, i.e., the first $$N$$ tokens of the prompt. A paragraph in the middle of a prompt has different keys and values depending on what comes before it. Matching is done on token IDs, not on raw text, so the same text tokenized differently (e.g., with an extra space) is a miss. Research systems reuse non-prefix chunks approximately: [CacheBlend](https://arxiv.org/abs/2405.16444) precomputes the cache of each retrieved document independently and recomputes a small fraction of tokens to restore the attention between documents, and [Prompt Cache](https://arxiv.org/abs/2311.04934) reuses precomputed modules of prompts written in a structured template.

**What is kept.** The KV cache holds the keys and values of every token processed so far, in every layer: the prompt and all the generated tokens, not only the last decode step, since every new token attends to all of them. When a request finishes, its blocks are not freed right away: they stay in memory, marked as reusable, until the memory is needed for something else. Generated tokens are cached too, which is what makes multi-turn chat fast: the prompt of the next turn is the previous prompt, plus the previous answer, plus the new message. Thus, we put the static content first in the prompt (system prompt, tools, documents), and the variable content last (the user's question, timestamps): anything that changes early in the prompt invalidates the cache for everything after it.

The following code implements the lookup of the cached blocks, and prefix caching on top of the KV cache. The keys and values of a prefix shared by all the requests are computed once and reused by every request, with the same outputs as without prefix caching:

```python
def block_hashes(tokens, block_size=16):
    """ Hashes of the full blocks of a list of token ids: each hash combines the tokens of its block with the hash of
        the previous block, so it identifies the whole prefix up to the end of its block """
    hashes, previous = [], None
    for start in range(0, len(tokens) - block_size + 1, block_size):
        previous = hash((previous, tuple(tokens[start:start + block_size])))
        hashes.append(previous)
    return hashes


def cached_prefix_length(tokens, cached_blocks, block_size=16):
    """ Number of leading tokens whose KV cache is cached: the lookup walks the blocks from the start of the prompt,
        and stops at the first miss. cached_blocks: the hashes of the cached blocks, e.g. a dict from hash to block """
    n = 0
    for block_hash in block_hashes(tokens, block_size):
        if block_hash not in cached_blocks:
            break
        n += block_size
    return n


def generate_with_prefix_cache(model, prefix, requests, n_tokens, seqlen):
    """ The KV cache of the prefix shared by all the requests, prefix: [1, P], is computed once, and every request,
        [1, S], only processes its own tokens on top of it. The attention appends to the cache with torch.cat, which
        creates new tensors, so the prefix cache is never modified """
    _, prefix_cache = model(prefix)
    return [generate_kvcache(model, request, n_tokens, seqlen, kv_cache=prefix_cache) for request in requests]
```

### Semantic caching

A semantic cache sits in front of the model, in the application. For every incoming query, it:

1. computes an *embedding*, a vector that represents the meaning of the query;
2. searches a vector index for the most similar past query;
3. if the similarity is above a threshold, returns the stored answer without calling the model;
4. otherwise, calls the model and stores the new (embedding, answer) pair.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/semantic_caching.svg"/>

**How embeddings are created.** Embeddings come from a separate and much smaller *embedding model*, typically a transformer encoder such as the [Sentence-BERT](https://arxiv.org/abs/1908.10084) family (e.g., all-MiniLM-L6-v2), E5, BGE or GTE, or from an embedding API. The encoder is trained with *contrastive learning*: pairs of texts with the same meaning (paraphrases, or a question and its answer) are pulled together, and the other texts of the batch are pushed apart. After training, the cosine similarity between two embeddings measures how close their meanings are. The raw hidden states of a generative LLM make poor embeddings unless the model is fine-tuned in the same contrastive way, which is how some of the best recent embedding models are built.

**Where embeddings are stored.** Any vector database/index works. [FAISS](https://github.com/facebookresearch/faiss) is a popular choice: an in-process library, for CPUs and GPUs, with exact indexes (e.g., `IndexFlatIP`, which searches by inner product, equal to the cosine similarity for normalized vectors) and approximate ones for millions of vectors (IVF, HNSW, product quantization). Since FAISS is a library and not a database, persistence, metadata, deletions and expiration are up to us; vector databases such as Milvus, Qdrant, Weaviate, pgvector or Redis provide them. Tools such as [GPTCache](https://github.com/zilliztech/GPTCache) implement the whole loop and support FAISS as a backend.

**Risks.** A semantic cache is approximate, so it can return a wrong answer: "how do I delete a file in Python?" and "how do I delete a folder in Python?" are very similar sentences with different answers. Answers also go stale, and queries that depend on the conversation history or on the user should not share answers. In practice, we tune the threshold on real traffic, scope the cache per user or tenant, add an expiration time, and optionally confirm hits with a more precise re-ranking model. The strict version of a semantic cache is the **exact-match cache**, which maps a hash of the full request (prompt, model and sampling parameters) to the response. It never returns a wrong answer, but only hits identical requests.

The following code implements a minimal semantic cache with FAISS and a pre-trained sentence encoder, in front of a function `generate_fn(query)` that calls the LLM. It also returns the similarity of each query to the closest cached one, which helps to choose the threshold:

```python
import faiss
import numpy as np


class SemanticCache:
    """ Cache of (query, answer) pairs, looked up by the cosine similarity of the query embeddings """

    def __init__(self, embed_fn, threshold):
        self.embed_fn = embed_fn  # maps a list of strings to unit-norm float32 embeddings of shape [n, dim]
        dim = embed_fn(["probe"]).shape[1]
        self.index = faiss.IndexFlatIP(dim)  # exact search; for unit vectors, inner product = cosine similarity
        self.answers = []
        self.threshold = threshold

    def __call__(self, query, generate_fn):
        """ Returns the answer to the query, the similarity to the closest cached query, and whether it was a hit """
        embedding = self.embed_fn([query])
        similarity = 0.0
        if self.index.ntotal > 0:
            similarities, ids = self.index.search(embedding, 1)  # the most similar cached query
            similarity = float(similarities[0, 0])
            if similarity >= self.threshold:
                return self.answers[ids[0, 0]], similarity, True
        answer = generate_fn(query)  # cache miss: call the model, and store the new pair
        self.index.add(embedding)
        self.answers.append(answer)
        return answer, similarity, False


def sentence_transformer_embedder(model_name="all-MiniLM-L6-v2"):
    """ Embedding function of a pre-trained sentence encoder (downloaded on first use) """
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_name)
    return lambda texts: encoder.encode(texts, normalize_embeddings=True).astype(np.float32)


# usage, with generate_fn(query) the call to the LLM:
# cache = SemanticCache(sentence_transformer_embedder(), threshold=0.85)
# answer, similarity, hit = cache(query, generate_fn)
```

## Batching improvements

Batching is the most effective way to increase throughput. Because decode is memory-bound, a decode step for a batch of 32 sequences takes almost the same time as for a single sequence: the weights are read once and used 32 times. Throughput grows almost linearly with the batch size, until the GPU becomes compute-bound or runs out of memory for the KV caches. The difficulty is that requests arrive at different times and generate different numbers of tokens.

### Static and continuous batching

The simplest approach groups $$B$$ requests and runs them together until all of them finish. Sequences that finish early keep their slot and generate tokens that are thrown away, and new requests wait until the whole batch is done. Prompts of different lengths are padded to the same length. We pad on the right: with causal attention, real tokens never attend to the padding that follows them, so no padding mask is needed, and each sequence reads its next-token prediction at its own last position. Padding on the left would require a padding mask and shifted positions instead. This **static batching** is not ideal, as it requires unnecessary computation and memory for the padding tokens and for the sequences that already finished.

**Continuous batching**, introduced by [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) as *iteration-level scheduling*, makes the scheduling decision at every decode step instead of once per batch: as soon as a sequence finishes, it leaves the batch, and a waiting request takes its slot. The GPU stays busy, and new requests do not wait for the longest sequence of the previous batch. The implementation must track each sequence separately, by keeping the tokens of every running request apart, and rebuild the right-padded batch at every step, so a new request never sees the tokens of the previous occupant of its slot.

The following code implements static and continuous batching. Both produce the same outputs, but continuous batching needs fewer steps, since its slots spend more time generating useful tokens. To keep the code short, every step processes the whole sequences, without a KV cache:

```python
def next_tokens(model, sequences, seqlen):
    """ Greedy next token of each sequence, given as a list of 1D tensors of different lengths. The batch is padded on
        the right: with causal attention, the real tokens never attend to the padding that follows them, so each
        sequence reads its prediction at its own last position """
    windows = [seq[-seqlen:] for seq in sequences]                # the last seqlen tokens of each sequence
    lengths = torch.tensor([len(w) for w in windows], device=windows[0].device)
    batch = nn.utils.rnn.pad_sequence(windows, batch_first=True)  # [B, S], padded with token 0
    logits, _ = model(batch)                                      # [B, S, V]
    return logits[torch.arange(len(windows), device=logits.device), lengths - 1].argmax(dim=-1)  # [B]


def static_batching(model, requests, batch_size, seqlen):
    """ Runs the requests, pairs (prompt, number of tokens to generate), in groups of batch_size, until the longest
        request of each group finishes """
    outputs, n_steps = [], 0
    for start in range(0, len(requests), batch_size):
        group = requests[start:start + batch_size]
        sequences = [prompt for prompt, _ in group]
        for _ in range(max(n_tokens for _, n_tokens in group)):
            tokens = next_tokens(model, sequences, seqlen)
            sequences = [torch.cat([seq, token[None]]) for seq, token in zip(sequences, tokens)]
            n_steps += 1
        # the requests that finished early generated extra tokens, which are thrown away
        outputs += [seq[:len(prompt) + n_tokens] for seq, (prompt, n_tokens) in zip(sequences, group)]
    return outputs, n_steps


def continuous_batching(model, requests, batch_size, seqlen):
    """ Keeps batch_size slots busy: as soon as a request finishes, the next waiting request takes its slot """
    outputs, n_steps = [None] * len(requests), 0
    waiting = list(range(len(requests)))  # ids of the requests waiting for a slot
    running = []                          # (request id, tokens so far) of the requests in the batch
    while waiting or running:
        while waiting and len(running) < batch_size:  # fill the free slots
            request_id = waiting.pop(0)
            running.append((request_id, requests[request_id][0]))
        tokens = next_tokens(model, [seq for _, seq in running], seqlen)
        running = [(request_id, torch.cat([seq, token[None]])) for (request_id, seq), token in zip(running, tokens)]
        n_steps += 1
        still_running = []
        for request_id, seq in running:  # the finished requests leave the batch
            prompt, n_tokens = requests[request_id]
            if len(seq) == len(prompt) + n_tokens:
                outputs[request_id] = seq
            else:
                still_running.append((request_id, seq))
        running = still_running
    return outputs, n_steps
```

### Chunked prefill

With continuous batching, a waiting request takes a free slot as soon as one is available, so its prefill runs in the same step as the decode of the other requests. The prefill of a short prompt costs about as much as a decode step, but the prefill of a long prompt can take many times longer, and all the other requests of the batch wait for it: their text pauses, and their time per output token spikes. **Chunked prefill** solves this by processing a long prompt in several smaller pieces (chunks), one per step, instead of in one forward pass. Every step then takes about the same time, and the other requests keep generating one token per step.

{: style="text-align:center; font-size: small;"}
<img width="95%" height="95%" src="{{ site.assets }}/inference/imgs/continuous_chunked_batching.svg"/>

{: style="text-align:center; font-size: small;"}
The same 6 requests (R1 to R6) on 3 slots, with static batching, continuous batching, and continuous batching with chunked prefill. The width of a step is its duration: a decode step and the prefill of a short prompt take one time unit, and the prefill of the long prompt of R4 takes three. Static batching runs the requests in groups of 3, until the longest request of each group finishes, and pads the prompts of a group to the longest one. Continuous batching gives a free slot to the next request at the next step, but the prefill of R4 stalls R1 and R3. With chunked prefill, R4 processes its prompt in 3 chunks of one time unit each, while R1 and R3 keep decoding.

Each chunk, say of 512 tokens, attends to the KV cache of the earlier chunks. Suppose an 8,000-token prompt arrives while 50 other sequences are decoding:
- Without chunking: one step must process all 8,000 prompt tokens plus the 50 decode tokens. That step takes far longer than a normal decode step, so all 50 users see their text pause.
- With chunking and a budget of 512 tokens per step: each step processes the 50 decode tokens plus 462 prompt tokens. The prompt finishes in about 18 steps, and the other users never notice.
- Note: because attention is causal, the **result is identical to processing the whole prompt at once**. In practice, it works by accumulating the KV cache chunk by chunk, and uses FlashAttention-style kernels to compute the partial attention correctly.
- With a causal mask, tokens only attend to previous tokens, so **attention is final after each chunk**. If the mask is not causal, the attention of the previous chunks needs an extra, final computation. This recomputation requires reloading the previous chunks into GPU SRAM, which increases the memory bandwidth overhead. For that reason, **chunked prefill is rarely used for non-causal models**.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/chunked_prefill.svg"/>

The following code implements chunked prefill:

```python
def chunked_prefill(model, prompt, chunk_size, kv_cache=None):
    """ Prefill of a long prompt: [1, S], in chunks of chunk_size tokens. Each chunk attends to the KV cache of the
        earlier chunks and to itself, so the cache and the logits of the last token are the same as with a single
        forward pass. Returns the logits of the last token: [1, V], and the KV cache """
    for start in range(0, prompt.size(1), chunk_size):
        logits, kv_cache = model(prompt[:, start:start + chunk_size], kv_cache)
    return logits[:, -1], kv_cache
```

### Ragged/packed batching

With continuous batching and chunked prefill, a step runs the prefill chunks of some requests and the decode of others. Engines don't pad the sequences into a rectangular batch, as sequences of different lengths would require substantial padding. Instead, the sequences are concatenated, and **each step is one forward pass over a flat list of tokens from all running sequences**. The next token of each sequence is predicted by the output at the index of its last token. This is ragged/packed batching:
- Linear layers: the MLP and the projections don't care which sequence a token belongs to, so they run as one large matrix multiplication over all the step's tokens.
- Attention: it is the only per-sequence operation. Variable-length kernels (FlashAttention's varlen mode, FlashInfer, PagedAttention) use per-token metadata (sequence id, position, KV block table) so that each token attends only to its own sequence. To separate the sequences, the attention uses a block-diagonal causal mask, and the kernels skip the blocks outside its diagonal, instead of computing a dense, wasteful square matrix.
- When the forward pass reaches the output layer, it doesn't compute predictions for every position. It looks up the metadata, isolates the indices of the last token of each sequence, and performs the argmax/sampling only on those tokens, to get the next token of every running sequence.

{: style="text-align:center; font-size: small;"}
<img width="91%" height="91%" src="{{ site.assets }}/inference/imgs/ragged_batching.svg"/>

The following code packs sequences of different lengths into one flat sequence, with the position of every token, the block-diagonal causal mask, and the indices of the last tokens:

```python
def pack(sequences):
    """ Packs sequences of different lengths, a list of 1D tensors, into one flat sequence without padding. Returns the
        tokens: [1, S], with S the total number of tokens; the position of every token in its own sequence: [S]; a
        block-diagonal causal mask: [S, S], True where a token can attend to itself or an earlier token of its sequence;
        and the index of the last token of every sequence: [B], where the logits that predict its next token are """
    device = sequences[0].device
    lengths = torch.tensor([len(seq) for seq in sequences], device=device)
    seq_ids = torch.repeat_interleave(torch.arange(len(sequences), device=device), lengths)  # [S]: sequence ids
    positions = torch.cat([torch.arange(n, device=device) for n in lengths.tolist()])        # [S]
    s = torch.arange(len(seq_ids), device=device)
    mask = (seq_ids[:, None] == seq_ids[None, :]) & (s[:, None] >= s[None, :])               # [S, S]
    return torch.cat(sequences)[None], positions, mask, lengths.cumsum(0) - 1


# usage: one forward pass for all the sequences, with a model that takes the mask and the positions. Real engines
# give the lengths of the sequences to variable-length attention kernels, instead of building the mask
# tokens, positions, mask, last = pack(sequences)
# logits = model(tokens, mask=mask, positions=positions)  # [1, S, V]
# next_tokens = logits[0, last].argmax(dim=-1)            # [B]: the next token of every sequence
```

### Scheduling

**Mixing prefill and decode** improves performance. This is the roofline picture again. Decode tokens alone are memory-bound and leave most of the GPU's compute idle. But adding prefill chunks to the same step means many more tokens share each weight read, which pushes the step toward the ridge point: this is what [Sarathi-Serve](https://arxiv.org/abs/2403.02310) calls "stall-free batching". To further improve GPU utilization:
- **Dynamic scheduling**. At every step the scheduler decides which requests run, how many tokens each contributes, which waiting requests to admit, and which ones to preempt. These decisions are bounded by the token budget and by the free KV cache blocks.
  - **Token budget**. The scheduler assigns each step a fixed number of tokens. It adds every decode token first, to keep the time per output token low, and then fills the rest with prefill chunks.
  - **Reducing CPU overhead** is done with CUDA graphs for the decode-only steps, and by scheduling the next step on the CPU while the GPU runs the current one (as SGLang does).

The following code selects the tokens of a step within a token budget:

```python
def schedule(decoding, prefilling, budget):
    """ Tokens of one step with chunked prefill: one token for every decoding sequence first, to keep the time per
        output token low, then chunks of the waiting prompts, up to the token budget. decoding: the ids of the decoding
        sequences; prefilling: pairs (id, number of prompt tokens left) of the waiting prompts. Returns pairs
        (id, number of tokens) """
    step = [(seq_id, 1) for seq_id in decoding]
    budget -= len(decoding)
    for seq_id, left in prefilling:
        if budget <= 0:
            break
        step.append((seq_id, min(left, budget)))
        budget -= min(left, budget)
    return step
```


## Prefill-decode disaggregation

Prefill is compute-bound and decode is memory-bound, so they interfere when they share GPUs, and each would prefer a different batch size and parallelism. Disaggregated serving runs them on separate pools of GPUs: prefill workers compute the KV cache of each prompt and send it over a fast interconnect to decode workers, which generate the answer. Each pool is sized and tuned for its own latency target: TTFT for prefill, TPOT for decode. This is the design of [DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677) and [Mooncake](https://arxiv.org/abs/2407.00079), the serving platform of Moonshot AI's Kimi, and it is supported by open-source stacks such as NVIDIA Dynamo and llm-d.

**The Core Issue: The KV Cache Handoff Bottleneck.** If a decode step runs on a separate NPU that lacks the KV cache, the model cannot generate the next token: the NPU needs the keys and values of the entire history up to that point. The system faces two choices:
- Recomputation (The Compute Burden): The decode NPU could recompute the prompt itself to rebuild the KV cache. However, this completely defeats the purpose of disaggregation, burning massive compute resources and spiking latency.
- Network Serialization (The Bandwidth Burden): The prefill NPU must serialize the multi-gigabyte KV cache tensor and send it over the network to the decode NPU.

Prefill-decode disaggregation requires the KV cache to be transmitted between prefill and decode nodes. To avoid sending large KV cache chunks and waiting for them, modern engines use **Layer-Wise Streaming (a pipeline overlap of computation and communication)**: as soon as Layer 1 finishes its prefill, it starts transmitting its KV cache chunks over the network, while Layer 2 is still computing. The destination can be the High-Bandwidth Memory (HBM) of an NPU or, in more advanced cases, a shared cluster acting as a **KV Cache Store**, e.g., [Mooncake](https://www.zhongzhuzhou.org/blog/2026-05-28-mooncake-technical-review-en/) or [LMCache](https://docs.lmcache.ai/mp/disaggregated_prefill.html).

{: style="text-align:center; font-size: small;"}
<img width="98%" height="98%" src="{{ site.assets }}/inference/imgs/disaggregation.svg"/>
