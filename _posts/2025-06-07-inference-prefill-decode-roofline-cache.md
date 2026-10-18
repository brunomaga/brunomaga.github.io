---
layout: post
title:  "Prefill vs decode, roofline, KV cache, batching and disaggregation"
categories: ["Efficient Inference"]
tags: ["LLM", "prefill", "decode", "roofline", "KV cache", "batching", "disaggregation"]
---

Inference optimization aims to generate text as fast and as cheaply as possible, with a small memory footprint and with little or no loss of quality. This matters for real-time and embedded systems, and for any service where the cost per generated token adds up. The methods below are about performance: they change how a model is served, not its weights. They start from the two phases of text generation, the prefill and the decode, and from the roofline model, which explains why the two phases use the hardware so differently. The post then covers caching, batching and scheduling, and the separation of the two phases on different GPUs. Most methods come with a short PyTorch snippet, which assumes `import torch`, `import torch.nn as nn` and `import torch.nn.functional as F`, and the snippets that run a model use the small [GPT](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer)-like model defined in the section on the KV cache.

## Prefill and decode phases

A GPT-like model generates text *autoregressively*: it reads the prompt, predicts the next token, appends this token to its input, and repeats until the answer is complete. Each new token depends on all the tokens before it, so the tokens of the answer can only be produced one at a time, while the tokens of the prompt are all known from the start. Inference therefore has two phases with very different behavior.

The **prefill** processes all the prompt tokens in one forward pass. Since these tokens are known in advance, they are processed in parallel, and the GPU performs large matrix-matrix multiplications, in which every weight read from memory serves many tokens. The prefill is therefore usually **compute-bound**: its speed is limited by the arithmetic throughput of the GPU. It ends with the first generated token, so its duration sets the **time to first token (TTFT)**.

The **decode** then generates the remaining tokens, one forward pass per token. Thanks to the key-value (KV) cache, described later, each pass processes only the newest token of each sequence. For a single sequence, the GPU then performs matrix-vector multiplications, which do little arithmetic with the data they read: each weight read from memory serves a single multiply-add, i.e. 2 FLOPs for the 2 bytes of a weight in [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format). The decode is therefore **memory-bandwidth-bound**: its speed is limited by how fast the weights arrive from memory, not by how fast the GPU computes. The duration of each decode step is the **time per output token (TPOT)**, also called the **inter-token latency (ITL)**. Since the prefill produces the first token of the answer and each decode step adds one more, an answer of $$n$$ output tokens takes a total runtime, or **time to last token (TTLT)**, of:

$$
\text{TTLT} = \text{TTFT} + \text{TPOT} \times ( n - 1)
$$

{: .picture}
<img width="96.4%" height="96.4%" src="{{ site.assets }}/inference/imgs/prefill_decode.svg"/>

The figure follows the prompt "The cat sat on" through the two phases. The prefill processes the four prompt tokens at once, writes their keys and values to the KV cache, and predicts "the". Each decode step then takes only the newest token as input: it reads the keys and values of all the earlier tokens from the cache, writes those of the new token, and predicts one more token, until "purred" completes an answer of $$n = 4$$ tokens. On the time axis, the long prefill sets the TTFT, and each of the three short decode steps adds one TPOT.

This difference between the two phases motivates most of the methods below. Almost every decode optimization does one of three things:
- it **reads fewer bytes** per token (quantization, smaller KV caches, smaller models);
- it **does more useful work per byte read** (batching, [speculative decoding](https://arxiv.org/abs/2211.17192));
- it **removes overheads** around the math (kernel fusion, CUDA graphs, better scheduling).


### Where does the compute go: MLP versus attention

How is the work of a forward pass split between the parts of a layer? Consider a model of width $$d$$, the embedding size, with an MLP of hidden size $$4d$$, as in GPT. Multiplying a token by a $$d \times d$$ weight matrix costs $$2d^2$$ FLOPs, one multiplication and one addition per weight. For each token, the linear layers of a layer therefore cost $$24d^2$$ FLOPs: $$6d^2$$ for the query, key and value projections, $$2d^2$$ for the output projection, and $$16d^2$$ for the two layers of the MLP, of sizes $$d \times 4d$$ and $$4d \times d$$. Attention adds the work of comparing the token with the $$T$$ tokens of its context: $$2Td$$ FLOPs for the scores $$QK^T$$, and $$2Td$$ for the weighted sum of the values, i.e. $$4Td$$ FLOPs in total.

The first term does not depend on the context length, while the second grows with every token of the context. Attention therefore becomes the dominant cost for long contexts, once $$4Td > 24d^2$$, i.e. for $$T > 6d$$: beyond 24,576 tokens for a width of $$d = 4096$$.

**Memory follows the same pattern.** A layer has about $$12d^2$$ weights, $$4d^2$$ in attention and $$8d^2$$ in the MLP, which a decode step reads once and shares among all the sequences of the batch. Each sequence, on the other hand, keeps the keys and values of all its tokens in its own KV cache: $$2Td$$ values per layer with multi-head attention, a number that grows with every generated token. The KV caches of a batch therefore outgrow the weights once the batch holds more than $$6d$$ tokens in total, since $$6d$$ tokens hold $$2 \times 6d \times d = 12d^2$$ values per layer. From then on, reading the KV caches dominates the memory traffic and the runtime of each decode step.

## Roofline model

Whether a program is limited by compute or by memory can be read from a simple picture, the [roofline model](https://en.wikipedia.org/wiki/Roofline_model). The model describes a program by its *arithmetic intensity* $$I$$, the number of FLOPs it performs per byte moved between memory and the processor, and bounds its performance $$P$$ by two limits of the hardware:

$$
P \le \min(P_\text{peak}, B_\text{peak} \times I)
$$

where $$P_\text{peak}$$ is the peak compute performance and $$B_\text{peak}$$ the peak memory bandwidth. The first limit says that a program cannot compute faster than the processor. The second says that it cannot compute faster than its data arrive: at a bandwidth of $$B_\text{peak}$$ bytes per second, with $$I$$ FLOPs per byte, the data feed at most $$B_\text{peak} \times I$$ FLOPs per second. A roofline chart plots the arithmetic intensity, in FLOPs/byte, on the x-axis, and the achieved performance, in GFLOPs/s, on the y-axis, both on a logarithmic scale.

{: .picture}
<img width="99%" height="99%" src="{{ site.assets }}/inference/imgs/roofline.svg"/>

{: .caption}
A roofline plot, on a log-log scale. Below the ridge point, the roof rises with the arithmetic intensity at the peak memory bandwidth; beyond it, the roof is flat at the peak performance. App<sub>1</sub> is memory-bound. App<sub>2</sub> is compute-bound and runs below the peak performance. The arrows show how an application at ❌ moves right (more FLOPs per byte), moves up (closer to the roof), or gains from a higher roof. Adapted from: [Roofline model, Wikipedia](https://en.wikipedia.org/wiki/Roofline_model)

On these log-log axes, the two terms of the minimum form a "roof": a slanted line for the bandwidth bound, and a flat line for the peak compute. The two lines meet at the **ridge point**, at the intensity $$P_\text{peak} / B_\text{peak}$$, which separates two regimes:
- Left of the ridge point, a program is **memory-bound**: the processor waits for data. The program gets faster only with fewer bytes moved, more FLOPs per byte, or an achieved bandwidth closer to the peak. The decode typically falls in this region.
- Right of the ridge point, a program is **compute-bound**: only faster math, or less math, helps. The prefill typically falls in this region, close to the roof.

In both regions, the vertical gap between a point and its roof, as under App<sub>1</sub> and App<sub>2</sub> in the figure, is the performance lost to other overheads, such as kernel launches, memory latencies or poor memory access patterns.

The roofline bound assumes an ideal program, in which data movement and computation overlap, so that the runtime is that of the slower of the two rather than their sum. [Double buffering](https://en.wikipedia.org/wiki/Multiple_buffering) is a common way to approach this ideal. With a single buffer, a loop alternates between loading a tile of data and computing on it, so the transfer and compute times add up. With two buffers, the program computes on one tile while it loads the next tile into the other buffer, and the two times overlap.

The arrows of the figure show how a program can get faster. A point moves **right**, i.e. performs more FLOPs per byte, through:
- a larger batch or more tokens per pass, so that each weight byte serves more tokens. For decode, this means batching, and speculative decoding, which verifies several draft tokens in one pass;
- better data reuse, with tiling in shared memory or SRAM;
- operator fusion, which avoids writing intermediate results to DRAM and reading them back;
- fewer bytes per FLOP: quantized weights and KV cache, and attention variants with a smaller KV cache, such as grouped-query attention ([GQA](https://arxiv.org/abs/2305.13245)) and multi-head latent attention ([MLA](https://arxiv.org/abs/2405.04434)). These matter most for decode, where the reads of the weights and of the KV cache dominate.

A point moves **up**, i.e. closer to its roof at the same intensity, when overheads are removed, with:
- tensor cores or other matrix units, and vectorized instructions;
- a high occupancy, and the overlap of compute and memory transfers;
- coalesced, aligned memory accesses, which bring the achieved bandwidth closer to the peak;
- fewer stalls from synchronization, bank conflicts, warp divergence and kernel launches. CUDA graphs reduce the launch overhead, which helps decode with its many small kernels.

The **compute-bound prefill** already runs near the flat roof, so its remaining gains come from two other places:
- **raising the roof**, with lower-precision tensor-core math, such as FP8, INT8 or FP4, which has a higher peak performance;
- **doing less work**, with sparse attention or smaller models. These reduce the runtime without necessarily moving the point on the plot.


### How to calculate the roofline point for an application

Placing a kernel on the plot, as the point ❌ above, takes two numbers. The $$X$$ coordinate is its arithmetic intensity, from the numbers of FLOPs and of bytes moved, which can be counted in the code or measured with hardware counters:

$$\text{Arithmetic Intensity } (X) = \frac{\text{Total FLOPs executed}}{\text{Total Bytes read and written}}$$

The $$Y$$ coordinate is its performance, which requires a measurement of the wall-clock runtime of the kernel. The factor $$10^9$$ converts FLOPs per second into GFLOPs per second:

$$\text{Performance } (Y) = \frac{\text{Total FLOPs executed}}{\text{Execution Time (Seconds)} \times 10^9} \quad \text{[GFLOPs/s]}$$

### Vector Sum Example

Consider first a loop over 1,000,000 elements that computes $$C[i] = A[i] + B[i]$$ with 32-bit single-precision floats ($$1 \text{ float} = 4 \text{ bytes}$$). Each iteration performs one addition, reads two floats and writes one, so:
  * **Total FLOPs:** 1 addition per iteration $$\times 1{,}000{,}000 = 1{,}000{,}000 \text{ FLOPs}$$.
  * **Total Bytes:** Read array $$A$$ ($$4\text{MB}$$), read array $$B$$ ($$4\text{MB}$$), write array $$C$$ ($$4\text{MB}$$) = $$12{,}000{,}000 \text{ bytes}$$.
  * **$$X$$ Coordinate:** $$1{,}000{,}000 / 12{,}000{,}000 = \mathbf{0.083 \text{ FLOPs/byte}}$$.

If the GPU runs the loop in $$0.0001 \text{ seconds}$$:
  * **$$Y$$ Coordinate:** $$1{,}000{,}000 / (0.0001 \times 10^9) = \mathbf{10 \text{ GFLOPs/s}}$$.

The roofline point ❌ of this example is therefore $$\mathbf{(0.083, 10)}$$. With one FLOP for every 12 bytes, the loop does almost no arithmetic with the data it moves, which makes it a typical memory-bound kernel.

### Matrix Multiplication (general formulation)

Matrix multiplication, which does most of the work of a transformer, is a more interesting case, because its intensity depends on the sizes of the matrices. Consider the product of an $$A \times B$$ matrix and a $$B \times C$$ matrix, with an output of shape $$A \times C$$ and $$dt$$ bytes per element. Its roofline point takes four steps: the FLOPs, the bytes, their ratio, and the runtime.

{: .picture}
<img width="36.9%" height="36.9%" src="{{ site.assets }}/inference/imgs/matmul.svg"/>

{: .caption}
**FLOPs:** $$AC$$ outputs $$\times$$ $$B$$ multiply-adds $$\times$$ 2 FLOPs $$= 2ABC$$. **Memory traffic, naive:** matrix 1 read $$C$$ times and matrix 2 read $$A$$ times, $$2ABC\text{ }dt$$. **Memory traffic, with cache:** inputs read once and output written once, $$(AB + BC + AC)\text{ }dt$$. **Arithmetic intensity:** $$2ABC / ((AB + BC + AC)\text{ }dt)$$, about $$2N / (3\text{ }dt)$$ for $$A = B = C = N$$.

**Step 1: FLOPs.** Each of the $$AC$$ elements of the output is the dot product of a row of the first matrix and a column of the second, as the figure shows, i.e. $$B$$ multiplications and $$B$$ additions. Counting each multiply-add as 2 FLOPs gives:

$$
\text{Total FLOPs} = 2ABC
$$

**Step 2: Memory traffic.** The number of bytes moved depends strongly on the implementation. A naive implementation, without data reuse, fetches every element from main memory each time it needs it. The first matrix is then loaded $$C$$ times, once for every column of the second matrix, and the second matrix $$A$$ times, once for every row of the first, for a total of $$2ABC\text{ }dt$$ bytes. An optimized implementation on an accelerator, such as a GPU with shared memory and tensor cores, loads tiles of both matrices into fast on-chip SRAM or cache, and reuses each tile for many multiply-adds. At best, each input element is then loaded once and each output element written once, for a total of $$(AB + BC + AC)\text{ }dt \text{ bytes}$$.

**Step 3: Arithmetic intensity.** The arithmetic intensity is the total number of FLOPs over the minimum memory traffic:

$$
\text{Arithmetic Intensity} = \frac{2ABC}{(AB + BC + AC)\text{ }dt}
$$

For square matrices with $$A = B = C = N$$, this is $$\frac{2N}{3\text{ }dt}$$ FLOPs per byte, which grows linearly with the size of the matrices: *matrix multiplication is compute-bound for large sizes and memory-bound for small sizes*. This explains the behavior of the two phases of inference. In decode, a linear layer multiplies the $$A$$ tokens of a step, one per sequence, by a $$B \times C$$ weight matrix, with $$A \ll B, C$$. The weights then dominate the memory traffic, and the intensity is about $$2ABC / (BC\text{ }dt) = 2A / dt$$, i.e. $$A$$ FLOPs per byte in BF16 ($$dt = 2$$). A single sequence thus gets 1 FLOP per byte, while the prefill of a long prompt reaches hundreds to thousands of FLOPs per byte.

**Step 4: Roofline point.** With a measured runtime $$\text{Time}_{\text{sec}}$$ in seconds, the performance follows as for the vector sum, and the roofline point ❌ of the product is:

$$
(X, Y) = \left( \frac{2ABC}{(AB + BC + AC)\text{ }dt}, \; \frac{2ABC}{\text{Time}_{\text{sec}} \times 10^9} \right)
$$

#### Tensor multiplication

Matrix multiplication extends to higher-dimensional tensors, for example of shapes $$A \times B \times D \times E \times F$$ and $$B \times C \times D \times E \times F$$. The extra dimensions act as batch axes: the product runs $$DEF$$ independent matrix products, which can execute in parallel. The FLOPs and the bytes both grow by the factor $$DEF$$, which therefore cancels out in the arithmetic intensity, so the $$X$$ coordinate is that of a single matrix product. The performance, on the other hand, counts $$DEF$$ times more FLOPs for the measured runtime, which gives the roofline point:

$$
(X, Y) = \left( \frac{2ABC}{(AB + BC + AC) \text{ } dt}, \; \frac{2ABC \text{ } DEF}{\text{Time}_{\text{sec}} \times 10^9} \right)
$$

#### Side note: FLOPs on forward and backward passes

The same counting applies to training, as [the scaling book](https://jax-ml.github.io/scaling-book/transformers/) shows in more detail. Training computes not only the output of the forward pass, but also the gradients of the backward pass. Consider a layer $$Y = XW$$, with an input $$X$$ of size $$A \times B$$, weights $$W$$ of size $$B \times C$$ and an output $$Y$$ of size $$A \times C$$. The backward pass receives the gradient $$\frac{\partial L}{\partial Y}$$ of the loss $$L$$ from the next layer, or from the loss itself for the last layer, and the chain rule turns it into the gradient of the weights:

$$\frac{\partial L}{\partial W} = X^T \left(\frac{\partial L}{\partial Y}\right)$$

Despite the calculus notation, the hardware computes this gradient with an ordinary matrix multiplication, of $$X^T$$ by the incoming gradient. It therefore costs as many FLOPs as any other matrix product of the same dimensions, $$2ABC$$, here contracted over the dimension $$A$$. The backward pass must also pass a gradient on to the previous layer, the gradient of the input, which is a second matrix product:

$$\frac{\partial L}{\partial X} = \left(\frac{\partial L}{\partial Y}\right) W^T$$

Since $$\frac{\partial L}{\partial Y}$$ has dimensions $$A \times C$$, this product is contracted over $$C$$, and it costs $$2ABC$$ FLOPs too. When both gradients are needed, the backward pass therefore costs $$4ABC$$ FLOPs, twice the forward pass, and a full training step costs $$6ABC$$. For a linear layer with $$BC$$ weights that processes $$A$$ tokens, this gives the common estimate of $$6 \times \text{number of weights}$$ FLOPs per token for training, against $$2 \times \text{number of weights}$$ for inference, for the matrix products only. Finally, the optimizer step of stochastic gradient descent, $$W \leftarrow W-\eta g$$, with a learning rate $$\eta$$ and gradients $$g$$, adds $$2BC$$ FLOPs: one multiplication and one subtraction per weight.

### How to draw the Roofline for a GPU Architecture

The roof of a GPU needs only two values from its specification sheet, for the chosen precision: the peak compute performance $$P_{\text{peak}}$$, in GFLOPs/s, and the peak memory bandwidth $$B_{\text{peak}}$$, in GB/s, whose ratio is the ridge point. For an NVIDIA A100 40GB in FP32, $$P_{\text{peak}} = 19{,}500 \text{ GFLOPs/s}$$ and $$B_{\text{peak}} = 1{,}555 \text{ GB/s}$$, so the ridge point is at $$19{,}500 / 1{,}555 \approx \mathbf{12.54 \text{ FLOPs/byte}}$$. The vector sum above, at 0.083 FLOPs/byte, lies far to the left of it and is memory-bound: its roof is at $$1{,}555 \times 0.083 \approx 130$$ GFLOPs/s, 13 times its measured performance. The ridge point also depends on the precision. In BF16, the tensor cores of the A100 reach $$P_{\text{peak}} = 312{,}000$$ GFLOPs/s with the same bandwidth, and the ridge point rises to about 200 FLOPs/byte. The linear layers of a decode step, at about $$A$$ FLOPs per byte for $$A$$ tokens, therefore stay memory-bound up to about 200 tokens per step, and until then, a larger batch makes each step do more work in about the same time.

On log-log axes, the memory ceiling $$y = B_\text{peak} \times x$$, left of the ridge point, is a line of slope 1, and the compute ceiling $$y = P_\text{peak}$$, right of it, is flat. The following code plots the roofline of the A100 in FP32, with the point of the vector sum:

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

Text generation does much of its work more than once: every decode step needs the keys and values of all the earlier tokens, many requests start with the same instructions, and many users ask the same questions. Caching avoids this repeated work at three layers of the stack, a view adapted from [this guide](https://machinelearningmastery.com/the-complete-guide-to-inference-caching-in-llms/):

- **KV caching** stores the attention keys and values of the tokens of a request, so that the model does not recompute them at every decode step. All serving engines use it.
- **Prefix caching**, also called **prompt caching** or **context caching**, extends KV caching across requests. When several requests share the same leading tokens, such as a system prompt, a reference document, few-shot examples or the previous turns of a conversation, the keys and values of this shared prefix are computed once and reused.
- **Semantic caching** is a cache in the application, which stores complete pairs of queries and answers and retrieves them by meaning. When a similar enough query was answered before, it skips the model call entirely.

The three layers complement each other rather than compete. The KV cache reuses work within one request and is always on. Prefix caching reuses work across requests, is exact, and pays off when requests share long prefixes. Semantic caching reuses whole answers, is approximate, and pays off when many queries repeat with the same meaning.

### KV cache

In the attention layer, the query of the new token attends to the keys and values of all the previous tokens. In a causal model, the keys and values of a past token never change, because token $$i$$ depends only on itself and on the tokens before it. Without a cache, each decode step nevertheless recomputes the queries, keys and values of the whole sequence, as the top row of the figure shows, although only the last row of the output is new. With a cache, each step computes the query, key and value of the new token only, appends the new key and value to the cache, and attends over all the cached entries, as the bottom row shows. Queries are not cached, because the query of a past token was needed only to compute the output at its own position. Step $$t$$ of a generation thus processes one token against a cache of $$t$$ entries, instead of $$t$$ tokens against each other, which removes a factor $$n$$ from the total cost of generating $$n$$ tokens: $$O(n)$$ instead of $$O(n^2)$$ in the projections and the MLP, and $$O(n^2)$$ instead of $$O(n^3)$$ in attention.

{: .picture}
<img width="81.8%" height="81.8%" src="{{ site.assets }}/inference/imgs/kv_cache.svg"/>

{: .caption}
**Without KV cache:** generating $$n$$ tokens costs $$O(n^2)$$ in the projections and the MLP, and $$O(n^3)$$ in attention. **With KV cache:** generating $$n$$ tokens costs $$O(n)$$ in the projections and the MLP, and $$O(n^2)$$ in attention.

The price of the cache is memory. For every token, every layer stores one key vector and one value vector per key-value head, hence the factor 2 in:

$$
\text{KV cache size} = 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times \text{bytes per value} \times \text{sequence length} \times \text{batch size}
$$

[Llama 3](https://en.wikipedia.org/wiki/Llama_(language_model)) 8B, for example, has 32 layers and 8 key-value heads of size 128. In BF16, its KV cache takes $$2 \times 32 \times 8 \times 128 \times 2$$ bytes, i.e. 128 KiB per token, or 1 GiB for a sequence of 8,192 tokens. This memory, not compute, often limits how many sequences fit in a batch. Many methods therefore shrink the KV cache, such as GQA, MLA and quantization, or waste less of it, such as PagedAttention below.

Two implementation details matter in practice. The first is how the cache grows. A **dynamic cache** concatenates the new keys and values to the cache at every step, which is simple, but every concatenation allocates a new tensor and copies the whole cache into it. A **static cache** instead preallocates a buffer for the maximum sequence length, and writes each new entry in place.

The second detail concerns the positions of the tokens. A model with learned absolute positional embeddings, such as GPT-2, has a context window of `seqlen` tokens. When a sequence grows beyond `seqlen` tokens, the model without a cache slides its window at every step, and re-encodes the last `seqlen` tokens at positions 0 to `seqlen-1`. A KV cache cannot do this, because its keys and values were computed at their original positions, and with the context available at that time. The model with a cache therefore matches the model without a cache only while the sequence has at most `seqlen` tokens.

The following code adds a KV cache to a small GPT-like model, and generates text with and without it. Two choices in the code follow from the details above. The cache keeps only the last `seqlen` keys and values, and once the window is full, every new token gets the last position, `seqlen-1`. Generation can then continue past the context window, but its output differs from that of the model without a cache. Relative positional encodings, such as [RoPE](https://arxiv.org/abs/2104.09864), suit a sliding window better, but dropping the oldest tokens still changes the output. The model also accepts several new tokens on top of a cache, as prefix caching, chunked prefill and speculative decoding require. For this, the causal mask is shifted, so that each new token attends to all the cached tokens and to the new tokens up to itself:


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

{: .picture}
<img width="59.3%" height="59.3%" src="{{ site.assets }}/inference/imgs/static_cache.svg"/>

The code above uses a dynamic cache, and the figure compares it with a static one. For every new token, the dynamic cache allocates a new tensor of $$P+1$$ rows, where $$P$$ is the number of cached tokens, and copies the $$P$$ cached rows into it. The static cache allocates a buffer of `max_len` rows once, writes the key or the value of the new token in place, at position $$P$$, and returns a view of its first $$P+1$$ rows, without a copy. The following class implements a static cache for one layer:

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

The two caches trade memory for copies. A dynamic cache uses memory only for the tokens so far, but its copies add up to $$O(n^2)$$ for $$n$$ tokens, since step $$t$$ copies $$t$$ entries. A static cache copies nothing, but it reserves the memory of `max_len` tokens from the start, whether the sequence needs them or not.

#### Static shapes for CUDA graphs

A decode step of a small model runs many small kernels, and the time the CPU needs to launch them can be a large part of the step. A CUDA graph removes this overhead: it records the kernels of a decode step once, and then replays them all with a single launch. The recording, however, fixes the shapes and the memory addresses of all tensors, so these must be the same at every step. The static cache above avoids the copies, but the keys and values it returns still grow by one token per step. The decode step must therefore attend over the *whole* preallocated buffer, and mask the positions that are not filled yet. For the same reason, the position of the new token is passed as a tensor, so that it is an input of the graph and not a constant recorded in it.

The following code wraps the GPT model above with a static cache for all layers. Its decode step is compiled with `torch.compile(mode="reduce-overhead")`, which records and replays it as a CUDA graph on a GPU. Because of the learned positional embeddings, `max_len` must not exceed the context window `seqlen`:

```python
class StaticCacheGPT(nn.Module):
    """ The GPT model above with a static KV cache for all layers, preallocated for max_len tokens. Attention runs over
        the whole buffer, with a mask for the positions not filled yet, so all tensors have the same shape at every
        decode step, as required by CUDA graphs """

    def __init__(self, model, batch_size, max_len):
        super().__init__()
        self.model = model
        attn, w = model.blocks[0].attn, model.head.weight
        shape = (len(model.blocks), batch_size, attn.num_heads, max_len, attn.head_dim)  # [L, B, H, max_len, h]
        self.register_buffer('k', torch.zeros(shape, dtype=w.dtype, device=w.device))
        self.register_buffer('v', torch.zeros_like(self.k))
        self.register_buffer('positions', torch.arange(max_len, device=w.device))

    def forward(self, tokens, input_pos):
        """ tokens: [B, S], the new tokens, at positions input_pos: [S] -> logits: [B, S, V] """
        m, (B, S) = self.model, tokens.shape
        x = m.token_embedding(tokens) + m.position_embedding(input_pos)  # [B, S, E]
        mask = input_pos[:, None] >= self.positions[None, :]            # [S, max_len], True where a query can attend
        for i, block in enumerate(m.blocks):
            a, H, h = block.attn, block.attn.num_heads, block.attn.head_dim
            y = block.ln1(x)
            q, k, v = (proj(y).view(B, S, H, h).transpose(1, 2) for proj in (a.wq, a.wk, a.wv))  # [B, H, S, h]
            self.k[i, :, :, input_pos] = k  # write the new keys and values in place
            self.v[i, :, :, input_pos] = v
            attn = F.scaled_dot_product_attention(q, self.k[i], self.v[i], attn_mask=mask)  # [B, H, S, h]
            x = x + a.out(attn.transpose(1, 2).reshape(B, S, H * h))
            x = x + block.mlp(block.ln2(x))
        return m.head(m.norm(x))


@torch.no_grad()
def generate_static(model, prompt, n_tokens, decode_step):
    """ greedy generation: prefill with the model, then one call to decode_step per new token """
    P = prompt.size(1)
    logits = model(prompt, torch.arange(P, device=prompt.device))  # prefill
    tokens = [logits[:, -1:].argmax(dim=-1)]
    for pos in range(P, P + n_tokens - 1):
        logits = decode_step(tokens[-1], torch.tensor([pos], device=prompt.device))  # decode
        tokens.append(logits[:, -1:].argmax(dim=-1))
    return torch.cat([prompt] + tokens, dim=1)


model_static = StaticCacheGPT(model, batch_size, max_len)
decode_step = torch.compile(model_static, mode="reduce-overhead", fullgraph=True)
tokens = generate_static(model_static, prompt, n_tokens, decode_step)
```

The prefill runs once per request, with a different shape, so it is not compiled here. The static shapes have a cost: the attention of every decode step now costs as much as that of the last step, since it covers all `max_len` positions. For small models and batches, this extra work is small next to the launch overhead that the graph removes.


#### PagedAttention: memory management for the KV cache

A static cache suits a single sequence, but a server runs many requests at once, and the final length of each sequence is unknown in advance. A naive engine therefore reserves a contiguous KV buffer of the maximum length for every request, and most of that memory stays empty: the authors of [vLLM](https://arxiv.org/abs/2309.06180) measured that only 20.4% to 38.2% of the KV cache memory of existing systems held actual tokens. The rest was reserved for future tokens, or lost to fragmentation. Their PagedAttention solves this problem the way operating systems do for the memory of processes, with the paging of [virtual memory](https://en.wikipedia.org/wiki/Virtual_memory). The KV cache is split into fixed-size *blocks*, the pages, which are allocated on demand, and a *block table* maps the logical blocks of each sequence to physical blocks anywhere in GPU memory. The attention kernel then reads the blocks of each sequence through its block table.

Paging also lets sequences share blocks, with [copy-on-write](https://en.wikipedia.org/wiki/Copy-on-write): a shared block is copied only when one of the sequences writes to it. Prefix caching, parallel sampling and [beam search](https://en.wikipedia.org/wiki/Beam_search) share their common prefix this way.

{: .picture}
<img width="98.9%" height="98.9%" src="{{ site.assets }}/inference/imgs/paged_attention.svg"/>

{: .caption}
The attention kernel reads the blocks of each sequence through its block table, and a sequence wastes at most 3 slots, in its last block. Inspired by: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

In the figure, the blocks hold 4 tokens. Sequence A holds 6 tokens in two logical blocks, which its block table maps to the physical blocks 5 and 1, and sequence B holds 9 tokens in three logical blocks, mapped to the physical blocks 2, 7 and 4. The physical blocks of a sequence need not be contiguous nor in order, and the free blocks 0, 3 and 6 are available to any sequence. Since only the last block of a sequence is partly empty, almost no memory is wasted, and more sequences fit in a batch: vLLM reported 2 to 4 times the throughput of FasterTransformer and Orca, at the same latency. Compared with a static cache, PagedAttention reserves memory one block at a time instead of `max_len` tokens at once, at the cost of an indirection through the block table in the attention kernel.

The following code implements a paged KV cache and the attention of one decode step over it:

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


### Prefix caching

Many requests start with the same tokens: a system prompt, the definitions of tools, a long document that users ask several questions about, or the history of a conversation. The keys and values of these tokens are the same in every request, so computing them again is redundant work. Prefix caching keeps the keys and values of these tokens after a request finishes, and the next request that starts with the same tokens skips their prefill and processes only its new part. Its first token arrives much sooner, i.e. its TTFT drops, and the GPU is free for other work.

**How the lookup works.** How does an engine find out that a new prompt starts with tokens it has already processed? vLLM splits the KV cache into fixed-size blocks, typically of 16 tokens, and labels each full block with a hash of its token IDs combined with the hash of the previous block. Because of this chain, the hash of a block identifies not only the tokens of the block, but the whole prefix up to its end. A new request then hashes its prompt block by block from the start, reuses the cached blocks up to the first miss, and prefills only the rest.

{: .picture}
<img width="93%" height="93%" src="{{ site.assets }}/inference/imgs/prefix_caching.svg"/>

{: .caption}
Automatic prefix caching, with blocks of 4 tokens (16 in vLLM). The hash of each block chains the hash of the previous block with the tokens of the block. Request 2 reuses the first two blocks of request 1 and prefills 4 tokens. Request 3 reuses only the first block: its last block holds the same tokens as that of request 1, but its hash differs because its second block differs, so request 3 prefills 8 tokens.

Request 3 of the figure shows why the hashes must be chained. Its last block holds the same tokens as that of request 1, but these tokens follow a different second block, so their keys and values differ too. A hash of the tokens of the block alone would wrongly report a hit, while the chained hash correctly reports a miss.

vLLM calls this mechanism Automatic Prefix Caching, and it needs little extra memory, because it builds on PagedAttention. The KV cache is already divided into blocks, so requests share the cached blocks through their block tables, without any copy. The memory cost is a hash table that maps block hashes to physical blocks, plus the cached blocks themselves. These are not freed when their request finishes: they stay in memory, marked as reusable, until new requests need the space, and they are then evicted, [least recently used](https://en.wikipedia.org/wiki/Cache_replacement_policies) first. [SGLang](https://arxiv.org/abs/2312.07104) introduced a different design, RadixAttention, which stores the cached sequences in a [radix tree](https://en.wikipedia.org/wiki/Radix_tree), a compressed prefix tree. A new request reuses the longest prefix it finds in the tree, and when memory runs out, the least recently used leaves are evicted first. The two designs differ in their granularity: the tree matches prefixes token by token, while the hash chain of vLLM matches whole blocks, so vLLM recomputes the last tokens of a shared prefix that do not fill a whole block.

**Why the prefix must match exactly.** It is tempting to look for *similar* prompts instead of identical ones, for example with a vector database, and to reuse their cache. This does not work for attention states. The key and value of token $$i$$ depend on all the tokens before it, through every layer of the model, so two prompts that differ in a single early token have different keys and values for every token after that difference, even if they have the same meaning. Reusing the cache of a similar prompt would silently give the model a different context than the one the user sent. An exact match, on the other hand, makes prefix caching *lossless*: the output is the same as without the cache. It also keeps the lookup cheap, a hash or tree search instead of an embedding and a vector search. Similarity search does make sense one level up, where whole answers are reused, and this is the idea of semantic caching, described below.

**Prefixes, not substrings.** For the same reason, only a prefix, i.e. the leading tokens of a prompt, can be reused: a paragraph in the middle of a prompt has different keys and values depending on what precedes it. Matching is also done on token IDs, not on the raw text, so the same text with a different tokenization, e.g. with an extra space, is a miss. Research systems do reuse other chunks, approximately. [CacheBlend](https://arxiv.org/abs/2405.16444) precomputes the cache of each retrieved document separately, and recomputes a small fraction of the tokens to restore the attention between the documents. [Prompt Cache](https://arxiv.org/abs/2311.04934) reuses precomputed modules of prompts written in a structured template.

**What is kept.** The cache holds the keys and values of every token processed so far, in every layer: the prompt, and also all the generated tokens, since every new token attends to all of them. Caching the generated tokens makes multi-turn chat fast, because the prompt of the next turn is the previous prompt, plus the previous answer, plus the new message, and all of it except the new message is already in the cache. Since any change early in the prompt invalidates the cache of everything after it, a prompt should place the static content first (the system prompt, tools and documents) and the variable content last (the question, timestamps).

The following code implements the block lookup, and prefix caching on top of the KV cache. The keys and values of a prefix shared by all requests are computed once and reused by every request, with the same outputs as without prefix caching:

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

Prefix caching saves the prefill of shared tokens, but every request still runs the model. Many applications, however, receive the same questions many times, in different words, and for these, the whole answer could be reused. This is the idea of a semantic cache. It sits in front of the model, in the application, and for every incoming query, it:

1. computes an [embedding](https://en.wikipedia.org/wiki/Sentence_embedding), a vector that represents the meaning of the query;
2. searches a vector index for the most similar past query;
3. if the similarity reaches a threshold, returns the stored answer of that query without calling the model;
4. otherwise, calls the model, and stores the new pair of embedding and answer.

{: .picture}
<img width="93.9%" height="93.9%" src="{{ site.assets }}/inference/imgs/semantic_caching.svg"/>

{: .caption}
A semantic cache. The vector index finds the past query whose embedding is closest to that of the new query. If their similarity reaches the threshold, as it may for a paraphrase such as "What is the price of corn in Rome?", the cache returns the stored answer of that query. Otherwise, the LLM answers, and the embedding and the answer are stored.

**How embeddings are created.** The embeddings come from a separate, much smaller *embedding model*, typically a transformer encoder of the [Sentence-BERT](https://arxiv.org/abs/1908.10084) family (e.g. all-MiniLM-L6-v2), E5, BGE or GTE, or from an embedding API. The encoder is trained with *contrastive learning*: pairs of texts with the same meaning, such as paraphrases or a question and its answer, are pulled together, while the other texts of the batch are pushed apart. After training, the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of two embeddings measures how close their meanings are. The raw hidden states of a generative LLM make poor embeddings, because the LLM was trained to predict the next token, not to compare meanings. An LLM fine-tuned in the same contrastive way, however, makes a good embedding model, as E5-Mistral and NV-Embed show.

**Where embeddings are stored.** Any [vector database](https://en.wikipedia.org/wiki/Vector_database) or index works. [FAISS](https://github.com/facebookresearch/faiss) is a common choice: an in-process library for CPUs and GPUs, with exact indexes, such as `IndexFlatIP`, which searches by inner product, equal to the cosine similarity for unit vectors, and approximate indexes, such as IVF, [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) and product quantization, which search millions of vectors or more. FAISS is a library and not a database: it stores only the vectors and their integer IDs, and the application keeps the answers and handles their expiration. Vector databases such as Milvus, Qdrant, Weaviate, pgvector or Redis store metadata with the vectors and run as services. Tools such as [GPTCache](https://github.com/zilliztech/GPTCache) implement the whole cache loop, with FAISS as one of their backends.

**Risks.** A semantic cache is approximate, so it can return a wrong answer: "how do I delete a file in Python?" and "how do I delete a folder in Python?" are similar sentences with different answers. Answers also become outdated, and queries that depend on the conversation history or on the user must not share answers. In practice, the threshold is therefore tuned on real traffic, the cache is scoped per user or tenant, the entries expire after some time, and a more precise re-ranking model can confirm each hit. The strict alternative is an **exact-match cache**, which maps a hash of the full request, i.e. the prompt, the model and the sampling parameters, to the response. It never returns the answer of another request, but it hits only identical requests.

The following code implements a minimal semantic cache with FAISS and a pre-trained sentence encoder, in front of a function `generate_fn(query)` that calls the LLM. It also returns the similarity of each query to the closest cached query, which helps to choose the threshold:

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

## Batching

The roofline analysis showed that a decode step for a single sequence leaves most of the compute of the GPU idle, and batching puts this compute to use. Since decode is memory-bound, a decode step for 32 sequences with short contexts takes about as long as a step for one sequence: the weights are read once and used 32 times. The throughput therefore grows almost linearly with the batch size, until the step becomes compute-bound or the KV caches fill the GPU memory. With long contexts, the gain is smaller, because each sequence also reads its own KV cache, which is not shared. The difficulty of batching lies elsewhere: requests arrive at different times, and they generate different numbers of tokens.

### Static and continuous batching

The simplest approach, **static batching**, groups the requests in batches and runs each batch until all its requests finish. Prompts of different lengths are padded to the same length. Sequences that finish early keep their slot and generate tokens that are thrown away, and new requests wait until the whole batch is done. Static batching therefore spends compute and memory on padding tokens and on sequences that have already finished, and this waste is small only when the requests of a batch have similar lengths.

**Continuous batching**, introduced by [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) as *iteration-level scheduling*, removes this waste by making the scheduling decision at every decode step instead of once per batch. As soon as a sequence finishes, it leaves the batch, and a waiting request takes its slot. The GPU stays busy, and new requests no longer wait for the longest sequence of the previous batch, whatever the mix of lengths and arrival times. The engine must, however, track each sequence separately, by keeping the tokens of every running request apart, and rebuild the batch at every step. This way, a new request never sees the tokens of the previous occupant of its slot. The top two schedules of the figure in the next section compare the two approaches on the same requests.

The following code implements both. They produce the same outputs, but continuous batching needs fewer steps, since its slots spend fewer steps idle or on discarded tokens. For brevity, every step processes the whole sequences, without a KV cache, and pads the batch again on the right. With causal attention, the real tokens never attend to the padding that follows them, so no padding mask is needed, and each sequence reads its next-token prediction at its own last position. Padding on the left would instead require a padding mask and shifted positions:

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

Continuous batching has a side effect. A waiting request takes a free slot as soon as one is available, so its prefill runs in the same step as the decode of the other requests. The prefill of a short prompt costs about as much as a decode step, but the prefill of a long prompt can take many times longer, and all the other requests of the batch wait for it: their output pauses, and their TPOT spikes. **Chunked prefill**, introduced by [SARATHI](https://arxiv.org/abs/2308.16369), solves this problem by splitting a long prompt into smaller pieces, the chunks, and processing one chunk per step. Every step then takes about the same time, and the other requests keep generating one token per step.

{: .picture}
<img width="99.4%" height="99.4%" src="{{ site.assets }}/inference/imgs/continuous_chunked_batching.svg"/>

{: .caption}
The same 6 requests (R1 to R6) on 3 slots, with static batching, continuous batching, and continuous batching with chunked prefill. The width of a step is its duration: a decode step and the prefill of a short prompt take one time unit, and the prefill of the long prompt of R4 takes three. Static batching runs the requests in groups of 3 until the longest request of each group finishes, and pads the prompts of a group to the longest one. Continuous batching gives a free slot to the next request at the next step, but the prefill of R4 stalls R1 and R3. With chunked prefill, R4 processes its prompt in 3 chunks of one time unit each, while R1 and R3 continue decoding.

The time axis of the figure shows the effect of each approach. The same requests take 11 time units with static batching, 9 with continuous batching, which fills the slot of R2 at once, and 8 with chunked prefill. More important than the total, the decode of R1 and R3 never stops with chunked prefill, while continuous batching makes them wait three time units for one token.

Why does chunking give the same result as a single forward pass? Each chunk attends to the KV cache of the earlier chunks and to itself, with a causal mask. A token attends only to the tokens before it, which are all either in the cache or in its own chunk, so it sees exactly the context it would see in a single pass. The KV cache is thus filled chunk by chunk, and attention kernels such as FlashAttention compute the attention of each chunk over the cache directly.

A larger example shows the effect on latency. Consider a prompt of 8,000 tokens that arrives while 50 other sequences are decoding:
- Without chunking, one step processes all 8,000 prompt tokens plus the 50 decode tokens. This step takes far longer than a decode step, and the output of all 50 sequences pauses meanwhile.
- With a budget of 512 tokens per step, each step processes the 50 decode tokens plus 462 prompt tokens, and the prompt finishes in 18 steps, the last of which holds the remaining 146 prompt tokens. The other sequences get a token at every step.

{: .picture}
<img width="99.6%" height="99.6%" src="{{ site.assets }}/inference/imgs/chunked_prefill.svg"/>

The budget sets the balance between the two kinds of requests: a smaller budget lowers the TPOT of the decoding sequences, and a larger one lowers the TTFT of the new request.

Chunking relies on the causal mask. In a bidirectional model, such as an encoder, the earlier tokens also attend to the later ones: the keys and values of a chunk then change with every new chunk, in every layer after the first, so the prompt cannot be processed piece by piece. An encoder also has no decode phase, so there are no decoding requests to protect from a long prefill.

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

With continuous batching and chunked prefill, a step runs the prefill chunks of some requests and the decode of others, and these have different lengths: chunks of hundreds of tokens next to sequences of a single token. Padding them into a rectangular batch would spend much of the compute on padding tokens. Engines therefore concatenate the sequences instead: **each step is one forward pass over a flat list of the tokens of all running sequences**. This is ragged, or packed, batching, and each part of the model handles it in its own way:
- The linear layers, i.e. the projections and the MLP, treat each token independently of the sequence it belongs to, so they run as one large matrix multiplication over all the tokens of the step.
- Attention is the only operation that mixes tokens. Variable-length kernels, such as the varlen mode of [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashInfer](https://arxiv.org/abs/2501.01005) and PagedAttention, use per-token metadata (sequence ID, position, KV block table) so that each token attends only to its own sequence. The mask is block-diagonal and causal, and the kernels skip the tiles outside the diagonal blocks, instead of computing the full square matrix.
- The output layer computes the logits only at the last token of each sequence, found with the metadata, and samples the next token of each sequence from them.

{: .picture}
<img width="99.7%" height="99.7%" src="{{ site.assets }}/inference/imgs/ragged_batching.svg"/>

The figure shows three sequences of 4, 1 and 3 tokens. Padded into a $$3 \times 4$$ batch, they spend 4 of the 12 slots on padding. Packed into a single sequence of 8 tokens, they spend none, and each token keeps its position within its own sequence, so the positions restart at 0 at the start of every sequence. The mask lets each query attend only to itself and to the earlier keys of its own sequence, which gives one causal triangle per sequence on the diagonal, and the next tokens are read at the last token of each sequence.

The following code packs sequences of different lengths into one flat sequence, and returns the position of every token, the block-diagonal causal mask, and the indices of the last tokens:

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

Mixing prefill and decode in one step also raises the utilization of the GPU, as the roofline model explains. Decode tokens alone are memory-bound and leave most of the compute idle. Prefill chunks in the same step share the weight reads of the decode tokens, which moves the step toward the ridge point, so that the step does much more useful work for almost the same memory traffic. [Sarathi-Serve](https://arxiv.org/abs/2403.02310) calls this *stall-free batching*.

At every step, a scheduler decides which requests run, how many tokens each one contributes, which waiting requests to admit, and which ones to preempt. Two limits bound these decisions:
- The **token budget** of a step. The scheduler adds one token for every decoding sequence first, to keep the TPOT low, and then fills the rest of the budget with prefill chunks.
- The **free KV cache blocks**. When they run out, the scheduler preempts a sequence: it frees the blocks of that sequence, and later either recomputes them or swaps them back in from CPU memory.

The CPU overhead of each step shrinks too: CUDA graphs run the decode-only steps, and the CPU schedules the next step while the GPU runs the current one, as in SGLang.

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

Chunked prefill bounds the interference between the two phases, but it does not remove it. Prefill is compute-bound and decode is memory-bound, so they still compete when they share GPUs, and each phase prefers a different batch size and parallelism. Disaggregated serving goes one step further and runs the two phases on separate pools of GPUs. Prefill workers compute the KV cache and the first token of each prompt, and send them to decode workers, which generate the remaining tokens. Each pool is then sized and tuned for its own latency target: the TTFT for the prefill, and the TPOT for the decode. This is the design of [DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677) and [Mooncake](https://arxiv.org/abs/2407.00079), the serving platform of Moonshot AI's Kimi, and open-source stacks such as NVIDIA Dynamo and llm-d also support it. The two approaches make different trade-offs: chunked prefill keeps both phases on the same GPUs and bounds their interference with the token budget, while disaggregation removes the interference, at the cost of moving the KV cache between GPUs and of balancing the sizes of the two pools.

**The KV cache handoff.** The decode GPU cannot generate the next token without the keys and values of the whole prompt, and there are two ways to provide them. The decode GPU could recompute them, but this would repeat the prefill and defeat the purpose of disaggregation. The prefill GPU therefore sends the KV cache instead, e.g. 1 GiB for 8,192 tokens of Llama 3 8B, over [NVLink](https://en.wikipedia.org/wiki/NVLink) or over the network with [RDMA](https://en.wikipedia.org/wiki/Remote_direct_memory_access). Waiting for the whole cache would still delay the first decode step, so engines hide the transfer with **layer-wise streaming**, a pipeline that overlaps computation and communication: the KV cache of a layer is sent as soon as the prefill of that layer finishes, while the next layers compute. The destination is the [HBM](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) of a decode GPU, or a shared **KV cache store**, such as [Mooncake](https://www.zhongzhuzhou.org/blog/2026-05-28-mooncake-technical-review-en/) or [LMCache](https://docs.lmcache.ai/mp/disaggregated_prefill.html), which keeps the KV caches of many requests in CPU memory and on SSDs.

{: .picture}
<img width="99.6%" height="99.6%" src="{{ site.assets }}/inference/imgs/disaggregation.svg"/>

{: .caption}
The prefill pool computes the KV cache and the first token of each request, and sends them to the decode pool. In the layer-wise streaming of the KV cache, the transfer of each layer overlaps with the compute of the next layers, and only the transfer of the last layer delays the decode. Inspired by: [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)

With four layers, as in the figure, streaming lets the decode start one layer transfer after the end of the prefill, while without it, the decode would wait for the transfers of all four layers.
