---
layout: post
title:  "Sequence and context parallelism with Ulysses and Ring attention"
categories: ["Distributed Training"]
tags: ["sequence parallelism", "context parallelism", "Ulysses", "Ring attention", "attention"]
---

This post describes and implements context and sequence parallelism, two techniques that parallelize computation along the sequence-length dimension. Sequence parallelism is one more dimension of model parallelism, alongside data, tensor, pipeline, context and expert parallelism.

Many model inputs and activations include an extra dimension that represents an (un)ordered sequence of tokens. Examples include text and other temporal data, with shape `(B, S, E)` for batch size `B`, sequence length (or token count) `S` and embedding size `E`. Videos have shape `(B, C, T, H, W)` for `C` channels, `T` frames, and `HxW` pixels per frame. Attention scores and probabilities have shape `(B, H, S, S)` for `H` heads and sequences of length `S`. In these cases, computation can be parallelized along the sequence dimension `S` (or `T` for videos). In the notation for text, sequence parallelism over `P` processes requires local storage of shape `(B, S/P, E)` per process. This post implements two existing techniques for sequence parallelism: Ulysses and Ring Attention. The use case is a model built from a sequence of transformer blocks, where each block is a multi-head attention (MHA) module followed by a feed-forward network (FFN), with some normalization and skip connections.

Two remarks before continuing:

- This post calls both algorithms **sequence parallelism**, although several sources use the term **context parallelism**. Strictly speaking, **sequence parallelism** splits the sequence (token) dimension across GPUs so that each device processes a different chunk of tokens, mainly to scale activation memory/compute with longer sequences. **Context parallelism** splits the attention context (keys/values) across GPUs, so that each device holds a slice of the KV context and computes partial attention that is combined via communication, mainly to scale attention/KV-cache memory for very large context windows. Many implementations (including the ones in this post) use both: the input tokens are split, and the KV elements needed by attention are also distributed.
- The focus is on **training and inference prefill**. Sequence/context parallelism for the **inference decode** step is a different setting: decode processes one token at a time and typically shards the KV cache rather than the input tokens.

## Data loader setup for sequence parallelism

Sequence parallelism first requires a data loader that loads data at the token level, matching the hybrid data- and sequence-parallel setup. This is done by configuring a `DataLoader` and a `DistributedSampler` that assign chunks of sequences (instead of full sequences) to the data loader workers.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/spdistributedsampler.svg"/>

{: style="text-align:center; font-size: small;"}
Data loading for hybrid data parallelism (DP) and sequence parallelism (SP) on 4 color-coded processes/GPUs and a dataset of 8 samples. First row: all 4 processes run distributed data parallelism. Second row: a custom `DistributedSampler` that yields chunks of sequences enables a hybrid data- and sequence-parallel execution with 2 groups of 2 sequence-parallel processes. Third row: no data parallelism; all 4 processes run sequence parallelism on the same sample.

Most layers (such as the feed-forward network) operate on the last (embedding) dimension of the input, so they run sequence-parallel training/prefill without changes. The attention layer is the exception and is covered next.

## Distributed attention for context parallelism

During training and inference, the runtime of most layers scales well with the sequence parallelism degree. The attention layer does not, because each GPU stores only a subset of the KV elements to reduce the activation memory footprint.

To compute attention, each process uses its own chunk of queries but **needs the keys and values of the full sequence**. Consider the attention head computation $$softmax \left(QK^T \right)V$$, where all tensors are shaped `(B, S, E)`. If each process holds a subset of rows of $$Q$$ as $$Q_p$$ with shape `(B, S/P, E)`, it needs to access all elements of $$K^T$$ and $$V$$ to perform the dot-product $$Q_p K^T$$, the row-wise $$softmax$$, and the dot-product with $$V$$, resulting in an attention output of shape `(B, S/P, E)` per process:

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="{{ site.assets }}/distributed-training/imgs/context_parallelism.svg"/>

Sequence parallelism therefore requires communication to make the relevant KV elements available on all GPUs. The following sections describe two alternative algorithms for sequence/context-parallel attention.

## Ulysses sequence parallelism

Using the previous notation, assume a mini-batch split along the sequence dimension, with local storage of `(B, S/P, E)` per process.
A feed-forward network applied to a tensor of this shape achieves parallelism of order `P`, since the linear layers treat the `S` dimension as a batch dimension. Multi-head attention, however, requires communication between processes, because the dependencies across tokens require access to the full attention context.

[(DeepSpeed) Ulysses parallelism](https://arxiv.org/abs/2309.14509) solves this by swapping the distributed view from sequence-split to head-split before the MHA module, and back after it, as in the following illustration:

{: style="text-align:center; font-size: small;"}
<img width="79%" height="79%" src="{{ site.assets }}/distributed-training/imgs/ulysses_sequence_parallelism.svg"/>

{: style="text-align:center; font-size: small;"}
Overview of Ulysses sequence parallelism. **Left:** the initial view of the input tensor, distributed across 4 color-coded GPUs and split along the sequence (`S`) dimension. **Center:** the *first all-to-all* changes the distributed view from sequence-split to head-split. Each process now holds full sequences for a subset of heads and computes attention locally. **Right:** the *second all-to-all* changes the view back from head-split to sequence-split.

Ulysses only adds steps that swap the distributed view of the input tensor, e.g. from `(H, B, S/P, E)` to `(H/P, B, S, E)` and back. The function `dist_view_swap()` takes a `tensor` distributed across the process group `group` and swaps its distributed view by changing the split dimension from `old_split_dim` to `new_split_dim`:

```python
def dist_view_swap(tensor: torch.Tensor, old_split_dim: int, new_split_dim: int, group: dist.ProcessGroup):
    '''Swap the split dimension of a tensor of full shape (H, B, S, E) distributed across P processes, between the
    heads (dim 0) and the sequence (dim 2): a local shape (H, B, S/P, E) becomes (H/P, B, S, E), and vice versa.'''
    full_shape, P = list(tensor.shape), group.size()
    full_shape[old_split_dim] *= P  # full distributed shape
    H, B, S, E = full_shape

    send = torch.cat([tensor.chunk(P, dim=new_split_dim)[r].contiguous() for r in range(P)])
    recv = torch.zeros_like(send)
    dist.all_to_all_single(output=recv, input=send, group=group)

    recv = torch.cat(
        [recv.chunk(P)[r].view(H // P, B, S // P, E) for r in range(P)],
        dim=old_split_dim,
    )
    return recv
```

The rest of the implementation follows directly. In the forward pass, the first (leftmost) all-to-all in the Ulysses diagram converts the distributed view from sequence-split to head-split. The backward pass performs the inverse swap:

```python
class first_alltoall(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, group=None):
        ctx.group = group  # save for backward pass
        return dist_view_swap(x, old_split_dim=2, new_split_dim=0, group=ctx.group)

    @staticmethod
    def backward(ctx, dout):
        dout = dist_view_swap(dout, old_split_dim=0, new_split_dim=2, group=ctx.group)
        return dout, None
```

The second all-to-all performs the opposite swaps in the forward and backward passes:

```python
class second_alltoall(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, group=None):
        ctx.group = group
        return dist_view_swap(x, old_split_dim=0, new_split_dim=2, group=ctx.group)

    @staticmethod
    def backward(ctx, dout):
        dout = dist_view_swap(dout, old_split_dim=2, new_split_dim=0, group=ctx.group)
        return dout, None
```

The `MultiHeadAttention` module calls both all-to-all functions:

```python
from flash_attn.flash_attn_interface import flash_attn_func

class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd=256, d_head=128, n_heads=8, dropout_p=0.1, group=None):
        '''A Ulysses multi-head attention. Variable names follow GPTlite's post.'''

        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.keys = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.queries = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.values = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, n_embd)
        self.dropout = nn.Dropout(dropout_p)

        self.group = group  # Ulysses group
        if self.group is None:
            self.group = dist.new_group(range(dist.get_world_size()))

    def forward(self, x):
        # x is local chunk: (B, S/P, E)
        P = self.group.size()
        B, S_local = x.shape[0], x.shape[1]
        S = S_local * P

        # Q, K and V embeddings: (B, S/P, E) -> (H, B, S/P, d_head)
        q = torch.stack([q(x) for q in self.queries], dim=0)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        if P > 1:  # (H, B, S/P, d_head) -> (H/P, B, S, d_head)
            q = first_alltoall.apply(q, self.group)
            k = first_alltoall.apply(k, self.group)
            v = first_alltoall.apply(v, self.group)

        # flash_attn expects (B, S, H_local, d_head)
        q_ = q.permute(1, 2, 0, 3)
        k_ = k.permute(1, 2, 0, 3)
        v_ = v.permute(1, 2, 0, 3)

        dropout_p, softmax_scale = 0.0, q_.shape[-1] ** (-0.5)
        out = flash_attn_func(q_, k_, v_, dropout_p, softmax_scale)  # (B, S, H/P, d_head)

        out = out.permute(2, 0, 1, 3)  # (H/P, B, S, d_head)

        if P > 1:  # (H/P, B, S, d_head) -> (H, B, S/P, d_head)
            out = second_alltoall.apply(out, self.group)

        out = out.permute(1, 2, 0, 3)  # (H, B, S/P, d_head) -> (B, S/P, H, d_head)
        out = out.reshape(B, S // P, -1)  # (B, S/P, H, d_head) -> (B, S/P, H*d_head)
        out = self.proj(out)  # (B, S/P, H*d_head) -> (B, S/P, E)
        out = self.dropout(out)
        return out
```

Other approaches to this problem include [ColAI-SP](https://arxiv.org/abs/2105.13120) and [Megatron-SP](https://arxiv.org/abs/2205.05198). The main advantage of [DeepSpeed Ulysses parallelism](https://arxiv.org/abs/2309.14509) is that it requires less communication than these alternatives. Its main limitations are that the maximum parallelism degree is bounded by the number of attention heads (typically 8), and that the all-to-all is a blocking collective that may incur a large overhead on slow interconnects. Ring Attention addresses these limitations.

## Ring Attention

Ring attention was presented in the paper [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889), building on [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370). It computes attention block by block, so that $$softmax \left(QK^T \right)V$$ is computed without materializing the full attention matrix or holding all keys/values at once. It can be viewed as a distributed variant of [Flash Attention](https://arxiv.org/abs/2205.14135). The core idea is also related to [Self-attention Does Not Need $$O(n^2)$$ Memory](https://arxiv.org/abs/2112.05682).

For a single query row with scores $$s_i = dot(q, k_i)$$, the attention output is:

$$
\text{Attention}(q, K, V) = \sum_i \text{softmax}(s)_i \, v_i
= \sum_i \left( \frac{\exp(s_i)}{\sum_j \exp(s_j)} \right) v_i
= \frac{\sum_i \exp(s_i)\, v_i}{\sum_j \exp(s_j)}.
$$

The output is thus a fraction whose numerator and denominator are sums of terms. Therefore, the full attention matrix $$QK^T$$ does not need to be stored. Instead:

1. In the forward pass, the algorithm iterates over blocks of keys/values and accumulates $$v^{\star} \leftarrow v^{\star} + \exp(s_i)\, v_i$$ (numerator) and $$s^{\star} \leftarrow s^{\star} + \exp(s_i)$$ (denominator);
2. After all keys and values are processed, the final attention output is $$\frac{v^{\star}}{s^{\star}}$$;
3. In the backward pass, gradients are computed blockwise, so the whole attention matrix from the forward pass is not stored.

### Improving numerical stability

The previous formulation is not numerically stable in floating-point arithmetic, because the softmax exponentials can become very large. A standard solution is to *subtract the maximum score from all scores* before computing the softmax. This does not change the result of the softmax, but avoids overflow/underflow. This is called the **safe softmax** trick, and is well explained [in this post](https://coconut-mode.com/posts/ring-attention/). The safe softmax formulation is:

$$
safe\_softmax(s_i) = \frac{\exp (s_i - s_{max})}{ \sum_j \exp( s_j - s_{max})}.
$$

When the computation is done in blocks, the global maximum is not known in advance. The solution is to keep the maximum over all blocks *seen so far* in the loop over blocks. If $$m_j$$ is the maximum over all blocks up to and including block $$j$$, and $$m_{j-1}$$ is the previous maximum (up to block $$j-1$$), the accumulators are updated as:

$$
\begin{align*}
v^{\star} \leftarrow & \, v^{\star} \cdot \exp ( m_{j-1} - m_{j}) + \sum_{i \in \text{block } j} \exp \left( s_i- m_{j} \right) \, v_i \\
s^{\star} \leftarrow & \, s^{\star} \cdot \exp( m_{j-1} - m_{j} ) + \sum_{i \in \text{block } j} \exp \left( s_i- m_{j} \right).
\end{align*}
$$

Finally, implementations often do not compute $$v^{\star}$$ and $$s^{\star}$$ directly, but use a small variation that accumulates log-values (LogSumExp, or LSE) for better stability (described [here](https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795)). The LSE is defined as:

$$
LSE (x_{1},\dots ,x_{n})=\log \left(\exp(x_{1})+\cdots +\exp(x_{n})\right).
$$

### Distributed algorithm

Since the attention output can be computed per block, attention is parallelized by assigning a sub-block to each process. The tensors $$q$$, $$k$$ and $$v$$ are first split along the sequence across $$P$$ processes, i.e., **each process holds a non-overlapping sequence block of the $$q$$, $$k$$ and $$v$$ tensors**. As in Ulysses, this gives `P`-order parallelism on the feed-forward network, but not on multi-head attention.

In ring attention, each process keeps its local chunk of queries and **rotates the key/value blocks** among all $$P$$ processes in a ring. At each of the $$P$$ communication steps, each process sends its key/value block to the next process in the ring and receives the key/value block of the previous one. After $$P$$ steps, every process has seen all key/value blocks and has computed attention for its local queries.

This pattern is illustrated below:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/ring_attention.svg"/>

{: style="text-align:center; font-size: small;"}
Overview of the Ring Attention algorithm. **Before Ring Attention:** the initial view of the input tensor, distributed across 4 color-coded GPUs and split along the sequence (`S`) dimension. **1st Ring Attention step:** each process holds its part of the Query, Key and Value tensors and computes block attention on them. Meanwhile, each process asynchronously sends its Key and Value tensors to the next process in the ring (clockwise). **2nd, 3rd, and 4th steps:** each process receives the Key and Value blocks of the previous process and computes the attention output for its original Query tensor and the received Key/Value tensors. **After Ring Attention:** the MHA output is already sequence-split across processes, as in the initial data layout.

After all ring steps, each process has kept its own sequence block of $$q$$ and has iterated over the full $$k$$ and $$v$$ tensors in chunks. For example, the third (red) process above sees the following data:

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/distributed-training/imgs/ring_attention_qkv.svg"/>

Ulysses communicates synchronously (all-to-all), so processes wait for the communication to complete. Ring attention instead uses asynchronous point-to-point communication. If the computation time of a block exceeds its transmission time, the communication overhead is hidden behind computation.

### Implementation

The implementation uses two helper functions:

- `isend_k_and_v(k, v, recv_k, recv_v, group)` asynchronously sends the tensors `k` and `v` to the next neighbour and asynchronously receives the previous neighbour's `k` and `v` into `recv_k` and `recv_v`. It returns the send/receive communication futures, which must be waited on with `req.wait()`.
- `accumulate_out_and_lse(out, lse, block_out, block_lse)` merges the attention output and LogSumExp of a block into the accumulated ones. It rescales both so that the softmax is normalized over all blocks seen so far.

```python
import flash_attn.flash_attn_interface as fa  # flash attention


def isend_k_and_v(k, v, recv_k, recv_v, group):
    P, rank = group.size(), group.rank()
    dst = (rank + 1) % P  # the rank of the next process
    src = (rank - 1) % P  # the rank of the previous process
    req_k_send = dist.P2POp(dist.isend, k, dst, group)
    req_v_send = dist.P2POp(dist.isend, v, dst, group)
    req_k_recv = dist.P2POp(dist.irecv, recv_k, src, group)
    req_v_recv = dist.P2POp(dist.irecv, recv_v, src, group)
    return dist.batch_isend_irecv([req_k_send, req_v_send, req_k_recv, req_v_recv])


def accumulate_out_and_lse(out, lse, block_out, block_lse):
    # source: https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)  # (B, H, S/P) -> (B, S/P, H, 1)
    if out is None:  # first block
        return block_out, block_lse

    # out = (1-w) * out + w * block_out, with w = exp(block_lse) / (exp(lse) + exp(block_lse)) = sigmoid(block_lse - lse)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    # lse = log(exp(lse) + exp(block_lse))
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse
```

The forward pass performs $$P$$ ring communication steps. At each step, it computes the attention output and the per-row LogSumExp of the current block (`block_out` and `block_lse`) and accumulates them. The keys and values of the next step are received into new buffers, because the current ones are still being sent and used:

```python
class RingAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, group):

        P = group.size()
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()  # (B, S/P, H, E)

        out = lse = None  # accumulators

        for step in range(P):  # do P ring steps

            # send K and V for next step, asynchronously, into new buffers: the current ones are still in use
            recv_k, recv_v = torch.empty_like(k), torch.empty_like(v)
            reqs_k_v = isend_k_and_v(k, v, recv_k, recv_v, group)

            # compute attention output and softmax lse for current block
            dropout_p, softmax_scale = 0.0, q.shape[-1] ** (-0.5)
            kwargs = dict(causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_softmax=False)
            block_out, _, _, _, _, block_lse, _, _ = fa._flash_attn_forward(q, k, v, dropout_p, softmax_scale, **kwargs)

            # update out and lse
            out, lse = accumulate_out_and_lse(out, lse, block_out, block_lse)

            # wait for new K and V before starting the next iteration
            [req.wait() for req in reqs_k_v]
            k, v = recv_k, recv_v

        ctx.group = group  # save for backward
        out = out.to(dtype=q.dtype)
        ctx.save_for_backward(q, k, v, out, lse)
        return out
```

The [backward pass](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#background) takes as input the gradient of the loss with respect to the output ($$\nabla_{out} Loss$$ or `dout` in the code below) and returns the gradients with respect to the inputs (gradients $$\nabla_{q} Loss$$, $$\nabla_{k} Loss$$, $$\nabla_{v} Loss$$, or `dq`, `dk`, `dv`):

```python
    @staticmethod
    def backward(ctx, dout, *args):

        P = ctx.group.size()
        q, k, v, out, softmax_lse = ctx.saved_tensors
        softmax_lse = softmax_lse.squeeze(dim=-1).transpose(1, 2)

        block_dq = torch.empty_like(q)
        block_dk = torch.empty_like(k)
        block_dv = torch.empty_like(v)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        for step in range(P):

            # send K and V for next step, asynchronously, into new buffers (as in the forward pass)
            recv_k, recv_v = torch.empty_like(k), torch.empty_like(v)
            reqs_k_v = isend_k_and_v(k, v, recv_k, recv_v, ctx.group)

            # compute gradients for the current block K/V and local Q
            dropout_p, softmax_scale = 0.0, q.shape[-1] ** (-0.5)
            kwargs = dict(causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                          deterministic=False, rng_state=None)
            fa._flash_attn_backward(
                dout, q, k, v, out, softmax_lse,
                block_dq, block_dk, block_dv,
                dropout_p, softmax_scale, **kwargs
            )

            if step > 0:
                # wait for dK and dV from the previous steps (rotated accumulators)
                [req.wait() for req in reqs_dk_dv]
                dk, dv = recv_dk, recv_dv

            dq += block_dq
            dk += block_dk
            dv += block_dv

            # rotate dK and dV alongside K and V, into new buffers
            recv_dk, recv_dv = torch.empty_like(dk), torch.empty_like(dv)
            reqs_dk_dv = isend_k_and_v(dk, dv, recv_dk, recv_dv, ctx.group)

            # wait for new K and V before the next iteration
            [req.wait() for req in reqs_k_v]
            k, v = recv_k, recv_v

        # before returning, wait for the last dK and dV for this process' original block
        [req.wait() for req in reqs_dk_dv]
        dk, dv = recv_dk, recv_dv
        return dq, dk, dv, None
```

In the backward pass, the gradients of a block refer to the current $$K$$ and $$V$$ blocks, which rotate across processes. The gradients `dk` and `dv` are therefore accumulated while they rotate alongside their `k` and `v` blocks.

Finally, the `forward` pass of multi-head attention calls ring attention instead of regular attention:

```python
class MultiHeadAttention(nn.Module):
     def forward(self, x):
        P = self.group.size()
        B, S_local = x.shape[0], x.shape[1]
        S = S_local * P

        # take Q, K and V, and collect all head embeddings: (B, S/P, E) -> (H, B, S/P, E)
        q = torch.stack([q(x) for q in self.queries], dim=0)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        # convert to flash-attn layout (B, S/P, H, E)
        q_ = q.permute(1, 2, 0, 3)
        k_ = k.permute(1, 2, 0, 3)
        v_ = v.permute(1, 2, 0, 3)

        if P == 1:
            out_ = fa.flash_attn_func(q_, k_, v_)
        else:
            out_ = RingAttention.apply(q_, k_, v_, self.group)

        out = out_.reshape(B, S // P, -1)  # (B, S/P, H, E) -> (B, S/P, H*E)
        out = self.proj(out)  # (B, S/P, H*E) -> (B, S/P, E)
        out = self.dropout(out)
        return out
```

A key disadvantage of ring attention is that the number of communication steps equals the number of processes. This may be a limiting factor on large networks: very fine-grained sequence splits lead to small per-process workloads, which makes it harder to hide communication behind computation.

Note that deterministic behaviour of sequence parallelism across networks with different node counts is difficult to achieve, because random number generators produce different values during model initialization and dropout.

## When to use Ulysses and Ring attention

Ulysses is suited to cases where the sequence-parallel group sits on a fast interconnect, such as NVLink within a node, and the parallelism degree is at most the number of attention heads. Therefore, Ulysses cannot be used with Multi-Query Attention (MQA), and its use with Grouped Query Attention (GQA) is limited. Its benefits are that any attention kernel works unchanged and that it works well with any mask. However, the all-to-all is hard to overlap with computation and is slow across nodes.

Ring attention is suited to cases that need a parallelism degree beyond the head count, or parallelism across nodes. It has no head limit and hides communication, but only if each local chunk is long enough for computation to cover the transfer. However, Ring Attention is not ideal for custom or arbitrary attention masks. The blockwise formulation supports any masking pattern mathematically, but practical implementations rely on overlapping communication with computation, so the mask should ideally yield computation at every step. In practice, most standard Ring Attention libraries support only causal masks (the triangular masks used in autoregressive models) or full masks (unmasked bidirectional attention). To address this, variants such as [**Striped Attention**](https://arxiv.org/abs/2311.09431 "Striped Attention Paper") and the zigzag chunks of [**Llama 3's context parallelism**](https://arxiv.org/abs/2407.21783 "Llama 3 Paper") permute the data chunks around the structure of a causal mask to balance the workload across devices.

Both methods have drawbacks. A common rule of thumb for multi-node runs is a hybrid algorithm: Ulysses within each node and Ring across nodes, which usually outperforms either method alone at scale. Hybrid algorithms for sequence parallelism are discussed in [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719), [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485v1), and [Megatron's a2a+p2p](https://docs.nvidia.com/nemo/megatron-bridge/0.5.0/training/hierarchical-context-parallel.html) for Hierarchical Context Parallelism (HCP, where p2p is the ring exchange and a2a is Ulysses-style parallelism).  


## Training with sequence- and multi-dimensional parallelism

PyTorch has no notion of *partial sequences*, in the way data parallelism has a notion of *partial batches*. Samples processed in parallel are therefore assumed to be full-length samples of a data-parallel execution. To correct this, sequence parallelism of order `P` requires `P` gradient accumulation steps with the corresponding **gradient scaling**, so that the intended batch size is processed and gradients are properly averaged.

Multi-dimensional parallelism (e.g., data + sequence) also requires process groups for the data-parallel processes (which load different samples) and for the sequence-parallel processes (which load different chunks of the same sample). These groups are created with PyTorch's [`DeviceMesh`](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html) or manually. As an example, $$2 \times 2$$ data and Ulysses sequence parallelism on 4 GPUs gives the following memory layout before and during multi-head attention:

{: style="text-align:center; font-size: small;"}
<img width="66%" height="66%" src="{{ site.assets }}/distributed-training/imgs/sequence_and_data_parallelism.svg"/>

{: style="text-align:center; font-size: small;"}
Allocation of activations on 4 GPUs with 2-GPU data parallelism and 2-GPU Ulysses sequence parallelism. Left: the blue and green processes form one sequence-parallel group and share one sample; the red and yellow processes form the other group and share the other sample. Right: the first all-to-all in Ulysses parallelism converts token-level distributed storage into head-level distributed storage. All four processes then compute attention over full sequences.
