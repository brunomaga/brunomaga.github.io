---
layout: post
title:  "Sequence and context parallelism with Ulysses and Ring attention"
categories: ["Distributed Training"]
tags: ["sequence parallelism", "context parallelism", "Ulysses", "Ring attention", "attention"]
---

Sequence parallelism splits the sequence dimension of the activations across processes. It is a parallelism dimension of distributed training, with data, tensor, pipeline and expert parallelism.

Text activations have the shape `(B, S, E)`, for batch size $$B$$, sequence length $$S$$ (the number of tokens) and embedding size $$E$$. Videos add a time dimension of $$T$$ frames, and attention scores have the shape `(B, H, S, S)` for $$H$$ heads. Sequence parallelism over $$P$$ processes splits $$S$$ (or $$T$$), so that each process stores a chunk of shape `(B, S/P, E)`. The code below implements two methods, Ulysses and Ring Attention, for a model of [GPTlite blocks]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}). Each block is a multi-head attention (MHA) module followed by a feed-forward network (FFN), with layer normalization and skip connections.

Two remarks:

- The names vary. [Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html) calls *sequence parallelism* its split of the layer norm and dropout activations inside a tensor-parallel group. It calls *context parallelism* the split of all activations along the sequence, with an exchange of keys and values in attention. DeepSpeed calls Ulysses sequence parallelism. Below, both Ulysses and Ring Attention are called sequence parallelism.
- The methods target training and the prefill phase of inference. The decode phase processes one token at a time, and splits the KV cache instead of the input tokens.

## Data Loading for Sequence Parallelism

A sequence-parallel run loads chunks of samples instead of full samples. A custom `DistributedSampler` assigns the chunks of each sample to the processes of a sequence-parallel group:

{: .picture}
<img width="97.1%" height="97.1%" src="{{ site.assets }}/distributed-training/imgs/spdistributedsampler.svg"/>

{: .caption}
Data loading for hybrid data parallelism (DP) and sequence parallelism (SP) on 4 color-coded processes/GPUs and a dataset of 8 samples. First row: all 4 processes run distributed data parallelism. Second row: a custom `DistributedSampler` that yields chunks of sequences enables a hybrid data- and sequence-parallel execution with 2 groups of 2 sequence-parallel processes. Third row: no data parallelism; all 4 processes run sequence parallelism on the same sample.

Layers that act on each token separately, such as the feed-forward network and the layer norm, run on a chunk without changes. Positional embeddings need the global position of each token, which is the offset of the chunk plus the local position. Attention is the exception: it needs all tokens.

## Distributed Attention

Each process holds the queries of its chunk, but attention needs the keys and values of the full sequence. For one head, attention computes $$\text{softmax}(Q K^T / \sqrt{d}) V$$, where $$Q$$, $$K$$ and $$V$$ have the shape `(B, S, d)` for a head size $$d$$ (written $$E$$ in the figure; most models have $$Hd = E$$). A process with the rows $$Q_p$$ of shape `(B, S/P, d)` computes $$Q_p K^T$$, the row-wise softmax and the product with $$V$$, so it needs all of $$K$$ and $$V$$. Its output has the shape `(B, S/P, d)`:

{: .picture}
<img width="77.1%" height="77.1%" src="{{ site.assets }}/distributed-training/imgs/context_parallelism.svg"/>

Sequence parallelism thus needs communication in the attention layers. Ulysses and Ring Attention provide it in two different ways.

## Ulysses Sequence Parallelism

[DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) (Jacobs et al., 2023) changes the split of the activations from the sequence to the heads before attention, and back after it:

{: .picture}
<img width="82.6%" height="82.6%" src="{{ site.assets }}/distributed-training/imgs/ulysses_sequence_parallelism.svg"/>

{: .caption}
Overview of Ulysses sequence parallelism. **Left:** the initial view of the input tensor, distributed across 4 color-coded GPUs and split along the sequence (`S`) dimension. **Center:** the *first all-to-all* changes the distributed view from sequence-split to head-split. Each process now holds full sequences for a subset of heads and computes attention locally. **Right:** the *second all-to-all* changes the view back from head-split to sequence-split. Inspired by: [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)

Before attention, an [all-to-all](https://en.wikipedia.org/wiki/All-to-all_(parallel_pattern)) turns the queries, keys and values of shape `(H, B, S/P, d)` into `(H/P, B, S, d)`. Each process then holds the full sequences of $$H/P$$ heads, and computes their attention without communication. A second all-to-all turns the output back into `(H, B, S/P, d)`. The function `dist_view_swap()` takes a `tensor` distributed across the process group `group`, and moves its split from the dimension `old_split_dim` to `new_split_dim`:

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

In the forward pass, the first all-to-all moves the split from the sequence to the heads. Its backward pass moves it back:

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

The second all-to-all does the opposite moves:

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

The `MultiHeadAttention` module calls both all-to-alls:

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

The code computes attention without a causal mask. For a decoder such as GPTlite, `flash_attn_func` takes `causal=True`, since each process holds full sequences.

[ColAI-SP](https://arxiv.org/abs/2105.13120) and [Megatron-SP](https://arxiv.org/abs/2205.05198) also split the sequence. In the forward pass of a transformer layer, Ulysses sends $$4SE/P$$ elements per link and per sample, against $$4SE$$ for Megatron-SP. Its communication volume thus stays constant when $$S$$ and $$P$$ grow together. Its parallelism degree $$P$$ cannot exceed the number of heads $$H$$. With [grouped-query attention](https://arxiv.org/abs/2305.13245) (GQA), the keys and values have fewer heads, 8 in Llama 3, which lowers this limit to 8 unless the key and value heads are replicated across processes. With [multi-query attention](https://arxiv.org/abs/1911.02150) (MQA), they have a single head. The attention needs the result of each all-to-all, so the all-to-all is hard to overlap with computation. Its cost is high on low-bandwidth links, such as the links between nodes.

## Ring Attention

[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023) builds on [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370). It computes attention block by block, without the full attention matrix and without all keys and values on one process. It is a distributed form of [Flash Attention](https://arxiv.org/abs/2205.14135), and follows the idea of [Self-attention Does Not Need $$O(n^2)$$ Memory](https://arxiv.org/abs/2112.05682).

For a query $$q$$ with the scores $$s_i = q \cdot k_i / \sqrt{d}$$, the attention output is:

$$
\text{Attention}(q, K, V) = \sum_i \text{softmax}(s)_i \, v_i
= \sum_i \left( \frac{\exp(s_i)}{\sum_j \exp(s_j)} \right) v_i
= \frac{\sum_i \exp(s_i)\, v_i}{\sum_j \exp(s_j)}.
$$

The numerator and the denominator are sums over the keys, so they can be accumulated block by block, without the full matrix $$QK^T$$:

1. The forward pass iterates over the blocks of keys and values, and accumulates the numerator $$\tilde{o} \leftarrow \tilde{o} + \exp(s_i)\, v_i$$ and the denominator $$\ell \leftarrow \ell + \exp(s_i)$$.
2. After the last block, the output is $$\tilde{o} / \ell$$.
3. The backward pass also computes the gradients block by block, and recomputes the scores of each block instead of storing the attention matrix.

### Numerical Stability

The exponentials overflow for large scores. The *safe softmax* subtracts the maximum score $$m$$ from all scores, which does not change the result ([Ring Attention Explained](https://coconut-mode.com/posts/ring-attention/)):

$$
\text{softmax}(s)_i = \frac{\exp (s_i - m)}{ \sum_j \exp( s_j - m)}.
$$

Block by block, the maximum of all scores is not known in advance, so the loop keeps the maximum of the blocks seen so far. With $$m_{j-1}$$ the maximum up to block $$j-1$$ and $$m_j$$ the maximum up to block $$j$$, the accumulators are updated as:

$$
\begin{align*}
\tilde{o} \leftarrow & \, \tilde{o} \cdot \exp ( m_{j-1} - m_{j}) + \sum_{i \in \text{block } j} \exp \left( s_i- m_{j} \right) \, v_i \\
\ell \leftarrow & \, \ell \cdot \exp( m_{j-1} - m_{j} ) + \sum_{i \in \text{block } j} \exp \left( s_i- m_{j} \right).
\end{align*}
$$

The code below keeps instead the normalized output $$o = \tilde{o} / \ell$$ and the [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) (LSE) of the scores seen so far, $$\text{lse} = m + \log \ell$$, as in [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795). The LSE is defined as:

$$
LSE (x_{1},\dots ,x_{n})=\log \left(\exp(x_{1})+\cdots +\exp(x_{n})\right).
$$

A block with the output $$o_b$$ and the LSE $$\text{lse}_b$$ updates them as:

$$
\begin{align*}
o \leftarrow & \, \frac{e^{\text{lse}} \, o + e^{\text{lse}_b} \, o_b}{e^{\text{lse}} + e^{\text{lse}_b}} = o - \sigma(\text{lse}_b - \text{lse}) \, (o - o_b) \\
\text{lse} \leftarrow & \, \log \left( e^{\text{lse}} + e^{\text{lse}_b} \right) = \text{lse} - \log \sigma(\text{lse} - \text{lse}_b),
\end{align*}
$$

where $$\sigma$$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function). The forms with $$\sigma$$ compute no large exponential.

### Distributed Algorithm

Each process holds a chunk of the sequence of $$Q$$, $$K$$ and $$V$$. It keeps its queries, and the blocks of keys and values move around a ring of the $$P$$ processes. At each of the $$P$$ steps, a process computes the attention of its queries with the key and value block it holds. Meanwhile, it sends this block to the next process and receives the block of the previous one:

{: .picture}
<img width="98.9%" height="98.9%" src="{{ site.assets }}/distributed-training/imgs/ring_attention.svg"/>

{: .caption}
Overview of the Ring Attention algorithm. **Before Ring Attention:** the initial view of the input tensor, distributed across 4 color-coded GPUs and split along the sequence (`S`) dimension. **1st Ring Attention step:** each process holds its part of the Query, Key and Value tensors (Q, K and V) and computes block attention on them. Meanwhile, each process asynchronously sends its Key and Value tensors to the next process in the ring (clockwise). **2nd, 3rd, and 4th steps:** each process receives the Key and Value blocks of the previous process and computes the attention output for its original Query tensor and the received Key/Value tensors. **After Ring Attention:** the MHA output is already sequence-split across processes, as in the initial data layout. Inspired by: [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

After $$P$$ steps, each process has combined its queries with all blocks of keys and values. For example, the third (red) process above sees:

{: .picture}
<img width="39.6%" height="39.6%" src="{{ site.assets }}/distributed-training/imgs/ring_attention_qkv.svg"/>

Ulysses waits for its all-to-alls to finish. Ring Attention overlaps its point-to-point transfers with computation, and hides them when the computation of a block takes longer than the transfer of the next one.

### Implementation

The implementation uses two helper functions:

- `isend_k_and_v(k, v, recv_k, recv_v, group)` sends `k` and `v` to the next process, and receives the `k` and `v` of the previous process into `recv_k` and `recv_v`, asynchronously. It returns the requests, which the caller completes with `req.wait()`.
- `accumulate_out_and_lse(out, lse, block_out, block_lse)` merges the output and the LSE of a block into the accumulated ones, with the update above.

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

The forward pass runs $$P$$ ring steps. At each step, it computes the output and the LSE of the current block (`block_out` and `block_lse`), and accumulates them. The keys and values of the next step arrive in new buffers, because the current ones are still being sent and used. After $$P$$ steps, each process holds its own keys and values again, and saves them for the backward pass:

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

The [backward pass](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#background) takes the gradient of the loss with respect to the output ($$\nabla_{out} Loss$$, or `dout` in the code below). It returns the gradients with respect to the inputs ($$\nabla_{q} Loss$$, $$\nabla_{k} Loss$$ and $$\nabla_{v} Loss$$, or `dq`, `dk` and `dv`). The gradients `dk` and `dv` of a block belong to the process that owns the block. They move around the ring with their keys and values, and each process adds its part:

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

The `forward` of the multi-head attention calls ring attention instead of the attention kernel:

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

The code calls the internal functions `_flash_attn_forward` and `_flash_attn_backward` of flash-attn 2.6, whose signatures changed in later versions. It computes attention without a causal mask. With a causal mask and contiguous chunks, the first process attends to one block and the last process to all $$P$$ blocks, so the load is unbalanced (next section).

Ring Attention needs $$P$$ communication steps. The computation of a step grows with $$(S/P)^2$$ and its transfer with $$S/P$$, so for a large $$P$$, the chunks are short and their computation no longer hides the transfers.

With both methods, the results change with the number of processes: the dropout masks depend on how the sequence is split, and the order of the floating-point additions changes.

## When to Use Ulysses and Ring Attention

The table compares the communication in the forward pass of one attention layer, per sample, with as many key-value heads as query heads and $$Hd = E$$. The third method is the all-gather of keys and values of Llama 3 (below):

| | Ulysses | Ring Attention | All-gather of keys and values |
|---|---|---|---|
| Communication | all-to-alls before and after attention | $$P$$ point-to-point steps | one all-gather |
| Elements sent per link | $$\approx 4SE/P$$ | $$\approx 2SE$$ | $$\approx 2SE$$ |
| Overlap with computation | no | yes, for long chunks | no |
| Limit on $$P$$ | number of heads (key-value heads with GQA) | none | none |
| Masks | any | any, but causal needs load balancing | any |

With GQA, the terms of the keys and values shrink by the ratio of key-value heads to query heads.

Ulysses suits a sequence-parallel group on fast links, such as [NVLink](https://en.wikipedia.org/wiki/NVLink) within a node, with $$P$$ at most the number of heads. Each process runs any attention kernel on full sequences, so any mask works.

Ring Attention suits a parallelism degree above the number of heads, and groups across nodes. Its transfers hide behind computation only when each chunk is long enough, and only when every step has computation, which a mask can prevent. With a causal mask and contiguous chunks, some processes skip most blocks while others compute all of them. [Striped Attention](https://arxiv.org/abs/2311.09431 "Striped Attention Paper") gives each process tokens spread uniformly over the sequence, which balances the work of a causal mask, and reports up to 1.45 times the throughput of Ring Attention. [Llama 3](https://arxiv.org/abs/2407.21783 "Llama 3 Paper") splits the sequence into $$2P$$ chunks and gives process $$i$$ the chunks $$i$$ and $$2P-1-i$$. It all-gathers the keys and values instead of passing them around a ring, which supports any mask, such as the document mask. With GQA, the keys and values are much smaller than the queries, and the all-gather grows with $$S$$ while attention grows with $$S^2$$, so the all-gather costs little.

The methods also combine. [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719) runs the all-to-alls of Ulysses on high-bandwidth links and the ring transfers on low-bandwidth links. On two nodes of 8 A800 GPUs, it was 12% to 13% faster than Ring Attention alone, and within one NVLink node, Ulysses alone was the fastest. [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485v1) and the [hierarchical context parallelism of Megatron](https://docs.nvidia.com/nemo/megatron-bridge/0.5.0/training/hierarchical-context-parallel.html) (`a2a+p2p`: all-to-all within a node, ring exchange across nodes) follow the same idea.

## Training with Multi-Dimensional Parallelism

A sequence-parallel group of $$P$$ processes trains on the samples that one data-parallel process would train on. The gradient all-reduce runs over all processes, including those of a sequence-parallel group, because each process holds a different part of the gradient. With the mean loss over the tokens of each chunk, the average over all processes is the gradient of the mean loss of the batch. For the same number of samples per optimizer step as a data-parallel run on the same GPUs, the training accumulates the gradients over $$P$$ steps.

Data and sequence parallelism together need two kinds of process groups: data-parallel groups, whose processes load different samples, and sequence-parallel groups, whose processes load different chunks of the same samples. PyTorch's [`DeviceMesh`](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html) creates them, or they are created manually. With 2-way data parallelism and 2-way Ulysses on 4 GPUs, the activations before and during attention are:

{: .picture}
<img width="64.8%" height="64.8%" src="{{ site.assets }}/distributed-training/imgs/sequence_and_data_parallelism.svg"/>

{: .caption}
Allocation of activations on 4 GPUs with 2-GPU data parallelism and 2-GPU Ulysses sequence parallelism. Left: the blue and green processes form one sequence-parallel group and share one sample; the red and yellow processes form the other group and share the other sample. Right: the first all-to-all in Ulysses parallelism converts token-level distributed storage into head-level distributed storage. All four processes then compute attention over full sequences.
