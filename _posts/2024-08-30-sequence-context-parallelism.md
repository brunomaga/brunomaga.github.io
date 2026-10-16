---
layout: post
title:  "Sequence and context parallelism with Ulysses and Ring attention"
categories: ["Distributed Training"]
tags: ["sequence parallelism", "context parallelism", "Ulysses", "Ring attention", "attention"]
---

In this post, we will discuss and implement context and sequence parallelism, techniques that parallelize computation across the sequence-length dimension. Sequence parallelism is just another dimension of model parallelism, in the universe of data, tensor, pipeline, context, sequence and expert parallelism. 

Many model inputs and activations include an extra dimension that represents an (un)ordered sequence of tokens. Examples include text and other temporal data, with shape `(B, S, E)` for batch size `B`, sequence length (or token count) `S` and embedding size `E`. Videos have shape `(B, C, T, H, W)` for `C` channels, `T` frames, and `HxW` pixels per frame. Attention scores and probabilities have shape `(B, H, S, S)` for `H` heads and sequences of length `S`. In these scenarios, we can exploit parallelism along the sequence dimension `S` (or `T` for videos). Using the notation for text, sequence parallelism over `P` processes requires local storage of shape `(B, S/P, E)` per process. With that in mind, in this post we will implement two existing techniques for sequence parallelism: Ulysses and Ring Attention. Our use case will be a model composed of a sequence of [GPTlite blocks]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}), where each block is a multi-head attention (MHA) module followed by a feed-forward network (FFN), with some normalization and skip connections.

Before we continue, we emphasize the following:

- We call our algorithms **sequence parallelism**, even though several sources also use the term **context parallelism**. Strictly speaking, **sequence parallelism** splits the sequence (token) dimension across GPUs so that each device processes a different chunk of tokens, mainly to scale activation memory/compute with longer sequences. **Context parallelism** splits the attention context (keys/values) across GPUs, so that each device holds a slice of the KV context and computes partial attention that’s combined via communication, mainly to scale attention/KV-cache memory for very large context windows. In many scenarios (including this post), we effectively use both: we split the input tokens and also distribute the KV elements needed by attention.
- We focus on **training and inference prefill**. This is not to be confused with sequence/context parallelism for the **inference decode** step, which is a different setting: decode processes one token at a time and typically shards the KV cache rather than the input tokens.

## Data loader setup for sequence parallelism

To implement sequence parallelism, we first have to adapt our data loader to load data at the token level, in a way that matches our hybrid data- and sequence-parallel setup. All we need to do is configure a `DataLoader` and a `DistributedSampler` that allocate chunks of sequences (instead of full sequences) to the data loader workers.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/spdistributedsampler.png"/>

{: style="text-align:center; font-size: small;"}
An example of data loading for hybrid data parallelism (DP) and sequence parallelism (SP) on 4 processes/GPUs (color-coded), on a dataset of 8 samples. First row: all 4 processes run distributed data parallelism. Second row: a custom `DistributedSampler` that yields chunks of sequences enables a hybrid data- and sequence-parallel execution with 2 groups of 2 sequence-parallel processes. Third row: no data parallelism — all 4 processes run sequence parallelism on the same sample.

Once this is in place, most layers (like the feed-forward network) operate on the last (embedding) dimension of the input, and can therefore run sequence-parallel training/prefill without changes. The attention layer is an exception, and is covered next.

## Distributed attention for context parallelism

During training/inference, the model runtime can scale well with the sequence parallelism degree, except for the attention layer. This is because, to reduce the activation memory footprint, each GPU stores only a subset of the KV elements.

In practice, to compute attention, each process uses its own subset/chunk of queries, but **needs the keys and values for the full sequence**. The rationale is the following: take the attention head computation $$softmax \left(QK^T \right)V$$, where all tensors are shaped `(B, S, E)`. If each process holds a subset of rows of $$Q$$ as $$Q_p$$ with shape `(B, S/P, E)`, it needs to access all elements of $$K^T$$ and $$V$$ to perform the dot-product $$Q_p K^T$$, the row-wise $$softmax$$, and the dot-product with $$V$$, resulting in an attention output of shape `(B, S/P, E)` per process:

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="{{ site.assets }}/distributed-training/imgs/context_parallelism.png"/>

Therefore, sequence parallelism requires some communication to make the relevant KV elements accessible across all GPUs. With that in mind, in the following sections we will look into two alternative algorithms for sequence/context-parallel attention.

## Ulysses sequence parallelism

Using the previous notation, assume a mini-batch split across the sequence dimension, with local storage of `(B, S/P, E)` per process.
If we pass a tensor of this shape to a feed-forward network, we achieve parallelism of order `P`, as the linear layers treat the `S` dimension as a batch dimension. However, multi-head attention requires communication between processes, because the time dependencies across tokens require access to the full attention context.

[(DeepSpeed) Ulysses parallelism](https://arxiv.org/abs/2309.14509) solves this by swapping the distributed view from sequence-split to head-split before the MHA module, and back after it, as in the following illustration:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/ulysses_sequence_parallelism.png"/>

{: style="text-align:center; font-size: small;"}
Overview of Ulysses sequence parallelism. **Left:** the initial view of the input tensor, distributed across 4 (color-coded) GPUs, split along the sequence (`S`) dimension. **Center:** the *first all-to-all* changes the distributed tensor view from sequence- to head-split. Each process now holds complete sequences (i.e., not sequence-split) for a subset of heads and can compute attention locally. **Right:** the *second all-to-all* reverts the view from head- to sequence-split.

In practice, implementing Ulysses only requires extra steps that swap the distributed view of the input tensor from, e.g., `(H, B, S/P, E)` to `(H/P, B, S, E)` and vice versa. We can implement a function `dist_view_swap()` that, given a `tensor` distributed across the process group `group`, swaps the distributed view by changing the split dimension from `old_split_dim` to `new_split_dim`:

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

From here, the implementation is straightforward. In the forward pass, the first (leftmost) all-to-all in the Ulysses diagram above converts the distributed view from sequence-split to head-split, while the backward pass performs the inverse view swap:

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

The second all-to-all is analogous, except that it performs the opposite view changes in the forward and backward passes:

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

Now we can implement the `MultiHeadAttention` module by calling both all-to-all functions:

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

And that is it. It's conceptually simple.

As an important note, there are other similar approaches to this problem, such as [ColAI-SP](https://arxiv.org/abs/2105.13120) and [Megatron-SP](https://arxiv.org/abs/2205.05198). The big advantage of [DeepSpeed Ulysses parallelism](https://arxiv.org/abs/2309.14509) is that it requires less communication than the alternatives. Its main downsides are that the maximum parallelism is dictated by the number of attention heads (typically 8), and that the all-to-all is a blocking collective communication that may incur heavy overhead on slow interconnects. That's where Ring Attention comes into play.

## Ring Attention

Ring attention was presented in the paper [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889), building on [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370). It performs a per-block computation of attention that allows one to compute $$softmax \left(QK^T \right)V$$ without materializing the full attention matrix or requiring all keys/values to be present at once. It can be viewed as a distributed variant of [Flash Attention](https://arxiv.org/abs/2205.14135). The core rationale is also related to the ideas in [Self-attention Does Not Need $$O(n^2)$$ Memory](https://arxiv.org/abs/2112.05682).

Given scores $$s_i = dot(q, k_i)$$ for a single query row, the attention output can be written as:

$$
\text{Attention}(q, K, V) = \sum_i \text{softmax}(s)_i \, v_i
= \sum_i \left( \frac{\exp(s_i)}{\sum_j \exp(s_j)} \right) v_i
= \frac{\sum_i \exp(s_i)\, v_i}{\sum_j \exp(s_j)}.
$$

This formulation shows that the output is a fraction whose numerator and denominator are sums of terms. Therefore, we do not need to store the full attention matrix $$QK^T$$. Instead:

1. In the forward pass, we iterate over blocks of keys/values and accumulate $$v^{\star} \leftarrow v^{\star} + \exp(s_i)\, v_i$$ (numerator) and $$s^{\star} \leftarrow s^{\star} + \exp(s_i)$$ (denominator);
2. After processing all keys and values, we compute $$\frac{v^{\star}}{s^{\star}}$$ to get the final attention output;
3. In the backward pass, we compute gradients blockwise, so we don't need to store the whole attention matrix from the forward pass.

### Improving numerical stability

The previous formulation is not numerically stable in floating-point arithmetic, because the softmax exponentials can become very large numbers. A standard solution is to compute the softmax *by subtracting the maximum score from all scores*. This does not change the result of the softmax but avoids overflow/underflow. This is called the **safe softmax** trick and is well explained [in this post](https://coconut-mode.com/posts/ring-attention/). The safe softmax formulation is:

$$
safe\_softmax(s_i) = \frac{\exp (s_i - s_{max})}{ \sum_j \exp( s_j - s_{max})}.
$$

The issue is how to compute the maximum when the computation is done in blocks. We do this by keeping the maximum over all blocks *seen so far* in the loop over blocks. Then, if $$m_j$$ is the maximum over all blocks up to and including block $$j$$, and $$m_{j-1}$$ is the previous maximum (up to block $$j-1$$), the accumulators are updated as:

$$
\begin{align*}
v^{\star} \leftarrow & \, v^{\star} \cdot \exp ( m_{j-1} - m_{j}) + \sum_{i \in \text{block } j} \exp \left( s_i- m_{j} \right) \, v_i \\
s^{\star} \leftarrow & \, s^{\star} \cdot \exp( m_{j-1} - m_{j} ) + \sum_{i \in \text{block } j} \exp \left( s_i- m_{j} \right).
\end{align*}
$$

Finally, in practice we often do not compute $$v^{\star}$$ and $$s^{\star}$$ directly, but apply a small variation that accumulates log-values (LogSumExp, or LSE) for better stability (described [here](https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795)), where the LSE is defined as:

$$
LSE (x_{1},\dots ,x_{n})=\log \left(\exp(x_{1})+\cdots +\exp(x_{n})\right).
$$

### Distributed algorithm

Now that we know how to compute the attention output per block, we can parallelize attention by assigning a sub-block to each process. We start with sequence parallelism of the tensors $$q$$, $$k$$ and $$v$$ across $$P$$ processes, i.e., **each process holds a non-overlapping sequence block of the $$q$$, $$k$$ and $$v$$ tensors**. Just like Ulysses, this allows for direct `P`-order parallelism on the feed-forward network, but not on multi-head attention.

During ring attention, each process keeps its local chunk of queries and **rotates the key/value blocks** among all $$P$$ processes in a ring: at each communication step (out of $$P$$ steps), each process sends its block of keys/values to the next process and receives the block of keys/values from the previous process in the ring. After $$P$$ steps, every process has seen all key/value blocks and can compute attention for its local queries.

This pattern can be illustrated as:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/distributed-training/imgs/ring_attention.png"/>

{: style="text-align:center; font-size: small;"}
Overview of the Ring Attention algorithm. **Before Ring Attention:** the initial view of the input tensor, distributed across 4 (color-coded) GPUs, split along the sequence (`S`) dimension. **1st Ring Attention step:** each process holds its part of the Query, Key and Value tensors and computes block attention for those tensors. Meanwhile, processes asynchronously send their Key and Value tensors to the next process in the ring (clockwise). **2nd, 3rd, and 4th steps:** each process receives the previous process' Key and Value blocks and computes the attention output for its original Query tensor and the received Key/Value tensors. **After Ring Attention:** the MHA output is already sequence-split across processes, similarly to the initial data format.

From a process standpoint, after all ring steps, each process has kept its own sequence block of $$q$$ and has iterated over the full $$k$$ and $$v$$ tensors (in chunks). For example, the third (red) process above sees the following data:

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/distributed-training/imgs/ring_attention_qkv.png"/>

An important point is that, while Ulysses communicates synchronously (all-to-all), forcing processes to wait for the communication to complete, ring attention can use asynchronous point-to-point communication. If the computation time of a block exceeds its transmission time, the communication overhead can be hidden behind computation.

### Implementation

We start with two helper functions: `isend_k_and_v(k, v, recv_k, recv_v, group)` asynchronously sends the tensors `k` and `v` to the next neighbour, asynchronously receives the previous neighbour's `k` and `v` into `recv_k` and `recv_v`, and returns the send/receive communication futures, which must be waited on with `req.wait()`; and `accumulate_out_and_lse(out, lse, block_out, block_lse)` merges the attention output and LogSumExp of a block into the accumulated ones, rescaling both so that the softmax is normalized over all the blocks seen so far:

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

The forward pass performs $$P$$ ring communication steps. On each step, it computes the attention output and the per-row LogSumExp of the current block (variables `block_out` and `block_lse`), and accumulates them. The keys and values of the next step are received into new buffers, as the current ones are still being sent and used:

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

The [backward pass](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#background) takes as input the gradient of the loss with respect to the output ($$\nabla_{out} Loss$$ or `dout` in the code below), and returns the gradients with respect to the inputs (gradients $$\nabla_{q} Loss$$, $$\nabla_{k} Loss$$, $$\nabla_{v} Loss$$, or `dq`, `dk`, `dv`):

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

A nuance in the backward pass is that the gradients of a given block refer to the current $$K$$ and $$V$$ blocks, which are being rotated across processes. Therefore, the gradients `dk` and `dv` are accumulated while being rotated alongside their `k` and `v` blocks.

Finally, the `forward` pass of multi-head attention is straightforward and simply calls ring attention instead of regular attention:

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

And we are done. As we can see, a key disadvantage of ring attention is that the number of communication steps equals the number of processes. This may be a limiting factor on large networks, where very fine-grained sequence splits lead to small per-process workloads, making it harder to fully hide communication behind computation.

This code has been added to the [distributed-training repo](https://github.com/{{ site.repository }}/tree/master/assets/distributed-training). Keep in mind that deterministic behaviour of sequence parallelism across networks with different node counts is difficult to achieve, because random number generators produce different values during model initialization and dropout.

## When to use Ulysses and Ring attention

Use Ulysses when the sequence-parallel group sits on a fast interconnect, like NVLink within a node, and the parallelism degree is at most the number of attention heads. That means we cannot use Ulysses with Multi-Query Attention (MQA), and its use with Grouped Query Attention (GQA) is limited. A benefit of Ulysses is that any attention kernel works unchanged, and that it works well for any mask. However, the all-to-all is hard to overlap with compute and slows down across nodes.

Use Ring attention when we need a parallelism degree beyond the head count, or parallelism across nodes. It has no head limit and hides communication, but only if each local chunk is long enough for compute to cover the transfer. However, Ring Attention is not ideal for custom or arbitrary attention masks: while the fundamental blockwise formulation can mathematically support any masking pattern, its real-world implementation depends on overlapping communication with computation (so the mask should ideally yield computation on each step). In practice, most standard Ring Attention libraries natively support only causal masks (the triangular masks used in autoregressive models) or full masks (unmasked bidirectional attention). To overcome this, variants like [**Striped Attention**](https://arxiv.org/abs/2311.09431 "Striped Attention Paper") and the zigzag chunks of [**Llama 3's context parallelism**](https://arxiv.org/abs/2407.21783 "Llama 3 Paper") explicitly permute and optimize data chunks around the specific mathematical structure of a causal mask to balance the hardware workload.

Both methods have drawbacks. The rule of thumb is to use a hybrid algorithm once we span multiple nodes: Ulysses within each node and Ring across nodes, which usually beats either one alone at scale. Hybrid algorithms for sequence parallelism are also discussed in [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719), [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485v1), and [Megatron's a2a+p2p](https://docs.nvidia.com/nemo/megatron-bridge/0.5.0/training/hierarchical-context-parallel.html) for Hierarchical Context Parallelism (HCP, where p2p is the ring exchange and a2a is Ulysses-style parallelism).  


## Training with sequence- and multi-dimensional parallelism

PyTorch does not have a notion of *partial sequences* in the same way that data parallelism has a notion of *partial batches*. Thus, samples processed in parallel are usually assumed to be full-length samples in a data-parallel execution. To overcome this, when we run sequence parallelism of order `P`, we must perform `P` gradient accumulation steps with the corresponding **gradient scaling**, so that we effectively process the intended batch size and gradients are properly averaged.

Moreover, when we perform multi-dimensional parallelism (e.g., data + sequence), we need to define process groups for the data-parallel processes (the ones that load different samples) and for the sequence-parallel processes (the ones that load different chunks of the same sample). We can do this with PyTorch's [`DeviceMesh`](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html) or by creating our own process groups manually. For the sake of illustration, if we implemented $$2 \times 2$$ data and Ulysses sequence parallelism on 4 GPUs, this would be the memory layout before and during multi-head attention:

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="{{ site.assets }}/distributed-training/imgs/sequence_and_data_parallelism.png"/>

{: style="text-align:center; font-size: small;"}
Allocation of activations on a 4-GPU execution with 2-GPU data parallelism and 2-GPU Ulysses sequence parallelism. Left: blue and green processes belong to the same sequence-parallel group and share one sample; red and yellow processes form the other sequence-parallel group and share the other sample. Right: the first all-to-all in Ulysses parallelism converts token-level distributed storage into head-level distributed storage. All four processes can then compute attention over full sequences.
