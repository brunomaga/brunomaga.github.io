---
layout: post
title:  "Distributed GPT model (part 4): sequence parallelism via Ulysses and Ring attention"
categories: [machine learning, distributed computing]
tags: [machinelearning]
---

We always thought about ML parallelism as a tridimensional problem, composed of [data parallelism]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-data-parallelism %}) (with or without sharding), [pipeline parallelism]({{ site.baseurl }}{% post_url 2023-08-30-GPT-lite-DeepSpeed-pipeline %}), and [model/tensor parallelism]({{ site.baseurl }}{% post_url 2023-09-02-GPT-lite-Megatron-LM-model-parallelism %}). In practice, if take an input of shape `(B, E)`, where `B` is the batch size and `E` is the size of the embeddings (channels, features), and we want to split that dataset across `P` processes, then:

1. data parallelism splits the data dimension across processors, effectively leading to local (per-process) storage requirement of size `(B/P, E)`;
2. pipeline parallelism requires the same `(B/P, E)` as input per processor, but processes each mini-batch as a pipeline of iterative micro-batches with gradient accumulation, leading to a memory requirement of `(B/P/Q, E)` , where `Q` is number of micro-batches;
3. model parallelism splits the embeddings across processors, requiring a local storage of shape  `(B, E/P)`.

However, many model inputs and activations include an extra dimension that represents an (un)ordered sequence of tokens. Few examples are temporal datasets with a shape  `(B, T, E)`, and attention mechanisms with an attention matrix of shape  `(B, T, T)`. In these scenarios, we can explore parallelism on the sequence/time dimension `T`. By doing that, we would require a local storage of `(B, T/P, E)` per process. With that in mind, in this post, we will implement two existing techniques for sequence parallelism: Ulysses and Ring Attention. Our use case will be a model composed of a sequence of [GPT-like blocks]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), where each block is multi-head attention (MHA) module followed by a feed-forward network (FFN), with some normalization and skip connections.

## Ulysses sentence parallelism

Take the previous notation and assume a data batch split across a network with a local storage of `(B, T/P, E)`. 
If you pass such shape to a feed-forward network, you can parallelize and speedup linearly the input across all processes. However, in the case of the multi-head attention, there is the need for process communication as the time dependency across the tokens require inter-token communication to compute the attention matrix of shape `(B, H, T, T)` for `H` heads. The [(DeepSpeed) Ulysses parallelism](https://arxiv.org/abs/2309.14509) approaches solves this by swapping the distributed view from time-split to head-split before and after the MHA attention module, as the following illustration:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/ulysses_sequence_parallelism.png"/>

{: style="text-align:center; font-size: small;"}
Overtiew of Ulysses sequence parallelism. **Left:** the initial view of the input tensor, distributed across 4 (color-coded) gpus, split by the time (T) dimension. **Center:** the *first all-to-all* changes the distributed tensor view from time- to head-split. Each process holds now a complete sententes and can compute attention independently. **Right**: the *second all-to-all* reverts the view to time-split.

There are other similar approaches to handle this problem, such as [ColAI-SP](https://arxiv.org/abs/2105.13120) and [Megatron-SP](https://arxiv.org/abs/2205.05198), however [DeepSpeed Ulysses parallelism](https://arxiv.org/abs/2309.14509)  provides less communication volume per step.

In practice, the implementation of Ulysses only requires the extra steps that swap the distributed view of the input tensor from e.g. `(H, B, T/P, E)` to  `(H/P, B, T, E)` and vice-versa. We can then implment the function that, given a `tensor` whose sentence distributed across the group `group` , swaps the distributed view by changing the split from `old_dim` to `new_dim` as:.


```python
def dist_view_swap(tensor: torch.Tensor, old_dim: int, new_dim: int, group: dist.ProcessGroup):
    """converts the distributed splie dimension of a tensor with shape (H, B, T, E) across P processes"""
    full_shape, P = list(tensor.shape), group.size()
    full_shape[old_dim]*=P # full distributed shape
    H, B, T, E = full_shape 
    send = torch.cat([tensor.chunk(P, dim=new_dim)[r].contiguous() for r in range(P)])
    recv = torch.zeros_like(send)
    dist.all_to_all_single(output=recv, input=send, group=group)
    recv = torch.cat([recv.chunk(P)[r].view(H // P, B, T // P, E) for r in range(P)], dim=old_dim)
    return recv
```

From here, the implementation is straightforward. The first all-to-all only needs to convert the distributed view from time-split to head-split ie from local shape `(H, B, T / P, E)` to `(H / P, B, T, E)`. The backward pass should then do the converse operation:

```python
class first_alltoall(torch.autograd.Function):  # noqa
    """the first all-to-all in Ulysses sequence parallelism"""

    @staticmethod
    def forward(ctx, x, ulysses_group=None):
        ctx.ulysses_group = ulysses_group  # save for backward pass
        return dist_view_swap(x, old_dim=0, new_dim=2, group=ctx.ulysses_group)

    @staticmethod
    def backward(ctx, dout):
        dout = dist_view_swap(dout, old_dim=2, new_dim=0, group=ctx.ulysses_group)
        return dout, None
```

The second all-to-all is analogous, except that it performs the opposite view changes in the forward and backward passes. Now we can implement the `MultiHeadAttention` module by calling the previous all-to-alls as modules:

```python
class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd=256, d_head=128, n_heads=8, dropout_p=0.1, ulysses_group=None):
        """ An Ulysses multi-head attention. Variable names follow GPT-lite's post """

        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.keys = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.queries = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.values = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, n_embd)
        self.dropout = nn.Dropout(dropout_p)
        self.ulysses_group = ulysses_group  # Ulysses sequence parallelism group
        if self.ulysses_group is None:
            self.ulysses_group = dist.new_group(range(dist.get_world_size())) 

    def forward(self, x):
        P = self.ulysses_group.size()
        B, T, H, E = x.shape[0], x.shape[1] * P, self.n_heads, self.d_head

        # take K, Q and V, and collect all head embeddings: (B, T/P, E) -> (H, B, T/P, E)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        q = torch.stack([q(x) for q in self.queries], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        if if P>1:  #  (H, B, T/P, E) -> (H/P, B, T, E)
            k = MultiHeadAttention.first_alltoall.apply(k, self.ulysses_group)
            q = MultiHeadAttention.first_alltoall.apply(q, self.ulysses_group)
            v = MultiHeadAttention.first_alltoall.apply(v, self.ulysses_group)

        # dropout in MHA randomly prevents some tokens from communicating with each other  # out: (H/P, B, T, E)
        out = nn.functional.scaled_dot_product_attention(k, k, v)

        if P>1:  # (H/P, B, T, E) -> (H, B, T/P, E)
            out = MultiHeadAttention.second_alltoall.apply(out, self.ulysses_group)

        out = out.permute(1, 2, 0, 3).reshape(B, T // P, H * E)  # (H, B, T/P, E) -> (B, T/P, H, E) -> (B, T/P, H*E)
        out = self.proj(out)  # (B, T/P, H*E) -> (B, T/P, E)
        out = self.dropout(out)
        return out
```

And that is it. It's pretty simple, but if you are looking for the full implementation, check [gptlite_ulisses_sequence_parallelism.py](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-distributed/gptlite_ulisses_sequence_parallelism.py). All in all, the only downsides are that the maximum parallelism is dictated by the number of attention heads (typically 8), and that the all-two-all requires blocking collective communication that may incur a heavy overhead. That's where Ring Attention comes into play.

## Ring Attention with Blockwise Transformers

Ring attention was presented in the paper  [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) based on [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370). It performs a per-block computation of the attention matrix (almost like [Flash Attention](https://arxiv.org/abs/2205.14135)), that allows one to compute the attention $$softmax \left(QK^T \right)$$ without having access to the full inputs $$Q$$, $$K$$ and $$V$$. The whole rationale was presented in the paper [Self-attention Does Not Need $$O(n^2)$$ Memory](https://arxiv.org/abs/2112.05682). In practice, the attention for a given head, given a query $$q$$, key $$k$$ and value $$v$$, the attention calculation can be reduced to:

$$
\begin{align*}
Attention(Q, K, V) & = softmax \left(QK^T \right) V \\
& = softmax \left( \sum_i dot(q, k_i) \right) v_i & \text{(expanding dot-product on k)}\\
& = softmax \left( \sum_i s_i  \right) v_i & \text{ (for simplicity, take } s_i = dot(q, k_i) \text{)} \\
& = \sum_i softmax \left(  s_i \right) v_i & \text{(sofmax of sum = sum of softmax)}\\
& = \sum_i \left( \frac{\exp \left(  s_i \right)}{ \sum_j \exp\left(  s_j \right)} \right) v_i & \text{(definition of softmax)}\\
& = \frac{ \sum_i \exp\left(  s_i \right) v_i }{ \sum_j \exp\left(  s_j \right) }.
\end{align*}
$$

The smart bit here is that we do not need to load the full $$v$$ and $$k$$ tensors or store the full attention matrix $$QK^T$$ in memory. Instead:
1. in the forward pass, we iterate over the $$i$$-th element of the tensors $$v$$ and $$k$$, and perform the accumulations $$v^{\star} \leftarrow v^{\star} + \exp(s_i) v_i$$ (top of the fraction), and $$s^{\star} \leftarrow s^{\star} + \exp(s_i)$$ (bottom of the fraction).
2. after processing all keys and values, we divide $$\frac{v^{\star}}{s^{\star}}$$ to get the final attention output value;
3. the backward pass we compute the gradients iteratively for the same blocks, therefore not needing to store the whole attention matrix from the forward pass.

### Improving numerical stability

The previous formulation is not numerically stable using floating point arithmetic because the softmax exponents can lead to very large numbers. The solution, quoting section 3 of the original paper, is to implement softmax *by subtracting the maximum score from all scores. This does not change the result of the softmax, but avoids this numerical problem.* This is called the **safe softmax** trick and is well explained [in this post](https://coconut-mode.com/posts/ring-attention/) as:

$$
safe\_softmax(s_i) = \frac{\exp (s_i)}{ \sum_j \exp(  s_j )}  \cdot \frac{\exp (-s_{max})}{\exp (-s_{max})} = \frac{\exp (s_i - s_{max})}{ \sum_j \exp(  s_j - s_{max})} 
$$

And how to compute the max value, when the computation is done in blocks? Simply by keeping the max value *so far* . Example, if the current maximum at block $$j$$ is $$m_j$$ and the previous maximum value until then was $$m_{j-1}$$, then we update our results as:

$$
\begin{align*}
v^{\star} \leftarrow & v^{\star} \cdot \exp ( m_{i-1} - m_{i}) + \exp \left( s_i- m_{i} \right) \, v_i \\
s^{\star} \leftarrow & s^{\star} \cdot \exp( m_{j-1} - m_{j} ) +  \exp \left( s_j- m_{j} \right)
\end{align*}
$$

### Distributed Algorithm

Now that we know how to compute the attention outputs per block, we can parallelize the computation of the attention by by delegating a sub-block to each processor. We start with sequence parallelism of the tensors $$q$$, $$k$$ and $$v$$ across $$P$$ processes, ie **each process hold a non-overlapping timeframe (block) of the $$q$$, $$k$$ and $$v$$ tensors**. Just like Ulysses, this allows for parallelism on the Feed-forward network, but not on the MultiHead attention. During the computation of the MHA, sub-blocks of the $$q$$ and $$v$$ tensors will be *rotated* among processes in a ring fashion, iteratively: at each communication step (out of `P` steps), each process sends its block of keys and values to the next process, and receives the keys and values of the the previous processor in the ring. After $$P$$ communication steps, all processes will have received the full $$k$$ and $$v$$ tensors, in chunks, and will have its original tensors returned to its local memory. This pattern can be illustrated as:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/ring_attention.png"/>

{: style="text-align:center; font-size: small;"}
Overview of the Ring Attention algorithm. **Before Ring Attention:** the initial view of the input tensor, distributed across 4 (color-coded) gpus, split by the time (T) dimension. **1st Ring Attention Step:** the first step of the ring attention. Each process holds its part of the Query, Value and Key tensors. Each process computes the block attention for those tensors. Asynchronously, processes perform an async send of the Key and Value tensors to the next process in the communication ring (clockwise). **2nd, 3rd, and 4th Ring Attention steps:** Each process receives the processes' Key and Value blocks and are now able to compute attention outpout for its original Query tensor and the received Key and Value tensors. **After Ring Attention**: the Multi-head attention output is already time-split across processes, similarly to the initial data format.

From a local memory standpoint, each process was presented with a timeframe of $$q$$ and the full $$k$$ and $$v$$ tensors (after all the ring steps). As an example, for the third (red) process above, we'd have the following data presented:

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="/assets/GPT-lite-distributed/ring_attention_qkv.png"/>

This allows the accumulation of $$v^{\star}$$, $$s^{\star}$$ as it was mentioned above. In practice, we apply a small variation where we accumulate on the [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) of values instead of $$v^{\star}$$ and $$s^{\star}$$ (described [here](https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795) ), due to: 

$$
LSE (x_{1},\dots ,x_{n})=\log \left(\exp(x_{1})+\cdots +\exp(x_{n})\right)
$$

### Implementation

The forward pass will perform $$P$$ ring communication steps, and on each step, it will compute for the current $$K$$, $$Q$$ $$V$$ block, the attention output and the [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) of each row of the matrix $$QK^T$$ (variables `block_out` and `block_lse`, in order to do the accumulation:

The [backward pass](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#background) will takes as input the gradient of the loss with respect to the output ($$\nabla_{out} Loss$$ or `dout` in the code below), and return the gradients of the loss with respect to the parameters of the functions (gradients $$\nabla_{q} Loss$$, $$\nabla_{k} Loss$$, $$\nabla_{v} Loss$$, or `dq`, `dk`, `dv`). Something similar to:

A small nuance in the backward pass is that the gradients of a given block will refer to the current $$K$$ and $$V$$ which is being *rotated* around processes. Therefore, its gradients `dv` and `dk` will also be accumulated by being rotated alongside the $$K$$ and $$V$$ blocks.

The big disavantadge in ring attention is the number of steps being proportional to the number of processes, and this may be a limiting factor on large compute networks where dividing sequence in such a granular fashion may leave to a small workload per process, and this not being able to mitigate the asynchronous computation. The ideal solution would then be a hybrid of Ulysses parallelism and Ring attention, which has been presented by [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719).