---
layout: post
title:  "Distributed GPT model (part 4): sequence parallelism with Ulysses and Ring attention"
categories: [machine learning, distributed computing]
tags: [machinelearning]
---

We always thought about ML parallelism as a tridimensional, composed of [data parallelism]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed-sharding %}) (with or without sharding), [pipeline parallelism]({{ site.baseurl }}{% post_url 2023-08-30-GPT-lite-DeepSpeed-pipeline %}), and [model/tensor parallelism]({{ site.baseurl }}{% post_url 2023-09-02-GPT-lite-Megatron-LM-model-parallelism %}). In practice, if take an input of shape `(B, E)`, where `B` is the batch size and `E` is the size of the embeddings (channels), and we want to split that dataset across `P` processes, then:

1. data parallelism splits the data dimension across processors, effectively leading to local (per-process) storage requirement of size `(B/P, E)`;
2. pipeline parallelism requires the same `(B/P, E)` as input per processor, but processes each mini-batches as a pipeline of iterative micro-batches with gradient accumulation, leading to a memory requirement of `(B/P/Q, E)` , where `Q` is the micro-batch size;
3. model parallelism splits the embeddings across processors, requiring a local storage of shape  `(B, E/P)`.

However, many model inputs and activations include an extra dimension that represents the sequence of tokens. Few examples are temporal datasets with a shape  `(B, T, E)`, and attention mechanisms with an attention matrix of shape  `(B, T, T)`. In these scenarios, we can explore parallelism on the sequence/time dimension `T`. In line with the previous syntax, this requires a local storage of `(B, T/P, E)` . With that in mind, in this post, we will detail and implement some existing techniques for sequence parallelism. We will focus our analysis on two modules that are the basis on any GPT block: a feed-forward network (FFN) and a multi-head attention (MHA) module.

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

From here, the implementation is straightforward. The first all-to-all only needs to convert the distributed view from time- (2nd dimension) to head-split (1st dimension) ie from `(H, B, T // P, E)` to `(H // P, B, T, E)`. The backward pass should then does the converse operation:

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

And that is it. If you are looking for the full implementation, check [gptlite_ulisses_sequence_parallelism.py](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-distributed/gptlite_ulisses_sequence_parallelism.py).

## Ring Attention with Blockwise Transformers

The whole rationale was presented in the paper [Self-attention Does Not Need $$O(n^2)$$ Memory](https://arxiv.org/abs/2112.05682). The rationale is the following. Usually the attention for a given head, given a query $$q$$, key $$k$$ and value $$v$$, the attention calculation can be reduced to:

$$
\begin{align*}
Attention(Q, K, V) & = softmax \left(QK^T \right) V \\
& = softmax \left( \sum_i dot(q, k_i) \right) v_i & \text{(expanding dot-product on k)}\\
& = \sum_i softmax \left( dot(q, k_i) \right) v_i & \text{(sofmax of sum = sum of softmax)}\\
& = \sum_i \left( \frac{\exp \left( dot(q, k_i) \right)}{ \sum_j \exp\left( dot(q, k_j) \right)} \right) v_i & \text{(definition of softmax)}\\
& = \frac{ \sum_i \exp\left( dot(q, k_i) \right) v_i }{ \sum_j \exp\left( dot(q, k_j) \right) }.
\end{align*}
$$

The smart bit here is that we do not need to load the full $$v$$ and $$k$$ tensors or store the full attention matrix $$QK^T$$ im memory. Instead:
1. we iterate over the $$i$$-th element of the tensors $$v$$ and $$k$$, and perform the accumulations $$v^{\star} \leftarrow v^{\star} + \exp(q^T k_i) v_i$$ (top of the fraction), and $$s^{\star} \leftarrow s^{\star} + \exp(q^T k_i)$$ (bottom of the fraction).
2. after processing all keys and values, we divide $$\frac{v^{\star}}{s^{\star}}$$ to get the final value.

There's also some numerical stability tricks explained on section 3.

Instead of having a single process iterating over the elements of the query and value tensors, we now want to perform sequence parallelism. We want to perform sequence parallelism, by splitting the tensors $$q$$, $$k$$ and $$v$$ across $$P$$ processes, in the time dimension. That's where ring attention comes into play - original paper [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) based on [Blockwise Parallel Transformer for Large Context Models
](https://arxiv.org/abs/2305.19370). To start, **each process hold a non-overlapping timeframe (block) of the $$q$$, $$k$$ and $$v$$ tensors**. After that, blocks for the $$q$$ and $$v$$ tensors will be sent to all processes, iteratively, in a ring fashion: at each step, each process sends its block of keys and values to the next process, and receives it from the previous. After $$P-1$$ communication steps, all processes will have received the full $$k$$ and $$v$$ tensors, in chunks. This pattern can be illustrated as:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/ring_attention.png"/>

{: style="text-align:center; font-size: small;"}
Overview of the Ring Attention algorithm. **Before Ring Attention:** the initial view of the input tensor, distributed across 4 (color-coded) gpus, split by the time (T) dimension. **1st Ring Attention Step:** the first step of the ring attention. Each process holds its part of the Query, Value and Key tensors. Each process computes the block attention for those tensors. Asynchronously, processes perform an async send/recv of the Key and Value tensors to the next/previous process in the communication ring (clockwise). **2nd, 3rd, and 4th Ring Attention steps:** Each process its original Query block, and the previous processes' Key and Value blocks. Processes compute again the block attention for its Query and the received Key and Values. **After Ring Attention**: the Multi-head attention output is time-split across processes, similarly to the initial data format.

From the standpoint of a process, it was presented a timeframe of $$q$$ and the full $$k$$ and $$v$$ (after all the ring steps). As an example, for the third (red) process above:


{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="/assets/GPT-lite-distributed/ring_attention_qkv.png"/>

This allows the computation of $$v^{\star}$$, $$s^{\star}$$ for the timeframe its $$q$$ refer to. (see the accumulation operation [here](https://github.com/zhuzilin/ring-flash-attention/pull/34) ). 

$$
LSE (x_{1},\dots ,x_{n})=\log \left(\exp(x_{1})+\cdots +\exp(x_{n})\right)
$$

Note that for a given $$\mathbf {x} =(x_{1},\dots ,x_{n})$$, we have the partial derivatives:

$$
{\frac {\partial }{\partial x_{i}}}{\mathrm {LSE} (\mathbf {x} )}={\frac {\exp x_{i}}{\sum _{j}\exp {x_{j}}}},
$$

which means the gradient of LogSumExp is the softmax function

The forward pass will simply take as input the query, key and values tensor for each process and compute the output of the attention and the [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) of the [Softmax function](https://en.wikipedia.org/wiki/Softmax_function), ie:

```python
block_out, block_lse, = attn_forward( q, k, v)
out, lse = update_out_and_lse(out, lse, block_out, block_lse)
```

Note that the [backward propagation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#background) takes as input the gradient of the loss with respect to the output ($$\nabla_{out} Loss$$ or `dout` in the code below), and returns the derivatives of the loss with respect to the parameters of the functions (gradients $$\nabla_{q} Loss$$, $$\nabla_{k} Loss$$, $$\nabla_{v} Loss$$, or `dq`, `dk`, `dv`). Something similar to:


```python
block_dq, block_dk, block_dv = attn_backward(dout, q, k, v, out, lse)
dq += block_dq
dv += block_qv
dk += block_qk 
```

Similarly to the forward pass, we will have to *rotate* `v` and `k` and also `dv` and `dk`. 

The queries tensor is always local to a process, and we can compute `dq` by summing `block_dk` for every step of the backward (with different `k` and `v`). The caveat is on computing `dk` and `dv`: because `k` and `v` *rotate*  in the process ring in every iteration, then the gradients `dk` and `dv` for a given timeframe (ie process) will be computed at the process that computes that timeframe's `dv` and `dk`. Because of that, we will have an accumulator of gradients that also *rotates in the circle*. After all rotations, it will return the correct value to the process holding that timeframe.

Note that the implementation of `backward` may be confusing. According to the [documentation](https://pytorch.org/docs/stable/generated/torch.autograd.Function.backward.html), "it must accept a context `ctx` as the first argument, followed by as many outputs as the `forward()` returned (`None` will be passed in for non tensor outputs of the forward function), and it should return as many tensors, as there were inputs to `forward()`".

## Final remarks


Note that you can add several improvements to the communication, such as sending `q`, `k` and `v` simultaneously, ou asynchronously. Also, runs are not deterministic across different process counts, due to the random number generators having different states. If you want determinism, for example for testing, remove all randomness in parameter initialization, dropouts, etc.

Finally, notice that Ulysses parallelism has the limitation of allowing a maximum parallelism which is given by the number of attention heads (typically 8). On the other hand, Ring Attention requires several steps of communication, that grow with the network size. A good approach is to combine both methods, by doing a step of ulysses and for each ulysses process group, perform ring attention. This is described in [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719).