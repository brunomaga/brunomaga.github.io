---
layout: post
title:  "Distributed training of a GPT model (part 4): Ulysses Sequence parallelism and Ring attention"
categories: [machine learning, distributed computing]
tags: [machinelearning]
---

## (DeepSpeed) Ulysses sentence parallelism

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/ulysses_sequence_parallelism.png"/>

{: style="text-align:center; font-size: small;"}
Overtiew of Ulysses sequence parallelism. **Left:** the initial view of the input tensor, distributed across 4 (color-coded) gpus, split by the time (T) dimension. **Center:** the *first all-to-all* changes the distributed tensor view from time- to head-split. Each process holds now a complete sententes and can compute attention independently. **Right**: the *second all-to-all* reverts the view to time-split.

The main complexity here lies on the implementation of the swap of the distributed representation from `(H/P, B, T, E)` to `(H, B, T/P, E)` and vice-versa. We can implement if for a given `tensor` whose sentence distributed across the group `group` in the two functions that follow.


```python
    @staticmethod
    def from_dist_H_to_T(tensor: torch.Tensor, group: dist.ProcessGroup = None):  # noqa
        """convert a distributed tensor from local shape (H/P, B, T, E) to (H, B, T/P, E)"""

        assert tensor.dim() == 4, f"expected 4D tensor, got {tensor.dim()}"
        P = group.size()  # noqa
        H, B, T, E = tensor.shape[0] * P, tensor.shape[1], tensor.shape[2], tensor.shape[3]  # noqa
        send = torch.cat([tensor.chunk(P, dim=2)[r].contiguous().flatten() for r in range(P)])
        recv = torch.zeros_like(send)
        dist.all_to_all_single(output=recv, input=send, group=group)
        recv = torch.cat([recv.chunk(P)[r].view(H // P, B, T // P, E) for r in range(P)], dim=0)
        assert recv.shape == (H, B, T // P, E), f"wrong shape after from_dist_H_to_T: {recv.shape}"
        return recv
```

```python
    @staticmethod
    def from_dist_T_to_H(tensor: torch.Tensor, group: dist.ProcessGroup = None):  # noqa
        """convert a distributed tensor from local shape (H, B, T/P, E) to (H/P, B, T, E)"""

        assert tensor.dim() == 4, f"expected 4D tensor, got {tensor.dim()}"
        P = group.size()  # noqa
        H, B, T, E = tensor.shape[0], tensor.shape[1], tensor.shape[2] * P, tensor.shape[3]  # noqa
        send = torch.cat([tensor.chunk(P, dim=0)[r].contiguous().flatten() for r in range(P)])
        recv = torch.zeros_like(send)
        dist.all_to_all_single(output=recv, input=send, group=group)
        recv = torch.cat([recv.chunk(P)[r].view(H // P, B, T // P, E) for r in range(P)], dim=2)
        return recv
```

From here, the implementation is straightforward. The all-to-all only needs to convert the distributed view from time- to head-split. The backward pass should do the converse operation:

```python
class first_alltoall(torch.autograd.Function):  # noqa
    """the first all-to-all in Figure 2 of DeepSpeed Ulysses paper"""

    @staticmethod
    def forward(ctx, x, ulysses_group=None):
        """receive a Tensor containing the input and return a Tensor containing the output"""
        ctx.ulysses_group = ulysses_group  # save for backward pass
        x = MultiHeadAttention.from_dist_T_to_H(x, group=ctx.ulysses_group)
        return x

    @staticmethod
    def backward(ctx, gradient):
        """receive gradient of loss with respect to output, and return gradient of loss with respect to input"""
        gradient = MultiHeadAttention.from_dist_H_to_T(gradient, group=ctx.ulysses_group)
        return gradient, None
```

Analogously, the second all-to-all makes a view change from head- to time-split in the forward pass, and its opposite change in the backward pass: 

```python
class second_alltoall(torch.autograd.Function):  # noqa
    """the second all-to-all in Figure 2 of DeepSpeed Ulysses papert"""

    @staticmethod
    def forward(ctx, x, ulysses_group=None):
        """receive a Tensor containing the input and return a Tensor containing the output"""
        ctx.ulysses_group = ulysses_group  # save for backward pass
        x = MultiHeadAttention.from_dist_H_to_T(x, group=ctx.ulysses_group)
        return x

    @staticmethod
    def backward(ctx, gradient):
        """receive gradient of loss with respect to output, and return gradient of loss with respect to input"""
        gradient = MultiHeadAttention.from_dist_T_to_H(gradient, group=ctx.ulysses_group)
        return gradient, None
```

Now we implement the `MultiHeadAttention` module simply by calling the previous attention, when needed

```python
class MultiHeadAttention(nn.Module):
    """the multi-head attention (MHA) in the DeepSpeed Ulysses paper"""

    def __init__(self, n_embd, d_head, n_heads=8, dropout_p=0.1, ulysses_group=None):
        super().__init__()

        self.n_embd = n_embd
        self.d_head = d_head
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.keys = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.queries = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.values = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, n_embd)
        self.dropout = nn.Dropout(dropout_p)
        self.ulysses_group = ulysses_group  # Ulysses sequence parallelism group

    def forward(self, x):
        P = self.ulysses_group.size() if self.ulysses_group is not None else 1  # noqa
        B, T, H, E = x.shape[0], x.shape[1] * P, self.n_heads, self.d_head  # noqa

        # take K, Q and V, and collect all head embeddings: (B, T/P, E) -> (H, B, T/P, E)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        q = torch.stack([q(x) for q in self.queries], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        if self.ulysses_group is not None:  #  (H, B, T/P, E) -> (H/P, B, T, E)
            k = MultiHeadAttention.first_alltoall.apply(k, self.ulysses_group)
            q = MultiHeadAttention.first_alltoall.apply(q, self.ulysses_group)
            v = MultiHeadAttention.first_alltoall.apply(v, self.ulysses_group)

        # dropout in MHA randomly prevents some tokens from communicating with each other  # out: (H/P, B, T, E)
        out = nn.functional.scaled_dot_product_attention(k, k, v, dropout_p=self.dropout_p)

        if self.ulysses_group is not None:  # (H/P, B, T, E) -> (H, B, T/P, E)
            out = MultiHeadAttention.second_alltoall.apply(out, self.ulysses_group)

        out = out.permute(1, 2, 0, 3).reshape(B, T // P, H * E)  # (H, B, T/P, E) -> (B, T/P, H, E) -> (B, T/P, H*E)
        out = self.proj(out)  # (B, T/P, H*E) -> (B, T/P, E)
        out = self.dropout(out)
        return out
```

Note that you can add several improvements to the communication, such as sending `q`, `k` and `v` simultaneously, ou asynchronously.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-distributed/ring_attention.png"/>

{: style="text-align:center; font-size: small;"}
Overview of the Ring Attention algorithm. **Before Ring Attention:** the initial view of the input tensor, distributed across 4 (color-coded) gpus, split by the time (T) dimension. **1st Ring Attention Step:** the first step of the ring attention. Each process holds its part of the Query, Value and Key tensors. Each process computes the block attention for those tensors. Asynchronously, processes perform an async send/recv of the Key and Value tensors to the next/previous process in the communication ring (clockwise). **2nd, 3rd, and 4th Ring Attention steps:** Each process its original Query block, and the previous processes' Key and Value blocks. Processes compute again the block attention for its Query and the received Key and Values. **After Ring Attention**: the Multi-head attention output is time-split across processes, similarly to the initial data format.
