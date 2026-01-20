"""
Megatron-LM style tensor parallelism helpers for the GPT-lite model.

This module implements the Fig. 3 "f" and "g" autograd functions and provides
tensor-parallel versions of the FFN and Attention blocks.

Assumptions (matching the blog post / code):
- gptlite exposes: block_size, n_embd, dropout, and a GPTlite(vocab_size) constructor.
- The base GPT-lite model defines a FeedForward and either a MultiHeadAttention module
  or a per-head Head module. This file will patch whichever exists.

Notes:
- Megatron_f: identity forward, all-reduce on backward (column-parallel input grad).
- Megatron_g: all-reduce on forward, identity backward (row-parallel output combine).
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Use base GPTlite model from the GPT-lite post
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, "..", "GPT-lite"))

import gptlite  # noqa: E402
from gptlite import block_size, n_embd, dropout  # noqa: E402


# ---- Distributed init (safe defaults) ----
def _maybe_init_process_group():
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


_maybe_init_process_group()


# ---- Megatron f/g (Fig. 3) ----
class Megatron_f(torch.autograd.Function):
    """The f function in Figure 3 in the Megatron-LM paper."""

    @staticmethod
    def forward(ctx, x, mp_comm_group=None):
        ctx.mp_comm_group = mp_comm_group
        return x

    @staticmethod
    def backward(ctx, gradient):
        # Sum grads across the tensor-parallel group
        if ctx.mp_comm_group is not None:
            dist.all_reduce(gradient, dist.ReduceOp.SUM, group=ctx.mp_comm_group)
        return gradient, None


class Megatron_g(torch.autograd.Function):
    """The g function in Figure 3 in the Megatron-LM paper."""

    @staticmethod
    def forward(ctx, x, mp_comm_group=None):
        # Sum partial outputs across the tensor-parallel group
        if mp_comm_group is not None:
            dist.all_reduce(x, dist.ReduceOp.SUM, group=mp_comm_group)
        return x

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


# ---- Tensor-parallel FFN (Fig. 3a) ----
class Megatron_FeedForward(nn.Module):
    """FFN/MLP block with Megatron-LM tensor parallelism (Fig. 3a)."""

    def __init__(self, n_embd: int, mp_comm_group=None):
        super().__init__()
        self.mp_comm_group = mp_comm_group

        # Fig 3a: split first GEMM across columns; second GEMM across rows
        n_embd_mid = n_embd * 4
        if self.mp_comm_group:
            n_embd_mid //= dist.get_world_size(group=self.mp_comm_group)

        self.fc1 = nn.Linear(n_embd, n_embd_mid)
        self.fc2 = nn.Linear(n_embd_mid, n_embd, bias=False)  # no bias here
        self.fc2_bias = nn.Parameter(torch.zeros(n_embd))     # bias added after all-reduce
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.mp_comm_group:
            x = Megatron_f.apply(x, self.mp_comm_group)

        # Transformers / Megatron use GeLU in the MLP
        y = F.gelu(self.fc1(x))

        z = self.fc2(y)  # partial matmul
        if self.mp_comm_group:
            z = Megatron_g.apply(z, self.mp_comm_group)

        z = z + self.fc2_bias  # bias AFTER all-reduce
        z = self.dropout(z)
        return z


# ---- Tensor-parallel Multi-Head Attention (Fig. 3b) ----
class Megatron_MHA(nn.Module):
    """
    Megatron-LM tensor-parallel Multi-Head Self-Attention (Fig. 3b style):

    - Q/K/V GEMM is column-parallel: each rank owns a subset of heads (local heads).
      Use Megatron_f on the input (identity forward, all-reduce backward) to match
      column-parallel input-gradient behavior.

    - Attention (softmax path) is fully local per rank (no communication).

    - Output projection is row-parallel: each rank projects its local concat heads
      to n_embd, then Megatron_g all-reduces the output (sum) across ranks.
    """

    def __init__(self, n_head: int, mp_comm_group=None):
        super().__init__()
        self.mp_comm_group = mp_comm_group
        self.n_head = n_head

        self.tp_size = dist.get_world_size(group=mp_comm_group) if mp_comm_group else 1
        if n_head % self.tp_size != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by tp_size ({self.tp_size}).")

        self.n_head_local = n_head // self.tp_size

        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head}).")
        self.head_dim = n_embd // n_head
        self.hidden_local = self.n_head_local * self.head_dim

        # Column-parallel QKV: each rank produces Q,K,V only for its local heads.
        self.qkv = nn.Linear(n_embd, 3 * self.hidden_local, bias=False)

        # Row-parallel output projection
        self.proj = nn.Linear(self.hidden_local, n_embd, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        if C != n_embd:
            raise ValueError(f"Expected x.shape[-1] == {n_embd}, got {C}.")

        if self.mp_comm_group:
            x = Megatron_f.apply(x, self.mp_comm_group)

        qkv = self.qkv(x)  # (B, T, 3*hidden_local)
        q, k, v = qkv.split(self.hidden_local, dim=-1)

        # (B, T, hidden_local) -> (B, n_head_local, T, head_dim)
        q = q.view(B, T, self.n_head_local, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head_local, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head_local, self.head_dim).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v  # (B, n_head_local, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_local)

        out = self.proj(out)  # partial contribution
        if self.mp_comm_group:
            out = Megatron_g.apply(out, self.mp_comm_group)

        out = self.out_dropout(out)
        return out


def get_megatron_tensor_parallel_model(vocab_size, mp_comm_group=None):
    """
    Returns a GPTlite model where the FFN and Attention are replaced with
    Megatron-LM tensor-parallel versions.

    If mp_comm_group is None, creates a tensor-parallel group over all ranks.
    """
    if mp_comm_group is None:
        mp_comm_group = dist.new_group(list(range(dist.get_world_size())))

    # Replace definition of base model's MLP with Megatron tensor-parallel version
    gptlite.FeedForward = lambda *args, **kwargs: Megatron_FeedForward(*args, mp_comm_group=mp_comm_group, **kwargs)
    gptlite.MHA = lambda *args, **kwargs: Megatron_MHA(*args, mp_comm_group=mp_comm_group, **kwargs)

    return gptlite.GPTlite(vocab_size)
