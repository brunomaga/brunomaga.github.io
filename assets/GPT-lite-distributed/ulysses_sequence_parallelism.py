import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from flash_attn.flash_attn_interface import flash_attn_func

# use FeedForwars from base GPTlite model from the GPT-lite post
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
from gptlite import FeedForward

class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd=256, d_head=128, n_heads=8, dropout_p=0.1, group=None):
        """ An Ulysses multi-head attention. Variable names follow GPT-lite's post """

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

    class first_alltoall(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, group=None):
            ctx.group = group  # save for backward pass
            return MultiHeadAttention.dist_view_swap(x, old_split_dim=2, new_split_dim=0, group=ctx.group)

        @staticmethod
        def backward(ctx, dout):
            dout = MultiHeadAttention.dist_view_swap(dout, old_split_dim=0, new_split_dim=2, group=ctx.group)
            return dout, None
        
    class second_alltoall(torch.autograd.Function): 

        @staticmethod
        def forward(ctx, x, group=None):
            ctx.group = group  # save for backward pass
            return MultiHeadAttention.dist_view_swap(x, old_split_dim=2, new_split_dim=0, group=ctx.group)

        @staticmethod
        def backward(ctx, dout):
            dout = MultiHeadAttention.dist_view_swap(dout, old_split_dim=0, new_split_dim=2, group=ctx.group)
            return dout, None

    @staticmethod
    def dist_view_swap(tensor: torch.Tensor, old_split_dim: int, new_split_dim: int, group: dist.ProcessGroup):
        """swaps the distributed split dimension of a tensor of shape (H, B, T, E) across P processes"""
        full_shape, P = list(tensor.shape), group.size()
        full_shape[old_split_dim]*=P # full distributed shape
        H, B, T, E = full_shape 
        send = torch.cat([tensor.chunk(P, dim=new_split_dim)[r].contiguous() for r in range(P)])
        recv = torch.zeros_like(send)
        dist.all_to_all_single(output=recv, input=send, group=group)
        recv = torch.cat([recv.chunk(P)[r].view(H // P, B, T // P, E) for r in range(P)], dim=old_split_dim)
        return recv

    def forward(self, x):
        P, B, T, = self.group.size(), x.shape[0], x.shape[1] * self.group.size()

        # Q, K and V embeddings: (B, T/P, E) -> (H, B, T/P, E)
        q = torch.stack([q(x) for q in self.queries], dim=0)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        if P>1:  #  (H, B, T/P, E) -> (H/P, B, T, E)
            q = MultiHeadAttention.first_alltoall.apply(q, self.group)
            k = MultiHeadAttention.first_alltoall.apply(k, self.group)
            v = MultiHeadAttention.first_alltoall.apply(v, self.group)

        dropout_p, softmax_scale = 0, q.shape[-1] ** (-0.5)
        out = flash_attn_func(q, k, v, dropout_p, softmax_scale)

        if P > 1:  # (H/P, B, T, E) -> (H, B, T/P, E)
            out = MultiHeadAttention.second_alltoall.apply(out, self.group)

        out = out.permute(1, 2, 0, 3)  # (H, B, T/P, E) -> (B, T/P, H, E)
        out = out.reshape(B, T // P, -1)  # (B, T/P, H, E) -> (B, T/P, H*E)
        out = self.proj(out)  # (B, T/P, H*E) -> (B, T/P, E)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, n_embd, d_head, n_heads=8, group=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, d_head, n_heads=n_heads, group=group)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffw = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


if __name__ == "__main__":

    # set up network variables
    dist.init_process_group()
    torch.manual_seed(42)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    group = dist.new_group(range(dist.get_world_size()))
    dtype = torch.bfloat16

    # model constants (use these or import from GPT-lite post). Naming matches post
    P = dist.get_world_size()
    B = 4 # batch size
    T = 2048 # the length of the sequence
    H = 8 # the number of heads
    E = 128 # the size of the head
    n_embd = 256 # the hidden size of the model
    n_blocks = 12 # number of transformer blocks

    # sanity checks
    assert T % P == 0, "seqlen must be divisible by number of processes"
    assert H % P == 0, "n_heads must be divisible by number of processes"

    x = torch.randint(0, 5, (B, T, n_embd)).to(device=device, dtype=dtype) # dummy input
    y = torch.ones_like(x) # dummy label

    # build model as sequence of blocks
    blocks = nn.Sequential(*[Block(n_embd, E, H, group=group) for _ in range(n_blocks)]).to(device=device, dtype=dtype)
    blocks = DistributedDataParallel(blocks, device_ids=[local_rank], static_graph=True, process_group=group)
    optimizer = torch.optim.Adam(blocks.parameters())

    # run few iterations
    for i in range(15):
        out = blocks(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"Iteration {i} loss: {loss}")

    dist.barrier(group=group)
    dist.destroy_process_group()
