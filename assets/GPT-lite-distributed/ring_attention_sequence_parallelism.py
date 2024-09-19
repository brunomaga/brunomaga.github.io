import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import flash_attn.flash_attn_interface as fa

# use FeedForwars from base GPTlite model from the GPT-lite post
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
from gptlite import FeedForward


class MultiHeadAttention(nn.Module):

    def __init__( self, n_embd, d_head, n_heads = 8, dropout_p = 0.1, group = None ):
        super().__init__()

        self.d_head = d_head
        self.n_heads = n_heads
        self.queries = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.keys = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.values = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, n_embd)
        self.dropout = nn.Dropout(dropout_p)
        self.group = group  # Ring Attention group group
        if self.group is None:
            self.group = dist.new_group(range(dist.get_world_size())) 

    class RingAttention(torch.autograd.Function):

        @staticmethod
        def acc_out_and_lse(out, lse, block_out, block_lse):
            # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795

            if out is None:  # first block, allocate results tensors
                out = block_out
                lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

            block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

            out = out - F.sigmoid(block_lse - lse) * (out - block_out)
            lse = lse - F.logsigmoid(lse - block_lse)
            return out, lse

        @staticmethod
        def forward( ctx, q, k, v, group ):

            P = group.size()
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous() # (B, T/P, H, E)

            out = lse = None  # accumulators
            recv_k, recv_v = torch.empty_like(k), torch.empty(v)  # recv buffers

            for _ in range(P): # do P ring steps
                # "overlapping the communication of key-value blocks with the computation of blockwise attention."
                all_reqs = MultiHeadAttention.isend_k_and_v(k, v, recv_k, recv_v, group)

                # forward pass of attention function for the K, V, and Q for this block
                block_out, _, _, _, _, block_lse, _, _ = fa._flash_attn_forward(q,k,v)

                # update out and lse
                out, lse = MultiHeadAttention.RingAttention.acc_out_and_lse(out, lse, block_out, block_lse)

                # wait for K and V for the next iteration (final iteration will revert K and V to original proc)
                for req in all_reqs:
                    req.wait()

            lse = lse.squeeze(dim=-1).transpose(1, 2)

            ctx.group = group # save for backward
            ctx.save_for_backward(q, k, v, out, lse)
            return out

        @staticmethod
        def backward(ctx, dout, *args):

            P = ctx.group.size()
            q, k, v, out, softmax_lse = ctx.saved_tensors

            block_dq, block_dk, block_dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

            dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

            recv_k, recv_v = torch.empty_like(k), torch.empty_like(v)  # recv buffers for K and V
            recv_dk, recv_dv = torch.empty_like(dk), torch.empty_like(dv)  # recv buffers for dK and dV

            for step in range(P):
                all_k_v_reqs = MultiHeadAttention.isend_k_and_v(k, v, recv_k, recv_v, group)

                fa._flash_attn_backward(dout=dout, q=q, k=k, v=k, out=out, softmax_lse=softmax_lse,
                                        dq=block_dq, dk=block_dk, dv=block_dv)

                # K and V are rotated, so dK and dV must also be rotated and accumulated in a ring fashion
                if step > 0:
                    for req in all_dk_dv_reqs:
                        req.wait()

                dq += block_dq
                dk += block_dk
                dv += block_dv

                all_dk_dv_reqs = MultiHeadAttention.isend_k_and_v(dk, dv, recv_dk, recv_dv, group)

                # wait for K and V for the next iteration
                for req in all_k_v_reqs:
                    req.wait()

            for req in all_dk_dv_reqs:
                req.wait()
            return dq, dk, dv, None

    @staticmethod
    def isend_k_and_v( k, v, recv_k, recv_v,  group):
        P, rank = group.size(), group.rank()
        dst = (rank + 1) % P
        src = (rank - 1) % P
        req_k_send = dist.P2POp(dist.isend, k, dst, group, 1)
        req_k_recv = dist.P2POp(dist.irecv, recv_k, src, group, 1)
        req_v_send = dist.P2POp(dist.isend, v, dst, group, 2)
        req_v_recv = dist.P2POp(dist.irecv, recv_v, src, group, 2)
        all_reqs = [req_k_send, req_k_recv, req_v_send, req_v_recv]
        dist.batch_isend_irecv(all_reqs)
        return all_reqs 

    def forward(self, x):
        P, B, T, = self.group.size(), x.shape[0], x.shape[1] * self.group.size()

        # take Q, K and V, and collect all head embeddings: (B, T/P, E) -> (H, B, T/P, E)
        q = torch.stack([q(x) for q in self.queries], dim=0)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        if P == 1:
            out = self.flash_attn_func(q, k, v)
        else:
            out = MultiHeadAttention.RingAttention.apply( q, k, v, self.group)

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

    # model constants (use these or import from GPT-lite post). Naming matches post
    P = dist.get_world_size()
    B = 4
    T = 2048 # the length of the sequence
    H = 8 # the number of heads
    E = 128 # the size of the head
    n_embd = 256 # the hidden size of the model
    n_blocks = 12 # number of transformer blocks

    # sanity checks
    assert T % P == 0, "seqlen must be divisible by number of processes"
    assert H % P == 0, "n_heads must be divisible by number of processes"

    x = torch.randint(0, 5, (B, T, n_embd)).to(device=device).float() # dummy input
    y = torch.ones_like(x).float() # dummy label

    # build model as sequence of blocks
    blocks = nn.Sequential(*[Block(n_embd, E, H, group=group) for _ in range(n_blocks)]).to(device=device)
    blocks = DistributedDataParallel(blocks, device_ids=[local_rank], static_graph=True, process_group=group)
    optimizer = torch.optim.Adam(blocks.parameters())

    # run 10 random iterations
    for i in range(10):
        out = blocks(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    dist.destroy_process_group()
