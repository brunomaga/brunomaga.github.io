import os
import sys

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
    """ A Ring Attention multi-head attention. Variable names follow GPT-lite's post """

    def __init__( self, n_embd, d_head, n_heads = 8, dropout_p = 0.1, group = None ):
        super().__init__()

        self.d_head = d_head
        self.n_heads = n_heads
        self.queries = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.keys = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.values = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, n_embd)
        self.dropout = nn.Dropout(dropout_p)
        self.group = group  # Ring Attention group
        if self.group is None:
            self.group = dist.new_group(range(dist.get_world_size())) 

    class RingAttention(torch.autograd.Function):

        @staticmethod
        def accumulate_out_and_lse(out, lse, block_out, block_lse):
            # source: https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795

            if out is None: 
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
            recv_k, recv_v = torch.empty_like(k), torch.empty_like(v)  # recv buffers

            for step in range(P): # do P ring steps
                # send already the K and V for next step, asynchronously
                reqs_k_v = MultiHeadAttention.isend_k_and_v(k, v, recv_k, recv_v, group)

                # compute attention output and softmax lse for current block
                dropout_p, softmax_scale = 0, q.shape[-1] ** (-0.5)
                kwargs = dict(causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_softmax=False)
                block_out, _, _, _, _, block_lse, _, _ = fa._flash_attn_forward(q,k,v, dropout_p, softmax_scale, **kwargs)

                # update out and lse
                out, lse = MultiHeadAttention.RingAttention.accumulate_out_and_lse(out, lse, block_out, block_lse)

                # wait for new K and V before starting the next iteration
                [ req.wait() for req in reqs_k_v]
                k, v = recv_k, recv_v

            ctx.group = group # save for backward
            out = out.to(dtype=q.dtype)
            ctx.save_for_backward(q, k, v, out, lse)
            return out

        @staticmethod
        def backward(ctx, dout, *args):

            P = ctx.group.size()
            q, k, v, out, softmax_lse = ctx.saved_tensors
            softmax_lse = softmax_lse.squeeze(dim=-1).transpose(1, 2)

            block_dq, block_dk, block_dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v) # output buffers
            dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v) # accumulators of gradients

            recv_k, recv_v = torch.empty_like(k), torch.empty_like(v)  # recv buffers for K and V
            recv_dk, recv_dv = torch.empty_like(dk), torch.empty_like(dv)  # recv buffers for dK and dV

            for step in range(P):

                # send already the K and V for next step, asynchronously
                reqs_k_v = MultiHeadAttention.isend_k_and_v(k, v, recv_k, recv_v, group)

                # compute the gradients for the current block K, V and Q
                dropout_p, softmax_scale = 0, q.shape[-1] ** (-0.5)
                kwargs = dict(causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, rng_state=None)
                fa._flash_attn_backward(dout, q, k, k, out, softmax_lse, block_dq, block_dk, block_dv, dropout_p, softmax_scale, **kwargs)

                if step > 0:
                    # wait for dK and dV from the previous steps, they're the dK and dV accumulators
                    [ req.wait() for req in reqs_dk_dv]
                    dk, dv = recv_dk, recv_dv

                dq += block_dq
                dk += block_dk
                dv += block_dv

                reqs_dk_dv = MultiHeadAttention.isend_k_and_v(dk, dv, recv_dk, recv_dv, group)

                # wait for new K and V before starting the next iteration
                [ req.wait() for req in reqs_k_v]
                k, v = recv_k, recv_v

            # before returning, wait for the last dK and dV, that relate to this process block
            [ req.wait() for req in reqs_dk_dv]
            dk, dv = recv_dk, recv_dv
            return dq, dk, dv, None

    @staticmethod
    def isend_k_and_v( k, v, recv_k, recv_v,  group):
        P, rank = group.size(), group.rank()
        dst = (rank + 1) % P  # the rank of the next process
        src = (rank - 1 -P) % P  # the rank of the previous process
        req_k_send = dist.P2POp(dist.isend, k, dst, group)
        req_v_send = dist.P2POp(dist.isend, v, dst, group)
        req_k_recv = dist.P2POp(dist.irecv, recv_k, src, group)
        req_v_recv = dist.P2POp(dist.irecv, recv_v, src, group)
        return dist.batch_isend_irecv([req_k_send, req_v_send, req_k_recv, req_v_recv])

    def forward(self, x):
        P, B, T, = self.group.size(), x.shape[0], x.shape[1] * self.group.size()

        # take Q, K and V, and collect all head embeddings: (B, T/P, E) -> (H, B, T/P, E)
        q = torch.stack([q(x) for q in self.queries], dim=0)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        if P == 1:
            out = fa.flash_attn_func(q, k, v)
        else:
            out = MultiHeadAttention.RingAttention.apply( q, k, v, self.group)

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
        if dist.get_rank() == 0:
            print(f"Iteration {i} loss: {loss}")

    dist.barrier(group=group)
    dist.destroy_process_group()
