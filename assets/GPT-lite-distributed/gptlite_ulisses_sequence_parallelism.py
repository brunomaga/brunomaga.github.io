import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


# use FeedForwars from base GPTlite model from the GPT-lite post
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
from gptlite import FeedForward

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

    @staticmethod
    def from_dist_H_to_T(tensor: torch.Tensor, group: dist.ProcessGroup = None):  # noqa
        """convert a distributed tensor from shape (H/P, B, T, E) to (H, B, T/P, E)"""

        assert tensor.dim() == 4, f"expected 4D tensor, got {tensor.dim()}"
        P = group.size()  # noqa
        H, B, T, E = tensor.shape[0] * P, tensor.shape[1], tensor.shape[2], tensor.shape[3]  # noqa
        send = torch.cat([tensor.chunk(P, dim=2)[r].contiguous().flatten() for r in range(P)])
        recv = torch.zeros_like(send)
        dist.all_to_all_single(output=recv, input=send, group=group)
        recv = torch.cat([recv.chunk(P)[r].view(H // P, B, T // P, E) for r in range(P)], dim=0)
        assert recv.shape == (H, B, T // P, E), f"wrong shape after from_dist_H_to_T: {recv.shape}"
        return recv

    @staticmethod
    def from_dist_T_to_H(tensor: torch.Tensor, group: dist.ProcessGroup = None):  # noqa
        """convert a distributed tensor from shape (H, B, T/P, E) to (H/P, B, T, E)"""

        assert tensor.dim() == 4, f"expected 4D tensor, got {tensor.dim()}"
        P = group.size()  # noqa
        H, B, T, E = tensor.shape[0], tensor.shape[1], tensor.shape[2] * P, tensor.shape[3]  # noqa
        send = torch.cat([tensor.chunk(P, dim=0)[r].contiguous().flatten() for r in range(P)])
        recv = torch.zeros_like(send)
        dist.all_to_all_single(output=recv, input=send, group=group)
        recv = torch.cat([recv.chunk(P)[r].view(H // P, B, T // P, E) for r in range(P)], dim=2)
        return recv

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


class Block(nn.Module):
    """a GPT-like block, parallelized a la DeepSpeed Ulysses"""

    def __init__(self, n_embd, d_head, n_heads=8, ulysses_group=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, d_head, n_heads=n_heads, ulysses_group=ulysses_group)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffw = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


if __name__ == "__main__":
    # launch with torchrun or deepspeed launcher

    dist.init_process_group()
    torch.manual_seed(0)

    # system constants
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # model constants (use these or import from GPT-lite post)
    batch_size = 4
    n_embd = 128
    seqlen = 2**10
    n_heads = 2**3
    n_blocks = 4
    d_head = n_embd // n_heads

    # comm group that participates in the seq parallelism of this data parallel group
    ulysses_group = dist.new_group(range(dist.get_world_size()))

    # sanity checks
    assert seqlen % dist.get_world_size() == 0, "seqlen must be divisible by number of processes"
    assert n_heads % dist.get_world_size() == 0, "n_heads must be divisible by number of processes"

    # all processes in a group need the same input (via the same seed or same dataloader id)
    x = torch.randint(0, 10, (batch_size, seqlen, n_embd)).to(dtype=dtype, device=device)
    if ulysses_group is not None:
        # split data samples across processes, on time dimension
        x = x.chunk(dist.get_world_size(), dim=1)[ulysses_group.rank()]  # from [B, T,E] to [B, T/P, E]
    y = torch.ones_like(x)

    # run (for simplicity, we only run GPT blocks, ie no input embeddings or output head)
    blocks = nn.Sequential(*[Block(n_embd, d_head, n_heads, ulysses_group=ulysses_group) for _ in range(n_blocks)])
    blocks = blocks.to(dtype=dtype, device=device)
    blocks = DistributedDataParallel(blocks, device_ids=[local_rank], static_graph=True, process_group=ulysses_group)
    optimizer = torch.optim.Adam(blocks.parameters())

    for i in range(5):
        out = blocks(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()  # loss per process
        optimizer.step
        optimizer.zero_grad(set_to_none=True)

    # cleanup
    dist.destroy_process_group()
