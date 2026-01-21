import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader

# use current directory as import path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_dir)

from moe import Router

# use base GPTlite model from the GPT-lite post
sys.path.insert(0, os.path.join(current_dir, "..", "GPT-lite"))
from gptlite import n_embd, dropout, FeedForward, GPTlite

# user helper functions from the GPT-lite deepspeed post
sys.path.insert(0, os.path.join(current_dir, "..", "GPT-lite-DeepSpeed"))
from gptlite_ds import get_dataset

assert "LOCAL_RANK" in os.environ and "RANK" in os.environ, "env vars not set. Launch with torchrun."
local_rank = int(os.environ["LOCAL_RANK"])  # set by torchrun
global_rank = int(os.environ["RANK"])  # set by torchrun

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
dist.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(local_rank)

# How to do data-parallelism of non-MoEs: pick DDP or FSDP
# DataParallel = lambda model: torch.distributed.fsdp.FullyShardedDataParallel(model.to(device), device_id=local_rank)
DataParallel = lambda model: torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank])


class MoE_dist(nn.Module):
    '''
    A simple (educational) distributed MoE:

      - One expert per rank (world_size == num_experts)
      - Router replicated via DDP
      - Token dispatch/return via all_to_all_single
      - Combines top-k expert contributions via scatter-add (index_add_)
    '''

    def __init__(self, k=2, capacity_factor=1.25, padding_val=0):
        super().__init__()
        self.capacity_factor = float(capacity_factor)
        self.padding_val = padding_val

        # One expert per GPU / rank.
        self.num_experts = dist.get_world_size()
        self.k = int(k)

        # Router is replicated (data-parallel). Expert is local (model-parallel across ranks).
        self.router = DataParallel(Router(n_embd, self.num_experts, dropout=dropout))
        self.expert = FeedForward(n_embd).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]  -> returns: [B, T, C]
        B, T, C = x.shape

        # 1) ROUTING
        probs = self.router(x)  # [B, T, E]
        topk_probs, topk_experts = torch.topk(probs, k=self.k, dim=-1)  # [B, T, k]

        # For each expert, collect the (b, t, kpos) indices that route to it,
        # and the corresponding routing probabilities.
        ids_per_expert = [(topk_experts == expert).nonzero(as_tuple=False) for expert in range(self.num_experts)]
        probs_per_expert = [topk_probs[topk_experts == expert] for expert in range(self.num_experts)]

        # 2) PERMUTATION (DISPATCH)
        # Exchange counts first.
        send_count = [torch.tensor([len(ids)], dtype=torch.int64, device=device) for ids in ids_per_expert]  # per-dst-rank
        recv_count = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(self.num_experts)]
        dist.all_to_all(recv_count, send_count)

        send_count_i = [int(s.item()) for s in send_count]
        recv_count_i = [int(r.item()) for r in recv_count]

        def scaled_counts(counts, scale=1):
            return [int(c) * int(scale) for c in counts]

        # Metadata: (row_id, tok_id, kpos)
        # We shift row_id by global_rank*B so experts can group by the "global row".
        M = 3
        if sum(send_count_i) > 0:
            send_ids = torch.cat(ids_per_expert, dim=0).to(device)  # [Nsend, 3] => (b, t, kpos)
            send_ids = send_ids.to(torch.int64)
            send_ids[:, 0] += global_rank * B
            send_ids_flat = send_ids.flatten()
        else:
            send_ids = torch.zeros((0, M), dtype=torch.int64, device=device)
            send_ids_flat = send_ids.flatten()

        recv_ids = torch.zeros(sum(recv_count_i) * M, dtype=torch.int64, device=device)
        dist.all_to_all_single(
            recv_ids,
            send_ids_flat,
            output_split_sizes=scaled_counts(recv_count_i, M),
            input_split_sizes=scaled_counts(send_count_i, M),
        )
        recv_ids = recv_ids.view(-1, M)  # [Nrecv, 3]

        # Dispatch tokens themselves (duplicate tokens if top-k routes to multiple experts)
        if sum(send_count_i) > 0:
            per_expert_toks = []
            for ids in ids_per_expert:
                if ids.numel() == 0:
                    continue
                b_idx = ids[:, 0]
                t_idx = ids[:, 1]
                per_expert_toks.append(x[b_idx, t_idx, :])  # [n, C]
            send_toks = torch.cat(per_expert_toks, dim=0).contiguous().to(device)  # [Nsend, C]
            send_toks_flat = send_toks.flatten()
        else:
            send_toks = torch.zeros((0, C), dtype=x.dtype, device=device)
            send_toks_flat = send_toks.flatten()

        recv_toks = torch.zeros(sum(recv_count_i) * C, dtype=x.dtype, device=device)
        dist.all_to_all_single(
            recv_toks,
            send_toks_flat,
            output_split_sizes=scaled_counts(recv_count_i, C),
            input_split_sizes=scaled_counts(send_count_i, C),
        )
        recv_toks = recv_toks.view(-1, C)  # [Nrecv, C]

        # 3) COMPUTATION (LOCAL EXPERT)
        # Group by row_id (sentence) and pad/crop per-row to a capacity.
        if recv_toks.numel() > 0:
            uniq_rows, recv_row_lens = recv_ids[:, 0].unique(sorted=True, return_counts=True)
            recv_row_offsets = torch.cat(
                [torch.zeros(1, dtype=torch.int64, device=device), torch.cumsum(recv_row_lens.to(torch.int64), dim=0)]
            ).tolist()

            def row_slice(i):
                return slice(int(recv_row_offsets[i]), int(recv_row_offsets[i + 1]))

            capacity = max(1, int((T / self.num_experts) * self.capacity_factor))

            def pad_or_crop(toks: torch.Tensor) -> torch.Tensor:
                toks = toks[:capacity]
                if toks.shape[0] < capacity:
                    toks = F.pad(toks, (0, 0, 0, capacity - toks.shape[0]), value=self.padding_val)
                return toks

            batch_toks = torch.stack([pad_or_crop(recv_toks[row_slice(i)]) for i in range(len(uniq_rows))], dim=0)
            batch_toks = self.expert(batch_toks)  # [Rows, Capacity, C]

            # Flatten the computed results back into recv_toks layout (without padded tail).
            out_recv_toks = torch.full_like(recv_toks, fill_value=float(self.padding_val))
            for i in range(len(uniq_rows)):
                s = row_slice(i)
                n = min(int(recv_row_lens[i].item()), capacity)
                out_recv_toks[s][:n] = batch_toks[i, :n, :]

            recv_toks = out_recv_toks

        # 4) UN-PERMUTATION (RETURN RESULTS)
        send_toks_back = torch.full((sum(send_count_i), C), fill_value=float(self.padding_val), dtype=x.dtype, device=device)
        dist.all_to_all_single(
            send_toks_back.flatten(),
            recv_toks.flatten(),
            output_split_sizes=scaled_counts(send_count_i, C),
            input_split_sizes=scaled_counts(recv_count_i, C),
        )
        x_entries = send_toks_back  # [Nsend, C] in the same order as send_ids

        # 5) SCALE + COMBINE (scatter-add over top-k routes)
        if sum(send_count_i) == 0:
            return x  # nothing routed (shouldn't happen in practice)

        probs_flat = torch.cat(probs_per_expert, dim=0).to(device).view(-1, 1)  # [Nsend, 1]
        x_entries = x_entries * probs_flat  # weight each routed copy

        # Map each routed entry back to a local (b, t) and sum contributions.
        local_b = (send_ids[:, 0] - (global_rank * B)).to(torch.int64)
        local_t = send_ids[:, 1].to(torch.int64)
        pos = local_b * T + local_t  # [Nsend]

        out = torch.zeros((B * T, C), dtype=x.dtype, device=device)
        out.index_add_(0, pos, x_entries)
        return out.view(B, T, C)


if __name__ == "__main__":
    torch.manual_seed(1234 + global_rank)

    vocab_size, batch_size = 65, 1
    n_epochs = 2  # keep tiny for a quick smoke test
    criterion = torch.nn.CrossEntropyLoss()

    dataset, _, vocab_size = get_dataset()
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), drop_last=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )

    # Instantiate model and apply DataParallel to all layers except our MoE FeedForward
    model = GPTlite(vocab_size).to(device)
    model.token_embedding_table = DataParallel(model.token_embedding_table)
    model.position_embedding_table = DataParallel(model.position_embedding_table)
    model.ln = DataParallel(model.ln)
    model.lm_head = DataParallel(model.lm_head)

    for block in model.blocks:
        block.sa = DataParallel(block.sa)
        block.ln1 = DataParallel(block.ln1)
        block.ln2 = DataParallel(block.ln2)
        block.ffwd = MoE_dist().to(device)  # replace FeedForward with MoE (model-parallel inside)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    model.train()

    for epoch in range(n_epochs):
        for step, data in enumerate(dataloader):
            inputs = data[0].to(device, non_blocking=True)
            labels = data[1].to(device, non_blocking=True)

            logits = model(inputs)  # [B, T, vocab]
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_rank == 0:
                print(f"Epoch: {epoch}, Iteration: {step}, Loss: {loss.item():.4f}")

        if global_rank == 0:
            print(f"Epoch done: {epoch}, Last loss: {loss.item():.4f}")
