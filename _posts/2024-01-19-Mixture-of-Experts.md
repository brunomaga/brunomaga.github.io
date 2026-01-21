---
layout: post
title:  "Mixture-of-Experts: a publications timeline, with serial and distributed implementations"
categories: [machine learning, Transformer, GPT, mixture-of-experts]
tags: [machinelearning]
---

Public details about GPT-4’s architecture have **not** been released by OpenAI. Various third‑party reports and leaks have *speculated* that GPT‑4 uses a Mixture‑of‑Experts (MoE) design with multiple large experts, but these claims are unverified. (If you're curious about where MoE fits into modern large-model training systems, OpenAI’s public overview of large‑scale training techniques includes MoE-style ideas as one ingredient among many.)

This post is instead about the **MoE idea itself**: how it evolved from early “mixture” models to modern sparse routing, what practical issues arise (routing, sparsity, load imbalance, auxiliary losses, expert capacity, numerical stability, and fine‑tuning), and how to implement both a **dense** MoE (all experts run) and a **sparse, distributed** MoE (only a few experts run, and experts are sharded across GPUs) using PyTorch distributed primitives.

## Early days: dense MoEs as weighted sums of expert outputs

In 1991, *Adaptive Mixtures of Local Experts* (Jacobs et al.) and *Task Decomposition Through Competition in a Modular Connectionist Architecture* (Jordan & Jacobs) introduced early versions of what we now call a Mixture‑of‑Experts.

The setup is simple:
- A set of independent **experts** (often small MLPs), each producing an output vector \(o_i=f_i(x)\).
- A **gating network** \(g(x)\) that produces a probability distribution \(p_i\) over experts (often a softmax).

A common formulation is:

\[
p = \mathrm{softmax}(x W_g), \qquad
z = \sum_i p_i \, f_i(x).
\]

This is a *dense* MoE: every expert runs for every input. Dense MoEs already show the key behavior that motivates MoEs today: different experts specialize in different parts of the input space, and the gate learns how to mix them.

### A minimal dense MoE implementation

Below is a small, readable PyTorch implementation. The main point is correctness and clarity (not performance).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, d_model: int, n_experts: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C] -> probs: [B, T, E]
        logits = self.proj(self.dropout(x))
        return F.softmax(logits, dim=-1)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, dropout: float = 0.0):
        super().__init__()
        self.router = Router(d_model, n_experts, dropout=dropout)
        self.experts = nn.ModuleList([FeedForward(d_model, dropout=dropout) for _ in range(n_experts)])

    def forward(self, x):
        # x: [B, T, C]
        # returns:
        #   y:     [B, T, C]
        #   probs: [B, T, E]
        #   outs:  [B, T, E, C]
        probs = self.router(x)  # [B, T, E]
        outs = torch.stack([ex(x) for ex in self.experts], dim=-2)  # [B, T, E, C]
        y = (outs * probs.unsqueeze(-1)).sum(dim=-2)               # [B, T, C]
        return y, probs, outs
```

### A note on the 1991 “anti-starvation” objective

A subtle issue appears even in dense MoEs: if one expert becomes slightly better early on, the gate may prefer it more, making it even better—a feedback loop that can “starve” other experts.

The 1991 work discusses objectives that reduce this collapse. Two simplified forms you often see discussed are:

1) **Mixture loss** (gate weights the *combined* prediction error):
\[
E = \left\|d - \sum_i p_i o_i \right\|^2.
\]

2) **Expected expert loss** (gate weights *each expert’s* error):
\[
E = \sum_i p_i \left\|d - o_i \right\|^2.
\]

The second form can provide a more direct learning signal to non‑dominant experts, depending on the task and parameterization.

Here is a small, shape-correct implementation of the “expected expert loss” for a token classification setting:

```python
def expected_expert_mse(probs, outs, labels):
    # probs:  [B, T, E]
    # outs:   [B, T, E, V]  (expert logits over V classes)
    # labels: [B, T]        (class indices)
    B, T, E, V = outs.shape
    one_hot = F.one_hot(labels, num_classes=V).to(outs.dtype)          # [B, T, V]
    mse = (outs - one_hot.unsqueeze(-2)).square().mean(dim=-1)         # [B, T, E]
    return (mse * probs).sum(dim=-1).mean()                            # scalar
```

## Deep Mixture of Experts

With the resurgence of deep networks, *Learning Factored Representations in a Deep Mixture of Experts* (2014) explored stacking multiple MoE layers (a “Deep MoE”). Stacking introduces two practical complications:

1) **Training collapse can happen at each layer.** The same “rich get richer” effect becomes multi-layered.
2) **The interface between layers matters.** Some designs pass only the gated weighted sum; others expose more of the expert structure (e.g., concatenations or richer routing).

The paper reports that deep mixtures can learn useful hierarchies (e.g., early experts focusing on “where” features and later experts on “what” features), but also that deep mixtures can overfit and may need regularization or training tricks.

### A shape-correct deep MoE skeleton

A common mistake is to try to put a tuple-returning MoE inside `nn.Sequential`. Below is a safe pattern that keeps routing outputs for analysis while passing only the mixture output forward.

```python
class DeepMoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        assert depth >= 1
        self.layers = nn.ModuleList([MoE(d_model, n_experts, dropout=dropout) for _ in range(depth)])

    def forward(self, x):
        all_probs = []
        for layer in self.layers:
            x, probs, _ = layer(x)
            all_probs.append(probs)
        return x, all_probs
```

## Sparsely-Gated Mixture of Experts

Dense MoEs scale poorly: *every* expert runs, even if only a few are useful for a token. The 2017 paper *Outrageously Large Neural Networks: The Sparsely‑Gated Mixture‑of‑Experts Layer* introduced modern **sparse routing** (a.k.a. conditional computation): only the top‑\(k\) experts are executed per token, allowing the total parameter count (number/size of experts) to grow without multiplying FLOPs proportionally.

The high-level change is:
- Router produces logits over experts.
- Keep only top‑\(k\) experts per token; other experts get probability 0 and are not executed.

### Noisy top-k gating and auxiliary losses (importance + load)

A key contribution is **noisy top‑\(k\) gating**: adding input‑dependent noise before selecting top‑\(k\), which improves exploration and helps avoid early collapse.

Just as important are the auxiliary losses the paper uses to keep routing balanced:
- **Importance loss**: encourages the *sum of gate values per expert* to be balanced.
- **Load loss**: encourages the *number of tokens routed per expert* to be balanced.

Both are expressed as the squared coefficient of variation \(CV^2\) (variance relative to mean), applied to vectors computed across the batch.

If you want an intuition: if one expert gets nearly all traffic, both importance and load vectors become highly skewed, increasing \(CV^2\), and the auxiliary penalty pushes the router back toward a more even distribution.

### Correcting a common arithmetic pitfall

When you compute \(CV=\sigma/\mu\), it is easy to make mental‑math mistakes. For example, for importances \([0.2, 0.1, 0.2, 2.4, 0.1]\), the mean is \(0.6\), the standard deviation is about \(0.90\), and \(CV \approx 1.50\) (not \(0.67\)).

## Conditional computation at scale: GShard

GShard (2020) scaled Transformer‑style sparse MoEs using TPU SPMD partitioning and sharding annotations in XLA. One architectural detail that is easy to miss is that in their largest models they **replace the FFN with an MoE in alternating layers** (“every other Transformer layer uses an MoE feed-forward”). This keeps compute and memory in check while still adding conditional capacity.

GShard also introduces a routing tweak sometimes described as **random routing** for the second expert in top‑2: the first expert always receives the token, while the second expert is selected stochastically based on its routing weight. This reduces unnecessary second‑expert compute when the second expert’s weight is tiny, and it can reduce overflow pressure under capacity constraints.

## Switch Transformers

Switch Transformers (2021) simplify sparse routing by using **top‑1** routing (“switch routing”): each token is sent to only one expert. The key insight is that a carefully designed top‑1 system can remain stable and efficient, while avoiding the extra compute/communication of top‑2.

Compared to prior sparse MoEs, Switch emphasizes:
- Simpler routing and dispatch (top‑1).
- A practical **capacity factor** to bound tokens per expert per batch, dropping or skipping overflow tokens (with residual connections).
- A per‑Switch‑layer **auxiliary load‑balancing loss** to reduce expert collapse.

In practice, Switch’s claim isn’t that “top‑1 always works,” but that with the right training recipe and losses, top‑1 can be a sweet spot between efficiency and quality.

## Distributed implementation in PyTorch

A distributed sparse MoE introduces an extra systems problem: tokens routed to different experts must be **shuffled across GPUs**, experts run locally, and results must be **shuffled back** and combined.

Conceptually, the forward pass is four steps (often illustrated in systems papers like MegaBlocks):

1) **Route**: compute top‑\(k\) expert ids and weights for each token  
2) **Dispatch / permute**: move tokens to the GPUs that host the selected experts  
3) **Expert compute**: run the local expert on received tokens  
4) **Combine / unpermute**: send results back and accumulate into the original token order  

Below is a compact, *shape-correct* implementation for the common teaching setup “one expert per GPU”. It uses `all_to_all_single` twice: once to dispatch token embeddings to the owning expert rank, once to return the expert outputs to the original ranks. For simplicity it drops overflow tokens beyond capacity (but keeps the bookkeeping correct).

> This is intentionally “bare metal” to show what frameworks like DeepSpeed or Tutel automate.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def _all_to_all_counts(send_counts: torch.Tensor, group=None) -> torch.Tensor:
    # send_counts: [world] int64 counts of items this rank will send to each rank
    # returns recv_counts: [world] counts this rank will receive from each rank
    recv = torch.empty_like(send_counts)
    dist.all_to_all_single(recv, send_counts, group=group)  # transpose counts
    return recv

def _prefix_sums(counts):
    offs = torch.zeros(counts.numel() + 1, device=counts.device, dtype=torch.long)
    offs[1:] = torch.cumsum(counts, dim=0)
    return offs

class MoE_dist(nn.Module):
    # One expert per rank, sparse top-k routing.
    # Inputs/outputs are local to each rank; tokens are exchanged via all-to-all.
    def __init__(self, d_model: int, k: int = 2, capacity_factor: float = 1.25, drop_overflow: bool = True):
        super().__init__()
        assert dist.is_initialized(), "torch.distributed must be initialized"
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        self.k = k
        self.capacity_factor = capacity_factor
        self.drop_overflow = drop_overflow

        self.router = nn.Linear(d_model, self.world, bias=False)
        self.expert = FeedForward(d_model)

    def forward(self, x):
        # x: [B, T, C] local tokens on this rank
        device = x.device
        B, T, C = x.shape
        N = B * T

        # 1) route
        logits = self.router(x)                     # [B, T, world]
        probs = F.softmax(logits, dim=-1)           # [B, T, world]
        topk_probs, topk_experts = torch.topk(probs, k=self.k, dim=-1)  # [B, T, k]

        x_flat = x.view(N, C)                       # [N, C]
        ex_flat = topk_experts.view(N, self.k)      # [N, k]
        pr_flat = topk_probs.view(N, self.k)        # [N, k]

        # Create one message per (token, chosen expert). meta=(token_index, which_of_k).
        msgs_x, msgs_meta, msgs_probs, dest = [], [], [], []
        tok_idx_all = torch.arange(N, device=device)

        for j in range(self.k):
            chosen = ex_flat[:, j]                  # [N]
            msgs_x.append(x_flat)
            msgs_meta.append(torch.stack([tok_idx_all, torch.full_like(tok_idx_all, j)], dim=-1))
            msgs_probs.append(pr_flat[:, j])
            dest.append(chosen)

        msgs_x = torch.cat(msgs_x, dim=0)           # [M, C]  where M=N*k
        msgs_meta = torch.cat(msgs_meta, dim=0)     # [M, 2]
        msgs_probs = torch.cat(msgs_probs, dim=0)   # [M]
        dest = torch.cat(dest, dim=0)               # [M]

        # Sort by destination expert (rank) so we can do a single all-to-all with splits.
        order = torch.argsort(dest)
        dest = dest[order]
        msgs_x = msgs_x[order]
        msgs_meta = msgs_meta[order]
        msgs_probs = msgs_probs[order]
        M = msgs_x.shape[0]

        send_counts = torch.bincount(dest, minlength=self.world).to(torch.long)    # [world]
        recv_counts = _all_to_all_counts(send_counts.to(torch.int64)).to(torch.long)

        send_off = _prefix_sums(send_counts)
        recv_off = _prefix_sums(recv_counts)

        # Optional capacity: keep at most cap messages per destination rank.
        if self.drop_overflow:
            cap = int((N / self.world) * self.capacity_factor)
            keep = torch.zeros(M, device=device, dtype=torch.bool)
            for r in range(self.world):
                s0, s1 = send_off[r].item(), send_off[r+1].item()
                if s1 > s0:
                    keep[s0 : min(s0 + cap, s1)] = True
            dest = dest[keep]
            msgs_x = msgs_x[keep]
            msgs_meta = msgs_meta[keep]
            msgs_probs = msgs_probs[keep]

            send_counts = torch.bincount(dest, minlength=self.world).to(torch.long)
            recv_counts = _all_to_all_counts(send_counts.to(torch.int64)).to(torch.long)
            send_off = _prefix_sums(send_counts)
            recv_off = _prefix_sums(recv_counts)

        # 2) dispatch
        recv_total = int(recv_counts.sum().item())

        # a) embeddings
        send_x = msgs_x.contiguous().view(-1)
        recv_x = torch.empty(recv_total * C, device=device, dtype=x.dtype)
        dist.all_to_all_single(
            recv_x, send_x,
            output_split_sizes=(recv_counts * C).tolist(),
            input_split_sizes=(send_counts * C).tolist(),
        )
        recv_x = recv_x.view(recv_total, C)

        # b) meta
        send_meta = msgs_meta.contiguous().view(-1)
        recv_meta = torch.empty(recv_total * 2, device=device, dtype=msgs_meta.dtype)
        dist.all_to_all_single(
            recv_meta, send_meta,
            output_split_sizes=(recv_counts * 2).tolist(),
            input_split_sizes=(send_counts * 2).tolist(),
        )
        recv_meta = recv_meta.view(recv_total, 2)

        # c) probs
        send_p = msgs_probs.contiguous()
        recv_p = torch.empty(recv_total, device=device, dtype=msgs_probs.dtype)
        dist.all_to_all_single(
            recv_p, send_p,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

        # 3) local expert compute
        out_local = self.expert(recv_x)                                 # [recv_total, C]

        # 4) return + combine (reverse all-to-all sizes)
        send_back = out_local.contiguous().view(-1)
        recv_back = torch.empty(int(send_counts.sum().item()) * C, device=device, dtype=out_local.dtype)
        dist.all_to_all_single(
            recv_back, send_back,
            output_split_sizes=(send_counts * C).tolist(),
            input_split_sizes=(recv_counts * C).tolist(),
        )
        recv_back = recv_back.view(int(send_counts.sum().item()), C)

        send_meta_back = recv_meta.contiguous().view(-1)
        recv_meta_back = torch.empty(int(send_counts.sum().item()) * 2, device=device, dtype=recv_meta.dtype)
        dist.all_to_all_single(
            recv_meta_back, send_meta_back,
            output_split_sizes=(send_counts * 2).tolist(),
            input_split_sizes=(recv_counts * 2).tolist(),
        )
        recv_meta_back = recv_meta_back.view(int(send_counts.sum().item()), 2)

        send_p_back = recv_p.contiguous()
        recv_p_back = torch.empty(int(send_counts.sum().item()), device=device, dtype=recv_p.dtype)
        dist.all_to_all_single(
            recv_p_back, send_p_back,
            output_split_sizes=send_counts.tolist(),
            input_split_sizes=recv_counts.tolist(),
        )

        # Combine: sum contributions per original token index
        y_flat = torch.zeros(N, C, device=device, dtype=out_local.dtype)
        tok_idx = recv_meta_back[:, 0].long()
        contrib = recv_back * recv_p_back.unsqueeze(-1)
        y_flat.index_add_(0, tok_idx, contrib)
        return y_flat.view(B, T, C)
```

### Applying the MoE to an existing LLM

An MoE module typically replaces the FFN sub‑block of a Transformer layer. For example, in a GPT‑like block, you keep attention and layer norms dense and replace `ffwd` with `MoE_dist()` (or a library implementation).

If you want a production‑ready implementation, it is usually better to use an MoE runtime like **DeepSpeed MoE**, **Tutel**, or frameworks that integrate fused dispatch/compute/combination and optimized kernels.

## Further reading

There is a lot of ongoing work on MoE training stability, dispatch efficiency, dropless routing, and fine‑tuning. If you want to go deeper, here are a few good entry points (with brief notes):

{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)</summary>

MegaBlocks is a system for efficient sparse MoE training on GPUs that targets the **model quality vs hardware efficiency** trade‑off caused by dynamic routing.

A common pain point is expert imbalance: to keep tensors rectangular, many implementations either **drop overflow tokens** or **pad** to a large capacity factor. Dropping hurts quality; padding hurts performance. MegaBlocks tackles this by reformulating MoE computation as **block‑sparse operations** and implementing kernels that can handle imbalanced routing efficiently.

The paper reports end‑to‑end speedups (relative to then‑state‑of‑the‑art baselines) by combining better routing/packing with custom block‑sparse kernels, enabling “dropless” behavior with less wasted compute.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2022 [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)</summary>

ST‑MoE is a deep dive into training stability for large sparse MoEs. It introduces the **router z‑loss**, which penalizes large router logits and improves stability.

The paper also documents practical stability/quality trade‑offs (dropout, noise, optimizer choice, precision, clipping), and provides a set of default recommendations (e.g., top‑2 routing, reasonable capacity factor, and one expert per core as a starting point).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2024 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)</summary>

Mixtral 8×7B is a widely used open MoE model family that replaces Transformer FFNs with 8 experts and uses **top‑2 routing**. In the “8×7B” variant, each token routes to two experts, so the token “has access” to a larger parameter set, while only a subset of parameters is active per token at inference.

The Mixtral technical report also discusses practical choices like SwiGLU‑style FFNs, gating design, and how top‑2 routing influences both quality and throughput.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">2024 [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)</summary>

Mixture‑of‑Depths (MoD) applies conditional computation across the **depth** of the Transformer: per layer, a router selects which tokens should receive full computation and which should skip via residuals, under a fixed compute budget.

A key idea is that if only a subset of tokens are “active” in a layer, both attention and FFN compute can be reduced while still preserving model quality. The work explores several routing strategies (token‑choice vs expert‑choice‑like selection) and analyzes the stability and efficiency implications.

</details>
{::options parse_block_html="false" /}
