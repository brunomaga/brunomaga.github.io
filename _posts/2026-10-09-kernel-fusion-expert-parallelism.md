---
layout: post
title:  "Kernel Fusion for MoE Expert Parallelism"
categories: ["Efficient Inference"]
tags: ["LLM", "kernel fusion", "MoE", "expert parallelism", "in-network computing", "collectives"]
---

This post covers a kernel that fuses a whole [Mixture-of-Experts (MoE)]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) layer, distributed with expert parallelism, into a single kernel, and moves its communication into the network. It follows the [post on kernel fusion for Megatron-LM tensor parallelism]({{ site.baseurl }}{% post_url 2026-08-18-kernel-fusion-tensor-parallelism %}), and summarizes the paper *MoeInc: an Expert Parallelism MegaKernel with Overlap of Matrix, Vector and In-Network Compute*.

## The problem: communication runs on the compute units

With expert parallelism, each of the $$W$$ ranks (accelerators) holds a slice of the $$E$$ experts. Every MoE layer runs two collectives:

- the **dispatch** sends each token to the ranks that hold its top-$$k$$ experts;
- the **combine** returns the weighted outputs of the experts to the rank that owns the token, where they are summed.

These two collectives take on average 47% of the runtime of popular MoE models. Current systems hide them by overlapping them with computation, or fuse them into the kernels of the experts. In both cases, the transfers are still issued by the accelerator's own compute units. For example, [DeepSeek-V3](https://arxiv.org/abs/2412.19437) dedicates about 15% of the compute units of each accelerator to communication. Compute units given to communication are taken from the experts.

The method in this post adds ranks whose only job is to run the collectives: *In-Network Computing* (INC) elements, which play the role of a programmable network switch. In the prototype, two extra accelerators act as the switch, one for the dispatch and one for the combine. All transfers use one-sided shmem operations on a memory heap shared by all ranks: a `put_signal` writes data to a remote rank and raises a flag there, a `signal` raises a flag only, a `get` reads remote data, and `wait_until` waits for a flag.

## Dispatch and combine, with and without the switch

The router produces a routing table that maps each token to its experts and to the ranks that hold them. In a regular all-to-all, a token selected by experts on $$d$$ ranks is sent $$d$$ times by its owner. The combine is the reverse all-to-all, and the owner then sums the $$W$$ partial results of each token. This puts $$W^2$$ transfers on the network per collective.

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/inference/imgs/ep_routing_table.svg"/>
<br/>
<img width="50%" height="50%" src="{{ site.assets }}/inference/imgs/ep_all_to_all.svg"/>
<br/>
<img width="48%" height="48%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_inc.svg"/>

{: style="text-align:center; font-size: small;"}
**Top:** the routing table of tokens A to C (owned by ranks $$P_0$$ to $$P_2$$) to experts $$E_0$$ to $$E_5$$, with top-$$k=4$$. **Middle:** the regular all-to-all dispatch and combine, through a switch that only relays packets. **Bottom:** the same exchange through the INC switch, with $$W$$ transfers in and $$W$$ out. $$A_0$$ is the sum of the expert outputs of rank 0 for token A, $$\Sigma_A$$ the sum across ranks, and $$\emptyset$$ a header-only packet.

The INC switch replaces the two collectives with two new ones:

- **All-to-All-Unique** (dispatch, *push-multicast*): each rank sends each token *once*, with its routing metadata, into the switch. Once all ranks have arrived, the switch sends to each rank the tokens its experts need. The replication and permutation of tokens happen in the switch.
- **All-to-All-Reduce** (combine, *pull-reduce*): each rank writes its partial results locally, signals the switch, and goes back to computing. The switch pulls the partials of each rank as soon as its flag rises, adds them to an accumulator as they arrive, and sends each owner its final result.

This cuts the packets crossing the switch from $$2W^2$$ to $$2W$$ per collective, and the data sent or received by each rank by up to a factor of $$W$$. Each rank issues a single `put_signal` for the dispatch and a single `signal` for the combine. All other transfers are issued by the switch.

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="{{ site.assets }}/inference/imgs/ep_timeline.svg"/>

{: style="text-align:center; font-size: small;"}
The dispatch and combine of the token $$T_C$$ of rank $$P_2$$. On the dispatch, $$P_2$$ puts $$T_C$$ and its expert ids $$E_{0123}$$ into the switch, which sends $$T_C$$ with $$E_{01}$$ to $$P_0$$ and with $$E_{23}$$ to $$P_1$$. On the combine, the switch pulls the local sum $$\sigma_C$$ of the expert outputs of each rank with a `get` when its flag rises, accumulates it, and sends the sum $$\Sigma_C$$ to $$P_2$$. Subscripts name the destination of each call, or the sender of a `signal`. Dotted arrows are flags raised with `signal`, and the asterisk marks nonblocking calls.

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_reduce.svg"/><span style="display:inline-block; width:5.5%"></span><img width="40%" height="40%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_unique.svg"/>

{: style="text-align:center; font-size: small;"}
The two INC collectives on $$W=4$$ ranks, from top to bottom. **Left:** All-to-All-Reduce. **Right:** All-to-All-Unique. $$T_x$$ are the tokens of rank $$P_x$$, $$E_x$$ their expert ids and weights, and $$f_x$$ its ready flag on the switch. In the All-to-All-Reduce, shown with $$B=2$$ buffers, the switch adds the local sum $$\sigma_x$$ of the expert outputs of each rank to a buffer $$\Sigma_b$$ as it arrives, then sums the buffers into $$\Sigma$$.

The same two ideas apply to the AllGather-ReduceScatter backend, which sends every token to every rank instead of only to the ranks that need it. The four combinations are shown below.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_communication_models.svg"/>

{: style="text-align:center; font-size: small;"}
The four communication models for $$W=3$$ ranks holding one token each. **Left:** a regular switch that relays packets. **Right:** an INC switch. **Top:** AllGather-ReduceScatter. **Bottom:** all-to-all.

## Pull-based reduction in the switch

The dispatch *pushes* its data into the switch, because only the sender knows how many tokens it sends. The combine *pulls* instead. Pulling lets the switch add each partial result to its accumulator as soon as it arrives, so it never holds all $$W$$ partials at once. This matters because the switch has much less memory than an accelerator. Pulling also lets the switch serve the ranks in the order they finish, and read a rank only when it has a free buffer.

The switch can fold the partials into $$B \le W$$ buffers. With $$B=1$$ (*get-reduce*) it uses the least memory, and with $$B=W$$ (*put-reduce*) it runs the most transfers in parallel. For a payload of $$S$$ bytes, the switch needs at most $$S(B+1)+4W$$ bytes, independently of the model.

## Fusing the layer into one kernel

The MoE layer is split into eight stages, each running on one class of unit: *matrix units* for matrix multiplications, *vector units* for all other computation, and the INC switch for the two collectives. The stages are `route`, `dispatch`, `regroup` (sort tokens by expert), the two matrix multiplications of the experts with a `SwiGLU` in between, `reduce` (weight and sum the expert outputs locally) and `combine`.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_stages.svg"/>

{: style="text-align:center; font-size: small;"}
The eight stages of the fused MoE layer, colored by the class of unit that runs them. The blue frame holds the experts.

A regular implementation launches one kernel per stage, and every stage stores its output to global memory (GM) and the next one loads it back. It also needs two extra stages: a `pack` that copies each token once per destination before the dispatch, and a `global reduce` that sums the $$W$$ partials after the combine. Fusion alone keeps the intermediate results of the experts in local (on-chip) memory, but still needs both extra stages. With fusion and INC, both extra stages disappear: the layer goes from ten kernel launches and ten round trips to global memory, to one launch and five.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_data_movement.svg"/>

{: style="text-align:center; font-size: small;"}
Data movement for one microbatch. Each row is a stage (left box) and the tensor it stores in global memory (right box). $$\star$$ marks memory of the shared heap. **(a)** One kernel per stage. **(b)** One fused kernel, where the ranks issue the transfers. **(c)** One fused kernel, where the INC switch issues the transfers.

## Pipelining: three-way overlap

A single microbatch uses one class of unit at a time. The kernel therefore runs several microbatches at once, each one stage behind the previous. While the matrix units multiply microbatch $$m$$, the vector units regroup microbatch $$m+1$$, and the switch dispatches or combines other microbatches. Matrix units, vector units and the switch are busy at the same time, on different microbatches. Each stage waits only on the previous stage of the same microbatch, so there is no global barrier. Also, only the buffers of the microbatches in flight are alive at any time, so memory grows with the depth of the pipeline, not with the size of the batch.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_pipeline.svg"/>

{: style="text-align:center; font-size: small;"}
Pipelined execution of four microbatches (mb 0 to 3) during prefill, to scale. No two matrix (purple) and no two vector (green) boxes overlap. The dispatch and combine run on different INC elements, so they may overlap.

## Results

The kernel was evaluated on OLMoE-1B-7B (64 experts, top-$$k=8$$) under vLLM, on 4 accelerators in one node, plus 2 accelerators acting as the INC switch. Three implementations were compared on the prefill step, with 4096 to 65536 tokens per rank:

- **Baseline**: one kernel per stage, with the collectives of the communication library;
- **MegaKernel**: one fused kernel, where the ranks issue the transfers through a switch that only relays packets;
- **MegaKernel+INC**: one fused kernel, with All-to-All-Unique and All-to-All-Reduce in the switch.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_results.svg"/>

{: style="text-align:center; font-size: small;"}
**Left:** prefill time per iteration against batch size. **Right:** decode time per output token against the number of concurrent sequences, with placeholder values that are not measured yet. Both axes are in $$\log_2$$ scale.

Fusion alone is 4.9% to 18.6% faster than the baseline, and INC adds a further 0.8% to 7.8%. The gain of INC grows with the batch size beyond 8192 tokens. At the largest batch, the throughput goes from 20,353 to 25,042 tokens per second. With the AllGather-ReduceScatter backend, MegaKernel+INC is 8.9% to 18.5% faster than the baseline.

Profiling shows that the compute ranks spend no time in cross-rank transfers. The matrix units are busy 75% of the time, the vector units 57%, and each INC element 22%. The layer is therefore *compute-bound*: its runtime is set by the matrix multiplications of the experts, not by communication. The low load on the INC elements leaves room for more ranks.

## Final remarks

Moving the collectives from the compute units to the network has two benefits. All compute units stay available to the experts, and a third class of unit (the switch) can work in parallel with the matrix and vector units. The switch also removes redundant work: each token is sent once, and partial results are summed during the transfer. Open directions include the decode step, which is limited by memory and latency rather than computation, sending tokens in lower precision, and evaluation under continuous batching.
