---
layout: post
title:  "Kernel Fusion for MoE Expert Parallelism"
categories: ["Efficient Inference"]
tags: ["LLM", "kernel fusion", "MoE", "expert parallelism", "in-network computing", "collectives"]
---

A [Mixture-of-Experts (MoE)]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) layer distributed with expert parallelism spends much of its time moving tokens between accelerators, on the same compute units as the experts. The method described below fuses the whole layer into a single kernel, and moves its communication into the network.

## The problem: communication runs on the compute units

With expert parallelism, each of the $$W$$ ranks (accelerators) holds a slice of the $$E$$ experts. A token may select experts on any rank, so every MoE layer runs two collectives:

- the **dispatch** sends each token to the ranks that hold its top-$$k$$ experts;
- the **combine** returns the weighted outputs of the experts to the rank that owns the token, where they are summed.

These two collectives can take 47% of the execution time of popular MoE models ([COMET](https://arxiv.org/abs/2502.19811)). Current systems hide them by overlapping them with computation, or fuse them into the kernels of the experts. In both cases, however, the compute units of the accelerator still issue the transfers, and every unit given to communication is taken from the experts. [DeepSeek-V3](https://arxiv.org/abs/2412.19437), for example, dedicates 20 of the 132 streaming multiprocessors (SMs) of an H800 GPU, about 15%, to communication, and its report asks for hardware, such as a network co-processor, that would take this work off the SMs.

The fused kernel therefore adds ranks whose only job is to run the collectives: *In-Network Computing* (INC) elements, which play the role of a programmable network switch that can copy, regroup and add the data it forwards. In the prototype, two extra accelerators act as the switch, one for the dispatch and one for the combine. All transfers are one-sided [SHMEM](https://en.wikipedia.org/wiki/SHMEM) operations, which access the memory of a remote rank without any action from that rank. They work on a *symmetric heap*, a memory region allocated at the same offset on every rank. A `put_signal` writes data to a remote rank and then raises a flag there, a `signal` raises a flag only, and a `get` reads remote data. A rank waits for one of its flags with `wait_until`, or for the first of several flags with `wait_until_some`.

## Dispatch and combine, with and without the switch

The router produces a routing table that maps each token to its experts and to the ranks that hold them. In the example below, with $$W = 3$$ ranks and top-4 routing, token A of $$P_0$$ selects the experts $$E_0$$, $$E_1$$, $$E_4$$ and $$E_5$$, so it must reach $$P_0$$ and $$P_2$$. In a regular all-to-all, the owner sends a separate copy of a token to every rank that holds one of its experts, so a token selected by experts on $$d$$ ranks is sent $$d$$ times, and each rank that gets no tokens from it receives a header-only packet ($$\emptyset$$) instead, since otherwise it would wait forever. The combine is the reverse all-to-all, after which the owner sums the partial results of each of its tokens, such as $$A_0$$ and $$A_2$$ into $$\Sigma_A$$. Each collective thus puts $$W^2$$ transfers on the network, one per pair of ranks.

{: .picture}
<img width="40%" height="40%" src="{{ site.assets }}/inference/imgs/ep_routing_table.svg"/>
<br/>
<img width="50%" height="50%" src="{{ site.assets }}/inference/imgs/ep_all_to_all.svg"/>
<br/>
<img width="48%" height="48%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_inc.svg"/>

{: .caption}
**Top:** the routing table of tokens A to C (owned by ranks $$P_0$$ to $$P_2$$) to experts $$E_0$$ to $$E_5$$, with top-$$k=4$$. **Middle:** the regular all-to-all dispatch and combine, through a switch that only relays packets. **Bottom:** the same exchange through the INC switch, with $$W$$ transfers in and $$W$$ out. $$A_0$$ is the sum of the expert outputs of rank 0 for token A, $$\Sigma_A$$ the sum across ranks, and $$\emptyset$$ a header-only packet.

The INC switch replaces the two collectives with two new ones:

- **All-to-All-Unique** (dispatch, *push-multicast*): each rank sends each token *once*, with its routing metadata, into the switch. Once all ranks have arrived, the switch sends to each rank the tokens that its experts need, A and C to $$P_0$$ in the example. The replication and the permutation of the tokens happen in the switch.
- **All-to-All-Reduce** (combine, *pull-reduce*): each rank writes its partial results locally, signals the switch, and goes back to computing. The switch pulls the partial results of each rank as soon as its flag rises, adds them to an accumulator as they arrive, and sends each owner its final result.

This cuts the packets that cross the switch, in and out, from $$2W^2$$ to $$2W$$ per collective. The data that a rank sends on the dispatch, or receives on the combine, also drop from up to $$\min(k, W)$$ copies of each token to one, and the operations that it issues drop from $$W$$ to one per collective: a single `put_signal` for the dispatch and a single `signal` for the combine. The switch issues all the other transfers, as the timeline below shows for a single token. The price is latency, because the INC switch waits for all the ranks before it sends, while a regular switch forwards each packet at once.

{: .picture}
<img width="45%" height="45%" src="{{ site.assets }}/inference/imgs/ep_timeline.svg"/>

{: .caption}
The dispatch and combine of the token $$T_C$$ of rank $$P_2$$. On the dispatch, $$P_2$$ puts $$T_C$$ and its expert ids $$E_{0123}$$ into the switch, which sends $$T_C$$ with $$E_{01}$$ to $$P_0$$ and with $$E_{23}$$ to $$P_1$$. On the combine, the switch pulls the local sum $$\sigma_C$$ of the expert outputs of each rank with a `get` when its flag rises, accumulates it, and sends the sum $$\Sigma_C$$ to $$P_2$$. Subscripts name the destination of each call, or the sender of a `signal`. Dotted arrows are flags raised with `signal`, and the asterisk marks nonblocking calls.

The next figure shows the two collectives inside the switch.

{: .picture}
<img width="40%" height="40%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_reduce.svg"/><span style="display:inline-block; width:5.5%"></span><img width="40%" height="40%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_unique.svg"/>

{: .caption}
The two INC collectives on $$W=4$$ ranks, from top to bottom. **Left:** All-to-All-Reduce. **Right:** All-to-All-Unique. $$T_x$$ are the tokens of rank $$P_x$$, $$E_x$$ their expert ids and weights, and $$f_x$$ its ready flag on the switch. In the All-to-All-Reduce, shown with $$B=2$$ buffers, the switch adds the local sum $$\sigma_x$$ of the expert outputs of each rank to a buffer $$\Sigma_b$$ as it arrives, then sums the buffers into $$\Sigma$$.

The same two ideas apply to the AllGather-ReduceScatter backend, which sends every token to every rank instead of only to the ranks that need it. This backend does not depend on the routing, which makes it simple, but it moves up to $$W$$ times more data than necessary. In the figure below, which shows the four combinations, the INC switch (right) carries one packet per link instead of $$W$$ (left), and the all-to-all (bottom) moves less data than the all-gather (top).

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_communication_models.svg"/>

{: .caption}
The four communication models for $$W=3$$ ranks holding one token each. **Left:** a regular switch that relays packets. **Right:** an INC switch. **Top:** AllGather-ReduceScatter. **Bottom:** all-to-all.

## Pull-based reduction in the switch

A switch can obtain its data in two ways. In a *push*, the senders write their data into its memory with a `put_signal`; in a *pull*, they only raise a flag with a `signal`, and the switch reads the data itself with a `get`. A pull needs the switch to know how much to read, and on the dispatch only the senders know how many tokens they send, so the dispatch pushes. The combine pulls instead. Pulling lets the switch add each partial result to its accumulator as soon as it arrives, so it never holds all $$W$$ partial results at once, which matters because a switch has much less memory than an accelerator. Pulling also lets the switch serve the ranks in the order in which they finish, so that a late rank does not hold back the others, and read a rank only when it has a free buffer.

The switch can fold the partial results into $$B \le W$$ buffers, which it sums at the end, as the All-to-All-Reduce above shows for $$B = 2$$. With $$B = 1$$ (*get-reduce*), it adds the partial results one after the other and uses the least memory, and with $$B = W$$ (*put-reduce*), it runs the most transfers in parallel. For a payload of $$S$$ bytes, the switch needs at most $$S(B+1)+4W$$ bytes, for $$B$$ accumulators and one result buffer of $$S$$ bytes, and $$W$$ flags of 4 bytes, independently of the model.

## Fusing the layer into one kernel

To fuse it into one kernel, the MoE layer is split into eight stages, each running on one class of unit: *matrix units* (Tensor Cores on NVIDIA GPUs, cube cores on Ascend [NPUs](https://en.wikipedia.org/wiki/Neural_processing_unit)) for the matrix multiplications, *vector units* (CUDA cores, vector cores) for all other computation, and the INC switch for the two collectives. The stages are `route`, `dispatch`, `regroup` (sort the tokens by expert), the two matrix multiplications of the experts with a `SwiGLU` in between, `reduce` (weight and sum the expert outputs locally) and `combine`.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_stages.svg"/>

{: .caption}
The eight stages of the fused MoE layer, colored by the class of unit that runs them. The blue frame holds the experts.

A regular implementation launches one kernel per stage: every stage stores its output to global memory (GM), the off-chip memory of the accelerator, and the next stage loads it back. Each collective is also a separate, synchronous call, across which the kernels cannot be fused, as the [post on kernel fusion for tensor parallelism]({{ site.baseurl }}{% post_url 2026-08-18-kernel-fusion-tensor-parallelism %}) explains. Two extra stages are needed as well: a `pack`, which copies each token once per destination before the dispatch, and a `global reduce`, which sums the partial results of each token after the combine. Fusion alone, in panel (b) of the figure below, keeps the intermediate results of the experts in local (on-chip) memory, which cuts the round trips to global memory from ten to seven, but it still needs both extra stages. With fusion and INC, in panel (c), both extra stages disappear, since All-to-All-Unique sends each token once and All-to-All-Reduce returns finished sums. The layer thus goes from ten kernel launches and ten round trips to global memory to one launch and five.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_data_movement.svg"/>

{: .caption}
Data movement for one microbatch. Each row is a stage (left box) and the tensor it stores in global memory (right box). $$\star$$ marks memory of the shared heap. **(a)** One kernel per stage. **(b)** One fused kernel, where the ranks issue the transfers. **(c)** One fused kernel, where the INC switch issues the transfers.

## Pipelining: three-way overlap

A single microbatch uses one class of unit at a time, since each of its stages waits for the previous one. The kernel therefore runs several microbatches at once, each one stage behind the previous one. While the matrix units multiply microbatch $$m$$, the vector units regroup microbatch $$m+1$$, and the switch dispatches or combines other microbatches. This extends the overlap of DeepSeek-V3, which also hides the dispatch and combine of one microbatch behind the computation of another, but runs them on its own SMs. Each stage waits only on the previous stage of the same microbatch, so there is no global barrier. Only the buffers of the microbatches in flight hold live data, so the memory that the schedule needs grows with the depth of the pipeline, not with the size of the batch, although the prototype still allocates buffers for all the microbatches. In the [prefill]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}#prefill-and-decode-phases) schedule below, the matrix multiplications, the longest stages, follow each other almost without gaps.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_pipeline.svg"/>

{: .caption}
Pipelined execution of four microbatches (mb 0 to 3) during prefill, to scale. No two matrix (purple) and no two vector (green) boxes overlap. The dispatch and combine run on different INC elements, so they may overlap.

## Results

The kernel was evaluated on [OLMoE-1B-7B](https://arxiv.org/abs/2409.02060), with 7B parameters, 1B of them active per token, and 64 experts with top-$$k = 8$$, under [vLLM](https://arxiv.org/abs/2309.06180), on 4 Ascend 910C NPUs in one node, plus 2 NPUs acting as the INC switch. Three implementations were compared on the prefill step, with 4096 to 65536 tokens per rank:

- **Baseline**: one kernel per stage, with the collectives of the communication library of the vendor (HCCL);
- **MegaKernel**: one fused kernel, where the ranks issue the transfers through a switch that only relays packets;
- **MegaKernel+INC**: one fused kernel, with All-to-All-Unique and All-to-All-Reduce in the switch.

In MegaKernel, the two extra NPUs act as a regular switch, so both fused implementations cross the same number of hops. The step from Baseline to MegaKernel thus isolates the fusion, and the step to MegaKernel+INC the in-network computing.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_results.svg"/>

{: .caption}
**Left:** prefill time per iteration against batch size. **Right:** decode time per output token against the number of concurrent sequences, with placeholder values that are not measured yet. Both axes are in $$\log_2$$ scale.

Fusion alone reduces the runtime by 4.9% to 18.6% compared with the baseline, and INC reduces it by a further 0.8% to 7.8%. The gain of INC grows with the batch size: at the smallest batch, the pipeline is too short to hide the round trip through the switch, and INC brings almost no gain. On the logarithmic axes, the gap between Baseline and MegaKernel+INC widens beyond 8192 tokens, which means a growing relative gain. At the largest batch, the throughput per compute rank goes from 20,353 tokens per second for the baseline to 25,042 for MegaKernel+INC. With the AllGather-ReduceScatter backend, MegaKernel+INC reduces the runtime of the baseline by 8.9% to 18.5%.

Profiling shows that the compute ranks spend no time in cross-rank transfers. The matrix units are busy 75% of the time, the vector units 57%, and each INC element 22%. The layer is therefore *compute-bound*: the matrix multiplications of the experts set its runtime, not the communication. The low load on the INC elements leaves room for more ranks.

## Future work

The decode step remains open: with one new token per sequence, its batch is too small to hide the collectives behind computation, and its runtime is limited by memory bandwidth and latency rather than by computation. Other directions are an allocator that reserves buffers only for the microbatches in flight, a dispatch in lower precision, and an evaluation under [continuous batching]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}#static-and-continuous-batching).

This post summarizes the paper *MoeInc: an Expert Parallelism MegaKernel with Overlap of Matrix, Vector and In-Network Compute*.
