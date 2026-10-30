---
layout: post
title:  "Kernel Fusion for MoE Expert Parallelism"
categories: ["Efficient Inference"]
tags: ["LLM", "kernel fusion", "MoE", "expert parallelism", "in-network computing", "collectives", "all-to-all"]
description: "A fused kernel for MoE expert parallelism in LLM inference: dispatch and combine on an in-network computing switch with the INC all-to-all and its phased variant, and three-way overlap of compute and network units."
---

A [Mixture-of-Experts]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) (MoE) layer holds many experts, of which each token activates only a few, so that the number of parameters of a model grows without growing its compute cost per token. In 2020, [GShard](https://arxiv.org/abs/2006.16668) placed the experts of such a layer on different accelerators, with **expert parallelism**, and both are now standard in large language models. The cost of expert parallelism is communication: every MoE layer sends each token to the accelerators that hold its experts, and brings their outputs back. According to [COMET](https://arxiv.org/abs/2502.19811), these two transfers take on average 47% of the execution time of popular MoE models and frameworks.

The usual remedies are of two kinds:

- scheduling the layer in chunks, so that the communication of one chunk overlaps the computation of another, as in the two-microbatch overlap of [DeepSeek-V3](https://arxiv.org/abs/2412.19437);
- fusing the communication into the compute kernels, so that the arrival of a chunk starts its matrix multiplication, as in COMET.

However, both run the transfers on the accelerator's own compute units, as a transfer must be issued by *some* execution unit of the device. A kernel that communicates must therefore reserve some of its units for it: thread blocks in COMET, and a configurable number of streaming multiprocessors (SMs) in DeepSeek's [DeepEP](https://github.com/deepseek-ai/DeepEP) library. As an example, DeepSeek-V3 gives about 15% of the SMs of each GPU to communication, and calls for hardware that takes this work off them. In practice, every unit given to communication is taken from the experts.

This post describes a kernel that takes the transfers off the compute units. It first compares the two common implementations of the MoE collectives, AllGather-ReduceScatter and All-to-All, and their forms on an **in-network computing** (INC) switch, i.e. a network switch that computes. It then covers two INC methods for the MoE layer, the INC all-to-all and its phased variant for switches with little memory, and a kernel that fuses the whole layer into a single launch of eight stages, [pipelined](https://en.wikipedia.org/wiki/Pipeline_(computing)) over microbatches so that three classes of hardware units work at the same time. The post ends with the results on a node with four compute devices.

{::options toc_levels="2..3" /}

* TOC
{:toc}

## Expert parallelism: dispatch and combine

Consider $$W$$ compute ranks, i.e. devices, and an MoE layer of $$E$$ experts, where each token selects its top-$$k$$ experts. With expert parallelism, each rank holds $$E_{\text{loc}} = E/W$$ experts, and the layer processes $$N$$ tokens of width $$H$$, split into microbatches of $$m_b$$ tokens. Every MoE layer then runs two [collectives](https://en.wikipedia.org/wiki/Collective_operation):

- the **dispatch** sends each token to the ranks that hold its $$k$$ experts, and each rank runs its local experts on the tokens it receives;
- the **combine** returns the weighted outputs of the experts to the rank that owns each token, where the $$k$$ contributions are summed.

Each rank is therefore the **token rank** of its tokens and the **expert rank** of its experts. The following table shows a small example of $$W=3$$ ranks, each holding one token and one expert, with top-$$k=2$$. The router produces this **routing table**, which maps each token to its experts and to the ranks that hold them. Each row is a token rank, and each column the expert of a rank:

| Token rank | $$E_0$$ on $$P_0$$ | $$E_1$$ on $$P_1$$ | $$E_2$$ on $$P_2$$ |
|:-:|:-:|:-:|:-:|
| $$P_0$$ | A | | A |
| $$P_1$$ | | B | B |
| $$P_2$$ | C | C | |

The collectives in this post are built with the one-sided operations of [SHMEM](https://en.wikipedia.org/wiki/SHMEM), which reach the memory of a remote rank without involving it. The ranks address each other through a **symmetric heap**, a memory region allocated at the same offset on every rank. Seven operations are needed, each given with the OpenSHMEM call that implements it:

- `put_signal*` (`shmem_putmem_signal_nbi`) writes a contiguous payload into the symmetric heap of a remote rank, and then raises a flag there, in one nonblocking call (the asterisk marks the nonblocking calls);
- `put*` (`shmem_putmem_nbi`) writes a contiguous payload into the symmetric heap of a remote rank, without a flag, in one nonblocking call;
- `signal` (`shmemx_signal_op`, an extension of OpenSHMEM) raises a remote flag without data, and a zero-length `put_signal*` can replace it;
- `get` (`shmem_getmem`) copies a region of the symmetric heap of a remote rank into local memory, and blocks until the copy is done;
- `wait_until` (`shmem_signal_wait_until`) blocks until a local flag satisfies a comparison, e.g. until it is raised;
- `wait_until_some` (`shmem_wait_until_some`) blocks until any of a set of local flags satisfies a comparison, and reports which;
- `quiet` (`shmem_quiet`) blocks until every nonblocking put that the caller issued has completed.

The INC all-to-all phased, described later, also uses an INC [barrier](https://en.wikipedia.org/wiki/Barrier_(computer_science)), at which every rank waits until all ranks have joined.

Two implementations of the dispatch and combine, or *backends*, are common: AllGather-ReduceScatter and All-to-All. Each of them runs through a regular switch that relays packets, or through an INC switch, so there are four **communication models**, and the last of them also has a phased variant. In this post, the dispatch (one-to-many) always follows a **push model**, in which each sender writes its data to the receiver. Without INC, the combine (many-to-one) follows a push model too, and the owner of each token sums the partial results that it receives (*rank reduce*); with INC, it follows a **pull model**, in which the INC switch reads the partial results itself and sums them (*INC reduce*).

Each model below comes with a table of its one-sided operations per microbatch: those that a token rank $$P_x$$, the switch and an expert rank $$P_y$$ issue on the dispatch, from $$P_x$$ to $$P_y$$, and on the combine, back to $$P_x$$. The tables also give the **steps**, i.e. the transfers on the critical path from the data being ready to its delivery, the **relays** per node, i.e. the packets that cross the switch for each rank, and what the switch holds and waits for. For a token rank $$P_x$$, $$T_x \le T$$ is the number of its $$T$$ tokens that it sends in the current microbatch, and $$W_x \le W$$ the number of ranks that hold the experts these tokens are routed to.

## AllGather-ReduceScatter

**AllGather-ReduceScatter** (AGRS) broadcasts all tokens to all ranks, and reduce-scatters the results. On the dispatch, every rank pushes its whole microbatch to every rank through the switch, so each rank holds all tokens. Each rank runs its experts on the tokens routed to them and sums their outputs per token, ignoring the tokens not routed to its experts and writing zero outputs ($$\emptyset$$) for them. On the combine, every rank pushes its outputs for the tokens of each token rank to that rank, $$\emptyset$$ where it holds none, and the token rank computes the final sum. The following picture illustrates both collectives on the example:

{: .picture}
<img width="73.5%" height="73.5%" src="{{ site.assets }}/inference/imgs/ep_all_gather_reduce_scatter.svg"/>

{: .caption}
AllGather-ReduceScatter through a switch that relays each packet, with the dispatch (All-Gather) on the left and the combine (Reduce-Scatter) on the right. $$A_0$$ is token A sent to the expert rank $$P_0$$, $$E_0(\text{AC}_0)$$ the expert $$E_0$$ of $$P_0$$ processing tokens A and C, $$\sigma^y_t$$ the partial sum of the expert outputs of $$P_y$$ for token $$t$$, $$\Sigma_t = \sigma^0_t + \sigma^1_t + \sigma^2_t$$ their sum across all ranks, i.e. the final output of token $$t$$, and $$\emptyset$$ a token without output. Under each collective, a bracket names its model and counts its steps per node $$P_x$$.

The one-sided operations per microbatch are:

| | All-Gather (dispatch, push) | Reduce-Scatter (combine, push) |
|:--|:--|:--|
| Token rank $$P_x$$ | `put_signal*` $$\times W$$ | `wait_until` $$\times W$$ + rank reduce |
| Switch | (`wait_until_some` + `put_signal*`) $$\times W^2$$ | (`wait_until_some` + `put_signal*`) $$\times W^2$$ |
| Expert rank $$P_y$$ | `wait_until_some` $$\times W$$ | `put_signal*` $$\sigma$$ $$\times W$$ |
| Steps | 2 | 2 |
| Relays per node | $$W$$ in, $$W$$ out | $$W$$ in, $$W$$ out |
| Switch holds | one packet | one packet |
| Switch waits for | none | none |

AllGather-ReduceScatter does not depend on the routing, which makes it simple, but it moves up to $$W$$ times more data than needed: each rank pushes its tokens $$W$$ times on the dispatch, once per destination, and its outputs $$W$$ times on the combine.

## All-to-All

**All-to-All** (A2A) avoids the worst case that AllGather always pays, i.e. sending every token to every rank whatever the routing. Its clear benefit is a *selective* send and receive: each token goes only to the ranks that hold its experts, so a token whose experts lie on $$d \le \min(k,W)$$ distinct ranks leaves its owner $$d$$ times, as many times as with AllGather only when $$d=W$$. Its combine is the reverse all-to-all, followed by a sum on the owner of each token:

{: .picture}
<img width="73.5%" height="73.5%" src="{{ site.assets }}/inference/imgs/ep_all_to_all.svg"/>

{: .caption}
All-to-All through a switch that relays each packet, with the dispatch on the left and the combine on the right, in the notation of AllGather-ReduceScatter. $$\emptyset$$ on a dotted arrow is a signal alone, sent where a rank has no data for the destination.

The one-sided operations per microbatch are the same as those of AllGather-ReduceScatter, but on $$T_x$$ tokens or outputs per destination instead of $$T$$:

| | All-to-All (dispatch, push) | All-to-All (combine, push) |
|:--|:--|:--|
| Token rank $$P_x$$ | `put_signal*` $$\times W$$ | `wait_until` $$\times W$$ + rank reduce |
| Switch | (`wait_until_some` + `put_signal*`) $$\times W^2$$ | (`wait_until_some` + `put_signal*`) $$\times W^2$$ |
| Expert rank $$P_y$$ | `wait_until_some` $$\times W$$ | `put_signal*` $$\sigma$$ $$\times W$$ |
| Steps | 2 | 2 |
| Relays per node | $$W$$ in, $$W$$ out | $$W$$ in, $$W$$ out |
| Switch holds | one packet | one packet |
| Switch waits for | none | none |

The cost of the selective send is that the message sizes depend on the routing, so every receiver must know how much data it gets from each sender. Two-sided libraries such as [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) therefore need an additional all-to-all beforehand, in which the ranks trade their send and receive counts, as in the [distributed implementation of the MoE post]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}#distributed-implementation-in-pytorch). This setup step adds the latency of one more collective to every layer, although the setup of a step can sometimes be overlapped with the computation of the previous one. The one-sided implementation of this post sends the counts with the tokens instead: the first chunk that a token rank puts to each rank holds the number of chunks and bytes and the top-$$k$$ experts of its tokens, and a rank with no tokens for a destination sends it a `put_signal*` with no payload, i.e. a signal alone ($$\emptyset$$ in the picture above), so that no receiver waits forever. One bit of the signal tag says whether a payload came with it. Thus, each collective still puts a packet from every rank to every rank, i.e. $$W^2$$ packets into the switch and $$W^2$$ packets out of it, although only those with tokens carry a payload, and the owner of each token sums the partial results that it receives.

On the other hand, AllGather-ReduceScatter needs no setup step, as the number of tokens and outputs that each rank sends and receives is known beforehand. It is simpler to implement, and for small MoE models, or on a small number of ranks, where each token is needed on most ranks anyway, it may be faster. All-to-All suits larger groups of ranks, where the selective send saves more than the setup costs.

## In-network computing

In-network computing is not new. In 2016, [SHArP](https://network.nvidia.com/sites/default/files/related-docs/solutions/hpc/paperieee_copyright.pdf) offloaded the reductions of MPI collectives into the switches of an InfiniBand network, and its second version, [SHARPv2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7295336/), added the streaming aggregation of large messages, which the gradients of deep learning need. In 2019, [SwitchML](https://arxiv.org/abs/1903.06701) aggregated the gradients of data-parallel training inside programmable switches, training up to 5.5 times faster. Inside a node, NVLink SHARP brings the same capabilities to the switches between the GPUs, which accelerate the all-reduce, all-gather and reduce-scatter collectives of the GPUs' [collective communication library](https://arxiv.org/abs/2507.04786).

An INC switch accelerates the gather, scatter and reduce operations in two ways:

- **[multicast](https://en.wikipedia.org/wiki/Multicast)** of unique tokens: each rank sends a token to the switch only once, and the switch replicates it to every rank that needs it, instead of the sender sending one copy per destination;
- **in-switch reduction**: the switch sums the partial results that it receives, instead of forwarding all of them to the destination rank, which sums them there.

In practice, both lower the bandwidth that each rank needs and the number of packets that the switch relays. The overview [In-Network Collective Operations: Game Changer or Challenge for AI Workloads?](https://arxiv.org/abs/2601.19132) covers the benefits of INC for AI workloads and the obstacles to its adoption in more detail.

## INC AllGather-ReduceScatter

An INC switch improves AllGather-ReduceScatter with fewer relays and less bandwidth. On the dispatch, each rank pushes a single copy of its tokens to the INC switch, which broadcasts their concatenation to all ranks once every rank has arrived. On the combine, the INC switch pulls the outputs of each rank once the rank signals them ready, sums them as they arrive, and returns to each token rank the outputs of its tokens:

{: .picture}
<img width="88.0%" height="88.0%" src="{{ site.assets }}/inference/imgs/ep_all_gather_reduce_scatter_inc.svg"/>

{: .caption}
INC AllGather-ReduceScatter, with INC All-Gather on the left and INC Reduce-Scatter on the right. $$A_{012}$$ is token A for the expert ranks $$P_0$$, $$P_1$$ and $$P_2$$, $$\text{ABC}_0$$ the tokens A, B and C for $$P_0$$, $$\sigma^0_{A\emptyset C}$$ the partial sums of $$P_0$$ for the tokens A and C, with no output for B, and $$r^0_{A\emptyset C}$$ the signal that they are ready (dotted). The first INC switch of the combine maps each ready signal, in a dashed box, to the pull of the partial sums that it names, $$r^y_t \rightarrow \sigma^y_t$$; the expert ranks are drawn again where the INC switch pulls from them.

The one-sided operations per microbatch are:

| | INC All-Gather (dispatch, INC push) | INC Reduce-Scatter (combine, INC pull) |
|:--|:--|:--|
| Token rank $$P_x$$ | `put_signal*` $$\times 1$$ | `wait_until` $$\Sigma$$ $$\times 1$$ |
| INC switch | `wait_until_some` $$\times W$$ + `put_signal*` $$\times W$$ | (`wait_until_some` $$r$$ + `get` $$\sigma$$ + add to $$\Sigma$$) $$\times W$$ + `put_signal*` $$\Sigma$$ $$\times W$$ |
| Expert rank $$P_y$$ | `wait_until_some` $$\times 1$$ | `signal` $$r$$ $$\times 1$$ |
| Steps | 2 | 3 |
| Relays per node | 1 in, 1 out | 1 in, 1 out, plus 1 signal |
| Switch holds | the tokens of every rank, until the last arrives | the sums $$\Sigma$$ of every rank |
| Switch waits for | all $$W$$ ranks | all $$W$$ ranks |

Compared with the table of AllGather-ReduceScatter, each rank issues one operation per collective instead of $$W$$, and one packet per rank crosses the switch in each direction, instead of $$W$$. As INC switches such as NVLink SHARP support it already, and as it is a single-step communication, with no setup step to exchange counts as in All-to-All, AllGather-ReduceScatter is still used for the dispatch and combine of many production systems. As an example, the MoE layers of [Megatron-Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher_inference.html) can dispatch and combine for inference with an all-gather and a reduce-scatter on NVLink SHARP, and [vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) offers an AllGather-ReduceScatter backend next to its all-to-all ones.

DeepSeek considers both sides. DeepSeek-V3 runs its dispatch and combine as an all-to-all without INC, with point-to-point transfers that its SMs issue, and asks for hardware that takes over this work, i.e. a network co-processor like SHARP, with read, write, multicast and reduce operations. The [Insights into DeepSeek-V3](https://arxiv.org/abs/2505.09343) paper names both INC opportunities, a dispatch that "resembles a small-scale multicast" and a combine that could use in-network aggregation, but judges a flexible combine hard to implement in the network, because of its small reduction scope and its imbalanced load. The pull model of the INC combine below, which adds the partial results in whatever order they become ready, is a response to this concern. Similarly, a [deployment of DeepSeek-V4-Pro in SGLang](https://www.lmsys.org/blog/2026-08-19-deepseek-v4-pro-engine-optimization-h20) runs the MoE layers of the prefill with an all-gather and a reduce-scatter, whose communication is predictable, and those of the decode with the all-to-all of DeepEP.

## INC all-to-all

The MoE collectives differ from the dense and symmetric collectives of training in every respect that matters to a switch. The dispatch is a multicast that depends on the routing, and the combine is a sparse sum, weighted per token and expert, and interleaved with a permutation. In practice, the opportunity is not to aggregate gradients, but to replicate each dispatched token inside the network rather than at its sender, and to fold the sum of the combine into the transfer that carries it. This is what the two methods of this post do. The first, the **INC all-to-all**, replaces the dispatch and the combine with two new INC collectives, INC All-to-All-Dispatch and INC All-to-All-Combine; the second, the INC all-to-all phased, runs them over phases of ranks, for switches with little memory.

### INC All-to-All-Dispatch

The dispatch is replaced by **INC All-to-All-Dispatch**, an all-to-all extended with an in-network multicast: a one-to-many collective on an INC push model, in which each sender sends a single copy of each token, however many ranks need it, as in INC All-Gather. Each token rank writes its microbatch, in token order and with its routing metadata, into its own slot on the INC switch, with one `put_signal*`. The switch finds the senders with `wait_until_some`, in whatever order their flags rise, and regroups the tokens of each sender by expert rank as they arrive.

Once every sender has arrived, the switch sends each expert rank the tokens that its experts need, in one coalesced packet with one `put_signal*`, which the rank waits for with `wait_until_some`. Thus, the replication and the permutation of the tokens run entirely on the switch.

### INC All-to-All-Combine

The conventional combine is wasteful in two ways: the $$W$$ partial results of a token cross the network independently, and their sum on the owner cannot start until all of them have arrived.

The transfer and the sum are fused into **INC All-to-All-Combine**, an all-to-all extended with an in-network selective reduction: a many-to-one collective on an INC pull model, as in INC Reduce-Scatter. Each expert rank writes its partial results into its own symmetric heap, raises its ready flag on the INC switch with one `signal`, and goes back to the experts. The switch serves the flags in the order they rise, found with `wait_until_some`: it pulls the partial results of each rank with one `get`, and adds them to the sum *as they land*. Once all ranks are in, it sends each token rank its rows of the sum with one `put_signal*`, which the rank waits for with `wait_until`.

The combine needs no metadata of its own, as it reuses the counts per sender of the dispatch, transposed, and the compute ranks run no cross-rank arithmetic. Note that each expert rank applies the routing weights to its expert outputs before it publishes its partial results, so the switch only adds them. The following picture shows both collectives on the example above:

{: .picture}
<img width="88%" height="88%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_inc.svg"/>

{: .caption}
The dispatch and combine of the example through the INC switch, with $$W$$ transfers in and $$W$$ out per collective. **Left:** INC All-to-All-Dispatch. **Right:** INC All-to-All-Combine. $$A_{02}$$ is token A for the expert ranks $$P_0$$ and $$P_2$$, $$\text{AC}_0$$ the tokens A and C for $$P_0$$, $$\sigma^y_{AC}$$ the partial sums of $$P_y$$ for tokens A and C, and $$r^y_{AC}$$ the signal that they are ready (dotted). The INC switch maps each ready signal to the pull of the partial sums that it names, $$r^y_t \rightarrow \sigma^y_t$$, and adds them into $$\Sigma_t$$.

The one-sided operations per microbatch are:

| | INC All-to-All-Dispatch (INC push) | INC All-to-All-Combine (INC pull) |
|:--|:--|:--|
| Token rank $$P_x$$ | `put_signal*` $$\times 1$$ | `wait_until` $$\Sigma$$ $$\times 1$$ |
| INC switch | (`wait_until_some` + group by expert rank) $$\times W$$ + `put_signal*` $$\times W$$ | (`wait_until_some` $$r$$ + `get` $$\sigma$$ + add to $$\Sigma$$) $$\times W$$ + `put_signal*` $$\Sigma$$ $$\times W$$ |
| Expert rank $$P_y$$ | `wait_until_some` $$\times 1$$ | `signal` $$r$$ $$\times 1$$ |
| Steps | 2 | 3 |
| Relays per node | 1 in, 1 out | 1 in, 1 out, plus 1 signal |
| Switch holds | the tokens of every token rank, until the last arrives | the sums $$\Sigma$$ of every token rank |
| Switch waits for | all $$W$$ token ranks | all $$W$$ expert ranks |

Every flag is a 32-bit integer, into which each sender writes the *magic* of its step, a value that strictly increases from step to step, so that a flag left from an earlier step never reads as an arrival. Slots are indexed by the microbatch modulo the pipeline depth, and a slot is written again only after the combine of the microbatch that used it has completed. As this combine follows every read of the slot, no further flow control is needed: in INC All-to-All-Combine, each rank receives its sums only after the switch has pulled the partial results of every rank. The INC switch also starts the next sum only after a `quiet` has completed the puts that read the previous one.

The timeline below follows the three tokens of the example through both collectives, with the operations of each device:

{: .picture}
<img width="63.1%" height="63.1%" src="{{ site.assets }}/inference/imgs/ep_timeline.svg"/>

{: .caption}
The dispatch and combine of the example through INC All-to-All-Dispatch and INC All-to-All-Combine, with the token ranks on the left and the expert ranks on the right. Once every token rank has put its tokens, the INC switch multicasts them grouped per expert rank, $$\text{AC}_0$$ to $$P_0$$, $$\text{BC}_1$$ to $$P_1$$ and $$\text{AB}_2$$ to $$P_2$$, and once every expert rank has signaled its partial sums ready, $$r^0_{AC}$$, $$r^1_{BC}$$ and $$r^2_{AB}$$, it pulls them and returns $$\Sigma_A$$, $$\Sigma_B$$ and $$\Sigma_C$$. The asterisk marks nonblocking calls, dotted arrows flags raised alone with `signal`, and a dot a flag awaited with `wait_until`.

No switch implements these two collectives today. Therefore, the implementation of this post emulates the INC switch with two extra devices of the same type as the compute ranks, one for the dispatch and one for the combine, which run the one-sided operations above on the symmetric heap. They hold no model parameters and run no experts.

### Bandwidth and packet count

With a conventional all-to-all dispatch, a token whose experts lie on $$d$$ distinct ranks is sent $$d$$ times by its owner, so each rank sends up to $$\min(k,W) \cdot m_b$$ rows per microbatch, and the network carries $$\mathcal{O}(W^2)$$ point-to-point flows. The conventional combine mirrors this, with each owner receiving up to $$\min(k,W)$$ partial results per token, and each rank issues $$W$$ operations per collective.

INC All-to-All-Dispatch and INC All-to-All-Combine remove the duplicates at either end. On the dispatch, each rank sends each token *exactly once*, in token order and with no packing per destination; on the combine, the switch sums the partial results in flight, and each owner receives one result per token. Therefore:

- the rows that a rank sends on the dispatch, and receives on the combine, fall from up to $$\min(k,W) \cdot m_b$$ to $$m_b$$, i.e. by up to a factor of $$W$$ when $$k \ge W$$, as in the setup below ($$k=8$$ and $$W=4$$);
- the packets that cross the switch fall from $$2W^2$$ to $$2W$$ per collective, over links whose endpoints are known in advance, so that the transfers can be scheduled ahead;
- the operations that each rank issues fall from $$W$$ to one per collective.

### Why the combine pulls

An INC switch can get its data in the two ways above. With a push model, the senders write their data into the memory of the switch with a `put_signal*`; with a pull model, they only `signal`, and the switch reads the data itself with a `get`. A pusher knows its own byte count, so a push needs no agreement on sizes beforehand, whereas a puller must know how much to read. Therefore, the dispatch, whose sizes only the sender knows, pushes.

The combine pulls, because pulling lets the switch add each partial result to the sum as soon as it lands, instead of holding all $$W$$ partial results before summing them. This keeps the reduction within the limited memory of the switch. Pulling also lets the switch choose the order, so that a late rank does not stall the sum of the ranks already present. Moreover, it gives the switch [flow control](https://en.wikipedia.org/wiki/Flow_control_(data)): it reads a rank only when it can fold it in, so a burst of partial results cannot overflow its memory.

The cost is that the switch starts transfers itself, rather than being a purely reactive switch, which departs from SHArP and from other in-switch designs. This is the price of fusing a reduction that is permuted, rather than symmetric.

## INC all-to-all phased

The weak point of the INC all-to-all is the memory of the switch. INC All-to-All-Dispatch and INC All-to-All-Combine, as INC All-Gather and INC Reduce-Scatter, are collectives: they are asynchronous on the compute ranks, but synchronous on the switch, which relays nothing until all $$W$$ ranks have sent their data into it, and holds all of it meanwhile. However, switches have limited memory for multicast and reduction.

For such switches, the second method of this post, the **INC all-to-all phased**, runs **INC All-to-All-Dispatch (Phased)** and **INC All-to-All-Combine (Phased)**: the variant of the INC all-to-all that executes both collectives iteratively, as **phased collectives** over $$G$$ ranks each, so that the switch multicasts to and reduces over one phase at a time, and holds only the data of that phase. $$G$$ ranges from 1 to $$W$$, where $$G=W$$ is the INC all-to-all, and the results below are for $$G=1$$. The dispatch multicasts the tokens of each token rank as they arrive. The combine waits for the request of every token rank, which carries the expert metadata of its tokens: only then does the switch know from how many expert ranks to pull with `get` for each sum $$\Sigma$$, and to which token rank to return it. The switch then serves the requests one phase at a time. Because of industry confidentiality, the specific implementation cannot be detailed; its details will be released in the near future. A phased INC All-Gather and Reduce-Scatter, which follows the same logic, was also implemented.

The following picture shows the INC all-to-all phased on the example, in a simplified view:

{: .picture}
<img width="92.6%" height="92.6%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_inc_phased.svg"/>

{: .caption}
INC All-to-All-Dispatch (Phased) on the left and INC All-to-All-Combine (Phased) on the right, with $$G=1$$, following token C of $$P_2$$ (black; the other phases are faded). The INC switch multicasts C to $$P_0$$ and $$P_1$$, which join an INC barrier (stopwatch), and once every token rank has sent its request, $$\text{rank}^{01}_C$$ for $$P_2$$, which names the expert ranks of token C, it pulls their partial sums for C and returns $$\Sigma_C$$ to $$P_2$$. Because of industry confidentiality, the drawing omits the details of the implementation. Brackets count the steps per node $$P_x$$, where $$W_x \le W$$ is the number of ranks that hold the experts of its tokens.

The one-sided operations per microbatch, for one token rank per phase ($$G=1$$), are:

| | INC All-to-All-Dispatch (Phased, INC push) | INC All-to-All-Combine (Phased, INC pull) |
|:--|:--|:--|
| Token rank $$P_x$$ | `put_signal*` $$\times 1$$ + barrier | `put_signal*` $$\text{rank}$$ $$\times 1$$ + `wait_until` $$\Sigma$$ $$\times 1$$ |
| INC switch | (`wait_until_some` + `put*` $$\times W_x$$ + `quiet`) $$\times W$$ + barrier | `wait_until` $$\text{rank}$$ $$\times W$$ + ((`get` $$\sigma$$ + add to $$\Sigma$$) $$\times W_x$$ + `put_signal*` $$\Sigma$$ + `quiet`) $$\times W$$ |
| Expert rank $$P_y$$ | barrier | none |
| Steps | 2 | 3 |
| Relays per node | 1 in, $$W_x$$ out | $$W_x$$ in, 1 out, plus 1 request |
| Switch holds | the tokens of one token rank, until they are multicast | the sum $$\Sigma$$ of one token rank |
| Switch waits for | none, until the barrier | the requests of all $$W$$ token ranks |

The timeline below follows the three tokens of the example through the INC all-to-all phased, in the same simplified view:

{: .picture}
<img width="59%" height="59%" src="{{ site.assets }}/inference/imgs/ep_timeline_phased.svg"/>

{: .caption}
The dispatch and combine of the example through INC All-to-All-Dispatch (Phased) and INC All-to-All-Combine (Phased), with $$G=1$$, in the notation of the timeline of the INC all-to-all. The INC switch multicasts each token as it lands, every rank joins an optional INC barrier (dashed line), and once every token rank has sent its request, as $$\text{rank}^{02}_A$$ for $$P_0$$, the switch serves one phase at a time: it pulls, accumulates and returns each sum, then runs a `quiet`.

The phased collectives need up to $$W$$ times less switch memory, their main benefit, which suits them to larger batches and to switches with little memory. Their dispatch also helps with expert load imbalance, as it multicasts the tokens as they arrive. The price is latency: they take more steps and relays than the INC all-to-all. The optional INC barrier after the dispatch guarantees that all tokens have been delivered, so that each expert still runs one large matrix multiplication over all its tokens, rather than several small ones that leave the matrix units under-utilized as the tokens of each phase arrive.

The routing density and the memory of the switch decide which method to use. Under uniform routing, each of the $$k\,m_b$$ expert choices of a microbatch misses a given rank with probability $$1-1/W$$, so the number of distinct ranks that the microbatch selects is approximately given by:

$$
W_x \approx W \left( 1 - \left( 1 - \frac{1}{W} \right)^{k\, m_b} \right)
$$

where $$W_x$$ tends to $$W$$ once $$k\,m_b \gg W$$: at $$k=8$$ and $$W=4$$, even a single token selects 3.6 ranks on average. Each phase then relays to all $$W$$ ranks, as the collective does. Thus, the INC all-to-all is the default, both in prefill, where the wait for every rank hides behind the matrix units, and in decode, where the latency of the network is the bottleneck. The INC all-to-all phased is the choice for a switch with little memory, at a cost that the results below measure.

## The hardware

The kernel is described in hardware-agnostic terms, which apply to the accelerators of different vendors. **Compute blocks** (e.g. the SMs of a GPU) schedule parallel threads and hold two classes of arithmetic units, which a single kernel launch programs and which run at the same time:

- **matrix units** (MUs) for two-dimensional matrix operations, such as matrix multiplications;
- **vector units** (VUs) for elementwise work, gathers, scatters and data movement.

Memory is divided into the off-chip **global memory** (GM), i.e. the [high-bandwidth memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) (HBM) of large capacity, and the on-chip **local memory** (LM), a [scratchpad](https://en.wikipedia.org/wiki/Scratchpad_memory) managed by software (e.g. the shared memory of a GPU). An interconnect fabric links the devices into one logical node, and a collective communication library (CCL) provides the collectives over it, next to the one-sided operations above.

Note that a cross-rank transfer is issued by a vector unit and staged through its local memory, so a rank that communicates always spends vector unit cycles on it. This hardware fact motivates the whole design.

## Fusing the layer into one kernel

Fusion is worth its effort because of what a non-fused layer costs. A conventional layer is a sequence of kernels, each loading its inputs from global memory, computing, and writing its outputs back. Fusion collapses this sequence into one launch and keeps the intermediate results on-chip, as [FlashAttention](https://arxiv.org/abs/2205.14135) did for attention, and **megakernels** extend it to a whole model, as in [a low-latency megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).

In a distributed layer the case is stronger, as each synchronous collective breaks the layer into separate kernels, with buffers sized for the whole batch in global memory between them. The MoE dispatch and combine are the hardest collectives to fuse: their transfers are sparse, their sizes depend on the routing, and the combine carries a weighted sum.

The MoE layer is split into eight stages, chosen so that each stage runs entirely on one class of unit, and consecutive stages alternate between classes wherever a dependency allows. The stages are:

1. `route` computes the routing metadata of a microbatch, and ends with one `put_signal*` of its tokens and metadata into the INC switch;
2. `dispatch` is an INC All-to-All-Dispatch: once the tokens of every compute rank have arrived, the INC switch sends each rank the tokens that its experts need;
3. `group` gathers the received rows into the expert-contiguous order that the grouped matrix multiplication needs;
4. `MatMul` is the gate and up projection of the experts, on the matrix units;
5. `SwiGLU` is the activation function ([SwiGLU](https://arxiv.org/abs/2002.05202)), on the vector units;
6. `MatMul` is the down projection of the experts, on the matrix units;
7. `reduce` scales each expert output by its routing weight, accumulates it into the partial result of the rank, and signals the INC switch;
8. `combine` sums the partial results of each token and returns the sum to its owner, as an INC All-to-All-Combine on the INC switch.

Only two of the eight stages are collectives, and both run on the INC switch. The following picture shows the stages in dependency order:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_stages.svg"/>

{: .caption}
The eight stages of the fused MoE layer, in dependency order, each on the class of unit that runs it: matrix multiplications on the matrix units (purple), all other computation on the vector units (green), and both collectives on the in-network element (orange). The blue frame holds the experts. The time axis sketches the decode step, whose batch is too small to be split into microbatches, so the matrix units idle while the network works.

Note that the INC switch puts the final result into the symmetric heap rather than into the output tensor of the layer, so one local copy to global memory remains. This copy is left out of the stages: it crosses no link of the network, its measured cost is close to zero, and it fuses into the next `route`, which reads those rows anyway.

Three implementations of the layer are compared in this post, from no fusion to fusion with INC:

- **Baseline**: each stage is its own kernel, so every boundary is a store to global memory followed by a load, and ten tensors are stored, among them the two intermediate results of the experts, the largest of the layer. Two extra stages are needed: a `pack` before the dispatch, which lays out a copy of each token contiguously per destination, and a `global reduce` after the combine, which sums the $$W$$ partial results of each token on its owner. The collectives are calls to the CCL, issued by the vector units of the rank.
- **MegaKernel**: the ten stages of the Baseline in one launch. The intermediate results of the experts stay in local memory, which cuts the round trips to global memory from ten to seven. However, the plain all-to-all still needs the `pack` and the `global reduce`, and its transfers become one-sided operations relayed by a switch, still issued by the vector units of the rank: one `put_signal*` per destination, i.e. $$W$$ per collective.
- **MegaKernel+INC**: the eight stages in one launch, with both collectives run by the INC switch, as the INC all-to-all or as its phased variant. INC removes the two extra stages: the `pack` disappears, as each rank sends one copy of each token, in token order, and the `global reduce` disappears, as the sum is folded into the transfer. Only four regions of global memory remain visible to the network, all in the symmetric heap. The layer needs one launch and five round trips to global memory, with one `put_signal*` and one `signal` per microbatch issued by each compute rank.

The following picture follows one microbatch through the memories that an MoE layer touches (the local memory of a rank, its global memory, and the far side of the network) in the Baseline and in MegaKernel+INC, and shows how INC simplifies the layer:

{: .picture}
<img width="84.7%" height="84.7%" src="{{ site.assets }}/inference/imgs/ep_data_movement.svg"/>

{: .caption}
Data movement for one microbatch. **Left:** Baseline, one kernel per stage. **Right:** MegaKernel+INC, one fused kernel with the INC transfers, without the `pack` and the `global reduce` stages. Each row is a stage (colored box) and the tensor it stores in global memory (white box), and $$\star$$ marks the symmetric heap. The blue frame holds the experts, dashed lines mark kernel boundaries, and the bracket one kernel launch. Each INC stage runs as the INC all-to-all or as its phased variant.

## Pipelining and three-way overlap

A single microbatch uses one class of unit at a time, as its chain of stages is strictly sequential. However, the chain crosses from one class of unit to another seven times, and every crossing frees the unit it leaves for another microbatch. Therefore, the kernel runs $$M$$ microbatches through the chain at the same time, each one stage behind the previous one, so that at any instant the matrix units, the vector units and both INC devices work on different microbatches. The following picture shows the resulting schedule during the prefill step:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_pipeline.svg"/>

{: .caption}
Pipelined execution of four microbatches (mb 0 to 3) during prefill, with time to scale from the cost of each stage. Each microbatch has three rows, one per class of unit, colored as in the stages above. A kernel uses all matrix and all vector units, so no two purple and no two green boxes overlap; the dispatch and the combine run on two independent INC devices, so a dispatch may overlap a combine.

In the schedule above, each microbatch is routed and put into the INC dispatch device, which fans it out once the copies of all ranks have arrived. The microbatch then passes through `group`, the two matrix multiplications, `SwiGLU` and `reduce`, before the combine device sums the partial results and returns them. While the matrix units multiply microbatch $$m$$, the vector units run the `group` stage of microbatch $$m+1$$. Note that stage $$s$$ of microbatch $$m$$ waits only on stage $$s-1$$ of the same microbatch, so there is no global barrier.

The schedule extends the two-microbatch overlap of DeepSeek-V3, which hides the dispatch and combine of one microbatch behind the computation of another. Here, a vertical cut through the schedule shows a dispatch, a computation and a combine in flight at once, each for a different microbatch, with both collectives on dedicated INC devices. Pipelining also bounds the memory: the buffers of a microbatch are alive only between its first and its last stage. Therefore, the number of buffer sets alive at once is the **pipeline depth** $$D$$, not the number of microbatches $$M$$, nor the whole batch that a non-fused layer holds at once.

## Results

The kernel was evaluated on [OLMoE-1B-7B](https://arxiv.org/abs/2409.02060), an open model with 7 billion parameters in total and 1 billion active per token, $$E=64$$ experts, top-$$k=8$$, $$H=2048$$, an intermediate width of 1024, and 16 MoE layers in [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), served with [vLLM](https://arxiv.org/abs/2309.06180). One node holds four devices as compute ranks, i.e. $$E_{\text{loc}}=16$$ experts per rank at $$W=4$$, plus the two devices that emulate the INC switch, one for the dispatch and one for the combine. Under MegaKernel, the same two devices act as a regular switch that forwards each packet as it lands, and computes nothing.

Four implementations are compared: the three above, with the INC all-to-all in MegaKernel+INC, and **MegaKernel+INC (Phased, $$G=1$$)**, with the INC all-to-all phased. Baseline and MegaKernel use a plain all-to-all, followed by a global reduction on the owner of each token. Thus, comparing the Baseline to MegaKernel isolates the effect of fusion, MegaKernel to MegaKernel+INC the effect of in-network computing, and MegaKernel+INC to its phased variant the price of a small switch memory.

Prefill runs a single forward pass over the whole batch, from 1,024 to 65,536 tokens per compute rank, which covers chunked prefill, with chunks of typically 1,024 to 4,096 tokens, and the large batches of training. Decode runs 1 to 64 concurrent sequences of 1,024 tokens. Both the MoE layers and the whole model are timed, and the time reported is the mean of five iterations after warm-up. In the Baseline, the MoE layers take 65% to 81% of the prefill time of the model, and 47% to 57% of its decode time. The runtimes of the four implementations are displayed below:

{: .picture}
<img width="90.9%" height="90.9%" src="{{ site.assets }}/inference/imgs/ep_results.svg"/>

{: .caption}
Runtime of the four implementations, for the MoE layers (top) and the whole model (bottom), in prefill (left) against the batch size, and in decode (right) against the number of concurrent sequences of 1,024 tokens, on $$\log_2$$ axes. The light blue band spans MegaKernel+INC with the INC all-to-all and with its phased variant ($$G=1$$). Baseline runs one microbatch per batch; the other implementations use, at each point, the microbatch size with the shortest time.

Note that on a $$\log_2$$ axis a constant speedup is a constant vertical offset. The following table gives the speedup of each implementation over the Baseline, as the range over the points of the picture above, i.e. 1k to 64k tokens in prefill and 1 to 64 sequences in decode:

| Speedup over the Baseline | MoE layers, prefill | MoE layers, decode | Whole model, prefill | Whole model, decode |
|:--|:-:|:-:|:-:|:-:|
| MegaKernel | 1.20 to 1.33 | 1.40 to 1.52 | 1.07 to 1.20 | 1.05 to 1.09 |
| MegaKernel+INC | **1.54 to 2.04** | **2.53 to 3.49** | **1.27 to 1.58** | **1.10 to 1.22** |
| MegaKernel+INC (Phased, $$G=1$$) | 1.48 to 1.85 | 2.27 to 3.13 | 1.17 to 1.46 | 1.11 to 1.17 |

### Fusion and in-network computing

The gap between the Baseline and MegaKernel is the return on software engineering alone, on the same network, collectives and relay switch: MegaKernel runs the MoE layers 1.20 to 1.33 times faster than the Baseline in prefill, and 1.40 to 1.52 times faster in decode, thanks to one launch per layer and to intermediate results of the experts that never leave the local memory. Without in-network computing, it is held back by its collectives: a plain all-to-all puts $$W^2$$ flows per collective through the relay switch, a quadratic number of relays that the network multiplies by splitting every large payload into packets.

The gap between MegaKernel and MegaKernel+INC is the return on network engineering, with the same kernel and the same number of data steps: only the protocol and the role of the switch change. MegaKernel+INC removes those relays, as it sends each token once and folds the sum into the transfer. It runs the MoE layers a further 1.28 to 1.54 times faster in prefill and 1.81 to 2.31 times faster in decode, i.e. 1.54 to 2.04 times and 2.53 to 3.49 times faster than the Baseline. The network speedup is larger than the software speedup at every point on the MoE layers, and on the whole model in prefill. The gain is largest at 1k to 4k tokens, the chunk sizes of chunked prefill, where the MoE layers run 1.75 to 2.04 times faster and the whole model up to 1.58 times faster. At 64k tokens, the prefill throughput of the model rises from 24,843 to 35,406 tokens per second per compute rank (the two INC devices are not counted in the denominator).

### Prefill

Profiling confirms the design: computation and communication overlap fully, so that no communication step adds to the runtime, apart from the dispatch of the first microbatch and the combine of the last, which fill and drain the pipeline. The whole share of a rank in the two collectives is one `put_signal*` and one `signal` per microbatch, each of them overlapped with its next stage. All matrix and vector units therefore stay available to the experts for the whole layer, and no units are reserved for communication, unlike the thread blocks of COMET and the SM budget of DeepEP.

The prefill gains come from the large matrix multiplications. At prefill batch sizes, the two grouped `MatMul` stages dominate the runtime of a microbatch, and the schedule fits all the remaining stages under them: the vector stages of microbatch $$m+1$$ and the INC transfers of $$m+2$$ run while the matrix units are busy with $$m$$. The matrix units are busy 75% of the time, the vector units 57%, and each INC device 22%. Thus, the layer is *compute-bound* (see the [roofline model]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %})): its runtime is set by the matrix multiplications, and the remaining levers are arithmetic intensity and expert load balance, neither of them a communication problem. Also, the low load of the INC devices leaves room for more ranks.

This makes large prefills and training the hardest regime to accelerate: with the matrix multiplications dominating, the speedup comes almost entirely from the overlap, and communication hides behind computation only if it is cheap, with few operations per rank and few packets on the network. From 16k tokens on, MegaKernel+INC holds a steady speedup of 1.54 to 1.58 times on the MoE layers: efficient communication is what keeps the collectives under the matrix multiplications as the batch grows.

In the schedule of the pipeline, dispatches never overlap each other and combines never overlap each other, but a dispatch overlaps a combine freely. An INC operation waits on the flags of the ranks with `wait_until_some`, moves the data (regrouping it on the dispatch, pulling it with blocking `get`s on the combine), and sends only once every rank has arrived. Therefore, the requests are served in order of arrival, and the work of one INC device is serialized. The dispatch and the combine overlap only because they run on *different* devices. A single device that serves both would remove this concurrency and lengthen the pipeline, although at $$W=4$$ both devices are busy less than a quarter of the time.

### Decode

The decode step is the opposite extreme. Each step adds one token per sequence, so its payload is a handful of rows: the grouped `MatMul` stages become skinny matrix multiplications that no longer fill the matrix units, while the weights of the experts must still be read from global memory in full at every step. With too few tokens to be split into microbatches, the collectives sit on the critical path, as the time axis of the stages picture shows. The decode time of the MoE layers hardly depends on the number of sequences: from 1 to 64 sequences, it is 0.91 to 1.14 s in the Baseline, and 0.30 to 0.39 s in MegaKernel+INC. Thus, a step is bound by latency, not by work, and both fusion and in-network computing pay most here.

Fusion pays here for a different reason than in prefill. A non-fused layer pays, at every step, a kernel launch and a round trip to global memory per stage, and runs its collectives as separate library calls whose latency is exposed. The fused kernel removes all three:

- **memory efficiency**: the intermediate results of the experts never leave the local memory, and the working set follows the tokens in flight rather than the batch;
- **kernel launches**: the eight stages run in one launch instead of ten;
- **fused communication**: the collectives are one-sided operations issued from inside the kernel, instead of library calls.

Thus, fusion runs the MoE layers 1.40 to 1.52 times faster than the Baseline in decode. The INC all-to-all then replaces the $$W^2$$ relayed flows and the reduction on the owner with one put and one signal per rank, and adds a further 1.81 to 2.31 times, i.e. 2.53 to 3.49 times faster than the Baseline. The whole model gains 1.10 to 1.22 times: the MoE layers take 47% to 57% of the decode time of the Baseline, and the time spent outside them is larger in the fused runs, 1.19 to 1.37 s with MegaKernel+INC against 0.83 to 1.04 s in the Baseline.

### The INC all-to-all phased

At $$G=1$$, the switch holds the data of one token rank at a time, $$W$$ times less than under the INC all-to-all. In prefill, the phased variant runs close to the INC all-to-all: from 8k tokens, it is 3.5% to 6.0% slower on the MoE layers and 2.7% to 5.2% slower on the whole model, and below 8k tokens, 7.5% to 17% slower on the MoE layers. The reason lies in the network: one-sided puts are optimized for latency, and the network splits a large payload into many small packets, so the extra signals and relays of the phased collectives add few packets to the many that each phase moves anyway. Decode reverses the proportion: its payloads are a few packets each, the signals and relays become a large part of every step, and the phased variant runs the MoE layers 4.4% to 22% slower. Thus, with the INC all-to-all phased, a switch with little memory loses little in prefill and training, while decode, which needs little switch memory, is best served by the INC all-to-all.

### The emulated switch

The setup understates the benefit of in-network computing in two ways, and favors the Baseline in a third:

1. The INC switch is emulated by devices, so an INC collective takes two steps in sequence, to the switch and from it, separated by its wait for every rank. The relay switch of MegaKernel forwards each packet as it lands instead, so its point-to-point relays proceed at the same time, in effectively one step. A switch that multicasts and reduces in its forwarding path removes the second step.
2. At $$W=4$$, the 16 relayed flows per collective of the plain all-to-all are not yet a bottleneck next to the linear relays of the INC methods, a gap that is expected to widen with $$W$$.
3. The collectives of the Baseline are calls to the CCL, highly optimized for this network, while the one-sided collectives and the emulated switch are a prototype.

Therefore, the kernel is expected to improve further on switch hardware. A fair comparison would implement both protocols in the same switch hardware, which is left for future work.

## Communication volume and memory

At $$W=4$$, the peak number of rows that a rank sends per microbatch falls from about $$4m_b$$ to exactly $$m_b$$. Moreover, the symmetric heap becomes a function of the shape of the layer rather than of the routing, i.e. $$W+2$$ buffer units per microbatch, which removes the over-allocation for the worst case that data-dependent routing forces.

The working set follows the pipeline, not the batch. A non-fused layer holds at once, each sized for the whole batch, the send buffer packed per destination, the receive buffer, the rows grouped by expert, the two intermediate results of the experts, and the $$W$$ partial results per token. Three mechanisms remove them:

- fusion keeps the two intermediate results of the experts, the largest tensors of the layer, in local memory;
- INC All-to-All-Dispatch removes the send buffer packed per destination, as each rank sends one copy per token, in token order;
- INC All-to-All-Combine removes the receive buffer of $$W$$ partial results, as each owner receives one finished result.

What remains belongs to one microbatch, and is alive only between its `route` and its `combine`, so at most $$D$$ sets are alive at once in a pipeline of depth $$D$$. The resident memory is therefore given by:

$$
\Theta \left( D \, (W+2) \, m_b H \right)
$$

i.e. a function of the $$D\,m_b$$ tokens in flight, not of the $$N$$ tokens of the batch, so doubling $$N$$ at fixed $$m_b$$ and $$D$$ leaves it unchanged. It grows linearly with the microbatches in flight, the same knob that buys the overlap of the pipeline.

However, the prototype allocates the buffers of all $$M$$ microbatches: in the sweep above, $$M$$ ranges from 8 to 64 against a depth of 5. An allocator bounded by the depth would hold five buffer sets where the current one reserves $$M$$. This allocator is not implemented, and its saving is not measured.

## Final remarks

Several directions remain open:

- switch hardware: implementing the INC collectives and the plain all-to-all in the same switch, at a larger $$W$$, would give the fair comparison that an emulated switch cannot;
- an allocator bounded by the pipeline depth would turn the memory bound above into a measured saving;
- a dispatch in reduced precision, possibly with [quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) in the INC switch, would halve the largest transfer, and composes with the INC pull model;
- expert load balance matters too, as many modern MoE models include shared experts, or are served with experts cloned across ranks, which the fused scheduler could accommodate;
- prefill and decode measured in isolation do not represent serving, so continuous batching is the next evaluation.

To find more about the topic, [DySHARP](https://computer-arch.hkust.edu.hk/publication/isca2026-tackling-moe-communication-bottleneck-wi/) (ISCA 2026) is the closest work. It removes the same two redundancies (a token sent once per destination on the dispatch, and partial results returned separately on the combine) with multicast and reduction inside the switch, evaluated in simulation. However, two of its four groups of SMs still issue the dispatch and the combine.

On the software side, the following works speed up or fuse the dispatch and the combine:

- DeepEP provides expert-parallel communication kernels;
- COMET fuses the dispatch and combine into the matrix multiplications of the experts;
- the Mega MoE kernel of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) fuses the dispatch, the experts and the combine into one kernel;
- [Triton-distributed](https://arxiv.org/abs/2504.19442) moves the transfers of the dispatch and combine inside its kernels.

All of them still run the transfers on the compute units.

This post summarizes the paper *MoeInc: An Expert Parallelism MegaKernel Overlapping Matrix and Vector Units with In-Network Dispatch-and-Combine*.
