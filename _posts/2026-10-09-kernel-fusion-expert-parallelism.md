---
layout: post
title:  "Kernel Fusion for MoE Expert Parallelism"
categories: ["Efficient Inference"]
tags: ["LLM", "kernel fusion", "MoE", "expert parallelism", "in-network computing", "collectives"]
---

A [Mixture-of-Experts]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) (MoE) layer holds many experts, of which each token activates only a few, so that the number of parameters of a model grows without growing its compute cost per token. In 2020, [GShard](https://arxiv.org/abs/2006.16668) placed the experts of such a layer on different accelerators, with **expert parallelism**, and both are now standard in large language models. The cost of expert parallelism is communication: every MoE layer sends each token to the accelerators that hold its experts, and brings their outputs back. According to [COMET](https://arxiv.org/abs/2502.19811), these two transfers take on average 47% of the execution time of popular MoE models and frameworks.

The usual remedies are of two kinds:

- scheduling the layer in chunks, so that the communication of one chunk overlaps the computation of another, as in the two-microbatch overlap of [DeepSeek-V3](https://arxiv.org/abs/2412.19437);
- fusing the communication into the compute kernels, so that the arrival of a chunk starts its matrix multiplication, as in COMET.

However, both run the transfers on the accelerator's own compute units, as a transfer must be issued by *some* execution unit of the device. A kernel that communicates must therefore reserve some of its units for it: thread blocks in COMET, and a configurable number of streaming multiprocessors (SMs) in DeepSeek's [DeepEP](https://github.com/deepseek-ai/DeepEP) library. As an example, DeepSeek-V3 gives about 15% of the SMs of each GPU to communication, and calls for hardware that takes this work off them. In practice, every unit given to communication is taken from the experts.

This post describes a kernel that takes the transfers off the compute units. It fuses the whole MoE layer into a single kernel of eight stages, and [pipelines](https://en.wikipedia.org/wiki/Pipeline_(computing)) the stages over microbatches so that three classes of hardware units work at the same time. It also moves the two collectives into an **in-network computing** (INC) element, i.e. a network switch that computes. The post then covers the two collectives that this element runs, their partial collective forms, the one-sided operations that implement them, and the results on a node with four compute devices.

## Expert parallelism: dispatch and combine

Consider $$W$$ compute ranks, i.e. devices, and an MoE layer of $$E$$ experts, where each token selects its top-$$k$$ experts. With expert parallelism, each rank holds $$E_{\text{loc}} = E/W$$ experts, and the layer processes $$N$$ tokens of width $$H$$, split into microbatches of $$m_b$$ tokens. Every MoE layer then runs two [collectives](https://en.wikipedia.org/wiki/Collective_operation):

- the **dispatch** sends each token to the ranks that hold its $$k$$ experts, and each rank runs its local experts on the tokens it receives;
- the **combine** returns the weighted outputs of the experts to the rank that owns each token, where the $$k$$ contributions are summed.

The following picture shows a small example of $$W=3$$ ranks, each holding one token and one expert, with top-$$k=2$$. The router produces a routing table, which maps each token to its experts and to the ranks that hold them:

{: .picture}
<img width="14.6%" height="14.6%" src="{{ site.assets }}/inference/imgs/ep_routing_table.svg"/>

{: .caption}
The routing table of the tokens A, B and C, owned by the ranks $$P_0$$, $$P_1$$ and $$P_2$$, which hold the experts $$E_0$$, $$E_1$$ and $$E_2$$, for top-$$k=2$$.

Two implementations of these collectives, or *backends*, are common: AllGather-ReduceScatter and All-to-All. Each of them runs through a regular switch that relays packets, or through an INC element, so there are four **communication models**. In the fused implementations of this post, the dispatch (one-to-many) is always a push, and a push-multicast with INC, and the combine (many-to-one) is a push without INC, and a pull-reduce with INC.

**AllGather-ReduceScatter** (AGRS) broadcasts all tokens to all ranks, and reduce-scatters the results. On the dispatch, every rank pushes its whole microbatch to every rank through the switch, and each rank sums the outputs of its experts per token, with zero outputs ($$\emptyset$$) for the tokens not routed to them. On the combine, every rank pushes its outputs for the tokens of each owner to that owner, $$\emptyset$$ where it holds none, and the owner computes the final sum. The following picture illustrates both collectives on the example:

{: .picture}
<img width="71.8%" height="71.8%" src="{{ site.assets }}/inference/imgs/ep_all_gather_reduce_scatter.svg"/>

{: .caption}
AllGather-ReduceScatter through a switch that relays each packet, with the dispatch (All-Gather) on the left and the combine (Reduce-Scatter) on the right. Each rank holds $$T=1$$ token and one expert, and is the token rank of its token and the expert rank of its expert. $$A_0$$ is token A sent to the expert rank $$P_0$$, $$E_0$$(AC) the expert $$E_0$$ processing tokens A and C, $$\sigma^y_t$$ the partial sum of the expert outputs of $$P_y$$ for token $$t$$, $$\Sigma_t = \sigma^0_t + \sigma^1_t + \sigma^2_t$$ their sum across all ranks, and $$\emptyset$$ a header-only packet or a token without output. Below each collective, the picture counts its steps per node, i.e. per rank $$P$$: the pushes or pulls of $$T$$ tokens or outputs between a rank and the switch, with $$T=m_b$$ in a microbatch.

AllGather-ReduceScatter does not depend on the routing, which makes it simple, but it moves up to $$W$$ times more data than needed: each rank pushes its tokens $$W$$ times on the dispatch, once per destination, and its outputs $$W$$ times on the combine.

An INC element improves it with fewer relays and less bandwidth. On the dispatch, each rank pushes a single copy of its tokens to the INC switch, which broadcasts their concatenation to all ranks once every rank has arrived. On the combine, the INC switch pulls the outputs of each rank once the rank signals them ready, sums them as they arrive, and returns to each owner the outputs of its tokens:

{: .picture}
<img width="71.7%" height="71.7%" src="{{ site.assets }}/inference/imgs/ep_all_gather_reduce_scatter_inc.svg"/>

{: .caption}
INC AllGather-ReduceScatter, with INC All-Gather on the left and INC Reduce-Scatter on the right. $$A_{012}$$ is token A for the expert ranks $$P_0$$, $$P_1$$ and $$P_2$$, $$ABC_0$$ the tokens A, B and C for $$P_0$$, $$\sigma^0_{A\emptyset C}$$ the partial sums of $$P_0$$ for the tokens A and C, with no output for B, and $$r^0_{A\emptyset C}$$ the signal that they are ready (dotted). A signal-pull is a pull after the ready signal of the rank.

**All-to-All** (A2A) avoids the worst case that AllGather always pays, i.e. sending every token to every rank whatever the routing. It sends each token only to the ranks that need it, at the cost of message sizes that depend on the routing: a token selected by experts on $$d$$ distinct ranks leaves its owner $$d$$ times, once per destination, as many times as with AllGather only when $$d=W$$. Its combine is the reverse all-to-all, followed by a sum on the owner of each token:

{: .picture}
<img width="71.7%" height="71.7%" src="{{ site.assets }}/inference/imgs/ep_all_to_all.svg"/>

{: .caption}
All-to-All through a switch that relays each packet, with the dispatch on the left and the combine on the right, in the notation of AllGather-ReduceScatter. $$T_E \le T$$ is the number of tokens and $$W_E \le W$$ the number of ranks that the routing table picks for the data of a node.

The diagram above shows that each rank sends a packet to every rank on each collective, a header-only packet ($$\emptyset$$) where it has nothing to send, so that no receiver waits forever. Thus, each collective sends $$W^2$$ packets into the switch and $$W^2$$ packets out of it, and the owner of each token sums the partial results that it receives. Therefore, AllGather-ReduceScatter suits a few ranks and many experts per token, as each token is then needed on most ranks anyway, and All-to-All suits larger groups of ranks.

An INC element improves All-to-All in the same way as AllGather, with fewer relays and less bandwidth. The INC forms of All-to-All, All-to-All-Unique and All-to-All-Reduce, are the subject of this post, and are described after the kernel that runs them.

## The hardware

The design is described in hardware-agnostic terms, which apply to the accelerators of different vendors. **Compute blocks** (e.g. the SMs of a GPU) schedule parallel threads and hold two classes of arithmetic units, which a single kernel launch programs and which run at the same time:

- **matrix units** (MUs) for two-dimensional matrix operations, such as matrix multiplications;
- **vector units** (VUs) for elementwise work, gathers, scatters and data movement.

Memory is divided into the off-chip **global memory** (GM), i.e. the [high-bandwidth memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) (HBM) of large capacity, and the on-chip **local memory** (LM), a [scratchpad](https://en.wikipedia.org/wiki/Scratchpad_memory) managed by software (e.g. the shared memory of a GPU). An interconnect fabric links the devices into one logical node, and a collective communication library (CCL) provides the collectives over it.

The collectives in this post use instead the one-sided operations of [SHMEM](https://en.wikipedia.org/wiki/SHMEM), which reach the memory of a remote rank without involving it. The ranks address each other through a **symmetric heap**, a memory region allocated at the same offset on every rank. Five operations are needed, each given with the OpenSHMEM call that implements it:

- `put_signal*` (`shmem_putmem_signal_nbi`) writes a contiguous payload into the symmetric heap of a remote rank, and then raises a flag there, in one nonblocking call (the asterisk marks the nonblocking calls);
- `signal` (an extension of OpenSHMEM) raises a remote flag without data, and a zero-length `put_signal*` can replace it;
- `get` (`shmem_getmem`) copies a region of the symmetric heap of a remote rank into local memory, and blocks until the copy is done;
- `wait_until` (`shmem_signal_wait_until`) blocks until a local flag satisfies a comparison, e.g. until it is raised;
- `wait_until_some` (`shmem_wait_until_some`) blocks until any of a set of local flags satisfies a comparison, and reports which.

Note that a cross-rank transfer is issued by a vector unit and staged through its local memory, so a rank that communicates always spends vector unit cycles on it. This hardware fact motivates the whole design.

## Fusing the layer into one kernel

Fusion is worth its effort because of what a non-fused layer costs. A conventional layer is a sequence of kernels, each loading its inputs from global memory, computing, and writing its outputs back. Fusion collapses this sequence into one launch and keeps the intermediate results on-chip, as [FlashAttention](https://arxiv.org/abs/2205.14135) did for attention, and **megakernels** extend it to a whole model, as in [a low-latency megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles).

In a distributed layer the case is stronger, as each synchronous collective breaks the layer into separate kernels, with buffers sized for the whole batch in global memory between them. The MoE dispatch and combine are the hardest collectives to fuse: their transfers are sparse, their sizes depend on the routing, and the combine carries a weighted sum.

The MoE layer is split into eight stages, chosen so that each stage runs entirely on one class of unit, and consecutive stages alternate between classes wherever a dependency allows. The stages are:

1. `route` computes the routing metadata of a microbatch, and ends with one `put_signal*` of its tokens and metadata into the INC element;
2. `dispatch` sends each rank the tokens that its experts need, and runs on the INC element;
3. `regroup` gathers the received rows into the expert-contiguous order that the grouped matrix multiplication needs;
4. `MatMul` is the gate and up projection of the experts, on the matrix units;
5. `SwiGLU` is the activation function ([SwiGLU](https://arxiv.org/abs/2002.05202)), on the vector units;
6. `MatMul` is the down projection of the experts, on the matrix units;
7. `reduce` scales each expert output by its routing weight, accumulates it into the partial result of the rank, and signals the INC element;
8. `combine` sums the partial results of each token and returns the sum to its owner, and runs on the INC element.

Only two of the eight stages are collectives, and both run on the INC element. The following picture shows the stages in dependency order:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_stages.svg"/>

{: .caption}
The eight stages of the fused MoE layer, in dependency order, each on the class of unit that runs it: matrix multiplications on the matrix units (purple), all other computation on the vector units (green), and both collectives on the in-network element (orange). The blue frame holds the experts. The time axis is the decode step, whose batch is too small to be split into microbatches, with placeholder costs that are not measured yet.

Note that the INC element puts the final result into the symmetric heap rather than into the output tensor of the layer, so one local copy to global memory remains. This copy is left out of the stages: it crosses no link of the network, its measured cost is close to zero, and it fuses into the next `route`, which reads those rows anyway.

The following picture follows one microbatch through the memories that an MoE layer touches (the local memory of a rank, its global memory, and the far side of the network), without fusion, and with fusion and INC:

{: .picture}
<img width="82.5%" height="82.5%" src="{{ site.assets }}/inference/imgs/ep_data_movement.svg"/>

{: .caption}
Data movement for one microbatch. **Left:** Baseline, one kernel per stage. **Right:** MegaKernel+INC, one fused kernel with the INC transfers. Each row is a stage (colored box) and the tensor it stores in global memory (white box), and $$\star$$ marks the symmetric heap. The blue frame holds the experts, dashed lines mark kernel boundaries, and the brace one kernel launch. Each INC stage runs as the collective or the partial collective operation.

Three implementations are compared in this post, from no fusion to fusion with INC:

- **Baseline**: each stage is its own kernel, so every boundary is a store to global memory followed by a load, and ten tensors are stored, among them the two intermediate results of the experts, the largest of the layer. Two extra stages are needed: a `pack` before the dispatch, which lays out a copy of each token contiguously per destination, and a `global reduce` after the combine, which sums the $$W$$ partial results of each token on its owner. The collectives are calls to the CCL, issued by the vector units of the rank.
- **MegaKernel**: the ten stages of the Baseline in one launch. The intermediate results of the experts stay in local memory, which cuts the round trips to global memory from ten to seven. However, the plain all-to-all still needs the `pack` and the `global reduce`, and its transfers become one-sided operations relayed by a switch, still issued by the vector units of the rank: one `put_signal*` per destination, i.e. $$W$$ per collective.
- **MegaKernel+INC**: the eight stages in one launch, with both collectives run by the INC element. The `pack` disappears, as each rank sends one copy of each token, in token order, and the `global reduce` disappears, as the sum is folded into the transfer. Only four regions of global memory remain visible to the network, all in the symmetric heap. The layer needs one launch and five round trips to global memory, with one `put_signal*` and one `signal` per microbatch issued by each compute rank.

## Pipelining and three-way overlap

A single microbatch uses one class of unit at a time, as its chain of stages is strictly sequential. However, the chain crosses from one class of unit to another seven times, and every crossing frees the unit it leaves for another microbatch. Therefore, the kernel runs $$M$$ microbatches through the chain at the same time, each one stage behind the previous one, so that at any instant the matrix units, the vector units and both INC elements work on different microbatches. The following picture shows the resulting schedule during the prefill step:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/inference/imgs/ep_pipeline.svg"/>

{: .caption}
Pipelined execution of four microbatches (mb 0 to 3) during prefill, with time to scale from the cost of each stage. Each microbatch has three rows, one per class of unit, colored as in the stages above. A kernel uses all matrix and all vector units, so no two purple and no two green boxes overlap; the two INC elements are independent, so a dispatch may overlap a combine.

In the schedule above, each microbatch is routed and put into the INC dispatch element, which fans it out once the copies of all ranks have arrived. The microbatch then passes through `regroup`, the two matrix multiplications, `SwiGLU` and `reduce`, before the combine element sums the partial results and returns them. While the matrix units multiply microbatch $$m$$, the vector units regroup microbatch $$m+1$$. Note that stage $$s$$ of microbatch $$m$$ waits only on stage $$s-1$$ of the same microbatch, so there is no global [barrier](https://en.wikipedia.org/wiki/Barrier_(computer_science)).

The schedule extends the two-microbatch overlap of DeepSeek-V3, which hides the dispatch and combine of one microbatch behind the computation of another. Here, a vertical cut through the schedule shows a dispatch, a computation and a combine in flight at once, each for a different microbatch, with both collectives on dedicated INC elements. Pipelining also bounds the memory: the buffers of a microbatch are alive only between its first and its last stage. Therefore, the number of buffer sets alive at once is the **pipeline depth** $$D$$, not the number of microbatches $$M$$, nor the whole batch that a non-fused layer holds at once.

## In-network computing

In-network computing is not new. In 2016, [SHArP](https://network.nvidia.com/sites/default/files/related-docs/solutions/hpc/paperieee_copyright.pdf) offloaded the reductions of [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) collectives into network switches. In 2019, [SwitchML](https://arxiv.org/abs/1903.06701) aggregated the gradients of data-parallel training inside programmable switches, training up to 5.5 times faster. However, both target dense and symmetric collectives known in advance, such as the all-reduce of the gradients.

The collectives of an MoE layer differ in every respect that matters to a switch. The dispatch is a [multicast](https://en.wikipedia.org/wiki/Multicast) that depends on the routing, and the combine is a sparse sum, weighted per token and expert, and interleaved with a permutation. DeepSeek names both opportunities in [Insights into DeepSeek-V3](https://arxiv.org/abs/2505.09343), a dispatch that "resembles a small-scale multicast" and a combine that could use in-network aggregation, but judges a flexible combine hard, because of its small reduction scope and its imbalanced load. In practice, the opportunity is not to aggregate gradients, but to replicate each dispatched token inside the network rather than at its sender, and to fold the sum of the combine into the transfer that carries it.

The INC element in this post is a programmable device that runs the one-sided operations above on the symmetric heap. In the prototype, two extra devices of the same type as the compute ranks play the role of the switch, one for the dispatch and one for the combine. They hold no model parameters and run no experts.

## All-to-All-Unique: the dispatch

The dispatch is replaced by **All-to-All-Unique**, a one-to-many collective on a *push-multicast* model, in which each sender sends a single copy of each token, however many ranks need it, as in INC All-Gather. Each compute rank writes its microbatch, in token order and with its routing metadata, into its own slot on the INC element, with one `put_signal*`. The element finds the senders with `wait_until_some`, in whatever order their flags rise, and regroups the tokens of each sender by destination as they arrive.

Once every sender has arrived, the element sends each rank the tokens that its experts need, with one `put_signal*`, which the rank waits for with `wait_until`. Thus, the replication and the permutation of the tokens run entirely on the element.

## All-to-All-Reduce: the combine

The conventional combine is wasteful in two ways: the $$W$$ partial results of a token cross the network independently, and their sum on the owner cannot start until all of them have arrived.

The transfer and the sum are fused into **All-to-All-Reduce**, a many-to-one collective on a *pull-reduce* model. Each compute rank writes its partial results into its own symmetric heap, raises its ready flag on the INC element with one `signal`, and goes back to the experts. The element serves the flags in the order they rise, found with `wait_until_some`: it pulls the partial results of each rank with one `get`, and adds them to the sum *as they land*. Once all ranks are in, it sends each owner its rows of the sum with one `put_signal*`.

The combine needs no metadata of its own, as it reuses the counts per sender of the dispatch, transposed, and the compute ranks run no cross-rank arithmetic. The following picture shows both collectives on the example above:

{: .picture}
<img width="73.4%" height="73.4%" src="{{ site.assets }}/inference/imgs/ep_all_to_all_inc.svg"/>

{: .caption}
The dispatch and combine of the example above through the INC switch, with $$W$$ transfers in and $$W$$ out per collective. **Left:** All-to-All-Unique. **Right:** All-to-All-Reduce. $$A_{02}$$ is token A for the expert ranks $$P_0$$ and $$P_2$$, $$AC_0$$ the tokens A and C for $$P_0$$, $$\sigma^y_{AC}$$ the partial sums of $$P_y$$ for tokens A and C, and $$r^y_{AC}$$ the signal that they are ready (dotted). The INC switch pulls the partial sums of each rank once the rank signals them ready, and adds them into $$\Sigma_t$$.

## Bandwidth and packet count

With a conventional all-to-all dispatch, a token selected by experts on $$d$$ distinct ranks is sent $$d$$ times by its owner, so each rank sends up to $$\min(k,W) \cdot m_b$$ rows per microbatch, and the network carries $$\mathcal{O}(W^2)$$ point-to-point flows. The conventional combine mirrors this, with each owner receiving up to $$\min(k,W)$$ partial results per token, and each rank issues $$W$$ operations per collective.

All-to-All-Unique and All-to-All-Reduce remove the duplicates at either end. On the dispatch, each rank sends each token *exactly once*, in token order and with no packing per destination; on the combine, the element sums the partial results in flight, and each owner receives one result per token. Therefore:

- the rows that a rank sends on the dispatch, and receives on the combine, fall from up to $$\min(k,W) \cdot m_b$$ to $$m_b$$, i.e. by up to a factor of $$W$$ when $$k \ge W$$, as in the setup below ($$k=8$$ and $$W=4$$);
- the packets that cross the switch fall from $$2W^2$$ to $$2W$$ per collective, over links whose endpoints are known in advance, so that the transfers can be scheduled ahead;
- the operations that each rank issues fall from $$W$$ to one per collective.

## Why the combine pulls

An INC element can get its data in two ways. With a *push* model, the senders write their data into the memory of the element with a `put_signal*`; with a *pull* model, they only `signal`, and the element reads the data itself with a `get`. A pusher knows its own byte count, so a push needs no agreement on sizes beforehand, whereas a puller must know how much to read. Therefore, the dispatch, whose sizes only the sender knows, pushes.

The combine pulls, because pulling lets the element add each partial result to the sum as soon as it lands, instead of holding all $$W$$ partial results before summing them. This keeps the reduction within the limited memory of the element. Pulling also lets the element choose the order, so that a late rank does not stall the sum of the ranks already present. Moreover, it gives the element [flow control](https://en.wikipedia.org/wiki/Flow_control_(data)): it reads a rank only when it can fold it in, so a burst of partial results cannot overflow its memory.

The cost is that the element starts transfers itself, rather than being a purely reactive switch, which departs from SHArP and from other in-switch designs. This is the price of fusing a reduction that is permuted, rather than symmetric.

## Collective or partial collective

The weak point of the INC collectives is the memory of the element. All-to-All-Unique and All-to-All-Reduce are collectives: they are asynchronous on the compute ranks, but synchronous on the element, which delivers nothing until all $$W$$ ranks have arrived, and holds the data of every rank until then. Their **partial collective** forms span only the $$d$$ ranks that a microbatch selects, so that the element holds the data of one token rank at a time:

- **One-to-Many-Unique** relays the tokens of each sender, still sent once, as soon as they land, without waiting for the other senders;
- **Many-to-One-Reduce** lets the owner, once signalled by its $$d$$ expert ranks that their outputs are ready ($$r$$), name them to the element ($$s$$). The element relays the request to each of them ($$s$$), pulls their partial results, and returns the sum to the owner alone.

The two $$s$$ steps carry the identity of the owner to its expert ranks, so that their results are relayed back to it. All-to-All-Reduce needs neither, as its collective call carries this information in its arguments. The following picture shows the two operations on the example above:

{: .picture}
<img width="99.7%" height="99.7%" src="{{ site.assets }}/inference/imgs/ep_one_to_many_inc.svg"/>

{: .caption}
The partial collective operations, One-to-Many-Unique (left) and Many-to-One-Reduce (right), with token C of $$P_2$$ and every rank and expert on its path in black, and the others in grey. The INC element relays the tokens of each sender as they land. $$P_0$$ and $$P_1$$ signal $$P_2$$ that their outputs are ready (dotted), $$P_2$$ signals the element $$s^{01}_C$$, naming them, and the element multicasts it to each, pulls $$\sigma^0_C$$ and $$\sigma^1_C$$, and returns their sum $$\Sigma_C$$ to $$P_2$$ alone. No transfer waits for another token rank.

The timelines below follow token C through both forms, with the operations of each device:

{: .picture}
<img width="55.9%" height="55.9%" src="{{ site.assets }}/inference/imgs/ep_timeline.svg"/>
<br/>
<img width="60.4%" height="60.4%" src="{{ site.assets }}/inference/imgs/ep_timeline_noncollective.svg"/>

{: .caption}
The dispatch and combine of token C, owned by $$P_2$$. **Top:** All-to-All-Unique and All-to-All-Reduce. **Bottom:** One-to-Many-Unique and Many-to-One-Reduce. $$P_2$$ sends $$C_{01}$$, i.e. token C for the expert ranks $$P_0$$ and $$P_1$$, which the INC element sends on as $$C_0$$ to $$P_0$$ and $$C_1$$ to $$P_1$$ once all ranks have arrived (top), or as soon as it lands (bottom). The element pulls the partial sums $$\sigma^0_C$$ and $$\sigma^1_C$$ as each rank signals it $$r^0_C$$ or $$r^1_C$$ (top), or once $$P_2$$, signalled $$r^0_C$$ and $$r^1_C$$, has signalled it $$s^{01}_C$$, which it relays as $$s^0_C$$ and $$s^1_C$$ (bottom), and sends their sum $$\Sigma_C$$ to $$P_2$$. The asterisk marks nonblocking calls, and dotted arrows are flags raised with `signal`.

The collective form takes fewer steps and relays: its combine takes three steps where the partial collective one takes five, and one relay per link and direction where the partial collective one takes up to $$W$$, plus $$\mathcal{O}(W^2)$$ signals. However, it pays in memory, $$W$$ times what the partial collective form holds, and in latency, as it waits for the slowest rank, which the partial collective form never does.

The routing density decides which form to use. Under uniform routing, each of the $$k\,m_b$$ expert choices of a microbatch misses a given rank with probability $$1-1/W$$, so the number of distinct ranks that the microbatch selects is approximately given by:

$$
d \approx W \left( 1 - \left( 1 - \frac{1}{W} \right)^{k\, m_b} \right)
$$

where $$d$$ tends to $$W$$ once $$k\,m_b \gg W$$. The join of the collective form then waits for no rank that the owner would not wait for anyway, and during prefill it hides behind the matrix units. Thus, the collective form wins and is the default: at $$k=8$$ and $$W=4$$, even a single token selects 3.6 ranks on average.

Sparse routing favors the partial collective form: with the 256 experts and $$k=8$$ of DeepSeek-V3 on 64 ranks, a decode microbatch of 4 tokens selects about 25 ranks, yet the collective form makes each rank wait for all 64. The partial collective form also wins with stragglers, with a switch of little memory, and when attention and experts run on separate ranks, as in [MegaScale-Infer](https://arxiv.org/abs/2504.02263), so that only some ranks send.

## One-sided operations of the dispatch and combine

The following tables list the one-sided operations per microbatch on $$W$$ compute ranks, through the relay switch of MegaKernel, and through the INC element of MegaKernel+INC, collective or partial collective. $$P_x$$ is the owner of a token and $$P_y$$ one of its expert ranks. In the partial collective operations, $$P_y$$ is one of the $$d_x \le \min(k\,m_b, W)$$ ranks that the microbatch of $$P_x$$ selects, and $$\sum d$$ is the sum of $$d_x$$ over the $$W$$ owners, at most $$W^2$$. Steps are the transfers on the critical path, from the data being ready to its delivery, and relays are the packets that cross the switch.

The dispatch, from $$P_x$$ to $$P_y$$, follows the push model:

| | All-to-All (MegaKernel) | All-to-All-Unique (MegaKernel+INC) | One-to-Many-Unique (MegaKernel+INC) |
|:--|:--|:--|:--|
| $$P_x$$ | `put_signal*` $$\times W$$ | `put_signal*` $$\times 1$$ | `put_signal*` $$\times 1$$ |
| Switch | (`wait_until_some` + `put_signal*`) $$\times W^2$$ | (`wait_until_some` + regroup by destination) $$\times W$$, then `put_signal*` $$\times W$$ | (`wait_until_some` + split by recipient + `put_signal*` $$\times d_x$$) $$\times W$$, and `signal` $$\times W$$ per layer |
| $$P_y$$ | `wait_until` $$\times W$$ | `wait_until` $$\times 1$$ | `wait_until_some` per sender that selects $$P_y$$ |
| Steps | 2 | 2 | 2 |
| Relays via switch | $$W^2$$ in, $$W^2$$ out | $$W$$ in, $$W$$ out | $$W$$ in, $$\sum d$$ out |
| Switch holds | one packet | the tokens of every sender, until the last arrives | the tokens of one sender, until relayed |
| Switch waits for | none | all $$W$$ senders | none |

The combine, from $$P_y$$ to $$P_x$$, follows the push model without INC and the pull model with INC:

| | All-to-All (MegaKernel) | All-to-All-Reduce (MegaKernel+INC) | Many-to-One-Reduce (MegaKernel+INC) |
|:--|:--|:--|:--|
| $$P_x$$ | `wait_until` $$\times W$$, then global reduce | `wait_until` $$\times 1$$ | `wait_until` $$\times d_x$$, `signal` $$\times 1$$, then `wait_until` $$\times 1$$ |
| Switch | (`wait_until_some` + `put_signal*`) $$\times W^2$$ | (`wait_until_some` + `get` + add to $$\Sigma$$) $$\times W$$, then `put_signal*` $$\times W$$ | (`wait_until_some` + `signal` $$\times d_x$$ + `get` $$\times d_x$$ + add to $$\Sigma_{[x]}$$ + `put_signal*`) $$\times W$$ |
| $$P_y$$ | `put_signal*` $$\times W$$ | `signal` $$\times 1$$ | `signal` per owner that it serves |
| Steps | 2 | 3 | 5 |
| Relays via switch | $$W^2$$ in, $$W^2$$ out | $$W$$ in, $$W$$ out, plus $$W$$ signals | $$\sum d$$ in, $$W$$ out, plus $$2\sum d + W$$ signals |
| Switch holds | one packet | the sums of every owner | the sums of one owner |
| Switch waits for | none | all $$W$$ ranks | none |

On the partial collective dispatch, $$P_y$$ keeps one slot and one flag per sender, so that packets from different senders never collide. As $$P_y$$ cannot know how many senders will select it, the join moves from the microbatch to the layer. Each sender marks its last microbatch, and once the element has relayed the last microbatch of every sender, it raises a layer-done flag on every rank, i.e. the $$W$$ signals per layer of the table.

On the partial collective combine, each $$P_y$$ signals $$P_x$$ through the switch, which forwards the signal unprocessed. The request of $$P_x$$ to the element carries neither a payload nor addresses: the element reuses the counts of the dispatch, and every rank keeps the sums of each owner at the same offset of its symmetric heap. Every partial result that a request names is ready when the request arrives, so no `get` waits on a straggler.

Every flag is a 32-bit integer, into which each sender writes the *magic* of its step, a value that strictly increases from step to step, so that a flag left from an earlier step never reads as an arrival. Slots are indexed by the microbatch modulo the pipeline depth, and a slot is written again only after the combine of the microbatch that used it has completed, so no further flow control is needed.

## Results

The kernel was evaluated on [OLMoE-1B-7B](https://arxiv.org/abs/2409.02060), an open model with 7 billion parameters in total and 1 billion active per token, $$E=64$$ experts, top-$$k=8$$, $$H=2048$$, an intermediate width of 1024, and 16 MoE layers in [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), served with [vLLM](https://arxiv.org/abs/2309.06180). One node holds four devices as compute ranks, i.e. $$E_{\text{loc}}=16$$ experts per rank at $$W=4$$, plus two devices of the same type as the INC elements, one for the dispatch and one for the combine. Under MegaKernel, the same two devices act as a regular switch that forwards each packet and computes nothing, so both fused implementations move the data in the same two steps per collective.

A forward pass was timed at 4,096 to 65,536 tokens per compute rank, which covers the chunks of chunked prefill and large prefill batches, and the time reported is the mean of five iterations after warm-up. Each iteration is a single forward pass over the whole batch that emits one token, i.e. the prefill step. Here, the MoE layer takes 87% to 96% of the time of an iteration, and this is the regime where the all-to-all of expert parallelism is the known bottleneck. The runtime per iteration of the three implementations is displayed below:

{: .picture}
<img width="93.8%" height="93.8%" src="{{ site.assets }}/inference/imgs/ep_results.svg"/>

{: .caption}
Runtime with the all-to-all backend, on $$\log_2$$ axes. **Left:** prefill time per iteration against the batch size. **Right:** decode time per output token against the number of concurrent sequences, with placeholder values that are not measured yet.

Comparing the Baseline to MegaKernel isolates the effect of fusion, and MegaKernel to MegaKernel+INC the effect of in-network computing, at the same number of data steps. Fusion alone reduces the runtime of the Baseline by 4.9% to 18.6%, and in-network computing reduces it by a further 0.8% to 7.8%. Overall, MegaKernel+INC is up to 24% faster than the Baseline end to end. At the largest batch, the throughput per compute rank rises from 20,353 to 25,042 tokens per second (the two INC ranks are not counted in the denominator).

The gain is present at every batch size, and grows beyond 8,192 tokens, the regime that matters for prefill-heavy serving. Note that on a $$\log_2$$ axis a constant speedup is a constant vertical offset, so the widening gap between the Baseline and MegaKernel+INC is a real growth of the relative advantage. At the smallest batch, the pipeline is too short to hide the round trip of the INC element, and MegaKernel+INC matches MegaKernel; from 16,384 tokens onward, the contribution of INC is positive and increasing.

Profiling confirms the design: on a compute rank, the time spent in cross-rank transfers is zero, as the whole share of a rank in the two collectives is one `put_signal*` and one `signal` per microbatch. Each of them is overlapped with its next stage. All matrix and vector units therefore stay available to the experts for the whole layer, and no units are reserved for communication, unlike the thread blocks of COMET and the SM budget of DeepEP.

At prefill batch sizes, the two grouped `MatMul` stages dominate the runtime of a microbatch, and the schedule fits everything else under them. The vector stages of microbatch $$m+1$$ and the INC transfers of $$m+2$$ run while the matrix units are busy with $$m$$.

The matrix units are busy 75% of the time, the vector units 57%, and each INC element 22%. Thus, the layer is *compute-bound* (see the [roofline model]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %})): its runtime is set by the matrix multiplications, and the remaining levers are arithmetic intensity and expert load balance, neither of them a communication problem. Also, the low load of the INC elements leaves room for more ranks.

In the schedule of the pipeline, dispatches never overlap each other and combines never overlap each other, but a dispatch overlaps a combine freely. An INC operation waits on the flags of the ranks with `wait_until_some`, moves the data (regrouping it on the dispatch, pulling it with blocking `get`s on the combine), and sends only once every rank has arrived. Therefore, the requests are served in order of arrival, and the work of one element is serialized.

The dispatch and the combine overlap only because they run on *different* devices. A single element that serves both would remove this concurrency and lengthen the pipeline, although at $$W=4$$ both elements are busy less than a quarter of the time.

With the AllGather-ReduceScatter backend, the results have the same shape and are slightly smaller: MegaKernel+INC improves on the Baseline by 8.9% to 18.5%, and on MegaKernel by up to 10.6%. As AllGather-ReduceScatter moves more data than needed, and its INC dispatch is a pure broadcast with no permutation, the two backends converge once the dispatch is unified. The times of MegaKernel+INC with the two backends differ by less than 2% at every batch size, and those of the Baseline by up to 7%.

## Communication volume and memory

At $$W=4$$, the peak number of rows that a rank sends per microbatch falls from about $$4m_b$$ to exactly $$m_b$$. Moreover, the symmetric heap becomes a function of the shape of the layer rather than of the routing, i.e. $$W+2$$ buffer units per microbatch, which removes the over-allocation for the worst case that data-dependent routing forces.

The working set follows the pipeline, not the batch. A non-fused layer holds at once, each sized for the whole batch, the send buffer packed per destination, the receive buffer, the rows grouped by expert, the two intermediate results of the experts, and the $$W$$ partial results per token. Three mechanisms remove them:

- fusion keeps the two intermediate results of the experts, the largest tensors of the layer, in local memory;
- All-to-All-Unique removes the send buffer packed per destination, as each rank sends one copy per token, in token order;
- All-to-All-Reduce removes the receive buffer of $$W$$ partial results, as each owner receives one finished result.

What remains belongs to one microbatch, and is alive only between its `route` and its `combine`, so at most $$D$$ sets are alive at once in a pipeline of depth $$D$$. The resident memory is therefore given by:

$$
\Theta \left( D \, (W+2) \, m_b H \right)
$$

i.e. a function of the $$D\,m_b$$ tokens in flight, not of the $$N$$ tokens of the batch, so doubling $$N$$ at fixed $$m_b$$ and $$D$$ leaves it unchanged. It grows linearly with the microbatches in flight, the same knob that buys the overlap of the pipeline.

However, the prototype allocates the buffers of all $$M$$ microbatches: in the sweep above, $$M$$ ranges from 8 to 64 against a depth of 5. An allocator bounded by the depth would hold five buffer sets where the current one reserves $$M$$. This allocator is not implemented, and its saving is not measured.

## Final remarks

Several directions remain open:

- an allocator bounded by the pipeline depth would turn the memory bound above into a measured saving;
- a dispatch in reduced precision, possibly with [quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) in the INC element, would halve the largest transfer, and composes with the pull-reduce model;
- expert load balance matters too, as many modern MoE models include shared experts, or are served with experts cloned across ranks, which the fused scheduler could accommodate;
- the decode step, which is limited by memory and latency rather than computation, is not measured yet;
- prefill and decode measured in isolation do not represent serving, so continuous batching is the next evaluation.

To find more about the topic, [DySHARP](https://computer-arch.hkust.edu.hk/publication/isca2026-tackling-moe-communication-bottleneck-wi/) (ISCA 2026) is the closest work. It removes the same two redundancies (a token sent once per destination on the dispatch, and partial results returned separately on the combine) with multicast and reduction inside the switch, evaluated in simulation. However, two of its four groups of SMs still issue the dispatch and the combine.

On the software side, the following works speed up or fuse the dispatch and the combine:

- DeepEP provides expert-parallel communication kernels;
- COMET fuses the dispatch and combine into the matrix multiplications of the experts;
- the Mega MoE kernel of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) fuses the dispatch, the experts and the combine into one kernel;
- [Triton-distributed](https://arxiv.org/abs/2504.19442) moves the transfers of the dispatch and combine inside its kernels.

All of them still run the transfers on the compute units.

This post summarizes the paper *MoeInc: an Expert Parallelism MegaKernel with Overlap of Matrix, Vector and In-Network Compute*.
