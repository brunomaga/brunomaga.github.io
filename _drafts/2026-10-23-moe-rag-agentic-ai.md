---
layout: post
title:  "From Mixture-of-Experts to RAG and Agentic AI Systems"
categories: ["Efficient Inference"]
tags: ["LLM", "MoE", "RAG", "agentic AI", "multi-agent systems", "latency", "continuous batching"]
---

Selecting a few specialized components for each input is a recurring idea in large-model systems. [Mixture-of-Experts]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) (MoE) models select experts per token inside the network. [Retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) (RAG) selects external documents to read as context. **Agentic AI systems** (AAIs) select external agents to execute. Along this progression, the selected units become coarser, more independent and more heterogeneous, which changes how latency must be modeled. Systems that combine such components are often called [compound AI systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/).

This post recaps the latency of an MoE model, describes RAG and two structures of agentic systems (static and dynamic), and derives the latency of each. It then discusses the limitations of agentic systems in production, with measurements of real agentic workloads, and ends with a comparison of MoE, RAG and agentic systems.

## Mixture-of-Experts

As a short reminder, an MoE system runs on a network of accelerators, and works in three steps:
- **Gating**: a gating (or routing) unit infers the ids and the weights of the top-k experts assigned to each token.
- **Dispatch**: a communication step selectively sends the tokens to the accelerators that hold their experts, with an [all-to-all]({{ site.baseurl }}{% post_url 2015-04-21-matrix-transpose %}) collective. Experts are typically feed-forward networks (FFNs) or a similar structure, with different parameters.
- **Combine**: the experts process the tokens they receive, and the outputs of each token are combined locally as a weighted sum over the expert outputs. These partial sums are then sent back to the accelerators that held the input tokens at the dispatch phase, where they are summed, as a global reduction step.

The combined prefill and decode latency of an MoE model of $$L$$ layers, for an input sequence of $$n_{\text{in}}$$ tokens and an output of $$n_{\text{out}}$$ tokens, is then given by:

$$
T_{\text{MoE}} = \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} L \cdot t_{\text{layer}}(n_{\text{in}},\; 0)}_{\text{MoE prefill}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} L \sum_{d=0}^{n_{\text{out}}-1} t_{\text{layer}}(1,\; n_{\text{in}} + d)}_{\text{MoE decode}}
\tag{1}
$$

where $$t_{\text{layer}}(m, c)$$ is the latency of one layer, a function of the size $$m$$ of the query tensor (the number of new tokens) and the size $$c$$ of the KV cache, which grows by one token at each decode step. The latency of a layer is computed as:

$$
t_{\text{layer}}(m, c) = t_{\text{attn}}(m, c) + t_{\text{gate}}(m) + \mathbf{t}_{\textbf{dispatch}}(m) + t_{\text{expert}}(m) + \mathbf{t}_{\textbf{combine}}(m)
\tag{2}
$$

Note that the terms of the dispatch and combine communication, $$\mathbf{t}_{\textbf{dispatch}}$$ and $$\mathbf{t}_{\textbf{combine}}$$, are in bold. They appear in every layer, in the prefill and at every decode step.

## Retrieval-augmented generation

[RAG](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020) extends a language model with an external knowledge base. Ahead of time, documents are split into chunks, embedded, and stored in a [vector database](https://en.wikipedia.org/wiki/Vector_database). At query time, the query is embedded with the same model, the most similar chunks are retrieved, and their text is appended to the prompt before generation. Vanilla RAG is therefore a fixed retrieve-then-generate pipeline, in which only the retrieved content varies per query. Its retrieval overhead (the query embedding plus the vector search) is small and fairly predictable. Later work made retrieval adaptive, letting the model decide when to retrieve during generation (e.g. [Self-RAG](https://arxiv.org/abs/2310.11511), Asai et al., 2024). When an agent decides what and when to retrieve, possibly over several rounds, the approach is commonly called **agentic RAG**.

The model is simple. The **embedding model** is usually a small transformer encoder. It runs once per chunk when the index is built, and once per query, a single forward pass that is cheap compared to the LLM. The rest is a vector database: an [approximate nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods) (e.g. [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world)) over high-dimensional vectors, usually by [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

As an important note, nothing is trained when documents are added or questions are asked. The embedding model was pretrained by its creator, typically with contrastive learning on many (question, relevant passage) pairs, so that texts with related meaning end up close together. The documents and the queries are only encoded with that frozen model. In a less common scenario, when retrieval is poor on the user's domain, the embedding model can be fine-tuned, but that is an extra step, not part of RAG itself: most systems today use a frozen, off-the-shelf embedding model.

## Continuous batching

Model servers process many requests together, in one group called a batch. With [continuous batching]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}), new requests join the group at every step and finished requests leave it. This keeps the server busy, and new requests do not wait for the whole group to finish.

## Agentic AI systems

An AAI can be viewed as an extension of an MoE to experts that are built or trained separately, and that are heterogeneous in compute and network bandwidth. Its agents serve different tasks (e.g. weather, code, image generation), and therefore have different compute runtimes. They may be placed at different physical locations, which leads to heterogeneous network latencies. They can fail, and they are subject to other external factors that make the system heterogeneous over both space and time. Two execution structures are considered here, which differ in when the agents invoked for a query are determined: static (router) and dynamic (orchestrator) multi-agent systems.

### Static AAI: the router architecture

In a **static AAI**, the agents are selected once, before execution, based only on the input query. A **router** selects a set of $$K$$ agents, out of a registry of $$N$$ agents, for the entire task. The router is a separate component. It can be a small trained classifier, an embedding-similarity search over the agent descriptions, a small LLM prompted with those descriptions, or plain rules. It may also work in two stages, with a cheap shortlist followed by an LLM pick. The embedding-similarity option is the retrieval step of RAG applied to agents, except that the selected units are executed rather than read.

Unlike an MoE gate, the router is not trained jointly with the units it routes to. It runs once per query on raw text, rather than per token on hidden states at every MoE layer. It also outputs agent ids (discarding their scores) rather than mixing weights. As a result, poor routing can often be fixed by editing the description of an agent, rather than by retraining.

The $$K$$ agents then run in parallel as a **hedged fan-out**: the system proceeds with the first $$k \le K$$ results, or with whatever has arrived when the TTLT deadline passes. The **Time-To-Last-Token** (TTLT) deadline is a [service-level objective](https://en.wikipedia.org/wiki/Service-level_objective) (SLO) on the time an agent takes to produce its last token. Launching more agents than needed hedges against slow or failed agents, much like the hedged requests of large-scale services described in [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) (Dean and Barroso, 2013). Finally, a **synthesizer** (or aggregator) reconciles the collected outputs (handling overlaps, contradictions, error messages, etc.) and produces the final response. There is no central LLM driving the process: the router, the agents and the synthesizer are independent units, and the synthesizer only aggregates the outputs it receives and cannot call further agents.

The latency is the routing time, plus the fan-out time, plus the synthesizer's prefill of its $$n_{\text{synth}}$$ input tokens and decode of the $$n_{\text{final}}$$ tokens of the response:

$$
T_{\text{static}} = t_{\text{route}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} \min\!\left(\max_{i \in [1,k]}\left(\mathbf{t}_{\textbf{dispatch}}^{(i)} + t_{\text{agent}}^{(i)} + \mathbf{t}_{\textbf{combine}}^{(i)}\right),\; \text{TTLT}\right)}_{\text{hedged fan-out of } K \text{ agents, first } k \text{ results within TTLT}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} t_{\text{synth}}\!\left(n_{\text{synth}},\; 0\right)}_{\text{synthesizer prefill}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} \sum_{d=0}^{n_{\text{final}}-1} t_{\text{synth}}\!\left(1,\; n_{\text{synth}} + d\right)}_{\text{synthesizer decode}}
\tag{3}
$$

where the agents are indexed by completion time, so that the maximum over the first $$k$$ is the $$k$$-th fastest of the $$K$$ agents. The time of each agent includes its own dispatch and combine, which, unlike in an MoE, differ from agent to agent.

### Dynamic AAI: the orchestrator architecture

In a **dynamic AAI**, the agents are selected during execution by a central LLM, the **orchestrator** (or lead agent), which inspects each output before deciding the next calls. At step $$s$$, the context length is $$c_s$$, with $$c_0 = 0$$. The orchestrator prefills the $$c_s - c_{s-1}$$ tokens appended since the previous step, reusing the KV cache of the first $$c_{s-1}$$ tokens. It then decodes $$n_s$$ tokens, which may include calls to a set of $$K_s$$ agents. As before, these calls run as a hedged fan-out that waits for the first $$k_s$$ results or until the TTLT deadline passes, and their outputs are appended to the context.

The loop ends when the orchestrator produces the final answer at step $$S$$, or when $$S$$ reaches the cap $$S_{\max}$$. The orchestrator thus combines the roles of the router and the synthesizer, and adds a continue-or-stop decision after each step. A static AAI therefore behaves like a dynamic one limited to a single round of agent calls, with the routing delegated to a separate, typically cheaper, component. A RAG pipeline can serve as one of the agents in either structure. In a dynamic AAI, the orchestrator can query it repeatedly with queries it writes itself, which is the agentic RAG setting described above. The total latency is then:

$$
T_{\text{dynamic}} = \sum_{s=1}^{S} \left[\, \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} t_{\text{orch}}\!\left(c_s - c_{s-1},\; c_{s-1}\right)}_{\text{orchestrator prefill}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} \sum_{d=0}^{n_s-1} t_{\text{orch}}\!\left(1,\; c_s + d\right)}_{\text{orchestrator decode}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} \min\!\left(\max_{j \in [1,k_s]}\left(\mathbf{t}_{\textbf{dispatch}}^{(j)} + t_{\text{agent}}^{(j)} + \mathbf{t}_{\textbf{combine}}^{(j)}\right),\; \text{TTLT}\right)}_{\text{hedged fan-out of } K_s \text{ agents, first } k_s \text{ results within TTLT}} \right]
\tag{4}
$$

The post [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) provides a diagram of the execution of such a dynamic multi-agent system. The last section of this post compares MoE, RAG, static and dynamic AAI systems in detail.

### Replicas, redundancy and load balancing

Production systems run several replicas of agents and orchestrators for two reasons: capacity, because one instance cannot serve all the traffic, and redundancy, so that the service survives failures and upgrades. A replica of a 70B-class model typically spans two to eight accelerators, depending on precision and parallelism. Several agents often share one model pool, and differ only in their prompts and tools. Replicas also enable hedging, where a call goes to two replicas and the first answer is used.

The count of replicas follows from the peak load plus spares. The [Site Reliability Engineering](https://sre.google/sre-book/production-environment/) book of Google provisions $$N + 2$$ replicas: $$N$$ to serve the peak load, plus one to cover an instance that is down for an update, and one for a failure during that update. In its example, a backend that handles about 100 queries per second and faces a peak of about 3,470 needs 35 tasks, so it gets 37. Its cluster scheduler also spreads the tasks of a job across racks, so that a single top-of-rack switch is not a single point of failure.

A [load balancer](https://en.wikipedia.org/wiki/Load_balancing_(computing)) in front of each replicated service spreads the calls across its replicas. The load balancer of the orchestrators does not see the queues of the agents, and vice versa. Moreover, each replica of an agent holds its own KV cache, so if a step falls on a different replica than the previous one, that replica has to recompute the KV cache.

### Combining RAG and agentic AI

Standard RAG relies mostly on embedding-similarity search to pull data, but advanced RAG frameworks add computation, logic and multi-step reasoning that blur the line between retrieval systems and AI agents. RAG can be combined with an AAI in three common ways:

- **As a RAG agent.** This is the most common use. A retrieval agent, or a "search the knowledge base" tool, is something the orchestrator calls like any other agent, which is agentic RAG. Specialized agents also often have their own RAG over their domain, like a code agent that retrieves from a repository, or a legal agent from contracts. Note the difference between an agentic RAG and an agent:
  - an agentic RAG is data-centric, bounded by the goal of finding, processing and answering questions based on specific documents or data stores, while an agent is action-centric, with the goal of accomplishing a broader task;
  - for a database search, an agentic RAG receives a query, processes it in a hardcoded pipeline and returns the result, while an agent autonomously chooses to create a query, runs it, interprets or changes the output, and returns it;
  - thus, agents have a higher level of understanding than RAG, as they can interpret the input, call the same algorithm as RAG, and then process its output.
- **In the router of a static AAI, as tool retrieval.** When there are too many agents or tools to list in a prompt, their descriptions are embedded, and the query is matched against them at query time to shortlist candidates before an LLM picks. This is called **tool retrieval** (e.g. the API retriever of [ToolLLM](https://arxiv.org/abs/2307.16789), Qin et al., 2023), and it is the embedding-similarity router of the static AAI above.
- **As a cache, i.e. long-term memory.** Agents store past results or the conversation history in a vector store, and retrieve the relevant parts later (e.g. [Generative Agents](https://arxiv.org/abs/2304.03442), Park et al., 2023, and [MemGPT](https://arxiv.org/abs/2310.08560), Packer et al., 2023). A shared vector store can also serve as a common memory between agents.

The rest of this post focuses on the large production setup: an orchestrator architecture (dynamic AAI), with possible replicas of orchestrators and agents, and possibly RAG agents.

## Limitations of agentic systems

In the AAI architecture, each query runs as a program of $$S$$ steps. Every step pays the orchestrator's prefill and decode, then a fan-out to agents whose dispatch, execution and combine all sit on the critical path. The network is crossed twice per agent call, and these delays accumulate over the whole trajectory.

At first sight, the network does not look like a bottleneck. Production systems often assign each task to a single agent ($$K = 1$$). In the fan-out of each step, the orchestrator launches $$K_s$$ agents, waits for the first $$k_s$$ results, and decides its next calls in the following step. Orchestrators often work sequentially, calling a single agent per step, so $$K_s = 1$$ is common. Step counts $$S$$ range from one or two for chat assistants to tens or more for coding and research agents.

Agents and orchestrators are not necessarily co-located. Within a data center, they are connected by 100 to 400 Gb/s links with [round-trip times](https://en.wikipedia.org/wiki/Round-trip_delay) (RTTs) of tens of microseconds, so the network transfer is negligible next to the model execution. Across availability zones of one region, RTTs grow to about a millisecond. Across regions of one continent, they reach tens of milliseconds, and between continents roughly 70 to over 200 ms, since light in fiber adds about 1 ms of RTT per 100 km. Third-party agents reached over the public internet add their own provider-side queueing on top. Bandwidth stays modest in every case, because agent messages are text of kilobytes to hundreds of kilobytes. Thus, for text-based agent traffic, the bottleneck is latency rather than bandwidth: round trips, queueing at shared agents and stragglers, not the bytes transferred. Bandwidth matters mainly for KV cache movement and for large multimodal payloads (images, video, audio).

Finally, because of continuous batching, the number of requests is a poor measure of how busy a server is. A server that runs many requests answers each of them more slowly, and one long new input slows down all the requests in its batch.

So the execution overhead lies in:
- **Queueing at shared agents.** Each agent serves many orchestrators from a finite pool of accelerators. Under load, $$t_{\text{agent}}$$ is dominated by queueing rather than by computation or transport, and it depends on the load of the replica a call lands on.
- **Per-call latency across locations.** Agents may run in the same rack, in another availability zone, in another region, or behind a third-party API, so RTTs range from tens of microseconds to hundreds of milliseconds. These requests and RTTs repeat at every step.
- **Stragglers and the TTLT cut-off.** A fan-out waits for the $$k$$-th fastest agent, capped at TTLT. Agents that miss TTLT are dropped: they have used their compute for nothing, and the synthesizer or orchestrator decides with fewer results. If an agent missed TTLT because of the network, the network directly lowered the quality of the decision.
- **Wasted work.**
  - Calls that lose a hedge or miss their deadline keep running on accelerators unless cancelled, and cancellation often does not propagate across agent hops.
  - Orchestrators get stuck repeating the same call.
  - Orchestrators re-prefill their whole context when a step lands on a replica that does not hold their KV cache.
  - While agents run, the orchestrator's KV cache sits idle: keeping it wastes accelerator memory, dropping it forces a re-prefill, and swapping it costs transfers. Recomputing the context after each external call accounts for 37 to 40% of the model forwarding time ([InferCept](https://arxiv.org/abs/2402.01869)).
- **Per-message overhead.** Token streaming turns each response into many small messages, so the cost of a proxy grows with the token rate rather than with the bytes.
- **Bandwidth where it does matter.** Bandwidth pressure comes from KV cache movement, multimodal payloads and redundant transfers, not from individual text messages. Redundant transfers include contexts re-sent at every step, duplicated hedged calls and repeated calls.

No single component sees enough to address these costs. A client sees only its own calls, an inference server sees only its own queue, and the orchestrator sees nothing until the results return. A switch on the path of all agent traffic, however, sees every call of a fan-out, every step of a session, and the load and latency of every agent replica. Moreover, in advanced scenarios, an accelerator inside the switch adds model inference next to the traffic: embeddings, vector search and small language models.

### Agentic workloads in practice

Agentic queries are long sequences of mostly sequential calls:
- A coding agent takes about 9 LLM and tool steps and 11 tool calls per request ([TraceLab](https://arxiv.org/abs/2606.30560), Zhu et al., 2026).
- In production GitHub Copilot traces, a session has a median of 15 LLM calls and a P90 above 100, with a P90 LLM concurrency of only 1.4 ([Agentic Coding in the Wild](https://arxiv.org/abs/2608.00101), Liu et al., 2026).
- On agentic benchmarks, tasks average 12 to 62 turns, with tails reaching 786 ([Agentic AI Workload Characteristics](https://arxiv.org/abs/2605.26297), Yuan et al., 2026).
- Multi-agent research systems are wider: a lead agent launches 3 to 5 sub-agents in parallel, sometimes more than ten, at roughly 15 times the tokens of a chat ([Anthropic, 2025](https://www.anthropic.com/engineering/multi-agent-research-system)).

Raw network speed is not the bottleneck. The median tool call takes 166 ms, and short tool batches are almost entirely hidden behind the LLM execution (Liu et al., 2026). A median completion is 247 tokens, about a kilobyte, against seconds of decode. The measured inefficiencies of agentic serving lie elsewhere:
- **Tails.** About 4% of the tool calls (those over a minute) account for about 85% of the tool time (Zhu et al., 2026). A lead agent also waits for the slowest sub-agent before proceeding (Anthropic, 2025).
- **Failures.** Tool failures occur in 9% of turns, and trigger retry loops that cost up to 4 times the compute (Liu et al., 2026). One configuration made 2,757 edit attempts at a 95.4% failure rate (Yuan et al., 2026).
- **Fragile state.** The KV cache hit rate falls from about 90% within a turn to 55% across turns, and to 8% after a model switch, which is mostly triggered by errors or rate limits (Liu et al., 2026). 81% of the appended prefill tokens are repeated work (Zhu et al., 2026).
- **Context growth.** Tool results are 28% of the prompt tokens while taking 4.7% of the time (Liu et al., 2026), so every unfiltered result inflates every later step.
- **Heterogeneity.** The token use per turn spans 50 times across user types, which makes uniform policies suboptimal (Liu et al., 2026).

## Comparison of MoE, RAG and agentic AI systems

The following tables compare MoE, RAG, static AAI and dynamic AAI systems in routing and architecture, execution, scaling of the pool of units, metrics, and failure handling. $$E$$ is the number of experts of the MoE, $$N$$ the number of agents in the registry, $$K$$ the number of agents launched and $$k$$ the number of results used.

**Routing and architecture**

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Routing | Every token, every MoE layer, in prefill and decode | Once per query, before generation | Once per query, before any agent runs | Every step, via tool calls emitted during decode |
| Router | Learned gate, trained jointly | Retriever: embedding model and vector search | Classifier, embeddings, LLM or rules | Orchestrator LLM, in tokens |
| Differentiable | Yes, via the gate weights (top-k itself is discrete) | Usually no (frozen retriever) | No | No |
| Routing granularity | Per token, per layer | Per query | Per query | Per step |
| Router output used as | Selection and blend weights | Ranking only (scores dropped) | Ranking only (scores dropped) | Not scored at all |
| Units active per query | Top-k of $$E$$ (e.g. 2 of 8) | Fixed top-k chunks | $$K$$ launched, first $$k$$ used | Varies per step; total unknown ahead |
| Total units | $$E$$ experts, fixed at training | Chunk corpus, grows freely | Editable registry of $$N$$ | Editable registry of $$N$$ |
| Selection at scale | Top-k over all $$E$$ experts | ANN index (HNSW, IVF), millions of chunks | Hierarchical, or retrieve-then-rerank past a few dozen | Same as static, plus the step budget $$S_{\max}$$ |

**Execution**

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Context layout | [sys][query][output]; experts add no tokens | [sys][c₁]…[cₖ][query][output] | [sys][query][a₁]…[aₖ][output] | [sys][query][d₁][a₁][d₂][a₂]…[output] |
| Combine | Weighted sum of activations | Concatenated into the prompt | Concatenation and synthesizer call | Concatenation, incrementally |
| Lossy combine | No: stays in hidden states | Partly: chunking and top-k drop context | Yes: summarized text, no vectors passed | Same as static |
| Parallelism | Experts in parallel within a layer | None: retrieve, then generate | All $$K$$ at once; waits for the $$k$$-th fastest, capped at TTLT | Within a step only; steps are sequential |
| Latency | One model call and an all-to-all per layer (Eq. 1) | Embedding and search (ms), and a longer prefill | 3 sequential stages (route, fan-out, synthesize); about 2 if routing is cheap (Eq. 3) | $$S$$ rounds; unbounded without $$S_{\max}$$ (Eq. 4) |
| Cost per query | Flat per token | Predictable: embedding, search and 1 LLM call | Predictable: router, $$K$$ agents and synthesizer | Highly variable |
| KV cache | Single sequence | Single sequence; chunks vary, little prefix reuse | Separate per agent and synthesizer; shared prefix reusable | Prefix reused, delta prefill per round, if the cache survives the agent wait |

**Scaling the pool of units**

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Can a unit be added? | Practically no, on a trained model | Yes: embed and insert | Yes | Yes |
| What it costs | GPU-hours to weeks, plus evaluation | Embedding compute, seconds to minutes | Minutes of human work | Minutes of human work |
| Retraining needed? | Yes: the new expert and at least the gate | No | Only if the router is a trained classifier | No, the orchestrator reads descriptions |
| Why | $$W_g \in \mathbb{R}^{d \times E}$$: $$E$$ changes, and the new column is untrained | The index is data, not weights | Configuration only | Configuration only |
| Takes effect | New weights, redeploy | After indexing | Hot-reloadable | Hot-reloadable |
| Removing a unit | Mask in the gate (quality drops) or fine-tune | Delete its vectors | Deregister instantly | Deregister instantly |
| Scaling ceiling | Memory, load imbalance, all-to-all bandwidth | Near-duplicate chunks crowd the top-k; index memory | Description collisions degrade routing | Same, plus context growth per step |

**Metrics**

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Router health | Gate entropy, load CV, max/mean load | Recall@k, MRR, nDCG | Routing accuracy against gold routes | Steps per query, replan rate |
| Capacity | Token drop rate, capacity factor | Search QPS, index size | Queue depth, concurrency saturation | Same, plus depth reached |
| Per-unit quality | Does not exist: no answer to grade | Per-chunk relevance | Per-agent success rate | Per-agent success rate |
| Latency metric | TTFT, TPOT, TTLT | Retrieval time and TTFT | Per-agent P99 TTLT; fan-out is the $$k$$-th fastest | End-to-end P99 TTLT, step count |
| Cost metric | Active FLOPs per token | Extra prompt tokens per query | Calls and tokens per query | Same, high variance |
| Quality | Perplexity, downstream evaluations | Faithfulness, answer relevance (e.g. [RAGAS](https://arxiv.org/abs/2309.15217)) | Answer quality, escalation rate | Trajectory success, groundedness |
| Health signal | Balance loss falling (training); even load (serving) | Rate of empty or low-score retrievals | Timeout rate, breaker trips | Loop rate, cap-hit rate |
| Skewed traffic | Hot experts: stragglers, dropped tokens | Fine; cache popular results | Expected (router working); can overload popular agents | Same as static |

**Failure handling**

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Can a unit fail? | Not independently; a GPU fault fails the whole model | The index or the embedding service can; content can be stale | Yes | Yes |
| Main failures | Capacity overflow | Irrelevant, missing or stale chunks | Timeout, bad answer, crash | Same, plus loops and drift |
| Slow unit | A straggler GPU delays every MoE layer | Slow search delays TTFT | TTLT timeout, synthesize without it | TTLT timeout, replan |
| Overflow behavior | Drop the token (residual passthrough) or reroute to the 2nd choice | Truncate chunks to the context budget | Queue or reject (rate limit); next-ranked agent | Same; the $$S_{\max}$$ cap forces an answer |
| Bad answer | Undetectable | Rerank, threshold or re-retrieve | Critic gate, retry elsewhere | Critic gate, or the orchestrator self-corrects |
| Dead unit | Not applicable (whole model down) | Answer without retrieval, or fail | [Circuit breaker](https://en.wikipedia.org/wiki/Circuit_breaker_design_pattern), pull from the routable set | Same |
| Fallback | None | The model's own knowledge, or "not found" | Next-ranked candidate, then generic, then human | Replan around it |
| Redundancy | Replicate hot experts (load balance) | Index replicas | Hedging: $$K > k$$, about $$K/k$$ times the agent cost | Per-step hedging (Eq. 4); cost compounds over steps |
| Retry safety | Not applicable | Safe (read-only) | Safe if read-only; else [idempotency](https://en.wikipedia.org/wiki/Idempotence) keys | Harder: side effects may need compensation |
| Blast radius | Silent, mild degradation | Wrong context, confidently wrong answer | One agent, single join point | A failed step invalidates later steps |
| Prevention | Load-balance auxiliary loss | Retrieval evaluation set, chunking, reranker | Routing evaluation set | Caps: $$S_{\max}$$, depth, TTLT, spend |
| Recovery path | None (preventive only) | Re-query or re-index | Scatter-gather, partial results usable | Replan, not retry |
