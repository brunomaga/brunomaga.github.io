---
layout: post
title:  "From Mixture-of-Experts to Retrieval-Augmented Generation (RAG) and Agentic AI"
categories: ["Efficient Inference"]
tags: ["LLM", "MoE", "RAG", "agentic AI", "multi-agent systems", "latency", "continuous batching"]
---

Selecting a few specialized components for each input is a recurring idea in large-model systems. [Mixture-of-Experts]({{ site.baseurl }}{% post_url 2024-01-19-mixture-of-experts %}) (MoE) models select experts per token inside the network, [retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) (RAG) selects documents to read as context, and **agentic AI systems** (AAIs) select external agents to run. Along this progression, the selected units become coarser, more independent and more heterogeneous, which changes how latency must be modeled. Systems that combine such components are often called **[compound AI systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)**.

This post recaps the latency of MoE models, describes RAG, explains why agents are used rather than a larger model, and derives the latency of static and dynamic agentic systems. It then discusses the limitations of agentic systems in production, and compares MoE, RAG and agentic systems.

## Mixture-of-Experts

As a short reminder, an MoE layer runs on a network of accelerators in three steps:
- **Gating**: a gate (or router) picks the top-k experts of each token, and their weights.
- **Dispatch**: an [all-to-all]({{ site.baseurl }}{% post_url 2015-04-21-matrix-transpose %}) collective sends each token to the accelerators that hold its experts. Experts are typically feed-forward networks (FFNs) with different parameters.
- **Combine**: the experts process their tokens, and each accelerator sums the outputs of each token, weighted by the gate. These partial sums are sent back to the accelerator that holds the token, where they are added up, as a global reduction.

The [prefill and decode]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}) latency of an MoE model of $$L_{\text{moe}}$$ layers, for $$n_{\text{in}}$$ input tokens and $$n_{\text{out}}$$ output tokens, is given by:

$$
T_{\text{MoE}} = \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} L_{\text{moe}} \cdot t_{\text{layer}}(n_{\text{in}},\; 0)}_{\text{MoE prefill}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} L_{\text{moe}} \sum_{d=0}^{n_{\text{out}}-2} t_{\text{layer}}(1,\; n_{\text{in}} + d)}_{\text{MoE decode}}
\tag{1}
$$

where $$t_{\text{layer}}(m, c)$$ is the latency of one layer for $$m$$ new tokens and a KV cache of $$c$$ tokens. The prefill produces the first output token, and each of the other $$n_{\text{out}} - 1$$ tokens takes one decode step, which adds one token to the KV cache. The latency of a layer is computed as:

$$
t_{\text{layer}}(m, c) = t_{\text{attn}}(m, c) + t_{\text{gate}}(m) + \mathbf{t}_{\textbf{dispatch}}(m) + t_{\text{expert}}(m) + \mathbf{t}_{\textbf{combine}}(m)
\tag{2}
$$

The communication terms, $$\mathbf{t}_{\textbf{dispatch}}$$ and $$\mathbf{t}_{\textbf{combine}}$$, are in bold: they are the only steps that cross the network, and they are paid in every layer, in the prefill and at every decode step. The following picture illustrates Eq. 2 for one layer on two accelerators, with two experts each:

{: .picture}
<img width="92.8%" height="92.8%" src="{{ site.assets }}/inference/imgs/moe_dispatch_combine.svg"/>

{: .caption}
One MoE layer on two accelerators, with two experts each, and the terms of Eq. 2 under its steps. The orange arrows are network transfers: at the dispatch, each gate keeps the tokens of its local experts and sends the other tokens to the other accelerator; at the combine, each accelerator sends the partial sums of the tokens of the other accelerator back to it, where they are added up.

## Retrieval-augmented generation

[RAG](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020) extends a language model with an external knowledge base. Ahead of time, the documents are split into chunks, and an **embedding model**, usually a small transformer encoder, maps each chunk to a vector, stored in a [vector database](https://en.wikipedia.org/wiki/Vector_database). At query time, the query is embedded with the same model, the chunks closest to it are retrieved, and their text is added to the prompt. The following picture illustrates the two phases of the pipeline:

{: .picture}
<img width="92.3%" height="92.3%" src="{{ site.assets }}/inference/imgs/rag_pipeline.svg"/>

{: .caption}
The RAG pipeline. Ahead of time, the chunks of the documents are embedded and stored in a vector database. At query time, the query is embedded with the same model, and the top-$$k$$ chunks $$c_1$$ to $$c_k$$ that the search returns are placed in the prompt of the LLM, between the system prompt (sys) and the query.

The retrieval is a **similarity search**: it returns the $$k$$ chunks whose vectors are nearest to the vector of the query, usually by [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). This is the nearest-neighbor query of [spatial databases](https://en.wikipedia.org/wiki/Spatial_database), but in hundreds to thousands of dimensions, where spatial indexes such as [k-d trees](https://en.wikipedia.org/wiki/K-d_tree) and [R-trees](https://en.wikipedia.org/wiki/R-tree) become slower than a scan of all the vectors. Vector databases therefore use an [approximate nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods) (e.g. [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world)), which returns most of the true nearest neighbors for a small fraction of the work. The reason is the *curse of dimensionality*, which the [unsupervised learning post]({{ site.baseurl }}{% post_url 2017-11-01-unsupervised-learning %}#the-curse-of-dimensionality-and-nearest-neighbor-search) covers together with the main methods of approximate search.

Vanilla RAG is thus a fixed retrieve-then-generate pipeline, in which only the retrieved content varies per query. Its overhead, one forward pass of the embedding model and one search, is small and predictable next to the LLM. Later work made retrieval adaptive, letting the model decide when to retrieve during generation (e.g. [Self-RAG](https://arxiv.org/abs/2310.11511), Asai et al., 2024). When an agent decides what and when to retrieve, possibly over several rounds, the approach is called **agentic RAG**.

Note that nothing is trained when documents are added or questions are asked. The embedding model was pretrained, typically with contrastive learning on (question, relevant passage) pairs, so that texts with related meaning end up close together, and most systems use it frozen. The original RAG model, for example, started from the [dense passage retriever](https://arxiv.org/abs/2004.04906) (DPR, Karpukhin et al., 2020), and fine-tuned its query encoder with the generator, but not its document encoder, so that the documents did not have to be embedded again.

In practice, the quality of the retrieval depends on more than the embedding model:
- **Chunk size**: short chunks, such as the 21 million chunks of 100 words of the Wikipedia index of the original RAG model, make each vector stand for one specific passage.
- **Hybrid search**: a keyword search such as [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), which matches rare words and names exactly, is often combined with the vector search.
- **Reranking**: a **reranker**, a model that reads the query and each candidate chunk together, reorders the candidates before the top-$$k$$ are kept (e.g. the BERT reranker of [Nogueira and Cho](https://arxiv.org/abs/1901.04085), 2019).
- **Position in the prompt**: language models use information at the beginning or at the end of a long context better than in its middle ([Lost in the Middle](https://arxiv.org/abs/2307.03172), Liu et al., 2023).

## Agentic AI systems

An AAI can be viewed as an MoE whose experts, the agents, are built or trained separately. The agents serve different tasks (e.g. weather, code, image generation), so their compute runtimes differ; they may run in different locations, so their network latencies differ; and they can fail. The system is therefore heterogeneous over both space and time.

A larger model, with more experts or wider FFNs, does not remove the need for agents. Its knowledge is fixed at training time, and language models "struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel" ([Toolformer](https://arxiv.org/abs/2302.04761), Schick et al., 2023). Agents bring what more parameters cannot: recent or private data, exact results from tools such as code interpreters, search engines and databases, actions on external systems, and units that teams develop and update independently. Moreover, the experts of an MoE are not task specialists: the routing analysis of [Mixtral](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024) found no obvious specialization of its experts by topic.

External units can even replace parameters: [RETRO](https://arxiv.org/abs/2112.04426) (Borgeaud et al., 2022) retrieves from a database of 2 trillion tokens, and performs comparably to GPT-3 on the Pile with 25 times fewer parameters. Splitting a task among agents also gives each one its own context window: in the [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) of Anthropic, a lead agent with parallel sub-agents outperformed a single agent by 90.2% on an internal evaluation, mainly because the system spends more tokens on the problem.

The price of agents is that of a distributed system. Its latency is modeled below for two structures, which differ in when the agents of a query are chosen: static (router) and dynamic (orchestrator) multi-agent systems. The static one combines the routing and parallelization workflows of [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic, 2024), and the dynamic one is close to its orchestrator-workers workflow.

### Static AAI: the router architecture

In a **static AAI**, the agents are selected once, before execution, from the input query alone. A **router**, a separate component, selects $$K$$ agents out of a registry of $$N$$ for the whole task. It can be a small trained classifier, a similarity search over the descriptions of the agents, a small LLM prompted with those descriptions, plain rules, or a cheap shortlist followed by an LLM pick. The similarity search is the retrieval step of RAG applied to agents, except that the selected units are executed rather than read.

Unlike an MoE gate, the router is not trained with the units it routes to, it runs once per query on raw text rather than per token on hidden states, and it outputs agent ids rather than mixing weights. As a result, poor routing can often be fixed by editing the description of an agent, rather than by retraining.

The $$K$$ agents then run in parallel as a **hedged fan-out**: the system continues with the first $$k \le K$$ results, or with whatever has arrived when the **Time-To-Last-Token** (TTLT) deadline passes. The TTLT deadline is a [service-level objective](https://en.wikipedia.org/wiki/Service-level_objective) (SLO) on the time an agent takes to produce its last token. Launching more agents than needed hedges against slow or failed agents, like the hedged requests of [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/) (Dean and Barroso, 2013). Finally, a **synthesizer** (or aggregator) reconciles the outputs (overlaps, contradictions, error messages) into the final response. There is no central LLM: the router, the agents and the synthesizer are independent units, and the synthesizer cannot call further agents.

The latency is the routing time, plus the fan-out time, plus the prefill of the $$n_{\text{synth}}$$ input tokens of the synthesizer and the decode of the $$n_{\text{final}}$$ tokens of the response:

$$
T_{\text{static}} = t_{\text{route}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} \min\!\left(\max_{i \in [1,k]}\left(\mathbf{t}_{\textbf{dispatch}}^{(i)} + t_{\text{agent}}^{(i)} + \mathbf{t}_{\textbf{combine}}^{(i)}\right),\; \text{TTLT}\right)}_{\text{hedged fan-out of } K \text{ agents, first } k \text{ results within TTLT}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} t_{\text{synth}}\!\left(n_{\text{synth}},\; 0\right)}_{\text{synthesizer prefill}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} \sum_{d=0}^{n_{\text{final}}-2} t_{\text{synth}}\!\left(1,\; n_{\text{synth}} + d\right)}_{\text{synthesizer decode}}
\tag{3}
$$

where the agents are indexed by completion time, so that the maximum over the first $$k$$ is the $$k$$-th fastest of the $$K$$ agents. Unlike in an MoE, each agent has its own dispatch and combine times. A dispatch sends a structured call, e.g. JSON with the name of a tool and its arguments, and a combine returns the result, usually as text. Both follow protocols such as the [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) (MCP, Anthropic, 2024) for tools and [Agent2Agent](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) (A2A, Google, 2025) for agents. The following picture illustrates Eq. 3 for $$K = 3$$ and $$k = 2$$:

{: .picture}
<img width="67.1%" height="67.1%" src="{{ site.assets }}/inference/imgs/aai_static.svg"/>

{: .caption}
A static AAI with $$K = 3$$ agents, of which the first $$k = 2$$ results are used, and the terms of Eq. 3 under its stages. The router selects the agents, the query is dispatched to the three of them in parallel, and the synthesizer combines the first two results; the result of the third agent comes too late and is dropped. The orange arrows are network transfers.

### Dynamic AAI: the orchestrator architecture

In a **dynamic AAI**, the agents are selected during execution by a central LLM, the **orchestrator** (or lead agent), which reads each output before deciding the next calls, as in [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2023), where a model interleaves reasoning with actions. At step $$s$$, the context has $$c_s$$ tokens, with $$c_0 = 0$$. The orchestrator prefills the $$c_s - c_{s-1}$$ tokens appended since the previous step, reusing the KV cache of the first $$c_{s-1}$$, and decodes $$n_s$$ tokens, which may include calls to $$K_s$$ agents. These calls run as a hedged fan-out that waits for the first $$k_s$$ results or for the TTLT deadline, and their outputs are appended to the context. The following picture illustrates this loop, and the context of the orchestrator after two steps:

{: .picture}
<img width="57.3%" height="57.3%" src="{{ site.assets }}/inference/imgs/aai_dynamic.svg"/>

{: .caption}
A dynamic AAI. At each step, the orchestrator decodes calls to agents, which are dispatched to them, and the outputs of the agents are combined and appended to its context for the next step, until the orchestrator decodes the final answer. The context (bottom) grows at each step $$s$$ by the tokens $$d_s$$ that the orchestrator decodes and the outputs $$a_s$$ of the agents. The orange arrows are network transfers.

The loop ends when the orchestrator produces the final answer at step $$S$$, or when $$S$$ reaches the cap $$S_{\max}$$. The orchestrator thus plays the roles of the router and the synthesizer, and decides after each step whether to continue. A static AAI is therefore a dynamic one limited to a single round of agent calls, with the routing done by a separate, typically cheaper, component. The total latency is:

$$
T_{\text{dynamic}} = \sum_{s=1}^{S} \left[\, \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} t_{\text{orch}}\!\left(c_s - c_{s-1},\; c_{s-1}\right)}_{\text{orchestrator prefill}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} \sum_{d=0}^{n_s-2} t_{\text{orch}}\!\left(1,\; c_s + d\right)}_{\text{orchestrator decode}} + \underbrace{\vphantom{\sum_{d=0}^{n}\max_{i}} \min\!\left(\max_{j \in [1,k_s]}\left(\mathbf{t}_{\textbf{dispatch}}^{(j)} + t_{\text{agent}}^{(j)} + \mathbf{t}_{\textbf{combine}}^{(j)}\right),\; \text{TTLT}\right)}_{\text{hedged fan-out of } K_s \text{ agents, first } k_s \text{ results within TTLT}} \right]
\tag{4}
$$

Note that a step that calls no agents, such as the last one, adds no fan-out time. The following picture illustrates Eq. 4 on a run of three steps:

{: .picture}
<img width="92.2%" height="92.2%" src="{{ site.assets }}/inference/imgs/aai_timeline.svg"/>

{: .caption}
A run of a dynamic AAI over $$S = 3$$ steps. In step 1, the orchestrator calls two agents and continues with the first result ($$K_1 = 2$$, $$k_1 = 1$$), so the result of agent 2 is not used. In step 2, it calls agent 2 alone, which misses the TTLT deadline, so the orchestrator continues without its result. In step 3, it decodes the final answer. The idle boxes are the time in which the orchestrator waits for the agents.

The post on the [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) of Anthropic shows a diagram of such a dynamic AAI in production.

### Replicas, redundancy and load balancing

Production systems run several replicas of agents and orchestrators, for capacity and for redundancy, so that the service survives failures and upgrades. A replica of a 70B-class model typically spans two to eight accelerators, and several agents often share one model pool, differing only in their prompts and tools. Replicas also enable hedging: a call goes to two replicas, and the first answer is used.

The number of replicas follows from the peak load plus spares. The [Site Reliability Engineering](https://sre.google/sre-book/production-environment/) book of Google provisions the replicas needed at the peak plus two, one for an instance that is down for an update and one for a failure during that update, e.g. 37 tasks for a peak load that needs 35. The tasks are also spread across racks, so that a top-of-rack switch is not a single point of failure.

A **[load balancer](https://en.wikipedia.org/wiki/Load_balancing_(computing))** in front of each replicated service spreads the calls across its replicas, but the load balancer of the orchestrators does not see the queues of the agents, and vice versa. Moreover, each replica holds its own KV cache, so a call that lands on a different replica than the previous call of its session has to recompute the KV cache of its context.

### Combining RAG and agentic AI

Standard RAG relies on similarity search, but advanced RAG frameworks add logic and multi-step reasoning that blur the line between retrieval systems and agents. RAG is combined with an AAI in three common ways:
- **As a RAG agent**, the most common use. A retrieval agent, or a "search the knowledge base" tool, is called by the orchestrator like any other agent, which is agentic RAG. Specialized agents also often have their own RAG over their domain, e.g. a code agent over a repository, or a legal agent over contracts. Note that RAG is data-centric: it runs a fixed pipeline to answer a query from its documents. An agent is action-centric: it decides which queries to write, runs them and interprets their results, to accomplish a broader task.
- **As tool retrieval, in the router of a static AAI.** When there are too many agents or tools to list in a prompt, their descriptions are embedded, and the query is matched against them to shortlist candidates before an LLM picks (e.g. the API retriever of [ToolLLM](https://arxiv.org/abs/2307.16789), Qin et al., 2023).
- **As long-term memory.** Agents store past results or the conversation history in a vector store, and retrieve the relevant parts later (e.g. [Generative Agents](https://arxiv.org/abs/2304.03442), Park et al., 2023, and [MemGPT](https://arxiv.org/abs/2310.08560), Packer et al., 2023). A shared vector store can also serve as a common memory between agents.

### Limitations of agentic systems

This section focuses on the large production setup: a dynamic AAI with replicas of orchestrators and agents, some of which may be RAG agents. Each query runs as a program of $$S$$ steps, and every step pays the prefill and decode of the orchestrator, then a fan-out whose dispatch, execution and combine all sit on the critical path. The network is crossed twice per agent call, at every step, and these delays add up over the whole run.

At first sight, the network does not look like a bottleneck. Orchestrators often call a single agent per step ($$K_s = 1$$), and the step count $$S$$ ranges from one or two for chat assistants to tens or more for coding and research agents. Agents and orchestrators are not necessarily co-located, and the [round-trip time](https://en.wikipedia.org/wiki/Round-trip_delay) (RTT) of a call depends on where they run:
- **Same data center**: tens of microseconds over 100 to 400 Gb/s links, negligible next to the model execution.
- **Availability zones of one region**: about a millisecond.
- **Regions of one continent**: tens of milliseconds.
- **Different continents**: roughly 70 to over 200 ms, as light in fiber adds about 1 ms of RTT per 100 km.

Third-party agents reached over the public internet add their own queueing on top. Bandwidth stays modest in every case, as agent messages are text of kilobytes to hundreds of kilobytes. Thus, for text-based agent traffic, the bottleneck is latency (round trips, queueing at shared agents and stragglers), not bandwidth, which matters mainly for KV cache movement and multimodal payloads (images, video, audio).

Finally, model servers run many requests together in a batch, and with **continuous batching**, introduced by [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) (Yu et al., 2022), requests join and leave the batch at every step (the [prefill and decode post]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}#static-and-continuous-batching) compares it with static batching). Therefore, the number of requests is a poor measure of how busy a server is: a server that runs many requests answers each of them more slowly, and one long new input slows down all the requests of its batch.

So the execution overhead lies in:
- **Queueing at shared agents.** Each agent serves many orchestrators from a finite pool of accelerators. Under load, $$t_{\text{agent}}$$ is dominated by queueing rather than by computation or transport, and depends on the load of the replica that a call lands on.
- **Stragglers and the TTLT cut-off.** A fan-out waits for the $$k$$-th fastest agent, up to TTLT. An agent that misses TTLT has used its compute for nothing, and the decision is made with fewer results: if it missed because of the network, the network lowered the quality of the answer.
- **Wasted work.** Calls that lose a hedge or miss their deadline keep running unless cancelled, and cancellations often do not propagate across agent hops. Orchestrators also get stuck repeating the same call. While agents run, the KV cache of the orchestrator sits idle: keeping it wastes memory, dropping it forces a re-prefill, and swapping it costs transfers. Recomputing the context after each external call takes 37 to 40% of the model forwarding time ([InferCept](https://arxiv.org/abs/2402.01869), Abhyankar et al., 2024).
- **Per-message overhead.** Token streaming turns each response into many small messages, so the cost of a proxy grows with the token rate rather than with the bytes.
- **Redundant transfers.** Where bandwidth matters, part of it carries contexts re-sent at every step, duplicated hedged calls and repeated calls.

No single component sees enough to address these costs: a client sees only its own calls, an inference server only its own queue, and the orchestrator nothing until the results return. A switch on the path of all agent traffic, however, sees every call of a fan-out, every step of a session, and the load and latency of every agent replica. In advanced scenarios, an accelerator inside the switch also runs model inference next to the traffic: embeddings, vector search and small language models.

### Agentic workloads in practice

Agentic queries are long sequences of mostly sequential calls:
- A coding agent takes on average about 9 steps (cycles of LLM and tool calls) and 11 tool calls per request ([TraceLab](https://arxiv.org/abs/2606.30560), Zhu et al., 2026).
- In production traces of GitHub Copilot, a session has a median of 15 LLM calls and a 90th percentile (P90) above 100. The LLM calls of a turn, the agent loop that answers one user message, hardly overlap: their P90 concurrency is only 1.4 ([Agentic Coding in the Wild](https://arxiv.org/abs/2608.00101), Liu et al., 2026).
- On agentic benchmarks, most configurations average 12 to 62 rounds of tool calls per task, with tails reaching 786 ([Agentic AI Workload Characteristics](https://arxiv.org/abs/2605.26297), Yuan et al., 2026).
- Multi-agent research systems are wider: a lead agent launches 3 to 5 sub-agents in parallel, sometimes more than ten, at roughly 15 times the tokens of a chat ([Anthropic, 2025](https://www.anthropic.com/engineering/multi-agent-research-system)).

Raw network speed is not the bottleneck. The median tool call takes 166 ms, and short tool batches hide behind the LLM execution: for batches under 500 ms, more than 99% of the tool time overlaps with an LLM call (Liu et al., 2026). A median completion is 247 tokens, about a kilobyte, against seconds of decode. The measured inefficiencies of agentic serving lie elsewhere:
- **Tails.** About 4% of the tool calls, those over a minute, take about 85% of the tool time (Zhu et al., 2026), and a lead agent waits for its slowest sub-agent (Anthropic, 2025).
- **Failures.** Tool failures occur in 9% of turns and trigger retry loops: these turns take a median of 36 LLM calls of more than 80K tokens each (Liu et al., 2026), and one benchmark configuration made 2,757 edit attempts at a 95.4% failure rate (Yuan et al., 2026).
- **Fragile state.** The KV cache hit rate is about 90% within a turn, 55% across turns, and 8% after a model switch, usually caused by errors or rate limits (Liu et al., 2026). Of the tokens that the traced coding agents prefill, 81% are repeated work ([Zhu et al., 2026](https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/)).
- **Context growth.** Tool-call results make up 28% of the prompt tokens, while tool calls take only 4.7% of the time (Liu et al., 2026), so every unfiltered result inflates every later step.
- **Heterogeneity.** The token use per turn spans 50 times across user types, from 23K to 1.1M tokens, which, as the paper argues, makes uniform policies suboptimal (Liu et al., 2026).

## Comparison of MoE, RAG and agentic AI systems

The four systems select their units in different ways, and each fits a different use:
- **MoE** selects experts inside one model, with a gate trained jointly with them. It scales the parameters of a model at a fixed compute per token, but its experts cannot change without training.
- **RAG** selects data rather than units that run. It gives a model knowledge that is too large, too private or too recent to be in its weights, at the small and predictable cost of an embedding and a search.
- **Static AAI** selects agents once per query, which keeps its cost and latency predictable, and fits tasks whose parts are known from the query.
- **Dynamic AAI** selects agents at every step, which fits open-ended tasks whose next step depends on the previous results, at the price of a cost and a latency that vary from query to query.

The following tables compare them in detail, where $$E$$ is the number of experts of the MoE, $$N$$ the number of agents in the registry, $$K$$ the number of agents launched and $$k$$ the number of results used.

### Routing and architecture

From the MoE to the dynamic AAI, the router moves out of the model, from a gate trained inside each layer to separate components that are configured rather than trained:

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Routing | Every token, every MoE layer, in prefill and decode | Once per query, before generation | Once per query, before any agent runs | Every step, via tool calls emitted during decode |
| Router | Learned gate, trained jointly | Retriever: embedding model and vector search | Classifier, embeddings, LLM or rules | Orchestrator LLM, in tokens |
| Differentiable | Yes, via the gate weights (top-k itself is discrete) | Usually no (frozen retriever) | No | No |
| Router output used as | Selection and blend weights | Ranking only (scores dropped) | Ranking only (scores dropped) | Not scored at all |
| Units active per query | Top-k of $$E$$ (e.g. 2 of 8) | Fixed top-k chunks | $$K$$ launched, first $$k$$ used | Varies per step; total unknown ahead |
| Total units | $$E$$ experts, fixed at training | Chunk corpus, grows freely | Editable registry of $$N$$ | Editable registry of $$N$$ |
| Selection at scale | Top-k over all $$E$$ experts | ANN index (HNSW, IVF), millions of chunks | Hierarchical, or retrieve-then-rerank past a few dozen | Same as static, plus the step budget $$S_{\max}$$ |

### Execution

The combine step is where the systems differ most: an MoE sums the activations of its experts, while RAG and the AAIs concatenate text into a prompt, which loses information and makes the context grow:

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Context layout | [sys][query][output]; experts add no tokens | [sys][c₁]…[cₖ][query][output] | [sys][query][a₁]…[aₖ][output] | [sys][query][d₁][a₁][d₂][a₂]…[output] |
| Combine | Weighted sum of activations | Concatenated into the prompt | Concatenation and synthesizer call | Concatenation, incrementally |
| Lossy combine | No: stays in hidden states | Partly: chunking and top-k drop context | Yes: summarized text, no vectors passed | Same as static |
| Parallelism | Experts in parallel within a layer | None: retrieve, then generate | All $$K$$ at once; waits for the $$k$$-th fastest, capped at TTLT | Within a step only; steps are sequential |
| Latency | One model call, with a dispatch and a combine all-to-all per layer (Eq. 1) | Embedding and search (ms), and a longer prefill | 3 sequential stages (route, fan-out, synthesize); about 2 if routing is cheap (Eq. 3) | $$S$$ rounds; unbounded without $$S_{\max}$$ (Eq. 4) |
| Cost per query | Flat per token | Predictable: embedding, search and 1 LLM call | Predictable: router, $$K$$ agents and synthesizer | Highly variable |
| KV cache | Single sequence | Single sequence; chunks vary, little prefix reuse | Separate per agent and synthesizer; shared prefix reusable | Prefix reused, delta prefill per round, if the cache survives the agent wait |

### Scaling the pool of units

Adding or removing a unit is where the agentic systems have the clearest advantage over an MoE: a new agent or document needs minutes of configuration or seconds of embedding, while a new expert needs training:

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Can a unit be added? | Practically no, on a trained model | Yes: embed and insert | Yes | Yes |
| What it costs | GPU-hours to weeks, plus evaluation | Embedding compute, seconds to minutes | Minutes of human work | Minutes of human work |
| Retraining needed? | Yes: the new expert and at least the gate | No | Only if the router is a trained classifier | No, the orchestrator reads descriptions |
| Why | $$W_g \in \mathbb{R}^{d_{\text{model}} \times E}$$, with $$d_{\text{model}}$$ the hidden size: $$E$$ changes, and the new column is untrained | The index is data, not weights | Configuration only | Configuration only |
| Takes effect | New weights, redeploy | After indexing | Hot-reloadable | Hot-reloadable |
| Removing a unit | Mask in the gate (quality drops) or fine-tune | Delete its vectors | Deregister instantly | Deregister instantly |
| Scaling ceiling | Memory, load imbalance, all-to-all bandwidth | Near-duplicate chunks crowd the top-k; index memory | Description collisions degrade routing | Same, plus context growth per step |

### Metrics

Each system needs its own metrics, as the quality of a unit can only be graded when the unit produces an answer, which the experts of an MoE do not. In the table, TTFT is the time to the first token, TPOT the time per output token, and P99 the 99th percentile:

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Router health | Gate entropy, load [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation), max/mean load | Recall@k, [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank), [nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) | Routing accuracy against gold routes | Steps per query, replan rate |
| Capacity | Token drop rate, capacity factor | Search queries per second, index size | Queue depth, concurrency saturation | Same, plus depth reached |
| Per-unit quality | Does not exist: no answer to grade | Per-chunk relevance | Per-agent success rate | Per-agent success rate |
| Latency metric | TTFT, TPOT, TTLT | Retrieval time and TTFT | Per-agent P99 TTLT; fan-out is the $$k$$-th fastest | End-to-end P99 TTLT, step count |
| Cost metric | Active FLOPs per token | Extra prompt tokens per query | Calls and tokens per query | Same, high variance |
| Quality | [Perplexity](https://en.wikipedia.org/wiki/Perplexity), downstream evaluations | Faithfulness, answer relevance (e.g. [RAGAS](https://arxiv.org/abs/2309.15217)) | Answer quality, escalation rate | Trajectory success, groundedness |
| Health signal | Balance loss falling (training); even load (serving) | Rate of empty or low-score retrievals | Timeout rate, [circuit breaker](https://en.wikipedia.org/wiki/Circuit_breaker_design_pattern) trips | Loop rate, cap-hit rate |
| Skewed traffic | Hot experts: stragglers, dropped tokens | Fine; cache popular results | Expected (router working); can overload popular agents | Same as static |

### Failure handling

Finally, the units of RAG and AAI systems fail independently, so a failure can be detected, retried or routed around, while in an MoE a fault of one accelerator fails the whole model, and an overflow of an expert degrades it silently:

| | MoE | RAG | Static AAI | Dynamic AAI |
|---|---|---|---|---|
| Can a unit fail? | Not independently; a GPU fault fails the whole model | The index or the embedding service can; content can be stale | Yes | Yes |
| Main failures | Capacity overflow | Irrelevant, missing or stale chunks | Timeout, bad answer, crash | Same, plus loops and drift |
| Slow unit | A straggler GPU delays every MoE layer | Slow search delays TTFT | TTLT timeout, synthesize without it | TTLT timeout, replan |
| Overflow behavior | Drop the token (residual passthrough) or reroute to the second choice | Truncate chunks to the context budget | Queue or reject (rate limit); next-ranked agent | Same; the $$S_{\max}$$ cap forces an answer |
| Bad answer | Undetectable | Rerank, threshold or re-retrieve | Critic gate, retry elsewhere | Critic gate, or the orchestrator self-corrects |
| Dead unit | Not applicable (whole model down) | Answer without retrieval, or fail | Circuit breaker, pull from the routable set | Same |
| Fallback | None | The model's own knowledge, or "not found" | Next-ranked candidate, then generic, then human | Replan around it |
| Redundancy | Replicate hot experts (load balance) | Index replicas | Hedging: $$K > k$$, about $$K/k$$ times the agent cost | Per-step hedging (Eq. 4); cost compounds over steps |
| Retry safety | Not applicable | Safe (read-only) | Safe if read-only; else [idempotency](https://en.wikipedia.org/wiki/Idempotence) keys | Harder: side effects may need compensation |
| Blast radius | Silent, mild degradation | Wrong context, confidently wrong answer | One agent, single join point | A failed step invalidates later steps |
| Prevention | Load-balance auxiliary loss | Retrieval evaluation set, chunking, reranker | Routing evaluation set | Caps: $$S_{\max}$$, depth, TTLT, spend |
| Recovery path | None (preventive only) | Re-query or re-index | Scatter-gather, partial results usable | Replan, not retry |

## Further reading

More on these topics can be found in the following works:
- **Retrieval.** [REALM](https://arxiv.org/abs/2002.08909) (Guu et al., 2020, Google) trains a retriever together with the language model during pre-training. [Atlas](https://arxiv.org/abs/2208.03299) (Izacard et al., 2023, Meta) reaches over 42% accuracy on Natural Questions from 64 examples, 3% more than a model of 540B parameters, with 50 times fewer parameters. [Faiss](https://arxiv.org/abs/2401.08281) (Douze et al., 2024, Meta) is a library of indexing methods to search, cluster, compress and transform vectors.
- **Agents that call models.** [HuggingGPT](https://arxiv.org/abs/2303.17580) (Shen et al., 2023) lets an LLM plan a task, select expert models of Hugging Face by their descriptions, run them and summarize their results, much like the router and the synthesizer of a static AAI in one LLM.
- **Multi-agent systems.** [AutoGen](https://arxiv.org/abs/2308.08155) (Wu et al., 2023, Microsoft) builds applications from agents that converse with each other. [Mixture-of-Agents](https://arxiv.org/abs/2406.04692) (Wang et al., 2024) stacks layers of LLM agents, in which each agent reads all the outputs of the previous layer, and scores 65.1% on AlpacaEval 2.0 with open models only, against 57.5% for GPT-4o. [Co-Scientist](https://arxiv.org/abs/2502.18864) (Gottweis et al., 2025, Google) is a multi-agent system built on Gemini, whose agents generate, critique and refine scientific hypotheses. [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) (Cemri et al., 2025) classifies 14 failure modes of multi-agent systems in 3 categories: system design, misalignment between agents, and task verification.
- **Models trained for agentic tasks.** [Kimi K2](https://arxiv.org/abs/2507.20534) (Kimi Team, 2025) and [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (DeepSeek-AI, 2025) are open MoE models whose post-training synthesizes agentic tasks at a large scale and trains on them with reinforcement learning. The pipeline of DeepSeek-V3.2 generates over 1,800 environments and 85,000 prompts.
- **Serving agentic workloads.** [SGLang](https://arxiv.org/abs/2312.07104) (Zheng et al., 2024) reuses the KV cache across the calls of a program with *RadixAttention*. [Parrot](https://arxiv.org/abs/2405.19888) (Lin et al., 2024) exposes the dependencies between the LLM calls of an application to the serving system, with *semantic variables*. [Autellix](https://arxiv.org/abs/2502.13965) (Luo et al., 2025) schedules the LLM calls of agentic programs by program, and reaches 4 to 15 times the throughput of vLLM at the same latency.
