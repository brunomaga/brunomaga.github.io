---
layout: post
title:  "Mixture-of-Experts: a historical overview, with implementations"
categories: [machine learning, Transformer, GPT, DeepSpeed, inference, mixture-of-experts]
tags: [machinelearning]
---

# 2017 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
covered in a [different post]({{ site.baseurl }}{% post_url 2024-01-19-Mixture-of-Experts %})

# 2014 [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)
covered in a [different post]({{ site.baseurl }}{% post_url 2024-01-19-Mixture-of-Experts %})

# 1991 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
covered in a [different post]({{ site.baseurl }}{% post_url 2024-01-19-Mixture-of-Experts %})

[GPT-4](https://en.wikipedia.org/wiki/GPT-4) details have been [leaked](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/), and it mentions the usage of Mixture-of-Experts (MoE), with 16 different experts of circa 110 billion parameters each, totalling about 1.8 trillion parameters. Briefly after, [Mistral 7x8B](https://mistral.ai/news/mixtral-of-experts/) has been released as a MoE model of 8 experts with 7 billion parameters each, claiming its performance beats [GPT-3.5](https://en.wikipedia.org/wiki/GPT-3) with circa 175 billion parameters. So what is Mixture of Experts, and how to implement it? In this post, we will go through an historical overview of MoEs. We will foccus on efforts towards large-scale distributed/parallel MoE implementations.

# 1991 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

In 1991, the Mixture of Experts (MoE) was introducesd in paper [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), as a system of parallel networks, of the same architecture but initialized differently, receiving the same input, alongside a gating network (a feed-forward network, also receiving the same input). The output of the networks are softmax-normalized and weighted by the probabilities output by the selector (gating mechanism) . In practice Mixtures of Experts combine the outputs of several "expert" networks, each of which specializes in a different part of the input space. This can also be seen as a probability model, where the final probability over classes is marginalized over the selection of expert. The final output for all experts $$f_i$$ and a gating network $$g$$ is then:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/Mixture-of-Experts/MoE_1991.png"/>

$$
F(x) = \sum_{i=1}^N g_i(x)  softmax(f_i(x))
$$

if each $$f_i$$ maps an output to $$C$$ outputs (class $$c$$), for all experts $$e_i$$ then the previous follows to:

$$
F(x) = \sum_{i=1}^N p(e_i \mid x) \, p(c \mid e_i, x) = p (c \mid x)
$$

### How and why training and inference works?

Training and inference MoEs is as follows: 
- During training, all experts and the router/gating are training simultaneously. 
- During inference, only a few (or all) experts are activated, depending on the routing algorithm.

Why do we train all experts instead only a few during training? Because we want to train experts and gating jointly, so if we have a high loss, either the gating will learn to pick a better expert, or the expert will learnt to perform that task better. It becomes interesting that experts can specialize in their own task, and do not collapse into a single model (task). The paper [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813) covers this topic and formally studies "how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model". It claims that:
- "cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE";
- there needs to be non-linearity in the experts for an MoE system to work; and 
- "the router can learn the cluster-center features, which helps divide the input complex problem into simpler linear classification sub-problems that individual experts can conquer".

This was demonstrated using a top-1 MoE model on a task classification problem with intrinsic cluster structures, which is hard to learn using a single expert. Each expert is a two-layer CNN.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2022_Zixiang_Chen.png"/>


# 2014 [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

Later, the 2014 paper [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314) "extends the Mixture of Experts work (MoE) to a stacked model, the Deep Mixture of Experts, with multiple sets of gating and experts. This exponentially increases the number of effective experts by associating each input with a combination of experts at each layer, yet maintains a modest model size." 
Formally, extends the [1991 paper Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) with 2 sets of experts ($$g^1$$, $$f_i^1$$) and ($$g^2$$, $$f_i^2$$) alongside a linear layer $$f^3$$, such as:

$$
z^1 = \sum_{i=1}^N g^1_i(x) f^1_i(x) \\
z^2 = \sum_{j=1}^N g^2_j(z^1) f^2_j(z^1) \\

F(x) = z^3 = softmax ( f^3 ( z^2 ) )
$$

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/Mixture-of-Experts/MoE_2014_Ilya.png"/>

In the paper, experts $$f_i^l$$ and $$f_j^l$$ are single-layer DNN and $$g^l_i$$ is a two-layer DNN, with rectification.  During training, they authors noticed that results in a **degenerate local minimum**: "experts at each layer that perform best for the first few examples end up overpowering the remaining experts”, because as they are better initially, then end up being picked more by the gates. This is overcome by capping (*max clipping*) the assignments to a maximum value, and re-normalizing across all (post max-clipping) gating probabilities. This **cap value is then removed in a second training stage**. 

They tested:
the effect of using a mixture at the second layer by comparing against using only a single fixed expert at the second layer, or concatenating the output of all experts (ie N · h hidden units). It’s expected for the concatenated model to perform better than the mixture, and the mixture to perform better than the single network.
they compared a two-layer MoE model against a one-layer MoE model
they compared also against a fullyconnected deep network with the same total number of parameters.
The authors tested the approach on the MNIST dataset and on a dataset of monophone speech samples. On the MNIST dataset, In most cases, the deeply stacked experts performs between the single and concatenated experts baselines on the training set. It also almost matches the concatenated version. However it suffers from overfitting, and the test set accuracy is lower. On the monophone speech dataset, the results are simillar, however it does not overfit on the test set and performance is simillar. 

Note: in brief, Mixture of Experts work because the total error takes into account the average of all expert errors, so the optimizer will try to fix/improve each expert.

### Picking variable number of experts and layer with Reinforcement Learning 

One of the fallacies of the previous approaches is that $$k$$ (the number of experts picked) is an user-defined parameter. This could lead to situations where we have more or less $$k$$s than needed, resulting in over- or under-computation assigned. 
As a follow up, the 2015 paper [Conditional Computation in Neural Networks for faster models](https://arxiv.org/abs/1511.06297) improves the formumation by allowing variable number of experts and layers to be picked, "capturing the idea of wanting to have parsimonious activations while maintaining prediction accuracy". It proposes using policy gradient to train activation-dependent policies for dropping out blocks of exprerts ("units"). The optimization of the input-dependent activation probabilities at each layer is setup as a  discrete time, continuous state and discrete action Markov Decision Process. Each node or block in a given layer is assigned a Bernoulli distribution, and a action $$u \in \{ 0,1 \}^k$$ (describing whether or not to put a mask over the experts of a given layer). Thus, they define a $$k$$-dimensional Bernoulli policy for every layer $$l$$ of $$k$$ experts. This is practice is an approach similar to a dropout (think Bernoulli distribution) but across experts, however here they $$p$$ parameter is trainable and differs accross experts.

# 2017 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

Towards scaling this approach,  the 2017 paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538),  introduces Sparsely-Gated Mixture-of-Experts layer (MoE). Sparsity uses the idea of **conditional computation**, where parts of the network are active on a per-example basis - ie instead of picking all experts, it picks only a few. I.e. when $$g_i(x)=0$$, then $$f_i(x)$$ is not computed. **"All parts of the network are trained jointly by back-propagation"**.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2017_Dean.png"/>

 If number is experts is two layer, we can use a two-layer hierarchical MoE, where each gating network chooses a sparse weighted combination of “experts” (not covered in main paper, hierarchical MoEs are in appendix B). Sparsity is great because it allows the model size to increase without increasing computation. Each expert is a feed-forward NN.  If we start with that regular softmax-based gating function $$g(x)$$:

$$
G_σ(x) = Softmax(x · W_g)
$$

Note that the authors added $$W_g$$, a linear transformation on the input before being passed to the gate. The proposed method is a **noisy top-k gating**, where they add **tunable gaussian noise** $$H(X)$$ (controlled by $$W_{noise}$$) before taking the softmax, and added sparsity (to save computation) by keeping only the top $$k$$ values, setting the rest to -∞ (which causes the corresponding gate values to equal 0)”:

$$
g(x) = softmax( \, KeepTopK ( \, h(x), k) )
h(x)i = (x · W_g)_i + StandardNormal() · Softplus ( (x · W_{noise})_i ) 
$$ 

During training, if we pick only a few experts, we still end up with a very large amount of activations from unused experts. This was fixed by:
- **mixing model and data parallelism** to allow memory- and compute-efficient scaling on expert on distributed systems: Each processor has a data-parallel replicas of the standard layers and the gating networks, and model-parallel shards for a subset of the experts.  Batch is passed in a data-parallel fashion - each device receives a different micro-batch - and forward batches run synchronously so that they can be combined for the MoE layer. For $$d$$ devices, it’s a factor of $$d$$ improvement in expert batch size.
- **Taking Advantage of Convolutionality (in language models only)** by applying apply same MoE to each time step of the previous layer.  E.g in one single batch run, the expert 1 takes timestep 1 of all inputs, expert 2 takes timestep 2 of all inputs,…, simultaneously, allowing one to apply the MoE to all the time steps together as one big batch. This reduces input size by a factor of unrolled time steps;
They built a 137B parameter recurrent language model where the *sparse* gating functions selected two experts to perform computation, and the expert outputs are modulated by the gating function.
- **Increasing Batch Size for a Recurrent MoE:** recurrent models e.g. LSTM/RNN would break the convolutionality trick before. ". Gruslys et al. (2016) describe a technique for drastically reducing the number of stored activations in an unrolled RNN, at the cost of recomputing forward activations. This would allow for a large increase in batch size".

To avoid the common effect of having few experts taking all the importance from the gating mechanism at early stages, they adopted a soft constraing approach (note: hard approach is fixing a maximum cap threshold). In practice, they add an additional **importance loss**, equal to the square of the coefficient of variation of the set of importance values, is added as:

$$
L_{importance} = w_{importance} \cdot CV \, (Importance (X))^2
$$ 

where the $$Importance$$ of an expert to be the batchwise sum of the gate values for that expert. This loss ensures equal importance, but experts may receive different numbers of examples.

The experiments fixed a compute budgets and compared a baseline model with several configurations of MoE (variable number of experts, with and without hierarchy etc), on several language tasks. Results show MoE models beating baseline LSTM models (less perplexity/confusion), with hierarchical MoE beating non-hierarchichal (with a higher parameter count, but same compute). Perplexit improved significantly by increasing data size (from 10B to 100B Word Google News Corpus). In the machine translation task, "Google Production dataset, our model achieved 1.01 higher test BLEU score even after training for only one sixth of the time."

# 2022 [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

With the explosing increase in interest on Transformer-based architectures, it is not a surprise that MoEs gained a special interest. In 2023, GshardGShard demonstrated scaling of MoEs on a model of 600 billion parameters sequence-to-sequence Transformer model with Sparsely-Gated Mixture-of-Experts layers. They claim that the their methods yield sub-linear computation cost and graph $$O(1)$$ compilation time on 2048 TPU devices.
This performance was achieved via:
- **Position-wise Sparsely-Gated MoEs**, just like the previous paper, that scales the capacity of RNN-based machine translation and language models and delivers sub-linear scaling;
- **GShard module that separates model description from the parallel partitioning**. It consists of "of a set of simple APIs for annotations, and a compiler extension in XLA for automatic parallelization." This allows developers to write the module and the runtime will automatically partition it accross compute unitrs;
- **compiler technique for SPMD (Single Program Multiple Data) transformation that generates a single program to run on all devices**, in order to keep the compilation time constant independent of the number of devices. 

The model used for the  sequence-to-sequence task (machine translation) presented is a Transformer-based language model, made of stacks of encoder and decoder layers. Each Transformer block (made of self-attention, feed-forward and on the decoder's case, attention mask) was converted to a conditional computation by replacing **every other feed-forward layer** with a MoE layer with a variant of **top-2 gating** in both the encoder and the decoder. The encoder part can be represented as:

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Mixture-of-Experts/MoE_GShard.png"/>

Strangely enough, the paper does not mention why picking "replacing every second FFN by an MoE" instead of picking every third or every single FFN instead. The parallelism follows the hybrib sharded-MoEs with data-parallel FFN and gating, as in the previous paper - confusingly, the "model-parallel MoE" in the picture refers to sharding and not to tensor parallelism. The underlying formulation of the gating mechanism is the following: A MoE layer for Transformer consists of $$E$$ feed-forward networks $$FFN_1$$ ... $$FFN_E$$ where:

$$
FFN_e(x_s) = wo_e · ReLU(wi_e · x_s)

y_x = \sum_{e=1}^{E} G_e(x_s) · FFN_e(x_s)
$$

where $$G_e(x_s)$$ is the probabilities (post-softmax) output by the gating mechanism for expert $$e$$ given input $$x_s$$, and $$y_x$$ is as before the weighted sum of products of the gating probabilities and the FFN outputs (FFN here refers to an expert). $$x_s$$ is the input token to the MoE layer, $$w_i$$ and $$w_o$$ are the input and output projection matrices for the feed-forward layer.

The avoid few experts *stealing* most of the attention at the early stages of training, an **auxiliary loss** similar to the previous was also added. **Load balancing** was regulated by assigning a maximum number of tokens to each expert (not that batches are of heterogenous number of tokens per sentence, thus here they choose to balance not by number of samples but by total number of tokens across samples). Finally, they introducted the concept of **random routing**, where in a top-2 setup, "if the weight for the 2nd expert is very small, we can simply ignore the 2nd expert to conserve the overall expert capacity". 

The parallel-partitioning is a combination of data-parallel and sharding. This is because (quoting the paper):
1. the attention layer is parallelized by splitting along the batch dimension and replicating its weights to all devices; and 
2. experts in the MoE layer are infeasible to be replicated in all the devices due to its sheer size and the only viable strategy is to shard experts into many devices.

Finally, they mention **Mixing manual and automatic sharding**, where it allows user to manually specify the sharding by telling the runtime "on how operators are partitioned, and one example is that the user has more run-time knowledge beyond the operators’ semantics", more concretely the "user might know that a specific Gather operator shuffles data only within each partition".
However, they claim that "the automatic sharding assignment is not the focus of this paper and we leave it as future work".


