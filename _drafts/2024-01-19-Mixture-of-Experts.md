---
layout: post
title:  "Mixture-of-Experts: a historical overview, with implementations"
categories: [machine learning, Transformer, GPT, DeepSpeed, inference, mixture-of-experts]
tags: [machinelearning]
---

[GPT-4](https://en.wikipedia.org/wiki/GPT-4) details have been [leaked](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/), and it mentions the usage of Mixture-of-Experts (MoE), with 16 different experts of circa 110 billion parameters each, totalling about 1.8 trillion parameters. Briefly after, [Mistral 7x8B](https://mistral.ai/news/mixtral-of-experts/) has been released as a MoE model of 8 experts with 7 billion parameters each, claiming its performance beats [GPT-3.5](https://en.wikipedia.org/wiki/GPT-3) with circa 175 billion parameters.

Behind this success is the fact that MoEs provide more efficient training (despite overfitting and lacking generalization during fine-tuning), much faster inference (as only some parameters are used, as we will see later), and better model accuracy for the same compute budget or parameter count.

So what is Mixture of Experts, and how to implement it? In this post, we will go through an historical overview of MoEs. We will foccus on efforts towards large-scale distributed/parallel MoE implementations.

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

The authors "demonstrate that the learning procedure divides up a vowel discrimination task into appropriate subtasks, each of which can be solved by a very simple expert network".

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
they compared also against a fully-connected deep network with the same total number of parameters.
The deep MoE model had 4 experts at the first layer and 16 at the second layer. Both layers had
128 hidden units. The gating networks were each two layers, with 64 units in the hidden layer.
The authors tested the approach on the MNIST dataset and on a dataset of monophone speech samples. On the MNIST dataset, In most cases, the deeply stacked experts performs between the single and concatenated experts baselines on the training set. It also almost matches the concatenated version. However it suffers from overfitting, and the test set accuracy is lower. On the monophone speech dataset, the results are simillar, however it does not overfit on the test set and performance is simillar. 

Note: in brief, Mixture of Experts work because the total error takes into account the average of all expert errors, so the optimizer will try to fix/improve each expert.

An important point is:
> we find that the Deep Mixture of Experts automatically learns to develop location-dependent ("where") experts at the first layer, and class-specific ("what") experts at the second layer.

This means that the first gating is learning to allocate different portions of the input to different experts (the "where") while the second is weighting the outputs by importance (the "what").

### Towards conditional computation, and Reinforcement Learning for better expert picking

The previous MoE implementations had a big drawback: all experts receive all inputs. In practice, the MoE was the whole model. This lead to efforts towards **Conditional Computation**, a method where only a subset of experts are picked.  In practice, the gating mechanism that was weighting individual experts outputs, was now moved to the begining of the workflow were it **routes**/gates inputs to different experts.

Moreover, one of the fallacies of the previous approaches is that $$k$$ (the number of experts picked) is an user-defined parameter. This could lead to situations where we have more or less $$k$$s than needed, resulting in over- or under-computation assigned.
As a follow up, the 2015 paper [Conditional Computation in Neural Networks for faster models](https://arxiv.org/abs/1511.06297) improves the formumation by allowing variable number of experts and layers to be picked, "capturing the idea of wanting to have parsimonious activations while maintaining prediction accuracy". It proposes using policy gradient to train activation-dependent policies for dropping out blocks of exprerts ("units"). The optimization of the input-dependent activation probabilities at each layer is setup as a  discrete time, continuous state and discrete action Markov Decision Process. Each node or block in a given layer is assigned a Bernoulli distribution, and a action $$u \in \{ 0,1 \}^k$$ (describing whether or not to put a mask over the experts of a given layer). Thus, they define a $$k$$-dimensional Bernoulli policy for every layer $$l$$ of $$k$$ experts. This is practice is an approach similar to a dropout (think Bernoulli distribution) but across experts, however here they $$p$$ parameter is trainable and differs accross experts.

# 2017 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

While [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314) included a input+MoE+gate+MoE+gate to represent a "where" and "what" structure, here they drop the first MoE and use the first gate to delegate inputs the different experts.

Towards scaling this approach,  the 2017 paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538),  introduces Sparsely-Gated Mixture-of-Experts layer (MoE). The main keypoints are:
- Sparsity uses the idea of **conditional computation**, where parts of the network are active on a per-example basis - ie instead of picking all experts, it picks only a few. I.e. when $$g_i(x)=0$$, then $$f_i(x)$$ is not computed.
- The model architure follow a stack of LSTM layers, where only a subset of experts is picked at every MoE pass.
- The MoE selects for each token in the text a possibly different combination of experts.
- Different experts specialize on different tasks based on syntax and semantics, as shows in Appendix E table 9:


{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2017_Appendix_E_table_9.png"/>

The gating network picks a sparse combination of the experts to use for each input. **"All parts of the network are trained jointly by back-propagation"**. "The different experts tend to become highly specialized based on syntax
and semantics (see Appendix E Table 9)"

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2017_Dean.png"/>

 If number is experts is two layer, we can use a two-layer hierarchical MoE, where each gating network chooses a sparse weighted combination of “experts” (not covered in main paper, hierarchical MoEs are in appendix B). Sparsity is great because it allows the model size to increase without increasing computation. Each expert is a feed-forward NN.  If we start with that regular softmax-based gating function $$g(x)$$:

$$
G_σ(x) = Softmax(x · W_g)
$$

Note that the authors added $$W_g$$, a linear transformation on the input before being passed to the gate. The proposed method is a **noisy top-k gating**, where they add **tunable gaussian noise** $$H(X)$$ (controlled by $$W_{noise}$$, to help with load balance, Appendix A) before taking the softmax. It also adds sparsity (picking only few experts, to save computation) by keeping only the top $$k$$ values, setting the rest to -∞ (which causes the corresponding gate values to equal 0)”:

$$
g(x) = softmax( \, KeepTopK ( \, h(x), k) )
h(x)_i = (x · W_g)_i + StandardNormal() · Softplus ( (x · W_{noise})_i ) 
$$ 

Note: $$SoftPlus$$ is a smooth approximation to ReLU and is used to constrain its output to be always positive.  
Sparsity is based on conditional computation. On large models it is important as it allows one to scale the model size without inreasing the computation.
During training, if we pick only a few experts, we still end up with a very large amount of activations from unused experts. This was fixed by:
- **mixing model and data parallelism** to allow memory- and compute-efficient scaling on expert on distributed systems: Each processor has a data-parallel replicas of the standard layers and the gating networks, and model-parallel shards for a subset of the experts.  Batch is passed in a data-parallel fashion - each device receives a different micro-batch - and forward batches run synchronously so that they can be combined for the MoE layer. For $$d$$ devices, it’s a factor of $$d$$ improvement in expert batch size.
- **Taking Advantage of Convolutionality (in language models only)** by applying apply same MoE to each time step of the previous layer.  E.g in one single batch run, the expert 1 takes timestep 1 of all inputs, expert 2 takes timestep 2 of all inputs,…, simultaneously, allowing one to apply the MoE to all the time steps together as one big batch. This reduces input size by a factor of unrolled time steps;
They built a 137B parameter recurrent language model where the *sparse* gating functions selected two experts to perform computation, and the expert outputs are modulated by the gating function.
- **Increasing Batch Size for a Recurrent MoE:** recurrent models e.g. LSTM/RNN would break the convolutionality trick before. ". Gruslys et al. (2016) describe a technique for drastically reducing the number of stored activations in an unrolled RNN, at the cost of recomputing forward activations. This would allow for a large increase in batch size".

> We save computation based on the sparsity of the output of $$G(x)$$. Wherever $$G(x)_i = 0$$, we need not compute $$E_i(x)$$

As in the previous paper, to avoid the common effect of having few experts taking all the importance from the gating mechanism at early stages, they adopted a soft constraing approach (note: hard approach is fixing a maximum cap threshold). In practice, they add an additional **importance loss**, equal to the square of the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) of the set of importance values, is added as:

$$
L_{importance} = w_{importance} \cdot CV \, (Importance (X))^2
$$ 

where the coefficient of variation ($$CV$$, or relative standard deviation) represents the dispersion of the probability distribution and is defined as the ratio of the standard deviation $$\sigma$$ to the mean $$\mu$$ as $$CV= \frac {\sigma }{\mu }$$.

where the $$Importance$$ of an expert to be the batchwise sum of the gate values for that expert, and $$w_{importance}$$ is a hand-tuned
scaling factor. This loss ensures equal importance, but experts may receive different numbers of examples.

Moreover, as we deal with sequences of varying lengths, experts may receive a different number of examples, that can cause memory and performance issues on distributed hardware. To solve this, a second loss $$L_{load}$$ is introduced to ensure load balancing. The additional load balancing loss encorages experts to receive a similar amount of training samples. However, note that the number of received tokens per expert is a constant and can not be backpropagated, so instead they use a smooth operator $$Load(X)$$ that can be back propagated, as:

$$
Load(X)_i = \sum_{x \in X} P(x,i)
$$

where $$P(x,i) = Pr(h(x)_i) \gt kth\_excluding(H(x), k, i))$$ denotes  the probability that $$G(x)_i$$ is **non-zero,  given a new random choice of noise on element $$i$$, but keeping the already-sampled choices of noise on the other elements** (see Appendix A for details). Note that $$G(x)_i$$ is nonzero if and only if $$H(x)_i$$ is greater than the $$k$$th-greatest element of H(x).

The experiments fixed a compute budgets and compared a baseline model with several configurations of MoE (variable number of experts, with and without hierarchy etc), on several language tasks. Results show MoE models beating baseline LSTM models (less perplexity/confusion), with hierarchical MoE beating non-hierarchichal (with a higher parameter count, but same compute). Perplexit improved significantly by increasing data size (from 10B to 100B Word Google News Corpus). In the machine translation task, "Google Production dataset, our model achieved 1.01 higher test BLEU score even after training for only one sixth of the time."

Finally, an important message in this paper is that it is suggested that learning to route does not work without the ability to compare at least two experts (due to instability purposes), i.e. we always need to backpropagate on least 2 experts on any MoE setup. The results cliams to obtain obtain "greater than 1000x improvements in model capacity with only minor losses in computational efficiency".

# 2020 [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

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

# 2021 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

A big issue with large MoEs are training instability. With that in mind [Switch Transformers](https://arxiv.org/abs/2101.03961) tackle that problem by simplifying the routing algorithm and improving model specifications. There is also a strong motivational quote about the usage of MoEs, particularly to solve low-data problems:

> The benefit of scale was exhaustively studied in Kaplan et al. (2020) which uncovered powerlaw scaling with model size, data set size and computational budget. Importantly, this work
advocates training large models on relatively small amounts of data as the computationally optimal approach.
Heeding these results, we investigate a fourth axis: increase the parameter count while
keeping the floating point operations (FLOPs) per example constant.

In order to achieve scaling, the sparsely activated layers split *unique weights* on different devices. Just like in GShard, the authors replaced the Feed Forward Network with a MoE. However, in Switch transformers, every FFN is replaced (instead of every other), and each Switch Transformer layer received two inputs (tokens) on for experts.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_Switch_Transformers.png"/>

The regular top-$$k$$ routing mechanism used in common use cases was replaced by a **Switch Routing**.  Here, in contradiction to the paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), where the authors claim that you need to compute a loss of at least 2 experts in order to train an MoE successfully, in **Switch Transformers they route to only a single expert**. The claim is that "simplification preserves model quality, reduces routing computation and performs better. This $$k=1$$ routing strategy is later referred to as a Switch layer". This leads to several benefits: reduced batch size as only a single expert is used, simpler routing and reduced communication.

They introduce the concept of **expert capacity factor**, which is the number of tokens that each expert computes, computed at batch size divided by number of experts (as before), times a capacity factor. If this capacity factor is exceeded (ie if the router sends too many inputs to a given expert), then extra tokens do not have computation associeated with them and are instead passed to the next layer via a residual connection.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_Switch_Transformers_2.png"/>

> Increasing the capacity factor increases the quality but also increases the communication overheard and the memory in the activations. A recommended initial setup is to use an MoE with a top-2 routing with 1.25 capacity factor, with one expert per core. At evaluation time, the capacity factor can be changed to reduce compute.

One of the downsides of MoEs is the large number of parameters. If one wants to run it on a smaller network we can perform distilation: 

> Successful distillation of sparse pre-trained and specialized fine-tuned models into
small dense models. We reduce the model size by up to 99% while preserving 30% of
the quality gains of the large sparse teacher.

# 2022 [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)

Training and inference MoEs is as follows: 
- During training, all experts and the router/gating are training simultaneously. 
- During inference, only a few (or all) experts are activated, depending on the routing algorithm.

How does it work? if we have a high loss, either the gating will learn to pick a better expert, or the expert will learnt to perform that task better. It becomes interesting that experts can specialize in their own task, and do not collapse into a single model (task).

The paper [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813) covers this topic and formally studies "how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model". It claims that:
- "cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE";
- there needs to be non-linearity in the experts for an MoE system to work; and 
- "the router can learn the cluster-center features, which helps divide the input complex problem into simpler linear classification sub-problems that individual experts can conquer".

This was demonstrated using a top-1 MoE model on a task classification problem with intrinsic cluster structures, which is hard to learn using a single expert. Each expert is a two-layer Conv Net, trained on the CIFAR-10 classification, and in the following image, a 4 cluster dataset:

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2022_Zixiang_Chen.png"/>


# 2022 [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)

Mixture of Experts are "hindered by training instabilities and uncertain quality during fine-tuning". 
We can use many methods to regularize the model to avoid instability. Dropout is generally a good option, except that reduces the model quality when experts are *small*. Adding complexity to the experts makes them mode capable but more prone to instability. Other options analysed that suffer from the same drawbacks are:
1. Remove multiplicative interactions, such as GEGLU and Root Mean Square Scare Parameters, detailed in section 3.1;
2. Inject model noise, section 3.2;
3. Constrain (clop) activations and gradients, where they shown that both update clipping and the router z-loss stabilize the model, but the update clipping significantly hurts the model quality, section 3.3, table 4.

> necessary pre-training was hampered by training instabilities previously undetected during smaller scale studies. These instabilities were later identified in other sparse models (see [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905) ). These results revealed a necessary balance of parameters and computation, but left an open question on how to reliably train these types of models. Our aim in this paper is to increase the practicality and reliability of sparse models. [...] We also put forth additional
analysis and a design guide (or at least, our heuristics) for sparse expert models. Furthermore, this work emphasizes jointly optimizing both the upstream pre-training and the downstream fine-tuning metrics to avoid discrepancies.

The sudy covers instability by training the baseline implementation with six random seeds, and the variants with three random seeds (to save compute).

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2022_ST-MoE.png"/>

This paper introduces also the **router z-loss**, that resolves instability issues and improves slightly the model quality, and a 269B sparse model -- **the Stable Transferable Mixture-of-Experts** or ST-MoE-32B - that achieves state-of-the-art performance across several NLP benchmarks.

$$
L_z(x) = \frac{1}{B} \sum_{i=1}^B \left( \log \sum_{j=1}^N e^{x_j^{(i)}} \right)
$$

where B is the number of tokens, N is the number of experts, and $$x \in \mathbb{R}^{B×N}$$ are the logits going into the router. The rationale behind z-loss is that it penalizes large logits into the gating network. Therefore the final loss is now a scaled (by user-defined hyper-parameters) sum of Cross-Entropy loss, load balancing sum, and z-loss sum:

$$
L_{tot} = L_{CE} + c_BL_B + c_zL_Z
$$

> The batch B of input tokens is broken into G unique groups across the data-parallelism dimension. each with size B/G. The expert capacity is equal to CF · tokens/experts where CF is the capacity factor hyperparameter (1.25 for train and 2.0 for eval), and tokens is the group size (note: the global batch size is split into smaller groups, each of size *data-parallel* Group Size). 

The authors also claim that sparse models are prone to overfit. In practice, sparse models do well when running on large datasets (pre-train), but can perform badly on smaller / finetuning datasets. This was demonstrated by running two tasks (Commitment Bank and ReCORD) of the SuperGLUE benchmakr. Each model contains 32 experts and was pre-trained on 500B tokens from the C4 corpus, and compared against a roughly equivalent (by FLOPs) dense T5 model.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2022_ST-MoE_5.png"/>

On both tasks, the sparse model converges faster to 100% train set accuracy supporting that sparse models optimize effectively under a data distribution shift.


They also tried updating a subset of model parameters, ie freezing the others, **during finetuning**. They performed 5 experiments:  5 different subsets of parameters: all parameters (All), only non MoE parameters (Non MoE), only MoE parameters (MoE), only the self-attention and enc-dec attention parameters (Attention) and only the non MoE FFN parameters (FFN). They observed that updating the non-MoE parameters roughly as as well as updating all the parameters, and that updating only the FFN parameters works a bit better.

The other note is that sparse and dense model require different vastly different performance across different batch sizes and learning rates for optimal fine-tuning:

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2022_ST-MoE_2.png"/>


Another message found is that sparse models are robust to dropped tokens (when input sentence exceeds temporal dimension in model) when fine-tuning, verified by the fact that "dropping 10-15% of tokens can perform approximately as well as models that drop < 1%". They also provide a guidance on how to design sparse models (quoting section 5):
- it is recommended top-2 routing with 1.25 capacity factor and at most one expert per core.
- they recommended (in their TPU system) 1 expert (or less) per core to keep a good compute-to-memory ratio. Using less experts lets the allocate more cores to the model parallelism “column” to have more FLOPs in the overral model.
- The capacity factor depends on the $$k$$ in top-$$k$$ and is set between 0.75 and 2 (see Table 8)
- the capacity factor can be changed during evaluation to adjust to new memory/compute requirements.
- Dense layer stacking and a multiplicative bias can boost quality (Appendix C).




How complex are the tasks that Experts learn? The authors showed that experts learn very shallow tasks, e.g. ponctuation, nouns, etc. Or rephrasing [this blog post](https://machinelearningmastery.com/mixture-of-experts/), MoEs performa divide-and-conquer approach that subdivides a problem into smaller problems that can solved by different experts combined.  They did this by "visualizing how tokens are routed among experts, [...] by passing a batch of tokens to the model and manually inspecting token assignment at each layer". At the encoder level, they observed that:
- at each layer, at least one expert specializes in **sentinel tokens** (mask tokens that represent blanks to fill-in).
- some encoder experts exhibit clear specialization, with some experts primarily operating on punctuation, verbs, proper names, counting, etc. 

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2022_ST-MoE_3.png"/>

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2022_ST-MoE_4.png"/>

At the decoder level, "expert specialization is far less noticeable in the decoder", and meaningful specialization (semantics or syntax) are not visible in decoder experts.

> We hypothesize that this lack of meaningful expert specialization is caused by the distribution of
target tokens induced by the span corruption objective. In particular, (a) a smaller number of tokens
are routed jointly in the decoder due to longer sequence lengths in the encoder (e.g. group size
is 2048 in the encoder vs 456 in the decoder in our setup) and (b) a higher proportion of tokens
are sentinel tokens in the decoder. As a result, target tokens in each group typically cover a smaller
semantic space (compared to the encoder), perhaps explaining the lack of expert specialization in the
decoder

### Improving MoE's finetuning

So to conclude, MoE models generalize better for large tasks and large datasets, but are not as good for fine-tuning. That's when [Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models](https://arxiv.org/abs/2305.14705) comes to rescue. They perform single-task fine-tuning, multi-task instruction-tuning and multi-task instruction-tuning followed by single-task fine-tuning. They showed that MoEs benefint more from instruction tuning than other models and benefit more from a higher number of tasks. In practice:

> in the absence of instruction tuning, MoE models fall short in performance when compared to dense models on downstream tasks. [...] When supplemented with instruction tuning, MoE models exceed the
performance of dense models on downstream tasks, as well as on held-out zero-shot
and few-shot tasks.


### When to use sparse MoE vs dense models?

from [this](https://huggingface.co/blog/moe#switch-transformers) blog post:

> Experts are useful for high throughput scenarios with many machines. Given a fixed compute budget for pretraining, a sparse model will be more optimal. For low throughput scenarios with little VRAM, a dense model will be better.
Note: one cannot directly compare the number of parameters between sparse and dense models, as both represent significantly different things.


# 2022 [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)


from [this](https://huggingface.co/blog/moe#switch-transformers) blog post:

> Megablocks (Nov 2022) explores efficient sparse pretraining by providing new GPU kernels that can handle the dynamism present in MoEs. Their proposal never drops tokens and maps efficiently to modern hardware, leading to significant speedups. What’s the trick? Traditional MoEs use batched matrix multiplication, which assumes all experts have the same shape and the same number of tokens. In contrast, Megablocks expresses MoE layers as block-sparse operations that can accommodate imbalanced assignment.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/Mixture-of-Experts/MoE_MegaBlocks.png"/>

# 2024 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

The current state-of-art MoE model is the Mixtral 8x7B, a sparse mixture of expert, with the same architecture as Mixtral 7B except that it supports a fully dense context length of 32k tokens and the feed-forward blocks are replaced by a Mixture of 8 feed-forward network experts. Model architecture is detailed in Table 1. It performs a top-2 routing, and because tokens can be allocated to different experts, "each token has access to 47B parameters, but only uses 13B active parameters during inference".

> Mixtral was trained with a context size of 32k tokens and it outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks. In particular, Mixtral vastly outperforms Llama 2 70B on mathematics, code generation, and multilingual benchmarks. We also provide a model fine-tuned to follow instructions, Mixtral 8x7B - Instruct, that surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B - chat model on human benchmark
.
The formulation of the model, in section 2.1, matches the GShard architecture (except that the MoE is in every layer instead of every second), with sparse top-$$k$$ MoE with gating $$G(x) = Softmax(TopK(x · Wg))$$ where $$TopK(ℓ)= −∞$$ for the non top-$$k$$ experts. As before, "in a Transformer model, the MoE layer is applied independently per token and replaces the feed-forward (FFN) sub-block of the transformer block". They use the [SwiGLU activation function](https://arxiv.org/abs/2002.05202v1) (same as Llama-2), and therefore for a given input $$x$$ the output is computed as:

$$
y = \sum_{i=0}^{n-1} Softmax(Top2(x · W_g))_i· SwiGLU_i(x).
$$
