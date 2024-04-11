---
layout: post
title:  "Mixture-of-Experts: a historical overview, with distributed DeepSpeed and Pytorch implementations"
categories: [machine learning, Transformer, GPT, DeepSpeed, mixture-of-experts]
tags: [machinelearning]
---

Details of [GPT-4](https://en.wikipedia.org/wiki/GPT-4) - the current reigning champion on the [ChatBot benchmark](https://chat.lmsys.org/) - were recently [leaked](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/). It mentions the usage of **Mixture-of-Experts (MoEs)**, with 16 different experts of circa 110 billion parameters each, totalling about 1.8 trillion parameters. Briefly after, [Mistral 7x8B](https://mistral.ai/news/mixtral-of-experts/) has been released as a MoE model of 8 experts with 7 billion parameters each, claiming its performance beats [GPT-3.5](https://en.wikipedia.org/wiki/GPT-3) with circa 175 billion parameters.

Behind this success is the fact that MoEs provide better training scaling and faster inference due to **sparsity**, yielding better model accuracy for the same compute budget or parameter count.
The scaling and analysis of training large non-sparse (dense) models have been already covered in [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), [Scaling Properties of Speech Language Models](https://arxiv.org/abs/2404.00685) and [Emergent Abilities of Large Language Models](https://openreview.net/forum?id=yzkSU5zdwD).
In this post, we will go through an historical overview of training and fine-tuning of sparse models and MoEs and provide an [implementation on PyTorch](#implementation-on-pytorch) and an [implementation on DeepSpeed](#implementation-on-deepspped) of a large-scale distributed Switch Transformer MoE. We will go through the following publications:
  
- [1991 Adaptive Mixture of Local Experts](#1991-adaptive-mixture-of-local-experts)
- [2014 Learning Factored Representations in a Deep Mixture of Experts](#2014-learning-factored-representations-in-a-deep-mixture-of-experts)
- [2017 Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](#2017-outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer)
- [2020 GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](#2020-gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding)
- [2021 Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](#2021-switch-transformers-scaling-to-trillion-parameter-models-with-simple-and-efficient-sparsity)
  - [Implementation on PyTorch](#implementation-on-pytorch)
  - [Implementation on DeepSpped](#implementation-on-deepspped)
- [2022 MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](#2022-megablocks-efficient-sparse-training-with-mixture-of-experts)
- [2022 Towards Understanding Mixture of Experts in Deep Learning](#2022-towards-understanding-mixture-of-experts-in-deep-learning)
- [2022 GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](#2022-glam-efficient-scaling-of-language-models-with-mixture-of-experts)
- [2022 ST-MoE: Designing Stable and Transferable Sparse Expert Models](#2022-st-moe-designing-stable-and-transferable-sparse-expert-models)
- [2023 Mixture-of-Experts Meets Instruction Tuning: a Winning Combination for Large Language Models](#2023-mixture-of-experts-meets-instruction-tuning-a-winning-combination-for-large-language-models)
- [2024 Mixtral of Experts](#2024-mixtral-of-experts)
- [2024 Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](#2024-mixture-of-depths-dynamically-allocating-compute-in-transformer-based-language-models)

## 1991 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

In 1991, the Mixture of Experts (MoE) was introduced in paper [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), as a system of parallel networks, of the same architecture but initialized differently, receiving the same input, alongside a **gating network** (a feed-forward network, also receiving the same input). The output of the networks are softmax-normalized and weighted by the probabilities output by the selector (gating mechanism) . In practice Mixtures of Experts combine the outputs of several "expert" networks, each of which specializes in a different part of the input space.

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/Mixture-of-Experts/MoE_1991.png" alt="MoE paper 1991"/>

This can also be seen as a probability model, where the final probability over classes is marginalized over the selection of expert. The final output for all experts $$f_i$$ and a gating network $$g$$ is then:

$$
F(x) = \sum_{i=1}^N g_i(x)  softmax(f_i(x))
$$

if each $$f_i$$ maps an output to $$C$$ outputs (class $$c$$), for all experts $$e_i$$ then the previous follows to:

$$
F(x) = \sum_{i=1}^N p(e_i \mid x) \, p(c \mid e_i, x) = p (c \mid x)
$$

The authors "demonstrate that the learning procedure divides up a vowel discrimination task into appropriate subtasks, each of which can be solved by a very simple expert network".

## 2014 [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

Later, the 2014 paper [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314) "extends the Mixture of Experts work (MoE) to a stacked model, the Deep Mixture of Experts, with multiple sets of gating and experts. This exponentially increases the number of effective experts by associating each input with a combination of experts at each layer, yet maintains a modest model size.".

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/Mixture-of-Experts/MoE_2014_Ilya.png" alt="MoE paper 2014"/>

Formally, it extends the [1991 paper Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) with a stack of 2 MoEs ($$g^1$$, $$f_i^1$$) and ($$g^2$$, $$f_i^2$$) alongside a linear layer $$f^3$$, such as:

$$
z^1 = \sum_{i=1}^N g^1_i(x) f^1_i(x) \\
z^2 = \sum_{j=1}^N g^2_j(z^1) f^2_j(z^1) \\

F(x) = z^3 = softmax ( f^3 ( z^2 ) )
$$

In the paper, experts $$f_i^l$$ and $$f_j^l$$ are single-layer DNN and $$g^l_i$$ is a two-layer DNN, with a ReLU activation.  During training, they authors noticed that training leads to a **degenerate local minimum**: "experts at each layer that perform best for the first few examples end up overpowering the remaining experts”, because as they are better initially, then end up being picked more by the gates. This is overcome by **capping (*max clipping*)** the assignments to a maximum value, and re-normalizing across all (post max-clipping) gating probabilities. This **cap value is then removed in a second training stage**.

The benchmark compares a single MoE (Fig 1a), a two-level MoE (Fig 1b), and a two-layer MoE where the information between layers is the concatenation of all layer-1 experts (instead of the weighted sum as in Fig 1b). They also compared it with a deep network with the same number of parameters. The deep MoE model had 4 experts at the first layer and 16 at the second layer. Both layers had
128 hidden units. Each gating networks is a two-layer feed forward networks, with 64 units in the hidden layer. Results shows that, "In most cases, the deeply stacked experts performs
between the single and concatenated experts baselines on the training set, as expected. However, the
deep models often suffer from **overfitting**: the mixture’s error on the test set is worse than that of the
single expert for two of the four model sizes. Encouragingly, the DMoE performs almost as well as
a fully-connected network (DNN) with the same number of parameters, even though this network
imposes fewer constraints on its structure".

In practice, the total error on a Mixture of Experts takes into account the average of all expert errors, so the optimizer will focus on improving the experts that were assigned the most weights, and/or improve the assignments (gating network). An important result multivated the exploration of stacked MoEs:
> we find that the Deep Mixture of Experts automatically learns to develop location-dependent ("where") experts at the first layer, and class-specific ("what") experts at the second layer.

<!-- 
## 2015 [Conditional Computation in Neural Networks for faster models](https://arxiv.org/abs/1511.06297) 

As a follow up, the 2015 paper [Conditional Computation in Neural Networks for faster models](https://arxiv.org/abs/1511.06297) improves the formulation by allowing variable number of experts and layers to be picked, "capturing the idea of wanting to have parsimonious activations while maintaining prediction accuracy". It proposes using policy gradient to train activation-dependent policies for dropping out blocks of experts ("units"). The optimization of the input-dependent activation probabilities at each layer is setup as a  discrete time, continuous state and discrete action Markov Decision Process. Each node or block in a given layer is assigned a Bernoulli distribution, and a action $$u \in \{ 0,1 \}^k$$ (describing whether or not to put a mask over the experts of a given layer). Thus, they define a $$k$$-dimensional Bernoulli policy for every layer $$l$$ of $$k$$ experts. This is practice is an approach similar to a dropout (think Bernoulli distribution) but across experts, however here they $$p$$ parameter is trainable and differs accross experts. -->

## 2017 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

The previous MoE implementations had a big drawback: all experts receive all inputs, leading to high training costs, even for the experts of low relevance for each input. This led to the introduction of **Conditional Computation (aka Sparsity), a method where only a subset of experts are picked**. In practice, the gating mechanism is now moved now moved to the begining of the workflow and works as a **routing mechanism**, as it provides the distribution of the experts relevance for different experts. A new hyper-parameter $$k$$ is introduced, referring to the number of experts picked per iteration. A low or high $$k$$ lead to a situation where too few or too many experts are utilized, resulting in under- or over-computation.

In practice, the MoE selects for each token in the text a possibly different combination of experts.
When $$g_i(x)=0$$ for a given expert $$i$$, then $$f_i(x)$$ is not computed. The simplest gating mechanism - **Softmax Gating** - is formulated as a softmax over an input $$x$$ multiplied by a trainable matrix $$W_g$$, as:

$$
G_σ(x) = Softmax(x · W_g)
$$

Towards scaling this approach,  the 2017 paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), introduces Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands experts (modelled as feed-forward networks) and a gating network. This MoE layer (with up to 137 billion) and is stacked recursively between stacked LSTM layers. Each MoE layer is The model architecture is composed of a stack of LSTMs. **"All parts of the network are trained jointly by back-propagation"**.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2017_Dean.png"/>

The authors don't use the typical softmax gating. Instead, the paper proposes **noisy top-k gating**, where they add **tunable gaussian noise** $$H(X)$$ (controlled by $$W_{noise}$$, to help with load balance, Appendix A) before taking the softmax. To handle sparsity, it the output value of the non-picked experts to -∞ (which causes the corresponding gate values to equal 0)”:

$$
G(x) = softmax( \, KeepTopK ( \, H(x), k) )
$$

where

$$
H(x)_i = (x · W_g)_i + StandardNormal() · Softplus ( (x · W_{noise})_i )
$$

Note: $$SoftPlus$$ is a smooth approximation to ReLU and is used to constrain its output to be always positive. In order to improve scaling, they **mix model and data parallelism** to allow memory- and compute-efficient scaling on expert on distributed systems: each processor has a data-parallel replica of the standard layers and the gating networks, and model-parallel shards of a subset of experts.  Batch is passed in a data-parallel fashion - each device receives a different micro-batch. This reduces the model size by a factor of the number of compute devices.

When applied to a language task, they take advantage of **Convolutionality**: " we apply the same MoE to each time step of the previous layer". As an example: in one single batch run, the expert 1 takes timestep 1 of all inputs, expert 2 takes timestep 2 of all inputs, etc, and this is repeated for each layer in the LSTM stack. "If we wait for the previous layer to finish, we can apply the MoE
to all the time steps together as one big batch. Doing so increases the size of the input batch to the
MoE layer by a factor of the number of unrolled time steps". By applying apply the same MoE to each time step of the previous layer.
 This allows one to apply the MoE to all the time steps together as one big batch., and reduces input the model size by a factor of unrolled time steps.

To avoid having few experts taking all the importance from the gating mechanism at early stages, they adopted a soft constraing approach (as a side note, a hard approach would be fixing a maximum cap threshold). In practice, they add an additional **importance loss**, equal to the square of the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation) of the set of importance values, as:

$$
L_{importance} = w_{importance} \cdot CV \, (Importance (X))^2
$$

where the coefficient of variation ($$CV$$, or relative standard deviation) represents the dispersion of the probability distribution and is defined as the ratio of the standard deviation $$\sigma$$ to the mean $$\mu$$ as $$CV= \frac {\sigma }{\mu }$$. The **Importance** of an expert is computed as the batchwise sum of the gate values for that expert, and $$w_{importance}$$ is a hand-tuned scaling factor. As an example, if you had a batch of 4 samples $$x_i$$ assigned across 5 experts $$E_j$$, with the following distribution:

|-|-|-|-|-|-|
| | $$E_1$$ | $$E_2$$ | $$E_3$$ | $$E_4$$ | $$E_5$$ |
|-|-|-|-|-|-|
| $$x_1$$ | 0.1 | 0.1 | 0 | 0.8 | 0 |
| $$x_2$$ | 0 | 0 | 0.2 | 0.7 | 0.1 |
| $$x_3$$ | 0.1 | 0 | 0 | 0.9 | 0 |
| $$x_4$$ | 0 | 0 | 0 | 1 | 0 |
| **Importance (sum)** | **0.2** | **0.1** | **0.2** | **2.4** | **0.1** |
|-|-|-|-|-|-|

, the mean of the importance is 0.6, and standard deviation is 0.9, thus the CV is 0.667. If experts would be assigned a similar importance, then the variance would've been smaller and the CV also smaller, thus reducing the importance loss. 

This loss tries to balance overall importance across experts, but experts may receive different numbers of examples. This may lead memory and compute imbalance on distributed hardware, and to having experts that are undertrained. To solve this, a second **load loss** $$L_{load}$$ is introduced to encourages experts to receive a similar amount of training samples. However, note that the number of received tokens per expert is a constant and can not be backpropagated, so instead they use a smooth operator $$Load(X)$$ that can be back propagated, as:

$$
L_{load}(X)_i = \sum_{x \in X} P(x,i)
$$

where $$P(x,i) = Pr(h(x)_i) \gt kth\_excluding(H(x), k, i)$$ denotes  the probability that $$G(x)_i$$ is **non-zero,  given a new random choice of noise on element $$i$$, but keeping the already-sampled choices of noise on the other elements** (see Appendix A for details). Note that $$G(x)_i$$ is nonzero if and only if $$H(x)_i$$ is greater than the $$k$$th-greatest element of H(x).

The experiments fixed a compute budgets and compared a baseline model with several configurations of MoE (variable number of experts, with and without hierarchy etc), on several language tasks. Results show MoE models beating baseline LSTM models (less perplexity/confusion), with hierarchical MoE beating non-hierarchichal (with a higher parameter count, but same compute). The perplexity improved significantly by increasing data size (from 10B to 100B Word Google News Corpus). In the machine translation task, on the Google Production dataset, the model achieved 1.01 higher test BLEU score even after training for only one sixth of the time.

An important message in the results is that **different experts specialize on different tasks based on syntax and semantics**, as shown in Appendix E table 9:

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_2017_Appendix_E_table_9.png"/>

Another important message in this paper is that it is suggested that learning to route does not work without the ability to compare at least two experts (due to instability purposes), i.e. we always need to backpropagate on least 2 experts on any MoE setup. In terms of scalability, the results claim to obtain "greater than 1000x improvements in model capacity with only minor losses in computational efficiency".

## 2020 [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

So far we covered deep MoE models based on recursive LSTM and stacks of MoEs. In 2020, GShard explored the scaling of Transformer-based Sparsely-Gated MoEs on a model of 600 billion parameters on a sequence-to-sequence (machine translation) task. They claim that the their methods yield sub-linear computation cost and graph $$O(1)$$ compilation time on 2048 TPU devices. This performance was achieved via:

- **Position-wise Sparsely-Gated MoEs**, just like the previous paper, that scales the capacity of RNN-based machine translation and language models and delivers sub-linear scaling;
- **GShard module that separates model description from the parallel partitioning**. It consists of "of a set of simple APIs for annotations, and a compiler extension in XLA for automatic parallelization." This allows developers to write the module and the runtime will automatically partition it accross compute unitrs;
- **compiler technique for SPMD (Single Program Multiple Data) transformation that generates a single program to run on all devices**, in order to keep the compilation time constant independent of the number of devices. 

The model used for the task presented is a Transformer-based language model, made of stacks of encoder and decoder layers. Each Transformer block (made of self-attention, feed-forward and on the decoder's case, attention mask) was converted into a conditional computation by **replacing the feed-forward layer in every second block with a MoE** layer with a variant of **top-2 gating** in both the encoder and the decoder. The encoder part can be represented as:

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/Mixture-of-Experts/MoE_GShard.png"/>

Strangely enough, the paper does not mention why they introduced MoE in every second block, instead of every third or every single block instead. The parallelism follows the hybrib sharded-MoEs with data-parallel FFN and gating, as in the previous paper. As a side node, the "model-parallel MoE" in the picture refers to sharding and not to tensor/activation parallelism. The underlying formulation of the gating mechanism is the following: a MoE layer for Transformer consists of $$E$$ feed-forward networks $$FFN_1$$ ... $$FFN_E$$, formulated as:

$$
FFN_e(x_s) = wo_e · ReLU(wi_e · x_s)
$$

and

$$
y_x = \sum_{e=1}^{E} G_e(x_s) · FFN_e(x_s)
$$

where $$G_e(x_s)$$ is the probabilities (post-softmax) output by the gating mechanism for expert $$e$$ given input $$x_s$$, and $$y_x$$ is as before the weighted sum of products of the gating probabilities and the FFN outputs (FFN here refers to an expert). $$x_s$$ is the input token to the MoE layer, $$w_i$$ and $$w_o$$ are the input and output projection matrices for the feed-forward layer.

The total loss is described as $$L = l_{nll} + k ∗ l_{aux}$$, where $$l_{nll}$$ is the negative log-likelihood loss, and $$l_{aux}$$ is an auxiliary loss to help avoiding few experts *stealing* most of the attention at the early stages of training. This auxiliary loss is formulated as

$$
l_{aux} = \frac{1}{E} \sum^E_{e=1} \frac{c_e}{S} \, m_e
$$

where the term $$c_e/S$$ represents the fraction of input routed to each expert and $$m_e$$ is the mean gates per expert (note that $$m_e$$ is added because $$c_e/S$$ is a constant and is not differentiable).

**Load balancing** was regulated by guaranteeing that the total number of tokens assigned to each expert is not below a given threshold (this is in contrast to have a threshold in the number of samples). Finally, they introducted the concept of **random routing**, where in a top-2 setup, "if the weight for the 2nd expert is very small, we can simply ignore the 2nd expert to conserve the overall expert capacity". 

The parallel-partitioning is a combination of data-parallel and sharding: (1) "the attention layer is parallelized by splitting along the batch dimension and replicating its weights to all devices"; and (2) "experts in the MoE layer are infeasible to be replicated in all the devices due to its sheer size and the only viable strategy is to shard experts into many devices".

<!--
Finally, they mention **Mixing manual and automatic sharding**, where it allows user to manually specify the sharding by telling the runtime "on how operators are partitioned, and one example is that the user has more run-time knowledge beyond the operators’ semantics", more concretely the "user might know that a specific Gather operator shuffles data only within each partition".
However, they claim that "the automatic sharding assignment is not the focus of this paper and we leave it as future work".
-->

## 2021 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

A big issue with large MoEs are training instability. With that in mind [Switch Transformers](https://arxiv.org/abs/2101.03961) tackle that problem by simplifying the routing algorithm and improving model specifications. There is also a strong motivational quote about the usage of MoEs, particularly to solve low-data problems:

> The benefit of scale was exhaustively studied in [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) which uncovered powerlaw scaling with model size, data set size and computational budget,  [and]
advocates training large models on relatively small amounts of data as the computationally optimal approach.
Heeding these results, we investigate a fourth axis: increase the parameter count while
keeping the floating point operations (FLOPs) per example constant.

In order to achieve scaling, the sparsely activated layers split *unique weights* on different devices. Just like in GShard, the authors replaced the Feed Forward Network with a MoE. However, in Switch transformers, every FFN is replaced (instead of every other), and each Switch Transformer layer received two inputs (tokens) on for experts.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_Switch_Transformers.png"/>

The regular top-$$k$$ routing mechanism used in common use cases was replaced by a **Switch Routing**.  Here, in contradiction to the paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), where the authors claim that you need to compute a loss of at least 2 experts in order to train an MoE successfully, in **Switch Transformers they route to only a single expert**. The claim is that "simplification preserves model quality, reduces routing computation and performs better. This $$k=1$$ routing strategy is later referred to as a Switch layer". This leads to several benefits: reduced batch size as only a single expert is used, simpler routing and reduced communication.

They introduce the concept of **expert capacity factor**, which is the number of tokens that each expert computes, computed as batch size divided by number of experts (as before), times a capacity factor. If this capacity factor is exceeded (ie if the router sends too many inputs to a given expert), then extra tokens do not have computation associeated with them and are instead passed to the next layer via a residual connection.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/MoE_Switch_Transformers_2.png"/>

> Increasing the capacity factor increases the quality but also increases the communication overheard and the memory in the activations. A recommended initial setup is to use an MoE with a top-2 routing with 1.25 capacity factor, with one expert per core. At evaluation time, the capacity factor can be changed to reduce compute.

One of the downsides of MoEs is the large number of parameters. If one wants to run it on a smaller network we can perform distilation: 

> Successful distillation of sparse pre-trained and specialized fine-tuned models into
small dense models. We reduce the model size by up to 99% while preserving 30% of
the quality gains of the large sparse teacher.

### Implementation on PyTorch

```python
def foo():
    pass
```

### Implementation on DeepSpped

```python
def foo():
    pass
```


## 2022 [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)

MegaBlocks is "a system for efficient Mixture-of-Experts (MoE) training on GPUs", that addresses the **model quality vs hardware efficiency tradeoff** on the dynamic routing of MoE layers. In detail, the load-imbalanced computation on MoEs forces one to either (1) drop tokens from the computation or (2) waste computation and memory on padding, a parameter that is usually improved via hyperparameter tuning. However, we know that "The loss reached by the MoE models decreases significantly as expert capacity is increased, but at the cost of additional computation" and "the lowest loss is achieved by the 'max expert capacity value', which avoids dropping tokens through the dynamic
capacity factor mechanism proposed by [Adaptive
mixture-of-experts at scale (2002)](#)". So not dropping tokens is ideal.

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/Mixture-of-Experts/MoE_MegaBlocks.png"/>

As a second motivation, "the sparse computation in MoEs does not map cleanly to the software primitives supported in major frameworks and libraries". 
 To overcome these limitations, MegaBlocks "reformulates MoE computation in terms of block-sparse operations and develop new block-sparse GPU kernels that efficiently handle the dynamism present in MoEs".
In practice, it proposes an approach of MoE routing based on **sparse primitives, that matches efficiently to GPUs and never drops tokens**, " enabling end-to-end training speedups of up to 40% and 2.4× over state of the art". This is achieved by performing block-sparse operations in the MoE layers to accommodate imbalanced assignment of tokens to experts. This is called **dropless-MoEs (dMoEs)** and contrasts with regular MoEs that use batched matrix multiplication instead. These dropless-MoEs kernels use blocked-CSR-COO encoding and transpose indices to enable efficient matrix products with sparse inputs.


## 2022 [Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)

Formally studies "how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model". It experiments on a 4-cluster classification vision task and claims that: (1) "cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE"; (2) there needs to be non-linearity in the experts for an MoE system to work; and (3) "the router can learn the cluster-center features, which helps divide the input complex problem into simpler linear classification sub-problems that individual experts can conquer".

## 2022 [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)

Introduces GLaM (Generalist Language Model), a family of dense and sparse decoder-only language models, "which uses a sparsely activated mixture-of-experts architecture to scale the model capacity while also incurring substantially less training cost compared to dense variants". GLam has 1.2 trillion parameters (7x larger than ChatGPT-3) but requires 1/3 of the energy for training, and half of the computation flops for inference, while "achieving better overall zero, one and few-shot performance across 29 NLP tasks" (Table 1, Figure 1).

Results were collected by training several variants of GLaM to study the behavior of MoE and dense models on the same training data (Table 4). The study highlights the that:
- "the quality of the pretrained data also plays a critical role, specifically, in the performance of downstream tasks" (section 6.2);
- sparse models scale better (section 6.3);
- "MoE models require significantly less data than dense models of comparable FLOPs to achieve similar zero, one, and fewshot performance" (section 6.4).

## 2022 [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)

Presents a study with experiments, results and techniques related to the stability of training and fine-tuning of very large sparse MoEs. Also introduces the **router z-loss**, that resolves instability issues and improves slightly the model quality, and a 269B sparse model -- **the Stable Transferable Mixture-of-Experts** or ST-MoE-32B - that achieves state-of-the-art performance across several NLP benchmarks. The z-loss formulation is:

$$
L_z(x) = \frac{1}{B} \sum_{i=1}^B \left( \log \sum_{j=1}^N e^{x_j^{(i)}} \right)
$$

where $$B$$ is the number of tokens, $$N$$ is the number of experts, and $$x \in \mathbb{R}^{B×N}$$ are the logits going into the router. The rationale behind z-loss is that it penalizes large logits into the gating network.  The paper also contains a very extensive analysis of quality-stability trade-off techniques and results:

- "Sparse models often suffer from training instabilities (Figure 1) worse than those observed in standard densely-activated Transformers. [...] Some architectural improvements involve more multiplications than additions or do not sum many items at once": the paper studies GELU Gated Linear Units (GEGLU) and Root Mean Square Scale Parameters.
- Adding noise (input-jitter) and adding dropout improves stability, but leads to a significant loss of model quality (Section 3.2).
- Gradient clipping in updates (in Adafactor optimizer) improves stability, but at a high loss of quality (Section 3.3).
- Sparse expert models are sensitive to roundoff errors (eg due to lower precision representation) because they have more exponential functions due to the routers (Section 3.4).
- Sparse models are prone to overfit (Fig. 3).
- Adding dropout at 0.1 provides modest boosts while higher rates penalizes performance; and selectively increasing experts dropout increase accuracy (Fig. 4).
- Updating the non MoE parameters (while freezing others) yields an accuracy similar to updating all the parameters; and updating only the FFN parameters works a bit better (Fig 5).
- "Sparse models benefit from noisier hyperparameters including small batch sizes and high learning rates. Dense models behave nearly oppositely" (Fig. 6).
- "Sparse models are robust to dropped tokens when fine-tuning. [...] dropping 10-15% of tokens can perform approximately as well as models that drop < 1%" (Table 5).

The paper provides the following 3 recommendations when designing sparse systems:
> 
- top-2 routing with 1.25 capacity factor and at most one expert per core.
- the capacity factor can be changed during evaluation to adjust to new memory/compute requirements.
- dense layer stacking and a multiplicative bias can boost quality (Appendix C).

Related to the functioning of mixture of experts, the authors showed that **encoder experts learn very shallow tasks**, e.g. ponctuation, nouns, etc. I.e., **MoEs subdivide a problem into smaller problems that can solved by different experts combined**. The authors show this by "visualizing how tokens are routed among (encoder) experts, [...] by passing a batch of tokens to the model and manually inspecting token assignment at each layer". They observed that at each layer, at least one expert specializes in **sentinel tokens** (mask tokens that represent blanks to fill-in); and some encoder experts exhibit clear specialization, with some experts primarily operating on punctuation, verbs, proper names, counting, etc. These are detailed in table 13 and 15.

At the decoder level, "expert specialization is far less noticeable in the decoder", and meaningful specialization (semantics or syntax) are not visible in decoder experts:
  > We hypothesize that this lack of meaningful expert specialization is caused by the distribution of
target tokens induced by the span corruption objective. In particular, (a) a smaller number of tokens
are routed jointly in the decoder due to longer sequence lengths in the encoder (e.g. group size
is 2048 in the encoder vs 456 in the decoder in our setup) and (b) a higher proportion of tokens
are sentinel tokens in the decoder. As a result, target tokens in each group typically cover a smaller
semantic space (compared to the encoder), perhaps explaining the lack of expert specialization in the
decoder

## 2023 [Mixture-of-Experts Meets Instruction Tuning: a Winning Combination for Large Language Models](https://arxiv.org/abs/2305.14705)

This paper claims that **Instruction tuning supplemented by further finetuning on individual downstream tasks outperforms fine-tuning or instruction-tuning alone**. To show that they perform single-task fine-tuning, multi-task instruction-tuning and multi-task instruction-tuning followed by single-task fine-tuning. They showed that MoEs benefit more from instruction tuning than other models and benefit more from a higher number of tasks. In practice:
  > in the absence of instruction tuning, MoE models fall short in performance when compared to dense models on downstream tasks. [...] When supplemented with instruction tuning, MoE models exceed the performance of dense models on downstream tasks, as well as on held-out zero-shot and few-shot tasks.

## 2024 [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

The current state-of-art MoE model is the Mixtral 8x7B, a sparse mixture of expert, with the same architecture as Mixtral 7B except that it supports a fully dense context length of 32k tokens and the feed-forward blocks are replaced by a Mixture of 8 feed-forward network experts. Model architecture is detailed in Table 1. It performs a top-2 routing, and because tokens can be allocated to different experts, "each token has access to 47B parameters, but only uses 13B active parameters during inference". 

The formulation of the model, in section 2.1, matches the GShard architecture (except that the MoE is in every layer instead of every second), with sparse top-$$k$$ MoE with gating $$G(x) = Softmax(TopK(x · Wg))$$ where $$TopK(ℓ)= −∞$$ for the non top-$$k$$ experts. As before, "in a Transformer model, the MoE layer is applied independently per token and replaces the feed-forward (FFN) sub-block of the transformer block". They use the [SwiGLU activation function](https://arxiv.org/abs/2002.05202v1) (same as Llama-2), and therefore for a given input $$x$$ the output is computed as:

$$
y = \sum_{i=0}^{n-1} Softmax(Top2(x · W_g))_i· SwiGLU_i(x).
$$

## 2024 [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

As background, remember that we mentioned before that MoEs work by subdividing the problem domain into smaller problems that are solved individually by different experts. Now think that, not all problems require the same amount of effort to train a model accurately. This is explored in this work, via **Conditional computation**, a technique that "tries to reduce total compute by expending it only when needed". Here, "the network must learn how to dynamically allocate the available compute by making decisions per-token, in each layer, about where to spend compute from the availablebudget" - where this compute budget is a total number of FLOPs pre-defined beforehand by the user and remains unchanged throught the execution.

In this work, for every token, the model decides to either apply a computation (as in the standard transformer), or skip the computation and passing it through a residual connection (remaining unchanged and saving compute). Moreover, contrarily to MoE, the **routing is applied to both the feedforward network and the multi-head attention**. In the multi-head routing, the router will not only decide on which tokens to update, but which tockens are made available to attend to. "We refer to this strategy as Mixture-of-Depths (MoD) to emphasize how individual tokens pass through different numbers of layers, or blocks, through the depth of the transformer". In practice: (1) the user picks a fixed compute budget beforehand; (2) during training, for every input, the router produces a scalar weight (importance) per token; and (3) we pick the top-$$k$$ tokens per sentence per block to participate in the transformer block computation. $$k$$ is a hyper-parameter that defined the max number of tokens passed to a block, thus the computation graph and tensor sizes remain static throughout the execution. As $$k$$ is set to be smaller than the sentence length, MoD allows one to trade off between performance and speed, while achieving a high level of accuracy for a given compute budget.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/Mixture_of_Depths.png"/>

The big challenge is the routing scheme. The authors considered (1) token-choice where  a router produces per-token probability distributions across computational paths (experts); and the orthogonal approach (2) expert-choice where instead of having tokens choose the expert they prefer, each expert instead picks the top-$$k$$ tokens based on the tokens’ preferences. They ultimately adopted the expert-choise because it does not require load balancing, and "routers can try to ensure that the most critical tokens are among the
top-$$k$$ by setting their weight appropriately".

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/Mixture-of-Experts/Mixture_of_Depths_2.png"/>

At the time of writing of this post, this is still very recent work, so future will tell if MoDs become useful for the general use case.

