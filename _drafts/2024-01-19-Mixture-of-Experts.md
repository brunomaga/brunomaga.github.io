---
layout: post
title:  "Mixture-of-Experts: a historical overview, with implementations"
categories: [machine learning, Transformer, GPT, DeepSpeed, inference, mixture-of-experts]
tags: [machinelearning]
---

# 2017 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
# 2014 [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)
# 1991 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

Covered in a [different post]({{ site.baseurl }}{% post_url 2024-01-19-Mixture-of-Experts %}) )


# 1991 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)

In 1991, the Mixture of Experts (MoE) was introducesd in paper [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), as a system of parallel networks, of the same architecture but initialized differently, receiving the same input, alongside a gating network (a feed-forward network, also receiving the same input). The output of the networks are softmax-normalized and weighted by the probabilities output by the selector (gating mechanism) . In practice Mixtures of Experts combine the outputs of several "expert" networks, each of which specializes in a different part of the input space. This can also be seen as a probability model, where the final probability over classes is marginalized over the selection of expert. The final output for all experts $$f_i$$ and a gating network $$g$$ is then:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/publications/MoE_1991.png"/>

$$
F(x) = \sum_{i=1}^N g_i(x)  softmax(f_i(x))
$$

if each $$f_i$$ maps an output to $$C$$ outputs (class $$c$$), for all experts $$e_i$$ then the previous follows to:

$$
F(x) = \sum_{i=1}^N p(e_i \mid x) \, p(c \mid e_i, x) = p (c \mid x)
$$

# 2014 [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314)

Later, the 2014 paper [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314) "extends the Mixture of Experts work (MoE) to a stacked model, the Deep Mixture of Experts, with multiple sets of gating and experts. This exponentially increases the number of effective experts by associating each input with a combination of experts at each layer, yet maintains a modest model size." 
Formally, extends the [1991 paper Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) with 2 sets of experts ($$g^1$$, $$f_i^1$$) and ($$g^2$$, $$f_i^2$$) alongside a linear layer $f^3$$, such as:

$$
z^1 = \sum_{i=1}^N g^1_i(x) f^1_i(x) \\
z^2 = \sum_{j=1}^N g^2_j(z^1) f^2_j(z^1) \\

F(x) = z^3 = softmax ( f^3 ( z^2 ) )
$$

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="/assets/publications/MoE_2014_Ilya.png"/>

In the paper, experts $$f_i^l$$ and $$f_j^l$$ are single-layer DNN and $$g^l_i$$ is a two-layer DNN, with rectification.  During training, they authors noticed that results in a **degenerate local minimum**: "experts at each layer that perform best for the first few examples end up overpowering the remaining experts”, because as they are better initially, then end up being picked more by the gates. This is overcome by capping (*max clipping*) the assignments to a maximum value, and re-normalizing across all (post max-clipping) gating probabilities. This **cap value is then removed in a second training stage**. 

They tested:
the effect of using a mixture at the second layer by comparing against using only a single fixed expert at the second layer, or concatenating the output of all experts (ie N · h hidden units). It’s expected for the concatenated model to perform better than the mixture, and the mixture to perform better than the single network.
they compared a two-layer MoE model against a one-layer MoE model
they compared also against a fullyconnected deep network with the same total number of parameters.
The authors tested the approach on the MNIST dataset and on a dataset of monophone speech samples. On the MNIST dataset, In most cases, the deeply stacked experts performs between the single and concatenated experts baselines on the training set. It also almost matches the concatenated version. However it suffers from overfitting, and the test set accuracy is lower. On the monophone speech dataset, the results are simillar, however it does not overfit on the test set and performance is simillar. 

Note: in brief, Mixture of Experts work because the total error takes into account the average of all expert errors, so the optimizer will try to fix/improve each expert.
 
# 2017 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

Towards scaling this approach,  the 2017 paper [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538),  introduces Sparsely-Gated Mixture-of-Experts layer (MoE). Sparsity uses the idea of **conditional computation**, where parts of the network are active on a per-example basis - ie instead of picking all experts, it picks only a few. I.e. when $$g_i(x)=0$$, then $$f_i(x)$$ is not computed.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/publications/MoE_2017_Dean.png"/>

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
L_{importance} = w_{importance} \dot CV \, (Importance (X))^2
$$ 

where the $$Importance$$ of an expert to be the batchwise sum of the gate values for that expert. This loss ensures equal importance, but experts may receive different numbers of examples.

The experiments fixed a compute budgets and compared a baseline model with several configurations of MoE (variable number of experts, with and without hierarchy etc), on several language tasks. Results show MoE models beating baseline LSTM models (less perplexity/confusion), with hierarchical MoE beating non-hierarchichal (with a higher parameter count, but same compute). Perplexit improved significantly by increasing data size (from 10B to 100B Word Google News Corpus). In the machine translation task, "Google Production dataset, our model achieved 1.01 higher test BLEU score even after training for only one sixth of the time."

