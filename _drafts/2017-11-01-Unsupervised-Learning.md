---
layout: post
title:  "Unsupervised Learning: an overview of methods"
date:   2017-11-01 12:01:42 +0100
categories: [machine learning, unsupervised learning]
tags: [machilelearning]
---

<div class="alert alert-primary" role="alert">
  This post provides a very high level description of unsupervised methods I came across, and is continuously updated.
</div>

### Hebbian Learning

Among several *biologically-inspired* learning models, Hebbian learning is probably the oldest. Hebbian learning as a general concept forms the basis for many learning algorithms, including backpropagation. Its rationale is based on the synaptic changes for neurons that spike in close instants in time (motto "*if they fire together, they wire together*"). It is knows from the Spike-Timing-Dependent Plasticity ([STDP, Markram et al. Frontiers](https://www.frontiersin.org/articles/10.3389/fnsyn.2012.00002/full)) that the increase/decrease of a given synaptic strenght is correlated with the time difference between a pre- and a post-synaptic neuron spike times. In essence, if a neuron spikes and if it leads to the firing of the output neuron, the synapse is strengthned.

In his own words:
<blockquote class="blockquote text-right">
  <p class="mb-0">When an axon of cell j repeatedly or persistently
takes part in firing cell i,<br/> then js efficiency as one
of the cells firing i is increased.</p>
  <footer class="blockquote-footer">Hebb, 1949</footer>
</blockquote>

Analogously, in artifical system, the weight of a connection between two artificial neurons adapts accordingly. Mathematically, we can describe the change of the weight $w$ at every iteration with the general Hebbian rule as:

$$
\frac{d}{dt}w_{ij} = \eta x_i y_j
$$

where $\eta$ is the learning rate, $y$ is the output, and $i$ and $j$ are the neuron ids. Hebbian is a **local** learning model i.e. only looks at the two involved neurons and does not take into account the activity in the overall system.  Hebbian is considered **Unsupervised Learning**, or learning that is acquired only with the existing dataset without any external stimuli or classifier of accuracy. Among others, a common formalization of the Hebbian rule is the [Oja's learning rule](http://www.scholarpedia.org/article/Oja_learning_rule):

$$
\frac{d}{dt}w_{ij} = \eta (x_i y_j - y_j^2 w_i).
$$ 

The squared output $y^2$ guarantees that the larger the output of the neuron becomes, the stronger is this balancing effect.
Another common formulation is the covarience rule

$$
\frac{d}{dt}w_{ij} = \eta (x_i - <x_i>) ( y_j - <y_j>) 
$$

assuming that $x$ and $y$ fluctuate around the mean values $ < x > $ and $ < y > $. Or the Bienenstock-Cooper-Munroe rule:

$$

$$

Typically the output of a neuron is a function of a linear multiplication of the weights and the input, as:

$$
y_j \propto g(\sum_i w_{ij} x_i)
$$

where $g$ is the pre-synaptic signal function, commonly referred to as the [activation function](https://en.wikipedia.org/wiki/Activation_function). In practice, we can say that $\Delta w \propto F(pre,post)$. When we add a reward signal to each operation, i.e. $\Delta w \propto F(pre, post, reward)$ then we enter the field of **reinforcement learning**. One could say that while unsupervised learning relates to the synaptic weight changes in the brain derived from [Long-Term Potentiation (LTP)](https://en.wikipedia.org/wiki/Long-term_potentiation) or [Long-Term Depression (LTD)](https://en.wikipedia.org/wiki/Long-term_depression), the reinforcement learning relates more to the Dopamine [reward system](https://en.wikipedia.org/wiki/Reward_system) in the brain, as it's an external factor that provides reward for learning for behaviour. 

### Principal Component Analysis

The Principal Component Analysis is a dimensionality (feature) reduction method that rotates data towards the direction that yields most data significance (variance), in such way that one can analyse data at the most meaningful dimensions and discards dimensions of low variance. Theoretically speaking, a quote from Gerstner's lecture notes in Unsupervised Learning:

<blockquote class="blockquote text-right">
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
</blockquote>

The algorithm is the following:
- center data by substracting mean: $x \leftarrow x - < x >$
- calculate [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix): $ C_{kj} = < ( x_k - < x_k > ) (x_j - < x_j >) >$
- calculate [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of the covariance matrix:
  - compute $det(C - \lambda I)$, where $\lambda_n$ is one eigenvalue;
  - compute the respective eigenvector for each eigenvalue by solving $C e_n = \lambda_n e_n$;
- compute the feature vector $F$, composed by the eigenvectors in the order of the largest to smallest corresponding eigenvalues.
- the final data is obtained by multiplying the feature vector transposed ($F^T$, i.e. with the most significant eigenvector is on top) by a matrix $D_c$ whose rows are the mean centered data $x -< x > $ and $y -< y > $ as: $D_n = F^T D_c$

A visual example is displayed below, where the output of the PCA rotates the data towards the new axis in the cluster of the datapoints:

<p align="center">
<img width="30%" height="30%" src="/assets/2017-Unsupervised-Learning/GaussianScatterPCA.svg.png"><br/><small>source: wikipedia</small>
</p>

We can now discard the least significant dimensions of the final post-PCA data, reducing the number of features in the model.

<b>Small trick:</b> to retrieve the original data, we simply need to invert the final transformation, i.e.:

$$
D_c = F^{-T} D_n = F D_n
$$

<div class="alert alert-warning" role="alert">
An interesting related topic is the Independent Component Analysis (ICA), a technique that allows one to separate a multidimensional signal into additive subcomponents, under the assumption that (1) the input signals are independent and (2) the values of signals do not follow a Gaussian distributions. I will add information about this in the future.
</div>
