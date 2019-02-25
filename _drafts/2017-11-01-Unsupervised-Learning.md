---
layout: post
title:  "Unsupervised Learning: an overview of methods"
date:   2017-11-01 12:01:42 +0100
categories: [machine learning, unsupervised learning]
tags: [machilelearning]
---

<div class="alert alert-primary" role="alert">
  This post provides a very high level description of unsupervised learning methods that I come across, and is continuously updated.
</div>

### Foreword: Hebbian Learning

Among several *biologically-inspired* learning models, Hebbian learning ([Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb),1949) is probably the oldest. Its general concept forms the basis for many learning algorithms, including backpropagation. The main rationale is based on the synaptic changes for neurons that spike in close instants in time (motto "*if they fire together, they wire together*"). It is known from the Spike-Timing-Dependent Plasticity ([STDP, Markram et al. Frontiers](https://www.frontiersin.org/articles/10.3389/fnsyn.2012.00002/full)) that the increase/decrease of a given synaptic strenght is correlated with the time difference between a pre- and a post-synaptic neuron spike times. In essence, if a neuron spikes and if it leads to the firing of the output neuron, the synapse is strengthned.

In his own words:
<blockquote class="blockquote text-right">
  <p class="mb-0">When an axon of cell j repeatedly or persistently
takes part in firing cell i,<br/> then js efficiency as one
of the cells firing i is increased.</p>
  <footer class="blockquote-footer">Hebb, 1949</footer>
</blockquote>

Analogously, in artifical system, the weight of a connection between two artificial neurons adapts accordingly. Mathematically, we can describe the change of the weight $w$ at every iteration with the general Hebb's rule based on the activity $x$ of neurons $i$ and $j$ as:

$$
\frac{d}{dt}w_{ij} = F (w_{ij}; x_i, x_j)
$$

or the simplest Hebbian learning rule as:

$$
\frac{d}{dt}w_{ij} = \eta x_i y_j
$$

where $\eta$ is the learning rate, $y$ is the output function of the neuron $x$, and $i$ and $j$ are the neuron ids. Hebbian is a **local** learning model i.e. only looks at the two involved neurons and does not take into account the activity in the overall system.  Hebbian is considered **Unsupervised Learning**, or learning that is acquired only with the existing dataset without any external stimuli or classifier of accuracy. Among others, a common formalization of the Hebbian rule is the [Oja's learning rule](http://www.scholarpedia.org/article/Oja_learning_rule):


$$
\frac{d}{dt}w_{ij} = \eta (x_i y_j - y_j^2 w_i).
$$ 

The squared output $y^2$ guarantees that the larger the output of the neuron becomes, the stronger is this balancing effect.
Another common formulation is the covarience rule

$$
\frac{d}{dt}w_{ij} = \eta (x_i - < x_i >) ( y_j - < y_j >) 
$$

assuming that $x$ and $y$ fluctuate around the mean values $ < x > $ and $ < y > $. Or the Bienenstock-Cooper-Munroe (BCM, 1982) rule:

$$
\frac{d}{dt}w_{ij} = \eta y_j x_i (x_i - < x_i > )
$$

a physical theory of learning in the visual cortex developed in 1981, proposing a sliding threshold for Long-term potentiation (LTP) depression (LTD) induction, and stating that synaptic plasticity is stabilized by a dynamic adaptation of the time-averaged postsynaptic activity.

Typically the output of a neuron is a function of a linear multiplication of the weights and the input, as:

$$
y \propto g(\sum_i w_{ij} x_i)
$$

where $g$ is the pre-synaptic signal function, commonly referred to as the [activation function](https://en.wikipedia.org/wiki/Activation_function). On the simplest use case of a **linear neuron**

$$
y = \sum_i w_i x_i
$$

Typically used are the **sigmoidal functions** (click [here](https://en.wikipedia.org/wiki/Sigmoid_function#Examples) for a list of examples), which are more *powerful* ways of describing an output, compared to a linear function. Generally speaking, we can say that in unsupervised learning, $\Delta w \propto F(pre,post)$. When we add a reward signal to each operation, i.e. $\Delta w \propto F(pre, post, reward)$ then we enter the field of **reinforcement learning**. While unsupervised learning relates to the synaptic weight changes in the brain derived from [Long-Term Potentiation (LTP)](https://en.wikipedia.org/wiki/Long-term_potentiation) or [Long-Term Depression (LTD)](https://en.wikipedia.org/wiki/Long-term_depression), the reinforcement learning relates more to the Dopamine [reward system](https://en.wikipedia.org/wiki/Reward_system) in the brain, as it's an external factor that provides reward for learning for behaviour. 

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

### Independent Component Analysis

<small>This section was largely inspired by the lecture notes of Tatsuya Yokota, Assistant Professor at Nagoya Institute of Technology</small>

An interesting related topic is the Independent Component Analysis (ICA), a method for [Blind Source Separation](https://en.wikipedia.org/wiki/Blind_signal_separation), a technique that allows one to separate a multidimensional signal into additive subcomponents, under the assumption that:
- the input signals $s_1$, ..., $s_n$ are statistically independent i.e. $$p(s_1, s_2, ..., s_n) = p(s_1)\text{ }p(s_2) \text{ } ... \text{ } p(s_n)$$ 
- the values of signals do not follow a Gaussian distributions, otherwise ICA is impossible. 

The ICA is based on the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) that states that the distribution of the sum of $N$ random variables approaches Gaussian as $N \rightarrow \infty$. As an example, the throw of a dice has an uniform distribution of 1 to 6, but the distribution of the sum of throws is not uniform --- it will be better approximated by a Gaussian as the number of throws increase.  *Non-Gaussianity is Independence*. 

A famous example is the [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect) problem. A guest will be listening to an observed signal $x$ from an original signal $s$, as in:

$$
x_1(t) = a_{11}s_1(t) + a_{12}s_2(t) + a_{13}s_3(t)
x_2(t) = a_{21}s_1(t) + a_{22}s_2(t) + a_{23}s_3(t)
x_3(t) = a_{31}s_1(t) + a_{32}s_2(t) + a_{33}s_3(t)
$$

We assume that $s_1$, $s_2$ and $s_3$ are statistically independent. The goal of the ICA is to estimate the independent components $s(t)$ from $x(t)$, with

$$
x(t) = As(t)
$$

$A$ is a regular matrix. Thus, we can write the problem as $s(t) = Bx(t)$, where $B = A^{-1}$. It is only necessary to estimate $B$ so that ${ s_i }$ are independent.

White signals are defined as $z$ such that it satisfies $E [z] = 0$, and $E [ z z^T ] = I$. Whitening is useful for the preprocessing of the ICA. The whitening of the signals $x(t)$ are given by $z (t) = V x(t)$, where $V$ is the whitening matrix. Model becomes:

$$
s(t) = U z (t) = U V x (t) = B x (t)
$$

where $U$ is an orthogonal transform matrix.  **Whitening simplifies the ICA**, so it is only necessary to estimate $U$. A visual illustration of the source, observerd and whitened data is diplayed below:

<p align="center">
<img width="20%" height="20%" src="/assets/2017-Unsupervised-Learning/ica_source.png">
<img width="20%" height="20%" src="/assets/2017-Unsupervised-Learning/ica_observed.png"> 
<img width="20%" height="20%" src="/assets/2017-Unsupervised-Learning/ica_whitening.png">
<br/><small>source: T. Yokota, Lecture notes, Nagoya Institute of Technology</small>
</p>

** Non-Guassianity is a measure of independence**. Kurtosis is a measure of non-Gaussianity, defined by:

$$
k(y) =  E [y^4] - 3 (E[y^2])^2.
$$

We assume that $y$ is white, i.e. $E[y]=0$ and $E[y^2]=1$, thus

$$
k(y) = E[y^4]-3.
$$

We solve the ICA problem with

$$
\hat{b} = max_b | k(b^T x(t)) |
$$ 

We aim to maximize the absolute value of the *kurtosis* , i.e. maximize $| k(w^Tz) |$ such that $w^T w=1$. The differential is:
<p align="center">
<img width="50%" height="50%" src="/assets/2017-Unsupervised-Learning/ica_kurtosis.png">
</p>

Now we can solve via Gradient descent:
- $w \leftarrow w + \Delta w$;
- with $w \propto sign [ k(w^T z)] [ E { z(w^T z)^3 } - 3w ] $. 

Or... because the algorithm converges when $w \propto \Delta w$, and  $w$ and $-w$ are equivalent, by the **FastICA** as:
- $w \leftarrow E { z ( w^T z)^3 } - 3w$

As a final remark, it is relevant to mention that *kurtosis is very weak with outliers because is a fourth order function*. An alternative often used method is the Neg-entropy, robust for outliers. I'll ommit it for brevity. If you're interested on details, check the [original lecture notes](https://www.slideshare.net/yokotatsuya/independent-component-analysis-11359849).
