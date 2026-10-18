---
layout: post
title:  "Variational Inference (ELBO, CAVI) and Gaussian Mixture Models"
categories: ["Machine Learning"]
tags: ["variational inference", "ELBO", "GMM", "MCMC"]
---

Bayesian inference computes the posterior of the latent variables $$z$$ given the observations $$x$$ with [Bayes' rule]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#bayes-rule):

$$
p(z \mid x) = \frac{p(z) \, p(x \mid z)}{p(x)}, \qquad p(x) = \int p(z, x) \, dz
$$

The **evidence** $$p(x)$$ integrates the joint density over all values of $$z$$. In the [Bayesian linear regression post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}), a Gaussian prior and a Gaussian likelihood give the posterior and the evidence in closed form. For many models with latent variables, the integral has no closed form or takes exponential time to compute, as in the mixture model below. Exact inference algorithms for graphical models, such as [variable elimination](https://en.wikipedia.org/wiki/Variable_elimination) and the [junction tree algorithm](https://en.wikipedia.org/wiki/Junction_tree_algorithm), take time exponential in the [treewidth](https://en.wikipedia.org/wiki/Treewidth) of the graph. Approximate inference uses loopy [belief propagation](https://en.wikipedia.org/wiki/Belief_propagation), sampling with Markov chain Monte Carlo (MCMC), or optimization with variational inference (VI). The sections below follow the review [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf) (Blei et al., 2017).

## Markov Chain Monte Carlo

[Monte Carlo methods](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimate a quantity from random samples. For example, points drawn uniformly in a square fall inside its inscribed circle with probability $$\pi / 4$$, so four times the fraction of points inside the circle estimates $$\pi$$. A [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) is a sequence of random states in which the distribution of the next state depends only on the current state.

[MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) builds a Markov chain whose stationary distribution is the posterior. After a burn-in period, the states of the chain are samples of $$p(z \mid x)$$. The [Metropolis algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) draws a proposal $$z'$$ from a symmetric distribution around the current state $$z$$, e.g. a Gaussian centered at $$z$$, and accepts it with probability $$\min(1, r)$$, where

$$
r = \frac{p(z' \mid x)}{p(z \mid x)} = \frac{p(x \mid z') \, p(z') / p(x)}{p(x \mid z) \, p(z) / p(x)} = \frac{p(x \mid z') \, p(z')}{p(x \mid z) \, p(z)}
$$

Otherwise, the chain stays at $$z$$. The evidence cancels in the ratio, so the algorithm needs only the prior and the likelihood. A proposal with a higher posterior density is always accepted, and one with a lower density is accepted with probability $$r$$, so the chain explores the whole posterior instead of climbing to its mode. [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling), another MCMC method, draws each latent variable in turn from its distribution given the data and the other latent variables.

MCMC and VI compare as follows:
- MCMC gives asymptotically exact samples of the posterior. It needs many iterations, each with a pass over the data to evaluate the likelihood, and consecutive samples are correlated.
- VI turns inference into an optimization problem. It is faster and scales to large datasets with stochastic optimization, but it finds only the closest member of a chosen family of distributions, and it generally underestimates the variance of the posterior.

VI thus suits large datasets and the fast comparison of many models, and MCMC suits smaller datasets, where precise samples justify the cost. [An Introduction to MCMC for Machine Learning]({{ site.assets }}/variational-inference-gmm/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf) (Andrieu et al., 2003) covers the theory of MCMC, and [Thomas Wiecki's post](https://twiecki.io/blog/2015/11/10/mcmc-sampling/) an implementation.

## Variational Inference

[Variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) chooses a family $$\mathcal{Q}$$ of distributions over the latent variables and finds its member closest to the posterior in [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

$$
q^*(z) = \underset{q \in \mathcal{Q}}{\arg\min} \; \mathrm{KL}\left( q(z) \Vert p(z \mid x) \right), \qquad \mathrm{KL}\left( q(z) \Vert p(z \mid x) \right) = \mathbb{E}_q\left[ \log \frac{q(z)}{p(z \mid x)} \right]
$$

where $$\mathbb{E}_q$$ is the expectation over $$z \sim q$$. The family must be simple enough for the expectations under $$q$$ to be computable, and rich enough to approximate the posterior. The KL divergence is not symmetric: $$\mathrm{KL}(q \Vert p)$$ penalizes $$q$$ where the posterior is small, so $$q$$ tends to fit one mode of the posterior and to underestimate its variance (see the [statistics post]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#cross-entropy-and-kl-divergence)). Other [f-divergences](https://en.wikipedia.org/wiki/F-divergence) give other variational objectives.

## Evidence Lower Bound

The KL divergence contains the posterior, whose evidence is unknown, so it cannot be minimized directly. With $$p(z \mid x) = p(z, x) / p(x)$$, it splits into two terms:

$$
\mathrm{KL}\left( q(z) \Vert p(z \mid x) \right) = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z, x)] + \log p(x) = \log p(x) - \mathrm{ELBO}(q)
$$

where the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) is

$$
\mathrm{ELBO}(q) = \mathbb{E}_q[\log p(z, x)] - \mathbb{E}_q[\log q(z)]
$$

- $$\log p(x)$$ does not depend on $$q$$, so maximizing the ELBO minimizes the KL divergence.
- The ELBO needs only the joint density $$p(z, x) = p(z) \, p(x \mid z)$$, the prior times the likelihood.
- The KL divergence is non-negative, so $$\mathrm{ELBO}(q) \le \log p(x)$$: the ELBO is a lower bound of the log evidence, with a gap equal to the KL divergence.

## Mean-Field Variational Family

The **mean-field** family makes the latent variables $$z = (z_1, \dots, z_m)$$ independent, each with its own factor:

$$
q(z) = \prod_{j=1}^m q_j(z_j)
$$

The name comes from [mean-field theory](https://en.wikipedia.org/wiki/Mean-field_theory) in physics, which replaces the effect of all other particles on one particle by a single averaged effect. Each factor can take any parametric form that suits its variable, e.g. a Gaussian for a continuous variable and a categorical distribution for a discrete one. The data $$x$$ does not appear in $$q$$: the ELBO connects $$q$$ to the data. The family cannot represent correlations between latent variables, so it usually does not contain the posterior. **Generalized mean field** factorizes $$q$$ over disjoint groups of latent variables and keeps the correlations within each group. Structured variational families, with dependencies between variables, and mixtures of variational densities approximate the posterior better, at the cost of a harder optimization (Blei et al., section 2.3).

## Coordinate Ascent Variational Inference

Coordinate ascent variational inference (CAVI) maximizes the ELBO over one factor $$q_j$$ at a time, with the other factors fixed, as in [coordinate ascent](https://en.wikipedia.org/wiki/Coordinate_descent). Let $$z_{-j}$$ be the latent variables other than $$z_j$$, and $$\mathbb{E}_{-j}$$ the expectation over $$\prod_{l \ne j} q_l$$. As a function of $$q_j$$, the ELBO is

$$
\mathrm{ELBO}(q_j) = \mathbb{E}_{q_j}\left[ \mathbb{E}_{-j}[\log p(z_j, z_{-j}, x)] \right] - \mathbb{E}_{q_j}[\log q_j(z_j)] + \text{const} = -\mathrm{KL}\left( q_j \Vert \tilde{q}_j \right) + \text{const}
$$

where $$\tilde{q}_j(z_j) \propto \exp\left( \mathbb{E}_{-j}[\log p(z_j, z_{-j}, x)] \right)$$ is a normalized density, and the constants do not depend on $$q_j$$. The KL divergence is zero for $$q_j = \tilde{q}_j$$, so the optimal factor is

$$
q_j^*(z_j) \propto \exp\left( \mathbb{E}_{-j}[\log p(z_j, z_{-j}, x)] \right) \propto \exp\left( \mathbb{E}_{-j}[\log p(z_j \mid z_{-j}, x)] \right)
$$

The two forms are equal up to a constant, since $$p(z_j, z_{-j}, x) = p(z_j \mid z_{-j}, x) \, p(z_{-j}, x)$$ and the last factor does not depend on $$z_j$$. The density $$p(z_j \mid z_{-j}, x)$$ is the **complete conditional** of $$z_j$$: Gibbs sampling draws $$z_j$$ from it, and CAVI sets $$q_j$$ to the normalized exponential of its expected logarithm.

CAVI cycles through the factors and applies this update to each, until the ELBO stops increasing. No update decreases the ELBO, so CAVI converges, but the ELBO is not concave in general: CAVI reaches a local optimum that depends on the initialization. Section 10.1.1 of [Bishop's PRML]({{ site.resources_permalink }}) derives the update in the same way, and the [CMU 10-708 lecture notes]({{ site.assets }}/variational-inference-gmm/10708-scribe-lecture13.pdf) derive it with the chain rule of probability.

When the complete conditional is in the [exponential family]({{ site.baseurl }}{% post_url 2019-03-20-exponential-family %}), with natural parameter $$\eta_j(z_{-j}, x)$$ and sufficient statistic $$T(z_j)$$, the optimal factor is in the same family, with the expected natural parameter:

$$
q_j^*(z_j) \propto h(z_j) \exp\left( \mathbb{E}_{-j}[\eta_j(z_{-j}, x)]^T \, T(z_j) \right)
$$

Each update then only computes an expectation, as in the mixture model below.

## Gaussian Mixture Model

The example follows sections 2.1, 3.1 and 3.2 of Blei et al. A [mixture model](https://en.wikipedia.org/wiki/Mixture_model) assumes that each observation comes from one of $$K$$ components. The Bayesian mixture of $$K$$ univariate Gaussians with unit variance has two kinds of latent variables: the means $$\mu = (\mu_1, \dots, \mu_K)$$ of the components, and the assignments $$c_1, \dots, c_n$$ of the $$n$$ observations to the components. Each $$c_i$$ is a one-hot [indicator vector](https://en.wikipedia.org/wiki/Indicator_vector) of length $$K$$, so $$c_i^T \mu$$ is the mean of the component of $$x_i$$. The variance $$\sigma^2$$ of the prior is a hyperparameter:

$$
\begin{align*}
\mu_k & \sim \mathcal{N}(0, \sigma^2), & k & = 1, \dots, K \\
c_i & \sim \text{Categorical}(1/K, \dots, 1/K), & i & = 1, \dots, n \\
x_i \mid c_i, \mu & \sim \mathcal{N}(c_i^T \mu, 1), & i & = 1, \dots, n
\end{align*}
$$

The joint density is $$p(\mu, c, x) = p(\mu) \prod_{i=1}^n p(c_i) \, p(x_i \mid c_i, \mu)$$, and the evidence sums over all assignments $$c = (c_1, \dots, c_n)$$:

$$
p(x) = \sum_c p(c) \int p(\mu) \prod_{i=1}^n p(x_i \mid c_i, \mu) \, d\mu
$$

The Gaussian prior is conjugate to the Gaussian likelihood, so each integral has a closed form, but there are $$K^n$$ assignments: the evidence costs $$O(K^n)$$ operations.

The mean-field family has a Gaussian factor for each mean and a categorical factor for each assignment:

$$
q(\mu, c) = \prod_{k=1}^K q(\mu_k; m_k, s_k^2) \prod_{i=1}^n q(c_i; \varphi_i)
$$

The factor $$q(\mu_k; m_k, s_k^2)$$ has mean $$m_k$$ and variance $$s_k^2$$, and $$\varphi_{ik}$$ is the probability that $$x_i$$ belongs to component $$k$$. With all expectations under $$q$$, the ELBO is

$$
\begin{align*}
\mathrm{ELBO}(m, s^2, \varphi) = & \sum_{k=1}^K \mathbb{E}\left[ \log p(\mu_k) \right] + \sum_{i=1}^n \left( \mathbb{E}\left[ \log p(c_i) \right] + \mathbb{E}\left[ \log p(x_i \mid c_i, \mu) \right] \right) \\
& - \sum_{i=1}^n \mathbb{E}\left[ \log q(c_i; \varphi_i) \right] - \sum_{k=1}^K \mathbb{E}\left[ \log q(\mu_k; m_k, s_k^2) \right]
\end{align*}
$$

### Update of the Assignments

The CAVI update of $$c_i$$ keeps the terms of $$\log p(\mu, c, x)$$ that depend on $$c_i$$:

$$
q^*(c_i; \varphi_i) \propto \exp\left( \log p(c_i) + \mathbb{E}\left[ \log p(x_i \mid c_i, \mu) \right] \right)
$$

The prior term $$\log p(c_i) = -\log K$$ is the same for all assignments. Since $$c_i$$ is one-hot, $$p(x_i \mid c_i, \mu) = \prod_{k=1}^K \mathcal{N}(x_i \mid \mu_k, 1)^{c_{ik}}$$, and

$$
\mathbb{E}\left[ \log p(x_i \mid c_i, \mu) \right] = \sum_{k=1}^K c_{ik} \, \mathbb{E}\left[ -\frac{(x_i - \mu_k)^2}{2} \right] + \text{const} = \sum_{k=1}^K c_{ik} \left( \mathbb{E}[\mu_k] \, x_i - \frac{\mathbb{E}[\mu_k^2]}{2} \right) + \text{const}
$$

The constants hold the terms that are the same for all assignments, such as $$-x_i^2 / 2$$, since $$\sum_k c_{ik} = 1$$. With $$\mathbb{E}[\mu_k] = m_k$$ and $$\mathbb{E}[\mu_k^2] = m_k^2 + s_k^2$$, the update is

$$
\varphi_{ik} \propto \exp\left( m_k x_i - \frac{m_k^2 + s_k^2}{2} \right)
$$

normalized so that $$\sum_k \varphi_{ik} = 1$$.

### Update of the Component Means

The CAVI update of $$\mu_k$$ keeps the prior of $$\mu_k$$ and the likelihood terms of component $$k$$, with $$\mathbb{E}[c_{ik}] = \varphi_{ik}$$:

$$
\begin{align*}
\log q^*(\mu_k) & = \log p(\mu_k) + \sum_{i=1}^n \varphi_{ik} \log \mathcal{N}(x_i \mid \mu_k, 1) + \text{const} \\
& = \left( \sum_{i=1}^n \varphi_{ik} x_i \right) \mu_k - \left( \frac{1}{2 \sigma^2} + \frac{1}{2} \sum_{i=1}^n \varphi_{ik} \right) \mu_k^2 + \text{const}
\end{align*}
$$

This is an exponential family density with sufficient statistics $$T(\mu_k) = [\mu_k, \mu_k^2]^T$$ and natural parameters $$\eta_1 = \sum_i \varphi_{ik} x_i$$ and $$\eta_2 = -\frac{1}{2 \sigma^2} - \frac{1}{2} \sum_i \varphi_{ik}$$: a Gaussian. Its mean $$-\eta_1 / (2 \eta_2)$$ and its variance $$-1 / (2 \eta_2)$$, derived in the [exponential family post]({{ site.baseurl }}{% post_url 2019-03-20-exponential-family %}#examples), give the update

$$
m_k = \frac{\sum_i \varphi_{ik} x_i}{1 / \sigma^2 + \sum_i \varphi_{ik}}, \qquad s_k^2 = \frac{1}{1 / \sigma^2 + \sum_i \varphi_{ik}}
$$

The mean $$m_k$$ is the average of the observations weighted by their assignment probabilities, shrunk toward the prior mean 0. The variance $$s_k^2$$ decreases as the expected number of observations in the component, $$\sum_i \varphi_{ik}$$, grows. The two updates resemble the two steps of the [expectation–maximization (EM) algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) for mixtures, soft assignments and weighted means, but CAVI also keeps the uncertainty of each mean.

### Algorithm

CAVI for the mixture repeats three steps until the ELBO converges:

1. for $$i = 1, \dots, n$$ and $$k = 1, \dots, K$$, set $$\varphi_{ik} \propto \exp\left( m_k x_i - (m_k^2 + s_k^2) / 2 \right)$$, normalized over $$k$$;
2. for $$k = 1, \dots, K$$, set $$m_k \leftarrow \frac{\sum_i \varphi_{ik} x_i}{1 / \sigma^2 + \sum_i \varphi_{ik}}$$ and $$s_k^2 \leftarrow \frac{1}{1 / \sigma^2 + \sum_i \varphi_{ik}}$$;
3. compute $$\mathrm{ELBO}(m, s^2, \varphi)$$.

Each iteration costs $$O(nK)$$ operations. The update of $$m$$ and $$s^2$$ needs the data only through the $$2K$$ sums $$\sum_i \varphi_{ik} x_i$$ and $$\sum_i \varphi_{ik}$$. When the observations are split across workers, each worker computes $$\varphi_i$$ and the partial sums for its observations, and one [all-reduce](https://en.wikipedia.org/wiki/Collective_operation#All-Reduce) of the $$2K$$ sums per iteration gives every worker the new $$m$$ and $$s^2$$. In the figure, the gray ellipses are the variational densities of the components, from the initialization to iteration 50:

{: .picture}
<img width="59.8%" height="59.8%" src="{{ site.assets }}/variational-inference-gmm/gmm_iterations.svg"/>

{: .caption}
The CAVI algorithm on a Gaussian mixture model, for a dataset of 2D points from five color-coded clusters: the initialization and the variational factors after 20, 28, 35 and 50 iterations. The ellipses are 2σ contours of the variational approximating factors. The figure uses the multivariate (2D) version of the model; the derivation above uses univariate Gaussians. Adapted from: [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)

## Stochastic Variational Inference

CAVI passes over the whole dataset in every iteration, which is slow for large datasets. [Stochastic variational inference](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) (SVI, Hoffman et al., 2013) instead updates the global variational parameters after each mini-batch of observations. It uses stochastic optimization ([Robbins and Monro, 1951](https://projecteuclid.org/euclid.aoms/1177729586)): it follows noisy but unbiased estimates of the natural gradient of the ELBO, with step sizes $$\rho_t$$ that satisfy $$\sum_t \rho_t = \infty$$ and $$\sum_t \rho_t^2 < \infty$$.

For the mixture, the global parameters are the natural parameters $$\lambda_k$$ of the factors $$q(\mu_k)$$, which CAVI sets to $$\left( \sum_i \varphi_{ik} x_i, \; -\frac{1}{2 \sigma^2} - \frac{1}{2} \sum_i \varphi_{ik} \right)$$. At step $$t$$, SVI draws a mini-batch $$B$$ of observations, computes $$\varphi_i$$ for $$i \in B$$ with step 1 of CAVI, and computes $$\hat{\lambda}_k$$ with the CAVI formula, in which each sum $$\sum_i$$ is replaced by $$\frac{n}{\vert B \vert} \sum_{i \in B}$$, as if the mini-batch were repeated to the size of the dataset. It then updates

$$
\lambda_k \leftarrow (1 - \rho_t) \, \lambda_k + \rho_t \, \hat{\lambda}_k
$$

This update is a step along the noisy natural gradient $$\hat{\lambda}_k - \lambda_k$$. It costs $$O(\vert B \vert K)$$ operations instead of $$O(nK)$$, and needs only the mini-batch in memory. SVI thus suits large and streaming datasets, while CAVI needs no step sizes. Section 4.3 of Blei et al. gives the general algorithm. Probabilistic programming libraries such as [Pyro](https://pyro.ai/examples/svi_part_i.html) extend SVI to models without conjugate complete conditionals, with Monte Carlo estimates of the ELBO gradient, as in the [variational autoencoder]({{ site.baseurl }}{% post_url 2019-10-12-vae-gan %}).
