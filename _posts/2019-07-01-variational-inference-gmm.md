---
layout: post
title:  "Variational Inference (ELBO, CAVI) and Gaussian Mixture Models"
categories: ["Machine Learning"]
tags: ["variational inference", "ELBO", "GMM", "MCMC"]
---

The [Bayesian linear regression post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}) showed how Bayesian inference turns prior knowledge into a posterior distribution once data are observed. The same idea applies to any model with **latent variables** $$z$$, the hidden quantities that explain the observations $$x$$. Their **posterior** follows from [Bayes' rule]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#bayes-rule), written here with and without its normalization term:

$$
p(z \mid x) = \frac{p(z) \, p(x \mid z)}{p(x)} \propto p(z) \, p(x \mid z)
$$

which reads as *the posterior is proportional to the prior times the likelihood*. Bayes' rule itself comes from the definition of the [conditional probability](https://en.wikipedia.org/wiki/Conditional_probability#Definition), $$p(z \mid x) = p(z, x) / p(x)$$. Its denominator is the marginal density of the observations, called the **evidence**, and it is computed by **marginalizing** the latent variables out of the joint density:

$$
p(x) = \int p(z, x) \, dz
$$

The inference problem is to compute the posterior $$p(z \mid x)$$, i.e. to condition on the data, and the evidence is what makes it hard. With a Gaussian prior and a Gaussian likelihood, as in Bayesian linear regression, the integral has a closed form. For many other models, however, it has no closed form, or it takes exponential time to compute, as the mixture model below shows.

There are several approaches to this problem. Exact inference either enumerates every value of the latent variables, by brute force, or exploits the structure of a graphical model, with [variable elimination](https://en.wikipedia.org/wiki/Variable_elimination), message passing (the sum-product algorithm, also called [belief propagation](https://en.wikipedia.org/wiki/Belief_propagation)) or the [junction tree algorithm](https://en.wikipedia.org/wiki/Junction_tree_algorithm). In general, though, the cost of these algorithms grows exponentially with the [treewidth](https://en.wikipedia.org/wiki/Treewidth) of the graph. When the posterior is **intractable**, approximate inference takes over. Its main approaches are loopy belief propagation, which runs message passing on graphs with cycles, sampling with Markov chain Monte Carlo (MCMC), and optimization with variational inference (VI). The sections below compare the last two, and then derive the simplest form of variational inference, based on the mean-field approximation, following the review [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf) (Blei et al., 2017).

## Markov Chain Monte Carlo

Before moving on to variational inference, it helps to know the method that dominated Bayesian inference for many years, [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC). Its name already says how it works.

[Monte Carlo methods](https://en.wikipedia.org/wiki/Monte_Carlo_method) estimate a quantity from random numbers. A classic example estimates the value of $$\pi$$: points drawn uniformly in a square fall inside its inscribed circle with probability $$\pi / 4$$, the ratio of the two areas, so four times the fraction of points that fall inside the circle approaches $$\pi$$ as the number of points grows. A [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) is a sequence of random states in which the distribution of the next state depends *only* on the current state, and not on the states before it. An example is a model of text that gives the probability of the next character, among 26 letters and the space, from the current character alone.

MCMC combines the two ideas. It builds a Markov chain whose states are values of the latent variables, for example candidate pairs of the mean and the variance of a normal distribution, and whose stationary distribution is the posterior. After a *burn-in* period, the states of the chain are samples of $$p(z \mid x)$$. The key point is that the chain never needs the evidence. The numerator of Bayes' rule, the prior times the likelihood, can be computed for any value of $$z$$, but the denominator is an intractable integral:

$$
p(z \mid x) = \frac{p(z) \, p(x \mid z)}{\int p(z) \, p(x \mid z) \, dz}
$$

The denominator is only a normalizer, which makes the posterior integrate to 1, and it cancels out when two states are compared through a ratio. The [Metropolis algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) relies on this property. It draws a proposal $$z'$$ from a symmetric distribution around the current state $$z$$, for example a Gaussian centered at $$z$$, and computes the ratio of their posteriors:

$$
r = \frac{p(z' \mid x)}{p(z \mid x)} = \frac{p(x \mid z') \, p(z') / p(x)}{p(x \mid z) \, p(z) / p(x)} = \frac{p(x \mid z') \, p(z')}{p(x \mid z) \, p(z)}
$$

It then accepts the proposal with probability $$\min(1, r)$$, and otherwise the chain stays at $$z$$. A proposal that explains the data better, with a higher posterior density, is always accepted. A proposal with a lower density is still accepted with probability $$r$$, and this is what makes the chain explore the whole posterior instead of climbing to its mode. [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling), another MCMC method, draws each latent variable in turn from its distribution given the data and the other latent variables.

The main advantage of MCMC is that it approximates the *true* posterior: its samples are asymptotically exact, i.e. their distribution converges to the posterior as the number of samples grows. The price is computing power, because MCMC needs many iterations, each with a pass over the data to evaluate the likelihood, and consecutive samples are correlated. Variational inference addresses this cost by turning inference into an optimization problem. It is faster and scales to large datasets with stochastic optimization, but it only finds the closest member of a chosen family of distributions, and it generally underestimates the variance of the posterior, i.e. it is sometimes *over-confident*. In practice, VI suits large datasets and the fast comparison of many models, while MCMC suits smaller datasets, where precise samples justify the cost. The paper [An Introduction to MCMC for Machine Learning]({{ site.assets }}/variational-inference-gmm/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf) (Andrieu et al., 2003) covers the theory of MCMC, and [Thomas Wiecki's post](https://twiecki.io/blog/2015/11/10/mcmc-sampling/) gives a practical explanation with code.

## Variational Inference

The idea behind [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) is to approximate the intractable posterior $$p(z \mid x)$$ with a simpler, parametric distribution $$q(z)$$. VI chooses a family $$\mathcal{Q}$$ of such distributions and looks for its member that is closest to the posterior. The family must be simple enough for the expectations under $$q$$ to be computable, and rich enough to approximate the posterior well. A common measure of how close two distributions are is the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), which for $$q$$ and the posterior is:

$$
\mathrm{KL}\left( q(z) \Vert p(z \mid x) \right) = \int q(z) \log \frac{q(z)}{p(z \mid x)} \, dz = \mathbb{E}_q\left[ \log \frac{q(z)}{p(z \mid x)} \right]
$$

where $$\mathbb{E}_q$$ is the expectation over $$z \sim q$$. Variational inference thus solves the optimization problem:

$$
q^*(z) = \underset{q \in \mathcal{Q}}{\arg\min} \; \mathrm{KL}\left( q(z) \Vert p(z \mid x) \right)
$$

The KL divergence is zero only when the two distributions are equal, but it is a dissimilarity function, not a metric, because it is not symmetric: the order of its arguments matters. $$\mathrm{KL}(q \Vert p)$$ averages over $$q$$, so it penalizes $$q$$ for placing probability where the posterior is small, but hardly penalizes it for missing parts of the posterior. As a result, $$q$$ tends to fit one mode of the posterior and to underestimate its variance, which explains the over-confidence of VI. The [statistics post]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#cross-entropy-and-kl-divergence) compares the two directions of the KL divergence, and other members of the family of [f-divergences](https://en.wikipedia.org/wiki/F-divergence) give other variational objectives.

There is one problem, however. The KL divergence contains the posterior $$p(z \mid x)$$, which is exactly the unknown quantity, so it cannot be computed, let alone minimized. Instead, variational inference maximizes a function that equals the negative KL divergence up to a constant, the evidence lower bound.

## Evidence Lower Bound

The way around the unknown posterior is to rewrite the KL divergence with the definition of the conditional probability, $$p(z \mid x) = p(z, x) / p(x)$$, and to separate the term that holds the evidence:

$$
\begin{align*}
& \mathrm{KL}\left( q(z) \Vert p(z \mid x) \right) \\
& \quad = \int q(z) \log \frac{q(z)}{p(z \mid x)} \, dz \\
& \quad = \int q(z) \log \frac{q(z) \, p(x)}{p(z, x)} \, dz && \text{(conditional probability)} \\
& \quad = \int q(z) \left( \log p(x) + \log \frac{q(z)}{p(z, x)} \right) dz && \text{(logarithm of a product)} \\
& \quad = \log p(x) \int q(z) \, dz + \int q(z) \log \frac{q(z)}{p(z, x)} \, dz && \text{(splitting the integral)} \\
& \quad = \log p(x) - \int q(z) \log \frac{p(z, x)}{q(z)} \, dz && \text{(} q \text{ integrates to 1; inverted fraction)} \\
& \quad = \log p(x) - \mathrm{ELBO}(q)
\end{align*}
$$

The integral that the last steps subtract from the log evidence is the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO), commonly written as the difference of two expectations:

$$
\mathrm{ELBO}(q) = \int q(z) \log \frac{p(z, x)}{q(z)} \, dz = \mathbb{E}_q[\log p(z, x)] - \mathbb{E}_q[\log q(z)]
$$

This replaces the KL divergence, which cannot be computed, with an expression that can be optimized, and three observations show why it solves the problem:
- The log evidence $$\log p(x)$$ does not depend on $$q$$, so maximizing the ELBO over $$q$$ minimizes the KL divergence.
- The ELBO needs only the joint density $$p(z, x) = p(z) \, p(x \mid z)$$, the prior times the likelihood, which are the inputs of the problem.
- The KL divergence is never negative, so $$\mathrm{ELBO}(q) \le \log p(x)$$: the ELBO is a lower bound of the log evidence, hence its name, and the gap between the two is exactly the KL divergence.

Splitting the joint density into the likelihood and the prior gives a third form, which shows what the ELBO rewards:

$$
\mathrm{ELBO}(q) = \mathbb{E}_q[\log p(x \mid z)] - \mathrm{KL}\left( q(z) \Vert p(z) \right)
$$

The first term, an expected log-likelihood, favors distributions that place their probability on values of $$z$$ that explain the data, and the second term favors distributions close to the prior. The ELBO thus balances the likelihood and the prior, as Bayes' rule does (Blei et al., section 2.2).

## Mean-Field Variational Family

A simple way to obtain a family that is easy to optimize over is to assume that some latent variables are independent of the others. The strongest form of this assumption makes all latent variables $$z = (z_1, \dots, z_m)$$ mutually independent, each with its own factor $$q_j$$:

$$
q(z) = \prod_{j=1}^m q_j(z_j)
$$

This is the **mean-field** family. Its name comes from [mean-field theory](https://en.wikipedia.org/wiki/Mean-field_theory) in physics, which studies a large system of interacting particles through a simpler model, in which the effect of all other particles on any given particle is replaced by a single averaged effect. A many-body problem thus becomes a one-body problem. In the same way, each factor $$q_j$$ sees the other latent variables only through averages, as the next section shows.

Each factor can take any parametric form that suits its variable, for example a Gaussian for a continuous variable and a categorical distribution for a discrete one. The data $$x$$ does not appear in $$q$$: the family is not a model of the data, and it is the ELBO that connects $$q$$ to the data.

The independence assumption has a price. The mean-field family cannot represent correlations between latent variables, so it usually does not contain the true posterior, and the best it can offer is the closest independent approximation. A middle ground, called **generalized mean field**, as opposed to the *naive* mean field above, splits the latent variables into disjoint groups and keeps one factor per group, so that the correlations within each group remain. Structured variational families, with explicit dependencies between variables, and mixtures of variational densities approximate the posterior even better, at the cost of a harder optimization (Blei et al., section 2.3).

## Coordinate Ascent Variational Inference

With the mean-field family, the goal is to choose the factors $$q_1, \dots, q_m$$ that maximize the ELBO. A common approach is **coordinate ascent variational inference** (CAVI), which, like [coordinate ascent](https://en.wikipedia.org/wiki/Coordinate_descent) for a function of several variables, optimizes one factor $$q_j$$ at a time while it holds the others fixed. The update of each factor turns out to have a simple closed form.

To derive it, let $$z_{-j}$$ denote all latent variables except $$z_j$$, and $$\mathbb{E}_{-j}$$ the expectation over them, under $$\prod_{l \ne j} q_l$$. The ELBO depends on $$q_j$$ through both of its terms. In the first one, the expectation over all factors can be taken in two steps, first over the other factors and then over $$q_j$$. The second one, the expectation of $$\log q(z) = \sum_l \log q_l(z_l)$$, splits into one term per factor, and only one of these terms depends on $$q_j$$. As a function of $$q_j$$, the ELBO is therefore:

$$
\mathrm{ELBO}(q_j) = \mathbb{E}_{q_j}\left[ \mathbb{E}_{-j}[\log p(z_j, z_{-j}, x)] \right] - \mathbb{E}_{q_j}[\log q_j(z_j)] + \text{const}
$$

where the constant holds the terms of the other factors. The inner expectation is a function of $$z_j$$ only, and its exponential, once normalized, defines a new density over $$z_j$$:

$$
\tilde{q}_j(z_j) = \frac{1}{Z_j} \exp\left( \mathbb{E}_{-j}[\log p(z_j, z_{-j}, x)] \right)
$$

where the normalization constant $$Z_j$$ does not depend on $$q_j$$. Replacing the inner expectation by $$\log \tilde{q}_j(z_j) + \log Z_j$$ turns the ELBO into a familiar form:

$$
\mathrm{ELBO}(q_j) = \mathbb{E}_{q_j}[\log \tilde{q}_j(z_j)] - \mathbb{E}_{q_j}[\log q_j(z_j)] + \text{const} = -\mathrm{KL}\left( q_j \Vert \tilde{q}_j \right) + \text{const}
$$

The KL divergence is never negative, and it is zero for $$q_j = \tilde{q}_j$$, which is therefore where the ELBO is largest. The optimal factor is:

$$
q_j^*(z_j) \propto \exp\left( \mathbb{E}_{-j}[\log p(z_j, z_{-j}, x)] \right) \propto \exp\left( \mathbb{E}_{-j}[\log p(z_j \mid z_{-j}, x)] \right)
$$

The two forms are equal up to a constant, because $$p(z_j, z_{-j}, x) = p(z_j \mid z_{-j}, x) \, p(z_{-j}, x)$$ and the last factor does not depend on $$z_j$$. The density $$p(z_j \mid z_{-j}, x)$$ is the **complete conditional** of $$z_j$$, and the update reveals a close link with Gibbs sampling: Gibbs sampling draws $$z_j$$ from its complete conditional, while CAVI sets $$q_j$$ to the normalized exponential of its expected logarithm. Section 10.1.1 of Bishop's [*Pattern Recognition and Machine Learning*]({{ site.resources_permalink }}) derives the update in the same way, and the [CMU 10-708 lecture notes]({{ site.assets }}/variational-inference-gmm/10708-scribe-lecture13.pdf) derive it with the [chain rule of probability](https://en.wikipedia.org/wiki/Chain_rule_(probability)).

CAVI cycles through the factors and applies this update to each one, until the ELBO stops increasing. Every update maximizes the ELBO over one factor, so no update can decrease it, and CAVI converges. The ELBO is not concave in general, however, so CAVI only reaches a local optimum, which depends on the initialization.

The update above is a template: the final form of the optimal factor depends on the model, and it decides how complex each update is. An important case is a complete conditional in the [exponential family]({{ site.baseurl }}{% post_url 2019-03-20-exponential-family %}), with base measure $$h(z_j)$$, natural parameter $$\eta_j(z_{-j}, x)$$, sufficient statistics $$T(z_j)$$ and log-partition function $$A$$:

$$
\log p(z_j \mid z_{-j}, x) = \log h(z_j) + \eta_j(z_{-j}, x)^T \, T(z_j) - A\left( \eta_j(z_{-j}, x) \right)
$$

The other variables enter only through the natural parameter, and the last term does not depend on $$z_j$$. The expectation in the CAVI update therefore only replaces the natural parameter with its expected value, and the optimal factor is in the same family:

$$
q_j^*(z_j) \propto h(z_j) \exp\left( \mathbb{E}_{-j}[\eta_j(z_{-j}, x)]^T \, T(z_j) \right)
$$

Each update then only has to compute an expectation, as the mixture model below shows.

## Gaussian Mixture Model

A [mixture model](https://en.wikipedia.org/wiki/Mixture_model) assumes that each observation comes from one of $$K$$ components, and a Gaussian mixture model assumes that each component is a Gaussian distribution with unknown parameters. Fitting it to a dataset means finding the Gaussian densities that best explain the data, together with the component that each data point comes from. The example below explains sections 2.1, 3.1 and 3.2 of [Blei et al.](https://arxiv.org/pdf/1601.00670.pdf) in more detail.

Consider a Bayesian mixture of $$K$$ univariate Gaussians with unit variance and means $$\mu = (\mu_1, \dots, \mu_K)$$. The means are drawn from a common Gaussian prior $$\mathcal{N}(0, \sigma^2)$$, whose variance $$\sigma^2$$ is a hyperparameter. Each of the $$n$$ observations $$x_i$$ is assigned to a component by $$c_i$$, an [indicator vector](https://en.wikipedia.org/wiki/Indicator_vector) of length $$K$$: all zeros, except for a one at the index of its component. The product $$c_i^T \mu$$ thus picks the mean of the component of $$x_i$$, and the model is:

$$
\begin{align*}
\mu_k & \sim \mathcal{N}(0, \sigma^2), & k & = 1, \dots, K \\
c_i & \sim \text{Categorical}(1/K, \dots, 1/K), & i & = 1, \dots, n \\
x_i \mid c_i, \mu & \sim \mathcal{N}(c_i^T \mu, 1), & i & = 1, \dots, n
\end{align*}
$$

The latent variables are $$z = \{\mu, c\}$$: the $$K$$ means of the components and the $$n$$ assignments $$c = (c_1, \dots, c_n)$$. The chain rule of probability, together with the independence of the pairs $$(c_i, x_i)$$ given the means, gives the joint density of the latent and observed variables:

$$
p(\mu, c, x) = p(\mu) \, p(c, x \mid \mu) = p(\mu) \prod_{i=1}^n p(c_i) \, p(x_i \mid c_i, \mu)
$$

The evidence marginalizes out both kinds of latent variables. The sum over the assignment of each observation can be placed inside the product, or the sum over all assignments $$c$$ can be placed outside the integral:

$$
\begin{align*}
p(x) & = \int p(\mu) \prod_{i=1}^n \sum_{c_i} p(c_i) \, p(x_i \mid c_i, \mu) \, d\mu \\
& = \sum_c p(c) \int p(\mu) \prod_{i=1}^n p(x_i \mid c_i, \mu) \, d\mu
\end{align*}
$$

The second form shows why the evidence is intractable. The Gaussian prior is [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior) to the Gaussian likelihood, so each integral has a closed form, but there is one integral for every configuration of the assignments, and there are $$K^n$$ of them: the evidence costs $$O(K^n)$$ operations. With only 3 components and 100 observations, this is already about $$5 \times 10^{47}$$ integrals.

The mean-field family avoids this sum. It has a Gaussian factor for each mean and a categorical factor for each assignment:

$$
q(\mu, c) = \prod_{k=1}^K q(\mu_k; m_k, s_k^2) \prod_{i=1}^n q(c_i; \varphi_i)
$$

The factors hold two kinds of variational parameters. The Gaussian factor $$q(\mu_k; m_k, s_k^2)$$ approximates the posterior of the mean of component $$k$$, with mean $$m_k$$ and variance $$s_k^2$$. The categorical factor $$q(c_i; \varphi_i)$$ approximates the posterior assignment of observation $$i$$: its parameter $$\varphi_i$$ is a vector of $$K$$ probabilities, in which $$\varphi_{ik}$$ is the probability that $$x_i$$ belongs to component $$k$$. Combining the joint density and the variational family gives the ELBO of the mixture, in which all expectations are under $$q$$:

$$
\begin{align*}
\mathrm{ELBO}(m, s^2, \varphi) = & \sum_{k=1}^K \mathbb{E}\left[ \log p(\mu_k) \right] + \sum_{i=1}^n \left( \mathbb{E}\left[ \log p(c_i) \right] + \mathbb{E}\left[ \log p(x_i \mid c_i, \mu) \right] \right) \\
& - \sum_{i=1}^n \mathbb{E}\left[ \log q(c_i; \varphi_i) \right] - \sum_{k=1}^K \mathbb{E}\left[ \log q(\mu_k; m_k, s_k^2) \right]
\end{align*}
$$

CAVI updates the two kinds of parameters in turn, so it needs two updates.

### Update of the Assignments

The CAVI update of the assignment $$c_i$$ keeps the terms of $$\log p(\mu, c, x)$$ that depend on $$c_i$$, its prior and the likelihood of $$x_i$$, and takes their expectation over the means:

$$
q^*(c_i; \varphi_i) \propto \exp\left( \log p(c_i) + \mathbb{E}\left[ \log p(x_i \mid c_i, \mu) \right] \right)
$$

The first term is the log prior of $$c_i$$, $$\log p(c_i) = \log \frac{1}{K} = -\log K$$, which is the same for all assignments. The second term is the expected log-likelihood of $$x_i$$. Because $$c_i$$ is an indicator vector, with $$c_{ik} \in \{0, 1\}$$, the likelihood can be written as a product in which every factor equals 1, except the factor of the component of $$x_i$$:

$$
p(x_i \mid c_i, \mu) = \prod_{k=1}^K \mathcal{N}(x_i \mid \mu_k, 1)^{c_{ik}}
$$

Its logarithm is a sum, and the expectation expands step by step:

$$
\begin{align*}
\mathbb{E}\left[ \log p(x_i \mid c_i, \mu) \right]
& = \sum_{k=1}^K c_{ik} \, \mathbb{E}\left[ \log \mathcal{N}(x_i \mid \mu_k, 1) \right] \\
& = \sum_{k=1}^K c_{ik} \, \mathbb{E}\left[ -\frac{1}{2} \log(2 \pi) - \frac{(x_i - \mu_k)^2}{2} \right] && \text{(Gaussian density)} \\
& = \sum_{k=1}^K c_{ik} \, \mathbb{E}\left[ -\frac{x_i^2}{2} + \mu_k x_i - \frac{\mu_k^2}{2} \right] + \text{const} && \text{(expanding the square)} \\
& = \sum_{k=1}^K c_{ik} \left( \mathbb{E}[\mu_k] \, x_i - \frac{\mathbb{E}[\mu_k^2]}{2} \right) + \text{const} && \text{(linearity of the expectation)}
\end{align*}
$$

The last two steps move the terms that are the same for all assignments, $$-\frac{1}{2} \log(2 \pi)$$ and then $$-x_i^2 / 2$$, into the constant. This is allowed because exactly one $$c_{ik}$$ equals 1, so that $$\sum_k c_{ik} \, a = a$$ for any term $$a$$ that does not depend on $$k$$. The result needs only $$\mathbb{E}[\mu_k] = m_k$$ and $$\mathbb{E}[\mu_k^2] = m_k^2 + s_k^2$$, the first two moments of the Gaussian factor of component $$k$$. For the assignment to component $$k$$, the sum keeps only its $$k$$-th term, so the update of the assignment probabilities is:

$$
\varphi_{ik} \propto \exp\left( m_k x_i - \frac{m_k^2 + s_k^2}{2} \right)
$$

normalized so that $$\sum_k \varphi_{ik} = 1$$. Completing the square, the exponent equals $$-(x_i - m_k)^2 / 2 - s_k^2 / 2$$, plus a term $$x_i^2 / 2$$ that is the same for all components. An observation is thus more likely to belong to a component whose mean is close to it, and less likely when that mean is uncertain.

### Update of the Component Means

The CAVI update of the mean $$\mu_k$$ keeps its prior and the likelihood of the observations, and takes their expectation over the assignments and the other means. Since $$c_{ik}$$ is either 0 or 1, its expected value is the probability that it equals 1, $$\mathbb{E}[c_{ik}] = \varphi_{ik}$$. Only the factors of component $$k$$ in the likelihood depend on $$\mu_k$$, so the logarithm of the optimal factor develops as follows:

$$
\begin{align*}
& \log q^*(\mu_k) \\
& \quad = \log p(\mu_k) + \sum_{i=1}^n \mathbb{E}\left[ \log p(x_i \mid c_i, \mu) \right] + \text{const} \\
& \quad = \log p(\mu_k) + \sum_{i=1}^n \mathbb{E}[c_{ik}] \log \mathcal{N}(x_i \mid \mu_k, 1) + \text{const} && \text{(terms of component } k \text{)} \\
& \quad = -\frac{\mu_k^2}{2 \sigma^2} + \sum_{i=1}^n \varphi_{ik} \left( -\frac{(x_i - \mu_k)^2}{2} \right) + \text{const} && \text{(Gaussian densities)} \\
& \quad = -\frac{\mu_k^2}{2 \sigma^2} + \sum_{i=1}^n \varphi_{ik} \left( x_i \mu_k - \frac{\mu_k^2}{2} \right) + \text{const} && \text{(expanding the square)} \\
& \quad = \left( \sum_{i=1}^n \varphi_{ik} x_i \right) \mu_k - \left( \frac{1}{2 \sigma^2} + \frac{1}{2} \sum_{i=1}^n \varphi_{ik} \right) \mu_k^2 + \text{const} && \text{(powers of } \mu_k \text{)}
\end{align*}
$$

The result is the logarithm of an exponential family density with sufficient statistics $$T(\mu_k) = [\mu_k, \mu_k^2]^T$$ and natural parameters

$$
\eta_1 = \sum_{i=1}^n \varphi_{ik} x_i, \qquad \eta_2 = -\frac{1}{2 \sigma^2} - \frac{1}{2} \sum_{i=1}^n \varphi_{ik}
$$

i.e. a Gaussian. Its mean and variance follow from the shape of the logarithm of a Gaussian, $$-(\mu_k - m_k)^2 / (2 s_k^2) + \text{const}$$, whose derivative is zero at the mean $$m_k$$ and whose second derivative is $$-1 / s_k^2$$. Applying both to $$\log q^*(\mu_k)$$ gives the update:

$$
\begin{align*}
& \sum_{i=1}^n \varphi_{ik} x_i - \left( \frac{1}{\sigma^2} + \sum_{i=1}^n \varphi_{ik} \right) m_k = 0 && \Leftrightarrow \quad m_k = \frac{\sum_i \varphi_{ik} x_i}{1 / \sigma^2 + \sum_i \varphi_{ik}} \\
& -\left( \frac{1}{\sigma^2} + \sum_{i=1}^n \varphi_{ik} \right) = -\frac{1}{s_k^2} && \Leftrightarrow \quad s_k^2 = \frac{1}{1 / \sigma^2 + \sum_i \varphi_{ik}}
\end{align*}
$$

in agreement with the mean $$-\eta_1 / (2 \eta_2)$$ and the variance $$-1 / (2 \eta_2)$$ of a Gaussian in its natural parameters, derived in the [exponential family post]({{ site.baseurl }}{% post_url 2019-03-20-exponential-family %}#examples). The mean $$m_k$$ is the average of the observations, weighted by their probabilities of belonging to component $$k$$, and shrunk toward the prior mean 0: the term $$1 / \sigma^2$$ in the denominator acts like extra observations at 0. The variance $$s_k^2$$ decreases as the expected number of observations in the component, $$\sum_i \varphi_{ik}$$, grows, so the more points a component explains, the more certain its mean becomes. The two updates resemble the two steps of the [expectation–maximization (EM) algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) for mixtures, soft assignments and weighted means, but CAVI also keeps the uncertainty of each mean.

### Algorithm

Putting it all together, CAVI for the mixture repeats three steps until the ELBO converges:

1. for $$i = 1, \dots, n$$ and $$k = 1, \dots, K$$, set $$\varphi_{ik} \propto \exp\left( m_k x_i - (m_k^2 + s_k^2) / 2 \right)$$, normalized over $$k$$;
2. for $$k = 1, \dots, K$$, set $$m_k \leftarrow \frac{\sum_i \varphi_{ik} x_i}{1 / \sigma^2 + \sum_i \varphi_{ik}}$$ and $$s_k^2 \leftarrow \frac{1}{1 / \sigma^2 + \sum_i \varphi_{ik}}$$;
3. compute $$\mathrm{ELBO}(m, s^2, \varphi)$$ to check the convergence.

Each iteration costs $$O(nK)$$ operations, linear in the size of the dataset. The update of $$m$$ and $$s^2$$ needs the data only through the $$2K$$ sums $$\sum_i \varphi_{ik} x_i$$ and $$\sum_i \varphi_{ik}$$, which makes the algorithm easy to parallelize. When the observations are split across workers, each worker computes $$\varphi_i$$ and the partial sums for its own observations, and one [all-reduce](https://en.wikipedia.org/wiki/Collective_operation#All-Reduce) of the $$2K$$ sums per iteration gives every worker the new $$m$$ and $$s^2$$.

{: .picture}
<img width="59.8%" height="59.8%" src="{{ site.assets }}/variational-inference-gmm/gmm_iterations.svg"/>

{: .caption}
The CAVI algorithm on a Gaussian mixture model, for a dataset of 2D points from five color-coded clusters: the initialization and the variational factors after 20, 28, 35 and 50 iterations. The ellipses are 2σ contours of the variational approximating factors. The figure uses the multivariate (2D) version of the model; the derivation above uses univariate Gaussians. Adapted from: [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)

The figure shows the algorithm at work on 2D points from five clusters, with a mixture of five components, each one drawn as a gray ellipse. At initialization, the components are wide Gaussians, nearly centered at the origin, which overlap and cover the whole dataset. After 20 iterations, they have split into two groups, one over the three upper clusters and one over the two lower clusters. The next iterations separate them further, and after 50 iterations each component covers one cluster.

## Stochastic Variational Inference

CAVI has one weakness: every iteration analyzes every data point before it can re-estimate the hidden structure, and this full pass over the data is slow for large datasets. [Stochastic variational inference](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) (SVI, Hoffman et al., 2013) avoids it with stochastic optimization ([Robbins and Monro, 1951](https://projecteuclid.org/euclid.aoms/1177729586)), which follows noisy but unbiased estimates of the gradient of an objective, as stochastic gradient descent does. SVI alternates between drawing a mini-batch of the data and updating the global variational parameters from that mini-batch alone, along a noisy estimate of the natural gradient of the ELBO. Its step sizes $$\rho_t$$ decrease over time and satisfy the conditions of Robbins and Monro, $$\sum_t \rho_t = \infty$$ and $$\sum_t \rho_t^2 < \infty$$.

For the mixture, the global parameters are the natural parameters $$\lambda_k$$ of the factors $$q(\mu_k)$$, which CAVI sets to $$\left( \sum_i \varphi_{ik} x_i, \; -\frac{1}{2 \sigma^2} - \frac{1}{2} \sum_i \varphi_{ik} \right)$$. At step $$t$$, SVI draws a mini-batch $$B$$ of observations and computes their local parameters $$\varphi_i$$, for $$i \in B$$, with step 1 of CAVI. It then computes an estimate $$\hat{\lambda}_k$$ with the CAVI formula, in which each sum $$\sum_i$$ over the dataset is replaced by $$\frac{n}{\vert B \vert} \sum_{i \in B}$$, as if the mini-batch were repeated to the size of the dataset. Finally, it moves the global parameters a step toward this estimate:

$$
\lambda_k \leftarrow (1 - \rho_t) \, \lambda_k + \rho_t \, \hat{\lambda}_k
$$

This update is a step of size $$\rho_t$$ along the noisy natural gradient $$\hat{\lambda}_k - \lambda_k$$, and with $$\rho_t = 1$$ and the whole dataset as the mini-batch, it is exactly the CAVI update. It costs $$O(\vert B \vert K)$$ operations instead of $$O(nK)$$, and it needs only the mini-batch in memory. SVI therefore suits large and streaming datasets, while CAVI is simpler to run, as it needs no step sizes. Section 4.3 of [Blei et al.](https://arxiv.org/pdf/1601.00670.pdf) gives the general algorithm. Probabilistic programming libraries such as [Pyro](https://pyro.ai/examples/svi_part_i.html) also use SVI, and extend it to models whose complete conditionals are not conjugate, with Monte Carlo estimates of the gradient of the ELBO, as in the [variational autoencoder]({{ site.baseurl }}{% post_url 2019-10-12-vae-gan %}).
