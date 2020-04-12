---
layout: post
title:  "[draft] Bayesian Optimization: Variational Inference and Monte-Carlo methods"
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---

We have mentioned on a [previous post]({{ site.baseurl }}{% post_url 2019-11-12-Bayesian-Linear-Regression %}) about Bayesian inference, that the goal of Bayesian inference is to compute the likelihood of observed data and the mode of the density of the likelihood, marginal distribution and conditional distributions. Take the formulation of the **posterior** of the latent variable $z$ and observations $x$, derived from the Bayes rule without the normalization term:

$$
p (z \mid x) \propto p(z) \, p(x \mid z)
$$

read as *the posterior is proportional to the prior times the likelihood*. Inference in Bayesian models amounts to conditioning on the data and compute the posterior $p(z \mid x)$.

The inference problem is to compute the **conditional probability** of the latent variables $z$ given the observations in $X$. By definition, we can write the conditional probability as:

$$
p (z \mid x ) = \frac{p(z,x)}{p(x)}
\label{eq_conditional}
$$

The denominator represents the marginal density of $x$ (ie our observations) and is also referred to as **evidence**, which can be calculated by marginalizing the latent variables in $z$ over their joint distribution:

$$
p(x) = \int p(z,x) dz
\label{eq_joint}
$$



There are several approaches to inference, comprising algorithms for exact inference (Brute force, The elimination algorithm, Message passing (sum-product algorithm, Belief propagation), Juntion tree algorithm), and for approximate inference (Loopy belief propagation, Variational (Bayesian) inference + mean field approximations, Stochastic simulation / sampling / Markov Chain Monte Carlo). Why de we need approximate methods after all? Simple because for many cases, we cannot directly compute the posterior distribution, i.e. the posterior is on an **intractable** form --- often involving integrals --- which cannot be (easily) computed.

TODO chapter comparing variational inference and MCMC

Compared to other approximate methods such as MCMC, the main advantage is that VI tends to be faster and easier to scale for large data. 

## Variational Inference

Variational Inference (VI, original paper [here](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf) ) is most used to infer the conditional distribution over the latent variables, given the observations (and parameters), also referred to as the *posterior distribution* of the latent variables. The idea behind VI is to propose a family of densities and find a member $q^\star$ of that family which is close to the target (posterior). I.e. we try to find the parameters $z$ of the new distribution $q^\star$ (the approximation to our real posterior) such that:

$$
q^\star(z) = arg\,min \, KL(q(z) || p(z\mid x))
$$

The main rationale is that we can then utilize the VI-approximated distribution $q^\star$ and not the real (untractable) posterior $q$ on the tasks at hand.

#### Kullback-Leibler Divergence

The logic is that we want to minimize the divergence between our real posterior $p$ and its approximated posterior $g$ (sometimes referred to as the *guide* distribution/posterior). So we need to first define a metric of *approximation* or proximity. To that extent, the closeness (*proximity*) of two distributins is a measurement of *probabilities similiarity* measured by the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) 

$$
KL (q || p ) =  \int_z q(z) \log \frac{q(z)}{p(z \mid x)} = \mathbf{E} \left[ \log \frac{q(z)}{p( z\mid x)} \right]
$$ 

If both $q$ and $p$ are high, then we achieve a good solution as the KL-divergence is low. If $q$ is high and $p$ is low, we have a high divergenve and the solution *may* not be very good. Otherwise, if *q* is low, we dont care abot *p*, since we achieve low divergence, independently of $p$. It would then make more sense to compute $KL(p\|\|q)$ instead, however we do not do this from computational reasons (TODO explain).

The logic is that we want to minimize the divergence between our real posterior $p$ and its approximated posterior $g$ (sometimes referred to as the *guide* distribution/posterior). We will show later why we cant minimize this directly, but we can minimize a function that is *equal to it (up to a constant)*, known as the Evidence Lower Bound (ELBO)

#### Evidence Lower Bound (ELBO)

<small>
Good to know: there are mainly two approaches to describe ELBO. I'll follow the paper [*Variational Inference: A Review for Statisticians*](https://arxiv.org/pdf/1601.00670.pdf). For an alternative approach, check the blog posts from [Zhiya Zuo](https://zhiyzuo.github.io/VI/) and [Chris Coy](https://chrischoy.github.io/research/Expectation-Maximization-and-Variational-Inference/) . 
</small>

Before we start, we introduce the concept of [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) (applied to random variables $X$),  that relates the value of a convex function $f$ of an integral to the integral of the convex function:

$$
f ( \mathbf{E}[X] ) \ge \mathbf{E}[f(X)]
\label{eq_jensens}
$$

We apply the Jensen's inequality to the log (marginal) probability of the observations to get the ELBO:

$$
%source: assets/Bayesian-SVI-MCMC/10708-scribe-lecture13.pdf
\begin{align*}
\log p(x) & = \log \int p(x,z) dz & \text{(joint probability, eq. \ref{eq_joint})}\\
  & = log \int p(x,z) \frac{q(z)}{q(z)} dz \\
  & = \log \left( \mathbf{E} \left[ \frac{p(x,z)}{q(z)} \right] \right)    & \text{(expected value: $\mathbf{E}[X] = \int x \, p_X(x) \, dx$)} \\
  & \ge \mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)] & \text{($\mathbf{E}[XY]=\mathbf{E}[X]*\mathbf{E}[Y]$; Jensen's inequality, eq. \ref{eq_jensens})} \\
\end{align*}
$$

Putting it all together, the Evidence Lower Bound for a probability model $p(x,z)$ and an approximation to the posterio $q(z)$  is:

$$
\log p(x) \ge \mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)]
\label{eq_elbo}
$$

This formulation justifies the name ELBO. In practice, $\mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)]$ is smaller-or-equal than --- or is the lower bound of --- the evidence $\log p(x)$.


We can rewrite the KL-divergence as:

$$
%source: assets/Bayesian-SVI-MCMC/10708-scribe-lecture13.pdf
\begin{align*}
KL (q || p) & = \mathbf{E} \left[ \log \frac{q(z)}{p(z \mid x)} \right]\\
 & = \mathbf{E} [\log q(z)] - \mathbf{E}[\log p(z\mid x)] \\  
 & = \mathbf{E} [\log q(z)] - \mathbf{E} [\log p(z,x)] + \log p(x)  &\text{(def. conditional prob., eq. \ref{eq_conditional})}\\  
 & = - (\mathbf{E} [\log q(z)] + \mathbf{E} [\log p(z,x)]) + \log p(x)\\
\end{align*}
$$

i.e. the KL divergence is the negative of the ELBO plus a constant that does not depend on $q$. The previous formulation is also commonly represented as:

$$
\log p(x) = ELBO(q) + KL (q || p)
$$

So we optimize the lower-bound term over quantities $q(z)$ to find a minimal approximation. Or in other words, since $\log p(x)$ does not depend on $q$, we *maximize the ELBO which is equivalent to miniziming the KL-divergence* to the posterior.

#### Mean-field Variational Inference and coordinate ascent

So far we foccused on a single posterior defined by a family with a single distributions $q$. We now extend our analysis to complex families. We focus on the simplest method for such scenarios, the *mean-field approximation, where we assume all latent variables are independent*. More complex scenarions such as variational family with dependencies between variables (structured variational inference) or with mixtures of variational families (covered in [Chris Bishop's PRML book]({{ site.resources_permalink }}))) are not covered. As a side note, because the latent variables are independent, the mean-field model usually does not contain the true posterior.

In this type of Variational Inference, due to variable independency, the variational distribution over the latent variables factorizes as:

$$
q(z) = q(z_1, ..., z_m) = \prod_{j=1}^{m} q(z_j)
$$

where each latent variable ($z_j$) is governed by its own density $q_j(z_j)$. Note that the variational family input does not depend on the input data $x$ which does not show up in the equation. 

We can also assume that all $m$ latent variables can be grouped in $K$ groups $$z_{1}, ..., z_{K}$$, and represented instead as:

$$
q(z) =  q(z_1, ..., z_m) =  q(z_{1}, ..., z_{K}) = \prod_{j=1}^{K} q(z_{j}) 
$$

This setup is often called of **generalized mean field** instead of **naive mean field**. The computation of the functional form of $q_i(z_j)$ is very lenghty and detailed on [Brian Keng's post](http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/), so I'll skip those details.


The probability chain rule gives:

$$
%source: file:///C:/Users/t-brdacu/Downloads/10708-scribe-lecture13.pdf
p(z_{1:m}, x_{1:n}) = p(x_{1_n})
$$

To optimize the ELBO in these conditions, we use typically the coordinate ascent method, by optimizing the variational approximation of each latent variable $q_{z_j}$, while holding the others fixed.

#### Example: Gaussian Mixed Models

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

Take a family of a mixture of $K$ univariate Gaussian distributions with means $\mu = \{ \mu_1, \mu_2, ..., \mu_K \}$. The means are drawn from a common prior $p(\mu_n)$, which we assume to be Gaussian $$\mathcal{N}(\mu, \sigma^2)$$, and $\sigma^2$ is a hyper-parameter. The cluster assignment of each point $c_i$ is described as a $K-$vector of zeros except a one on the index of the allocated cluster. Each new observation $x_i$ is drawn from the corresponding Gaussian \mathcal{N}(c_i^T, \mu, 1). The full model is then described as:

$$
%source: section 2.1: https://arxiv.org/pdf/1601.00670.pdf
\begin{align}
\mu_k              & \thicksim \mathcal{N}(0, \sigma^2),             & & k=1,...,K, \\
c                  & \thicksim \text{Categorical} (1/K, ..., 1/K),   & & i=1,...,n, \\
x_i \mid c_i, \mu & \thicksim \mathcal{N}(c_i^T, \mu, 1)            & & i=1,...,n. \\
\end{align}
$$

quote-begin
For a sample of size $n$, the joint density of latent and observed variables is:

$$
p (\mu, c, x) = p(\mu) \prod_{i=1}^n p(c_i) p(x_i \mid c_i, \mu)
$$

The latent variables are $z=\{\mu, c\}$, the $K$ classes and $n$ class assignments.


#### Expectation Maximization

when we don't know the source, e.g. if we don't know beforehand which elements belong to each Gaussian Model. In this scenarion, we'd need to find as well the probability of each datapoint belonging to each Gaussian group. This problem becomes then a kind of a *chiken and egg* problem:
- we need the parameters of each Gaussian to know where each point belongs; but
- we need to know where each point belongs to know the parameters of each Gaussian;

The approach to this optimization is the [Expectaction Maximization (EM) method](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm), iterative algorithm that iterates over the following two steps until convergence. I'll try to update this post and include it it in the near future. 

disadvantages: over-confident, works well only if we know the parametric distribution of the posterior, requires the posterior to follow a parametric distribution, otherwise we can use the sampling.

## Monte Carlo sampling for exact posteriors 

Before VI, the dominant approach to compute the posterior was the Markov chain Monte Carlo methods (MCMC), based on sampling. 

[quote]
". Landmark developments include the Metropolis-Hastings algorithm (Metropolis et al.,
1953; Hastings, 1970), the Gibbs sampler (Geman and Geman, 1984) and its application to
Bayesian statistics (Gelfand and Smith, 1990). MCMC algorithms are under active investigation. They have been widely studied, extended, and applied; see Robert and Casella (2004)
for a perspective."

 
advantages: *real* posterior.
disadvantages: very high computation cost, what to do with an exact posterior which doesnt follow a parametric ditribution.

## Related topics

About the quality of Bayesian methods on Deep Neural Networks:
- https://arxiv.org/abs/1906.09686
- https://arxiv.org/abs/2002.02405

Forward vd Reverse KL:
- https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/
- https://blog.evjang.com/2016/08/variational-bayes.html

Stochastic Variational Inference and the variational autoencoders
- http://krasserm.github.io/2018/04/03/variational-inference/

PyTorch code exmaples:
- https://pyro.ai/examples/bayesian\_regression.html
- https://pyro.ai/examples/svi\_part\_i.html
- http://krasserm.github.io/2019/11/21/latent-variable-models-part-1/

