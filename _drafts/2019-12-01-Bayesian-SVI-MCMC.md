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



There are several approaches to inference, comprising algorithms for exact inference (Brute force, The elimination algorithm, Message passing (sum-product algorithm, Belief propagation), Juntion tree algorithm), and for approximate inference (Loopy belief propagation, Variational (Bayesian) inference + mean field approximations, Stochastic simulation / sampling / Markov Chain Monte Carlo). Why de we need approximate methods after all? Simple because for many cases, we cannot directly compute the posterior distribution, i.e. the posteior is in an **intractable** form --- often involving integrals --- which cannot be (easily) computed.

TODO chapter comparing variational inference and MCMC

Compared to other approximate methods such as MCMC, the main advantage is that VI tends to be faster and easier to scale for large data. 

## Variational Inference

Variational Inference (VI, original paper [here](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf) ) is most used to infer the conditional distribution over the latent variables, given the observations (and parameters), also referred to as the *posterior distribution* of the latent variables. The idea behind VI is to propose a family of densities and find a member $q^\star$ of that family which is close to the target (posterior). I.e. we try to find the parameters $z$ of the new distribution $q^\star$ (the approximation to our real posterior) such that:

$$
q^\star(z) = arg\,min \, KL(q(z) || p(z\mid x))
$$

The main rationale is that we can then utilize the VI-approximated distribution $q^\star$ and not the real (untractable) posterior $q$ on the tasks at hand.

#### KL-divergence

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

Before we start, we introduce the concept of [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)(applied to random variables $X$),  that relates the value of a convex function $f$ of an integral to the integral of the convex function:

$$
f ( \mathbf{E}[X] ) \ge \mathbf{E}[f(X)]
\label{eq_jensens}
$$

We apply the Jensen's inequality to the log (marginal) probability of the observations to get the ELBO:

$$
\begin{align*}
\log p(x) & = \log \int p(x,z) dz & \text{(joint probability, eq. \ref{eq_joint})}\\
  & = log \int p(x,z) \frac{q(z)}{q(z)} dz \\
  & = \log \left( \mathbf{E} \left[ \frac{p(x,z)}{q(z)} \right] \right)    & \text{(expected value: $\mathbf{E}[X] = \int x \, p_X(x) \, dx$)} \\
  & \ge \mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)] & \text{($\mathbf{E}[XY]=\mathbf{E}[X]*\mathbf{E}[Y]$; Jensen's inequality, eq. \ref{eq_jensens})} \\
\end{align*}
$$

Putting it all together, the Evidence Lower Bound for a probability model $p(x,z)$ and an approximation to the posterio $q(z)$  is:

$$
\log p(x) \ge \mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)]$
\label{eq_elbo}
$$

This formulation justifies the name ELBO. In practice, $\mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)]$ is smaller-or-equal than --- or is the lower bound of --- the evidence $\log p(x)$. So we optimize the lower-bound term over quantities $q(z)$ to find a minimal approximation. In practice, we maximize the ELBO, which is equivalent to miniziming the KL-divergence.

We can rewrite the KL-divergence as:

$$
\begin{align*}
KL (q || p) & = \mathbf{E} \left[ \log \frac{q(z)}{p(z \mid x)} \right]\\
 & = \mathbf{E} [\log q(z)] - \mathbf{E}[\log p(z\mid x)] \\  
 & = \mathbf{E} [\log q(z)] - \mathbf{E} [\log p(z,x)] + \mathbf{E}[\log p(x)]  &\text{(def. conditional prob., eq. \ref{eq_conditional})}\\  
 & = - (\mathbf{E} [\log q(z)] + \mathbf{E} [\log p(z,x)]) + \mathbf{E}[\log p(x)]\\
\end{align*}
$$

i.e. the KL divergence is the negative of the ELBO plus a constant that does not depend on $q$.
https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/

#### Gaussian Mixed Models and the EM algorithm

http://krasserm.github.io/2019/11/21/latent-variable-models-part-1/

#### Stochastic Variational Inference and the variational autoencoders

http://krasserm.github.io/2018/04/03/variational-inference/

 

The following sections provide a simpler explanation of the Stochastic Variational Inference. More information can be found in the [original paper](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf).

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

## Deep Neural Net

https://arxiv.org/abs/1906.09686
https://arxiv.org/abs/2002.02405

### TRASH


$$
\begin{align*}
%source: https://zhiyzuo.github.io/VI/
ln~p(\mathbf{x}) & = \int_{\mathbf{z}} q(\mathbf{z}) d\mathbf{z}~~ln~p(\mathbf{x}) \\ 
&  = \int_{\mathbf{z}} q(\mathbf{z}) ln~ \frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{z} \vert \mathbf{x})} d\mathbf{z} & \text{( $\int_{\mathbf{z}} q(\mathbf{z}) d\mathbf{z} = 1$)} \\
& = \int_{\mathbf{z}} q(\mathbf{z}) ln~ \frac{p(\mathbf{x}, \mathbf{z})~q(\mathbf{z})}{p(\mathbf{z} \vert \mathbf{x}) ~q(\mathbf{z})} d\mathbf{z}\\
& = \int_{\mathbf{z}} q(\mathbf{z}) ln~ \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})} d\mathbf{z} + \int_{\mathbf{z}} q(\mathbf{z}) ln~ \frac{q(\mathbf{z})}{p(\mathbf{z} \vert \mathbf{x})} d\mathbf{z}\\
& = \mathcal{L}(\mathbf{x}) + KL(q\vert \vert p) \\
\end{align*}
$$

where $\mathcal{L}$ is called is the ELBO, and the KL-divergence is bounded and nonnegative. By decomposing ELBO, we have:

$$
\begin{align*}
%source: https://zhiyzuo.github.io/VI/
\mathcal{L}(\mathbf{x}) & = \int_{\mathbf{z}} q(\mathbf{z}) ln~ \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})} d\mathbf{z} \\
& = \int_{\mathbf{z}} q(\mathbf{z}) ln~p(\mathbf{x} \vert \mathbf{z}) - q(\mathbf{z})ln~\frac{q(\mathbf{z})}{p(\mathbf{z})} d\mathbf{z}\\ 
& = E_q\big[ ln~p(\mathbf{z} \vert \mathbf{x}) \big] - KL(q(\mathbf{z})||p(\mathbf{z}))\\ 
& = \int_{\mathbf{z}} q(\mathbf{z}) ln~p(\mathbf{x}, \mathbf{z}) - q(\mathbf{z})ln~q(\mathbf{z}) d\mathbf{z}\\
& = E_q\big[ ln~p(\mathbf{x}, \mathbf{z}) \big] + \mathcal{H}(q) & \text{(Entropy of } q\text{)}\\
\end{align*}
$$
