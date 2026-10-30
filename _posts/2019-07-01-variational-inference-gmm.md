---
layout: post
title:  "Variational Inference (ELBO, CAVI) and Gaussian Mixture Models"
categories: ["Machine Learning"]
tags: ["variational inference", "ELBO", "KL divergence", "mean-field approximation", "CAVI", "SVI", "GMM", "MCMC"]
description: "Variational inference as an alternative to MCMC: the KL divergence and the ELBO, the mean-field approximation, coordinate ascent (CAVI) on a Gaussian mixture model, and stochastic variational inference (SVI) for large datasets."
---

Consider 100 points and a Bayesian model that groups them into 5 clusters, each a Gaussian distribution with an unknown mean (a Gaussian mixture model). The exact posterior of this model, i.e. the distribution of the cluster means and of the cluster of each point given the data, is normalized by a sum over every possible assignment of the points to the clusters: $$5^{100} \approx 7.9 \times 10^{69}$$ terms, which would take $$2.5 \times 10^{53}$$ years at one nanosecond per term. Variational inference avoids this sum: it approximates the posterior by the closest member of a simpler family of distributions, which is found by optimization.

In 1987, the paper [A Mean Field Theory Learning Algorithm for Neural Networks](https://www.complex-systems.com/abstracts/v01_i05_a06/) sped up the training of a neural network by replacing its slow stochastic sampling with the deterministic equations of mean-field theory, a method of statistical physics. This idea was then extended to a wide class of probabilistic models, as presented in the tutorial [An Introduction to Variational Methods for Graphical Models](https://doi.org/10.1023/A:1007665907178) in 1999.

**This post focuses on the simplest approach to Variational Inference, based on the mean-field approximation**. After a refresher on Bayesian inference and a detour on Markov Chain Monte Carlo (MCMC), it derives the Evidence Lower Bound (ELBO) and the coordinate ascent variational inference (CAVI) algorithm, and applies it to a Gaussian mixture model, where it is compared with the [expectation–maximization (EM)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm. It ends with Stochastic Variational Inference (SVI), which scales CAVI to large datasets, and with a comparison of MCMC, CAVI and SVI.

## The Inference Problem

A [previous post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}) about Bayesian inference showed that its goal is to compute the **posterior**, the distribution of the unknown variables (there, the weights of a linear regression) given the observed data. More generally, inference also computes the likelihood of the observed data (the evidence, defined below), the marginal distribution of a subset of the variables, and the modes of these distributions. Recall the formulation of the posterior of the [latent variable](https://en.wikipedia.org/wiki/Latent_and_observable_variables) $$z$$ given the observations $$x$$, derived from the Bayes rule, with and without the normalization term:

$$
p (z \mid x) = \frac{p(z) \, p(x \mid z)}{p(x)} \propto p(z) \, p(x \mid z)
\label{eq_bayes}
$$

read as *the posterior is proportional to the prior times the likelihood*. 
The same post also showed that the Bayes rule derives from the definition of [**conditional probability**](https://en.wikipedia.org/wiki/Conditional_probability#Definition) $$p(z\mid x)$$ expressed as:

$$
p (z \mid x ) = \frac{p(z,x)}{p(x)}
\label{eq_conditional}
$$

where the denominator represents the marginal density of $$x$$ (i.e. the observations), also referred to as the **[evidence](https://en.wikipedia.org/wiki/Marginal_likelihood)**, which can be calculated by **marginalizing** out the latent variables $$z$$ from the joint distribution:

$$
p(x) = \int p(z,x) dz
\label{eq_joint}
$$

The inference problem is to compute the conditional probability of the latent variables $$z$$ given the observations $$x$$, i.e. to condition on the data and compute the posterior $$p(z \mid x)$$. For many models, this evidence integral has no closed-form solution or requires exponential time to compute. 

**There are several approaches to inference**. Algorithms for exact inference include brute force, the [elimination algorithm](https://en.wikipedia.org/wiki/Variable_elimination), message passing (the sum-product algorithm, or [belief propagation](https://en.wikipedia.org/wiki/Belief_propagation)) and the [junction tree algorithm](https://en.wikipedia.org/wiki/Junction_tree_algorithm). Algorithms for approximate inference include loopy belief propagation, [variational (Bayesian) inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) and stochastic simulation, i.e. sampling methods such as Markov Chain Monte Carlo. Approximate methods are needed because, in many cases, the posterior distribution cannot be computed directly, i.e. the posterior has an **intractable** form (often involving integrals) that cannot be (easily) computed.

## Detour: Markov Chain Monte Carlo

Before moving on to Variational Inference, it helps to understand the place of VI in this type of inference. For many years, the dominant approach was [Markov chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo). A simple high-level understanding of MCMC follows from the name itself.

[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) methods are a simple way of estimating parameters by generating random numbers. A simple example is to compute the area of a circle by generating random 2D coordinates within the bounding square around the circle. The value of $$\pi$$ or the area of the circle is then estimated from the proportion of generated datapoints that fall inside vs outside the circle in the square.

For a circle of radius 1 and the square $$[-1, 1]^2$$ around it, the proportion of points inside the circle tends to the ratio of the two areas, $$\pi / 4$$. A run with 10,000 random points puts 7,901 of them inside the circle, which estimates $$\pi \approx 4 \times 0.7901 = 3.1604$$. The error of such an estimate falls as $$1 / \sqrt{N}$$ with the number of points $$N$$: its standard deviation is 0.16 for 100 points, 0.016 for 10,000 points and 0.0016 for a million, so each additional correct digit costs 100 times more points.

A [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain) is a stochastic model describing a sequence of events in which the probability of moving to the next state depends *only* on the current state and not on the previous ones. An example is the probability of the next character in a word, for all *27* possible characters, given the current character.

Therefore, MCMC methods approximate the posterior by random sampling: each new value of the latent variables is drawn from a probabilistic space defined by the current value only, and the values form a Markov chain whose *stationary distribution* (the distribution that the chain reaches after many steps) is the posterior. The posterior is then approximated by the samples collected along the chain.

MCMC is based on the assumption that the prior-likelihood product $$p(x \mid z) \, p(z)$$ can be computed, i.e. is known, for a given $$z$$, without any knowledge of the posterior as a function of $$z$$. This product is the numerator of the posterior (eq. \ref{eq_bayes}), while the denominator, the evidence $$p(x)$$, is unknown or intractable. Since the denominator is only a *normalizer* (i.e. it guarantees that the posterior sums to 1), MCMC can discard it and work with $$p(z \mid x) \propto p(z) \, p(x \mid z)$$.

The rationale is then to propose, at each step, a new value $$z$$ of the latent variables (for example, the mean or the variance of a normal distribution) near the current value $$z_0$$, and to move to it more often when it explains the data better, i.e. when it has a higher posterior probability. A simple way to express this is to compute the **acceptance ratio** of the proposed over the current posterior (which cancels out the term $$p(x)$$ that could not be computed):

$$
  \frac{ p(z \mid x) }{ p(z_0 \mid x) }
=  \frac{ \hspace{0.5cm}{ } \frac{p(x \mid z) \, p(z)}{ p(x) } \hspace{0.5cm} } { \,\, \frac{p(x \mid z_0) \, p(z_0)}{ p(x) } \,\, }
= \frac{ p(x \mid z) \, p(z) }{ p(x \mid z_0) \, p(z_0)}
$$

The proposed state $$z$$ is accepted with probability $$\min(1, \text{ratio})$$, otherwise the chain keeps the current state $$z_0$$ (the [Metropolis algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), with a symmetric proposal distribution). A state that explains the data better (a ratio above 1) is always accepted, and a worse one is accepted with a probability equal to the ratio. Thus, the chain does not stop at the most likely state: it visits each region of the latent space in proportion to its posterior probability, and, after enough steps, the states that it visits are samples of the posterior.

The main advantage of MCMC is that it approximates the *true* posterior, whatever its form. However, it only provides samples of it, and there are many disadvantages in handling a posterior that is known only through samples and does not follow a parametric distribution (such as being hard to interpret in high dimensions). Moreover, it requires massive computational power for problems with large latent spaces. That is where Variational Inference enters the picture: it is faster and works well if the posterior is close to a known parametric distribution, but it is sometimes *over-confident* when the posterior does not exactly match the approximation.

So as a golden rule, variational methods are the default choice. If in doubt, the results should be double-checked against an MCMC estimate of the posterior, which is also approximate but converges to the true posterior as the number of samples grows. In practice, the paper [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) suggests variational inference for large datasets and for the fast exploration of many models, and MCMC for smaller datasets, where a higher computational cost buys more precise samples.

The theory behind MCMC is covered in the paper [An Introduction to MCMC for Machine Learning]({{ site.assets }}/variational-inference-gmm/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf), and a practical explanation with code is available in [Thomas Wiecki's post](https://twiecki.io/blog/2015/11/10/mcmc-sampling/).

## Variational Inference

**The idea behind Variational Inference (VI) methods is to find a parametric distribution $$q(z)$$ that is an approximation of an intractable/non-parametric posterior $$p(z \mid x)$$**. Inference thus becomes an optimization problem: instead of sampling, VI searches a family of distributions for the member closest to the posterior. To do that, the first step is to define a function that measures the closeness between the two distributions $$p$$ and $$q$$. So the starting point is the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between two distributions $$f(x)$$ and $$g(x)$$, defined as:

$$
KL (f(x) || g(x)) =  \int f(x) \log \frac{f(x)}{g(x)} dx= \mathbb{E} \left[ \log \frac{f(x)}{g(x)} \right] = \mathbb{E} \left[ \log f(x) \right] - \mathbb{E} \left[ \log g(x) \right]
$$ 

when applied to this use case:

$$
KL (q(z) || p(z \mid x)) =  \int q(z) \log \frac{q(z)}{p(z \mid x)} dz= \mathbb{E} \left[ \log \frac{q(z)}{p( z\mid x)} \right]
\label{eq_KLdiv}
$$ 

Note that the **KL divergence is a dissimilarity function and not a metric**, as it is not symmetric. Also, there are other divergences besides the KL divergence, part of the [f-divergence family](https://en.wikipedia.org/wiki/F-divergence).

The order of the two distributions matters. Eq. \ref{eq_KLdiv}, written $$KL(q \Vert p)$$ for short, is called the **reverse KL divergence**: it is an expectation over $$q$$, which is chosen simple enough for this expectation to be computed, and the posterior only appears inside its logarithm. The **forward KL divergence** $$KL(p \Vert q)$$ would instead be an expectation over the posterior $$p(z \mid x)$$, the very distribution that is unknown. The reverse divergence also shapes the approximation: its log-ratio $$\log (q/p)$$ is large where $$q$$ puts mass and $$p$$ has little, so $$q$$ pays a high price for placing mass where the posterior is small, while it pays nothing where $$q$$ itself is small, whatever the posterior is there.

As an example, take a 2D Gaussian posterior with unit variances and a correlation of 0.9, approximated by the product of two independent Gaussians (the mean-field family of the next sections). The product that minimizes the reverse KL has the mean of the posterior, but a variance of only $$1 - 0.9^2 = 0.19$$ in each dimension instead of 1: the variance of one dimension when the other is known ([Chris Bishop's PRML book]({{ site.resources_permalink }}), section 10.1.2). It fits inside the narrow ridge of the posterior, with a standard deviation of 0.44 instead of 1. The product that minimizes the forward KL instead matches the marginal variances of 1, but puts mass outside the ridge, where the posterior has little.

This is why VI is *over-confident*: it underestimates the variance of the posterior, as Figure 1 of the review paper above shows. Similarly, when the posterior has several modes, the reverse KL fits $$q$$ to one of them (it is *mode-seeking*), while the forward KL spreads $$q$$ over all of them.

Looking at equation \ref{eq_KLdiv}, there is an issue: it includes the true posterior $$p(z \mid x)$$, which is exactly the value that is not known. So it cannot be minimized directly, but it is possible to maximize a function that is *equal to its negative (up to a constant)*, known as the [Evidence Lower Bound](https://en.wikipedia.org/wiki/Evidence_lower_bound).

## Evidence Lower Bound (ELBO)

The KL divergence can be rewritten as:

$$
\begin{align*}
KL (q(z) || p(z \mid x)) 
  & = \int q(z) \log \frac{q(z)}{p(z \mid x)} dz \\
  & = \int q(z) \log \frac{q(z)p(x)}{p(z, x)} dz \\
    & \qquad \text{(definition of conditional prob.)} \\
  & = \int q(z) \left( \log p(x) + \log \frac{q(z)}{p(z, x)} \right) dz \\
    & \qquad \text{(expanding } \log \text{)} \\
  & = \log p(x) \int q(z) dz + \int q(z) \log \frac{q(z)}{p(z, x)} dz \\
    & \qquad \text{(splitting the integral)} \\
  & = \log p(x) - \int q(z) \log \frac{p(z, x)}{q(z)} dz \\
    & \qquad \text{(negate log; } q(z) \text{ is a dist. thus } \int q(z) dz =1 \text{)} \\
  &= \log p(x) - ELBO(q,p)\\
\end{align*}
$$

In practice, the original $$KL$$-divergence formulation, which could not be minimized, was replaced by an expression that can be optimized: minimizing the KL divergence is equivalent to maximizing the ELBO. Looking at the right-hand side of the expression:
- The goal is to minimize the KL divergence over $$q$$, so the term $$\log p(x)$$ can be ignored as it does not depend on $$q$$;
- All terms inside $$ELBO(q,p)$$ are known, as $$p(z,x)$$ is the prior times the likelihood $$p(z,x) = p(z) \, p(x \mid z)$$, which are the inputs of the problem;
- The KL divergence cannot be negative. Thus $$\log p(x) \ge ELBO$$, i.e. the ELBO is never larger than the log evidence $$\log p(x)$$, and this explains the name **Evidence Lower Bound**.

Note that the ELBO can also be written, and is usually found expressed, in terms of two expectations:

$$
ELBO(q,p) = \int q(z) \log \frac{p(z, x)}{q(z)} dz =  \int q(z) \log p(z, x) dz - \int q(z) \log q(z) dz  = \mathbb{E} [\log p(x,z)] - \mathbb{E}[\log q(z)]
\label{eq_elbo}
$$

The second term, $$-\mathbb{E}[\log q(z)]$$, is the [**entropy**](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of $$q$$, a measure of how spread out it is. So the ELBO rewards a $$q$$ that places its mass where the joint $$p(z, x)$$ is high, and its entropy term keeps $$q$$ spread out, rather than collapsed on the single most likely value of $$z$$. Splitting the joint as $$p(z, x) = p(z) \, p(x \mid z)$$ also gives the ELBO as the expected log-likelihood of the data minus the KL divergence between $$q$$ and the prior (section 2.2 of the review paper):

$$
ELBO(q) = \mathbb{E} [\log p(x \mid z)] - KL (q(z) || p(z))
$$

The first term encourages $$q$$ to place its mass on the values of the latent variables that explain the observed data, and the second keeps $$q$$ close to the prior: the usual balance between likelihood and prior of Bayesian inference. The [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) maximizes the ELBO in this form.

There are two common options to solve the previous optimization problem: [coordinate ascent](https://en.wikipedia.org/wiki/Coordinate_descent), which optimizes the approximation of one latent variable at a time, as in the CAVI algorithm below (slow, and requires deriving the formulation of the updates by hand), or stochastic optimization, as in Stochastic Variational Inference (faster on large datasets). Both optimize over a family of distributions simple enough for the expectations of the ELBO to be computed, such as the mean-field family of the next section.

## Mean-field approximation

So far, $$q(z)$$ was one distribution over all latent variables; now it is restricted to a family that makes the optimization tractable. One way to do so is to approximate the posterior probability using a simpler model, for example by assuming that a set of latent variables is independent of the other variables. Or full independence can be enforced among all latent variables given $$x$$. This assumption is known as the **mean-field approximation**.

In practice, this method originates from mean-field theory. From [Wikipedia](https://en.wikipedia.org/wiki/Mean-field_theory): 
> "Mean-field theory [...] studies the behavior of high-dimensional random (stochastic) models by studying a simpler model that approximates the original by averaging over degrees of freedom. [...] The effect of all the other individuals on any given individual is approximated by a single averaged effect, thus reducing a many-body problem to a one-body problem."

More complex scenarios, such as a variational family with dependencies between variables (structured variational inference) or with mixtures of variational families (covered in [Chris Bishop's PRML book]({{ site.resources_permalink }})), are not covered in this post. As a side note, **because the latent variables are assumed to be independent, the mean-field approach that follows usually does not contain the true posterior**: in the Gaussian mixture model below, for example, the cluster assignments of the points depend on each other and on the cluster means, given the data.

In this type of Variational Inference, due to the independence of the variables, the variational distribution over the latent variables factorizes as:

$$
q(z) = q(z_1, ..., z_m) = \prod_{j=1}^{m} q(z_j)
\label{eq_posterior_mf}
$$

where each latent variable ($$z_j$$) is governed by its own density $$q(z_j)$$. Note that the variational family does not depend on the input data $$x$$, which does not show up in the equation: it is the ELBO that connects $$q$$ to the data and the model.

The latent variables can also be partitioned into $$R$$ groups $$z_{G_1}, ..., z_{G_R}$$, with the approximation represented instead as:

$$
q(z) =  q(z_1, ..., z_m) =  q(z_{G_1}, ..., z_{G_R}) = \prod_{j=1}^{R} q_j(z_{G_j}) 
$$

This setup is often called **generalized mean field**, as opposed to **naive mean field**. In the naive mean field, each latent variable $$z_j$$ is governed by its own **variational factor**, the density $$q_j(z_j)$$. For the formulation to be complete, the parametric form of the individual variational factors would have to be specified. In principle, each can take on any parametric distribution appropriate to the corresponding random variable (e.g. a combination of Gaussian and [Categorical](https://en.wikipedia.org/wiki/Categorical_distribution) distributions).


## Coordinate Ascent Variational Inference 

The main objective is to **optimize the ELBO in mean-field variational inference**, or equivalently, to choose the variational factors that maximize the ELBO (eq. \ref{eq_elbo}). A common approach is to use the **coordinate ascent** method, optimizing the variational approximation of each latent variable $$q(z_j)$$ while holding the others fixed. The derivation below follows the lecture notes [Variational Inference: Mean Field Approximation]({{ site.assets }}/variational-inference-gmm/10708-scribe-lecture13.pdf) of the CMU course Probabilistic Graphical Models (10-708).

Recall the [probability chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)) for more than two events $$X_1, \ldots , X_n$$:

$$
P(X_n, \ldots , X_1)  = P(X_n | X_{n-1}, \ldots , X_1) \cdot P(X_{n-1}, \ldots , X_1)
$$

in its recursive form, or the following in its general form: 

$$
P\left(\bigcap_{k=1}^n X_k\right) = \prod_{k=1}^n P\left(X_k \,\Bigg|\, \bigcap_{j=1}^{k-1} X_j\right)
$$

Remember from eq. \ref{eq_elbo} that $$ELBO(q) = \mathbb{E} [\log p(z, x)] - \mathbb{E}[\log q(z)]$$.
Now the chain rule is applied to the joint probability $$p(z_{1:m}, x_{1:n})$$ for $$m$$ variational factors and $$n$$ datapoints, which gives:

$$
%source: assets: 10708-scribe-lecture13.pdf
p(z_{1:m}, x_{1:n}) = p(x_{1:n}) \prod_{l=1}^m p(z_l \mid z_{1: (l-1)}, x_{1:n})
$$

and the negative entropy, $$\mathbb{E}_q [\log q(z_{1:m})]$$, can be decomposed using the mean-field variational approximation as:

$$
\mathbb{E}_q [ \log q(z_{1:m})] = \sum_{l=1}^m \mathbb{E}_{q_l}[\log q(z_l)]
$$

Inserting the two previous decompositions into the ELBO equation, it can now be expressed for the mean-field variational approximation as:

$$
%source: assets: 10708-scribe-lecture13.pdf
\begin{align*}
ELBO(q) & = \mathbb{E}_q [\log p(x_{1:n},z_{1:m})] - \mathbb{E}_q[\log q(z_{1:m})] \\
        & = \mathbb{E}_q \left[ \log \left ( p(x_{1:n}) \prod_{l=1}^m p(z_l \mid z_{1: (l-1)}, x_{1:n}) \right) \right] - \sum_{l=1}^m \mathbb{E}_{q_l}[\log q(z_l)] \\
        & = \log p(x_{1:n}) + \sum_{l=1}^m \mathbb{E}_q \left[ \log p(z_l \mid z_{1: (l-1)}, x_{1:n}) \right] - \sum_{l=1}^m \mathbb{E}_{q_l}[\log q(z_l)] \\
\end{align*}
$$

Coordinate ascent variational inference (CAVI) is now applied, where the derivative with respect to each variational factor is computed and the factors are updated alternately, each to the value that maximizes the ELBO while the other factors are fixed. Before proceeding to the derivative, the following notation is introduced:

$$
p (z_j \mid z_{\neq j}, x) = p (z_j \mid z_1, ..., z_{j-1}, z_{j+1}, ..., z_m, x)
$$

where $$\neq j$$ means *all indices except the $$j^{th}$$*, so that $$p(z_j \mid z_{\neq j}, x)$$ is the **complete conditional** of $$z_j$$, its distribution given all the other latent variables and the observations. Now the goal is to derive the coordinate ascent update for a latent variable, while keeping the others fixed, i.e. to compute $$arg\,max_{q_j} \, ELBO$$. The order of the latent variables in the chain rule is arbitrary, so $$z_j$$ can be placed last, and then only the term $$p(z_j \mid z_{\neq j}, x)$$ of the product depends on $$z_j$$:

$$
\begin{align*}
  & arg\,max_{q_j} \, ELBO (q)\\
= & arg\,max_{q_j} \, \left( \log p(x_{1:n}) + \sum_{l=1}^m \mathbb{E}_q \left[ \log p(z_l \mid z_{1: (l-1)}, x_{1:n}) \right] - \sum_{l=1}^m \mathbb{E}_{q_l}[\log q(z_l)] \right)\\  
= & arg\,max_{q_j} \, \left( \mathbb{E}_q \left[ \log p(z_j \mid z_{\neq j}, x) \right] - \mathbb{E}_{q_j}[\log q(z_j)] \right) \\
  & \qquad \text{(removing vars that do not depend on } q(z_j) \text{)} \\
= & arg\,max_{q_j} \, \left( \int q(z_j) \, \mathbb{E}_{q_{\neq j}} \left[ \log p(z_j \mid z_{\neq j}, x) \right] dz_j - \int q(z_j) \log q(z_j) dz_j \right) \\
  & \qquad \text{(def. Expected Value)} \\
\end{align*}
$$

To find this argmax, one takes the derivative of the ELBO, denoted $$L$$, with respect to $$q(z_j)$$, uses [Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier) (as $$q(z_j)$$ must integrate to one), and sets the derivative to zero:

$$
\frac{d\,L}{d\,q(z_j)} = \mathbb{E}_{q_{\neq j}} \left[ \log p(z_j \mid z_{\neq j}, x) \right] - \log q(z_j) - 1 + \lambda = 0
$$
 
where $$\lambda$$ is the Lagrange multiplier of the constraint $$\int q(z_j) \, dz_j = 1$$. Solving for $$q(z_j)$$ gives $$q(z_j) = e^{\lambda - 1} \exp \{ \mathbb{E}_{q_{\neq j}} [ \log p(z_j \mid z_{\neq j}, x)] \}$$, in which the factor $$e^{\lambda - 1}$$ is the normalization constant that makes $$q(z_j)$$ integrate to one. This leads to the final coordinate ascent update:

$$
\begin{align}
q^{\star}(z_j) & \propto \exp \{ \mathbb{E}_{q_{\neq j}} [ \log p(z_j \mid z_{\neq j}, x)] \} \\
               & \propto \exp \{ \mathbb{E}_{q_{\neq j}} [ \log p(z_j, z_{\neq j}, x)] \} \label{eq_CAVI} \\
                 & \qquad \text{(} p(z_{\neq j}, x) \text{ does not depend on } z_j \text{)} \\
\end{align}
$$

However, this provides the template of the computation, but not the final form (i.e. the application to a specific distribution family) of the optimal $$q_j$$: the form follows from the expectation in eq. \ref{eq_CAVI}, once it is computed for a specific model, and it determines the complexity or *easiness* of the coordinate update $$q^{\star}(z_j)$$. When the complete conditional $$p(z_j \mid z_{\neq j}, x)$$ is in the [exponential family](https://en.wikipedia.org/wiki/Exponential_family), the optimal factor is in the same family, and the update reduces to a closed-form formula for its parameters (section 4.1 of the review paper). The Gaussian mixture model of the next section is such a model.

## Gaussian Mixture Models

Credit: this section is a more verbose explanation of sections 2.1, 3.1 and 3.2 of the paper [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf).

A [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model) is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. The problem at hand is to fit a set of Gaussian density functions to the input dataset.

Take a mixture of $$K$$ univariate Gaussian distributions with means $$\mu = \mu_{1:K} = \{ \mu_1, \mu_2, ..., \mu_K \}$$. The means are drawn from a common prior $$p(\mu_k)$$, which is assumed to be Gaussian $$\mathcal{N}(0, \sigma^2)$$. **The latent variables are the means $$\mu_{1:K}$$ and the cluster assignments $$c_{1:n}$$** for each observation $$x_i$$. **$$\sigma^2$$ is a hyper-parameter**.

$$c_i$$ is described as an indicator $$K$$-vector, i.e. all zeros except for a one at the index of the allocated cluster. Each new observation $$x_i$$ (from a total of $$n$$ observations) is drawn from the corresponding Gaussian $$\mathcal{N}(c_i^T \mu, 1)$$. The model can be expressed as:

$$
%source: section 2.1: https://arxiv.org/pdf/1601.00670.pdf
\begin{align}
\mu_k              & \thicksim \mathcal{N}(0, \sigma^2),             & & k=1,...,K, \\
c_i                & \thicksim \text{Categorical} (1/K, ..., 1/K),   & & i=1,...,n, \\
x_i \mid c_i, \mu & \thicksim \mathcal{N}(c_i^T \mu, 1)            & & i=1,...,n. \label{eq_gmm_likelihood}\\
\end{align}
$$

In this case, the joint probability of the latent and observed variables for a sample of size $$n$$ is:

$$
p (\mu, c, x) = p(\mu) p(c,x \mid \mu) = p(\mu) \, \prod_{i=1}^n p(c_i) p(x_i \mid c_i, \mu)
\label{eq_7_source}
$$

Here the probability chain rule is applied again. The latent variables are $$z=\{\mu, c\}$$, holding the $$K$$-class means and the $$n$$-point class assignments. The evidence is:

$$
\begin{align*}
p(x) & = \int p(\mu) \prod_{i=1}^n \sum_{c_i} p(c_i) \, p(x_i \mid c_i, \mu) d\mu \\
     & = \sum_c p(c) \int p(\mu) \prod_{i=1}^n p(x_i \mid c_i, \mu) d\mu \\
\end{align*}
$$

I.e. the evidence is now a sum over all possible configurations of cluster assignments, with a computational complexity of $$O(K^n)$$, due to the $$K$$ classes that have to be assigned to $$n$$ points, such as the $$5^{100}$$ terms of the opening for 100 points and 5 clusters.
Each integral in the previous equation can be computed, since the Gaussian prior and likelihood are [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior).
However, the evidence is still computationally intractable due to the $$K^n$$ complexity, exponential in the number of points.

An alternative solution is to use the mean-field approximation method discussed above. As shown before (eq. \ref{eq_posterior_mf}), the mean-field approximation yields approximate posterior probabilities of the form:

$$
q(\mu,c) = \prod_{k=1}^K q(\mu_k; m_k, s^2_k) \prod_{i=1}^n q(c_i; \varphi_i).
\label{eq_variational_family}
$$

Rephrasing what was mentioned before, each latent variable is governed by its own variational factor. The factor $$q(\mu_k; m_k, s^2_k)$$ is a Gaussian distribution on the mean of the $$k^{th}$$ mixture component, with mean $$m_k$$ and variance $$s^2_k$$. The factor $$q(c_i; \varphi_i)$$ is a distribution on the $$i^{th}$$ observation's mixture assignment; its assignment probabilities are a $$K$$-vector $$\varphi_i$$. These forms are the optimal ones for this model: as the updates below show, the coordinate ascent update of eq. \ref{eq_CAVI} gives a Categorical factor for each assignment and a Gaussian factor for each mean.

So there are two types of variational parameters: (1) the Gaussian parameters $$m_k$$ and $$s^2_k$$ to approximate the posterior of the $$k^{th}$$ component; and (2) the Categorical parameters $$\varphi_i$$ to approximate the posterior cluster assignment of the $$i^{th}$$ data point. The joint density of the latent and observed variables (eq. \ref{eq_7_source}) and the variational family (eq. \ref{eq_variational_family}) are now combined to form the ELBO for the mixture of Gaussians:

$$
%source https://arxiv.org/pdf/1601.00670.pdf
\begin{align}
ELBO(m, s^2, \varphi) = & \sum_{k=1}^K \mathbb{E} \left[ \log p(\mu_k); m_k, s_k^2 \right] \\
                        & + \sum_{i=1}^n \left( \mathbb{E} \left[ \log p(c_i); \varphi_i \right] + \mathbb{E} \left[ \log p(x_i \mid c_i, \mu); \varphi_i, m, s^2 \right] \right) \\
                        & - \sum_{i=1}^n \mathbb{E} \left[ \log q(c_i; \varphi_i) \right] - \sum_{k=1}^K \mathbb{E} \left[ \log q(\mu_k ; m_k, s_k^2) \right] \label{eq_algo_elbo} \\
\end{align}
$$

The CAVI algorithm updates each variational parameter in turn, so both updates need to be computed.

### Update 1: Variational update for cluster assignment

The derivation starts with the variational update for the cluster assignment $$c_i$$. Using the mean-field recipe from equation \ref{eq_CAVI}:

$$
q^{\star}(c_i; \varphi_i) \propto \exp \left( \log p(c_i) + \mathbb{E} [ \log p(x_i \mid c_i, \mu); m, s^2] \right).
$$

The first term is the log prior of $$c_i$$, which is the same for all values of $$c_i$$: $$\log p(c_i)= \log \frac{1}{K} = \log 1 - \log K = -\log K$$. The second term is the expected log probability of the $$c_i^{th}$$ Gaussian density. Because $$c_i$$ is an [indicator vector](https://en.wikipedia.org/wiki/Indicator_vector), the likelihood can be written as:

$$
p (x_i \mid c_i, \mu) = \prod_{k=1}^K p(x_i \mid \mu_k)^{c_{ik}}
$$

i.e. because $$c_{ik} \in \{0,1\}$$, this is a product of terms that are equal to one, except when cluster $$k$$ is allocated to the datapoint $$x_i$$. This is used to continue the expansion of the second term:

$$
\begin{align*}
  & \mathbb{E} [ \log p(x_i \mid c_i, \mu); m, s^2] \\
= & \sum_k c_{ik} \mathbb{E} [ \log p(x_i \mid \mu_k); m_k, s_k^2] \\
  & \qquad \text{(sum of matching assignments } ik \text{)} \\
= & \sum_k c_{ik} \mathbb{E} \left[ \log \mathcal{N}(x_i \mid \mu_k, 1) \right] \\
  & \qquad \text{(likelihood eq. \ref{eq_gmm_likelihood})} \\
= & \sum_k c_{ik} \mathbb{E} \left[ \log \left( \frac{1}{1 \sqrt{2\pi}} \right) - \frac{1}{2} \left( \frac{x_i - \mu_k}{1} \right)^2 ; m_k, s_k^2 \right] \\
  & \qquad \text{(log of normal distribution)} \\
= & \sum_k c_{ik} \mathbb{E} \left[ -\frac{1}{2}(x_i - \mu_k)^2; m_k, s_k^2 \right] + const \\
  & \qquad \text{(removed terms that are constant with respect to } c_i \text{)} \\
= & \sum_k c_{ik} \mathbb{E} \left[ -\frac{1}{2} x_i^2 + \mu_k x_i -\frac{1}{2} \mu_k^2; m_k, s_k^2 \right] + const \\
  & \qquad \text{(decomposed square of sum)} \\
= & \sum_k c_{ik} \left( x_i \, \mathbb{E} [ \mu_k; m_k, s_k^2] - \frac{1}{2} \mathbb{E}[ \mu^2_k; m_k, s^2_k] \right) + const \\
  & \qquad \text{(removed terms that are constant with respect to } c_i \text{)} \\
\end{align*}
$$

The calculation requires $$\mathbb{E} [\mu_k]$$ and $$\mathbb{E} [\mu_k^2]$$, computable from the variational Gaussian on the $$k^{th}$$ mixture component: $$\mathbb{E} [\mu_k] = m_k$$ and $$\mathbb{E} [\mu_k^2] = s_k^2 + m_k^2$$. Thus the variational update for the $$i^{th}$$ cluster assignment, expressed as a function of the variational parameters for the mixture component only, is:

$$
\varphi_{ik} \propto \exp \left\{ x_i \, \mathbb{E} [ \mu_k; m_k, s_k^2] - \frac{1}{2} \mathbb{E}[ \mu^2_k; m_k, s^2_k] \right\}
\label{eq_algo_first_step}
$$

Substituting these expectations, and normalizing over the $$K$$ clusters so that the probabilities $$\varphi_{i1}, ..., \varphi_{iK}$$ sum to 1, the update is the [softmax](https://en.wikipedia.org/wiki/Softmax_function) of the $$K$$ exponents:

$$
\varphi_{ik} = \frac{\exp \{ x_i m_k - (s_k^2 + m_k^2)/2 \}}{\sum_{l=1}^K \exp \{ x_i m_l - (s_l^2 + m_l^2)/2 \}}
$$

As an example, take $$K = 2$$ clusters whose factors have the means $$m = (0, 3)$$ and the variances $$s^2 = (1, 1)$$, and a point $$x_i = 2$$. The two exponents are $$-0.5$$ and $$1$$, so $$\varphi_i = (0.18, 0.82)$$: the point is closer to the second mean, which takes 82% of it. In practice, since $$x_i m_k - m_k^2/2 = x_i^2/2 - (x_i - m_k)^2/2$$, and $$x_i^2/2$$ is the same for all clusters, the exponent of cluster $$k$$ is the log-likelihood of the point under the mean $$m_k$$, up to a constant, minus $$s_k^2/2$$: of two clusters at the same distance from the point, the one whose mean is less certain gets the lower probability.

### Update 2: Variational update for cluster mean

The derivation now turns to the variational density $$q(\mu_k; m_k, s^2_k)$$ in equation \ref{eq_variational_family}. Equation \ref{eq_CAVI} (the exponentiated log of the joint) is again used for the mean values $$\mu_k$$:

$$
q(\mu_k) \propto \exp \{ \log p(\mu_k) + \sum_{i=1}^n \mathbb{E} [ \log p(x_i \mid c_i, \mu); \varphi_i, m_{\neq k}, s^2_{\neq k} ] \}
$$

where, as before, $$\varphi_{ik}$$ represents the probability that the $$i^{th}$$ observation comes from the $$k^{th}$$ cluster, and $$c_i$$ is a one-hot (or indicator) vector, so that $$\mathbb{E}[c_{ik}; \varphi_i] = \varphi_{ik}$$. The computation of the log probability follows equations 28 to 33 of [the paper](https://arxiv.org/pdf/1601.00670.pdf), and shows that $$\log q(\mu_k)$$ is a quadratic function of $$\mu_k$$:

$$
\begin{align*}
\log q(\mu_k) & = \log p(\mu_k) + \sum_i \mathbb{E} [ \log p(x_i \mid c_i, \mu); \varphi_i, m_{\neq k}, s^2_{\neq k}] + const \\
  & = \log p(\mu_k) + \sum_i \mathbb{E} [ c_{ik} \log p(x_i \mid \mu_k); \varphi_i] + const \\
  & \qquad \text{(only the term of cluster } k \text{ depends on } \mu_k \text{)} \\
  & = -\frac{\mu_k^2}{2 \sigma^2} + \sum_i \varphi_{ik} \log p(x_i \mid \mu_k) + const \\
  & \qquad \text{(prior } \mathcal{N}(0, \sigma^2) \text{, and } \mathbb{E}[c_{ik}; \varphi_i] = \varphi_{ik} \text{)} \\
  & = -\frac{\mu_k^2}{2 \sigma^2} - \sum_i \varphi_{ik} \frac{(x_i - \mu_k)^2}{2} + const \\
  & \qquad \text{(log of normal distribution)} \\
  & = -\frac{\mu_k^2}{2 \sigma^2} + \sum_i \varphi_{ik} \left( x_i \mu_k - \frac{\mu_k^2}{2} \right) + const \\
  & \qquad \text{(decomposed square; } x_i^2 \text{ does not depend on } \mu_k \text{)} \\
  & = \left( \sum_i \varphi_{ik} x_i \right) \mu_k - \left( \frac{1}{2 \sigma^2} + \frac{1}{2} \sum_i \varphi_{ik} \right) \mu_k^2 + const.
\end{align*}
$$

In practice, this means that the CAVI optimal variational density of $$\mu_k$$ is a member of the exponential family of distributions with sufficient statistics $$\{\mu_k, \mu_k^2\}$$ and natural parameters $$\left\{\sum_{i=1}^n \varphi_{ik} x_i, -\frac{1}{2 \sigma^2} - \frac{1}{2}\sum_{i=1}^n \varphi_{ik} \right\}$$, i.e. a Gaussian. A detailed discussion of the exponential family, sufficient statistics and natural parameters is available in the previous [post]({{ site.baseurl }}{% post_url 2019-03-20-exponential-family %}). The updates for the mean and the variance can now be expressed as:

$$
\begin{align*}
0 & = \frac{d\, \log q(\mu_k)}{d\mu_k} \\
0 & = \sum_i \varphi_{ik}x_i - 2 \left( \frac{1}{2 \sigma^2} + \frac{1}{2} \sum_i \varphi_{ik} \right) \mu_k \\
\hat{\mu_k} & = \frac{\sum_i \varphi_{ik} x_i}{1/\sigma^2 + \sum_i \varphi_{ik}}\\ 
\end{align*}
$$

The final variable $$\hat{\mu_k}$$ is referred to as $$m_k$$, in line with the paper and to avoid confusion. Similarly, the update of the variance follows from the coefficient of $$\mu_k^2$$, which is $$-\frac{1}{2 s_k^2}$$ for a Gaussian:

$$
s^2_k = \frac{1}{1/\sigma^2 + \sum_i \varphi_{ik}}\\
\label{eq_algo_second_step}
$$

These updates relate closely to the complete conditional of $$\mu_k$$, the Gaussian posterior of the mean given the points of cluster $$k$$: the variational update is a weighted complete conditional, where each point is weighted by its probability $$\varphi_{ik}$$ of coming from cluster $$k$$ (section 3.2 of the review paper). In practice, the mean $$m_k$$ is the weighted average of the points, pulled towards the prior mean 0 by the term $$1/\sigma^2$$, and the variance $$s_k^2$$ falls as the cluster gains points, as its mean is then better known.

As an example, two points $$x_1 = x_2 = 3$$ that come from cluster $$k$$ with probability $$\varphi_{ik} = 1$$, with a prior variance $$\sigma^2 = 100$$, give $$m_k = 6 / 2.01 = 2.985$$, slightly below their average of 3, and $$s_k^2 = 1 / 2.01 = 0.50$$, half the unit variance of a single observation.

### Algorithm and Visualization

Putting it all together, the algorithm is as follows:

While the ELBO has not converged:

1. for all $$i \in \{1, ..., n\}$$, set (equation \ref{eq_algo_first_step}):

   $$
   \varphi_{ik} \propto \exp \left\{ x_i \, \mathbb{E} [ \mu_k; m_k, s_k^2] - \frac{1}{2} \mathbb{E}[ \mu^2_k; m_k, s^2_k] \right\}
   $$

2. for $$k \in \{1, ..., K\}$$, set (equation \ref{eq_algo_second_step}):

   $$
   \begin{align*}
   m_k & \leftarrow \frac{\sum_i \varphi_{ik} x_i}{1/\sigma^2 + \sum_i \varphi_{ik}} \\
   s^2_k & \leftarrow \frac{1}{1/\sigma^2 + \sum_i \varphi_{ik}}
   \end{align*}
   $$

3. compute $$ELBO(m, s^2, \varphi)$$ (equation \ref{eq_algo_elbo})

The algorithm is iterative: each update can only increase the ELBO, and the algorithm stops when this increase falls below a threshold, once each Gaussian fits its data points. Note that the ELBO is in general not convex, so CAVI only converges to a local optimum, which depends on the initialization. The concept can be easily understood by looking at the illustration of several iterations of the fitting of Gaussian models to five clusters of points:

{: .picture}
<img width="59.8%" height="59.8%" src="{{ site.assets }}/variational-inference-gmm/gmm_iterations.svg"/>

{: .caption}
The CAVI algorithm on a Gaussian mixture model, for a dataset of 2D points from five color-coded clusters: the initialization and the variational factors after 20, 28, 35 and 50 iterations. The ellipses are 2σ contours of the variational approximating factors. The figure uses the multivariate (2D) version of the model; the derivation above uses univariate Gaussians. Adapted from: [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)

The picture above shows that the Gaussians start near the center of the dataset, with a wide variance, and, as the iterations progress, each one moves towards a cluster and narrows around it.

### CAVI and the EM Algorithm

The two updates of CAVI look like the two steps of the EM algorithm, which fits the means of the same mixture by maximum likelihood. The update of the assignments plays the role of the E-step, which computes the probability (the *responsibility*) of each cluster for each point, and the update of the means plays the role of the M-step, which sets each mean to the average of the points, weighted by these probabilities. The difference is in what is estimated: EM treats the means as fixed parameters and computes a single value of each (a point estimate), while CAVI treats them as latent variables with a prior, and approximates their posterior by a Gaussian of variance $$s_k^2$$. EM also maximizes the ELBO, but with the exact posterior of the assignments given the means as $$q$$, for which the ELBO equals the log-likelihood $$\log p(x \mid \mu)$$ (section 2.2 of the review paper).

This difference shows in the updates: in EM, the update of the assignments has no term $$-s_k^2/2$$, as each mean is a single value, and the update of the means has no term $$1/\sigma^2$$, as the means have no prior. As an example, a point at $$x_i = 1.5$$, halfway between the means $$m = (0, 3)$$, is split as $$(0.5, 0.5)$$ by EM. If the first cluster holds 1 point and the second 10 points, update 2 gives them the variances $$s^2 = (0.99, 0.10)$$ (with $$\sigma^2 = 100$$, as before), and CAVI splits the point as $$(0.39, 0.61)$$, in favor of the cluster whose mean is better known.

With a flat prior ($$\sigma^2 \rightarrow \infty$$) and many points in each cluster ($$s_k^2 \rightarrow 0$$), the updates of CAVI become those of EM. Both algorithms converge to a local optimum, and both read every data point at each iteration, the cost that SVI removes.

## Stochastic Variational Inference

The coordinate ascent variational inference method presented does not scale well to very large datasets.
The underlying reason is that CAVI requires iterating between re-analyzing every data point in the dataset and re-estimating its hidden structure. This is inefficient for large datasets, because it requires a full pass through the data at each iteration.

In the Gaussian mixture model, the means $$\mu_k$$ are **global** latent variables, shared by all the data points, while each assignment $$c_i$$ is a **local** latent variable, which only governs the point $$x_i$$ (section 4.2 of the review paper). Each iteration of CAVI updates the $$n$$ local factors, with $$K$$ exponents for each point, and then the $$K$$ global factors, whose updates sum over the $$n$$ points: it costs $$O(nK)$$ operations for a single update of the means.

Another famous method is [Stochastic Variational Inference](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) (SVI), introduced by Hoffman, Blei, Wang and Paisley in 2013. It uses a more efficient algorithm based on stochastic optimization ([Robbins and Monro, 1951](https://projecteuclid.org/euclid.aoms/1177729586)), a technique that follows noisy estimates of the gradient of the objective.

SVI applies the idea of [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to variational inference: it climbs the ELBO with steps along a gradient estimated on a single mini-batch at each iteration, which allows the algorithm to scale to the large datasets that are encountered in modern machine learning. Unlike the usual stochastic gradient descent, it follows the **natural gradient** of the ELBO instead of its usual (Euclidean) gradient.

The natural gradient ([Amari, 1998](https://doi.org/10.1162/089976698300017746)) measures the length of a step by how much it changes the distribution, instead of how much it changes its parameters. In the Gaussian mixture, a step of 0.1 on the mean $$m_k$$ changes the factor $$q(\mu_k)$$ by a KL divergence of $$0.1^2 / (2 s_k^2)$$: 0.5 for a cluster of 100 points ($$s_k^2 \approx 0.01$$, with $$\sigma^2 = 100$$), and 0.00005 for a cluster with no points ($$s_k^2 = \sigma^2 = 100$$). The usual gradient treats both steps alike, while the natural gradient, the usual gradient multiplied by the inverse of the [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) matrix of $$q$$, takes steps that change the distribution by the same amount in every direction, measured by the symmetrized KL divergence (section 4.3 of the review paper).

For conditionally conjugate models, whose complete conditionals are in the exponential family, such as the Gaussian mixture, the natural gradient has a simple form. The factor $$q(\mu_k)$$, a Gaussian of mean $$m_k$$ and variance $$s_k^2$$, has the natural parameters $$\{ m_k / s_k^2, -1/(2 s_k^2) \}$$, which update 2 sets to $$\left\{ \sum_i \varphi_{ik} x_i, -\frac{1}{2\sigma^2} - \frac{1}{2} \sum_i \varphi_{ik} \right\}$$. The natural gradient of the ELBO with respect to these parameters is the difference between their coordinate ascent update and their current value (Eq. 52 in the review paper). Thus, a step of size $$\rho_t$$ along it is a weighted average of the current value and the update, and a step of size 1 is the CAVI update itself.

SVI computes this update on a mini-batch $$\mathcal{B}_t$$ of $$B$$ points instead of the whole dataset, as if the dataset held $$n / B$$ copies of the mini-batch, which gives an unbiased estimate of it. At each iteration $$t$$:

1. sample a mini-batch $$\mathcal{B}_t$$ of $$B$$ points from the dataset;
2. for all $$i \in \mathcal{B}_t$$, set $$\varphi_{ik}$$ with update 1 (equation \ref{eq_algo_first_step});
3. for $$k \in \{1, ..., K\}$$, update the natural parameters of $$q(\mu_k)$$, from which the new $$m_k$$ and $$s_k^2$$ follow:

   $$
   \begin{align*}
   \frac{1}{s_k^2} & \leftarrow (1 - \rho_t) \frac{1}{s_k^2} + \rho_t \left( \frac{1}{\sigma^2} + \frac{n}{B} \sum_{i \in \mathcal{B}_t} \varphi_{ik} \right) \\
   \frac{m_k}{s_k^2} & \leftarrow (1 - \rho_t) \frac{m_k}{s_k^2} + \rho_t \frac{n}{B} \sum_{i \in \mathcal{B}_t} \varphi_{ik} x_i
   \end{align*}
   $$

where the step size $$\rho_t$$ decreases with the iterations, under the conditions of Robbins and Monro, $$\sum_t \rho_t = \infty$$ and $$\sum_t \rho_t^2 < \infty$$: for example $$\rho_t = (t + \tau)^{-\kappa}$$, with a *forgetting rate* $$\kappa \in (0.5, 1]$$ and a *delay* $$\tau \ge 0$$ (Eq. 26 in the SVI paper). The first condition lets the steps travel any distance, and the second makes the noise of the mini-batches average out.

SVI thus iterates between subsampling the data and adjusting the hidden structure based only on the subsample, making it more efficient than traditional variational inference. Each iteration costs $$O(BK)$$ operations instead of $$O(nK)$$: for $$n = 10^6$$ points and mini-batches of $$B = 100$$ points, SVI updates the means $$10^4$$ times in the time that CAVI takes to update them once. For this reason, several ML libraries such as [Pyro](https://pyro.ai/examples/svi_part_i.html) use SVI, in a more general form: they estimate the usual gradient of the ELBO by sampling from $$q$$, which also covers models without closed-form updates, and follow it with optimizers such as Adam.

The derivation of SVI for the general class of conditionally conjugate models, of which the Gaussian mixture is an example, is in section 4.3 of the paper [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf). The SVI paper applies it to topic models of millions of documents, such as 1.8 million articles of the New York Times and 3.8 million articles of Wikipedia.

## Comparison of MCMC, CAVI and SVI

The following table sums up the three methods of this post, with the costs of a Gaussian mixture model of $$n$$ points and $$K$$ clusters:

| Method | Approximation of the posterior | Cost of one update | When to use it |
|---|---|---|---|
| MCMC (e.g. Metropolis) | Samples of the true posterior, of any form, which converge to it as the number of samples grows | A pass over the data for each sample, and many samples | Small datasets, where precise estimates of the posterior are worth a high computational cost |
| CAVI | A member of the mean-field family, at a local optimum of the reverse KL, which underestimates the posterior variance | $$O(nK)$$: a pass over the data for each update of the means | Models with closed-form updates, on datasets small enough for a full pass at each iteration |
| SVI | A member of the same family as CAVI, reached with noisy steps along the natural gradient | $$O(BK)$$: a mini-batch of $$B$$ points for each update of the means | The same models, on large datasets of millions of points |

To find more about the topic, chapter 10 of Chris Bishop's PRML book covers variational inference in more depth, including a variational mixture of Gaussians that also infers the mixing coefficients and the covariances of the components (section 10.2).
