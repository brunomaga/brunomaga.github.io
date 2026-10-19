---
layout: post
title:  "Variational Inference (ELBO, CAVI) and Gaussian Mixture Models"
categories: ["Machine Learning"]
tags: ["variational inference", "ELBO", "GMM", "MCMC"]
---

In 1987, the paper [A Mean Field Theory Learning Algorithm for Neural Networks](https://www.complex-systems.com/abstracts/v01_i05_a06/) sped up the training of a neural network by replacing its slow stochastic sampling with the deterministic equations of mean-field theory, a method of statistical physics. This idea was then extended to a wide class of probabilistic models, as presented in the tutorial [An Introduction to Variational Methods for Graphical Models](https://doi.org/10.1023/A:1007665907178) in 1999.

A [previous post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}) about Bayesian inference showed that its goal is to compute the likelihood of the observed data and the mode of the density of the likelihood, marginal distribution and conditional distributions. Recall the formulation of the **posterior** of the [latent variable](https://en.wikipedia.org/wiki/Latent_and_observable_variables) $$z$$ given the observations $$x$$, derived from the Bayes rule, with and without the normalization term:

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

The inference problem is to compute the conditional probability of the latent variables $$z$$ given the observations $$x$$ in $$X$$, i.e. to condition on the data and compute the posterior $$p(z \mid x)$$. For many models, this evidence integral has no closed-form solution or requires exponential time to compute. 

**There are several approaches to inference**, comprising algorithms for exact inference (Brute force, the [elimination algorithm](https://en.wikipedia.org/wiki/Variable_elimination), Message passing (sum-product algorithm, [Belief propagation](https://en.wikipedia.org/wiki/Belief_propagation)), [Junction tree algorithm](https://en.wikipedia.org/wiki/Junction_tree_algorithm)). Other algorithms perform approximate inference (Loopy belief propagation, [Variational (Bayesian) inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods), Stochastic simulation / sampling / Markov Chain Monte Carlo). Approximate methods are needed because, in many cases, the posterior distribution cannot be computed directly, i.e. the posterior has an **intractable** form (often involving integrals) that cannot be (easily) computed.

**This post focuses on the simplest approach to Variational Inference, based on the mean-field approximation**. After a detour on Markov Chain Monte Carlo, it derives the Evidence Lower Bound (ELBO) and the coordinate ascent variational inference (CAVI) algorithm, applies it to a Gaussian mixture model, and ends with Stochastic Variational Inference.

## Detour: Markov Chain Monte Carlo

Before moving on to Variational Inference, it helps to understand the place of VI in this type of inference. For many years, the dominant approach was [Markov chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo). A simple high-level understanding of MCMC follows from the name itself.

[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) methods are a simple way of estimating parameters by generating random numbers. A simple example is to compute the area of a circle by generating random 2D coordinates within the bounding square around the circle. The value of $$\pi$$ or the area of the circle is then estimated from the proportion of generated datapoints that fall inside vs outside the circle in the square.

A [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain) is a stochastic model describing a sequence of events in which the probability of moving to the next state depends *only* on the current state and not on the previous ones. An example is the probability of the next character in a word, for all *27* possible characters, given the current character.

Therefore, MCMC methods are used to approximate the distribution (state) of parameters at the next iteration by random sampling from a probabilistic space defined by the current state. MCMC is based on the assumption that the prior-likelihood product $$ p(x \mid z) \, p(z)$$ can be computed, i.e. is known, for a given $$z$$. However, this can be done without any knowledge of the function of $$z$$. But, from a mathematical perspective (eq. \ref{eq_bayes}), the posterior density is expressed as:

$$
\begin{align*}
p (z \mid x) & = \frac{p(z) \, p(x \mid z)}{p(x)} \\
  & \qquad \text{(Bayes equation)} \\
             & = \frac{p(z) \, p(x \mid z)}{\int p(z) p(x \mid z) \,dz} \\
               & \qquad \text{(marginalizing } z \text{)} \\
\end{align*}
$$

where the top term of the division can be computed, but the bottom one is unknown or intractable. Since the bottom term is a *normalizer* (i.e. it guarantees that the posterior sums to 1), it can be discarded, leading to the expression $$ p (z \mid x) \propto p(z) \, p(x \mid z)$$.

Therefore, the rationale is to take several samples of the latent variables (for example, the mean $$\mu$$ or the variance $$\sigma^2$$ of a normal distribution) from the current space. The model is then updated with the values that explain the data better than the values at the current state, i.e. that have a higher posterior probability. A simpler way to express this is to compute the **acceptance ratio** of the proposed over the current posterior (which cancels out the term $$p(x)$$ that could not be computed):

$$
  \frac{ p(z \mid x) }{ p(z_0 \mid x) }
=  \frac{ \hspace{0.5cm}{ } \frac{p(x \mid z) \, p(z)}{ p(x) } \hspace{0.5cm} } { \,\, \frac{p(x \mid z_0) \, p(z_0)}{ p(x) } \,\, }
= \frac{ p(x \mid z) \, p(z) }{ p(x \mid z_0) \, p(z_0)}
$$

and accept the proposed state $$z$$ with probability $$\min(1, \text{ratio})$$, otherwise keep the current state $$z_0$$ (the [Metropolis algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), with a symmetric proposal distribution). 

The main advantage of MCMC is that it provides an approximation of the *true* posterior. However, there are many disadvantages in handling an exact posterior that does not follow a parametric distribution (such as being hard to interpret in high dimensions). Moreover, it requires massive computational power for problems with large latent spaces. That is where Variational Inference enters the picture: it is faster and works well if the parametric distribution of the posterior is known, but it is sometimes *over-confident* when the posterior does not exactly match the approximation.

So as a golden rule, variational methods are the default choice. If in doubt, the results should be double-checked against an MCMC estimate of the posterior, which is also approximate but converges to the true posterior as the number of samples grows. In practice, the paper [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) suggests variational inference for large datasets and for the fast exploration of many models, and MCMC for smaller datasets, where a higher computational cost buys more precise samples.

The theory behind MCMC is covered in the paper [An Introduction to MCMC for Machine Learning]({{ site.assets }}/variational-inference-gmm/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf), and a practical explanation with code is available in [Thomas Wiecki's post](https://twiecki.io/blog/2015/11/10/mcmc-sampling/).

## Variational Inference

**The idea behind Variational Inference (VI) methods is to find a parametric distribution $$q(z)$$ that is an approximation of an intractable/non-parametric posterior $$p(z \mid x)$$**. To do that, the first step is to define a function that gives the approximation/similarity/closeness/proximity between the two distributions $$p$$ and $$q$$. So the starting point is the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between two distributions $$f(x)$$ and $$g(x)$$, defined as:

$$
KL (f(x) || g(x)) =  \int f(x) \log \frac{f(x)}{g(x)} dx= \mathbb{E} \left[ \log \frac{f(x)}{g(x)} \right] = \mathbb{E} \left[ \log f(x) \right] - \mathbb{E} \left[ \log g(x) \right]
$$ 

when applied to this use case:

$$
KL (q(z) || p(z \mid x)) =  \int q(z) \log \frac{q(z)}{p(z \mid x)} dz= \mathbb{E} \left[ \log \frac{q(z)}{p( z\mid x)} \right]
\label{eq_KLdiv}
$$ 

Note that the **KL divergence is a dissimilarity function and not a metric**, as it is not symmetric. Also, there are other divergences besides the KL divergence, part of the [f-divergence family](https://en.wikipedia.org/wiki/F-divergence).

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

There are several options to solve the previous optimization problem: [coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent) in all variables $$z_i$$ (slow, and requires deriving the formulation of the gradients by hand), Stochastic Variational Inference (faster), Mean-field approximation, etc.

## Mean-field approximation

So far, the focus was on a single posterior defined by a family with a single distribution $$q$$, whose posterior $$p(z \mid x)$$ is tractable. The analysis now extends to complex families, where the posterior is not tractable.

One way to overcome this is to approximate the posterior probability using a simpler model, for example by assuming that a set of latent variables is independent of the other variables. Or full independence can be enforced among all latent variables given $$x$$. This assumption is known as the **mean-field approximation**.

In practice, this method originates from mean-field theory. From [Wikipedia](https://en.wikipedia.org/wiki/Mean-field_theory): 
> "Mean-field theory [...] studies the behavior of high-dimensional random (stochastic) models by studying a simpler model that approximates the original by averaging over degrees of freedom. [...] The effect of all the other individuals on any given individual is approximated by a single averaged effect, thus reducing a many-body problem to a one-body problem."

More complex scenarios, such as a variational family with dependencies between variables (structured variational inference) or with mixtures of variational families (covered in [Chris Bishop's PRML book]({{ site.resources_permalink }})), are not covered in this post. As a side note, **because the latent variables are assumed to be independent, the mean-field approach that follows usually does not contain the true posterior**.

In this type of Variational Inference, due to the independence of the variables, the variational distribution over the latent variables factorizes as:

$$
q(z) = q(z_1, ..., z_m) = \prod_{j=1}^{m} q(z_j)
\label{eq_posterior_mf}
$$

where each latent variable ($$z_j$$) is governed by its own density $$q(z_j)$$. Note that the variational family does not depend on the input data $$x$$, which does not show up in the equation.

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

and the entropy term can be decomposed using the mean-field variational approximation as:

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

where $$\neq j$$ means *all indices except the $$j^{th}$$*. Now the goal is to derive the coordinate ascent update for a latent variable, while keeping the others fixed, i.e. to compute $$arg\,max_{q_j} \, ELBO$$. The order of the latent variables in the chain rule is arbitrary, so $$z_j$$ can be placed last, and then only the term $$p(z_j \mid z_{\neq j}, x)$$ of the product depends on $$z_j$$:

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

However, this provides the factorization, or the template of the computation, but not the final form (i.e. the application to a specific distribution family) of the optimal $$q_j$$. The form that is chosen influences the complexity or *easiness* of the coordinate update $$q^{\star}(z_j)$$.

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

I.e. the evidence is now a sum over all possible configurations of cluster assignments, with a computational complexity of $$O(K^n)$$, due to the $$K$$ classes that have to be assigned to $$n$$ points.
Each integral in the previous equation can be computed, since the Gaussian prior and likelihood are [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior).
However, the evidence is still computationally intractable due to the $$K^n$$ complexity.

An alternative solution is to use the mean-field approximation method discussed above. As shown before (eq. \ref{eq_posterior_mf}), the mean-field approximation yields approximate posterior probabilities of the form:

$$
q(\mu,c) = \prod_{k=1}^K q(\mu_k; m_k, s^2_k) \prod_{i=1}^n q(c_i; \varphi_i).
\label{eq_variational_family}
$$

Rephrasing what was mentioned before, each latent variable is governed by its own variational factor. The factor $$q(\mu_k; m_k, s^2_k)$$ is a Gaussian distribution on the mean of the $$k^{th}$$ mixture component, with mean $$m_k$$ and variance $$s^2_k$$. The factor $$q(c_i; \varphi_i)$$ is a distribution on the $$i^{th}$$ observation's mixture assignment; its assignment probabilities are a $$K$$-vector $$\varphi_i$$.

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


### Update 2: Variational update for cluster mean

The derivation now turns to the variational density $$q(\mu_k; m_k, s^2_k)$$ in equation \ref{eq_variational_family}. Equation \ref{eq_CAVI} (the exponentiated log of the joint) is again used for the mean values $$\mu_k$$:

$$
q(\mu_k) \propto \exp \{ \log p(\mu_k) + \sum_{i=1}^n \mathbb{E} [ \log p(x_i \mid c_i, \mu); \varphi_i, m_{\neq k}, s^2_{\neq k} ] \}
$$

where, as before, $$\varphi_{ik}$$ represents the probability that the $$i^{th}$$ observation comes from the $$k^{th}$$ cluster, and $$c_i$$ is a one-hot (or indicator) vector. The computation of the log probability follows as:

$$
\log q(\mu_k) = \log p(\mu_k) + \sum_i \mathbb{E} [ \log p(x_i \mid c_i, \mu); \varphi_i, m_{\neq k}, s^2_{\neq k}] + const\\
$$

This reduction is omitted here; the details are in equations 28-32 of [the paper](https://arxiv.org/pdf/1601.00670.pdf). This calculation leads to:

$$
\log q(\mu_k) = (\sum_i \varphi_{ik} x_i) \mu_k - (\frac{1}{2 \sigma^2} + \frac{1}{2} \sum_i \varphi_{ik} ) \mu_k^2 + const.\\
$$

In practice, this means that the CAVI optimal variational density of $$\mu_k$$ is a member of the Exponential Family of Distributions with sufficient statistics $$\{\mu_k, \mu_k^2\}$$ and natural parameters $$\left\{\sum_{i=1}^n \varphi_{ik} x_i, -\frac{1}{2 \sigma^2} - \frac{1}{2}\sum_{i=1}^n \varphi_{ik} \right\}$$. 
 A detailed discussion of the exponential family, sufficient statistics and natural parameters is available in the previous [post]({{ site.baseurl }}{% post_url 2019-03-20-exponential-family %}). The updates for the mean and the variance can now be expressed as:

$$
\begin{align*}
0 & = \frac{d\, \log q(\mu_k)}{d\mu_k} \\
0 & = \sum_i \varphi_{ik}x_i - 2 \left( \frac{1}{2 \sigma^2} + \frac{1}{2} \sum_i \varphi_{ik} \right) \mu_k \\
\hat{\mu_k} & = \frac{\sum_i \varphi_{ik} x_i}{1/\sigma^2 + \sum_i \varphi_{ik}}\\ 
\end{align*}
$$

The final variable $$\hat{\mu_k}$$ is referred to as $$m_k$$, in line with the paper and to avoid confusion. Similarly, the update of the variance follows from the coefficient of $$\mu_k^2$$, which is $$-\frac{1}{2 s_k^2}$$ for a Gaussian:

$$
s^2_k = \hat{\sigma}^2 = \frac{1}{1/\sigma^2 + \sum_i \varphi_{ik}}\\
\label{eq_algo_second_step}
$$

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
   s^2_k & \leftarrow \hat{\sigma}^2 = \frac{1}{1/\sigma^2 + \sum_i \varphi_{ik}}
   \end{align*}
   $$

3. compute $$ELBO(m, s^2, \varphi)$$ (equation \ref{eq_algo_elbo})

The algorithm is iterative and will converge on a certain loss threshold (ELBO), based on the fitting of each Gaussian to the data points. Note that the ELBO is in general not convex, so CAVI only converges to a local optimum, which depends on the initialization. The concept can be easily understood by looking at the illustration of several iterations of the fitting of Gaussian models to five clusters of points:

{: .picture}
<img width="59.8%" height="59.8%" src="{{ site.assets }}/variational-inference-gmm/gmm_iterations.svg"/>

{: .caption}
The CAVI algorithm on a Gaussian mixture model, for a dataset of 2D points from five color-coded clusters: the initialization and the variational factors after 20, 28, 35 and 50 iterations. The ellipses are 2σ contours of the variational approximating factors. The figure uses the multivariate (2D) version of the model; the derivation above uses univariate Gaussians. Adapted from: [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)

The picture above shows that the Gaussians start near the center of the dataset, with a wide variance, and, as the iterations progress, each one moves towards a cluster and narrows around it.


## Stochastic Variational Inference

The coordinate ascent variational inference method presented does not scale well to very large datasets.
The underlying reason is that CAVI requires iterating between re-analyzing every data point in the dataset and re-estimating its hidden structure. This is inefficient for large datasets, because it requires a full pass through the data at each iteration.

Another famous method is [Stochastic Variational Inference](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) (SVI), introduced by Hoffman, Blei, Wang and Paisley in 2013. It uses a more efficient algorithm based on stochastic optimization ([Robbins and Monro, 1951](https://projecteuclid.org/euclid.aoms/1177729586)), a technique that follows noisy estimates of the gradient of the objective.

SVI is in fact [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) applied to variational inference. It allows the algorithm to scale to large datasets that are encountered in modern machine learning, since the ELBO and its gradient can be estimated on a single mini-batch at each iteration.

SVI iterates between subsampling the data and adjusting the hidden structure based only on the subsample, making it more efficient than traditional variational inference. For this reason, several ML libraries such as [Pyro](https://pyro.ai/examples/svi_part_i.html) use SVI.

The details of SVI are omitted here; they are in section 4.3 of the paper [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf).

To find more about the topic, chapter 10 of Chris Bishop's PRML book covers variational inference in more depth, including a variational mixture of Gaussians that also infers the mixing coefficients and the covariances of the components (section 10.2).
