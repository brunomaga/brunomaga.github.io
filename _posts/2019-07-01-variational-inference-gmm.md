---
layout: post
title:  "Variational Inference (ELBO, Mean-Field Approximation, CAVI) and Gaussian Mixture Models"
categories: ["Machine Learning"]
tags: ["variational inference", "ELBO", "GMM", "MCMC"]
---

The goal of Bayesian inference is to compute the likelihood of the observed data and the mode of the density of the likelihood, marginal distribution and conditional distributions. Recall the formulation of the **posterior** of the latent variable $$z$$ given the observations $$x$$, derived from the Bayes rule, with and without the normalization term:

$$
p (z \mid x) = \frac{p(z) \, p(x \mid z)}{p(x)} \propto p(z) \, p(x \mid z)
\label{eq_bayes}
$$

read as *the posterior is proportional to the prior times the likelihood*. 
The Bayes rule derives from the definition of [**conditional probability**](https://en.wikipedia.org/wiki/Conditional_probability#Definition) $$p(z\mid x)$$, expressed as:

$$
p (z \mid x ) = \frac{p(z,x)}{p(x)}
\label{eq_conditional}
$$

where the denominator is the marginal density of $$x$$ (i.e. the observations), also called the **evidence**. It is calculated by **marginalizing** out the latent variables $$z$$ from the joint distribution:

$$
p(x) = \int p(z,x) dz
\label{eq_joint}
$$

The inference problem is to compute the conditional probability of the latent variables $$z$$ given the observations $$x$$ in $$X$$, i.e. to condition on the data and compute the posterior $$p(z \mid x)$$. For many models, the evidence integral has no closed-form solution or requires exponential time to compute. 

**There are several approaches to inference**. Algorithms for exact inference include Brute force, the elimination algorithm, Message passing (sum-product algorithm, Belief propagation) and the Junction tree algorithm. Algorithms for approximate inference include Loopy belief propagation, Variational (Bayesian) inference and Stochastic simulation / sampling / Markov Chain Monte Carlo. Approximate methods are needed because in many cases the posterior distribution cannot be computed directly, i.e. the posterior has an **intractable** form --- often involving integrals --- that cannot be (easily) computed. **This post focuses on the simplest approach to Variational Inference, based on the mean-field approximation**.

## Detour: Markov Chain Monte Carlo

Before moving on to Variational Inference, it helps to understand the place of VI in this type of inference. For many years, the dominant approach was [Markov chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo). The name itself gives a high-level understanding of MCMC.

[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) methods estimate parameters by generating random numbers. For example, to compute the area of a circle, random 2D coordinates are generated within the square that bounds the circle. The proportion of points that fall inside vs outside the circle gives an estimate of the area of the circle, or of the value of $$\pi$$.

A [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain) is a stochastic model describing a sequence of events in which the probability of moving to the next state depends *only* on the current state and not on the previous ones. An example is the probability of the next character in a word, for all *27* possible characters, given the current character.

MCMC methods approximate the distribution (state) of parameters at the next iteration by random sampling from a probabilistic space defined by the current state. MCMC assumes that the prior-likelihood product $$ P(x \mid z) \, P(z)$$ can be computed, i.e. is known, for a given $$z$$. This does not require any knowledge of the function of $$z$$. From a mathematical perspective (eq. \ref{eq_bayes}), the posterior density is expressed as:

$$
\begin{align*}
p (z \mid x) & = \frac{p(z) \, p(x \mid z)}{p(x)} \\
  & \qquad \text{(Bayes equation)} \\
             & = \frac{p(z) \, p(x \mid z)}{\int p(z) p(x \mid z) \,dz} \\
               & \qquad \text{(marginalizing } z \text{)} \\
\end{align*}
$$

where the numerator can be computed, but the denominator is unknown or intractable. Since the denominator is a *normalizer* --- i.e. it guarantees that the posterior sums to 1 --- it can be discarded, leading to $$ p (z \mid x) \propto p(z) \, p(x \mid z)$$. The rationale is to take several samples of the latent variables (for example, the mean $$\mu$$ or the variance $$\sigma^2$$ of a normal distribution) from the current space, and to update the model with the values that explain the data better than the values at the current state, i.e. that have a higher posterior probability. In practice, this is done with the acceptance ratio of the proposed over the current posterior, which cancels out the term $$P(x)$$ that cannot be computed:

$$
  \frac{ P(z \mid x) }{ P(z_0 \mid x) }
=  \frac{ \hspace{0.5cm}{ } \frac{P(x \mid z) \, P(z)}{ P(x) } \hspace{0.5cm} } { \,\, \frac{P(x \mid z_0) \, P(z_0)}{ P(x) } \,\, }
= \frac{ P(x \mid z) \, P(z) }{ P(x \mid z_0) \, P(z_0)}
$$

and accept the proposed state $$z$$ with probability $$\min(1, \text{ratio})$$, otherwise keep the current state $$z_0$$ (Metropolis algorithm, with a symmetric proposal distribution). 

The main advantage of MCMC is that it approximates the *true* posterior. However, an exact posterior that does not follow a parametric distribution has many disadvantages, such as being hard to interpret in high dimensions. MCMC also requires massive computational power for problems with large latent spaces. Variational Inference addresses these issues: it is faster and works well if the parametric distribution of the posterior is known, but it is sometimes *over-confident* when the posterior does not exactly match the approximation.

As a rule of thumb, variational methods are the default choice, but in case of doubt the results should be checked against an MCMC estimate of the posterior. MCMC is also approximate, but it converges to the true posterior as the number of samples grows.

The theory behind MCMC is covered in the paper [An Introduction to MCMC for Machine Learning]({{ site.assets }}/variational-inference-gmm/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf), and a practical explanation with code is available in [Thomas Wiecki's post](https://twiecki.io/blog/2015/11/10/mcmc-sampling/).

## Variational Inference

**The idea behind Variational Inference (VI) methods is to find a parametric distribution $$q(z)$$ that is an approximation of an intractable/non-parametric posterior $$p(z \mid x)$$**. This requires a function that measures the approximation/similarity/closeness/proximity between the two distributions $$p$$ and $$q$$. A common choice is the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), defined between two distributions $$f(x)$$ and $$g(x)$$ as:

$$
KL (f(x) || g(x)) =  \int f(x) \log \frac{f(x)}{g(x)} dx= \mathbb{E} \left[ \log \frac{f(x)}{g(x)} \right] = \mathbb{E} \left[ \log f(x) \right] - \mathbb{E} \left[ \log g(x) \right]
$$ 

and, applied to the problem at hand:

$$
KL (q(z) || p(z \mid x)) =  \int q(z) \log \frac{q(z)}{p(z \mid x)} dz= \mathbb{E} \left[ \log \frac{q(z)}{p( z\mid x)} \right]
\label{eq_KLdiv}
$$ 

Note that the **KL divergence is a dissimilarity function and not a metric**, as it is not symmetric. Other divergences besides the KL divergence also exist, as part of the [f-divergence family](https://en.wikipedia.org/wiki/F-divergence).

Equation \ref{eq_KLdiv} includes the true posterior $$p(z \mid x)$$, which is exactly the unknown value. Therefore, it cannot be minimized directly. Instead, the optimization targets a function that is *equal to its negative (up to a constant)*, known as the Evidence Lower Bound.

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

This replaces the original $$KL$$-divergence formulation, which cannot be minimized, by an expression that can be optimized: minimizing the KL divergence is equivalent to maximizing the ELBO. On the right-hand side of the expression:
- The KL divergence is minimized over $$q$$, so the term $$\log p(x)$$ can be ignored as it does not depend on $$q$$;
- All terms inside $$ELBO(q,p)$$ are known, as $$p(z,x)$$ is the prior times the likelihood $$p(z,x) = p(z) \, p(x \mid z)$$, which are the inputs of the problem;
- The KL divergence cannot be negative. Thus $$\log p(x) \ge ELBO$$, i.e. the ELBO is never larger than the log evidence $$\log p(x)$$. This explains the name **Evidence Lower Bound**.

The ELBO is also commonly written in terms of two expectations:

$$
ELBO(q,p) = \int q(z) \log \frac{p(z, x)}{q(z)} dz =  \int q(z) \log p(z, x) dz - \int q(z) \log q(z) dz  = \mathbb{E} [\log p(x,z)] - \mathbb{E}[\log q(z)]
\label{eq_elbo}
$$

There are several options to solve this optimization problem: coordinate descent in all variables $$z_i$$ (slow, and requires deriving the gradients by hand), Stochastic Variational Inference (faster), Mean-field approximation, etc.

## Mean-field approximation

So far, the analysis assumed a single posterior defined by a family with a single distribution $$q$$, whose posterior $$p(z \mid x)$$ is tractable. This section extends the analysis to complex families, where the posterior is not tractable.

One solution is to approximate the posterior probability with a simpler model, for example by assuming that a set of latent variables is independent of the other variables. Another is to enforce full independence among all latent variables given $$x$$. This assumption is known as the **mean-field approximation**.

This method originates from mean-field theory. From [Wikipedia](https://en.wikipedia.org/wiki/Mean-field_theory): 
> "Mean-field theory [...] studies the behavior of high-dimensional random (stochastic) models by studying a simpler model that approximates the original by averaging over degrees of freedom. [...] The effect of all the other individuals on any given individual is approximated by a single averaged effect, thus reducing a many-body problem to a one-body problem."

More complex scenarios, such as a variational family with dependencies between variables (structured variational inference) or with mixtures of variational families (covered in [Chris Bishop's PRML book]({{ site.resources_permalink }})), are not covered in this post. Note that **because the latent variables are assumed to be independent, the mean-field approach that follows usually does not contain the true posterior**.

In this type of Variational Inference, the latent variables are independent, so the variational distribution over them factorizes as:

$$
q(z) = q(z_1, ..., z_m) = \prod_{j=1}^{m} q(z_j)
\label{eq_posterior_mf}
$$

where each latent variable $$z_j$$ is governed by its own density $$q(z_j)$$. Note that the variational family does not depend on the input data $$x$$, which does not appear in the equation.

The latent variables can also be partitioned into $$K$$ groups $$z_{G_1}, ..., z_{G_K}$$, and the approximation is then:

$$
q(z) =  q(z_1, ..., z_m) =  q(z_{G_1}, ..., z_{G_K}) = \prod_{j=1}^{K} q_j(z_{G_j}) 
$$

This setup is often called **generalized mean field**, as opposed to **naive mean field**. Each latent variable $$z_j$$ is governed by its own variational factor, the density $$q_j(z_j)$$. A complete formulation also requires the parametric form of each variational factor. In principle, each factor can take any parametric distribution appropriate to the corresponding random variable (e.g. a combination of Gaussian and Categorical distributions).


## Coordinate Ascent Variational Inference 

The main objective is to **optimize the ELBO in mean-field variational inference**, i.e. to choose the variational factors that maximize the ELBO (eq. \ref{eq_elbo}). A common approach is the **coordinate ascent** method, which optimizes the variational approximation of each latent variable $$q_{z_j}$$ while holding the others fixed.

Recall the [probability chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)) for more than two events $$X_1, \ldots , X_n$$:

$$
P(X_n, \ldots , X_1)  = P(X_n | X_{n-1}, \ldots , X_1) \cdot P(X_{n-1}, \ldots , X_1)
$$

in its recursive form, or the following in its general form: 

$$
P\left(\bigcap_{k=1}^n X_k\right) = \prod_{k=1}^n P\left(X_k \,\Bigg|\, \bigcap_{j=1}^{k-1} X_j\right)
$$

Recall from eq. \ref{eq_elbo} that $$ELBO(q) = \mathbb{E} [\log p(z, x)] - \mathbb{E}[\log q(z)]$$.
Applying the chain rule to the joint probability $$p(z_{1:m}, x_{1:n})$$ for $$m$$ variational factors and $$n$$ datapoints gives:

$$
%source: assets: 10708-scribe-lecture13.pdf
p(z_{1:m}, x_{1:n}) = p(x_{1:n}) \prod_{j=1}^m p(z_j \mid z_{1: (j-1)}, x_{1:n})
$$

and the entropy term decomposes under the mean-field variational approximation as:

$$
\mathbb{E}_q [ \log q(z_{1:m})] = \sum_j^m \mathbb{E}_{q_j}[\log q(z_j)]
$$

Inserting both decompositions into the ELBO equation gives the ELBO for the mean-field variational approximation:

$$
%source: assets: 10708-scribe-lecture13.pdf
\begin{align*}
ELBO(q) & = \mathbb{E}_q [\log p(x_{1:n},z_{1:m})] - \mathbb{E}_q[\log q(z_{1:m})] \\
        & = \mathbb{E}_q \left[ \log \left ( p(x_{1:n}) \prod_{j=1}^m p(z_j \mid z_{1: (j-1)}, x_{1:n}) \right) \right] - \sum_j^m \mathbb{E}_{q_j}[\log q(z_j)] \\
        & = \log p(x_{1:n}) + \sum_j^m \mathbb{E}_q \left[ \log p(z_j \mid z_{1: (j-1)}, x_{1:n}) \right] - \sum_j^m \mathbb{E}_{q_j}[\log q(z_j)] \\
\end{align*}
$$

Coordinate ascent variational inference (CAVI) computes the derivative with respect to each variational factor and updates one factor at a time to the value that maximizes the ELBO, with the other factors fixed. The derivative uses the notation:

$$
p (z_j \mid z_{\neq j}, x) = p (z_j \mid z_1, ..., z_{j-1}, z_{j+1}, ..., z_m, x)
$$

where $$\neq j$$ means *all indices except the $$j^{th}$$*. The coordinate ascent update for a latent variable, with the others fixed, is $$arg\,max_{q_j} \, ELBO$$:

$$
\begin{align*}
  & arg\,max_{q_j} \, ELBO (q)\\
= & arg\,max_{q_j} \, \left( \log p(x_{1:n}) + \sum_j^m \mathbb{E}_q \left[ \log p(z_j \mid z_{1: (j-1)}, x_{1:n}) \right] - \sum_j^m \mathbb{E}_{q_j}[\log q(z_j)] \right)\\  
= & arg\,max_{q_j} \, \left( \mathbb{E}_q \left[ \log p(z_j \mid z_{\neq j}, x) \right] - \mathbb{E}_{q_j}[\log q(z_j)] \right) \\
  & \qquad \text{(removing vars that don't depend on } q(z_j) \text{)} \\
= & arg\,max_{q_j} \, \left( \int q(z_j) \, \mathbb{E}_{q_{\neq j}} \left[ \log p(z_j \mid z_{\neq j}, x) \right] dz_j - \int q(z_j) \log q(z_j) dz_j \right) \\
  & \qquad \text{(def. Expected Value)} \\
\end{align*}
$$

This argmax is found by taking the derivative of the loss with respect to $$q(z_j)$$, using Lagrange multipliers, and setting the derivative to zero:

$$
\frac{d\,L}{d\,q(z_j)} = \mathbb{E}_{q_{\neq j}} \left[ \log p(z_j \mid z_{\neq j}, x) \right] - \log q(z_j) -1 = 0
$$
 
This leads to the final coordinate ascent update:

$$
\begin{align}
q^{\star}(z_j) & \propto \exp \{ \mathbb{E}_{q_{\neq j}} [ \log p(z_j \mid z_{\neq j}, x)] \} \\
               & \propto \exp \{ \mathbb{E}_{q_{\neq j}} [ \log p(z_j, z_{\neq j}, x)] \} \label{eq_CAVI} \\
                 & \qquad \text{(} p(z_{\neq j}, x) \text{ does not depend on } z_j \text{)} \\
\end{align}
$$

However, this gives the factorization, or the template of the computation, but not the final form of the optimal $$q_j$$ (i.e. its application to a specific distribution family). The chosen form determines the complexity of the coordinate update $$q^{\star}(z_j)$$.

## Gaussian Mixture Models

Credit: this section is a more detailed explanation of sections 2.1, 3.1 and 3.2 of the paper [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf).

A Gaussian mixture model is a probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. The problem is to fit a set of Gaussian density functions to the input dataset.

Consider a mixture of $$K$$ univariate Gaussian distributions with means $$\mu = \mu_{1:K} = \{ \mu_1, \mu_2, ..., \mu_K \}$$. The means are drawn from a common prior $$p(\mu_k)$$, assumed to be Gaussian $$\mathcal{N}(0, \sigma^2)$$. **The latent variables are the means $$\mu_{1:K}$$ and the cluster assignments $$c_{1:n}$$** for each observation $$x_i$$. **$$\sigma^2$$ is a hyper-parameter**. $$c_i$$ is an indicator $$K$$-vector, i.e. all zeros except for a one at the index of the allocated cluster. Each observation $$x_i$$ (from a total of $$n$$ observations) is drawn from the corresponding Gaussian $$\mathcal{N}(c_i^T \mu, 1)$$. The model is:

$$
%source: section 2.1: https://arxiv.org/pdf/1601.00670.pdf
\begin{align}
\mu_k              & \thicksim \mathcal{N}(0, \sigma^2),             & & k=1,...,K, \\
c_i                & \thicksim \text{Categorical} (1/K, ..., 1/K),   & & i=1,...,n, \\
x_i \mid c_i, \mu & \thicksim \mathcal{N}(c_i^T \mu, 1)            & & i=1,...,n. \label{eq_gmm_likelihood}\\
\end{align}
$$

The joint probability of the latent and observed variables for a sample of size $$n$$ is then:

$$
p (\mu, c, x) = p(\mu) p(c,x \mid \mu) = p(\mu) \, \prod_{i=1}^n p(c_i) p(x_i \mid c_i, \mu)
\label{eq_7_source}
$$

This again applies the probability chain rule. The latent variables are $$z=\{\mu, c\}$$, holding the $$K$$-class means and the $$n$$-point class assignments. The evidence is:

$$
\begin{align*}
p(x) & = \int p(\mu) \prod_{i=1}^n \sum_{c_i} p(c_i) \, p(x_i \mid c_i, \mu) d\mu \\
     & = \sum_c p(c) \int p(\mu) \prod_{i=1}^n p(x_i \mid c_i, \mu) d\mu \\
\end{align*}
$$

That is, the evidence is a sum over all possible configurations of cluster assignments, with a computational complexity of $$O(K^n)$$, due to the $$K$$ classes that have to be assigned to $$n$$ points.
Each integral in the previous equation can be computed, since the Gaussian prior and likelihood are conjugate.
However, the evidence is still computationally intractable due to the $$K^n$$ complexity.

An alternative is the mean-field approximation described earlier. From eq. \ref{eq_posterior_mf}, the mean-field approximation yields approximate posterior probabilities of the form:

$$
q(\mu,c) = \prod_{k=1}^K q(\mu_k; m_k, s^2_k) \prod_{i=1}^n q(c_i; \varphi_i).
\label{eq_variational_family}
$$

As before, each latent variable is governed by its own variational factor. The factor $$q(\mu_k; m_k, s^2_k)$$ is a Gaussian distribution on the mean of the $$k^{th}$$ mixture component, with mean $$m_k$$ and variance $$s^2_k$$. The factor $$q(c_i; \varphi_i)$$ is a distribution on the $$i^{th}$$ observation's mixture assignment; its assignment probabilities are a $$K$$-vector $$\varphi_i$$.

There are two types of variational parameters: (1) the Gaussian parameters $$m_k$$ and $$s^2_k$$ to approximate the posterior of the $$k^{th}$$ component; and (2) the Categorical parameters $$\varphi_i$$ to approximate the posterior cluster assignment of the $$i^{th}$$ data point. Combining the joint density of the latent and observed variables (eq. \ref{eq_7_source}) and the variational family (eq. \ref{eq_variational_family}) gives the ELBO for the mixture of Gaussians:

$$
%source https://arxiv.org/pdf/1601.00670.pdf
\begin{align}
ELBO(m, s^2, \varphi) = & \sum_{k=1}^K \mathbb{E} \left[ \log p(\mu_k); m_k, s_k^2 \right] \\
                        & + \sum_{i=1}^n \left( \mathbb{E} \left[ \log p(c_i); \varphi_i \right] + \mathbb{E} \left[ \log p(x_i \mid c_i, \mu); \varphi_i, m, s^2 \right] \right) \\
                        & - \sum_{i=1}^n \mathbb{E} \left[ \log q(c_i; \varphi_i) \right] - \sum_{k=1}^K \mathbb{E} \left[ \log q(\mu_k ; m_k, s_k^2) \right] \label{eq_algo_elbo} \\
\end{align}
$$

The CAVI (coordinate ascent variational inference) algorithm updates each variational parameter in turn, so both updates must be computed.

### Update 1: Variational update for cluster assignment

The first update is for the cluster assignment $$c_i$$. Using the mean-field recipe from equation \ref{eq_CAVI}:

$$
q^{\star}(c_i; \varphi_i) \propto \exp \left( \log p(c_i) + \mathbb{E} [ \log p(x_i \mid c_i, \mu); m, s^2] \right).
$$

The first term is the log prior of $$c_i$$, which is the same for all values of $$c_i$$: $$\log p(c_i)= \log \frac{1}{K} = \log 1 - \log K = -\log K$$. The second term is the expected log probability of the $$c_i^{th}$$ Gaussian density. Because $$c_i$$ is an [indicator vector](https://en.wikipedia.org/wiki/Indicator_vector), the likelihood can be written as:

$$
p (x_i \mid c_i, \mu) = \prod_{k=1}^K p(x_i \mid \mu_k)^{c_{ik}}
$$

i.e. because $$c_{ik} \in \{0,1\}$$, this is a product of terms that are equal to one, except when cluster $$k$$ is allocated to the datapoint $$x_i$$. This gives the following expansion of the second term:

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

The calculation requires $$\mathbb{E} [\mu_k]$$ and $$\mathbb{E} [\mu_k^2]$$, which are computable from the variational Gaussian on the $$k^{th}$$ mixture component. Thus the variational update for the $$i^{th}$$ cluster assignment --- expressed only as a function of the variational parameters of the mixture component --- is:

$$
\varphi_{ik} \propto \exp \left\{ x_i \, \mathbb{E} [ \mu_k; m_k, s_k^2] - \frac{1}{2} \mathbb{E}[ \mu^2_k; m_k, s^2_k] \right\}
\label{eq_algo_first_step}
$$


### Update 2: Variational update for cluster mean

The second update is for the variational density $$q(\mu_k; m_k, s^2_k)$$ in equation \ref{eq_variational_family}. Equation \ref{eq_CAVI} (the exponentiated log of the joint) is again applied to the mean values $$\mu_k$$:

$$
q(\mu_k) \propto \exp \{ \log p(\mu_k) + \sum_{i=1}^n \mathbb{E} [ \log p(x_i \mid c_i, \mu); \varphi_i, m_{\neq k}, s^2_{\neq k} ] \}
$$

where, as before, $$\varphi_{ik}$$ is the probability that the $$i^{th}$$ observation comes from the $$k^{th}$$ cluster, and $$c_i$$ is a one-hot (or indicator) vector. The log probability is then:

$$
\log q(\mu_k) = \log p(\mu_k) + \sum_i \mathbb{E} [ \log p(x_i \mid c_i, \mu); \varphi_i, m_{\neq k}, s^2_{\neq k}] + const\\
$$

This reduction is omitted here; the details are in equations 28-32 of [the paper](https://arxiv.org/pdf/1601.00670.pdf). The result is:

$$
\log q(\mu_k) = (\sum_i \varphi_{ik} x_i) \mu_k - (\frac{1}{2 \sigma^2} + \frac{1}{2} \sum_i \varphi_{ik} ) \mu_k^2 + const.\\
$$

Therefore, the CAVI optimal variational density of $$\mu_k$$ is a member of the Exponential Family of Distributions with sufficient statistics $$\{\mu_k, \mu_k^2\}$$ and natural parameters $$\left\{\sum_{i=1}^n \varphi_{ik} x_i, -\frac{1}{2 \sigma^2} - \frac{1}{2}\sum_{i=1}^n \varphi_{ik} \right\}$$. 
 For a detailed discussion of the exponential family, sufficient statistics and natural parameters, see the previous [post]({{ site.baseurl }}{% post_url 2019-03-20-exponential-family %}). The updates for the mean and the variance are then:

$$
\begin{align*}
0 & = \frac{d\, \log q(\mu_k)}{d\mu_k} \\
0 & = \sum_i \varphi_{ik}x_i - 2 \left( \frac{1}{2 \sigma^2} + \frac{1}{2} \sum_i \varphi_{ik} \right) \mu_k \\
\hat{\mu_k} & = \frac{\sum_i \varphi_{ik} x_i}{1/\sigma^2 + \sum_i \varphi_{ik}}\\ 
\end{align*}
$$

The final variable $$\hat{\mu_k}$$ is referred to as $$m_k$$, in line with the paper and to avoid confusion. Similarly, the update of the variance is:

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

The algorithm is iterative and converges on a certain loss threshold (ELBO), based on the fit of each Gaussian to the data points. The illustration below shows several iterations of the fitting of Gaussian models to five clusters of points:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/variational-inference-gmm/gmm_iterations.svg"/>

{: style="text-align:center; font-size: small;"}
Application of the CAVI algorithm to Gaussian Mixture Models, on a dataset of 2D points assigned to five color-coded clusters. The figure uses the multivariate (2D) version of the model; the derivation above uses univariate Gaussians.
(source: <a href="https://arxiv.org/pdf/1601.00670.pdf">Variational Inference: A Review for Statisticians</a>)


## Stochastic Variational Inference

CAVI does not scale well to very large datasets.
CAVI iterates between re-analyzing every data point in the dataset and re-estimating its hidden structure. This is inefficient for large datasets, because each iteration requires a full pass through the data.

Another well-known method, [Stochastic Variational Inference](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) (SVI), uses a more efficient algorithm based on stochastic optimization ([Robbins and Monro, 1951](https://projecteuclid.org/euclid.aoms/1177729586)), a technique that follows noisy estimates of the gradient of the objective.

SVI is stochastic gradient descent applied to variational inference. It scales to the large datasets of modern machine learning, since the ELBO can be computed on a single mini-batch at each iteration.

SVI iterates between subsampling the data and adjusting the hidden structure based only on the subsample, which makes it more efficient than traditional variational inference. For this reason, several ML libraries such as [Pyro](https://pyro.ai/examples/svi_part_i.html) use SVI.

The details of SVI are not covered here; see section 4.3 of the paper [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf).
