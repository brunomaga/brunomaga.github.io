---
layout: post
title:  "Variational Inference: Mean-Field Approximation on Gaussian Mixture Models"
categories: [machine learning, unsupervised learning, probabilistic programming]
tags: [machinelearning]
---

We learnt in a [previous post]({{ site.baseurl }}{% post_url 2019-11-12-Bayesian-Linear-Regression %}) about Bayesian inference, that the goal of Bayesian inference is to compute the likelihood of observed data and the mode of the density of the likelihood, marginal distribution and conditional distributions. Recall the formulation of the **posterior** of the latent variable $z$ and observations $x$, derived from the Bayes rule without the normalization term:

$$
p (z \mid x) = \frac{p(z) \, p(x \mid z)}{p(x)} \propto p(z) \, p(x \mid z)
\label{eq_bayes}
$$

read as *the posterior is proportional to the prior times the likelihood*. 
We also saw that the Bayes rule derives from the formulation of [**conditional probability**](https://en.wikipedia.org/wiki/Conditional_probability#Definition) $p(z\mid x)$ expressed as:

$$
p (z \mid x ) = \frac{p(z,x)}{p(x)}
\label{eq_conditional}
$$

where the denominator represents the marginal density of $x$ (ie our observations) and is also referred to as **evidence**, which can be calculated by **marginalizing** the latent variables in $z$ over their joint distribution:

$$
p(x) = \int p(z,x) dz
\label{eq_joint}
$$

The inference problem is to compute the conditional probability of the latent variables $z$ given the observations $x$ in $X$, i.e. conditioning on the data and compute the posterior $p(z \mid x)$. For many models, this evidence integral does not have a closed-form solution or requires an exponential time to compute. 

**There are several approaches to inference**, comprising algorithms for exact inference (Brute force, The elimination algorithm, Message passing (sum-product algorithm, Belief propagation), Juntion tree algorithm), and for approximate inference (Loopy belief propagation, Variational (Bayesian) inference, Stochastic simulation / sampling / Markov Chain Monte Carlo). Why do we need approximate methods after all? Simply because for many cases, we cannot directly compute the posterior distribution, i.e. the posterior is on an **intractable** form --- often involving integrals --- which cannot be (easily) computed. **This post focuses on the simplest approach to Variational Inference based on mean-field approximation and coordinate-ascent optimization**.

### Detour: Markov Chain Monte Carlo

Before moving into Variational Inference, let's understand the place of VI in this type of inference. For many years, the dominant approach was the [Markov chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo). A simple high-level understanding of MCMC follows from the name itself:
- [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) methods are a simple way of estimating parameters via generating random numbers. A simple example is to compute the area of a circle inside by generating random 2D coordinates whitin the bounding square around the circle, and estimate the value of $\pi$ or the area of the circle from the proportion of generated datapoints that fall inside vs outside the circlle in the square;
- [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain) is a stochastic model describing a sequence of events in which the probability of moving to a next state depends *only* on the state of the current state and not on the previous ones; an example would be the probability of the next character in a word for all *27* possible characters, given the current character.

Therefore, MCMC methods are used to approximate the distribution (state) of parameters at a next iteration by random sampling from a probabilitistic space defined in the current state. MCMC is based on the assumption that the prior-likelihood product $ P(x \mid z) \, P(z)$ can be computed, i.e. is known, for a given $z$. However, we can do this without having any knowledge of the function of $z$. But we know that from a mathematical perspective (eq. \ref{eq_bayes}) the posterior density is expressed as:

$$
\begin{align*}
p (z \mid x) & = \frac{p(z) \, p(x \mid z)}{p(x)} & \text{(Bayes, eq. \ref{eq_bayes})} \\
             & = \frac{p(z) \, p(x \mid z)}{\int p(z) p(x \mid z) \,dz} & \text{(marginalizing $z$, eq. \ref{eq_joint})} \\
\end{align*}
$$

where the top term of the division can be computed, but the bottom one is unknown or intractable. Since the bottom term is a *normalizer* --- i.e. guarantees that the posterior sums to 1, we can discard it, leading to the expression $ p (z \mid x) \propto p(z) \, p(x \mid z)$. Therefore, the rationale is that we can take several samples of the latent variables (as en example the mean $\mu$ or variance $\sigma^2$ in normal distribution) from the current space, and update our model with the values that explain the data better than the values at the current state, i.e. higher posterior probability. A simpler way to represent this description is to compute the acceptance ratio of the proposed over the current posterior (allowing us to discard the term $P(x)$ which we couldn't compute):

$$
  \frac{ P(z \mid x) }{ P(z_0 \mid x) }
=  \frac{ \hspace{0.5cm}{ } \frac{P(x \mid z) \, P(z)}{ P(x) } \hspace{0.5cm} } { \text{ } \frac{P(x \mid z_0) \, P(z_0)}{ P(x) } \text{ } }
= \frac{ P(x \mid z) \, P(z) }{ P(x \mid z_0) \, P(z_0)}
$$

and perform a step change towards the state of highest ratio. 

The main advantage of MCMC is that it provides an approximation of the *true* posterior, although there are many disavantadges on handling a exact posterior which doesn't follow a parametric ditribution (such as hard interpretability in high dimensions). Moreover, it requires a massive computation power for problems with large latent spaces. That's where Variational Inference enters the picture: it's faster and works well if we know the parametric distribution of the posterior, however being sometimes *over-confident* when posterior are not exact to the approximated. 

I will try to post about this topic in the near future, but if you're curious you can always check the paper [An Introduction to MCMC for Machine Learning]({{ site.assets }}/Bayesian-Variational-Inference-GMM/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf) if you are interested on the theory behind it, or a practical explanation with code in [Thomas Wiecki's post](https://twiecki.io/blog/2015/11/10/mcmc-sampling/).

### Variational Inference

**The idea behind Variational Inference (VI) methods is to propose a family of densities and find a member $q^\star$ of that family which is close to the target posterior $p(z \mid x)$**. I.e. instead of computing the *real* posterior, we try to find the parameters $z$ of a new distribution $q^\star$ (the approximation to our real posterior) such that:

$$
q^\star(z) = arg\,min \, KL(q(z) || p(z\mid x))
$$

The logic behing it is that we want to minimize the divergence between a real posterior and its approximated posterior, sometimes referred to as the *guide* distribution/posterior. So we need to first define a metric of *approximation* or proximity. To that extent, the closeness (*proximity*) of two distributins $p$ and $q$ is a measurement of *probabilities similiarity* measured by the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence): 

$$
KL (q(z) || p(z \mid x)) =  \int_z q(z) \log \frac{q(z)}{p(z \mid x)} = \mathbf{E} \left[ \log \frac{q(z)}{p( z\mid x)} \right]
\label{eq_KLdiv}
$$ 

If both $q$ and $p$ are high, then we achieve a good solution as the KL-divergence is low. If $q$ is high and $p$ is low, we have a high divergence and the solution *may* not be very good. Otherwise, if *q* is low, we dont care about *p*, since we achieve low divergence, independently of $p$. It would then make more sense to compute $KL(p\|\|q)$ instead, however we do not do this due to computational reasons, as we will see later.

The logic is that we want to minimize the divergence between our real posterior $p(z \mid x)$ and its approximated posterior $q(z)$. We cannot minimize this directly, but we can minimize a function that is *equal to it (up to a constant)*, known as the Evidence Lower Bound.

### Evidence Lower Bound (ELBO)

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
\log p(x) & = \log \int p(x,z) dz & \text{(marginalizing $z$, eq. \ref{eq_joint})}\\
  & = log \int p(x,z) \frac{q(z)}{q(z)} dz \\
  & = \log \left( \mathbf{E} \left[ \frac{p(x,z)}{q(z)} \right] \right)    & \text{(expected value: $\mathbf{E}[X] = \int x \, p_X(x) \, dx$)} \\
  & \ge \mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)] & \text{($\mathbf{E}[XY]=\mathbf{E}[X]*\mathbf{E}[Y]$; Jensen's inequality, eq. \ref{eq_jensens})} \\
\end{align*}
$$

Putting it all together, the Evidence Lower Bound for a probability model $p(x,z)$ and the approximation to the posterior $q(z)$  is:

$$
ELBO(q) = \mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)]
\label{eq_elbo}
$$

This formulation justifies the name ELBO. In practice, $\mathbf{E} [\log p(x,z)] - \mathbf{E}[\log q(z)]$ is smaller-or-equal than --- or is the lower bound of --- the evidence $\log p(x)$.


We can use the previous result and rewrite the KL-divergence as:

$$
%source: assets/Bayesian-SVI-MCMC/10708-scribe-lecture13.pdf
\begin{align*}
KL (q(z) || p(z \mid x)) & = \mathbf{E} \left[ \log \frac{q(z)}{p(z \mid x)} \right] & \text{(def of KL-divergence, eq. \ref{eq_KLdiv})}\\
 & = \mathbf{E} [\log q(z)] - \mathbf{E}[\log p(z\mid x)] \\  
 & = \mathbf{E} [\log q(z)] - \mathbf{E} [\log p(z,x)] + \mathbf{E} [\log p(x)]  &\text{(def. conditional prob., eq. \ref{eq_conditional})}\\  
 & = - (\mathbf{E} [\log q(z)] + \mathbf{E} [\log p(z,x)]) + \log p(x)\\
 & = - ELBO(q) + \log p(x) &\text{(def. of ELBO eq. \label{eq_elbo})}\\
\end{align*}
$$

i.e. the KL divergence is the negative of the ELBO plus a constant that does not depend on $q$. The previous formulation is also commonly represented as:

$$
\log p(x) = ELBO(q) + KL (q(z) || p(z \mid x))
$$

So we optimize the lower-bound term over quantities $q(z)$ to find a minimal approximation. Or in other words, since $\log p(x)$ does not depend on $q$, we **maximize the ELBO which is equivalent to miniziming the KL-divergence** of the real posterior $p(z \mid x)$ and its approximation $q(z)$.

### Mean-field approximation

So far we foccused on a single posterior defined by a family with a single distributions $q$, whose posterior $p(z \mid x)$ is tractable. We now extend our analysis to complex families, where the posterior is not tractable. 

One way to overcome it is to approximate the posterior probability using a simpler model. For example, assuming that a set of a set of latent variables is independent of other variables. Or we can enfurce full independence among all latent variables given $x$. This assumption is known as the **mean-field approximation**.

In practice, this methods originates from mean-field theory. From [wikipedia](https://en.wikipedia.org/wiki/Mean-field_theory): 
> "Mean-field theory [...] studies the behavior of high-dimensional random (stochastic) models by studying a simpler model that approximates the original by averaging over degrees of freedom. [...] The effect of all the other individuals on any given individual is approximated by a single averaged effect, thus reducing a many-body problem to a one-body problem."

More complex scenarios such as variational family with dependencies between variables (structured variational inference) or with mixtures of variational families (covered in [Chris Bishop's PRML book]({{ site.resources_permalink }})) will be covered in a future post. As a side note, because the latent variables are independent, the mean-field approach that follows usually does not contain the true posterior.

In this type of Variational Inference, due to variable independency, the variational distribution over the latent variables factorizes as:

$$
q(z) = q(z_1, ..., z_m) = \prod_{j=1}^{m} q(z_j)
\label{eq_posterior_mf}
$$

where each latent variable ($z_j$) is governed by its own density $q(z_j)$. Note that the variational family input does not depend on the input data $x$ which does not show up in the equation. We can also partition the latent variables into $K$ groups $$z_{G_1}, ..., z_{G_K}$$, and represent the approximation instead as:

$$
q(z) =  q(z_1, ..., z_m) =  q(z_{G_1}, ..., z_{G_K}) = \prod_{j=1}^{K} q_j(z_{G_j}) 
$$

This setup is often called **generalized mean field** instead of **naive mean field**. Each latent variable $z_j$ is governed by its own variational factor, the density $q_j(z_j)$. For the formulation to be complete, we'd have to specify the parametric form of the individual variational factors. In principle, each can take on any parametric distribution to the corresponding random variable (e.g. a combination of Gaussian and Categorical distributions). The computation of the (template of the) functional form of $q_j(z_j)$ is very lenghty and is detailed on [Brian Keng's post](http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/), so I'll skip those details and show later an application of this template to Gaussian Mixture Models.

Back to the main topic, *the main objective now is to optimize the ELBO in the mean field variational inference*, or equivalently, to choose the variational factors that maximizes the ELBO (eq. \ref{eq_elbo}). A common approach is to  use the **coordinate ascent** method, by optimizing the variational approximation of each latent variable $q_{z_j}$, while holding the others fixed.

Recall the [probability chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)) for more than two events $X_1, \ldots , X_n$:

$$
P(X_n, \ldots , X_1)  = P(X_n | X_{n-1}, \ldots , X_1) \cdot P(X_{n-1}, \ldots , X_1)
$$

on its recursive notation; or the following on its general form: 

$$
P\left(\bigcap_{k=1}^n X_k\right) = \prod_{k=1}^n P\left(X_k \,\Bigg|\, \bigcap_{j=1}^{k-1} X_j\right)
$$

We now apply the chain rule to equation \ref{eq_joint} but applied to our use case, i.e. to the joint probablility $p(z_{1:m}, x_{1:n})$ for $m$ variational factors and $n$ datapoints we get:

$$
%source: assets: 10708-scribe-lecture13.pdf
p(z_{1:m}, x_{1:n}) = p(x_{1:n}) \prod_{j=1}^m p(z_j \mid z_{1: (j-1)}, x_{1:n})
$$

We can decompose the entropy term of the ELBO (using the mean field variational approximation) as: 

$$
\mathbf{E}_q [ \log q(z_{1:m})] = \sum_j^m \mathbf{E}_{q_j}[\log q(z_j)]
$$

Combining the two previous factors, we can now express the ELBO in equation \ref{eq_elbo} for the mean field-variational approximation as:

$$
%source: assets: 10708-scribe-lecture13.pdf
\begin{align*}
ELBO(q) & = \mathbf{E}_q [\log p(x_{1:n},z_{1:m})] - \mathbf{E}_{q_j}[\log q(z_{1:m})] \\
        & = \mathbf{E}_q \left[ \log \left ( p(x_{1:n}) \prod_{j=1}^m p(z_j \mid z_{1: (j-1)}, x_{1:n}) \right) \right] - \sum_j^m \mathbf{E}_{q_j}[\log q(z_j)] \\
        & = \log p(x_{1:n}) + \sum_j^m \mathbf{E}_q \left[ \log p(z_j \mid z_{1: (j-1)}, x_{1:n}) \right] - \sum_j^m \mathbf{E}_{q_j}[\log q(z_j)] \\
\end{align*}
$$

### Coordinate Ascent Mean-Field Variational Inference 

A common method for iteratively finding the optimal solution is the coordinate ascent variational inference (CAVI), where we compute the derivative of each variational factor and alternatively update the weights towards the lowest loss acording to the factor being iterated. Before we proceed to the derivative, we introduce the notation:

$$
p (z_j \mid z_{\neq j}, x) = p (z_j \mid z_1, ..., z_{j-1}, z_{j+1}, ..., z_m, x)
$$

where $\neq j$ means *all indices except the $j^{th}$*. Now we want to derive the coordinate ascent update for a latent variable, while keeping the others fixed, i.e. we compute $arg\,max_{q_j} \, ELBO$:

$$
\begin{align*}
  & arg\,max_{q_j} \, ELBO \\
= & arg\,max_{q_j} \, \left( \log p(x_{1:n}) + \sum_j^m \mathbf{E}_q \left[ \log p(z_j \mid z_{1: (j-1)}, x_{1:n}) \right] - \sum_j^m \mathbf{E}_{q_j}[\log q(z_j)] \right)\\  
= & arg\,max_{q_j} \, \left( \mathbf{E}_q \left[ \log p(z_j \mid z_{\neq j}, x) \right] - \mathbf{E}_{q_j}[\log q(z_j)] \right) & \text{(removing vars that don't depend on $q(z_j)$)} \\
= & arg\,max_{q_j} \, \left( \int q(z_j) \log p(z_j \mid z_{\neq j}, x) dz_j - \int q(z_j) \log q(z_j) dz_j \right) & \text{(def. Expected Value)} \\
\end{align*}
$$

To find this argmax, we take the derivative of the loss with respect to $q(z_j)$, use Lagrange multipliers, and set the derivative to zero:

$$
\frac{d\,L}{d\,q(z_j)} = \mathbf{E}_{q_{\neq j}} \left[ \log p(z_j \mid z_{\neq j}, x) \right] - \log q(z_j) -1 = 0
$$
 
Leading to the final coordinate ascent update:

$$
\begin{align}
q^{\star}(z_j) & \propto exp \{ \mathbf{E}_{q_{\neq j}} [ \log p(z_j \mid z_{\neq j}, x)] \} \\
               & \propto exp \{ \mathbf{E}_{q_{\neq j}} [ \log p(z_j, z_{\neq j}, x)] \} & \text{(mean-field assumption: latent vars are independent)} \label{eq_CAVI}\\
\end{align}
$$

However, this provides the factorization or the template of the computation, but not the final form (i.e. application to a specific distribution family) of the optimal $q_j$. The form we choose influences the complexity or *easiness* of the coordinate update $q^{\star}(z_j)$. As a side note, this formulation can be seen as a Message Passing algorthm, widely used in Graphical Models (see [John Winn's MBML book]({{ site.resources_permalink }})), and this has enabled the design of large-scale models.

To finalize, we derive the coordinate update of the previous equation, by re-writing the ELBO in eq. \ref{eq_elbo} as a function of the $j^{th}$ variational factor $q_j(z_j)$ and *absorbing* into a constant the terms that do not depend on it:

$$
%source: [*Variational Inference: A Review for Statisticians*](https://arxiv.org/pdf/1601.00670.pdf)
ELBO(q_j) = \mathbf{E_j} [ \mathbf{E}_{\neq j}[ \log p (z_j, z_{\neq j}, x) ]] - \mathbf{E}_j [ \log q_j(z_j)] + const
$$ 

where the first term on the right-hand side was rewritten using [iterated expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation), and the second term was decomposed using the mean-field variable independency and keeping only the terms dependent on $q_j(z_j)$. 
An alternative resolution based on gradients is also proposed in [Chris Bishop's PRML book]({{ site.resources_permalink }})).

Finally, to help with numerical stability, it's recommended to work with logarithms of probabilities. A useful *log-sum-trick* is the definition of the logarithm of the probability as:

$$
\log \left[ \sum_i \exp(x_i) \right] = \alpha + \log \left[ \sum_i \exp (x_i - \alpha) \right]
$$ 

where the constant $\alpha$ is typically set to $max_i x_i$, providing numerical stability to most common VI computations.

### Gaussian Mixture Models

<small>Credit: this section is a more verbose and extended explanation of sections 2.1, 3.1 and 3.2 of the paper [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf).</small>

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. The problem at hand is to fit a set of Gaussian density functions to the input dataset. 

Take a family of a mixture of $K$ univariate Gaussian distributions with means $\mu = \mu_{1:K} = \{ \mu_1, \mu_2, ..., \mu_K \}$. The means are drawn from a common prior $p(\mu_n)$, which we assume to be Gaussian $$\mathcal{N}(\mu, \sigma^2)$$. **The latent space are the means $\mu_{1:K}$ and the cluster assignments $c_{1:n}$** for each observation $x_i$. **$\sigma^2$ is a hyper-parameter**. $c_i$ is described as an indicator $K-$vector, i.e. all zeros except a one on the index of the allocated cluster. Each new observation $x_i$ (from a total of $n$ observations) is drawn from the corresponding Gaussian $\mathcal{N}(c_i^T, \mu, 1)$. The model can be expressed as:

$$
%source: section 2.1: https://arxiv.org/pdf/1601.00670.pdf
\begin{align}
\mu_k              & \thicksim \mathcal{N}(0, \sigma^2),             & & k=1,...,K, \\
c                  & \thicksim \text{Categorical} (1/K, ..., 1/K),   & & i=1,...,n, \\
x_i \mid c_i, \mu & \thicksim \mathcal{N}(c_i^T \mu, 1)            & & i=1,...,n. \label{eq_gmm_likelihood}\\
\end{align}
$$

In this case, the joint probability of the latent and observed variables  for a sample of size $n$ is:

$$
p (\mu, c, x) = p(\mu) p(c,x) = p(\mu) \, \prod_{i=1}^n p(c_i) p(x_i \mid c_i, \mu)
\label{eq_7_source}
$$

Here we applied again the probability chain rule. The latent variables are $z=\{\mu, c\}$, holding the $K$-class means and the $n$-point class assigments. The evidence is:

$$
\begin{align*}
p(x) & = \int p(\mu) \prod_{i=1}^n \sum_{c_i} p(c_i) \, p(x_i \mid c_i, \mu) d\mu \\
     & = \sum_c p(c) \int p(\mu) \prod_{i=1}^n p(x_i \mid c_i, \mu) d\mu \\
\end{align*}
$$

I.e the evidence is now a sum of all possible configurations of clusters assignment, with a computational complexity $O(K^n)$ due to $K$ classes that have to be assigned to $n$ points.
We can compute this previous equation, since the gaussian prior and likelihood are conjugates.
However, it is still computationally intractable due to the $K^n$ complexity. 

An alternative solution is to use the mean-field approximation method we've discussed. We know from before (eq. \ref{eq_posterior_mf}) that the mean-field yields approximate posterior probabilities of the form:

$$
q(\mu,c) = \prod_{k=1}^K q(\mu_k; m_k, s^2_k) \prod_{i=1}^n q(c_i; \varphi_i).
\label{eq_variational_family}
$$

and re-phrasing what was mentioned before, each latent variable is governed by its own variational factor. The factor $q(\mu_k; m_k, s^2_k)$ is a Gaussian distribution of the $k^{th}$ mixure component, including its mean $m_k$ and variance $s^2_k$. The factor $q(c_i; \varphi_i)$ is a distribution on the $i^{th}$ observation's mixture assignment; its assignment probabilities are a $K-$vector $\varphi_i$.

So we have two types of variational parameters: (1) the Gaussian parameters $m_k$ and $s^2_k$ to approximate the posterior of the $k^{th}$ component; and (2) the Categorical parameters $\varphi_i$ to approximate the posterior cluster assignement of the $i^{th}$ data point. We now combine the joint density of the latent and observed variables (eq. \ref{eq_7_source}) and the variational family (eq. \ref{eq_variational_family}) to form the ELBO for the mixture of Gaussians:

$$
%source https://arxiv.org/pdf/1601.00670.pdf
\begin{align}
ELBO(m, s^2, \varphi) = & \sum_{k=1}^K \mathbf{E} \left[ \log p(\mu_k); m_k, s_k^2 \right] \\
                        & + \sum_{i=1}^n \left( \mathbf{E} \left[ \log p(c_i); \varphi_i \right] + \mathbf{E} \left[ \log p(x_i \mid c_i, \mu); \varphi_i, m, s^2 \right] \right) \\
                        & - \sum_{i=1}^n \left( \mathbf{E} \left[ \log q(c_i; \varphi_i) \right] - \sum_{k=1}^K \mathbf{E} \left[ \log q(\mu_k ; m_k, s_k^2) \right] \right) \label{eq_algo_elbo} \\
\end{align}
$$

The CAVI (coordinate ascent variational inference) updates each variational parameter in turn, so we need to comput both updates.

##### Update 1: Variational update for cluster assignment

We start with the variational update for the cluster assignment $c_i$. Using the mean-field recipe from equation \ref{eq_CAVI}:

$$
q^{\star}(c_i; \varphi_i) \propto exp \left( \log p(c_i) + \mathbf{E} [ \log p(x_i \mid c_i, \mu); m, s^2] \right).
$$

The first term is the log prior of $c_i$, the same value for all values of $c_i$: $\log p(c_i)= \log \frac{1}{K} = \log 1 - \log K = -\log K$.  The second term is the expected log probability of the $c_i^{th}$ Gaussian density. Because $c_i$ is an [indicator vector](https://en.wikipedia.org/wiki/Indicator_vector) we can write:

$$
p (x_i \mid c_i, \mu) = \prod_{k=1}^K p(x_i \mid \mu_k)^{c_{ik}}
$$

i.e. because $$c_{ik} = \{0,1\}$$, this is a multiplication of terms that are ones except when cluster $k$ is allocated to the datapoint $x_i$. We use this to continue the expansion of the second term:

$$
\begin{align*}
  & \mathbf{E} [ \log p(x_i \mid c_i, \mu); m, s^2] \\
= & \sum_k c_{ik} \mathbf{E} [ \log p(x_i \mid \mu_k); m_k, s_k^2] & \text{(sum of matching assignments $ik$)} \\
= & \sum_k c_{ik} \mathbf{E} \left[ \log \mathcal{N}(x_i^T \mu, 1) \right] & \text{(likelihood eq. \ref{eq_gmm_likelihood})} \\
= & \sum_k c_{ik} \mathbf{E} \left[ \log \left( \frac{1}{1 \sqrt{2\pi}} \right) - \frac{1}{2} \left( \frac{x_i - \mu_k}{1} \right)^2 ; m_k, s_k^2 \right] & \text{(log of normal distribution)} \\
= & \sum_k c_{ik} \mathbf{E} \left[ -\frac{1}{2}(x_i - \mu_k)^2; m_k, s_k^2 \right] + const & \text{(removed terms that are constant with respect to $c_i$)} \\
= & \sum_k c_{ik} \mathbf{E} \left[ -\frac{1}{2} x_i^2 + \mu x_i -\frac{1}{2} \mu^2; m_k, s_k^2 \right] + const & \text{(decomposed square of sum)} \\
= & \sum_k c_{ik} \left( x_i \, \mathbf{E} [ \mu_k; m_k, s_k^2] - \frac{1}{2} \mathbf{E}[ \mu^2_k; m_k, s^2_k] \right) + const  & \text{(removed terms that are constant with respect to $c_i$)}  \\
\end{align*}
$$

The calculation requires $\mathbf{E} [\mu_k]$ and $\mathbf{E} [\mu_k^2]$, computable from the variational Gaussian on the $k^{th}$ mixture component. Thus the variational update for the $i^{th}$ cluster assignment --- expressed as a function of the variational parameters for the mixture component only --- is:

$$
\varphi_{ik} \propto exp \left\{ x_i \, \mathbf{E} [ \mu_k; m_k, s_k^2] - \frac{1}{2} \mathbf{E}[ \mu^2_k; m_k, s^2_k \right\}
\label{eq_algo_first_step}
$$


##### Update 2: Variational update for cluster mean

We now turn to the variational density $$q(\mu_k; m_k, s^2_k)$$ on equation \ref{eq_variational_family}. We use again \ref{eq_CAVI} (the exponentiated log of te joint) for the mean values $\mu_k$:

$$
q(\mu_k) \propto \exp \{ \log p(\mu_k) + \sum_{i=1}^n \mathbf{E} [ \log p(x_i \mid c_i, \mu); \varphi_i, m_{k\neq}, s^2_{k\neq} ] \}
$$

where the term $\varphi_{ik}$ represents as before the probabliity that the $i^{th}$ observation comes from the $k^{th}$ cluster, and $c_i$ is a one-hot (or indicator) vector. The computation of the log probability follows as:

$$
\log q(\mu_k) = \log p(\mu_k) + \sum_i \mathbf{E} [ \log p(x_i \mid c_i, \mu); \varphi_i, m_{\neq k}, s^2_{\neq k}] + const\\
$$

We will ommit this reduction, if you are looking for details, check equations 28-32 [here](https://arxiv.org/pdf/1601.00670.pdf). This calculation leads to :

$$
\log q(\mu_k) = (\sum_i \varphi_{ik} x_i) \mu_k - (\frac{1}{2} \sigma^2 + \frac{1}{2} \sum_i \varphi_{ik} ) \mu_k^2 + const.\\
$$

In practice it means that the CAVI optimal variational density of $\mu_k$ is a member of the [Exponential Family of Distributions]({{ site.baseurl }}{% post_url 2019-11-20-Exponential-Family-Distributions %}) with sufficient statistics $$\{\mu, \mu^2\}$$  and natural parameters $$\left\{\sum_{i=1}^n \varphi_{ik} x_i, -\frac{1}{2} \sigma^2 - \frac{1}{2}\sum_{i=1}^n \varphi_{ik} \right\}$$. 
 For a detailed discussion on exponential family, sufficient statistics and natural parameters check the previous [post]({{ site.baseurl }}{% post_url 2019-11-20-Exponential-Family-Distributions %}) for details. We can now express the updates for the mean and standard deviation we have:

$$
\begin{align*}
0 & = \frac{d\, \log q(\mu_k)}{d\mu} \\
0 & = \sum_i \varphi_{ik}x_i - 2 \left( \frac{1}{2} \sigma^2 + \frac{1}{2} \sum_i \varphi_{ik} \right) \mu_k \\
\hat{\mu_k} & = \frac{\sum_i \varphi_{ik} x_i}{1/\sigma^2 + \sum_i \varphi_{ik}}\\ 
\end{align*}
$$

We'll refer to the final variable $\hat{\mu_k}$ as $m_k$,  in line with the paper and to avoid confusions. Similarly, the update of the the standard deviation is:

$$
s^2_k = \hat{\sigma}^2 = \frac{1}{1/\sigma^2 + \sum_i \varphi_{ik}}\\
\label{eq_algo_second_step}
$$

###### Algorithm and Vizualization

Putting it all together, the algorithm follows as:

- While ELBO has not converged:
	1. for all $i \in \{1, ..., n\}$, set $$\varphi_{ik} \propto exp \left\{ x_i \, \mathbf{E} [ \mu_k; m_k, s_k^2] - \frac{1}{2} \mathbf{E}[ \mu^2_k; m_k, s^2_k \right\} $$ (equation \ref{eq_algo_first_step})
	2. for $k \in \{1, ..., K\}$ do:
		- set $$m_k \leftarrow \frac{\sum_i \varphi_{ik} x_i}{1/\sigma^2 + \sum_i \varphi_{ik}}$$ (equation \ref{eq_algo_second_step}) 
		- set $$s^2_k \leftarrow \hat{\sigma}^2 = \frac{1}{1/\sigma^2 + \sum_i \varphi_{ik}}$$ (equation \ref{eq_algo_second_step})
	3. compute $ELBO(m, s^2, \varphi)$ (equation \ref{eq_algo_elbo})

The algorithm is iterative and will converge on a certain loss threshold (ELBO), based on the fitting of each Gaussian to the data points. The concept can be easily understood by looking at the illustration of several iterations of the fitting of Gaussian models to 4 clusters of points:   

<p align="center">
<img width="60%" height="60%" src="/assets/Bayesian-Variational-Inference-GMM/GMM_iterations.JPG"/><br/>
<br/><br/><small>Example of the application of the CAVI algorithm to Gaussian Mixture Models. Dataset described by 2D points assigned to five color-coded clusters.
<br/>(source: <a href="https://arxiv.org/pdf/1601.00670.pdf">Variational Inference: A Review for Statisticians</a>)</small>
</p>


### Further Reading

There is a universe of methods and techniques that are interesting and were not covered in this post. To name a few:
- An application of the CAVI to other members of the [Exponential Family of Distributions]({{ site.baseurl }}{% post_url 2019-11-20-Exponential-Family-Distributions %}) and Stochastic Variational Inference is detailed in the papers [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf) and [Stochastic Variational Inference](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf);
- The KL divergence formula is non-symmetric for the two parameter distributions. For details on the application of forward- and reverse KL divergence, relating to computing variational loss based on $$KL( q(z) \mid\mid p(z\mid x))$$ vs $$KL(p(z\mid x) \mid \mid q(z) )$$, refer to [Augustinus Kristiadi's plost](https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/), [Eric Jang's post](https://blog.evjang.com/2016/08/variational-bayes.html) and [Brian Keng's post](http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/);
- Pytorch provides detailed examples with explanations of implementations of Variational Inference in Pytorch. A good place to start is [*SVI Part I: An Introduction to Stochastic Variational Inference in Pyro*](https://pyro.ai/examples/svi_part_i.html);
- Finally, a bit off topic, there is some discussion on whether methods based on approximated posteriors and Bayesian methods are any good for more complicated models, particularly for Deep Neural Nets. If you're curious, I recommend the papers [Quality of Uncertainty Quantification for Bayesian Neural Network Inference](https://arxiv.org/abs/1906.09686) and [How Good is the Bayes Posterior in Deep Neural Networks Really?](https://arxiv.org/abs/2002.02405).
