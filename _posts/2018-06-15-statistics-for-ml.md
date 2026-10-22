---
layout: post
title:  "Statistics for ML"
redirect_from: /statistics
categories: ["Machine Learning"]
tags: ["statistics", "probability", "Bayes rule", "Gaussian distribution", "KL divergence"]
---

In 1763, two years after the death of Thomas Bayes, his essay [An Essay Towards Solving a Problem in the Doctrine of Chances](https://en.wikipedia.org/wiki/An_Essay_Towards_Solving_a_Problem_in_the_Doctrine_of_Chances) was edited and published by his friend Richard Price. It used conditional probability to compute limits on an unknown parameter from evidence, the idea behind [Bayes rule](https://en.wikipedia.org/wiki/Bayes%27_theorem). In 1933, Andrey Kolmogorov founded modern probability theory on three [axioms](https://en.wikipedia.org/wiki/Probability_axioms).

This post is a brief summary of the topics in statistics and probability that are relevant to ML engineers on a daily basis, extracted from the books in the <a href="{{ site.resources_permalink }}">resources</a> section. It covers the rules of probability, moments, independence, Gaussian and exponential family distributions, limit theorems, inequalities, testing, information theory, combinatorics, sets, Bayesian networks and kernel density estimation.

### Basic definitions

The **sample space $$Ω$$** is the set of all *possible* outcomes of an experiment. The **event space $$\mathcal{A}$$** is the space of *potential* results of the experiment, i.e. a [σ-algebra](https://en.wikipedia.org/wiki/%CE%A3-algebra) of subsets of $$Ω$$ (the events) to which probabilities are assigned. The **probability of $$A$$**, or $$P(A)$$, is the degree of belief that the event $$A$$ will occur, in the interval $$[0,1]$$.

A **random variable** is a function $$X : Ω → T$$ that takes an outcome in $$Ω$$ and returns a particular quantity of interest $$x$$ in the target space $$T$$. Note: the name "random variable" creates confusion because it is neither random nor a variable: it is a function! For example, in the case of tossing two coins and counting the number of heads, a random variable $$X$$ maps to the three possible outcomes: $$X(hh) = 2$$, $$X(ht) = 1$$, $$X(th) = 1$$, and $$X(tt) = 0$$.

The pdf of the **sum of independent random variables** is the **[convolution](https://en.wikipedia.org/wiki/Convolution_of_probability_distributions)** of their pdfs:

$$
p_{X+Y}(u) = \int_{-\infty}^{+\infty} p_X(u-v) \, p_Y(v) \, dv
$$


### Statistical distributions

Statistical distributions can be either <a href="{{ site.epfl_statistics }}/formula-sheet-continuous-distributions.pdf">continuous</a> or <a href="{{ site.epfl_statistics }}/formula-sheet-discrete-distributions.pdf">discrete</a>. Most of the commonly used parametric continuous distributions belong to the [exponential family of distributions](https://en.wikipedia.org/wiki/Exponential_family).

$$p(x)$$ is called a **probability mass function (pmf)** or a **probability density function (pdf)** for a discrete or continuous variable $$x$$, respectively. Any function over a discrete (continuous) domain can be a probability distribution, as long as it only has non-negative values and all values sum (integrate) to 1. The **cumulative distribution function (cdf)** is

$$
P(X \le x) = \int^x_{-\infty} p(t) dt
$$

and there are cdfs which do not have a corresponding pdf. The probability of a subset/range of values is the sum of the probabilities of all its values, i.e. $$P( a \le X \le b) = \int^b_a p(t) dt$$, or the equivalent sum for the discrete case.

For discrete probabilities, the **joint probability** of $$x \in X$$ and $$y \in Y$$ (not necessarily independent) is $$P(X=x_i, Y=y_j) = \frac{n_{ij}}{N}$$ for $$N$$ observations, where $$n_{ij}$$ can be taken as the $$ij$$-cell in the confusion matrix.


### Sum rule

The sum rule (or **marginalization property**) states that the probability of $$x$$ is the sum of all joint probabilities of $$x$$ and another variable $$y$$, i.e. $$p(x) = \sum_{y \in Y} p(x,y)$$ in the discrete case and $$p(x) = \int_Y p(x,y) dy$$ in the continuous case.


### Product rule

The product rule relates the joint distribution and the conditional distribution, and states that every joint distribution can be factorized into two other distributions:

$$
p(x,y) = p(y \mid x) p(x) = p( x \mid y) p(y)
$$


### Bayes rule

Bayes rule describes the relationship between some prior knowledge $$p(x)$$ about an unobserved random variable $$x$$ and some relationship $$p(y \mid x)$$ between $$x$$ and a second variable $$y$$:

$$
p(x \mid y) = \frac{ p(y \mid x) p(x)}{p(y)}
$$

read as: $$\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$. It is derived from the product rule on both terms:

$$
p(y \mid x) p(x) = p(x \mid y) p(y) \Leftrightarrow p(x \mid y) = \frac{p(y \mid x) p(x)}{p(y)}
$$

The **evidence** is the **marginal likelihood** $$p(y) = \int p(y \mid x) p(x) dx = \mathbb{E}[ p(y \mid x)]$$. In ML, the posterior $$p(\theta \mid x)$$ is the quantity of interest, as it tells what is known about the parameters $$\theta$$ (i.e. their distribution) after observing the population $$x$$ (given its likelihood and a known/estimated prior).

---


### Expected Value

The expected value of a function $$g$$ of a univariate random variable $$X$$ is $$\mathbb{E}_X [g(x)] = \int_X g(x) p(x) dx$$, or the equivalent weighted sum for the discrete case. For a multivariate random variable, seen as a finite set of univariate ones, i.e. $$X = \left[X_1, X_2, ..., X_D \right]$$, it is:

$$
\mathbb{E}_X [g(x)] = \left[ \mathbb{E}_{X_1} [g(x_1)], \mathbb{E}_{X_2} [g(x_2)], ..., \mathbb{E}_{X_D} [g(x_D)] \right] \in \mathbb{R}^D
$$


### Covariance

The covariance of two random variables is the expected product of their deviations from their respective means, i.e.:

$$
\begin{align*}
\mathbb{Cov}[X,Y] & = \mathbb{E} \left[ (X- \mathbb{E}[X]) \, (Y- \mathbb{E}[Y]) \right] \\
& = \mathbb{E}\left[ XY -  X\mathbb{E}[Y] - \mathbb{E}[X]Y +  \mathbb{E}[X]\,\mathbb{E}[Y] \right] \\
& = \mathbb{E}[XY] - \mathbb{E}[X] \,\mathbb{E}[Y] - \mathbb{E}[X] \,\mathbb{E}[Y] +  \mathbb{E}[X] \, \mathbb{E}[Y] \\
& = \mathbb{E}[XY] -  \mathbb{E}[X] \,\mathbb{E}[Y]
\end{align*}
$$

In the univariate case:

$$
\mathbb{Cov}_{X,Y}[x,y] = \mathbb{E}_{X,Y} \left[ (x-\mathbb{E}_X[x]) (y-\mathbb{E}_Y[y]) \right] = \mathbb{Cov}_{Y,X}[y,x]
$$

and for multivariate random variables $$X$$ and $$Y$$ with states $$x \in \mathbb{R}^D$$ and $$y \in \mathbb{R}^E$$:

$$
\mathbb{Cov}_{X,Y}[x,y] = \mathbb{E}[xy^{\intercal}] - \mathbb{E}[x] \,\mathbb{E}[y]^{\intercal} = \mathbb{Cov}[y,x]^{\intercal} \in \mathbb{R}^{D \times E}
$$


### Variance

In the univariate case:

$$
\mathbb{Var}_X[x] = \mathbb{Cov}_X[x,x] = \mathbb{E}_{X} \left[ (x-\mathbb{E}_X[x])^2 \right] = \mathbb{E}_X[x^2] - \mathbb{E}_X[x]^2
$$

and in the multivariate case:

$$
\mathbb{Var}_X[x] = \mathbb{Cov}_X[x,x] = \mathbb{E}_X[(x-\mu)(x-\mu)^{\intercal} ] = \mathbb{E}_X[xx^{\intercal}] - \mathbb{E}_X[x] \, \mathbb{E}_X[x]^{\intercal}
$$

This is a $$D \times D$$ matrix also called the **Covariance Matrix** of the multivariate random variable $$X$$:
- it is symmetric and positive semidefinite;
- the diagonal terms are the variances $$\mathbb{Var}[x_i]$$, i.e. the covariance of each random variable with itself;
- the off-diagonal terms are $$\mathbb{Cov}[x_i, x_j]$$ for $$i,j = 1, ..., D$$ and $$i \neq j$$.

[Law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance):

$$
\mathbb{Var}[Y] = \mathbb{E} \left [ \mathbb{Var}[Y \mid X] \right] + \mathbb{Var}\left[ \mathbb{E}[Y \mid X] \right]
$$


### Correlation

[Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between random variables is a measure of covariance standardized to a limited interval $$[-1,1]$$. It is computed from the covariance (matrix) as:

$$
corr[x,y] = \frac{\mathbb{Cov}[x,y]}{\sqrt{\mathbb{Var}[x] \mathbb{Var}[y]}}  \in [-1, 1]
$$

Positive correlation means $$y$$ tends to increase as $$x$$ increases (1=perfect correlation). Negative means $$y$$ tends to decrease as $$x$$ increases. Zero means no linear correlation.


### Rules of transformations of Random Variables

The expected value and the variance of the sum and of the difference of two random variables are given by:

$$
\begin{align*}
\mathbb{E}[x+y] & = \mathbb{E}[x] + \mathbb{E}[y] \\
\mathbb{E}[x-y] & = \mathbb{E}[x] - \mathbb{E}[y] \\
\mathbb{Var}[x+y] & = \mathbb{Var}[x] + \mathbb{Var}[y] + \mathbb{Cov}[x,y] + \mathbb{Cov}[y,x] \\
\mathbb{Var}[x-y] & = \mathbb{Var}[x] + \mathbb{Var}[y] - \mathbb{Cov}[x,y] - \mathbb{Cov}[y,x]
\end{align*}
$$

For the affine transformation $$y = Ax + b$$ of $$x$$, where $$\mu$$ is the mean vector and $$\Sigma$$ is the covariance matrix of $$x$$:

$$
\begin{align*}
\mathbb{E}[Ax+b] & = A \mathbb{E}[x] + b = A \mu + b \\
\mathbb{Var}[Ax+b] & = \mathbb{Var}[Ax] = A \mathbb{Var}[x] A^{\intercal} = A \Sigma A^{\intercal}
\end{align*}
$$


### Statistical Independence

Statistical independence of random variables $$X$$ and $$Y$$ holds iff $$p(x,y)=p(x)p(y)$$. Independence implies $$p(y \mid x) = p(y)$$, $$p(x \mid y) = p(x)$$ and $$\mathbb{E}[xy] = \mathbb{E}[x] \, \mathbb{E}[y]$$, thus:

$$
\mathbb{Cov}_{X,Y}[x,y] = \mathbb{E}[xy] - \mathbb{E}[x] \, \mathbb{E}[y] = 0
$$

and thus:

$$
\mathbb{Var}_{X,Y}[x+y] = \mathbb{Var}_X[x] + \mathbb{Var}_Y[y]
$$

Important: the converse does not hold, i.e. zero covariance ($$\mathbb{Cov}_{X,Y}[x,y]=0$$) does not mean independence!


### Conditional Independence

Conditional independence of random variables $$X$$ and $$Y$$ given $$Z$$, i.e. $$X \perp Y \mid Z$$, holds iff $$p(x,y \mid z) = p(x \mid z) \, p(y \mid z)$$ for all $$z \in Z$$.

---


### Gaussian Distribution

The univariate Gaussian with mean $$\mu$$ and variance $$\sigma^2$$ is:

$$
p(x \mid \mu, \sigma^2 ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } \,\, \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)
$$

and the multivariate Gaussian with mean vector $$\mu \in \mathbb{R}^D$$ and covariance matrix $$\Sigma$$ is:

$$
p(x \mid \mu, \Sigma ) = (2 \pi)^{-\frac{D}{2}} \mid\Sigma\mid^{-\frac{1}{2}} \exp \left( -\frac{1}{2}(x-\mu)^{\intercal} \Sigma^{-1}(x-\mu) \right)
$$

The marginals and conditionals of multivariate Gaussians are also Gaussian. The joint (bivariate) Gaussian distribution of two jointly Gaussian random variables $$X$$ and $$Y$$ is:

$$
p(x,y) = \mathcal{N} \left( \begin{bmatrix} \mu_x \\ \mu_y  \end{bmatrix}, \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy} \end{bmatrix}   \right)
$$

The conditional is $$p(x \mid y) = \mathcal{N} ( \mu_{x \mid y} , \Sigma_{x \mid y})$$, and the marginal is $$p(x) = \int p(x,y) dy = \mathcal{N} ( x \mid \mu_x, \Sigma_{xx})$$. The conditional mean and covariance are given by $$\mu_{x \mid y} = \mu_x + \Sigma_{xy} \Sigma_{yy}^{-1} (y - \mu_y)$$ and $$\Sigma_{x \mid y} = \Sigma_{xx} - \Sigma_{xy} \Sigma_{yy}^{-1} \Sigma_{yx}$$ (<a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>, eq. 353). An illustration of a bivariate Gaussian is displayed below:

{: .picture}
<img width="42.3%" height="42.3%" src="{{ site.assets }}/statistics-for-ml/bivariate_gaussian.svg"/>

{: .caption}
A bivariate Gaussian with mean $$\mu = (0, 0)$$ and covariance entries $$\Sigma_{xx} = 1$$, $$\Sigma_{xy} = \Sigma_{yx} = 0.6$$ and $$\Sigma_{yy} = 2$$. Dots: samples from the joint density $$p(x,y)$$. Green: the 3σ ellipse of the joint density. Blue: the marginal density $$p(x)$$. Red: the marginal density $$p(y)$$. The bars are histograms of the samples. A conditional distribution, a slice of the joint density along $$x$$ or $$y$$, is also Gaussian. Adapted from: [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)

The product of two Gaussian pdfs $$\mathcal{N} (x \mid a, A) \, \mathcal{N}(x \mid b, B)$$ is a Gaussian scaled by a constant $$c \in \mathbb{R}$$.

If $$X,Y$$ are independent univariate Gaussian random variables, then $$p(x,y)=p(x) p(y)$$, and $$p(x+y) = \mathcal{N}(\mu_x + \mu_y, \Sigma_x + \Sigma_y)$$. For the weighted sum, $$p(ax + by) = \mathcal{N}(a\mu_x + b\mu_y, a^2 \Sigma_x + b^2 \Sigma_y)$$.

Any linear/affine transformation of a Gaussian random variable is also Gaussian. Take $$y=Ax$$ to be the transformed version of $$x$$:

$$
\begin{align*}
\mathbb{E}[y] & = \mathbb{E}[Ax] = A \mathbb{E}[x] = A\mu \\
\mathbb{Var}[y] & = \mathbb{Var}[Ax] = A \mathbb{Var}[x]A^{\intercal} = A \Sigma A^{\intercal}
\end{align*}
$$

thus $$p(y) = \mathcal{N}(y \mid A\mu, A \Sigma A^{\intercal})$$.

The sum of the squares of $$n$$ independent standard Gaussians $$Z_i \sim \mathcal{N}(0,1)$$ follows a [Chi-Square distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution) with $$n$$ degrees of freedom: $$Z_1^2 + Z_2^2 + ... + Z_n^2 \sim \chi_n^2 $$.


### Conjugacy

If the posterior distribution $$p(\theta \mid x)$$ is in the same probability distribution *family* (not necessarily the same distribution) as the prior $$p(\theta )$$, the prior and posterior are called **conjugate distributions**. The prior is then called a **[conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)** for the likelihood function $$p(x\mid \theta )$$. A conjugate prior is an algebraic convenience, giving a closed-form expression for the posterior; otherwise, numerical integration may be necessary. Every member of the exponential family has a conjugate prior, e.g. the Beta distribution for the Bernoulli distribution (MML book, Example 6.12).


### Exponential Family

The exponential family is the family of distributions that can be expressed in the form:

$$
p(x \mid \theta) = h(x) \exp \left(\eta(\theta)^{\intercal} ϕ(x) -A(\theta)\right)
$$

where:
- $$θ$$ are the **parameters** of the family;
- $$A(θ)$$ is the **log-partition function**, a normalization constant that ensures that the distribution sums up or integrates to one;
- $$ϕ(x)$$ is a **[sufficient statistic](https://en.wikipedia.org/wiki/Sufficient_statistic)** of the distribution;
- $$\eta$$ is the **natural parameter**.

Information about data (population) can be captured in $$ϕ(x)$$. Sufficient statistics carry *all* the information needed to make inference about the population, that is, they are the statistics that are sufficient to represent the distribution.

The **Fisher-Neyman theorem** states: Let $$X$$ have probability density function $$p(x \mid θ)$$. Then the statistics $$ϕ(x)$$ are sufficient for $$θ$$ if and only if $$p(x \mid θ)$$ can be written in the form $$p(x \mid \theta) = h(x) g_{\theta}(ϕ(x))$$. Here, $$h(x)$$ is a non-negative function independent of $$θ$$, and $$g_θ$$ captures all the dependence on $$θ$$ via the sufficient statistics $$ϕ(x)$$. Note that the form of the exponential family is essentially a particular expression of $$g_θ(ϕ(x))$$ in the Fisher-Neyman theorem.

As an intuition, the family can be seen as $$p(x \mid \eta) \propto \exp(\eta^{\intercal} ϕ(x))$$, ignoring $$h(x)$$ and the normalization $$A(\eta)$$ (MML book, section 6.6.3). An alternative notation in **natural form** is $$p(x \mid \eta) = h(x) \exp \left(\eta^{\intercal} T(x) -A(\eta)\right)$$, where $$T(x)$$ is the sufficient statistic $$ϕ(x)$$.

Why use the exponential family:
- they have finite-dimensional sufficient statistics;
- conjugate distributions are easy to write down, and the conjugate distributions also come from an exponential family;
- [Maximum Likelihood Estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) behaves nicely because empirical estimates of sufficient statistics are optimal estimates of the population values of sufficient statistics (recall the mean and covariance of a Gaussian);
- from an optimization perspective, the log-likelihood function is concave in the natural parameters, allowing for efficient optimization approaches to be applied.


### Moment Generating Function

The [moment generating function](https://en.wikipedia.org/wiki/Moment-generating_function) is an alternative formulation of a pmf or pdf $$p(x)$$:

$$
M(t) = \mathbb{E} \left[ e^{tX} \right]
$$

i.e. $$\sum_{x \in S} e^{tx}\, p(x)$$ for a pmf, or $$\int e^{tx}\, p(x) \, dx$$ for a pdf. If the random variables are independent, $$M_{X+Y}(t) = M_X(t) \, M_Y(t)$$. It can be used to show e.g. that the sum of independent Gaussians is a Gaussian.

---


### Law of Large Numbers

Let $$Y_n$$ be [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) random variables (of any form) with $$\mathbb{E}[Y_k]=\mu$$ for all $$k$$. Then $$\frac{1}{n} (Y_1 + ... + Y_n) \rightarrow \mu$$. In practice, this is why the average loss of a given model over many samples approximates its expected loss over the data distribution.


### Central Limit Theorem

Let $$\{ Y_n \}$$ be a sequence of iid random variables (of any form) with mean $$\mu$$ and variance $$\sigma^2$$. Then:

$$
\sqrt{n} \left( \frac{1}{n} \sum_{i=1}^n (Y_i - \mu) \right) \rightarrow \mathcal{N}(0, \sigma^2)
$$

Thus, the mean of many iid samples is approximately Gaussian, whatever their distribution, which is the basis of the confidence intervals of the Testing section.


### Inequality rules

**[Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)**, where $$\phi$$ is a convex function:

$$
\phi (\mathbb{E}[X]) \le \mathbb{E}[ \phi(X)]
$$

**[Markov's inequality](https://en.wikipedia.org/wiki/Markov%27s_inequality)**, where $$X$$ is non-negative and $$\epsilon \gt 0$$:

$$
P(X \ge \epsilon) \le \frac{\mathbb{E}[X]}{\epsilon}
$$

**[Chebyshev's inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality)**:

$$
P \left( \mid X - \mathbb{E}[X] \mid \ge \epsilon \right) \le \frac{\mathbb{Var}[X]}{\epsilon^2}
$$

**[Cauchy-Schwarz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality)**:

$$
\mathbb{Var}[X] \, \mathbb{Var}[Y] \ge \mathbb{Cov}^2[X,Y]
$$

**ELBO inequality**, for any distribution $$q_{\phi}(z)$$ of the latent variables $$z$$, where the left-hand side is the log evidence $$\ln p_{\theta}(x)$$, and the right-hand side is the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) for $$x$$:

$$
\ln p_{\theta }(x)\geq \mathbb {\mathbb {E} } _{z\sim q_{\phi }}\left[\ln {\frac {p_{\theta }(x,z)}{q_{\phi }(z)}}\right]
$$


### Kullback–Leibler / KL divergence

The [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) measures how much a distribution $$q$$ differs from a reference distribution $$p$$, and is defined as:

$$
KL ( p \| q ) =  \int_\mathbb{R} p(x) \log \frac{p(x)}{q(x)} dx
$$

From Jensen's inequality, it follows that:

$$
KL ( p \| q ) = \mathbb{E} \left[ - \log \frac{q(X)}{p(X)} \right] \ge -\log \mathbb{E} \left[ \frac{q(X)}{p(X)} \right] =0
$$

The KL divergence is not a metric because it is not symmetric: $$  KL ( q \| p ) \neq  KL ( p \| q )$$. Also, $$p=q \Leftrightarrow KL ( q \| p ) = 0$$.

---


### Testing

The **[quantile](https://en.wikipedia.org/wiki/Quantile_function)** is the inverse of the cdf. It gives the point at which the cdf (or the integral of the pdf) equals a given value $$u \in [0,1]$$. The **[p-value](https://en.wikipedia.org/wiki/P-value)** (or observed significance level) is the smallest value of $$α$$ for which the [null hypothesis](https://en.wikipedia.org/wiki/Null_hypothesis) would be rejected at level $$α$$.

The **[confidence interval](https://en.wikipedia.org/wiki/Confidence_interval)** refers to a set of values, computed from the data, that contains the population parameter for a certain proportion of times when the experiment is repeated. For example, for a $$95\%$$ **confidence level**:

{: .picture}
<img width="29.6%" height="29.6%" src="{{ site.assets }}/statistics-for-ml/confidence_interval.svg"/>

{: .caption}
A 95% confidence interval for a normally distributed estimate centred at $$\bar{X}$$: the lower and upper limits lie 1.96 standard errors either side of $$\bar{X}$$, so the interval holds a probability of 0.95, and each tail 0.025.

---


### Entropy

[Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) measures the average level of *surprise* or *uncertainty* inherent to the possible outcomes of a random variable:

$$
H(X) = -\int_{x} p(x)\log p(x) dx=\mathbb {E} [-\log p(x)]
$$

Conditional Entropy:

$$
H( Y \mid X) = - \int_x p(x) \int_y p(y \mid x) \log p(y \mid x) \,dy\,dx
$$

Joint Entropy:

$$
H(X, Y) = - \int_{xy} p(x,y) \log p(x,y) \,dx\,dy
$$

Entropy Chain Rule:

$$
H(X,Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)
$$


### Mutual Information

[Mutual information](https://en.wikipedia.org/wiki/Mutual_information) measures the reduction in uncertainty for one variable given a known value of the other variable:

$$
\begin{align*}
I(X,Y) & = \int_{x,y} p(x,y) \log \frac{p (x,y)}{p(x) \, p(y)} dx\,dy = \mathbb{E} \left[ KL \left( p(x \mid y) \,\|\, p(x) \right) \right] \\
I(X,Y) & = H(Y) - H(Y \mid X) = H(X) - H(X \mid Y)
\end{align*}
$$

---


### Arrangements

Arrangements of $$k$$ items out of a set of size $$n$$:
- **Arrangement** is the arrangement of *some* items in which order matters: $$^nA_k=n^k$$ with replacement or $$\frac{n!}{(n-k)!}$$ without.
- **Permutation** is the arrangement of *all* items in which order matters: $$^nP_n = n!$$, the case $$k=n$$ of $$^nP_k=\frac{n!}{(n-k)!}$$.
- **Combination** is the arrangement of *some* items in which order *does not* matter (without replacement): $$\binom nk = \, ^nC_k = \frac{1}{k!} ^nP_k = \frac{n!}{k!(n-k)!}$$.

---


### Sets

The **union** of two subsets is written as $$F_1 \cup F_2 = \{ ω ∈ Ω : ω ∈ F_1 \text{ or } ω ∈ F_2 \}$$, and the **intersection** is $$F_1 ∩ F_2 = \{ ω ∈ Ω : ω ∈ F_1 \text{ and } ω ∈ F_2 \}$$. Two events $$F_1$$ and $$F_2$$ are **disjoint** if they have no elements in common, or $$F_1 ∩ F_2 = ∅$$.

The **complement** of $$F$$ is written as $$F^c$$ and contains all elements of $$\Omega$$ that are not in $$F$$, i.e. $$F^c = \{ ω ∈ Ω : ω \not\in F \}$$. From this, $$F ∪ F^c = Ω$$.

A **partition** $$\{ F_n \}$$  for $$n \ge 1$$ is a collection of events such that $$F_i ∩ F_j = ∅$$ for all $$i \neq j$$ and $$\cup_{n≥1} F_n = Ω$$. The **difference** between $$F_1$$ and $$F_2$$ is defined as $$F_1 \backslash F_2 = F_1 ∩ F_2^c$$.

The following **properties** hold:
- associativity: $$(F_1 ∪ F_2) ∪ F_3 = F_1 ∪ (F_2 ∪ F_3) = F_1 ∪ F_2 ∪ F_3$$
- associativity: $$(F_1 ∩ F_2) ∩ F_3 = F_1 ∩ (F_2 ∩ F_3) = F_1 ∩ F_2 ∩ F_3$$
- distributivity: $$F_1 ∩ (F_2 ∪ F_3) = (F_1 ∩ F_2) ∪ (F_1 ∩ F_3)$$
- distributivity: $$F_1 ∪ (F_2 ∩ F_3) = (F_1 ∪ F_2) ∩ (F_1 ∪ F_3)$$
- [De Morgan's Laws](https://en.wikipedia.org/wiki/De_Morgan%27s_laws): $$(F_1 ∪ F_2)^c = F^c_1 ∩ F^c_2$$  and  $$(F_1 ∩ F_2)^c = F^c_1 ∪ F^c_2$$


Using these properties and the axioms of probability, it can be shown that:

$$
\begin{align*}
& P(F_1 ∪ F_2) = P(F_1) − P(F_1 ∩ F_2) + P(F_2) \\
& P(F_1 ∩ F_2) ≤ \min\{P(F_1), P(F_2)\} \\
& F ∪ F^c = Ω, \quad 1 = P(Ω) = P(F) + P(F^c), \text{ thus } P(F^c) = 1 − P(F)
\end{align*}
$$

---


### Causal Inference

A **[Bayesian network](https://en.wikipedia.org/wiki/Bayesian_network)** is a probabilistic model for which a directed acyclic graph (DAG) expresses the conditional dependence structure between random variables (a [causal model](https://en.wikipedia.org/wiki/Causal_model) when the edges represent direct causes). The model represents a factorization of the joint probability of all random variables:

$$
p(X_1,\ldots ,X_n) =\prod_{i=1}^{n} p \left( X_i \mid \text{pa}(X_i) \right)
$$

where $$\text{pa}(X_i)$$ are the parents of $$X_i$$. Example:

$$
p(A,B,C,D) = p(A)\cdot p(B \mid A) \cdot p( C \mid A) \cdot p ( D \mid A,C )
$$

The following diagram illustrates the corresponding graph, with an arrow from each parent to its child:

{: .picture}
<img width="10.5%" height="10.5%" src="{{ site.assets }}/statistics-for-ml/directed_graph.svg"/>

{: .caption}
A directed acyclic graph of the factorization $$p(A,B,C,D) = p(A) \cdot p(B \mid A) \cdot p(C \mid A) \cdot p(D \mid A,C)$$: each node depends only on its parents. Adapted from: [Graphical model](https://en.wikipedia.org/wiki/Graphical_model)

---


### Extra: Non-parametric statistics (Smoothing)

A **[Kernel Density Estimator](https://en.wikipedia.org/wiki/Kernel_density_estimation)** is used to convert a discrete distribution into a continuous distribution. Formulation: given a discrete cumulative distribution function (the [empirical distribution function](https://en.wikipedia.org/wiki/Empirical_distribution_function)) with $$n$$ steps at $$Y_1, ..., Y_n$$, defined as:

$$
\hat{F}_n (y) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(Y_i \le y)
$$  

the kernel density estimator $$\hat{f}$$ is the random density function:

$$
\hat{f}(x) \, = \, \frac{1}{nh} \sum_{i=1}^n K \left( \frac{x-Y_i}{h}  \right)
$$

where $$K$$ is the kernel, a non-negative function that integrates to 1, and $$h \gt 0$$ is the smoothing parameter (*bandwidth*). $$h$$ regulates the bias-variance trade-off: a large vs. small $$h$$ yields a flat vs. wiggly estimator. An example of $$K$$ is the standard normal pdf, as in the following picture:

{: .picture}
<img width="61.4%" height="61.4%" src="{{ site.assets }}/statistics-for-ml/kde.svg"/>

{: .caption}
Histogram with bins of width 2 (left) and Gaussian kernel density estimate with bandwidth $$h = 1.5$$ (right) of the same six data points. The estimate is the sum of six kernels centred at the points, each divided by $$n$$. Adapted from: [Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation)

Compared to the histogram on the left, the kernel density estimate is smooth and converges faster to the true density of a continuous random variable.

The following table summarizes the method used for each use case:

|-|-|-|
| **Distribution / function $$g$$** | **parametric $$g(x_i) = x_i^{\intercal} \beta$$** | **non-parametric $$g$$** |
| Gaussian | Linear Regression | Smoothing (e.g. [kernel regression](https://en.wikipedia.org/wiki/Kernel_regression)) |
| Exponential Family | [Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model) | [Generalized Additive Models](https://en.wikipedia.org/wiki/Generalized_additive_model) |

To find more about the topic, the book [Mathematics for Machine Learning](https://mml-book.github.io/) by Deisenroth, Faisal and Ong covers probability and distributions in chapter 6. The book [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) by Bishop covers information theory (section 1.6), the Gaussian distribution and the exponential family (sections 2.3 and 2.4), kernel density estimators (section 2.5.1) and Bayesian networks (section 8.1).
