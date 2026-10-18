---
layout: post
title:  "Statistics for ML"
redirect_from: /statistics
categories: ["Machine Learning"]
tags: ["statistics", "probability", "Bayes rule", "Gaussian distribution", "KL divergence"]
---

A summary of the probability and statistics used in ML, taken from the books in the <a href="{{ site.resources_permalink }}">resources</a> section, mainly chapter 6 of *Mathematics for Machine Learning* by Deisenroth, Faisal and Ong.

## Probability

### Events and Probability

The **sample space** $$\Omega$$ is the set of all possible outcomes of an experiment. An **event** is a subset of $$\Omega$$, and the **event space** $$\mathcal{A}$$, a [σ-algebra](https://en.wikipedia.org/wiki/%CE%A3-algebra) on $$\Omega$$, is the set of events that have a probability. Events $$A$$ and $$B$$ combine by **union** $$A \cup B$$, **intersection** $$A \cap B$$, **complement** $$A^c$$ and **difference** $$A \setminus B = A \cap B^c$$. They are **disjoint** if $$A \cap B = \emptyset$$, and a **partition** of $$\Omega$$ is a set of pairwise disjoint events whose union is $$\Omega$$. Union and intersection are associative, each distributes over the other, and [De Morgan's laws](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) give $$(A \cup B)^c = A^c \cap B^c$$ and $$(A \cap B)^c = A^c \cup B^c$$.

The **probability** $$P(A)$$ measures how likely the event $$A$$ is, or the degree of belief in $$A$$. It satisfies the [axioms of probability](https://en.wikipedia.org/wiki/Probability_axioms): $$P(A) \ge 0$$, $$P(\Omega) = 1$$, and the probability of a countable union of pairwise disjoint events is the sum of their probabilities. The axioms imply:

$$
\begin{align*}
P(A^c) & = 1 - P(A) \\
P(A \cup B) & = P(A) + P(B) - P(A \cap B) \le P(A) + P(B) \\
P(A \cap B) & \le \min \{ P(A), P(B) \}
\end{align*}
$$

### Counting

When all outcomes are equally likely, $$P(A) = \vert A \vert / \vert \Omega \vert$$, and a probability is a ratio of counts. The number of ways to select $$k$$ items from a set of $$n$$ items is:

|   | With replacement | Without replacement |
|---|---|---|
| Ordered | $$n^k$$ | $$n! / (n-k)!$$ |
| Unordered | $$\binom{n+k-1}{k}$$ | $$\binom{n}{k} = \frac{n!}{k! \, (n-k)!}$$ |

Unordered selections are **combinations**, and a **permutation** orders all $$n$$ items, in $$n!$$ ways.

### Random Variables

A **random variable** is a function $$X : \Omega \to \mathcal{T}$$ that maps each outcome to a value $$x$$ in a target space $$\mathcal{T}$$: despite its name, it is neither random nor a variable. For two coin tosses and $$X$$ the number of heads, $$\Omega = \{hh, ht, th, tt\}$$, $$\mathcal{T} = \{0, 1, 2\}$$, $$X(hh) = 2$$, $$X(ht) = X(th) = 1$$ and $$X(tt) = 0$$.

A discrete random variable has a **probability mass function** (pmf) $$p(x) = P(X = x)$$. A continuous one has a **probability density function** (pdf) $$p(x)$$, with $$P(a \le X \le b) = \int_a^b p(x) \, dx$$. Any non-negative function that sums (or integrates) to 1 over its domain is a valid pmf (or pdf). A pdf value can exceed 1: only its integral over an interval is a probability. The **cumulative distribution function** (CDF) is $$F(x) = P(X \le x)$$, or $$\int_{-\infty}^{x} p(t) \, dt$$ for a continuous variable. It also exists when there is no pmf or pdf. The formula sheets of the EPFL course *Statistics for Data Science* list common <a href="{{ site.epfl_statistics }}/formula-sheet-discrete-distributions.pdf">discrete</a> and <a href="{{ site.epfl_statistics }}/formula-sheet-continuous-distributions.pdf">continuous</a> distributions.

### Sum and Product Rules

The **sum rule**, or marginalization, gives the distribution of $$x$$ from the joint distribution of $$x$$ and $$y$$. The **product rule** factorizes a joint distribution into a conditional and a marginal distribution:

$$
\begin{align*}
p(x) & = \sum_y p(x, y) \quad \text{or} \quad p(x) = \int p(x, y) \, dy \\
p(x, y) & = p(y \mid x) \, p(x) = p(x \mid y) \, p(y)
\end{align*}
$$

For two discrete variables observed $$N$$ times, the frequency $$n_{ij} / N$$ of cell $$(i, j)$$ in the [contingency table](https://en.wikipedia.org/wiki/Contingency_table), with $$n_{ij}$$ observations of $$X = x_i$$ and $$Y = y_j$$, estimates $$P(X = x_i, Y = y_j)$$. By the sum rule, the sum of row $$i$$ divided by $$N$$ estimates the marginal $$P(X = x_i)$$.

### Bayes' Rule

The product rule in both orders, $$p(x \mid \theta) \, p(\theta) = p(\theta \mid x) \, p(x)$$, gives **Bayes' rule**. It updates the **prior** $$p(\theta)$$, the knowledge about an unobserved variable $$\theta$$, after observing $$x$$:

$$
p(\theta \mid x) = \frac{p(x \mid \theta) \, p(\theta)}{p(x)}, \qquad \text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}
$$

The **evidence**, or **marginal likelihood**, $$p(x) = \int p(x \mid \theta) \, p(\theta) \, d\theta = \mathbb{E}_{\theta \sim p(\theta)}[p(x \mid \theta)]$$ normalizes the posterior. In Bayesian ML, $$\theta$$ holds the model parameters, $$x$$ the data, and the posterior $$p(\theta \mid x)$$ is the quantity of interest. The evidence integral is usually intractable: conjugate priors give it in closed form, and variational inference bounds it (both below).

## Moments and Independence

### Expected Value

The **expected value** of a function $$g$$ of a random variable $$X$$ with pdf $$p(x)$$ is

$$
\mathbb{E}_X[g(x)] = \int g(x) \, p(x) \, dx,
$$

or $$\sum_x g(x) \, p(x)$$ for a pmf. For a random vector $$x \in \mathbb{R}^D$$, the expectation of a vector-valued $$g$$ is taken element-wise, and the **mean** is $$\mu = \mathbb{E}[x] = [\mathbb{E}[x_1], \dots, \mathbb{E}[x_D]]^\intercal$$, where each $$\mathbb{E}[x_d]$$ uses the marginal $$p(x_d)$$. The [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation) computes an expectation in two stages: $$\mathbb{E}[y] = \mathbb{E}\left[ \mathbb{E}[y \mid x] \right]$$.

### Covariance and Variance

The **covariance** of two random variables is the expected product of their deviations from their means:

$$
\begin{align*}
\mathrm{Cov}[x, y] & = \mathbb{E}\left[ (x - \mathbb{E}[x]) \, (y - \mathbb{E}[y]) \right] \\
& = \mathbb{E}\left[ xy - x \, \mathbb{E}[y] - \mathbb{E}[x] \, y + \mathbb{E}[x] \, \mathbb{E}[y] \right] \\
& = \mathbb{E}[xy] - \mathbb{E}[x] \, \mathbb{E}[y]
\end{align*}
$$

For random vectors $$x \in \mathbb{R}^D$$ and $$y \in \mathbb{R}^E$$, it is the $$D \times E$$ matrix $$\mathrm{Cov}[x, y] = \mathbb{E}[x y^\intercal] - \mathbb{E}[x] \, \mathbb{E}[y]^\intercal = \mathrm{Cov}[y, x]^\intercal$$.

The **variance** is the covariance of a variable with itself, $$\mathrm{Var}[x] = \mathrm{Cov}[x, x] = \mathbb{E}\left[ (x - \mu)^2 \right] = \mathbb{E}[x^2] - \mu^2$$. For a random vector, it is the $$D \times D$$ **covariance matrix**

$$
\Sigma = \mathrm{Var}[x] = \mathbb{E}\left[ (x - \mu)(x - \mu)^\intercal \right] = \mathbb{E}[x x^\intercal] - \mu \mu^\intercal,
$$

which is symmetric and positive semidefinite, with the variances $$\mathrm{Var}[x_d]$$ on its diagonal and the covariances $$\mathrm{Cov}[x_d, x_{d'}]$$, $$d \neq d'$$, off the diagonal. The [law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance) states:

$$
\mathrm{Var}[y] = \mathbb{E}\left[ \mathrm{Var}[y \mid x] \right] + \mathrm{Var}\left[ \mathbb{E}[y \mid x] \right]
$$

From $$N$$ samples $$x_1, \dots, x_N$$, the **empirical mean** and the **empirical covariance** are

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i, \qquad \hat{\Sigma} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})(x_i - \bar{x})^\intercal.
$$

Since $$\mathbb{E}[\hat{\Sigma}] = \frac{N - 1}{N} \Sigma$$, dividing by $$N - 1$$ instead of $$N$$ gives an unbiased estimate ([Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)).

Computed from running sums of $$x_i$$ and $$x_i^2$$, the form $$\mathbb{E}[x^2] - \mu^2$$ loses precision in floating point when the variance is small relative to $$\mu^2$$ ([catastrophic cancellation](https://en.wikipedia.org/wiki/Catastrophic_cancellation)). [Welford's algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance) instead updates the mean and the sum of squared deviations $$S = \sum_i (x_i - \bar{x})^2$$ one sample at a time, and the merge formula of Chan et al. combines these statistics for two disjoint sets of samples $$A$$ and $$B$$:

$$
N = N_A + N_B, \qquad \delta = \bar{x}_B - \bar{x}_A, \qquad \bar{x} = \bar{x}_A + \delta \, \frac{N_B}{N}, \qquad S = S_A + S_B + \delta^2 \, \frac{N_A N_B}{N}
$$

Welford's update is the case $$N_B = 1$$. GPU kernels for layer normalization, for example, run Welford's update in each thread and this merge in a tree reduction.

### Correlation

The [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is the covariance standardized to $$[-1, 1]$$:

$$
\mathrm{corr}[x, y] = \frac{\mathrm{Cov}[x, y]}{\sqrt{\mathrm{Var}[x] \, \mathrm{Var}[y]}}
$$

A positive (negative) correlation means that $$y$$ tends to increase (decrease) as $$x$$ increases. $$\vert \mathrm{corr}[x, y] \vert = 1$$ if and only if $$y = ax + b$$ for constants $$a \neq 0$$ and $$b$$, and $$\mathrm{corr}[x, y] = 0$$ means no *linear* relation.

### Linear Transformations

For random variables $$x$$ and $$y$$ of the same dimension, dependent or not:

$$
\begin{align*}
\mathbb{E}[x \pm y] & = \mathbb{E}[x] \pm \mathbb{E}[y] \\
\mathrm{Var}[x \pm y] & = \mathrm{Var}[x] + \mathrm{Var}[y] \pm \left( \mathrm{Cov}[x, y] + \mathrm{Cov}[y, x] \right)
\end{align*}
$$

For an affine map $$Wx + b$$, such as a linear layer, of a random vector $$x$$ with mean $$\mu$$ and covariance $$\Sigma$$:

$$
\mathbb{E}[Wx + b] = W \mu + b, \qquad \mathrm{Var}[Wx + b] = W \Sigma W^\intercal
$$

### Independence

$$X$$ and $$Y$$ are **independent** if and only if $$p(x, y) = p(x) \, p(y)$$. Then $$p(y \mid x) = p(y)$$, $$p(x \mid y) = p(x)$$ and $$\mathbb{E}[xy] = \mathbb{E}[x] \, \mathbb{E}[y]$$, so $$\mathrm{Cov}[x, y] = 0$$ and $$\mathrm{Var}[x + y] = \mathrm{Var}[x] + \mathrm{Var}[y]$$. The converse does not hold: for $$x \sim \mathcal{N}(0, 1)$$ and $$y = x^2$$, $$\mathrm{Cov}[x, y] = \mathbb{E}[x^3] = 0$$, yet $$y$$ is a function of $$x$$.

The pdf of the sum of two independent random variables is the [convolution](https://en.wikipedia.org/wiki/Convolution_of_probability_distributions) of their pdfs:

$$
p_{X+Y}(u) = \int_{-\infty}^{+\infty} p_X(u - v) \, p_Y(v) \, dv
$$

$$X$$ and $$Y$$ are **conditionally independent** given $$Z$$, written $$X \perp Y \mid Z$$, if and only if $$p(x, y \mid z) = p(x \mid z) \, p(y \mid z)$$ for all $$z$$.

## Distributions

### Gaussian Distribution

The univariate Gaussian with mean $$\mu$$ and variance $$\sigma^2$$, and the multivariate Gaussian with mean $$\mu \in \mathbb{R}^D$$ and covariance $$\Sigma$$, have the pdfs:

$$
\begin{align*}
\mathcal{N}(x \mid \mu, \sigma^2) & = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2 \sigma^2} \right) \\
\mathcal{N}(x \mid \mu, \Sigma) & = \frac{1}{\sqrt{(2 \pi)^D \det(\Sigma)}} \exp\left( -\frac{1}{2} (x - \mu)^\intercal \Sigma^{-1} (x - \mu) \right)
\end{align*}
$$

Two Gaussian random vectors $$x$$ and $$y$$ are not always jointly Gaussian. When they are, their joint distribution is

$$
p(x, y) = \mathcal{N}\left( \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy} \end{bmatrix} \right),
$$

and the marginal and conditional distributions are also Gaussian:

$$
\begin{align*}
p(x) & = \int p(x, y) \, dy = \mathcal{N}(x \mid \mu_x, \Sigma_{xx}) \\
p(x \mid y) & = \mathcal{N}(x \mid \mu_{x \mid y}, \Sigma_{x \mid y}) \\
\mu_{x \mid y} & = \mu_x + \Sigma_{xy} \Sigma_{yy}^{-1} (y - \mu_y) \\
\Sigma_{x \mid y} & = \Sigma_{xx} - \Sigma_{xy} \Sigma_{yy}^{-1} \Sigma_{yx}
\end{align*}
$$

The conditional covariance, the [Schur complement](https://en.wikipedia.org/wiki/Schur_complement) of $$\Sigma_{yy}$$, does not depend on the observed $$y$$. Marginalizing only drops blocks, while conditioning solves linear systems in $$\Sigma_{yy}$$, usually with a [Cholesky factorization](https://en.wikipedia.org/wiki/Cholesky_decomposition) in $$O(E^3)$$ operations for $$y \in \mathbb{R}^E$$: the main cost of [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) regression. The same formulas give the update step of the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter).

{: .picture}
<img width="42.3%" height="42.3%" src="{{ site.assets }}/statistics-for-ml/bivariate_gaussian.svg"/>

{: .caption}
A bivariate Gaussian with mean $$\mu = (0, 0)$$ and covariance entries $$\Sigma_{xx} = 1$$, $$\Sigma_{xy} = \Sigma_{yx} = 0.6$$ and $$\Sigma_{yy} = 2$$. Dots: samples from the joint density $$p(x,y)$$. Green: the 3σ ellipse of the joint density. Blue: the marginal density $$p(x)$$. Red: the marginal density $$p(y)$$. The bars are histograms of the samples. A conditional distribution, a slice of the joint density along $$x$$ or $$y$$, is also Gaussian. Adapted from: [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)

The product of two Gaussian pdfs in $$x$$ is a Gaussian scaled by a constant $$c$$ (equation 371 of [The Matrix Cookbook]({{ site.assets }}/resources/the_matrix_cookbook.pdf)):

$$
\begin{align*}
\mathcal{N}(x \mid \mu_1, \Sigma_1) \, \mathcal{N}(x \mid \mu_2, \Sigma_2) & = c \, \mathcal{N}(x \mid \mu_c, \Sigma_c), \qquad c = \mathcal{N}(\mu_1 \mid \mu_2, \Sigma_1 + \Sigma_2) \\
\Sigma_c & = \left( \Sigma_1^{-1} + \Sigma_2^{-1} \right)^{-1} \\
\mu_c & = \Sigma_c \left( \Sigma_1^{-1} \mu_1 + \Sigma_2^{-1} \mu_2 \right)
\end{align*}
$$

The precisions (inverse covariances) add, and the mean is the precision-weighted average of the two means. This gives the posterior for a Gaussian prior and a Gaussian likelihood.

An affine map of a Gaussian vector is Gaussian, with the moments given by the transformation rules above: $$Wx + b \sim \mathcal{N}(W \mu + b, W \Sigma W^\intercal)$$. For independent Gaussian vectors $$x$$ and $$y$$, $$x + y \sim \mathcal{N}(\mu_x + \mu_y, \Sigma_{xx} + \Sigma_{yy})$$.

The sum of the squares of $$D$$ independent standard Gaussians follows the [chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution) $$\chi^2_D$$ with $$D$$ degrees of freedom. Hence the squared [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) $$(x - \mu)^\intercal \Sigma^{-1} (x - \mu)$$ of $$x \sim \mathcal{N}(\mu, \Sigma)$$ follows $$\chi^2_D$$, which gives the probability inside each elliptical contour of the pdf: in two dimensions, the contour at distance 3 holds $$1 - e^{-4.5} \approx 98.9\%$$, against $$99.7\%$$ within 3 standard deviations in one dimension.

### Conjugate Priors

If the posterior $$p(\theta \mid x)$$ is in the same family of distributions as the prior $$p(\theta)$$, the prior is a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) for the likelihood $$p(x \mid \theta)$$. The posterior then has a closed form, and the evidence needs no numerical integration. Common pairs (likelihood: conjugate prior) are Bernoulli or binomial: beta; categorical or multinomial: Dirichlet; Poisson: gamma; Gaussian with a known variance: Gaussian. Every likelihood in the exponential family has a conjugate prior, which is also in the exponential family.

### Exponential Family

A distribution with parameters $$\theta$$ is in the [exponential family](https://en.wikipedia.org/wiki/Exponential_family) if its pmf or pdf can be written as

$$
p(x \mid \eta) = h(x) \exp\left( \eta^\intercal T(x) - A(\eta) \right),
$$

where $$\eta = \eta(\theta)$$ are the **natural parameters**, $$T(x)$$ the **sufficient statistics**, $$h(x)$$ the base measure, and $$A(\eta)$$ the **log-partition function**, the logarithm of the normalization constant. For a Bernoulli variable $$x \in \{0, 1\}$$ with mean $$\mu = P(x = 1)$$:

$$
p(x \mid \mu) = \mu^x (1 - \mu)^{1 - x} = \exp\left( x \log \frac{\mu}{1 - \mu} + \log(1 - \mu) \right)
$$

so $$T(x) = x$$, $$h(x) = 1$$, the natural parameter is the [logit](https://en.wikipedia.org/wiki/Logit) $$\eta = \log \frac{\mu}{1 - \mu}$$, and $$A(\eta) = -\log(1 - \mu) = \log(1 + e^\eta)$$. Inverting the logit gives the logistic sigmoid $$\mu = 1 / (1 + e^{-\eta})$$ of logistic regression. The Gaussian has $$T(x) = [x, x^2]^\intercal$$.

A statistic $$T(x)$$ is [sufficient](https://en.wikipedia.org/wiki/Sufficient_statistic) for $$\theta$$ if the distribution of the data given $$T(x)$$ does not depend on $$\theta$$, i.e. $$T(x)$$ holds all the information that the data carry about $$\theta$$. By the **Fisher–Neyman factorization theorem**, this holds if and only if $$p(x \mid \theta) = h(x) \, g_\theta(T(x))$$ for some $$h(x) \ge 0$$ independent of $$\theta$$. The exponential family has this form, with $$g_\theta(t) = \exp(\eta^\intercal t - A(\eta))$$.

The derivatives of the log-partition function are the moments of the sufficient statistics, $$\nabla_\eta A(\eta) = \mathbb{E}[T(x)]$$ and $$\nabla_\eta^2 A(\eta) = \mathrm{Var}[T(x)]$$, so $$A$$ is convex. For $$N$$ independent samples, this gives the family its main advantages:
- the log-likelihood $$\sum_i \log h(x_i) + \eta^\intercal \sum_i T(x_i) - N A(\eta)$$ is concave in $$\eta$$, so every local maximum is global;
- at the maximum, the moments match, $$\mathbb{E}_\eta[T(x)] = \frac{1}{N} \sum_i T(x_i)$$: for a Gaussian, the estimates are the empirical mean and the empirical covariance $$\hat{\Sigma}$$;
- the data enter only through $$\sum_i T(x_i)$$, whose size does not grow with $$N$$, so a streaming estimator keeps only this sum. Among families whose support does not depend on $$\theta$$, only the exponential family has sufficient statistics of a fixed size (Pitman–Koopman–Darmois theorem).

Members include the Gaussian, exponential, gamma, chi-squared, beta, Dirichlet, Bernoulli, categorical and Poisson distributions, and the binomial and multinomial with a fixed number of trials. Student's t, most mixtures (such as Gaussian mixture models) and the uniform distribution with unknown bounds are not members. The [exponential family post]({{ site.baseurl }}{% post_url 2019-03-20-exponential-family %}) derives the Gaussian and Bernoulli cases and their maximum likelihood estimates.

### Moment Generating Function

The [moment generating function](https://en.wikipedia.org/wiki/Moment-generating_function) (MGF) of $$X$$ is $$M_X(t) = \mathbb{E}[e^{tX}]$$. Its $$k$$-th derivative at $$t = 0$$ is the $$k$$-th moment, $$M_X^{(k)}(0) = \mathbb{E}[X^k]$$. When the MGF exists in an interval around 0 (it does not for the log-normal distribution), it determines the distribution. For independent $$X$$ and $$Y$$, $$M_{X+Y}(t) = M_X(t) \, M_Y(t)$$. With $$M_X(t) = \exp(\mu t + \sigma^2 t^2 / 2)$$ for $$X \sim \mathcal{N}(\mu, \sigma^2)$$, this shows that the sum of independent Gaussians is Gaussian, with the means and the variances added.

## Limit Theorems and Tests

### Inequalities

For a convex function $$f$$, [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) states:

$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]
$$

For a non-negative $$X$$ and $$\epsilon > 0$$, [Markov's inequality](https://en.wikipedia.org/wiki/Markov%27s_inequality) bounds the tail probability. Applied to $$(X - \mathbb{E}[X])^2$$, it gives [Chebyshev's inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality) for any $$X$$ with a finite variance:

$$
P(X \ge \epsilon) \le \frac{\mathbb{E}[X]}{\epsilon}, \qquad P\left( \vert X - \mathbb{E}[X] \vert \ge \epsilon \right) \le \frac{\mathrm{Var}[X]}{\epsilon^2}
$$

The [Cauchy–Schwarz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality) for random variables, $$\mathrm{Cov}[X, Y]^2 \le \mathrm{Var}[X] \, \mathrm{Var}[Y]$$, bounds the correlation to $$[-1, 1]$$.

### Law of Large Numbers

Let $$X_1, \dots, X_N$$ be independent and identically distributed (iid) random variables with a finite mean $$\mu$$, and let $$\bar{X}_N = \frac{1}{N} \sum_{i=1}^{N} X_i$$ be their sample mean. The [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) states that $$\bar{X}_N \to \mu$$ as $$N \to \infty$$, in probability (weak law) and almost surely (strong law). With a finite variance $$\sigma^2$$, the weak law follows from $$\mathrm{Var}[\bar{X}_N] = \sigma^2 / N$$ and Chebyshev's inequality:

$$
P\left( \vert \bar{X}_N - \mu \vert \ge \epsilon \right) \le \frac{\sigma^2}{N \epsilon^2} \to 0
$$

### Central Limit Theorem

If the iid $$X_i$$ also have a finite variance $$\sigma^2 > 0$$, the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) states that the scaled error of the sample mean converges in distribution to a Gaussian, whatever the distribution of the $$X_i$$:

$$
\sqrt{N} \left( \bar{X}_N - \mu \right) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

The standard error of the mean, $$\sigma / \sqrt{N}$$, falls with the square root of the sample size. Halving the error of a Monte Carlo estimate, such as a mini-batch gradient, takes four times as many samples.

### Confidence Intervals and Hypothesis Tests

The [quantile function](https://en.wikipedia.org/wiki/Quantile_function) $$Q = F^{-1}$$ inverts the CDF: $$Q(u)$$ is the value below which a fraction $$u$$ of the probability lies, e.g. $$Q(0.975) \approx 1.96$$ for the standard Gaussian.

A [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval) at level $$1 - \alpha$$ is a random interval, computed from the data, that contains the fixed true parameter with probability $$1 - \alpha$$ over repeated samples; a single computed interval contains it or not. For the mean, the central limit theorem gives the approximate interval $$\bar{X}_N \pm Q(1 - \alpha / 2) \, \sigma / \sqrt{N}$$, with $$Q$$ the standard Gaussian quantile function. At the $$95\%$$ level, this is $$\bar{X}_N \pm 1.96 \, \sigma / \sqrt{N}$$, whose limits cut $$2.5\%$$ of the probability from each tail (figure). With an unknown $$\sigma$$, the empirical standard deviation with Bessel's correction replaces $$\sigma$$, and the quantile of [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) with $$N - 1$$ degrees of freedom replaces $$1.96$$.

{: .picture}
<img width="29.6%" height="29.6%" src="{{ site.assets }}/statistics-for-ml/confidence_interval.svg"/>

{: .caption}
A 95% confidence interval for a normally distributed estimate centred at $$\bar{X}$$: the lower and upper limits lie 1.96 standard errors either side of $$\bar{X}$$, so the interval holds a probability of 0.95, and each tail 0.025.

A [hypothesis test](https://en.wikipedia.org/wiki/Statistical_hypothesis_test) rejects a null hypothesis $$H_0$$ when the data give enough evidence against it. The **significance level** $$\alpha$$ is the probability of rejecting $$H_0$$ when it is true (a type I error). The [p-value](https://en.wikipedia.org/wiki/P-value) is the probability, under $$H_0$$, of a test statistic at least as extreme as the observed one, i.e. the smallest $$\alpha$$ at which the test rejects $$H_0$$. The test rejects $$H_0$$ when the p-value is at most $$\alpha$$.

## Information Theory

### Entropy

The [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of a discrete random variable is the expected surprise $$-\log p(x)$$ of its outcomes:

$$
H(X) = -\sum_x p(x) \log p(x) = \mathbb{E}[-\log p(x)]
$$

in bits for $$\log_2$$ and in nats for $$\ln$$. It is maximal, $$\log n$$, for the uniform distribution over $$n$$ outcomes. For a continuous variable, the integral $$-\int p(x) \log p(x) \, dx$$ is the [differential entropy](https://en.wikipedia.org/wiki/Differential_entropy), which can be negative. Among all distributions with variance $$\sigma^2$$, $$\mathcal{N}(\mu, \sigma^2)$$ has the largest one, $$\frac{1}{2} \log(2 \pi e \sigma^2)$$. The **joint** and **conditional** entropies, and the **chain rule** that relates them, are:

$$
\begin{align*}
H(X, Y) & = \mathbb{E}[-\log p(x, y)], \qquad H(Y \mid X) = \mathbb{E}[-\log p(y \mid x)] \\
H(X, Y) & = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)
\end{align*}
$$

### Cross-Entropy and KL Divergence

The [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) of a distribution $$q$$ relative to a distribution $$p$$ is the expected surprise of $$q$$ for samples of $$p$$. The [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) is its excess over the entropy $$H(p)$$ of $$p$$:

$$
H(p, q) = \mathbb{E}_{x \sim p}[-\log q(x)], \qquad \mathrm{KL}(p \Vert q) = \mathbb{E}_{x \sim p}\left[ \log \frac{p(x)}{q(x)} \right] = H(p, q) - H(p)
$$

Jensen's inequality with the convex function $$-\log$$ shows that the KL divergence is non-negative:

$$
\mathrm{KL}(p \Vert q) = \mathbb{E}_{x \sim p}\left[ -\log \frac{q(x)}{p(x)} \right] \ge -\log \mathbb{E}_{x \sim p}\left[ \frac{q(x)}{p(x)} \right] = -\log \int q(x) \, dx = 0
$$

with equality if and only if $$p = q$$ almost everywhere. The KL divergence is not a distance: it is not symmetric, $$\mathrm{KL}(p \Vert q) \neq \mathrm{KL}(q \Vert p)$$ in general, and it does not satisfy the triangle inequality.

In classification, $$p$$ is the one-hot distribution of the true label $$y$$ and $$q$$ the predicted distribution, so the cross-entropy loss $$-\log q(y)$$ is the negative log-likelihood of the label, and equals $$\mathrm{KL}(p \Vert q)$$. Uniform predictions over $$n$$ classes give a loss of $$\ln n$$, a quick check of the loss at initialization ($$\ln 10 \approx 2.30$$ for 10 classes).

When a model $$q$$ is fitted to a target $$p$$, the direction of the divergence matters. Minimizing $$\mathrm{KL}(p \Vert q)$$, as maximum likelihood does, spreads $$q$$ over all the probability of $$p$$ (mass-covering). Minimizing $$\mathrm{KL}(q \Vert p)$$, as variational inference does, penalizes $$q$$ where $$p$$ is small, so $$q$$ tends to concentrate on one mode of $$p$$ (mode-seeking).

For a model $$p(x, z)$$ with latent variables $$z$$ and any distribution $$q(z)$$, the log evidence splits into the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) and a KL divergence:

$$
\log p(x) = {\underbrace{\mathbb{E}_{z \sim q}\left[ \log \frac{p(x, z)}{q(z)} \right]}_{\text{ELBO}}} + \mathrm{KL}\left( q(z) \Vert p(z \mid x) \right) \ge \text{ELBO}
$$

Maximizing the ELBO over $$q$$ thus minimizes $$\mathrm{KL}(q(z) \Vert p(z \mid x))$$, the divergence of $$q$$ from the posterior, without computing the evidence. This is the objective of [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) and of [variational autoencoders]({{ site.baseurl }}{% post_url 2019-10-12-vae-gan %}) ([Kingma and Welling, 2014](https://arxiv.org/abs/1312.6114)). The [variational inference post]({{ site.baseurl }}{% post_url 2019-07-01-variational-inference-gmm %}) applies it to Gaussian mixtures.

### Mutual Information

The [mutual information](https://en.wikipedia.org/wiki/Mutual_information) measures how much knowing one variable reduces the uncertainty about the other. It is the KL divergence between the joint distribution and the product of the marginals:

$$
\begin{align*}
I(X; Y) & = \mathrm{KL}\left( p(x, y) \Vert p(x) \, p(y) \right) = \mathbb{E}_{y}\left[ \mathrm{KL}\left( p(x \mid y) \Vert p(x) \right) \right] \\
& = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)
\end{align*}
$$

Hence $$I(X; Y) \ge 0$$, with equality if and only if $$X$$ and $$Y$$ are independent: unlike the covariance, the mutual information detects any dependence.

## Bayesian Networks

A [Bayesian network](https://en.wikipedia.org/wiki/Bayesian_network) is a directed acyclic graph (DAG) whose nodes are random variables and whose edges give the conditional dependencies. It factorizes the joint distribution into the distribution of each variable given its parents $$\text{pa}(X_i)$$:

$$
p(X_1, \ldots, X_n) = \prod_{i=1}^{n} p\left( X_i \mid \text{pa}(X_i) \right)
$$

For the network in the figure:

$$
p(A, B, C, D) = p(A) \, p(B \mid A) \, p(C \mid A) \, p(D \mid A, C)
$$

{: .picture}
<img width="10.5%" height="10.5%" src="{{ site.assets }}/statistics-for-ml/directed_graph.svg"/>

{: .caption}
A directed acyclic graph of the factorization $$p(A,B,C,D) = p(A) \cdot p(B \mid A) \cdot p(C \mid A) \cdot p(D \mid A,C)$$: each node depends only on its parents. Adapted from: [Graphical model](https://en.wikipedia.org/wiki/Graphical_model)

Each variable is conditionally independent of its non-descendants given its parents: in the figure, $$B \perp C \mid A$$ and $$D \perp B \mid A, C$$. The factorization also saves memory. For $$n$$ binary variables, the full joint table has $$2^n - 1$$ free parameters, while the network has $$\sum_i 2^{\vert \text{pa}(X_i) \vert} \le n \, 2^k$$ for at most $$k$$ parents per variable: $$1 + 2 + 2 + 4 = 9$$ instead of 15 in the figure. When each edge is a direct cause, the network is a [causal model](https://en.wikipedia.org/wiki/Causal_model), which also predicts the effect of interventions.

## Kernel Density Estimation

A [kernel density estimator](https://en.wikipedia.org/wiki/Kernel_density_estimation) (KDE) estimates a continuous pdf from $$N$$ samples $$x_1, \dots, x_N$$ without a parametric model. The samples define the [empirical CDF](https://en.wikipedia.org/wiki/Empirical_distribution_function), a step function that rises by $$1/N$$ at each sample:

$$
\hat{F}_N(x) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(x_i \le x)
$$

The KDE smooths these steps by placing a kernel on each sample:

$$
\hat{p}(x) = \frac{1}{N h} \sum_{i=1}^{N} K\left( \frac{x - x_i}{h} \right)
$$

where the kernel $$K$$ is a non-negative function that integrates to 1, such as the standard Gaussian pdf, and $$h > 0$$ is the **bandwidth**. The bandwidth sets the bias-variance trade-off: a large $$h$$ gives a flat, biased estimate, and a small $$h$$ a noisy one. For a Gaussian kernel and near-Gaussian data, a rule of thumb is $$h \approx 1.06 \, \hat{\sigma} N^{-1/5}$$, with $$\hat{\sigma}$$ the empirical standard deviation. The figure shows a histogram (left) and a KDE (right) of the same six samples, with the scaled kernels dashed.

{: .picture}
<img width="60.4%" height="60.4%" src="{{ site.assets }}/statistics-for-ml/kde.svg"/>

{: .caption}
Histogram with bins of width 2 (left) and Gaussian kernel density estimate with bandwidth $$h = 1.5$$ (right) of the same six data points. The estimate is the sum of six kernels centred at the points, each divided by $$n$$. Adapted from: [Kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation)

Evaluating $$\hat{p}$$ at $$m$$ points costs $$O(Nm)$$ kernel evaluations. For large $$N$$, binning the samples on a grid of $$G$$ points and convolving the bins with the kernel by FFT reduces the cost to $$O(N + G \log G)$$.

Smoothing also applies to regression. The table places four regression models by the distribution of the response $$y_i$$ and by its predictor: linear in the coefficients $$\beta$$ (parametric) or a smooth function of the inputs $$x_i$$ (non-parametric). For an exponential family response, a link function maps the mean of $$y_i$$ to the predictor, e.g. the logit in logistic regression.

| Distribution of $$y_i$$ | Parametric predictor $$x_i^\intercal \beta$$ | Non-parametric predictor |
|---|---|---|
| Gaussian | Linear regression | Smoothing, e.g. [kernel regression](https://en.wikipedia.org/wiki/Kernel_regression) |
| Exponential family | [Generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) | [Generalized additive models](https://en.wikipedia.org/wiki/Generalized_additive_model) |
