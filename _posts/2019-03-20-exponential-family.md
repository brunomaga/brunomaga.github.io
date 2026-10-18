---
layout: post
title:  "Exponential Family of Distributions"
categories: ["Machine Learning"]
tags: ["probability distributions", "exponential family", "MLE"]
---

The [Bayesian linear regression post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}) showed that the maximum likelihood and the maximum a posteriori estimates of a model with Gaussian noise become much easier to compute after taking the logarithm of the likelihood, the *log trick*. The logarithm is monotonically increasing, so it keeps the maximum in place, and it turns the product of Gaussian densities, a product of exponentials, into a sum, here a quadratic function of the weights with a closed-form maximum.

This property is not exclusive to the Gaussian distribution. Many common distributions, including the exponential, log-normal, gamma, chi-squared, beta, Dirichlet, Bernoulli, categorical, Poisson, geometric, inverse Gaussian, von Mises and von Mises–Fisher distributions, have a logarithm that is linear in a vector of statistics of the data, up to terms that depend only on the data or only on the parameters. They form the [exponential family](https://en.wikipedia.org/wiki/Exponential_family), whose members share four properties used throughout ML:
- the mean and the variance of the statistics are derivatives of one function;
- maximum likelihood matches the averages of the statistics;
- the estimates depend on the data only through sums of a fixed size;
- every member has a conjugate prior.

The [statistics post]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#exponential-family) summarizes these properties, and the sections below derive them, following chapter 8 of Jordan's notes on graphical models, <a href="{{ site.assets }}/exponential-family/chapter8.pdf">The exponential family: Basics</a>, and Jebara's [tutorial](http://www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf).

## Common Distributions

Many of the distributions used in ML are related to each other. The picture relates 15 common ones: the discrete distributions, described by a probability mass function, are drawn as bars, and the continuous ones, described by a probability density function, as filled curves.

{: .picture}
<img width="86.4%" height="86.4%" src="{{ site.assets }}/exponential-family/common_distributions.svg"/>

{: .caption}
The 15 most common probability distributions and their relationships. Adapted from: [Common probability distributions: the data scientist's crib sheet, S. Owen](https://medium.com/@srowen/common-probability-distributions-347e6b945ce4)

The original [post](https://medium.com/@srowen/common-probability-distributions-347e6b945ce4) by Sean Owen describes these relationships, summarized below:
- **Bernoulli and uniform**: the discrete uniform distribution gives the same probability to each outcome, as to the faces of a fair die, while the Bernoulli distribution gives the probabilities $$p$$ and $$1 - p$$ to two outcomes, as to the sides of a biased coin.
- **Binomial and hypergeometric**: the binomial distribution counts the successes in $$n$$ independent Bernoulli trials with the success probability $$p$$, e.g. the number of sixes in 30 rolls of a die ($$n = 30$$, $$p = 1/6$$). Its trials are draws with replacement, while the hypergeometric distribution counts the successes in $$n$$ draws *without* replacement from a finite population. The two agree when the population is much larger than $$n$$.
- **Poisson and binomial**: the Poisson distribution also counts, here the events in a fixed interval of time, for independent events that occur at a constant rate $$\lambda$$. It is the limit of the binomial distribution for many trials that each rarely succeed: $$n \to \infty$$ and $$p \to 0$$, with the mean $$np = \lambda$$ fixed.
- **Geometric and negative binomial**: these distributions fix when the trials stop, rather than how many there are. In independent trials with the success probability $$p$$, the negative binomial distribution counts the successes before the $$r$$-th failure, and the geometric distribution is its case $$r = 1$$ (other texts swap success and failure, which replaces $$p$$ with $$1 - p$$).
- **Exponential and Weibull**: the exponential distribution, the continuous counterpart of the geometric, is the time until an event or a failure that occurs at a constant rate $$\lambda$$, e.g. the time between two events of a Poisson process. The Weibull distribution, which includes the exponential, models a failure rate that increases or decreases over time.
- **Normal, log-normal, Student's t and chi-squared**: by the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), the sum of many independent variables with the same distribution and a finite variance is approximately normal, whatever this distribution is. The binomial and the hypergeometric distributions approach the normal distribution for a large $$n$$, and the Poisson distribution for a large $$\lambda$$. A variable is log-normal if its logarithm is normal. Student's t distribution has heavier tails than the normal distribution, and approaches it as its number of degrees of freedom grows. The chi-squared distribution is the distribution of a sum of squares of independent standard normal variables, as many as its degrees of freedom.
- **Gamma and beta**: the gamma distribution generalizes the exponential and the chi-squared distributions, and models the waiting time until a given number of events of a Poisson process. It is the conjugate prior of the Poisson rate and of the precision, the inverse variance, of a Gaussian. For independent gamma variables $$X$$ and $$Y$$ with the same scale, $$X / (X + Y)$$ follows a beta distribution, the conjugate prior of the success probability of the Bernoulli, binomial, negative binomial and geometric distributions.

Twelve of these 15 distributions are in the exponential family, the binomial, negative binomial and Weibull distributions only for a fixed $$n$$, $$r$$ or shape. The uniform distribution with unknown bounds, the hypergeometric distribution and Student's t distribution are not.

## Definition

A family of distributions with parameters $$\theta$$ is an exponential family if its pmf or pdf can be written as:

$$
p(x \mid \theta) = h(x) \exp\left( \eta(\theta)^T T(x) - A(\theta) \right)
$$

for known functions $$h$$, $$T$$, $$\eta$$ and $$A$$, where $$\eta(\theta)^T T(x) = \sum_{i=1}^{s} \eta_i(\theta) \, T_i(x)$$ sums over the $$s$$ components of $$T$$. Since $$A(\theta)$$ only normalizes the density, the parameters enter it only through $$\eta(\theta)$$, and the density can also be written with $$\eta$$ as its parameter, whatever the map from $$\theta$$ to $$\eta$$. This gives the **natural form**, or **canonical form**, of the family:

$$
p(x \mid \eta) = h(x) \exp\left( \eta^T T(x) - A(\eta) \right)
$$

whose terms have the following roles:
- $$\eta$$ holds the **natural parameters**, and the set of $$\eta$$ for which $$A(\eta)$$ is finite, the **natural parameter space**, is always convex.
- $$T(x)$$ holds the [sufficient statistics](https://en.wikipedia.org/wiki/Sufficient_statistic). The density has the form $$h(x) \, g_\eta(T(x))$$ of the Fisher–Neyman factorization theorem, with $$g_\eta(t) = \exp\left( \eta^T t - A(\eta) \right)$$, so $$T(x)$$ holds all the information that $$x$$ carries about $$\eta$$, and $$x$$ can be discarded for inference once $$T(x)$$ is known. As a consequence, two datasets $$x$$ and $$y$$ with $$T(x) = T(y)$$ have the same likelihood ratio, $$p(x \mid \eta_1) / p(x \mid \eta_2) = p(y \mid \eta_1) / p(y \mid \eta_2)$$, for any $$\eta_1$$ and $$\eta_2$$.
- $$h(x)$$ is the **base measure**.
- $$A(\eta)$$ is the **log-partition function**, the logarithm of the normalization factor that makes the density integrate to 1, or sum to 1 for a discrete $$x$$:

  $$
  A(\eta) = \log \int h(x) \exp\left( \eta^T T(x) \right) dx
  $$

The density must thus factorize: its logarithm is a sum of terms in $$x$$ only, terms in $$\theta$$ only, and products of a function of $$x$$ and a function of $$\theta$$. Since the exponential is never zero, the support of the density, the set of $$x$$ where it is positive, cannot depend on $$\theta$$, which excludes the uniform distribution with unknown bounds.

## Moments from the Log-Partition Function

The mean and the variance of a distribution are usually integrals, but in the exponential family they follow from the derivatives of $$A$$. The first derivative uses the derivative of the logarithm, which is the derivative of its argument divided by the argument (first line). The denominator is $$\exp(A(\eta))$$, by the definition of $$A$$, so dividing by it subtracts $$A(\eta)$$ in the exponent (second line). The integrand then is $$T(x)$$ times the density, and its integral is the expected value of $$T(X)$$ (third line):

$$
\begin{align*}
\nabla_\eta A(\eta) & = \frac{\int T(x) \, h(x) \exp\left( \eta^T T(x) \right) dx}{\int h(x) \exp\left( \eta^T T(x) \right) dx} \\
& = \int T(x) \, h(x) \exp\left( \eta^T T(x) - A(\eta) \right) dx \\
& = \mathbb{E}[T(X)]
\end{align*}
$$

Differentiating once more, the exponent brings down the factor $$T(x) - \nabla_\eta A(\eta) = T(x) - \mathbb{E}[T(X)]$$, and the integral is the covariance matrix of $$T(X)$$:

$$
\begin{align*}
\nabla_\eta^2 A(\eta) & = \int T(x) \left( T(x) - \mathbb{E}[T(X)] \right)^T h(x) \exp\left( \eta^T T(x) - A(\eta) \right) dx \\
& = \mathbb{E}\left[ T(X) \, T(X)^T \right] - \mathbb{E}[T(X)] \, \mathbb{E}[T(X)]^T = \mathrm{Var}[T(X)]
\end{align*}
$$

The higher derivatives give the higher [cumulants](https://en.wikipedia.org/wiki/Cumulant) of $$T(X)$$. Since a covariance matrix is positive semidefinite, $$A$$ is convex. When the representation is **minimal**, i.e. when there are no linear constraints among the components of $$\eta$$, nor among those of $$T(x)$$, $$A$$ is strictly convex, and the map from $$\eta$$ to the **mean parameters** $$\mathbb{E}[T(X)] = \nabla_\eta A(\eta)$$ is invertible. A distribution of the family can thus be described by its natural parameters or by its mean parameters.

## Examples

The univariate Gaussian with mean $$\mu$$ and standard deviation $$\sigma$$ is brought to this form by expanding the square in the exponent, and by writing the factor $$1 / \sigma$$ as $$\exp(-\log \sigma)$$:

$$
\begin{align*}
p(x \mid \mu, \sigma^2) & = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2 \sigma^2} \right) \\
& = \frac{1}{\sqrt{2 \pi}} \exp\left( \frac{\mu}{\sigma^2} x - \frac{1}{2 \sigma^2} x^2 - \frac{\mu^2}{2 \sigma^2} - \log \sigma \right)
\end{align*}
$$

The terms that multiply $$x$$ and $$x^2$$ form $$\eta^T T(x)$$, and the other terms form $$-A$$:
- $$h(x) = 1 / \sqrt{2 \pi}$$;
- $$T(x) = [x, x^2]^T$$;
- $$\eta = [\mu / \sigma^2, -1 / (2 \sigma^2)]^T$$, so that $$\mu = -\eta_1 / (2 \eta_2)$$ and $$\sigma^2 = -1 / (2 \eta_2)$$;
- $$A(\eta) = \frac{\mu^2}{2 \sigma^2} + \log \sigma = -\frac{\eta_1^2}{4 \eta_2} - \frac{1}{2} \log(-2 \eta_2)$$.

The derivatives of $$A$$ with respect to $$\eta_1$$ give the mean and the variance of $$x$$, the first sufficient statistic:

$$
\frac{\partial A}{\partial \eta_1} = -\frac{\eta_1}{2 \eta_2} = \mu, \qquad \frac{\partial^2 A}{\partial \eta_1^2} = -\frac{1}{2 \eta_2} = \sigma^2
$$

For a Bernoulli variable $$x \in \{0, 1\}$$, the success probability is also the mean, written $$\mu$$ rather than $$p$$, since $$p$$ already names the density. Writing the pmf as the exponential of its logarithm, and collecting the terms in $$x$$, gives:

$$
\begin{align*}
p(x \mid \mu) & = \mu^x (1 - \mu)^{1 - x} \\
& = \exp\left( x \log \mu + (1 - x) \log(1 - \mu) \right) \\
& = \exp\left( x \log \frac{\mu}{1 - \mu} + \log(1 - \mu) \right) \\
& = \exp\left( x \eta - \log(1 + e^\eta) \right)
\end{align*}
$$

so that $$h(x) = 1$$, $$T(x) = x$$, the natural parameter is the [logit](https://en.wikipedia.org/wiki/Logit), or log-odds, $$\eta = \log \frac{\mu}{1 - \mu}$$, and $$A(\eta) = -\log(1 - \mu) = \log(1 + e^\eta)$$. The derivatives of $$A$$ are again the mean and the variance:

$$
\frac{dA}{d\eta} = \frac{e^\eta}{1 + e^\eta} = \frac{1}{1 + e^{-\eta}} = \mu, \qquad \frac{d^2 A}{d\eta^2} = \frac{d\mu}{d\eta} = \mu (1 - \mu)
$$

The map from $$\eta$$ to $$\mu$$, the inverse of the logit, is the logistic sigmoid of logistic regression.

## Parameters of Common Distributions

The table lists the parameters of the most common members of the family, with the inverse mapping from $$\eta$$ back to the usual parameters, and the log-partition function as a function of $$\eta$$ and of the usual parameters $$\theta$$. The [Wikipedia article](https://en.wikipedia.org/wiki/Exponential_family) has a longer list.

<font size="2">
<table class="table table-striped table-hover" style="border-spacing: 3px;">
<thead class="thead-light">
<tr>
<th scope="col">Distribution</th>
<th scope="col">Probability Density/Mass Function</th>
<th scope="col">Natural parameter(s) <span class="math inline">\(\boldsymbol\eta\)</span></th>
<th scope="col">Inverse parameter mapping</th>
<th scope="col">Base measure <span class="math inline">\(h(x)\)</span></th>
<th scope="col">Sufficient statistic <span class="math inline">\(T(x)\)</span></th>
<th scope="col">Log-partition <span class="math inline">\(A(\boldsymbol\eta)\)</span></th>
<th scope="col">Log-partition <span class="math inline">\(A(\boldsymbol\theta)\)</span></th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="https://en.wikipedia.org/wiki/Bernoulli_distribution" title="wikilink">Bernoulli distribution</a></td>
<td><span class="math inline">\(f(x ; p) = p^x (1-p)^{1-x}\)</span></td>
<td><span class="math inline">\(\log\frac{p}{1-p}\)</span>
(<a href="https://en.wikipedia.org/wiki/logit_function" title="wikilink">logit function</a>)
</td>
<td><span class="math inline">\(\frac{1}{1+e^{-\eta}} = \frac{e^\eta}{1+e^{\eta}}\)</span>
(<a href="https://en.wikipedia.org/wiki/logistic_function" title="wikilink">logistic function</a>)
</td>
<td><span class="math inline">\(1\)</span></td>
<td><span class="math inline">\(x\)</span></td>
<td><span class="math inline">\(\log (1+e^{\eta})\)</span></td>
<td><span class="math inline">\(-\log (1-p)\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/binomial_distribution" title="wikilink">binomial distribution</a><br />
with known number of trials <em>n</em></td>
<td><span class="math inline">\(f(x; n,p) = \binom{n}{x}p^x(1-p)^{n-x} \)</span></td>
<td><span class="math inline">\(\log\frac{p}{1-p}\)</span></td>
<td><span class="math inline">\(\frac{1}{1+e^{-\eta}} = \frac{e^\eta}{1+e^{\eta}}\)</span></td>
<td><span class="math inline">\({n \choose x}\)</span></td>
<td><span class="math inline">\(x\)</span></td>
<td><span class="math inline">\(n \log (1+e^{\eta})\)</span></td>
<td><span class="math inline">\(-n \log (1-p)\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/Poisson_distribution" title="wikilink">Poisson distribution</a></td>
<td><span class="math inline">\( f(x; \lambda)=\frac{\lambda^x e^{-\lambda}}{x!}\)</span></td>
<td><span class="math inline">\(\log\lambda\)</span></td>
<td><span class="math inline">\(e^\eta\)</span></td>
<td><span class="math inline">\(\frac{1}{x!}\)</span></td>
<td><span class="math inline">\(x\)</span></td>
<td><span class="math inline">\(e^{\eta}\)</span></td>
<td><span class="math inline">\(\lambda\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/negative_binomial_distribution" title="wikilink">negative binomial distribution</a><br />
with known number of failures <em>r</em></td>
<td><span class="math inline">\( f(x; r, p) = \binom{x+r-1}{x} p^x(1-p)^r \)</span></td>
<td><span class="math inline">\(\log p\)</span></td>
<td><span class="math inline">\(e^\eta\)</span></td>
<td><span class="math inline">\({x+r-1 \choose x}\)</span></td>
<td><span class="math inline">\(x\)</span></td>
<td><span class="math inline">\(-r \log (1-e^{\eta})\)</span></td>
<td><span class="math inline">\(-r \log (1-p)\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/exponential_distribution" title="wikilink">exponential distribution</a></td>
<td><span class="math inline">\( f(x;\lambda) = \begin{cases} \lambda e^{-\lambda x} & x \ge 0, \\ 0 & x < 0. \end{cases} \)</span></td>
<td><span class="math inline">\(-\lambda\)</span></td>
<td><span class="math inline">\(-\eta\)</span></td>
<td><span class="math inline">\(1\)</span></td>
<td><span class="math inline">\(x\)</span></td>
<td><span class="math inline">\(-\log(-\eta)\)</span></td>
<td><span class="math inline">\(-\log\lambda\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/Pareto_distribution" title="wikilink">Pareto distribution</a><br />
with known minimum value <span class="math inline">\( x_m \) </span></td>
<td><span class="math inline">\( f(x;\alpha) = \begin{cases} \frac{\alpha x_\mathrm{m}^\alpha}{x^{\alpha+1}} & x\ge x_\mathrm{m}, \\ 0 & x < x_\mathrm{m}. \end{cases} \)</span></td>
<td><span class="math inline">\(-\alpha-1\)</span></td>
<td><span class="math inline">\(-1-\eta\)</span></td>
<td><span class="math inline">\(1\)</span></td>
<td><span class="math inline">\(\log x\)</span></td>
<td><span class="math inline">\(-\log (-1-\eta) + (1+\eta) \log x_{\mathrm m}\)</span></td>
<td><span class="math inline">\(-\log \alpha - \alpha \log x_{\mathrm m}\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/Laplace_distribution" title="wikilink">Laplace distribution</a><br />
with known mean <em>μ</em></td>
<td><span class="math inline">\(
f(x ; \mu,b) = \frac{1}{2b} \exp \left( -\frac{|x-\mu|}{b} \right) \\
 = \frac{1}{2b}
    \left\{\begin{matrix}
      \exp \left( -\frac{\mu-x}{b} \right) & \text{if }x < \mu
      \\[8pt]
      \exp \left( -\frac{x-\mu}{b} \right) & \text{if }x \geq \mu
    \end{matrix}\right.
\)</span></td>
<td><span class="math inline">\(-\frac{1}{b}\)</span></td>
<td><span class="math inline">\(-\frac{1}{\eta}\)</span></td>
<td><span class="math inline">\(1\)</span></td>
<td><span class="math inline">\(|x-\mu|\)</span></td>
<td><span class="math inline">\(\log\left(-\frac{2}{\eta}\right)\)</span></td>
<td><span class="math inline">\(\log 2b\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/normal_distribution" title="wikilink">normal distribution</a><br />
known variance</td>
<td><span class="math inline">\( f(x; \mu, \sigma^2) = \frac{1}{\sigma \sqrt{2\pi} } e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} \)</span></td>
<td><span class="math inline">\(\mu/\sigma\)</span></td>
<td><span class="math inline">\(\sigma\eta\)</span></td>
<td><span class="math inline">\(\frac{e^{-\frac{x^2}{2\sigma^2}}}{\sqrt{2\pi}\sigma}\)</span></td>
<td><span class="math inline">\(\frac{x}{\sigma}\)</span></td>
<td><span class="math inline">\(\frac{\eta^2}{2}\)</span></td>
<td><span class="math inline">\(\frac{\mu^2}{2\sigma^2}\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/normal_distribution" title="wikilink">normal distribution</a></td>
<td><span class="math inline">\( f(x; \mu, \sigma^2) = \frac{1}{\sigma \sqrt{2\pi} } e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} \)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \dfrac{\mu}{\sigma^2} \\[10pt] -\dfrac{1}{2\sigma^2} \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} -\dfrac{\eta_1}{2\eta_2} \\[15pt] -\dfrac{1}{2\eta_2} \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\frac{1}{\sqrt{2\pi}}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} x \\ x^2 \end{bmatrix}\)</span></td>
<td><span class="math inline">\(-\frac{\eta_1^2}{4\eta_2} - \frac12\log(-2\eta_2)\)</span></td>
<td><span class="math inline">\(\frac{\mu^2}{2\sigma^2} + \log \sigma\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/lognormal_distribution" title="wikilink">lognormal distribution</a></td>
<td><span class="math inline">\( \ln(X) \sim \mathcal N(\mu,\sigma^2) \)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \dfrac{\mu}{\sigma^2} \\[10pt] -\dfrac{1}{2\sigma^2} \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} -\dfrac{\eta_1}{2\eta_2} \\[15pt] -\dfrac{1}{2\eta_2} \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\frac{1}{\sqrt{2\pi}x}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \log x \\ (\log x)^2 \end{bmatrix}\)</span></td>
<td><span class="math inline">\(-\frac{\eta_1^2}{4\eta_2} - \frac12\log(-2\eta_2)\)</span></td>
<td><span class="math inline">\(\frac{\mu^2}{2\sigma^2} + \log \sigma\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/gamma_distribution" title="wikilink">gamma distribution</a><br>shape \(\alpha\), rate \(\beta\)</td>
<td><span class="math inline">\( f(x;\alpha,\beta) = \frac{ \beta^\alpha x^{\alpha-1} e^{-\beta x}}{\Gamma(\alpha)} \\ (\text{ for } x > 0 \quad \alpha, \beta > 0) \)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \alpha-1 \\ -\beta \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \eta_1+1 \\ -\eta_2 \end{bmatrix}\)</span></td>
<td><span class="math inline">\(1\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \log x \\ x \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\log \Gamma(\eta_1+1)-(\eta_1+1)\log(-\eta_2)\)</span></td>
<td><span class="math inline">\(\log \Gamma(\alpha)-\alpha\log\beta\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/gamma_distribution" title="wikilink">gamma distribution</a><br>shape \(k\), scale \(\theta\)</td>
<td><span class="math inline">\( f(x;k,\theta) =  \frac{x^{k-1}e^{-\frac{x}{\theta}}}{\theta^k\Gamma(k)} \\ (\text{ for } x > 0 \text{ and } k, \theta > 0) \)</span></td>
<td><span class="math inline">\(\begin{bmatrix} k-1 \\[5pt] -\dfrac{1}{\theta} \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \eta_1+1 \\[5pt] -\dfrac{1}{\eta_2} \end{bmatrix}\)</span></td>
<td><span class="math inline">\(1\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \log x \\ x \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\log \Gamma(\eta_1+1)-(\eta_1+1)\log(-\eta_2)\)</span></td>
<td><span class="math inline">\(\log \Gamma(k)+k\log\theta\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/beta_distribution" title="wikilink">beta distribution</a></td>
<td><span class="math inline">\( f(x;\alpha,\beta) = \frac{1}{B(\alpha,\beta)} x^{\alpha-1}(1-x)^{\beta-1}  \)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \alpha \\ \beta \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \eta_1 \\ \eta_2 \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\frac{1}{x(1-x)}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \log x \\ \log (1-x)  \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\log \Gamma(\eta_1) + \log \Gamma(\eta_2)\\- \log \Gamma(\eta_1+\eta_2)\)</span></td>
<td><span class="math inline">\(\log \Gamma(\alpha) + \log \Gamma(\beta)\\- \log \Gamma(\alpha+\beta)\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/multivariate_normal_distribution" title="wikilink">multivariate normal distribution</a></td>
<td><span class="math inline">\( f(x; \mu, \Sigma) = (2\pi)^{-\frac{k}{2}}\det(\Sigma)^{-\frac{1}{2}} \\ \,\, \exp \left( -\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu) \right)\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \boldsymbol\Sigma^{-1}\boldsymbol\mu \\[5pt] -\frac12\boldsymbol\Sigma^{-1} \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} -\frac12\boldsymbol\eta_2^{-1}\boldsymbol\eta_1 \\[5pt] -\frac12\boldsymbol\eta_2^{-1} \end{bmatrix}\)</span></td>
<td><span class="math inline">\((2\pi)^{-\frac{k}{2}}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \mathbf{x} \\[5pt] \mathbf{x}\mathbf{x}^\mathrm{T} \end{bmatrix}\)</span></td>
<td><span class="math inline">\(-\frac{1}{4}\boldsymbol\eta_1^{\rm T}\boldsymbol\eta_2^{-1}\boldsymbol\eta_1 - \frac12\log\left|-2\boldsymbol\eta_2\right|\)</span></td>
<td><span class="math inline">\(\frac12\boldsymbol\mu^{\rm T}\boldsymbol\Sigma^{-1}\boldsymbol\mu + \frac12 \log |\boldsymbol\Sigma|\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/multinomial_distribution" title="wikilink">multinomial distribution</a><br />
with known number of trials <em>n</em></td>
<td>
  <span class="math inline">\(
     f(x_1,\ldots,x_k;n,p_1,\ldots,p_k) = \\
   \frac{n!}{x_1!\cdots x_k!} p_1^{x_1} \cdots p_k^{x_k} \\
   \text{(where } \sum_{i=1}^k p_i = 1 \text{)} 
  \)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \log p_1 \\ \vdots \\ \log p_k \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} e^{\eta_1} \\ \vdots \\ e^{\eta_k} \end{bmatrix}\)</span><br />
<br />
where <span class="math inline">\(\textstyle\sum_{i=1}^k e^{\eta_i}=1\)</span></td>
<td><span class="math inline">\(\frac{n!}{\prod_{i=1}^{k} x_i!}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} x_1 \\ \vdots \\ x_k \end{bmatrix}\)</span></td>
<td><span class="math inline">\(0\)</span></td>
<td><span class="math inline">\(0\)</span></td>
</tr>
</tbody>
</table>
</font>

## Maximum Likelihood

Since the Gaussian belongs to the exponential family, its maximum likelihood estimate generalizes to every member, and the first step is again the log trick. For $$N$$ independent samples $$x_1, \dots, x_N$$, the logarithm turns the product of their densities into a sum:

$$
\begin{align*}
\ell(\eta) = \log p(x_1, \dots, x_N \mid \eta) & = \log \left( \prod_{i=1}^N h(x_i) \exp\left( \eta^T T(x_i) - A(\eta) \right) \right) \\
& = \sum_{i=1}^N \log h(x_i) + \eta^T \sum_{i=1}^N T(x_i) - N A(\eta)
\end{align*}
$$

The first term does not depend on $$\eta$$, the second is linear in $$\eta$$, and the third is concave, so the log-likelihood is concave, and any stationary point is the global maximum. Setting the gradient to zero gives:

$$
\nabla_\eta \ell(\eta) = \sum_{i=1}^N T(x_i) - N \nabla_\eta A(\eta) = 0 \quad \Leftrightarrow \quad \nabla_\eta A(\hat{\eta}) = \frac{1}{N} \sum_{i=1}^N T(x_i)
$$

Since $$\nabla_\eta A(\hat{\eta}) = \mathbb{E}_{\hat{\eta}}[T(X)]$$, the estimate sets the mean parameters to the average of the sufficient statistics over the data (**moment matching**), and $$\hat{\eta}$$ follows by inverting $$\nabla_\eta A$$. The data enter the estimate only through the sum $$\sum_i T(x_i)$$, which gives sufficiency its meaning: to estimate the parameters, only the sufficient statistics need to be kept.

For $$T(x) = x$$, as in the Bernoulli, Poisson and multinomial distributions, the estimate of the mean is the sample mean, and for the Gaussian, $$T(x) = [x, x^2]^T$$ gives the sample mean and the empirical variance with the factor $$1/N$$. When the average lies on the boundary of the possible means, e.g. when a Bernoulli variable is only observed as ones, $$\hat{\eta}$$ is infinite. The average of the $$T(x_i)$$ is an unbiased estimate of the mean parameters, and its variance $$\mathrm{Var}[T(X)] / N$$ attains the [Cramér–Rao bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound).

The size of the sum $$\sum_i T(x_i)$$ does not grow with $$N$$, and by the Pitman–Koopman–Darmois theorem, among the families whose support does not depend on the parameters, only exponential families have sufficient statistics of a bounded size. This suits large and streaming datasets: a stream keeps a running sum, and parallel workers add the sums of their shards with one all-reduce, of $$k + k(k + 1)/2$$ numbers and the sample count for a Gaussian in $$k$$ dimensions. In floating point, however, the sums of $$x$$ and $$x^2$$ lose precision when the variance is small relative to the squared mean, and Welford's update and Chan's merge, in the [statistics post]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#covariance-and-variance), compute the same statistics stably.

## Conjugate Priors and Products

Every exponential family likelihood has a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior), a prior whose posterior stays in its family, and this prior is itself in the exponential family, with the hyperparameters $$\chi$$, of the size of $$T$$, and $$\nu > 0$$:

$$
p(\eta \mid \chi, \nu) \propto \exp\left( \eta^T \chi - \nu A(\eta) \right)
$$

Multiplying it by the likelihood of $$N$$ samples adds the exponents, so the posterior has the same form, with:

$$
\chi' = \chi + \sum_{i=1}^N T(x_i), \qquad \nu' = \nu + N
$$

The prior thus acts as $$\nu$$ pseudo-observations whose sufficient statistics sum to $$\chi$$, and the Bayesian update only adds sums. For the Bernoulli likelihood, this prior is the beta distribution: a prior $$\text{Beta}(a, b)$$ on $$\mu$$ gives the posterior $$\text{Beta}(a + \sum_i x_i, \, b + N - \sum_i x_i)$$, with $$a$$ and $$b$$ as pseudo-counts of ones and zeros. When the normalization constant of the prior has a closed form, so do the evidence and the posterior predictive distribution.

In the same way, the product of two densities of the same family adds their natural parameters:

$$
h(x) \exp\left( \eta_1^T T(x) - A(\eta_1) \right) \cdot h(x) \exp\left( \eta_2^T T(x) - A(\eta_2) \right) = h(x)^2 \exp\left( (\eta_1 + \eta_2)^T T(x) - A(\eta_1) - A(\eta_2) \right)
$$

After normalization, the product is an exponential family density with the base measure $$h(x)^2$$ and the natural parameter $$\eta_1 + \eta_2$$. For Gaussians, whose base measure is constant, the precisions add, as in the posterior of Bayesian linear regression.

These properties make the exponential family central to ML. [Generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) give the output an exponential family distribution whose mean is a function of a linear combination of the inputs. With the canonical link, the natural parameter is this linear combination, and the link function, which maps the mean to the linear combination, is the inverse of $$\nabla_\eta A$$: the logit for logistic regression, as in the Bernoulli example, and the logarithm for Poisson regression. In models whose complete conditional distributions are in the exponential family, the mean-field updates of [variational inference]({{ site.baseurl }}{% post_url 2019-07-01-variational-inference-gmm %}) have a closed form. Lecture 11 of Jordan's <a href="{{ site.assets }}/exponential-family/lec-10-5.pdf">Statistical Learning Theory course</a> proves the moment, convexity, sufficiency and maximum likelihood results.
