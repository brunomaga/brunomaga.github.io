---
layout: post
title:  "Exponential Family of Distributions"
categories: ["Machine Learning"]
tags: ["probability distributions", "exponential family", "MLE"]
---

The Gaussian likelihood makes the maximum likelihood and the Bayesian estimates of the [Bayesian linear regression post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}) closed-form: its logarithm is a quadratic function of the weights. Many of these results extend to the [exponential family](https://en.wikipedia.org/wiki/Exponential_family), a large set of distributions whose log-density is linear in a vector of statistics of the data, up to terms that depend only on the data or only on the parameters. Its members share four properties used throughout ML:
- the mean and the variance of the statistics are derivatives of one function;
- maximum likelihood matches the averages of the statistics;
- the estimates depend on the data only through sums of a fixed size;
- every member has a conjugate prior.

The [statistics post]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#exponential-family) summarizes these properties; the sections below derive them, following chapter 8 of Jordan's notes on graphical models, <a href="{{ site.assets }}/exponential-family/chapter8.pdf">The exponential family: Basics</a>, and Jebara's [tutorial](http://www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf).

## Common Distributions

The figure relates 15 common distributions. Bars mark the discrete ones, defined by a probability mass function, and filled curves the continuous ones, defined by a probability density function.

{: .picture}
<img width="70%" height="70%" src="{{ site.assets }}/exponential-family/common_distributions.svg"/>

{: .caption}
The 15 most common probability distributions and their relationships.
(source: post <a href="https://medium.com/@srowen/common-probability-distributions-347e6b945ce4">Common probability distributions</a> by Sean Owen)

- **Bernoulli and uniform**: the Bernoulli distribution gives the probabilities $$p$$ and $$1 - p$$ to two outcomes, e.g. the toss of a biased coin. The discrete uniform distribution gives the same probability to each of $$n$$ outcomes, e.g. the roll of a fair die.
- **Binomial and hypergeometric**: the binomial distribution counts the successes in $$n$$ independent Bernoulli trials with probability $$p$$, e.g. the number of sixes in 30 rolls of a die ($$n = 30$$, $$p = 1/6$$). The trials are draws with replacement. The hypergeometric distribution counts the successes in $$n$$ draws *without* replacement from a finite population. The two agree when the population is much larger than $$n$$.
- **Poisson**: the number of events in a fixed interval, for independent events at a constant rate $$\lambda$$. It is the limit of the binomial for $$n \to \infty$$ and $$p \to 0$$ with $$np = \lambda$$ fixed.
- **Geometric and negative binomial**: in independent trials with success probability $$p$$, the negative binomial distribution counts the successes before the $$r$$-th failure, and the geometric distribution is its case $$r = 1$$.
- **Exponential and Weibull**: the exponential distribution is the continuous counterpart of the geometric: the waiting time between the events of a Poisson process with rate $$\lambda$$. The Weibull distribution, with the exponential as a special case, models a failure rate that increases or decreases over time.
- **Normal, log-normal, Student's t and chi-squared**: by the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), the sum of many independent, identically distributed variables with a finite variance is approximately normal. The binomial and the hypergeometric approach it for large $$n$$, and the Poisson for large $$\lambda$$. A variable is log-normal if its logarithm is normal. Student's t distribution has heavier tails than the normal and approaches it as its number of degrees of freedom grows. The chi-squared distribution with $$k$$ degrees of freedom is the distribution of the sum of the squares of $$k$$ independent standard normal variables.
- **Gamma and beta**: the gamma distribution generalizes the exponential and the chi-squared distributions, and models the waiting time until the $$k$$-th event of a Poisson process. It is the conjugate prior of the Poisson rate and of the precision of a Gaussian. If $$X$$ and $$Y$$ are independent gamma variables with the same scale, $$X / (X + Y)$$ follows a beta distribution. The beta distribution is the conjugate prior of the success probability of the Bernoulli, binomial, negative binomial and geometric distributions.

Twelve of these distributions are in the exponential family, the binomial, negative binomial and Weibull only for a fixed $$n$$, $$r$$ or shape. The uniform distribution with unknown bounds, the hypergeometric and Student's t are not. Other members are the categorical, multinomial, Dirichlet, inverse Gaussian, von Mises and von Mises–Fisher distributions.

## Definition

A family of distributions with parameters $$\theta$$ is an exponential family if its pmf or pdf can be written as

$$
p(x \mid \theta) = h(x) \exp\left( \eta(\theta)^T T(x) - A(\theta) \right)
$$

for known functions $$h$$, $$T$$, $$\eta$$ and $$A$$. Written in terms of $$\eta$$, this is the **canonical form**:

$$
p(x \mid \eta) = h(x) \exp\left( \eta^T T(x) - A(\eta) \right)
$$

- $$\eta$$ holds the **natural parameters**. The **natural parameter space**, the set of $$\eta$$ for which $$A(\eta)$$ is finite, is convex.
- $$T(x)$$ holds the [sufficient statistics](https://en.wikipedia.org/wiki/Sufficient_statistic). The parameters interact with $$x$$ only through $$T(x)$$, so $$T(x)$$ holds all the information that $$x$$ carries about $$\eta$$ (Fisher–Neyman factorization theorem): $$x$$ can be discarded once $$T(x)$$ is known. Two datasets $$x$$ and $$y$$ with $$T(x) = T(y)$$ have the same likelihood ratio $$p(x \mid \eta_1) / p(x \mid \eta_2)$$ for any $$\eta_1$$ and $$\eta_2$$.
- $$h(x)$$ is the **base measure**.
- $$A(\eta)$$ is the **log-partition function**, the logarithm of the normalization constant:

  $$
  A(\eta) = \log \int h(x) \exp\left( \eta^T T(x) \right) dx,
  $$

  with a sum instead of the integral for a discrete $$x$$.

The density must thus factor into a function of $$x$$, a function of $$\theta$$, and the exponential of $$\eta(\theta)^T T(x)$$. In particular, the support of $$p$$ cannot depend on $$\theta$$, which excludes the uniform distribution with unknown bounds.

## Moments from the Log-Partition Function

The derivatives of $$A$$ give the moments of $$T(X)$$, without an integral. The first derivative follows from the derivative of the logarithm:

$$
\begin{align*}
\nabla_\eta A(\eta) & = \frac{\int T(x) \, h(x) \exp\left( \eta^T T(x) \right) dx}{\int h(x) \exp\left( \eta^T T(x) \right) dx} \\
& = \int T(x) \, h(x) \exp\left( \eta^T T(x) - A(\eta) \right) dx \\
& = \mathbb{E}[T(X)]
\end{align*}
$$

Differentiating again, with $$\nabla_\eta A(\eta) = \mathbb{E}[T(X)]$$:

$$
\begin{align*}
\nabla_\eta^2 A(\eta) & = \int T(x) \left( T(x) - \mathbb{E}[T(X)] \right)^T h(x) \exp\left( \eta^T T(x) - A(\eta) \right) dx \\
& = \mathbb{E}\left[ T(X) \, T(X)^T \right] - \mathbb{E}[T(X)] \, \mathbb{E}[T(X)]^T = \mathrm{Var}[T(X)]
\end{align*}
$$

The higher derivatives give the higher [cumulants](https://en.wikipedia.org/wiki/Cumulant) of $$T(X)$$. A covariance matrix is positive semidefinite, so $$A$$ is convex. A representation is **minimal** when no linear relation holds among the components of $$\eta$$ or of $$T(x)$$. $$A$$ is then strictly convex, and the map from $$\eta$$ to the **mean parameters** $$\mu = \mathbb{E}[T(X)] = \nabla_\eta A(\eta)$$ is invertible. A distribution can thus be parametrized by $$\eta$$ or by $$\mu$$.

## Examples

The univariate Gaussian with mean $$\mu$$ and variance $$\sigma^2$$ expands to

$$
p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2 \sigma^2} \right) = \frac{1}{\sqrt{2 \pi}} \exp\left( \frac{\mu}{\sigma^2} x - \frac{1}{2 \sigma^2} x^2 - \frac{\mu^2}{2 \sigma^2} - \log \sigma \right)
$$

so that:
- $$h(x) = 1 / \sqrt{2 \pi}$$;
- $$T(x) = [x, x^2]^T$$;
- $$\eta = [\mu / \sigma^2, -1 / (2 \sigma^2)]^T$$;
- $$A(\eta) = \frac{\mu^2}{2 \sigma^2} + \log \sigma = -\frac{\eta_1^2}{4 \eta_2} - \frac{1}{2} \log(-2 \eta_2)$$.

The derivatives of $$A$$ with respect to $$\eta_1$$ give the mean and the variance of $$x$$, the first sufficient statistic:

$$
\frac{\partial A}{\partial \eta_1} = -\frac{\eta_1}{2 \eta_2} = \mu, \qquad \frac{\partial^2 A}{\partial \eta_1^2} = -\frac{1}{2 \eta_2} = \sigma^2
$$

A Bernoulli variable $$x \in \{0, 1\}$$ with mean $$\mu$$ expands to

$$
p(x \mid \mu) = \mu^x (1 - \mu)^{1 - x} = \exp\left( x \log \frac{\mu}{1 - \mu} + \log(1 - \mu) \right) = \exp\left( x \eta - \log(1 + e^\eta) \right)
$$

so that $$h(x) = 1$$, $$T(x) = x$$, the natural parameter is the [logit](https://en.wikipedia.org/wiki/Logit) $$\eta = \log \frac{\mu}{1 - \mu}$$, and $$A(\eta) = \log(1 + e^\eta)$$. The derivatives give the mean and the variance of the Bernoulli variable:

$$
\frac{dA}{d\eta} = \frac{1}{1 + e^{-\eta}} = \mu, \qquad \frac{d^2 A}{d\eta^2} = \frac{d\mu}{d\eta} = \mu (1 - \mu)
$$

The map from $$\eta$$ to $$\mu$$ is the logistic sigmoid of logistic regression.

## Parameters of Common Distributions

The table lists the parameters of common members. The [Wikipedia article](https://en.wikipedia.org/wiki/Exponential_family) has a longer list.

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

For $$N$$ independent samples $$x_1, \dots, x_N$$, the log-likelihood is

$$
\ell(\eta) = \log p(x_1, \dots, x_N \mid \eta) = \sum_{i=1}^N \log h(x_i) + \eta^T \sum_{i=1}^N T(x_i) - N A(\eta)
$$

It is concave in $$\eta$$, since $$A$$ is convex, so any stationary point is the global maximum. The gradient is zero at:

$$
\nabla_\eta \ell(\eta) = \sum_{i=1}^N T(x_i) - N \nabla_\eta A(\eta) = 0 \quad \Leftrightarrow \quad \hat{\mu}_{\text{MLE}} = \nabla_\eta A(\hat{\eta}) = \frac{1}{N} \sum_{i=1}^N T(x_i)
$$

The maximum likelihood estimate of the mean parameters is the average of the sufficient statistics (**moment matching**), and $$\hat{\eta}$$ follows by inverting $$\nabla_\eta A$$. When the average lies on the boundary of the possible means, e.g. only ones for a Bernoulli variable, $$\hat{\eta}$$ is infinite. For $$T(x) = x$$, as in the Bernoulli, Poisson and multinomial distributions, $$\hat{\mu}_{\text{MLE}}$$ is the sample mean. For the Gaussian, $$T(x) = [x, x^2]^T$$ gives the sample mean and the empirical variance with the factor $$1/N$$. The estimate $$\hat{\mu}_{\text{MLE}}$$ is unbiased, and its variance $$\mathrm{Var}[T(X)] / N$$ attains the [Cramér–Rao bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound).

The data enter the estimate only through the sum $$\sum_i T(x_i)$$, whose size does not grow with $$N$$. Among families whose support does not depend on the parameters, only exponential families have sufficient statistics of a bounded size (Pitman–Koopman–Darmois theorem). This suits large and streaming data: a stream keeps a running sum, and parallel workers compute the sums of their shards and add them with one all-reduce of $$\dim T$$ numbers, e.g. $$D + D(D + 1)/2$$ for a Gaussian in $$D$$ dimensions. For the Gaussian, the sums of $$x$$ and $$x^2$$ lose precision in floating point when the variance is small relative to the squared mean; Welford's update and Chan's merge, in the [statistics post]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#covariance-and-variance), compute the same statistics stably.

## Conjugate Priors and Products

Every exponential family likelihood has a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior), also in the exponential family, with hyperparameters $$\chi$$ (of the size of $$T$$) and $$\nu > 0$$:

$$
p(\eta \mid \chi, \nu) \propto \exp\left( \eta^T \chi - \nu A(\eta) \right)
$$

Multiplying it by the likelihood of $$N$$ samples gives a posterior of the same form, with:

$$
\chi' = \chi + \sum_{i=1}^N T(x_i), \qquad \nu' = \nu + N
$$

The prior acts as $$\nu$$ pseudo-observations whose sufficient statistics sum to $$\chi$$, and the Bayesian update only adds sums. For the Bernoulli likelihood, this prior is the beta distribution: a prior $$\text{Beta}(a, b)$$ on $$\mu$$ gives the posterior $$\text{Beta}(a + \sum_i x_i, \, b + N - \sum_i x_i)$$, with $$a$$ and $$b$$ as pseudo-counts of ones and zeros. When the normalization constant of the prior has a closed form, so do the evidence and the posterior predictive distribution.

The product of two densities of the same family adds their natural parameters:

$$
h(x) \exp\left( \eta_1^T T(x) - A(\eta_1) \right) \cdot h(x) \exp\left( \eta_2^T T(x) - A(\eta_2) \right) = h(x)^2 \exp\left( (\eta_1 + \eta_2)^T T(x) - A(\eta_1) - A(\eta_2) \right)
$$

After normalization, it is an exponential family density with base measure $$h(x)^2$$ and natural parameter $$\eta_1 + \eta_2$$. For Gaussians, whose base measure is constant, the precisions add, as in the Bayesian linear regression posterior.

These properties make the exponential family central to ML models. [Generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) give the output an exponential family distribution whose mean is a function of a linear combination of the inputs. With the canonical link, the natural parameter is that linear combination, and the link function, from the mean to the linear combination, is the inverse of $$\nabla_\eta A$$: the logit for logistic regression and the logarithm for Poisson regression. In models whose complete conditional distributions are in the exponential family, the mean-field updates of [variational inference]({{ site.baseurl }}{% post_url 2019-07-01-variational-inference-gmm %}) have a closed form. Lecture 11 of Jordan's <a href="{{ site.assets }}/exponential-family/lec-10-5.pdf">Statistical Learning Theory course</a> proves the moment, convexity, sufficiency and maximum likelihood results.
