---
layout: post
title:  "Exponential Family of Distributions"
categories: ["Machine Learning"]
tags: ["probability distributions", "exponential family", "MLE"]
---

In the [previous post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}), we saw that computing the Maximum Likelihood estimator and the Maximum-a-Posteriori on a normally-distributed set of parameters becomes much easier once we apply the log-trick. The rationale is that since $$\log$$ is a monotonically increasing function, the maximum and minimum of the function to be optimized are attained at the same points as those of the original function inside the $$\log$$ operator. Thus, by applying the $$\log$$ function, the normal distribution becomes simpler and faster to compute, as we convert a product of exponentials into a sum. 

However, this is not a property of the Gaussian distribution only. In fact, most common distributions, including the exponential, log-normal, gamma, chi-squared, beta, Dirichlet, Bernoulli, categorical, Poisson, geometric, inverse Gaussian, von Mises and von Mises-Fisher distributions, can be represented with a similar syntax, making them simple to compute as well. We call the set of such distributions the **Exponential Family of Distributions**, and we will discuss it next.

### Detour: relationship between common probability distributions

Probability distributions describe the probabilities of each outcome, with the common property that the probabilities of all outcomes add up to 1. They can also be classified into two subsets: the ones described by a probability **mass function**, if specified for discrete values, or by a probability **density function**, if described within some continuous interval. There are dozens (hundreds?) of different distributions, even though only 15 of them are often mentioned and used, and they have some kind of relationship among themselves: 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="{{ site.assets }}/exponential-family/common_distributions.png"/>

{: style="text-align:center; font-size: small;"}
The 15 most common probability distributions and their relationships.
(source: post <a href="https://medium.com/@srowen/common-probability-distributions-347e6b945ce4">Common probability distributions</a> by Sean Owen)


A brief summary of their relationships follows. For more details, check the original [post](https://medium.com/@srowen/common-probability-distributions-347e6b945ce4) by Sean Owen:
- *Bernoulli and Uniform*: the uniform distribution assigns equal probability to each discrete outcome, e.g. a coin toss or a dice roll; the Bernoulli assigns unequal probabilities $$p$$ and $$1-p$$ to two discrete outcomes, e.g. an unfair coin toss;
- *Binomial and Hypergeometric*: the binomial can be seen as the distribution of the sum of outcomes of something that follows a Bernoulli distribution, e.g. if we roll a dice 30 times, how many times do we get a six? This count follows the binomial distribution, with parameters $$n$$, the number of trials, and $$p$$, the probability of success (a la Bernoulli);
- *Poisson and Binomial*: like the binomial distribution, the Poisson distribution is a distribution of a count --- the number of times some event happened over a fixed interval of time, given a rate for the event to occur. It's parametrized as $$\lambda = np$$ (the $$n$$ and $$p$$ parameters of the binomial);  
- *Geometric and Negative Binomial*: while in the binomial we count the number of times an event *succeeds* after a number of trials, in the geometric distribution we count how many failed trials occur until our event succeeds; the negative binomial distribution is a simple generalization of the geometric, measuring the number of failures until $$r$$ successes have occurred, not just 1;
- *Exponential and Weibull*: the exponential distribution is the geometric on a continuous interval, parametrized by $$\lambda$$, like the Poisson. While it describes "time until event or failure" at a constant rate, the Weibull distribution models increases or decreases of the rate of failures over time (i.e. models *time-to-failure*);  
- *Normal, Log-Normal, Student’s t, and Chi-squared*: if we take a set of values following the same (any) distribution and sum them, that sum approximately follows the normal distribution --- this is true regardless of the underlying distribution, and this phenomenon is called the [**Central Limit Theorem**](https://en.wikipedia.org/wiki/Central_limit_theorem). The log-normal distribution relates to variables whose logarithm is normally distributed. The exponentiation of a normally-distributed variable is log-normally distributed. Student's t-distributions are normal-like distributions with a *fatter* tail, although they approach the normal distribution as the parameter increases. The chi-squared distribution is the distribution of the sum of squares of normally-distributed values;
- *Gamma and Beta*: the gamma distribution is a generalization of the exponential and the chi-squared distributions. Like the exponential distribution, it is used to model waiting times, e.g. the time until the next $$n$$ events occur. It appears in machine learning as the conjugate prior to some distributions. The beta distribution is the conjugate prior to most of the other distributions mentioned here.

## Exponential Family of distributions

The exponential family of distributions is the set of distributions parametrized by $$\theta \in \mathbf{R}^D$$ that can be described in the form:

$$
p(x \mid \theta) = h(x) \exp \left(\eta(\theta)^T T(x) -A(\theta)\right)
\label{eq_main_theta}
$$

or in a more extensive notation:

$$
p(x \mid \theta) = h(x) \exp\left(\sum_{i=1}^s \eta_i({\boldsymbol \theta}) T_i(x) - A({\boldsymbol \theta}) \right)
$$

where $$T(x)$$, $$h(x)$$, $$\eta(\theta)$$, and $$A(\theta)$$ are known functions. An alternative notation to equation \ref{eq_main_theta} describes $$A$$ as a function of $$\eta$$, regardless of the transformation from $$\theta$$ to $$\eta$$. We call this new expression an exponential family in its **natural form**, and it looks like:

$$
p(x \mid \eta) = h(x) \exp \left(\eta^T T(x) -A(\eta)\right)
\label{eq_main_eta}
$$

The term $$T(x)$$ is a **sufficient statistic** of the distribution. The sufficient statistic is a function of the data that holds all the information the data $$x$$ provides with regard to the unknown parameter values:
- The intuitive notion of sufficiency is that $$T(X)$$ is sufficient for $$\theta$$ if there is no information in $$X$$ regarding $$\theta$$ beyond that in $$T(X)$$. That is, having observed $$T(X)$$, we can throw away $$X$$ for the purposes of inference with respect to $$\theta$$;
- Moreover, this means that the likelihood *ratio* is the same for any two datasets $$x$$ and $$y$$ with the same sufficient statistic, i.e. if $$T(x)=T(y)$$, then $$\frac{p(x \mid \theta_1)}{p(x \mid \theta_2)} = \frac{p(y \mid \theta_1)}{p(y \mid \theta_2)}$$;

The term $$\eta$$ is the **natural parameter**, and the set of values of $$\eta$$ for which $$A(\eta)$$ is finite is called the **natural parameter space**, which is always convex. 

The term $$A(\eta)$$ is the **log-partition function**, because it is the logarithm of a normalization factor that ensures that the distribution $$p(x \mid \theta)$$ sums up or integrates to one (without which $$p(x \mid \theta)$$ would not be a probability distribution), i.e.:

$$
A(\eta) =  \log \int_x h(x) \exp (\eta^T T(x)) \, \mathrm{d}x;
$$

Another important point is that the mean and variance of $$T(x)$$ can be derived by differentiating $$A(\eta)$$ and computing its first and second derivatives, respectively:

$$
\begin{align*}
\frac{dA}{d\eta^T} & = \frac{d}{d\eta^T} \left(  \log \int_x h(x) \exp (\eta^T T(x)) \, \mathrm{d}x  \right) \\
                   & = \frac{ \int_x T(x) h(x) \exp (\eta^T T(x)) \, \mathrm{d}x}{\int_x h(x) \exp (\eta^T T(x)) \, \mathrm{d}x} \\
                     & \qquad \text{(log derivative rule: } \frac{d}{dx} log(x) = \frac{x'}{x} \text{)} \\
                   & = \int_x T(x) h(x) \exp(\eta^TT(x) - A(\eta))  \, \mathrm{d}x \\
                     & \qquad \text{(definition of } p(x \mid \eta) \text{, and } \frac{e^a}{e^b} = e^{a-b} \text{)} \\
                   & = \mathbf{E}[T(X)] \\
                     & \qquad \text{(def. expected value: } \mathbf{E}[X] = \int_x x \, f(x) dx \text{, for density func. } f(x) \text{)} \\
\end{align*}
$$ 

for the complete dataset $$X=(x_1, x_2, ..., x_m)$$, i.e. the first derivative of $$A(\eta)$$ is equal to the mean of the sufficient statistic. We can now look at the second derivative:

$$
\begin{align*}
\frac{d^2A}{d\eta\,d\eta^T} & = \int_x h(x) T(x) \left( T(x) -\frac{d}{d\eta^T}A(\eta)\right)^T \, \exp\left(\eta^TT(x) - A(\eta)\right)  \, dx \\
                            & = \int_x h(x) T(x) \left( T(x) - \mathbf{E}[T(X)] \right)^T \, \exp\left(\eta^TT(x) - A(\eta)\right)  \, dx  & (\frac{dA}{d\eta^T} =  \mathbf{E}[T(X)])\\
                            & = \mathbf{E}[T(X)T(X)^T] -\mathbf{E}[T(X)] \, \mathbf{E}[T(X)]^T  & (\mathbf{E}[X] = \int_x x \, f(x) dx)\\
                            & = \mathbf{Var}[T(X)] & (Var(X)=E[X^2]-E[X]^2)\\
\end{align*}
$$

and, as expected, the second derivative is equal to the variance of $$T(X)$$.

One requirement of exponential family distributions is that the parameters *must* factorize (i.e. must be separable into products, each of which involves only one type of variable), as either the power or the base of an exponentiation operation. I.e. the factors must be one of the following:

$$
f(x), g(\theta), c^{f(x)}, c^{g(\theta)}, {[f(x)]}^c, {[g(\theta)]}^c, {[f(x)]}^{g(\theta)}, {[g(\theta)]}^{f(x)}, {[f(x)]}^{h(x)g(\theta)}, \text{ or } {[g(\theta)]}^{h(x)j(\theta)}
$$

where $$f$$ and $$h$$ are arbitrary functions of $$x$$, $$g$$ and $$j$$ are arbitrary functions of $$\theta$$; and c is an arbitrary constant expression.

Another important point is that the product of two exponential-family distributions is also part of the exponential family, but unnormalized:

$$
\left[ h(x) \exp \left(\eta_1^T T(x) -A(\eta_1)\right) \right] 
* \left[ h(x) \exp \left(\eta_2^T T(x) -A(\eta_2)\right) \right]
= \widetilde{h}(x) \exp \left(\left(\eta_1+\eta_2\right)^T T(x) -\widetilde{A}(\eta_1, \eta_2)\right)
$$


Finally, *exponential family distributions have conjugate priors* (i.e. prior and posterior distributions have distributions from the exponential family), and the *posterior predictive distribution always has a closed-form solution* (provided that the normalizing factor can also be stated in closed form), both important properties for Bayesian statistics.

## Example: Univariate Gaussian distribution

The univariate Gaussian distribution is defined for an input $$x$$ as:

$$
p(x \mid \mu, \sigma^2 ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } \,\, \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)
$$

for a distribution with mean $$\mu$$ and standard deviation $$\sigma$$. By moving the terms around, we get:

$$
\begin{align*}
%% source: http://www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf
p(x \mid \mu, \sigma ) & = \frac{1}{ \sqrt{2 \pi \sigma^2} } \,\, \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right) \\
                       & = \frac{1}{ \sqrt{2 \pi } } \,\, \exp \left( - \log \sigma - \frac{x^2}{2 \sigma^2} + \frac{\mu x}{\sigma^2} - \frac{\mu^2}{2 \sigma^2} \right) \\
                       & = \frac{1}{ \sqrt{2 \pi } } \,\, \exp \left( \eta(\theta)^TT(x) - \log \sigma - \frac{\mu^2}{2\sigma^2} \right) \\
                       & = h(x) \,\, \exp \left( \eta(\theta)^TT(x) - A(\theta) \right) \\
\end{align*}
$$

where:
- $$h(x) = \frac{1}{ \sqrt{2 \pi}}$$,
- $$T = { x \choose x^2 }$$,
- $$\eta(\theta) = { \mu/(\sigma^2) \choose -1/(2\sigma^2) }$$, 
- $$A(\eta) = \frac{\mu^2}{2\sigma^2} + \log \sigma = - \frac{ \eta^2_1}{4 \eta_2} - \frac{1}{2} \log(-2\eta_2)$$.

We will now use the first and second derivatives of $$A(\eta)$$ to compute the mean and the variance of the sufficient statistic $$T(x)$$:

$$
\begin{align*}
%% source: https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf
\frac{dA}{d\eta_1} & = \frac{d}{d\eta_1} \left( - \frac{ \eta^2_1}{4 \eta_2} - \frac{1}{2} \log(-2\eta_2) \right) \\
                   & = -\frac{\eta_1}{2\eta_2} \\
                   & = \frac{\mu / \sigma^2}{1/\sigma^2} \\
		   & = \mu \\
\end{align*}
$$

which is the mean of $$x$$, the first component of the sufficient statistic. Taking the second derivative, we get:

$$
%% source: https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf
\frac{d^2A}{d\eta_1^2} = - \frac{1}{2\eta_2} = \sigma^2
$$

which is the variance of our normal distribution, by definition.

## Example: Bernoulli distribution

Similarly, to compute the exponential family parameters of the Bernoulli distribution, we proceed as follows:

$$
%% source: http://www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf
\begin{align*}
p(x \mid \alpha) & = \alpha^x (1-\alpha)^{1-x} \hspace{0.5cm}, x \in \{0,1\} \\
             & = \exp \left( \log (\alpha^x (1-\alpha)^{1-x}) \right) \\
             & = \exp \left( x \log \alpha + (1-x) \log (1-\alpha) \right) \\
             & = \exp \left( x \log \frac{\alpha}{1-\alpha} + \log (1-\alpha) \right) \\
             & = \exp \left( x \eta - \log (1+e^\eta) \right) \\
\end{align*}
$$

where:
- $$h(x) = 1$$,
- $$T(x) = x$$,
- $$\eta = \log \frac{\alpha}{1-\alpha}$$,
- $$A(\eta) = \log ( 1+e^\eta)$$.

We now compute the mean of $$T(x)$$ as:

$$
\begin{align*}
\frac{d A}{d \eta} = \frac{e^\eta}{1+e^\eta} = \frac{1}{1 + e^{-\eta}} = \mu \\
\end{align*}
$$

which is the mean of a Bernoulli variable. Taking a second derivative yields:

$$
%% source: https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf
\frac{d^2A}{d\eta^2} = \frac{d \mu}{d \eta} = \mu (1-\mu)
$$

which is the variance of a Bernoulli variable.

## Parameters for common distributions

The following table provides a summary of the most common distributions in the exponential family and their exponential-family parameters. For a more exhaustive list, check the [Wikipedia entry for Exponential Family](https://en.wikipedia.org/wiki/Exponential_family).

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
<td><span class="math inline">\( \overline{f}(x;\alpha) = \Pr(X>x) = \\ \begin{cases} \left(\frac{x_\mathrm{m}}{x}\right)^\alpha & x\ge x_\mathrm{m}, \\ 1 & x < x_\mathrm{m}, \end{cases} \)</span></td>
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
<td><a href="https://en.wikipedia.org/wiki/gamma_distribution" title="wikilink">gamma distribution</a><br>shape $$\alpha$$, rate $$\beta$$</td>
<td><span class="math inline">\( f(x;\alpha,\beta) = \frac{ \beta^\alpha x^{\alpha-1} e^{-\beta x}}{\Gamma(\alpha)} \\ (\text{ for } x > 0 \quad \alpha, \beta > 0) \)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \alpha-1 \\ -\beta \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \eta_1+1 \\ -\eta_2 \end{bmatrix}\)</span></td>
<td><span class="math inline">\(1\)</span></td>
<td><span class="math inline">\(\begin{bmatrix} \log x \\ x \end{bmatrix}\)</span></td>
<td><span class="math inline">\(\log \Gamma(\eta_1+1)-(\eta_1+1)\log(-\eta_2)\)</span></td>
<td><span class="math inline">\(\log \Gamma(\alpha)-\alpha\log\beta\)</span></td>
</tr>
<tr>
<td><a href="https://en.wikipedia.org/wiki/gamma_distribution" title="wikilink">gamma distribution</a><br>shape $$k$$, scale $$\theta$$</td>
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
<td><span class="math inline">\( f(x; \mu, \Sigma) = (2\pi)^{-\frac{k}{2}}\det(\Sigma)^{-\frac{1}{2}} \\ \,\, exp \left( -\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu) \right)\)</span></td>
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
   = \frac{n!}{x_1!\cdots x_k!} p_1^{x_1} \cdots p_k^{x_k} \\
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

In the [previous post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}), we computed the Maximum Likelihood Estimator (MLE) for a Gaussian distribution. In this post, we have seen that the Gaussian --- alongside plenty of other distributions --- belongs to the Exponential Family of Distributions. We will now show that the MLE can be generalized across all distributions in the Exponential Family.

As in the Gaussian case, to compute the MLE, we start by applying the log-trick to the general expression of the exponential family, and obtain the following log-likelihood:

$$
%% source: http://www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf
\begin{align*}
 lik(\eta \mid X) &= \log p(X \mid \eta) \\
                  & =  \log \left( \prod_{i=1}^N h(x_i) \exp \left(\eta^T T(x_i) -A(\eta) \right) \right) \\
                  & =  \log \left( \prod_{i=1}^N h(x_i) \right) + \eta^T \left( \sum_{i=1}^N T(x_i) \right) - NA(\eta) \\
%\log p(x \mid \eta) & = \sum_{i=1}^N \log p(x_i \mid \theta) \\
%                 & = \sum_{i=1}^M \log \left( h(x_i) \exp \left(\eta(\theta)^T T(x_i) -A(\theta) \right) \right) \\
%                 & = \sum_{i=1}^N \left( \log h(x_i) + \eta(\theta)^T T(x_i) - A(\theta) \right) \\ 
%                 & = \sum_{i=1}^N \left( \log h(x_i) + \eta(\theta)^T T(x_i) \right) - NA(\theta) \\ 
\end{align*}
$$ 

We then compute the derivative with respect to $$\eta$$ and set it to zero:

$$
%% source: http://www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf
\begin{align*}
& \frac{d}{d \eta} lik(\eta \mid X) = 0 \\
\Leftrightarrow & \sum_{i=1}^N T(x_i) - N \nabla_\eta A (\eta) =0 \\ 
\Leftrightarrow & \nabla_\eta A (\eta) = \frac{1}{N} \sum_{i=1}^N T(x_i)\\
\Leftrightarrow & \mu_{MLE} = \frac{1}{N} \sum_{i=1}^N T(x_i) \\
  & \qquad \text{(from above:  } \frac{dA(\eta)}{d \eta} = E[T(X)] \text{ and } E[T(X)] = \mu \text{)} \\
\end{align*}
$$

Not surprisingly, the result relates to the data only via the sufficient statistic $$\sum_{i=1}^N T(x_i)$$, giving a meaning to our notion of sufficiency --- *in order to estimate parameters, we retain only the sufficient statistic*. For distributions in which $$T(x) = x$$, which include the Bernoulli, Poisson and multinomial distributions, it shows that the sample mean is the maximum likelihood estimate of the mean. For the univariate Gaussian distribution, the sample mean is the maximum likelihood estimate of the mean and the sample variance is the maximum likelihood estimate of the variance.

