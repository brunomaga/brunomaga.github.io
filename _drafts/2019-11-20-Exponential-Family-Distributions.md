---
layout: post
title:  "Exponential Family of Distributions"
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---

In our [previous post]({{ site.baseurl }}{% post_url 2018-02-17-Supervised-Learning %}), we saw that computing the Maximum Likelihood estimator and the Maximum-a-Posterior becomes much easier once we apply the log-trick. The rationale was that by multiplying the solution by the $\log$, the normal distribution becomes easier to compute as we convert a multiplication with an exponential terms into a $log$ of sums. 

However, this is not an unique property of the Gaussian distribution. In this post we will show that several other distributions can be represented in a similar syntax, making it simple to compute as well. To the set of such distribution we call the **Exponential Family of Distributions**.

### Background: Probability Distributions 

A [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. A few examples of commonly used distributions are (for an input $x$):
- Gaussian distribution:
  - $$p(y \mid \mu, \sigma^2 ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } \text{ } exp \left( - \frac{(y - \mu)^2}{2 \sigma^2} \right)$$ for a distribution with mean $\mu$ and standard deviation $\sigma$, or
  - $$p(y \mid \mu, \Sigma ) = \left(\frac{1}{2 \pi}\right)^{-D/2} \text{ } det(\Sigma)^{-1/2} exp \left( - \frac{1}{2}(y - \mu)^T \Sigma^{-1} (y-\mu) \right)$$ with means-vector $\mu$ and covariance matrix $\Sigma$ on its multivariate notation
- Laplace:
 $$ p( y_n \mid x_n, w ) = \frac{1}{2b} e^{-\frac{1}{b} \mid y_n - X_n^T w \mid } $$
- Bernoulli: 
  - $p(x) = \alpha^x (1-\alpha)^{1-x}, x \in \{0,1\}$

- TODO add more distributions

### Exponential Family of distributions

The exponential family of distribution is the set of distributions that can be described as:

Exponential families include many of the most common distributions. Among many others, exponential families includes the following:

$$
p(x, \theta) = h(x) \text{ } exp ( \theta^T T(x) - A(\theta) )
$$

In fact, the most commonly distributions belong to the family of distributions, including:
- normal
exponential
gamma
chi-squared
beta
Dirichlet
Bernoulli
categorical
Poisson
Wishart
inverse Wishart
geometric

This property is very important, as it allows us to perform optimization by applying the log-likelihood method described next. The rationale is that since $log$ is an increasingly monotonic function, the maximum and minimum values of the loss functions are the same as in the original function. Moreover, it simplifies massively the computation as:

### Appendix: Exponential family parameters for common distributions

The following table provides a summary of most common distributions in the exponential family and their exponential-family parameters. For a more exhaustive list, check the [Wikipedia entry for Exponential Family](https://en.wikipedia.org/wiki/Exponential_family).

<font size="2.7">
<table class="table table-striped table-hover table-sm" style="border-spacing: 3px;">
<thead class="thead-light">
<tr>
<th scope="col"><p>Distribution</p></th>
<th scope="col"><p>Probability Density/<br/>Mass Function</p></th>
<th scope="col"><p>Parameter(s) <span class="math inline">\(\boldsymbol\theta\)</span></p></th>
<th scope="col"><p>Natural parameter(s) <span class="math inline">\(\boldsymbol\eta\)</span></p></th>
<th scope="col"><p>Inverse parameter mapping</p></th>
<th scope="col"><p>Base measure <span class="math inline">\(h(x)\)</span></p></th>
<th scope="col"><p>Sufficient statistic <span class="math inline">\(T(x)\)</span></p></th>
<th scope="col"><p>Log-partition <span class="math inline">\(A(\boldsymbol\eta)\)</span></p></th>
<th scope="col"><p>Log-partition <span class="math inline">\(A(\boldsymbol\theta)\)</span></p></th>
</tr>
</thead>
<tbody>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/Bernoulli_distribution" title="wikilink">Bernoulli distribution</a></p></td>
<td><p><span class="math inline">\(f(x, p) = p^x (1-p)^{1-x}\)</span></p></td>
<td><p>p</p></td>
<td><p><span class="math inline">\(\log\frac{p}{1-p}\)</span></p>
<small>(<a href="https://en.wikipedia.org/wiki/logit_function" title="wikilink">logit function</a>)</small>
</td>
<td><p><span class="math inline">\(\frac{1}{1+e^{-\eta}} = \frac{e^\eta}{1+e^{\eta}}\)</span></p>
<small>(<a href="https://en.wikipedia.org/wiki/logistic_function" title="wikilink">logistic function</a>)</small>
</td>
<td><p><span class="math inline">\(1\)</span></p></td>
<td><p><span class="math inline">\(x\)</span></p></td>
<td><p><span class="math inline">\(\log (1+e^{\eta})\)</span></p></td>
<td><p><span class="math inline">\(-\log (1-p)\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/binomial_distribution" title="wikilink">binomial distribution</a><br />
with known number of trials <em>n</em></p></td>
<td><p><span class="math inline">\(f(x,n,p) = \binom{n}{x}p^x(1-p)^{n-x} \)</span></p></td>
<td><p>p</p></td>
<td><p><span class="math inline">\(\log\frac{p}{1-p}\)</span></p></td>
<td><p><span class="math inline">\(\frac{1}{1+e^{-\eta}} = \frac{e^\eta}{1+e^{\eta}}\)</span></p></td>
<td><p><span class="math inline">\({n \choose x}\)</span></p></td>
<td><p><span class="math inline">\(x\)</span></p></td>
<td><p><span class="math inline">\(n \log (1+e^{\eta})\)</span></p></td>
<td><p><span class="math inline">\(-n \log (1-p)\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/Poisson_distribution" title="wikilink">Poisson distribution</a></p></td>
<td><p><span class="math inline">\( f(x; \lambda)=\frac{\lambda^x e^{-\lambda}}{x!}\)</span></p></td>
<td><p>λ</p></td>
<td><p><span class="math inline">\(\log\lambda\)</span></p></td>
<td><p><span class="math inline">\(e^\eta\)</span></p></td>
<td><p><span class="math inline">\(\frac{1}{x!}\)</span></p></td>
<td><p><span class="math inline">\(x\)</span></p></td>
<td><p><span class="math inline">\(e^{\eta}\)</span></p></td>
<td><p><span class="math inline">\(\lambda\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/negative_binomial_distribution" title="wikilink">negative binomial distribution</a><br />
with known number of failures <em>r</em></p></td>
<td><p><span class="math inline">\( f(x; r, p) = \binom{x+r-1}{x} p^r(1-p)^x \)</span></p></td>
<td><p>p</p></td>
<td><p><span class="math inline">\(\log p\)</span></p></td>
<td><p><span class="math inline">\(e^\eta\)</span></p></td>
<td><p><span class="math inline">\({x+r-1 \choose x}\)</span></p></td>
<td><p><span class="math inline">\(x\)</span></p></td>
<td><p><span class="math inline">\(-r \log (1-e^{\eta})\)</span></p></td>
<td><p><span class="math inline">\(-r \log (1-p)\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/exponential_distribution" title="wikilink">exponential distribution</a></p></td>
<td><p><span class="math inline">\( f(x;\lambda) = \begin{cases} \lambda e^{-\lambda x} & x \ge 0, \\ 0 & x < 0. \end{cases} \)</span></p></td>
<td><p>λ</p></td>
<td><p><span class="math inline">\(-\lambda\)</span></p></td>
<td><p><span class="math inline">\(-\eta\)</span></p></td>
<td><p><span class="math inline">\(1\)</span></p></td>
<td><p><span class="math inline">\(x\)</span></p></td>
<td><p><span class="math inline">\(-\log(-\eta)\)</span></p></td>
<td><p><span class="math inline">\(-\log\lambda\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/Pareto_distribution" title="wikilink">Pareto distribution</a><br />
with known minimum value <span class="math inline">\( x_m \) </span></p></td>
<td><p><span class="math inline">\( \overline{F}(x) = \Pr(X>x) = \\ \begin{cases} \left(\frac{x_\mathrm{m}}{x}\right)^\alpha & x\ge x_\mathrm{m}, \\ 1 & x < x_\mathrm{m}, \end{cases} \)</span></p></td>
<td><p>α</p></td>
<td><p><span class="math inline">\(-\alpha-1\)</span></p></td>
<td><p><span class="math inline">\(-1-\eta\)</span></p></td>
<td><p><span class="math inline">\(1\)</span></p></td>
<td><p><span class="math inline">\(\log x\)</span></p></td>
<td><p><span class="math inline">\(-\log (-1-\eta) + (1+\eta) \log x_{\mathrm m}\)</span></p></td>
<td><p><span class="math inline">\(-\log \alpha - \alpha \log x_{\mathrm m}\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/Laplace_distribution" title="wikilink">Laplace distribution</a><br />
with known mean <em>μ</em></p></td>
<td><p><span class="math inline">\(
f(x\mid\mu,b) = \frac{1}{2b} \exp \left( -\frac{|x-\mu|}{b} \right) \\
 = \frac{1}{2b}
    \left\{\begin{matrix}
      \exp \left( -\frac{\mu-x}{b} \right) & \text{if }x < \mu
      \\[8pt]
      \exp \left( -\frac{x-\mu}{b} \right) & \text{if }x \geq \mu
    \end{matrix}\right.
\)</span></p></td>
<td><p>b</p></td>
<td><p><span class="math inline">\(-\frac{1}{b}\)</span></p></td>
<td><p><span class="math inline">\(-\frac{1}{\eta}\)</span></p></td>
<td><p><span class="math inline">\(1\)</span></p></td>
<td><p><span class="math inline">\(|x-\mu|\)</span></p></td>
<td><p><span class="math inline">\(\log\left(-\frac{2}{\eta}\right)\)</span></p></td>
<td><p><span class="math inline">\(\log 2b\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/normal_distribution" title="wikilink">normal distribution</a><br />
known variance</p></td>
<td><p><span class="math inline">\( f(x) = \frac{1}{\sigma \sqrt{2\pi} } e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} \)</span></p></td>
<td><p>μ</p></td>
<td><p><span class="math inline">\(\frac{\mu}{\sigma}\)</span></p></td>
<td><p><span class="math inline">\(\sigma\eta\)</span></p></td>
<td><p><span class="math inline">\(\frac{e^{-\frac{x^2}{2\sigma^2}}}{\sqrt{2\pi}\sigma}\)</span></p></td>
<td><p><span class="math inline">\(\frac{x}{\sigma}\)</span></p></td>
<td><p><span class="math inline">\(\frac{\eta^2}{2}\)</span></p></td>
<td><p><span class="math inline">\(\frac{\mu^2}{2\sigma^2}\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/normal_distribution" title="wikilink">normal distribution</a></p></td>
<td><p><span class="math inline">\( f(x) = \frac{1}{\sigma \sqrt{2\pi} } e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} \)</span></p></td>
<td><p>μ,σ<sup>2</sup></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \dfrac{\mu}{\sigma^2} \\[10pt] -\dfrac{1}{2\sigma^2} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} -\dfrac{\eta_1}{2\eta_2} \\[15pt] -\dfrac{1}{2\eta_2} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\frac{1}{\sqrt{2\pi}}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} x \\ x^2 \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(-\frac{\eta_1^2}{4\eta_2} - \frac12\log(-2\eta_2)\)</span></p></td>
<td><p><span class="math inline">\(\frac{\mu^2}{2\sigma^2} + \log \sigma\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/lognormal_distribution" title="wikilink">lognormal distribution</a></p></td>
<td><p><span class="math inline">\( \ln(X) \sim \mathcal N(\mu,\sigma^2) \)</span></p></td>
<td><p>μ,σ<sup>2</sup></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \dfrac{\mu}{\sigma^2} \\[10pt] -\dfrac{1}{2\sigma^2} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} -\dfrac{\eta_1}{2\eta_2} \\[15pt] -\dfrac{1}{2\eta_2} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\frac{1}{\sqrt{2\pi}x}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \log x \\ (\log x)^2 \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(-\frac{\eta_1^2}{4\eta_2} - \frac12\log(-2\eta_2)\)</span></p></td>
<td><p><span class="math inline">\(\frac{\mu^2}{2\sigma^2} + \log \sigma\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/gamma_distribution" title="wikilink">gamma distribution</a><br>shape $\alpha$, rate $\beta$</p></td>
<td><p><span class="math inline">\( f(x;\alpha,\beta) = \frac{ \beta^\alpha x^{\alpha-1} e^{-\beta x}}{\Gamma(\alpha)} \\ (\text{ for } x > 0 \quad \alpha, \beta > 0) \)</span></p></td>
<td><p>α,β</p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \alpha-1 \\ -\beta \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \eta_1+1 \\ -\eta_2 \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(1\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \log x \\ x \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\log \Gamma(\eta_1+1)-(\eta_1+1)\log(-\eta_2)\)</span></p></td>
<td><p><span class="math inline">\(\log \Gamma(\alpha)-\alpha\log\beta\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/gamma_distribution" title="wikilink">gamma distribution</a><br>shape $k$, scale $\theta$</p></td>
<td><p><span class="math inline">\( f(x;k,\theta) =  \frac{x^{k-1}e^{-\frac{x}{\theta}}}{\theta^k\Gamma(k)} \\ (\text{ for } x > 0 \text{ and } k, \theta > 0) \)</span></p></td>
<td><p><em>k</em>, <em>θ</em></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} k-1 \\[5pt] -\dfrac{1}{\theta} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \eta_1+1 \\[5pt] -\dfrac{1}{\eta_2} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(1\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \log x \\ x \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\log \Gamma(\eta_1+1)-(\eta_1+1)\log(-\eta_2)\)</span></p></td>
<td><p><span class="math inline">\(\log \Gamma(k)-k\log\theta\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/beta_distribution" title="wikilink">beta distribution</a></p></td>
<td><p><span class="math inline">\( f(x;\alpha,\beta) = \frac{1}{B(\alpha,\beta)} x^{\alpha-1}(1-x)^{\beta-1}  \)</span></p></td>
<td><p>α,β</p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \alpha \\ \beta \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \eta_1 \\ \eta_2 \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\frac{1}{x(1-x)}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \log x \\ \log (1-x)  \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\log \Gamma(\eta_1) + \log \Gamma(\eta_2)\\- \log \Gamma(\eta_1+\eta_2)\)</span></p></td>
<td><p><span class="math inline">\(\log \Gamma(\alpha) + \log \Gamma(\beta)\\- \log \Gamma(\alpha+\beta)\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/multivariate_normal_distribution" title="wikilink">multivariate normal distribution</a></p></td>
<td><p><span class="math inline">\( (2\pi)^{-\frac{k}{2}}\det(\Sigma)^{-\frac{1}{2}} \\ \text{ } exp \left( -\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu) \right)\)</span></p></td>
<td><p><strong>μ</strong>,<strong>Σ</strong></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \boldsymbol\Sigma^{-1}\boldsymbol\mu \\[5pt] -\frac12\boldsymbol\Sigma^{-1} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} -\frac12\boldsymbol\eta_2^{-1}\boldsymbol\eta_1 \\[5pt] -\frac12\boldsymbol\eta_2^{-1} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\((2\pi)^{-\frac{k}{2}}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \mathbf{x} \\[5pt] \mathbf{x}\mathbf{x}^\mathrm{T} \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(-\frac{1}{4}\boldsymbol\eta_1^{\rm T}\boldsymbol\eta_2^{-1}\boldsymbol\eta_1 - \frac12\log\left|-2\boldsymbol\eta_2\right|\)</span></p></td>
<td><p><span class="math inline">\(\frac12\boldsymbol\mu^{\rm T}\boldsymbol\Sigma^{-1}\boldsymbol\mu + \frac12 \log |\boldsymbol\Sigma|\)</span></p></td>
</tr>
<tr>
<td><p><a href="https://en.wikipedia.org/wiki/multinomial_distribution" title="wikilink">multinomial distribution</a><br />
with known number of trials <em>n</em></p></td>
<td><p><span class="math inline">\( f(x_1,\ldots,x_k;n,p_1,\ldots,p_k) = \\= \frac{n!}{x_1!\cdots x_k!} p_1^{x_1} \cdots p_k^{x_k} \)</span></p></td>
<td><p>p<sub>1</sub>,...,p<sub>k</sub><br />
<br />
where <span class="math inline">\(\textstyle\sum_{i=1}^k p_i=1\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} \log p_1 \\ \vdots \\ \log p_k \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} e^{\eta_1} \\ \vdots \\ e^{\eta_k} \end{bmatrix}\)</span><br />
<br />
where <span class="math inline">\(\textstyle\sum_{i=1}^k e^{\eta_i}=1\)</span></p></td>
<td><p><span class="math inline">\(\frac{n!}{\prod_{i=1}^{k} x_i!}\)</span></p></td>
<td><p><span class="math inline">\(\begin{bmatrix} x_1 \\ \vdots \\ x_k \end{bmatrix}\)</span></p></td>
<td><p><span class="math inline">\(0\)</span></p></td>
<td><p><span class="math inline">\(0\)</span></p></td>
</tr>
</tbody>
</table>
</font>
