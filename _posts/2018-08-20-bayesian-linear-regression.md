---
layout: post
title:  "Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori"
categories: ["Machine Learning"]
tags: ["Bayesian inference", "linear regression", "MLE", "MAP"]
---

Linear regression models an output $$y \in \mathbb{R}$$ as a linear function of an input $$x \in \mathbb{R}^D$$ plus Gaussian noise:

$$
y = x^T w + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

with weights $$w \in \mathbb{R}^D$$ and noise variance $$\sigma^2$$. A constant input equal to 1 adds a bias weight. The model stays linear in $$w$$ when $$x$$ is replaced by a vector of features $$\phi(x)$$, such as powers of the input or Gaussian basis functions, so all results below hold with $$\phi(x)$$ in place of $$x$$. For $$N$$ independent samples, the rows of $$X \in \mathbb{R}^{N \times D}$$ are the inputs $$x_n^T$$, and $$y \in \mathbb{R}^N$$ holds the outputs $$y_n$$.

A frequentist estimator returns one value of $$w$$. Bayesian inference instead treats $$w$$ as a random variable: a **prior** distribution holds the knowledge about $$w$$ before the data, and the data turn it into a **posterior** distribution. The posterior gives the uncertainty of each prediction, and it can be updated with new data only. Three estimates follow, from the simplest to the most complete: the maximum likelihood estimate (MLE), the maximum a posteriori (MAP) estimate, and the full posterior.

## Bayes' Rule

[Bayes' rule]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#bayes-rule) gives the posterior of the weights from the likelihood of the data and the prior:

$$
p(w \mid X, y) = \frac{p(y \mid X, w) \, p(w)}{p(y \mid X)} \propto p(y \mid X, w) \, p(w)
$$

The inputs $$X$$ are given, not modelled, so the distributions of $$y$$ are conditioned on them. The evidence $$p(y \mid X)$$ does not depend on $$w$$, so a maximization over $$w$$ ignores it.

As $$N$$ grows, the data outweigh the prior. The [Bernstein–von Mises theorem](https://en.wikipedia.org/wiki/Bernstein%E2%80%93von_Mises_theorem) states that the posterior converges to a Gaussian centered at the maximum likelihood estimate, with a covariance that shrinks as $$1/N$$. It holds when the model is correct and regular, and the prior density is continuous and positive at the true parameters.

## Maximum Likelihood

The **likelihood** $$p(y \mid X, w)$$ is the probability density of the observed outputs for given weights. For fixed data, it is a function of $$w$$, not a probability distribution over $$w$$. The [maximum likelihood estimate](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) $$w_{\text{MLE}} = \arg\max_w p(y \mid X, w)$$ is the $$w$$ under which the observed data are most probable. With independent noise, the likelihood is a product of Gaussians:

$$
p(y \mid X, w) = \prod_{n=1}^{N} \mathcal{N}(y_n \mid x_n^T w, \sigma^2)
$$

The logarithm turns the product into a sum. It is monotonically increasing, so it does not move the maximum. It also avoids underflow: the product of 1000 densities of 0.01 is $$10^{-2000}$$, far below the smallest positive double, about $$5 \times 10^{-324}$$. The negative log-likelihood is the squared error, up to a scale and a constant:

$$
-\log p(y \mid X, w) = \frac{1}{2 \sigma^2} \sum_{n=1}^{N} (y_n - x_n^T w)^2 + \frac{N}{2} \log(2 \pi \sigma^2) = \frac{1}{2 \sigma^2} \Vert y - Xw \Vert^2 + \frac{N}{2} \log(2 \pi \sigma^2)
$$

Maximum likelihood under Gaussian noise is thus least squares. With $$\nabla_w (a^T w) = a$$ and $$\nabla_w (w^T A w) = (A + A^T) w$$ (equations 69 and 81 of [The Matrix Cookbook]({{ site.assets }}/resources/the_matrix_cookbook.pdf)), the gradient is zero at:

$$
\frac{1}{\sigma^2} \left( X^T X w - X^T y \right) = 0 \quad \Leftrightarrow \quad w_{\text{MLE}} = (X^T X)^{-1} X^T y
$$

The solution is unique when $$X$$ has full column rank, which requires $$N \ge D$$, and it does not depend on $$\sigma^2$$. The [linear regression post]({{ site.baseurl }}{% post_url 2017-08-17-linear-regression-matrix-factorization %}) details the solvers: forming $$X^T X$$ costs $$O(ND^2)$$ operations, and its Cholesky factorization $$O(D^3)$$.

The noise variance is also a parameter of the likelihood. The derivative of the log-likelihood with respect to $$\sigma^2$$ is zero at:

$$
-\frac{N}{2 \sigma^2} + \frac{1}{2 \sigma^4} \Vert y - Xw \Vert^2 = 0 \quad \Leftrightarrow \quad \sigma^2_{\text{MLE}} = \frac{1}{N} \Vert y - X w_{\text{MLE}} \Vert^2
$$

Since $$w_{\text{MLE}}$$ does not depend on $$\sigma^2$$, the joint maximum takes one step: $$w$$ first, then $$\sigma^2$$, the mean squared residual. This estimate is biased low: the residuals of $$D$$ fitted weights have $$N - D$$ degrees of freedom, and dividing by $$N - D$$ instead of $$N$$ gives an unbiased estimate.

## Maximum a Posteriori

Maximum likelihood overfits when $$N$$ is not much larger than $$D$$: the weights fit the noise, and they often take large values (sections 9.2.2 and 9.2.3 of *Mathematics for Machine Learning*, MML). A prior on the weights counters this. The [maximum a posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) (MAP) estimate maximizes the posterior, i.e. the sum of the log-likelihood and the log-prior:

$$
w_{\text{MAP}} = \arg\max_w \left( \log p(y \mid X, w) + \log p(w) \right)
$$

For the prior $$w \sim \mathcal{N}(0, b^2 I)$$, $$-\log p(w) = \frac{1}{2 b^2} w^T w + \text{const}$$, and the gradient of the negative log-posterior is zero at:

$$
\frac{1}{\sigma^2} \left( X^T X w - X^T y \right) + \frac{1}{b^2} w = 0 \quad \Leftrightarrow \quad w_{\text{MAP}} = \left( X^T X + \frac{\sigma^2}{b^2} I \right)^{-1} X^T y
$$

This is [ridge regression](https://en.wikipedia.org/wiki/Ridge_regression) with $$\lambda = \sigma^2 / b^2$$: the Gaussian prior is an L2 penalty, and a [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution) prior gives the L1 penalty of LASSO in the same way. A narrow prior (small $$b$$) pulls the weights toward 0, and a flat prior ($$b \to \infty$$) gives back $$w_{\text{MLE}}$$. The matrix $$X^T X + \lambda I$$ is invertible for any $$\lambda > 0$$, so the MAP estimate also exists when $$N < D$$. Its cost is that of the MLE.

## Posterior Distribution

MLE and MAP return one weight vector. Bayesian linear regression keeps the whole posterior, and it averages predictions over all weights, weighted by their posterior probability. With a Gaussian prior and a Gaussian likelihood, the posterior is Gaussian and has a closed form.

For the prior $$p(w) = \mathcal{N}(w \mid m_0, S_0)$$, with mean $$m_0$$ and covariance $$S_0$$, the log-posterior is, up to terms that do not depend on $$w$$:

$$
\begin{align*}
\log p(w \mid X, y) & = -\frac{1}{2 \sigma^2} (y - Xw)^T (y - Xw) - \frac{1}{2} (w - m_0)^T S_0^{-1} (w - m_0) + \text{const} \\
& = -\frac{1}{2} w^T \left( \sigma^{-2} X^T X + S_0^{-1} \right) w + \left( \sigma^{-2} X^T y + S_0^{-1} m_0 \right)^T w + \text{const}
\end{align*}
$$

It is quadratic in $$w$$, so the posterior is a Gaussian $$\mathcal{N}(w \mid m_N, S_N)$$. [Completing the square](https://en.wikipedia.org/wiki/Completing_the_square), i.e. matching the expression with $$\log \mathcal{N}(w \mid m_N, S_N) = -\frac{1}{2} w^T S_N^{-1} w + m_N^T S_N^{-1} w + \text{const}$$, gives $$S_N$$ from the quadratic terms and $$m_N$$ from the linear terms (equations 3.50 and 3.51 of Bishop's *Pattern Recognition and Machine Learning*, PRML):

$$
S_N^{-1} = S_0^{-1} + \sigma^{-2} X^T X, \qquad m_N = S_N \left( S_0^{-1} m_0 + \sigma^{-2} X^T y \right)
$$

The precisions (inverse covariances) of the prior and of the data add, as in the product of two Gaussian densities. For $$m_0 = 0$$ and $$S_0 = b^2 I$$, the posterior mean is $$w_{\text{MAP}}$$, since the mean and the mode of a Gaussian coincide. The posterior adds the covariance $$S_N$$, the remaining uncertainty about $$w$$.

The cost is that of the MAP estimate: $$O(ND^2)$$ to form $$X^T X$$ and $$O(D^3)$$ to factorize $$S_N^{-1}$$. Linear in $$N$$, it scales to large datasets, but $$S_N$$ needs $$O(D^2)$$ memory: 80 GB in double precision for $$D = 10^5$$ features.

### Online Learning

Bayes' rule can process the data one batch at a time: the posterior after a batch is the prior for the next one. For a batch $$(X_t, y_t)$$:

$$
S_t^{-1} = S_{t-1}^{-1} + \sigma^{-2} X_t^T X_t, \qquad S_t^{-1} m_t = S_{t-1}^{-1} m_{t-1} + \sigma^{-2} X_t^T y_t
$$

After the last batch, the result equals the posterior of all data at once, in any order of the batches. Training thus needs neither the whole dataset in memory nor a second pass over old data, which suits data streams and datasets that are deleted after use. The state, $$S_t^{-1}$$ and $$S_t^{-1} m_t$$, has size $$O(D^2)$$ whatever the number of samples. The same sums parallelize over the data: each worker computes $$X_k^T X_k$$ and $$X_k^T y_k$$ on its shard, and an [all-reduce](https://en.wikipedia.org/wiki/Collective_operation#All-Reduce) adds them.

{: .picture}
<img width="43.9%" height="43.9%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_posterior_1.svg"/>

{: .picture}
<img width="43.9%" height="43.9%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_posterior_3.svg"/>

{: .picture}
<img width="43.9%" height="43.9%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_posterior_4.svg"/>

{: .picture}
<img width="43.9%" height="43.9%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_posterior_5.svg"/>

{: .caption}
Online learning for the linear model $$y = w_0 + w_1x$$, after 1, 2, 3 and 20 data points (the last update adds 17 points at once). Left: the posterior density of the weights $$(w_0, w_1)$$; the cross marks the true weights $$(-0.3, 0.5)$$. Right: the data points and six lines drawn from the posterior. Inspired by: [Pattern Recognition and Machine Learning, C. Bishop, Figure 3.7](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

The figure shows online learning for the line $$y = w_0 + w_1 x$$, with data from $$w_0 = -0.3$$ and $$w_1 = 0.5$$ (cross) and Gaussian noise of standard deviation $$\sigma = 0.1$$, which the model knows. The prior is $$\mathcal{N}(0, I)$$. The rows show the posterior (left) and the samples with six lines drawn from the posterior (right), after 1, 2, 3 and 20 samples. One sample constrains only a line in the weight space, $$w_0 + w_1 x_1 = y_1$$, so the first posterior is a ridge along it. Two samples set both weights, and the posterior shrinks to a small ellipse, off the true weights because of the noise in two samples. After 20 samples, the posterior is concentrated around the true weights, and the sampled lines almost coincide.

## Predictive Distribution

For a new input $$x_*$$, the predictive distribution averages the likelihood over the posterior:

$$
p(y_* \mid x_*, X, y) = \int p(y_* \mid x_*, w) \, p(w \mid X, y) \, dw = \mathcal{N}\left( y_* \mid x_*^T m_N, \; x_*^T S_N x_* + \sigma^2 \right)
$$

The output $$y_* = x_*^T w + \varepsilon$$ is a linear function of the Gaussian $$w$$ plus independent Gaussian noise, so it is Gaussian, with mean $$x_*^T m_N$$ and variance $$x_*^T S_N x_* + \sigma^2$$ (PRML, equations 3.58 and 3.59). The variance has two terms: the noise $$\sigma^2$$, which more data cannot reduce, and the uncertainty about the weights, $$x_*^T S_N x_*$$, which falls as data arrive and is largest far from the training inputs. A prediction costs $$O(D)$$ for the mean and $$O(D^2)$$ for the variance.

{: .picture}
<img width="34.2%" height="34.2%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_predictive_distribution_0.svg"/>

{: .picture}
<img width="34.2%" height="34.2%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_predictive_distribution_1.svg"/>

{: .picture}
<img width="34.2%" height="34.2%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_predictive_distribution_2.svg"/>

{: .picture}
<img width="34.2%" height="34.2%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_predictive_distribution_3.svg"/>

{: .caption}
Bayesian linear regression of synthetic sinusoidal data with 9 Gaussian basis functions, after 1, 2, 4 and 25 data points. Each plot shows the ground truth $$\sin(2 \pi x)$$ in blue, the predictive mean in orange, and the predictive mean plus and minus one predictive standard deviation as a light-orange area. The model has a prior precision $$\alpha = 10^{-3}$$ and a noise precision $$\beta = 2$$. Inspired by: [Pattern Recognition and Machine Learning, C. Bishop, Figure 3.8](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

The figure shows the predictive distribution of a model with 9 Gaussian basis functions and a bias as features, fitted to noisy samples of $$\sin(2 \pi x)$$, after 1, 2, 4 and 25 samples. The band is narrow near the samples and wide away from them, where only the prior constrains the weights.

## Model Comparison

The posterior odds of two models $$M_1$$ and $$M_2$$, such as two sets of features, for the same data $$\mathcal{D}$$ are:

$$
\frac{p(M_1 \mid \mathcal{D})}{p(M_2 \mid \mathcal{D})} = \frac{p(\mathcal{D} \mid M_1)}{p(\mathcal{D} \mid M_2)} \, \frac{p(M_1)}{p(M_2)}
$$

The ratio of the evidences is the [Bayes factor](https://en.wikipedia.org/wiki/Bayes_factor). With equal prior probabilities, the model with the larger evidence is preferred. For Bayesian linear regression, $$y = Xw + \varepsilon$$ is a linear function of Gaussians, so the evidence has a closed form:

$$
p(y \mid X) = \mathcal{N}\left( y \mid X m_0, \; X S_0 X^T + \sigma^2 I \right)
$$

The evidence averages the likelihood over the prior, so it penalizes models that are too complex as well as models that are too simple, without a validation set. Maximizing it over $$\sigma^2$$ and the prior covariance, instead of fixing them, is the *evidence approximation*, or empirical Bayes (PRML, section 3.5).

## Choosing an Estimate

|   | MLE | MAP | Posterior |
|---|---|---|---|
| Result | $$w_{\text{MLE}}$$ | $$w_{\text{MAP}}$$ | $$\mathcal{N}(m_N, S_N)$$ |
| Prior | none | $$\mathcal{N}(0, b^2 I)$$ | $$\mathcal{N}(m_0, S_0)$$ |
| Defined for $$N < D$$ | no | yes | yes |
| Overfitting | when $$N$$ is close to $$D$$ | reduced by the prior | reduced by the prior |
| Predictive variance | $$\sigma^2$$ only | $$\sigma^2$$ only | $$x_*^T S_N x_* + \sigma^2$$ |
| Training | $$O(ND^2 + D^3)$$ | $$O(ND^2 + D^3)$$ | $$O(ND^2 + D^3)$$ |
| Prediction | $$O(D)$$ | $$O(D)$$ | $$O(D)$$ mean, $$O(D^2)$$ variance |

The three estimates cost the same to train; the posterior also keeps the $$D \times D$$ matrix $$S_N$$ for predictions. The MLE suffices when $$N \gg D$$ and a point prediction is enough. The MAP estimate is the better choice when $$N$$ is small or the inputs are correlated, at no extra cost. The full posterior is the choice when predictions need an uncertainty, as in [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) and [Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization), for online updates, and for model comparison.

## Related Models

With features $$\phi(x)$$ and the prior $$\mathcal{N}(0, S_0)$$, Bayesian linear regression is a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) with the kernel $$k(x, x') = \phi(x)^T S_0 \phi(x')$$ ([Rasmussen and Williams, 2006, section 2.1](https://gaussianprocess.org/gpml/chapters/RW2.pdf)). The two forms give the same predictions at different costs: the weight form factorizes a $$D \times D$$ matrix, in $$O(D^3)$$, and the kernel form an $$N \times N$$ matrix, in $$O(N^3)$$, which is cheaper when $$N < D$$. For outputs that are not Gaussian but in the exponential family, such as counts or classes, [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) apply; their posterior has no closed form.

Chapter 3 of PRML and section 9.3 of MML, both in the <a href="{{ site.resources_permalink }}">resources</a> section, cover these models in more depth. The <a href="{{ site.assets }}/bayesian-linear-regression/bayesian_linear_regression.ipynb">Bayesian linear regression notebook</a> holds the code of the figures.
