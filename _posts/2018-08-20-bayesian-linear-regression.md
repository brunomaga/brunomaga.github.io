---
layout: post
title:  "Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori"
categories: ["Machine Learning"]
tags: ["Bayesian inference", "linear regression", "MLE", "MAP"]
---

Most regression methods return a single set of best-fit parameters. This is the *frequentist* view: the parameters have one true value, and the data serve to estimate it. Bayesian methods look at the problem differently. They treat the parameters as random variables, and instead of a single answer they return a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) over the parameters, called the **posterior distribution**. The spread of this distribution measures how confident the model is, given the data it has observed, and this *uncertainty* carries over to every prediction the model makes.

As an example, consider the line $$y = w_0 + w_1 x$$, with intercept $$w_0$$ and slope $$w_1$$. The frequentist approach looks for the constant values of $$w_0$$ and $$w_1$$ that fit the data best. The Bayesian approach uses the same line, but its weights are no longer constants: they are drawn from probability distributions. Before any data are seen, these distributions express the **prior** knowledge about the weights. A common choice is that both weights are independent and normally distributed around zero, for example $$w_0, w_1 \sim \mathcal{N}(0, 1)$$. Training then turns this prior into the posterior, which tells which weights remain plausible once the data are known. The model can also describe *how noisy* the data are, with a noise term $$\varepsilon \sim \mathcal{N}(0, \sigma^2)$$, so that $$y = w_0 + w_1 x + \varepsilon$$.

Besides quantifying uncertainty, Bayesian methods allow **online learning**. The posterior summarizes everything the model has learnt so far, so a trained model can be updated with the new data alone, without going back to the old data. This is useful for data streams, and for datasets that are deleted after some time.

The rest of the post uses the general form of this model. An output $$y \in \mathbb{R}$$ is a linear function of an input $$x \in \mathbb{R}^D$$, plus Gaussian noise:

$$
y = x^T w + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

with weights $$w \in \mathbb{R}^D$$ and noise variance $$\sigma^2$$. The intercept is one of the weights, multiplied by an input that is always 1. The model only has to be linear in $$w$$, not in $$x$$, so the input can be replaced by a vector of features $$\phi(x)$$, such as powers of the input or Gaussian basis functions, and every result below still holds with $$\phi(x)$$ in place of $$x$$. For a dataset of $$N$$ independent samples, the rows of the matrix $$X \in \mathbb{R}^{N \times D}$$ are the inputs $$x_n^T$$, and the vector $$y \in \mathbb{R}^N$$ holds the outputs $$y_n$$. All the distributions in this post are [normal distributions](https://en.wikipedia.org/wiki/Normal_distribution), whose properties the [statistics post]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#gaussian-distribution) reviews.

Three estimates of the weights follow from this model, each one building on the previous one: the maximum likelihood estimate (MLE), the maximum a posteriori (MAP) estimate, and the full posterior distribution.

## Bayes' Rule

All three estimates start from [Bayes' rule]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#bayes-rule). It follows from the product rule of probability, $$p(A, B) = p(A) \, p(B \mid A)$$, which by symmetry can also be written as $$p(A, B) = p(B) \, p(A \mid B)$$. Equating the two right-hand sides and dividing by $$p(B)$$ gives:

$$
p(A \mid B) = \frac{p(B \mid A) \, p(A)}{p(B)}, \qquad \text{i.e.} \qquad \text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}
$$

In machine learning, the interest is in how the parameters of a model relate to the data. For linear regression, Bayes' rule gives the posterior of the weights:

$$
p(w \mid X, y) = \frac{p(y \mid X, w) \, p(w)}{p(y \mid X)} \propto p(y \mid X, w) \, p(w)
$$

Bayes' rule inverts the relation between the parameters and the data. The likelihood $$p(y \mid X, w)$$ says how probable the observed outputs are for given weights, and the posterior says how probable the weights are for the observed outputs. The inputs $$X$$ are given rather than modelled, so every distribution is conditioned on them. The denominator, called the **evidence**, is the likelihood averaged over the prior:

$$
p(y \mid X) = \int p(y \mid X, w) \, p(w) \, dw = \mathbb{E}_{w} \left[ p(y \mid X, w) \right]
$$

It does not depend on $$w$$, so when the goal is to find the most probable weights, the evidence can be dropped, and only the numerator $$p(y \mid X, w) \, p(w)$$ needs to be maximized. The evidence comes back at the end of the post, where it serves to compare models.

How much does the choice of the prior matter? Less and less as data arrive. The [Bernstein–von Mises theorem](https://en.wikipedia.org/wiki/Bernstein%E2%80%93von_Mises_theorem) makes this precise: as $$N$$ grows, the posterior converges to a Gaussian centered at the maximum likelihood estimate, with a covariance that shrinks as $$1/N$$. For a large dataset, the data alone carry enough information, and the prior has little influence. The theorem holds when the model is correct and regular, and when the prior density is continuous and positive at the true parameters. The last condition makes sense: where the prior is zero, the posterior is zero too, and no amount of data can change that.

## Maximum Likelihood

The simplest estimate ignores the prior and only looks at the likelihood. The **likelihood** $$p(y \mid X, w)$$ is the probability density of the observed outputs for given weights, and it can be read in two ways. For fixed weights, it is a distribution over the data: it describes how the outputs scatter around the regression line. Once the data are observed, they are fixed, and the likelihood becomes a function of the weights instead, one that scores how well each setting of $$w$$ explains the observations. This function is not a probability distribution over $$w$$, as it does not integrate to 1 over the weights. The [maximum likelihood estimate](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is the setting with the highest score, i.e. the weights under which the observed data are most probable (section 8.3.1 of [*Mathematics for Machine Learning*]({{ site.resources_permalink }}), MML):

$$
w_{\text{MLE}} = \arg\max_w \, p(y \mid X, w)
$$

The noise of each sample is independent, so the likelihood of the dataset is the product of the likelihoods of its samples, each a Gaussian centered at the prediction $$x_n^T w$$:

$$
p(y \mid X, w) = \prod_{n=1}^{N} \mathcal{N}(y_n \mid x_n^T w, \sigma^2)
$$

A product of exponentials is awkward to differentiate, so the usual trick is to maximize the logarithm of the likelihood instead. This is allowed because the logarithm is monotonically increasing, so it does not move the location of the maximum. And it helps in three ways: it turns the product into a sum, it brings the exponent of each Gaussian down as a plain quadratic term, and it avoids numerical [underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow). The last point matters in practice: the product of 1,000 densities of 0.01 is $$10^{-2000}$$, far below the smallest positive double-precision number, about $$5 \times 10^{-324}$$. For one sample, the log-likelihood is:

$$
\log p(y_n \mid x_n, w) = \log \left( \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(y_n - x_n^T w)^2}{2 \sigma^2} \right) \right) = - \frac{1}{2 \sigma^2} (y_n - x_n^T w)^2 - \frac{1}{2} \log(2 \pi \sigma^2)
$$

Summing over the samples and changing the sign gives the negative log-likelihood, which is the loss function $$L(w)$$ to minimize:

$$
L(w) = -\log p(y \mid X, w) = \frac{1}{2 \sigma^2} \sum_{n=1}^{N} (y_n - x_n^T w)^2 + \frac{N}{2} \log(2 \pi \sigma^2) = \frac{1}{2 \sigma^2} \Vert y - Xw \Vert^2 + \frac{N}{2} \log(2 \pi \sigma^2)
$$

Up to a scale and a constant, this is the sum of squared errors. In other words, maximum likelihood under Gaussian noise is exactly the method of least squares. The loss is a quadratic function of $$w$$, so its minimum is where its gradient is zero. Expanding the squared norm, and using two rules of matrix calculus, $$\nabla_w (a^T w) = a$$ and $$\nabla_w (w^T A w) = (A + A^T) w$$ (equations 69 and 81 of <a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>), the derivation takes a few lines:

$$
\begin{align*}
& \nabla_w L(w) = 0 \\
\Leftrightarrow \quad & \frac{1}{2 \sigma^2} \nabla_w \left( y^T y - 2 y^T X w + w^T X^T X w \right) = 0 \\
\Leftrightarrow \quad & \frac{1}{2 \sigma^2} \left( 2 X^T X w - 2 X^T y \right) = 0 \\
\Leftrightarrow \quad & X^T X w = X^T y \\
\Leftrightarrow \quad & w_{\text{MLE}} = (X^T X)^{-1} X^T y
\end{align*}
$$

This is the closed-form solution of least squares, derived in the [linear regression post]({{ site.baseurl }}{% post_url 2017-08-17-linear-regression-matrix-factorization %}), which also details how to compute it: forming $$X^T X$$ costs $$O(ND^2)$$ operations, and its [Cholesky factorization](https://en.wikipedia.org/wiki/Cholesky_decomposition) costs $$O(D^3)$$. Two properties of the solution are worth noticing. First, the inverse exists only when $$X$$ has full column rank, which requires at least as many samples as weights, $$N \ge D$$; with fewer samples, infinitely many weights fit the data equally well. Second, the noise variance $$\sigma^2$$ has disappeared: the best line does not depend on how noisy the data are.

The noise variance is still a parameter of the likelihood, though, and the same principle estimates it. The terms of the log-likelihood that depend on $$\sigma^2$$ are:

$$
\log p(y \mid X, w, \sigma^2) = -\frac{N}{2} \log \sigma^2 - \frac{1}{2 \sigma^2} \sum_{n=1}^{N} (y_n - x_n^T w)^2 + \text{const}
$$

and setting their derivative with respect to $$\sigma^2$$ to zero gives:

$$
-\frac{N}{2 \sigma^2} + \frac{1}{2 \sigma^4} \sum_{n=1}^{N} (y_n - x_n^T w)^2 = 0 \quad \Leftrightarrow \quad \sigma^2_{\text{MLE}} = \frac{1}{N} \sum_{n=1}^{N} (y_n - x_n^T w_{\text{MLE}})^2
$$

In words, the noise variance is the mean of the squared distances between the observations and the predictions of the fitted line. Because $$w_{\text{MLE}}$$ does not depend on $$\sigma^2$$, the two estimates are found one after the other: first the weights, then the variance of their residuals. This estimate is slightly too small on average, however. The $$D$$ fitted weights have already adapted to the noise of the samples, so the residuals keep only $$N - D$$ degrees of freedom, and dividing by $$N - D$$ instead of $$N$$ gives an [unbiased](https://en.wikipedia.org/wiki/Bias_of_an_estimator) estimate.

## Maximum a Posteriori

Maximum likelihood trusts the data completely, and this is also its weakness. When the number of samples $$N$$ is not much larger than the number of weights $$D$$, the weights start to fit the noise instead of the underlying function, and they often take very large values to do so (sections 9.2.2 and 9.2.3 of MML). This is [overfitting](https://en.wikipedia.org/wiki/Overfitting). A natural remedy is to tell the model what reasonable weights look like before it sees any data, through a prior distribution $$p(w)$$, and then to ask for the weights that are most probable once the data are taken into account. These weights maximize the posterior, and they are called the [maximum a posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) (MAP) estimate. By Bayes' rule, and with the same logarithm trick as before, the log-posterior is the sum of the log-likelihood and the *log-prior*, plus a constant:

$$
w_{\text{MAP}} = \arg\max_w \, \log p(w \mid X, y) = \arg\max_w \left( \log p(y \mid X, w) + \log p(w) \right)
$$

The MAP estimate is therefore a compromise between two forces. The likelihood pulls the weights toward the values that fit the data best, and the prior pulls them toward the values it considers plausible (MML, section 8.3.2). For a zero-mean Gaussian prior $$w \sim \mathcal{N}(0, b^2 I)$$, the log-prior is $$-\frac{1}{2 b^2} w^T w$$ plus a constant, a penalty that grows as the weights move away from the origin. Setting the gradient of the negative log-posterior to zero, with the same two rules as before, gives:

$$
\begin{align*}
& -\nabla_w \log p(w \mid X, y) = 0 \\
\Leftrightarrow \quad & \nabla_w \left( \frac{1}{2 \sigma^2} (y - Xw)^T (y - Xw) + \frac{1}{2 b^2} w^T w \right) = 0 \\
\Leftrightarrow \quad & \frac{1}{\sigma^2} \left( X^T X w - X^T y \right) + \frac{1}{b^2} w = 0 \\
\Leftrightarrow \quad & \left( X^T X + \frac{\sigma^2}{b^2} I \right) w = X^T y \\
\Leftrightarrow \quad & w_{\text{MAP}} = \left( X^T X + \frac{\sigma^2}{b^2} I \right)^{-1} X^T y
\end{align*}
$$

The only difference from $$w_{\text{MLE}}$$ is the term $$\frac{\sigma^2}{b^2} I$$ inside the inverse, and it acts as a regularizer. In fact, this is exactly [ridge regression](https://en.wikipedia.org/wiki/Ridge_regression), the least-squares solution with an L2 penalty $$\lambda \Vert w \Vert^2$$, for $$\lambda = \sigma^2 / b^2$$. The Gaussian prior and the L2 regularizer are the same idea seen from two angles, and a [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution) prior gives the L1 penalty of [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) in the same way. Unlike the MLE, the solution now depends on the noise: the noisier the data (large $$\sigma^2$$) or the narrower the prior (small $$b$$), the stronger the pull toward zero, and a flat prior ($$b \to \infty$$) gives back $$w_{\text{MLE}}$$. The regularizer also has a numerical benefit. The matrix $$X^T X + \lambda I$$ is invertible for any $$\lambda > 0$$, so the MAP estimate exists even when $$N < D$$, and it costs the same to compute as the MLE.

## Posterior Distribution

MLE and MAP both return a single weight vector, and with it they discard the information about how uncertain those weights are. [Bayesian linear regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression) takes the idea of the prior one step further and keeps the whole posterior distribution. No single set of weights is chosen: a prediction averages over all plausible weights, each one weighted by its posterior probability. This may sound expensive, but with a Gaussian prior and Gaussian noise the posterior is itself Gaussian and has a closed form, which this section derives, following the notation of Bishop's [*Pattern Recognition and Machine Learning*]({{ site.resources_permalink }}) (PRML).

The weights start from a Gaussian prior with mean vector $$m_0$$ and a positive definite covariance matrix $$S_0$$, whose density in $$D$$ dimensions is:

$$
p(w) = \mathcal{N}(w \mid m_0, S_0) = (2 \pi)^{-\frac{D}{2}} \det(S_0)^{-\frac{1}{2}} \exp \left( - \frac{1}{2} (w - m_0)^T S_0^{-1} (w - m_0) \right)
$$

The weights are independent if $$S_0$$ is diagonal, and correlated otherwise. The likelihood is the product of Gaussians of the previous sections, written in vector form:

$$
p(y \mid X, w) = (2 \pi \sigma^2)^{-\frac{N}{2}} \exp \left( - \frac{1}{2 \sigma^2} (y - Xw)^T (y - Xw) \right)
$$

Bayes' rule multiplies the two. The factors in front of the exponentials do not depend on $$w$$, and neither does the evidence, so they all go into a constant. Taking the logarithm and expanding the products then sorts the remaining terms by their degree in $$w$$:

$$
\begin{align*}
\log p(w \mid X, y) & = -\frac{1}{2} \left( \sigma^{-2} (y - Xw)^T (y - Xw) + (w - m_0)^T S_0^{-1} (w - m_0) \right) + \text{const} \\
& = -\frac{1}{2} \left( \sigma^{-2} y^T y - 2 \sigma^{-2} y^T X w + \sigma^{-2} w^T X^T X w + w^T S_0^{-1} w - 2 m_0^T S_0^{-1} w + m_0^T S_0^{-1} m_0 \right) + \text{const} \\
& = -\frac{1}{2} w^T \left( \sigma^{-2} X^T X + S_0^{-1} \right) w + \left( \sigma^{-2} X^T y + S_0^{-1} m_0 \right)^T w + \text{const}
\end{align*}
$$

where the last constant also absorbs $$y^T y$$ and $$m_0^T S_0^{-1} m_0$$, which do not depend on $$w$$. The log-posterior is a negative quadratic function of $$w$$, and the exponential of a negative quadratic function is, up to normalization, a Gaussian. So the posterior is a Gaussian $$\mathcal{N}(w \mid m_N, S_N)$$, and only its mean $$m_N$$ and its covariance $$S_N$$ remain to be found. The method of [completing the square](https://en.wikipedia.org/wiki/Completing_the_square) finds them. The logarithm of the target Gaussian expands in the same way:

$$
\log \mathcal{N}(w \mid m_N, S_N) = -\frac{1}{2} (w - m_N)^T S_N^{-1} (w - m_N) + \text{const} = -\frac{1}{2} w^T S_N^{-1} w + m_N^T S_N^{-1} w + \text{const}
$$

and the two expressions agree for every $$w$$ only if their quadratic terms and their linear terms match. The quadratic terms give the covariance, and the linear terms then give the mean:

$$
S_N^{-1} = S_0^{-1} + \sigma^{-2} X^T X, \qquad m_N = S_N \left( S_0^{-1} m_0 + \sigma^{-2} X^T y \right)
$$

in agreement with equations 3.50 and 3.51 of PRML. These two formulas say a lot about how Bayesian learning works. The inverse covariance, or *precision*, of the posterior is the precision of the prior plus a term contributed by the data, $$\sigma^{-2} X^T X$$: every sample adds information, and the posterior can never become wider. The posterior mean combines the prior mean and the data, each weighted by its precision. And for the prior of the MAP section, $$m_0 = 0$$ and $$S_0 = b^2 I$$, the posterior mean is exactly $$w_{\text{MAP}}$$, since the mean and the mode of a Gaussian coincide. The full posterior therefore contains the MAP estimate, plus something that no point estimate can give: the covariance $$S_N$$, which measures the uncertainty that remains about the weights.

The cost is that of the MAP estimate: $$O(ND^2)$$ operations to form $$X^T X$$, and $$O(D^3)$$ to factorize $$S_N^{-1}$$. It grows only linearly with the number of samples, so the method scales well to large datasets. The dimension is the limit: $$S_N$$ needs $$O(D^2)$$ memory, which is already 80 GB in double precision for $$D = 10^5$$ features.

### Online Learning

The closed form makes online learning straightforward. Nothing in Bayes' rule requires the prior to be the initial one: the posterior after one batch of data can serve as the prior for the next batch, so that what the model has learnt so far becomes its new initial knowledge. For a batch $$(X_t, y_t)$$, the formulas above turn into an update of the posterior of the previous batch:

$$
S_t^{-1} = S_{t-1}^{-1} + \sigma^{-2} X_t^T X_t, \qquad S_t^{-1} m_t = S_{t-1}^{-1} m_{t-1} + \sigma^{-2} X_t^T y_t
$$

Since $$X^T X$$ and $$X^T y$$ are sums over the samples, the result after the last batch is exactly the posterior of all the data at once, whatever the order of the batches. Training therefore needs neither the whole dataset in memory nor a second pass over old data, which suits data streams and datasets that are deleted after use. The state of the model, $$S_t^{-1}$$ and $$S_t^{-1} m_t$$, keeps a size of $$O(D^2)$$, however many samples it has seen. The same sums also parallelize over the data: each worker computes $$X_k^T X_k$$ and $$X_k^T y_k$$ on its shard of the data, and an [all-reduce](https://en.wikipedia.org/wiki/Collective_operation#All-Reduce) adds them up.

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

The figure shows this process for the line $$y = w_0 + w_1 x$$ of the introduction. The data come from the weights $$w_0 = -0.3$$ and $$w_1 = 0.5$$, marked by the cross, plus Gaussian noise with a standard deviation $$\sigma = 0.1$$ that the model knows. The prior is centered at zero, with a standard deviation of 1 for both weights. Each row shows the posterior on the left, and on the right the samples seen so far, together with six lines drawn from the posterior. The first sample (top row) constrains only a line in the weight space, $$w_0 + w_1 x_1 = y_1$$, so the posterior is a long ridge along that line, and the sampled lines all pass near the sample, but in many directions. The second sample is combined with this posterior, which now acts as the prior. Two samples are enough to determine both weights, and the posterior shrinks to a small ellipse. It still sits off the true weights, because two noisy samples are not yet representative, and a third sample, close to the first one, changes little. After 20 samples, the posterior is concentrated around the true weights, and the sampled lines almost coincide.

## Predictive Distribution

The posterior over the weights is a means, not an end: what matters in practice is the prediction for a new input $$x_*$$. The prediction $$y_* = x_*^T w + \varepsilon$$ inherits two sources of randomness: the remaining uncertainty about the weights, described by the posterior $$\mathcal{N}(m_N, S_N)$$, and the noise $$\varepsilon \sim \mathcal{N}(0, \sigma^2)$$. Averaging the likelihood over the posterior gives the **predictive distribution**:

$$
p(y_* \mid x_*, X, y) = \int p(y_* \mid x_*, w) \, p(w \mid X, y) \, dw
$$

Since $$y_*$$ is a linear function of the Gaussian $$w$$ plus independent Gaussian noise, it is Gaussian too, and its mean and variance are enough to describe it. The [expectation](https://en.wikipedia.org/wiki/Expected_value) is linear and the noise has zero mean, so:

$$
\mathbb{E}[y_*] = \mathbb{E}[x_*^T w + \varepsilon] = x_*^T \mathbb{E}[w] + \mathbb{E}[\varepsilon] = x_*^T m_N
$$

The mean prediction is thus an ordinary linear regression, with the weights set to their posterior mean. The [variance](https://en.wikipedia.org/wiki/Variance) follows from the rules for sums and [linear transformations]({{ site.baseurl }}{% post_url 2018-06-15-statistics-for-ml %}#linear-transformations) of random variables:

$$
\begin{align*}
\mathrm{Var}[y_*] & = \mathrm{Var}[x_*^T w] + \mathrm{Var}[\varepsilon] + 2 \, \mathrm{Cov}[x_*^T w, \varepsilon] \\
& = \mathrm{Var}[x_*^T w] + \mathrm{Var}[\varepsilon] && \text{(} w \text{ and } \varepsilon \text{ are independent)} \\
& = x_*^T \, \mathrm{Var}[w] \, x_* + \sigma^2 \\
& = x_*^T S_N x_* + \sigma^2
\end{align*}
$$

Together, these give the predictive distribution of equations 3.58 and 3.59 of PRML:

$$
p(y_* \mid x_*, X, y) = \mathcal{N} \left( y_* \mid x_*^T m_N, \; x_*^T S_N x_* + \sigma^2 \right)
$$

The two terms of the variance have different origins, and the difference is what makes the predictive distribution useful. The noise $$\sigma^2$$ is a property of the data, and no amount of training can reduce it. The term $$x_*^T S_N x_*$$ is the uncertainty about the weights: it shrinks as data arrive, and it is largest for inputs far from the training data, where the weights are least constrained. A prediction costs $$O(D)$$ operations for the mean and $$O(D^2)$$ for the variance.

The next figure shows the effect on a harder problem: fitting noisy samples of $$\sin(2 \pi x)$$ with a model whose features are 9 Gaussian basis functions and a bias.

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

As in the previous figure, every new sample improves the model. With few samples, the band is narrow only near them: away from the samples, only the prior constrains the weights, and the band stays wide. With 25 samples spread over the whole interval, the predictive mean follows the sine closely, and the width of the band comes mostly from the noise.

## Model Comparison

Bayes' rule can also compare whole models, for example two different sets of features for the same data $$\mathcal{D}$$. Applied to two models $$M_1$$ and $$M_2$$ instead of the weights, it gives the ratio of their posteriors, called the **posterior odds**:

$$
\frac{p(M_1 \mid \mathcal{D})}{p(M_2 \mid \mathcal{D})} = \frac{p(\mathcal{D} \mid M_1) \, p(M_1) / p(\mathcal{D})}{p(\mathcal{D} \mid M_2) \, p(M_2) / p(\mathcal{D})} = \frac{p(\mathcal{D} \mid M_1)}{p(\mathcal{D} \mid M_2)} \, \frac{p(M_1)}{p(M_2)}
$$

The probability of the data, $$p(\mathcal{D})$$, cancels out. If both models are equally probable a priori, the comparison reduces to the ratio of their evidences, called the [Bayes factor](https://en.wikipedia.org/wiki/Bayes_factor): a ratio above 1 favours $$M_1$$, and a ratio below 1 favours $$M_2$$. For Bayesian linear regression, the evidence is the denominator of Bayes' rule from the beginning of the post, and it has a closed form. The outputs $$y = Xw + \varepsilon$$ are a linear function of Gaussians, so:

$$
p(y \mid X) = \mathcal{N}\left( y \mid X m_0, \; X S_0 X^T + \sigma^2 I \right)
$$

Since the evidence is the likelihood averaged over the prior, it has a remarkable property. A model that is too simple fits the data poorly for every setting of its weights, so its average likelihood is low. A model that is too complex can fit many different datasets, so it spreads its probability thinly, and the observed data receive little of it. The evidence therefore penalizes both underfitting and overfitting, without the need for a validation set. Going one step further, the evidence can be maximized over $$\sigma^2$$ and the prior covariance, instead of fixing them by hand. This is the *evidence approximation*, also known as empirical Bayes (PRML, section 3.5).

## Choosing an Estimate

The three estimates of this post build on each other, and the table compares them.

|   | MLE | MAP | Posterior |
|---|---|---|---|
| Result | $$w_{\text{MLE}}$$ | $$w_{\text{MAP}}$$ | $$\mathcal{N}(m_N, S_N)$$ |
| Prior | none | $$\mathcal{N}(0, b^2 I)$$ | $$\mathcal{N}(m_0, S_0)$$ |
| Defined for $$N < D$$ | no | yes | yes |
| Overfitting | when $$N$$ is close to $$D$$ | reduced by the prior | reduced by the prior |
| Predictive variance | $$\sigma^2$$ only | $$\sigma^2$$ only | $$x_*^T S_N x_* + \sigma^2$$ |
| Training | $$O(ND^2 + D^3)$$ | $$O(ND^2 + D^3)$$ | $$O(ND^2 + D^3)$$ |
| Prediction | $$O(D)$$ | $$O(D)$$ | $$O(D)$$ mean, $$O(D^2)$$ variance |

All three cost the same to train, and the posterior also keeps the $$D \times D$$ matrix $$S_N$$ for its predictions. In practice, the MLE is enough when there are many more samples than weights ($$N \gg D$$) and a point prediction is all that is needed. The MAP estimate is the better choice when data are scarce or the inputs are correlated, and it comes at no extra cost. The full posterior is the right tool whenever a prediction needs an uncertainty, as in [active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) and [Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization), and whenever models are compared through their evidence.

## Related Models

Bayesian linear regression leads to two larger families of models. The first one is the [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process). With features $$\phi(x)$$ and the prior $$\mathcal{N}(0, S_0)$$, Bayesian linear regression is a Gaussian process with the kernel $$k(x, x') = \phi(x)^T S_0 \phi(x')$$ ([Rasmussen and Williams, 2006, section 2.1](https://gaussianprocess.org/gpml/chapters/RW2.pdf)). The two views give the same predictions, but at different costs: the weight view factorizes a $$D \times D$$ matrix, in $$O(D^3)$$, while the kernel view factorizes an $$N \times N$$ matrix, in $$O(N^3)$$, which is cheaper when there are fewer samples than features. The second family covers outputs that are not Gaussian but still belong to the exponential family, such as counts or classes. These are the [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model), whose posterior has no closed form.

Chapter 3 of PRML and section 9.3 of MML, both on the [resources page]({{ site.resources_permalink }}), cover these models in more depth. The code of the closed-form solutions and of the figures above is in the <a href="{{ site.assets }}/bayesian-linear-regression/bayesian_linear_regression.ipynb">Bayesian linear regression notebook</a>.
