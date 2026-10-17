---
layout: post
title:  "Bayesian Linear Regression, Maximum Likelihood and Maximum-A-Posteriori"
categories: ["Machine Learning"]
tags: ["Bayesian inference", "linear regression", "MLE", "MAP"]
---

Bayesian methods model the relationship between an input and an output while also providing a measure of *uncertainty*, i.e. how confident the model is, based on the observed data. Most common *frequentist* methods output a set of best-fit parameters. In contrast, the output of a Bayesian regression is a probability distribution for each model parameter, called the **posterior distribution**.

As a comparison, consider the simple linear regression $$y = mx + b$$. The frequentist approach finds constant values for the slope $$m$$ and the bias $$b$$, with $$m \in \mathbb{R}$$ and $$b \in \mathbb{R}$$. The Bayesian approach also computes $$y = mx + b$$. However, $$b$$ and $$m$$ are not constant values: they are drawn from probability distributions. The parameters of those distributions are the values learnt (or *tuned*) during training. A common choice of **prior knowledge** is that $$m$$ and $$b$$ are independent and normally distributed, i.e. $$b \thicksim \mathcal{N}(\mu_{b}, \sigma^2_{b})$$ and $$m \thicksim \mathcal{N}(\mu_{m}, \sigma^2_{m})$$. The parameters to be learnt are then all $$\mu$$ and $$\sigma$$. A Bayesian approach can also model the noise (whose variance is the inverse of the precision) with a term $$\varepsilon$$ that captures *how noisy* the data is. The model then becomes $$y = mx + b + \varepsilon$$, with $$\varepsilon \thicksim \mathcal{N}(0, \sigma^2_{\varepsilon})$$.
 
Besides uncertainty quantification, Bayesian methods also allow **online learning**, i.e. the continuous update of a trained model (from previously seen data) using only the new data. This is useful e.g. for datasets that are purged periodically.

### Refresher: Normal distribution

A [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) is a mathematical function that gives the probabilities of occurrence of the possible outcomes of an experiment. The methods in this post are based solely on the [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution), defined for an input $$x$$ by:

$$
p(x \mid \mu, \sigma ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } \,\, \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)
$$

for a distribution with mean $$\mu$$ and standard deviation $$\sigma$$ in univariate notation, or

$$
p(x \mid \mu, \Sigma ) = \left(\frac{1}{2 \pi}\right)^{D/2} \,\, \det(\Sigma)^{-1/2} \exp \left( - \frac{1}{2}(x - \mu)^T \Sigma^{-1} (x-\mu) \right)
$$

with mean vector $$\mu$$ and covariance matrix $$\Sigma$$ in its multivariate (*matrix*) notation.

## Bayes Theorem

In probability, the **product rule** states that $$p(A, B) = p(A) p(B\mid A)$$. By symmetry, $$p(B,A) = p(B) p(A\mid B)$$. Equating the right-hand sides of both equations and rearranging the terms gives **Bayes Theorem**:

\begin{equation}
\text{posterior} = \frac{\text{likelihood}\, \times \, \text{prior}}{\text{evidence}} = p(A\mid B) = \frac{p(B\mid A) p(A)}{p(B)}\end{equation}

In Machine Learning, the main interest is how the parameters of the model $$\theta$$ relate to the data $$X$$:

$$
p(\theta \mid X) = \frac{p(X\mid \theta) p(\theta)}{p(X)} \propto p(X \mid \theta) p(\theta)
\label{eq_prior_AB}
$$

where the evidence is

$$
p(X) = \int p(X \mid \theta) p(\theta) d\theta = \mathbb{E}_{\theta}[ p(X \mid \theta) ]
$$   

In ML, Bayes Theorem inverts the relationship between the parameters $$θ$$ and the data $$X$$ to obtain the posterior.

An important result, the **Bernstein-von Mises Theorem**, states that:
- for a sufficiently large sample size, the posterior distribution becomes independent of the prior (as long as the prior is neither 0 nor 1)
  - i.e. for a sufficiently large dataset, the prior has little influence, as the data alone carries enough information;
  - as the number of datapoints goes to infinity, the posterior distribution converges to a normal distribution whose mean is the maximum likelihood estimator;
    - this is the Bayesian counterpart of the asymptotic normality of the maximum likelihood estimator: the posterior concentrates around the maximum likelihood estimate, and its variance shrinks as $$1/N$$;
  - i.e. the effect of the prior decreases as the data increases;
- since the goal is typically to maximize the posterior, the evidence term can be dropped (as it is independent of $$\theta$$), and only the maximum of the term $$p(X \mid \theta) p(\theta)$$ is computed.

Bayesian methods involve two main optimization problems: the Maximum Likelihood Estimator (MLE) and the Maximum-a-Posteriori (MAP).

## Maximum Likelihood Estimator (MLE)


(The rules used in the following derivations are listed in the [appendix at the bottom of the page](#appendix-lin-alg). <a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a> gives a more detailed overview of matrix operations.)



Finding *how likely it is for an output $$y$$ to belong to a model defined by data $$X$$, weights $$w$$ and model parameters $$\sigma$$ (if any)*, i.e. **maximizing the likelihood $$p(y\mid w, X, \sigma^2)$$**, yields the **Maximum Likelihood Estimator (MLE)**. Maximizing the likelihood means maximizing the probability of the training data, given the model parameters:

$$ w_{MLE} = argmax_w \,\, p(y \mid w, X)  $$

Consider what the probability density $$p(x \mid θ)$$ models for a fixed value of θ. It is a distribution that models the uncertainty of the data for a given parameter setting. In a complementary view, the data is fixed (because it has been observed) and the parameters θ vary. In this view, the MLE measures how likely a particular setting of θ is for the observations x. The maximum likelihood estimator is therefore the most likely parameter θ for the set of data (MML book, section 8.3.1).

When the prior and posterior distributions are computationally tractable, their parameters can be optimized with [coordinate ascent](https://en.wikipedia.org/wiki/Coordinate_descent), a variant of gradient-based optimization covered [in another post]({% post_url 2017-08-17-linear-regression-matrix-factorization %}). Instead of updating all parameters at once, each iteration updates one parameter (or block of parameters) at a time, e.g. the mean $$\mu$$ and then the standard deviation $$\sigma$$ in a Gaussian setting. Each update follows the partial derivative of the objective with respect to that parameter, while the other parameters stay fixed.

To simplify the computation, the *log-trick* places the term to optimize inside a log function. This is valid because $$log$$ is a monotonically increasing function, so applying it does not change the input values at which the minimum or maximum is attained (i.e. where the gradient is zero). Moreover, since the Gaussian distribution is a product of exponentials, the $$log$$ function *brings the power term out* of the exponential, which makes the computation simpler and faster. Applying the log-trick to the function to minimize gives:

$$
  \begin{align*}
-\log \,\, p(y \mid w, X) & = -log \prod_{n=1}^N p(y_n \mid w, X_n) \\
                                       & = - \sum_{n=1}^N log \,\, p(y_n \mid w, X_n)\\
  \end{align*}
$$

That is, the log of products becomes a sum of logs. In the linear regression model, the likelihood is Gaussian, due to the Gaussian noise term $$\varepsilon \thicksim \mathcal{N}(0, \sigma^2_{\varepsilon})$$. Therefore:

$$
\begin{aligned}
log \,\,  p(y_n \mid w, x_n, \sigma^2) & = \log \left( \frac{1}{ \sqrt{2 \pi \sigma^2} } \,\, \exp \left( - \frac{(y_n - x_n^Tw)^2}{2 \sigma^2} \right) \right)\\
& =  - \frac{1}{2 \sigma^2} (y_n - x^T_n w)^2 + const \\
\end{aligned}
\label{eq_lr_likelihood}
$$ 

where $$const$$ includes all terms independent of $$w$$. Taking the negative of this log-likelihood, and ignoring the constants (whose derivative is zero), gives the following loss function:

$$
  \begin{align*}
L(w) & = \frac{1}{2 \sigma^2} \sum_{n=1}^N (y_n - x_n^Tw)^2 \\
     & = \frac{1}{2 \sigma^2} (y - Xw)^T (y-Xw) \\ 
     & = \frac{1}{2 \sigma^2} \| y-Xw \|^2 \\
  \end{align*}
$$

The second trick is that this loss is quadratic in $$w$$. Thus, the minimization has a unique global solution, found where the derivative is zero:

$$
  \begin{align*}
       & \frac{dL}{dw} = 0 \\
   \Leftrightarrow & 0 = \frac{d}{dw} \left( \frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) \right) \\
   \Leftrightarrow & 0 =  \frac{1}{2 \sigma^2} \frac{d}{dw} \left( y^Ty - 2y^TXw + w^TX^TXw \right) \\
   \Leftrightarrow & 0 =  \frac{1}{2 \sigma^2} \left( -2y^TX + 2w^TX^TX \right) \\
   \Leftrightarrow & y^TX = w^TX^TX \\
   \Leftrightarrow & w^T = y^TX (X^TX)^{-1} \\
   \Leftrightarrow & w_{MLE} = (X^TX)^{-1}X^Ty \\
  \end{align*}
$$

This solution is independent of the noise variance $$\sigma^2$$. It also matches the closed-form solution $$ w = (X^TX)^{-1} X^Ty$$ of the Least Squares problem, derived in a [previous post]({{ site.baseurl }}{% post_url 2017-08-17-linear-regression-matrix-factorization %}). In practice, minimizing the Least Squares problem is equivalent to finding the most likely $$w$$ under the assumption that $$y$$ contains Gaussian noise, i.e. $$y = wx + b + \varepsilon$$, with $$\varepsilon \thicksim \mathcal{N}(0, \sigma^2)$$.


### Adding regularization

Regularizers can be added as in non-Bayesian regression, and may also have an analytical solution. As an example, an $$L_2$$/Ridge regularizer gives:

$$
  \begin{align*}
 & 0 = \frac{d}{dw} \left( -L - \frac{\alpha}{2} w^Tw \right)\\
 \Leftrightarrow & 0 = \frac{d}{dw} \left( - \frac{N}{2} log ( 2 \pi \sigma^2) - \frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) - \frac{\alpha}{2} w^Tw \right)\\
 \Leftrightarrow & w_{MLE\_reg} = (X^TX + \lambda I )^{-1} X^Ty\\
  \end{align*}
$$

where $$\lambda = \alpha \sigma^2$$. Contrary to the previous case, the solution now depends on the noise. The regularizer constant $$\alpha/2$$ was chosen in the first step to simplify the maths, as it *cancels out* the 2 from the derivative of $$w^Tw$$.

### Estimating noise variance

So far, the noise variance $$\sigma^2$$ was assumed to be known. However, the same Maximum Likelihood principle gives the estimator $$\sigma^2_{MLE}$$ for the noise:

$$
  \begin{align*}
  log \,\, p(y \mid X, w, \sigma^2) = & \sum_{n=1}^N log \,\, \mathcal{N} (y_n \mid X_nw, \sigma^2) \\
 = & \sum_{n=1}^N \left( -\frac{1}{2} log(2\pi) -\frac{1}{2} log(\sigma^2) -\frac{1}{2 \sigma^2}(y_n-X_nw)^2 \right)\\
 = & -\frac{N}{2} log \sigma^2 -\frac{1}{2 \sigma^2} \sum_{n=1}^N (y_n - X_nw)^2 + const\\ 
  \end{align*}
$$

The partial derivative of the log-likelihood with respect to $$\sigma^2$$, and the MLE of $$\sigma^2$$, are then:

$$
  \begin{align*}
   & \frac{d \,\, log \,\, p(y \mid X, w, \sigma^2)}{d \sigma^2} = -\frac{N}{2 \sigma^2} + \frac{1}{2 \sigma^4} \sum_{n=1}^N (y_n - X_nw)^2 = 0\\
   \Leftrightarrow & \frac{N}{2 \sigma^2} = \frac{1}{2 \sigma^4} \sum_{n=1}^N (y_n - X_nw)^2 \\
   \Leftrightarrow & \sigma^2_{MLE} = \frac{1}{N} \sum_{n=1}^N (y_n - X_nw)^2 \\
  \end{align*}
$$

i.e. $$\sigma^2$$ is the mean of the squared distances between the observations and the noise-free values.

## Maximum-a-Posteriori (MAP)

Maximum likelihood without a regularizer is prone to overfitting (details in section 9.2.2 of the [Mathematics for Machine Learning book]({{ site.resources_permalink }})). Overfitting leads to very large parameter values. To mitigate this, a prior can be placed on the parameter space. The goal is then to find the parameters that maximize the posterior distribution.

Prior knowledge about the distribution of the parameters is included by multiplying the likelihood by an additional term: a prior probability distribution on the parameters. Given a prior, how should the distribution of the parameters be updated after observing some data $$x$$? In other words, how should the more specific knowledge of the parameters, gained after observing data $$x$$, be represented?

This is the Maximum-a-Posteriori (MAP) estimation, obtained by applying Bayes Theorem.

The problem is to find the parameters of the distribution that best represent the data. Adapting equation \ref{eq_prior_AB} to regression, the goal is to compute:

$$
p(w\mid X, y) = \frac{p(y\mid X, w) p(w)}{p(y \mid X)} \propto p(y\mid X, w) p(w)
\label{posterior_w}
$$

The computation follows the same log-trick as in the MLE case. The *log-posterior* is:

$$
log \,\, p(w\mid X, y) = log \,\, p(y \mid X, w) + log \,\, p(w) + const
$$

This is the sum of the log-likelihood $$ log \,\, p(y \mid X, w)$$ and the *log-prior* $$log \,\, p(w)$$, so the MAP estimate is a *compromise* between the prior and the likelihood. An alternative view is regularization: the prior adds a term that biases the parameters to be close to the origin (MML book section 8.3.2). As with the MLE, the derivative of the negative log-posterior with respect to $$w$$ is:

$$
  \begin{align*}
   & - \frac{d \,\,  log \,\, p( w \mid X,y)}{dw} \\
 = & - \frac{d \,\,  log \,\, p( y \mid X,w)}{dw} - \frac{d \,\,  log \,\, p(w)}{dw} \\ 
 = & \frac{d}{dw} \left( \frac{1}{2 \sigma^2}(y-Xw)^T(y-Xw) \right) + \frac{d}{dw} \left( \frac{1}{2b^2}w^Tw \right) \\
   & \qquad \text{(Linear Reg. likelihood (eq \ref{eq_lr_likelihood}), and prior } \mathcal{N}(0, b^2 I) \text{)} \\
 = & \frac{1}{\sigma^2}(w^TX^TX-y^TX) + \frac{1}{b^2}w^T \\
   & \qquad \text{(First-order derivatives, rule 5)} \\
  \end{align*}
\label{eq_prior_w}
$$

The minimum is found by setting the derivative to zero:

$$
\begin{align*}
 & \frac{1}{\sigma^2}(w^TX^TX-y^TX) + \frac{1}{b^2}w^T = 0 \\
\Leftrightarrow & w^T \left( \frac{1}{\sigma^2}X^TX + \frac{1}{b^2}I \right) - \frac{1}{\sigma^2}y^TX = 0\\
\Leftrightarrow & w^T \left( X^TX + \frac{\sigma^2}{b^2}I \right) = y^TX \\
\Leftrightarrow & w^T = y^TX \left( X^TX + \frac{\sigma^2}{b^2}I \right)^{-1} \\
\Leftrightarrow & w_{MAP} = \left( X^TX + \frac{\sigma^2}{b^2}I \right)^{-1} X^Ty\\   
\end{align*}
$$

The only difference between the weights estimated with MAP ($$w_{MAP}$$) and with MLE ($$w_{MLE}$$) is the additional term $$\frac{\sigma^2}{b^2}I$$ in the inverse matrix, which acts as a regularizer.

## Closed-form solution for Bayesian Linear Regression

Instead of computing a point estimate via MLE or MAP, a special case of Bayesian inference is linear regression with normal priors and posterior. In this case, *the posterior has an analytical solution*. This approach is very common, mainly because the result is not an estimate, and the analytical solution for the posteriors is *extremely fast* to compute, even for very large datasets and dimensionality.

Bayesian linear regression is a type of conditional modeling in which the mean of one variable is described by a linear combination of other variables. The goal is to obtain the posterior probability of the regression coefficients (as well as other parameters describing the distribution of the regressand), and ultimately to allow the out-of-sample prediction of the regressand (often labelled $$y$$) conditional on observed values of the regressors (source: [Wikipedia page for Bayesian Linear Regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression)). In other words, Bayesian linear regression takes the idea of the parameter prior one step further. It does not compute a point estimate of the parameters. Instead, the full posterior distribution over the parameters is used when making predictions. No parameters are fitted: predictions are a mean over all plausible parameter settings (according to the posterior).

All weights are assumed to be drawn from a Gaussian distribution. They are independent if the covariance matrix is diagonal, and dependent otherwise. The starting point is the prior $$p(w) \thicksim \mathcal{N}(m_0, S_0)$$, with mean vector $$m_0$$ and (positive definite) covariance matrix $$S_0$$ (following the variable notation of [Chris Bishop's PRML book]({{ site.resources_permalink }})).

The prior in multivariate notation is:

$$
p(w \mid m_0, S_0 ) = (2 \pi)^{-\frac{D}{2}} \,\, \det(S_0)^{- \frac{1}{2}} \,\, \exp \left( - \frac{1}{2}(w - m_0)^T S_0^{-1} (w-m_0) \right)
$$ 

The likelihood is:

$$
p(y \mid X, w) = \frac{1}{(2 \pi \sigma^2)^{N/2}} \exp \left(  -\frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) \right)
$$

Replacing these terms in Equation \ref{posterior_w} gives:

$$
\begin{align*}
p(w \mid X, y) & = \frac{p(y\mid X, w) p(w)}{p(y \mid X )} \\
                & \propto p(y\mid X, w) p(w) \\
                  & \qquad \text{(dropped constant term in division)} \\
                & \propto \exp \left(  -\frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) - \frac{1}{2}(w-m_0)^TS_0^{-1}(w-m_0) \right) \\
                  & \qquad \text{(dropped constant in multiplication)} \\
                & = \exp \left( -\frac{1}{2} \left( \sigma^{-2} (y-Xw)^T(y-Xw) + (w-m_0)^TS_0^{-1}(w-m_0) \right) \right)\\
\end{align*}
$$

Applying the log-trick and factorizing the expression isolates the quadratic, linear and independent terms in $$w$$:

$$
  \begin{align}
   & log \,\, p(w \mid X, y)  \\
 = & -\frac{1}{2} (\sigma^{-2}(y-Xw)^T(y-Xw) + (w-m_0)^TS_0^{-1}(w-m_0)) + const \\
   & \qquad \text{(ignore const due to zero-derivative)} \\
 = & - \frac{1}{2} \left( \sigma^{-2} y^T y - 2\sigma^{-2} y^T Xw + \sigma^{-2} w^T X^T Xw + w^TS_0^{-1}w - 2 m_0^T S_0^{-1} w + m_0^T S_0^{-1} m_0 \right) \\
   \label{eq1_sq}
 = & - \frac{1}{2} \left( w^T ( \sigma^{-2} X^TX + S_0^{-1}) w \right)   & \hspace{2cm}\text{(terms quadratic in } w \text{)} \\
   \label{eq1_lin}
   & + \left( \sigma^{-2} X^Ty + S_0^{-1} m_0 \right)^T w               &  \hspace{2cm}\text{(terms linear in } w \text{)} \\
   & - \frac{1}{2} \left( \sigma^{-2} y^T y + m_0^T S_0^{-1} m_0 \right) & \hspace{2cm}\text{(const terms independent of } w \text{)} \\
  \end{align}
$$

As a side note, the last equation is quadratic in $$w$$. Since the unnormalized log-posterior is a negative quadratic, *the posterior is Gaussian*, i.e.:

$$
  \begin{align*}
 p(w\mid X, y) = & \exp ( log \,\, p(w\mid y, X) ) \\
\propto & \exp ( log \,\, p(y\mid X, w) + log \,\, p(w) ) \\
  & \qquad \text{(Bayes equation)} \\
\propto & \exp \left( -\frac{1}{2} w^T ( \sigma^{-2} X^TX + S_0^{-1}) w + \left( \sigma^{-2} X^Ty + S_0^{-1} m_0 \right)^T w \right) \\
  & \qquad \text{(quadratic and linear terms of } log \,\, p(w \mid X, y) \text{)} \\
  \end{align*}
$$

where only the linear and quadratic terms in $$w$$ are kept. The $$const$$ terms are ignored, as their derivative is zero and they do not change the proportionality.

Back to the main problem: this unnormalized Gaussian must now be brought into a form proportional to $$ \mathcal{N} ( w \mid m_N, S_N )$$. The method of [completing the square](https://en.wikipedia.org/wiki/Completing_the_square) gives the values of $$m_N$$ and $$S_N$$. The target log-posterior is:

$$
  \begin{align}
  log \,\, \mathcal{N} ( w \mid m_N, S_N ) = & -\frac{1}{2}(w - m_N)^T S_N^{-1}(w-m_N) + const    \\ 
   \label{eq2_sq}
 = & -\frac{1}{2} \left(w^T S_N^{-1}w \right)   & \hspace{2cm}\text{(terms quadratic in } w \text{)} \\
   \label{eq2_lin}
   & + m^T_N S^{-1}_N w                         &  \hspace{2cm}\text{(terms linear in } w \text{)} \\
   & - \frac{1}{2} \left( m_N^T S_N^{-1} m_N \right) & \hspace{2cm}\text{(const terms independent of } w \text{)} \\
  \end{align}
$$

Matching the quadratic (\ref{eq1_sq}, \ref{eq2_sq}) and linear (\ref{eq1_lin}, \ref{eq2_lin}) terms of the computed and target equations gives $$S_N$$ and $$m_N$$:

$$
 \begin{align*}
                   & w^T S_N^{-1} w = \sigma^{-2} w^TX^TXw+w^TS_0^{-1} w \\
  \Leftrightarrow  & w^T S_N^{-1} w = w^T ( \sigma^{-2} X^TX+ S^{-1}_0) w \\
  \Leftrightarrow  & S_N^{-1} = \sigma^{-2} X^T X + S^{-1}_0 \\
  \Leftrightarrow  & S_N = ( \sigma^{-2} X^TX + S^{-1}_0)^{-1} \\
 \end{align*}
$$

and

$$
 \begin{align*}
                   & m_N^T S_N^{-1}w = (\sigma^{-2}X^Ty + S_0^{-1} m_0)^T w \\
  \Leftrightarrow  & m_N^T S_N^{-1} = (\sigma^{-2}X^Ty + S_0^{-1} m_0)^T \\
  \Leftrightarrow  & m_N = S_N (\sigma^{-2}X^Ty + S_0^{-1} m_0) \\
 \end{align*}
$$

in line with equations 3.50 and 3.51 of [Chris Bishop's PRML book]({{ site.resources_permalink }}).

## Online learning

Online learning updates the posterior continuously as new data is observed. Following the $$posterior \propto likelihood * prior$$ principle, the posterior of each iteration becomes the prior of the next one, i.e. the new initial knowledge is what was learnt previously. The main advantage is that training does not require all the data at once, so the model can learn from datasets that are not fully available at a given time. The principle is illustrated below:

{: .picture}
<img width="45%" height="35%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_posterior_1.svg"/>

{: .picture}
<img width="45%" height="35%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_posterior_3.svg"/>

{: .picture}
<img width="45%" height="35%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_posterior_4.svg"/>

{: .picture}
<img width="45%" height="35%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_posterior_5.svg"/>

{: .caption}
Illustration of four steps of online learning for the linear model $$y = w_0 + w_1x$$.
(inspired by fig 3.7, <a href="{{ site.resources_permalink }}">Pattern Recognition and Machine Learning, Chris Bishop</a>)


The initial prior assumes that both weights ($$w_0$$ and $$w_1$$) are zero-centered, i.e. with mean 0 and a standard deviation of 1. When a new datapoint is introduced (blue circle on the top-right plot), the new posterior (top-left) is computed from the likelihood and the initial prior. Models drawn from the current posterior give an inaccurate regression (orange lines on the top-right plot). Another point is then introduced (2nd row, right), leading to a new posterior (second row, left), computed from the likelihood and the prior (i.e. the posterior of the previous iteration). The model is now more accurate, and the search space (covariance of the weights) is smaller (second row, left). After two more iterations, the model is already very accurate, with a reduced variance on the weights (bottom left) and better-tuned linear regression models (bottom right).

## Predictive Variance

Recall that the goal is to model a noisy output defined by $$y = X^Tw + \varepsilon$$, where $$X$$ is a single input (column) vector, with model parameters $$w \thicksim \mathcal{N}(\mu, \Sigma)$$ and noise parameter $$\varepsilon \thicksim \mathcal{N}(0, \sigma^2)$$. An earlier section gave the closed-form solution for the distribution of $$w$$. This section computes the distribution of $$y$$. For a fixed $$w$$, the likelihood $$p(y \mid X, w, \sigma^2)$$ is the distribution of $$y = X^Tw + \varepsilon$$. With $$w$$ drawn from its posterior $$\mathcal{N}(m_N, S_N)$$, $$y$$ follows the predictive distribution, whose mean and variance are computed below.

The [expectation](https://en.wikipedia.org/wiki/Expected_value) of $$y$$ is:

$$
 \begin{align*}
  \mathbf{E} \left[ y \right] & = \mathbf{E} \left[ X^Tw + \varepsilon \right] \\
                              & = X^T \mathbf{E} \left[ w \right] + \mathbf{E} \left[ \varepsilon \right] \\
                              & = X^T m_N \\
 \end{align*}
$$

i.e. because the noise has zero mean, the result equals a linear regression with the weights set to the closed-form solution of their mean value. The [variance](https://en.wikipedia.org/wiki/Variance) of $$y$$ follows analogously (the Variance rules are listed in the appendix at the end of the post):

$$
  \begin{align*}
  \mathbf{Var} \left[ y \right] & = \mathbf{Var} \left[ X^Tw + \varepsilon \right] \\
                                & = \mathbf{Var} \left[ X^Tw  \right] + \mathbf{Var} \left[ \varepsilon \right] + 2\mathbf{Cov} \left[ X^Tw, \varepsilon \right] \\
                                & = \mathbf{Var} \left[ X^Tw  \right] + \mathbf{Var} \left[ \varepsilon \right] \\
                                  & \qquad \text{(independent variables)} \\
                                & = X^T \mathbf{Var} \left[ w  \right] X + \mathbf{Var} \left[ \varepsilon \right] \\
                                  & \qquad \text{(Var linear transformation, rule 6)} \\
                                & = X^T S_N X + \sigma^2 & ( \varepsilon \thicksim \mathcal{N}(0, \sigma^2 ) \text{ and noise variance } \sigma^2 \text{)}\\
 \end{align*}
$$

As in the previous visualization, introducing new datapoints improves the accuracy of the model:


{: .picture}
<img width="35%" height="35%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_predictive_distribution_0.svg"/>

{: .picture}
<img width="35%" height="35%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_predictive_distribution_1.svg"/>

{: .picture}
<img width="35%" height="35%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_predictive_distribution_2.svg"/>

{: .picture}
<img width="35%" height="35%" src="{{ site.assets }}/bayesian-linear-regression/linear_bayesian_predictive_distribution_3.svg"/>

{: .caption}
Illustration of four steps modelling synthetic sinusoidal data. Each plot shows the ground truth in blue, the model mean approximation in orange, and the model mean plus and minus one predictive standard deviation as a light-orange area. (inspired by fig 3.8, <a href="{{ site.resources_permalink }}">Pattern Recognition and Machine Learning, Chris Bishop</a>)


## Comparing two Bayesian models

Two probabilistic models $$M_1,M_2$$ are compared on the same dataset $$D$$ through the ratio of their posteriors, a.k.a. the **posterior odds**:

$$
\frac{p(M_1 \mid D)}{p(M_2 \mid D)} = \frac{ \frac{p(D \mid M_1) \, p(M_1)}{p(D)} }{ \frac{p(D \mid M_2) \, p(M_2)}{p(D)} } =  \frac{p(D \mid M_1) \, p(M_1)}{p(D \mid M_2) \, p(M_2)} 
$$

If both models have the same prior, the equation simplifies to $$\frac{p(D \mid M_1)}{p(D \mid M_2)}$$. If this ratio is greater than $$1$$, model $$M_1$$ is chosen, otherwise model $$M_2$$ is chosen.

## Final Remarks

More advanced topics on Bayesian Linear Regression are covered in chapter 3 of the Pattern Recognition and Machine Learning book by Chris Bishop and in section 9.3 of Mathematics for Machine Learning, both available on the [resources page]({{ site.resources_permalink }}). The source code of the closed-form solutions and of the examples above is available in the <a href="{{ site.assets }}/bayesian-linear-regression/bayesian_linear_regression.ipynb">Bayesian Linear Regression notebook</a>.

Finally, it has been shown that a kernelized Bayesian Linear Regression with the kernel $$K(x, x′)=x^Tx′$$ is equivalent to a [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process).
 
### Misc: linear models of the exponential family

For a setup where the output is non-Gaussian but belongs to the exponential family, see [Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model).




---

### Appendix: refresher on Linear Algebra
<div id="appendix-lin-alg" />

The algebraic rules used in this document are listed below. A more detailed list of matrix operations is available in <a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>.

1. **Multiplication:**
	1. $$A(BC) = (AB)C$$;
	2. $$A(B+C)=AB+AC$$;
	3. $$(B+C)A=BA+CA$$;
	4. $$r(AB)=(rA)B=A(rB)$$ for a scalar $$r$$;
	5. $$I_mA=A=AI_n$$;
	6. $$(y - Xw)^2 = (y-Xw)^T(y-Xw)$$.

2. **Transpose:**
	1. $$(A^T)^T = A$$;
	2. $$(A+B)^T=A^T+B^T$$;
	3. $$(rA)^T = rA^T$$ for a scalar $$r$$;
	4. $$(AB)^T=B^TA^T$$;

3. **Division:**
	1. if $$rA=B$$ and $$A$$ is invertible, then $$r=BA^{-1}$$;
	2. if $$Ar=B$$ and $$A$$ is invertible, then $$r=A^{-1}B$$;
	3. $$Ax=b$$ is the system of linear equations $$a_{1,1}x_1 + a_{1,2}x_2 + ... + a_{1,n}x_n = b_1$$ for row $$1$$, repeated for every row;
		- therefore, $$x = A^{-1}b$$, if matrix $$A$$ has an inverse;

4. **Inverse:** 
	1. $$AA^{-1}=A^{-1}A=I$$;
	2. If $$A$$ is invertible, its inverse is unique;
	3. If $$A$$ is invertible, then $$Ax=b$$ has a unique solution;
	4. If $$A$$ is invertible, $$(A^{-1})^{-1}=A$$;
	5. $$rA^{-1} = (\frac{1}{r}A)^{-1}$$ for a scalar $$r$$;

5. **First Order derivatives**:
	1. $$\frac{d w^TX}{dw} = \frac{d X^Tw}{dw} = X$$; 
	2. $$\frac{d X^TwY}{dw} = XY^T$$;
	3. $$\frac{d X^Tw^TY}{dw} = YX^T$$; 
	4. $$\frac{d X^TwX}{dw} = \frac{d X^Tw^TX}{dw} = XX^T$$; 
	5. $$\frac{d \, w^TAw}{dw} = w^T(A+A^T)$$;

6. **Variance and Covariance:**
	1. $$\mathbb{Var}(X) = \mathbb{Cov}(X,X)$$;
	2. $$\mathbb{Var}(rX) = r^2 \mathbb{Var}(X)$$, for scalar $$r$$;
	3. $$\mathbb{Var}(X+Y) = \mathbb{Var}(X) + \mathbb{Var}(Y) + 2\mathbb{Cov}(X,Y)$$;
	4. $$\mathbb{Var}(X+Y) = \mathbb{Var}(X) + \mathbb{Var}(Y)$$, if X and Y are independent (i.e. zero covariance);
	5. $$\mathbb{Var}(X) = \mathbf{E}[(X-\mathbf{E}[X])^2 ] = \mathbf{E} \left[ X^2  - 2X\mathbf{E}[X] + \mathbf{E}[X]^2 \right] = \mathbf{E}[X^2] - 2\mathbf{E}[X]\mathbf{E}[X] + \mathbf{E}[X]^2 =  \mathbf{E}[X^2] -  \mathbf{E}[X]^2$$;
	6. If $$y$$ is a linear transformation of univariate form $$y=mx+b$$ or multivariate form $$y=Xw+b$$, then:
		- Variance in univariate form: $$\mathbb{Var}(y)=m^2 \,\, \mathbb{Var}(x)$$;
		- Variance in multivariate form: $$Var (Xw + b) = Var (Xw) = X \mathbb{Var}(w) X^T = X \Sigma X^T$$, where $$\Sigma$$ is the covariance matrix of $$w$$;
		- Expected Value in multivariate form: $$\mathbf{E}[Xw + b] = X \mathbf{E}[w] + b = X \mu + b$$, where $$\mu$$ is the mean of $$w$$ 
			- formulas 6.50 and 6.51 in [Mathematics for Machine Learning]({{ site.resources_permalink }})
 
