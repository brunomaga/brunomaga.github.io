---
layout: post
title:  "Bayesian Optimization"
date:   2019-10-12 01:00:00 +0100
categories: [machine learning, probabilistic programming]
#tags: [machinelearning]
---

Bayesian methods allows us to perform modelling of an input to an output by providing a measure of *uncertainty* or "how sure we are", based on the seen data. Unlike most *frequentist* methods commonly used, where the outpt of the method is a set of best fit parameters, the output of a Bayesian regression is a probability distribution of each model parameter, called the **posterior distribution**. 
Unlike most *frequentist* methods commonly used, where the outpt of the method is a set of best fit parameters (e.g. the values of the weight on linear regression), the output of a Bayesian regression is a probability distribution of each model parameter, called the *posterior distribution*. 

For the sake of comparison, take the example of the simple linear regression $y = mx + b$. On the frequentist approach, one tries to find the constant that define the slope $m$ and bias $b$ values, with $m \in \mathbb{R}$ and $b \in \mathbb{R}$, using optimization methods. On the other hand, the Bayesian approach would also compute $y = mx + b$, however, $b$ and $m$ are not fixed values but drawn from probabilities that we *tune* during training. As an example, we can assume as prior knowledge that $m$ and $b$ are independent and normally distributed --- i.e. $b \thicksim \mathcal{N}(\mu, \sigma^2)$ and $w \thicksim \mathcal{N}(\mu, \sigma^2)$ --- and the parameters to be learnt would then be all $\mu$ and $\sigma$. Additionally, we can also add the model of the noise (or inverse of the precision) constant $\varepsilon$ that models as well *how noisy* our data may be, by computing $y = mx + b + \varepsilon$ instead, with $\varepsilon \thicksim \mathcal{N}(0, \sigma^2)$.
 
Take the following example of few observations of $x$ in a linear space, plotted in pink, with a yellow line representing a linear regression of the data. The value of $x$ can be estimated by the value in the regression model represented by a blue cross. However, because it is so different from other observations, we may not be certain of how accurate is our prediction. The Bayesian model helps in this decision by computing the uncertainty (or error) of our decision, as plotted in green:

<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Optimization/linear_bayesian_2.png">
</p>

Apart from the uncertainty quantification, another benefit of Bayesian is the possibility of **online learning**, i.e. a continuous update of the trained model (from previously-seen data) by looking at only the new data. This is a handy feature for e.g. datasets that are purged periodically.

In this post we will discuss how to perform Bayesian optimization, with a particular emphasis on linear regression and normal distrutions.

### Basic concepts

From the field of probability, the **product rule** tells us that the **joint distribution** of two given events $A$ and $B$ can be written as the product of the distribution of $a$ and the **conditional distribution** of $B$ given a value of $A$, i.e: $P(A, B) = P(A) P(B\|A)$. By symmetry we have that $P(B,A) = P(B) P(A\|B)$. By equating both right hand sides of the equations and re-arranging the terms we obtain the **Bayes Theorem**:

\begin{equation}
P (A\|B) = \frac{P(B\|A) P(A)}{P(B)} \propto P(B\|A) P(A)
\label{eq_prior_AB}
\end{equation}

This equation is commonly read as "the **posterior** $P(A\| B)$ is proportional to the product of the **prior** $P(A)$ and the **likelihood** $P(B\|A)$" -- note that we dropped the **normalizer** term $P(B)$ as it is constant, making the left-hand term proportional to $P (A\|B)$.
 
The prior distribution $P(A)$ is a shorthand for $P(A \| I)$ where $I$ is all information we have before start collecting data. If we have no information about the parameters then $P(A\|I)$ is a constant --- called an *uninformative prior* or *objective prior* --- and the posterior equals the likelihood function. Otherwise, we call it a *substantive/informative* prior.

The posterior distribution describes how much the data has changed our *prior* beliefs. An important theoream called the *Bernstein-von Mises Theorem* states that:
- for a sufficiently large sample size, the posterior distribution becomes independent of the prior (as long as the prior is neither 0 or 1)
  - ie for a sufficiently large dataset, the prior just doesnt matter much anymore as the current data has enough information;
  - in other others, if we let the datapoints go to infitiny, the posterior distribution will go to normal distribution, where the mean will be the maximum likelihood estimator;
    - this is a restatement of the central limit theorem, where the posterior distribution becomes the likelihood function;
  - ie the effect of the prior decreases as the data increases;


### Probability Distributions and the Exponential Family 

A [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. Examples:
- Gaussian distribution, for an input $y$:
  $$p(y | \mu, \sigma^2 ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } exp [ - \frac{(y - \mu)^2}{2 \sigma^2} ]$$ for a distribution with mean $\mu$ and standard deviation $\sigma$, or
  $$p(y | \mu, \Sigma ) = \frac{1}{ \sqrt{ (2 \pi)^D det(\Sigma)} } exp [ - \frac{1}{2}(y - \mu)^T \Sigma^{-1} (y-\mu)]$$ with means-vector $\mu$ and covariance matrix $\Sigma$ on its matrix form

- Laplace:
 $$ p( y_n | x_n, w ) = \frac{1}{2b} e^{-\frac{1}{b} | y_n - X_n^T w | } $$

- TODO add more distributions
- TODO add explanation of exponential family
- TODO add Least Square solution to the Unsupervised Learning notebook
- TOOD add analytical solution to Mean Square Error:
  - $argmin E_2$ = $(y - Xw)^2 = (y-Xw)^T(y-Xw)$ , which (when equal to zero) has a closed for solution of $w = (X^TX)^{-1} X^Ty$
  - ie minimizing $E_2$ is equivalent to determining  the most likely $w$ under the assumption that $y$ contains gaussian noise  i.e. $y = mx + b + \varepsilon$ instead, with $\varepsilon \thicksim \mathcal{N}(0, \sigma^2)$
- TODO add Adams optimizer to Unsupervised Learning notebook
- TODO remove Date and Categories from final URL (use tags instead?)


### Maximum Likelihood (MLE) and Maximum-a-Posteriori (MAP)

The problem in hand is to find the parameters of the distribution that best represent the data. Adapting the previous equation \ref{eq_prior_AB} of the prior to the problem at hand, we aim at computing:

$$
P (w \| y, X, \sigma^2) = \frac{P(y\|w, X, \sigma^2) P(w)}{P(y, X, \sigma^2)} \propto P(y\|w, X, \sigma^2) P(w)
\label{eq_prior_w}
$$

for a given input set $X$, with labels $y$, and model parameters $\sigma$.

There are two main optimization problems that we discuss commonly on Bayesian methods:
- When we try to find *how likely is that an output $y$ belongs to a model defined by $X$, $w$ and $\sigma$*, or **maximize the likelihood $P(y\|w, X, \sigma^2)$**, we perform a Maximum Likelihood Estimator (MLE);
- When we try to maximize the posterior, or the probability of the model parameters $w$ given the model $X$, $y$ and $\sigma$, or **maximize the posterior $P (w \| y, X, \sigma^2)$**, we perform a Maximum-a-Posteriori (MAP); 


To compute that, we perform the *log-trick* and place the term to optimize into a log function. We can do this because $log$ is a monotonically-increasing function, thus applying it to any function won't change the input values where the minimum or maximum of the solution is found (ie where gradient is zero). Moreover, since most distributions are part of the **exponential family** of distributions, they can all be represented as an exponential, and applying the log will bring the power term *out* of the exponential, and make it computationally simpler are faster. 

The **log-likelihood** is the log of the [likelihood](https://www.statisticshowto.datasciencecentral.com/likelihood-function/) of observing --- given observed data --- a parameter value  of the statistical model used to describe that data. I.e.:

$$
L_{lik} (w) = log p (y | X, w)
$$

This can be used to estimate the cost. The log-likelihood if (typically?) convex in the weight vector $w$, as it's a sum of convex functions. The **Maximum likelihood estimator (MLE)** states that:

$$
argmin_w L_{MSE}(w) = argmax_w L_{lik} (w)
$$

i.e. the solution that minimizes the weights in the Mean Square Error problem, maximizes the log-likelihood of the data. MLE is a sample approximation of the *expected* log-likelihood i.e.

$$
L_{lik}(w) \approx E_{p(x,y)} [ log p(y | x,w ) ]
$$

[coordinate-ascent]: {{ site.baseurl }}{% post_url 2018-02-17-Supervised-Learning %}

When the distribution of the prior and posterior are computationally tractable, the optimization of the parameters that define the distribution can be performed using the [coordinate ascent][coordinate-ascent] method, detailed briefly previously. In practice we perform and iterative partial derivatives of the prior/posterior for the model parameters. For the sake of comparison, while we minimize the negation of the loss on $w$ when performing linear regression, on a Bayesian model with a Gaussian distribution on all priors, we'd maximize based on the coordinate ascent of the parameters mean $\mu$ and standard deviation $\sigma$.

### Bayesian Linear Optimization

A particular exception of Bayesian optimization that requires non-iterative methods is the linear regression with normal priors and posterior. In this case, *the posterior has an analytical solution*. This approach is utilized very commonly, mainly due to two reasons:
1. defining the appropriate parametric distribution of the weights (i.e. the prior) is a *hard* problem and requires domain knowledge that many times is not easy to grasp;
2. the analytical solution for the posteriors is *extremelly fast* to compute even for very large datasets and dimensionality, compared to aternative methods that we will cover later;

Take the linear regression model $y = w_1x_1 + ... + w_dx_d$ or its alternative matrix representation $y = \sum_{d=1}^D \text{ } w_d^TX$ for a problem with dimensionlity $D$.

Going back to the previous equation \ref{eq_prior_w} 

### Stochastic Variational Inference for approximate posterior

advantage: faster than sampling methods; allows non-linear regression methods;
disadvantages: over-confident, works well only if we know the parametric distribution of the posterior, requires the posterior to follow a parametric distribution, otherwise we can use the sampling.

### Monte Carlo sampling for exact posteriors 

advantages: *real* posterior.
disadvantages: very high computation cost, what to do with an exact posterior which doesnt follow a parametric ditribution.

---

### Refresher: Linear Algebra

Here's a reminder of some operations on linear algebra, in case the previous reductions cause some confusion:

1. **Multiplication:**
	1. $A(BC) = (AB)C$;
	2. $A(B+C)=AB+AC$;
	3. $(B+C)A=BA+CA$;
	4. $r(AB)=(rA)B=A(rB)$ for a scalar $r$;
	5. $I_mA=AI=AI_n$;

2. **Transpose:**
	1. $(A^T)^T = A$;
	2. $(A+B)^T=A^T+B^T$;
	3. $(rA)^T = rA^T$ for a scalar $r$;
	4. $(AB)^T=B^TA^T$;

3. **Division:**
	1. if $rA=B$, then $r=BA^{-1}$, for a scalar $r$;
	2. if $Ar=B$, then $r=A^{-1}B$, for a scalar $r$;
	3. $Ax=b$ is the system of linear equations $a_{1,1}x_1 + a_{1,2}x_2 + ... + a_{1,n}x_n = b_1$ for row $1$, repeated for every row;
		- therefore, $x = A^{-1}b$, if matrix has $A$ an inverse;

4. **Inverse:** 
	1. $AA^{-1}=A^{-1}A=I$;
	2. If $A$ is invertible, its inverse is unique;
	3. If $A$ is invertible, then $Ax=b$ has an unique solution;
	4. If $A$ is invertible, $(A^{-1})^{-1}=A$;
	5. $rA^{-1} = (\frac{1}{r}A)^{-1}$ for a scalar $r$;
