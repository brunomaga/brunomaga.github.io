---
layout: post
title:  "Exponential Family of Distributions"
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---

### Background: Probability Distributions 

A [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. A few examples of commonly used distributions are (and an input $x$):
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

$$
log (exp(x)) = x
$$l

