---
layout: post
title:  "[draft] Bayesian Regression"
date:   2019-03-12 12:01:42 +0100
categories: [machine learning]
tags: [machinelearning]
---

Regular regression models, particularly linear regression, have a major problem: overfitting. An alternative would be to use Maxiumum a porteriori (MAP), however we have no representation of our uncertainty.

To that extent, the main advantadge of Bayesian regression is that provides a measure of uncertainty or "how sure we are", based on the seen data. Take the following example of few observations of $x$ in a linear space, plotted in pink, with a yellow line represeting a linear regression of the data:
  
<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Regression/linear_bayesian_1.png">
<p align="center">

The value of $x$ can be estimated by the value in the regression model represented by a blue cross. However, because it is so different from other observations, we may not be certain of how accurate is our prediction. The Bayesian model helps in this decision by computing the uncertainty (or error) of our decision, as plotted in green:

<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Regression/linear_bayesian_2.png">
</p>

In practice, Bayesian uses decision theory to *optimize the loss function*. That is possible because Bayesian gives us what we need to optimize the loss function: the *predictive distribution* $p( y | x, D)$. 

Here is the setup: 
- Data is given by $D = ((x_1, y_1), ..., (x_n, y_n)), x+i \in R^d, y_i \in R$;
- $y$ are conditionally independent given $w$, where $y_i \approx \mathcal{N}(w^T x_i, a^{-1})$ for normal distribution $\mathcal{N}$ and $a$ being the precision $a = 1/ \delta^2$;
- $w$ is a multivariate normal with $w \approx \mathcal{N}(0, b^{-1} I)$, the mean given by zero-vector $0$, covariance matrix $b^{-1} I$, and identity matrix $I$. $b>0$ is also the precision value.
- $a$ and $b$ are known, so the only unknown is $w$.

