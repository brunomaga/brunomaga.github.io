---
layout: post
title:  "[draft] Bayesian Linear Regression"
date:   2019-03-12 12:01:42 +0100
categories: [machine learning]
tags: [machinelearning]
---

Regular regression models, particularly linear regression, have a major problem: overfitting. An alternative would be to use Maxiumum a porteriori (MAP), however we have no representation of our uncertainty.

To that extent, the main advantadge of Bayesian regression is that provides a measure of uncertainty or "how sure we are", based on the seen data. Take the following example of few observations of $x$ in a linear space, plotted in pink, with a yellow line represeting a linear regression of the data:
  
<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Regression/linear_bayesian_1.png">
</p>

The value of $x$ can be estimated by the value in the regression model represented by a blue cross. However, because it is so different from other observations, we may not be certain of how accurate is our prediction. The Bayesian model helps in this decision by computing the uncertainty (or error) of our decision, as plotted in green:

<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Regression/linear_bayesian_2.png">
</p>

In practice, Bayesian uses decision theory to *optimize the loss function*. That is possible because Bayesian gives us what we need to optimize the loss function: the *predictive distribution* $p( y \| x, D)$. 

### Introduction

We know that $P(A \cap B) = \frac{P(A\|B) P(B)}{P(B)}$ and $P(B \cap A) = \frac{P(B\|A)P(A)}{P(A)}$. From there it follows the *Bayes Theorem*:
\begin{equation}
P (A\|B) = \frac{P(B\|A) P(A)}{P(B)} \propto P(B\|A) P(A)
\end{equation}

Unlike most frequentist methods presented, where the outpt of the method is a set of best fit parameters, the output of a Bayesian is a probability distribution for each parameter $\theta$, called the *posterior distribution*. In practice, we need to summarize the distribution, using the mode, mean, median or range mipoint, or an equivalent *best estimate*. We also need the distribution to calculation a *credible interval* (quantiles) of each parameter.

The posterior distribution describes how much the data has changed our *prior* beliefs. An important theoream called the *Bernstein-von Mises Theorem* states that:
- for a sufficiently large sample size, the posterior distribution becomes independent of the prior (as long as the prior is neither 0 or 1)
  - ie for a sufficiently large dataset, the prior just doesnt matter much anymore as the current data has enough information;
  - in other others, if we let the datapoints go to infitiny, the posterior distribution will go to normal distribution, where the mean will be the maximum likelihood estimator;
    - this is a restatement of the central limit theorem, where the posterior distribution becomes the likelihood function;
  - ie the effect of the prior decreases as the data increases;

The prior distribution $P(\theta)$ is a shorthand for $P(\theta \| I)$ where $I$ is all information we have before start collecting data. If we have no information about the parameters then $P(\theta\|I)$ is a constant --- called an *uninformative prior* or *objective prior* --- and the posterior equals the likelihood function. Otherwise, we call it a *substantive/informative* prior.

##### Prior choice example

C consider a linear regression with $y = mx+b$. We can set our prior for the slope to favor the *Null hypothesis* that there's no relationship between $y$ and $x$: $P(b) \approx \mathcal{N} (0, \sigma^2)$, where $\sigma$ is large anough to allow for the expected range of possible slopes. The main equestion is: 
- Does the data contain enough information to push us way from that prior belief that $x$ and $y$ are unrelated?


##### Conjugate priors

For a given likelihood distribution, analytical solutions of the posterior distribution are only possible for special cases fo priors, called *conjugate priors*. The most common example, for iid normal errors, the conjugate prior for $\beta$ if normal, for for $\sigma^2$ is inverse gamma.

Usually, we need to solve Bayes' equation, numerically. The most common method is the Markov Chain Monte Carlo (MCMC) sampling, where we randomly pull random numbers for the values of the parameters on these distributions. Result is a set of points from the posterior distribution that we then summarize (by the mean, maximum, a posteriori -- MAP -- estimate of the mode). 

---

Reminder of some rules in matrices of linear algebra operations, in case the previous reductions cause some confusion:
- Multiplication: |
  - $A(BC) = (AB)C$;
  - $A(B+C)=AB+AC$;
  - $(B+C)A=BA+CA$;
  - $r(AB)=(rA)B=A(rB)$ for a scalar $r$;
  - $I_mA=AI=AI_n$;
- Transpose:
  - $(A^T)^T = A$;
  - $(A+B)^T=A^T+B^T$;
  - $(rA)^T = rA^T$ for a scalar $r$;
  - $(AB)^T=B^TA^T$;
- Division:
  - if $rA=B$, then $r=BA^{-1}$, for a scalar $r$;
  - if $Ar=B$, then $r=A^{-1}B$, for a scalar $r$;
  - $Ax=b$ is the system of linear equations $a_{1,1}x_1 + a_{1,2}x_2 + ... + a_{1,n}x_n = b_1$ for row $1$, repeated for every row.
    - therefore, $x = A^{-1}b$, if matrix has $A$ an inverse.
- Inverse: $AA^{-1}=A^{-1}A=I$;
  - If $A$ is invertible, its inverse is unique;
  - If $A$ is invertible, then $Ax=b$ has an unique solution;
  - If $A$ is invertible, $(A^{-1})^{-1}=A$;
  - $rA^{-1} = (\frac{1}{r}A)^{-1}$ for a scalar $r$;
