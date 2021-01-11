---
layout: post
title:  "Expectation-Maximization and Gaussian Mixture Models"
categories: [machine learning, unsupervised learning, probabilistic programming]
tags: [machinelearning]
---

TODO this post is currently being written

<p align="center"><img width="30%" height="30%" src="/assets/Expectation-Maximization-GMM/univariate-gaussian-mixture.png"><br/><small>A distribution modelled by 3 univariate gaussian distributions</small></p>


# From wikipedia

The EM algorithm is used to find (local) maximum likelihood parameters of a statistical model in cases where the equations cannot be solved directly. Typically these models involve latent variables in addition to unknown parameters and known data observations.

That is, either missing values exist among the data, or the model can be formulated more simply by assuming the existence of further unobserved data points. For example, a mixture model can be described more simply by assuming that each observed data point has a corresponding unobserved data point, or latent variable, specifying the mixture component to which each data point belongs.

Finding a maximum likelihood solution typically requires taking the derivatives of the likelihood function with respect to all the unknown values, the parameters and the latent variables, and simultaneously solving the resulting equations. **In statistical models with latent variables, this is usually impossible**. Instead, the result is typically a set of interlocking equations in which the solution to the parameters requires the values of the latent variables and vice versa, but substituting one set of equations into the other produces an unsolvable equation.


The EM algorithm proceeds from the observation that there is a way to solve these two sets of equations numerically. One can simply pick arbitrary values for one of the two sets of unknowns, use them to estimate the second set, then use these new values to find a better estimate of the first set, and then keep alternating between the two until the resulting values both converge to fixed points.



# Variational Bayes vs Expectation Maximization

Variational Bayes (VB) is often compared with expectation maximization (EM). The actual numerical procedure is quite similar, in that both are alternating iterative procedures that successively converge on optimum parameter values. The initial steps to derive the respective procedures are also vaguely similar, both starting out with formulas for probability densities and both involving significant amounts of mathematical manipulations.

However, there are a number of differences. Most important is what is being computed.

EM computes point estimates of posterior distribution of those random variables that can be categorized as "parameters", but only estimates of the actual posterior distributions of the latent variables (at least in "soft EM", and often only when the latent variables are discrete). The point estimates computed are the modes of these parameters; no other information is available.
VB, on the other hand, computes estimates of the actual posterior distribution of all variables, both parameters and latent variables. When point estimates need to be derived, generally the mean is used rather than the mode, as is normal in Bayesian inference. Concomitant with this, the parameters computed in VB do not have the same significance as those in EM. EM computes optimum values of the parameters of the Bayes network itself. VB computes optimum values of the parameters of the distributions used to approximate the parameters and latent variables of the Bayes network. For example, a typical Gaussian mixture model will have parameters for the mean and variance of each of the mixture components. EM would directly estimate optimum values for these parameters. VB, however, would first fit a distribution to these parameters — typically in the form of a prior distribution, e.g. a normal-scaled inverse gamma distribution — and would then compute values for the parameters of this prior distribution, i.e. essentially hyperparameters. In this case, VB would compute optimum estimates of the four parameters of the normal-scaled inverse gamma distribution that describes the joint distribution of the mean and variance of the component.

# more notes

EM is qeuivalent to VB under the constraint that the approxiamte posterior for $\theta$ is a point mass. See proof [here](https://stats.stackexchange.com/questions/105661/relation-between-variational-bayes-and-em)

Therefore, I read somewhere that Variational Bayes method is a generalization of the EM algorithm. 

# more notes

However, in contrast to the EM algorithm which only gives you a point estimate, it is always better for Bayesians if, hopefully, the whole posterior distribution is available. This is different from just obtaining one point estimate because then, you don’t have any measure of uncertainty that your estimate conveys with it. This is where the variational Bayes (or variational inference, variational approximations) kicks in.

The difference of EM and VB is the kind of results they provide, EM is just a point, VB is a distribution. However, they also have similarities. EM and VB can both be interpreted as minimizing some sort of distance between the true value and our estimate, which is the Kullback-Leibler divergence.

So EM and VB are not really distinguished as to how complex they’re used for is, but rather what kind of result it returns in the end.

# more notes

Variational Inference is used for more complex models. Here is how I think about them.

1 - For the simplest of Gaussian models, the Maximum Likelihood estimation method yields a closed form solution, in which case the only unknowns are the parameters.

2 - When the Gaussian model involves latent variables and parameters only, Expectation Maximization is enough to solve the model.

3 - If, on top of the latent variables, the parameters becomes random with prior distributions, the Variational Inference (or Variational Bayes) method is used

