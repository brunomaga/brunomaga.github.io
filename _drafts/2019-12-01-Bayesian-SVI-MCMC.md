---
layout: post
title:  "[draft] Bayesian Optimization: Variational Inference and Monte-Carlo methods"
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---


## Stochastic Variational Inference for approximate posterior

advantage: faster than sampling methods; allows non-linear regression methods;
disadvantages: over-confident, works well only if we know the parametric distribution of the posterior, requires the posterior to follow a parametric distribution, otherwise we can use the sampling.

## Monte Carlo sampling for exact posteriors 

advantages: *real* posterior.
disadvantages: very high computation cost, what to do with an exact posterior which doesnt follow a parametric ditribution.

## Deep Neural Net

https://arxiv.org/abs/1906.09686
https://arxiv.org/abs/2002.02405

