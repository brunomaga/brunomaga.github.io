---
layout: post
title:  "Variational Autoencoders: a brief overview"
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---

This post is a brief summary of Auto-Encoding Variational Bayes (Variational Autoencoders), Universiteit van Amsterdam, 2013 and An Introduction to Variational Autoencoders from the same authors.

The paper introduces Variational Autoencoders, aiming at performing efficient inference and learning in probabilistic models whose latent variables and parameters have intractable posterior distributions, and on large datasets. The SGVB (Stochastic Gradient Variational Bayes) estimator presented can be used for efficient approximation of posterior inference in almost any model with continuous latent variables, and because it’s differentiable, it can be optimized with e.g. gradient descent. The AutoEncoding Variabional Bayes (AEVB) algorithm presented uses the SGVB estimator to optimize a model that does efficient approximate posterior inference using simple sampling (from parametric distributions).

Input datapoints x are generated as following:

a value z(i) is generated from some prior distribution pθ∗(z); and
a value x(i) is generated from the likelihood distribution pθ∗(x∥z).
The assumptions are that both the prior and likelihood functions are parametric differentiable distributions, and that θ∗ and z(i) are unknown. There’s also the assumption of intractability as in: the integral of the marginal likelihood ∫pθ(z)pθ(x∥z)dz is intractable,the true posterior density pθ(z∥x) is intractable — so the expectation maximization algorithm cant be used — and the algorithms for any mean-field approximation algorithm are also intractable. This is the level of intractability common to DNNs with nonlinear hidden layers.

The VAE methods propose solutions to an efficient approximation of MAP estimation of parameters θ, of the posterior inference of the latent variable z and the marginal inference of the variable x. Similarly to other variational inference methods, the intractable true posterior pθ(z∥x) is approximated by qϕ(z∥x) (the Encoder), whose parameters ϕ are not computed by a closed-form expectation but by the Encoder DNN instead.

pθ(x∥z) is the Decoder, that given a z will produce/generate the output which is a distribution over the possible values of x. Given a datapoint x the encoder produces produces a distribution (e.g. a Gaussian) over the possible values of the code z from which the datapoint x could have been generated.


VAE structure (image credit: [Data Science Blog: Variational Autoencoders, by [Sunil Yadav](https://data-science-blog.com/blog/2022/04/19/variational-autoencoders/))

Therefore the VAE can be viewed as two coupled, independent parameterized models: the encoder/recognition models, and the decoder/generative model (trained together), where the encoder delivers to the decoder an approximation to its posterior over latente random variables.

The VAE proposed in the paper includes a DNN decoder, a DNN decoder, with parameters θ and ϕ optimized together with the AEVB algorithm, where pθ(x∥z) is a Gaussian/Bernoulli with distribution parameters computed from z.

Why VAEs instead of Variational Inference?
One advantage of the VAE framework, relative to ordinary Variational Inference, is that the encoder is now a (stochastic) function of the input variables, in contrast to VI where each data-case has a separate variational distribution, which is inefficient for large datasets.

The reparametrization trick
In the paper, the authors noticed that the sampling induces sampling noise in the gradients required for learning (or that because z is randomly generated and cannot be backpropagated), and to can counteract that variance they use the “reparameterization trick”. It goes as follows: the sample vector z that is typically sampled from the mean vector μ and variance σ in the Gaussian scenario in now described as z=μ+σ⋅ϵ where ϵ is always the standard gaussian ie ϵ N(0,1).



Loss Funtion
The loss function is a sum of two terms:



The first term is the reconstruction loss (or expected negative log-likelihood of the i-th datapoint). The expectation is taken with respect to the encoder’s distribution over the representations, encouraging the VAE to generate valid datapoints. This loss compares the model output with the model input and can be the losses we used in the autoencoders, such as L2 loss.

The second term is the Kullback-Leibler divergence between the encoder’s distribution qθ(z∣x)q and p(z). This divergence measures how much information is lost (in units of nats) when using q to represent p. It is one measure of how close q is to p.

Results
The paper tested the VAE on the MNIST and Frey Face datasets and compared the variational lower bound and estimated marginal likelihood, demonstrating improved results over the wake-sleep algorithm.

