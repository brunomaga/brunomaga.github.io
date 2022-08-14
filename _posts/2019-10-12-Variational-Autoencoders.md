---
layout: post
title:  "Variational Autoencoders: a brief summary"
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---

This post is a brief summary of [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) and [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) from Diederik P. Kingma, Max Welling at the Uversiteit van Amsterdam. 

The papers introduces Variational Autoencoders, aiming at performing efficient inference and learning in probabilistic models whose latent variables and parameters have intractable posterior distributions, and on large datasets. The SGVB (Stochastic Gradient Variational Bayes) estimator presented can be used for efficient approximation of posterior inference in almost any model with continuous latent variables, and because it’s differentiable, it can be optimized with e.g. gradient descent. The AutoEncoding Variabional Bayes (AEVB) algorithm presented uses the SGVB estimator to optimize a model that does efficient approximate posterior inference using simple sampling (from parametric distributions).

Input datapoints $$x$$ are generated as following:
- a value $$z(i)$$ is generated from some prior distribution $$p_{\theta^*}(z)$$; and
- a value $$x(i)$$ is generated from the likelihood distribution $$p_{\theta^∗}(x \mid z)$$.

The assumptions are that both the prior and likelihood functions are parametric differentiable distributions, and that $$\theta^*$$ and $$z(i)$$ are unknown. There’s also the assumption of intractability as in:
- the integral of the marginal likelihood $$\int p_\theta(z) p\theta(x \mid z) dz$$ is intractable;
- the true posterior density $$p\theta (z \mid x)$$ is intractable, so the expectation maximization algorithm cant be used; and
- the algorithms for any mean-field approximation algorithm are also intractable.
This is the level of intractability common to DNNs with nonlinear hidden layers.

The paper propose solutions to an efficient approximation of MAP estimation of parameters $$\theta$$, of the posterior inference of the latent variable $$z$$ and the marginal inference of the variable $$x$$. Similarly to other variational inference methods, the intractable true posterior $$p_{\theta}(z \mid x)$$ is approximated by $$q_\phi(z  \mid  x)$$ (the Encoder), whose parameters $$\phi$$ are not computed by a closed-form expectation but by the Encoder DNN instead. $$p_\theta(x \mid z)$$ is the Decoder, that given a $$z$$ will produce/generate the output which is a distribution over the possible values of x. Given a datapoint $$x$$ the encoder produces produces a distribution (e.g. a Gaussian) over the possible values of the code $$z$$ from which the datapoint $$x$$ could have been generated.

### A VAE as two Encoder-Decoder DNNs

The VAE proposed includes a DNN decoder, a DNN decoder, with parameters $$\theta$$ and $$\phi$$ optimized together with the AEVB algorithm, where $$p_\theta(x \mid z)$$ is a Gaussian/Bernoulli with distribution parameters computed from $$z$$. Therefore the VAE can be viewed as two coupled, *independent* parameterized models: the encoder/recognition models, and the decoder/generative model (trained together), where the encoder delivers to the decoder an approximation to its posterior over latente random variables.

The difference between Variational Autoencoders and regular Autoencoders (such as seq-to-seq encoder-decoder) is that:
- in regular AEs the latent representation of $$z$$ is both the output of the encoder and input of the decoder;
- in VAEs the output of the decoder are the parameters of the distributions of $$z$$ -- e.g. mean and variance in case of a Gaussian distribution -- and $$z$$ is drawn from those parameters and then passed to the decoder;

In practice, they can be vizualised  as:
 
<p align="center"><img width="95%" height="95%" src="/assets/Variational-Autoencoders/vae.png"/><br/>
<small>VAE vs AE structure (image credit: <a href="https://data-science-blog.com/blog/2022/04/19/variational-autoencoders/">Variational Autoencoders, Data Science Blog, by Sunil Yadav</a>))</small></p>

### Why VAE instead of Variational Inference

One advantage of the VAE framework, relative to ordinary Variational Inference, is that the encoder is now a (stochastic) function of the input variables, in contrast to VI where each data-case has a separate variational distribution, which *is inefficient for large datasets*.

### The parametrization Trick

The authors noticed that the sampling induces sampling noise in the gradients required for learning (or that because $$z$$ is randomly generated and cannot be backpropagated), and to can counteract that variance they use the “reparameterization trick”.

It goes as follows: the sample vector $$z$$ that is typically sampled from the mean vector $$\mu$$ and variance $$\sigma$$ in the Gaussian scenario in now described as $$ z = \mu + \sigma \cdot \epsilon$$ where $$\epsilon$$ is always the standard gaussian ie $$\epsilon ~ N(0,1)$$.

<p align="center"><img width="70%" height="70%" src="/assets/Variational-Autoencoders/vae2.png"/></p>


### The loss function

The loss function is a sum of two terms:

<p align="center"><img width="70%" height="70%" src="/assets/Variational-Autoencoders/vae_loss.png"/></p>

The first term is the reconstruction loss (or expected negative log-likelihood of the i-th datapoint). The expectation is taken with respect to the encoder’s distribution over the representations, encouraging the VAE to generate valid datapoints. This loss compares the model output with the model input and can be the losses we used in the autoencoders, such as L2 loss.

The second term is the Kullback-Leibler divergence between the encoder’s distribution qθ(z∣x)q and p(z). This divergence measures how much information is lost (in units of nats) when using q to represent p. It is one measure of how close q is to p.

### Results

The paper tested the VAE on the MNIST and Frey Face datasets and compared the variational lower bound and estimated marginal likelihood, demonstrating improved results over the wake-sleep algorithm.

