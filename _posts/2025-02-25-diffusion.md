---
layout: post
title:  "Diffusion models, from DDPMs to DiTs"
categories: ["ML Architectures"]
tags: ["generative models", "diffusion", "DDPM", "DiT", "video generation"]
---


[Diffusion models](https://en.wikipedia.org/wiki/Diffusion_model) are [generative models](https://en.wikipedia.org/wiki/Generative_model): like [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) and [VAEs](https://en.wikipedia.org/wiki/Variational_autoencoder), they learn to turn random noise into new samples of the data, such as images. They date back to [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) (2015), but they only gained momentum with [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM, 2020). The idea, inspired by non-equilibrium statistical physics, is to destroy the structure of the data slowly, by adding a little Gaussian noise many times over until only noise remains, and to learn how to undo this destruction, one small denoising step at a time. A diffusion model is therefore made of two [Markov chains](https://en.wikipedia.org/wiki/Markov_chain): a forward process that gradually adds noise to the data, and a **learned** reverse process that removes it. The transitions of the reverse chain are learned with [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods), and once they are learned, a sampling algorithm generates new data by running the reverse chain from pure noise.

The rest of the post derives the mathematics of DDPM and implements it with two networks, a [U-Net](https://en.wikipedia.org/wiki/U-Net) and a Transformer. It then turns to high-dimensional inputs such as videos, which video diffusion models handle with compressed latent representations and with attention across space and time.

{: .picture}
<img width="66.3%" height="66.3%" src="{{ site.assets }}/diffusion/diffusion.svg"/> 

{: .caption}
A diffusion model is a $$T$$-step Markov chain, defined by a forward process $$q$$ and a trainable reverse process $$p_\theta$$. Adapted from: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

The figure shows both chains on the image of a face. The forward process $$q$$, the dashed arrow, runs from right to left: each of its steps adds noise, until the face disappears in the pure noise of $$\mathbf{x}_T$$. The reverse process $$p_\theta$$ runs from left to right, and each of its steps removes a little noise, until a clean image $$\mathbf{x}_0$$ appears.

The derivation and the U-Net code follow DDPM, the Hugging Face post [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) and the post [Lil'Log: what are diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) of Lilian Weng. The learned variance and the noise schedule for a smaller number of steps follow [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Improved DDPM) and its [official implementation](https://github.com/openai/improved-diffusion). Equation numbers refer to DDPM unless stated otherwise, and the full code is in [diffusion.py]({{ site.assets }}/diffusion/diffusion.py) and [dit.py]({{ site.assets }}/diffusion/dit.py).

## Forward Process

The forward process $$q$$ is a Markov chain of $$T$$ steps. It starts from a data sample $$\mathbf{x}_0$$, such as an image, and produces noisier and noisier versions of it, $$\mathbf{x}_1, \dots, \mathbf{x}_T$$. Step $$t$$ adds Gaussian noise of variance $$\beta_t \in (0,1)$$, and the variances usually grow with the steps, $$\beta_1 \lt \beta_2 \lt \dots \lt \beta_T$$, so that the first steps change the data only a little. This sequence of variances is the *noise schedule*. It can be learned or, as here, fixed as a set of hyper-parameters. Each step draws the next sample from a Gaussian centered on a scaled copy of the previous one (Eq. 2):

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

In other words, $$\mathbf{x}_t$$ has the mean $$\sqrt{1-\beta_t} \mathbf{x}_{t-1}$$ and the covariance $$\beta_t\mathbf{I}$$, so every dimension receives independent noise of variance $$\beta_t$$. Sampling from this distribution is the same as drawing $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$ and setting $$\mathbf{x}_t = \sqrt{1-\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t}\epsilon$$. The factor $$\sqrt{1-\beta_t}$$, which shrinks the previous sample, keeps the scale of the data under control: if $$\mathbf{x}_{t-1}$$ has unit variance, then $$\mathbf{x}_t$$ has the variance $$(1-\beta_t) + \beta_t = 1$$ as well. Step after step, the signal fades away and the noise takes its place.

Training needs noisy samples $$\mathbf{x}_t$$ at random steps $$t$$, and running the chain up to step $$t$$ for each of them would be slow. Fortunately, the steps can be merged, because a sum of independent Gaussian variables is again a Gaussian. With $$\alpha_t = 1 - \beta_t$$ and the independent noises $$\epsilon_{t-1}, \epsilon_{t-2} \sim \mathcal{N}(0, \mathbf{I})$$, two consecutive steps give:

$$
\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1-\alpha_t} \, \epsilon_{t-1} = \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t (1-\alpha_{t-1})} \, \epsilon_{t-2} + \sqrt{1-\alpha_t} \, \epsilon_{t-1}
$$

The last two terms are independent Gaussians with the variances $$\alpha_t (1-\alpha_{t-1})$$ and $$1-\alpha_t$$, so their sum is a single Gaussian with the variance $$\alpha_t (1-\alpha_{t-1}) + 1-\alpha_t = 1 - \alpha_t \alpha_{t-1}$$. Repeating this argument down to $$\mathbf{x}_0$$ gives the distribution of $$\mathbf{x}_t$$ at any step $$t$$, directly from the data and without running the chain (Eq. 4):

$$
q (\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N} (\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

where $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$. After $$t$$ steps, the data are therefore scaled by $$\sqrt{\bar{\alpha}_t}$$, and the noise added so far has the variance $$1-\bar{\alpha}_t$$.

Eq. 4 also tells how much of the signal is left at the last step. When $$T \rightarrow \infty$$, $$\bar{\alpha}_T$$ goes to zero, and $$\mathbf{x}_T$$ becomes pure noise $$\mathcal{N}(0, \mathbf{I})$$, an [isotropic Gaussian](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic), i.e. a Gaussian whose covariance matrix is $$\sigma^2\mathbf{I}$$ for a scalar variance $$\sigma^2$$, here $$\sigma^2 = 1$$. With a finite number of steps, $$\mathbf{x}_T$$ is close to pure noise only if $$\bar{\alpha}_T \approx 0$$, and the noise schedule must ensure it. DDPM uses $$T=1000$$ steps and a $$\beta_t$$ that grows linearly from $$\beta_1=10^{-4}$$ to $$\beta_T=0.02$$, which gives $$\sqrt{\bar{\alpha}_T} \approx 0.006$$. For faster sampling, the code uses $$T=100$$ steps. With the same $$\beta_1$$ and $$\beta_T$$, however, $$\sqrt{\bar{\alpha}_T} \approx 0.6$$: the training inputs $$\mathbf{x}_T$$ would keep 60% of the amplitude of $$\mathbf{x}_0$$, while sampling starts from pure noise, an input unlike any that the model saw in training. As in the [official implementation](https://github.com/openai/improved-diffusion) of Improved DDPM, both values are therefore scaled by $$1000/T$$, which gives $$\sqrt{\bar{\alpha}_T} \approx 0.005$$. Of the [many $$\beta_t$$ schedules](https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process), the code uses the linear one, and it stores $$\alpha_t$$ and $$\bar{\alpha}_t$$ for later use:

```python
    T = 100 # the paper uses T=1000
    β_1, β_T = 0.0001 * 1000 / T, 0.02 * 1000 / T # the paper's values, scaled by 1000/T
    # arrays are 0-indexed: index i holds the value of the paper's timestep t=i+1
    β = torch.linspace(β_1, β_T, T, device=device)
    α = 1. - β
    α_cumprod = torch.cumprod(α, axis=0)
```

The function `q_sample` below samples Eq. 4 in the same way as Eq. 2, as $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$ with $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$:

```python
    @torch.no_grad()
    def q_sample(x0, t, noise=None):
        # note that in Eq 4, it samples from a non-standard gaussian distribution, but here we implement it
        # by sampling from the standard normal (0,1) and shifting and scaling it later, which is equivalent
        # Detailed in the paragraph between equations 8 and 9, and applied in algorithm 1, step 4, the epsilon.
        if noise is None:
            noise = torch.randn_like(x0) # samples from N(0,1)
        sqrt_α_cumprod = torch.sqrt(α_cumprod)
        sqrt_α_cumprod_t = sqrt_α_cumprod[t][:, None, None, None]
        sqrt_1_minus_α_cumprod = torch.sqrt(1. - α_cumprod)
        sqrt_1_minus_α_cumprod_t = sqrt_1_minus_α_cumprod[t][:, None, None, None]
        return sqrt_α_cumprod_t * x0 + sqrt_1_minus_α_cumprod_t * noise
```

The forward process has a second useful property. Running it backwards, from $$\mathbf{x}_t$$ to $$\mathbf{x}_{t-1}$$, is hard in general, because the answer depends on the whole data distribution. Once $$\mathbf{x}_0$$ is known, however, Bayes' rule combines the Gaussians of Eqs. 2 and 4 into the posterior of the forward process, which is again a Gaussian with a closed form (Eq. 6):

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0 ) = \mathcal{N} (\mathbf{x}_{t-1} ; \tilde{\mu}_t  (\mathbf{x}_t, \mathbf{x}_0), \, \tilde{\beta}_t \mathbf{I})
$$

Its variance, the **posterior variance** $$\tilde{\beta}_t$$, depends only on the noise schedule (Eq. 7):

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

The code computes it, together with its logarithm, which the DiT section uses to learn the variance. Since $$\bar{\alpha}_0 = 1$$, the first value is $$\tilde{\beta}_1 = 0$$, whose logarithm is not finite, so the log is clipped to its value at the second step, as in the official implementation:

```python
    # posterior variance, and its log clipped at index 0 (where posterior_β is 0)
    α_cumprod_prev = F.pad(α_cumprod[:-1], (1, 0), value=1.0)
    posterior_β = β * (1. - α_cumprod_prev) / (1. - α_cumprod) # Eq 7
    posterior_log_β = torch.log(torch.cat([posterior_β[1:2], posterior_β[1:]]))
```

Its mean, the **posterior mean** $$\tilde{\mu}_t$$, is a weighted sum of the clean sample $$\mathbf{x}_0$$ and the noisy sample $$\mathbf{x}_t$$ (Eq. 7):

$$
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t }{1-\bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}  \mathbf{x}_t
$$

implemented as:

```python
    @torch.no_grad()
    def posterior_μ(x_0, x_t, t):
        """ return posterior mean at step t, Equation 7 """
        α_t = α[t][:, None, None, None]
        β_t = β[t][:, None, None, None]
        α_cumprod_t = α_cumprod[t][:, None, None, None]
        α_cumprod_prev_t = α_cumprod_prev[t][:, None, None, None]
        coef1 =  β_t * torch.sqrt(α_cumprod_prev_t) / (1.0 - α_cumprod_t)
        coef2 = (1.0 - α_cumprod_prev_t) * torch.sqrt(α_t) / (1.0 - α_cumprod_t)
        return coef1 * x_0 + coef2 * x_t
```

## Reverse Process

Generating data means running the chain the other way, from noise to data. The reverse process is a Markov chain $$p_\theta(\mathbf{x}_{0:T})$$ with **learned Gaussian transitions**, which starts from pure noise, $$p(\mathbf{x}_T) = \mathcal{N} (\mathbf{x}_T ; \mathbf{0}, \mathbf{I})$$ (Eq. 1):

$$
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) )
$$

The true denoising step $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ is unknown: unlike the posterior of Eq. 6, it is not conditioned on $$\mathbf{x}_0$$, so it depends on the whole data distribution. A neural network with the parameters $$\theta$$ approximates it instead, with a Gaussian of mean $$\mu_\theta$$ and covariance $$\Sigma_\theta$$. A Gaussian is a good choice when the $$\beta_t$$ are small, because the reverse of a small Gaussian step is close to a Gaussian, and this is one reason why the forward process takes many small steps.

DDPM does not learn the covariance. It sets $$\Sigma_\theta(\mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$$, with either $$\sigma^2_t = \beta_t$$ or $$\sigma^2_t = \tilde{\beta}_t$$, and both choices gave similar results. They are two extremes: the first one is optimal for data $$\mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I})$$, and the second one for data fixed to one point. [Improved DDPM](https://arxiv.org/abs/2102.09672) later learned $$\Sigma_\theta$$, and the DiT section follows this approach.

How can the network be trained? The pair $$q$$ and $$p_\theta$$ forms a [variational auto-encoder](https://arxiv.org/abs/1312.6114), in which the forward process is a fixed encoder and the noisy samples $$\mathbf{x}_{1:T}$$ are the latent variables. As in a VAE, the likelihood of the data cannot be computed, because it integrates over all the latent variables, but it has a lower bound that can: the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO), derived in the [variational inference post]({{ site.baseurl }}{% post_url 2019-07-01-variational-inference-gmm %}#evidence-lower-bound). Training maximizes the ELBO or, equivalently, minimizes the variational bound $$L$$ on the negative log-likelihood of the data (Eq. 3):

$$
\mathbb{E} \left[ - \log p_\theta(\mathbf{x}_0) \right] \le \mathbb{E}_q \left[ - \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \right] = \mathbb{E}_q \left[ - \log p(\mathbf{x}_T) - \sum_{t \ge 1} \log \frac{p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)}{q(\mathbf{x}_t \vert \mathbf{x}_{t-1})} \right] =: L
$$

In this form, each term of the sum compares the reverse step $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ with the forward step $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$, two distributions over different variables, and the bound can only be estimated from samples, with a high variance. DDPM rewrites it in two steps. First, the forward process is a Markov chain, so conditioning a forward step on $$\mathbf{x}_0$$ as well changes nothing, $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)$$, and Bayes' rule then reverses the direction of the step:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) = \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \, q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}
$$

Second, in the sum over $$t$$, the logarithms of the ratios $$q(\mathbf{x}_t \mid \mathbf{x}_0) / q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$$ cancel out in pairs, and only $$q(\mathbf{x}_T \mid \mathbf{x}_0)$$ and $$q(\mathbf{x}_1 \mid \mathbf{x}_0)$$ remain: the first one pairs with the prior $$p(\mathbf{x}_T)$$, and the second one cancels with the forward step of $$t = 1$$. What is left is one term per step, and each KL divergence now compares two distributions over the same variable (Eq. 5):

$$
L = \mathbb{E}_q \bigg[ \underbrace{D_{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p(\mathbf{x}_T))}_{L_T} + \sum_{t > 1} \underbrace{D_{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} \bigg]
$$

The three kinds of terms have different roles. $$L_T$$ compares the end of the forward process with the starting noise of the reverse process; it has no learnable parameters, and it is a constant. $$L_0$$ is the reconstruction term of the last denoising step, from $$\mathbf{x}_1$$ to the data. Each $$L_{t-1}$$ is a [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between two Gaussians, which has a [closed form](https://huggingface.co/blog/annotated-diffusion#defining-an-objective-function-by-reparametrizing-the-mean), so that these terms are computed exactly instead of estimated from samples.

Consider first the terms $$L_{t-1}$$. With $$\Sigma_\theta = \sigma_t^2 \mathbf{I}$$, both Gaussians have fixed variances, and their KL divergence depends on $$\theta$$ only through the distance between their means. For $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t \mathbf{I})$$ and $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 \mathbf{I})$$, it is (Eq. 8):

$$
L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2 \sigma_t^2} \| \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) - \mu_\theta(\mathbf{x}_t, t) \|^2 \right] + C
$$

where $$C$$ is a constant that does not depend on $$\theta$$. The most direct parameterization is therefore a model $$\mu_\theta$$ that predicts the posterior mean $$\tilde{\mu}_t$$. DDPM goes one step further and writes this mean in terms of the noise. It writes Eq. 4 as $$\mathbf{x}_t(\mathbf{x}_0, \epsilon) = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$, with $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$, solves it for $$\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \epsilon \right)$$, and replaces $$\mathbf{x}_0$$ in the posterior mean $$\tilde{\mu}_t$$ of Eq. 7 (Eqs. 9 and 10):

$$
L_{t-1} - C = \mathbb{E}_{\mathbf{x}_0, \epsilon} \left[ \frac{1}{2 \sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t(\mathbf{x}_0, \epsilon) - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right) - \mu_\theta(\mathbf{x}_t(\mathbf{x}_0, \epsilon), t) \right\|^2 \right]
$$

So $$\mu_\theta$$ must predict $$\frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)$$ from $$\mathbf{x}_t$$. In this expression, $$\mathbf{x}_t$$ is an input of the model and the coefficients are known, so the only unknown is the noise $$\epsilon$$ that was added to the data. The network therefore only needs to predict this noise, $$\epsilon_\theta(\mathbf{x}_t, t)$$, and the mean follows from it (Eq. 11):

$$
\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right)
$$

In other words, the network is a denoiser: given a noisy image, it estimates the noise in it. Its inputs are the noisy image $$\mathbf{x}_t$$ and the [timestep $$t$$, which it turns into a timestep embedding](https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/unets/unet_2d.py#L243), so that a single network serves all noise levels. The code is:

```python
    def eq11_μ_θ(x, ε_θ, t):
        # Equation 11: use model (noise predictor) to predict the mean
        β_t = β[t][:, None, None, None]
        one_over_sqrt_α = torch.sqrt(1.0 / α)
        one_over_sqrt_α_t = one_over_sqrt_α[t][:, None, None, None]
        sqrt_1_minus_α_cumprod = torch.sqrt(1. - α_cumprod)
        sqrt_1_minus_α_cumprod_t = sqrt_1_minus_α_cumprod[t][:, None, None, None]
        μ_θ = one_over_sqrt_α_t * (x - β_t * ε_θ  / sqrt_1_minus_α_cumprod_t)
        return μ_θ
```

With Eq. 11, each $$L_{t-1}$$ becomes a weighted mean squared error between the sampled and the predicted noise (Eq. 12):

$$
L_{t-1} - C = \mathbb{E}_{\mathbf{x}_0, \epsilon} \left[ \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1-\bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$

DDPM notes that this objective resembles denoising score matching over several noise levels, and that sampling with Eq. 11 resembles [Langevin dynamics](https://en.wikipedia.org/wiki/Langevin_dynamics), with $$\epsilon_\theta$$ as a learned gradient of the data density.

The last term, $$L_0$$, needs a different treatment, because images are discrete data. Pixel values are integers in $$\{0, 1, \dots, 255\}$$, scaled linearly to $$[-1, 1]$$, and a probability density alone does not give them a probability. DDPM therefore ends the reverse process with a discrete decoder: the likelihood of a pixel value is the probability mass of the Gaussian $$p_\theta$$ in a bin of width $$2/255$$ around it, where the first and last bins extend to infinity (Eq. 13):

$$
p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) = \prod_{i=1}^D \int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}(x; \mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2) \, dx
\quad \text{with} \quad
\delta_+(x) = \begin{cases} \infty & \text{if } x = 1 \\ x + \frac{1}{255} & \text{if } x < 1 \end{cases}
\quad \text{and} \quad
\delta_-(x) = \begin{cases} -\infty & \text{if } x = -1 \\ x - \frac{1}{255} & \text{if } x > -1 \end{cases}
$$

where $$D$$ is the data dimension and $$i$$ indexes one coordinate. The negative log-likelihood per pixel value, for the Gaussian `p` of $$p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$$, is:

```python
    def eq13_decoder_nll(x_0, p):
        """ Equation 13: negative log-likelihood of the discretized decoder p_θ(x_0|x_1), per pixel value """
        cdf_upper = p.cdf(x_0 + 1/255) # CDF at δ+(x_0), for x_0 < 1
        cdf_lower = p.cdf(x_0 - 1/255) # CDF at δ-(x_0), for x_0 > -1
        # the CDF at -∞ and +∞ is 0 and 1 (evaluating it at ±∞ would make the gradients NaN)
        prob = torch.where(x_0 < -0.999, cdf_upper, torch.where(x_0 > 0.999, 1 - cdf_lower, cdf_upper - cdf_lower))
        return -torch.log(prob.clamp(min=1e-12))
```

## Sampling

After training, new images are generated by running the reverse process, from $$t = T$$ down to $$t = 1$$ (Algorithm 2):

$$
\begin{array}{l}
\hline
\textbf{Algorithm 2} \text{ Sampling} \\
\hline
\text{1:} \;\; \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
\text{2:} \;\; \textbf{for}\ t = T, \ldots, 1\ \textbf{do} \\
\text{3:} \quad\; \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \text{ if } t > 1 \text{, else } \mathbf{z} = \mathbf{0} \\
\text{4:} \quad\; \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z} \\
\text{5:} \;\; \textbf{end for} \\
\text{6:} \;\; \textbf{return}\ \mathbf{x}_0 \\
\hline
\end{array}
$$

The algorithm starts from pure noise, and each pass through step 4 samples $$\mathbf{x}_{t-1}$$ from $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$. Since $$1 - \alpha_t = \beta_t$$, the first term of step 4 is the predicted mean $$\mu_\theta(\mathbf{x}_t, t)$$ of Eq. 11, and the second term adds the noise $$\mathbf{z}$$, scaled by the standard deviation $$\sigma_t$$. The last step adds no noise, so that the generated image is the predicted mean itself. The U-Net uses $$\sigma^2_t = \tilde{\beta}_t$$, and the DiT its learned variance $$\Sigma_\theta$$. As in DDPM, the `model` that samples is an [exponential moving average](https://openreview.net/forum?id=2M9CUnYnBA) (EMA) of the trained weights, described in the next section:

```python
    @torch.no_grad()
    def alg2_p_sampling(model, shape):
        img = torch.randn(shape, device=device) # step 1
        for t_index in reversed(range(0, T)): # step 2
            t = torch.full((shape[0],), t_index, device=device, dtype=torch.long)
            if model_name == 'UNet':
                ε_θ = model(img, t).sample # extract tensor from UNet2DOutput class
                σ2_t = posterior_β[t][:, None, None, None] # σ_t^2 = β̃_t
            else:
                ε_θ, v = model(img, t)
                σ2_t = learned_Σ_θ(v, t) # σ_t^2 = Σ_θ, learned
            img = eq11_μ_θ(img, ε_θ, t) # Eq. 11
            if t_index > 0: # step 3: z = 0 at the last step (paper's t=1)
                img += torch.sqrt(σ2_t) * torch.randn_like(img) # step 4
        return img
```

## Training Algorithm

The model trains on the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) dataset, which contains 32x32 RGB images of 10 classes, with a batch of 48 images per GPU. The pixel values are scaled to $$[-1, 1]$$, the range that the decoder of Eq. 13 expects:

```python
    # load dataset from the hub
    dataset = load_dataset("uoft-cs/cifar10", split='train')
    transform = lambda image: (pil_to_tensor(image)/255)*2-1 # normalize to [-1,1]
    images = [ transform(image) for image in dataset["img"]]
    batch_size, channels, img_size = 48, images[0].shape[0], images[0].shape[-1]
    sampler = DistributedSampler(images)
    dataloader = DataLoader(images, batch_size=batch_size, sampler=sampler, drop_last=True)
```

The model is the U-Net of the `diffusers` package, which is similar to the network of DDPM (the DiT comes in the next section). As in DDPM, the optimizer is Adam with a learning rate of $$2 \times 10^{-4}$$, and training keeps an EMA of the weights with a decay of $$0.9999$$. The EMA averages the weights over roughly the last 10,000 steps, the inverse of $$1 - 0.9999$$, and smooths out the noise of the individual updates. With a fixed decay, however, the poor weights of the first training step would still make up $$0.9999^{10000} \approx 37\%$$ of the average after 10,000 steps. As in the default `EMAModel` of `diffusers`, update $$n$$ of the EMA therefore uses the decay $$\min \left( \frac{1+n}{10+n}, 0.9999 \right)$$, so that the average follows the trained model closely at the start of training:

```python
    # load model, optimizer and the Exponential Moving Average (EMA) of the model weights
    if model_name == 'UNet':
        model = diffusers.UNet2DModel(in_channels=channels, out_channels=channels)
    elif model_name == 'DiT':
        model = DiT(T, channels, img_size, patch_size=4, n_blocks=4)
    else:
        raise ValueError(f"Model name {model_name} not recognized")
    model = model.to(device=device)
    ema_decay = lambda n: ((1 + n) / (10 + n)).clamp(max=0.9999) # 0.9999 as in the paper, with a warmup as in diffusers
    ema = AveragedModel(model, avg_fn=lambda ema_p, p, n: ema_decay(n) * ema_p + (1 - ema_decay(n)) * p).eval()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    optimizer = Adam(model.parameters(), lr=2e-4)
```

As in DDPM, the loss is the mean squared error between the sampled noise $$\epsilon$$ and the predicted noise $$\epsilon_\theta$$ (Eq. 14):

$$
\mathcal{L}_\text{simple}(\theta) = \mathop{\mathbb{E}}_{t, \mathbf{x}_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t)\| ^2 \right]
$$

where $$t$$ is drawn uniformly from $$\{1, \dots, T\}$$. $$\mathcal{L}_\text{simple}$$ is a simplified version of the bound $$L$$: the terms $$t > 1$$ are the terms $$L_{t-1}$$ of Eq. 12 without their weights, the term $$t = 1$$ approximates $$L_0$$, and the constant $$L_T$$ is dropped. Dropping the weights changes more than it seems. For the schedule of DDPM with $$\sigma_t^2 = \beta_t$$, the weight of Eq. 12 is 0.5 at $$t = 1$$, but at most 0.01 for $$t \ge 100$$. Compared with the bound, $$\mathcal{L}_\text{simple}$$ therefore gives much less weight to the terms of small $$t$$, where the noise is small and denoising is easy, and it moves the training effort to the harder denoising tasks at large $$t$$. DDPM found that this loss gives better samples, and it is also simpler to implement. The Huber loss and the mean absolute error are also common choices:

```python
    def unet_loss(x_t, t, noise):
        """ L_simple, Equation 14 """
        ε_θ = model(x_t, t).sample # extract tensor from UNet2DOutput class
        return F.mse_loss(ε_θ, noise) # Huber loss and MAE are also ok
```

Together, these pieces give the training algorithm (Algorithm 1):

$$
\begin{array}{l}
\hline
\textbf{Algorithm 1} \text{ Training} \\
\hline
\text{1:} \;\; \textbf{repeat} \\
\text{2:} \quad\; \mathbf{x}_0 \sim q(\mathbf{x}_0) \\
\text{3:} \quad\; t \sim \mathrm{Uniform}(\{1, \ldots, T\}) \\
\text{4:} \quad\; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
\text{5:} \quad\; \text{Take gradient descent step on} \\
\qquad\qquad\; \nabla_\theta \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, t) \right\|^2 \\
\text{6:} \;\; \textbf{until} \text{ converged} \\
\hline
\end{array}
$$

Each iteration takes a clean image, picks a random step and a random noise, builds the noisy image of Eq. 4, and trains the network to recover the noise. The inner loop below runs steps 2 to 5 for each batch, and after each epoch, the EMA model generates and saves new images:

```python
    # training loop
    for epoch in range(20):
        sampler.set_epoch(epoch) # shuffle the data differently at every epoch
        for step, x_0 in enumerate(dataloader):
            x_0 = x_0.to(device=device)
            t = torch.randint(0, T, (batch_size,), device=device).long() # Algorithm 1, step 3 (0-indexed)
            noise = torch.randn_like(x_0) # step 4
            x_t = q_sample(x0=x_0, t=t, noise=noise)
            loss = unet_loss(x_t, t, noise) if model_name == 'UNet' else dit_loss(x_0, x_t, t, noise) # step 5
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update_parameters(model.module)
            if dist.get_rank() == 0:
                print(f"epoch {epoch}, step {step} loss: {loss.item()}")

        # save images generated with the EMA model
        if dist.get_rank() == 0:
            results_folder = Path("./results")
            results_folder.mkdir(exist_ok = True)
            img = alg2_p_sampling(ema, shape=x_0.shape)
            img = (img + 1) * 0.5 # from [-1,1] to [0,1]
            save_image(img, str(results_folder / f'sample-{epoch}.png'), nrow = batch_size)
```

## Diffusion Transformers

As Transformers became the main building block of sequence models, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) asked whether diffusion models need the U-Net at all. Its answer is that the convolutional inductive bias of the U-Net is not essential to their performance, and that a standard Transformer can replace it: the diffusion transformer (DiT). DiT is a [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT), which splits its input into patches and processes them as a sequence of tokens (Figure 4 in the paper). ViTs scale well: when pre-trained on large datasets, they match or outperform convolutional networks in image classification, with less training compute, and DiT brings this scalability to diffusion. Its input is not the image itself, but the latent representation of the image produced by a pre-trained VAE, as in the latent diffusion models of the video section.

The paper measures the quality of the samples with the [Fréchet inception distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) (FID), which compares the feature statistics of generated and real images, lower being better. Its largest model, DiT-XL/2, outperforms the earlier U-Net diffusion models on class-conditional ImageNet, with an FID of 2.27 at 256x256 (with the classifier-free guidance of the conditioning section). The scaling study of the paper shows where this quality comes from. The FID falls as the Gflops of the model grow, whether they come from more layers, wider layers or more tokens. A smaller patch size, in particular, adds tokens and Gflops but no parameters, and it still lowers the FID: the Gflops, not the parameter count, set the quality. Larger DiT models also use compute more efficiently, as they reach a given FID with less training compute.

{: .picture}
<img width="82.4%" height="82.4%" src="{{ site.assets }}/diffusion/dit.svg"/> 

{: .caption}
<b>The Diffusion Transformer (DiT) architecture.</b> <i>Top left:</i> a conditional latent DiT; the input latent is decomposed into patches and processed by several DiT blocks. <i>Top right and bottom:</i> variants of the DiT block that add the conditioning by adaptive layer norm (adaLN-Zero), cross-attention or extra input tokens (in-context conditioning); adaptive layer norm works best. Adapted from: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).

The left part of the figure shows the latent DiT. A noised latent of 32x32x4 values, the VAE encoding of a 256x256 image, is cut into patches by `Patchify` and becomes a sequence of tokens. A stack of DiT blocks processes these tokens, conditioned on the embeddings of the timestep $$t$$ and of the class label $$y$$, and a final layer norm and a linear layer decode each token back into a patch of the predicted noise and of the covariance $$\Sigma$$. The other parts show the block designs that the paper compares. In the adaLN-Zero block, at the top right, adaptive layer normalization conditions each block on $$t$$ and $$y$$: the scale $$\gamma$$ and shift $$\beta$$ of each layer norm are not learned parameters, but outputs of an MLP that takes the sum of the embeddings of $$t$$ and $$y$$. adaLN-Zero also regresses scales $$\alpha$$, applied before each residual connection, and initializes them to zero, so that each block starts as the identity function, an initialization that earlier work found beneficial for residual networks. These $$\alpha$$, $$\beta$$ and $$\gamma$$ are unrelated to the $$\alpha_t$$ and $$\beta_t$$ of the noise schedule. The two faded blocks at the bottom are the alternatives, which add the conditioning through cross-attention or as extra tokens.

The code implements a simpler DiT: a ViT made of positional embeddings, a stack of Transformer blocks and a decoder. It leaves out the four block designs of Section 3.2 of the paper (in-context conditioning, cross-attention, adaLN and adaLN-Zero). Instead, it adds a timestep embedding to every patch, and each block is a regular Transformer `Block`, i.e. multi-head self-attention followed by a feed-forward network, as in the [GPT-lite post]({{ site.baseurl }}{% post_url 2023-02-28-gptlite %}). Unlike the DiT of the paper, it works on the pixels of the small CIFAR-10 images directly, without a VAE. Each token is a flattened image patch, with no linear patch embedding, so the embedding size is $$p \times p \times C$$, for a patch size $$p$$ and $$C$$ channels. The positional and timestep embeddings are learned look-up tables (`nn.Embedding`), where DiT uses frequency-based (sine-cosine) positional embeddings. Finally, the decoder is a layer norm followed by a linear layer, which outputs $$p \times p \times 2C$$ values per token: a predicted noise and a value that sets the variance, for each channel and pixel of the patch.

```python
class DiT(nn.Module):
    """ A Diffusion Transformer (DiT) model """

    def __init__(self, timesteps, num_channels, img_size, patch_size=4, n_blocks=12, num_labels=None):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        n_embd = patch_size*patch_size*num_channels # values per img patch

        # timestep and positional embeddings
        n_pos_emb = (img_size//patch_size)*(img_size//patch_size) # number of patches per image
        self.t_embedding = nn.Embedding(timesteps, n_embd)
        self.pos_embedding = nn.Embedding(n_pos_emb, n_embd)

        # class embeddings, only for a class-conditional model
        self.class_embedding = nn.Embedding(num_labels, n_embd) if num_labels else None

        # DiT blocks: a regular transformer Block, as in GPT-lite (see dit.py)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd) for _ in range(n_blocks)])

        # decoder: "standard linear decoder to do this; we apply the layer norm and linearly decode each token into a p×p×2C tensor"
        self.decoder = nn.Sequential( nn.LayerNorm(n_embd), nn.Linear(n_embd, n_embd*2) )
```

The methods `patchify` and `unpatchify` split an image into the patches that form the token sequence, and turn the output patches back into images, here the noise $$\epsilon_\theta$$ and the output `v` that sets the variance:

```python
    def patchify(self, x):
        """ break image (B, C, H, W) into patches (B, C, NH, NW, PH, PW) for NH*NW patches of size PHxPW """
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # linearize patches and flatten embeddings: (B, NH*NW, PH*PW*C)
        _, _, NH, NW, PH, PW = x.shape
        x = x.permute(0, 2, 3, 4, 5, 1) # (B, NH, NW, PH, PW, C)
        x = x.reshape(B, NH*NW, PH*PW*C)
        return x, dict(B=B, C=C, H=H, W=W, NH=NH, NW=NW, PH=PH, PW=PW)

    def unpatchify(self, x, shapes):
        """ convert patches (B, NH*NW, PH*PW*C*2) back into the noise ε_θ and the vector v that parameterizes
            the variance Σ_θ, each of shape (B, C, H, W) = (B, C, NH*PH, NW*PW) """
        B, C, H, W, NH, NW, PH, PW, = shapes.values()
        assert x.shape == (B, NH*NW, PH*PW*C*2)
        x = x.reshape(B, NH, NW, PH, PW, C, 2).permute(0, 5, 1, 3, 2, 4, 6) # (B, C, NH, PH, NW, PW, 2)
        x = x.reshape(B, C, NH*PH, NW*PW, 2)
        ε_θ, v = x[...,0], x[...,1]
        assert ε_θ.shape == v.shape == (B, C, H, W) # original shape
        return ε_θ, v
```

The forward pass adds the positional and timestep embeddings to the patches (and the class embedding of the next section), then runs the Transformer blocks and the decoder:

```python
    def forward(self, x, t, label=None):
        x, shapes = self.patchify(x) # (B, C, H, W) -> (B, N, E), for N patches of E=PH*PW*C values
        B, N, E = x.shape
        x += self.pos_embedding(torch.arange(N, device=x.device)).reshape(1, N, E) # positional embeddings
        x += self.t_embedding(t).reshape(B, 1, E) # timestep embedding, added to all patches of an image
        if label is not None: # class embedding, added to all patches of an image
            x += self.class_embedding(label).reshape(B, 1, E)
        x = self.blocks(x)
        x = self.decoder(x) # (B, N, E) -> (B, N, 2E)
        return self.unpatchify(x, shapes) # ε_θ and v
```

Unlike the U-Net, the DiT also learns the covariance $$\Sigma_\theta$$, and this requires a change of the loss: $$\mathcal{L}_\text{simple}$$ does not depend on $$\Sigma_\theta$$, so only the bound $$L$$ can train it. Following [Improved DDPM](https://arxiv.org/abs/2102.09672), the network outputs a vector $$v$$ with one value per dimension, which interpolates the log-variance between the two extreme choices of DDPM, $$\beta_t$$ and $$\tilde{\beta}_t$$ (Eq. 15 of Improved DDPM):

$$
\Sigma_\theta(\mathbf{x}_t, t) = \exp \left( v \log \beta_t + (1-v) \log \tilde{\beta}_t \right)
$$

Why not predict the variance directly? The two bounds are almost equal, except at the first steps, so the range of reasonable variances is narrow, and it is hard for a network to predict the variance directly, even in the log domain. A position between the two bounds is much easier to learn. As in the official implementation, the code maps the raw network output `v` from $$[-1, 1]$$ to `frac` $$\in [0, 1]$$, which is the $$v$$ of Eq. 15:

```python
    def learned_Σ_θ(v, t):
        """ Improved DDPM, Equation 15: interpolate log(Σ_θ) between log(β_t) and log(posterior_β_t) """
        frac = (v + 1) / 2 # from the model output in [-1, 1] to [0, 1], as in the official implementation
        log_β_t = torch.log(β)[t][:, None, None, None]
        posterior_log_β_t = posterior_log_β[t][:, None, None, None]
        return torch.exp(frac * log_β_t + (1 - frac) * posterior_log_β_t)
```

Improved DDPM then trains $$\epsilon_\theta$$ with $$\mathcal{L}_\text{simple}$$, as before, and $$\Sigma_\theta$$ with the bound $$L_\text{vlb}$$, the $$L$$ of Eq. 5. The two combine into a hybrid loss (Eq. 16 of Improved DDPM):

$$
\mathcal{L}_\text{hybrid} = \mathcal{L}_\text{simple} + \lambda L_\text{vlb}
$$

where $$\lambda = 0.001$$ keeps $$L_\text{vlb}$$ from overwhelming $$\mathcal{L}_\text{simple}$$. A stop-gradient on $$\mu_\theta$$ makes $$L_\text{vlb}$$ train only $$\Sigma_\theta$$, so that $$\mathcal{L}_\text{simple}$$ remains the main source of influence over the mean. Each $$L_{t-1}$$ is the closed-form KL divergence between two Gaussians ([Meta's DiT implementation](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py#L682) has another implementation of it), and $$L_0$$ is the negative log-likelihood of the decoder of Eq. 13. One detail remains: each image samples a single timestep, while $$L_\text{vlb}$$ sums the terms of $$T$$ timesteps. The code therefore estimates the bound as $$T$$ times the term of the sampled timestep, measured in bits per dimension as in the paper:

```python
    def dit_loss(x_0, x_t, t, noise):
        """ L_hybrid = L_simple + λ L_vlb, Improved DDPM, Equation 16 """
        ε_θ, v = model(x_t, t)
        Σ_θ = learned_Σ_θ(v, t)
        μ_θ = eq11_μ_θ(x_t, ε_θ.detach(), t) # stop-gradient: L_vlb only trains Σ_θ
        p = torch.distributions.Normal(μ_θ, Σ_θ.sqrt()) # Eq. 1
        q = torch.distributions.Normal(posterior_μ(x_0, x_t, t), torch.exp(0.5 * posterior_log_β[t][:, None, None, None])) # Eq. 6
        kl = torch.distributions.kl_divergence(q, p).mean(dim=(1, 2, 3)) # L_{t-1}
        nll = eq13_decoder_nll(x_0, p).mean(dim=(1, 2, 3)) # L_0
        L_vlb = torch.where(t == 0, nll, kl) / math.log(2) # per sample, in bits per dimension (t=0 is the paper's t=1)
        L_simple = F.mse_loss(ε_θ, noise)
        return L_simple + T / 1000 * L_vlb.mean() # λ = 1/1000, times T as L_vlb sums the losses of T timesteps
```

## Conditioning

The model above is unconditional. Trained on CIFAR-10, which has 10 classes of objects, it generates an image of a random class, and the class cannot be chosen. Most applications need to guide the generation instead, with a class, a text, an image or audio. A **conditional** model takes this extra information $$y$$ as an input and learns $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, y)$$, where $$\epsilon_\theta$$ and $$\Sigma_\theta$$ depend on $$y$$. The DiT of the paper is class-conditional, through the adaLN-Zero blocks of the previous section. The DiT code takes an optional class `label` in a simpler way: when it is created with `num_labels` classes, it learns a class embedding and adds it to all patches of an image, like the timestep embedding:

```python
    def __init__(self, timesteps, num_channels, img_size, patch_size=4, n_blocks=12, num_labels=None):
        # [...]
        # class embeddings, only for a class-conditional model
        self.class_embedding = nn.Embedding(num_labels, n_embd) if num_labels else None
        # [...]

    def forward(self, x, t, label=None):
        # [...]
        if label is not None: # class embedding, added to all patches of an image
            x += self.class_embedding(label).reshape(B, 1, E)
        # [...]
```

The training script trains an unconditional model, and passing the CIFAR-10 label of each image to the model trains a class-conditional one.

How closely should the samples follow their condition? [Classifier-free guidance](https://arxiv.org/abs/2207.12598) turns this into a choice at sampling time, and it improves the sample quality of a conditional model at the cost of sample diversity. Training replaces the label $$y$$ by a learned null embedding $$\emptyset$$ in a fraction of the samples (of the fractions tested, 10% and 20% work best), so that one model learns both the conditional prediction $$\epsilon_\theta(\mathbf{x}_t, y)$$ and the unconditional one $$\epsilon_\theta(\mathbf{x}_t, \emptyset)$$. Sampling then extrapolates from the unconditional prediction towards the conditional one, with a guidance scale $$s \gt 1$$, in the notation of DiT:

$$
\hat{\epsilon}_\theta(\mathbf{x}_t, y) = \epsilon_\theta(\mathbf{x}_t, \emptyset) + s \left( \epsilon_\theta(\mathbf{x}_t, y) - \epsilon_\theta(\mathbf{x}_t, \emptyset) \right)
$$

For $$s = 1$$, this is plain conditional sampling. The difference between the two predictions is, up to a scale factor, the gradient of $$\log p(y \mid \mathbf{x}_t)$$ of an implicit classifier, so a larger $$s$$ pushes the samples towards images that clearly belong to class $$y$$. Each sampling step runs the model twice, with and without the label, so guidance doubles the cost of sampling.

## Video Diffusion and 3D Attention

Running diffusion directly on pixels is already expensive for high-resolution images, as every denoising step processes every pixel. This led to the [latent diffusion model](https://arxiv.org/abs/2112.10752) (LDM), which runs diffusion in the latent space of a pre-trained [autoencoder](https://en.wikipedia.org/wiki/Autoencoder), such as a VAE, instead of on the image itself. The latent is much smaller than the image, which cuts the cost of training and sampling, and the decoder of the autoencoder turns the generated latent back into an image. When diffusion moved to video, the large amount of data per sample made this compression even more important, and video models now compress their input in both space and time:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/diffusion/sora_vae.svg"/> 

{: .caption}
An overview of the latent space compression in Sora. The pre-processing step "turns videos into [visual] patches by first compressing videos into a lower-dimensional latent space and subsequently decomposing the representation into spacetime patches". Adapted from: [Video generation models as world simulators (OpenAI)](https://openai.com/index/video-generation-models-as-world-simulators/)

In the figure, a visual encoder compresses the frames of the video into a smaller latent volume, which is then cut into spacetime patches and flattened into a sequence of tokens.

The second challenge of video data is attention, which must relate the patches within a frame (space) and across frames (time). Consider a latent video of shape $$B \times F \times H \times W \times C$$: a batch of $$B$$ videos of $$F$$ frames, each frame made of $$H \times W$$ patches with $$C$$ channels. The frames are counted by $$F$$, as $$T$$ already counts the diffusion steps. There are two main designs:
- **Spatio-temporal attention** factorizes the attention into two parts. A spatial attention relates the patches of each frame, on the input reshaped to $$(B F) \times (H W) \times C$$, i.e. one sequence of $$H W$$ patches per frame. A temporal attention then relates the patches at the same position across frames, on the input reshaped to $$(B H W) \times F \times C$$, i.e. one sequence of $$F$$ patches per position. Two patches at different positions of different frames are connected in two hops, one spatial and one temporal, and repeating the pair over several U-Net or DiT blocks correlates the patches across both space and time.
- **Full 3D attention** relates all patches of all frames at once, on the input reshaped to $$B \times (F H W) \times C$$, i.e. one long sequence per video.

Which design is better? The cost of attention grows with the square of the sequence length. Per layer, spatio-temporal attention computes $$B F (H W)^2 + B H W F^2$$ attention scores, while full 3D attention computes $$B (F H W)^2$$, which is $$F H W / (H W + F)$$ times more, or about $$F$$ times more when $$H W \gg F$$: for 16 frames of 32x32 patches, about 16 times more. Spatio-temporal attention is therefore cheaper, and its spatial layers can come from a pre-trained text-to-image model. Full 3D attention, on the other hand, relates any two patches in a single layer. Large recent models such as CogVideoX and Movie Gen (below) use it, and their long sequences call for the [sequence and context parallelism]({{ site.baseurl }}{% post_url 2024-08-30-sequence-context-parallelism %}) of long-context language models.

Training can also shorten the sequence by dropping tokens. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (MAE) showed that a ViT encoder learns good representations from a random 25% of the patches of each image, which speeds up training by 3x or more. [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113) extended this to videos, where 10% of the spacetime patches are enough. [Fast Training of Diffusion Models with Masked Transformers](https://arxiv.org/abs/2306.09305) (MaskDiT) brings the idea to diffusion: it masks a large fraction of the patches during training (e.g. 50%), and matches DiT in about 30% of its training time. [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304) also drops random tokens, together with sequence packing, to speed up training.

{: .picture}
<img width="90.6%" height="90.6%" src="{{ site.assets }}/diffusion/masked_autoencoders_cropped.svg"/> 

{: .caption}
An illustration of a masked autoencoder that randomly picks 10% of the spacetime patches of a video. These patches retain enough representative power to reconstruct the original video. Adapted from: [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113)

In the figure, the encoder sees only the visible patches (green), and the decoder receives their encodings (blue) together with mask tokens (grey) for the missing patches, from which it reconstructs the whole video.

## Further Reading

The papers below go further. The first ones are U-Net diffusion models for text-to-image and text-to-video generation:

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Imagen: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)</summary>
A U-Net text-to-image model, whose "key discovery is that generic large language models (e.g. T5), pretrained on text-only corpora, are surprisingly effective at encoding text for image synthesis: increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model".
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127)</summary>
A U-Net latent diffusion model for text-to-video and image-to-video generation, based on Stable Diffusion 2.1. It trains in three stages: (1) text-to-image pretraining, (2) video pretraining on a large dataset at low resolution, and (3) high-resolution finetuning on a much smaller set of high-quality videos. The paper also shows the importance of data curation, and describes how it removes scene cuts, static scenes (with low optical flow) and clips with much written text (found by OCR).
<!-- Captions are generated by using a model to describe the mid frame of the video, and V-BLIP to generate captions from video, and an LLM to summarize the previous 2 captions. -->
To train on videos, it adds temporal layers to a pre-trained image model, as in [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818):
> We first pre-train an LDM on images only; then, we turn the image generator into a video generator by introducing a temporal dimension to the latent space diffusion model and fine-tuning on encoded image sequences, i.e., videos.

{: .picture}
<img width="93.8%" height="93.8%" src="{{ site.assets }}/diffusion/align_your_latents.svg"/> 

{: .caption}
**Left:** a pre-trained LDM is turned into a video generator by inserting temporal layers that learn to align frames into temporally consistent sequences. During optimization, the image backbone $$\theta$$ remains fixed and only the parameters $$\phi$$ of the temporal layers $$l^i_\phi$$ are trained. **Right:** During training, the base model $$\theta$$ interprets the input sequence of length $$T$$ as a batch of images. For the temporal layers $$l^i_\phi$$, these batches are reshaped into video format. Their output $$z'$$ is combined with the spatial output $$z$$, using a learned merge parameter $$\alpha$$. During inference, skipping the temporal layers ($$\alpha^i_\phi=1$$) yields the original image model. For illustration purposes, only a single U-Net Block is shown. $$B$$ denotes the batch size, $$T$$ the sequence length, $$C$$ the input channels and $$H$$ and $$W$$ the spatial dimensions of the input. $$c_S$$ is an optional context-frame conditioning, used when training prediction models. Adapted from: [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818).
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation](https://arxiv.org/abs/2311.17117)</summary>
A U-Net model that animates the person in a reference image with a pose sequence, i.e. a moving stick figure. A VAE and CLIP encode the reference image. The model has two U-Nets: ReferenceNet merges detail features of the reference image into the denoising U-Net through spatial attention, and the denoising U-Net generates the video from the pose sequence, which a lightweight *pose guider* encodes. Each block of the denoising U-Net runs spatial attention, cross-attention to the CLIP features and temporal attention, i.e. the spatio-temporal attention of the video section.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/diffusion/anymate_anyone.svg"/> 

{: .caption}
Overview of Animate Anyone. The pose sequence is initially encoded using Pose Guider and fused with multi-frame noise, followed by the Denoising UNet conducting the denoising process for video generation. The computational block of the Denoising UNet consists of Spatial-Attention, Cross-Attention, and Temporal-Attention, as illustrated in the dashed box on the right. The integration of reference image involves two aspects. Firstly, detailed features are extracted through ReferenceNet and utilized for Spatial-Attention. Secondly, semantic features are extracted through the CLIP image encoder for Cross-Attention. Temporal-Attention operates in the temporal dimension. Finally, the VAE decoder decodes the result into a video clip. Adapted from: [Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation](https://arxiv.org/abs/2311.17117).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[CyberHost: Taming Audio-driven Avatar Diffusion Model with Region Codebook Attention](https://arxiv.org/abs/2409.01876)</summary>
An audio-driven U-Net model that animates a person from a reference image, with the reference net of Animate Anyone. Its Region Codebook Attention mechanism "improves the generation quality of facial and hand animations by integrating fine-grained local features with learned motion pattern priors".
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack](https://arxiv.org/abs/2309.15807)</summary>
A U-Net latent diffusion model for text-to-image, which shows how much fine-tuning on a small set of 2,000 high-quality images improves the samples: "in order to align the model towards highly aesthetic generations – quality matters significantly more than quantity in the fine-tuning dataset".
</details>
{::options parse_block_html="false" /}

<br/>
The next ones are conditional models based on DiT:

{::options parse_block_html="true" /}
<details> <summary markdown="span">[OmniGen: Unified Image Generation](https://arxiv.org/abs/2409.11340)</summary>
OmniGen is made of only a VAE and a Transformer, initialized from the Phi-3 LLM, and a single model performs text-to-image generation, image editing, subject-driven generation and visual-conditional generation.

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/diffusion/omnigen.svg"/> 

{: .caption}
The framework of OmniGen. Texts are tokenized into tokens, while input images are transformed into embedding via VAE. OmniGen can accept free-form multi-modal prompts and generate images through the rectified flow approach. The attention mask (left) is causal across the input sequence and bidirectional within each image. Adapted from: [OmniGen: Unified Image Generation](https://arxiv.org/abs/2409.11340).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models](https://arxiv.org/abs/2409.10695)</summary>
A text-to-image model that replaces the usual T5 and CLIP text encoders with a frozen Llama3-8B LLM. Its Transformer blocks mirror those of the LLM, and each image block takes the hidden states of the corresponding LLM layer as conditioning.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/diffusion/playground_v3.svg"/> 

{: .caption}
Simplified structure of Playground v3 (PGv3). Text conditioning comes only from Llama3-8B, with no T5 or CLIP encoder: each PGv3 block attends jointly to the image tokens and to the keys and values of the matching Llama3 block. Adapted from: [Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models](https://arxiv.org/abs/2409.10695).

</details>
{::options parse_block_html="false" /}
 
<br/>
The following works address video diffusion and 3D attention:

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)</summary>
NaViT (Native Resolution ViT) packs the patches of several images into one sequence, hence "Patch n' Pack", so that it trains on images of any resolution and aspect ratio without resizing them. Packing also allows a different token dropping rate for each image.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072)</summary>
A DiT for text-to-video generation. T5 encodes the text, and a 3D causal VAE compresses the video in space and time. The video patches are unfolded into one long sequence and concatenated with the text embeddings, and a stack of *expert* Transformer blocks processes this sequence, with separate adaptive layer norms for the text and the video tokens. The model uses full 3D attention instead of separate spatial and temporal attention, and the 3D VAE decodes its unpatchified output into a video.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048v1)</summary>

A DiT for unconditional, class-conditional and text-to-video generation. It extracts spatio-temporal tokens from the input video and models their distribution in latent space with a series of Transformer blocks. The paper tests the main design choices (patch embedding, model variant, timestep and class injection, temporal positional embedding and learning strategy) and reports the best ones. Its four model variants differ in the way they combine space and time:
- Variant 1 interleaves spatial and temporal Transformer blocks.
- Variant 2 runs all spatial blocks before all temporal blocks ("late fusion").
- Variant 3 computes spatial and then temporal attention inside each block.
- Variant 4 splits the attention heads of each block between the spatial and the temporal dimension.

Of the four, variant 1 performs best.

{: .picture}
<img width="77.9%" height="77.9%" src="{{ site.assets }}/diffusion/latte.svg"/> 

{: .caption}
<b>The pipeline of Latte for video generation</b>. Four model variants of Latte are proposed to efficiently capture spatio-temporal information in videos. Each block depicted in light orange represents a Transformer block. The standard Transformer blocks are used in (a) and (b), while (c) and (d) use Latte's own block variants. For the sake of simplicity, encoding and decoding processes for VAE are not shown in the diagram. Adapted from: [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048).

The paper also compares two ways to embed the patches of a video. The uniform frame patch embedding (a) embeds each frame into tokens as in ViT, while the compression frame patch embedding (b) extends the ViT patch embedding to 3D and extracts tubes along the time dimension. Embedding (a) performs better:


{: .picture}
<img width="70%" height="70%" src="{{ site.assets }}/diffusion/latte2.svg"/> 

{: .caption}
<b>The video clip patch embedding</b>. (a) Uniform frame patch embedding: <i>F</i> frames are sampled and each frame is embedded into tokens as in ViT. (b) Compression frame patch embedding: to capture temporal information, the ViT patch embedding is extended from 2D to 3D and tubes are extracted along the temporal dimension. The original video clip is shown for clarity; the patch embedding in the latent space of videos follows the same approach. Adapted from: [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://arxiv.org/abs/2407.21705) ([webpage](https://ali-videoai.github.io/tora_video/))</summary>

Tora generates videos guided by trajectories, images, texts or their combinations. It "employs the Spatial-Temporal Diffusion Transformer (ST-DiT) from OpenSora as its foundational model": each block runs a spatial and then a temporal attention, as variant 3 of [Latte](https://arxiv.org/abs/2401.03048v1) above. Its Trajectory Extractor "converts the trajectory into motion patches, which inhabit the same latent space as the video patches", and T5 encodes the text.

{: .picture}
<img width="98.5%" height="98.5%" src="{{ site.assets }}/diffusion/tora.svg"/> 

{: .caption}
Overview of the Tora architecture. Tora adds two modules: the Trajectory Extractor and the Motion-guidance Fuser. The Trajectory Extractor uses a 3D motion VAE to embed trajectory vectors into the same latent space as video patches, preserving motion information across frames. It then employs stacked convolutional layers to extract hierarchical motion features. The Motion-guidance Fuser uses adaptive normalization layers to integrate these multi-level motion conditions into the corresponding DiT blocks, so that generated videos follow the defined trajectories. Tora uses the scalability of DiT to create motion-controllable videos of extended duration. Adapted from: [Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://arxiv.org/abs/2407.21705).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113) ([github](https://github.com/facebookresearch/mae_st))
</summary>
The paper applies masked autoencoders to video, with a large gain in compute efficiency:
>  We randomly mask out
spacetime patches in videos and learn an autoencoder to reconstruct them in pixels.
Interestingly, we show that our MAE method can learn strong representations
with almost no inductive bias on spacetime (only except for patch and positional
embeddings), and spacetime-agnostic random masking performs the best. We
observe that the optimal masking ratio is as high as 90% (vs. 75% on images [31]),
supporting the hypothesis that this ratio is related to information redundancy of the
data. A high masking ratio leads to a large speedup, e.g., > 4× in wall-clock time
or even more.

{: .picture}
<img width="90.6%" height="90.6%" src="{{ site.assets }}/diffusion/masked_autoencoders.svg"/> 

{: .caption}
<b>Masked Autoencoders as spatiotemporal learners</b>. A large subset (<i>e.g.</i>, 90%) of random patches in spacetime is masked. An encoder operates on the set of visible patches. A small decoder then processes the full set of encoded patches and mask tokens to reconstruct the input. Except for patch and positional embeddings, <i>neither the encoder, the decoder, nor the masking strategy, has any spatiotemporal inductive bias</i>. Adapted from: [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113).

</details>
{::options parse_block_html="false" /}

<br/>
The last ones address multi-dimensional parallelism for large models such as Sora:

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Scaling Diffusion Transformers to 16 Billion Parameters](https://arxiv.org/abs/2407.11633)</summary>

DiT-MoE is a sparse [mixture of experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) version of DiT, with shared expert routing and an expert-level balance loss. It scales to 16.5B parameters, matches dense DiTs with less compute at inference, and reaches an FID of 1.80 on ImageNet 512x512.

{: .picture}
<img width="96.6%" height="96.6%" src="{{ site.assets }}/diffusion/dit_moe.svg"/> 

{: .caption}
<b>Overview of the DiT-MoE architecture.</b> DiT-MoE is built upon the DiT and composed of MoE-inserted Transformer blocks. In each block, the MLP is replaced by a sparsely activated mixture of MLPs. The right subfigure shows the MoE layer with the shared expert strategy. Adapted from: [Scaling Diffusion Transformers to 16 Billion Parameters](https://arxiv.org/abs/2407.11633).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://www.arxiv.org/abs/2408.10188)</summary>

LongVILA trains long-context visual-language models for video understanding in five stages. The first three come from [VILA: On Pre-training for Visual Language Models](https://arxiv.org/abs/2312.07533v2): multi-modal alignment, large-scale pre-training and short supervised fine-tuning. Stage 4 extends the context of the LLM progressively, from 8K to 262K tokens, and stage 5 fine-tunes the model on long videos with Multi-Modal Sequence Parallelism (MM-SP), which follows the 2D attention of [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719) and [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485). MM-SP trains with a context of 2M tokens on 256 GPUs. It is 2.1x to 5.7x faster than ring-style sequence parallelism, and 1.1x to 1.4x faster than Megatron with context and tensor parallelism.

{: .picture}
<img width="99.1%" height="99.1%" src="{{ site.assets }}/diffusion/longvilla.svg"/> 

{: .caption}
Workflow of Multi-Modal Sequence Parallelism (MM-SP) with a Batch Size (bs) of 4 and a Sequence Parallel Size (SP\_Size) of 3. To support sequence parallelism for multi-modal inputs, MM-SP uses a custom sharding strategy that balances the workload and is compatible with SP communication. Adapted from: [LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://arxiv.org/abs/2408.10188).

</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span"> [PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models](https://arxiv.org/abs/2405.14430) and [xDiT github](https://github.com/xdit-project/).</summary>

PipeFusion is a pipeline-parallel inference method for DiTs. It splits the image into patches and the network layers across GPUs, and pipelines the patches through the layers. Consecutive diffusion steps have similar inputs, so each patch can take the context of the other patches from the previous step, i.e. one-step stale feature maps, which needs less communication than tensor parallelism, sequence parallelism and DistriFusion. xDiT is a parallel inference engine for DiTs. It combines Unified Sequence Parallelism (USP: Ulysses and Ring Attention), PipeFusion, CFG parallelism (the two model runs of classifier-free guidance on different GPUs) and data parallelism, and it runs image models such as Stable Diffusion 3, Flux, PixArt and HunyuanDiT, and video models such as CogVideoX and Latte.

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/diffusion/pipefusion.svg"/> 

{: .caption}
(a) DistriFusion replicates DiT parameters on two devices. It splits an image into 2 patches and employs asynchronous allgather for activations of every layer. (b) PipeFusion shards DiT parameters on two devices. It splits an image into 4 patches and employs asynchronous P2P for activations across two devices. In both panels, the orange and blue blocks run on the two devices. Adapted from: [PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models](https://arxiv.org/abs/2405.14430).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> [Open-Sora, from HPC-AI Tech](https://github.com/hpcaitech/Open-Sora/tree/main)</summary>

An open-source reproduction of [Sora](https://openai.com/index/video-generation-models-as-world-simulators/). The notes below refer to version 1.2, and come from its [docs](https://github.com/hpcaitech/Open-Sora/tree/v1.2.0/docs), in particular the [technical report](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md):
- [Acceleration](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/acceleration.md#accelerated-transformer) comes from FlashAttention, a fused layer norm kernel, kernels compiled by ColossalAI, ZeRO and gradient checkpointing. [Sequence parallelism](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md#sequence-parallelism) follows Ulysses, and it is used only for inference.
- The [attention](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/acceleration.md#efficient-stdit) is ST-DiT rather than full 3D attention, as ST-DiT is more efficient as the number of frames grows.
- Data processing is described in the [Data Processing](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/data_processing.md) and [Datasets](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/datasets.md) pages.
- T5 encodes the text, and a VAE the video. The 2D VAE is initialized from the VAE of SDXL, and the 3D VAE follows the architecture of Magvit-v2 (see the [VAE Report](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/vae.md)). The [video compression network](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md#video-compression-network) of the previous version was an 83M 2D VAE that compressed space by 8x8, and kept 1 frame in every 3. To improve quality, version 1.2 compresses space by 8x8 and then time by 4x.
- The VAE trains in three stages: (1) the 2D VAE is frozen, and an identity loss trains the 3D VAE to match its features; (2) without the identity loss, only the 3D VAE trains; (3) the whole VAE trains to reconstruct the videos. The diffusion model then trains in [3 stages](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md#more-data-and-better-multi-stage-training) of increasing data quality, to make the best use of a limited compute budget.
- Diffusion uses [rectified flow](https://arxiv.org/abs/2209.03003) instead of [DDPM](https://arxiv.org/abs/2006.11239) ([source](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md#rectified-flow-and-model-adaptation)).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> [Movie Gen: A Cast of Media Foundation Models research paper](https://ai.meta.com/static-resource/movie-gen-research-paper)</summary>

One of the most detailed technical reports on a large video generation model. The model is a transformer that follows LLaMa3, with cross-attention to the text, adaptive layer norm for the timestep and full bidirectional attention, and it is trained with flow matching instead of the DDPM loss, on up to 6,144 H100 GPUs. The models of the report perform text-to-video synthesis, video personalization, video editing, video-to-audio generation and text-to-audio generation. "Our largest video generation model is a 30B parameter transformer trained with a maximum context length of 73K video tokens, corresponding to a generated video of 16 seconds at 16 frames-per-second". Section 3.1.6, "Model Scaling and Training Efficiency" (details in Appendix A.2), combines 4 parallelism methods: fully sharded data parallelism, tensor parallelism, sequence parallelism and [context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html), and the implementation overlaps communication with computation to minimize the exposed communication time.
</details>
{::options parse_block_html="false" /}
