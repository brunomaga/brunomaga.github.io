---
layout: post
title:  "Diffusion models, from DDPMs to DiTs"
categories: ["ML Architectures"]
tags: ["generative models", "diffusion", "DDPM", "DiT", "video generation"]
---


[Diffusion models](https://en.wikipedia.org/wiki/Diffusion_model) date back to [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) (2015), and became popular with [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM, 2020). Like [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) and [VAEs](https://en.wikipedia.org/wiki/Variational_autoencoder), they are [generative models](https://en.wikipedia.org/wiki/Generative_model) that learn to turn noise into data samples. A diffusion model has two [Markov chains](https://en.wikipedia.org/wiki/Markov_chain): a forward process that gradually adds noise to the data, and a **learned** reverse process that removes it. The reverse transitions are learned with [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods). After training, a sampling algorithm runs the reverse chain from pure noise to generate new data.

{: .picture}
<img width="80%" height="80%" src="{{ site.assets }}/diffusion/diffusion.svg"/> 

{: .caption}
A diffusion model is a $$T$$-step Markov chain, defined by a forward process $$q$$ and a trainable reverse process $$p_\theta$$. Source: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

The derivation and the [U-Net](https://en.wikipedia.org/wiki/U-Net) code follow DDPM, the Hugging Face post [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) and the post [Lil'Log: what are diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) of Lilian Weng. The learned variance and the noise schedule for fewer steps follow [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Improved DDPM) and its [official implementation](https://github.com/openai/improved-diffusion). Equation numbers refer to DDPM unless stated otherwise. The full code is in [diffusion.py]({{ site.assets }}/diffusion/diffusion.py) and [dit.py]({{ site.assets }}/diffusion/dit.py).

## Forward Process

The forward process $$q$$ is a Markov chain of $$T$$ steps. Step $$t$$ adds Gaussian noise of variance $$\beta_t \in (0,1)$$, usually with $$\beta_1 \lt \beta_2 \lt \dots \lt \beta_T$$. The variances can be learned or, as here, fixed hyper-parameters. Starting from the data $$\mathbf{x}_0$$, the chain produces the noisy samples $$\mathbf{x}_1, \dots, \mathbf{x}_T$$ (Eq. 2):

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

Each $$\mathbf{x}_t$$ is drawn from a Gaussian with mean $$\sqrt{1-\beta_t} \mathbf{x}_{t-1}$$ and covariance $$\beta_t\mathbf{I}$$: each dimension gets independent noise of variance $$\beta_t$$. This is the same as drawing $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$ and setting $$\mathbf{x}_t = \sqrt{1-\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t}\epsilon$$. When $$T \rightarrow \infty$$, $$\mathbf{x}_T$$ becomes an [isotropic Gaussian](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic), whose covariance matrix is $$\sigma^2\mathbf{I}$$ for a scalar variance $$\sigma^2$$ and the identity matrix $$\mathbf{I}$$.

The sum of independent Gaussian variables is Gaussian. Therefore $$\mathbf{x}_t$$ can be sampled at any step $$t$$ directly from $$\mathbf{x}_0$$, without running the chain (Eq. 4):

$$
q (\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N} (\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})
$$

where $$\alpha_t = 1 - \beta_t$$ and $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$.

Eq. 4 also gives the signal left at the last step: $$\mathbf{x}_T$$ is close to pure noise $$\mathcal{N}(0, \mathbf{I})$$ only if $$\bar{\alpha}_T \approx 0$$. DDPM uses $$T=1000$$ steps and a $$\beta_t$$ that grows linearly from $$\beta_1=10^{-4}$$ to $$\beta_T=0.02$$, which gives $$\sqrt{\bar{\alpha}_T} \approx 0.006$$. The code uses $$T=100$$ steps for faster sampling. With the same $$\beta_1$$ and $$\beta_T$$, $$\sqrt{\bar{\alpha}_T} \approx 0.6$$: the training inputs $$\mathbf{x}_T$$ would keep 60% of the amplitude of $$\mathbf{x}_0$$, while sampling starts from pure noise. As in the [official implementation](https://github.com/openai/improved-diffusion) of Improved DDPM, both values are therefore scaled by $$1000/T$$, which gives $$\sqrt{\bar{\alpha}_T} \approx 0.005$$. Of the [many $$\beta_t$$ schedules](https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process), the code uses the linear one, and stores $$\alpha_t$$ and $$\bar{\alpha}_t$$:

```python
    T = 100 # the paper uses T=1000
    β_1, β_T = 0.0001 * 1000 / T, 0.02 * 1000 / T # the paper's values, scaled by 1000/T
    # arrays are 0-indexed: index i holds the value of the paper's timestep t=i+1
    β = torch.linspace(β_1, β_T, T, device=device)
    α = 1. - β
    α_cumprod = torch.cumprod(α, axis=0)
```

The function below samples Eq. 4 as $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$, with $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$:

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

Conditioned on $$\mathbf{x}_0$$, the posterior of the forward process is also a Gaussian with a closed form (Eq. 6):

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0 ) = \mathcal{N} (\mathbf{x}_{t-1} ; \tilde{\mu}_t  (\mathbf{x}_t, \mathbf{x}_0), \, \tilde{\beta}_t \mathbf{I})
$$

The **posterior variance** $$\tilde{\beta}_t$$ depends only on the noise schedule (Eq. 7):

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

The code computes it, and its logarithm, which the DiT section uses to learn the variance. With $$\bar{\alpha}_0 = 1$$, $$\tilde{\beta}_1 = 0$$, so the log is clipped to its value at the second step, as in the official implementation:

```python
    # posterior variance, and its log clipped at index 0 (where posterior_β is 0)
    α_cumprod_prev = F.pad(α_cumprod[:-1], (1, 0), value=1.0)
    posterior_β = β * (1. - α_cumprod_prev) / (1. - α_cumprod) # Eq 7
    posterior_log_β = torch.log(torch.cat([posterior_β[1:2], posterior_β[1:]]))
```

The **posterior mean** is (Eq. 7):

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

The reverse process is a Markov chain $$p_\theta(\mathbf{x}_{0:T})$$ with **learned Gaussian transitions**, starting at $$p(\mathbf{x}_T) = \mathcal{N} (\mathbf{x}_T ; \mathbf{0}, \mathbf{I})$$ (Eq. 1):

$$
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) )
$$

The true reverse step $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ depends on the whole data distribution and is unknown. A neural network approximates it with a Gaussian of mean $$\mu_\theta$$ and covariance $$\Sigma_\theta$$. The approximation is good when the $$\beta_t$$ are small, because the reverse of a small Gaussian step is close to Gaussian.

DDPM does not learn the covariance: it sets $$\Sigma_\theta(\mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$$, with $$\sigma^2_t = \beta_t$$ or $$\sigma^2_t = \tilde{\beta}_t$$, which gave similar results. The first value is optimal for data $$\mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I})$$, and the second for data fixed to one point. Improved DDPM learns $$\Sigma_\theta$$, as the DiT section shows.

The pair $$q$$ and $$p_\theta$$ forms a [variational auto-encoder](https://arxiv.org/abs/1312.6114), with a fixed encoder $$q$$ and the latent variables $$\mathbf{x}_{1:T}$$. Training minimizes the variational bound $$L$$ on the negative log-likelihood of the data, i.e. it maximizes the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) (Eq. 3):

$$
\mathbb{E} \left[ - \log p_\theta(\mathbf{x}_0) \right] \le \mathbb{E}_q \left[ - \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \right] = \mathbb{E}_q \left[ - \log p(\mathbf{x}_T) - \sum_{t \ge 1} \log \frac{p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)}{q(\mathbf{x}_t \vert \mathbf{x}_{t-1})} \right] =: L
$$

Bayes' rule rewrites each $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$ with the posteriors $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$$, which turns the bound into one term per step (Eq. 5):

$$
L = \mathbb{E}_q \bigg[ \underbrace{D_{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p(\mathbf{x}_T))}_{L_T} + \sum_{t > 1} \underbrace{D_{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} \bigg]
$$

$$L_T$$ has no learnable parameters and is a constant. $$L_0$$ is the reconstruction term of the last denoising step. Each $$L_{t-1}$$ is a [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between two Gaussians, which has a [closed form](https://huggingface.co/blog/annotated-diffusion#defining-an-objective-function-by-reparametrizing-the-mean).

With $$\Sigma_\theta = \sigma_t^2 \mathbf{I}$$, the KL divergence between $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t \mathbf{I})$$ and $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 \mathbf{I})$$ is (Eq. 8):

$$
L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2 \sigma_t^2} \| \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) - \mu_\theta(\mathbf{x}_t, t) \|^2 \right] + C
$$

where $$C$$ does not depend on $$\theta$$. A model $$\mu_\theta$$ could predict the posterior mean $$\tilde{\mu}_t$$ directly. DDPM goes one step further. It writes Eq. 4 as $$\mathbf{x}_t(\mathbf{x}_0, \epsilon) = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$, with $$\epsilon \sim \mathcal{N}(0, \mathbf{I})$$, and replaces $$\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1-\bar{\alpha}_t} \epsilon \right)$$ in $$\tilde{\mu}_t$$ (Eqs. 9 and 10):

$$
L_{t-1} - C = \mathbb{E}_{\mathbf{x}_0, \epsilon} \left[ \frac{1}{2 \sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t(\mathbf{x}_0, \epsilon) - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right) - \mu_\theta(\mathbf{x}_t(\mathbf{x}_0, \epsilon), t) \right\|^2 \right]
$$

$$\mu_\theta$$ must therefore predict $$\frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)$$ from $$\mathbf{x}_t$$. As $$\mathbf{x}_t$$ is an input of the model, the network only needs to predict the noise $$\epsilon_\theta(\mathbf{x}_t, t)$$, and the mean follows from it (Eq. 11):

$$
\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right)
$$

The network takes the noisy image $$\mathbf{x}_t$$ and the [timestep $$t$$, which it turns into a timestep embedding](https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/unets/unet_2d.py#L243). The code is:

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

DDPM notes that this objective resembles denoising score matching over several noise levels. Sampling with Eq. 11 resembles Langevin dynamics, with $$\epsilon_\theta$$ as a learned gradient of the data density.

The last term $$L_0$$ uses a discrete decoder. Pixel values are integers in $$\{0, 1, \dots, 255\}$$, scaled linearly to $$[-1, 1]$$. The likelihood of a pixel value is the probability mass of $$p_\theta$$ in a bin of width $$2/255$$ around it. The first and last bins extend to infinity (Eq. 13):

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

After training, new images are generated by running the reverse process from $$t = T$$ to $$t = 1$$ (Algorithm 2):

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

Step 4 samples $$\mathbf{x}_{t-1}$$: it adds the noise $$\mathbf{z}$$, scaled by the standard deviation $$\sigma_t$$, to the predicted mean $$\mu_\theta(\mathbf{x}_t, t)$$ of Eq. 11. The U-Net uses $$\sigma^2_t = \tilde{\beta}_t$$, and the DiT its learned variance $$\Sigma_\theta$$. As in DDPM, the sampling `model` is an [exponential moving average](https://openreview.net/forum?id=2M9CUnYnBA) (EMA) of the trained weights (see the training section):

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

The model trains on the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) dataset of 32x32 RGB images in 10 classes, with 48 images per GPU:

```python
    # load dataset from the hub
    dataset = load_dataset("uoft-cs/cifar10", split='train')
    transform = lambda image: (pil_to_tensor(image)/255)*2-1 # normalize to [-1,1]
    images = [ transform(image) for image in dataset["img"]]
    batch_size, channels, img_size = 48, images[0].shape[0], images[0].shape[-1]
    sampler = DistributedSampler(images)
    dataloader = DataLoader(images, batch_size=batch_size, sampler=sampler, drop_last=True)
```

The model is the U-Net of the `diffusers` package, which is similar to the DDPM network (the DiT is in the next section). As in DDPM, the optimizer is Adam with a learning rate of $$2 \times 10^{-4}$$, and the EMA of the weights has a decay of $$0.9999$$. As in the default `EMAModel` of `diffusers`, update $$n$$ of the EMA uses the decay $$\min \left( \frac{1+n}{10+n}, 0.9999 \right)$$. The average therefore follows the trained model closely at the start of training:

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

where $$t$$ is uniform in $$\{1, \dots, T\}$$. $$\mathcal{L}_\text{simple}$$ simplifies the bound $$L$$: the terms $$t > 1$$ are the terms $$L_{t-1}$$ of Eq. 12 without their weights, the term $$t = 1$$ approximates $$L_0$$, and the constant $$L_T$$ is dropped. The weights of Eq. 12 are larger at small $$t$$, so dropping them moves the training effort to the harder denoising tasks at large $$t$$. DDPM found that this loss gives better samples and is simpler to implement:

```python
    def unet_loss(x_t, t, noise):
        """ L_simple, Equation 14 """
        ε_θ = model(x_t, t).sample # extract tensor from UNet2DOutput class
        return F.mse_loss(ε_θ, noise) # Huber loss and MAE are also ok
```

The training algorithm is (Algorithm 1):

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

The inner loop below runs steps 2 to 5 for each batch. After each epoch, the EMA model generates and saves new images:

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

[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) replaced the U-Net with a Transformer, the diffusion transformer (DiT). DiT is a [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT): it splits its input into patches and processes them as a sequence of tokens (Figure 4 in the paper). Its input is the latent representation of the image produced by a pre-trained VAE, as in the latent diffusion models of the video section. DiT-XL/2 outperforms the earlier U-Net diffusion models on class-conditional ImageNet, with a [Fréchet inception distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) (FID) of 2.27 at 256x256. The FID falls as the model Gflops grow, from more layers, wider layers or more tokens. A smaller patch size adds tokens and Gflops but no parameters, and lowers the FID: the Gflops, not the parameter count, set the quality. Larger DiT models also reach a given FID with less training compute.

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/diffusion/dit.svg"/> 

{: .caption}
<b>The Diffusion Transformer (DiT) architecture.</b> <i>Left:</i> We train conditional latent DiT models. The input latent is decomposed into patches and processed by several DiT blocks. <i>Right:</i> Details of our DiT blocks. We experiment with variants of standard transformer blocks that incorporate conditioning via adaptive layer norm, cross-attention and extra input tokens. Adaptive layer norm works best.

In the adaLN-Zero block of the figure, adaptive layer normalization conditions each block on the timestep $$t$$ and the class label $$y$$. The scale $$\gamma$$ and shift $$\beta$$ of each layer norm are not learned parameters: an MLP regresses them from the sum of the embeddings of $$t$$ and $$y$$. adaLN-Zero also regresses scales $$\alpha$$, applied before each residual connection, and initializes them to zero, so that each block starts as the identity function. These $$\alpha$$ and $$\beta$$ are unrelated to the $$\alpha_t$$ and $$\beta_t$$ of the noise schedule.

The code implements a simpler DiT: a ViT with positional embeddings, a stack of Transformer blocks and a decoder. It leaves out the four block designs of Section 3.2 of the paper (in-context conditioning, cross-attention, adaLN and adaLN-Zero). Instead, it adds a timestep embedding to every patch, and each block is a regular Transformer `Block`: multi-head self-attention followed by a feed-forward network, as in the [GPT-lite post]({{ site.baseurl }}{% post_url 2023-02-28-gptlite %}). Each token is a flattened patch of $$p \times p \times C$$ values, for a patch size $$p$$ and $$C$$ channels, with no linear patch embedding. The positional and timestep embeddings are learned look-up tables (`nn.Embedding`), where DiT uses sine-cosine positional embeddings. The decoder is a layer norm and a linear layer. It outputs $$p \times p \times 2C$$ values per token: the predicted noise and a value that sets the variance, for each channel and pixel of the patch.

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

`patchify` splits an image into the patches that form the token sequence, and `unpatchify` turns the output patches back into images:

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

The forward pass adds the positional and timestep embeddings to the patches (and the class embedding of the next section), and runs the Transformer blocks and the decoder:

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

The DiT also learns the covariance $$\Sigma_\theta$$. $$\mathcal{L}_\text{simple}$$ does not depend on $$\Sigma_\theta$$, so the model must also optimize the bound $$L$$. Following Improved DDPM, the network outputs a vector $$v$$ with one value per dimension. $$v$$ interpolates the log-variance between the two choices of DDPM, $$\beta_t$$ and $$\tilde{\beta}_t$$ (Eq. 15 of Improved DDPM):

$$
\Sigma_\theta(\mathbf{x}_t, t) = \exp \left( v \log \beta_t + (1-v) \log \tilde{\beta}_t \right)
$$

As in the official implementation, the code maps the raw network output `v` from $$[-1, 1]$$ to `frac` $$\in [0, 1]$$, the $$v$$ of Eq. 15:

```python
    def learned_Σ_θ(v, t):
        """ Improved DDPM, Equation 15: interpolate log(Σ_θ) between log(β_t) and log(posterior_β_t) """
        frac = (v + 1) / 2 # from the model output in [-1, 1] to [0, 1], as in the official implementation
        log_β_t = torch.log(β)[t][:, None, None, None]
        posterior_log_β_t = posterior_log_β[t][:, None, None, None]
        return torch.exp(frac * log_β_t + (1 - frac) * posterior_log_β_t)
```

Improved DDPM trains $$\epsilon_\theta$$ with $$\mathcal{L}_\text{simple}$$, and $$\Sigma_\theta$$ with the bound $$L_\text{vlb}$$, the $$L$$ of Eq. 5 (Eq. 16 of Improved DDPM):

$$
\mathcal{L}_\text{hybrid} = \mathcal{L}_\text{simple} + \lambda L_\text{vlb}
$$

$$\lambda = 0.001$$ keeps $$L_\text{vlb}$$ from overwhelming $$\mathcal{L}_\text{simple}$$, and a stop-gradient on $$\mu_\theta$$ makes $$L_\text{vlb}$$ train only $$\Sigma_\theta$$. Each $$L_{t-1}$$ is the closed-form KL divergence between two Gaussians ([Meta's DiT implementation](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py#L682) has another implementation), and $$L_0$$ is the negative log-likelihood of the decoder of Eq. 13. Each image samples one timestep, while $$L_\text{vlb}$$ sums $$T$$ terms. The code therefore estimates the bound as $$T$$ times the term of the sampled timestep, in bits per dimension as in the paper:

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

The model above is unconditional: trained on CIFAR-10, it generates an image of a random class, and the class cannot be chosen. A **conditional** model also takes extra information $$y$$, such as a class, a text, an image or audio, and learns $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, y)$$, where $$\epsilon_\theta$$ and $$\Sigma_\theta$$ depend on $$y$$. The DiT of the paper is class-conditional. The DiT code takes an optional class `label`. When created with `num_labels` classes, it learns a class embedding and adds it to all patches of an image, like the timestep embedding:

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

The training script trains an unconditional model. Passing the CIFAR-10 label of each image to the model trains a class-conditional one.

[Classifier-free guidance](https://arxiv.org/abs/2207.12598) improves the sample quality of a conditional model, at the cost of sample diversity. Training replaces the label $$y$$ by a learned null embedding $$\emptyset$$ in a fraction of the samples (10% to 20% work best). One model thus learns both the conditional $$\epsilon_\theta(\mathbf{x}_t, y)$$ and the unconditional $$\epsilon_\theta(\mathbf{x}_t, \emptyset)$$. Sampling extrapolates from the unconditional prediction towards the conditional one, with a guidance scale $$s \gt 1$$ ($$s = 1$$ is plain conditional sampling), in the notation of DiT:

$$
\hat{\epsilon}_\theta(\mathbf{x}_t, y) = \epsilon_\theta(\mathbf{x}_t, \emptyset) + s \left( \epsilon_\theta(\mathbf{x}_t, y) - \epsilon_\theta(\mathbf{x}_t, \emptyset) \right)
$$

Each sampling step runs the model twice, with and without the label, which doubles the sampling cost.

## Video Diffusion and 3D Attention

Diffusion on pixels is expensive at high resolution. The [latent diffusion model](https://arxiv.org/abs/2112.10752) (LDM) runs diffusion in the latent space of a pre-trained [autoencoder](https://en.wikipedia.org/wiki/Autoencoder), such as a VAE. The latent is smaller than the image, which cuts the cost of training and sampling. A video holds many frames per sample, so video models compress their input in both space and time:

{: .picture}
<img width="90%" height="90%" src="{{ site.assets }}/diffusion/sora_vae.svg"/> 

{: .caption}
An overview of the latent space compression in SORA. The pre-processing step "turns videos into [visual] patches by first compressing videos into a lower-dimensional latent space and subsequently decomposing the representation into spacetime patches". Source: [Video generation models as world simulators, OpenAI](https://openai.com/index/video-generation-models-as-world-simulators/)

The second problem is attention, which must relate patches within a frame (space) and across frames (time). For a latent video of shape $$B \times F \times H \times W \times C$$ (batch, frames, height and width in patches, and channels), there are two main designs:
- **Spatio-temporal attention** factorizes the attention. A spatial attention relates the patches of each frame, on the input reshaped to $$(B F) \times (H W) \times C$$. A temporal attention then relates the patches at the same position across frames, on the input reshaped to $$(B H W) \times F \times C$$. One spatial and one temporal attention connect any two patches in two hops.
- **Full 3D attention** relates all patches of all frames at once, on the input reshaped to $$B \times (F H W) \times C$$.

The cost of attention grows with the square of the sequence length. Per layer, spatio-temporal attention computes $$B F (H W)^2 + B H W F^2$$ attention scores, and full 3D attention computes $$B (F H W)^2$$. That is $$F H W / (H W + F)$$ times more, or about $$F$$ times more when $$H W \gg F$$. Spatio-temporal attention is cheaper, and its spatial layers can come from a pre-trained text-to-image model. Full 3D attention relates any two patches in one layer. Large recent models use it (CogVideoX and Movie Gen, below), and their long sequences need sequence or context parallelism.

Training can also shorten the sequence by dropping tokens. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (MAE) trains a ViT encoder on a random 25% of the patches of each image, which speeds up training by 3x or more. [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113) keeps only 10% of the spacetime patches of a video. [Fast Training of Diffusion Models with Masked Transformers](https://arxiv.org/abs/2306.09305) (MaskDiT) applies the idea to diffusion: it masks a large fraction of the patches during training (e.g. 50%), and matches DiT in about 30% of its training time. [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304) also drops random tokens, together with sequence packing, to speed up training.

{: .picture}
<img width="70%" height="70%" src="{{ site.assets }}/diffusion/masked_autoencoders_cropped.svg"/> 

{: .caption}
An illustration of a masked autoencoder that randomly picks 10% of the spacetime patches of a video. These patches retain enough representative power to reconstruct the original video. Source: [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113)

## Further Reading

The following are U-Net diffusion models for text-to-image and text-to-video:

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Imagen: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)</summary>
A U-Net text-to-image model. Its "key discovery is that generic large language models (e.g. T5), pretrained on text-only corpora, are surprisingly effective at encoding text for image synthesis: increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model".
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127)</summary>
A U-Net latent diffusion model for text-to-video and image-to-video generation, based on Stable Diffusion 2.1. It trains in 3 stages: (1) text-to-image pretraining, (2) video pretraining on a large low-resolution dataset, and (3) high-resolution finetuning on a small set of high-quality videos. The paper describes its data curation, which removes scene cuts, static scenes (low optical flow) and clips with much written text (found by OCR).
<!-- Captions are generated by using a model to describe the mid frame of the video, and V-BLIP to generate captions from video, and an LLM to summarize the previous 2 captions. -->
To train on videos, it adds temporal layers to a pre-trained image model, as in [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818):
> We first pre-train an LDM on images only; then, we turn the image generator into a video generator by introducing a temporal dimension to the latent space diffusion model and fine-tuning on encoded image sequences, i.e., videos.

{: .picture}
<img width="68%" height="68%" src="{{ site.assets }}/diffusion/align_your_latents.svg"/> 

{: .caption}
**Left:** We turn a pre-trained LDM into a video generator by inserting temporal layers that learn to align frames into temporally consistent sequences. During optimization, the image backbone $$\theta$$ remains fixed and only the parameters $$\phi$$ of the temporal layers $$l^i_\phi$$ are trained, cf. Eq. (2). **Right:** During training, the base model $$\theta$$ interprets the input sequence of length $$T$$ as a batch of images. For the temporal layers $$l^i_\phi$$, these batches are reshaped into video format. Their output $$z'$$ is combined with the spatial output $$z$$, using a learned merge parameter $$\alpha$$. During inference, skipping the temporal layers ($$\alpha^i_\phi=1$$) yields the original image model. For illustration purposes, only a single U-Net Block is shown. $$c_S$$ is optional context frame conditioning, when training prediction models (Sec. 3.2). Caption and figure adapted from: [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818).
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation](https://arxiv.org/abs/2311.17117)</summary>
A U-Net model that animates the person in a reference image with a pose sequence (a moving stick figure). A VAE and CLIP encode the reference image. The model has two U-Nets. ReferenceNet merges detail features of the reference image into the denoising U-Net through spatial attention. The denoising U-Net generates the video from the pose sequence, encoded by a *pose guider*. Each of its blocks runs spatial attention, cross-attention to the CLIP features, and temporal attention: the spatio-temporal attention of the video section.

{: .picture}
<img width="90%" height="90%" src="{{ site.assets }}/diffusion/anymate_anyone.svg"/> 

{: .caption}
The overview of our method. The pose sequence is initially encoded using Pose Guider and fused with multi-frame noise, followed by the Denoising UNet conducting the denoising process for video generation. The computational block of the Denoising UNet consists of Spatial-Attention, Cross-Attention, and Temporal-Attention, as illustrated in the dashed box on the right. The integration of reference image involves two aspects. Firstly, detailed features are extracted through ReferenceNet and utilized for Spatial-Attention. Secondly, semantic features are extracted through the CLIP image encoder for Cross-Attention. Temporal-Attention operates in the temporal dimension. Finally, the VAE decoder decodes the result into a video clip.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[CyberHost: Taming Audio-driven Avatar Diffusion Model with Region Codebook Attention](https://arxiv.org/abs/2409.01876)</summary>
An audio-driven U-Net model that animates a person from a reference image. It reuses the reference net of Animate Anyone. Its Region Codebook Attention mechanism "improves the generation quality of facial and hand animations by integrating fine-grained local features with learned motion pattern priors".
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack](https://arxiv.org/abs/2309.15807)</summary>
A U-Net latent diffusion model for text-to-image. Fine-tuning on a small set of 2,000 high-quality images improves its samples: "in order to align the model towards highly aesthetic generations – quality matters significantly more than quantity in the fine-tuning dataset".
</details>
{::options parse_block_html="false" /}

<br/>
The following are conditional models based on DiT:

{::options parse_block_html="true" /}
<details> <summary markdown="span">[OmniGen: Unified Image Generation](https://arxiv.org/abs/2409.11340)</summary>
OmniGen is only a VAE and a Transformer, initialized from the Phi-3 LLM. One model performs text-to-image generation, image editing, subject-driven generation and visual-conditional generation.

{: .picture}
<img width="68%" height="68%" src="{{ site.assets }}/diffusion/omnigen.svg"/> 

{: .caption}
The framework of OmniGen. Texts are tokenized into tokens, while input images are transformed into embedding via VAE. OmniGen can accept free-form multi-modal prompts and generate images through the rectified flow approach.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models](https://arxiv.org/abs/2409.10695)</summary>
A text-to-image model whose Transformer blocks mirror those of a frozen Llama3-8B LLM: each image block takes the hidden states of the corresponding LLM layer as conditioning.

{: .picture}
<img width="68%" height="68%" src="{{ site.assets }}/diffusion/playground_v3.svg"/> 

{: .caption}
PGv3 model structure overview simplified. There have been a lot of research efforts aimed at leveraging LLMs to improve t2i models’ prompt-following abilities. These efforts typically involve either using LLMs as the text encoder to replace the commonly-used T5 and CLIP, or employing LLMs to adapt or revise prompts for a pretrained T5/CLIP-based t2i model \[19, 56, 47, 24, 38, 3\]. In PGv3, we adopt the former approach, i.e. we don’t use any NLP encoders like T5 and CLIP, and solely rely on Llama3-8B to provide text conditioning to our diffusion model, from the initial stage of the model training.

</details>
{::options parse_block_html="false" /}
 
<br/>
The following works address video diffusion and 3D attention:

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)</summary>
NaViT (Native Resolution ViT) packs the patches of several images into one sequence ("Patch n' Pack"). It thus trains on images of any resolution and aspect ratio, without resizing. Packing also allows token dropping at a different rate per image.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072)</summary>
A DiT for text-to-video. T5 encodes the text, and a 3D causal VAE compresses the video in space and time. The video patches are unfolded into one long sequence and concatenated with the text embeddings. *Expert* Transformer blocks process this sequence, with separate adaptive layer norms for the text and the video tokens. The model uses full 3D attention instead of separate spatial and temporal attention. Its output is unpatchified and decoded by the 3D VAE.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048v1)</summary>

A DiT for video generation: unconditional, class-conditional and text-to-video. It extracts spatio-temporal tokens from the input video and models their distribution in latent space with Transformer blocks. The paper tests the design choices (patch embedding, model variant, timestep and class injection, temporal positional embedding and learning strategy) and reports the best ones. It compares four model variants:
- Variant 1 interleaves spatial and temporal Transformer blocks.
- Variant 2 runs all spatial blocks before all temporal blocks ("late fusion").
- Variant 3 computes spatial and then temporal attention inside each block.
- Variant 4 splits the attention heads of each block between the spatial and the temporal dimension.

Variant 1 performs best.

{: .picture}
<img width="70%" height="70%" src="{{ site.assets }}/diffusion/latte.svg"/> 

{: .caption}
<b>The pipeline of Latte for video generation</b>. Four model variants of Latte are proposed to efficiently capture spatio-temporal information in videos. Each block depicted in light orange represents a Transformer block. The standard Transformer blocks (described in Fig. 4b) are employed in (a) and (b). Meanwhile, (c) and (d) employ our respective Transformer block variants. For the sake of simplicity, encoding and decoding processes for VAE are not shown in the diagram.

The paper also compares two video patch embeddings. The uniform frame patch embedding (a) embeds each frame into tokens as in ViT. The compression frame patch embedding (b) extends the ViT patch embedding to 3D, and extracts tubes along the time dimension. Embedding (a) performs better:


{: .picture}
<img width="57%" height="57%" src="{{ site.assets }}/diffusion/latte2.svg"/> 

{: .caption}
<b>The video clip patch embedding</b>. (a) We sample <i>F</i> frames and embed each individual video frame into tokens using the method described in ViT. (b) We consider capturing temporal information and then extending the ViT patch embedding method from 2D to 3D and subsequently extracting tubes along the temporal dimension. For ease of understanding, we use the original video clip here to demonstrate the patch embedding method. The patch embedding in the latent space of videos follows the same processing approach.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://arxiv.org/abs/2407.21705) ([webpage](https://ali-videoai.github.io/tora_video/))</summary>

Tora generates videos guided by trajectories, images, texts or their combinations. It "employs the Spatial-Temporal Diffusion Transformer (ST-DiT) from OpenSora as its foundational model": each block runs a spatial and then a temporal attention, as variant 3 of Latte. Its Trajectory Extractor "converts the trajectory into motion patches, which inhabit the same latent space as the video patches". T5 encodes the text.

{: .picture}
<img width="76%" height="76%" src="{{ site.assets }}/diffusion/tora.svg"/> 

{: .caption}
Overview of the Tora Architecture. We introduce two novel modules: the Trajectory Extractor and the Motion-guidance Fuser. The Trajectory Extractor uses a 3D motion VAE to embed trajectory vectors into the same latent space as video patches, preserving motion information across frames. It then employs stacked convolutional layers to extract hierarchical motion features. The Motion-guidance Fuser utilizes adaptive normalization layers to integrate these multi-level motion conditions into the corresponding DiT blocks, ensuring that generated videos consistently follow defined trajectories. Our method leverages the scalability of DiT, enabling the creation of motion-controllable videos of extended duration.

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
<img width="66%" height="66%" src="{{ site.assets }}/diffusion/masked_autoencoders.svg"/> 

{: .caption}
<b>Masked Autoencoders as spatiotemporal learners</b>. We mask a large subset (<i>e.g.</i>, 90%) of random patches in spacetime. An encoder operates on the set of visible patches. A small decoder then processes the full set of encoded patches and mask tokens to reconstruct the input. Except for patch and positional embeddings, <i>neither the encoder, the decoder, nor the masking strategy, has any spatiotemporal inductive bias</i>.

</details>
{::options parse_block_html="false" /}

<br/>
The following works address multi-dimensional parallelism for large models such as Sora:

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Scaling Diffusion Transformers to 16 Billion Parameters](https://arxiv.org/abs/2407.11633)</summary>

DiT-MoE is a sparse [mixture of experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) version of DiT, with shared expert routing and an expert-level balance loss. It scales to 16.5B parameters, matches dense DiTs with less compute at inference, and reaches an FID of 1.80 on ImageNet 512x512.

{: .picture}
<img width="70%" height="70%" src="{{ site.assets }}/diffusion/dit_moe.svg"/> 

{: .caption}
<b>Overview of the DiT-MoE architecture.</b> Generally, DiT-MoE is built upon the DiT and composed of MoE-inserted Transformer blocks. In between, we replace the MLP with a sparsely activated mixture of MLPs. The right subfigure demonstrates the details of MoE layer integration of the shared expert strategy.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://www.arxiv.org/abs/2408.10188)</summary>

LongVILA trains long-context visual-language models, for video understanding, in five stages. The first three come from [VILA: On Pre-training for Visual Language Models](https://arxiv.org/abs/2312.07533v2): multi-modal alignment, large-scale pre-training and short supervised fine-tuning. Stage 4 extends the context of the LLM progressively, from 8K to 262K tokens. Stage 5 fine-tunes on long videos with Multi-Modal Sequence Parallelism (MM-SP), which follows the 2D attention of [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719) and [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485). MM-SP trains with a context of 2M tokens on 256 GPUs. It is 2.1x to 5.7x faster than ring-style sequence parallelism, and 1.1x to 1.4x faster than Megatron with context and tensor parallelism.

{: .picture}
<img width="68%" height="68%" src="{{ site.assets }}/diffusion/longvilla.svg"/> 

{: .caption}
Workflow of Multi-Modal Sequence Parallelism with a Batch Size (bs) of 4 and a Sequence Parallel Size (SP\_Size) of 3. To facilitate sequence parallelism for multi-modal inputs, we developed a customized sharding strategy that ensures balanced workload distribution and compatibility with SP communcation.

</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span"> [PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models](https://arxiv.org/abs/2405.14430) and [xDiT github](https://github.com/xdit-project/).</summary>

PipeFusion is a pipeline-parallel inference method for DiTs. It splits the image into patches and the network layers across GPUs, and pipelines the patches through the layers. Consecutive diffusion steps have similar inputs, so each patch takes the context of the other patches from the previous step (one-step stale feature maps). This needs less communication than tensor parallelism, sequence parallelism and DistriFusion. xDiT is a parallel inference engine for DiTs. It combines Unified Sequence Parallelism (USP: Ulysses and Ring Attention), PipeFusion, CFG parallelism (the two model runs of classifier-free guidance on different GPUs) and data parallelism.

{: .picture}
<img width="70%" height="70%" src="{{ site.assets }}/diffusion/pipefusion.svg"/> 

{: .caption}
(a) DistriFusion replicates DiT parameters on two devices. It splits an image into 2 patches and employs asynchronous allgather for activations of every layer. (b) PipeFusion shards DiT parameters on two devices. It splits an image into 4 patches and employs asynchronous P2P for activations across two devices.

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> [Open-Sora, from HPC-AI Tech](https://github.com/hpcaitech/Open-Sora/tree/main)</summary>

An open-source reproduction of [Sora](https://openai.com/index/video-generation-models-as-world-simulators/). The notes below refer to version 1.2, from its [docs](https://github.com/hpcaitech/Open-Sora/tree/v1.2.0/docs), in particular the [technical report](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md):
- [Acceleration](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/acceleration.md#accelerated-transformer) comes from FlashAttention, a fused layer norm kernel, kernels compiled by ColossalAI, ZeRO and gradient checkpointing. [Sequence parallelism](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md#sequence-parallelism) follows Ulysses and is used only for inference.
- [Attention](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/acceleration.md#efficient-stdit) is ST-DiT instead of full 3D attention, as ST-DiT is more efficient as the number of frames grows.
- Data processing is described in the [Data Processing](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/data_processing.md) and [Datasets](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/datasets.md) pages.
- T5 encodes the text, and a VAE the video. The 2D VAE is initialized from the VAE of SDXL, and the 3D VAE follows the architecture of Magvit-v2 (see the [VAE Report](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/vae.md)). The [video compression network](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md#video-compression-network) of the previous version was an 83M 2D VAE that compressed space by 8x8, with 1 frame kept in every 3. Version 1.2 compresses space by 8x8 and then time by 4x.
- The VAE trains in 3 stages. (1) The 2D VAE is frozen, and an identity loss trains the 3D VAE to match its features. (2) Without the identity loss, only the 3D VAE trains. (3) The whole VAE trains to reconstruct the videos. The diffusion model then trains in [3 stages](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md#more-data-and-better-multi-stage-training) of increasing data quality.
- Diffusion uses [rectified flow](https://arxiv.org/abs/2209.03003) instead of [DDPM](https://arxiv.org/abs/2006.11239) ([source](https://github.com/hpcaitech/Open-Sora/blob/v1.2.0/docs/report_03.md#rectified-flow-and-model-adaptation)).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> [Movie Gen: A Cast of Media Foundation Models research paper](https://ai.meta.com/static-resource/movie-gen-research-paper)</summary>

One of the most detailed technical reports on a large video generation model. The model is a transformer that follows LLaMa3, with cross-attention to the text, adaptive layer norm for the timestep and full bidirectional attention. It is trained with flow matching instead of the DDPM loss, on up to 6,144 H100 GPUs. The models perform text-to-video synthesis, video personalization, video editing, video-to-audio generation and text-to-audio generation. "Our largest video generation model is a 30B parameter transformer trained with a maximum context length of 73K video tokens, corresponding to a generated video of 16 seconds at 16 frames-per-second". Section 3.1.6, "Model Scaling and Training Efficiency" (details in Appendix A.2), combines 4 parallelism methods: fully sharded data parallelism, tensor parallelism, sequence parallelism and [context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html). The implementation also overlaps communication with computation, to minimize the exposed communication time.
</details>
{::options parse_block_html="false" /}
