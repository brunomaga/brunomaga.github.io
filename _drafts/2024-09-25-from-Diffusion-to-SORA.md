---
layout: post
title:  "[DRAFT] from diffusion models to large-scale SORA"
categories: [machine learning, diffusion, SORA]
tags: [machinelearning]
---


The diffusion models have first been introducted by [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Since then, many models revolutionized ML, including 
> include GLIDE and DALL-E 2 by OpenAI, Latent Diffusion by the University of Heidelberg and ImageGen by Google Brain.


This first part is a summary of the [huggingface post](https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process) and [Lillian Weng post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) on diffusion models.

Quoting the original paper:

> A diffusion probabilistic model
(which we will call a “diffusion model” for brevity) is a parameterized Markov chain trained using
variational inference to produce samples matching the data after finite time. Transitions of this chain
are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the
data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of
small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional
Gaussians too, allowing for a particularly simple neural network parameterization.

> Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise.


{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/from-Diffusion-to-SORA/diffusion.png"/> 

{: style="text-align:center; font-size: small;"}
The diffusion model. source: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).


A diffusion model is a $$T$$-step Markov chain, that trains a forward and a backward step as:
- a forward diffusion pass $$q$$ that performs $$T$$ steps ($$T=1000$$ in the paper), where each step gradually adds gaussian noise to previous step (for $$t=0$$ we use the initial image $$x_0$$), so that the output of the forward pass will be noise. Given a data point $$\mathbf{x}_0 \sim q(\mathbf{x})$$, we add a small gaussian noise at each of the $$T$$ steps, producing noisy samples $$x_0, ..., x_T$$. When $$T \rightarrow \infty$$, then $$x_T$$ is equivalent to an isotropic Gaussian distribution. The step sizes are controlled by a variance $$\beta_t \in (0,1)$$ at every step $$t$$. The forward ($$q$$) function is:

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$

- a **learnable** backward/reverse diffusion step $$p_{\theta}$$, where the network learns to gradually (per step) denoise the pure noise, in a way that we can recreate a true sample from a Gaussian noise input $$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$:

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$ 

There's an interesting property in the forward pass, demonstrated by [Sohl-Dickstein et al.](https://arxiv.org/abs/1503.03585): because the sum of Gaussians is also a gaussian, we can sample $$x_t$$ at any time $$t$$ direction conditioned on $$x_0$$ instead of $$x_{t-1}$$. Ie we don't have to compute $$q$$ for every timestep until we reach $$x_t$$. So we have that:

$$
q (x_t \mid x_0) = \mathcal{N} (x_t ; \sqrt{\hat{\alpha}} x_0, (1-\hat{\alpha}) \mathbf{I})
$$

where $$\alpha_t = 1 - \beta_t$$ and $$\hat{\alpha}_t = \prod_{s=1}^t \alpha_t$$.

The rationale of the diffusion model is that, given a large-enough $$T$$ and a well-trained model, we end up with an isotropic Gaussian distribution at $$t=T$$. An [isotropic gaussian](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) is one where the covariance matrix is represented by the simplified matrix $$Σ=𝜎^2I$$, where $$𝜎^2 \in \mathbb{R}$$ is the variance and $$I$$ is the identity matrix. There are few motivations for wanting this property: (1) have independent dimensions in the multivariate distribution; (2) reducing the computation by having a covariance matrix whose free parameers grow linearly - not quadratically - with the dimensionality, therefore easier to learn; and (3) the conjugate prior of a multivariate normal distribution with a unit variance is an isotropic normal. 

The mathematical formulation is detailed in section 2 of the paper and the [hugging face post](https://huggingface.co/blog/annotated-diffusion#in-more-mathematical-form), but the best resource I found is [Lillian Weng's post on diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

Note that the original assumed $$𝜎=1$$, however this parameter was also learnt in [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). But for the time being, assume that the neural network only needs to learn mean of the gaussian distribution. 

In order to train $$p_\theta$$ and $$q$$, we assume the combination of $$q$$ and $$p_\theta$$ to be a [variational auto-encoder](https://arxiv.org/abs/1312.6114), and we can then use the evidence lower bound (ELBO) aka variational lower bound to minimize the log-likelihood of the model output with respect to the input $$x_0$$. The log of the product of losses can then be represented as a sum of terms, which [can be minimized by the KL divergence between 2 gaussian distributions](https://huggingface.co/blog/annotated-diffusion#defining-an-objective-function-by-reparametrizing-the-mean).

## Original implementation

We'll define our data loader from the `tiny-imagenet` dataset containining several 64x64 RGB images:

```python
    dataset = load_dataset("zh-plus/tiny-imagenet", split='train')
    transform = lambda image: (pil_to_tensor(image)/255)*2-1 # normalize [-1,1]
    images = [ transform(image) for image in dataset["image"]]
    batch_size, channels, image_size = 16, images[0].shape[0], images[0].shape[1]
    dataloader = DataLoader(images, batch_size=batch_size, sampler=DistributedSampler(images))
```

Now we'll use the U-net model of the originl publications available in the `diffusers` package, and define an the optimizer:

```python
    model = diffusers.UNet2DModel(in_channels=channels, out_channels=channels).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = SGD(model.parameters(), lr=1e-2)
``` 

We start by defining the method `diffusion_constants` that define all diffusion hyper-parameters. There are [many $$\beta_t$$ variance schedulers]( https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process), here we'll use a linear $$\beta_t$$, and define the alpha variables $$\alpha_t = 1 - \beta_t$$ and $$\hat{\alpha}_t = \prod_{s=1}^t \alpha_t$$:

```python
    betas = torch.tensor([ beta_1 + (beta_T - beta_1) / timesteps * t for t in range(timesteps) ])
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
```

Now we define Sohl-Dickstein's forward diffusion pass $$q (x_t \mid x_0) = \mathcal{N} (x_t ; \sqrt{\hat{\alpha}} x_0, (1-\hat{\alpha}) \mathbf{I})$$:

```python
    @torch.no_grad()
    def q_sample(x0, t, noise=None):
        noise = noise or torch.randn_like(x0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_t = extract( sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise 
```

Now we look at the reverse process. First we define the function that predicts the model mean $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\hat{\alpha_t}}} \epsilon_\theta(x_t, t) \right)$$ as in Eq. 11:
```python
    @torch.no_grad()
    def predicted_mean(model, x, t):
        # Equation 11: use model (noise predictor) to predict the mean
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        one_over_sqrt_alphas = torch.sqrt(1.0 / torch.cumprod(alphas, axis=0))
        one_over_sqrt_alphas_t = extract(one_over_sqrt_alphas, t, x.shape)
        model_mean = one_over_sqrt_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        return model_mean
```

Now we compute the posterior variance   $$\tilde{\beta_t} = \frac{1-\hat{\alpha}_{t-1}}{1-\hat{\alpha_t}} \beta_t$$ 
```python
    @torch.no_grad()
    def posterior_variance(x, t):
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_var = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_var_t = extract(posterior_var, t, x.shape)
        return posterior_var_t
```

and finally the sampling function $$p_θ(x_{t−1}\mid x_t) = \mathcal{N}(x_{t−1}; µ_θ (x_t, t), Σ_θ(x_t, t))$$, where $$Σ_θ(x_t, t) = 𝜎^2I$$ as in section 3.2:

```python
    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        model_mean = predicted_mean(model, x, t)

        if t_index == 0:
            return model_mean
        
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
```

Finally, to generate new images we must reverse the diffusion process (from time T to time 1):

```python
    @torch.no_grad()
    def p_sample_loop(model, shape):
        img = torch.randn(shape, device=device) # start from pure noise
        for i in range(timesteps,0,-1):
            img = p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
        return img
```

In their implementation, the authors picked the U-Net as the encoder-decoder architecture. The encoder-encoder inputs an image, passes through several downsampling and upsampling layers (with residual connections) and tries to reconstruct the original picture. 

We first start by taking a regular U-Net code, and extract dowsampling (encoder) and upsampling (decoder) blocks as in [here](https://github.com/clemkoa/u-net/blob/master/unet/unet.py). Then we add sinusoidal positional embeddings as in [here](https://huggingface.co/blog/annotated-diffusion#position-embeddings) or [RotaryPositionalEmbeddings](https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html#rotarypositionalembeddings) (original [paper](https://arxiv.org/abs/2104.09864)).