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

and finally the sampling function $$p_θ(x_{t−1}\mid x_t) = \mathcal{N}(x_{t−1}; µ_θ (x_t, t), Σ_θ(x_t, t))$$, where $$Σ_θ(x_t, t) = 𝜎^2I$$ due to being an isotropic Gaussian distributed, as in section 3.2:

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

Now that we have the forward and reverse diffusion processes, we can perform every diffusion iteration as:

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/from-Diffusion-to-SORA/diffusion_alg1.png"/> 
<img width="45%" height="45%" src="/assets/from-Diffusion-to-SORA/diffusion_alg2.png"/> 

```python
    # Algorithm 1 line 3: sample t uniformally for every example in the batch
    t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    noise = torch.randn_like(batch)
    x_noisy = q_sample(x0=batch, t=t, noise=noise)
    predicted_noise = model(x_noisy, t)[0]
    loss = F.smooth_l1_loss(noise, predicted_noise) # Huber loss
```

## Diffusion transformers

The publication [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) introduced diffusion transformers (DiT) as a replacement that outperformes the UNet-based diffusion in scaling and accuracy measured by [Fréchet inception distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance). Because the work presented is based on image diffusion, DiT is based on [Vision Transformers (ViTs)](https://arxiv.org/abs/2010.11929), that operate on patches of images (Figure 4). VITs also also shown to have better scaling properties and accuracy than convolutional neural networks. Moreover, related to ViTs scaling, it shows that (1) ViT Gflops are strongly correlated with FID, (2) DiT Gflops are critical to improving performance, and (3) larger DiT models use large compute more efficiently.

>  We study the scaling behavior of transformers with respect to network complexity (in GFlops) vs. sample quality. We show that by constructing and benchmarking the DiT design space under the Latent Diffusion Models (LDMs) [48] framework, where diffusion models are trained within a VAE’s latent space, we can successfully replace the U-Net backbone with a transformer. 

> We further show that DiTs are scalable architectures for diffusion models: there is a strong correlation between the network complexity (measured by Gflops) vs. sample quality (measured by FID).

> Background: Gaussian diffusion models assume a forward noising process which gradually applies noise to real data $$x_0: q(x_t \mid x_0 ) = \mathcal{N} (x_t ; \sqrt{α¯}_t x_ 0,(1 − α¯t)I)$$.  By applying the reparameterization trick, we can sample $$x_t = \sqrt{α¯_t} x_0 + \sqrt{1 − α¯_t} \epsilon_t, where \epsilon_t ∼ \mathcal{N} (0, I)$$. ETC section 3.1

Here they use implement a **conditional diffusion model** that takes as input extra information such as class $$c$$, and the reverse process becomes $$p_θ(x_{t−1} \mid  x_t, c)$$, where $$\epsilon_θ$$ and $$Σ_θ$$ are conditioned on $$c$$.

> In this setting, **[classifier-free guidance](https://arxiv.org/abs/2207.12598)** can be used to encourage the sampling procedure to find $$x$$ such that $$\log p(c \mid x)$$ is high. By Bayes Rule, $$\log p(c \mid x) ∝ \log p(x \mid c) − \log p(x)$$, and hence $$∇_x \log p(c \mid x) ∝ ∇_x \log p(x \mid c) − ∇_x \log p(x)$$.  By interpreting the output of diffusion models as the score function, the DDPM sampling procedure can be guided to sample $$x$$ with high $$p(x \mid c)$$ by: $$\hat{\epsilon}_θ(x_t, c) = \epsilon_θ(x_t, ∅) + s · ∇_x \log p(x \mid c) ∝ \epsilon_θ(x_t, ∅) + s · (\epsilon_θ(x_t, c)−\epsilon_θ(x_t, ∅))$$, where $$s \gt 1$$ indicates the scale of the guidance (note that $$s = 1$$ recovers standard sampling). Evaluating the diffusion model with $$c = ∅$$ is done by randomly dropping out $$c$$ during training and replacing it with a learned “null” embedding $$∅$$. 

The paper also mentions the notion of **[lattent diffusion model](https://arxiv.org/abs/2112.10752)** where diffusion is applied on the latent space of pretrained autoencoders (e.g. VAE) instaed of the image directly. This reduces computation by training diffusion on high-resolution images by training on its compressed representation instead. 

To keep our architecture simple as simple as possible, we will ingnore the 4 variants described in DiT block design (in Section 3.2, in-context conditioning, cross-attention block, adaptive layer norm block and adaLN-Zero block) and we will use the regular PyTorch embedding `nn.Embedding` (a look-up table) instaed of the  frequency-based positional
embeddings (the sine-cosine version).

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/from-Diffusion-to-SORA/DiT.png"/> 

So we start the implementation with the boilerplace code that crops the input image into patches to be used by the visual attention module:

```python
    @staticmethod
    def patchify(x, patch_size):
        """ converts an image x into a list of patches of size patch_size x patch_size """
        B, C, H, W = x.shape
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(B, C, -1, patch_size, patch_size)
        x = x.permute(0, 2, 1, 3, 4)
        return x
```

Our ViT is simply a positional embedding layer, a block of GPT blocks and a decoder. The decoder is a layer-norm and a linear layer that outputs the shape $$p \times p \times 2C$$ (ie a mean and variance for each channel and patch). Optionally, we add VAE that we may want to use to get a latent encoding and decoding:

```python
class ViT(nn.Module):
    def __init__(self, channels, patch_size=4, n_embd=64, n_blocks=12, use_vae=False):
        super().__init__()
        self.patch_size = patch_size
        self.pos_emb = nn.Embedding(64, n_embd)
        self.blocks = nn.Sequential(*[Block(64, 64) for _ in range(n_blocks)])
        self.decoder = nn.Sequential( nn.LayerNorm(n_embd), nn.Linear(n_embd, channels) )
        self.vae = diffusers.models.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse") if use_vae else None

    def forward(self):
        if self.vae:  x = self.vae.tiled_encode(x)
        x = ViT.patchify(x, self.patch_size)
        x += self.pos_emb(torch.arange(x.shape[0], device=x.device))
        x = self.blocks(x)
        x = self.decoder(x)
        if self.vae:  x = self.vae.tiled_decode(x)
        return x
```

## SORA

We now move to the movie domain and study a text-to-video model, which is a basic implementation of [SORA](https://openai.com/index/video-generation-models-as-world-simulators/), a text-to-video, image-to-video, text-to-image (single frame video), and video-to-video model (for video editing, extension, etc).

Just as in the previous example, we also use a VAE to compress the input into a smaller latent space. However, we compress videos into spatial and temporal patches.

[Masked autoencoders (MAE)](Masked Autoencoders As Spatiotemporal Learners) have shown to be  scalable self-supervised learners for computer vision, in [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) and [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304).

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/from-Diffusion-to-SORA/sora_vae.png"/> 

{: style="text-align:center; font-size: small;"}
An overview of the lattent space in SORA. The pre-processing step "turns videos into [visual] patches by first compressing videos into a lower-dimensional latent space and subsequently decomposing the representation into spacetime patches". source: [Video generation models as world simulators, OpenAI](https://openai.com/index/video-generation-models-as-world-simulators/)

## Further reading and resources

- [Scaling Diffusion Transformers to 16 Billion Parameters](https://arxiv.org/abs/2407.11633): presents Mixture of Experts of DiTs (MoE-DiT), delivering better scaling properties, an accuracy comparable to dense DiTs, and highly optimized inference; 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/from-Diffusion-to-SORA/dit_moe.png"/> 

- [LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://www.arxiv.org/abs/2408.10188) details a pipeline of 5 steps for training long-context visual-language models. The first 3 stages are multi-modal alignment, large-scale pre-training and short supervised fine-tuning from [VILA: On Pre-training for Visual Language Models](https://arxiv.org/abs/2312.07533v2). Stage 4 is context extension for LLMs, by increasing the sequence length of input samples (ie curriculum learning) up to 262K tokens. In Stage 5, the model is fine-tuned for long video understanding with Multi-Modal Sequence Parallelism (MM-SP) based on [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485).

{: style="text-align:center; font-size: small;"}
<img width="68%" height="68%" src="/assets/from-Diffusion-to-SORA/longvilla.png"/> 

- [OmniGen: Unified Image Generation](https://arxiv.org/abs/2409.11340), for text-to-image generation
- [DALL-E 3](https://openai.com/index/dall-e-3/), for text-to-image generation
- OpenSORA implementation from HPC AI Tech : [github](https://github.com/hpcaitech/Open-Sora/tree/main) and [report](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md).
- [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
- [CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072)
- [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048v1)
- [Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://arxiv.org/abs/2407.21705) ([webpage](https://ali-videoai.github.io/tora_video/))
- [Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113) ([github](https://github.com/facebookresearch/mae_st))
- [PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models](https://arxiv.org/abs/2405.14430) and [xDIT github](https://github.com/xdit-project/). PipeFusion splits images into patches and distributes the network layers across multiple devices. It employs a pipeline parallel manner to orchestrate communication and computations. xDIT is a parallel **inference** engine of DiTs using Universal Sequence Parallelism (including Ulysses attention and Ring attention), PipeFusion, and hybrid parallelism. It applies and benchmarks xDIT to the following DiT implementations: CogVideo, Flux, Latte, HunyuanDiT, Stable Diffusion 3, Pixart-Sigma, Pixart-alpha. 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/from-Diffusion-to-SORA/pipefusion.png"/> 