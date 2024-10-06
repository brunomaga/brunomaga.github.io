---
layout: post
title:  "[DRAFT] from diffusion models to large-scale SORA"
categories: [machine learning, diffusion, SORA]
tags: [machinelearning]
---

Despite being an old idea first introduced back to 2015, diffusion models (DMs) started to be taken seriously after the refinements in the paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). Just like GANs or VAEs, the diffusion models learn to convert noise from a distribution into a data sample - the "denoising" process. Quoting the paper:

> A diffusion model is a parameterized Markov chain trained using
variational inference to produce samples matching the data after finite time. Transitions of this chain
are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the
data in the opposite direction of sampling until signal is destroyed.

> When the diffusion consists of
small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional
Gaussians too, allowing for a particularly simple neural network parameterization.

The diffusion model is characterized by three main elements: a forward process that gradually adds noise to data, a learnable reverse diffusion process that learns the denoising process, and a sampling algorithm that will generates data from pure noise using the trained model.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/from-Diffusion-to-SORA/diffusion.png"/> 

{: style="text-align:center; font-size: small;"}
A diffusion model is a $$T$$-step Markov chain, characterized by a forward process $$q$$ and a trainable reverse process $$q$$. Source: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

Let's look at those processes in detail. Credit: mathematical formulation inspired by [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), [https://huggingface.co/blog/annotated-diffusion](https://huggingface.co/blog/annotated-diffusion) and [Lil'Log: waht are diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

## Forward Process

The forward diffusion process $$q$$ performs $$T$$ steps, where each step gradually adds gaussian noise to the previous step. Given a data point $$\mathbf{x}_0$$ for $$t=0$$, we sample and add gaussian noise at each step $$t$$ of the $$T$$ steps, producing noisy samples $$\mathbf{x}_0, ..., \mathbf{x}_T$$. An important property, is that **when $$T \rightarrow \infty$$, we end up with an isotropic Gaussian distribution at $$t=T$$**. An [isotropic gaussian](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) is one where the covariance matrix $$Σ$$ is represented by $$Σ=𝜎^2I$$, where $$𝜎^2 \in \mathbb{R}$$ is the variance constant and $$I$$ is the identity matrix.

The step sizes are controlled by a variance $$\beta_t \in (0,1)$$ at every step $$t$$, that is set by an user-defined variance scheduler that sets $$\beta_t$$ such that $$0 \lt \beta_1 \lt \beta_2 \lt  ... \lt \beta_T < 1$$. The forward process $$q$$ is then represented as (Eq. 2 in paper):

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
$$

ie each new sample $$\mathbf{x}_t$$ is drawn for a Gaussian distribution with mean $$μ_t = \sqrt{1−β_t} \mathbf{x}_{t−1}$$ and variance $$\sigma^2_t = \beta_t $$. This is equivalent sampling $$\epsilon ∼ \mathcal{N}$$ and then setting $$\mathbf{x}_t = \sqrt{1−β_t} \mathbf{x}_{t−1} + \sqrt{\beta_t}\epsilon$$.


Another property in the forward process, demonstrated by [Sohl-Dickstein et al.](https://arxiv.org/abs/1503.03585), is that because the sum of Gaussians is also a gaussian, we can sample $$\mathbf{x}_t$$ at any time $$t$$, conditioned directly on $$\mathbf{x}_0$$, instead of conditioned on $$\mathbf{x}_{t-1}$$ and computed iteratively. So we have that (Eq. 4 in paper):

$$
q (\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N} (\mathbf{x}_t ; \sqrt{\bar{\alpha}} \mathbf{x}_0, (1-\bar{\alpha}) \mathbf{I})
$$

where $$\alpha_t = 1 - \beta_t$$ and $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_t$$.


We can start implementing our diffusion algorithm by defining the hyper-parameters $$\alpha$$ and for convenience, an additional $$\bar{\alpha}_t = \prod_{s=1}^t \alpha_t$$. There are [many $$\beta_t$$ variance schedulers]( https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process), but for simplicity, we will implement a linear scheduler $$\beta_t$$:

```python
    timesteps, beta_1, beta_T = 100, 0.0001, 0.02
    betas = torch.tensor([ beta_1 + (beta_T - beta_1) / timesteps * t for t in range(timesteps) ], device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
```

Now we define the Sohl-Dickstein's forward process $$q (\mathbf{x}_t \mid \mathbf{x}_0)$$ as:

```python
    @torch.no_grad()
    def q_sample(x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise 
```

## Reverse process

We now look at the **learnable** backward/reverse diffusion step. We do not know the distribution of the denoising step $$p (\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$, so we will use a neural network $$p_{\theta}$$ to approximate it. We will assume this distribution to be of a Gaussian shape with learnable mean $$\mu_\theta$$ and $$\Sigma_\theta$$ (Eq. 1 in paper):

$$
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) )
$$ 

In order to train $$p_\theta$$, we can treat the combination of $$q$$ and $$p_\theta$$ as a [variational auto-encoder](https://arxiv.org/abs/1312.6114), and we can then use the evidence lower bound (ELBO) to minimize the log-likelihood of the model output with respect to the input $$\mathbf{x}_0$$. The log of the product of losses can then be represented as a sum of terms, which [can be minimized by the KL divergence between 2 gaussian distributions](https://huggingface.co/blog/annotated-diffusion#defining-an-objective-function-by-reparametrizing-the-mean).

Just like in VAEs, we perform a **reparameterization trick so that we learn the added noise $$\epsilon_\theta(\mathbf{x}, t)$$ for noise level $$t$$** instead of prediction the mean. We can then compute the mean as (Eq 11):

 In the original paper, only the mean is trained ("We ignore the fact that the forward process variances βt are learnable by reparameterization and
instead fix them to constants (see Section 4 for details)"). The variance is set as $$\Sigma_\theta(\mathbf{x}_t, t) = \sigma^2_t I$$ and $$\sigma^2_t = \beta_t$$ or $$\sigma^2_t = \tilde{\beta}_t$$ (similar results), where $$\tilde{\beta}_t$$ is the **posterior variance** (Eq. 7 in paper):

$$
\tilde{\beta_t} = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha_t}} \beta_t
$$

simply coded as:

```python
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
```

As a side note, learning a variance was later explored by [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

To represent the mean $$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$$, the authors propose a parameterization trick (section 3.2) that allows for our model to learn the noise $$\epsilon_\theta(\mathbf{x}_t, t)$$ for step $$t$$ instead of predicting the mean. The mean can then be computed as (Eq. 11 in paper):

 $$
 \mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha_t}}} \epsilon_\theta(\mathbf{x}_t, t) \right)
 $$ 

 and coded as:
```python
    @torch.no_grad()
    def predicted_mean(model, x, t):
        betas_t = betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        one_over_sqrt_alphas = torch.sqrt(1.0 / torch.cumprod(alphas, axis=0))
        one_over_sqrt_alphas_t = one_over_sqrt_alphas[t][:, None, None, None]
        epsilon = model(x, t)
        model_mean = one_over_sqrt_alphas_t * (x - betas_t * epsilon  / sqrt_one_minus_alphas_cumprod_t)
        return model_mean
```

where `model(x, t)` is the model output $$\epsilon_\theta(\mathbf{x}_t, t)$$.

## Sampling

Once the model is trained, to generate new images we must reverse the diffusion process (from time T to time 1):

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/from-Diffusion-to-SORA/diffusion_alg2.png"/> 

The step 6 samples $$\mathbf{x}_t$$ for a step $$t$$, by summing the predicted mean $$ \mu_\theta(\mathbf{x}_t, t)$$ with noise $$\mathbf{z}_t$$ multiplied by the variance $$\sigma$$:

```python
    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        model_mean = predicted_mean(model, x, t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
```

The remaining steps are coded in a simple for loop:

```python
    @torch.no_grad()
    def p_sample_loop(model, shape):
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, timesteps)):
            img = p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
        return img
```

## Training algorithm

We'll train our model with the CIFAR10 dataset containing several 32x32 RGB images of 10 means of transportation. We'll set our batch size to 64 images per batch.

```python
    dataset = load_dataset("uoft-cs/cifar10", split='train')
    transform = lambda image: (pil_to_tensor(image)/255)*2-1 # normalize to [-1,1]
    images = [ transform(image) for image in dataset["img"]]
    batch_size, channels = 64, images[0].shape[0]
    dataloader = DataLoader(images, batch_size=batch_size, sampler=DistributedSampler(images), drop_last=True)
```

Our model will be U-net model of the original publication available in the `diffusers` package, trained with an SGD optimizer:

```python
    model = diffusers.UNet2DModel(in_channels=channels, out_channels=channels).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = SGD(model.parameters(), lr=1e-2)
``` 

As loss function, we follow the paper and use a simple mean square error (Eq. 14) between the sampled noise $$\epsilon$$ and the predicted noise $$\epsilon\_theta$$ (Huber loss or MAE are also popular choices):

$$
\mathcal{L_{simple}(\theta)} = \mathop{\mathbb{E}}_{t, \mathbf{x_0}, \epsilon}  \| \epsilon - \epsilon_\theta(\sqrt{\bar{α}_t} \mathbf{x}_0 + \sqrt{1-\bar{α}_t} \epsilon, t)\| ^2
$$

We can now put together our final training algorithm: 

{: style="text-align:center; font-size: small;"}
<img width="45%" height="45%" src="/assets/from-Diffusion-to-SORA/diffusion_alg1.png"/> 

where each training iteration (steps 2 to 6) can be coded as:

```python
    t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    noise = torch.randn_like(batch)
    x_noisy = q_sample(x0=batch, t=t, noise=noise)
    predicted_noise = model(x_noisy, t)[0]
    loss = F.mse_loss(noise, predicted_noise) # Huber loss also ok
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### U-net based diffusion models for text-to-image and text-to-video tasks

The previous implementation was generating/sampling new images from a diffusion process. One can add conditioning for ...

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Imagen: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)</summary>
A Unet-based text-to-image diffusion model, where "key discovery is that generic large language models (e.g. T5), pretrained on text-only corpora, are surprisingly effective at encoding text for image synthesis: increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model".
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127)</summary>
Presents a U-Net based diffusion model for  Text-to-Video, and  (Text-to-)Image-to-Video genarative model. Trained on 3 stages: (1) text-to-image pretraining of a diffusion model, (2) video pretraining on a large dataset at low resolution, and (3) high-resolution video finetuning on a much smaller dataset with higher-quality videos. It also emphasizes the importance and methods for data curation: e.g. avoiding cutscenes or static scenes, removing videos with large ammount of written text. 
<!-- Captions are generated by using a model to describ the mid frame of the video, and V-BLIP to generate captions from video, and an LLM to summarize the previous 2 captions. -->
To train on videos and have 3D attention they use the method presented in [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818):
> first pre-train the diffusion model on images only; then, turn the image generator into a video generator by introducing a temporal dimension to the latent space diffusion model and fine-tuning on encoded image sequences, i.e., videos.

{: style="text-align:center; font-size: small;"}
<img width="68%" height="68%" src="/assets/from-Diffusion-to-SORA/align_your_latents.png"/> 

{: style="text-align:center; font-size: small;"}
**Left:** We turn a pre-trained LDM into a video generator by inserting temporal layers that learn to align frames into temporally consistent sequences. During optimization, the image backbone $$θ$$ remains fixed and only the parameters $$ϕ$$ of the temporal layers $$l^i_ϕ$$ are trained, cf . Eq. (2). **Right:** During training, the base model $$θ$$ interprets the input sequence of length $$T$$ as a batch of images. For the temporal layers $$l^i_ϕ$$, these batches are reshaped into video format. Their output $$z'$$ is combined with the spatial output $$z$$, using a learned merge parameter $$α$$. During inference, skipping the temporal layers ($$α^i_ϕ=1$$) yields the original image model. For illustration purposes, only a single U-Net Block is shown. $$c_S$$ is optional context frame conditioning, when training prediction models (Sec. 3.2). Source and caption: [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818).
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation](https://arxiv.org/abs/2311.17117)</summary>
An U-net based diffusion model that takes as input a reference image (photo of a human) and a video of a moving human annotation (*stick man*, the pose sequence) and outputs the video that animates the human with the movements of the stick man. The reference image is encoded with VAE and CLIP embeddings. The model structure includes 2 U-nets, the reference unet that *merges detail features via spatial attention* and a denoising/diffusion U-Net that is applied to the pose sequence to generate the final video. 

3D attention is achieved by a spatial attention converts input $$B \times T \times H \times W \times C$$ into $$(B \times T ) \times (H \times W) \times C$$ to perform attention of patches within the same frame, followed by a temporal attention that converts input $$B \times T \times H \times W \times C$$ into $$(B \times H \times W) \times T \times C$$  that performs attention of the same patch across time.

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/from-Diffusion-to-SORA/anymate_anyone.png"/> 

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[CyberHost: Taming Audio-driven Avatar Diffusion Model with Region Codebook Attention](https://arxiv.org/abs/2409.01876)</summary>
CyberHost is a U-net based diffusion model that takes audio as an input and generats valid human movements. The novelty, compared to AnymateAnyone is the Region Codebook Attention which *improves the generation
quality of facial and hand animations by integrating fine-grained local features with
learned motion pattern priors*.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack](https://arxiv.org/abs/2309.15807)</summary>
Demonstrates the importance on fine-tuning text-to-image tasks with a very high quality dataset in order to achieve superior model quality: "in order to align the model towards highly aesthetic generations, quality matters significantly more than quantity in the fine-tuning dataset".
</details>
{::options parse_block_html="false" /}


## Diffusion transformers

The publication [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) introduced diffusion transformers (DiT) as a replacement that outperforms the UNet-based diffusion in scaling and accuracy measured by [Fréchet inception distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance). Because the work presented is based on image diffusion, DiT is based on [Vision Transformers (ViTs)](https://arxiv.org/abs/2010.11929), that operate on patches of images (Figure 4). VITs also also shown to have better scaling properties and accuracy than convolutional neural networks. Moreover, related to ViTs scaling, it shows that (1) ViT Gflops are strongly correlated with FID, (2) DiT Gflops are critical to improving performance, and (3) larger DiT models use large compute more efficiently.

>  We study the scaling behavior of transformers with respect to network complexity (in GFlops) vs. sample quality. We show that by constructing and benchmarking the DiT design space under the Latent Diffusion Models (LDMs) [48] framework, where diffusion models are trained within a VAE’s latent space, we can successfully replace the U-Net backbone with a transformer. 

> We further show that DiTs are scalable architectures for diffusion models: there is a strong correlation between the network complexity (measured by Gflops) vs. sample quality (measured by FID).

> Background: Gaussian diffusion models assume a forward noising process which gradually applies noise to real data $$\mathbf{x}_0: q(\mathbf{x}_t \mid \mathbf{x}_0 ) = \mathcal{N} (\mathbf{x}_t ; \sqrt{α¯}_t \mathbf{x}_ 0,(1 − α¯t)I)$$.  By applying the reparameterization trick, we can sample $$\mathbf{x}_t = \sqrt{α¯_t} \mathbf{x}_0 + \sqrt{1 − α¯_t} \epsilon_t, where \epsilon_t ∼ \mathcal{N} (0, I)$$. ETC section 3.1

Here they use implement a **conditional diffusion model** that takes as input extra information such as class $$c$$, and the reverse process becomes $$p_θ(\mathbf{x}_{t−1} \mid  \mathbf{x}_t, c)$$, where $$\epsilon_θ$$ and $$Σ_θ$$ are conditioned on $$c$$.

> In this setting, **[classifier-free guidance](https://arxiv.org/abs/2207.12598)** can be used to encourage the sampling procedure to find $$x$$ such that $$\log p(c \mid x)$$ is high. By Bayes Rule, $$\log p(c \mid x) ∝ \log p(x \mid c) − \log p(x)$$, and hence $$∇_x \log p(c \mid x) ∝ ∇_x \log p(x \mid c) − ∇_x \log p(x)$$.  By interpreting the output of diffusion models as the score function, the DDPM sampling procedure can be guided to sample $$x$$ with high $$p(x \mid c)$$ by: $$\bar{\epsilon}_θ(\mathbf{x}_t, c) = \epsilon_θ(\mathbf{x}_t, ∅) + s · ∇_x \log p(x \mid c) ∝ \epsilon_θ(\mathbf{x}_t, ∅) + s · (\epsilon_θ(\mathbf{x}_t, c)−\epsilon_θ(\mathbf{x}_t, ∅))$$, where $$s \gt 1$$ indicates the scale of the guidance (note that $$s = 1$$ recovers standard sampling). Evaluating the diffusion model with $$c = ∅$$ is done by randomly dropping out $$c$$ during training and replacing it with a learned “null” embedding $$∅$$. 

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

### Example of DiT-based diffusion models for text-to-image tasks

{::options parse_block_html="true" /}
<details> <summary markdown="span">[OmniGen: Unified Image Generation](https://arxiv.org/abs/2409.11340)</summary>
OmniGen is is a diffusion model based on DiT and VAE for text-to-image tasks, able to perform several tasks. Text input is tokenized, and image input are transformed into embedding via VAE.

{: style="text-align:center; font-size: small;"}
<img width="68%" height="68%" src="/assets/from-Diffusion-to-SORA/omnigen.png"/> 
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models](https://arxiv.org/abs/2409.10695)</summary>
A text-to-image diffusion model, that replaces the commonly used T5 and CLIP for input text encoding with the latents on a a decoder-only LLM (Llama3-8B).

{: style="text-align:center; font-size: small;"}
<img width="68%" height="68%" src="/assets/from-Diffusion-to-SORA/playground_v3.png"/> 
</details>
{::options parse_block_html="false" /}
 
## Video Diffusion and SORA

We now move to the movie domain and study a text-to-video model. 

Several advancements later, it led to [SORA](https://openai.com/index/video-generation-models-as-world-simulators/), a text-to-video, image-to-video, text-to-image (single frame video), and video-to-video model (for video editing, extension, etc)
Just as in the previous example, we also use a VAE to compress the input into a smaller latent space. However, we compress videos into spatial and temporal patches.

[Masked autoencoders (MAE)](Masked Autoencoders As Spatiotemporal Learners) have shown to be  scalable self-supervised learners for computer vision, in [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) and [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304).

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/from-Diffusion-to-SORA/sora_vae.png"/> 

{: style="text-align:center; font-size: small;"}
An overview of the lattent space in SORA. The pre-processing step "turns videos into [visual] patches by first compressing videos into a lower-dimensional latent space and subsequently decomposing the representation into spacetime patches". source: [Video generation models as world simulators, OpenAI](https://openai.com/index/video-generation-models-as-world-simulators/)

### Other video diffusion models

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)</summary>
NaViT (Native Resolution ViT) uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios. This is done by packing multiple patches from different images into a single sequence (the "Patch n’ Pack" method) which enables variable resolution while preserving the aspect ratio.
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072)</summary>
CogVideoX a large-scale DiT model for text-to-video generation. Model input is a pair of video and text. Text input is enconded with T5. Video input is passed through a 3D causal VAE that compresses the video into the latent space, and then all video patches are unfolded into a long sequence. Text and video embeddings are then concatenated as input, and passed to a stack of *expert* transformer blocks. The model output is then unpatchified to restore the original latent shape, and decoded using a 3D causal VAE to reconstruct the video. The attention is provided by a 3D attention model (that unfolds all patches of all frames) instead of a separate spatial and temporal attention. 
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048v1)</summary>

Latte is a DiT-based text-to-image and text-to-video diffusion model. Latte first extracts spatio-temporal tokens from input videos and then adopts a series of Transformer blocks to model video distribution in the latent space. The paper experiments several methods for embedding, clip patch embedding, model variants, timestep-class information injection, temporal positional embedding, and learning strategies, etc and provides a report. An an example, it tests four variants of 3D transformer block: (1) with spatial Transformer blocks and temporal Transformer blocks, (2) a "late fusion" approach to combine spatial-temporal information, that consists of an equal number of Transformer blocks as in Variant 1, (3) that "initially computes self-attention only on the spatial dimension, followed by the temporal dimension, and as a result, each Transformer block captures both spatial and temporal information", and (4) one that uses use different attenion heads to handle tokens separately in spatial and temporal dimensions.

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/from-Diffusion-to-SORA/latte.png"/> 

The paper also analyses 2 distinct methods for video patching embedding: (1) collect all patches of a frame, and then collect the patches for the following frame etc, as in ViT, (2) extracting patches in the temporal dimension as well (in a "tube") and move that tube in the spatial dimension: 


{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/from-Diffusion-to-SORA/latte2.png"/> 

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Tora: Trajectory-oriented Diffusion Transformer for Video Generation](https://arxiv.org/abs/2407.21705) ([webpage](https://ali-videoai.github.io/tora_video/))</summary>

Tora is capable of generating videos guided by trajectories, images, texts, or combinations thereof. "Spatial-Temporal Diffusion Transformer (ST-DiT) from OpenSora as its foundational model", ie 1 spacial attention followed by a temporal attention, just like variant 1 in [Latte](https://arxiv.org/abs/2401.03048v1) (above). **The big advantadge in using ST-DiT compared to using 3D attention is that it saves on computation and it utilizes pre-trained text-to-image models.** "The trajectory encoder converts the trajectory into motion patches, which inhabit the same latent space as the video patches".  Text encoding is provided by T5.

{: style="text-align:center; font-size: small;"}
<img width="80%" height="80%" src="/assets/from-Diffusion-to-SORA/tora.png"/> 

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113) ([github](https://github.com/facebookresearch/mae_st))
</summary>
It applies Masked AutoEncoders (MAE) to the video domain, demonstrating high compute efficiency:
>  We randomly mask out
spacetime patches in videos and learn an autoencoder to reconstruct them in pixels.
Interestingly, we show that our MAE method can learn strong representations
with almost no inductive bias on spacetime (only except for patch and positional
embeddings), and spacetime-agnostic random masking performs the best. We
observe that the optimal masking ratio is as high as 90% (vs. 75% on images [31]),
supporting the hypothesis that this ratio is related to information redundancy of the
data. A high masking ratio leads to a large speedup, e.g., > 4× in wall-clock time
or even more.

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/from-Diffusion-to-SORA/masked_autoencoders.png"/> 

</details>
{::options parse_block_html="false" /}

## Multi-dimensional SORA parallelism

{::options parse_block_html="true" /}
<details> <summary markdown="span">[Scaling Diffusion Transformers to 16 Billion Parameters](https://arxiv.org/abs/2407.11633)</summary>

Presents Mixture of Experts of DiTs (MoE-DiT), delivering better scaling properties, an accuracy comparable to dense DiTs, and highly optimized inference; 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/from-Diffusion-to-SORA/dit_moe.png"/> 
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span">[LongVILA: Scaling Long-Context Visual Language Models for Long Videos](https://www.arxiv.org/abs/2408.10188)</summary>

LongVilla details a pipeline of 5 steps for training long-context visual-language models. The first 3 stages are multi-modal alignment, large-scale pre-training and short supervised fine-tuning from [VILA: On Pre-training for Visual Language Models](https://arxiv.org/abs/2312.07533v2). Stage 4 is context extension for LLMs, by increasing the sequence length of input samples (ie curriculum learning) up to 262K tokens. In Stage 5, the model is fine-tuned for long video understanding with Multi-Modal Sequence Parallelism (MM-SP) based on [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485).

{: style="text-align:center; font-size: small;"}
<img width="68%" height="68%" src="/assets/from-Diffusion-to-SORA/longvilla.png"/> 

</details>
{::options parse_block_html="false" /}


{::options parse_block_html="true" /}
<details> <summary markdown="span"> [PipeFusion: Displaced Patch Pipeline Parallelism for Inference of Diffusion Transformer Models](https://arxiv.org/abs/2405.14430) and [xDIT github](https://github.com/xdit-project/).</summary>

PipeFusion splits images into patches and distributes the network layers across multiple devices. It employs a pipeline parallel manner to orchestrate communication and computations. xDIT is a parallel **inference** engine of DiTs using Universal Sequence Parallelism (including Ulysses attention and Ring attention), PipeFusion, and hybrid parallelism. It applies and benchmarks xDIT to the following DiT implementations: CogVideo, Flux, Latte, HunyuanDiT, Stable Diffusion 3, Pixart-Sigma, Pixart-alpha. 

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/from-Diffusion-to-SORA/pipefusion.png"/> 
</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> [OpenSORA implementation from HPC AI Tech](https://github.com/hpcaitech/Open-Sora/tree/main)</summary>

An attempt to create an open-source implementation of [SORA](https://openai.com/index/video-generation-models-as-world-simulators/). Currently in version 1.2. Details collected from the [docs](https://github.com/hpcaitech/Open-Sora/tree/main/docs) section, particularly the [technical reports](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md):
- [acceleration](https://github.com/hpcaitech/Open-Sora/blob/main/docs/acceleration.md#accelerated-transformer) provided by kernel optimization (flash attention), fused layernorm kernel, and ones compiled by colossalAI. [Sequence parallelism](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md#sequence-parallelism) is provided by Ulysses only;
- [Spatial attention](https://github.com/hpcaitech/Open-Sora/blob/main/docs/acceleration.md#efficient-stdit) is provided by ST-DiT instead of full 3D attention as STDiT is more (compute) efficient as the number of frames increases.
- Data processing are explained in the [Data Processing](https://github.com/hpcaitech/Open-Sora/blob/main/docs/data_processing.md) and [Datasets](https://github.com/hpcaitech/Open-Sora/blob/main/docs/datasets.md) pages;
- Texts are encoded by T5 and videos by VAE: the 2D VAE is initialized with SDXL's VAE, and the 3D VAE is initialized with Magvit-v2. See the [VAE Report](https://github.com/hpcaitech/Open-Sora/blob/main/docs/vae.md) for additional info. The [video compression network](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md#video-compression-network) used was 83M 2D VAE in previous version, yielding an 8x compression, with 1 frames picked in every 3 (to reduce the temporal dimension). To improve quality, in the 1.2, the authors first compress the video in the spatial dimension by 8x8 times, then compress the video in the temporal dimension by 4x times.
- The training includes 3 steps: (1) freeze the 2D VAE in order to train features from the 3D VAE similar to the features from the 2D VAE; (2) remove the identity loss and just learn the 3D VAE; and (3) remove the loss and train the whole VAE to reconstruct the original videos. Training is performed with a curriculum of increasing video quality, in three stages, to better utilize compute ([source](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md#more-data-and-better-multi-stage-training));
- It used [rectified flow](https://arxiv.org/abs/2209.03003) instead of [DDPM](https://arxiv.org/abs/2006.11239) for diffusion ([source](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md#rectified-flow-and-model-adaptation)).

</details>
{::options parse_block_html="false" /}

{::options parse_block_html="true" /}
<details> <summary markdown="span"> [Meta Movie Gen research paper](https://ai.meta.com/static-resource/movie-gen-research-paper)</summary>

One of the most detailed technical reports of a very large DiT-based diffusion model, trained on 6,144 H100 GPUs, able to solve multiple tasks: text-to-video synthesis, video personalization, video editing, video-to-audio generation, and text-to-audio generation. "The largest video generation
model is a 30B parameter transformer trained with a maximum context length of 73K video tokens,
corresponding to a generated video of 16 seconds at 16 frames-per-second". Appendix A.2 "Model scaling and training efficiency" details 4-way parallelism via Data Parallelism with Sharding, Tensor parallelism, Sequence parallelism and [Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html). Also includes details on the overlapping of communication and computation, and the usage of activation checkpointing for improved memory efficiency.
</details>
{::options parse_block_html="false" /}