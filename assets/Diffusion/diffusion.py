import os
import sys
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from torch.optim import Adam
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor
import diffusers

# import files in current folder
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir))
from dit import DiT


def main(model_name='UNet'):

    assert 'RANK' in os.environ, "distributed training requires setting the RANK environment variable, launch with torchrun"
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # linear variance scheduler. See here for more implementations: https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process
    T = 100 # the paper uses T=1000
    β_1, β_T = 0.0001 * 1000 / T, 0.02 * 1000 / T # the paper's values, scaled by 1000/T
    # arrays are 0-indexed: index i holds the value of the paper's timestep t=i+1
    β = torch.linspace(β_1, β_T, T, device=device)
    α = 1. - β
    α_cumprod = torch.cumprod(α, axis=0)

    # posterior variance, and its log clipped at index 0 (where posterior_β is 0)
    α_cumprod_prev = F.pad(α_cumprod[:-1], (1, 0), value=1.0)
    posterior_β = β * (1. - α_cumprod_prev) / (1. - α_cumprod) # Eq 7
    posterior_log_β = torch.log(torch.cat([posterior_β[1:2], posterior_β[1:]]))

    # load dataset from the hub
    dataset = load_dataset("uoft-cs/cifar10", split='train')
    transform = lambda image: (pil_to_tensor(image)/255)*2-1 # normalize to [-1,1]
    images = [ transform(image) for image in dataset["img"]]
    batch_size, channels, img_size = 48, images[0].shape[0], images[0].shape[-1]
    sampler = DistributedSampler(images)
    dataloader = DataLoader(images, batch_size=batch_size, sampler=sampler, drop_last=True)

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

    def eq11_μ_θ(x, ε_θ, t):
        # Equation 11: use model (noise predictor) to predict the mean
        β_t = β[t][:, None, None, None]
        one_over_sqrt_α = torch.sqrt(1.0 / α)
        one_over_sqrt_α_t = one_over_sqrt_α[t][:, None, None, None]
        sqrt_1_minus_α_cumprod = torch.sqrt(1. - α_cumprod)
        sqrt_1_minus_α_cumprod_t = sqrt_1_minus_α_cumprod[t][:, None, None, None]
        μ_θ = one_over_sqrt_α_t * (x - β_t * ε_θ  / sqrt_1_minus_α_cumprod_t)
        return μ_θ

    def eq13_decoder_nll(x_0, p):
        """ Equation 13: negative log-likelihood of the discretized decoder p_θ(x_0|x_1), per pixel value """
        cdf_upper = p.cdf(x_0 + 1/255) # CDF at δ+(x_0), for x_0 < 1
        cdf_lower = p.cdf(x_0 - 1/255) # CDF at δ-(x_0), for x_0 > -1
        # the CDF at -∞ and +∞ is 0 and 1 (evaluating it at ±∞ would make the gradients NaN)
        prob = torch.where(x_0 < -0.999, cdf_upper, torch.where(x_0 > 0.999, 1 - cdf_lower, cdf_upper - cdf_lower))
        return -torch.log(prob.clamp(min=1e-12))

    def learned_Σ_θ(v, t):
        """ Improved DDPM, Equation 15: interpolate log(Σ_θ) between log(β_t) and log(posterior_β_t) """
        frac = (v + 1) / 2 # from the model output in [-1, 1] to [0, 1], as in the official implementation
        log_β_t = torch.log(β)[t][:, None, None, None]
        posterior_log_β_t = posterior_log_β[t][:, None, None, None]
        return torch.exp(frac * log_β_t + (1 - frac) * posterior_log_β_t)

    def unet_loss(x_t, t, noise):
        """ L_simple, Equation 14 """
        ε_θ = model(x_t, t).sample # extract tensor from UNet2DOutput class
        return F.mse_loss(ε_θ, noise) # Huber loss and MAE are also ok

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

    dist.destroy_process_group()


if __name__ == "__main__":
    # main("UNet")
    main("DiT")
