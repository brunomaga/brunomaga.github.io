import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim import SGD
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DistributedSampler
import diffusers

#import files in current folder
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir))
from dit import DiT


def main(model_name='UNet'):

    assert 'RANK' in os.environ, "distributed training requires setting the RANK environment variable, launch with torchrun"
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # linear variance scheduler. See here for more implementations: https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process
    T, β_1, β_T = 100, 0.0001, 0.02
    β = torch.tensor([ β_1 + (β_T - β_1) / T * t for t in range(T) ], device=device)
    α = 1. - β
    α_cumprod = torch.cumprod(α, axis=0)
    α_cumprod_prev = F.pad(α_cumprod[:-1], (1, 0), value=1.0)
    posterior_β = β * (1. - α_cumprod_prev) / (1. - α_cumprod) # Eq 7
    # compute log but clip first element because posterior_β is 0 at the beginning
    posterior_log_β = torch.tensor([posterior_β[1].item()] + posterior_β[1:].tolist()).log().to(device)

    # load dataset from the hub
    dataset = load_dataset("uoft-cs/cifar10", split='train')
    transform = lambda image: (pil_to_tensor(image)/255)*2-1 # normalize to [-1,1]
                
    images = [ transform(image) for image in dataset["img"]]
    batch_size, channels, img_size = 48, images[0].shape[0], images[0].shape[-1]
    dataloader = DataLoader(images, batch_size=batch_size, sampler=DistributedSampler(images), drop_last=True)

    # load model and optimizer
    if model_name == 'UNet':
        model = diffusers.UNet2DModel(in_channels=channels, out_channels=channels)
    elif model_name == 'DiT':
        model = DiT(T, channels, img_size, patch_size=4, n_blocks=4)
    else:
        raise ValueError(f"Model name {model_name} not recognized")
    model = model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = SGD(model.parameters(), lr=1e-2)

    @torch.no_grad()
    def posterior_µ(x_0, x_t, t):
        """ return posterior mean at step t, Equation 7 """
        α_t = α[t][:, None, None, None]
        β_t = β[t][:, None, None, None]
        α_cumprod_t = α_cumprod[t][:, None, None, None]
        α_cumprod_prev_t = α_cumprod_prev[t][:, None, None, None]
        coef1 =  β_t * torch.sqrt(α_cumprod_prev_t) / (1.0 - α_cumprod_t)
        coef2 = (1.0 - α_cumprod_prev_t) * torch.sqrt(α_t) / (1.0 - α_cumprod_t)
        return coef1 * x_0 + coef2 * x_t

    @torch.no_grad()
    def predicted_mean(model, x, t):
        # Equation 11: use model (noise predictor) to predict the mean
        β_t = β[t][:, None, None, None]
        sqrt_one_minus_α_cumprod = torch.sqrt(1. - α_cumprod)
        sqrt_one_minus_α_cumprod_t = sqrt_one_minus_α_cumprod[t][:, None, None, None]
        one_over_sqrt_α = torch.sqrt(1.0 / torch.cumprod(α, axis=0))
        one_over_sqrt_α_t = one_over_sqrt_α[t][:, None, None, None]
        model_output = model(x, t)
        if model_name == 'UNet':
            ε_θ = model_output.sample # remove tensor from Unet2DOutput class
        elif model_name == 'DiT':
            ε_θ, _ = model_output
        model_mean = one_over_sqrt_α_t * (x - β_t * ε_θ  / sqrt_one_minus_α_cumprod_t)
        return model_mean

    # Generating new images from a diffusion model happens by reversing the diffusion process: we start from T and go to 1 (alg 2)
    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        model_mean = predicted_mean(model, x, t)
        if t_index == 0:
            return model_mean
        else:
            posterior_β_t = posterior_β[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_β_t) * noise 
        

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(model, shape):
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, T)):
            img = p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
        return img

    # forward diffusion (using Sohl-Dickstein property)
    @torch.no_grad()
    def q_sample(x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_α_cumprod = torch.sqrt(α_cumprod)
        sqrt_α_cumprod_t = sqrt_α_cumprod[t][:, None, None, None]
        sqrt_one_minus_α_cumprod = torch.sqrt(1. - α_cumprod)
        sqrt_one_minus_α_cumprod_t = sqrt_one_minus_α_cumprod[t][:, None, None, None]
        return sqrt_α_cumprod_t * x0 + sqrt_one_minus_α_cumprod_t * noise 

    @torch.no_grad()
    def eq11_from_ε_to_μ(x, ε_θ, t):
        # Equation 11: use model (noise predictor) to predict the mean
        β_t = β[t][:, None, None, None]
        one_over_sqrt_α = torch.sqrt(1.0 / torch.cumprod(α, axis=0))
        one_over_sqrt_α_t = one_over_sqrt_α[t][:, None, None, None]
        sqrt_one_minus_α_cumprod = torch.sqrt(1. - α_cumprod)
        sqrt_one_minus_α_cumprod_t = sqrt_one_minus_α_cumprod[t][:, None, None, None]
        model_mean = one_over_sqrt_α_t * (x - β_t * ε_θ  / sqrt_one_minus_α_cumprod_t)
        return model_mean
    

    # training loop
    for epoch in range(20):
        for step, x_0 in enumerate(dataloader):
            x_0 = x_0.to(device=device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, T, (batch_size,), device=device).long()
            noise = torch.randn_like(x_0)
            x_t = q_sample(x0=x_0, t=t, noise=noise)
            model_output = model(x_t, t)
            if model_name == 'UNet':
                ε_θ = model_output.sample # epsilon
                loss = F.mse_loss(noise, ε_θ) # Huber or MAE loss also ok
            elif model_name == 'DiT':
                # "We follow Nichol and Dhariwal’s approach [36]: train ε_θ with L_simple, and train Σ_θ with the full KL divergence."
                ε_θ, Σ_θ = model_output
                posterior_µ_t = posterior_µ(x_0, x_t, t)
                posterior_β_t = posterior_β[t][:, None, None, None]     

                # avoid negative or very high values for variance
                # according to paper: beta_t and posterior_beta_t are the two extreme choices corresponding
                # to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance 
                # source: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py#L682
                if False:
                    log_β_t = torch.log(β)[t][:, None, None, None]
                    posterior_log_β_t = posterior_log_β[t][:, None, None, None]
                    min_log, max_log = posterior_log_β_t, log_β_t
                    frac = (Σ_θ + 1) / 2 # from [-1,1] to [0,1]
                    log_Σ_θ = frac * max_log + (1 - frac) * min_log # from [0,1] to [min_log, max_log]
                    Σ_θ = torch.exp(log_Σ_θ) # TODO use this instead of Σ_θ_t
                else:
                    Σ_θ = torch.clamp(Σ_θ, min=1e-5, max=1e5)
                    posterior_β_t = torch.clamp(posterior_β_t, min=1e-5, max=1e5)

                    # normal_kl = lambda mean1, logvar1, mean2, logvar2 : \
                    #     0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2) )
                    # kl = normal_kl(posterior_mean_t, posterior_log_variance_t, ε_θ, model_log_variance_t)
                    # kl = kl.mean(dim=(1, 2, 3)) # mean over of non-batch dimensions

                μ_θ = eq11_from_ε_to_μ(x_t, ε_θ, t) # Eq. 11
                p = torch.distributions.Normal(μ_θ, Σ_θ.sqrt()) # Eq. 1
                q = torch.distributions.Normal(posterior_µ_t, posterior_β_t.sqrt()) # Eq. 6
                kl = torch.distributions.kl_divergence(p, q).mean(dim=(1, 2, 3))

                # for t=0, return Negative Log Likelihood (NLL) of the decoder, otherwise return KL divergence
                decoder_nll = F.gaussian_nll_loss(input=μ_θ, var=Σ_θ, target=x_0, reduction='none').mean(dim=(1, 2, 3))
                loss = torch.where((t == 0), decoder_nll, kl) # loss per sample
                loss = loss.mean()
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if dist.get_rank() == 0:
                print(f"epoch {epoch}, step {step} loss: {loss.item()}")

        # save generated images
        results_folder = Path("./results")
        results_folder.mkdir(exist_ok = True)
        img = p_sample_loop(model, shape=x_0.shape)
        img = (img + 1) * 0.5 # from [-1,1] to [0,1]
        save_image(img, str(results_folder / f'sample-{epoch}.png'), nrow = batch_size)
            

if __name__ == "__main__":
    # main("UNet")
    main("DiT")