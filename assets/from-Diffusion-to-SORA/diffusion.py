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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import diffusers

#import files in current folder
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir))
from vit import ViT


def main(model_name='UNet'):

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # linear variance scheduler. See here for more implementations: https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process
    timesteps, beta_1, beta_T = 100, 0.0001, 0.02
    betas = torch.tensor([ beta_1 + (beta_T - beta_1) / timesteps * t for t in range(timesteps) ], device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # load dataset from the hub
    dataset = load_dataset("uoft-cs/cifar10", split='train')
    transform = lambda image: (pil_to_tensor(image)/255)*2-1 # normalize to [-1,1]
    images = [ transform(image) for image in dataset["img"]]
    batch_size, channels, img_size = 64, images[0].shape[0], images[0].shape[-1]
    dataloader = DataLoader(images, batch_size=batch_size, sampler=DistributedSampler(images), drop_last=True)

    # load model and optimizer
    if model_name == 'UNet':
        model = diffusers.UNet2DModel(in_channels=channels, out_channels=channels)
    elif model_name == 'ViT':
        patch_size = 4
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        model = ViT(channels=channels, patch_size=patch_size, img_size=img_size, num_channels=channels, n_embd=64, n_blocks=12, timesteps=timesteps)
    else:
        raise ValueError(f"Model name {model_name} not recognized")
    model = DDP(model.to(device), device_ids=[local_rank])
    optimizer = SGD(model.parameters(), lr=1e-2)

    @torch.no_grad()
    def predicted_mean(model, x, t):
        # Equation 11: use model (noise predictor) to predict the mean
        betas_t = betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        one_over_sqrt_alphas = torch.sqrt(1.0 / torch.cumprod(alphas, axis=0))
        one_over_sqrt_alphas_t = one_over_sqrt_alphas[t][:, None, None, None]
        epsilon = model(x, t)
        if model_name == 'UNet':
            epsilon = epsilon.sample # remove tensor from Unet2DOutput class
        model_mean = one_over_sqrt_alphas_t * (x - betas_t * epsilon  / sqrt_one_minus_alphas_cumprod_t)
        return model_mean

    # Generating new images from a diffusion model happens by reversing the diffusion process: we start from T and go to 1 (alg 2)
    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        model_mean = predicted_mean(model, x, t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(model, shape):
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, timesteps)):
            img = p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
        return img

    # forward diffusion (using Sohl-Dickstein property)
    @torch.no_grad()
    def q_sample(x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise 


    # training loop
    for epoch in range(20):
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(batch)
            x_noisy = q_sample(x0=batch, t=t, noise=noise)
            predicted_noise = model(x_noisy, t)[0]
            loss = F.mse_loss(noise, predicted_noise) # Huber loss also ok
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if dist.get_rank() == 0:
                print(f"epoch {epoch}, step {step} loss: {loss.item()}")

        # save generated images
        results_folder = Path("./results")
        results_folder.mkdir(exist_ok = True)
        img = p_sample_loop(model, shape=batch.shape)
        img = (img + 1) * 0.5 # from [-1,1] to [0,1]
        save_image(img, str(results_folder / f'sample-{epoch}.png'), nrow = batch_size)
            

if __name__ == "__main__":
    # main("UNet")
    main("ViT")