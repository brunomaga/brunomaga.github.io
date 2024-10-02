import os
import torch
import torch.distributed
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
from vit import ViT


def extract(a, t, x_shape):
    """ allow us to extract the appropriate t index for a batch of indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def main(model_name='UNet'):

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)

    # load dataset from the hub
    dataset = load_dataset("zh-plus/tiny-imagenet", split='train')
    transform = lambda image: (pil_to_tensor(image)/255)*2-1 # normalize [-1,1]
    images = [ transform(image) for image in dataset["image"]]
    batch_size, channels, image_size = 16, images[0].shape[0], images[0].shape[1]
    images = [img for img in images if img.shape == (channels, image_size, image_size)] # remove single-channel images
    dataloader = DataLoader(images, batch_size=batch_size, sampler=DistributedSampler(images))

    # load model and optimizer
    if model_name == 'UNet':
        model = diffusers.UNet2DModel(in_channels=channels, out_channels=channels)
    elif model_name == 'ViT':
        model = ViT(channels=channels, patch_size=4, n_embd=64, n_blocks=12)
    model = DDP(model.to(device), device_ids=[local_rank])
    optimizer = SGD(model.parameters(), lr=1e-2)

    # linear variance scheduler. See here for more implementations: https://huggingface.co/blog/annotated-diffusion#defining-the-forward-diffusion-process
    timesteps, beta_1, beta_T = 100, 10**-4, = 0.02
    betas = torch.tensor([ beta_1 + (beta_T - beta_1) / timesteps * t for t in range(timesteps) ])
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok = True)


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

    @torch.no_grad()
    def posterior_variance(x, t):
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_var = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_var_t = extract(posterior_var, t, x.shape)
        return posterior_var_t

    # Generating new images from a diffusion model happens by reversing the diffusion process: we start from T and go to 1 (alg 2)
    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        model_mean = predicted_mean(model, x, t)

        if t_index == 0:
            return model_mean

        posterior_variance_t = posterior_variance(x,t)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(model, shape):
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in range(timesteps,0,-1):
            img = p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    # forward diffusion (using Sohl-Dickstein property)
    @torch.no_grad()
    def q_sample(x0, t, noise=None):
        noise = noise or torch.randn_like(x0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_one_minus_alphas_cumprod_t = extract( sqrt_one_minus_alphas_cumprod, t, x0.shape )
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise 


    # training loop
    for step, batch in enumerate(dataloader):
        batch = batch.to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(batch)
        x_noisy = q_sample(x0=batch, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)[0]
        loss = F.smooth_l1_loss(noise, predicted_noise) # Huber loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"step {step} loss: {loss.item()}")

        @torch.no_grad()
        def num_to_groups(num, divisor):
            groups = num // divisor
            remainder = num % divisor
            arr = [divisor] * groups
            if remainder > 0:
                arr.append(remainder)
            return arr
        
        # save generated images
        if step != 0 and step % 10 == 0:
            batches = num_to_groups(4, batch_size)
            all_images_list = list(map(lambda n: p_sample_loop(model, shape=(n, channels, image_size, image_size)), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{step}.png'), nrow = 6)
            

if __name__ == "__main__":
    main("Unet")
    main("ViT")