import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import diffusers

class Block(nn.Module):

    def __init__(self, n_embd, d_head=128, n_heads=8, dropout_p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiheadAttention(
            embed_dim=d_head, num_heads=n_heads, # output of size of each head and number of heads
            kdim=n_embd, vdim=n_embd, # number of feature in key and value
            dropout=dropout_p,
            batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffw = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

    
class ViT(nn.Module):
    """ A Vision Transformer model. """

    def __init__(self, channels, patch_size=4, img_size=32, num_channels=3, n_embd=64, n_blocks=12, timesteps=100, use_vae=False):
        super().__init__()
        self.patch_size = patch_size
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd) for _ in range(n_blocks)])
        self.decoder = nn.Sequential( nn.LayerNorm(n_embd), nn.Linear(n_embd, channels*2) )
        self.vae = diffusers.models.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse") if use_vae else None
        
        sample_emb_size = patch_size*patch_size*num_channels # values per img patch 
        self.n_pos_emb = (img_size//patch_size)*(img_size//patch_size)
        self.timesteps = timesteps
        self.t_embedding = nn.Embedding(self.timesteps, sample_emb_size)
        self.pos_emb = nn.Embedding(self.n_pos_emb, sample_emb_size)

    def forward(self, x, t):
        if self.vae:
            x = self.vae.tiled_encode(x)

        # break image (B, C, H, W) into patches (B, C, PH, PW, H, W) for PH*PW patches
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # (B, C, PH, PW, H, W) -> (B, C, PH*PW, H, W)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)

        # (B, C, PH*PW, H, W) -> (B, PH*PW, C, H, W) 
        x = x.permute(0, 2, 1, 3, 4)

        # flatten patches: (B, PH*PW, C, H, W) -> (B, PH*PW, C*H*W)
        _, N, C, H, W = x.shape
        x = x.contiguous().view(B, N, -1)

        # add positional embeddings
        assert N == self.n_pos_emb, f"Number of patches {N} must be equal to number of positional embeddings {self.n_pos_emb}" 
        x += self.pos_emb(torch.arange(N, device=x.device)).view(1, N, C*H*W)

        # add temporal embeddings
        x += self.t_embedding(t).view(B, 1, C*H*W)

        x = self.blocks(x)
        x = self.decoder(x)
        if self.vae:
            x = self.vae.tiled_decode(x)
        return x, #singleton tuple
