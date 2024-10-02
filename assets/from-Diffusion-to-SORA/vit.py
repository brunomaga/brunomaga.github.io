import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import diffusers

class Block(nn.Module):

    def __init__(self, n_embd, d_head=128, n_heads=8, dropout_p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiheadAttention(n_embd, d_head, n_heads=n_heads, dropout=dropout_p)
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

    def __init__(self, channels, patch_size=4, n_embd=64, n_blocks=12) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pos_emb = nn.Embedding(64, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd) for _ in range(n_blocks)])
        self.decoder = nn.Sequential( nn.LayerNorm(n_embd), nn.Linear(n_embd, channels*2) )
        self.vae = diffusers.models.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

    @staticmethod
    def patchify(x, patch_size):
        """ converts an image x into a list of patches of size patch_size x patch_size """
        B, C, _, _ = x.shape # batch, channels, height, width
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(B, C, -1, patch_size, patch_size)
        x = x.permute(0, 2, 1, 3, 4)
        return x

    def forward(self):
        if self.vae:
            x = self.vae.tiled_encode(x)
        x = ViT.patchify(x, self.patch_size)
        x += self.pos_emb(torch.arange(x.shape[0], device=x.device))
        x = self.blocks(x)
        x = self.decoder(x)
        if self.vae:
            x = self.vae.tiled_decode(x)
        return x
