import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as flash_attn_func

class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd=256, d_head=128, n_heads=8, dropout_p=0.1):
        """ An Ulysses multi-head attention. Variable names follow GPT-lite's post """

        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.keys = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.queries = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.values = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, n_embd)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        B, N, _ = x.shape

        # Q, K and V embeddings: (B, N, E) -> (H, B, N, E)
        q = torch.stack([q(x) for q in self.queries], dim=0)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        softmax_scale = q.shape[-1] ** (-0.5)
        out = flash_attn_func(q, k, v,dropout_p=self.dropout_p, scale=softmax_scale)

        out = out.permute(1, 2, 0, 3)  # (H, B, N, E) -> (B, N, H, E)
        out = out.reshape(B, N, -1)  # (B, N, H, E) -> (B, N, H*E)
        out = self.proj(out)  # (B, N, H*E) -> (B, N, E)
        out = self.dropout(out)
        return out
    

class Block(nn.Module):

    def __init__(self, n_embd, d_head=128, n_heads=8, dropout_p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, d_head, n_heads=n_heads)
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
 
    
class DiT(nn.Module):
    """ A Vision Transformer model. """

    def __init__(self, timesteps, num_channels, img_size, patch_size=4, n_blocks=12):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        n_embd = patch_size*patch_size*num_channels # values per img patch 

        # temporal and positional embeddings
        n_pos_emb = (img_size//patch_size)*(img_size//patch_size) # number of patches per image
        self.t_embedding = nn.Embedding(timesteps, n_embd)
        self.pos_embedding = nn.Embedding(n_pos_emb, n_embd)

        # DiT blocks
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd) for _ in range(n_blocks)])

        # decoder: "standard linear decoder to do this; we apply the layer norm and linearly decode each token into a p×p×2C tensor"
        self.decoder = nn.Sequential( nn.LayerNorm(n_embd), nn.Linear(n_embd, n_embd*2) )
        

    def patchify(self, x, t):
        """ break image (B, C, H, W) into patches (B, C, NH, NW, PH, PW) for NH*NW patches of size PHxPW """
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # linearize patches and linearize patches and flatthen embeddings: (B, NH*NW, PH*PW*C)
        _, _, NH, NW, PH, PW = x.shape    
        x = x.permute(0, 2, 3, 4, 5, 1) # (B, NH, NW, PH, PW, C)
        x = x.reshape(B, NH*NW, PH*PW*C)
        return x, dict(B=B, C=C, H=H, W=W, NH=NH, NW=NW, PH=PH, PW=PW)

    def unpatchify(self, x, shapes):
        """ convert patches (B, NH*NW, C*PH*PW*2) back into mu and var of shape (B, C, H, W) = (B, C, NH*PH, NW*PW) """
        B, C, H, W, NH, NW, PH, PW, = shapes.values()
        assert x.shape == (B, NH*NW, PH*PW*C*2)
        x = x.reshape(B, NH, NW, PH, PW, C, 2).permute(0, 5, 1, 3, 2, 4, 6) # (B, C, NH, PH, NW, PW, 2)
        x = x.reshape(B, C, NH*PH, NW*PW, 2)
        ε_θ, Σ_θ = x[...,0], x[...,1]
        assert ε_θ.shape == Σ_θ.shape == (B, C, H, W) # original shape
        return ε_θ, Σ_θ

    def forward(self, x, t):

        # convert images (B, C, H, W) to patches (B, N, C*H*W) for N patches
        x, shapes = self.patchify(x, t) # add embeddings and unfold/linearize patches
        B, N, E = x.shape

        # add positional embeddings
        x += self.pos_embedding(torch.arange(N, device=x.device)).reshape(1, N, E)

        # add temporal embeddings
        x += self.t_embedding(t).reshape(B, 1, E)

        x = self.blocks(x)

        # decode patches, as per paper, output is 2x the size of input (mean and variance)
        # (B, C, NH*PH, NW*PW) -> (B, C, NH*PH, NW*PW*2) 
        x = self.decoder(x)

        ε_θ, Σ_θ = self.unpatchify(x, shapes)
        return ε_θ, Σ_θ

