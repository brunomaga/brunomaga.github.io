import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as flash_attn_func

class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd=256, d_head=128, n_heads=8, dropout_p=0.1):
        """ A multi-head attention module. Variable names follow GPT-lite's post """

        super().__init__()
        self.dropout_p = dropout_p
        self.keys = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.queries = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.values = nn.ModuleList([nn.Linear(n_embd, d_head, bias=False) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, n_embd)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        B, N, _ = x.shape

        # Q, K and V embeddings: (B, N, E) -> (H, B, N, D) for H heads of size D
        q = torch.stack([q(x) for q in self.queries], dim=0)
        k = torch.stack([k(x) for k in self.keys], dim=0)
        v = torch.stack([v(x) for v in self.values], dim=0)

        softmax_scale = q.shape[-1] ** (-0.5)
        dropout_p = self.dropout_p if self.training else 0.0 # the functional API does not know about eval mode
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, scale=softmax_scale)

        out = out.permute(1, 2, 0, 3)  # (H, B, N, D) -> (B, N, H, D)
        out = out.reshape(B, N, -1)  # (B, N, H, D) -> (B, N, H*D)
        out = self.proj(out)  # (B, N, H*D) -> (B, N, E)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    """ A pre-layer-norm transformer block: multi-head self-attention followed by a feed-forward network """

    def __init__(self, n_embd, d_head=128, n_heads=8, dropout_p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, d_head, n_heads=n_heads, dropout_p=dropout_p)
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
    """ A Diffusion Transformer (DiT) model """

    def __init__(self, timesteps, num_channels, img_size, patch_size=4, n_blocks=12, num_labels=None):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        n_embd = patch_size*patch_size*num_channels # values per img patch

        # timestep and positional embeddings
        n_pos_emb = (img_size//patch_size)*(img_size//patch_size) # number of patches per image
        self.t_embedding = nn.Embedding(timesteps, n_embd)
        self.pos_embedding = nn.Embedding(n_pos_emb, n_embd)

        # class embeddings, only for a class-conditional model
        self.class_embedding = nn.Embedding(num_labels, n_embd) if num_labels else None

        # DiT blocks
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd) for _ in range(n_blocks)])

        # decoder: "standard linear decoder to do this; we apply the layer norm and linearly decode each token into a p×p×2C tensor"
        self.decoder = nn.Sequential( nn.LayerNorm(n_embd), nn.Linear(n_embd, n_embd*2) )

    def patchify(self, x):
        """ break image (B, C, H, W) into patches (B, C, NH, NW, PH, PW) for NH*NW patches of size PHxPW """
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # linearize patches and flatten embeddings: (B, NH*NW, PH*PW*C)
        _, _, NH, NW, PH, PW = x.shape
        x = x.permute(0, 2, 3, 4, 5, 1) # (B, NH, NW, PH, PW, C)
        x = x.reshape(B, NH*NW, PH*PW*C)
        return x, dict(B=B, C=C, H=H, W=W, NH=NH, NW=NW, PH=PH, PW=PW)

    def unpatchify(self, x, shapes):
        """ convert patches (B, NH*NW, PH*PW*C*2) back into the noise ε_θ and the vector v that parameterizes
            the variance Σ_θ, each of shape (B, C, H, W) = (B, C, NH*PH, NW*PW) """
        B, C, H, W, NH, NW, PH, PW, = shapes.values()
        assert x.shape == (B, NH*NW, PH*PW*C*2)
        x = x.reshape(B, NH, NW, PH, PW, C, 2).permute(0, 5, 1, 3, 2, 4, 6) # (B, C, NH, PH, NW, PW, 2)
        x = x.reshape(B, C, NH*PH, NW*PW, 2)
        ε_θ, v = x[...,0], x[...,1]
        assert ε_θ.shape == v.shape == (B, C, H, W) # original shape
        return ε_θ, v

    def forward(self, x, t, label=None):
        x, shapes = self.patchify(x) # (B, C, H, W) -> (B, N, E), for N patches of E=PH*PW*C values
        B, N, E = x.shape
        x += self.pos_embedding(torch.arange(N, device=x.device)).reshape(1, N, E) # positional embeddings
        x += self.t_embedding(t).reshape(B, 1, E) # timestep embedding, added to all patches of an image
        if label is not None: # class embedding, added to all patches of an image
            x += self.class_embedding(label).reshape(B, 1, E)
        x = self.blocks(x)
        x = self.decoder(x) # (B, N, E) -> (B, N, 2E)
        return self.unpatchify(x, shapes) # ε_θ and v
