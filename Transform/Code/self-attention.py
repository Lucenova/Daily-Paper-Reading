import numpy as np
import torch
from einops import rearrange
from torch import nn


class SelfAttentionAISummer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim * 3, bias=True)
        self.scale_factor = dim ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3

        qvk = self.to_qvk(x)
        q, v, k = tuple(rearrange(qvk, 'b t (d k) -> k b t d', k=3))

        scaled_dot_prod = torch.einsum('b i d, b j d -> b i j', q,
                                       k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod[1:]
            scaled_dot_prod = scaled_dot_prod.mask_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        return torch.einsum('b i j, b j d -> b i d', attention, v)


class MultiHeadSelfAttentionAISummmer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        super().__init__()
        self.dim_head = (int(dim / heads )) if dim_head is None else dim_head
        _dim = dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qvk = self.to_qvk(x)

        q, v, k = tuple(rearrange(qvk, 'b t (d k h) -> k b h t d', k=3, h=self.heads))

        scaled_dot_prod = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.mask_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)

        out = rearrange(out, 'b h t d -> b t (h d)')

        return self.W_0(out)



