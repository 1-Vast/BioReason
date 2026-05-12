"""Latent reasoning backbone components."""

import torch
import torch.nn as nn
from .base import MLP, ResidualBlock


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: y = gamma(cond)*x + beta(cond)"""

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, dim)
        self.beta_proj = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class LatentBlock(nn.Module):
    """Single-step latent reasoning block.

    Supported modes:
    - transformer: context-aware self-attention block
    - mlp: residual MLP block for ablation
    """

    def __init__(self, dim, hidden=None, heads=4, dropout=0.1, mode="transformer"):
        super().__init__()
        self.mode = mode
        if mode == "transformer":
            self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            h = hidden or dim * 4
            self.ffn = nn.Sequential(
                nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(h, dim), nn.Dropout(dropout),
            )
        elif mode == "mlp":
            self.block = ResidualBlock(dim, hidden=hidden, dropout=dropout)
        else:
            raise ValueError(f"Unsupported LatentBlock mode: {mode}")

    def forward(self, z, context=None):
        """z: [B, dim] -> [B, dim]"""
        if self.mode == "transformer":
            z_3d = z.unsqueeze(1)
            if context is not None:
                z_3d = torch.cat([z_3d, context.unsqueeze(1)], dim=1)
            attn_out, _ = self.attn(z_3d, z_3d, z_3d)
            z_out = self.norm1(z + attn_out[:, 0])
            z_out = self.norm2(z_out + self.ffn(z_out))
            return z_out
        elif self.mode == "mlp":
            return self.block(z)
