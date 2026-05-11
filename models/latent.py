"""Latent reasoning backbone components.

- LatentBlock: single-step latent state update
- CrossBlock: cross-attention between latent and condition
- FiLM: feature-wise linear modulation for conditioning
"""

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
    """Single-step latent reasoning block. Supports transformer/mlp/gru modes."""

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
        elif mode == "gru":
            self.gru = nn.GRUCell(dim, dim)

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
        elif self.mode == "gru":
            return self.gru(z, z)


class CrossBlock(nn.Module):
    """Cross-attention: latent attends to condition tokens."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attn_out, _ = self.attn(query, key_value, key_value)
        x = self.norm(query + attn_out)
        return self.norm2(x + self.ffn(x))
