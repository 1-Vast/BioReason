"""ExprDecoder: decode latent state to expression difference.

Input:  cell_emb [B, dim] + z_bio [B, dim]
Output: delta  [B, G]  predicted expression change
"""

import torch
import torch.nn as nn
from .base import MLP


class ExprDecoder(nn.Module):
    """Decodes cell embedding + latent biological reasoning state
    into predicted expression difference (delta)."""

    def __init__(self, dim, output_dim, hidden=None, dropout=0.1):
        super().__init__()
        hidden = hidden or dim * 2
        self.mlp = MLP(dim * 2, output_dim, hidden=hidden, layers=3, dropout=dropout)

    def forward(self, cell_emb, z_bio):
        """cell_emb: [B, dim], z_bio: [B, dim] -> [B, G]"""
        x = torch.cat([cell_emb, z_bio], dim=-1)
        return self.mlp(x)
