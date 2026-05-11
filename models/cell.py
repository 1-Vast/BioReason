"""CellEncoder and CovEncoder.

Input:  x [B, G]  gene expression vector
Output: z [B, dim]  cell embedding
"""

import torch
import torch.nn as nn
from .base import MLP


class CellEncoder(nn.Module):
    """Encodes gene expression into a latent cell embedding.

    Supports MLP encoder with optional layer/batch norm.
    """

    def __init__(self, input_dim, dim, hidden=None, layers=3, dropout=0.1, norm="layer"):
        super().__init__()
        hidden = hidden or dim * 2
        self.input_norm = nn.LayerNorm(input_dim) if norm == "layer" else (
            nn.BatchNorm1d(input_dim) if norm == "batch" else nn.Identity()
        )
        self.mlp = MLP(input_dim, dim, hidden=hidden, layers=layers, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: [B, G] -> [B, dim]"""
        x = self.input_norm(x)
        x = self.mlp(x)
        x = self.out_norm(x)
        return x


class CovEncoder(nn.Module):
    """Encodes covariate metadata (cell_type, dose, time, batch)."""

    def __init__(self, cov_dims, dim, dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            key: nn.Embedding(num, dim) for key, num in cov_dims.items()
        })
        self.proj = nn.Linear(dim * len(cov_dims), dim) if len(cov_dims) > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, cov):
        """cov: dict[str, LongTensor[B]] -> [B, dim] or None"""
        if not cov:
            return None
        embs = [self.embeddings[key](cov[key]) for key in cov if key in self.embeddings]
        if not embs:
            return None
        x = torch.cat(embs, dim=-1)
        x = self.proj(x)
        return self.dropout(x)
