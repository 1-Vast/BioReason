"""PertEncoder: encode perturbation condition into embedding.

Supports:
  - categorical perturbation id (single gene/drug)
  - multi-hot perturbation vector (gene combinations)
  - continuous perturbation descriptor (drug dose/time)
  - aggregation: mean, sum, attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import MLP


class PertEncoder(nn.Module):
    """Encodes perturbation condition into a latent embedding.

    pert_mode:
      "id"         - single categorical ID
      "multihot"   - multi-hot vector of gene IDs [B, N]
      "continuous" - continuous descriptor [B, F]

    pert_agg (for multihot):
      "mean", "sum", "attention"
    """

    def __init__(self, num_perts, dim, hidden=None, pert_mode="id", pert_agg="mean",
                 dropout=0.1):
        super().__init__()
        self.pert_mode = pert_mode
        self.pert_agg = pert_agg
        self.dim = dim

        if pert_mode in ("id", "multihot"):
            self.embed = nn.Embedding(num_perts, dim, padding_idx=-1)
            self.dropout = nn.Dropout(dropout)

        if pert_agg == "attention":
            self.attn_proj = nn.Linear(dim, 1)

        if pert_mode == "continuous":
            self.cont_encoder = MLP(hidden or dim, dim, hidden=hidden, layers=2, dropout=dropout)

        self.out_proj = MLP(dim, dim, hidden=hidden, layers=2, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, pert):
        """pert -> [B, dim]"""
        if self.pert_mode == "id":
            x = self.embed(pert)

        elif self.pert_mode == "multihot":
            mask = (pert >= 0).float().unsqueeze(-1)
            x = self.embed(pert.clamp(min=0))

            if self.pert_agg == "mean":
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            elif self.pert_agg == "sum":
                x = (x * mask).sum(dim=1)
            elif self.pert_agg == "attention":
                w = self.attn_proj(x).squeeze(-1)
                w = w.masked_fill(pert < 0, -1e9)
                w = F.softmax(w, dim=-1).unsqueeze(-1)
                x = (x * w).sum(dim=1)

        elif self.pert_mode == "continuous":
            x = self.cont_encoder(pert)

        else:
            raise ValueError(f"Unknown pert_mode: {self.pert_mode}")

        x = self.out_proj(x)
        x = self.out_norm(x)
        return x
