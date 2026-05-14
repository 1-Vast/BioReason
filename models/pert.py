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
      "id"              - single categorical ID
      "id_plus_evidence" - categorical ID + evidence projection
      "multihot"        - multi-hot vector of gene IDs [B, N]
      "continuous"      - continuous descriptor [B, F]

    pert_agg (for multihot):
      "mean", "sum", "attention"
    """

    def __init__(self, num_perts, dim, hidden=None, pert_mode="id", pert_agg="mean",
                 dropout=0.1, evidence_dim=None, evidence_pert_alpha=0.5,
                 evidence_delta_cap_ratio=0.1):
        super().__init__()
        self.pert_mode = pert_mode
        self.pert_agg = pert_agg
        self.dim = dim
        self.evidence_pert_alpha = evidence_pert_alpha
        self.evidence_delta_cap_ratio = evidence_delta_cap_ratio

        if pert_mode in ("id", "id_plus_evidence", "multihot"):
            self.embed = nn.Embedding(num_perts, dim, padding_idx=-1)
            self.dropout = nn.Dropout(dropout)

        if pert_agg == "attention":
            self.attn_proj = nn.Linear(dim, 1)

        if pert_mode == "continuous":
            self.cont_encoder = MLP(hidden or dim, dim, hidden=hidden, layers=2, dropout=dropout)

        if pert_mode == "id_plus_evidence":
            if evidence_dim is None:
                raise ValueError("evidence_dim must be set for pert_mode='id_plus_evidence'")
            self.evidence_to_pert = MLP(dim, dim, hidden=dim, layers=2, dropout=dropout)

        self.out_proj = MLP(dim, dim, hidden=hidden, layers=2, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, pert, evidence_emb=None):
        """pert -> [B, dim]

        Args:
            pert: perturbation ID tensor [B] or multi-hot [B, N] or continuous [B, F]
            evidence_emb: optional evidence embedding [B, dim] (used in id_plus_evidence mode)
        """
        if self.pert_mode == "id":
            x = self.embed(pert)

        elif self.pert_mode == "id_plus_evidence":
            pert_id_emb = self.embed(pert)
            if evidence_emb is not None:
                evidence_pert = self.evidence_to_pert(evidence_emb)
                cap = pert_id_emb.detach().norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.evidence_delta_cap_ratio
                scale = (cap / evidence_pert.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
                evidence_pert = evidence_pert * scale
                x = pert_id_emb + self.evidence_pert_alpha * evidence_pert
            else:
                x = pert_id_emb

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
