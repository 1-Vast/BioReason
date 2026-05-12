"""
Core idea:
BioReason learns evidence-guided latent biological reasoning states
for single-cell perturbation prediction. During training, biological
evidence can supervise and shape the latent reasoning states. During
inference, the model predicts perturbation responses without evidence.

Architecture:
  x_control, perturbation, covariates
    -> z_bio (latent biological reasoning)
    -> delta_x
    -> x_perturbed

Classes:
  ReasonStep   - single-step latent biological reasoning block
  EvidenceGate - injects biological evidence into latent (train only)
  Reasoner     - multi-step latent biological reasoning engine
  BioReason    - full model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MLP
from .cell import CellEncoder, CovEncoder
from .pert import PertEncoder
from .decoder import ExprDecoder
from .latent import FiLM, LatentBlock


class ReasonStep(nn.Module):
    """Single-step latent biological reasoning.

    FiLM modulation by context, then self-attention / MLP update.
    """

    def __init__(self, dim, hidden=None, heads=4, dropout=0.1, mode="transformer"):
        super().__init__()
        self.latent_block = LatentBlock(dim, hidden=hidden, heads=heads,
                                         dropout=dropout, mode=mode)
        self.film = FiLM(dim, dim)

    def forward(self, z, context):
        return self.latent_block(self.film(z, context))


class EvidenceGate(nn.Module):
    """Injects biological evidence into latent state.

    Training: evidence provided (DEG, pathway, TF scores).
    Inference: evidence=None → identity (no evidence leak).
    """

    def __init__(self, dim, dropout=0.1, mode="film"):
        super().__init__()
        self.mode = mode
        if mode == "film":
            self.gate = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim, dim * 2),
            )
        elif mode == "add":
            self.gate = MLP(dim, dim, hidden=dim * 2, layers=2, dropout=dropout)
        elif mode == "cross":
            self.gate = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=dropout)
            self.norm = nn.LayerNorm(dim)

    def forward(self, z, evidence=None):
        if evidence is None:
            return z
        if self.mode == "film":
            gamma, beta = self.gate(evidence).chunk(2, dim=-1)
            return gamma * z + beta
        elif self.mode == "add":
            return z + self.gate(evidence)
        elif self.mode == "cross":
            zq = z.unsqueeze(1)
            ek = evidence.unsqueeze(1)
            attn_out, _ = self.gate(zq, ek, ek)
            return self.norm(z + attn_out.squeeze(1))
        return z


class Reasoner(nn.Module):
    """Multi-step latent biological reasoning engine.

    Fuses cell + pert + cov into context, initializes latent,
    iterates reasoning steps with optional evidence injection.
    """

    def __init__(self, dim, steps=8, hidden=None, heads=4, dropout=0.1,
                 evidence_mode="film"):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.context_fuse = MLP(dim * 3, dim, hidden=hidden, layers=2, dropout=dropout)
        self.init_proj = nn.Linear(dim, dim)
        self.reason_steps = nn.ModuleList([
            ReasonStep(dim, hidden=hidden, heads=heads, dropout=dropout)
            for _ in range(steps)
        ])
        self.evidence_gate = EvidenceGate(dim, dropout=dropout, mode=evidence_mode)
        self.out_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, cell_emb, pert_emb, cov_emb=None, evidence=None, return_all=False):
        B = cell_emb.size(0)
        parts = [cell_emb, pert_emb]
        parts.append(cov_emb if cov_emb is not None
                      else torch.zeros(B, self.dim, device=cell_emb.device))
        context = self.context_fuse(torch.cat(parts, dim=-1))
        z = self.init_proj(context)
        z_all = []
        for i in range(self.steps):
            z = self.evidence_gate(z, evidence)
            z = self.reason_steps[i](z, context)
            if return_all:
                z_all.append(z)
        z = self.out_proj(z)
        return (z, z_all) if return_all else z


class BioReason(nn.Module):
    """Full BioReason model.

    Parameters:
      evidence_dim: if None, evidence encoder/head are not created.
    """

    def __init__(self, input_dim, dim=256, hidden=512, steps=8, heads=4,
                 dropout=0.1, residual=True, pert_mode="id", pert_agg="mean",
                 num_perts=10000, cov_dims=None, evidence_dim=None,
                 evidence_mode="film"):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.residual = residual
        self.evidence_dim = evidence_dim

        self.cell_encoder = CellEncoder(input_dim, dim, hidden=hidden, dropout=dropout)
        self.pert_encoder = PertEncoder(num_perts, dim, hidden=hidden,
                                         pert_mode=pert_mode, pert_agg=pert_agg,
                                         dropout=dropout)
        self.cov_encoder = CovEncoder(cov_dims or {}, dim, dropout=dropout)

        # Evidence encoder: only if evidence_dim is set
        self.evidence_encoder = None
        if evidence_dim is not None:
            self.evidence_encoder = MLP(evidence_dim, dim, hidden=hidden,
                                         layers=2, dropout=dropout)

        self.reasoner = Reasoner(dim, steps=steps, hidden=hidden, heads=heads,
                                  dropout=dropout, evidence_mode=evidence_mode)
        self.decoder = ExprDecoder(dim, input_dim, hidden=hidden, dropout=dropout)

        # Evidence prediction head: only if evidence_dim is set
        self.evidence_head = None
        if evidence_dim is not None:
            self.evidence_head = MLP(dim, evidence_dim, hidden=hidden, layers=1,
                                      dropout=dropout)

    def forward(self, x, pert, cov=None, evidence=None, target_latent=None,
                return_latent=True, detach_latent=False):
        """
        Returns dict with fixed keys: pred, delta, latent, evidence_pred, target_latent
        """
        cell_emb = self.cell_encoder(x)
        pert_emb = self.pert_encoder(pert)
        cov_emb = self.cov_encoder(cov) if cov else None

        # Evidence
        evidence_emb = None
        if evidence is not None:
            if self.evidence_encoder is None:
                raise ValueError(
                    "evidence_dim must be set in BioReason() when evidence is used. "
                    f"Got evidence tensor of shape {tuple(evidence.shape)}."
                )
            evidence_emb = self.evidence_encoder(evidence)

        z = self.reasoner(cell_emb, pert_emb, cov_emb, evidence_emb)

        # detach_latent: detach z from the decoder, not cell_emb
        z_for_dec = z.detach() if detach_latent else z
        delta = self.decoder(cell_emb, z_for_dec)
        pred = x + delta if self.residual else delta

        # Evidence prediction
        evidence_pred = None
        if evidence is not None and self.evidence_head is not None:
            evidence_pred = self.evidence_head(z)

        return {
            "pred": pred,
            "delta": delta,
            "latent": z if return_latent else None,
            "evidence_pred": evidence_pred,
            "target_latent": target_latent,
        }

    def forward_latent(self, x, pert, cov=None, evidence=None):
        """Return only z_bio. Used for stage 2 latent export / stage 3 alignment."""
        cell_emb = self.cell_encoder(x)
        pert_emb = self.pert_encoder(pert)
        cov_emb = self.cov_encoder(cov) if cov else None
        evidence_emb = (self.evidence_encoder(evidence) if evidence is not None
                         and self.evidence_encoder is not None else None)
        return self.reasoner(cell_emb, pert_emb, cov_emb, evidence_emb)

    def encode(self, x, pert, cov=None):
        return self.forward_latent(x, pert, cov=cov, evidence=None)

    def predict(self, x, z):
        cell_emb = self.cell_encoder(x)
        delta = self.decoder(cell_emb, z)
        return x + delta if self.residual else delta

    def freeze_except_reasoner(self):
        """Freeze all modules except pert_encoder, reasoner, evidence heads."""
        for name, param in self.named_parameters():
            if any(x in name for x in ("pert_encoder", "reasoner",
                                         "evidence_encoder", "evidence_head")):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
