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
  BioReason    - full model (CellEncoder + PertEncoder + Reasoner + ExprDecoder)
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
    """Single-step latent biological reasoning block.

    Updates latent state using self-attention or MLP, conditioned on
    the context embedding (cell + perturbation + covariates fused).
    """

    def __init__(self, dim, hidden=None, heads=4, dropout=0.1, mode="transformer"):
        super().__init__()
        self.latent_block = LatentBlock(dim, hidden=hidden, heads=heads,
                                         dropout=dropout, mode=mode)
        self.film = FiLM(dim, dim)

    def forward(self, z, context):
        """z: [B, dim], context: [B, dim] -> [B, dim]"""
        z = self.film(z, context)
        z = self.latent_block(z)
        return z


class EvidenceGate(nn.Module):
    """Gate that injects biological evidence into the latent state.

    Training: evidence is provided (DEG, pathway, TF scores).
    Inference: evidence=None -> gate returns identity (no evidence leak).
    """

    def __init__(self, dim, dropout=0.1, mode="film"):
        super().__init__()
        self.mode = mode
        if mode == "film":
            self.gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim * 2),
            )
        elif mode == "add":
            self.gate = MLP(dim, dim, hidden=dim * 2, layers=2, dropout=dropout)
        elif mode == "cross":
            self.gate = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=dropout)
            self.norm = nn.LayerNorm(dim)

    def forward(self, z, evidence=None):
        """z: [B, dim], evidence: [B, dim] or None -> [B, dim]"""
        if evidence is None:
            return z  # inference mode: no evidence

        if self.mode == "film":
            gb = self.gate(evidence)
            gamma, beta = gb.chunk(2, dim=-1)
            z = gamma * z + beta
        elif self.mode == "add":
            z = z + self.gate(evidence)
        elif self.mode == "cross":
            zq = z.unsqueeze(1)
            ek = evidence.unsqueeze(1)
            attn_out, _ = self.gate(zq, ek, ek)
            z = self.norm(z + attn_out.squeeze(1))
        return z


class Reasoner(nn.Module):
    """Multi-step latent biological reasoning engine.

    Takes cell embedding, perturbation embedding, and optional covariates,
    iteratively refines a latent biological reasoning state z_bio over
    multiple steps.

    Key params:
      dim, steps, heads, dropout
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

        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, cell_emb, pert_emb, cov_emb=None, evidence=None,
                return_all=False):
        """cell_emb: [B, dim], pert_emb: [B, dim], cov_emb: [B, dim] or None
        evidence: [B, dim] or None -> [B, dim]"""
        B = cell_emb.size(0)
        device = cell_emb.device

        parts = [cell_emb, pert_emb]
        if cov_emb is not None:
            parts.append(cov_emb)
        else:
            parts.append(torch.zeros(B, self.dim, device=device))
        context = self.context_fuse(torch.cat(parts, dim=-1))

        z = self.init_proj(context)
        z_all = []

        for i in range(self.steps):
            z = self.evidence_gate(z, evidence)
            z = self.reason_steps[i](z, context)
            if return_all:
                z_all.append(z)

        z = self.out_proj(z)

        if return_all:
            return z, z_all
        return z


class BioReason(nn.Module):
    """Full BioReason model for single-cell perturbation prediction.

    Flow:
      x (control)       -> CellEncoder    -> cell_emb
      pert              -> PertEncoder    -> pert_emb
      cov (optional)    -> CovEncoder     -> cov_emb
      evidence (opt)    -> (EvidenceGate) -> injected into z_bio
      z_bio = Reasoner(cell_emb, pert_emb, cov_emb, evidence)
      delta = ExprDecoder(cell_emb, z_bio)
      pred = x + delta

    Three training stages:
      Stage 1: basic warm-up (no evidence)
      Stage 2: evidence-guided latent training
      Stage 3: latent alignment (no evidence, align with teacher latents)
    """

    def __init__(self, input_dim, dim=256, hidden=512, steps=8, heads=4,
                 dropout=0.1, residual=True, pert_mode="id", pert_agg="mean",
                 num_perts=10000, cov_dims=None, evidence_mode="film"):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.residual = residual

        self.cell_encoder = CellEncoder(input_dim, dim, hidden=hidden, dropout=dropout)
        self.pert_encoder = PertEncoder(num_perts, dim, hidden=hidden,
                                         pert_mode=pert_mode, pert_agg=pert_agg,
                                         dropout=dropout)
        self.cov_encoder = CovEncoder(cov_dims or {}, dim, dropout=dropout)
        self.evidence_encoder = MLP(dim, dim, hidden=hidden, layers=2, dropout=dropout)

        self.reasoner = Reasoner(dim, steps=steps, hidden=hidden,
                                  heads=heads, dropout=dropout,
                                  evidence_mode=evidence_mode)

        self.decoder = ExprDecoder(dim, input_dim, hidden=hidden, dropout=dropout)
        self.evidence_head = MLP(dim, dim, hidden=hidden, layers=1, dropout=dropout)

    def forward(self, x, pert, cov=None, evidence=None, target_latent=None,
                return_latent=True, detach_latent=False):
        """
        x:             [B, G]  control cell expression
        pert:          perturbation ids or multi-hot [B] / [B, N]
        cov:           dict[str, LongTensor[B]]  optional covariates
        evidence:      [B, E] or None  biological evidence (train only)
        target_latent: [B, dim] or None  teacher latent for alignment
        return_latent: bool
        detach_latent: bool  detach cell_emb gradient before decoder,
                       prevents the model from bypassing z_bio via a
                       direct cell_emb→delta shortcut path.

        Returns dict: pred, delta, latent, evidence_pred
        """
        cell_emb = self.cell_encoder(x)
        pert_emb = self.pert_encoder(pert)
        cov_emb = self.cov_encoder(cov) if cov else None

        evidence_emb = None
        if evidence is not None:
            evidence_emb = self.evidence_encoder(evidence)

        z = self.reasoner(cell_emb, pert_emb, cov_emb, evidence_emb)

        if detach_latent:
            cell_emb_det = cell_emb.detach()
        else:
            cell_emb_det = cell_emb

        delta = self.decoder(cell_emb_det, z)
        pred = x + delta if self.residual else delta

        evidence_pred = self.evidence_head(z) if evidence is not None else None

        out = {"pred": pred, "delta": delta}
        if return_latent:
            out["latent"] = z
        if evidence_pred is not None:
            out["evidence_pred"] = evidence_pred

        return out

    def encode(self, x, pert, cov=None):
        """Encode inputs to latent state (inference, no evidence)."""
        cell_emb = self.cell_encoder(x)
        pert_emb = self.pert_encoder(pert)
        cov_emb = self.cov_encoder(cov) if cov else None
        return self.reasoner(cell_emb, pert_emb, cov_emb, evidence=None)

    def predict(self, x, z):
        """Predict perturbed expression from expression + latent state."""
        cell_emb = self.cell_encoder(x)
        delta = self.decoder(cell_emb, z)
        return x + delta if self.residual else delta
