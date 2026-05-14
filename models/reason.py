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
        return self.latent_block(self.film(z, context), context=context)


class EvidenceGate(nn.Module):
    """Confidence-aware evidence gate. Trust score modulates injection strength.

    Training: evidence provided (DEG, pathway, TF scores).
    Inference: evidence=None → identity (no evidence leak).
    """

    def __init__(self, dim, dropout=0.1, mode="film", use_conf=True,
                 adaptive=False, evidence_gate_init_bias=-1.5,
                 evidence_delta_cap_ratio=0.1, use_reliability=True):
        super().__init__()
        self.mode = mode
        self.use_conf = use_conf
        self.adaptive = adaptive
        self.evidence_delta_cap_ratio = evidence_delta_cap_ratio
        self.use_reliability = use_reliability
        self._last_gate = None
        self._last_reliability = None
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
        elif mode == "gate_add":
            # MLP to project evidence into dim
            self.gate = MLP(dim, dim, hidden=dim * 2, layers=2, dropout=dropout)
            # Gate projection: sigmoid(W * [z, e_proj, context]) -> [B, dim]
            # Always use dim*3 to handle the presence of context; zero-pad when absent
            self.gate_proj = nn.Linear(dim * 3, dim)
            # Initialize gate bias to -2.0 so initial sigmoid(-2) ≈ 0.12
            nn.init.constant_(self.gate_proj.bias, -2.0)
            # Residual projection: evidence -> delta_z
            self.delta_proj = nn.Linear(dim, dim)
            # Adaptive gate: per-sample modulation
            if adaptive:
                self.adapt_proj = nn.Linear(dim * 2 + 1, 1)
                nn.init.constant_(self.adapt_proj.bias, evidence_gate_init_bias)
            if use_reliability:
                self.reliability_proj = nn.Linear(dim * 2 + 1, 1)
                nn.init.constant_(self.reliability_proj.bias, evidence_gate_init_bias)
        else:
            raise ValueError(f"Unsupported evidence mode: {mode}")
        if use_conf:
            self.score = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid()
            )

    def forward(self, z, evidence=None, context=None, return_score=False,
                evidence_conf=None):
        if evidence is None:
            self._last_gate = None
            self._last_reliability = None
            return (z, None) if return_score else z
        if self.mode == "film":
            gamma, beta = self.gate(evidence).chunk(2, dim=-1)
            if self.use_conf and hasattr(self, 'score'):
                trust = self.score(torch.cat([z, evidence], dim=-1))
                gamma = gamma * trust
                beta = beta * trust
            else:
                trust = torch.ones(z.size(0), 1, device=z.device)
            if evidence_conf is not None:
                conf = evidence_conf.unsqueeze(-1) if evidence_conf.dim() <= 1 else evidence_conf
                gamma = gamma * conf
                beta = beta * conf
            z = gamma * z + beta
        elif self.mode == "add":
            delta_e = self.gate(evidence)
            if evidence_conf is not None:
                conf = evidence_conf.unsqueeze(-1) if evidence_conf.dim() <= 1 else evidence_conf
                delta_e = delta_e * conf
            z = z + delta_e
            trust = torch.ones(z.size(0), 1, device=z.device) if return_score else None
        elif self.mode == "cross":
            zq = z.unsqueeze(1)
            ek = evidence.unsqueeze(1)
            attn_out, _ = self.gate(zq, ek, ek)
            z = self.norm(z + attn_out.squeeze(1))
            trust = torch.ones(z.size(0), 1, device=z.device) if return_score else None
        elif self.mode == "gate_add":
            e_proj = self.gate(evidence)  # MLP: [B, dim] -> [B, dim]
            # Gate: sigmoid(W_gate * [z, e_proj, context])
            if context is not None:
                gate_input = torch.cat([z, e_proj, context], dim=-1)
            else:
                # Zero-pad for absent context
                gate_input = torch.cat([z, e_proj, torch.zeros_like(z)], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, dim]
            # Adaptive per-sample modulation
            if self.adaptive and hasattr(self, 'adapt_proj'):
                ev_conf_vec = evidence_conf.view(-1, 1) if evidence_conf is not None else torch.ones(z.size(0), 1, device=z.device)
                adapt_factor = torch.sigmoid(self.adapt_proj(torch.cat([z, evidence, ev_conf_vec], -1)))
                gate = gate * adapt_factor
            if self.use_reliability and hasattr(self, 'reliability_proj'):
                ev_conf_vec = evidence_conf.view(-1, 1) if evidence_conf is not None else torch.ones(z.size(0), 1, device=z.device)
                reliability = torch.sigmoid(self.reliability_proj(torch.cat([z, evidence, ev_conf_vec], -1)))
                gate = gate * reliability
                self._last_reliability = reliability
            else:
                self._last_reliability = None
            if evidence_conf is not None:
                conf = evidence_conf.unsqueeze(-1) if evidence_conf.dim() <= 1 else evidence_conf
                gate = gate * conf
            self._last_gate = gate  # store for diagnostics
            delta_e = self.delta_proj(e_proj)  # [B, dim]
            cap = z.detach().norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.evidence_delta_cap_ratio
            scale = (cap / delta_e.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
            delta_e = delta_e * scale
            z = z + gate * delta_e
            trust = torch.ones(z.size(0), 1, device=z.device) if return_score else None
        return (z, trust) if return_score else z


class Reasoner(nn.Module):
    """Multi-step latent biological reasoning engine.

    Fuses cell + pert + cov into context, initializes latent,
    iterates reasoning steps with optional evidence injection.
    """

    def __init__(self, dim, steps=8, hidden=None, heads=4, dropout=0.1,
                 evidence_mode="film", reason_mode="transformer",
                 use_evidence_conf=True, adaptive_evidence_gate=False,
                 evidence_gate_init_bias=-1.5, evidence_delta_cap_ratio=0.1,
                 use_evidence_reliability=True):
        super().__init__()
        self.dim = dim
        self.steps = steps
        if reason_mode not in ("transformer", "mlp"):
            raise ValueError(f"Unsupported reason_mode: {reason_mode}")
        self.reason_mode = reason_mode
        self.context_fuse = MLP(dim * 3, dim, hidden=hidden, layers=2, dropout=dropout)
        self.init_proj = nn.Linear(dim, dim)
        self.reason_steps = nn.ModuleList([
            ReasonStep(dim, hidden=hidden, heads=heads, dropout=dropout, mode=reason_mode)
            for _ in range(steps)
        ])
        self.evidence_gate = EvidenceGate(dim, dropout=dropout, mode=evidence_mode, use_conf=use_evidence_conf,
                                          adaptive=adaptive_evidence_gate, evidence_gate_init_bias=evidence_gate_init_bias,
                                          evidence_delta_cap_ratio=evidence_delta_cap_ratio,
                                          use_reliability=use_evidence_reliability)
        self.out_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, cell_emb, pert_emb, cov_emb=None, evidence=None, return_trust=False,
                evidence_conf=None):
        B = cell_emb.size(0)
        parts = [cell_emb, pert_emb]
        parts.append(cov_emb if cov_emb is not None
                      else torch.zeros(B, self.dim, device=cell_emb.device))
        context = self.context_fuse(torch.cat(parts, dim=-1))
        z = self.init_proj(context)
        
        # FIX: Apply evidence ONLY at step 0 as initialization, not at every step.
        # Applying FiLM at each step with near-identical evidence vectors (hash collision)
        # drowns out perturbation-specific signals. Single-step injection preserves
        # perturbation discriminability.
        trust_all = []
        if evidence is not None:
            if return_trust:
                z, trust = self.evidence_gate(z, evidence, context=context,
                                              return_score=True,
                                              evidence_conf=evidence_conf)
                trust_all.append(trust)
            else:
                z = self.evidence_gate(z, evidence, context=context,
                                       evidence_conf=evidence_conf)
        else:
            self.evidence_gate._last_gate = None
        
        for step_idx in range(self.steps):
            # Skip evidence injection at reasoning steps (already applied at init)
            z = self.reason_steps[step_idx](z, context)
        z = self.out_proj(z)
        if return_trust:
            t = torch.stack(trust_all, dim=1).mean(dim=1) if trust_all else None
            return z, t
        return z


class BioReason(nn.Module):
    """Full BioReason model.

    Parameters:
      evidence_dim: if None, evidence encoder/head are not created.
    """

    def __init__(self, input_dim, dim=256, hidden=512, steps=8, heads=4,
                 dropout=0.1, residual=True, pert_mode="id", pert_agg="mean",
                 num_perts=10000, cov_dims=None, evidence_dim=None,
                 evidence_mode="film", reason_mode="transformer",
                 use_evidence_conf=True, pert_embed_strength=1.0,
                 evidence_strength=0.2, evidence_pert_alpha=0.5,
                 use_evidence_as_pert_init=False,
                 adaptive_evidence_gate=False, evidence_gate_init_bias=-1.5,
                 evidence_delta_cap_ratio=0.1, use_evidence_reliability=True):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.residual = residual
        self.evidence_dim = evidence_dim
        self.reason_mode = reason_mode
        self.pert_embed_strength = pert_embed_strength
        self.evidence_strength = evidence_strength
        self.evidence_pert_alpha = evidence_pert_alpha
        self.use_evidence_as_pert_init = use_evidence_as_pert_init
        self.evidence_delta_cap_ratio = evidence_delta_cap_ratio

        self.cell_encoder = CellEncoder(input_dim, dim, hidden=hidden, dropout=dropout)
        self.pert_encoder = PertEncoder(num_perts, dim, hidden=hidden,
                                         pert_mode=pert_mode, pert_agg=pert_agg,
                                         dropout=dropout, evidence_dim=evidence_dim,
                                         evidence_pert_alpha=evidence_pert_alpha,
                                         evidence_delta_cap_ratio=evidence_delta_cap_ratio)
        self.cov_encoder = CovEncoder(cov_dims or {}, dim, dropout=dropout)

        # Evidence encoder: only if evidence_dim is set
        self.evidence_encoder = None
        if evidence_dim is not None:
            # SMALL linear encoder: prevents hash-similar evidence from being
            # amplified into large FiLM parameters. Keeps evidence as a subtle hint.
            self.evidence_encoder = nn.Sequential(
                nn.Linear(evidence_dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, dim),
            )

        self.reasoner = Reasoner(dim, steps=steps, hidden=hidden, heads=heads,
                                    dropout=dropout, evidence_mode=evidence_mode,
                                    reason_mode=reason_mode,
                                    use_evidence_conf=use_evidence_conf,
                                    adaptive_evidence_gate=adaptive_evidence_gate,
                                    evidence_gate_init_bias=evidence_gate_init_bias,
                                    evidence_delta_cap_ratio=evidence_delta_cap_ratio,
                                    use_evidence_reliability=use_evidence_reliability)
        self.decoder = ExprDecoder(dim, input_dim, hidden=hidden, dropout=dropout)

        # Evidence prediction head: only if evidence_dim is set
        self.evidence_head = None
        if evidence_dim is not None:
            # Small head: prevents evidence prediction from dominating latent
            self.evidence_head = nn.Linear(dim, evidence_dim)

    def forward(self, x, pert, cov=None, evidence=None, target_latent=None,
                return_latent=True, detach_latent=False, evidence_conf=None):
        """
        Returns dict with fixed keys: pred, delta, latent, evidence_pred, target_latent, trust
        """
        cell_emb = self.cell_encoder(x)
        
        # Evidence: encode first so it can feed into pert_encoder for id_plus_evidence
        evidence_emb = None
        if evidence is not None:
            if self.evidence_encoder is None:
                raise ValueError(
                    "evidence_dim must be set in BioReason() when evidence is used. "
                    f"Got evidence tensor of shape {tuple(evidence.shape)}."
                )
            evidence_emb = self.evidence_encoder(evidence)
            evidence_emb = evidence_emb * self.evidence_strength

        # Pert: pass evidence_emb for id_plus_evidence mode
        if self.pert_encoder.pert_mode == "id_plus_evidence":
            pert_emb = self.pert_encoder(pert, evidence_emb=evidence_emb)
        else:
            pert_emb = self.pert_encoder(pert)
        pert_emb = pert_emb * self.pert_embed_strength
        cov_emb = self.cov_encoder(cov) if cov else None

        z, trust = self.reasoner(cell_emb, pert_emb, cov_emb, evidence_emb,
                                   return_trust=True, evidence_conf=evidence_conf)

        # detach_latent: detach z from the decoder, not cell_emb
        z_for_dec = z.detach() if detach_latent else z
        delta = self.decoder(cell_emb, z_for_dec)
        pred = x + delta if self.residual else delta

        # Evidence prediction
        evidence_pred = None
        if evidence is not None and self.evidence_head is not None:
            evidence_pred = self.evidence_head(z)

        # Evidence gate for diagnostics
        evidence_gate = getattr(self.reasoner.evidence_gate, '_last_gate', None)
        evidence_reliability = getattr(self.reasoner.evidence_gate, '_last_reliability', None)

        return {
            "pred": pred,
            "delta": delta,
            "latent": z if return_latent else None,
            "evidence_pred": evidence_pred,
            "evidence_emb": evidence_emb,  # encoded evidence before reasoner
            "evidence_gate": evidence_gate,
            "evidence_reliability": evidence_reliability,
            "target_latent": target_latent,
            "trust": trust,
        }

    def forward_latent(self, x, pert, cov=None, evidence=None,
                       evidence_conf=None):
        """Return only z_bio. Used for stage 2 latent export / stage 3 alignment."""
        cell_emb = self.cell_encoder(x)
        
        # Evidence: encode first for potential pert_encoder consumption
        evidence_emb = (self.evidence_encoder(evidence) if evidence is not None
                         and self.evidence_encoder is not None else None)
        if evidence_emb is not None:
            evidence_emb = evidence_emb * self.evidence_strength

        # Pert: pass evidence_emb for id_plus_evidence mode
        if self.pert_encoder.pert_mode == "id_plus_evidence":
            pert_emb = self.pert_encoder(pert, evidence_emb=evidence_emb)
        else:
            pert_emb = self.pert_encoder(pert)
        pert_emb = pert_emb * self.pert_embed_strength
        cov_emb = self.cov_encoder(cov) if cov else None

        return self.reasoner(cell_emb, pert_emb, cov_emb, evidence_emb,
                              evidence_conf=evidence_conf)

    def encode(self, x, pert, cov=None):
        return self.forward_latent(x, pert, cov=cov, evidence=None)

    def predict(self, x, z):
        cell_emb = self.cell_encoder(x)
        delta = self.decoder(cell_emb, z)
        return x + delta if self.residual else delta

    def freeze_except_reasoner(self):
        """Backward-compatible alias for latent-only BP training."""
        self.freeze_main_path_for_latent()

    def set_trainable(self, names):
        """Only modules whose name contains one of `names` remain trainable."""
        names = tuple(names or ())
        for name, param in self.named_parameters():
            param.requires_grad = any(key in name for key in names)

    def freeze_main_path_for_latent(self):
        """Freeze cell_encoder and decoder for latent-only backpropagation.

        Monet latent-only backpropagation maps here to BioReason latent-only BP:
        evidence/alignment losses update the latent generation path instead of
        being minimized through cell_encoder or decoder shortcuts.
        """
        self.set_trainable((
            "reasoner",
            "evidence_encoder",
            "evidence_head",
            "pert_encoder",
            "cov_encoder",
        ))

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
