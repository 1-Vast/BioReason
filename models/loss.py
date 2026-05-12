"""BioLoss: perturbation prediction loss functions.

Stage-gated, missing-field-safe. Uses target_latent_mask for stage 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mmd_loss(x, y, bandwidths=(0.1, 1.0, 10.0)):
    """MMD with multi-bandwidth RBF kernel."""
    xx = torch.mm(x, x.t()); yy = torch.mm(y, y.t()); xy = torch.mm(x, y.t())
    rx = (x * x).sum(1).unsqueeze(1); ry = (y * y).sum(1).unsqueeze(1)
    d_xx = (rx + rx.t() - 2 * xx).clamp(min=0)
    d_yy = (ry + ry.t() - 2 * yy).clamp(min=0)
    d_xy = (rx + ry.t() - 2 * xy).clamp(min=0)
    loss = 0.0
    for bw in bandwidths:
        k_xx = torch.exp(-d_xx / (2 * bw**2))
        k_yy = torch.exp(-d_yy / (2 * bw**2))
        k_xy = torch.exp(-d_xy / (2 * bw**2))
        n, m = x.size(0), y.size(0)
        loss += k_xx.sum()/(n*n) + k_yy.sum()/(m*m) - 2*k_xy.sum()/(n*m)
    return loss / len(bandwidths)


def get_deg_mask(batch, top_k=50):
    if "deg_mask" in batch:
        return batch["deg_mask"]
    with torch.no_grad():
        delta_true = batch["y"] - batch["x"]
        abs_delta = delta_true.abs().mean(dim=0)
        _, top = torch.topk(abs_delta, min(top_k, abs_delta.size(0)))
        mask = torch.zeros(abs_delta.size(0), device=abs_delta.device)
        mask[top] = 1.0
    return mask


class BioLoss(nn.Module):
    """Combined loss with stage-gating and target_latent_mask support."""

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            "expr": 1.0, "delta": 1.0, "deg": 2.0,
            "latent": 1.0, "evidence": 1.0, "mmd": 0.1,
        }
        self.top_deg = self.weights.pop("top_deg", 50)
        self.latent_metric = self.weights.pop("latent_metric", "cosine")
        self.mse_none = nn.MSELoss(reduction="none")
        self.mse_mean = nn.MSELoss(reduction="mean")

    def forward(self, out, batch, stage=1):
        pred = out["pred"]; delta_pred = out["delta"]
        x, y = batch["x"], batch["y"]; delta_true = y - x
        dev = pred.device
        losses = {}

        # 1-2. Expression + delta
        losses["expr"] = self.mse_mean(pred, y)
        losses["delta"] = self.mse_mean(delta_pred, delta_true)

        # 3. DEG-weighted
        deg_mask = get_deg_mask(batch, top_k=self.top_deg)
        err = (pred - y) ** 2
        losses["deg"] = (err * deg_mask.unsqueeze(0)).sum() / (deg_mask.sum() * pred.size(0)).clamp_min(1.0)

        # 4. Latent alignment (stage 3) — uses target_latent_mask
        losses["latent"] = torch.tensor(0.0, device=dev)
        target = batch.get("target_latent")
        mask = batch.get("target_latent_mask")
        z = out.get("latent")
        if stage == 3 and target is not None and z is not None:
            if mask is not None:
                z_valid = z[mask]
                t_valid = target[mask]
                if z_valid.numel() == 0:
                    pass  # latent_loss stays 0
                elif self.latent_metric == "cosine":
                    losses["latent"] = 1.0 - F.cosine_similarity(z_valid, t_valid.detach(), dim=-1).mean()
                else:
                    losses["latent"] = self.mse_mean(z_valid, t_valid.detach())
            else:
                if self.latent_metric == "cosine":
                    losses["latent"] = 1.0 - F.cosine_similarity(z, target.detach(), dim=-1).mean()
                else:
                    losses["latent"] = self.mse_mean(z, target.detach())

        # 5. Evidence (stage 2)
        losses["evidence"] = torch.tensor(0.0, device=dev)
        ev_pred = out.get("evidence_pred")
        ev_true = batch.get("evidence")
        if stage == 2 and ev_pred is not None and ev_true is not None:
            if ev_pred.shape != ev_true.shape:
                raise ValueError(f"evidence_pred {tuple(ev_pred.shape)} != evidence {tuple(ev_true.shape)}")
            losses["evidence"] = self.mse_mean(ev_pred, ev_true)

        # 6. MMD(pred, y)
        losses["mmd"] = torch.tensor(0.0, device=dev)
        if pred.size(0) >= 2:
            losses["mmd"] = mmd_loss(pred, y)

        # Total
        total = torch.tensor(0.0, device=dev)
        for k, v in losses.items():
            w = self.weights.get(k, 0.0)
            if w > 0:
                total = total + w * v
        losses["loss"] = total
        return losses
