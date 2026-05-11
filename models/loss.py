"""BioLoss: perturbation prediction loss functions.

Losses:
  - expression reconstruction loss (MSE)
  - delta loss (MSE)
  - DEG weighted loss
  - latent alignment loss
  - evidence prediction loss
  - MMD distribution loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mmd_loss(x, y, kernel="rbf", bandwidths=(0.1, 1.0, 10.0)):
    """Maximum Mean Discrepancy between distributions x, y: [B, D]."""
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = (x * x).sum(1).unsqueeze(1)
    ry = (y * y).sum(1).unsqueeze(1)

    dist_xx = rx + rx.t() - 2 * xx
    dist_yy = ry + ry.t() - 2 * yy
    dist_xy = rx + ry.t() - 2 * xy

    loss = 0.0
    for bw in bandwidths:
        k_xx = torch.exp(-dist_xx / (2 * bw ** 2))
        k_yy = torch.exp(-dist_yy / (2 * bw ** 2))
        k_xy = torch.exp(-dist_xy / (2 * bw ** 2))
        n, m = x.size(0), y.size(0)
        loss += k_xx.sum() / (n * n) + k_yy.sum() / (m * m) - 2 * k_xy.sum() / (n * m)
    return loss / len(bandwidths)


def top_deg_mask(batch, top_k=50):
    """Create mask selecting top DEGs by absolute mean delta across batch."""
    if "deg_mask" in batch:
        return batch["deg_mask"]
    with torch.no_grad():
        y = batch.get("y", batch["x"] + batch.get("delta", 0).to(batch["x"].device))
        x = batch["x"]
        abs_delta = (y - x).abs().mean(dim=0)
        _, top_idx = torch.topk(abs_delta, min(top_k, abs_delta.size(0)))
        mask = torch.zeros(abs_delta.size(0), device=abs_delta.device)
        mask[top_idx] = 1.0
    return mask


class BioLoss(nn.Module):
    """Combined loss for BioReason training.

    Args:
      weights: dict[str, float]  loss weights from config
    """

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            "expr": 1.0, "delta": 1.0, "deg": 2.0,
            "latent": 1.0, "evidence": 1.0, "mmd": 0.1,
        }
        self.mse = nn.MSELoss()

    def forward(self, out, batch, stage=1):
        """
        out:   dict  model output (pred, delta, latent, evidence_pred)
        batch: dict  data batch (x, y, evidence, target_latent)
        stage: int   1, 2, or 3

        Returns dict: loss, expr, delta, deg, latent, evidence, mmd
        """
        pred = out["pred"]
        delta_pred = out["delta"]
        y = batch["y"]
        x = batch["x"]
        delta_true = y - x

        losses = {}

        # 1. Expression loss
        losses["expr"] = self.mse(pred, y)

        # 2. Delta loss
        losses["delta"] = self.mse(delta_pred, delta_true)

        # 3. DEG-weighted loss
        deg_mask = top_deg_mask(batch, top_k=50)
        losses["deg"] = (self.mse(pred, y) * deg_mask.unsqueeze(0)).mean()

        # 4. Latent alignment (stage 3)
        losses["latent"] = torch.tensor(0.0, device=pred.device)
        if stage == 3 and batch.get("target_latent") is not None:
            losses["latent"] = self.mse(out.get("latent", pred[:, :1]),
                                         batch["target_latent"])

        # 5. Evidence prediction (stage 2)
        losses["evidence"] = torch.tensor(0.0, device=pred.device)
        if stage == 2 and out.get("evidence_pred") is not None:
            targets = batch.get("evidence")
            if targets is not None:
                losses["evidence"] = self.mse(out["evidence_pred"], targets)

        # 6. MMD
        losses["mmd"] = torch.tensor(0.0, device=pred.device)
        if out.get("latent") is not None and stage >= 2:
            z = out["latent"]
            if z.size(0) >= 2:
                rand_idx = torch.randperm(z.size(0), device=z.device)
                losses["mmd"] = mmd_loss(z, z[rand_idx])

        # Total
        total = torch.tensor(0.0, device=pred.device)
        for k, v in losses.items():
            w = self.weights.get(k, 0.0)
            if w > 0:
                total = total + w * v

        losses["loss"] = total
        return losses
