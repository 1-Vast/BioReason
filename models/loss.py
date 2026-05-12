"""BioLoss — with AMP-safe MMD (float32, max_samples, nan guard)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mmd_loss(x, y, bandwidths=(0.1, 1.0, 10.0), max_samples=None):
    """MMD with multi-bandwidth RBF kernel. Forces float32 for AMP stability."""
    x = x.float(); y = y.float()
    if max_samples is not None and x.size(0) > max_samples:
        idx = torch.randperm(x.size(0), device=x.device)[:max_samples]
        x = x[idx]; y = y[idx]
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
    loss = loss / len(bandwidths)
    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)


def get_deg_mask(batch, top_k=50):
    if "deg_mask" in batch: return batch["deg_mask"]
    with torch.no_grad():
        delta_true = batch["y"] - batch["x"]
        abs_delta = delta_true.abs().mean(dim=0)
        _, top = torch.topk(abs_delta, min(top_k, abs_delta.size(0)))
        mask = torch.zeros(abs_delta.size(0), device=abs_delta.device)
        mask[top] = 1.0
    return mask


class BioLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {"expr":1.0,"delta":1.0,"deg":2.0,"latent":1.0,"evidence":1.0,"mmd":0.1}
        self.top_deg = self.weights.pop("top_deg", 50)
        self.latent_metric = self.weights.pop("latent_metric", "cosine")
        self.mmd_max_samples = self.weights.pop("mmd_max_samples", 128)
        self.mmd_every = self.weights.pop("mmd_every", 1)
        self.mse_none = nn.MSELoss(reduction="none")
        self.mse_mean = nn.MSELoss(reduction="mean")
        self._step_counter = 0

    def forward(self, out, batch, stage=1, step=None):
        if step is not None: self._step_counter = step
        else: self._step_counter += 1

        pred = out["pred"]; delta_pred = out["delta"]
        x, y = batch["x"], batch["y"]; delta_true = y - x
        dev = pred.device; losses = {}

        losses["expr"] = self.mse_mean(pred, y)
        losses["delta"] = self.mse_mean(delta_pred, delta_true)

        deg_mask = get_deg_mask(batch, top_k=self.top_deg)
        err = (pred - y)**2
        losses["deg"] = (err * deg_mask.unsqueeze(0)).sum() / (deg_mask.sum() * pred.size(0)).clamp_min(1.0)

        losses["latent"] = torch.tensor(0.0, device=dev)
        target = batch.get("target_latent"); mask = batch.get("target_latent_mask"); z = out.get("latent")
        if stage == 3 and target is not None and z is not None:
            if mask is not None:
                z_valid = z[mask]; t_valid = target[mask]
                if z_valid.numel() > 0:
                    losses["latent"] = (1.0 - F.cosine_similarity(z_valid, t_valid.detach(), dim=-1).mean()) if self.latent_metric == "cosine" else self.mse_mean(z_valid, t_valid.detach())
            else:
                losses["latent"] = (1.0 - F.cosine_similarity(z, target.detach(), dim=-1).mean()) if self.latent_metric == "cosine" else self.mse_mean(z, target.detach())

        losses["evidence"] = torch.tensor(0.0, device=dev)
        ev_pred = out.get("evidence_pred"); ev_true = batch.get("evidence")
        if stage == 2 and ev_pred is not None and ev_true is not None:
            if ev_pred.shape != ev_true.shape:
                raise ValueError(f"evidence_pred {tuple(ev_pred.shape)} != evidence {tuple(ev_true.shape)}")
            losses["evidence"] = self.mse_mean(ev_pred, ev_true)

        # MMD: skip if not at mmd_every interval
        losses["mmd"] = torch.tensor(0.0, device=dev)
        do_mmd = (self._step_counter % self.mmd_every == 0)
        if pred.size(0) >= 2 and do_mmd:
            losses["mmd"] = mmd_loss(pred, y, max_samples=self.mmd_max_samples)

        total = torch.tensor(0.0, device=dev)
        for k, v in losses.items():
            w = self.weights.get(k, 0.0)
            if w > 0: total = total + w * v
        losses["loss"] = total
        return losses
