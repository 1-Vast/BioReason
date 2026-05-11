"""Test: verify BioLoss returns proper dict."""
print("Testing BioLoss...")

import torch
from models.loss import BioLoss

B, G = 4, 100
dim = 32

loss_fn = BioLoss()

# Toy data
x = torch.randn(B, G)
y = x + torch.randn(B, G) * 0.1
out = {
    "pred": x + torch.randn(B, G) * 0.05,
    "delta": torch.randn(B, G) * 0.05,
    "latent": torch.randn(B, dim),
    "evidence_pred": torch.randn(B, dim),
}
batch = {"x": x, "y": y, "evidence": torch.randn(B, dim)}

# Stage 1
losses1 = loss_fn(out, batch, stage=1)
assert "loss" in losses1, "loss missing"
assert "expr" in losses1, "expr missing"
assert "delta" in losses1, "delta missing"
assert "deg" in losses1, "deg missing"
assert isinstance(losses1["loss"].item(), float)
print(f"  Stage 1: loss={losses1['loss'].item():.4f}")

# Stage 2
losses2 = loss_fn(out, batch, stage=2)
assert "evidence" in losses2
assert "mmd" in losses2
print(f"  Stage 2: loss={losses2['loss'].item():.4f}, evidence={losses2['evidence'].item():.4f}")

# Stage 3 with target latent
batch3 = {**batch, "target_latent": torch.randn(B, dim)}
losses3 = loss_fn(out, batch3, stage=3)
assert "latent" in losses3
print(f"  Stage 3: loss={losses3['loss'].item():.4f}, latent={losses3['latent'].item():.4f}")

# Missing fields should not crash
batch_min = {"x": x, "y": y}
out_min = {"pred": x, "delta": torch.zeros_like(x)}
losses_min = loss_fn(out_min, batch_min, stage=1)
assert isinstance(losses_min["loss"].item(), float)
print(f"  Minimal batch: loss={losses_min['loss'].item():.4f}")

print("TEST PASSED")
