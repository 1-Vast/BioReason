"""Test: BioLoss with proper DEG, cosine alignment, MMD."""
print("--- test_loss ---")
import torch, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from models.loss import BioLoss

B, G, D = 4, 100, 32
loss_fn = BioLoss()
x = torch.randn(B, G)
y = x + torch.randn(B, G) * 0.5
out = {"pred": y + torch.randn(B, G) * 0.05,
       "delta": torch.randn(B, G) * 0.05,
       "latent": torch.randn(B, D),
       "evidence_pred": torch.randn(B, 16),
       "target_latent": None}
batch = {"x": x, "y": y, "evidence": torch.randn(B, 16)}

# Stage 1
L1 = loss_fn(out, batch, stage=1)
for k in ["loss", "expr", "delta", "deg", "mmd"]:
    assert L1[k].item() >= 0, f"{k}={L1[k]}"
print(f"  1) stage 1 loss={L1['loss']:.4f} OK")

# Stage 2 with evidence
L2 = loss_fn(out, batch, stage=2)
assert L2["evidence"].item() > 0
print(f"  2) stage 2 evidence={L2['evidence']:.4f} OK")

# Stage 3 cosine alignment
batch3 = {"x": x, "y": y, "target_latent": torch.randn(B, D)}
out3 = dict(out, latent=torch.randn(B, D))
L3 = loss_fn(out3, batch3, stage=3)
assert L3["latent"].item() >= 0
print(f"  3) stage 3 latent(cosine)={L3['latent']:.4f} OK")

# Stage 3 MSE alignment
loss_mse = BioLoss({"expr": 1.0, "latent": 1.0, "latent_metric": "mse", "top_deg": 50})
L3m = loss_mse(out3, batch3, stage=3)
assert L3m["latent"].item() > 0

# Minimal batch
Lm = loss_fn({"pred": x, "delta": x*0, "latent": None, "evidence_pred": None, "target_latent": None},
             {"x": x, "y": x}, stage=1)
assert Lm["loss"].item() >= 0
print("  4) minimal batch OK")

# Evidence shape mismatch
try:
    loss_fn({"pred": x, "delta": x*0, "latent": None, "evidence_pred": torch.randn(B, 5), "target_latent": None},
            {"x": x, "y": x, "evidence": torch.randn(B, 16)}, stage=2)
    assert False
except ValueError as e:
    assert "!=" in str(e)
    print("  5) evidence shape mismatch → ValueError OK")

print("ALL OK")
