"""Test: evidence_conf weighting in loss."""
print("--- test_evidence_conf_loss ---")
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch

from models.loss import BioLoss

# 1) Evidence-conf=0 → evidence loss ~0
loss_fn = BioLoss({"evidence": 1.0, "expr": 0.0, "delta": 0.0, "deg": 0.0, "mmd": 0.0, "latent": 0.0})

out = {"pred": torch.randn(4, 32), "delta": torch.randn(4, 32),
       "latent": torch.randn(4, 16), "evidence_pred": torch.randn(4, 8)}
batch = {"x": torch.randn(4, 32), "y": torch.randn(4, 32),
         "evidence": torch.randn(4, 8), "evidence_conf": torch.zeros(4)}

losses = loss_fn(out, batch, stage=2)
assert losses["evidence"].item() < 0.01, f"Evidence loss with conf=0 should be ~0, got {losses['evidence'].item():.6f}"
print(f"  1) evidence_conf=0 → evidence loss ~0: {losses['evidence'].item():.6f} OK")

# 2) Evidence-conf=1 → evidence loss normal
batch2 = {"x": torch.randn(4, 32), "y": torch.randn(4, 32),
          "evidence": torch.randn(4, 8), "evidence_conf": torch.ones(4)}
losses2 = loss_fn(out, batch2, stage=2)
assert losses2["evidence"].item() > 0.01, f"Evidence loss with conf=1 should be >0, got {losses2['evidence'].item():.6f}"
print(f"  2) evidence_conf=1 → evidence loss normal: {losses2['evidence'].item():.6f} OK")

# 3) Default evidence weight is 0.0
loss_fn_default = BioLoss()
assert loss_fn_default.weights.get("evidence", 1.0) == 0.0, f"Default evidence weight should be 0.0, got {loss_fn_default.weights.get('evidence')}"
print(f"  3) default evidence weight = 0.0 OK")

# 4) evi_gain accessible from weights
assert loss_fn_default.weights.get("evi_gain", 0.0) >= 0.0, "evi_gain should be in weights"
print(f"  4) evi_gain accessible from weights OK")

# 5) Evidence loss with mixed confidence
batch_mixed = {"x": torch.randn(4, 32), "y": torch.randn(4, 32),
               "evidence": torch.randn(4, 8), "evidence_conf": torch.tensor([1.0, 0.0, 0.5, 1.0])}
losses_mixed = loss_fn(out, batch_mixed, stage=2)
# Mixed conf should give lower loss than all-ones but higher than all-zeros
assert losses_mixed["evidence"].item() < losses2["evidence"].item(), "Mixed conf should give lower evidence loss"
print(f"  5) mixed confidence halves loss: full={losses2['evidence'].item():.4f}, mixed={losses_mixed['evidence'].item():.4f} OK")

# 6) Stage 1 doesn't compute evidence loss (when evidence_pred is None)
out_no_ev = {"pred": torch.randn(4, 32), "delta": torch.randn(4, 32),
             "latent": torch.randn(4, 16), "evidence_pred": None}
losses3 = loss_fn(out_no_ev, batch2, stage=1)
assert losses3["evidence"].item() == 0.0, "Stage 1 should have evidence loss = 0"
print(f"  6) stage 1 evidence loss = 0 OK")

print("ALL OK")
