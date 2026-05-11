"""Test: verify BioReason forward pass with toy data."""
print("Testing BioReason forward pass...")

import torch
from models.reason import BioReason

B, G = 4, 100
dim = 32
hidden = 64

x = torch.randn(B, G)
pert = torch.randint(0, 10, (B,))

model = BioReason(
    input_dim=G, dim=dim, hidden=hidden, steps=2,
    heads=2, dropout=0.0, pert_mode="id", num_perts=10,
)

# Test forward WITHOUT evidence (inference mode)
out = model(x, pert, cov=None, evidence=None, return_latent=True)
assert out["pred"].shape == (B, G), f"pred shape: {out['pred'].shape}"
assert out["delta"].shape == (B, G), f"delta shape: {out['delta'].shape}"
assert out["latent"].shape == (B, dim), f"latent shape: {out['latent'].shape}"
assert out["evidence_pred"] is None, "evidence_pred should be None without evidence"
print(f"  Forward (no evidence): pred={out['pred'].shape}, delta={out['delta'].shape}, latent={out['latent'].shape}")

# Test forward WITH evidence (training mode)
evidence = torch.randn(B, dim)
out2 = model(x, pert, cov=None, evidence=evidence, return_latent=True)
assert out2["pred"].shape == (B, G)
assert out2["evidence_pred"] is not None, "evidence_pred should not be None with evidence"
print(f"  Forward (with evidence): evidence_pred present")

# Test detach_latent
out3 = model(x, pert, detach_latent=True)
assert out3["pred"].shape == (B, G)
print(f"  Forward (detach_latent): OK")

# Test encode + predict
z = model.encode(x, pert)
assert z.shape == (B, dim)
pred2 = model.predict(x, z)
assert pred2.shape == (B, G)
print(f"  Encode + predict: {pred2.shape}")

print("TEST PASSED")
