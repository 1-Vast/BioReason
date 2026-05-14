"""Test latent-only BP freezes main prediction path and restores it."""
print("--- test_latent_only_bp ---")
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from models.reason import BioReason

B, G, D, ED = 4, 20, 16, 8
model = BioReason(input_dim=G, dim=D, hidden=32, steps=1, heads=2, dropout=0.0,
                  pert_mode="id", num_perts=5, evidence_dim=ED,
                  cov_dims={"cell_type": 2})

model.freeze_main_path_for_latent()

assert not any(p.requires_grad for p in model.cell_encoder.parameters())
assert not any(p.requires_grad for p in model.decoder.parameters())
assert any(p.requires_grad for p in model.reasoner.parameters())
assert any(p.requires_grad for p in model.pert_encoder.parameters())
assert any(p.requires_grad for p in model.cov_encoder.parameters())
assert model.evidence_encoder is not None and any(p.requires_grad for p in model.evidence_encoder.parameters())
assert model.evidence_head is not None and any(p.requires_grad for p in model.evidence_head.parameters())
print("  1) freeze_main_path_for_latent requires_grad OK")

x = torch.randn(B, G)
pert = torch.randint(0, 5, (B,))
evidence = torch.randn(B, ED)
target = torch.randn(B, D)
out = model(x, pert, evidence=evidence, return_latent=True)
loss = ((out["latent"] - target) ** 2).mean()
loss.backward()

assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.reasoner.parameters())
assert not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
assert not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cell_encoder.parameters())
print("  2) latent-only backward gradients OK")

model.unfreeze_all()
assert all(p.requires_grad for p in model.parameters())
print("  3) unfreeze_all restores trainability OK")

print("ALL OK")
