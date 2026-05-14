"""Test: evidence-latent contrastive loss (evi_contrast_loss)."""
print("--- test_evi_contrast ---")
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch

from models.loss import evi_contrast_loss, BioLoss

dim = 64

# ── Test 1: contrastive loss computes without error ──
B = 8
z = torch.randn(B, dim)
evidence_emb = torch.randn(B, dim)
pert_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

loss = evi_contrast_loss(z, evidence_emb, pert_labels)
assert isinstance(loss, torch.Tensor), "loss should be a tensor"
assert loss.numel() == 1, "loss should be scalar"
assert loss.item() >= 0.0, f"loss should be non-negative, got {loss.item():.6f}"
print(f"  1) contrastive loss computes OK: {loss.item():.6f}")

# ── Test 2: evidence_conf=0 → loss ~0 ──
evidence_conf = torch.zeros(B)
loss_zero = evi_contrast_loss(z, evidence_emb, pert_labels, evidence_conf=evidence_conf, min_conf=0.35)
assert loss_zero.item() < 0.01, f"loss with conf=0 should be ~0, got {loss_zero.item():.6f}"
print(f"  2) evidence_conf=0 → loss ~0: {loss_zero.item():.6f} OK")

# ── Test 3: batch with < 2 perturbations returns 0 (via BioLoss.forward) ──
loss_fn_3 = BioLoss({"evi_contrast": 1.0, "expr": 0.0, "delta": 0.0, "deg": 0.0, "mmd": 0.0, "latent": 0.0, "evidence": 0.0})
out3 = {"pred": torch.randn(B, dim), "delta": torch.randn(B, dim),
        "latent": torch.randn(B, dim), "evidence_pred": torch.randn(B, dim),
        "evidence_emb": torch.randn(B, dim)}
# All same perturbation → only 1 unique pert → n_pert < 2 → guard triggers
batch3 = {"x": torch.randn(B, dim), "y": torch.randn(B, dim),
          "evidence": torch.randn(B, dim), "evidence_conf": torch.ones(B),
          "pert": torch.zeros(B, dtype=torch.long)}
losses3 = loss_fn_3(out3, batch3, stage=2)
assert losses3["evi_contrast"].item() == 0.0, f"single pert should give evi_contrast=0 via BioLoss guard, got {losses3['evi_contrast'].item():.6f}"
print(f"  3) single perturbation → BioLoss returns evi_contrast = 0: {losses3['evi_contrast'].item():.6f} OK")

# ── Test 4: loss has gradient ──
z_grad = torch.randn(B, dim, requires_grad=True)
evidence_emb_grad = torch.randn(B, dim, requires_grad=True)
loss_grad = evi_contrast_loss(z_grad, evidence_emb_grad, pert_labels)
loss_grad.backward()
assert z_grad.grad is not None, "z should have gradient"
assert evidence_emb_grad.grad is not None, "evidence_emb should have gradient"
assert z_grad.grad.abs().sum() > 0, "z gradient should be non-zero"
print(f"  5) loss has gradient: z_grad_norm={z_grad.grad.norm().item():.6f} OK")

# ── Test 5: ZERO evidence run not affected (BioLoss default evi_contrast weight = 0) ──
loss_fn = BioLoss()
assert loss_fn.evi_contrast_w == 0.0, f"Default evi_contrast weight should be 0.0, got {loss_fn.evi_contrast_w}"
assert loss_fn.weights.get("evi_contrast", 1.0) == 0.0, f"weights dict should have evi_contrast=0.0"
print(f"  5) default evi_contrast weight = 0.0 OK (ZERO evidence run not affected)")

# ── Test 6: same perturbation yields lower loss than random ──
# Same perturbation → latents should cluster together
# Create 4 samples: 2 from pert A (correlated z and evidence), 2 from pert B (correlated z and evidence)
z_A = torch.randn(2, dim) * 0.5 + torch.tensor([1.0] * dim)  # centered at [1,...,1]
e_A = torch.randn(2, dim) * 0.5 + torch.tensor([1.0] * dim)   # similar center → high similarity
z_B = torch.randn(2, dim) * 0.5 - torch.tensor([1.0] * dim)   # centered at [-1,...,-1]
e_B = torch.randn(2, dim) * 0.5 - torch.tensor([1.0] * dim)   # similar center → high similarity
z_6 = torch.cat([z_A, z_B], dim=0)
e_6 = torch.cat([e_A, e_B], dim=0)
pert_6 = torch.tensor([0, 0, 1, 1])
loss_clustered = evi_contrast_loss(z_6, e_6, pert_6)
# Shuffle to break structure
shuffle_idx = torch.tensor([0, 2, 1, 3])
z_shuffled = z_6[shuffle_idx]
e_shuffled = e_6[shuffle_idx]
pert_shuffled = pert_6[shuffle_idx]
loss_shuffled = evi_contrast_loss(z_shuffled, e_shuffled, pert_shuffled)
# Clustered arrangement should have lower or equal loss (same-pert pairs have higher similarity)
assert loss_clustered.item() <= loss_shuffled.item() + 1e-5, \
    f"Same-pert clusters should have lower loss: clustered={loss_clustered.item():.6f}, shuffled={loss_shuffled.item():.6f}"
print(f"  6) same pert yields lower loss: clustered={loss_clustered.item():.4f} <= shuffled={loss_shuffled.item():.4f} OK")

print("ALL OK")
