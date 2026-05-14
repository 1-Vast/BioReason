"""Test: adaptive evidence gate in EvidenceGate (gate_add mode)."""
print("--- test_adaptive_evidence_gate ---")
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from models.reason import EvidenceGate, BioReason

dim = 64
B = 8

# ── Test 1: adaptive gate computes without error ──
gate = EvidenceGate(dim, mode="gate_add", use_conf=True, adaptive=True, evidence_gate_init_bias=-1.5)
z = torch.randn(B, dim)
evidence = torch.randn(B, dim)
context = torch.randn(B, dim)
evidence_conf = torch.rand(B) * 0.8 + 0.2  # random in [0.2, 1.0]

out = gate(z, evidence=evidence, context=context, evidence_conf=evidence_conf)
assert out.shape == (B, dim), f"Expected ({B},{dim}), got {out.shape}"
print(f"  1) adaptive gate computes OK: {out.shape}")

# ── Test 2: evidence_conf=0 → gate close to 0 (output ≈ z) ──
z_ref = torch.randn(B, dim)
out_zero_conf = gate(z_ref, evidence=evidence, context=context, evidence_conf=torch.zeros(B))
out_no_ev = gate(z_ref, evidence=None)
# With evidence_conf=0, the gate should be zero, so output should match no-evidence
assert torch.allclose(out_zero_conf, out_no_ev, atol=1e-5), \
    f"evidence_conf=0 should match no-evidence, max_diff={(out_zero_conf - out_no_ev).abs().max().item():.6f}"
print("  2) evidence_conf=0 → gate close to 0 OK")

# ── Test 3: high-conf evidence → gate > 0 (latent changes) ──
# Force evidence to have a clear effect by using a fresh gate with fresh weights
gate3 = EvidenceGate(dim, mode="gate_add", use_conf=True, adaptive=True, evidence_gate_init_bias=-1.5)
out_conf1 = gate3(z, evidence=evidence, context=context, evidence_conf=torch.ones(B))
diff = torch.norm(out_conf1 - out_no_ev, dim=-1).mean()
assert diff > 0.01, f"high-conf evidence should change latent, got diff={diff:.6f}"
print(f"  3) high-conf evidence changes latent: diff={diff:.6f} OK")

# ── Test 4: gate has gradient ──
gate4 = EvidenceGate(dim, mode="gate_add", use_conf=True, adaptive=True, evidence_gate_init_bias=-1.5)
z_grad = torch.randn(B, dim, requires_grad=True)
evidence_grad = torch.randn(B, dim, requires_grad=True)
# Use evidence_conf that passes through
out_grad = gate4(z_grad, evidence=evidence_grad, context=context, evidence_conf=torch.ones(B))
loss = out_grad.sum()
loss.backward()
assert z_grad.grad is not None and z_grad.grad.abs().sum() > 0, "z should have gradient"
assert evidence_grad.grad is not None and evidence_grad.grad.abs().sum() > 0, "evidence should have gradient"
print(f"  4) gate has gradient: z_grad_norm={z_grad.grad.norm().item():.4f} OK")

# ── Test 5: no-evidence run not affected (gate with evidence=None returns z unchanged) ──
gate5 = EvidenceGate(dim, mode="gate_add", use_conf=True, adaptive=True, evidence_gate_init_bias=-1.5)
out_none = gate5(z, evidence=None)
assert torch.allclose(out_none, z, atol=1e-7), "evidence=None should return z unchanged"
print("  5) no-evidence run not affected OK")

# ── Test 6: gate stored in output (via BioReason) ──
bio = BioReason(input_dim=100, dim=dim, hidden=dim * 2, steps=2, heads=2,
                pert_mode="id", num_perts=20, evidence_dim=dim,
                evidence_mode="gate_add", use_evidence_conf=True,
                adaptive_evidence_gate=True, evidence_gate_init_bias=-1.5)
x = torch.randn(B, 100)
pert = torch.randint(0, 20, (B,))
ev = torch.randn(B, dim)
out_bio = bio(x, pert, evidence=ev, return_latent=True)
assert "evidence_gate" in out_bio, "evidence_gate key missing from output"
assert out_bio["evidence_gate"] is not None, "evidence_gate should not be None when evidence provided"
assert out_bio["evidence_gate"].shape == (B, dim), f"gate shape {out_bio['evidence_gate'].shape} != ({B},{dim})"
gate_vals = out_bio["evidence_gate"]
# Gate values should be in [0, 1] (product of two sigmoids and confidence)
assert (gate_vals >= 0).all() and (gate_vals <= 1.01).all(), \
    f"gate values out of range: [{gate_vals.min().item():.4f}, {gate_vals.max().item():.4f}]"
print(f"  6) gate stored in output: shape={out_bio['evidence_gate'].shape}, range=[{gate_vals.min().item():.4f}, {gate_vals.max().item():.4f}] OK")

# ── Test 7: evidence_gate is None when evidence not provided ──
out_noev = bio(x, pert, evidence=None, return_latent=True)
assert "evidence_gate" in out_noev, "evidence_gate key should exist"
assert out_noev["evidence_gate"] is None, f"evidence_gate should be None when no evidence, got {out_noev['evidence_gate']}"
print("  7) evidence_gate=None when no evidence OK")

print("ALL OK")
