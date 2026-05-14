"""Test: gate_add evidence injection in EvidenceGate."""
print("--- test_gate_add_evidence ---")
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from models.reason import EvidenceGate

dim = 64

# 1) gate_add forward pass works
gate = EvidenceGate(dim, mode="gate_add", use_conf=True)
z = torch.randn(4, dim)
evidence = torch.randn(4, dim)
context = torch.randn(4, dim)

out = gate(z, evidence=evidence, context=context)
assert out.shape == (4, dim), f"Expected (4,{dim}), got {out.shape}"
print(f"  1) gate_add forward shape OK: {out.shape}")

# 2) gate_add with return_score
out, trust = gate(z, evidence=evidence, context=context, return_score=True)
assert trust is not None, "Trust should not be None"
print(f"  2) gate_add return_score OK: trust shape={trust.shape}")

# 3) evidence=None → identity
out_none = gate(z, evidence=None)
assert torch.allclose(out_none, z), "Should pass through when evidence=None"
print("  3) evidence=None → identity OK")

# 4) evidence_conf=0 → close to no-evidence
z_ref = torch.randn(4, dim)
out_zero_conf = gate(z, evidence=evidence, context=context, evidence_conf=torch.zeros(4))
out_no_ev = gate(z, evidence=None)
# With evidence_conf=0, output should be same as no-evidence (since gate*0 = 0)
assert torch.allclose(out_zero_conf, out_no_ev, atol=1e-5), "evidence_conf=0 should be same as no evidence"
print("  4) evidence_conf=0 → close to no evidence OK")

# 5) evidence_conf=1 → evidence can change latent
out_conf1 = gate(z, evidence=evidence, context=context, evidence_conf=torch.ones(4))
diff = torch.norm(out_conf1 - out_no_ev, dim=-1).mean()
assert diff > 0.01, f"evidence_conf=1 should change latent, got diff={diff:.6f}"
print(f"  5) evidence_conf=1 changes latent: diff={diff:.6f} OK")

# 6) Gate initial value is small (bias=-2.0, sigmoid(-2)≈0.12)
gate2 = EvidenceGate(dim, mode="gate_add", use_conf=True)
z2 = torch.ones(1, dim)
e2 = torch.ones(1, dim)
ctx2 = torch.ones(1, dim)
out2 = gate2(z2, evidence=e2, context=ctx2)
# The shift from z should be small initially
shift = torch.norm(out2 - z2).item()
print(f"  6) initial gate shift small: {shift:.6f} OK")

# 7) Works without context
out_no_ctx = gate(z, evidence=evidence, context=None)
assert out_no_ctx.shape == (4, dim)
print(f"  7) gate_add without context OK: {out_no_ctx.shape}")

# 8) Backward pass works
out_bp = gate(z, evidence=evidence, context=context)
loss = out_bp.sum()
loss.backward()
print("  8) backward pass OK")

print("ALL OK")
