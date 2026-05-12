"""Test: BioReason with evidence_dim, check out["trust"] shape, values, None case."""
print("--- test_trust_gate ---")
import torch
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from models.reason import BioReason, EvidenceGate, Reasoner

B, G, D, ED = 4, 100, 32, 16
x = torch.randn(B, G)
pert = torch.randint(0, 10, (B,))
ev = torch.randn(B, ED)

# ── 1) EvidenceGate confidence with return_score ──
gate = EvidenceGate(dim=D, dropout=0.0, mode="film", use_conf=True)
z_in = torch.randn(B, D)
# Evidence must match gate dimension (in real usage it goes through evidence_encoder first)
ev_D = torch.randn(B, D)
z_out, trust = gate(z_in, ev_D, return_score=True)
assert z_out.shape == (B, D), f"z_out shape {z_out.shape}"
assert trust.shape == (B, 1), f"trust shape {trust.shape} != ({B}, 1)"
# trust from sigmoid should be in [0, 1]
assert (trust >= 0).all() and (trust <= 1).all(), f"trust out of range: {trust.min()}-{trust.max()}"
print(f"  1) EvidenceGate return_score OK: trust [{trust.min().item():.3f}, {trust.max().item():.3f}]")

# 2) EvidenceGate without return_score → just z
z_out2 = gate(z_in, ev_D)
assert z_out2.shape == (B, D)
print("  2) EvidenceGate no return_score OK")

# 3) EvidenceGate with evidence=None and return_score
z_out3, trust3 = gate(z_in, None, return_score=True)
assert z_out3.shape == (B, D)
assert trust3 is None, f"trust should be None when evidence is None, got {trust3}"
print("  3) EvidenceGate evidence=None → trust=None OK")

# ── 4) BioReason with evidence → trust present ──
m = BioReason(input_dim=G, dim=D, hidden=64, steps=2, heads=2, dropout=0.0,
               pert_mode="id", num_perts=10, evidence_dim=ED,
               use_evidence_conf=True)
out = m(x, pert, evidence=ev, return_latent=True)
assert "trust" in out, "trust not in output"
assert out["pred"].shape == (B, G)
assert out["trust"].shape == (B, 1), f"trust shape {out['trust'].shape} != ({B}, 1)"
assert (out["trust"] >= 0).all() and (out["trust"] <= 1).all(), f"trust out of range"
print(f"  4) BioReason trust OK: shape {tuple(out['trust'].shape)}, range [{out['trust'].min().item():.3f}, {out['trust'].max().item():.3f}]")

# ── 5) BioReason without evidence → trust is None ──
out5 = m(x, pert, evidence=None)
assert "trust" in out5, "trust key should exist"
# trust is None when evidence is None (trust_all is empty → None)
assert out5["trust"] is None, f"trust should be None, got {out5['trust']}"
print("  5) BioReason evidence=None → trust=None OK")

# ── 6) BioReason with evidence_dim but use_evidence_conf=False still returns trust ──
m6 = BioReason(input_dim=G, dim=D, hidden=64, steps=2, heads=2, dropout=0.0,
                pert_mode="id", num_perts=10, evidence_dim=ED,
                use_evidence_conf=False)
out6 = m6(x, pert, evidence=ev)
assert out6["trust"] is not None  # Reasoner always returns trust if evidence provided
assert out6["trust"].shape == (B, 1), f"trust shape {out6['trust'].shape}"
print(f"  6) BioReason no_conf trust OK: shape {tuple(out6['trust'].shape)}")

# ── 7) Reasoner return_trust ──
r = Reasoner(dim=D, steps=2, hidden=64, heads=2, dropout=0.0, use_evidence_conf=True)
z_r, trust_r = r(cell_emb=torch.randn(B, D), pert_emb=torch.randn(B, D),
                  cov_emb=None, evidence=torch.randn(B, D), return_trust=True)
assert z_r.shape == (B, D)
assert trust_r.shape == (B, 1)
print(f"  7) Reasoner return_trust OK: trust {tuple(trust_r.shape)}")

# ── 8) Reasoner return_trust=False → just z ──
z_r8 = r(cell_emb=torch.randn(B, D), pert_emb=torch.randn(B, D),
          cov_emb=None, evidence=torch.randn(B, D), return_trust=False)
assert z_r8.shape == (B, D)
print("  8) Reasoner return_trust=False OK")

print("ALL OK")
