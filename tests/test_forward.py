"""Test: BioReason forward pass with evidence_dim support."""
print("--- test_forward ---")
import torch
sys_path = __import__("sys").path
sys_path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from models.reason import BioReason

B, G, D, ED = 4, 100, 32, 16
x = torch.randn(B, G)
pert = torch.randint(0, 10, (B,))
ev = torch.randn(B, ED)

# 1) No evidence_dim → no evidence allowed
m1 = BioReason(input_dim=G, dim=D, hidden=64, steps=2, heads=2, dropout=0.0,
                pert_mode="id", num_perts=10, evidence_dim=None)
out = m1(x, pert)
assert out["pred"].shape == (B, G)
assert out["evidence_pred"] is None
print("  1) no evidence_dim, no evidence → OK")

# 2) evidence_dim=ED → evidence works
m2 = BioReason(input_dim=G, dim=D, hidden=64, steps=2, heads=2, dropout=0.0,
                pert_mode="id", num_perts=10, evidence_dim=ED)
out2 = m2(x, pert, evidence=ev, return_latent=True)
assert out2["pred"].shape == (B, G)
assert out2["evidence_pred"].shape == ev.shape, f"{out2['evidence_pred'].shape} != {ev.shape}"
assert out2["latent"].shape == (B, D)
assert out2["target_latent"] is None
print(f"  2) evidence_dim={ED} → evidence_pred {tuple(out2['evidence_pred'].shape)} OK")

# 3) evidence=None → evidence_pred is None
out3 = m2(x, pert, evidence=None)
assert out3["evidence_pred"] is None
print("  3) evidence=None → evidence_pred is None OK")

# 4) evidence_dim not set but evidence given → error
try:
    m1(x, pert, evidence=ev)
    assert False, "should have raised"
except ValueError as e:
    assert "evidence_dim" in str(e)
    print("  4) evidence_dim missing → ValueError OK")

# 5) detach_latent
out5 = m2(x, pert, detach_latent=True, return_latent=True)
assert out5["pred"].shape == (B, G)
print("  5) detach_latent OK")

# 6) forward_latent
z = m2.forward_latent(x, pert, evidence=ev)
assert z.shape == (B, D)
print(f"  6) forward_latent {tuple(z.shape)} OK")

# 7) freeze_except_reasoner
m2.freeze_except_reasoner()
frozen = sum(p.numel() for p in m2.cell_encoder.parameters() if p.requires_grad)
assert frozen == 0
m2.unfreeze_all()
print("  7) freeze/unfreeze OK")

print("ALL OK")
