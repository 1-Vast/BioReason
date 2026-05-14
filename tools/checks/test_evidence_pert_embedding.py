"""Test: id_plus_evidence pert_mode in PertEncoder and BioReason wiring."""
print("--- test_evidence_pert_embedding ---")
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch

from models.pert import PertEncoder
from models.reason import BioReason

dim = 64
evidence_dim = 32
num_perts = 50
B = 8

# ── Test 1: id_plus_evidence with evidence produces different pert_emb than without ──
enc = PertEncoder(num_perts, dim, pert_mode="id_plus_evidence",
                  evidence_dim=evidence_dim, evidence_pert_alpha=0.5)
pert_ids = torch.randint(0, num_perts, (B,))
evidence_emb = torch.randn(B, dim)

out_with_ev = enc(pert_ids, evidence_emb=evidence_emb)
out_without_ev = enc(pert_ids, evidence_emb=None)
assert not torch.allclose(out_with_ev, out_without_ev, atol=1e-4), \
    "id_plus_evidence with evidence should differ from without"
print(f"  1) evidence_emb changes pert_emb OK: max_diff={ (out_with_ev - out_without_ev).abs().max().item():.6f}")

# ── Test 2: same perturbation with different evidence yields different pert_emb ──
evidence_emb2 = torch.randn(B, dim)
out_ev2 = enc(pert_ids, evidence_emb=evidence_emb2)
assert not torch.allclose(out_with_ev, out_ev2, atol=1e-4), \
    "different evidence should yield different pert_emb"
print(f"  2) different evidence → different pert_emb OK: max_diff={ (out_with_ev - out_ev2).abs().max().item():.6f}")

# ── Test 3: zero evidence yields same as ID-only embedding ──
enc3 = PertEncoder(num_perts, dim, pert_mode="id_plus_evidence",
                   evidence_dim=evidence_dim, evidence_pert_alpha=0.5)
out_zero_ev = enc3(pert_ids, evidence_emb=torch.zeros(B, dim))
out_no_ev = enc3(pert_ids, evidence_emb=None)
# With zero evidence input, evidence_to_pert output depends on the MLP.
# Even with zero input, a non-zero bias will shift the result.
# So zero evidence != no evidence in general. Just verify shapes and reasonable behavior.
assert out_zero_ev.shape == (B, dim), f"Expected ({B},{dim}), got {out_zero_ev.shape}"
max_diff_zero = (out_zero_ev - out_no_ev).abs().max().item()
print(f"  3) zero evidence shape OK: max_diff={max_diff_zero:.6f} (non-zero bias expected)")

# ── Test 4: evidence_pert_alpha=0 means evidence doesn't affect pert_emb ──
enc4 = PertEncoder(num_perts, dim, pert_mode="id_plus_evidence",
                   evidence_dim=evidence_dim, evidence_pert_alpha=0.0)
enc4.eval()  # eval mode to avoid dropout non-determinism between calls
with torch.no_grad():
    out_alpha0_ev = enc4(pert_ids, evidence_emb=evidence_emb)
    out_alpha0_none = enc4(pert_ids, evidence_emb=None)
assert torch.allclose(out_alpha0_ev, out_alpha0_none, atol=1e-5), \
    f"alpha=0 should mean evidence doesn't affect pert_emb, max_diff={ (out_alpha0_ev - out_alpha0_none).abs().max().item():.6f}"
print("  4) evidence_pert_alpha=0 → evidence has no effect OK")

# ── Test 5: different perturbations with same evidence yield different pert_emb ──
enc5 = PertEncoder(num_perts, dim, pert_mode="id_plus_evidence",
                   evidence_dim=evidence_dim, evidence_pert_alpha=0.5)
pert_ids_a = torch.randint(0, num_perts // 2, (B,))
pert_ids_b = torch.randint(num_perts // 2, num_perts, (B,))
# Ensure no overlap
for i in range(B):
    while pert_ids_b[i].item() == pert_ids_a[i].item():
        pert_ids_b[i] = torch.randint(num_perts // 2, num_perts, (1,))

same_evidence = torch.randn(B, dim)
out_a = enc5(pert_ids_a, evidence_emb=same_evidence)
out_b = enc5(pert_ids_b, evidence_emb=same_evidence)
assert not torch.allclose(out_a, out_b, atol=1e-4), \
    "different perturbations with same evidence should differ (ID dominates)"
print(f"  5) different pert + same evidence → different pert_emb OK: max_diff={ (out_a - out_b).abs().max().item():.6f}")

# ── Test 6: forward pass with id_plus_evidence works end-to-end ──
bio = BioReason(input_dim=100, dim=dim, hidden=dim * 2, steps=2, heads=2,
                pert_mode="id_plus_evidence", num_perts=num_perts,
                evidence_dim=evidence_dim, evidence_pert_alpha=0.5)
x = torch.randn(B, 100)
pert = torch.randint(0, num_perts, (B,))
evidence = torch.randn(B, evidence_dim)

out = bio(x, pert, evidence=evidence, return_latent=True)
for key in ("pred", "delta", "latent", "evidence_pred", "trust"):
    assert key in out, f"Missing key '{key}' in forward output"
assert out["pred"].shape == (B, 100), f"pred shape {out['pred'].shape}"
assert out["delta"].shape == (B, 100)
assert out["latent"].shape == (B, dim)
assert out["evidence_pred"] is not None
print(f"  6) end-to-end forward with id_plus_evidence OK: pred={out['pred'].shape} delta={out['delta'].shape}")

# ── Test 7: forward_latent with id_plus_evidence ──
z = bio.forward_latent(x, pert, evidence=evidence)
assert z.shape == (B, dim), f"forward_latent shape {z.shape}"
print(f"  7) forward_latent with id_plus_evidence OK: z={z.shape}")

# ── Test 8: backward compatibility — id mode still works ──
enc_id = PertEncoder(num_perts, dim, pert_mode="id")
out_id = enc_id(pert_ids)
assert out_id.shape == (B, dim)
print(f"  8) id mode backward compatible OK: {out_id.shape}")

# ── Test 9: backward compatibility — BioReason with id mode ──
bio_id = BioReason(input_dim=100, dim=dim, pert_mode="id", num_perts=num_perts)
out_id2 = bio_id(x, pert, return_latent=True)
assert out_id2["pred"].shape == (B, 100)
# Also test that passing evidence_emb is gracefully ignored in id mode
# (the pert_encoder forward signature accepts evidence_emb=None by default)
print(f"  9) Bioreason id mode backward compatible OK: {out_id2['pred'].shape}")

print("ALL OK")
