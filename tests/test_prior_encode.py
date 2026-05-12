"""Test: prior_to_text, TextEncoder hash mode, deterministic output."""
print("--- test_prior_encode ---")
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from utils.evi import prior_to_text, zero_prior, validate_prior
from utils.text import TextEncoder

# ── Test prior_to_text ──
prior = {
    "description": "TP53 knockout disrupts DNA damage response.",
    "confidence_score": 0.95,
    "source": "local",
    "pathway_impact": [
        {"pathway": "DNA damage response", "direction": "down", "confidence": 0.95},
    ],
    "tf_activity": [{"tf": "TP53", "direction": "down"}],
    "marker_genes": [{"gene": "CDKN1A", "direction": "down"}],
}
text = prior_to_text(prior)
assert "Description:" in text
assert "Pathways:" in text
assert "TF activity:" in text
assert "Markers:" in text
print("  1) prior_to_text OK:", text[:80])

# ── Test TextEncoder hash mode ──
enc = TextEncoder(dim=128, mode="hash", seed=42)

# 2) Encode single text
v1 = enc.encode([text])
assert v1.shape == (1, 128), f"Expected (1,128), got {v1.shape}"
print(f"  2) encode shape OK: {v1.shape}")

# 3) Encode multiple texts
texts = [text, "MYC overexpression activates cell cycle.", ""]
v3 = enc.encode(texts)
assert v3.shape == (3, 128)
print(f"  3) batch encode shape OK: {v3.shape}")

# 4) Deterministic output (same text, same encoder → same result)
v4a = enc.encode(["hello world"])
v4b = enc.encode(["hello world"])
assert np.allclose(v4a, v4b), "Hash should be deterministic"
print("  4) deterministic OK")

# 5) L2 normalized
norms = np.linalg.norm(v1, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), f"Norms not 1: {norms}"
print("  5) L2 normalized OK")

# 6) Different texts produce different vectors
v_a = enc.encode(["TP53 knockout"])
v_b = enc.encode(["MYC overexpression"])
assert not np.allclose(v_a, v_b), "Different texts should differ"
print("  6) distinct texts OK")

# ── Test zero_prior ──
zp = zero_prior("test_pert", "test_reason")
assert zp["confidence_score"] == 0.0
assert zp["source"] == "zero"
assert zp["reason"] == "test_reason"
print("  7) zero_prior OK")

# ── Test validate_prior ──
valid, reason, cleaned = validate_prior(prior, min_conf=0.5)
assert valid, f"Should be valid, got {reason}"
assert cleaned["confidence_score"] == 0.95
print("  8) validate_prior valid OK")

# 9) validate_prior with low confidence
low_prior = {**prior, "confidence_score": 0.3}
valid, reason, cleaned = validate_prior(low_prior, min_conf=0.5)
assert not valid, "Low confidence should be invalid"
assert "low_confidence" in reason
print(f"  9) validate_prior low conf OK: {reason}")

# 10) validate_prior with missing keys
partial_prior = {"description": "test"}
valid, reason, cleaned = validate_prior(partial_prior, min_conf=0.5)
# Fills in missing keys, but confidence defaults to 0.0 → invalid
assert "low_confidence" in reason or not valid
print(f"  10) validate_prior partial OK: {reason}")

print("ALL OK")
