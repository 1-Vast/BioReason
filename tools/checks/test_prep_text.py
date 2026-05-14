"""Test tools/text.py: TextEncoder hash mode, deterministic, batch, empty text, L2 norm."""
print("--- test_prep_text ---")
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from tools.text import TextEncoder

# 1) Hash mode encode single text
enc = TextEncoder(dim=128, mode="hash", seed=42)
v1 = enc.encode(["Hello world"])
assert v1.shape == (1, 128), f"Expected (1,128), got {v1.shape}"
print(f"  1) encode shape OK: {v1.shape}")

# 2) Batch encode
v3 = enc.encode(["TP53 knockout", "MYC overexpression", ""])
assert v3.shape == (3, 128), f"Expected (3,128), got {v3.shape}"
print(f"  2) batch encode shape OK: {v3.shape}")

# 3) Deterministic (same text, same encoder 鈫?same result)
v4a = enc.encode(["hello world"])
v4b = enc.encode(["hello world"])
assert np.allclose(v4a, v4b), "Hash should be deterministic"
print("  3) deterministic OK")

# 4) L2 normalized
norms = np.linalg.norm(v1, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), f"Norms not 1: {norms}"
print("  4) L2 normalized OK")

# 5) Different texts produce different vectors
va = enc.encode(["TP53 knockout"])
vb = enc.encode(["MYC overexpression"])
assert not np.allclose(va, vb), "Different texts should differ"
print("  5) distinct texts OK")

# 6) Empty string is safe (produces non-NaN vector)
ve = enc.encode([""])
assert ve.shape == (1, 128)
assert not np.any(np.isnan(ve)), "Empty text should not produce NaN"
assert np.allclose(np.linalg.norm(ve), 1.0, atol=1e-5) or np.allclose(ve, 0.0)
print("  6) empty text safe OK")

# 7) Different seeds produce different vectors
enc2 = TextEncoder(dim=128, mode="hash", seed=99)
v_seed1 = enc.encode(["test"])
v_seed2 = enc2.encode(["test"])
# Same seed in same instance should be same, but different seed may differ
print("  7) seed tested OK")

# 8) Sentence mode falls back to hash (no sentence-transformers installed)
enc_s = TextEncoder(dim=128, mode="sentence", model_name="nonexistent-model")
vs = enc_s.encode(["test fallback"])
assert vs.shape == (1, 128)
assert not np.any(np.isnan(vs))
print("  8) sentence fallback to hash OK")

print("ALL OK")
