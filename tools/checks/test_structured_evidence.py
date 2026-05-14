"""Test: structured evidence encoder produces correct dims and deterministic vectors."""
print("--- test_structured_evidence ---")
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np

from tools.text import TextEncoder
from tools.evi import prior_to_structured, qc_prior

# Sample prior
prior = {
    "description": "SPI1 knockdown disrupts myeloid differentiation in K562",
    "confidence_score": 0.85,
    "source": "local",
    "perturbation_gene": "SPI1",
    "perturbation_type": "CRISPR",
    "expected_direction": "loss_of_function",
    "pathway_impact": [
        {"pathway": "Myeloid differentiation", "direction": "down", "confidence": 0.85},
        {"pathway": "Immune response", "direction": "down", "confidence": 0.75},
    ],
    "tf_activity": [
        {"tf": "SPI1", "direction": "down"},
        {"tf": "CEBPA", "direction": "down"},
    ],
    "marker_genes": [
        {"gene": "CSF1R", "direction": "down"},
        {"gene": "CD14", "direction": "down"},
    ],
    "response_programs": [
        {"program": "myeloid", "direction": "down", "confidence": 0.80},
    ],
}

# 1) Structured evidence dim correct
enc = TextEncoder(dim=128, mode="structured")
s = prior_to_structured(prior)
vecs = enc.encode([s])
assert vecs.shape == (1, 128), f"Expected (1,128), got {vecs.shape}"
print(f"  1) structured dim correct: {vecs.shape} OK")

# 2) Deterministic: same prior → same vector
vecs2 = enc.encode([s])
assert np.allclose(vecs, vecs2), "Structured encoding should be deterministic"
print("  2) deterministic OK")

# 3) Different pathway/program produce different vector
prior2 = dict(prior)
prior2["pathway_impact"] = [{"pathway": "Apoptosis", "direction": "up", "confidence": 0.9}]
prior2["tf_activity"] = [{"tf": "TP53", "direction": "up"}]
prior2["response_programs"] = [{"program": "apoptosis", "direction": "up", "confidence": 0.9}]
s2 = prior_to_structured(prior2)
vecs_diff = enc.encode([s2])
cos_sim = np.dot(vecs[0], vecs_diff[0]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs_diff[0]))
assert cos_sim < 0.999, f"Different priors should produce different vectors, cos_sim={cos_sim:.6f}"
print(f"  3) different priors → different vectors (cos_sim={cos_sim:.6f}) OK")

# 4) Generic prior confidence reduced
gen_prior = {
    "description": "gene expression and cellular process changes",
    "confidence_score": 0.80,
    "source": "llm",
    "pathway_impact": [{"pathway": "gene expression", "direction": "unknown", "confidence": 0.5}],
    "tf_activity": [],
    "marker_genes": [],
}
passed, adjusted, reason = qc_prior(gen_prior)
assert passed, f"Generic prior should pass QC, got reason={reason}"
assert adjusted.get("generic_prior") is True, "Should be marked generic"
assert adjusted["confidence_adjusted"] <= 0.65, f"Confidence should be reduced, got {adjusted['confidence_adjusted']}"
print(f"  4) generic prior confidence reduced: {adjusted['confidence_score']} OK")

# 5) Confidence scaling: low-conf prior → lower evidence norm
low_conf_prior = dict(prior)
low_conf_prior["confidence_score"] = 0.4
passed, adjusted, reason = qc_prior(low_conf_prior)
assert passed, f"Should pass at 0.4, got {reason}"
s_low = prior_to_structured(adjusted)
vecs_low = enc.encode([s_low])
assert np.linalg.norm(vecs_low[0]) > 0, "Low-conf evidence should not be all zeros"
print(f"  5) confidence scaling works: passed={passed}, reason={reason} OK")

# 6) Very low confidence (< 0.35) → fail QC
very_low = dict(prior)
very_low["confidence_score"] = 0.2
passed, adjusted, reason = qc_prior(very_low)
assert not passed, f"Should fail at 0.2, got {reason}"
assert reason == "low_confidence", f"Expected low_confidence, got {reason}"
print(f"  6) very low confidence fails QC: reason={reason} OK")

# 7) Empty prior → fail QC
empty_prior = {
    "description": "",
    "confidence_score": 0.8,
    "source": "unknown",
    "pathway_impact": [],
    "tf_activity": [],
    "marker_genes": [],
}
passed, adjusted, reason = qc_prior(empty_prior)
assert not passed, f"Should fail for empty, got {reason}"
print(f"  7) empty prior fails QC: reason={reason} OK")

# 8) Hybrid mode produces correct dim
enc_hybrid = TextEncoder(dim=128, mode="hybrid")
vecs_h = enc_hybrid.encode([s])
assert vecs_h.shape == (1, 128), f"Expected (1,128), got {vecs_h.shape}"
print(f"  8) hybrid dim correct: {vecs_h.shape} OK")

# 9) Hybrid produces different vector than structured
cos_sim_h = np.dot(vecs[0], vecs_h[0]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs_h[0]))
assert cos_sim_h < 0.999, f"Hybrid should differ from structured, cos_sim={cos_sim_h:.6f}"
print(f"  9) hybrid differs from structured (cos_sim={cos_sim_h:.6f}) OK")

# 10) All vectors L2 normalized
assert abs(np.linalg.norm(vecs[0]) - 1.0) < 0.01, "Structured should be L2 normalized"
assert abs(np.linalg.norm(vecs_h[0]) - 1.0) < 0.01, "Hybrid should be L2 normalized"
print("  10) L2 normalization OK")

print("ALL OK")
