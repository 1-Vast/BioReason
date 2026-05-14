"""Test tools/evi.py: validation, text conversion, build/write evidence."""
print("--- test_prep_evi ---")
import sys
import tempfile
import json

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed")
    sys.exit(0)

from tools.evi import validate_prior, prior_to_text, zero_prior, build_evidence, write_evidence

prior = {
    "description": "TP53 knockout disrupts DNA damage response.",
    "confidence_score": 0.9,
    "source": "local",
    "pathway_impact": [{"pathway": "DNA damage response", "direction": "down", "confidence": 0.9}],
    "tf_activity": [{"tf": "TP53", "direction": "down"}],
    "marker_genes": [{"gene": "CDKN1A", "direction": "down"}],
}

valid, cleaned, reason = validate_prior(prior, min_conf=0.5)
assert valid and reason == "ok"
assert "Description:" in prior_to_text(cleaned)
print("  1) validate/text OK")

valid, cleaned, reason = validate_prior({"confidence_score": 0.1}, min_conf=0.5)
assert not valid and reason == "low_confidence"
assert cleaned["description"] == ""
print("  2) low confidence OK")

zp = zero_prior("X", "no_prior")
assert zp["source"] == "zero" and zp["reason"] == "no_prior"
print("  3) zero_prior OK")

X = np.random.randn(4, 6).astype(np.float32)
obs = {"perturbation": ["control", "TP53_KO", "LOW_KO", "MISS_KO"]}
adata = anndata.AnnData(X=X, obs=obs)
kb = {"TP53_KO": prior, "LOW_KO": {**prior, "confidence_score": 0.1}}

with tempfile.TemporaryDirectory() as tmp:
    kb_path = f"{tmp}/prior.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f)
    evidence, conf, source, audit = build_evidence(
        adata, "perturbation", "control", kb_path, False, 0.5, 16, "hash"
    )

assert evidence.shape == (4, 16)
assert conf.shape == (4,)
assert len(source) == 4
assert len(audit) == 4
assert np.linalg.norm(evidence[1]) > 0
assert np.allclose(evidence[2], 0)
assert np.allclose(evidence[3], 0)
print("  4) build_evidence OK")

write_evidence(adata, evidence, conf, source, audit)
assert "evidence" in adata.obsm
assert "evidence_conf" in adata.obs
assert "evidence_source" in adata.obs
assert "evidence_audit" in adata.uns
print("  5) write_evidence OK")

print("ALL OK")
