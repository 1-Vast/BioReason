"""Test Stage 0 LLM JSON cache behavior."""
print("--- test_llm_cache ---")
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed")
    sys.exit(0)

from tools.evi import build_evidence


def prior(name, source, conf=0.95):
    return {
        "description": f"{name} prior",
        "confidence_score": conf,
        "source": source,
        "pathway_impact": [],
        "tf_activity": [],
        "marker_genes": [{"gene": name.replace("_KO", ""), "direction": "down"}],
    }


adata = anndata.AnnData(
    X=np.random.randn(4, 6).astype(np.float32),
    obs={"perturbation": ["control", "A_KO", "B_KO", "C_KO"]},
)
kb = {"A_KO": prior("A_KO", "local")}

with tempfile.TemporaryDirectory() as tmp:
    cache_path = Path(tmp) / "llm_cache.json"
    cache_path.write_text(json.dumps({
        "B KO": {
            "prior": prior("B_KO", "llm"),
            "model": "mock",
            "llm_max_tokens": 512,
        }
    }), encoding="utf-8")

    with patch("tools.kb.load_kb", return_value=kb), \
         patch("utils.llm.has_llm_key", return_value=True), \
         patch("utils.llm.build_perturbation_prior", return_value=prior("C_KO", "llm")) as mock_llm:
        evidence, conf, source, audit = build_evidence(
            adata, "perturbation", "control", None, True, 0.5, 8, "hash",
            max_llm_calls=10, max_llm_tokens=5000, llm_max_tokens=512,
            llm_cache=str(cache_path),
        )

    assert mock_llm.call_count == 1
    assert bool(audit.loc[audit["pert"] == "B_KO", "llm_cache_hit"].item()) is True
    assert bool(audit.loc[audit["pert"] == "C_KO", "llm_called"].item()) is True
    updated = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "C KO" in updated
    print("  1) cache hit + cache update OK")

    with patch("tools.kb.load_kb", return_value=kb), \
         patch("utils.llm.has_llm_key", return_value=True), \
         patch("utils.llm.build_perturbation_prior") as mock_llm2:
        evidence, conf, source, audit = build_evidence(
            adata, "perturbation", "control", None, True, 0.5, 8, "hash",
            max_llm_calls=10, max_llm_tokens=5000, llm_max_tokens=512,
            llm_cache=str(cache_path),
        )

    assert mock_llm2.call_count == 0
    assert bool(audit.loc[audit["pert"] == "C_KO", "llm_cache_hit"].item()) is True
    print("  2) second run uses cache OK")

print("ALL OK")
