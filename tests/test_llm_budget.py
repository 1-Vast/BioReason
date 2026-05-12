"""Test Stage 0 LLM call/token budget and dry-run safeguards."""
print("--- test_llm_budget ---")
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed")
    sys.exit(0)

from tools.evi import build_evidence


def prior(name, source="local", conf=0.95):
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
kb = {"A_KO": prior("A_KO")}

mock_prior = prior("B_KO", source="llm")
with patch("tools.kb.load_kb", return_value=kb), \
     patch("utils.llm.has_llm_key", return_value=True), \
     patch("utils.llm.build_perturbation_prior", return_value=mock_prior) as mock_llm:
    evidence, conf, source, audit = build_evidence(
        adata, "perturbation", "control", None, True, 0.5, 8, "hash",
        max_llm_calls=1, max_llm_tokens=512, llm_max_tokens=512,
        llm_cache=None,
    )

assert mock_llm.call_count == 1
assert evidence.shape == (4, 8)
assert int(audit["llm_called"].sum()) == 1
assert audit.loc[audit["pert"] == "A_KO", "llm_skipped_reason"].item() == "local_hit"
assert audit.loc[audit["pert"] == "C_KO", "reason"].item() == "llm_call_limit"
assert bool(audit.loc[audit["pert"] == "C_KO", "zero_evidence"].item()) is True
print("  1) max_llm_calls hard limit OK")

with patch("tools.kb.load_kb", return_value=kb), \
     patch("utils.llm.build_perturbation_prior") as mock_dry:
    evidence, conf, source, audit = build_evidence(
        adata, "perturbation", "control", None, True, 0.5, 8, "hash",
        max_llm_calls=50, max_llm_tokens=30000, llm_max_tokens=512,
        llm_cache=None, dry_run=True,
    )

assert mock_dry.call_count == 0
miss = audit[audit["pert"].isin(["B_KO", "C_KO"])]
assert set(miss["reason"]) == {"dry_run"}
assert int(audit["llm_called"].sum()) == 0
assert evidence.shape == (4, 8)
print("  2) dry-run avoids API calls OK")

print("ALL OK")
