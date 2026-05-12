"""Test tools/prep.py CLI on a toy AnnData file."""
print("--- test_prep_cli ---")
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed")
    sys.exit(0)

root = Path(__file__).parent.parent

help_result = subprocess.run(
    [sys.executable, str(root / "tools" / "prep.py"), "--help"],
    cwd=root, text=True, capture_output=True, check=True,
)
for flag in ("--max_llm_calls", "--max_llm_tokens", "--llm_max_tokens",
             "--llm_cache", "--llm_sleep", "--dry_run"):
    assert flag in help_result.stdout, f"{flag} missing from prep.py --help"
print("  1) prep help LLM safety flags OK")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    h5ad = tmp / "perturb.h5ad"
    out = tmp / "perturb_evi.h5ad"
    kb_path = tmp / "prior.json"
    audit = tmp / "audit.csv"

    adata = anndata.AnnData(
        X=np.random.randn(3, 5).astype(np.float32),
        obs={"perturbation": ["control", "TP53_KO", "MISS_KO"]},
    )
    adata.write_h5ad(h5ad)
    kb_path.write_text(json.dumps({
        "TP53_KO": {
            "description": "TP53 knockout disrupts DNA damage response.",
            "confidence_score": 0.95,
            "source": "local",
            "pathway_impact": [{"pathway": "DNA damage response", "direction": "down", "confidence": 0.95}],
            "tf_activity": [{"tf": "TP53", "direction": "down"}],
            "marker_genes": [{"gene": "CDKN1A", "direction": "down"}],
        }
    }), encoding="utf-8")

    cmd = [
        sys.executable, str(root / "tools" / "prep.py"),
        "--h5ad", str(h5ad),
        "--out", str(out),
        "--kb", str(kb_path),
        "--evidence_dim", "12",
        "--encoder", "hash",
        "--audit", str(audit),
    ]
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=True)
    assert "BioReason Stage 0 preprocessing" in result.stdout
    assert "llm calls: 0" in result.stdout
    assert out.exists()
    assert audit.exists()

    loaded = anndata.read_h5ad(out)
    assert loaded.obsm["evidence"].shape == (3, 12)
    assert "evidence_conf" in loaded.obs
    print("  2) prep CLI OK")

print("ALL OK")
