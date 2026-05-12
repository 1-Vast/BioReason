"""Test: Toy AnnData, local KB, build_evidence + write_evidence, check evidence in obsm.
Avoids write_h5ad via run_prior which fails due to h5py audit serialization.
"""
print("--- test_prior_pipeline ---")
import sys, os, json, tempfile
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import anndata
from utils.evi import build_evidence, write_evidence
from utils.kb import load_kb

# ── Create toy AnnData ──
n_cells = 20
n_genes = 50
X = np.random.rand(n_cells, n_genes).astype(np.float32)
adata = anndata.AnnData(X)
adata.obs["perturbation"] = ["TP53_KO"] * 10 + ["control"] * 10

# ── Create temp KB ──
kb = {
    "TP53_KO": {
        "description": "TP53 knockout disrupts DNA damage response.",
        "confidence_score": 0.95,
        "source": "local",
        "pathway_impact": [
            {"pathway": "DNA damage response", "direction": "down", "confidence": 0.95},
        ],
        "tf_activity": [{"tf": "TP53", "direction": "down"}],
        "marker_genes": [{"gene": "CDKN1A", "direction": "down"}],
    }
}

with tempfile.TemporaryDirectory() as tmpdir:
    kb_path = os.path.join(tmpdir, "kb.json")
    with open(kb_path, "w") as f:
        json.dump(kb, f)

    kb_loaded = load_kb(kb_path)

    pert_list = adata.obs["perturbation"].astype(str).tolist()

    # Use build_evidence directly
    evidence, confidence, sources, audit = build_evidence(
        pert_list, control_label="control", kb=kb_loaded,
        use_llm=False, min_conf=0.5, evidence_dim=128,
        encoder_mode="hash",
    )

    # Write evidence (without audit → avoid h5py serialization issue)
    adata_copy = adata.copy()
    write_evidence(adata_copy, evidence, confidence, sources, audit=None)

    # 1) Check evidence in obsm
    assert "evidence" in adata_copy.obsm, "Evidence not in obsm"
    ev = adata_copy.obsm["evidence"]
    assert ev.shape == (n_cells, 128), f"Evidence shape {ev.shape} != ({n_cells}, 128)"
    print(f"  1) evidence in obsm OK: {ev.shape}")

    # 2) Check evidence_conf in obs
    assert "evidence_conf" in adata_copy.obs, "evidence_conf not in obs"
    conf = adata_copy.obs["evidence_conf"].values
    assert len(conf) == n_cells
    assert abs(float(conf[0]) - 0.95) < 0.01, f"TP53_KO conf {conf[0]} != 0.95"
    assert abs(float(conf[10]) - 1.0) < 0.01, f"control conf {conf[10]} != 1.0"
    print(f"  2) evidence_conf OK: {list(conf[:2])}...")

    # 3) Check evidence_source in obs
    assert "evidence_source" in adata_copy.obs
    sources_obs = adata_copy.obs["evidence_source"].values
    assert str(sources_obs[0]) == "local", f"TP53_KO source {sources_obs[0]}"
    print(f"  3) evidence_source OK: first={sources_obs[0]}")

    # 4) Check evidence vectors are not all zero
    assert ev[:10].std() > 0, "TP53_KO evidence should be non-zero"
    assert np.allclose(ev[10:], 0.0, atol=1e-6), "control evidence should be zero"
    print("  4) evidence values OK")

    # 5) Audit list is correct
    assert len(audit) >= 1, "Audit should have entries"
    local_audit = [a for a in audit if a.get("source") == "local" and a.get("valid")]
    assert len(local_audit) >= 1
    print(f"  5) audit OK: {len(audit)} entries, {len(local_audit)} local valid")

    print("ALL OK")
