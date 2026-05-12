"""Stage 0: Evidence prior construction pipeline.

Local KB → LLM fallback → JSON validate → text encode → write evidence.
"""

import os, json
import numpy as np
from pathlib import Path


def run_prior(h5ad_path, out_path, pert_key="perturbation", control_label="control",
              kb_path=None, use_llm=False, min_conf=0.5, evidence_dim=128,
              encoder="hash", model_name=None, audit_out=None):
    """Run Stage 0 evidence construction pipeline."""
    import scanpy as sc
    from .kb import load_kb
    from .evi import build_evidence, write_evidence

    print(f"\n{'='*50}")
    print(f"  BioReason Stage 0 — Evidence Prior Construction")
    print(f"{'='*50}")

    adata = sc.read_h5ad(h5ad_path)
    pert_list = adata.obs[pert_key].astype(str).tolist()
    pert_unique = sorted(set(pert_list))
    n_perts = len(pert_unique) - (1 if control_label in pert_unique else 0)
    print(f"  cells: {adata.n_obs} | pert labels: {n_perts} | evidence_dim: {evidence_dim}")

    kb = load_kb(kb_path)
    print(f"  kb entries: {len(kb)} | use_llm: {use_llm} | min_conf: {min_conf}")

    evidence, confidence, sources, audit = build_evidence(
        pert_list, control_label=control_label, kb=kb,
        use_llm=use_llm, min_conf=min_conf, evidence_dim=evidence_dim,
        encoder_mode=encoder, model_name=model_name, audit_out=audit_out,
    )

    write_evidence(adata, evidence, confidence, sources, audit)

    # Summary
    local_count = sum(1 for a in audit if a.get("source") == "local" and a.get("valid"))
    llm_count = sum(1 for a in audit if a.get("source") == "llm" and a.get("valid"))
    zero_count = sum(1 for a in audit if not a.get("valid", False))

    print(f"  local hits: {local_count} | llm calls: {llm_count} | zeroed: {zero_count}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"  output: {out_path}")

    if audit_out:
        Path(audit_out).parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.DataFrame(audit).to_csv(audit_out, index=False)
        print(f"  audit: {audit_out}")

    print(f"{'='*50}\n")
    return adata
