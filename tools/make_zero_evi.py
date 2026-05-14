"""Create zero-evidence h5ad: copies input but sets evidence to zeros."""
import argparse, sys
from pathlib import Path
import numpy as np, scanpy as sc

def main():
    ap = argparse.ArgumentParser(description="Make zero-evidence h5ad")
    ap.add_argument("--h5ad", required=True, help="Input h5ad with or without evidence")
    ap.add_argument("--out", required=True, help="Output h5ad with zero evidence")
    ap.add_argument("--evidence_dim", type=int, default=128, help="Evidence dimension if not present in input")
    args = ap.parse_args()

    a = sc.read_h5ad(args.h5ad)

    # Determine evidence dim
    if "evidence" in a.obsm:
        ev_dim = a.obsm["evidence"].shape[1]
        print(f"Using existing evidence_dim={ev_dim}")
    else:
        ev_dim = args.evidence_dim
        print(f"No evidence found, creating zero evidence with dim={ev_dim}")

    a.obsm["evidence"] = np.zeros((a.n_obs, ev_dim), dtype=np.float32)
    a.obs["evidence_conf"] = np.zeros(a.n_obs, dtype=np.float32)
    a.obs["evidence_conf_raw"] = np.zeros(a.n_obs, dtype=np.float32)
    a.obs["evidence_source"] = np.array(["zero"] * a.n_obs, dtype=object)
    a.obs["evidence_qc_flags"] = np.array(["zero_evidence"] * a.n_obs, dtype=object)

    # Update audit metadata
    audit = []
    unique_perts = a.obs["perturbation"].unique()
    for pert in unique_perts:
        audit.append({
            "pert": str(pert),
            "source": "zero",
            "evidence_encoder_mode": "zero",
            "confidence_raw": 0.0,
            "confidence_adjusted": 0.0,
            "used_evidence": False,
            "filtered_reason": "zero_evidence",
            "evidence_norm": 0.0,
        })
    a.uns["evidence_audit"] = {k: [r.get(k, "") for r in audit] for k in audit[0]} if audit else {}
    a.uns["evidence_schema"] = "zero"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(args.out)
    print(f"Zero-evidence h5ad saved: {args.out}")
    print(f"  cells={a.n_obs} genes={a.n_vars} evidence_dim={ev_dim}")

if __name__ == "__main__":
    main()
