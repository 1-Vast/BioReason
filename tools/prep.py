"""BioReason Stage 0: offline evidence preprocessing."""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="BioReason Stage 0: evidence preprocessing")
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pert_key", default="perturbation")
    parser.add_argument("--control_label", default="control")
    parser.add_argument("--kb", default=None, help="Path to local KB JSON")
    parser.add_argument("--use_llm", action="store_true", default=False)
    parser.add_argument("--min_conf", type=float, default=0.5)
    parser.add_argument("--evidence_dim", type=int, default=128)
    parser.add_argument("--encoder", default="hash", choices=["hash", "sentence"])
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--audit", default="output/prior_audit.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import scanpy as sc
    from tools.evi import build_evidence, write_evidence
    from utils.llm import has_llm_key

    use_llm = args.use_llm
    if use_llm and not has_llm_key():
        warnings.warn("--use_llm set but no API key. Using KB and zero fallback only.")
        use_llm = False

    adata = sc.read_h5ad(args.h5ad)
    evidence, conf, source, audit = build_evidence(
        adata,
        pert_key=args.pert_key,
        control_label=args.control_label,
        kb_path=args.kb,
        use_llm=use_llm,
        min_conf=args.min_conf,
        evidence_dim=args.evidence_dim,
        encoder=args.encoder,
        model_name=args.model_name,
    )
    write_evidence(adata, evidence, conf, source, audit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)

    audit_path = Path(args.audit) if args.audit else None
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)

    non_control = audit[audit["pert"] != args.control_label] if not audit.empty else audit
    local_hits = int(non_control["local_hit"].sum()) if "local_hit" in non_control else 0
    llm_calls = int(non_control["llm_called"].sum()) if "llm_called" in non_control else 0
    zero_evidence = int(non_control["zero_evidence"].sum()) if "zero_evidence" in non_control else 0
    perts = adata.obs[args.pert_key].astype(str).nunique()

    print("BioReason Stage 0 preprocessing")
    print(f"cells: {adata.n_obs}")
    print(f"genes: {adata.n_vars}")
    print(f"perturbations: {perts}")
    print(f"local hits: {local_hits}")
    print(f"llm calls: {llm_calls}")
    print(f"zero evidence: {zero_evidence}")
    print(f"evidence_dim: {args.evidence_dim}")
    print(f"output: {args.out}")
    print(f"audit: {args.audit if args.audit else ''}")


if __name__ == "__main__":
    main()
