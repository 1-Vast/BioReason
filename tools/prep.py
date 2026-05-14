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
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--cell_line", default=None)
    parser.add_argument("--perturbation_type", default=None)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--organism", default=None)
    parser.add_argument("--evidence_schema", default="bio_v2")

    parser.add_argument("--use_llm", action="store_true", default=False)
    parser.add_argument("--min_conf", type=float, default=0.35)
    parser.add_argument("--evidence_dim", type=int, default=128)
    parser.add_argument("--encoder", default="hash",
                        choices=["hash", "sentence", "structured", "hybrid"])
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--audit", default="output/prior_audit.csv")
    parser.add_argument("--max_llm_calls", type=int, default=50)
    parser.add_argument("--max_llm_tokens", type=int, default=30000)
    parser.add_argument("--llm_max_tokens", type=int, default=512)
    parser.add_argument("--llm_cache", default="output/llm_cache.json")
    parser.add_argument("--llm_sleep", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--gene_vocab", default=None, help="Path to gene vocab file (one per line)")
    parser.add_argument("--pathway_vocab", default=None, help="Path to pathway vocab file (one per line)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import scanpy as sc
    from tools.evi import build_evidence, write_evidence

    use_llm = args.use_llm

    # load optional vocab files
    gene_vocab = None
    if args.gene_vocab:
        try:
            with open(args.gene_vocab, "r", encoding="utf-8") as f:
                gene_vocab = set(line.strip() for line in f if line.strip())
        except Exception:
            gene_vocab = None

    pathway_vocab = None
    if args.pathway_vocab:
        try:
            with open(args.pathway_vocab, "r", encoding="utf-8") as f:
                pathway_vocab = set(line.strip() for line in f if line.strip())
        except Exception:
            pathway_vocab = None

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
        max_llm_calls=args.max_llm_calls,
        max_llm_tokens=args.max_llm_tokens,
        llm_max_tokens=args.llm_max_tokens,
        llm_cache=args.llm_cache,
        llm_sleep=args.llm_sleep,
        dry_run=args.dry_run,
        evidence_schema=args.evidence_schema,
        gene_vocab=gene_vocab,
        pathway_vocab=pathway_vocab,
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
    cache_hits = int(non_control["llm_cache_hit"].sum()) if "llm_cache_hit" in non_control else 0
    llm_calls = int(non_control["llm_called"].sum()) if "llm_called" in non_control else 0
    budget_used = int(non_control["llm_budget_used"].max()) if "llm_budget_used" in non_control and not non_control.empty else 0
    dry_run_planned = int((non_control["llm_skipped_reason"] == "dry_run").sum()) if "llm_skipped_reason" in non_control else 0
    zero_evidence = int((~non_control["used_evidence"].astype(bool)).sum()) if "used_evidence" in non_control else int(non_control["zero_evidence"].sum()) if "zero_evidence" in non_control else 0
    perts = adata.obs[args.pert_key].astype(str).nunique()
    encoder_mode = args.encoder

    print("BioReason Stage 0 preprocessing")
    print(f"cells: {adata.n_obs}")
    print(f"genes: {adata.n_vars}")
    print(f"perturbations: {perts}")
    print(f"local hits: {local_hits}")
    print(f"cache hits: {cache_hits}")
    print(f"llm calls: {llm_calls} / {args.max_llm_calls}")
    print(f"llm token budget used: {budget_used} / {args.max_llm_tokens}")
    print(f"zero evidence: {zero_evidence}")
    print(f"encoder: {encoder_mode}")
    print(f"evidence_schema: {args.evidence_schema}")
    print(f"dry run: {str(args.dry_run).lower()}")
    if args.dry_run:
        print(f"dry run planned llm calls: {dry_run_planned}")
    print(f"evidence_dim: {args.evidence_dim}")
    print(f"output: {args.out}")
    print(f"audit: {args.audit if args.audit else ''}")


if __name__ == "__main__":
    main()
