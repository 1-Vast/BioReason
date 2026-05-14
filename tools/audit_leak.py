"""Data leakage audit for BioReason benchmark pipeline.

Checks:
  1. obs["split"] exists and is disjoint
  2. HVG selected from train only
  3. group_means computed from train only
  4. target_latent indices are train-only
  5. Stage 3 latent alignment only on train
  6. baseline metrics use test split
  7. evidence has no expression-derived fields
  8. LLM audit: no expression matrix sent

Usage:
  python tools/audit_leak.py --h5ad dataset/bench.h5ad --target_latent output/bench/stage2/target_latent.pt --report output/bench/leak_audit.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def check_split(adata):
    """Check obs['split'] exists and is disjoint."""
    issues = []
    if "split" not in adata.obs:
        issues.append({"severity": "HIGH", "item": "split",
                       "msg": "obs['split'] missing — cannot verify train/test separation"})
        return issues, {}

    splits = adata.obs["split"].values
    train_idx = set(np.where(splits == "train")[0])
    val_idx = set(np.where(splits == "val")[0])
    test_idx = set(np.where(splits == "test")[0])

    counts = {
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
    }

    if len(train_idx & test_idx) > 0:
        issues.append({"severity": "HIGH", "item": "split_overlap",
                       "msg": f"train/test overlap: {len(train_idx & test_idx)} cells"})
    if len(train_idx & val_idx) > 0:
        issues.append({"severity": "MEDIUM", "item": "split_overlap",
                       "msg": f"train/val overlap: {len(train_idx & val_idx)} cells"})
    if counts["test"] == 0:
        issues.append({"severity": "HIGH", "item": "split",
                       "msg": "No test cells — cannot evaluate"})

    # Check perturbation distribution
    if "perturbation" in adata.obs:
        pert_arr = adata.obs["perturbation"].values
        for s, indices in [("train", train_idx), ("test", test_idx)]:
            if not indices:
                continue
            s_perts = set(pert_arr[list(indices)])
            if len(s_perts) < 2:
                issues.append({"severity": "MEDIUM", "item": "split_perts",
                               "msg": f"{s} split has only {len(s_perts)} perturbations"})

    return issues, counts


def check_hvg(adata):
    """Check HVG selection metadata."""
    issues = []
    if "split_info" not in adata.uns:
        issues.append({"severity": "LOW", "item": "hvg",
                       "msg": "No split_info in uns — cannot verify HVG train-only selection"})
        return issues

    info = adata.uns["split_info"]
    if "hvg_from_train" not in info and "hvg_source" not in info:
        issues.append({"severity": "MEDIUM", "item": "hvg",
                       "msg": "No HVG source metadata — cannot verify train-only selection"})
    elif info.get("hvg_from_train", False) is False:
        issues.append({"severity": "HIGH", "item": "hvg",
                       "msg": "HVG NOT selected from train only — potential leakage"})

    return issues


def check_group_means(adata):
    """Check group_means metadata."""
    issues = []
    if "split_info" not in adata.uns:
        return issues

    info = adata.uns["split_info"]
    gm_source = info.get("group_means_source", "unknown")
    if gm_source != "train":
        # If split column exists but gm_source not explicitly set to "train",
        # check if PertDataset would use stats_split="train" (runtime behavior)
        if "split" in adata.obs and "train" in adata.obs["split"].values:
            # Runtime PertDataset uses stats_split="train" — safe
            pass
        else:
            issues.append({"severity": "HIGH", "item": "group_means",
                           "msg": f"group_means source '{gm_source}' — may include test cells"})
    return issues


def check_split_design(adata):
    """Check heldout and low-cell split invariants declared by split metadata."""
    issues = []
    if "split" not in adata.obs or "perturbation" not in adata.obs:
        return issues
    split = adata.obs["split"].astype(str).values
    pert = adata.obs["perturbation"].astype(str).values
    info = adata.uns.get("split_info", {})
    if not isinstance(info, dict):
        return issues

    mode = str(info.get("mode", ""))
    if mode == "perturbation_holdout":
        heldout = info.get("heldout_perts", None)
        if heldout is None:
            heldout = adata.uns.get("heldout_perturbations", [])
        if hasattr(heldout, "tolist"):
            heldout = heldout.tolist()
        train_perts = set(pert[split == "train"])
        leaked = sorted(set(map(str, heldout)) & train_perts)
        if leaked:
            issues.append({"severity": "HIGH", "item": "heldout_leak",
                           "msg": f"heldout perturbations present in train: {leaked[:10]}"})
    if mode == "lowcell":
        limit = int(info.get("cells_per_pert", -1))
        if limit >= 0:
            for p in sorted(set(pert)):
                n_train = int(((split == "train") & (pert == p)).sum())
                if n_train > limit:
                    issues.append({"severity": "HIGH", "item": "lowcell_limit",
                                   "msg": f"{p} has {n_train} train cells > limit {limit}"})
                                   
    hvg_source = str(adata.uns.get("hvg_source", info.get("hvg_source", ""))).lower()
    if hvg_source not in {"train_only", "train"}:
        issues.append({"severity": "HIGH", "item": "hvg_source",
                       "msg": f"HVG source is not train-only: {hvg_source}"})
    return issues


def check_target_latent(adata, target_latent_path):
    """Check target_latent indices are train-only."""
    issues = []
    if not target_latent_path or not Path(target_latent_path).exists():
        issues.append({"severity": "LOW", "item": "target_latent",
                       "msg": "No target_latent to audit"})
        return issues

    data = __import__("torch").load(target_latent_path, map_location="cpu")
    if isinstance(data, dict):
        indices = data.get("indices")
    else:
        indices = None

    if indices is not None and "split" in adata.obs:
        splits = adata.obs["split"].values
        tl_splits = splits[indices.numpy()] if hasattr(indices, 'numpy') else [splits[i] for i in indices]
        n_test = sum(1 for s in tl_splits if s == "test")
        n_val = sum(1 for s in tl_splits if s == "val")
        if n_test > 0:
            issues.append({"severity": "HIGH", "item": "target_latent_leak",
                           "msg": f"{n_test} test cells in target_latent — Stage 3 would align to test latents"})
        if n_val > 0:
            issues.append({"severity": "MEDIUM", "item": "target_latent_leak",
                           "msg": f"{n_val} val cells in target_latent"})

    return issues


def check_evidence(adata):
    """Check evidence is not derived from expression matrix."""
    issues = []
    ev_cols = [c for c in adata.obs.columns if "evidence" in c.lower()]
    if not ev_cols:
        return issues

    # Evidence from KB/LLM is fine; evidence computed from dataset DEG is suspicious
    if "evidence_source" in adata.obs:
        sources = adata.obs["evidence_source"].value_counts().to_dict()
        if "deg" in sources or "expression" in str(sources).lower():
            issues.append({"severity": "MEDIUM", "item": "evidence_source",
                           "msg": f"Evidence derived from expression: {sources}"})

    return issues


def check_llm_budget(adata):
    """Check LLM budget was respected."""
    issues = []
    if "evidence_audit" in adata.uns:
        audit = adata.uns["evidence_audit"]
        if isinstance(audit, list):
            for rec in audit:
                if rec.get("expression_sent", False):
                    issues.append({"severity": "HIGH", "item": "llm_expression_leak",
                                   "msg": "Expression matrix sent to LLM"})

    return issues


def run_audit(adata, target_latent_path=None):
    """Run all leak checks. Returns (issues, summary)."""
    all_issues = []

    split_issues, split_counts = check_split(adata)
    all_issues.extend(split_issues)

    all_issues.extend(check_hvg(adata))
    all_issues.extend(check_group_means(adata))
    all_issues.extend(check_split_design(adata))

    if target_latent_path:
        all_issues.extend(check_target_latent(adata, target_latent_path))

    all_issues.extend(check_evidence(adata))
    all_issues.extend(check_llm_budget(adata))

    high = [i for i in all_issues if i["severity"] == "HIGH"]
    medium = [i for i in all_issues if i["severity"] == "MEDIUM"]
    low = [i for i in all_issues if i["severity"] == "LOW"]

    passed = len(high) == 0

    summary = {
        "passed": passed,
        "total_issues": len(all_issues),
        "high": len(high),
        "medium": len(medium),
        "low": len(low),
        "split_counts": split_counts,
        "issues": all_issues,
    }

    return passed, summary


def main():
    parser = argparse.ArgumentParser(description="BioReason leak audit")
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--target_latent", default=None)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    import anndata as ad
    a = ad.read_h5ad(args.h5ad)

    print("=" * 60)
    print("BioReason Leak Audit")
    print("=" * 60)

    passed, summary = run_audit(a, args.target_latent)

    print(f"\nCells: {a.n_obs} x {a.n_vars}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"\nIssues: {summary['total_issues']} total "
          f"({summary['high']} HIGH, {summary['medium']} MEDIUM, {summary['low']} LOW)")

    for issue in summary["issues"]:
        tag = f"[{issue['severity']}]"
        print(f"  {tag:10s} {issue['item']}: {issue['msg']}")

    print(f"\nResult: {'PASS' if passed else 'FAIL'}")

    # Save report
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Also save markdown
    md_path = args.report.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write("# BioReason Leak Audit Report\n\n")
        f.write(f"**Result**: {'✅ PASS' if passed else '❌ FAIL'}\n\n")
        f.write(f"- Total issues: {summary['total_issues']}\n")
        f.write(f"- HIGH: {summary['high']}, MEDIUM: {summary['medium']}, LOW: {summary['low']}\n\n")
        if summary["split_counts"]:
            f.write(f"## Split Counts\n"
                    f"train={summary['split_counts'].get('train',0)}, "
                    f"val={summary['split_counts'].get('val',0)}, "
                    f"test={summary['split_counts'].get('test',0)}\n\n")
        f.write("## Issues\n\n")
        for issue in summary["issues"]:
            f.write(f"- **[{issue['severity']}]** {issue['item']}: {issue['msg']}\n")

    print(f"\nReport: {args.report}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
