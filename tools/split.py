"""Stratified data split with obs["split"] annotation.

Supports:
  cell mode: stratify by perturbation, split cells
  perturbation_holdout: hold out entire perturbations

Usage:
  python tools/split.py --h5ad dataset/bench.h5ad --out dataset/bench_split.h5ad
"""

import argparse
import numpy as np
import anndata as ad
from pathlib import Path


def stratified_cell_split(adata, pert_key="perturbation", control_label="control",
                           cell_type_key="cell_type", test_frac=0.2, val_frac=0.1, seed=42):
    """Stratified split: each perturbation gets train/val/test cells."""
    np.random.seed(seed)
    pert_arr = adata.obs[pert_key].astype(str).values

    split_arr = np.full(len(adata), "train", dtype=object)
    counts = {"train": 0, "val": 0, "test": 0}
    pert_counts = {}

    for p in sorted(set(pert_arr)):
        p_idx = np.where(pert_arr == p)[0]
        n_total = len(p_idx)
        n_test = max(1, int(n_total * test_frac))
        n_val = max(0, int(n_total * val_frac)) if n_total >= 10 else 0

        np.random.shuffle(p_idx)
        test_idx = p_idx[:n_test]
        val_idx = p_idx[n_test:n_test + n_val] if n_val > 0 else np.array([], dtype=int)
        train_idx = p_idx[n_test + n_val:]

        split_arr[test_idx] = "test"
        if n_val > 0:
            split_arr[val_idx] = "val"

        counts["train"] += len(train_idx)
        counts["val"] += len(val_idx)
        counts["test"] += len(test_idx)
        pert_counts[p] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}

        if n_total < 10:
            print(f"  WARNING: {p} has only {n_total} cells — insufficient for reliable split")

    adata.obs["split"] = split_arr

    # Verify disjoint
    train_set = set(np.where(split_arr == "train")[0])
    val_set = set(np.where(split_arr == "val")[0])
    test_set = set(np.where(split_arr == "test")[0])
    assert len(train_set & val_set) == 0, "train/val overlap!"
    assert len(train_set & test_set) == 0, "train/test overlap!"
    assert len(val_set & test_set) == 0, "val/test overlap!"

    adata.uns["split_info"] = {
        "mode": "cell",
        "pert_key": pert_key,
        "control_label": control_label,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "seed": seed,
        "counts": counts,
        "pert_counts": pert_counts,
        "hvg_from_train": True,          # HVG must be selected from train split only
        "group_means_source": "train",   # group_means computed from train split only
    }

    return adata, counts, pert_counts


def perturbation_holdout_split(adata, pert_key="perturbation", control_label="control",
                                n_holdout=2, seed=42):
    """Hold out entire perturbations for OOD testing."""
    np.random.seed(seed)
    pert_arr = adata.obs[pert_key].astype(str).values
    pert_names = sorted(set(pert_arr))

    non_control = [p for p in pert_names if p != control_label]
    np.random.shuffle(non_control)
    holdout_perts = non_control[:n_holdout]
    train_perts = non_control[n_holdout:]

    split_arr = np.full(len(adata), "train", dtype=object)
    for p in holdout_perts:
        split_arr[pert_arr == p] = "test"

    adata.obs["split"] = split_arr

    adata.uns["split_info"] = {
        "mode": "perturbation_holdout",
        "pert_key": pert_key,
        "heldout_perts": holdout_perts,
        "train_perts": train_perts + [control_label],
        "seed": seed,
    }

    counts = {
        "train": int((split_arr == "train").sum()),
        "test": int((split_arr == "test").sum()),
    }

    return adata, counts, {"heldout": holdout_perts, "train": train_perts}


def main():
    parser = argparse.ArgumentParser(description="Stratified data split")
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pert_key", default="perturbation")
    parser.add_argument("--control_label", default="control")
    parser.add_argument("--cell_type_key", default="cell_type")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="cell", choices=["cell", "perturbation_holdout"])
    parser.add_argument("--n_holdout", type=int, default=2)
    args = parser.parse_args()

    a = ad.read_h5ad(args.h5ad)
    print(f"Input: {a.n_obs} cells x {a.n_vars} genes")

    if args.mode == "cell":
        a, counts, pert_counts = stratified_cell_split(
            a, args.pert_key, args.control_label,
            args.cell_type_key, args.test_frac, args.val_frac, args.seed
        )
        print(f"\nSplit counts:")
        print(f"  train: {counts['train']}, val: {counts['val']}, test: {counts['test']}")
        print(f"\nPer-perturbation:")
        for p, c in pert_counts.items():
            print(f"  {p}: train={c['train']}, val={c['val']}, test={c['test']}")
    else:
        a, counts, pert_info = perturbation_holdout_split(
            a, args.pert_key, args.control_label, args.n_holdout, args.seed
        )
        print(f"\nHeld-out perturbations: {pert_info['heldout']}")
        print(f"Train perturbations: {pert_info['train']}")
        print(f"Train cells: {counts['train']}, Test cells: {counts['test']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
