"""Build leakage-safe tensor cache from an h5ad file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from models.data import build_train_val_datasets, read_h5ad, PertDataset, align_adata_to_genes, _ensure_source_idx


def _save_split(ds, out_dir: Path, split: str, dtype: torch.dtype) -> int:
    rows = [ds[i] for i in range(len(ds))]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack([r["x"] for r in rows]).to(dtype), out_dir / f"x_{split}.pt")
    torch.save(torch.stack([r["y"] for r in rows]).to(dtype), out_dir / f"y_{split}.pt")
    torch.save(torch.stack([r["pert"] for r in rows]).long(), out_dir / f"pert_{split}.pt")
    ev = [r["evidence"] for r in rows]
    if ev and ev[0] is not None:
        torch.save(torch.stack(ev).to(dtype), out_dir / f"evidence_{split}.pt")
    else:
        torch.save(torch.zeros((len(rows), int(ds.evidence_dim or 1)), dtype=dtype), out_dir / f"evidence_{split}.pt")
    conf = [r["evidence_conf"] if r["evidence_conf"] is not None else torch.tensor(1.0) for r in rows]
    torch.save(torch.stack(conf).float(), out_dir / f"evidence_conf_{split}.pt")
    cov = {k: torch.stack([r["cov"][k] for r in rows]).long() for k in ds.cov_dims}
    torch.save(cov, out_dir / f"cov_{split}.pt")
    idx = [int(r["meta"]["idx"]) for r in rows]
    torch.save(torch.tensor(idx, dtype=torch.long), out_dir / f"idx_{split}.pt")
    pe = [r["perturbation_effect"] for r in rows]
    torch.save(torch.stack(pe).to(dtype), out_dir / f"perturbation_effect_{split}.pt")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--target_mode", default="group_mean")
    ap.add_argument("--control_input_mode", default="control_mean")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--device_preload", default="false")
    ap.add_argument("--label_key", default="perturbation")
    ap.add_argument("--control_label", default="control")
    ap.add_argument("--cell_type_key", default="cell_type")
    ap.add_argument("--evidence_key", default="evidence")
    ap.add_argument("--n_hvg", type=int, default=2000)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    out_dir = Path(args.out)
    adata = _ensure_source_idx(read_h5ad(args.h5ad))
    cfg = {
        "label_key": args.label_key,
        "control_label": args.control_label,
        "cell_type_key": args.cell_type_key,
        "evidence_key": args.evidence_key,
        "target_mode": args.target_mode,
        "pair_by": args.cell_type_key,
        "use_control_as_input": True,
        "use_hvg": True,
        "n_hvg": args.n_hvg,
        "cache_group_means": True,
        "control_input_mode": args.control_input_mode,
    }
    train_ds, _ = build_train_val_datasets(args.h5ad, cfg)
    counts = {"train": _save_split(train_ds, out_dir, "train", dtype)}

    split_values = adata.obs["split"].astype(str).values if "split" in adata.obs else None
    for split in [s.strip() for s in args.splits.split(",") if s.strip() and s.strip() != "train"]:
        if split_values is None:
            continue
        mask = split_values == split
        if not mask.any():
            continue
        split_adata = align_adata_to_genes(adata[mask].copy(), train_ds.selected_var_names)
        split_cfg = dict(cfg)
        split_cfg.update({"use_hvg": False, "stats_split": None, "target_mode": "per_cell",
                          "cache_group_means": False, "warn_on_missing_stats": False})
        ds = PertDataset(split_adata, **split_cfg)
        from models.data import _copy_aligned_metadata
        _copy_aligned_metadata(ds, train_ds)
        ds.selected_var_names = list(train_ds.selected_var_names)
        counts[split] = _save_split(ds, out_dir, split, dtype)

    meta = {
        "h5ad": args.h5ad,
        "selected_var_names": train_ds.selected_var_names,
        "split_counts": counts,
        "perturbation_vocab": train_ds.pert_to_id,
        "pert_to_id": train_ds.pert_to_id,
        "id_to_pert": {str(k): v for k, v in train_ds.id_to_pert.items()},
        "pert_cats": train_ds.pert_cats,
        "input_dim": train_ds.input_dim,
        "evidence_dim": int(train_ds.evidence_dim or 1),
        "cov_dims": dict(train_ds.cov_dims),
        "dtype": args.dtype,
        "target_mode": args.target_mode,
        "control_input_mode": args.control_input_mode,
        "stats_source": "train",
        "leakage_safe": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"cache_dir": str(out_dir), "counts": counts, "leakage_safe": True}, indent=2))


if __name__ == "__main__":
    main()
