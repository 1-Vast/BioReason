"""Strict train/validation leakage checks."""

import tempfile
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

print("--- test_strict_no_leak ---")

from models.data import build_train_val_datasets


def _write_adata(X, split, pert, var_names):
    obs = pd.DataFrame({"split": split, "perturbation": pert})
    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs, var=var)
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "toy.h5ad"
    adata.write_h5ad(path)
    return tmp, path


def test_hvg_uses_train_only():
    X = [
        [0, 0, 0, 0, 0],
        [10, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [10, 0, 0, 0, 0],
        [0, 0, 0, 0, 1000],
        [0, 0, 0, 0, -1000],
    ]
    split = ["train", "train", "train", "train", "val", "val"]
    pert = ["control", "A_KO", "control", "A_KO", "A_KO", "A_KO"]
    tmp, path = _write_adata(X, split, pert, [f"g{i}" for i in range(5)])
    try:
        train_ds, val_ds = build_train_val_datasets(
            path,
            {"use_hvg": True, "n_hvg": 1, "target_mode": "group_mean"},
        )
        assert train_ds.selected_var_names == ["g0"]
        assert val_ds.selected_var_names == ["g0"]
        assert train_ds.input_dim == 1
        assert val_ds.input_dim == 1
    finally:
        tmp.cleanup()


def test_group_mean_uses_train_only():
    X = [[0, 0], [2, 0], [2, 0], [0, 0], [100, 0], [100, 0]]
    split = ["train", "train", "train", "train", "val", "val"]
    pert = ["control", "A_KO", "A_KO", "control", "A_KO", "A_KO"]
    tmp, path = _write_adata(X, split, pert, ["g0", "g1"])
    try:
        _, val_ds = build_train_val_datasets(
            path,
            {"use_hvg": False, "target_mode": "group_mean"},
        )
        item = val_ds[0]
        assert item["pert_str"] == "A_KO"
        assert abs(float(item["y"][0]) - 2.0) < 1e-6
        assert item["meta"]["idx"] == 4
        assert item["meta"]["source_idx"] == 4
    finally:
        tmp.cleanup()


test_hvg_uses_train_only()
print("  1) HVG uses train split only")
test_group_mean_uses_train_only()
print("  2) group mean targets use train split only")
print("ALL OK")
