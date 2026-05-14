"""Test: stratified split produces disjoint train/val/test."""
print("--- test_split_no_leak ---")
import numpy as np
import anndata as ad
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from tools.split import stratified_cell_split, perturbation_holdout_split

# Create toy data
n = 300
adata = ad.AnnData(
    X=np.random.randn(n, 50).astype(np.float32),
    obs={"perturbation": ["control"] * 100 + ["A_KO"] * 100 + ["B_KO"] * 100}
)

# Test 1: cell mode split
a, counts, pert_counts = stratified_cell_split(
    adata.copy(), pert_key="perturbation", test_frac=0.2, val_frac=0.1, seed=42
)
assert "split" in a.obs, "split column missing"
splits = a.obs["split"].values
train = set(np.where(splits == "train")[0])
val = set(np.where(splits == "val")[0])
test = set(np.where(splits == "test")[0])
assert len(train & test) == 0, "train/test overlap"
assert len(train & val) == 0, "train/val overlap"
assert len(val & test) == 0, "val/test overlap"
assert counts["train"] > 0 and counts["test"] > 0
for p, c in pert_counts.items():
    assert c["train"] > 0, f"{p} has no train cells"
    assert c["test"] > 0, f"{p} has no test cells"
print("  1) cell split: train/val/test disjoint OK")

# Test 2: each perturbation appears in both train and test
for p in ["control", "A_KO", "B_KO"]:
    p_idx = np.where(a.obs["perturbation"].values == p)[0]
    p_splits = splits[p_idx]
    assert "train" in p_splits, f"{p} missing from train"
    assert "test" in p_splits, f"{p} missing from test"
print("  2) perturbation stratification OK")

# Test 3: split_info in uns
assert "split_info" in a.uns
info = a.uns["split_info"]
assert info["mode"] == "cell"
assert info["test_frac"] == 0.2
print("  3) split_info metadata OK")

# Test 4: perturbation_holdout mode
a2, c2, pi = perturbation_holdout_split(adata.copy(), n_holdout=1, seed=42)
assert len(pi["heldout"]) == 1
assert "split" in a2.obs
test_cells = (a2.obs["split"] == "test").sum()
assert test_cells > 0
print(f"  4) perturbation_holdout: heldout={pi['heldout']}, test_cells={test_cells} OK")

# Test 5: seed reproducibility
a3, _, _ = stratified_cell_split(adata.copy(), test_frac=0.2, seed=42)
a4, _, _ = stratified_cell_split(adata.copy(), test_frac=0.2, seed=42)
assert (a3.obs["split"] == a4.obs["split"]).all(), "split not reproducible"
print("  5) seed reproducibility OK")

print("ALL OK")
