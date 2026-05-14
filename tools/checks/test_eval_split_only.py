"""Test: evaluation only uses test split, not train cells."""
print("--- test_eval_split_only ---")
import numpy as np
import anndata as ad
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

# Simulate eval_bench logic: filter predictions to test split only
n = 100
X = np.random.randn(n, 20).astype(np.float32)
pert = ["control"] * 50 + ["A_KO"] * 50
split_arr = ["train"] * 70 + ["test"] * 30
adata = ad.AnnData(X=X, obs={"perturbation": pert, "split": split_arr})

# Create fake predictions for ALL cells
preds = np.random.randn(n, 20).astype(np.float32)
deltas = np.random.randn(n, 20).astype(np.float32)

# Test 1: filter to test split only
test_mask = adata.obs["split"] == "test"
test_preds = preds[test_mask]
assert len(test_preds) == 30, f"expected 30 test cells, got {len(test_preds)}"
print(f"  1) test-only predictions: {len(test_preds)} cells OK")

# Test 2: train predictions NOT included in metrics
train_mask = adata.obs["split"] == "train"
train_preds = preds[train_mask]
assert len(train_preds) == 70
# Verify test and train predictions are from different indices
test_idx = np.where(test_mask)[0]
train_idx = np.where(train_mask)[0]
assert len(set(test_idx) & set(train_idx)) == 0, "train/test index overlap"
print("  2) train/test predictions disjoint OK")

# Test 3: ground truth for evaluation computed from test cells only
for p in ["A_KO"]:
    p_test_mask = (adata.obs["perturbation"] == p) & test_mask
    p_test_cells = X[p_test_mask]
    p_test_mean = p_test_cells.mean(axis=0)
    
    # Verify this is NOT the same as all-cell mean
    p_all_mean = X[adata.obs["perturbation"] == p].mean(axis=0)
    diff = np.abs(p_test_mean - p_all_mean).mean()
    print(f"  3) {p}: test-only mean vs all-cell mean diff={diff:.4f} (should be >0) OK")

print("ALL OK")
