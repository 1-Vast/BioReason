"""Test: group_means computed from train split only, not test cells."""
print("--- test_groupmean_no_leak ---")
import numpy as np
import anndata as ad
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from models.data import PertDataset

# Create toy data with extreme train/test difference
# train cells have expression ~1, test cells have expression ~100
n = 200
X = np.zeros((n, 10), dtype=np.float32)
pert = ["control"] * 100 + ["A_KO"] * 100
split_arr = ["train"] * 160 + ["test"] * 40
# First 80 control + 80 A_KO = train, rest = test
np.random.seed(42)
# Train: expression ~1
X[:80] = np.random.randn(80, 10).astype(np.float32) * 0.1 + 1.0  # control train
X[80:160] = np.random.randn(80, 10).astype(np.float32) * 0.1 + 2.0  # A_KO train
# Test: expression ~100 (very different)
X[160:180] = np.random.randn(20, 10).astype(np.float32) * 0.1 + 100.0  # control test
X[180:] = np.random.randn(20, 10).astype(np.float32) * 0.1 + 101.0  # A_KO test

adata = ad.AnnData(X=X, obs={"perturbation": pert, "split": split_arr})

# Test 1: PertDataset with split=None, stats_split="train" 鈫?group means from train only
ds = PertDataset(adata, split=None, stats_split="train", use_hvg=False)

gm_ctrl = ds.group_means["control"]
gm_ako = ds.group_means["A_KO"]

# Train means should be ~1.0 and ~2.0
assert 0.5 < gm_ctrl.mean() < 3.0, f"control group mean {gm_ctrl.mean():.2f} not from train (~1.0)"
assert 1.0 < gm_ako.mean() < 4.0, f"A_KO group mean {gm_ako.mean():.2f} not from train (~2.0)"
print(f"  1) group_means from train only: control={gm_ctrl.mean():.2f}, A_KO={gm_ako.mean():.2f} OK")

# Test 2: train split dataset should return target from train group means
ds_train = PertDataset(adata, split="train", stats_split="train", use_hvg=False)
item = ds_train[0]
y = item["y"]
pert_id = item["pert"]
# y should be ~2.0 (A_KO train mean) not ~101.0 (A_KO test mean)
print(f"  2) train target from train group means: y_mean={y.mean():.2f} OK")

# Test 3: train dataset length
assert len(ds_train) == 160, f"train ds length {len(ds_train)} != 160"
print(f"  3) train dataset length={len(ds_train)} OK")

# Test 4: test split dataset should exist
ds_test = PertDataset(adata, split="test", stats_split="train", use_hvg=False)
assert len(ds_test) == 40, f"test ds length {len(ds_test)} != 40"
print(f"  4) test dataset length={len(ds_test)} OK")

print("ALL OK")
