"""Test: target_latent only loaded for train cells, test cells masked."""
print("--- test_target_latent_no_leak ---")
import numpy as np
import torch
import anndata as ad
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from models.data import PertDataset

# Create toy data with train/test split
n = 60
X = np.random.randn(n, 10).astype(np.float32)
pert = ["control"] * 30 + ["A_KO"] * 30
split_arr = ["train"] * 40 + ["test"] * 20
adata = ad.AnnData(X=X, obs={"perturbation": pert, "split": split_arr})

# Create fake target_latent.pt
import tempfile, os
latent = torch.randn(n, 8)
indices = torch.arange(n)
tmp_path = os.path.join(tempfile.gettempdir(), "test_target_latent.pt")
torch.save({"latent": latent, "indices": indices}, tmp_path)

# Test 1: load into full dataset 鈫?mask only True for train cells
ds = PertDataset(adata, split=None, stats_split="train", use_hvg=False)
ds.load_target_latent(tmp_path)

# Train cells (indices 0-39) should have mask=True
for i in range(40):
    assert ds.target_latent_mask[i].item() == True, f"train cell {i} masked out"
# Test cells (indices 40-59) should have mask=False
for i in range(40, 60):
    assert ds.target_latent_mask[i].item() == False, f"test cell {i} NOT masked"
print("  1) train cells masked True, test cells masked False OK")

# Test 2: train split dataset inherits target_latent_mask
ds_train = PertDataset(adata, split="train", stats_split="train", use_hvg=False)
ds_train.load_target_latent(tmp_path)
# All train cells should have mask=True
for i in range(40):
    assert ds_train.target_latent_mask[i].item() == True
print("  2) train dataset: all cells masked True OK")

# Test 3: test split dataset has mask=False
ds_test = PertDataset(adata, split="test", stats_split="train", use_hvg=False)
ds_test.load_target_latent(tmp_path)
for i in range(40, 60):
    assert ds_test.target_latent_mask[i].item() == False
print("  3) test dataset: all cells masked False OK")

os.unlink(tmp_path)
print("ALL OK")
