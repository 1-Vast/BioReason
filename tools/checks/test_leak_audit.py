"""Test: leak audit catches common leakage patterns."""
print("--- test_leak_audit ---")
import numpy as np
import anndata as ad
import torch
import json
import tempfile
import os
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from tools.audit_leak import run_audit

# Test 1: clean data passes audit
n = 100
X = np.random.randn(n, 20).astype(np.float32)
adata = ad.AnnData(X=X, obs={
    "perturbation": ["control"] * 50 + ["A_KO"] * 50,
    "split": ["train"] * 70 + ["test"] * 30,
})
adata.uns["split_info"] = {
    "hvg_from_train": True,
    "group_means_source": "train",
}

passed, summary = run_audit(adata)
assert passed, f"Clean data should pass audit, but got {summary['high']} HIGH issues"
print(f"  1) clean data: PASS (issues={summary['total_issues']}) OK")

# Test 2: no split column 鈫?HIGH issue
adata2 = ad.AnnData(X=np.random.randn(50, 10).astype(np.float32))
passed2, s2 = run_audit(adata2)
assert not passed2 or s2["high"] > 0, "No-split data should have issues"
print(f"  2) no split: {'PASS' if passed2 else 'FAIL'} (high={s2['high']}) OK")

# Test 3: target_latent with test cells 鈫?HIGH issue
adata3 = ad.AnnData(X=np.random.randn(100, 10).astype(np.float32), obs={
    "perturbation": ["control"] * 50 + ["A_KO"] * 50,
    "split": ["train"] * 70 + ["test"] * 30,
})

# Create target_latent with test indices
tmp = os.path.join(tempfile.gettempdir(), "test_leak_tl.pt")
latent = torch.randn(10, 8)
indices = torch.arange(70, 80)  # test indices (70-79)
torch.save({"latent": latent, "indices": indices}, tmp)

passed3, s3 = run_audit(adata3, tmp)
assert not passed3, "target_latent with test indices should FAIL"
print(f"  3) target_latent with test cells: FAIL (high={s3['high']}) OK")

os.unlink(tmp)

# Test 4: hvg not from train 鈫?MEDIUM issue
adata4 = ad.AnnData(X=np.random.randn(50, 10).astype(np.float32), obs={
    "split": ["train"] * 35 + ["test"] * 15,
})
adata4.uns["split_info"] = {"hvg_from_train": False}
passed4, s4 = run_audit(adata4)
assert s4["medium"] > 0 or not passed4, "HVG not from train should be flagged"
print(f"  4) HVG not from train: flagged (medium={s4['medium']}) OK")

print("ALL OK")
