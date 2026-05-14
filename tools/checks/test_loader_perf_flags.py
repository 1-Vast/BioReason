"""Test: build_loader handles num_workers=0 and >0 correctly."""
print("--- test_loader_perf_flags ---")
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

try:
    import anndata
    X = np.random.randn(20, 10)
    adata = anndata.AnnData(X=X, obs={"perturbation": ["control"]*20})
    from models.data import PertDataset, build_loader
    ds = PertDataset(adata)
except ImportError:
    print("SKIP: anndata not installed"); sys.exit(0)

# num_workers=0: prefetch/persistent not passed (no error)
l0 = build_loader(ds, num_workers=0, batch_size=4)
b0 = next(iter(l0))
assert b0["x"].shape == (4, 10)
print("  1) num_workers=0 OK (prefetch/persistent not passed)")

# pin_memory flag
l1 = build_loader(ds, num_workers=0, batch_size=4, pin_memory=True)
b1 = next(iter(l1))
print("  2) pin_memory=True OK")

# drop_last True
l2 = build_loader(ds, num_workers=0, batch_size=7, drop_last=True)
cnt = len(list(l2))
assert cnt == 2, f"expected 2 batches (14/7), got {cnt}"
print("  3) drop_last=True OK")

print("ALL OK")
