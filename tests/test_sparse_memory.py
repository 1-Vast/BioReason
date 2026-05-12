"""Test: sparse CSR AnnData stays sparse through PertDataset."""
print("--- test_sparse_memory ---")
import numpy as np, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

try:
    from scipy.sparse import issparse, csr_matrix, random as sp_random
    import anndata
except ImportError:
    print("SKIP: scipy/anndata not installed"); sys.exit(0)

# Sparse CSR: 1000 cells x 5000 genes, density 0.001
X = sp_random(1000, 5000, density=0.001, format="csr", dtype=np.float32)
obs = {"perturbation": ["control"] * 500 + ["A_KO"] * 500}
obsm = {"evidence": np.random.randn(1000, 8).astype(np.float32)}
adata = anndata.AnnData(X=X, obs=obs, obsm=obsm)

from models.data import PertDataset, _issparse

# No full densify
ds = PertDataset(adata, use_hvg=False, cache_group_means=True)
assert _issparse(ds.X), "X should remain sparse"
print("  1) X remains sparse OK")

# Row access returns dense
row = ds[0]["x"]
assert isinstance(row.numpy(), (np.ndarray, float))
assert len(row.shape) == 1
print("  2) __getitem__ returns dense row OK")

# Group means shape correct
assert "A_KO" in ds.group_means
assert ds.group_means["A_KO"].shape == (5000,)
print(f"  3) group_means shape={ds.group_means['A_KO'].shape} OK")

# No cache mode
ds2 = PertDataset(adata, use_hvg=False, cache_group_means=False)
row2 = ds2[50]
assert row2["x"].shape[0] == 5000
print("  4) cache_group_means=False OK")

print("ALL OK")
