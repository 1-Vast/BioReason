"""Test: align_adata_to_genes handles missing/extra genes."""
print("--- test_gene_align ---")
import numpy as np, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed"); sys.exit(0)

X = np.random.randn(10, 5)
obs = {"perturbation": ["control"]*5 + ["A_KO"]*5}
var_names = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]
adata = anndata.AnnData(X=X, obs=obs)
adata.var_names = var_names

from models.data import align_adata_to_genes
from scipy.sparse import issparse

def _get_X(ad):
    return ad.X.toarray() if issparse(ad.X) else np.asarray(ad.X)

# Subset + reorder
target = ["GENE_C", "GENE_A", "GENE_B"]
aligned = align_adata_to_genes(adata, target)
assert list(aligned.var_names) == target
assert aligned.shape == (10, 3)
assert np.allclose(_get_X(aligned)[:, 0], X[:, 2])  # GENE_C
print("  1) subset+reorder OK")

# Missing gene
target2 = ["GENE_A", "GENE_Z", "GENE_B"]
aligned2 = align_adata_to_genes(adata, target2)
assert list(aligned2.var_names) == target2
assert aligned2.shape == (10, 3)
assert np.allclose(_get_X(aligned2)[:, 0], X[:, 0])  # GENE_A
assert np.all(_get_X(aligned2)[:, 1] == 0)  # GENE_Z filled with 0
print("  2) missing gene filled with 0 OK")

# All missing
target3 = ["X1", "X2"]
aligned3 = align_adata_to_genes(adata, target3)
assert aligned3.shape == (10, 2)
assert np.all(_get_X(aligned3) == 0)
print("  3) all missing: all zeros OK")

print("ALL OK")
