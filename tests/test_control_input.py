"""Test: _get_control_input uses control pool for non-control cells."""
print("--- test_control_input ---")
import numpy as np, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed"); sys.exit(0)

np.random.seed(42)
X = np.random.randn(10, 20)
obs = {
    "perturbation": ["control","control","A_KO","A_KO","B_KO","B_KO",
                      "control","A_KO","B_KO","control"],
    "cell_type":     ["T","T","T","B","T","B","B","T","B","T"],
}
adata = anndata.AnnData(X=X, obs=obs)

from models.data import PertDataset

# use_control_as_input=True (default)
ds = PertDataset(adata, target_mode="group_mean", use_control_as_input=True, pair_by="cell_type")

# Non-control cell (idx=3, A_KO, cell_type B) should get input from control pool
item = ds[3]
assert item["pert_str"] == "A_KO"
# The input x should not equal the original X[3]
ctrl_pool_x = [ds[i]["x"].numpy() for i in ds.control_indices]
found = any(np.allclose(item["x"].numpy(), cx) for cx in ctrl_pool_x)
assert found, "Non-control cell did not get input from control pool"
print("  1) use_control_as_input=True: non-control gets control input OK")

# Control cell should use its own expression
item_ctrl = ds[0]
assert item_ctrl["pert_str"] == "control"
assert np.allclose(item_ctrl["x"].numpy(), X[0])
print("  2) control cell uses own expression OK")

# use_control_as_input=False
ds2 = PertDataset(adata, target_mode="group_mean", use_control_as_input=False)
item2 = ds2[3]
assert np.allclose(item2["x"].numpy(), X[3])
print("  3) use_control_as_input=False: uses own expression OK")

# identity mode
ds3 = PertDataset(adata, target_mode="identity", use_control_as_input=True)
item3 = ds3[3]
assert np.allclose(item3["x"].numpy(), X[3])
print("  4) identity mode: uses own expression OK")

print("ALL OK")
