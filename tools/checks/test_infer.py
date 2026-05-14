"""Test: Inference counterfactual logic."""
print("--- test_infer ---")
import torch, sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

# Toy model
G, D = 20, 16
from models.reason import BioReason
model = BioReason(input_dim=G, dim=D, hidden=32, steps=2, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5)
model.eval()

# Toy AnnData
try:
    import anndata
    X = np.random.randn(8, G)
    obs = {"perturbation": ["control"]*4 + ["A_KO"]*4}
    adata = anndata.AnnData(X=X, obs=obs)
    from models.data import PertDataset
    ds = PertDataset(adata, target_mode="group_mean")
except ImportError:
    print("SKIP: anndata not installed")
    sys.exit(0)

# Counterfactual
from models.infer import predict_counterfactual
preds, deltas, latents, metas, pert_arr, pert_strs = predict_counterfactual(
    model, ds, "A_KO", batch_size=4, device="cpu")
assert preds.shape == (8, G)
assert np.all(pert_arr == ds.pert_to_id["A_KO"])
assert all(m["pert_str"] == "A_KO" for m in metas)
assert all(ps == "A_KO" for ps in pert_strs)
print(f"  1) counterfactual: preds={preds.shape}, all pert=A_KO OK")

# Clear and re-test with control
ds.clear_target_pert()
preds2, _, _, metas2, pert_arr2, pert_strs2 = predict_counterfactual(model, ds, "control", batch_size=4, device="cpu")
assert np.all(pert_arr2 == ds.pert_to_id["control"])
print("  2) control counterfactual OK")

print("ALL OK")
