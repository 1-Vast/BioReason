"""Test: PertDataset target construction and counterfactual."""
print("--- test_data ---")
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

# Build toy AnnData
try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed")
    sys.exit(0)

X = np.random.randn(10, 20)
obs = {
    "perturbation": ["control", "control", "A_KO", "A_KO", "B_KO", "B_KO",
                      "control", "A_KO", "B_KO", "control"],
    "cell_type":     ["T", "T", "T", "B", "T", "B", "B", "T", "B", "T"],
}
obsm = {"evidence": np.random.randn(10, 8)}
adata = anndata.AnnData(X=X, obs=obs, obsm=obsm)

from models.data import PertDataset, bio_collate_fn, build_loader

# group_mean
ds = PertDataset(adata, target_mode="group_mean", evidence_key="evidence")
assert ds.input_dim == 20
assert ds.n_perts == 3
assert ds.evidence_dim == 8
for i in range(len(ds)):
    item = ds[i]
    if item["pert_str"] != "control":
        assert not np.allclose(item["x"].numpy(), item["y"].numpy()), f"y==x for {item['pert_str']}"
print(f"  1) group_mean: {len(ds)} samples, evidence_dim={ds.evidence_dim} OK")

# Counterfactual
ds.set_target_pert("A_KO")
item = ds[0]
assert item["pert_str"] == "A_KO"
assert item["pert"] == ds.pert_to_id["A_KO"]
print("  2) set_target_pert OK")
ds.clear_target_pert()

# identity (debug)
ds_id = PertDataset(adata, target_mode="identity")
item = ds_id[2]
assert np.allclose(item["x"].numpy(), item["y"].numpy())
print("  3) identity OK")

# control_to_pert
ds_cp = PertDataset(adata, target_mode="control_to_pert", pair_by="cell_type")
has_diff = False
for i in range(len(ds_cp)):
    if ds_cp[i]["pert_str"] != "control":
        if not np.allclose(ds_cp[i]["x"].numpy(), ds_cp[i]["y"].numpy()):
            has_diff = True
            break
assert has_diff
print("  4) control_to_pert OK")

# collate_fn with evidence=None
batch = [ds[i] for i in range(4)]
collated = bio_collate_fn(batch)
assert collated["evidence"] is not None
assert collated["target_latent"] is None

# Build loader
loader = build_loader(ds, batch_size=4)
b = next(iter(loader))
assert b["x"].shape == (4, 20)
print("  5) build_loader + bio_collate_fn OK")

print("ALL OK")
