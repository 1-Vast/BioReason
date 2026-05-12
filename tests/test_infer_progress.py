"""Test: inference with progress=True/False."""
print("--- test_infer_progress ---")
import torch, sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

try:
    import anndata
    X = np.random.randn(12, 15)
    adata = anndata.AnnData(X=X, obs={"perturbation": ["control"]*6+["A_KO"]*6})
    from models.data import PertDataset, build_loader
    ds = PertDataset(adata)
except ImportError:
    print("SKIP: anndata not installed"); sys.exit(0)

from models.reason import BioReason
from models.infer import predict
G = 15; D = 16
model = BioReason(input_dim=G, dim=D, hidden=32, steps=1, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5).cpu()
loader = build_loader(ds, batch_size=6, shuffle=False)

# progress=False
print("  progress=False ...", end=" ", flush=True)
preds, deltas, latents, metas, pert_arr, pert_strs = predict(
    model, loader, device="cpu", use_amp=False, progress=False)
assert preds.shape == (12, G)
print("OK")

# progress=True
print("  progress=True ...", end=" ", flush=True)
preds2, _, _, _, _, _ = predict(model, loader, device="cpu", use_amp=False, progress=True)
assert preds2.shape == (12, G)
print("OK")

print("ALL OK")
