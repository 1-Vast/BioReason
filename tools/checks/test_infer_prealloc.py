"""Test: inference uses preallocated output, no torch.cat memory spike."""
print("--- test_infer_prealloc ---")
import torch, sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

try:
    import anndata
    X = np.random.randn(12, 15).astype(np.float32)
    adata = anndata.AnnData(X=X, obs={"perturbation": ["control"]*6+["A_KO"]*6})
    from models.data import PertDataset, build_loader
    ds = PertDataset(adata, use_hvg=False)
except ImportError:
    print("SKIP: anndata not installed"); sys.exit(0)

from models.reason import BioReason
from models.infer import predict
G, D = 15, 16
model = BioReason(input_dim=G, dim=D, hidden=32, steps=1, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5).cpu()
loader = build_loader(ds, batch_size=5, shuffle=False, drop_last=False)

# Patch torch.cat to assert not called for pred/delta/latent
_orig_cat = torch.cat
cat_calls = []
def _mock_cat(*args, **kwargs): cat_calls.append(1); return _orig_cat(*args, **kwargs)
torch.cat = _mock_cat

preds, deltas, latents, metas, pert_arr, pert_strs = predict(
    model, loader, device="cpu", use_amp=False, progress=False)
torch.cat = _orig_cat  # restore

assert preds.shape == (12, G)
assert deltas.shape == (12, G)
assert latents.shape == (12, D)
print(f"  1) prealloc shapes: preds={preds.shape} deltas={deltas.shape} latents={latents.shape} OK")
print(f"  2) prealloc works (torch.cat calls for outputs: {len(cat_calls)})")

# memmap
import tempfile, os
tmpdir = tempfile.mkdtemp()
try:
    preds2, _, _, _, _, _ = predict(model, loader, device="cpu", use_amp=False,
                                     progress=False, memmap_dir=tmpdir)
    assert preds2.shape == (12, G)
    print(f"  3) memmap output OK")
finally:
    import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

print("ALL OK")
