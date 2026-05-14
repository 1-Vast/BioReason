"""Test: batch moves to GPU correctly, shapes and devices match."""
print("--- test_device_flow ---")
import torch, sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from utils.device import get_device, move_to_device, tensor_device_summary, cuda_available

try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed"); sys.exit(0)

X = np.random.randn(16, 20)
obs = {"perturbation": ["control","A_KO"]*8, "cell_type": ["T"]*16}
adata = anndata.AnnData(X=X, obs=obs)

from models.data import PertDataset, build_loader, batch_summary
ds = PertDataset(adata, target_mode="group_mean")
loader = build_loader(ds, batch_size=8, shuffle=False)
batch = next(iter(loader))

# CPU summary
print(f"  CPU batch: x={batch['x'].shape}, device={batch['x'].device}")

# Move to device
device = get_device("cuda" if cuda_available() else "cpu")
batch_gpu = move_to_device(batch, device, non_blocking=(device.type != "cpu"))
print(f"  GPU batch: x={batch_gpu['x'].shape}, device={batch_gpu['x'].device}")

assert batch_gpu["x"].device.type == device.type
assert batch_gpu["pert"].device.type == device.type
# evidence may be None
if batch_gpu.get("evidence") is not None:
    assert batch_gpu["evidence"].device.type == device.type
# meta should be a list
assert isinstance(batch_gpu["meta"], list)

# batch_summary
s = batch_summary(batch)
assert s["x"] == (8, 20)
print(f"  batch_summary: {s}")

# Forward with device batch
from models.reason import BioReason
model = BioReason(input_dim=20, dim=16, hidden=32, steps=2, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5).to(device)
out = model(batch_gpu["x"], batch_gpu["pert"], cov=batch_gpu.get("cov"))
assert out["pred"].device.type == device.type
print(f"  forward: pred device={out['pred'].device}, shape={out['pred'].shape}")

print("ALL OK")
