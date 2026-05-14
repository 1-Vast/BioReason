"""Test: progress=True/False both run without error."""
print("--- test_progress_smoke ---")
import torch, sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

try:
    import anndata
    X = np.random.randn(20, 10)
    obs = {"perturbation": ["control","A_KO"]*10}
    adata = anndata.AnnData(X=X, obs=obs)
    from models.data import PertDataset, build_loader
    ds = PertDataset(adata)
except ImportError:
    print("SKIP: anndata not installed"); sys.exit(0)

from models.reason import BioReason
from models.loss import BioLoss
from models.train import train_epoch, validate
from utils.device import get_device, get_scaler

G, D = 10, 16
device = get_device("cpu")
model = BioReason(input_dim=G, dim=D, hidden=32, steps=1, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5).to(device)
loss_fn = BioLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = get_scaler(device, False)
loader = build_loader(ds, batch_size=8, shuffle=True)

# progress=False
print("  progress=False ...", end=" ", flush=True)
s = train_epoch(model, loader, loss_fn, optimizer, scaler, device, stage=1,
                use_amp=False, progress=False, log_every=100)
assert s["loss"] > 0
print("OK")

# progress=True
print("  progress=True ...", end=" ", flush=True)
s2 = train_epoch(model, loader, loss_fn, optimizer, scaler, device, stage=1,
                 use_amp=False, progress=True, log_every=100)
assert s2["loss"] > 0
print("OK")

# Validation with progress
v = validate(model, loader, loss_fn, device, stage=1, use_amp=False, progress=True)
assert v["loss"] > 0
print("  validate progress=True OK")

print("ALL OK")
