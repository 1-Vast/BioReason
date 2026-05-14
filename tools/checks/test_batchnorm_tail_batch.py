"""Test: build_loader avoids batch size 1 when avoid_single_batch=True."""
print("--- test_batchnorm_tail_batch ---")
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

try:
    import anndata
    X = np.random.randn(9, 10).astype(np.float32)  # 9 samples
    adata = anndata.AnnData(X=X, obs={"perturbation": ["control"]*9})
    from models.data import PertDataset, build_loader
    ds = PertDataset(adata, use_hvg=False)
except ImportError:
    print("SKIP: anndata not installed"); sys.exit(0)

# batch_size=4 鈫?9%4=1 鈫?should drop_last=True with warning
loader = build_loader(ds, batch_size=4, shuffle=False, avoid_single_batch=True)
batches = list(loader)
sizes = [b["x"].shape[0] for b in batches]
assert 1 not in sizes, f"Found batch size 1: {sizes}"
print(f"  1) avoid_single_batch: batches sizes={sizes} OK")

# Without avoid_single_batch 鈫?last batch size 1
loader2 = build_loader(ds, batch_size=4, shuffle=False, avoid_single_batch=False, drop_last=False)
batches2 = list(loader2)
sizes2 = [b["x"].shape[0] for b in batches2]
assert 1 in sizes2, f"Expected last batch size 1, got {sizes2}"
print(f"  2) no avoid: batch sizes={sizes2} OK")

# BatchNorm + tail batch = no crash in train_epoch (if drop_last)
import torch
from models.reason import BioReason
from models.loss import BioLoss
from models.train import train_epoch
from utils.device import get_device, get_scaler

device = get_device("cpu")
model = BioReason(input_dim=10, dim=16, hidden=32, steps=1, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5).to(device)
loader3 = build_loader(ds, batch_size=4, shuffle=True, avoid_single_batch=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = get_scaler(device, False)
loss_fn = BioLoss()
stats = train_epoch(model, loader3, loss_fn, optimizer, scaler, device, stage=1,
                    use_amp=False, progress=False, log_every=100)
assert stats["loss"] >= 0
print(f"  3) train_epoch with avoid_single_batch OK")

print("ALL OK")
