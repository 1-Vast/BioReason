"""Test Stage 2 evidence warm-up branch and fallback behavior."""
print("--- test_stage2_warm ---")
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

try:
    import anndata
except ImportError:
    print("SKIP: anndata not installed")
    sys.exit(0)

from models.data import PertDataset, build_loader
from models.loss import BioLoss
from models.reason import BioReason
from models.train import train_epoch
from utils.device import get_scaler

torch.manual_seed(1)
np.random.seed(1)

B, G, ED = 8, 20, 6
X = np.random.randn(B, G).astype(np.float32)
obs = {
    "perturbation": ["control", "control", "A_KO", "A_KO", "B_KO", "B_KO", "A_KO", "B_KO"],
    "cell_type": ["T"] * B,
}
obsm = {"evidence": np.random.randn(B, ED).astype(np.float32)}
adata = anndata.AnnData(X=X, obs=obs, obsm=obsm)

ds = PertDataset(adata, target_mode="group_mean", evidence_key="evidence", use_hvg=False)
loader = build_loader(ds, batch_size=4, shuffle=False, drop_last=True)
model = BioReason(input_dim=G, dim=16, hidden=32, steps=1, heads=2, dropout=0.0,
                  pert_mode="id", num_perts=ds.n_perts, evidence_dim=ED)
loss_fn = BioLoss({"expr": 1.0, "delta": 1.0, "deg": 1.0, "evidence": 1.0,
                   "mmd": 0.0, "evi_gain": 0.5, "evi_shift": 0.0})
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
stats = train_epoch(
    model, loader, loss_fn, opt, get_scaler(torch.device("cpu"), enabled=False),
    torch.device("cpu"), stage=2, use_amp=False, progress=False,
    epoch=1, evi_warm=True, evi_warm_epochs=2, evi_warm_margin=0.02,
)

assert "evi_gain" in stats
assert "z_shift" in stats
assert "evi_rec" in stats
assert stats["z_shift"] >= 0.0
assert stats["evi_rec"] >= 0.0
print("  1) stage=2 warm stats OK")

batch = next(iter(loader))
with torch.no_grad():
    out_ev = model(batch["x"], batch["pert"], cov=batch.get("cov"), evidence=batch["evidence"], return_latent=True)
    out_no = model(batch["x"], batch["pert"], cov=batch.get("cov"), evidence=None, return_latent=True)
    L_ev = loss_fn(out_ev, {"x": batch["x"], "y": batch["y"], "evidence": batch["evidence"]}, stage=2)
    L_no = loss_fn(out_no, {"x": batch["x"], "y": batch["y"], "evidence": None}, stage=1)
    evi_gain = L_no["deg"] - L_ev["deg"]
    z_shift = torch.norm(out_ev["latent"] - out_no["latent"], dim=-1).mean()
    evi_rec = L_ev["evidence"]

assert out_ev["pred"].shape == out_no["pred"].shape == (4, G)
assert out_ev["latent"].shape == out_no["latent"].shape
assert evi_gain.numel() == z_shift.numel() == evi_rec.numel() == 1
print("  2) out_ev/out_no metrics OK")

adata_no = anndata.AnnData(X=X, obs=obs)
ds_no = PertDataset(adata_no, target_mode="group_mean", evidence_key="evidence", use_hvg=False)
loader_no = build_loader(ds_no, batch_size=4, shuffle=False, drop_last=True)
model_no = BioReason(input_dim=G, dim=16, hidden=32, steps=1, heads=2, dropout=0.0,
                     pert_mode="id", num_perts=ds_no.n_perts, evidence_dim=ED)
opt_no = torch.optim.AdamW(model_no.parameters(), lr=1e-3)
stats_no = train_epoch(
    model_no, loader_no, loss_fn, opt_no, get_scaler(torch.device("cpu"), enabled=False),
    torch.device("cpu"), stage=2, use_amp=False, progress=False,
    epoch=1, evi_warm=True, evi_warm_epochs=2,
)
assert stats_no["loss"] > 0
assert stats_no["evi_gain"] == 0.0
print("  3) evidence=None fallback OK")

for stage in (1, 3):
    model_s = BioReason(input_dim=G, dim=16, hidden=32, steps=1, heads=2, dropout=0.0,
                        pert_mode="id", num_perts=ds.n_perts, evidence_dim=ED)
    opt_s = torch.optim.AdamW(model_s.parameters(), lr=1e-3)
    stats_s = train_epoch(
        model_s, loader, loss_fn, opt_s, get_scaler(torch.device("cpu"), enabled=False),
        torch.device("cpu"), stage=stage, use_amp=False, progress=False,
        epoch=1, evi_warm=True, evi_warm_epochs=2,
    )
    assert stats_s["loss"] > 0
    assert stats_s["evi_gain"] == 0.0
print("  4) Stage 1/3 unchanged OK")

print("ALL OK")
