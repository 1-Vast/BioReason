"""Test reason_mode ablation wiring."""
print("--- test_reason_mode ---")
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.reason import BioReason
from models.infer import load_model
from models.train import save_ckpt

B, G, D, ED = 3, 12, 8, 5
x = torch.randn(B, G)
pert = torch.randint(0, 4, (B,))
evidence = torch.randn(B, ED)

for mode in ("transformer", "mlp"):
    model = BioReason(input_dim=G, dim=D, hidden=16, steps=1, heads=2,
                      dropout=0.0, pert_mode="id", num_perts=4,
                      evidence_dim=ED, reason_mode=mode)
    out = model(x, pert, evidence=evidence)
    assert out["pred"].shape == (B, G)
    assert out["latent"].shape == (B, D)
print("  1) transformer/mlp forward OK")

try:
    BioReason(input_dim=G, dim=D, hidden=16, steps=1, heads=2,
              dropout=0.0, pert_mode="id", num_perts=4,
              evidence_dim=ED, reason_mode="gru")
    assert False, "gru reason_mode should raise"
except ValueError:
    print("  2) invalid reason_mode raises OK")

cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "model.yaml").read_text())
assert cfg["model"]["reason_mode"] in ("transformer", "mlp")
assert cfg["model"]["reason_mode"] == "transformer"
print("  3) config reason_mode readable OK")

with tempfile.TemporaryDirectory() as tmp:
    model = BioReason(input_dim=G, dim=D, hidden=16, steps=1, heads=2,
                      dropout=0.0, pert_mode="id", num_perts=4,
                      evidence_dim=ED, reason_mode="mlp")
    config = {"model": {
        "input_dim": G, "dim": D, "hidden": 16, "latent_steps": 1,
        "heads": 2, "dropout": 0.0, "pert_mode": "id",
        "pert_agg": "mean", "num_perts": 4, "evidence_dim": ED,
        "cov_dims": {}, "residual": True, "reason_mode": "mlp",
        "evidence_mode": "film", "use_evidence_conf": True,
    }}
    path = os.path.join(tmp, "model.pt")
    save_ckpt(model, None, 1, config, path)
    loaded, loaded_cfg = load_model(path, device="cpu")
    assert loaded.reason_mode == "mlp"
    assert loaded_cfg["model"]["reason_mode"] == "mlp"
    out = loaded(x, pert, evidence=evidence)
    assert out["pred"].shape == (B, G)
print("  4) checkpoint/load_model preserves reason_mode OK")

print("ALL OK")
