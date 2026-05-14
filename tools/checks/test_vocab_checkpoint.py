"""Test: checkpoint saves and restores vocab, dims, gene order."""
print("--- test_vocab_checkpoint ---")
import torch, sys, tempfile, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from models.reason import BioReason
from models.infer import load_model
from models.train import save_ckpt

G, D = 20, 16
model = BioReason(input_dim=G, dim=D, hidden=32, steps=2, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5, evidence_dim=8, cov_dims={"cell_type": 3})

# Run a dummy forward to initialize all parameters
x = torch.randn(2, G); pert = torch.randint(0, 5, (2,))
_ = model(x, pert)

config = {
    "model": {"input_dim": G, "dim": D, "hidden": 32, "latent_steps": 2,
               "heads": 2, "dropout": 0.0, "pert_mode": "id", "pert_agg": "mean",
               "num_perts": 5, "evidence_dim": 8, "cov_dims": {"cell_type": 3},
               "residual": True},
    "data_meta": {
        "pert_to_id": {"control": 0, "A_KO": 1, "B_KO": 2},
        "id_to_pert": {0: "control", 1: "A_KO", 2: "B_KO"},
        "pert_cats": ["control", "A_KO", "B_KO"],
        "selected_var_names": [f"gene_{i}" for i in range(G)],
        "input_dim": G,
        "evidence_dim": 8,
        "cov_dims": {"cell_type": 3},
    },
}

with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = os.path.join(tmp, "test.pt")
    save_ckpt(model, None, 1, config, ckpt_path)

    loaded_model, loaded_config = load_model(ckpt_path, device="cpu")
    data_meta = loaded_config.get("data_meta", {})
    assert data_meta["pert_to_id"] == {"control": 0, "A_KO": 1, "B_KO": 2}
    assert data_meta["selected_var_names"] == [f"gene_{i}" for i in range(G)]
    assert data_meta["input_dim"] == G
    assert data_meta["evidence_dim"] == 8
    assert data_meta["cov_dims"] == {"cell_type": 3}

    model_cfg = loaded_config.get("model", {})
    assert model_cfg["num_perts"] == 5
    assert model_cfg["evidence_dim"] == 8
    assert model_cfg["cov_dims"] == {"cell_type": 3}

print("  1) checkpoint vocab/dims/gene_order preserved OK")
print("  2) load_model restores all metadata OK")
print("ALL OK")
