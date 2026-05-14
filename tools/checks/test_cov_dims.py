"""Test: cov_dims passed to BioReason, CovEncoder produces cov_emb."""
print("--- test_cov_dims ---")
import torch, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from models.reason import BioReason

B, G, D = 4, 20, 16
cov_dims = {"cell_type": 3, "batch": 2}

model = BioReason(input_dim=G, dim=D, hidden=32, steps=2, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5, cov_dims=cov_dims)

x = torch.randn(B, G)
pert = torch.randint(0, 5, (B,))
cov = {"cell_type": torch.randint(0, 3, (B,)), "batch": torch.randint(0, 2, (B,))}

# Forward with covariates
out = model(x, pert, cov=cov)
assert out["pred"].shape == (B, G)
print("  1) forward with cov_dims OK")

# Forward without covariates 鈫?CovEncoder returns None
out2 = model(x, pert, cov=None)
assert out2["pred"].shape == (B, G)
print("  2) forward without cov OK")

# No cov_dims model
model_no_cov = BioReason(input_dim=G, dim=D, hidden=32, steps=2, heads=2, dropout=0.0,
                          pert_mode="id", num_perts=5, cov_dims={})
out3 = model_no_cov(x, pert, cov=None)
assert out3["pred"].shape == (B, G)
print("  3) empty cov_dims OK")

print("ALL OK")
