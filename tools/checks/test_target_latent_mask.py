"""Test: target_latent_mask prevents zero-latent alignment."""
print("--- test_target_latent_mask ---")
import torch, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from models.loss import BioLoss

B, G, D = 6, 100, 32
loss_fn = BioLoss({"expr": 1.0, "latent": 1.0, "top_deg": 50, "latent_metric": "cosine"})

x = torch.randn(B, G)
y = x + torch.randn(B, G) * 0.5
z = torch.randn(B, D)
target = torch.randn(B, D)

# All valid mask 鈫?latent loss > 0
out_all = {"pred": y, "delta": y - x, "latent": z, "evidence_pred": None, "target_latent": None}
batch_all = {"x": x, "y": y, "target_latent": target, "target_latent_mask": torch.ones(B, dtype=torch.bool)}
L_all = loss_fn(out_all, batch_all, stage=3)
assert L_all["latent"].item() > 0
print(f"  all valid mask: latent={L_all['latent'].item():.4f}")

# All invalid mask 鈫?latent loss = 0
batch_none = {"x": x, "y": y, "target_latent": target, "target_latent_mask": torch.zeros(B, dtype=torch.bool)}
L_none = loss_fn(out_all, batch_none, stage=3)
assert abs(L_none["latent"].item()) < 1e-10
print(f"  all invalid mask: latent={L_none['latent'].item():.4f}")

# Mixed mask 鈫?only valid ones contribute
mask_mix = torch.tensor([True, True, False, False, False, False])
batch_mix = {"x": x, "y": y, "target_latent": target, "target_latent_mask": mask_mix}
L_mix = loss_fn(out_all, batch_mix, stage=3)
assert L_mix["latent"].item() > 0
print(f"  mixed mask (2/6 valid): latent={L_mix['latent'].item():.4f}")

# No mask at all 鈫?all used (backward compat)
batch_no_mask = {"x": x, "y": y, "target_latent": target}
L_nm = loss_fn(out_all, batch_no_mask, stage=3)
assert L_nm["latent"].item() > 0
print(f"  no mask (backward compat): latent={L_nm['latent'].item():.4f}")

print("ALL OK")
