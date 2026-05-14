"""Test: MMD under float16 input does not return NaN/Inf."""
print("--- test_mmd_amp_stability ---")
import torch, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from models.loss import mmd_loss

B, D = 64, 200
# float16 with values ~10 to trigger overflow risk
x = torch.randn(B, D, dtype=torch.float16) * 10
y = torch.randn(B, D, dtype=torch.float16) * 10

loss = mmd_loss(x, y, max_samples=32)
assert torch.isfinite(loss), f"MMD loss is not finite: {loss}"
print(f"  1) float16 input, float32 compute: loss={loss.item():.4f} OK")

# Zero input
x0 = torch.zeros(B, D, dtype=torch.float32)
y0 = torch.zeros(B, D, dtype=torch.float32)
loss0 = mmd_loss(x0, y0)
assert torch.isfinite(loss0)
print(f"  2) zero input: loss={loss0.item():.6f} OK")

# Large input
xL = torch.randn(B, D, dtype=torch.float32) * 100
yL = torch.randn(B, D, dtype=torch.float32) * 100
lossL = mmd_loss(xL, yL, max_samples=32)
assert torch.isfinite(lossL)
print(f"  3) large input (x100): loss={lossL.item():.4f} OK")

print("ALL OK")
