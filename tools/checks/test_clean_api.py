"""Test cleaned public API after dead-code removal."""
print("--- test_clean_api ---")
import inspect
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from models.latent import LatentBlock
from models.reason import BioReason, Reasoner

try:
    from models.latent import CrossBlock  # noqa: F401
    assert False, "CrossBlock should not be importable"
except ImportError:
    print("  1) CrossBlock removed OK")

try:
    from models.base import LayerNormBlock  # noqa: F401
    assert False, "LayerNormBlock should not be importable"
except ImportError:
    print("  2) LayerNormBlock removed OK")

try:
    LatentBlock(8, mode="gru")
    assert False, "gru mode should raise"
except ValueError:
    print("  3) gru mode rejected OK")

sig = inspect.signature(Reasoner.forward)
assert "return_all" not in sig.parameters
try:
    r = Reasoner(dim=8, steps=1, hidden=16, heads=2, dropout=0.0)
    r(torch.randn(2, 8), torch.randn(2, 8), return_all=True)
    assert False, "return_all should not be accepted"
except TypeError:
    print("  4) return_all removed OK")

model = BioReason(input_dim=10, dim=8, hidden=16, steps=1, heads=2,
                  dropout=0.0, pert_mode="id", num_perts=3, evidence_dim=4)
out = model(torch.randn(2, 10), torch.randint(0, 3, (2,)), evidence=torch.randn(2, 4))
assert out["latent"].shape == (2, 8)
assert out["pred"].shape == (2, 10)
print("  5) BioReason final latent output OK")

print("ALL OK")
