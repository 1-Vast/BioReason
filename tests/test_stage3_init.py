"""Test Stage 3 initialization policy."""
print("--- test_stage3_init ---")
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.reason import BioReason
from models.train import initialize_stage3_model


def make_model():
    return BioReason(input_dim=12, dim=8, hidden=16, steps=1, heads=2,
                     dropout=0.0, pert_mode="id", num_perts=4, evidence_dim=5)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    stage1_path = tmp / "stage1.pt"
    src = make_model()
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.123)
    torch.save({"model_state_dict": src.state_dict(), "config": {"stage": 1}}, stage1_path)

    dst = make_model()
    before = dst.reasoner.init_proj.weight.detach().clone()
    loaded = initialize_stage3_model(dst, {
        "stage": 3,
        "stage3_init": "stage1",
        "stage1_ckpt": str(stage1_path),
    }, device="cpu")
    after = dst.reasoner.init_proj.weight.detach()
    assert loaded == "stage1"
    assert not torch.allclose(before, after)
    assert torch.allclose(after, src.reasoner.init_proj.weight.detach())
    print("  1) stage1 checkpoint init OK")

    missing = initialize_stage3_model(make_model(), {
        "stage": 3,
        "stage3_init": "stage1",
        "stage1_ckpt": str(tmp / "missing.pt"),
    }, device="cpu")
    assert missing is None
    print("  2) missing stage1 checkpoint warns/skips OK")

    stage2_path = tmp / "stage2.pt"
    torch.save({"model_state_dict": src.state_dict(), "config": {"stage": 2}}, stage2_path)
    loaded = initialize_stage3_model(make_model(), {
        "stage": 3,
        "stage3_init": "stage2",
        "stage2_ckpt": str(stage2_path),
    }, device="cpu")
    assert loaded == "stage2"
    print("  3) stage2 ablation init OK")

print("ALL OK")
