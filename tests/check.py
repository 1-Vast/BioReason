"""BioReason verification: import, forward, loss checks.

Usage:
  python tests/check.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_import():
    """Verify all key classes import cleanly."""
    print("--- import ---")
    from models import BioReason, Reasoner, BioLoss
    from models import CellEncoder, PertEncoder, ExprDecoder
    from models import EvidenceGate, ReasonStep
    from models import MLP, ResidualBlock
    print("  OK")


def check_forward():
    """Verify BioReason forward with toy data (no evidence / with evidence)."""
    print("--- forward ---")
    import torch
    from models.reason import BioReason

    B, G = 4, 100
    dim, hidden = 32, 64

    x = torch.randn(B, G)
    pert = torch.randint(0, 10, (B,))

    model = BioReason(input_dim=G, dim=dim, hidden=hidden, steps=2,
                       heads=2, dropout=0.0, pert_mode="id", num_perts=10)

    # 1) no evidence (inference)
    out = model(x, pert, evidence=None, return_latent=True)
    assert out["pred"].shape == (B, G)
    assert out["delta"].shape == (B, G)
    assert out["latent"].shape == (B, dim)
    assert out.get("evidence_pred") is None
    print("  no evidence   pred=%s delta=%s latent=%s" %
          (str(tuple(out["pred"].shape)), str(tuple(out["delta"].shape)),
           str(tuple(out["latent"].shape))))

    # 2) with evidence (training)
    ev = torch.randn(B, dim)
    out2 = model(x, pert, evidence=ev, return_latent=True)
    assert out2["evidence_pred"] is not None
    print("  with evidence evidence_pred present")

    # 3) detach_latent
    out3 = model(x, pert, detach_latent=True)
    assert out3["pred"].shape == (B, G)
    print("  detach_latent OK")

    # 4) encode + predict
    z = model.encode(x, pert)
    p = model.predict(x, z)
    assert p.shape == (B, G)
    print("  encode+predict pred=%s" % str(tuple(p.shape)))

    print("  OK")


def check_loss():
    """Verify BioLoss returns correct dict at each stage."""
    print("--- loss ---")
    import torch
    from models.loss import BioLoss

    B, G = 4, 100
    dim = 32
    loss_fn = BioLoss()

    x = torch.randn(B, G)
    y = x + torch.randn(B, G) * 0.1
    out = {"pred": y + torch.randn(B, G) * 0.01,
           "delta": torch.randn(B, G) * 0.01,
           "latent": torch.randn(B, dim),
           "evidence_pred": torch.randn(B, dim)}
    batch = {"x": x, "y": y, "evidence": torch.randn(B, dim)}

    for stage, keys in [(1, ["expr", "delta", "deg"]),
                         (2, ["evidence", "mmd"]),
                         (3, ["latent"])]:
        b = dict(batch)
        if stage == 3:
            b["target_latent"] = torch.randn(B, dim)
        L = loss_fn(out, b, stage=stage)
        for k in keys:
            assert L[k].item() >= 0
        print("  stage %d loss=%.4f" % (stage, L["loss"].item()))

    # minimal batch — should not crash
    L_min = loss_fn({"pred": x, "delta": x * 0}, {"x": x, "y": x}, stage=1)
    print("  minimal batch loss=%.4f" % L_min["loss"].item())

    print("  OK")


if __name__ == "__main__":
    try:
        check_import()
        check_forward()
        check_loss()
        print("\nALL CHECKS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
