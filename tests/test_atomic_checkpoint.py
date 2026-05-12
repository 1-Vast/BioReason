"""Test: atomic checkpoint save/resume, tmp file cleanup."""
print("--- test_atomic_checkpoint ---")
import torch, sys, os, tempfile
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from models.reason import BioReason
from utils.io import atomic_torch_save, safe_torch_load

G, D = 20, 16
model = BioReason(input_dim=G, dim=D, hidden=32, steps=1, heads=2, dropout=0.0,
                   pert_mode="id", num_perts=5)
_ = model(torch.randn(2, G), torch.randint(0, 5, (2,)))

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model.pt")
    ckpt = {"model_state_dict": model.state_dict(), "epoch": 1, "config": {}}

    # Atomic save
    atomic_torch_save(ckpt, path)
    assert os.path.isfile(path), "checkpoint not created"
    assert not os.path.isfile(path + ".tmp"), "tmp file should be gone"
    print("  1) atomic save OK, no residual tmp")

    # Load
    loaded = safe_torch_load(path)
    assert loaded is not None
    assert loaded["epoch"] == 1
    print("  2) safe_torch_load OK")

    # Corrupted file
    with open(os.path.join(tmp, "bad.pt"), "w") as f:
        f.write("garbage")
    bad = safe_torch_load(os.path.join(tmp, "bad.pt"))
    assert bad is None
    print("  3) corrupted file returns None OK")

    # Tmp exists -> overwrite works
    tmp_path = path + ".tmp"
    torch.save({"test": True}, tmp_path)
    assert os.path.isfile(tmp_path)
    atomic_torch_save(ckpt, path)
    assert not os.path.isfile(tmp_path)
    assert safe_torch_load(path) is not None
    print("  4) overwrite with existing tmp OK")

print("ALL OK")
