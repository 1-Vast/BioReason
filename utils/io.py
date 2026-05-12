"""Atomic I/O utilities.

atomic_torch_save: write to .tmp then os.replace to prevent corruption.
"""

import torch
import os
from pathlib import Path


def atomic_torch_save(obj, path, _already_checked_parent=False):
    """Write checkpoint atomically: tmp file → os.replace.

    If interrupted mid-write, the original checkpoint is preserved.
    """
    path = Path(path)
    if not _already_checked_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def safe_torch_load(path, map_location="cpu"):
    """Load checkpoint. Returns None if file missing or corrupted."""
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        return None
