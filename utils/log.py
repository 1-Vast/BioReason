"""Lightweight logging and progress bar utilities.

Uses tqdm for progress bars. All output concise — no per-step printing.
"""

import logging
import time
import sys


def setup_logger(name="BioReason", level=None):
    """Configure root logger with clean format."""
    if level is None:
        level = logging.INFO
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def short_float(x, width=4):
    """Format float to fixed width."""
    if abs(x) < 1e-10:
        return "0.0000"
    if abs(x) >= 10000:
        return f"{x:.2e}"
    return f"{x:.{width}f}"


def format_loss(loss_dict, prefix=""):
    """Format loss dict to compact string."""
    parts = []
    for k in ["loss", "expr", "delta", "deg", "latent", "evidence", "mmd"]:
        v = loss_dict.get(k)
        if v is not None and abs(v) > 1e-10:
            parts.append(f"{prefix}{k}={short_float(v)}")
    return " ".join(parts)


def format_speed(samples, elapsed):
    """Format throughput: cells/sec or samples/sec."""
    if elapsed < 0.001:
        return "--- cells/s"
    rate = samples / elapsed
    return f"{int(rate)} cells/s"


def make_bar(iterable, enable=True, desc="", total=None):
    """Create tqdm bar or return iterable unchanged."""
    if not enable:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=False, ncols=120)
    except ImportError:
        return iterable


def update_bar_postfix(bar, loss_dict=None, extra=None):
    """Update tqdm postfix with loss + speed info."""
    if bar is None or not hasattr(bar, "set_postfix"):
        return
    info = {}
    if loss_dict:
        info["loss"] = short_float(loss_dict.get("loss", 0))
    if extra:
        info.update(extra)
    bar.set_postfix(info, refresh=False)
