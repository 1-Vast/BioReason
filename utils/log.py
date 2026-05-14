"""Lightweight logging and progress utilities."""

import logging
import sys


def setup_logger(name="BioReason", level=None):
    """Configure project logger with compact plain-text output."""
    if level is None:
        level = logging.INFO
    fmt = logging.Formatter("%(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    return logger


def short_float(x, width=4):
    """Format float to fixed width."""
    if abs(x) < 1e-10:
        return "0.0000"
    if abs(x) >= 10000:
        return f"{x:.2e}"
    return f"{x:.{width}f}"


def format_loss(loss_dict, prefix=""):
    """Format only key training signals."""
    parts = []
    for k in ["loss", "deg", "latent", "evidence", "evi_gain", "z_shift",
              "evi_rec", "latent_align", "mmd"]:
        v = loss_dict.get(k)
        if v is not None and abs(v) > 1e-10:
            parts.append(f"{prefix}{k}={short_float(v)}")
    return " ".join(parts)


def format_speed(samples, elapsed):
    """Format throughput as cells/sec."""
    if elapsed < 0.001:
        return "--- cells/s"
    return f"{int(samples / elapsed)} cells/s"


def make_bar(iterable, enable=True, desc="", total=None):
    """Create a quiet ASCII tqdm bar only for real interactive terminals."""
    if not enable or not sys.stderr.isatty():
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=False, ncols=96,
                    ascii=True, dynamic_ncols=False, mininterval=1.0)
    except ImportError:
        return iterable


def update_bar_postfix(bar, loss_dict=None, extra=None):
    """Update tqdm postfix with compact metrics."""
    if bar is None or not hasattr(bar, "set_postfix"):
        return
    info = {}
    if loss_dict:
        info["loss"] = short_float(loss_dict.get("loss", 0))
    if extra:
        info.update(extra)
    bar.set_postfix(info, refresh=False)
