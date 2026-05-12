"""Unified GPU/device utilities.

Handles: device selection, non_blocking transfer, AMP autocast,
GradScaler, GPU memory summary. No heavy dependencies.
"""

import torch


def cuda_available():
    return torch.cuda.is_available()


def get_device(device_str="cuda"):
    """Resolve device. Falls back to CPU if CUDA unavailable."""
    if device_str == "cuda" and not cuda_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def move_to_device(batch, device, non_blocking=True):
    """Recursively move tensors to device. Preserves None, dict, list structure."""
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cpu":
        non_blocking = False

    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    elif batch is None:
        return None
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device, non_blocking) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device, non_blocking) for v in batch)
    return batch


def get_autocast(device, enabled=True):
    """Return autocast context manager. Uses torch.amp with fallback."""
    if not enabled or not cuda_available():
        from contextlib import nullcontext
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def get_scaler(device, enabled=True):
    """Return GradScaler if CUDA+AMP, else disabled."""
    if not enabled or not cuda_available():
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def sync_if_cuda(device):
    """Synchronize if CUDA device."""
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)
    elif isinstance(device, str) and device == "cuda" and cuda_available():
        torch.cuda.synchronize()


def format_bytes(num):
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def gpu_summary():
    """Return compact GPU info string."""
    if not cuda_available():
        return "GPU: CPU mode"
    try:
        props = torch.cuda.get_device_properties(0)
        alloc = torch.cuda.memory_allocated(0)
        total = props.total_memory
        name = props.name
        return f"GPU: {name} | mem {format_bytes(alloc)}/{format_bytes(total)}"
    except Exception:
        return f"GPU: {torch.cuda.get_device_name(0)}"


def memory_summary(device=None):
    """Return allocated and reserved memory."""
    if not cuda_available():
        return {"allocated": 0, "reserved": 0}
    idx = device.index if isinstance(device, torch.device) else 0
    return {
        "allocated": torch.cuda.memory_allocated(idx),
        "reserved": torch.cuda.memory_reserved(idx),
    }


def gpu_mem_gb():
    """Return allocated GPU memory in GB."""
    if not cuda_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / (1024 ** 3)


def tensor_device_summary(batch):
    """Return compact dict of key tensor shapes and devices."""
    if not isinstance(batch, dict):
        return {"error": "not a dict"}
    summary = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            summary[k] = f"{str(v.device)} {list(v.shape)}"
        elif isinstance(v, dict):
            summary[k] = {ck: f"{cv.device} {list(cv.shape)}" for ck, cv in v.items() if isinstance(cv, torch.Tensor)}
        elif v is None:
            summary[k] = None
        elif isinstance(v, list):
            summary[k] = f"list[{len(v)}]"
    return summary
