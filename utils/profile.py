"""Optional profiling tools for data loading and training steps.

Uses torch.cuda.Event for GPU timing when available.
No nvidia-smi dependency. Never called during normal training.
"""

import time
import torch
from .device import cuda_available, move_to_device, gpu_mem_gb


def _timer():
    """Return (start_fn, elapsed_fn) pair. Uses CUDA events if available."""
    if cuda_available():
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        return (lambda: start_ev.record(),
                lambda: (end_ev.record(), torch.cuda.synchronize(), start_ev.elapsed_time(end_ev) / 1000)[2])
    else:
        t = [0.0]
        return (lambda: t.__setitem__(0, time.perf_counter()),
                lambda: time.perf_counter() - t[0])


def profile_loader(loader, device, batches=20):
    """Profile DataLoader: measure data wait time and H2D transfer."""
    print(f"\nProfiling DataLoader ({batches} batches)...")
    stats = {"data_wait": [], "h2d_time": [], "batch_shapes": []}
    start_tick, tick = _timer()
    for i, batch in enumerate(loader):
        start_tick()
        if i >= batches:
            break
        data_wait = tick()
        stats["data_wait"].append(data_wait)
        stats["batch_shapes"].append({k: v.shape if isinstance(v, torch.Tensor) else type(v).__name__
                                       for k, v in batch.items() if k not in ("meta", "pert_str", "cov")})

        start_tick()
        _ = move_to_device(batch, device, non_blocking=True)
        h2d = tick()
        stats["h2d_time"].append(h2d)

    print(f"  avg data wait: {sum(stats['data_wait'])/len(stats['data_wait'])*1000:.1f} ms")
    print(f"  avg H2D transfer: {sum(stats['h2d_time'])/len(stats['h2d_time'])*1000:.1f} ms")
    return stats


def profile_train_step(model, loader, loss_fn, optimizer, device, stage=1, batches=20):
    """Profile a few training steps: forward, backward, step times."""
    print(f"\nProfiling train steps ({batches} batches)...")
    from .device import get_autocast, get_scaler
    scaler = get_scaler(device, True)
    model.train()
    stats = {"forward": [], "backward": [], "step": [], "mem": []}
    start_tick, tick = _timer()

    for i, batch in enumerate(loader):
        if i >= batches:
            break
        b = move_to_device(batch, device)
        evidence = b.get("evidence") if stage == 2 else None
        optimizer.zero_grad(set_to_none=True)

        start_tick()
        with get_autocast(device, True):
            out = model(b["x"], b["pert"], cov=b.get("cov"), evidence=evidence, return_latent=True)
            L = loss_fn(out, {"x": b["x"], "y": b["y"], "evidence": evidence}, stage=stage)
        forward_t = tick()
        stats["forward"].append(forward_t)

        start_tick()
        scaler.scale(L["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        step_t = tick()
        stats["step"].append(step_t)
        stats["backward"].append(step_t - forward_t)
        stats["mem"].append(gpu_mem_gb())

    avg_fwd = sum(stats["forward"]) / len(stats["forward"]) * 1000
    avg_bwd = sum(stats["backward"]) / len(stats["backward"]) * 1000
    avg_step = sum(stats["step"]) / len(stats["step"]) * 1000
    avg_mem = sum(stats["mem"]) / len(stats["mem"])
    print(f"  avg forward: {avg_fwd:.1f} ms | backward: {avg_bwd:.1f} ms | step: {avg_step:.1f} ms")
    print(f"  avg GPU mem: {avg_mem:.2f} GB")
    return stats


def suggest_loader_settings(loader_stats, step_stats=None):
    """Suggest DataLoader settings based on profiling results."""
    suggestions = []
    avg_data = sum(loader_stats["data_wait"]) / len(loader_stats["data_wait"]) * 1000
    avg_h2d = sum(loader_stats["h2d_time"]) / len(loader_stats["h2d_time"]) * 1000

    if step_stats:
        avg_step = sum(step_stats["step"]) / len(step_stats["step"]) * 1000
        if avg_data / avg_step > 0.3:
            suggestions.append("Data wait > 30% of step time → increase num_workers, enable pin_memory")
    else:
        if avg_data > 100:
            suggestions.append("Data wait > 100ms → increase num_workers, enable pin_memory")

    if avg_h2d > 10:
        suggestions.append("H2D transfer > 10ms → enable pin_memory=True, non_blocking=True")

    return suggestions
