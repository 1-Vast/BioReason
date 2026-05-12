"""Training loop for BioReason — GPU-optimized with tqdm progress.

Stages 1-3. Non-blocking GPU transfer, AMP, per-epoch checkpoint only.
No per-step .cpu()/.numpy()/.item() spam.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging, time, math

logger = logging.getLogger(__name__)


def move_batch(batch, device, non_blocking=True):
    """Move batch to GPU. Lightweight wrapper over utils.device."""
    from utils.device import move_to_device
    return move_to_device(batch, device, non_blocking=non_blocking)


def prepare_stage_batch(batch, stage):
    if stage == 2:
        return batch.get("evidence"), None, batch.get("target_latent_mask")
    elif stage == 3:
        return None, batch.get("target_latent"), batch.get("target_latent_mask")
    return None, None, None


def attach_target_latent(dataset, path):
    dataset.load_target_latent(path)
    n_valid = int(dataset.target_latent_mask.sum().item()) if dataset.target_latent_mask is not None else 0
    logger.info(f"Target latent loaded: {n_valid}/{len(dataset)} valid")
    return dataset


def export_target_latent(model, dataloader, device, save_path, use_amp=True, non_blocking=True, progress=True):
    from utils.log import make_bar, format_speed
    model.eval()
    latents, indices_list, pert_list = [], [], []
    bar = make_bar(dataloader, enable=progress, desc="Export latent")
    t0 = time.perf_counter()
    n_cells = 0
    with torch.no_grad():
        for batch in bar:
            b = move_batch(batch, device, non_blocking=non_blocking)
            with torch.amp.autocast("cuda", enabled=use_amp) if torch.cuda.is_available() else __import__("contextlib").nullcontext():
                z = model.forward_latent(b["x"], b["pert"], cov=b.get("cov"), evidence=b.get("evidence"))
            latents.append(z.cpu())
            pert_list.append(b["pert"].cpu())
            for m in b.get("meta", []):
                indices_list.append(m.get("idx", len(indices_list)))
            n_cells += z.size(0)
    elapsed = time.perf_counter() - t0
    all_z = torch.cat(latents, dim=0)
    all_pert = torch.cat(pert_list, dim=0)
    all_idx = torch.tensor(indices_list, dtype=torch.long)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": all_z, "indices": all_idx, "pert": all_pert,
                "mask": torch.ones(len(all_idx), dtype=torch.bool)}, save_path)
    logger.info(f"Target latent saved: {save_path}  shape={all_z.shape}  {format_speed(n_cells, elapsed)}")


# ── Epoch training ───────────────────────────────────────────────

def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device, stage=1,
                grad_clip=1.0, use_amp=True, non_blocking=True, progress=True,
                log_every=20, profile=False):
    from utils.log import make_bar, update_bar_postfix, short_float
    from utils.device import gpu_mem_gb, get_autocast

    model.train()
    n_batches = len(dataloader)
    loss_sum = {k: 0.0 for k in ["loss", "expr", "delta", "deg", "latent", "evidence", "mmd"]}
    n_samples = 0
    t_data = 0.0; t_step_start = time.perf_counter()

    bar = make_bar(dataloader, enable=progress, desc=f"Train stg{stage}", total=n_batches)

    for bi, batch_cpu in enumerate(bar):
        t_data_end = time.perf_counter()
        t_data += t_data_end - (t_step_start if bi == 0 else t_step_start)

        b = move_batch(batch_cpu, device, non_blocking=non_blocking)
        evidence, target_latent, tmask = prepare_stage_batch(b, stage)
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = get_autocast(device, use_amp) if use_amp else __import__("contextlib").nullcontext()
        with autocast_ctx:
            out = model(b["x"], b["pert"], cov=b.get("cov"), evidence=evidence, return_latent=True)
            L = loss_fn(out, {"x": b["x"], "y": b["y"], "evidence": evidence,
                               "target_latent": target_latent, "target_latent_mask": tmask}, stage=stage)

        scaler.scale(L["loss"]).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()

        # Accumulate with detach only (no .item() every step)
        n_samples += b["x"].size(0)
        for k in loss_sum:
            v = L.get(k)
            if v is not None and v.numel() > 0:
                loss_sum[k] += v.detach().item() if isinstance(v, torch.Tensor) else float(v)

        # Update progress bar at log_every intervals
        if progress and bar is not None and (bi + 1) % log_every == 0:
            avg_loss = loss_sum["loss"] / (bi + 1)
            update_bar_postfix(bar, loss_dict={"loss": avg_loss},
                               extra={"mem": f"{gpu_mem_gb():.1f}GB"})

        t_step_start = time.perf_counter()

    # Epoch stats
    n = max(n_batches, 1)
    stats = {k: v / n for k, v in loss_sum.items()}
    stats["data_time"] = t_data
    stats["samples"] = n_samples
    return stats


def validate(model, dataloader, loss_fn, device, stage=1, use_amp=True, non_blocking=True, progress=True):
    from utils.log import make_bar, update_bar_postfix, short_float
    from utils.device import gpu_mem_gb

    model.eval()
    n_batches = len(dataloader)
    loss_sum = {k: 0.0 for k in ["loss", "expr", "delta", "deg", "latent", "evidence", "mmd"]}
    bar = make_bar(dataloader, enable=progress, desc=f"Val   stg{stage}", total=n_batches)

    with torch.no_grad():
        for bi, batch_cpu in enumerate(bar):
            b = move_batch(batch_cpu, device, non_blocking=non_blocking)
            evidence, target_latent, tmask = prepare_stage_batch(b, stage)
            with torch.amp.autocast("cuda", enabled=use_amp) if torch.cuda.is_available() else __import__("contextlib").nullcontext():
                out = model(b["x"], b["pert"], cov=b.get("cov"), evidence=evidence, return_latent=True)
                L = loss_fn(out, {"x": b["x"], "y": b["y"], "evidence": evidence,
                                   "target_latent": target_latent, "target_latent_mask": tmask}, stage=stage)
            for k in loss_sum:
                v = L.get(k)
                if v is not None and v.numel() > 0:
                    loss_sum[k] += v.detach().item()

            if progress and (bi + 1) % 20 == 0:
                update_bar_postfix(bar, loss_dict={"loss": loss_sum["loss"] / (bi + 1)},
                                   extra={"mem": f"{gpu_mem_gb():.1f}GB"})

    n = max(n_batches, 1)
    return {k: v / n for k, v in loss_sum.items()}


# ── Checkpoint ───────────────────────────────────────────────────

def save_ckpt(model, optimizer, epoch, config, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_ckpt(path, model=None, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if model is not None: model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ── Full training ────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, config, loss_fn=None):
    from utils.device import get_device, get_scaler, gpu_summary, gpu_mem_gb, sync_if_cuda
    from utils.log import format_loss, format_speed, short_float

    device = get_device(config.get("device", "cuda"))
    model = model.to(device)

    if loss_fn is None:
        from .loss import BioLoss
        loss_fn = BioLoss(config.get("loss", {}))

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.get("lr", 5e-4),
                                   weight_decay=config.get("weight_decay", 1e-4))
    scaler = get_scaler(device, config.get("amp", True))
    stage = config.get("stage", 1)
    epochs = config.get("epochs", 100)
    grad_clip = config.get("grad_clip", 1.0)
    use_amp = config.get("amp", True)
    non_blocking = config.get("non_blocking", True)
    progress = config.get("progress", True)
    log_every = config.get("log_every", 20)
    save_dir = Path(config.get("save_dir", "output"))

    # torch.compile (optional)
    if config.get("compile", False):
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    logger.info(f"BioReason train | stage={stage} | device={device} | amp={use_amp} | epochs={epochs}")
    logger.info(gpu_summary())

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        sync_if_cuda(device)

        train_stats = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device,
                                   stage=stage, grad_clip=grad_clip, use_amp=use_amp,
                                   non_blocking=non_blocking, progress=progress, log_every=log_every)
        val_stats = validate(model, val_loader, loss_fn, device, stage=stage,
                              use_amp=use_amp, non_blocking=non_blocking,
                              progress=progress) if val_loader else {}
        elapsed = time.perf_counter() - t0
        sync_if_cuda(device)

        # Epoch summary line
        parts = [f"Epoch {epoch:03d}/{epochs}"]
        parts.append(f"tr {format_loss(train_stats)}")
        if val_stats:
            parts.append(f"vl {format_loss(val_stats, prefix='vl_')}")
        parts.append(format_speed(train_stats.get("samples", 0), elapsed))
        parts.append(f"gpu {gpu_mem_gb():.1f}GB")
        logger.info(" | ".join(parts))

        # Checkpoint (epoch end only)
        if epoch % 10 == 0 or epoch == epochs:
            save_ckpt(model, optimizer, epoch, config, save_dir / f"stage{stage}" / f"ckpt_epoch{epoch}.pt")

        # Optional empty_cache
        if config.get("empty_cache", False) and device.type == "cuda":
            torch.cuda.empty_cache()

    final_path = save_dir / f"stage{stage}" / "model.pt"
    save_ckpt(model, optimizer, epochs, config, final_path)

    if stage == 2:
        export_path = save_dir / "stage2" / "target_latent.pt"
        export_target_latent(model, train_loader, device, export_path, use_amp=use_amp,
                              non_blocking=non_blocking, progress=progress)

    logger.info(f"Training done → {final_path}")
    return model
