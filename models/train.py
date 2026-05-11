"""Training loop for BioReason.

Supports: 3-stage training, GPU/AMP, gradient clipping, checkpoint save/resume.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device,
                stage=1, grad_clip=1.0, use_amp=True):
    """Train one epoch. Returns dict of average losses."""
    model.train()
    losses_sum = {}
    n_batches = len(dataloader)

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pert = batch["pert"].to(device)
        cov = {k: v.to(device) for k, v in batch.get("cov", {}).items()}
        evidence = batch.get("evidence")
        if evidence is not None:
            evidence = evidence.to(device)

        with autocast(enabled=use_amp):
            out = model(x, pert, cov=cov, evidence=evidence)
            losses = loss_fn(out, {"x": x, "y": y, "evidence": evidence}, stage=stage)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(losses["loss"]).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in losses.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + v.item()

    for k in losses_sum:
        losses_sum[k] /= n_batches
    return losses_sum


def validate(model, dataloader, loss_fn, device, stage=1, use_amp=True):
    """Validate. Returns dict of average losses."""
    model.eval()
    losses_sum = {}
    n_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pert = batch["pert"].to(device)
            cov = {k: v.to(device) for k, v in batch.get("cov", {}).items()}
            evidence = batch.get("evidence")
            if evidence is not None:
                evidence = evidence.to(device)

            with autocast(enabled=use_amp):
                out = model(x, pert, cov=cov, evidence=evidence)
                losses = loss_fn(out, {"x": x, "y": y, "evidence": evidence}, stage=stage)

            for k, v in losses.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + v.item()

    for k in losses_sum:
        losses_sum[k] /= n_batches
    return losses_sum


def save_ckpt(model, optimizer, epoch, config, path):
    """Save checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def load_ckpt(path, model=None, optimizer=None, device="cpu"):
    """Load checkpoint. Returns ckpt dict."""
    ckpt = torch.load(path, map_location=device)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def save_target_latent(model, dataloader, device, save_path, use_amp=True):
    """Save teacher latent states for stage 3 alignment."""
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            pert = batch["pert"].to(device)
            cov = {k: v.to(device) for k, v in batch.get("cov", {}).items()}
            evidence = batch.get("evidence")
            if evidence is not None:
                evidence = evidence.to(device)
            with autocast(enabled=use_amp):
                z = model.encode(x, pert, cov=cov)
            latents.append(z.cpu())
    all_latents = torch.cat(latents, dim=0)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": all_latents}, save_path)
    logger.info(f"Target latent saved: {save_path}  shape={all_latents.shape}")


def train_model(model, train_loader, val_loader, config, loss_fn=None):
    """Full training loop.

    config: dict with stage, epochs, lr, weight_decay, amp, grad_clip, device, save_dir
    """
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if loss_fn is None:
        from .loss import BioLoss
        loss_fn = BioLoss(config.get("loss", {}))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 5e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scaler = GradScaler(enabled=config.get("amp", True))
    stage = config.get("stage", 1)
    epochs = config.get("epochs", 100)
    grad_clip = config.get("grad_clip", 1.0)
    use_amp = config.get("amp", True)
    save_dir = Path(config.get("save_dir", "output"))

    # Stage 2: save teacher latent
    if stage == 2:
        save_target_latent(model, train_loader, device, save_dir / "stage2" / "target_latent.pt", use_amp)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_losses = train_epoch(model, train_loader, loss_fn, optimizer,
                                    scaler, device, stage=stage,
                                    grad_clip=grad_clip, use_amp=use_amp)
        val_losses = {}
        if val_loader:
            val_losses = validate(model, val_loader, loss_fn, device,
                                   stage=stage, use_amp=use_amp)
        elapsed = time.time() - t0

        log_parts = [f"Epoch {epoch}/{epochs} | {elapsed:.1f}s"]
        for name in ["loss", "expr", "delta", "deg", "latent", "evidence", "mmd"]:
            tv = train_losses.get(name)
            if tv is not None and tv != 0:
                log_parts.append(f"tr_{name}={tv:.4f}")
            vv = val_losses.get(name)
            if vv is not None and vv != 0:
                log_parts.append(f"vl_{name}={vv:.4f}")
        logger.info(" | ".join(log_parts))

        if epoch % 10 == 0 or epoch == epochs:
            save_ckpt(model, optimizer, epoch, config,
                       save_dir / f"stage{stage}" / f"ckpt_epoch{epoch}.pt")

    final_path = save_dir / f"stage{stage}" / "model.pt"
    save_ckpt(model, optimizer, epochs, config, final_path)
    logger.info(f"Training complete. Model saved: {final_path}")

    return model
