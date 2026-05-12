"""Training loop for BioReason.

Stages:
  1: warm-up (no evidence, expr + delta + deg)
  2: evidence-guided (saves teacher latent AFTER training)
  3: evidence-free alignment (loads target latent, cosine alignment)

Exports: train_model, train_epoch, validate, save_ckpt, load_ckpt,
         export_target_latent, attach_target_latent, prepare_stage_batch,
         move_batch
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────

def move_batch(batch, device):
    """Move batch tensors to device. Handles evidence=None, cov dicts."""
    out = {}
    for k, v in batch.items():
        if k == "cov":
            out[k] = {ck: cv.to(device) for ck, cv in v.items()}
        elif k == "evidence":
            out[k] = v.to(device) if v is not None else None
        elif k == "target_latent":
            out[k] = v.to(device) if v is not None else None
        elif k == "meta" or k == "pert_str":
            out[k] = v
        else:
            out[k] = v.to(device)
    return out


def prepare_stage_batch(batch, stage):
    """Return (evidence, target_latent) appropriate for stage."""
    if stage == 2:
        return batch.get("evidence"), None
    elif stage == 3:
        return None, batch.get("target_latent")
    return None, None


def attach_target_latent(dataset, path):
    """Load target latent file into dataset for stage 3 alignment."""
    data = torch.load(path, map_location="cpu")
    indices = data.get("indices", torch.arange(len(data["latent"])))
    latent = data["latent"]
    n = max(indices.max().item() + 1, len(dataset))
    dataset.target_latents = torch.zeros(n, latent.shape[1])
    dataset.target_latents[indices] = latent
    logger.info(f"Attached target latent from {path}: {dataset.target_latents.shape}")
    return dataset


def export_target_latent(model, dataloader, device, save_path, use_amp=True):
    """Export evidence-guided latent states after stage 2 training."""
    model.eval()
    latents, indices_list, pert_list = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            b = move_batch(batch, device)
            evidence = b.get("evidence")
            with autocast(enabled=use_amp):
                z = model.forward_latent(b["x"], b["pert"], cov=b.get("cov"),
                                          evidence=evidence)
            latents.append(z.cpu())
            pert_list.append(b["pert"].cpu())
            for m in b.get("meta", []):
                indices_list.append(m.get("idx", len(indices_list)))
    all_z = torch.cat(latents, dim=0)
    all_pert = torch.cat(pert_list, dim=0)
    all_idx = torch.tensor(indices_list, dtype=torch.long)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": all_z, "indices": all_idx, "pert": all_pert}, save_path)
    logger.info(f"Target latent saved: {save_path}  shape={all_z.shape}")


# ── Training core ────────────────────────────────────────────────

def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device,
                stage=1, grad_clip=1.0, use_amp=True):
    model.train()
    losses_sum = {}
    n_batches = len(dataloader)
    for batch in dataloader:
        b = move_batch(batch, device)
        evidence, _ = prepare_stage_batch(b, stage)
        with autocast(enabled=use_amp):
            out = model(b["x"], b["pert"], cov=b.get("cov"),
                         evidence=evidence, return_latent=True)
            L = loss_fn(out, {"x": b["x"], "y": b["y"],
                               "evidence": evidence,
                               "target_latent": b.get("target_latent")},
                         stage=stage)
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(L["loss"]).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            L["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        for k, v in L.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + v.item()
    for k in losses_sum:
        losses_sum[k] /= n_batches
    return losses_sum


def validate(model, dataloader, loss_fn, device, stage=1, use_amp=True):
    model.eval()
    losses_sum = {}
    n_batches = len(dataloader)
    with torch.no_grad():
        for batch in dataloader:
            b = move_batch(batch, device)
            evidence, _ = prepare_stage_batch(b, stage)
            with autocast(enabled=use_amp):
                out = model(b["x"], b["pert"], cov=b.get("cov"),
                             evidence=evidence, return_latent=True)
                L = loss_fn(out, {"x": b["x"], "y": b["y"],
                                   "evidence": evidence,
                                   "target_latent": b.get("target_latent")},
                             stage=stage)
            for k, v in L.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + v.item()
    for k in losses_sum:
        losses_sum[k] /= n_batches
    return losses_sum


# ── Checkpoint ───────────────────────────────────────────────────

def save_ckpt(model, optimizer, epoch, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path)
    logger.info(f"Checkpoint: {path}")


def load_ckpt(path, model=None, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ── Full training ────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, config, loss_fn=None):
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if loss_fn is None:
        from .loss import BioLoss
        loss_fn = BioLoss(config.get("loss", {}))

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.get("lr", 5e-4),
                                   weight_decay=config.get("weight_decay", 1e-4))
    scaler = GradScaler(enabled=config.get("amp", True))
    stage = config.get("stage", 1)
    epochs = config.get("epochs", 100)
    grad_clip = config.get("grad_clip", 1.0)
    use_amp = config.get("amp", True)
    save_dir = Path(config.get("save_dir", "output"))
    latent_path = config.get("target_latent")

    # Stage 3: warn if no target latent
    if stage == 3:
        if latent_path and Path(latent_path).exists():
            logger.info(f"Stage 3: target latent loaded from {latent_path}")
        else:
            logger.warning("Stage 3: no target latent found. latent loss will be 0.")

    # Train
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_losses = train_epoch(model, train_loader, loss_fn, optimizer,
                                    scaler, device, stage=stage,
                                    grad_clip=grad_clip, use_amp=use_amp)
        val_losses = validate(model, val_loader, loss_fn, device,
                               stage=stage, use_amp=use_amp) if val_loader else {}
        elapsed = time.time() - t0

        log_parts = [f"Epoch {epoch}/{epochs} | {elapsed:.1f}s"]
        for name in ["loss", "expr", "delta", "deg", "latent", "evidence", "mmd"]:
            tv = train_losses.get(name)
            if tv is not None and abs(tv) > 1e-10:
                log_parts.append(f"tr_{name}={tv:.4f}")
            vv = val_losses.get(name)
            if vv is not None and abs(vv) > 1e-10:
                log_parts.append(f"vl_{name}={vv:.4f}")
        logger.info(" | ".join(log_parts))

        if epoch % 10 == 0 or epoch == epochs:
            save_ckpt(model, optimizer, epoch, config,
                       save_dir / f"stage{stage}" / f"ckpt_epoch{epoch}.pt")

    final_path = save_dir / f"stage{stage}" / "model.pt"
    save_ckpt(model, optimizer, epochs, config, final_path)

    # Stage 2: export evidence-guided target latent AFTER training
    if stage == 2:
        export_path = save_dir / "stage2" / "target_latent.pt"
        export_target_latent(model, train_loader, device, export_path, use_amp)

    logger.info(f"Training complete. Model: {final_path}")
    return model
