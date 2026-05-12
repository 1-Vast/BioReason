"""Training loop for BioReason.

Stages 1-3 with proper target latent export and checkpoint metadata.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import logging, time

logger = logging.getLogger(__name__)


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if k in ("meta", "pert_str"):
            out[k] = v
        elif k == "cov":
            out[k] = {ck: cv.to(device) for ck, cv in v.items()}
        elif k in ("evidence", "target_latent", "target_latent_mask"):
            out[k] = v.to(device) if v is not None else None
        else:
            out[k] = v.to(device)
    return out


def prepare_stage_batch(batch, stage):
    if stage == 2:
        return batch.get("evidence"), None, batch.get("target_latent_mask")
    elif stage == 3:
        return None, batch.get("target_latent"), batch.get("target_latent_mask")
    return None, None, None


def attach_target_latent(dataset, path):
    dataset.load_target_latent(path)
    n_valid = int(dataset.target_latent_mask.sum().item()) if dataset.target_latent_mask is not None else 0
    logger.info(f"Target latent from {path}: {n_valid} valid / {len(dataset)} total")
    return dataset


def export_target_latent(model, dataloader, device, save_path, use_amp=True):
    model.eval()
    latents, indices_list, pert_list = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            b = move_batch(batch, device)
            with autocast(enabled=use_amp):
                z = model.forward_latent(b["x"], b["pert"], cov=b.get("cov"),
                                          evidence=b.get("evidence"))
            latents.append(z.cpu())
            pert_list.append(b["pert"].cpu())
            for m in b.get("meta", []):
                indices_list.append(m.get("idx", len(indices_list)))
    all_z = torch.cat(latents, dim=0)
    all_pert = torch.cat(pert_list, dim=0)
    all_idx = torch.tensor(indices_list, dtype=torch.long)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": all_z, "indices": all_idx, "pert": all_pert,
                "mask": torch.ones(len(all_idx), dtype=torch.bool)}, save_path)
    logger.info(f"Target latent: {save_path}  shape={all_z.shape}")


def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device,
                stage=1, grad_clip=1.0, use_amp=True):
    model.train()
    losses_sum = {}; n_batches = len(dataloader)
    for batch in dataloader:
        b = move_batch(batch, device)
        evidence, target_latent, tmask = prepare_stage_batch(b, stage)
        with autocast(enabled=use_amp):
            out = model(b["x"], b["pert"], cov=b.get("cov"),
                         evidence=evidence, return_latent=True)
            L = loss_fn(out, {"x": b["x"], "y": b["y"], "evidence": evidence,
                               "target_latent": target_latent,
                               "target_latent_mask": tmask}, stage=stage)
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(L["loss"]).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            L["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        for k, v in L.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + v.item()
    return {k: v/n_batches for k, v in losses_sum.items()}


def validate(model, dataloader, loss_fn, device, stage=1, use_amp=True):
    model.eval()
    losses_sum = {}; n_batches = len(dataloader)
    with torch.no_grad():
        for batch in dataloader:
            b = move_batch(batch, device)
            evidence, target_latent, tmask = prepare_stage_batch(b, stage)
            with autocast(enabled=use_amp):
                out = model(b["x"], b["pert"], cov=b.get("cov"),
                             evidence=evidence, return_latent=True)
                L = loss_fn(out, {"x": b["x"], "y": b["y"], "evidence": evidence,
                                   "target_latent": target_latent,
                                   "target_latent_mask": tmask}, stage=stage)
            for k, v in L.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + v.item()
    return {k: v/n_batches for k, v in losses_sum.items()}


def save_ckpt(model, optimizer, epoch, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)
    logger.info(f"CKPT: {path}")


def load_ckpt(path, model=None, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


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
    stage = config.get("stage", 1); epochs = config.get("epochs", 100)
    grad_clip = config.get("grad_clip", 1.0); use_amp = config.get("amp", True)
    save_dir = Path(config.get("save_dir", "output"))

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

    if stage == 2:
        export_path = save_dir / "stage2" / "target_latent.pt"
        export_target_latent(model, train_loader, device, export_path, use_amp)

    logger.info(f"Training done. Model: {final_path}")
    return model
