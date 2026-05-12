"""Training loop for BioReason.

Stages 1-3 keep GPU data flow, AMP, tqdm progress, and epoch-level checkpointing.
Stage 2 evidence warm-up and Stage 3 latent alignment use latent-only BP for
evidence/alignment losses so cell_encoder and decoder cannot absorb shortcuts.
"""

import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def move_batch(batch, device, non_blocking=True):
    from utils.device import move_to_device
    return move_to_device(batch, device, non_blocking=non_blocking)


def prepare_stage_batch(batch, stage):
    if stage == 2:
        return batch.get("evidence"), None, batch.get("target_latent_mask")
    if stage == 3:
        return None, batch.get("target_latent"), batch.get("target_latent_mask")
    return None, None, None


def attach_target_latent(dataset, path):
    dataset.load_target_latent(path)
    n_valid = int(dataset.target_latent_mask.sum().item()) if dataset.target_latent_mask is not None else 0
    logger.info(f"Target latent loaded: {n_valid}/{len(dataset)} valid")
    return dataset


def export_target_latent(model, dataloader, device, save_path, use_amp=True, non_blocking=True, progress=True):
    from contextlib import nullcontext
    from utils.log import make_bar, format_speed
    from utils.device import get_autocast

    model.eval()
    latents, indices_list, pert_list = [], [], []
    bar = make_bar(dataloader, enable=progress, desc="Export latent")
    t0 = time.perf_counter()
    n_cells = 0
    warned_no_evidence = False
    with torch.no_grad():
        for batch in bar:
            b = move_batch(batch, device, non_blocking=non_blocking)
            if b.get("evidence") is None and not warned_no_evidence:
                logger.warning("Stage 2 target latent exported without evidence; evidence-guided distillation is disabled.")
                warned_no_evidence = True
            ctx = get_autocast(device, use_amp) if use_amp else nullcontext()
            with ctx:
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


def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device, stage=1,
                grad_clip=1.0, use_amp=True, non_blocking=True, progress=True,
                log_every=20, profile=False, epoch=1, evi_warm=False,
                evi_warm_epochs=0, evi_warm_margin=0.02, evi_log=True,
                latent_only_bp=True, latent_only_modules=None):
    from contextlib import nullcontext
    from utils.log import make_bar, update_bar_postfix, short_float
    from utils.device import gpu_mem_gb, get_autocast

    model.train()
    n_batches = len(dataloader)
    loss_keys = ["loss", "expr", "delta", "deg", "latent", "evidence", "mmd",
                 "evi_gain", "z_shift", "evi_rec", "latent_align", "latent_only"]
    loss_sum = {k: 0.0 for k in loss_keys}
    n_samples = 0
    t_data = 0.0
    t_step_start = time.perf_counter()
    warned_no_evidence = False
    warm_active = stage == 2 and evi_warm and epoch <= evi_warm_epochs
    desc = f"Train stg{stage}" + (" warm" if warm_active else "")
    bar = make_bar(dataloader, enable=progress, desc=desc, total=n_batches)

    def optimizer_step(loss):
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return False
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        return True

    def freeze_latent_path():
        if hasattr(model, "set_trainable") and latent_only_modules:
            model.set_trainable(latent_only_modules)
        elif hasattr(model, "freeze_main_path_for_latent"):
            model.freeze_main_path_for_latent()

    def unfreeze_all():
        if hasattr(model, "unfreeze_all"):
            model.unfreeze_all()

    for bi, batch_cpu in enumerate(bar):
        t_data_end = time.perf_counter()
        t_data += t_data_end - t_step_start
        b = move_batch(batch_cpu, device, non_blocking=non_blocking)
        evidence, target_latent, tmask = prepare_stage_batch(b, stage)
        autocast_ctx = get_autocast(device, use_amp) if use_amp else nullcontext()
        loss_batch = {"x": b["x"], "y": b["y"], "evidence": evidence,
                      "target_latent": target_latent, "target_latent_mask": tmask,
                      "evidence_conf": b.get("evidence_conf")}

        if warm_active and evidence is not None:
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                out_ev = model(b["x"], b["pert"], cov=b.get("cov"), evidence=evidence, return_latent=True)
                L_ev = loss_fn(out_ev, loss_batch, stage=2)
                with torch.no_grad():
                    out_no = model(b["x"], b["pert"], cov=b.get("cov"), evidence=None, return_latent=True)
                    L_no = loss_fn(out_no, {"x": b["x"], "y": b["y"], "evidence": None}, stage=1)
            optimizer_step(L_ev["loss"])
            L = dict(L_ev)

            if latent_only_bp:
                optimizer.zero_grad(set_to_none=True)
                freeze_latent_path()
                latent_only_applied = 0.0
                try:
                    with autocast_ctx:
                        out_lat = model(b["x"], b["pert"], cov=b.get("cov"), evidence=evidence, return_latent=True)
                        L_lat = loss_fn(out_lat, loss_batch, stage=2)
                        with torch.no_grad():
                            out_no_ref = model(b["x"], b["pert"], cov=b.get("cov"), evidence=None, return_latent=True)
                        w_gain = getattr(loss_fn, "weights", {}).get("evi_gain", 0.0)
                        w_recon = getattr(loss_fn, "weights", {}).get("evi_recon", getattr(loss_fn, "weights", {}).get("evidence", 1.0))
                        w_shift = getattr(loss_fn, "weights", {}).get("evi_shift", 0.0)
                        loss_evi_gain = torch.relu(L_lat["deg"] - L_no["deg"].detach() + evi_warm_margin)
                        z_shift = torch.norm(out_lat["latent"] - out_no_ref["latent"].detach(), dim=-1).mean()
                        loss_evi_shift = torch.relu(torch.tensor(0.1, device=device) - z_shift) if w_shift > 0 else torch.tensor(0.0, device=device)
                        latent_only_loss = w_gain * loss_evi_gain + w_recon * L_lat["evidence"] + w_shift * loss_evi_shift
                    latent_only_applied = 1.0 if optimizer_step(latent_only_loss) else 0.0
                finally:
                    unfreeze_all()
                L["loss"] = L_ev["loss"].detach() + latent_only_loss.detach()
                L["evi_gain"] = L_no["deg"].detach() - L_lat["deg"].detach()
                L["z_shift"] = z_shift.detach()
                L["evi_rec"] = L_lat["evidence"].detach()
                L["latent_only"] = torch.tensor(latent_only_applied, device=device)
            else:
                z_shift = torch.norm(out_ev["latent"] - out_no["latent"].detach(), dim=-1).mean()
                L["evi_gain"] = L_no["deg"].detach() - L_ev["deg"].detach()
                L["z_shift"] = z_shift.detach()
                L["evi_rec"] = L_ev.get("evidence", torch.tensor(0.0, device=device)).detach()
                L["latent_only"] = torch.tensor(0.0, device=device)

        elif stage == 3 and latent_only_bp:
            optimizer.zero_grad(set_to_none=True)
            pred_batch = {"x": b["x"], "y": b["y"], "evidence": None}
            with autocast_ctx:
                out = model(b["x"], b["pert"], cov=b.get("cov"), evidence=None, return_latent=True)
                L_pred = loss_fn(out, pred_batch, stage=1)
            optimizer_step(L_pred["loss"])
            L = dict(L_pred)

            optimizer.zero_grad(set_to_none=True)
            freeze_latent_path()
            latent_only_applied = 0.0
            try:
                with autocast_ctx:
                    out_lat = model(b["x"], b["pert"], cov=b.get("cov"), evidence=None, return_latent=True)
                    L_lat = loss_fn(out_lat, loss_batch, stage=3)
                    latent_only_loss = getattr(loss_fn, "weights", {}).get("latent", 1.0) * L_lat["latent"]
                latent_only_applied = 1.0 if optimizer_step(latent_only_loss) else 0.0
            finally:
                unfreeze_all()
            L["loss"] = L_pred["loss"].detach() + latent_only_loss.detach()
            L["latent"] = L_lat["latent"].detach()
            L["latent_align"] = L_lat["latent"].detach()
            L["latent_only"] = torch.tensor(latent_only_applied, device=device)

        else:
            if warm_active and evidence is None and not warned_no_evidence:
                logger.warning("Stage 2 warm-up skipped because batch evidence is None.")
                warned_no_evidence = True
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                out = model(b["x"], b["pert"], cov=b.get("cov"), evidence=evidence, return_latent=True)
                L = loss_fn(out, loss_batch, stage=stage)
            optimizer_step(L["loss"])

        n_samples += b["x"].size(0)
        for k in loss_sum:
            v = L.get(k)
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                loss_sum[k] += v.detach().item()
            elif v is not None:
                loss_sum[k] += float(v)

        if progress and bar is not None and (bi + 1) % log_every == 0:
            avg_loss = loss_sum["loss"] / (bi + 1)
            extra = {"mem": f"{gpu_mem_gb():.1f}GB"}
            if warm_active and evi_log:
                extra.update({
                    "evi_gain": short_float(loss_sum["evi_gain"] / (bi + 1)),
                    "z_shift": short_float(loss_sum["z_shift"] / (bi + 1)),
                    "evi_rec": short_float(loss_sum["evi_rec"] / (bi + 1)),
                    "latent_only": short_float(loss_sum["latent_only"] / (bi + 1)),
                })
            elif stage == 3 and latent_only_bp:
                extra.update({
                    "latent_align": short_float(loss_sum["latent_align"] / (bi + 1)),
                    "latent_only": short_float(loss_sum["latent_only"] / (bi + 1)),
                })
            update_bar_postfix(bar, loss_dict={"loss": avg_loss}, extra=extra)
        t_step_start = time.perf_counter()

    n = max(n_batches, 1)
    stats = {k: v / n for k, v in loss_sum.items()}
    stats["data_time"] = t_data
    stats["samples"] = n_samples
    return stats


def validate(model, dataloader, loss_fn, device, stage=1, use_amp=True, non_blocking=True, progress=True):
    from contextlib import nullcontext
    from utils.log import make_bar, update_bar_postfix
    from utils.device import gpu_mem_gb, get_autocast

    model.eval()
    n_batches = len(dataloader)
    loss_sum = {k: 0.0 for k in ["loss", "expr", "delta", "deg", "latent", "evidence", "mmd"]}
    bar = make_bar(dataloader, enable=progress, desc=f"Val   stg{stage}", total=n_batches)

    with torch.no_grad():
        for bi, batch_cpu in enumerate(bar):
            b = move_batch(batch_cpu, device, non_blocking=non_blocking)
            evidence, target_latent, tmask = prepare_stage_batch(b, stage)
            ctx = get_autocast(device, use_amp) if use_amp else nullcontext()
            with ctx:
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


def save_ckpt(model, optimizer, epoch, config, path):
    from utils.io import atomic_torch_save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    atomic_torch_save(ckpt, path)
    logger.info(f"CKPT: {path}")


def load_ckpt(path, model=None, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def initialize_stage3_model(model, config, device="cpu"):
    """Initialize Stage 3 from the configured warm-up checkpoint.

    Mainline follows Monet's Stage 3 principle: start from the Stage 1 warm-up
    model and learn evidence-free latent generation toward Stage 2 target_latent.
    Stage 2 init remains available as an explicit ablation.
    """
    if config.get("stage") != 3:
        return None

    init = config.get("stage3_init", "stage1")
    if init in (None, "none", "current"):
        logger.info("Stage 3 init: current model")
        return "current"
    if init == "stage1":
        path = Path(config.get("stage1_ckpt", "output/stage1/model.pt"))
    elif init == "stage2":
        path = Path(config.get("stage2_ckpt", "output/stage2/model.pt"))
    else:
        logger.warning(f"Unknown Stage 3 init '{init}', using current model.")
        return "current"

    if not path.exists():
        logger.warning(f"Stage 3 requested Stage 1 init, but checkpoint not found: {path}" if init == "stage1"
                       else f"Stage 3 requested Stage 2 init, but checkpoint not found: {path}")
        return None

    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    current = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    skipped = len(state) - len(filtered)
    model.load_state_dict(filtered, strict=False)
    if skipped:
        logger.warning(f"Stage 3 init={init}: skipped {skipped} incompatible checkpoint tensors.")
    logger.info(f"Stage 3 init={init}: loaded {path}")
    return init


def train_model(model, train_loader, val_loader, config, loss_fn=None):
    from utils.device import get_device, get_scaler, gpu_summary, gpu_mem_gb, sync_if_cuda
    from utils.log import format_loss, format_speed

    device = get_device(config.get("device", "cuda"))
    model = model.to(device)
    init_from = initialize_stage3_model(model, config, device=device)
    if init_from:
        config["stage3_loaded_from"] = init_from

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
        train_stats = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device,
            stage=stage, grad_clip=grad_clip, use_amp=use_amp,
            non_blocking=non_blocking, progress=progress, log_every=log_every,
            epoch=epoch, evi_warm=config.get("evi_warm", True),
            evi_warm_epochs=config.get("evi_warm_epochs", 10),
            evi_warm_margin=config.get("evi_warm_margin", 0.02),
            evi_log=config.get("evi_log", True),
            latent_only_bp=config.get("latent_only_bp", True),
            latent_only_modules=config.get("latent_only_modules"),
        )
        val_stats = validate(model, val_loader, loss_fn, device, stage=stage,
                             use_amp=use_amp, non_blocking=non_blocking,
                             progress=progress) if val_loader else {}
        elapsed = time.perf_counter() - t0
        sync_if_cuda(device)

        parts = [f"Epoch {epoch:03d}/{epochs}"]
        if stage == 2 and config.get("evi_warm", True) and epoch <= config.get("evi_warm_epochs", 10):
            parts.append("stg2 warm")
        if stage == 3:
            parts.append(f"init={config.get('stage3_loaded_from', config.get('stage3_init', 'stage1'))}")
        parts.append(f"tr {format_loss(train_stats)}")
        if val_stats:
            parts.append(f"vl {format_loss(val_stats, prefix='vl_')}")
        parts.append(format_speed(train_stats.get("samples", 0), elapsed))
        parts.append(f"gpu {gpu_mem_gb():.1f}GB")
        logger.info(" | ".join(parts))

        if epoch % 10 == 0 or epoch == epochs:
            save_ckpt(model, optimizer, epoch, config, save_dir / f"stage{stage}" / f"ckpt_epoch{epoch}.pt")
        if config.get("empty_cache", False) and device.type == "cuda":
            torch.cuda.empty_cache()

    final_path = save_dir / f"stage{stage}" / "model.pt"
    save_ckpt(model, optimizer, epochs, config, final_path)

    if stage == 2:
        export_path = save_dir / "stage2" / "target_latent.pt"
        export_target_latent(model, train_loader, device, export_path, use_amp=use_amp,
                             non_blocking=non_blocking, progress=progress)

    logger.info(f"Training done -> {final_path}")
    return model
