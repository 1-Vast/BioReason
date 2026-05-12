"""Inference for BioReason — GPU-optimized counterfactual prediction.

Uses inference_mode, tqdm, per-batch-only CPU transfer.
"""

import torch
import numpy as np
from pathlib import Path


def load_model(checkpoint_path, model_class=None, device=None):
    if model_class is None:
        from .reason import BioReason
        model_class = BioReason
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})
    model = model_class(
        input_dim=model_cfg.get("input_dim", 2000),
        dim=model_cfg.get("dim", 256),
        hidden=model_cfg.get("hidden", 512),
        steps=model_cfg.get("latent_steps", 8),
        heads=model_cfg.get("heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        residual=model_cfg.get("residual", True),
        pert_mode=model_cfg.get("pert_mode", "id"),
        pert_agg=model_cfg.get("pert_agg", "mean"),
        num_perts=model_cfg.get("num_perts", 10000),
        evidence_dim=model_cfg.get("evidence_dim"),
        cov_dims=model_cfg.get("cov_dims", {}),
    )
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    if device:
        model = model.to(device)
    model.eval()
    return model, config


def predict(model, dataloader, device="cuda", use_amp=True, target_pert=None,
            non_blocking=True, progress=True):
    from utils.device import move_to_device, gpu_mem_gb
    from utils.log import make_bar, update_bar_postfix, format_speed
    import time

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device); model.eval()
    preds, deltas, lats, metas, pert_list, pert_strs = [], [], [], [], [], []
    n_cells = 0

    bar = make_bar(dataloader, enable=progress, desc="Infer")

    with torch.inference_mode():
        for batch_cpu in bar:
            b = move_to_device(batch_cpu, device, non_blocking=non_blocking)
            pert = b["pert"] if target_pert is None else \
                   torch.full_like(b["pert"], target_pert)

            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if torch.cuda.is_available() and use_amp else __import__("contextlib").nullcontext()
            with autocast_ctx:
                out = model(b["x"], pert, cov=b.get("cov"), evidence=None, return_latent=True)

            # Move to CPU only after batch forward
            preds.append(out["pred"].detach().cpu())
            deltas.append(out["delta"].detach().cpu())
            if out.get("latent") is not None:
                lats.append(out["latent"].detach().cpu())
            metas.extend(b.get("meta", []))
            pert_list.extend(pert.detach().cpu().tolist())
            pert_strs.extend(b.get("pert_str", []))
            n_cells += b["x"].size(0)

            if progress:
                update_bar_postfix(bar, extra={"cells": n_cells, "mem": f"{gpu_mem_gb():.1f}GB"})

    # Concat on CPU once
    preds = torch.cat(preds, dim=0).numpy()
    deltas = torch.cat(deltas, dim=0).numpy()
    latents = torch.cat(lats, dim=0).numpy() if lats else np.array([])

    return preds, deltas, latents, metas, np.array(pert_list), pert_strs


def predict_counterfactual(model, dataset, pert, batch_size=128, device="cuda",
                            **kwargs):
    dataset.set_target_pert(pert)
    from .data import build_loader
    loader = build_loader(dataset, shuffle=False, batch_size=batch_size)
    return predict(model, loader, device=device, target_pert=dataset._target_pert_id, **kwargs)


def save_pred(preds, deltas, latents, metas, pert_arr, pert_strs, output_dir, prefix="pred"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_dir}/{prefix}.npz ... ", end="", flush=True)

    np.savez(output_dir / f"{prefix}.npz",
             preds=preds, deltas=deltas, latents=latents,
             pert=pert_arr, pert_str=np.array(pert_strs),
             indices=np.array([m.get("idx", i) for i, m in enumerate(metas)]))
    print("OK")

    try:
        import scanpy as sc
        adata = sc.AnnData(X=preds)
        adata.obs["predicted_perturbation_id"] = pert_arr
        adata.obs["predicted_perturbation"] = pert_strs
        adata.obs["source_idx"] = [m.get("source_idx", m.get("idx", i)) for i, m in enumerate(metas)]
        if deltas is not None: adata.obsm["delta"] = deltas
        if latents is not None and latents.size > 0: adata.obsm["latent"] = latents
        print(f"Saving h5ad ... ", end="", flush=True)
        adata.write_h5ad(output_dir / f"{prefix}.h5ad")
        print("OK")
    except ImportError:
        pass


save_predictions = save_pred
