"""Inference for BioReason.

Input: control cells + perturbation condition (NO evidence).
Output: predicted perturbed expression (npz / h5ad).
"""

import torch
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast


def load_model(checkpoint_path, model_class=None, device=None):
    """Load BioReason from checkpoint."""
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
    )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)

    if device:
        model = model.to(device)
    model.eval()
    return model, config


def predict(model, dataloader, device="cuda", use_amp=True):
    """Run inference on dataloader. No evidence.

    Returns: preds, deltas, latents, metas
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    preds_list, deltas_list, latents_list, metas_list = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            pert = batch["pert"].to(device)
            cov = {k: v.to(device) for k, v in batch.get("cov", {}).items()}

            with autocast(enabled=use_amp):
                out = model(x, pert, cov=cov, evidence=None, return_latent=True)

            preds_list.append(out["pred"].cpu().numpy())
            deltas_list.append(out["delta"].cpu().numpy())
            if "latent" in out:
                latents_list.append(out["latent"].cpu().numpy())
            metas_list.extend(batch.get("meta", [{}]))

    preds = np.concatenate(preds_list, axis=0)
    deltas = np.concatenate(deltas_list, axis=0)
    latents = np.concatenate(latents_list, axis=0) if latents_list else np.array([])

    return preds, deltas, latents, metas_list


def save_pred(preds, deltas, latents, metas, output_dir, prefix="pred"):
    """Save predictions to disk (npz + optional h5ad)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_dir / f"{prefix}.npz",
             preds=preds, deltas=deltas, latents=latents)

    try:
        import scanpy as sc
        adata = sc.AnnData(X=preds)
        adata.obs["is_predicted"] = True
        if deltas is not None:
            adata.obsm["delta"] = deltas
        if latents is not None and latents.size > 0:
            adata.obsm["latent"] = latents
        adata.write_h5ad(output_dir / f"{prefix}.h5ad")
    except ImportError:
        pass

    print(f"Predictions saved to {output_dir}")


save_predictions = save_pred  # alias
