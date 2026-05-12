"""Inference for BioReason — counterfactual perturbation prediction.

No evidence required at inference.
Supports counterfactual: input control cells + target perturbation → predicted expression.
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
        evidence_dim=model_cfg.get("evidence_dim"),
    )
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    if device:
        model = model.to(device)
    model.eval()
    return model, config


def predict(model, dataloader, device="cuda", use_amp=True, target_pert=None):
    """Run inference. If target_pert is an int, force that pert id."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    preds, deltas, lats, metas, pert_list = [], [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            pert = batch["pert"].to(device) if target_pert is None else \
                   torch.full_like(batch["pert"].to(device), target_pert)
            cov = {k: v.to(device) for k, v in batch.get("cov", {}).items()}
            with autocast(enabled=use_amp):
                out = model(x, pert, cov=cov, evidence=None, return_latent=True)
            preds.append(out["pred"].cpu().numpy())
            deltas.append(out["delta"].cpu().numpy())
            if out.get("latent") is not None:
                lats.append(out["latent"].cpu().numpy())
            metas.extend(batch.get("meta", [{}]))
            pert_list.extend(pert.cpu().tolist())
    return (np.concatenate(preds, axis=0),
            np.concatenate(deltas, axis=0),
            np.concatenate(lats, axis=0) if lats else np.array([]),
            metas,
            np.array(pert_list))


def predict_counterfactual(model, dataset, pert, batch_size=128, device="cuda"):
    """Counterfactual inference: override all cells to target perturbation."""
    dataset.set_target_pert(pert)
    from .data import build_loader
    loader = build_loader(dataset, shuffle=False, batch_size=batch_size)
    preds, deltas, latents, metas, pert_arr = predict(model, loader,
                                                       device=device,
                                                       target_pert=dataset._target_pert_id)
    return preds, deltas, latents, metas, pert_arr


def save_pred(preds, deltas, latents, metas, pert_arr, output_dir, prefix="pred"):
    """Save predictions to npz + optional h5ad."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"{prefix}.npz",
             preds=preds, deltas=deltas, latents=latents,
             pert=pert_arr,
             indices=np.array([m.get("idx", i) for i, m in enumerate(metas)]))
    try:
        import scanpy as sc
        adata = sc.AnnData(X=preds)
        adata.obs["predicted_perturbation"] = pert_arr.astype(str)
        adata.obs["source_idx"] = [m.get("source_idx", m.get("idx", i))
                                    for i, m in enumerate(metas)]
        if deltas is not None:
            adata.obsm["delta"] = deltas
        if latents is not None and latents.size > 0:
            adata.obsm["latent"] = latents
        adata.write_h5ad(output_dir / f"{prefix}.h5ad")
    except ImportError:
        pass
    print(f"Predictions saved to {output_dir}")


save_predictions = save_pred
