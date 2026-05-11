"""Evaluation metrics for perturbation prediction.

Metrics: MSE, MAE, Pearson, Spearman, R2, DEG_Pearson, top_DEG_overlap, MMD
Output: json + csv
"""

import numpy as np
import json
from pathlib import Path


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson(y_true, y_pred):
    """Per-gene Pearson correlation, averaged."""
    from scipy import stats
    corrs = []
    for g in range(y_true.shape[1]):
        if np.std(y_true[:, g]) > 1e-8 and np.std(y_pred[:, g]) > 1e-8:
            c, _ = stats.pearsonr(y_true[:, g], y_pred[:, g])
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def spearman(y_true, y_pred):
    """Per-gene Spearman correlation, averaged."""
    from scipy import stats
    corrs = []
    for g in range(y_true.shape[1]):
        if np.std(y_true[:, g]) > 1e-8 and np.std(y_pred[:, g]) > 1e-8:
            c, _ = stats.spearmanr(y_true[:, g], y_pred[:, g])
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def r2(y_true, y_pred):
    """Per-gene R2, averaged."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2_per_gene = 1 - ss_res / (ss_tot + 1e-8)
    return float(np.mean(r2_per_gene))


def deg_pearson(y_true, y_pred, top_k=50):
    """Pearson on top DEGs only."""
    abs_delta = np.abs(y_true - y_true.mean(axis=0))
    top_genes = np.argsort(-abs_delta.mean(axis=0))[:top_k]
    return pearson(y_true[:, top_genes], y_pred[:, top_genes])


def top_deg_overlap(y_true, y_pred, top_k=50):
    """Overlap of top DEGs between true and predicted."""
    true_delta = np.abs(y_true - y_true.mean(axis=0)).mean(axis=0)
    pred_delta = np.abs(y_pred - y_pred.mean(axis=0)).mean(axis=0)
    true_top = set(np.argsort(-true_delta)[:top_k])
    pred_top = set(np.argsort(-pred_delta)[:top_k])
    return len(true_top & pred_top) / top_k


def compute_metrics(y_true, y_pred, delta_true=None, delta_pred=None, latents=None,
                    top_deg=50):
    """Compute all evaluation metrics.

    Returns dict of metrics.
    """
    metrics = {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "deg_pearson": deg_pearson(y_true, y_pred, top_k=top_deg),
        "top_deg_overlap": top_deg_overlap(y_true, y_pred, top_k=top_deg),
    }

    if delta_true is not None and delta_pred is not None:
        metrics["delta_mse"] = mse(delta_true, delta_pred)
        metrics["delta_pearson"] = pearson(delta_true, delta_pred)

    if latents is not None and latents.size > 0 and latents.shape[0] >= 2:
        from .loss import mmd_loss
        import torch
        n = min(100, latents.shape[0] // 2)
        z = torch.tensor(latents[:n * 2], dtype=torch.float32)
        metrics["mmd"] = float(mmd_loss(z[:n], z[n:]))

    return metrics


def save_metrics(metrics, output_dir, prefix="metrics"):
    """Save metrics to json and csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{prefix}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import csv
    with open(output_dir / f"{prefix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"Metrics saved to {output_dir}")


def evaluate(y_true, y_pred, output_dir=None, **kwargs):
    """Evaluate and optionally save. Convenience wrapper."""
    metrics = compute_metrics(y_true, y_pred, **kwargs)
    if output_dir:
        save_metrics(metrics, output_dir)
    return metrics
