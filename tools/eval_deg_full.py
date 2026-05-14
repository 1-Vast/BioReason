"""Full DEG evaluation: load checkpoint, infer all perturbations, compute metrics.
Computes: DEG Pearson, Top-50 overlap, direction accuracy, delta cosine, MSE.
"""
import sys, json, argparse, gc
from pathlib import Path
import numpy as np
import torch
import anndata as ad
from scipy.sparse import issparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.reason import BioReason
from utils.log import setup_logger

setup_logger(level="WARNING")


def load_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    
    dim = model_config.get("dim", 64)
    hidden = model_config.get("hidden", 128)
    latent_steps = model_config.get("latent_steps", 3)
    heads = model_config.get("heads", 2)
    dropout = model_config.get("dropout", 0.1)
    reason_mode = model_config.get("reason_mode", "transformer")
    evidence_mode = model_config.get("evidence_mode", "gate_add")
    use_evidence_conf = model_config.get("use_evidence_conf", True)
    residual = model_config.get("residual", True)
    pert_mode = model_config.get("pert_mode", "id")
    evidence_strength = model_config.get("evidence_strength", 0.2)
    pert_embed_strength = model_config.get("pert_embed_strength", 1.0)
    use_control_as_input = config.get("data", {}).get("use_control_as_input", True)
    
    n_genes = checkpoint.get("n_genes", 1000)
    n_covs = checkpoint.get("n_covs", 0)
    pert_names = checkpoint.get("pert_names", [])
    gene_names = checkpoint.get("gene_names", [])
    evidence_dim = model_config.get("evidence_dim", 128)
    
    model = BioReason(
        n_genes=n_genes, n_perts=len(pert_names), dim=dim,
        hidden=hidden, latent_steps=latent_steps, heads=heads,
        dropout=dropout, reason_mode=reason_mode,
        evidence_mode=evidence_mode, evidence_dim=evidence_dim,
        use_evidence_conf=use_evidence_conf, residual=residual,
        pert_mode=pert_mode, evidence_strength=evidence_strength,
        pert_embed_strength=pert_embed_strength, n_covs=n_covs,
        use_control_as_input=use_control_as_input,
    )
    
    # Load state dict
    model_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            model_state[k[6:]] = v
        elif k.startswith("reason.") or k.startswith("cell_encoder.") or k.startswith("pert_encoder.") or k.startswith("decoder.") or k.startswith("cov_encoder.") or k.startswith("evidence_gate."):
            model_state[k] = v
    
    try:
        model.load_state_dict(model_state, strict=False)
    except Exception as e:
        # Try loading with prefix
        if not model_state:
            try:
                model.load_state_dict(state, strict=False)
            except Exception as e2:
                raise RuntimeError(f"Cannot load state dict: {e2}")
    
    model.to(device)
    model.eval()
    
    pert_to_idx = {p: i for i, p in enumerate(pert_names)}
    
    return model, pert_names, pert_to_idx, gene_names, config


def compute_group_mean_control(adata, control_label="control"):
    """Compute control mean from train split."""
    split = adata.obs["split"].values if "split" in adata.obs else None
    pert = adata.obs["perturbation"].astype(str).values
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    
    if split is not None:
        train_mask = split == "train"
        ctrl_mask = (pert == control_label) & train_mask
    else:
        ctrl_mask = pert == control_label
    
    return X[ctrl_mask].mean(axis=0)


def build_pert_evidence_map(adata):
    """Map perturbation -> evidence vector from adata."""
    evi_map = {}
    conf_map = {}
    if "evidence" in adata.obsm:
        evi = adata.obsm["evidence"]
        pert_arr = adata.obs["perturbation"].astype(str).values
        if "evidence_conf" in adata.obs:
            conf_arr = adata.obs["evidence_conf"].values
        else:
            conf_arr = np.ones(len(pert_arr))
        
        for i, p in enumerate(pert_arr):
            if p not in evi_map:
                evi_map[p] = evi[i]
                conf_map[p] = conf_arr[i]
    
    return evi_map, conf_map


def infer_perturbation(model, pert_idx, control_mean, pert_to_idx, device, batch_size=64, evidence=None, evidence_conf=None):
    """Infer perturbation response for a specific perturbation."""
    n_cells = len(control_mean)
    n_genes = control_mean.shape[1] if control_mean.ndim > 1 else control_mean.shape[0]
    
    all_preds = []
    all_deltas = []
    
    pert_tensor = torch.tensor([pert_idx] * len(control_mean), dtype=torch.long, device=device)
    
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            x_ctrl = torch.as_tensor(control_mean[start:end], dtype=torch.float32, device=device)
            p = pert_tensor[start:end]
            
            if x_ctrl.ndim == 1:
                x_ctrl = x_ctrl.unsqueeze(0)
                p = p[:1]
            
            evi_batch = None
            evi_conf_batch = None
            
            pred, delta = model.infer(x_ctrl, p, evidence=evi_batch, evidence_conf=evi_conf_batch)
            all_preds.append(pred.cpu().numpy().reshape(end - start, -1))
            all_deltas.append(delta.cpu().numpy().reshape(end - start, -1))
    
    return np.concatenate(all_preds, axis=0), np.concatenate(all_deltas, axis=0)


def evaluate(checkpoint_path, truth_h5ad_path, device="cuda", top_k=50, control_label="control"):
    """Main evaluation function."""
    print(f"Loading checkpoint: {checkpoint_path}")
    model, pert_names, pert_to_idx, gene_names, config = load_model(checkpoint_path, device)
    print(f"Model: {len(pert_names)} perturbations, {model.n_genes} genes")
    
    print(f"Loading truth: {truth_h5ad_path}")
    truth = ad.read_h5ad(truth_h5ad_path)
    X = truth.X.toarray() if issparse(truth.X) else np.asarray(truth.X, dtype=np.float32)
    pert_arr = truth.obs["perturbation"].astype(str).values
    split_arr = truth.obs["split"].values
    
    # Compute control mean
    control_mean_all = compute_group_mean_control(truth, control_label)
    print(f"Control mean computed: shape={control_mean_all.shape}")
    
    # Get test data
    test_mask = split_arr == "test"
    test_pert = pert_arr[test_mask]
    test_X = X[test_mask]
    
    # Filter perturbations in the model
    test_pert_set = sorted(set(test_pert))
    available_perts = [p for p in test_pert_set if p in pert_to_idx and p != control_label]
    print(f"Test perturbations: {available_perts}")
    
    results = {}
    all_metrics = []
    
    for p in available_perts:
        p_mask_test = test_pert == p
        n_cells = p_mask_test.sum()
        if n_cells < 5:
            print(f"  {p}: {n_cells} cells (skipped)")
            continue
        
        # True perturbation mean
        true_mean = test_X[p_mask_test].mean(axis=0)
        
        # Infer
        p_idx = pert_to_idx[p]
        pred, delta = infer_perturbation(model, p_idx, control_mean_all, pert_to_idx, device)
        
        # Predicted pert mean = control mean + delta
        pred_mean = control_mean_all + delta.mean(axis=0)
        
        # Compute metrics
        true_abs = np.abs(true_mean)
        n_top = min(top_k, len(true_abs))
        true_top = np.argsort(true_abs)[-n_top:]
        pred_abs = np.abs(pred_mean)
        pred_top = np.argsort(pred_abs)[-n_top:]
        
        overlap = len(set(true_top) & set(pred_top)) / n_top if n_top > 0 else 0
        
        deg_p = 0.0
        if true_mean[true_top].std() > 1e-10:
            deg_p = np.corrcoef(pred_mean[true_top], true_mean[true_top])[0, 1]
            deg_p = max(-1.0, min(1.0, deg_p))  # clip
        
        dir_acc = np.mean(np.sign(pred_mean[true_top]) == np.sign(true_mean[true_top])) if n_top > 0 else 0
        
        dn = np.linalg.norm(pred_mean) + 1e-10
        tn = np.linalg.norm(true_mean) + 1e-10
        cos_sim = np.dot(pred_mean, true_mean) / (dn * tn)
        
        mse = np.mean((pred_mean - true_mean) ** 2)
        
        metrics = {
            "top_deg_overlap": round(overlap, 4),
            "deg_pearson": round(deg_p, 4),
            "direction_acc": round(dir_acc, 4),
            "delta_cosine": round(cos_sim, 4),
            "mse": round(float(mse), 4),
            "n_cells": int(n_cells),
        }
        results[p] = metrics
        all_metrics.append(metrics)
        
        print(f"  {p}: Pearson={deg_p:.4f} Overlap@{top_k}={overlap:.4f} DirAcc={dir_acc:.4f} Cos={cos_sim:.4f} MSE={mse:.6f}")
    
    # Compute averages
    if all_metrics:
        avg = {
            "top_deg_overlap_mean": round(np.mean([m["top_deg_overlap"] for m in all_metrics]), 4),
            "deg_pearson_mean": round(np.mean([m["deg_pearson"] for m in all_metrics]), 4),
            "direction_acc_mean": round(np.mean([m["direction_acc"] for m in all_metrics]), 4),
            "delta_cosine_mean": round(np.mean([m["delta_cosine"] for m in all_metrics]), 4),
            "mse_mean": round(np.mean([m["mse"] for m in all_metrics]), 6),
            "n_perts": len(all_metrics),
        }
    else:
        avg = {"error": "no perturbations evaluated"}
    
    return results, avg, pert_names, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()
    
    results, avg, pert_names, config = evaluate(
        args.checkpoint, args.truth, args.device, args.top_k
    )
    
    output = {
        "checkpoint": args.checkpoint,
        "truth": args.truth,
        "perturbations": pert_names,
        "per_perturbation": results,
        "average": avg,
    }
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {args.out}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in avg.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
