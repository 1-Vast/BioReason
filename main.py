"""BioReason: Evidence-guided latent biological reasoning
for single-cell perturbation prediction.

Thin CLI entry point. All logic lives in models/.

Usage:
  python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 1
  python main.py infer --config config/default.yaml --checkpoint output/model.pt --h5ad dataset/perturb.h5ad --pert TP53_KO --out output/infer
  python main.py eval --pred output/infer/pred.npz --truth dataset/perturb.h5ad --out output/eval
"""

import argparse
import sys
import os
from pathlib import Path

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def load_yaml(path):
    """Load a YAML file safely."""
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("PyYAML required. Install: pip install pyyaml")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file not found: {path}")
        return {}


def merge_configs(*paths):
    """Merge multiple YAML configs. Later values override earlier ones."""
    merged = {}
    for path in paths:
        if os.path.isfile(path):
            cfg = load_yaml(path)
            for key, val in cfg.items():
                if key == "includes":
                    continue
                if isinstance(val, dict) and isinstance(merged.get(key), dict):
                    merged[key].update(val)
                else:
                    merged[key] = val
    return merged


# ── CLI ───────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="BioReason: evidence-guided latent biological reasoning for sc-perturbation prediction"
    )
    sub = parser.add_subparsers(dest="command", help="Command: train | infer | eval")

    # train
    p_train = sub.add_parser("train", help="Train BioReason")
    p_train.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    p_train.add_argument("--h5ad", required=True, help="Path to h5ad data file")
    p_train.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    p_train.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    p_train.add_argument("--save_dir", default=None, help="Override output dir")
    p_train.add_argument("--target_latent", default=None, help="Target latent path (stage 3)")

    # infer
    p_infer = sub.add_parser("infer", help="Run inference with BioReason")
    p_infer.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    p_infer.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    p_infer.add_argument("--h5ad", required=True, help="Path to h5ad data file")
    p_infer.add_argument("--pert", default=None, help="Perturbation to predict")
    p_infer.add_argument("--out", default="output/infer", help="Output directory")
    p_infer.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    p_infer.add_argument("--batch_size", type=int, default=128, help="Batch size")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate predictions")
    p_eval.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    p_eval.add_argument("--pred", required=True, help="Predictions npz path")
    p_eval.add_argument("--truth", required=True, help="Ground truth h5ad path")
    p_eval.add_argument("--out", default="output/eval", help="Output directory")
    p_eval.add_argument("--top_deg", type=int, default=50, help="Top N DEGs for metrics")

    return parser


def cmd_train(args):
    cfg = merge_configs(
        "config/default.yaml",
        "config/model.yaml",
        "config/train.yaml",
        "config/loss.yaml",
        args.config,
    )
    train_cfg = cfg.get("train", {})
    train_cfg["stage"] = args.stage
    train_cfg["device"] = args.device
    if args.save_dir:
        train_cfg["save_dir"] = args.save_dir

    import torch
    from models.data import build_dataset
    from torch.utils.data import DataLoader, random_split
    from models.reason import BioReason
    from models.loss import BioLoss
    from models.train import train_model

    data_cfg = cfg.get("data", {})
    dataset = build_dataset(args.h5ad, data_cfg)

    n_train = int(len(dataset) * 0.9)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 128),
                               shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.get("batch_size", 128),
                             shuffle=False, num_workers=2)

    model_cfg = cfg.get("model", {})
    model = BioReason(
        input_dim=model_cfg.get("input_dim", dataset.X.shape[1]),
        dim=model_cfg.get("dim", 256),
        hidden=model_cfg.get("hidden", 512),
        steps=model_cfg.get("latent_steps", 8),
        heads=model_cfg.get("heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        residual=model_cfg.get("residual", True),
        pert_mode=model_cfg.get("pert_mode", "id"),
        pert_agg=model_cfg.get("pert_agg", "mean"),
        num_perts=dataset.n_perts,
    )

    loss_fn = BioLoss(cfg.get("loss", {}))

    full_cfg = {**cfg, **train_cfg}
    train_model(model, train_loader, val_loader, full_cfg, loss_fn=loss_fn)


def cmd_infer(args):
    cfg = merge_configs(
        "config/default.yaml",
        "config/model.yaml",
        args.config,
    )

    import torch
    from models.infer import load_model, predict, save_predictions
    from models.data import build_dataset
    from torch.utils.data import DataLoader

    model, ckpt_cfg = load_model(args.checkpoint, device=args.device)
    data_cfg = cfg.get("data", {})
    dataset = build_dataset(args.h5ad, data_cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    preds, deltas, latents, metas = predict(model, loader, device=args.device)
    save_predictions(preds, deltas, latents, metas, args.out)

    print(f"Inference done: {preds.shape[0]} cells x {preds.shape[1]} genes")


def cmd_eval(args):
    import numpy as np
    from models.eval import compute_metrics, save_metrics
    from models.data import load_h5ad

    data = np.load(args.pred)
    preds = data["preds"]
    deltas = data.get("deltas", None)
    latents = data.get("latents", None)

    from scipy.sparse import issparse
    adata = load_h5ad(args.truth)
    X_dense = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    y_true = X_dense[:preds.shape[0]]

    metrics = compute_metrics(y_true, preds, delta_pred=deltas,
                               latents=latents, top_deg=args.top_deg)
    save_metrics(metrics, args.out)

    print("\n=== BioReason Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("============================\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
