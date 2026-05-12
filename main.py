"""BioReason: Evidence-guided latent biological reasoning
for single-cell perturbation prediction.

Thin CLI entry point. All logic lives in models/.
Config loading in utils/.

Usage:
  python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 1
  python main.py infer --checkpoint output/stage3/model.pt --h5ad dataset/perturb.h5ad --pert TP53_KO --out output/infer
  python main.py eval --pred output/infer/pred.npz --truth dataset/perturb.h5ad --out output/eval
  python main.py api-test
"""

import argparse
import sys
from pathlib import Path
from utils.config import load_env, merge as merge_cfg
load_env()


# ── CLI ──────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="BioReason: evidence-guided latent biological reasoning"
    )
    sub = parser.add_subparsers(dest="command", help="train | infer | eval | api-test")

    p_train = sub.add_parser("train", help="Train BioReason")
    p_train.add_argument("--config", default="config/default.yaml")
    p_train.add_argument("--h5ad", required=True)
    p_train.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    p_train.add_argument("--device", default="cuda")
    p_train.add_argument("--save_dir", default=None)
    p_train.add_argument("--target_latent", default=None, help="Stage 3 teacher latent path")

    p_infer = sub.add_parser("infer", help="Counterfactual inference")
    p_infer.add_argument("--config", default="config/default.yaml")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--h5ad", required=True)
    p_infer.add_argument("--pert", required=True, help="Target perturbation (e.g., TP53_KO)")
    p_infer.add_argument("--out", default="output/infer")
    p_infer.add_argument("--device", default="cuda")
    p_infer.add_argument("--batch_size", type=int, default=128)

    p_eval = sub.add_parser("eval", help="Evaluate predictions")
    p_eval.add_argument("--config", default="config/default.yaml")
    p_eval.add_argument("--pred", required=True)
    p_eval.add_argument("--truth", required=True)
    p_eval.add_argument("--out", default="output/eval")
    p_eval.add_argument("--top_deg", type=int, default=50)

    p_api = sub.add_parser("api-test", help="Test LLM API connectivity")
    return parser


# ── Commands ─────────────────────────────────────────────────────

def cmd_train(args):
    cfg = merge_cfg("config/default.yaml", "config/model.yaml",
                     "config/train.yaml", "config/loss.yaml", args.config)
    train_cfg = cfg.get("train", {})
    train_cfg["stage"] = args.stage
    train_cfg["device"] = args.device
    if args.save_dir:
        train_cfg["save_dir"] = args.save_dir
    if args.target_latent:
        train_cfg["target_latent"] = args.target_latent

    import torch
    from models.data import build_dataset, build_loader, split_data
    from models.reason import BioReason
    from models.loss import BioLoss
    from models.train import train_model, attach_target_latent

    data_cfg = cfg.get("data", {})
    dataset = build_dataset(args.h5ad, data_cfg)
    train_ds, val_ds = split_data(dataset)
    train_loader = build_loader(train_ds, batch_size=train_cfg.get("batch_size", 128),
                                 shuffle=True, num_workers=train_cfg.get("num_workers", 0))
    val_loader = build_loader(val_ds, batch_size=train_cfg.get("batch_size", 128),
                               shuffle=False, num_workers=train_cfg.get("num_workers", 0))

    # Stage 3: attach target latent
    if args.stage == 3:
        path = args.target_latent or train_cfg.get("target_latent")
        if path and Path(path).exists():
            attach_target_latent(dataset, path)
            print(f"Stage 3: target latent loaded from {path}")
        else:
            print(f"WARNING: Stage 3 but no target latent found at {path}. latent loss = 0.")

    model_cfg = cfg.get("model", {})
    evidence_dim = dataset.evidence_dim or model_cfg.get("evidence_dim")
    model = BioReason(
        input_dim=dataset.input_dim,
        dim=model_cfg.get("dim", 256),
        hidden=model_cfg.get("hidden", 512),
        steps=model_cfg.get("latent_steps", 8),
        heads=model_cfg.get("heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        residual=model_cfg.get("residual", True),
        pert_mode=model_cfg.get("pert_mode", "id"),
        pert_agg=model_cfg.get("pert_agg", "mean"),
        num_perts=dataset.n_perts,
        evidence_dim=evidence_dim,
    )
    loss_fn = BioLoss(cfg.get("loss", {}))
    train_model(model, train_loader, val_loader, {**cfg, **train_cfg}, loss_fn=loss_fn)


def cmd_infer(args):
    cfg = merge_cfg("config/default.yaml", "config/model.yaml", args.config)
    from models.infer import load_model, predict_counterfactual, save_predictions
    from models.data import build_dataset

    model, _ = load_model(args.checkpoint, device=args.device)
    data_cfg = cfg.get("data", {})
    dataset = build_dataset(args.h5ad, data_cfg)

    # Verify perturbation is known
    if args.pert not in dataset.pert_to_id:
        raise ValueError(
            f"Unknown perturbation '{args.pert}'. "
            f"Available ({len(dataset.pert_cats)}): {list(dataset.pert_cats)[:20]}..."
        )

    preds, deltas, latents, metas, pert_arr = predict_counterfactual(
        model, dataset, args.pert, batch_size=args.batch_size, device=args.device,
    )
    save_predictions(preds, deltas, latents, metas, pert_arr, args.out)
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
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    y_true = X[:preds.shape[0]]

    metrics = compute_metrics(y_true, preds, delta_pred=deltas,
                               latents=latents, top_deg=args.top_deg)
    save_metrics(metrics, args.out)
    print("\n=== BioReason Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("============================\n")


def cmd_api_test(args):
    from utils.llm import get_llm_config, test_llm_connection
    cfg = get_llm_config()
    print(f"Provider: {cfg['provider']}")
    print(f"Base URL: {cfg['base_url']}")
    print(f"Model:    {cfg['model']}")
    print(f"Key:      {'***configured***' if cfg['api_key'] else 'MISSING'}")
    result = test_llm_connection()
    print(f"\nConnection test: {'OK' if result['ok'] else 'FAILED'}")
    print(f"Reason: {result['reason']}")
    if result.get("response"):
        print(f"Response: {result['response']}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "api-test":
        cmd_api_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
