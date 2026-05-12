"""BioReason: Evidence-guided latent biological reasoning for single-cell perturbation prediction.

Thin CLI entry. All logic in models/. Config in utils/.
Usage:
  python main.py train --h5ad dataset/perturb.h5ad --stage 1
  python main.py infer --checkpoint output/model.pt --h5ad dataset/perturb.h5ad --pert TP53_KO --out output/infer
  python main.py eval --pred output/infer/pred.npz --truth dataset/perturb.h5ad --out output/eval
  python main.py api-test
"""

import argparse, sys, os
from pathlib import Path
from utils.config import load_env, merge as merge_cfg
load_env()


def build_parser():
    p = argparse.ArgumentParser(description="BioReason: evidence-guided latent biological reasoning")
    sub = p.add_subparsers(dest="command", help="train | infer | eval | api-test")

    pt = sub.add_parser("train", help="Train BioReason")
    pt.add_argument("--config", default="config/default.yaml")
    pt.add_argument("--h5ad", required=True)
    pt.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    pt.add_argument("--device", default="cuda")
    pt.add_argument("--save_dir", default=None)
    pt.add_argument("--target_latent", default=None, help="Stage 3 teacher latent path")

    pi = sub.add_parser("infer", help="Counterfactual inference")
    pi.add_argument("--config", default="config/default.yaml")
    pi.add_argument("--checkpoint", required=True)
    pi.add_argument("--h5ad", required=True)
    pi.add_argument("--pert", required=True, help="Target perturbation, e.g. TP53_KO")
    pi.add_argument("--out", default="output/infer")
    pi.add_argument("--device", default="cuda")
    pi.add_argument("--batch_size", type=int, default=128)

    pe = sub.add_parser("eval", help="Evaluate predictions")
    pe.add_argument("--config", default="config/default.yaml")
    pe.add_argument("--pred", required=True)
    pe.add_argument("--truth", required=True)
    pe.add_argument("--out", default="output/eval")
    pe.add_argument("--top_deg", type=int, default=50)

    pa = sub.add_parser("api-test", help="Test LLM API connectivity")
    return p


# ── Train ────────────────────────────────────────────────────────

def cmd_train(args):
    cfg = merge_cfg("config/default.yaml", "config/model.yaml",
                     "config/train.yaml", "config/loss.yaml", args.config)
    train_cfg = cfg.get("train", {})
    train_cfg["stage"] = args.stage
    train_cfg["device"] = args.device
    if args.save_dir: train_cfg["save_dir"] = args.save_dir
    if args.target_latent: train_cfg["target_latent"] = args.target_latent

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

    # Build data_meta for reproducibility
    data_meta = {
        "pert_to_id": dataset.pert_to_id,
        "id_to_pert": dataset.id_to_pert,
        "pert_cats": dataset.pert_cats,
        "selected_var_names": dataset.selected_var_names,
        "input_dim": dataset.input_dim,
        "evidence_dim": dataset.evidence_dim,
        "cov_dims": dict(dataset.cov_dims),
    }
    # Inject into config for checkpoint
    cfg["data_meta"] = data_meta
    model_cfg = cfg.setdefault("model", {})
    model_cfg["input_dim"] = dataset.input_dim
    model_cfg["num_perts"] = dataset.n_perts
    evidence_dim = dataset.evidence_dim or model_cfg.get("evidence_dim")
    model_cfg["evidence_dim"] = evidence_dim
    model_cfg["cov_dims"] = dict(dataset.cov_dims)

    # Stage 3: attach target latent
    if args.stage == 3:
        path = args.target_latent or train_cfg.get("target_latent")
        if path and Path(path).exists():
            attach_target_latent(dataset, path)
            print(f"Stage 3: target latent loaded from {path}")
        else:
            print(f"WARNING: Stage 3, no target latent at {path}. latent loss = 0.")

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
        cov_dims=dict(dataset.cov_dims),
    )
    loss_fn = BioLoss(cfg.get("loss", {}))
    full_cfg = {**cfg, **train_cfg}
    train_model(model, train_loader, val_loader, full_cfg, loss_fn=loss_fn)


# ── Infer ────────────────────────────────────────────────────────

def cmd_infer(args):
    cfg = merge_cfg("config/default.yaml", "config/model.yaml", args.config)
    from models.infer import load_model, predict_counterfactual, save_predictions
    from models.data import build_dataset, align_adata_to_genes, read_h5ad

    model, ckpt_cfg = load_model(args.checkpoint, device=args.device)
    data_meta = ckpt_cfg.get("data_meta", {})

    # Load adata and align genes
    adata = read_h5ad(args.h5ad)
    target_genes = data_meta.get("selected_var_names")
    if target_genes:
        adata = align_adata_to_genes(adata, target_genes)

    data_cfg = cfg.get("data", {})
    dataset = build_dataset(args.h5ad, data_cfg)
    # Use checkpoint data's adata after gene alignment
    dataset.adata = adata
    dataset.X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X) if 'np' in dir() else __import__('numpy').asarray(adata.X)
    import numpy as np
    from scipy.sparse import issparse
    dataset.X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    dataset.input_dim = dataset.X.shape[1]
    dataset._build_group_means()

    # Use checkpoint vocabulary
    if data_meta.get("pert_to_id"):
        dataset.set_vocab(data_meta["pert_to_id"], data_meta.get("id_to_pert"))

    # Verify perturbation
    if args.pert not in dataset.pert_to_id:
        raise ValueError(
            f"Unknown perturbation '{args.pert}'. "
            f"Available ({len(dataset.pert_cats)}): {list(dataset.pert_cats)[:20]}..."
        )

    preds, deltas, latents, metas, pert_arr, pert_strs = predict_counterfactual(
        model, dataset, args.pert, batch_size=args.batch_size, device=args.device)
    save_predictions(preds, deltas, latents, metas, pert_arr, pert_strs, args.out)
    print(f"Inference done: {preds.shape[0]} cells x {preds.shape[1]} genes")


# ── Eval ─────────────────────────────────────────────────────────

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
    metrics = compute_metrics(y_true, preds, delta_pred=deltas, latents=latents, top_deg=args.top_deg)
    save_metrics(metrics, args.out)
    print("\n=== BioReason Evaluation ===")
    for k, v in metrics.items(): print(f"  {k}: {v:.6f}")
    print("============================\n")


# ── API test ─────────────────────────────────────────────────────

def cmd_api_test(args):
    from utils.llm import get_llm_config, test_llm_connection
    cfg = get_llm_config()
    print(f"Provider: {cfg['provider']}")
    print(f"Base URL: {cfg['base_url']}")
    print(f"Model:    {cfg['model']}")
    print(f"Key:      {'***configured***' if cfg['api_key'] else 'MISSING'}")
    result = test_llm_connection()
    print(f"\nConnection: {'OK' if result['ok'] else 'FAILED'} ({result['reason']})")
    if result.get("response"):
        print(f"Response: {result['response']}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()
    cmds = {"train": cmd_train, "infer": cmd_infer, "eval": cmd_eval, "api-test": cmd_api_test}
    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
