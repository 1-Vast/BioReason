"""BioReason: Evidence-guided latent biological reasoning for single-cell perturbation prediction.

Thin CLI entry. All logic in models/. Config/GPU in utils/.
"""

import argparse, sys, os
from pathlib import Path
from utils.config import load_env, merge as merge_cfg
load_env()


def build_parser():
    p = argparse.ArgumentParser(description="BioReason: evidence-guided latent biological reasoning")
    sub = p.add_subparsers(dest="command", help="train | infer | eval | api-test | prior")

    pt = sub.add_parser("train", help="Train BioReason")
    pt.add_argument("--config", default="config/default.yaml")
    pt.add_argument("--h5ad", required=True)
    pt.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    pt.add_argument("--device", default="cuda")
    pt.add_argument("--save_dir", default=None)
    pt.add_argument("--target_latent", default=None, help="Stage 3 teacher latent path")
    pt.add_argument("--batch_size", type=int, default=None)
    pt.add_argument("--num_workers", type=int, default=None)
    pt.add_argument("--amp", dest="amp", action="store_true", default=None)
    pt.add_argument("--no_amp", dest="amp", action="store_false")
    pt.add_argument("--progress", dest="progress", action="store_true", default=None)
    pt.add_argument("--no_progress", dest="progress", action="store_false")
    pt.add_argument("--profile", action="store_true", default=False)
    pt.add_argument("--compile", action="store_true", default=False)
    pt.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=None)
    pt.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")

    pi = sub.add_parser("infer", help="Counterfactual inference")
    pi.add_argument("--config", default="config/default.yaml")
    pi.add_argument("--checkpoint", required=True)
    pi.add_argument("--h5ad", required=True)
    pi.add_argument("--pert", required=True, help="Target perturbation, e.g. TP53_KO")
    pi.add_argument("--out", default="output/infer")
    pi.add_argument("--device", default="cuda")
    pi.add_argument("--batch_size", type=int, default=128)
    pi.add_argument("--num_workers", type=int, default=None)
    pi.add_argument("--amp", dest="amp", action="store_true", default=None)
    pi.add_argument("--no_amp", dest="amp", action="store_false")
    pi.add_argument("--progress", dest="progress", action="store_true", default=None)
    pi.add_argument("--no_progress", dest="progress", action="store_false")
    pi.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=None)
    pi.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    pi.add_argument("--memmap", action="store_true", default=False, help="Use disk memmap for large output")
    pi.add_argument("--memmap_dir", default="output/infer_memmap", help="Memmap directory")

    pe = sub.add_parser("eval", help="Evaluate predictions")
    pe.add_argument("--config", default="config/default.yaml")
    pe.add_argument("--pred", required=True)
    pe.add_argument("--truth", required=True)
    pe.add_argument("--out", default="output/eval")
    pe.add_argument("--top_deg", type=int, default=50)

    pa = sub.add_parser("api-test", help="Test LLM API connectivity")

    pp = sub.add_parser("prior", help="Stage 0: build evidence prior")
    pp.add_argument("--h5ad", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--pert_key", default="perturbation")
    pp.add_argument("--control_label", default="control")
    pp.add_argument("--kb", default=None, help="Path to local KB JSON")
    pp.add_argument("--use_llm", action="store_true", default=False)
    pp.add_argument("--min_conf", type=float, default=0.5)
    pp.add_argument("--evidence_dim", type=int, default=128)
    pp.add_argument("--encoder", default="hash", choices=["hash", "sentence"])
    pp.add_argument("--model_name", default=None)
    pp.add_argument("--audit", default="output/prior_audit.csv")

    return p


def print_run_summary(args, cfg, kind="train"):
    train_cfg = cfg.get("train", {})
    print(f"\n{'='*60}")
    print(f"  BioReason {kind}")
    print(f"  stage: {getattr(args, 'stage', 'N/A')} | device: {getattr(args, 'device', train_cfg.get('device','cuda'))} | amp: {train_cfg.get('amp',True)} | batch: {getattr(args, 'batch_size', train_cfg.get('batch_size',128)) or train_cfg.get('batch_size',128)} | workers: {train_cfg.get('num_workers',0)}")
    print(f"  data: {getattr(args, 'h5ad', 'N/A')}")
    if hasattr(args, 'save_dir') and args.save_dir: print(f"  output: {args.save_dir}")
    elif hasattr(args, 'out'): print(f"  output: {args.out}")
    from utils.device import gpu_summary
    print(f"  {gpu_summary()}")
    print(f"{'='*60}\n")


# ── Train ────────────────────────────────────────────────────────

def cmd_train(args):
    cfg = merge_cfg("config/default.yaml", "config/model.yaml", "config/train.yaml", "config/loss.yaml", args.config)
    train_cfg = cfg.setdefault("train", {})
    train_cfg["stage"] = args.stage
    train_cfg["device"] = args.device
    if args.save_dir: train_cfg["save_dir"] = args.save_dir
    if args.target_latent: train_cfg["target_latent"] = args.target_latent

    # CLI overrides
    for key in ("batch_size", "num_workers", "amp", "progress", "pin_memory", "compile"):
        v = getattr(args, key, None)
        if v is not None:
            train_cfg[key] = v
    if args.profile: train_cfg["profile"] = True

    print_run_summary(args, cfg, "train")

    import torch
    from models.data import build_dataset, build_loader, split_data
    from models.reason import BioReason
    from models.loss import BioLoss
    from models.train import train_model, attach_target_latent

    data_cfg = cfg.get("data", {})
    dataset = build_dataset(args.h5ad, data_cfg)
    train_ds, val_ds = split_data(dataset)

    bs = train_cfg.get("batch_size", 128); nw = train_cfg.get("num_workers", 0)
    pm = train_cfg.get("pin_memory", False); pw = train_cfg.get("persistent_workers", False)
    pf = train_cfg.get("prefetch_factor", 2)
    train_loader = build_loader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                                 pin_memory=pm, persistent_workers=pw, prefetch_factor=pf, drop_last=True)
    val_loader = build_loader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                               pin_memory=pm, persistent_workers=False, prefetch_factor=pf)

    # Profile
    if train_cfg.get("profile"):
        from utils.profile import profile_loader, profile_train_step, suggest_loader_settings
        loader_stats = profile_loader(train_loader, args.device, batches=train_cfg.get("profile_batches", 20))
        step_stats = profile_train_step(
            BioReason(input_dim=dataset.input_dim, dim=cfg["model"].get("dim",256),
                       hidden=cfg["model"].get("hidden",512),
                       steps=cfg["model"].get("latent_steps",8),
                       heads=cfg["model"].get("heads",4), dropout=0, pert_mode="id",
                       num_perts=dataset.n_perts, evidence_dim=dataset.evidence_dim).to(args.device),
            train_loader, BioLoss(cfg.get("loss",{})),
            torch.optim.AdamW([torch.zeros(1, requires_grad=True)]),
            args.device, stage=args.stage, batches=train_cfg.get("profile_batches", 20))
        suggestions = suggest_loader_settings(loader_stats, step_stats)
        if suggestions:
            print("\nSuggestions:"); [print(f"  - {s}") for s in suggestions]
        print()

    # Data meta for reproducibility
    data_meta = {
        "pert_to_id": dataset.pert_to_id, "id_to_pert": dataset.id_to_pert,
        "pert_cats": dataset.pert_cats, "selected_var_names": dataset.selected_var_names,
        "input_dim": dataset.input_dim, "evidence_dim": dataset.evidence_dim,
        "cov_dims": dict(dataset.cov_dims),
    }
    cfg["data_meta"] = data_meta
    model_cfg = cfg.setdefault("model", {})
    model_cfg["input_dim"] = dataset.input_dim
    model_cfg["num_perts"] = dataset.n_perts
    evidence_dim = dataset.evidence_dim or model_cfg.get("evidence_dim")
    model_cfg["evidence_dim"] = evidence_dim
    model_cfg["cov_dims"] = dict(dataset.cov_dims)

    if args.stage == 3:
        path = args.target_latent or train_cfg.get("target_latent")
        if path and Path(path).exists():
            attach_target_latent(dataset, path)
            print(f"Stage 3: target latent loaded from {path}")
        else:
            print(f"WARNING: Stage 3, no target latent → latent loss = 0")

    model = BioReason(input_dim=dataset.input_dim, dim=model_cfg.get("dim", 256),
                       hidden=model_cfg.get("hidden", 512), steps=model_cfg.get("latent_steps", 8),
                       heads=model_cfg.get("heads", 4), dropout=model_cfg.get("dropout", 0.1),
                       residual=model_cfg.get("residual", True), pert_mode=model_cfg.get("pert_mode", "id"),
                       pert_agg=model_cfg.get("pert_agg", "mean"), num_perts=dataset.n_perts,
                       evidence_dim=evidence_dim, cov_dims=dict(dataset.cov_dims))
    loss_fn = BioLoss(cfg.get("loss", {}))
    train_model(model, train_loader, val_loader, {**cfg, **train_cfg}, loss_fn=loss_fn)


# ── Infer ────────────────────────────────────────────────────────

def cmd_infer(args):
    cfg = merge_cfg("config/default.yaml", "config/model.yaml", args.config)
    train_cfg = cfg.get("train", {})
    print_run_summary(args, cfg, "infer")

    import numpy as np
    from models.infer import load_model, predict_counterfactual, save_predictions
    from models.data import build_dataset, build_loader, align_adata_to_genes, read_h5ad

    model, ckpt_cfg = load_model(args.checkpoint, device=args.device)
    data_meta = ckpt_cfg.get("data_meta", {})

    adata = read_h5ad(args.h5ad)
    target_genes = data_meta.get("selected_var_names")
    if target_genes:
        adata = align_adata_to_genes(adata, target_genes)

    data_cfg = cfg.get("data", {})
    dataset = build_dataset(args.h5ad, data_cfg)
    from scipy.sparse import issparse
    dataset.adata = adata
    dataset.X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    dataset.input_dim = dataset.X.shape[1]
    dataset._build_group_means()

    if data_meta.get("pert_to_id"):
        dataset.set_vocab(data_meta["pert_to_id"], data_meta.get("id_to_pert"))

    if args.pert not in dataset.pert_to_id:
        raise ValueError(f"Unknown perturbation '{args.pert}'. Available: {list(dataset.pert_cats)[:20]}...")

    bs = args.batch_size; nw = getattr(args, 'num_workers', train_cfg.get("num_workers", 0)) or 0
    pm = args.pin_memory if args.pin_memory is not None else train_cfg.get("pin_memory", False)
    progress = args.progress if args.progress is not None else train_cfg.get("progress", True)
    use_amp = args.amp if args.amp is not None else train_cfg.get("amp", True)

    preds, deltas, latents, metas, pert_arr, pert_strs = predict_counterfactual(
        model, dataset, args.pert, batch_size=bs, device=args.device,
        non_blocking=pm, progress=progress, use_amp=use_amp)
    save_predictions(preds, deltas, latents, metas, pert_arr, pert_strs, args.out)
    print(f"Inference done: {preds.shape[0]} cells x {preds.shape[1]} genes")


# ── Eval / API-test ──────────────────────────────────────────────

def cmd_eval(args):
    import numpy as np
    from models.eval import compute_metrics, save_metrics
    from models.data import load_h5ad
    data = np.load(args.pred)
    preds = data["preds"]; deltas = data.get("deltas"); latents = data.get("latents")
    from scipy.sparse import issparse
    adata = load_h5ad(args.truth)
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    metrics = compute_metrics(X[:preds.shape[0]], preds, delta_pred=deltas, latents=latents, top_deg=args.top_deg)
    save_metrics(metrics, args.out)
    print("\n=== BioReason Evaluation ===")
    for k, v in metrics.items(): print(f"  {k}: {v:.6f}")
    print("============================\n")


def cmd_api_test(args):
    from utils.llm import get_llm_config, test_llm_connection
    cfg = get_llm_config()
    print(f"Provider: {cfg['provider']}  Model: {cfg['model']}")
    print(f"Key: {'***configured***' if cfg['api_key'] else 'MISSING'}")
    result = test_llm_connection()
    print(f"Connection: {'OK' if result['ok'] else 'FAILED'} ({result['reason']})")


def cmd_prior(args):
    from utils.prior import run_prior
    if args.use_llm:
        from utils.llm import has_llm_key
        if not has_llm_key():
            print("WARNING: --use_llm set but no API key found. Using local KB only.")
            args.use_llm = False
    run_prior(args.h5ad, args.out, pert_key=args.pert_key, control_label=args.control_label,
              kb_path=args.kb, use_llm=args.use_llm, min_conf=args.min_conf,
              evidence_dim=args.evidence_dim, encoder=args.encoder,
              model_name=args.model_name, audit_out=args.audit)


def main():
    parser = build_parser()
    args = parser.parse_args()
    cmds = {"train": cmd_train, "infer": cmd_infer, "eval": cmd_eval, "api-test": cmd_api_test, "prior": cmd_prior}
    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
