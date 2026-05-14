"""Profile data wait, H2D, forward, backward, and optimizer timing."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from utils.config import merge as merge_cfg
from models.data import build_train_val_datasets, build_loader
from models.cache import load_cached_train_val
from models.reason import BioReason
from models.loss import BioLoss
from models.train import move_batch
from utils.device import get_autocast, get_scaler


def gpu_util() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad")
    ap.add_argument("--cache_dir")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--batch_size", default="256")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preload_cache_to_gpu", action="store_true")
    args = ap.parse_args()

    cfg = merge_cfg("config/default.yaml", "config/model.yaml", "config/train.yaml", "config/loss.yaml", args.config)
    train_cfg = cfg.get("train", {})
    bs = int(train_cfg.get("batch_size", 256) if args.batch_size == "auto" else args.batch_size)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.cache_dir:
        train_ds, _ = load_cached_train_val(args.cache_dir, preload_to_gpu=args.preload_cache_to_gpu, device=args.device)
        nw = 0 if args.preload_cache_to_gpu else int(train_cfg.get("num_workers", 4))
        pin = False if args.preload_cache_to_gpu else bool(train_cfg.get("pin_memory", True))
    else:
        train_ds, _ = build_train_val_datasets(args.h5ad, cfg.get("data", {}))
        nw = int(train_cfg.get("num_workers", 4))
        pin = bool(train_cfg.get("pin_memory", True))
    loader = build_loader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                          pin_memory=pin, persistent_workers=nw > 0,
                          prefetch_factor=int(train_cfg.get("prefetch_factor", 4)), drop_last=True)

    model_cfg = cfg["model"]
    model = BioReason(input_dim=train_ds.input_dim, dim=model_cfg.get("dim", 128),
                      hidden=model_cfg.get("hidden", 256), steps=model_cfg.get("latent_steps", 4),
                      heads=model_cfg.get("heads", 4), dropout=model_cfg.get("dropout", 0.1),
                      residual=model_cfg.get("residual", True), pert_mode=model_cfg.get("pert_mode", "id_plus_evidence"),
                      num_perts=train_ds.n_perts, evidence_dim=train_ds.evidence_dim,
                      cov_dims=dict(train_ds.cov_dims), reason_mode=model_cfg.get("reason_mode", "transformer"),
                      evidence_mode=model_cfg.get("evidence_mode", "gate_add"),
                      use_evidence_conf=model_cfg.get("use_evidence_conf", True)).to(device)
    loss_fn = BioLoss(cfg.get("loss", {}))
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 5e-4))
    scaler = get_scaler(device, train_cfg.get("amp", True))

    sums = {k: 0.0 for k in ["data_wait", "h2d_time", "forward_time", "backward_time", "optimizer_time", "step_time"]}
    n = 0
    cells = 0
    util_samples = []
    t_next = time.perf_counter()
    for batch in loader:
        if n >= args.batches:
            break
        t0 = time.perf_counter()
        sums["data_wait"] += t0 - t_next
        t = time.perf_counter()
        b = move_batch(batch, device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        sums["h2d_time"] += time.perf_counter() - t
        opt.zero_grad(set_to_none=True)
        t = time.perf_counter()
        with get_autocast(device, train_cfg.get("amp", True)):
            out = model(b["x"], b["pert"], cov=b.get("cov"), evidence=b.get("evidence"), evidence_conf=b.get("evidence_conf"), return_latent=True)
            losses = loss_fn(out, {"x": b["x"], "y": b["y"], "evidence": b.get("evidence"), "pert": b["pert"]}, stage=1)
        if device.type == "cuda": torch.cuda.synchronize()
        sums["forward_time"] += time.perf_counter() - t
        t = time.perf_counter()
        scaler.scale(losses["loss"]).backward()
        if device.type == "cuda": torch.cuda.synchronize()
        sums["backward_time"] += time.perf_counter() - t
        t = time.perf_counter()
        scaler.step(opt); scaler.update()
        if device.type == "cuda": torch.cuda.synchronize()
        sums["optimizer_time"] += time.perf_counter() - t
        sums["step_time"] += time.perf_counter() - t0
        cells += int(b["x"].shape[0])
        u = gpu_util()
        if u is not None: util_samples.append(u)
        n += 1
        t_next = time.perf_counter()

    result = {f"{k}_mean": (v / max(n, 1)) for k, v in sums.items()}
    result["cells_per_sec"] = cells / max(sums["step_time"], 1e-9)
    result["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    result["gpu_memory_reserved"] = torch.cuda.memory_reserved(0) if torch.cuda.is_available() else 0
    result["gpu_util_mean"] = sum(util_samples) / len(util_samples) if util_samples else None
    result["batches"] = n
    result["batch_size"] = bs
    result["diagnosis"] = "data_bottleneck" if result["data_wait_mean"] > result["forward_time_mean"] + result["backward_time_mean"] else "compute_or_balanced"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
