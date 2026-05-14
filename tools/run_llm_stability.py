"""Remote LLM stability runner.

Runs leak audit, evidence construction, Stage 1/2/3 training, lightweight
evaluation from training metrics, GPU monitoring, and per-run cleanup. The
script deliberately does not call LLMs inside the training loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

CONFIGS = {
    "best_previous": dict(strength=0.4, alpha=0.8, dropout=0.2, contrast=0.0, adaptive=True),
    "contrastive": dict(strength=0.4, alpha=0.8, dropout=0.2, contrast=0.05, adaptive=True),
    "adaptive_gate": dict(strength=0.4, alpha=0.8, dropout=0.2, contrast=0.0, adaptive=True),
    "conservative": dict(strength=0.3, alpha=0.8, dropout=0.2, contrast=0.0, adaptive=True),
    "strong_evidence": dict(strength=0.5, alpha=1.0, dropout=0.2, contrast=0.0, adaptive=True),
}


def run(cmd: list[str], log: Path, timeout: int = 3600) -> tuple[bool, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    text = ""
    started = time.time()
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=timeout)
            text = p.stdout or ""
            f.write(text)
            ok = p.returncode == 0
        except subprocess.TimeoutExpired as exc:
            text = (exc.stdout or "") + "\nTIMEOUT\n"
            f.write(text)
            ok = False
    elapsed = time.time() - started
    return ok, text + f"\n[elapsed_sec] {elapsed:.2f}\n"


def parse_train_metrics(text: str) -> dict:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if "[train] ep" not in line:
            continue
        section = "train"
        for token in line.replace(",", " ").split():
            if token == "val":
                section = "val"
                continue
            if token == "train":
                section = "train"
                continue
            if "=" not in token:
                continue
            key, val = token.split("=", 1)
            out_key = f"{section}_{key.strip()}" if key.strip() in {"loss", "deg"} else key.strip()
            try:
                metrics[out_key] = float(val.strip())
            except ValueError:
                pass
    return metrics


def write_config(path: Path, cfg: dict, out_dir: Path, batch_size: int, workers: int) -> None:
    data = {
        "data": {
            "h5ad": "",
            "label_key": "perturbation",
            "control_label": "control",
            "cell_type_key": "cell_type",
            "evidence_key": "evidence",
            "target_mode": "group_mean",
            "pair_by": "cell_type",
            "use_control_as_input": True,
            "use_hvg": True,
            "n_hvg": 2000,
            "cache_group_means": True,
            "sparse_output": True,
            "control_input_mode": "control_mean",
        },
        "model": {
            "dim": 128,
            "hidden": 256,
            "latent_steps": 4,
            "heads": 4,
            "dropout": 0.1,
            "reason_mode": "transformer",
            "evidence_mode": "gate_add",
            "evidence_strength": cfg["strength"],
            "evidence_pert_alpha": cfg["alpha"],
            "evidence_dropout": cfg["dropout"],
            "pert_mode": "id_plus_evidence",
            "use_evidence_as_pert_init": True,
            "evidence_dim": 128,
            "adaptive_evidence_gate": cfg["adaptive"],
            "evidence_gate_init_bias": -1.5,
        },
        "train": {
            "epochs": 20,
            "batch_size": batch_size,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "amp": True,
            "grad_clip": 1.0,
            "device": "cuda",
            "num_workers": workers,
            "pin_memory": True,
            "persistent_workers": workers > 0,
            "prefetch_factor": 4,
            "progress": True,
            "profile": True,
            "compile": True,
            "evi_warm": True,
            "evi_warm_epochs": 6,
            "latent_only_bp": True,
            "use_evidence_policy": "quality_gate",
            "min_evidence_conf_train": 0.35,
            "evidence_warm_start_epoch": 2,
            "stage3_init": "stage1",
            "stage1_ckpt": str(out_dir / "tmp_run/shared/stage1/model.pt"),
            "stage2_ckpt": str(out_dir / "tmp_run/shared/stage2/model.pt"),
        },
        "loss": {
            "expr": 1.0,
            "delta": 1.0,
            "deg": 2.0,
            "evidence": 0.0,
            "evi_gain": 0.2,
            "latent": 1.0,
            "mmd": 0.02,
            "mmd_max_samples": 64,
            "evi_contrast": cfg["contrast"],
            "evi_contrast_temp": 0.2,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def clean_transients(out_dir: Path) -> None:
    for rel in ["tmp_run"]:
        p = out_dir / rel
        if p.exists():
            shutil.rmtree(p)
    for pat in ["*.pt", "*.pth", "*.ckpt", "*.npz", "pred*.h5ad"]:
        for p in out_dir.rglob(pat):
            try:
                p.unlink()
            except OSError:
                pass


def env_report(out_dir: Path) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run(["bash", "-lc", "pwd; git status --short; git rev-parse HEAD || true"], log_dir / "env.txt")
    run([sys.executable, "--version"], log_dir / "env.txt")
    run([sys.executable, "-c", "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('torch_cuda:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"], log_dir / "env.txt")
    run(["bash", "-lc", "nvidia-smi"], log_dir / "gpu_before.txt")


def ensure_evidence(args, base_h5ad: Path, seed: str, split: str, evidence: str, out_dir: Path) -> Path:
    ev_dir = Path(args.processed_dir)
    out = ev_dir / f"{args.prefix}_seed{seed}_{split}_{evidence}.h5ad"
    if out.exists():
        return out
    audit = out_dir / "evidence_audit" / f"seed{seed}_{split}_{evidence}.csv"
    if evidence == "zero":
        ok, _ = run([sys.executable, "tools/make_zero_evi.py", "--h5ad", str(base_h5ad),
                     "--out", str(out), "--evidence_dim", "128"], out_dir / "logs/evidence.log", 1800)
    else:
        encoder = "structured" if evidence == "struct_llm" else "hybrid"
        dry = out_dir / "evidence_audit" / f"seed{seed}_{split}_{evidence}.dryrun.csv"
        run([sys.executable, "main.py", "prior", "--h5ad", str(base_h5ad), "--out", str(out),
             "--pert_key", "perturbation", "--control_label", "control", "--use_llm",
             "--dataset_name", "AdamsonWeissman2016", "--cell_line", "K562",
             "--perturbation_type", "CRISPRi", "--platform", "Perturb-seq",
             "--organism", "human", "--evidence_schema", "bio_v2",
             "--encoder", encoder, "--evidence_dim", "128", "--max_llm_calls", "60",
             "--max_llm_tokens", "30000", "--llm_max_tokens", "512",
             "--llm_cache", str(out_dir / "llm_cache.json"), "--audit", str(dry), "--dry_run"],
            out_dir / "logs/evidence.log", 1800)
        ok, _ = run([sys.executable, "main.py", "prior", "--h5ad", str(base_h5ad), "--out", str(out),
             "--pert_key", "perturbation", "--control_label", "control", "--use_llm",
             "--dataset_name", "AdamsonWeissman2016", "--cell_line", "K562",
             "--perturbation_type", "CRISPRi", "--platform", "Perturb-seq",
             "--organism", "human", "--evidence_schema", "bio_v2",
             "--encoder", encoder, "--evidence_dim", "128", "--max_llm_calls", "60",
             "--max_llm_tokens", "30000", "--llm_max_tokens", "512",
             "--llm_cache", str(out_dir / "llm_cache.json"), "--audit", str(audit)],
            out_dir / "logs/evidence.log", 7200)
    if not ok:
        raise RuntimeError(f"evidence build failed: {out}")
    return out


def audit_or_die(h5ad: Path, out_dir: Path) -> None:
    report = out_dir / "audit" / (h5ad.stem + ".json")
    ok, _ = run([sys.executable, "tools/audit_leak.py", "--h5ad", str(h5ad), "--report", str(report)],
                out_dir / "logs/audit.log", 1800)
    if not ok:
        raise RuntimeError(f"leak audit failed: {h5ad}")


def train_run(args, h5ad: Path, seed: str, split: str, evidence: str, config_name: str,
              cfg: dict, out_dir: Path, batch_size: int, workers: int) -> dict:
    exp = f"s{seed}_{split}_{evidence}_{config_name}"
    log = out_dir / "logs" / f"{exp}.summary.txt"
    config_path = out_dir / "configs" / f"{exp}.yaml"
    save_dir = out_dir / "tmp_run" / exp
    shared = out_dir / "tmp_run/shared"
    if log.exists():
        log.unlink()
    write_config(config_path, cfg, out_dir, batch_size, workers)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "split": split,
        "evidence": evidence,
        "config": config_name,
        "experiment": exp,
        "batch_size": batch_size,
        "num_workers": workers,
        "status": "running",
    }
    stage_metrics = {}
    for stage in (1, 2, 3):
        cmd = [sys.executable, "main.py", "train", "--h5ad", str(h5ad), "--config", str(config_path),
               "--stage", str(stage), "--save_dir", str(save_dir), "--device", "cuda",
               "--batch_size", str(batch_size), "--num_workers", str(workers), "--progress", "--profile", "--compile"]
        if stage == 3 and (shared / "stage2/target_latent.pt").exists():
            cmd += ["--target_latent", str(shared / "stage2/target_latent.pt")]
        ok, text = run(cmd, log, args.run_timeout)
        stage_metrics.update({f"s{stage}_{k}": v for k, v in parse_train_metrics(text).items()})
        stage_dir = save_dir / f"stage{stage}"
        if ok and stage_dir.exists():
            dst = shared / f"stage{stage}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(stage_dir, dst)
        if not ok:
            row.update(stage_metrics)
            row["status"] = f"failed_stage{stage}"
            return row
    row.update(stage_metrics)
    row["status"] = "pass"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="dataset/processed")
    ap.add_argument("--prefix", default="remote_stable")
    ap.add_argument("--out_dir", default="output/llm_stability_remote")
    ap.add_argument("--seeds", default="42,123,2024")
    ap.add_argument("--splits", default="heldout,lowcell_5,lowcell_10,lowcell_20,lowcell_50")
    ap.add_argument("--evidence", default="zero,struct_llm,hybrid_llm")
    ap.add_argument("--run_timeout", type=int, default=7200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_report(out_dir)

    from tools.gpu_watch import GpuWatch
    watch = GpuWatch(out_dir / "logs/gpu_monitor.csv", interval=30, threshold=30.0)
    watch.start()

    results_path = out_dir / "results.csv"
    fields = ["timestamp", "seed", "split", "evidence", "config", "experiment", "batch_size",
              "num_workers", "status", "gpu_avg_util", "s1_val_deg", "s2_val_deg", "s3_val_deg", "s3_delta",
              "delta_deg", "delta_top50", "delta_delta_pearson"]
    rows = []
    zero_by_key: dict[tuple[str, str, str], dict] = {}
    batch_size, workers = 512, 6
    try:
        for seed in [s.strip() for s in args.seeds.split(",") if s.strip()]:
            for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
                base = Path(args.processed_dir) / f"{args.prefix}_seed{seed}_{split}.h5ad"
                if not base.exists():
                    rows.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                                 "seed": seed, "split": split, "status": "missing_h5ad",
                                 "experiment": str(base)})
                    continue
                audit_or_die(base, out_dir)
                for evidence in [e.strip() for e in args.evidence.split(",") if e.strip()]:
                    h5ad = ensure_evidence(args, base, seed, split, evidence, out_dir)
                    audit_or_die(h5ad, out_dir)
                    for config_name, cfg in CONFIGS.items():
                        row = train_run(args, h5ad, seed, split, evidence, config_name, cfg,
                                        out_dir, batch_size, workers)
                        key = (seed, split, config_name)
                        if evidence == "zero" and row.get("status") == "pass":
                            zero_by_key[key] = row
                        elif row.get("status") == "pass" and key in zero_by_key:
                            z = zero_by_key[key]
                            zdeg = float(z.get("s3_val_deg", 0.0) or 0.0)
                            ldeg = float(row.get("s3_val_deg", 0.0) or 0.0)
                            row["delta_deg"] = zdeg - ldeg
                        rows.append(row)
                        with results_path.open("w", newline="", encoding="utf-8") as f:
                            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                            w.writeheader()
                            w.writerows(rows)
                        clean_transients(out_dir)
    finally:
        gpu_summary = watch.stop()
        (out_dir / "gpu_summary.csv").write_text(
            "samples,avg_gpu_util,max_memory_mib,warning,threshold\n"
            f"{gpu_summary['samples']},{gpu_summary['avg_gpu_util']},"
            f"{gpu_summary['max_memory_mib']},{gpu_summary['warning']},{gpu_summary['threshold']}\n",
            encoding="utf-8",
        )

    run([sys.executable, "tools/summarize_llm_stability.py", "--results", str(results_path),
         "--out_dir", str(out_dir)], out_dir / "logs/summarize.summary.txt", 1800)


if __name__ == "__main__":
    main()
