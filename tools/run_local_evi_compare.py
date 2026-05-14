"""Run local evidence comparison experiments.

Runs lightweight training for each evidence variant:
  A. ZERO
  B. LOCAL_STRUCT
  C. DATASET_STRUCT_LLM
  D. DATASET_HYBRID_LLM
  E. DATASET_STRUCT_LLM + film
  F. DATASET_STRUCT_LLM + cross

Each run: Stage 1 (8 epochs) → Stage 2 (8 epochs) → Stage 3 (8 epochs) → evaluate → extract metrics.
After each run: cleanup checkpoints, append to compare.csv.
"""

import argparse, csv, json, gc, os, sys, shutil
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd, timeout=1800):
    import subprocess
    print(f"  CMD: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[-1000:]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return False


def run_stage(stage, h5ad, config_path, save_dir, stage1_ckpt=None, stage2_ckpt=None, target_latent=None):
    """Run one training stage and return success."""
    cmd = (
        f'conda run -n DL python main.py train '
        f'--h5ad "{h5ad}" '
        f'--config "{config_path}" '
        f'--stage {stage} '
        f'--epochs 8 '
        f'--batch_size 64 '
        f'--save_dir "{save_dir}" '
        f'--device cuda '
        f'--amp false '
        f'--num_workers 0 '
        f'--progress true '
        f'--log_every 10 '
    )
    if stage1_ckpt:
        cmd += f' --stage1_ckpt "{stage1_ckpt}"'
    if stage2_ckpt:
        cmd += f' --stage2_ckpt "{stage2_ckpt}"'
    if target_latent:
        cmd += f' --target_latent "{target_latent}"'
    return run_command(cmd)


def run_eval(pred_npz, truth_h5ad, out_file, split="test"):
    """Run evaluation."""
    cmd = (
        f'conda run -n DL python main.py eval '
        f'--pred "{pred_npz}" '
        f'--truth "{truth_h5ad}" '
        f'--split {split} '
        f'--out "{out_file}" '
    )
    result = run_command(cmd)
    # Try to read metrics from eval output
    try:
        with open(out_file, "r") as f:
            lines = f.readlines()
        metrics = {}
        for line in lines:
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, val = line.split(":", 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    metrics[key.strip()] = val.strip()
        return metrics
    except Exception:
        return {}


def run_infer(checkpoint, h5ad, out_dir, device="cuda"):
    """Run counterfactual inference for all perturbations."""
    cmd = (
        f'conda run -n DL python main.py infer '
        f'--checkpoint "{checkpoint}" '
        f'--h5ad "{h5ad}" '
        f'--out "{out_dir}" '
        f'--device {device} '
        f'--batch_size 128 '
    )
    return run_command(cmd)


def extract_metrics_from_logs(log_dir):
    """Extract key metrics from training logs."""
    metrics = {}
    for stage in [1, 2, 3]:
        log_file = Path(log_dir) / "logs" / f"stage{stage}.log"
        if log_file.exists():
            with open(log_file) as f:
                content = f.read()
            # Extract last epoch's metrics
            lines = content.strip().split("\n")
            for line in reversed(lines):
                if "deg=" in line and "train" in line:
                    # Parse: train loss=0.xxxx deg=0.xxxx ...
                    import re
                    m = re.search(r'deg=([\d.]+)', line)
                    if m:
                        metrics[f"stage{stage}_train_deg"] = float(m.group(1))
                    m = re.search(r'delta=([\d.]+)', line)
                    if m:
                        metrics[f"stage{stage}_train_delta"] = float(m.group(1))
                    m = re.search(r'z_shift=([\d.]+)', line)
                    if m:
                        metrics[f"stage{stage}_z_shift"] = float(m.group(1))
                    m = re.search(r'evi_gain=([\d.-]+)', line)
                    if m:
                        metrics[f"stage{stage}_evi_gain"] = float(m.group(1))
                    break
            # Find val deg
            for line in reversed(lines):
                if "val" in line and "deg=" in line:
                    import re
                    m = re.search(r'deg=([\d.]+)', line)
                    if m:
                        metrics[f"stage{stage}_val_deg"] = float(m.group(1))
                    break
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Run local evidence comparison experiments")
    ap.add_argument("--config", default="output/local_evi_fix/configs/config_evi_fix.yaml")
    ap.add_argument("--out", default="output/local_evi_fix/compare.csv")
    ap.add_argument("--log_dir", default="output/local_evi_fix/logs")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    config_path = root / args.config
    out_csv = root / args.out
    log_dir = root / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define experiments
    experiments = [
        {
            "name": "A_ZERO",
            "h5ad": "dataset/processed/local_zeroevi.h5ad",
            "evidence_mode": "gate_add",
            "evidence_dim": 128,
            "description": "Zero evidence baseline",
        },
        {
            "name": "B_LOCAL_STRUCT",
            "h5ad": "dataset/processed/local_struct_evi.h5ad",
            "evidence_mode": "gate_add",
            "evidence_dim": 128,
            "description": "Local KB structured evidence",
        },
        {
            "name": "C_DATASET_STRUCT_LLM",
            "h5ad": "dataset/processed/local_dataset_struct_llm_evi.h5ad",
            "evidence_mode": "gate_add",
            "evidence_dim": 128,
            "description": "Dataset-aware LLM structured evidence",
        },
        {
            "name": "D_DATASET_HYBRID_LLM",
            "h5ad": "dataset/processed/local_dataset_hybrid_llm_evi.h5ad",
            "evidence_mode": "gate_add",
            "evidence_dim": 128,
            "description": "Dataset-aware LLM hybrid evidence",
        },
        {
            "name": "E_DATASET_STRUCT_LLM_FILM",
            "h5ad": "dataset/processed/local_dataset_struct_llm_evi.h5ad",
            "evidence_mode": "film",
            "evidence_dim": 128,
            "description": "Dataset-aware LLM structured + FiLM",
        },
        {
            "name": "F_DATASET_STRUCT_LLM_CROSS",
            "h5ad": "dataset/processed/local_dataset_struct_llm_evi.h5ad",
            "evidence_mode": "cross",
            "evidence_dim": 128,
            "description": "Dataset-aware LLM structured + cross-attn",
        },
    ]

    all_metrics = []

    for exp in experiments:
        name = exp["name"]
        h5ad_path = root / exp["h5ad"]

        if not h5ad_path.exists():
            print(f"SKIP {name}: h5ad not found: {h5ad_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Experiment: {name} - {exp['description']}")
        print(f"{'='*60}")

        save_base = root / f"output/local_evi_fix/tmp_run/{name}"

        if args.dry_run:
            print(f"  DRY RUN - would train {name}")
            continue

        # Stage 1
        stage1_dir = save_base / "stage1"
        ok = run_stage(1, h5ad_path, config_path, stage1_dir)
        if not ok:
            print(f"  FAIL: Stage 1 for {name}")
            continue

        stage1_ckpt = stage1_dir / "model.pt"

        # Stage 2
        stage2_dir = save_base / "stage2"
        ok = run_stage(2, h5ad_path, config_path, stage2_dir, stage1_ckpt=stage1_ckpt)
        if not ok:
            print(f"  FAIL: Stage 2 for {name}")
            continue

        stage2_ckpt = stage2_dir / "model.pt"
        target_latent = stage2_dir / "target_latent.pt"

        # Stage 3
        stage3_dir = save_base / "stage3"
        ok = run_stage(3, h5ad_path, config_path, stage3_dir,
                       stage1_ckpt=stage1_ckpt,
                       stage2_ckpt=stage2_ckpt,
                       target_latent=target_latent)
        if not ok:
            print(f"  FAIL: Stage 3 for {name}")
            continue

        stage3_ckpt = stage3_dir / "model.pt"

        # Inference
        infer_dir = save_base / "infer"
        ok = run_infer(stage3_ckpt, h5ad_path, infer_dir)
        if not ok:
            print(f"  FAIL: Inference for {name}")
            continue

        # Evaluation
        pred_npz = infer_dir / "pred.npz"
        eval_out = save_base / "eval_metrics.txt"
        metrics = run_eval(pred_npz, h5ad_path, eval_out, split="test")

        # Extract training metrics from logs
        train_metrics = extract_metrics_from_logs(save_base)

        # Compile metrics
        row = {
            "experiment": name,
            "description": exp["description"],
            "evidence_mode": exp["evidence_mode"],
            "timestamp": datetime.now().isoformat(),
        }
        row.update(metrics)
        row.update(train_metrics)
        all_metrics.append(row)

        # Save summary
        summary_path = log_dir / f"{name}.summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Experiment: {name}\n")
            f.write(f"Description: {exp['description']}\n")
            f.write(f"Evidence mode: {exp['evidence_mode']}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            for k, v in sorted(row.items()):
                f.write(f"{k}: {v}\n")

        print(f"  Summary: {summary_path}")

        # Write compare.csv incrementally
        if all_metrics:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
                writer.writeheader()
                writer.writerows(all_metrics)

        # Cleanup this experiment
        print(f"  Cleaning up {name}...")
        for pattern in ["*.pt", "*.pth", "*.ckpt", "*.npz"]:
            for f in save_base.rglob(pattern):
                f.unlink()
        # Remove large files
        for f in save_base.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".h5ad",):
                if f.stat().st_size > 5 * 1024 * 1024:  # >5MB
                    f.unlink()

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    print(f"\n{'='*60}")
    print(f"All experiments complete.")
    print(f"Compare CSV: {out_csv}")
    print(f"Logs: {log_dir}")

    # Final cleanup
    from tools.cleanup_local_outputs import cleanup
    cleanup()


if __name__ == "__main__":
    main()
