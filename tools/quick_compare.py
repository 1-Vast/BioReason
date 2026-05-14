"""Quick orchestrator for local evidence fix experiments."""
import subprocess, sys, json, csv, shutil, gc, re
from pathlib import Path
from datetime import datetime

ROOT = Path("D:/BioReason")
CONFIG = ROOT / "output/local_evi_fix/configs/config_evi_fix.yaml"
SHARED = ROOT / "output/local_evi_fix/tmp_run/shared"
LOG_DIR = ROOT / "output/local_evi_fix/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SHARED.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {"name": "A_ZERO", "h5ad": "dataset/processed/local_zeroevi.h5ad", "desc": "Zero evidence"},
    {"name": "B_LOCAL_STRUCT", "h5ad": "dataset/processed/local_struct_evi.h5ad", "desc": "Local KB structured"},
    {"name": "C_DATASET_STRUCT_LLM", "h5ad": "dataset/processed/local_dataset_struct_llm_evi.h5ad", "desc": "LLM structured"},
    {"name": "D_DATASET_HYBRID_LLM", "h5ad": "dataset/processed/local_dataset_hybrid_llm_evi.h5ad", "desc": "LLM hybrid"},
]

def run_cmd(cmd, timeout=600):
    print(f"  CMD: {cmd[:150]}...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        err = r.stderr[-800:] if r.stderr else ""
        print(f"  FAIL: {err[:300]}")
    return r.returncode == 0, r.stdout, r.stderr

def parse_metrics(stdout):
    """Parse deg/delta metrics from training output."""
    metrics = {}
    for line in stdout.split("\n"):
        # Look for: train loss=X deg=Y delta=Z
        m = re.search(r'train loss=([\d.]+)\s+deg=([\d.]+)', line)
        if m:
            metrics['train_deg'] = float(m.group(2))
            metrics['train_loss'] = float(m.group(1))
        m = re.search(r'delta=([\d.]+)', line)
        if m:
            metrics['train_delta'] = float(m.group(1))
        m = re.search(r'val loss=([\d.]+)\s+deg=([\d.]+)', line)
        if m:
            metrics['val_deg'] = float(m.group(2))
            metrics['val_loss'] = float(m.group(1))
    return metrics

def train_stage(exp_name, stage, h5ad):
    """Run one training stage."""
    save_dir = ROOT / f"output/local_evi_fix/tmp_run/{exp_name}"
    target_latent_flag = ""
    
    if stage == 3:
        tl_path = SHARED / "stage2" / "target_latent.pt"
        if tl_path.exists():
            target_latent_flag = f'--target_latent "{tl_path}"'
    
    cmd = (
        f'conda run -n DL python "{ROOT}/main.py" train '
        f'--h5ad "{ROOT}/{h5ad}" '
        f'--config "{CONFIG}" '
        f'--stage {stage} '
        f'--batch_size 64 '
        f'--save_dir "{save_dir}" '
        f'--device cuda --no_amp --num_workers 0 --progress '
        f'{target_latent_flag}'
    )
    ok, out, err = run_cmd(cmd, timeout=600)
    
    # Copy checkpoints to shared for next stages
    if ok:
        stage_dir = save_dir / f"stage{stage}"
        shared_stage = SHARED / f"stage{stage}"
        if stage_dir.exists():
            if shared_stage.exists():
                shutil.rmtree(shared_stage)
            shutil.copytree(stage_dir, shared_stage)
    
    return ok, out, err

def run_experiment(exp):
    name = exp["name"]
    h5ad = exp["h5ad"]
    print(f"\n{'='*60}")
    print(f"Experiment: {name} - {exp['desc']}")
    print(f"{'='*60}")
    
    all_metrics = {"experiment": name, "desc": exp["desc"]}
    
    # Stage 1
    ok, out, _ = train_stage(name, 1, h5ad)
    if not ok:
        print(f"  {name}: Stage 1 FAILED")
        return {"experiment": name, "status": "FAIL_stage1", "desc": exp["desc"]}
    s1 = parse_metrics(out)
    for k, v in s1.items():
        all_metrics[f"s1_{k}"] = v
    
    # Stage 2
    ok, out, _ = train_stage(name, 2, h5ad)
    if not ok:
        print(f"  {name}: Stage 2 FAILED")
        return {"experiment": name, "status": "FAIL_stage2", **all_metrics}
    s2 = parse_metrics(out)
    for k, v in s2.items():
        all_metrics[f"s2_{k}"] = v
    
    # Stage 3
    ok, out, _ = train_stage(name, 3, h5ad)
    if not ok:
        print(f"  {name}: Stage 3 FAILED")
        return {"experiment": name, "status": "FAIL_stage3", **all_metrics}
    s3 = parse_metrics(out)
    for k, v in s3.items():
        all_metrics[f"s3_{k}"] = v
    
    all_metrics["status"] = "PASS"
    
    # Save summary
    summary_path = LOG_DIR / f"{name}.summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {name}\n")
        f.write(f"Description: {exp['desc']}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        for k, v in sorted(all_metrics.items()):
            f.write(f"{k}: {v}\n")
    
    print(f"  {name}: PASS - s3_train_deg={all_metrics.get('s3_train_deg', 'N/A')}")
    return all_metrics


# Run all experiments
all_results = []
for exp in EXPERIMENTS:
    result = run_experiment(exp)
    all_results.append(result)
    # Cleanup checkpoints for this experiment
    exp_dir = ROOT / f"output/local_evi_fix/tmp_run/{exp['name']}"
    for pattern in ["*.pt", "*.pth", "*.ckpt"]:
        for f in exp_dir.rglob(pattern) if exp_dir.exists() else []:
            try: f.unlink()
            except: pass
    gc.collect()

# Write compare.csv
csv_path = ROOT / "output/local_evi_fix/compare.csv"
if all_results:
    keys = set()
    for r in all_results:
        keys.update(r.keys())
    keys = sorted(keys)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved: {csv_path}")

# Print summary
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
for r in all_results:
    s3_deg = r.get('s3_train_deg', 'N/A')
    s3_val = r.get('s3_val_deg', 'N/A')
    print(f"  {r['experiment']}: {r.get('status','?')} | train_deg={s3_deg} | val_deg={s3_val}")

# Cleanup shared
if SHARED.exists():
    shutil.rmtree(SHARED)
print("\nDone.")
