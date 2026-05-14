"""Run E (film) and F (cross) ablation experiments + DEG evaluation.
Each experiment: Stage 1 (8ep) -> Stage 2 (8ep) -> Stage 3 (8ep) -> eval.
After each run: cleanup checkpoints, collect metrics.
"""
import subprocess, sys, json, csv, gc, re, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import anndata as ad
from scipy.sparse import issparse

ROOT = Path("D:/BioReason")
H5AD = ROOT / "dataset/processed/local_dataset_struct_llm_evi.h5ad"
TRUTH_H5AD = ROOT / "dataset/processed/local_evi_fix_small.h5ad"
LOG_DIR = ROOT / "output/local_evi_fix/logs"
SHARED = ROOT / "output/local_evi_fix/tmp_run/shared"
COMPARE_CSV = ROOT / "output/local_evi_fix/compare.csv"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "E_FILM": ROOT / "output/local_evi_fix/configs/config_evi_fix_film.yaml",
    "F_CROSS": ROOT / "output/local_evi_fix/configs/config_evi_fix_cross.yaml",
}

DESCS = {
    "E_FILM": "LLM structured + film (ablation)",
    "F_CROSS": "LLM structured + cross (ablation)",
}

def run_cmd(cmd, timeout=600):
    print(f"  CMD: {cmd[:200]}...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    ok = r.returncode == 0
    if not ok:
        print(f"  FAIL (rc={r.returncode}): {r.stderr[-500:] if r.stderr else ''}")
    return ok, r.stdout, r.stderr

def parse_metrics(stdout):
    metrics = {}
    for line in stdout.split("\n"):
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
        m = re.search(r'evi_gain=([\d.]+)', line)
        if m:
            metrics['evi_gain'] = float(m.group(1))
        m = re.search(r'z_shift=([\d.]+)', line)
        if m:
            metrics['z_shift'] = float(m.group(1))
    return metrics

def train_stage(exp_name, stage, config_path, stage1_ckpt=None, stage2_ckpt=None, target_latent=None):
    save_dir = ROOT / f"output/local_evi_fix/tmp_run/{exp_name}"
    target_latent_flag = ""
    if target_latent and Path(target_latent).exists():
        target_latent_flag = f'--target_latent "{target_latent}"'

    cmd = (
        f'conda run -n DL python main.py train '
        f'--h5ad "{H5AD}" '
        f'--config "{config_path}" '
        f'--stage {stage} '
        f'--batch_size 64 '
        f'--save_dir "{save_dir}" '
        f'--device cuda --no_amp --num_workers 0 --progress '
    )
    if stage1_ckpt:
        cmd += f' --stage1_ckpt "{stage1_ckpt}"'
    if stage2_ckpt:
        cmd += f' --stage2_ckpt "{stage2_ckpt}"'
    if target_latent_flag:
        cmd += f' {target_latent_flag}'

    ok, out, err = run_cmd(cmd, timeout=900)
    return ok, out, err

def evaluate_prediction(pred_npz_path, truth_h5ad_path, top_k=50):
    """Compute DEG-level metrics: Pearson, overlap@50, direction accuracy, delta cosine."""
    pred_data = np.load(pred_npz_path)
    pred = pred_data["pred"]
    pred_delta = pred_data.get("delta", np.zeros_like(pred))
    
    truth = ad.read_h5ad(truth_h5ad_path)
    X = truth.X.toarray() if issparse(truth.X) else np.asarray(truth.X)
    pert_arr = truth.obs["perturbation"].astype(str).values
    split_arr = truth.obs["split"].values
    
    test_mask = split_arr == "test"
    test_X = X[test_mask]
    test_pert = pert_arr[test_mask]
    
    pert_names = sorted(set(test_pert))
    pert_names = [p for p in pert_names if p != "control"]
    
    results = {}
    all_overlaps, all_pearsons, all_dirs, all_cosines = [], [], [], []
    
    for p in pert_names:
        p_mask = test_pert == p
        if p_mask.sum() < 5:
            continue
            
        true_mean = test_X[p_mask].mean(axis=0)
        pred_mean = pred[test_mask][p_mask].mean(axis=0)
        
        true_abs = np.abs(true_mean)
        n_top = min(top_k, len(true_abs))
        true_top = np.argsort(true_abs)[-n_top:]
        pred_abs = np.abs(pred_mean)
        pred_top = np.argsort(pred_abs)[-n_top:]
        
        overlap = len(set(true_top) & set(pred_top)) / n_top if n_top > 0 else 0
        deg_p = np.corrcoef(pred_mean[true_top], true_mean[true_top])[0, 1] if true_mean[true_top].std() > 1e-10 else 0
        dir_acc = np.mean(np.sign(pred_mean[true_top]) == np.sign(true_mean[true_top])) if n_top > 0 else 0
        dn = np.linalg.norm(pred_mean)
        tn = np.linalg.norm(true_mean)
        cos_sim = np.dot(pred_mean, true_mean) / (dn * tn + 1e-10)
        
        all_overlaps.append(overlap)
        all_pearsons.append(max(0, deg_p))  # clip negative
        all_dirs.append(dir_acc)
        all_cosines.append(cos_sim)
        
        results[p] = {
            "top_deg_overlap": round(overlap, 4),
            "deg_pearson": round(deg_p, 4),
            "direction_acc": round(dir_acc, 4),
            "delta_cosine": round(cos_sim, 4),
            "n_cells": int(p_mask.sum()),
        }
    
    avg = {}
    if all_overlaps:
        avg["top_deg_overlap_mean"] = round(np.mean(all_overlaps), 4)
        avg["deg_pearson_mean"] = round(np.mean(all_pearsons), 4)
        avg["direction_acc_mean"] = round(np.mean(all_dirs), 4)
        avg["delta_cosine_mean"] = round(np.mean(all_cosines), 4)
        avg["n_perts_evaluated"] = len(all_overlaps)
    else:
        avg = {"error": "no perturbations with sufficient test cells"}
    
    return results, avg


def run_experiment(exp_name):
    config = CONFIGS[exp_name]
    desc = DESCS[exp_name]
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name} - {desc}")
    print(f"Config: {config.name}")
    print(f"{'='*60}")
    
    save_dir = ROOT / f"output/local_evi_fix/tmp_run/{exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {"experiment": exp_name, "desc": desc}
    
    # Stage 1
    print(f"\n  --- Stage 1 ---")
    ok, out, _ = train_stage(exp_name, 1, config)
    if not ok:
        print(f"  {exp_name}: Stage 1 FAILED")
        return {**all_metrics, "status": "FAIL_stage1"}
    s1 = parse_metrics(out)
    for k, v in s1.items():
        all_metrics[f"s1_{k}"] = v
    print(f"  Stage 1 done: train_deg={s1.get('train_deg', '?')}")
    
    # Stage 2
    print(f"\n  --- Stage 2 ---")
    ok, out, _ = train_stage(exp_name, 2, config)
    if not ok:
        print(f"  {exp_name}: Stage 2 FAILED")
        return {**all_metrics, "status": "FAIL_stage2"}
    s2 = parse_metrics(out)
    for k, v in s2.items():
        all_metrics[f"s2_{k}"] = v
    print(f"  Stage 2 done: train_deg={s2.get('train_deg', '?')}")
    
    # Stage 3
    print(f"\n  --- Stage 3 ---")
    tl_path = SHARED / "stage2" / "target_latent.pt"
    ok, out, _ = train_stage(exp_name, 3, config, target_latent=str(tl_path) if tl_path.exists() else None)
    if not ok:
        print(f"  {exp_name}: Stage 3 FAILED")
        return {**all_metrics, "status": "FAIL_stage3"}
    s3 = parse_metrics(out)
    for k, v in s3.items():
        all_metrics[f"s3_{k}"] = v
    print(f"  Stage 3 done: train_deg={s3.get('train_deg', '?')}")
    
    # Run inference + evaluation
    print(f"\n  --- Inference + Evaluation ---")
    pred_npz = save_dir / "pred_test.npz"
    
    # Inference
    inf_cmd = (
        f'conda run -n DL python main.py infer '
        f'--h5ad "{H5AD}" '
        f'--model "{save_dir}/stage3/model.pt" '
        f'--save_dir "{save_dir}" '
        f'--out "{pred_npz}" '
        f'--device cuda --no_amp '
        f'--target_pert all '
    )
    ok_inf, out_inf, _ = run_cmd(inf_cmd, timeout=600)
    if not ok_inf or not pred_npz.exists():
        print(f"  Inference FAILED or no prediction file")
        all_metrics["status"] = "FAIL_infer"
    else:
        # Evaluate DEG metrics
        pert_results, avg_metrics = evaluate_prediction(str(pred_npz), str(TRUTH_H5AD))
        all_metrics.update(avg_metrics)
        all_metrics["status"] = "PASS"
        print(f"  DEG Pearson mean={avg_metrics.get('deg_pearson_mean', 'N/A')}")
        print(f"  Top-50 overlap mean={avg_metrics.get('top_deg_overlap_mean', 'N/A')}")
        print(f"  Direction acc mean={avg_metrics.get('direction_acc_mean', 'N/A')}")
    
    # Save summary
    summary_path = LOG_DIR / f"{exp_name}.summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Description: {desc}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        for k, v in sorted(all_metrics.items()):
            f.write(f"{k}: {v}\n")
    
    print(f"  Summary saved: {summary_path}")
    return all_metrics


def cleanup_experiment(exp_name):
    exp_dir = ROOT / f"output/local_evi_fix/tmp_run/{exp_name}"
    if exp_dir.exists():
        # Delete checkpoints
        for pattern in ["*.pt", "*.pth", "*.ckpt", "*.npz"]:
            for f in exp_dir.rglob(pattern):
                try:
                    f.unlink()
                except:
                    pass
        # Remove empty stage dirs
        for d in list(exp_dir.glob("stage*")):
            try:
                shutil.rmtree(d)
            except:
                pass
        print(f"  Cleaned: {exp_name}")
    gc.collect()


def update_compare_csv(new_results):
    """Append new results to compare.csv."""
    all_results = []
    if COMPARE_CSV.exists():
        with open(COMPARE_CSV, "r") as f:
            reader = csv.DictReader(f)
            all_results = list(reader)
    
    for r in new_results:
        all_results.append(r)
    
    if all_results:
        keys = set()
        for r in all_results:
            keys.update(r.keys())
        keys = sorted(keys)
        with open(COMPARE_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
        print(f"Updated: {COMPARE_CSV} ({len(all_results)} rows)")


if __name__ == "__main__":
    experiments_to_run = list(CONFIGS.keys())
    print(f"Running {len(experiments_to_run)} experiments: {experiments_to_run}")
    
    new_results = []
    for exp_name in experiments_to_run:
        try:
            result = run_experiment(exp_name)
            new_results.append(result)
        except Exception as e:
            print(f"  {exp_name}: EXCEPTION: {e}")
            new_results.append({"experiment": exp_name, "status": "EXCEPTION", "error": str(e)})
        
        # Cleanup
        cleanup_experiment(exp_name)
        
        # Update CSV after each experiment
        update_compare_csv([result])
        print(f"  Progress: {len(new_results)}/{len(experiments_to_run)} complete")
    
    # Final summary
    print(f"\n{'='*60}")
    print("RUN COMPLETE")
    print(f"{'='*60}")
    for r in new_results:
        exp = r.get('experiment', '?')
        status = r.get('status', '?')
        deg_p = r.get('deg_pearson_mean', 'N/A')
        overlap = r.get('top_deg_overlap_mean', 'N/A')
        print(f"  {exp}: {status} | DEG_Pearson={deg_p} | Overlap@50={overlap}")
    
    # Cleanup shared
    if SHARED.exists():
        shutil.rmtree(SHARED)
    print("Done.")
