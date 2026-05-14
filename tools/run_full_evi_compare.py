"""Full evidence comparison pipeline with DEG evaluation.
Runs 6 experiments (A-F) with train->eval->cleanup.
"""
import subprocess, sys, json, csv, gc, shutil, re, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import anndata as ad
from scipy.sparse import issparse

ROOT = Path("D:/BioReason")
LOG_DIR = ROOT / "output/local_evi_fix/logs"
SHARED = ROOT / "output/local_evi_fix/tmp_run/shared"
COMPARE_CSV = ROOT / "output/local_evi_fix/compare.csv"
CONFIG_DIR = ROOT / "output/local_evi_fix/configs"
TRUTH_H5AD = ROOT / "dataset/processed/local_evi_fix_small.h5ad"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Add project root
sys.path.insert(0, str(ROOT))

EXPERIMENTS = [
    {
        "name": "A_ZERO",
        "desc": "Zero evidence",
        "h5ad": "dataset/processed/local_zeroevi.h5ad",
        "config": CONFIG_DIR / "config_evi_fix.yaml",
        "evidence_mode": "gate_add",
    },
    {
        "name": "B_LOCAL_STRUCT",
        "desc": "Local KB structured",
        "h5ad": "dataset/processed/local_struct_evi.h5ad",
        "config": CONFIG_DIR / "config_evi_fix.yaml",
        "evidence_mode": "gate_add",
    },
    {
        "name": "C_DATASET_STRUCT_LLM",
        "desc": "LLM structured",
        "h5ad": "dataset/processed/local_dataset_struct_llm_evi.h5ad",
        "config": CONFIG_DIR / "config_evi_fix.yaml",
        "evidence_mode": "gate_add",
    },
    # D_DATASET_HYBRID_LLM disabled — mainline uses struct_llm only
    {
        "name": "E_FILM",
        "desc": "LLM structured + film",
        "h5ad": "dataset/processed/local_dataset_struct_llm_evi.h5ad",
        "config": CONFIG_DIR / "config_evi_fix_film.yaml",
        "evidence_mode": "film",
    },
    {
        "name": "F_CROSS",
        "desc": "LLM structured + cross",
        "h5ad": "dataset/processed/local_dataset_struct_llm_evi.h5ad",
        "config": CONFIG_DIR / "config_evi_fix_cross.yaml",
        "evidence_mode": "cross",
    },
]

def run_cmd(cmd, timeout=900):
    t0 = time.time()
    print(f"  [{time.strftime('%H:%M:%S')}] CMD: {cmd[:180]}...")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        ok = r.returncode == 0
        if not ok:
            err = r.stderr[-600:] if r.stderr else ""
            print(f"  FAIL ({elapsed:.0f}s, rc={r.returncode}): {err[:300]}")
        return ok, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False, "", "timeout"


def parse_train_metrics(stdout):
    m = {}
    for line in stdout.split("\n"):
        mm = re.search(r'train loss=([\d.]+)\s+deg=([\d.]+)', line)
        if mm:
            m['train_deg'] = float(mm.group(2))
            m['train_loss'] = float(mm.group(1))
        mm = re.search(r'delta=([\d.]+)', line)
        if mm:
            m['train_delta'] = float(mm.group(1))
        mm = re.search(r'val loss=([\d.]+)\s+deg=([\d.]+)', line)
        if mm:
            m['val_deg'] = float(mm.group(2))
            m['val_loss'] = float(mm.group(1))
        mm = re.search(r'evi_gain=([\d.]+)', line)
        if mm:
            m['evi_gain'] = float(mm.group(1))
        mm = re.search(r'z_shift=([\d.]+)', line)
        if mm:
            m['z_shift'] = float(mm.group(1))
    return m


def train_stage(exp, stage, stage1_ckpt=None, stage2_ckpt=None, target_latent=None):
    name = exp["name"]
    save_dir = ROOT / f"output/local_evi_fix/tmp_run/{name}"
    h5ad_path = ROOT / exp["h5ad"]
    
    cmd = (
        f'conda run -n DL python main.py train '
        f'--h5ad "{h5ad_path}" '
        f'--config "{exp["config"]}" '
        f'--stage {stage} '
        f'--batch_size 64 '
        f'--save_dir "{save_dir}" '
        f'--device cuda --no_amp --num_workers 0 --progress '
    )
    if stage1_ckpt:
        cmd += f' --stage1_ckpt "{stage1_ckpt}"'
    if stage2_ckpt:
        cmd += f' --stage2_ckpt "{stage2_ckpt}"'
    if target_latent and Path(target_latent).exists():
        cmd += f' --target_latent "{target_latent}"'
    
    ok, out, err = run_cmd(cmd, timeout=900)
    return ok, out


def evaluate_checkpoint(checkpoint_path, truth_path, device="cuda", top_k=50):
    """Evaluate model checkpoint on truth data using DEG metrics."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        return {"error": f"load_failed: {e}"}, {}
    
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
    config = checkpoint.get("config", {})
    mc = config.get("model", {})
    dm = config.get("data_meta", {})
    
    # Build model
    from models.reason import BioReason
    
    input_dim = mc.get("input_dim", 1000)
    pert_names = dm.get("pert_names", [])
    num_perts = mc.get("num_perts", len(pert_names))
    dim = mc.get("dim", 64)
    hidden = mc.get("hidden", 128)
    steps = mc.get("steps", mc.get("latent_steps", 3))
    heads = mc.get("heads", 2)
    dropout = mc.get("dropout", 0.1)
    reason_mode = mc.get("reason_mode", "transformer")
    evidence_mode = mc.get("evidence_mode", "gate_add")
    evidence_dim = mc.get("evidence_dim", 128)
    use_evidence_conf = mc.get("use_evidence_conf", True)
    residual = mc.get("residual", True)
    pert_mode = mc.get("pert_mode", "id")
    evidence_strength = mc.get("evidence_strength", 0.2)
    pert_embed_strength = mc.get("pert_embed_strength", 1.0)
    
    model = BioReason(
        input_dim=input_dim, dim=dim,
        hidden=hidden, steps=steps, heads=heads,
        dropout=dropout, reason_mode=reason_mode,
        evidence_mode=evidence_mode, evidence_dim=evidence_dim,
        use_evidence_conf=use_evidence_conf, residual=residual,
        pert_mode=pert_mode, evidence_strength=evidence_strength,
        pert_embed_strength=pert_embed_strength, num_perts=num_perts,
    )
    
    model.load_state_dict(state, strict=False)
    
    model.to(device)
    model.eval()
    
    pert_to_idx = {p: i for i, p in enumerate(pert_names)}
    
    # Load truth data
    truth = ad.read_h5ad(truth_path)
    X = truth.X.toarray() if issparse(truth.X) else np.asarray(truth.X, dtype=np.float32)
    pert_arr = truth.obs["perturbation"].astype(str).values
    split_arr = truth.obs["split"].values
    
    # Control mean from train
    train_mask = split_arr == "train"
    ctrl_mask = (pert_arr == "control") & train_mask
    control_mean = X[ctrl_mask].mean(axis=0)
    
    # Test data
    test_mask = split_arr == "test"
    test_pert = pert_arr[test_mask]
    test_X = X[test_mask]
    
    available_perts = sorted(set(p for p in test_pert if p in pert_to_idx and p != "control"))
    
    results = {}
    all_overlaps, all_pearsons, all_dirs, all_cosines, all_mses = [], [], [], [], []
    
    ctrl_tensor = torch.as_tensor(control_mean, dtype=torch.float32, device=device).unsqueeze(0)  # [1, n_genes]
    
    with torch.no_grad():
        for p in available_perts:
            p_mask = test_pert == p
            n_cells = p_mask.sum()
            if n_cells < 5:
                continue
            
            true_mean = test_X[p_mask].mean(axis=0)
            p_idx = pert_to_idx[p]
            p_t = torch.full((1,), p_idx, dtype=torch.long, device=device)
            
            out = model(ctrl_tensor, p_t, return_latent=False)
            delta = out["delta"].cpu().numpy().squeeze(0)  # [n_genes]
            pred_mean = control_mean + delta
            
            true_abs = np.abs(true_mean)
            n_top = min(top_k, len(true_abs))
            true_top = np.argsort(true_abs)[-n_top:]
            pred_abs = np.abs(pred_mean)
            pred_top = np.argsort(pred_abs)[-n_top:]
            
            overlap = len(set(true_top) & set(pred_top)) / n_top if n_top > 0 else 0
            
            deg_p = 0.0
            if true_mean[true_top].std() > 1e-10:
                deg_p = np.corrcoef(pred_mean[true_top], true_mean[true_top])[0, 1]
                deg_p = max(-1.0, min(1.0, float(deg_p)))
            
            dir_acc = float(np.mean(np.sign(pred_mean[true_top]) == np.sign(true_mean[true_top]))) if n_top > 0 else 0
            
            dn = np.linalg.norm(pred_mean) + 1e-10
            tn = np.linalg.norm(true_mean) + 1e-10
            cos_sim = float(np.dot(pred_mean, true_mean) / (dn * tn))
            
            mse = float(np.mean((pred_mean - true_mean) ** 2))
            
            all_overlaps.append(overlap)
            all_pearsons.append(deg_p)
            all_dirs.append(dir_acc)
            all_cosines.append(cos_sim)
            all_mses.append(mse)
    
    avg = {}
    if all_overlaps:
        avg["top_deg_overlap_mean"] = round(np.mean(all_overlaps), 4)
        avg["deg_pearson_mean"] = round(np.mean(all_pearsons), 4)
        avg["direction_acc_mean"] = round(np.mean(all_dirs), 4)
        avg["delta_cosine_mean"] = round(np.mean(all_cosines), 4)
        avg["mse_mean"] = round(np.mean(all_mses), 6)
        avg["n_perts"] = len(all_overlaps)
    else:
        avg = {"error": "no perts", "top_deg_overlap_mean": 0, "deg_pearson_mean": 0}
    
    return results, avg


def run_experiment(exp):
    name = exp["name"]
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Experiment: {name} - {exp['desc']}")
    print(f"{'='*60}")
    
    save_dir = ROOT / f"output/local_evi_fix/tmp_run/{name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {"experiment": name, "desc": exp["desc"]}
    
    # Stage 1
    print(f"  --- Stage 1 ---")
    ok, out = train_stage(exp, 1)
    if not ok:
        print(f"  {name}: Stage 1 FAILED")
        return {**all_metrics, "status": "FAIL_stage1"}
    s1 = parse_train_metrics(out)
    for k, v in s1.items():
        all_metrics[f"s1_{k}"] = v
    print(f"  Stage 1: deg={s1.get('train_deg','?')}")
    
    # Stage 2
    print(f"  --- Stage 2 ---")
    ok, out = train_stage(exp, 2)
    if not ok:
        print(f"  {name}: Stage 2 FAILED")
        return {**all_metrics, "status": "FAIL_stage2"}
    s2 = parse_train_metrics(out)
    for k, v in s2.items():
        all_metrics[f"s2_{k}"] = v
    print(f"  Stage 2: deg={s2.get('train_deg','?')}")
    
    # Stage 3
    print(f"  --- Stage 3 ---")
    tl_path = SHARED / "stage2" / "target_latent.pt"
    ok, out = train_stage(exp, 3, target_latent=str(tl_path) if tl_path.exists() else None)
    if not ok:
        print(f"  {name}: Stage 3 FAILED")
        return {**all_metrics, "status": "FAIL_stage3"}
    s3 = parse_train_metrics(out)
    for k, v in s3.items():
        all_metrics[f"s3_{k}"] = v
    print(f"  Stage 3: deg={s3.get('train_deg','?')}")
    
    # Evaluate
    print(f"  --- DEG Evaluation ---")
    ckpt_path = save_dir / "stage3" / "model.pt"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        all_metrics["status"] = "FAIL_no_ckpt"
    else:
        pert_results, avg = evaluate_checkpoint(str(ckpt_path), str(TRUTH_H5AD))
        all_metrics.update(avg)
        print(f"  DEG Pearson={avg.get('deg_pearson_mean','N/A')} Overlap@50={avg.get('top_deg_overlap_mean','N/A')} DirAcc={avg.get('direction_acc_mean','N/A')}")
    
    all_metrics["status"] = all_metrics.get("status", "PASS")
    
    # Save summary
    summary_path = LOG_DIR / f"{name}.summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {name}\nDescription: {exp['desc']}\nTimestamp: {datetime.now().isoformat()}\n\n")
        for k, v in sorted(all_metrics.items()):
            f.write(f"{k}: {v}\n")
    
    # Cleanup checkpoints
    for pattern in ["*.pt", "*.pth", "*.ckpt", "*.npz"]:
        for fp in save_dir.rglob(pattern):
            try: fp.unlink()
            except: pass
    for d in list(save_dir.glob("stage*")):
        try: shutil.rmtree(d)
        except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_metrics


def update_csv(new_results):
    all_results = []
    if COMPARE_CSV.exists():
        with open(COMPARE_CSV, "r") as f:
            all_results = list(csv.DictReader(f))
    
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
        print(f"CSV updated: {len(all_results)} rows")


def main():
    print("=" * 60)
    print("FULL EVIDENCE COMPARE PIPELINE")
    print(f"Experiments: {[e['name'] for e in EXPERIMENTS]}")
    print("=" * 60)
    
    t_total = time.time()
    
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n>>> Experiment {i+1}/{len(EXPERIMENTS)}: {exp['name']}")
        t0 = time.time()
        
        result = run_experiment(exp)
        update_csv([result])
        
        elapsed = time.time() - t0
        print(f"<<< {exp['name']}: {result.get('status','?')} ({elapsed:.0f}s) | "
              f"DEG_P={result.get('deg_pearson_mean','?')} "
              f"O@50={result.get('top_deg_overlap_mean','?')}")
    
    # Cleanup shared
    if SHARED.exists():
        shutil.rmtree(SHARED)
    
    # Final summary
    all_results = list(csv.DictReader(open(COMPARE_CSV, "r"))) if COMPARE_CSV.exists() else []
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY ({time.time()-t_total:.0f}s total)")
    print(f"{'='*60}")
    print(f"{'Exp':20s} {'Status':12s} {'DEG_Pearson':>12s} {'Overlap@50':>12s} {'DirAcc':>10s} {'Cos':>8s} {'MSE':>12s}")
    print("-" * 80)
    
    # Get latest results (last 6 entries)
    latest = {}
    for r in all_results:
        latest[r.get("experiment", "?")] = r
    
    for exp in EXPERIMENTS:
        r = latest.get(exp["name"], {})
        print(f"{exp['name']:20s} {r.get('status','?'):12s} "
              f"{str(r.get('deg_pearson_mean','N/A')):>12s} "
              f"{str(r.get('top_deg_overlap_mean','N/A')):>12s} "
              f"{str(r.get('direction_acc_mean','N/A')):>10s} "
              f"{str(r.get('delta_cosine_mean','N/A')):>8s} "
              f"{str(r.get('mse_mean','N/A')):>12s}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
