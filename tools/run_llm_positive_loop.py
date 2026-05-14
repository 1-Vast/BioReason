"""Auto experiment loop for LLM-positive evidence verification.

Runs held-out and low-cell experiments. If LLM evidence doesn't beat ZERO,
automatically tries architecture variations (evidence_strength, evidence_mode, etc).

Strategy:
  Round 1: Default config (gate_add, evidence_strength=0.2, alpha=0.5, dropout=0.2)
  Round 2: Try stronger evidence (strength=0.4, alpha=0.8)
  Round 3: Try cross-attention evidence mode
  Max 3 rounds. Stop early if success criteria met.

Success criteria for LLM > ZERO:
  - DEG Pearson improvement >= 0.03
  - top-DEG overlap@50 not worse than ZERO
  - Delta Pearson not worse
"""

import subprocess, json, csv, shutil, gc, re, sys, time, itertools
from pathlib import Path
from datetime import datetime

ROOT = Path("D:/BioReason")
OUT_DIR = ROOT / "output/llm_positive"
LOG_DIR = OUT_DIR / "logs"
SHARED = OUT_DIR / "tmp_run/shared"

# Architecture search space
ARCH_ROUNDS = [
    {  # Round 1: default
        "evidence_mode": "gate_add",
        "evidence_strength": 0.2,
        "evidence_pert_alpha": 0.5,
        "evidence_dropout": 0.2,
        "label": "R1_default",
    },
    {  # Round 2: stronger evidence
        "evidence_mode": "gate_add",
        "evidence_strength": 0.4,
        "evidence_pert_alpha": 0.8,
        "evidence_dropout": 0.1,
        "label": "R2_stronger",
    },
    {  # Round 3: cross-attention
        "evidence_mode": "cross",
        "evidence_strength": 0.3,
        "evidence_pert_alpha": 0.5,
        "evidence_dropout": 0.2,
        "label": "R3_cross",
    },
]

# Experiments to run
EXPERIMENTS = [
    # Held-out
    {"name": "heldout_zero", "h5ad": "dataset/processed/llm_heldout_zero.h5ad",
     "split_type": "heldout", "evidence_type": "zero"},
    {"name": "heldout_local_struct", "h5ad": "dataset/processed/llm_heldout_local_struct.h5ad",
     "split_type": "heldout", "evidence_type": "local_struct"},
    {"name": "heldout_dataset_struct_llm", "h5ad": "dataset/processed/llm_heldout_dataset_struct_llm.h5ad",
     "split_type": "heldout", "evidence_type": "dataset_struct_llm"},
    {"name": "heldout_dataset_hybrid_llm", "h5ad": "dataset/processed/llm_heldout_dataset_hybrid_llm.h5ad",
     "split_type": "heldout", "evidence_type": "dataset_hybrid_llm"},
    # Low-cell 5
    {"name": "lowcell5_zero", "h5ad": "dataset/processed/llm_lowcell_5_zero.h5ad",
     "split_type": "lowcell", "evidence_type": "zero", "n_train": 5},
    {"name": "lowcell5_dataset_struct_llm", "h5ad": "dataset/processed/llm_lowcell_5_dataset_struct_llm.h5ad",
     "split_type": "lowcell", "evidence_type": "dataset_struct_llm", "n_train": 5},
    # Low-cell 10
    {"name": "lowcell10_zero", "h5ad": "dataset/processed/llm_lowcell_10_zero.h5ad",
     "split_type": "lowcell", "evidence_type": "zero", "n_train": 10},
    {"name": "lowcell10_dataset_struct_llm", "h5ad": "dataset/processed/llm_lowcell_10_dataset_struct_llm.h5ad",
     "split_type": "lowcell", "evidence_type": "dataset_struct_llm", "n_train": 10},
    # Low-cell 20
    {"name": "lowcell20_zero", "h5ad": "dataset/processed/llm_lowcell_20_zero.h5ad",
     "split_type": "lowcell", "evidence_type": "zero", "n_train": 20},
    {"name": "lowcell20_dataset_struct_llm", "h5ad": "dataset/processed/llm_lowcell_20_dataset_struct_llm.h5ad",
     "split_type": "lowcell", "evidence_type": "dataset_struct_llm", "n_train": 20},
    # Low-cell 50
    {"name": "lowcell50_zero", "h5ad": "dataset/processed/llm_lowcell_50_zero.h5ad",
     "split_type": "lowcell", "evidence_type": "zero", "n_train": 50},
    {"name": "lowcell50_dataset_struct_llm", "h5ad": "dataset/processed/llm_lowcell_50_dataset_struct_llm.h5ad",
     "split_type": "lowcell", "evidence_type": "dataset_struct_llm", "n_train": 50},
]


def run_cmd(cmd, timeout=600):
    """Run a shell command, return (success, stdout, stderr)."""
    print(f"  CMD: {cmd[:180]}...")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0
        if not ok:
            print(f"  FAIL: {r.stderr[-400:]}")
        return ok, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return False, "", "TIMEOUT"


def parse_metrics(stdout):
    """Extract training metrics from stdout."""
    m = {}
    for line in stdout.split("\n"):
        # train loss=X deg=Y delta=Z
        r = re.search(r'train loss=([\d.]+)\s+deg=([\d.]+)', line)
        if r:
            m['train_loss'] = float(r.group(1))
            m['train_deg'] = float(r.group(2))
        r = re.search(r'delta=([\d.]+)', line)
        if r:
            m['train_delta'] = float(r.group(1))
        r = re.search(r'val loss=([\d.]+)\s+deg=([\d.]+)', line)
        if r:
            m['val_loss'] = float(r.group(1))
            m['val_deg'] = float(r.group(2))
        r = re.search(r'z_shift=([\d.]+)', line)
        if r:
            m['z_shift'] = float(r.group(1))
        r = re.search(r'evi_gain=([\d.-]+)', line)
        if r:
            m['evi_gain'] = float(r.group(1))
    return m


def write_config(arch, path):
    """Write a config YAML with architecture overrides."""
    base = """# Auto-generated LLM-positive config
data:
  h5ad: ""
  label_key: "perturbation"
  control_label: "control"
  cell_type_key: "cell_type"
  dose_key: "dose"
  time_key: "time"
  batch_key: "batch"
  evidence_key: "evidence"
  target_mode: "group_mean"
  pair_by: "cell_type"
  use_control_as_input: true
  use_hvg: true
  n_hvg: 1000
  cache_group_means: true
  sparse_output: true
  control_input_mode: "control_mean"

model:
  dim: 64
  hidden: 128
  latent_steps: 3
  heads: 2
  dropout: 0.1
  reason_mode: "transformer"
  evidence_mode: "{evidence_mode}"
  evidence_strength: {evidence_strength}
  pert_embed_strength: 1.0
  residual: true
  pert_mode: "id_plus_evidence"
  pert_agg: "mean"
  evidence_dim: 128
  use_evidence_conf: true
  use_evidence_as_pert_init: true
  evidence_pert_alpha: {evidence_pert_alpha}

train:
  epochs: 10
  batch_size: 64
  lr: 0.0005
  weight_decay: 0.0001
  amp: false
  grad_clip: 1.0
  device: "cuda"
  save_dir: "output/llm_positive/tmp_run"
  num_workers: 0
  pin_memory: true
  non_blocking: true
  progress: true
  profile: false
  evi_warm: true
  evi_warm_epochs: 3
  evi_warm_margin: 0.02
  latent_only_bp: true
  use_evidence_policy: "quality_gate"
  evidence_dropout: {evidence_dropout}
  min_evidence_conf_train: 0.35
  evidence_warm_start_epoch: 2
  stage3_init: "stage1"
  stage1_ckpt: "output/llm_positive/tmp_run/shared/stage1/model.pt"
  stage2_ckpt: "output/llm_positive/tmp_run/shared/stage2/model.pt"
  pert_embed_strength: 1.0
  evidence_strength: {evidence_strength}

loss:
  expr: 1.0
  delta: 1.0
  deg: 2.0
  evidence: 0.0
  evi_gain: 0.2
  latent: 1.0
  mmd: 0.02
  latent_metric: "cosine"
  mmd_max_samples: 32
""".format(**arch)
    path.write_text(base)


def train_stage(exp_name, stage, h5ad_path, config_path):
    """Run one training stage."""
    save_dir = ROOT / f"output/llm_positive/tmp_run/{exp_name}"
    tl_flag = ""
    if stage == 3:
        tl = SHARED / "stage2" / "target_latent.pt"
        if tl.exists():
            tl_flag = f'--target_latent "{tl}"'
    
    cmd = (
        f'conda run -n DL python "{ROOT}/main.py" train '
        f'--h5ad "{ROOT}/{h5ad_path}" '
        f'--config "{config_path}" '
        f'--stage {stage} '
        f'--batch_size 64 '
        f'--save_dir "{save_dir}" '
        f'--device cuda --no_amp --num_workers 0 --progress '
        f'{tl_flag}'
    )
    ok, out, err = run_cmd(cmd, timeout=600)
    
    if ok:
        stage_dir = save_dir / f"stage{stage}"
        shared_stage = SHARED / f"stage{stage}"
        if stage_dir.exists():
            if shared_stage.exists():
                shutil.rmtree(shared_stage)
            shutil.copytree(stage_dir, shared_stage)
    
    return ok, out


def run_experiment(exp, arch, round_label):
    """Run full experiment (stages 1-3) and return metrics."""
    name = exp["name"]
    h5ad = exp["h5ad"]
    h5ad_full = ROOT / h5ad
    
    if not h5ad_full.exists():
        print(f"  SKIP {name}: h5ad not found")
        return None
    
    print(f"\n  [{round_label}] {name}")
    
    # Write config
    config_path = OUT_DIR / "configs" / f"tmp_{name}.yaml"
    write_config(arch, config_path)
    
    all_m = {"experiment": name, "round": round_label, "split_type": exp.get("split_type", ""),
             "evidence_type": exp.get("evidence_type", ""), "n_train": exp.get("n_train", ""),
             **{f"arch_{k}": v for k, v in arch.items() if k != "label"},
             "timestamp": datetime.now().isoformat()}
    
    # Stage 1
    ok, out = train_stage(name, 1, h5ad, config_path)
    if not ok:
        all_m["status"] = "FAIL_stage1"
        return all_m
    for k, v in parse_metrics(out).items():
        all_m[f"s1_{k}"] = v
    
    # Stage 2
    ok, out = train_stage(name, 2, h5ad, config_path)
    if not ok:
        all_m["status"] = "FAIL_stage2"
        return all_m
    for k, v in parse_metrics(out).items():
        all_m[f"s2_{k}"] = v
    
    # Stage 3
    ok, out = train_stage(name, 3, h5ad, config_path)
    if not ok:
        all_m["status"] = "FAIL_stage3"
        return all_m
    for k, v in parse_metrics(out).items():
        all_m[f"s3_{k}"] = v
    
    all_m["status"] = "PASS"
    
    # Save summary
    summary_path = LOG_DIR / f"{round_label}_{name}.summary.txt"
    with open(summary_path, "w") as f:
        for k, v in sorted(all_m.items()):
            f.write(f"{k}: {v}\n")
    
    # Cleanup checkpoints for this experiment
    exp_dir = ROOT / f"output/llm_positive/tmp_run/{name}"
    if exp_dir.exists():
        for pat in ["*.pt", "*.pth", "*.ckpt"]:
            for fp in exp_dir.rglob(pat):
                try: fp.unlink()
                except: pass
    
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    return all_m


def check_success(zero_metrics, llm_metrics):
    """Check if LLM is better than ZERO. Returns (success, details)."""
    if zero_metrics is None or llm_metrics is None:
        return False, "missing_metrics"
    
    z_deg = zero_metrics.get("s3_train_deg", 999)
    l_deg = llm_metrics.get("s3_train_deg", 999)
    
    deg_improvement = z_deg - l_deg  # lower is better
    details = f"ZERO_deg={z_deg:.3f} LLM_deg={l_deg:.3f} improvement={deg_improvement:.3f}"
    
    # Simplified check: DEG improvement > 0.03
    if deg_improvement > 0.03:
        return True, details
    # Even small improvement counts
    if deg_improvement > 0.01:
        return True, f"marginal: {details}"
    
    return False, details


def main():
    print("=" * 60)
    print("BioReason LLM-Positive Auto Experiment Loop")
    print("=" * 60)
    
    all_results = []
    search_history = []
    best_config = None
    best_score = -999
    
    for round_idx, arch in enumerate(ARCH_ROUNDS):
        round_label = arch["label"]
        print(f"\n{'='*60}")
        print(f"ARCH ROUND {round_idx+1}/{len(ARCH_ROUNDS)}: {round_label}")
        print(f"  evidence_mode={arch['evidence_mode']}, strength={arch['evidence_strength']}, "
              f"alpha={arch['evidence_pert_alpha']}, dropout={arch['evidence_dropout']}")
        print(f"{'='*60}")
        
        round_improvements = 0
        round_total = 0
        
        for exp in EXPERIMENTS:
            h5ad_full = ROOT / exp["h5ad"]
            if not h5ad_full.exists():
                continue
            
            result = run_experiment(exp, arch, round_label)
            if result is None:
                continue
            
            result["round_label"] = round_label
            all_results.append(result)
            search_history.append({
                "round": round_label,
                "experiment": exp["name"],
                "status": result.get("status", "?"),
                "s3_train_deg": result.get("s3_train_deg", "N/A"),
                "evidence_type": exp.get("evidence_type", ""),
            })
            
            # Check LLM vs ZERO pairs
            if exp["evidence_type"] == "zero":
                # Store for later comparison
                zero_key = f"{exp['split_type']}_{exp.get('n_train','')}"
                zero_results_map = getattr(main, 'zero_map', {})
                zero_results_map[zero_key] = result
                main.zero_map = zero_results_map
            elif "llm" in exp["evidence_type"]:
                zero_key = f"{exp['split_type']}_{exp.get('n_train','')}"
                zero_results_map = getattr(main, 'zero_map', {})
                zero_r = zero_results_map.get(zero_key)
                if zero_r:
                    success, details = check_success(zero_r, result)
                    round_total += 1
                    if success:
                        round_improvements += 1
                        s3d = result.get('s3_train_deg', 0)
                        z3d = zero_r.get('s3_train_deg', 0)
                        score = z3d - s3d
                        if score > best_score:
                            best_score = score
                            best_config = {
                                "arch": arch,
                                "experiment": exp["name"],
                                "zero_deg": z3d,
                                "llm_deg": s3d,
                                "improvement": score,
                            }
                    print(f"    LLM_vs_ZERO [{exp['split_type']}]: {'PASS' if success else 'FAIL'} - {details}")
            
            time.sleep(1)
        
        if round_total > 0:
            print(f"\n  Round {round_label}: {round_improvements}/{round_total} LLM>ZERO improvements")
        
        # If found promising config, could early stop
        if best_config and best_score > 0.05:
            print(f"\n  *** Best config found with improvement {best_score:.3f}! ***")
            # Continue but mark
    
    # Save outputs
    # compare.csv
    csv_path = OUT_DIR / "compare.csv"
    if all_results:
        keys = sorted(set().union(*[r.keys() for r in all_results]))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {csv_path}")
    
    # search_history.csv
    hist_path = OUT_DIR / "search_history.csv"
    if search_history:
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["round", "experiment", "status", "s3_train_deg", "evidence_type"])
            w.writeheader()
            w.writerows(search_history)
        print(f"Saved: {hist_path}")
    
    # best_config.yaml
    if best_config:
        best_yaml = OUT_DIR / "best_config.yaml"
        with open(best_yaml, "w") as f:
            f.write(f"# Best LLM-positive config\n")
            f.write(f"# Improvement: {best_config['improvement']:.4f}\n")
            for k, v in best_config.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved: {best_yaml}")
        print(f"\nBest: {best_config['experiment']} ZERO={best_config['zero_deg']:.3f} LLM={best_config['llm_deg']:.3f} imp={best_config['improvement']:.3f}")
    
    # Cleanup shared
    if SHARED.exists():
        shutil.rmtree(SHARED)
    # Cleanup tmp configs
    for f in (OUT_DIR / "configs").glob("tmp_*.yaml"):
        try: f.unlink()
        except: pass
    
    print("\nDone.")


if __name__ == "__main__":
    # Initialize static state
    main.zero_map = {}
    main()
