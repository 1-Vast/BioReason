"""Focused stability experiment: config A across 3 seeds, heldout + lowcell_10 + lowcell_50."""
import subprocess, shutil, re, gc, csv, time
from pathlib import Path
from datetime import datetime

ROOT = Path("D:/BioReason")
OUT_DIR = ROOT / "output/llm_stability"
LOG_DIR = OUT_DIR / "logs"
SHARED = OUT_DIR / "tmp_run/shared"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = ["42", "123", "2024"]
SPLITS = ["heldout", "lowcell_10", "lowcell_50"]
EVIDENCE = ["zero", "struct_llm", "hybrid_llm"]

def write_config(path):
    path.write_text("""data:
  h5ad: ""
  label_key: "perturbation"
  control_label: "control"
  cell_type_key: "cell_type"
  evidence_key: "evidence"
  target_mode: "group_mean"
  pair_by: "cell_type"
  use_control_as_input: true
  use_hvg: true
  n_hvg: 1200
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
  evidence_mode: "gate_add"
  evidence_strength: 0.4
  pert_embed_strength: 1.0
  residual: true
  pert_mode: "id_plus_evidence"
  pert_agg: "mean"
  evidence_dim: 128
  use_evidence_conf: true
  use_evidence_as_pert_init: true
  evidence_pert_alpha: 0.8
  adaptive_evidence_gate: true
  evidence_gate_init_bias: -1.5

train:
  epochs: 8
  batch_size: 256
  lr: 0.0005
  weight_decay: 0.0001
  amp: true
  grad_clip: 1.0
  device: "cuda"
  save_dir: "output/llm_stability/tmp_run"
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
  non_blocking: true
  progress: true
  profile: true
  compile: true
  evi_warm: true
  evi_warm_epochs: 3
  evi_warm_margin: 0.02
  latent_only_bp: true
  use_evidence_policy: "quality_gate"
  evidence_dropout: 0.1
  min_evidence_conf_train: 0.35
  evidence_warm_start_epoch: 2
  stage3_init: "stage1"
  stage1_ckpt: "output/llm_stability/tmp_run/shared/stage1/model.pt"
  stage2_ckpt: "output/llm_stability/tmp_run/shared/stage2/model.pt"

loss:
  expr: 1.0
  delta: 1.0
  deg: 2.0
  evidence: 0.0
  evi_gain: 0.2
  evi_contrast: 0.05
  evi_contrast_temp: 0.2
  latent: 1.0
  mmd: 0.02
  latent_metric: "cosine"
  mmd_max_samples: 32
""")

def run_cmd(cmd, timeout=600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, r.stdout, r.stderr

def train_stage(name, stage, h5ad_path, config_path):
    save = ROOT / f"output/llm_stability/tmp_run/{name}"
    tl = ""
    if stage == 3:
        tp = SHARED / "stage2" / "target_latent.pt"
        if tp.exists():
            tl = f'--target_latent "{tp}"'
    cmd = (
        f'conda run -n DL python "D:/BioReason/main.py" train '
        f'--h5ad "{h5ad_path}" --config "{config_path}" '
        f'--stage {stage} --batch_size 256 '
        f'--save_dir "{save}" --device cuda --amp --num_workers 4 --progress --compile '
        f'{tl}'
    )
    ok, out, err = run_cmd(cmd, timeout=900)
    if ok:
        src = save / f"stage{stage}"
        dst = SHARED / f"stage{stage}"
        if src.exists():
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(src, dst)
    return ok, out

def parse_deg(stdout):
    for line in reversed(stdout.split("\n")):
        m = re.search(r'deg=([\d.]+)', line)
        if m and "train" in line:
            return float(m.group(1))
    return -1

config_path = OUT_DIR / "configs/config_A.yaml"
write_config(config_path)

all_results = []

for seed in SEEDS:
    print(f"\n{'#'*50}")
    print(f"SEED {seed}")
    print(f"{'#'*50}")
    
    for split in SPLITS:
        print(f"\n  SPLIT: {split}")
        
        # Clear shared between splits
        if SHARED.exists():
            shutil.rmtree(SHARED)
        SHARED.mkdir(parents=True, exist_ok=True)
        
        for ev in EVIDENCE:
            h5ad = ROOT / f"dataset/processed/stable_seed{seed}_{split}_{ev}.h5ad"
            if not h5ad.exists():
                print(f"    SKIP {ev}: file not found")
                continue
            
            name = f"s{seed}_{split}_{ev}"
            exp_save = ROOT / f"output/llm_stability/tmp_run/{name}"
            
            ok = True
            for stage in [1, 2, 3]:
                ok, out = train_stage(name, stage, h5ad, config_path)
                if not ok:
                    print(f"    {ev} stage{stage}: FAIL")
                    break
            
            deg = parse_deg(out) if ok else -1
            tag = "FAIL" if not ok else "OK"
            print(f"    {ev}: {tag} deg={deg:.3f}" if deg > 0 else f"    {ev}: {tag}")
            
            result = {"seed": seed, "split": split, "evidence": ev,
                      "config": "A", "status": "PASS" if ok else "FAIL",
                      "s3_train_deg": deg, "timestamp": datetime.now().isoformat()}
            all_results.append(result)
            
            # Cleanup checkpoints for this experiment
            if exp_save.exists():
                for pat in ["*.pt", "*.pth", "*.ckpt"]:
                    for fp in exp_save.rglob(pat):
                        try: fp.unlink()
                        except: pass
            
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except: pass
            
            time.sleep(0.5)

# Compute ZERO vs LLM deltas
zero_map = {}
for r in all_results:
    if r["evidence"] == "zero" and r["status"] == "PASS":
        key = f"{r['seed']}_{r['split']}"
        zero_map[key] = r["s3_train_deg"]

for r in all_results:
    if r["evidence"] != "zero" and r["status"] == "PASS":
        key = f"{r['seed']}_{r['split']}"
        zd = zero_map.get(key, 0)
        r["zero_deg"] = zd
        r["delta_deg"] = zd - r["s3_train_deg"] if zd > 0 else 0

# Save results
csv_path = OUT_DIR / "results.csv"
if all_results:
    keys = ["seed", "split", "evidence", "config", "status", "s3_train_deg", "delta_deg", "timestamp"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

# Print summary
print(f"\n{'='*60}")
print("STABILITY SUMMARY")
print(f"{'='*60}")
pos = sum(1 for r in all_results if r.get("delta_deg", 0) > 0)
tot = sum(1 for r in all_results if "delta_deg" in r)
print(f"LLM > ZERO: {pos}/{tot} ({100*pos/max(tot,1):.0f}%)")

for seed in SEEDS:
    sd = [r for r in all_results if r["seed"] == seed and "delta_deg" in r]
    if sd:
        mean_d = sum(r["delta_deg"] for r in sd) / len(sd)
        p = sum(1 for r in sd if r["delta_deg"] > 0)
        print(f"  Seed {seed}: mean delta={mean_d:.3f}, pos={p}/{len(sd)}")

print(f"\nSaved: {csv_path}")

if SHARED.exists():
    shutil.rmtree(SHARED)
print("Done")
