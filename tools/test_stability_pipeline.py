"""Quick single-experiment test to verify stability pipeline."""
import subprocess, shutil, re, gc
from pathlib import Path

ROOT = Path("D:/BioReason")
SHARED = ROOT / "output/llm_stability/tmp_run/shared"
SHARED.mkdir(parents=True, exist_ok=True)

config = ROOT / "output/llm_stability/configs/test_config.yaml"

# Write test config
config.write_text("""# Test config
data:
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
  epochs: 3
  batch_size: 64
  lr: 0.0005
  weight_decay: 0.0001
  amp: false
  grad_clip: 1.0
  device: "cuda"
  save_dir: "output/llm_stability/tmp_run"
  num_workers: 0
  pin_memory: true
  non_blocking: true
  progress: true
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

h5ad = ROOT / "dataset/processed/stable_seed42_heldout_zero.h5ad"
save = ROOT / "output/llm_stability/tmp_run/test_run"

for stage in [1, 2, 3]:
    tl = ""
    if stage == 3:
        tp = SHARED / "stage2" / "target_latent.pt"
        if tp.exists():
            tl = f'--target_latent "{tp}"'
    
    cmd = (
        f'conda run -n DL python "D:/BioReason/main.py" train '
        f'--h5ad "{h5ad}" --config "{config}" '
        f'--stage {stage} --batch_size 64 '
        f'--save_dir "{save}" --device cuda --no_amp --num_workers 0 --progress '
        f'{tl}'
    )
    print(f"Stage {stage}...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    if r.returncode == 0:
        print(f"  Stage {stage} OK")
        # Copy to shared
        src = save / f"stage{stage}"
        dst = SHARED / f"stage{stage}"
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    else:
        print(f"  Stage {stage} FAILED")
        print(f"  STDERR: {r.stderr[-500:]}")
        break

# Parse final metrics
for line in r.stdout.split("\n") if r.returncode == 0 else []:
    m = re.search(r'deg=([\d.]+)', line)
    if m and "train" in line:
        print(f"Final train deg: {float(m.group(1)):.3f}")
        break

if SHARED.exists():
    shutil.rmtree(SHARED)
print("Test done")
