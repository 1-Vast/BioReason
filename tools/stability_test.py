"""Seed 123 stability test."""
import subprocess, shutil, re
from pathlib import Path

ROOT = Path("D:/BioReason")
SHARED = ROOT / "output/llm_positive/tmp_run/shared"
SHARED.mkdir(parents=True, exist_ok=True)
config = ROOT / "output/llm_positive/configs/config_llm_positive.yaml"

def train_stage(name, stage, h5ad, save):
    tl = ""
    if stage == 3:
        tl_path = SHARED / "stage2" / "target_latent.pt"
        if tl_path.exists():
            tl = f'--target_latent "{tl_path}"'
    cmd = (
        f'conda run -n DL python "{ROOT}/main.py" train '
        f'--h5ad "{h5ad}" --config "{config}" '
        f'--stage {stage} --batch_size 64 '
        f'--save_dir "{save}" --device cuda --no_amp --num_workers 0 --progress '
        f'{tl}'
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    if r.returncode == 0:
        src = save / f"stage{stage}"
        dst = SHARED / f"stage{stage}"
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    return r.returncode == 0, r.stdout

exps = [
    ("s123_heldout_zero", ROOT / "dataset/processed/llm_seed123_heldout_zero.h5ad"),
    ("s123_heldout_llm", ROOT / "dataset/processed/llm_seed123_heldout_dataset_struct_llm.h5ad"),
]

for exp_name, h5ad_path in exps:
    save = ROOT / f"output/llm_positive/tmp_run/{exp_name}"
    for stage in [1, 2, 3]:
        ok, out = train_stage(exp_name, stage, h5ad_path, save)
        if not ok:
            print(f"{exp_name} stage {stage}: FAIL")
            break
    else:
        # Parse final deg
        for line in out.split("\n"):
            m = re.search(r'deg=([\d.]+)', line)
            if m and "train" in line:
                deg = float(m.group(1))
        print(f"{exp_name}: s3_train_deg={deg:.3f}")

if SHARED.exists():
    shutil.rmtree(SHARED)
print("Seed 123 done")
