#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/BioReason

OUT="output/llm_stability_remote"
LOG="$OUT/logs"
PY="${PYTHON_BIN:-/root/miniconda3/bin/python}"
export PATH="/root/miniconda3/bin:$PATH"
mkdir -p "$LOG" "$OUT/audit" "$OUT/evidence_audit"

{
  echo "== repo =="
  pwd
  git status --short || true
  git rev-parse HEAD || true
  echo "== python =="
  "$PY" --version
  "$PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
PY
  echo "== packages =="
  "$PY" - <<'PY'
for pkg in ["numpy", "pandas", "scipy", "sklearn", "anndata", "scanpy", "openai", "yaml", "tqdm"]:
    try:
        m = __import__(pkg)
        print(pkg, "ok", getattr(m, "__version__", ""))
    except Exception as e:
        print(pkg, "MISSING", e)
PY
} | tee "$LOG/env_check.txt"

"$PY" - <<'PY'
missing = []
for pkg, pip_name in [("scanpy", "scanpy"), ("anndata", "anndata"), ("sklearn", "scikit-learn"), ("openai", "openai"), ("yaml", "pyyaml"), ("tqdm", "tqdm")]:
    try:
        __import__(pkg)
    except Exception:
        missing.append(pip_name)
if missing:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "python-dotenv"])
PY

nvidia-smi | tee "$LOG/gpu_before.txt"

ls -lh dataset/raw/adamson_10X010.h5ad dataset/raw/simulated_multicondition.h5ad | tee "$LOG/data_files.txt"
md5sum dataset/raw/adamson_10X010.h5ad dataset/raw/simulated_multicondition.h5ad | tee "$LOG/data_md5.txt"
grep -q "2fa44ea61a8dd35742af618638ec65fc" "$LOG/data_md5.txt"
grep -q "16f0c511595fd11fc1759d90d129be01" "$LOG/data_md5.txt"

if ! ls dataset/processed/remote_stable_seed42_heldout.h5ad >/dev/null 2>&1; then
  mkdir -p dataset/processed
  "$PY" tools/make_llm_stable_bench.py \
    --h5ad dataset/raw/adamson_10X010.h5ad \
    --out_dir dataset/processed \
    --prefix remote_stable \
    --top_perts 40 \
    --max_cells 20000 \
    --n_hvg 2000 \
    --min_cells_per_pert 80 \
    --heldout_perts 8 \
    --lowcell_list 5,10,20,50 \
    --seeds 42,123,2024 \
    --report "$OUT/preprocess_report.md" || {
      echo "Primary preprocessing failed; falling back to smaller benchmark." | tee "$OUT/preprocess_downgrade.txt"
      "$PY" tools/make_llm_stable_bench.py \
        --h5ad dataset/raw/adamson_10X010.h5ad \
        --out_dir dataset/processed \
        --prefix remote_stable \
        --top_perts 25 \
        --max_cells 12000 \
        --n_hvg 1500 \
        --min_cells_per_pert 80 \
        --heldout_perts 5 \
        --lowcell_list 5,10,20,50 \
        --seeds 42,123,2024 \
        --report "$OUT/preprocess_report.md"
    }
fi

"$PY" tools/run_llm_stability.py \
  --processed_dir dataset/processed \
  --prefix remote_stable \
  --out_dir "$OUT" \
  --seeds 42,123,2024 \
  --splits heldout,lowcell_5,lowcell_10,lowcell_20,lowcell_50 \
  --evidence zero,struct_llm,hybrid_llm

"$PY" tools/summarize_llm_stability.py \
  --results "$OUT/results.csv" \
  --out_dir "$OUT"

find "$OUT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.npz' -o -name 'pred*.h5ad' \) -delete
rm -rf "$OUT/tmp_run" "$OUT"/runs/*/stage1 "$OUT"/runs/*/stage2 "$OUT"/runs/*/stage3

echo "Remote LLM stability run complete: $OUT/report.md"
