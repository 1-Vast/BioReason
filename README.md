# BioReason

BioReason is a single-cell perturbation response model. It predicts perturbed expression from a control expression profile and a perturbation label, with optional offline biological evidence used during training.

The main principle is simple: keep data preparation, evidence construction, training, inference, evaluation, and leakage audit as separate steps.

## Pipeline

1. Stage 0: offline evidence preprocessing
2. Stage 1: perturbation prediction warm-up without evidence
3. Stage 2: evidence-guided training and target latent export
4. Stage 3: evidence-free latent alignment
5. Inference: counterfactual prediction without LLM calls
6. Evaluation: index-aligned, split-aware metrics
7. Audit: strict leakage checks before trusting results

LLM calls are never used in `model.forward`, training, inference, or evaluation. LLM fallback is only allowed in Stage 0 when `--use_llm` is explicitly passed.

## Data Preparation

Raw h5ad files should be converted into a model-ready file before training.

Recommended steps:

```bash
python main.py prior \
  --h5ad dataset/raw.h5ad \
  --out dataset/perturb_evi.h5ad \
  --kb dataset/kb/prior.json \
  --evidence_dim 128 \
  --encoder hash
```

Optional LLM fallback is budgeted and cached:

```bash
python main.py prior \
  --h5ad dataset/raw.h5ad \
  --out dataset/perturb_evi.h5ad \
  --kb dataset/kb/prior.json \
  --use_llm \
  --max_llm_calls 20 \
  --max_llm_tokens 10000 \
  --llm_max_tokens 512 \
  --llm_cache output/llm_cache.json
```

Dry-run before spending API budget:

```bash
python main.py prior \
  --h5ad dataset/raw.h5ad \
  --out dataset/perturb_evi_dry.h5ad \
  --kb dataset/kb/prior.json \
  --use_llm \
  --dry_run
```

Stage 0 writes:

- `adata.obsm["evidence"]`
- `adata.obs["evidence_conf"]`
- `adata.obs["evidence_source"]`
- prior audit CSV/JSON metadata

## Train/Validation/Test Policy

Training and validation datasets are built from AnnData-level splits before any fitted statistic is computed.

Protected against leakage:

- HVG selection uses train cells only.
- perturbation group means use train cells only.
- validation data is aligned to the train gene order.
- model perturbation and covariate dimensions come from train metadata.
- Stage 3 target latent masks are valid only for train source indices.
- evaluation aligns predictions to saved source indices and defaults to the test split.

If `adata.obs["split"]` contains `train` and `val`, those labels are used. If no validation split exists, BioReason creates a seeded train/val split before constructing datasets.

For reliable reporting, keep a separate test set or a separate small evaluation h5ad that is never used for training.

## Training

Stage 1:

```bash
python main.py train --h5ad dataset/perturb_evi.h5ad --stage 1 --device cuda
```

Stage 2:

```bash
python main.py train --h5ad dataset/perturb_evi.h5ad --stage 2 --device cuda
```

Stage 3:

```bash
python main.py train \
  --h5ad dataset/perturb_evi.h5ad \
  --stage 3 \
  --target_latent output/stage2/target_latent.pt \
  --device cuda
```

The CLI keeps only important runtime overrides. Main defaults are in YAML configs. GPU is the default device, AMP is enabled by default, and the default training data loader is configured for higher throughput:

- `batch_size: 256`
- `num_workers: 2`
- `pin_memory: true`
- `persistent_workers: true`
- `prefetch_factor: 4`
- `progress: false`

Use `--batch_size` and config `num_workers` to tune GPU occupancy. Low GPU utilization usually means the model is small, batch size is too low, CPU data loading is slow, or Stage 2/3 two-step latent-only updates add synchronization overhead.

## Inference

Inference never uses evidence or LLM calls.

```bash
python main.py infer \
  --checkpoint output/stage3/model.pt \
  --h5ad dataset/eval.h5ad \
  --pert TP53_KO \
  --out output/infer \
  --device cuda \
  --batch_size 512
```

Inference output is intentionally different from training output. It reports data size, device, completion, and saved files.

## Evaluation

Evaluation uses prediction indices saved in `pred.npz`. By default it evaluates only `obs["split"] == "test"`.

```bash
python main.py eval \
  --pred output/infer/pred.npz \
  --truth dataset/eval.h5ad \
  --split test \
  --out output/eval
```

Use `--split all` only for debugging. Do not report train-split metrics as final performance.

## Leakage Audit

Run a strict audit before trusting high metrics:

```bash
python main.py audit \
  --h5ad dataset/perturb_evi.h5ad \
  --target_latent output/stage2/target_latent.pt \
  --report output/leak_audit.json
```

The audit checks split existence, split disjointness, HVG metadata, group-mean source, target latent indices, evidence source, and LLM audit fields.

## Output Style

Training output is epoch-level and compact:

```text
[train] BioReason
[train] stage=2
[train] ep 003/100 | stg2-warm | train loss=0.4210 deg=0.1300 evi_gain=0.0410 z_shift=0.3700 | val loss=0.4380 deg=0.1410 | 2200 cells/s | gpu 6.1GB
```

Inference output is task-level:

```text
[infer] cells=12000 genes=2000 latent=256 device=cuda amp=True
[infer] done cells=12000 genes=2000 mem_est=0.2GB
[infer] saved=output/infer/pred.npz, output/infer/pred.h5ad
```

## Speed Checklist

Training speed:

- increase batch size until GPU memory is near the desired limit
- use AMP on CUDA
- use `pin_memory`, `non_blocking`, and data loader workers
- keep `progress: false` for clean logs and less terminal overhead
- use Stage 1 for fast warm-up; Stage 2/3 are slower because latent-only BP does extra forward/backward passes

Inference speed:

- use larger batch size than training when memory allows
- keep evidence disabled
- use memmap for very large outputs
- avoid writing h5ad if only npz metrics are needed

## Checks and Tools

Verification entrypoint:

```bash
python tools/check.py
```

All verification checks now live under `tools/checks`; the old `tests` directory has been removed.

Main tool files:

- `tools/prep.py`: Stage 0 evidence preprocessing
- `tools/kb.py`: local biological prior lookup
- `tools/evi.py`: evidence validation, encoding, and audit metadata
- `tools/text.py`: hash/sentence text encoder
- `tools/split.py`: train/val/test split annotation
- `tools/audit_leak.py`: leakage audit
- `tools/eval_deg.py`: DEG-oriented evaluation
- `tools/check.py`: verification runner

## API Safety

Do not put API keys in code, README, audit logs, or committed files. Use `.env` locally. LLM use is disabled by default and only applies to Stage 0 prior construction.
