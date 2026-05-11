# BioReason

**Evidence-guided latent biological reasoning for single-cell perturbation prediction.**

## Overview

BioReason learns to predict single-cell perturbation responses through an intermediate **latent biological reasoning state** (`z_bio`). Instead of mapping control expression + perturbation directly to perturbed expression, BioReason first reasons about the biological mechanism of perturbation response.

```
control expression + perturbation
        ↓
latent biological reasoning (z_bio)
        ↓
perturbed expression prediction
```

## Model Architecture

| Module | Role |
|--------|------|
| `CellEncoder` | Encodes gene expression into cell embedding |
| `PertEncoder` | Encodes perturbation condition (gene/drug/combo) |
| `Reasoner` | Multi-step latent biological reasoning engine |
| `EvidenceGate` | Injects biological evidence during training (bypassed at inference) |
| `ExprDecoder` | Decodes latent state to expression difference |
| `BioLoss` | Combined loss with DEG weighting, latent alignment, MMD |

## Three-Stage Training

**Stage 1 — Warm-up**
Basic perturbation prediction with expression + delta loss.

**Stage 2 — Evidence-Guided Latent Training**
Biological evidence (DEG, pathway, TF scores) supervises latent reasoning states. EvidenceGate injects evidence during training.

**Stage 3 — Evidence-Free Reasoning**
Evidence is removed. Model generates latent reasoning autonomously. Latent states aligned with Stage 2 teacher.

## Data Format

Input: h5ad (AnnData) with:
- `adata.X` — expression matrix [N cells x G genes]
- `adata.obs["perturbation"]` — perturbation labels
- `adata.obs["cell_type"]` — cell type (optional)
- `adata.obs["dose"]`, `adata.obs["time"]`, `adata.obs["batch"]` — metadata

## Quick Start

### Training

```bash
# Stage 1
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 1 --device cuda

# Stage 2
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 2 --device cuda

# Stage 3
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 3 --target_latent output/stage2/target_latent.pt --device cuda
```

### Inference

```bash
python main.py infer --config config/default.yaml --checkpoint output/stage3/model.pt --h5ad dataset/perturb.h5ad --pert TP53_KO --out output/infer
```

### Evaluation

```bash
python main.py eval --pred output/infer/pred.npz --truth dataset/perturb.h5ad --out output/eval
```

## Configuration

Edit config files in `config/`:
- `default.yaml` — project settings, data paths
- `model.yaml` — model architecture
- `train.yaml` — training hyperparameters
- `loss.yaml` — loss weights

## Environment

Copy `.env.example` to `.env` and set API keys:
```
OPENAI_API_KEY=
OPENAI_BASE_URL=
LLM_MODEL=
```
`.env` is git-ignored. Never commit real keys.

## Output

| File | Description |
|------|-------------|
| `output/stageN/model.pt` | Checkpoint |
| `output/stage2/target_latent.pt` | Teacher latent for Stage 3 |
| `output/infer/pred.npz` | Predictions |
| `output/eval/metrics.json` | Evaluation metrics |

**Note:** `dataset/` and `output/` are git-ignored (except README). Place data files locally.

## Inspiration

BioReason is inspired by latent reasoning paradigms. It adapts the principle of evidence-guided latent state learning to single-cell perturbation biology, where biological evidence shapes reasoning during training but is not required at inference.

## License

MIT
