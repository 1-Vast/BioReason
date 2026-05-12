# BioReason

**Evidence-guided latent biological reasoning for single-cell perturbation prediction.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Model Framework](#model-framework)
- [Architecture](#architecture)
- [Three-Stage Training](#three-stage-training)
- [Installation](#installation)
- [Data Format](#data-format)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [API Reference](#api-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Output Files](#output-files)
- [Tests](#tests)
- [License](#license)
- [Citation](#citation)

---

## Overview

**BioReason** predicts single-cell perturbation responses without relying on biological evidence at inference time. Instead of a direct `x → y` mapping, it introduces an intermediate **latent biological reasoning state** (`z_bio`) that captures the mechanism of perturbation response — pathway shifts, TF activity changes, gene module responses, GRN propagation, cell-state transitions.

```
control expression + perturbation
        ↓
latent biological reasoning (z_bio)
        ↓
perturbed expression prediction
```

**Key innovation**: During training, optional biological evidence (DEG, pathway scores, TF activity, gene module scores) can supervise and shape the latent reasoning states. During inference, evidence is **not required** — the model generates `z_bio` autonomously.

---

## Motivation

Existing perturbation prediction models face two limitations:

1. **Direct mapping** (`x, pert → y`) lacks interpretable intermediate states.
2. **Evidence-dependent models** require auxiliary data (DEG, pathway) at inference time, limiting practical deployment.

BioReason addresses both:

- **Interpretable bottleneck**: `z_bio` represents the biological mechanism of perturbation.
- **Evidence-free inference**: Evidence only used during training for distillation; inference uses only `x + pert`.

---

## Model Framework

### Forward Pass

```
┌─────────────────────────────────────────────────────────────────┐
│                          BioReason                              │
│                                                                 │
│  x [B,G] ──► CellEncoder ──► cell_emb [B,D] ──┐                │
│                                                 │               │
│  pert ─────► PertEncoder ──► pert_emb [B,D] ──►├──► context ──► │
│                                                 │               │
│  cov ──────► CovEncoder ───► cov_emb  [B,D] ──┘               │
│                                                                 │
│                       ┌──────────────────┐                      │
│  evidence [B,E] ──►──►   EvidenceGate    │  (train only)        │
│                       └────────┬─────────┘                      │
│                                │                                │
│                       ┌────────▼─────────┐                      │
│                       │     Reasoner     │  × N steps           │
│                       │  ReasonStep × N  │                      │
│                       └────────┬─────────┘                      │
│                                │                                │
│                                ▼  z_bio [B,D]                   │
│                       ┌──────────────────┐                      │
│  cell_emb ───────────►│   ExprDecoder    │                      │
│                       └────────┬─────────┘                      │
│                                │                                │
│                                ▼  delta [B,G]                   │
│                       ┌──────────────────┐                      │
│  x ──────────────────►│  pred = x + delta│                      │
│                       └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### Training vs Inference

```
          TRAINING                          INFERENCE
          ────────                          ─────────
   x + pert + evidence              x + pert (no evidence)
         │                                   │
         ▼                                   ▼
     Reasoner                             Reasoner
         │                                   │
         ▼                                   ▼
   ExprDecoder                          ExprDecoder
         │                                   │
    pred + L_evidence                       pred
```

---

## Architecture

### Core Modules

| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| `CellEncoder` | `x [B,G]` | `cell_emb [B,D]` | MLP encoder with LayerNorm, maps gene expression to cell embedding |
| `PertEncoder` | `pert [B]` or `[B,N]` or `[B,F]` | `pert_emb [B,D]` | Supports categorical id, multi-hot gene vector, continuous descriptor. Aggregation: mean / sum / attention |
| `CovEncoder` | `cov` dict | `cov_emb [B,D]` | Embeds cell_type, dose, time, batch metadata |
| `Reasoner` | `cell_emb + pert_emb + cov_emb` | `z_bio [B,D]` | Multi-step reasoning engine. Shared-context FiLM + self-attention at each step |
| `ReasonStep` | `z [B,D], context [B,D]` | `z' [B,D]` | Single reasoning step: FiLM modulation → Transformer/MLP/GRU update |
| `EvidenceGate` | `z [B,D], evidence [B,D]` | `z' [B,D]` | FiLM/additive/cross-attention gate. Idempotent when `evidence=None` |
| `ExprDecoder` | `cell_emb [B,D] + z_bio [B,D]` | `delta [B,G]` | 3-layer MLP, predicts expression difference |
| `BioLoss` | `out, batch, stage` | `loss dict` | expr + delta + DEG-weighted + latent-alignment + evidence + MMD |

### BioReason Constructor

```python
BioReason(
    input_dim=2000,     # number of genes
    dim=256,            # latent dimension
    hidden=512,         # MLP hidden dim
    steps=8,            # reasoning steps
    heads=4,            # attention heads
    dropout=0.1,
    residual=True,      # pred = x + delta
    pert_mode="id",     # "id" | "multihot" | "continuous"
    pert_agg="mean",    # "mean" | "sum" | "attention"
    num_perts=10000,
)
```

### BioReason.forward()

```python
def forward(self, x, pert, cov=None, evidence=None, target_latent=None,
            return_latent=True, detach_latent=False) -> dict:
    """
    Returns: {"pred": [B,G], "delta": [B,G], "latent": [B,D], "evidence_pred": [B,D] or None}
    """
```

---

## Three-Stage Training

### Stage 1 — Warm-up

Simple perturbation prediction. No evidence. Model learns basic expression patterns.

```
Loss = L_expr + L_delta + L_deg
```

### Stage 2 — Evidence-Guided Latent Distillation

Biological evidence (DEG, pathway scores, TF activity) is injected via `EvidenceGate`. Model learns to represent biological mechanisms in `z_bio`. Teacher latent states are saved for Stage 3.

```
Loss = L_expr + L_delta + L_deg + L_evidence + L_mmd
```

### Stage 3 — Evidence-Free Alignment

Evidence removed. Student latent states are aligned with teacher latents from Stage 2. Model learns autonomous reasoning.

```
Loss = L_expr + L_delta + L_deg + L_latent_align + L_mmd
```

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 1.13+
- CUDA (optional, for GPU training)

### Setup

```bash
git clone git@github.com:1-Vast/BioReason.git
cd BioReason
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.13
numpy
pandas
scipy
scikit-learn
anndata
scanpy
pyyaml
python-dotenv
tqdm
```

---

## Data Format

### h5ad (AnnData)

| Component | Type | Required | Description |
|-----------|------|----------|-------------|
| `adata.X` | `[N, G]` sparse/dense | Yes | Expression matrix |
| `adata.obs["perturbation"]` | str | Yes | Perturbation label (e.g., "TP53_KO", "control") |
| `adata.obs["cell_type"]` | str | No | Cell type annotation |
| `adata.obs["dose"]` | float/str | No | Perturbation dose |
| `adata.obs["time"]` | float/str | No | Time point |
| `adata.obs["batch"]` | str | No | Batch ID |
| `adata.obsm["evidence"]` | `[N, E]` | No | Precomputed evidence matrix |

### Control Label

Default: `"control"`. Configurable via `config/default.yaml` → `data.control_label`.

### PertDataset Output

```python
{
    "x":        torch.Tensor [G],    # expression
    "y":        torch.Tensor [G],    # target expression
    "pert":     int,                 # perturbation id
    "pert_str": str,                 # perturbation label
    "cov":      dict[str, Tensor],   # covariate values
    "evidence": Tensor [E] or None,  # biological evidence
    "meta":     dict,                # index, is_control
}
```

---

## Quick Start

### 1. Prepare Data

Place your h5ad file in `dataset/`:

```
dataset/perturb.h5ad
```

### 2. Configure

Edit `config/default.yaml` to match your data columns, or create a custom config.

### 3. Train

```bash
# Stage 1: warm-up
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 1 --device cuda

# Stage 2: evidence-guided
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 2 --device cuda

# Stage 3: evidence-free alignment
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 3 --target_latent output/stage2/target_latent.pt --device cuda
```

### 4. Infer

```bash
python main.py infer --config config/default.yaml --checkpoint output/stage3/model.pt --h5ad dataset/perturb.h5ad --pert TP53_KO --out output/infer
```

### 5. Evaluate

```bash
python main.py eval --pred output/infer/pred.npz --truth dataset/perturb.h5ad --out output/eval
```

### 6. Verify

```bash
python tests/check.py
```

---

## Configuration

All config in `config/`. Files are merged in order, later values override earlier.

### config/default.yaml

```yaml
project:
  name: "BioReason"
  task: "single-cell perturbation prediction"

data:
  h5ad: ""
  label_key: "perturbation"
  control_label: "control"
  cell_type_key: "cell_type"
  dose_key: "dose"
  time_key: "time"
  batch_key: "batch"
  use_hvg: true
  n_hvg: 2000

eval:
  top_deg: 50
```

### config/model.yaml

```yaml
model:
  input_dim: 2000
  dim: 256
  hidden: 512
  latent_steps: 8
  heads: 4
  dropout: 0.1
  residual: true
  pert_mode: "id"
  pert_agg: "mean"
```

### config/train.yaml

```yaml
train:
  stage: 1
  epochs: 100
  batch_size: 128
  lr: 0.0005
  weight_decay: 0.0001
  amp: true
  grad_clip: 1.0
  device: "cuda"
  seed: 42
  save_dir: "output"
```

### config/loss.yaml

```yaml
loss:
  expr: 1.0
  delta: 1.0
  deg: 2.0
  latent: 1.0
  evidence: 1.0
  mmd: 0.1
```

---

## CLI Reference

### `python main.py train`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config/default.yaml` | Config path |
| `--h5ad` | str | (required) | Data file |
| `--stage` | int | `1` | 1, 2, or 3 |
| `--device` | str | `cuda` | `cuda` or `cpu` |
| `--save_dir` | str | (from config) | Output directory |
| `--target_latent` | str | `None` | Teacher latent path (stage 3) |

### `python main.py infer`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config/default.yaml` | Config path |
| `--checkpoint` | str | (required) | Model `.pt` file |
| `--h5ad` | str | (required) | Data file |
| `--pert` | str | `None` | Perturbation label |
| `--out` | str | `output/infer` | Output directory |
| `--device` | str | `cuda` | `cuda` or `cpu` |
| `--batch_size` | int | `128` | Batch size |

### `python main.py eval`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config/default.yaml` | Config path |
| `--pred` | str | (required) | Predictions `.npz` file |
| `--truth` | str | (required) | Ground truth `.h5ad` file |
| `--out` | str | `output/eval` | Output directory |
| `--top_deg` | int | `50` | Top N DEGs |

### `python tests/check.py`

Runs import + forward + loss verification. No arguments.

---

## API Reference

### Python Usage

```python
from models import BioReason, BioLoss
from models.data import build_dataset
from models.train import train_model

# Build model
model = BioReason(input_dim=2000, dim=256, num_perts=100)

# Forward
out = model(x, pert, evidence=None)
# out["pred"] [B,G], out["delta"] [B,G], out["latent"] [B,D]

# Inference without evidence
z = model.encode(x, pert)
pred = model.predict(x, z)

# Loss
loss_fn = BioLoss()
losses = loss_fn(out, batch, stage=1)
# losses["loss"], losses["expr"], losses["delta"], losses["deg"], ...
```

### Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `BioReason` | `models.reason` | Full model: encode → reason → decode |
| `Reasoner` | `models.reason` | Multi-step latent reasoning engine |
| `ReasonStep` | `models.reason` | Single reasoning step (FiLM + attn) |
| `EvidenceGate` | `models.reason` | Evidence injection (idempotent if None) |
| `BioLoss` | `models.loss` | 6-loss composite with stage gating |
| `CellEncoder` | `models.cell` | Expression → embedding |
| `PertEncoder` | `models.pert` | Perturbation → embedding |
| `ExprDecoder` | `models.decoder` | Embedding pair → delta |
| `PertDataset` | `models.data` | h5ad → PyTorch Dataset |
| `MLP` | `models.base` | Configurable MLP block |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `mse` | Mean squared error (cell × gene) |
| `mae` | Mean absolute error |
| `pearson` | Per-gene Pearson correlation, averaged |
| `spearman` | Per-gene Spearman correlation, averaged |
| `r2` | Per-gene R² score, averaged |
| `deg_pearson` | Pearson on top-K DEGs only |
| `top_deg_overlap` | Jaccard of top-K DEG sets |
| `mmd` | Maximum Mean Discrepancy (latent distribution) |

Output: `output/eval/metrics.json` + `metrics.csv`.

---

## Project Structure

```
BioReason/
│
├── main.py                 # CLI entry (train / infer / eval)
├── README.md               # ← this file
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore              # Excludes .env, data, outputs, binaries
├── LICENSE                 # MIT
│
├── models/                 # Core model logic (NO CLI code)
│   ├── __init__.py         # Public API exports
│   ├── reason.py           # ★ BioReason, Reasoner, EvidenceGate, ReasonStep
│   ├── base.py             # MLP, ResidualBlock, LayerNormBlock, EmbeddingBlock
│   ├── cell.py             # CellEncoder, CovEncoder
│   ├── pert.py             # PertEncoder (id / multihot / continuous)
│   ├── latent.py           # LatentBlock, FiLM, CrossBlock
│   ├── decoder.py          # ExprDecoder
│   ├── loss.py             # BioLoss (6 losses, stage-gated)
│   ├── data.py             # PertDataset, build_dataset, build_loader, split_data
│   ├── train.py            # train_model, train_epoch, save_ckpt, load_ckpt
│   ├── infer.py            # load_model, predict, save_pred
│   └── eval.py             # evaluate, save_metrics, all metric functions
│
├── utils/                  # Shared utilities
│   ├── __init__.py
│   └── config.py           # YAML loading, .env loading, config merging
│
├── config/                 # YAML configuration files
│   ├── default.yaml        # Project, data, eval settings
│   ├── model.yaml          # Model hyperparameters
│   ├── train.yaml          # Training hyperparameters
│   └── loss.yaml           # Loss weights
│
├── scripts/                # Convenience batch launchers
│   ├── run_train.bat       # Stage 1 training
│   ├── run_infer.bat       # Inference
│   └── run_eval.bat        # Evaluation
│
├── tests/
│   └── check.py            # Import + forward + loss verification
│
├── dataset/                # Data files (git-ignored except README)
│   ├── README.md           # Data format documentation
│   └── .gitkeep
│
└── output/                 # Results (git-ignored except README)
    ├── README.md           # Output structure documentation
    └── .gitkeep
```

---

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Variables:

| Key | Description |
|-----|-------------|
| `OPENAI_API_KEY` | API key for LLM features |
| `OPENAI_BASE_URL` | API endpoint |
| `OPENAI_MODEL` | Model name |
| `LLM_PROVIDER` | Provider identifier |
| `LLM_BASE_URL` | Alternative LLM endpoint |
| `LLM_API_KEY` | Alternative API key |
| `LLM_MODEL` | Alternative model name |

> **IMPORTANT**: Never commit `.env` to version control. It is listed in `.gitignore`.

---

## Output Files

| File | Stage | Description |
|------|-------|-------------|
| `output/stage1/model.pt` | Train | Stage 1 checkpoint |
| `output/stage2/model.pt` | Train | Stage 2 checkpoint |
| `output/stage2/target_latent.pt` | Train | Teacher latent for Stage 3 |
| `output/stage3/model.pt` | Train | Stage 3 checkpoint |
| `output/infer/pred.npz` | Infer | Predicted expressions |
| `output/infer/pred.h5ad` | Infer | Predictions as AnnData |
| `output/eval/metrics.json` | Eval | Evaluation metrics (JSON) |
| `output/eval/metrics.csv` | Eval | Evaluation metrics (CSV) |

---

## Tests

```bash
python tests/check.py
```

Verifies:

1. **Import** — all classes import correctly
2. **Forward** — BioReason runs with toy data:
   - No evidence (inference mode)
   - With evidence (training mode)
   - `detach_latent=True`
   - `encode()` + `predict()`
3. **Loss** — BioLoss returns correct dict for stages 1, 2, 3 and minimal batch

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

If you use BioReason in your research, please cite:

```bibtex
@software{biorreason2026,
  title     = {BioReason: Evidence-guided latent biological reasoning for single-cell perturbation prediction},
  year      = {2026},
  url       = {https://github.com/1-Vast/BioReason}
}
```
