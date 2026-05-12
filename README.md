# BioReason

**Evidence-guided latent biological reasoning for single-cell perturbation prediction.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Table of Contents

- [Overview](#overview)
- [Framework](#framework)
- [Architecture](#architecture)
- [Three-Stage Training](#three-stage-training)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Design Notes](#design-notes)
- [File Index](#file-index)
- [License](#license)

---

## Overview

BioReason predicts single-cell perturbation responses through an intermediate **latent biological reasoning state** (`z_bio`). It adapts evidence-guided latent distillation to perturbation biology:

- **Training**: optional biological evidence (DEG, pathway, TF scores) shapes `z_bio`
- **Inference**: no evidence required — model reasons autonomously
- **Counterfactual**: predict unseen perturbations by overriding the perturbation condition

```
x_control + perturbation → z_bio → delta → x_perturbed
```

---

## Framework

```
                         ┌─────────────────────┐
  control  ─────────────►│    CellEncoder      │──┐
  expression             └─────────────────────┘  │
                                                  │  cell_emb
  perturbation ────────►│    PertEncoder      │──┤
  (gene / drug / combo)  └─────────────────────┘  │
                                                  ├──► context ──┐
  covariates  ──────────►│    CovEncoder       │──┘              │
  (cell_type/dose/time)  └─────────────────────┘                 │
                                                                 ▼
                                          ┌──────────────────────────┐
  evidence  ──────► EvidenceGate ────────►│       Reasoner           │
  (DEG/pathway)     (train only)          │  ReasonStep × N         │
                                          └──────────┬───────────────┘
                                                     │
                                                     ▼  z_bio
                                          ┌──────────────────────────┐
                     cell_emb ───────────►│     ExprDecoder          │
                                          └──────────┬───────────────┘
                                                     │
                                                     ▼  delta_x
                                          ┌──────────────────────────┐
                     x_control ──────────►│     pred = x + delta     │
                                          └──────────────────────────┘
```

---

## Architecture

| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| `CellEncoder` | `x [B,G]` | `[B,D]` | Gene expression → cell embedding |
| `PertEncoder` | pert id/multihot/continuous | `[B,D]` | Perturbation → embedding |
| `CovEncoder` | cov dict | `[B,D]` | Metadata → embedding |
| `Reasoner` | cell+pert+cov | `z_bio [B,D]` | Multi-step latent reasoning |
| `EvidenceGate` | z + evidence | `z' [B,D]` | Evidence injection (idempotent if None) |
| `ExprDecoder` | cell_emb + z_bio | `delta [B,G]` | Latent → expression change |
| `BioLoss` | out + batch + stage | loss dict | Stage-gated 5-loss composite |

---

## Three-Stage Training

```
Stage 1              Stage 2                 Stage 3
warm-up              evidence-guided         evidence-free
                     latent distillation     latent alignment

x + pert             x + pert + evidence     x + pert (no evidence)
   │                    │                       │
   ▼                    ▼                       ▼
Reasoner             Reasoner                Reasoner
   │                    │                       │
   ▼                    ▼                       ▼
ExprDecoder          ExprDecoder            ExprDecoder
   │                    │                       │
L_expr+L_delta       +L_evidence+L_mmd       +L_latent(cosine)
                     (export target_latent)   (align with teacher)
```

---

## Installation

```bash
git clone git@github.com:1-Vast/BioReason.git
cd BioReason
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `scipy`, `anndata`, `scanpy`, `pyyaml`, `python-dotenv`, `tqdm`, `openai` (optional).

---

## Quick Start

### 1. Data

Place h5ad file in `dataset/`:

```
dataset/perturb.h5ad
```

Required: `adata.X`, `adata.obs["perturbation"]`. Optional: `adata.obsm["evidence"]`.

### 2. Train

```bash
# Stage 1: warm-up
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 1 --device cuda

# Stage 2: evidence-guided (saves target_latent.pt automatically)
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 2 --device cuda

# Stage 3: evidence-free alignment
python main.py train --config config/default.yaml --h5ad dataset/perturb.h5ad --stage 3 --target_latent output/stage2/target_latent.pt --device cuda
```

### 3. Counterfactual Inference

```bash
python main.py infer --checkpoint output/stage3/model.pt --h5ad dataset/perturb.h5ad --pert TP53_KO --out output/infer
```
All cells are predicted as if `TP53_KO` was applied. No evidence required.

### 4. Evaluate

```bash
python main.py eval --pred output/infer/pred.npz --truth dataset/perturb.h5ad --out output/eval
```

### 5. API Test

```bash
python main.py api-test
```

### 6. Verify

```bash
python tests/check.py
```

---

## CLI Reference

### `python main.py train`

| Argument | Description |
|----------|-------------|
| `--config` | Config YAML path (default: config/default.yaml) |
| `--h5ad` | Data file (required) |
| `--stage` | 1, 2, or 3 |
| `--device` | cuda or cpu |
| `--save_dir` | Override output directory |
| `--target_latent` | Teacher latent path (stage 3) |

### `python main.py infer`

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Model .pt file (required) |
| `--h5ad` | Data file (required) |
| `--pert` | Target perturbation, e.g., TP53_KO (required) |
| `--out` | Output directory |
| `--device` | cuda or cpu |
| `--batch_size` | Batch size |

### `python main.py eval`

| Argument | Description |
|----------|-------------|
| `--pred` | Predictions .npz (required) |
| `--truth` | Ground truth .h5ad (required) |
| `--out` | Output directory |
| `--top_deg` | Top N DEGs (default 50) |

### `python main.py api-test`

Tests LLM API connectivity. No data sent.

---

## Design Notes

### Important Design Choices

1. **LLM is optional and restricted**. LLM is only used for offline biological prior construction and API connectivity tests. It is **never** called inside `model.forward()` or the training loop.

2. **No multimodal/visual dependencies**. BioReason is a single-cell perturbation model. Input is gene expression + perturbation condition. No image, MLLM, Qwen, or visual reasoning code.

3. **Evidence is optional at inference**. The model learns to reason without evidence during Stage 3 alignment.

4. **Evidence can include**: pathway scores, TF activity vectors, DEG score vectors, gene module scores — any precomputed biological signal.

5. **Counterfactual inference**: `--pert TP53_KO` overrides perturbation labels across all cells, enabling prediction of unseen perturbations from control cells.

6. **Dataset target construction**: default is `group_mean` (y = group mean expression). Other modes: `control_to_pert` (x from control, y from perturbed group), `identity` (debug only).

7. **API keys**: read from `.env` via `python-dotenv`. Never hardcoded. `.env` is git-ignored.

---

## File Index

### `models/` — Core

| File | Purpose |
|------|---------|
| `reason.py` ★ | `BioReason`, `Reasoner`, `EvidenceGate`, `ReasonStep` |
| `data.py` | `PertDataset` with real target construction, counterfactual, collate |
| `loss.py` | `BioLoss` — stage-gated, cosine alignment, per-gene DEG weighting |
| `train.py` | Train loop, stage logic, target latent export/attach |
| `infer.py` | Counterfactual prediction, model loading |
| `eval.py` | 8 evaluation metrics |
| `base.py` | `MLP`, `ResidualBlock` |
| `cell.py` | `CellEncoder`, `CovEncoder` |
| `pert.py` | `PertEncoder` (id/multihot/continuous) |
| `latent.py` | `LatentBlock`, `FiLM`, `CrossBlock` |
| `decoder.py` | `ExprDecoder` |

### `utils/`

| File | Purpose |
|------|---------|
| `config.py` | YAML loading, .env |
| `llm.py` | Offline LLM config test, prior builder |

### `config/`

| File | Purpose |
|------|---------|
| `default.yaml` | Project + data settings |
| `model.yaml` | Architecture hyperparameters |
| `train.yaml` | Training hyperparameters |
| `loss.yaml` | Loss weights + metrics |

### `tests/`

| File | Purpose |
|------|---------|
| `check.py` | Run all tests |
| `test_forward.py` | Forward pass with evidence_dim |
| `test_loss.py` | Stage-gated loss computation |
| `test_data.py` | Dataset target construction |
| `test_infer.py` | Counterfactual inference |
| `test_llm_config.py` | API config (no real key) |

---

## License

MIT
