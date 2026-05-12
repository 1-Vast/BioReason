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

## Stage 0: Evidence Prior Construction

Before training, biological evidence priors can be constructed from a **local knowledge base (KB)** with an optional **LLM fallback**.

### Design Principles

- **Local KB first**: A JSON dictionary of perturbation → biological mechanism is the primary source. LLM is only invoked when a perturbation is missing from the KB and `--use_llm` is set.
- **LLM never in training/forward**: LLM calls are restricted to offline prior construction. They are never used during model training or inference.
- **Confidence filtering**: Priors with `confidence_score < min_conf` (default 0.5) are zeroed out — no evidence is injected for low-confidence perturbations.
- **Hash encoder default**: Text-to-vector encoding uses `sklearn.feature_extraction.text.HashingVectorizer` (no extra dependencies, deterministic).

### Usage

```bash
# Local KB only (recommended)
python main.py prior --h5ad dataset/perturb.h5ad --out dataset/perturb_with_evidence.h5ad --kb dataset/kb/prior.json

# With LLM fallback for missing perturbations
python main.py prior --h5ad dataset/perturb.h5ad --out dataset/perturb_with_evidence.h5ad --kb dataset/kb/prior.json --use_llm

# Custom settings
python main.py prior --h5ad dataset/perturb.h5ad --out output/with_evidence.h5ad \
  --kb dataset/kb/prior.json --min_conf 0.6 --evidence_dim 256 --encoder hash \
  --audit output/prior_audit.csv
```

### KB Format

```json
{
  "TP53_KO": {
    "description": "TP53 knockout disrupts DNA damage response and cell-cycle checkpoint.",
    "confidence_score": 0.95,
    "source": "local",
    "pathway_impact": [
      {"pathway": "DNA damage response", "direction": "down", "confidence": 0.95}
    ],
    "tf_activity": [{"tf": "TP53", "direction": "down"}],
    "marker_genes": [{"gene": "CDKN1A", "direction": "down"}]
  }
}
```

### Output

The pipeline writes into the output `.h5ad`:
- `adata.obsm["evidence"]` — evidence vectors [N_cells, evidence_dim]
- `adata.obs["evidence_conf"]` — confidence scores per cell
- `adata.obs["evidence_source"]` — "local", "llm", "control", or "miss"
- `adata.uns["evidence_audit"]` — per-perturbation audit records

---

## Trust Gate — Confidence-Aware Evidence Injection

BioReason includes a **confidence-aware evidence gate** that modulates how strongly biological evidence is injected into the latent reasoning state.

- **Trust score**: A learned sigmoid head predicts a per-sample confidence score `trust ∈ [0, 1]` from the latent state and evidence vector.
- **Modulation**: The trust score scales the FiLM gate parameters (`gamma`, `beta`), reducing evidence influence when the model is uncertain.
- **Training target** (optional): If `trust_w > 0`, the trust score is regularized against `evidence_conf` from the prior construction step.
- **Inference**: Evidence is never provided at inference time, so trust defaults to `None`.

```yaml
# config/model.yaml
model:
  evidence_gate: "confidence"   # enables trust scoring
  use_evidence_conf: true       # uses confidence head

# config/loss.yaml
loss:
  trust: 0.0                    # weight for trust regularization
  trust_target: null            # unused (targets from evidence_conf)
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

## Performance and GPU Usage

### Recommended Commands

**Linux server (multi-worker):**
```bash
python main.py train --h5ad dataset/perturb.h5ad --stage 1 --device cuda \
  --batch_size 256 --num_workers 4 --pin_memory --progress
```

**Windows:**
```bash
python main.py train --h5ad dataset/perturb.h5ad --stage 1 --device cuda \
  --batch_size 128 --num_workers 0 --progress
```

### Profiling

```bash
python main.py train --h5ad dataset/perturb.h5ad --stage 1 --profile --profile_batches 20
```

Outputs: data wait time, H2D transfer time, forward/backward timings, GPU memory, and optimization suggestions.

### GPU Data Flow

```
DataLoader (CPU, pin_memory) → non_blocking .to(device) → forward (GPU)
→ loss (GPU) → backward (GPU) → optimizer step (GPU)
→ per-epoch: .detach() summary → checkpoint save
```

- **No `.cpu().numpy()` inside training step** — all computation stays on GPU.
- **No per-step `.item()` spam** — losses accumulated via `.detach().item()` at `log_every` intervals.
- **Checkpoint saved only at epoch end** — zero I/O inside training loop.
- **`.empty_cache()` only if configured** — not called per step.
- **`tqdm` progress bar** with live loss + GPU memory postfix. No per-batch `print()`.
- **`torch.inference_mode()` for validation and inference** — disables autograd tracking.

### CLI Flags

| Flag | Description |
|------|-------------|
| `--batch_size N` | Batch size |
| `--num_workers N` | DataLoader workers (0 for Windows) |
| `--pin_memory` / `--no_pin_memory` | Enable/disable pinned memory |
| `--amp` / `--no_amp` | Enable/disable automatic mixed precision |
| `--progress` / `--no_progress` | Show/hide tqdm progress bars |
| `--profile` | Run profiling before training |
| `--compile` | Enable torch.compile (experimental) |

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

8. **Control input**: `use_control_as_input=True` draws input from the control pool (cell-type matched when possible), not the perturbed cell itself. This ensures realistic baseline for counterfactual prediction.

---

## Reproducibility Notes

BioReason checkpoints are self-contained for reproducible inference:

1. **Perturbation vocabulary** is saved in checkpoint (`pert_to_id`, `id_to_pert`, `pert_cats`). Inference uses the checkpoint vocabulary, not the inference h5ad vocabulary. Unknown perturbations in the checkpoint vocab can still be targeted.

2. **Gene order** (`selected_var_names`) is saved in checkpoint. Inference aligns input genes to training order via `align_adata_to_genes()`. Missing genes are zero-filled; extra genes are ignored.

3. **Model dimensions** (`input_dim`, `num_perts`, `evidence_dim`, `cov_dims`) are frozen in the checkpoint. `load_model()` initializes from checkpoint values to prevent embedding shape mismatch.

4. **Covariate vocabularies** are frozen in checkpoint. Unknown inference covariates are mapped to 0.

5. **Target latent mask** prevents Stage 3 latent loss from aligning against zero vectors. Only samples with valid teacher latents contribute to the alignment loss.

---

## Large-scale Stability Notes

BioReason is designed for real-world single-cell datasets (10⁵–10⁶ cells):

1. **Sparse-first**: `AnnData.X` stays sparse in memory. Dense conversion only row-wise / batch-wise. No full `.toarray()` in dataset init.

2. **Sparse HVG**: Uses sparse variance (`X.power(2).mean() - X.mean()**2`), not full densify.

3. **Sparse gene alignment**: `align_adata_to_genes()` returns sparse CSR via column slicing. Missing genes are sparse zero columns.

4. **MMD float32**: Under AMP, `mmd_loss` casts to `.float()` internally. `torch.nan_to_num` guards overflow. `mmd_max_samples` (128) limits O(B²).

5. **Atomic checkpoint**: Writes to `.tmp` then `os.replace()`. Interrupted writes never corrupt existing checkpoint.

6. **Preallocated inference**: NumPy preallocation or `--memmap` for million-cell output. No `torch.cat()` memory spike.

7. **BatchNorm safety**: `avoid_single_batch=True` auto-drops tail batch if size=1. **LayerNorm recommended** for single-cell.

8. **Memmap inference**:
   ```bash
   python main.py infer ... --memmap --memmap_dir output/infer_memmap
   ```

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
| `check.py` | Run all 10 tests |
| `test_forward.py` | Forward pass with evidence_dim |
| `test_loss.py` | Stage-gated loss, mask, cosine |
| `test_data.py` | Dataset target construction |
| `test_infer.py` | Counterfactual inference |
| `test_llm_config.py` | API config (no real key) |
| `test_control_input.py` | Control pool input sampling |
| `test_vocab_checkpoint.py` | Checkpoint vocab/dims preservation |
| `test_gene_align.py` | Gene alignment with missing/extra genes |
| `test_target_latent_mask.py` | Latent mask for stage 3 alignment |
| `test_cov_dims.py` | Covariate dimensions in model |

---

## License

MIT
