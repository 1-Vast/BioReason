# BioReason

**Evidence-guided latent biological reasoning for single-cell perturbation prediction.**

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
  (DEG/pathway)     (train only)          │  multi-step reasoning    │
                                          │  ReasonStep × N         │
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

## Architecture

| Module | Role |
|--------|------|
| `CellEncoder` | Gene expression → cell embedding |
| `PertEncoder` | Perturbation id / multi-hot / continuous → pert embedding |
| `Reasoner` | Multi-step latent biological reasoning (× N ReasonStep) |
| `ReasonStep` | Single reasoning step: FiLM + self-attention update |
| `EvidenceGate` | Injects biological evidence into latent state (train only; inference idempotent) |
| `ExprDecoder` | cell_emb + z_bio → delta expression |
| `BioLoss` | expr + delta + DEG-weighted + latent-alignment + evidence + MMD |

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
  pred                pred + L_evidence      pred + L_align
                     (save teacher latent)   (align with teacher)
```

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

### Tests

```bash
python tests/check.py
```

## Data Format

h5ad (AnnData): `adata.X` [N×G], `adata.obs["perturbation"]`, `adata.obs["cell_type"]` (optional).

## Configuration

| File | Content |
|------|---------|
| `config/default.yaml` | Project, data, eval settings |
| `config/model.yaml` | Model dims, steps, heads, dropout |
| `config/train.yaml` | Epochs, lr, batch size, AMP, grad clip |
| `config/loss.yaml` | Loss weights |

## Environment

Copy `.env.example` to `.env`. Never commit `.env`.

## Project Structure

```
BioReason/
├── main.py              # CLI entry
├── models/              # Core model code
│   ├── reason.py        # BioReason, Reasoner, EvidenceGate, ReasonStep
│   ├── base.py          # MLP, ResidualBlock
│   ├── cell.py          # CellEncoder, CovEncoder
│   ├── pert.py          # PertEncoder
│   ├── latent.py        # LatentBlock, FiLM, CrossBlock
│   ├── decoder.py       # ExprDecoder
│   ├── loss.py          # BioLoss
│   ├── data.py          # PertDataset
│   ├── train.py         # Training loop
│   ├── infer.py         # Inference
│   └── eval.py          # Metrics
├── utils/               # Shared utilities
│   └── config.py        # YAML config loading
├── config/              # YAML configuration
├── tests/
│   └── check.py         # Import, forward, loss verification
├── scripts/             # Batch launch scripts
├── dataset/             # Data (git-ignored)
└── output/              # Results (git-ignored)
```

## License

MIT
