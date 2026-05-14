# BioReason Experiment Summary

This file summarizes the current local benchmark results and their limitations.

## Evidence and LLM Status

- LLM is not a prediction model.
- LLM is only a Stage 0 offline biological evidence builder.
- LLM converts perturbation labels, gene names, or pathway hints into structured biological prior JSON.
- Stage 0 encodes that prior into `adata.obsm["evidence"]`.
- Stage 2 consumes the evidence vector during training.
- Stage 3 and inference run without LLM and without evidence.

Current local benchmark runs did not use real API calls. The reported results are therefore local-KB / offline-evidence results, not proof that LLM-generated evidence improves performance.

## Leakage Audit

Latest available audit:

- `output/bench/leak_audit_final.json`
- result: PASS
- high issues: 0
- medium issues: 0
- low issues: 0
- split counts: train 7000, val 1000, test 2000

Implemented leak protections:

- train/validation split happens before HVG selection and group mean construction
- group means are computed from train cells only
- validation data is aligned to train gene order
- target latent masks are valid only for train source indices
- evaluation defaults to test split and aligns predictions by saved source indices

## Main DEG Result

Source files:

- `output/bench/deg_report.json`
- `output/bench/deg_metrics.csv`

Summary:

| Metric | BioReason | GroupMean |
|---|---:|---:|
| Top-DEG Overlap@50 | 0.950 | 0.907 |
| DEG Pearson | 0.980 | 0.924 |
| Direction Accuracy | 1.000 | 1.000 |
| Delta Cosine | 0.9995 | 0.9990 |

Interpretation:

- BioReason preserves perturbation DEG structure on the synthetic benchmark.
- GroupMean remains strong because the benchmark has clear perturbation means.
- These results do not prove LLM benefit because no real LLM calls were used.

## Cell-Type Result

Source file:

- `output/bench/report_percell.json`

Summary:

| Metric | BioReason | GroupMean |
|---|---:|---:|
| MSE | 10.4423 | 13.0778 |
| MAE | 2.4787 | 2.6894 |
| Pearson | 0.6567 | 0.3792 |
| DEG Pearson | 0.9549 | 0.9363 |
| Top-DEG Overlap@50 | 0.850 | 0.8067 |

Interpretation:

- BioReason improves cell-level and cell-type-sensitive prediction versus GroupMean in this benchmark.
- The main advantage is preserving per-cell/cell-type identity rather than only predicting a perturbation mean.

## Ablation Result

Source files:

- `output/bench/deg_ablation.json`
- `output/bench/ablation_results.json`

DEG ablation:

| Setting | Overlap@50 |
|---|---:|
| Full | 0.935 |
| No evidence warm-up | 0.935 |
| No latent-only BP | 0.940 |
| MLP reasoner | 0.930 |
| No evidence | 0.935 |
| Stage 1 only | 0.875 |

Interpretation:

- Stage 2 + Stage 3 improve over Stage 1 only on this synthetic DEG benchmark.
- Evidence-specific ablations are weakly separated because perturbation ID is already highly informative.
- A stronger held-out perturbation benchmark is required to prove evidence-guided reasoning.

## Current Conclusion

Supported:

- Stage 0/1/2/3 pipeline runs end to end.
- Leak checks pass on current benchmark metadata.
- BioReason can outperform GroupMean on cell-level synthetic evaluation.
- Stage 2/3 improve over Stage 1-only in DEG overlap.

Not yet supported:

- Real LLM-generated evidence improves performance.
- Generalization to unseen perturbations.
- Real Perturb-seq performance.

Required next experiments:

1. Run Stage 0 with `--use_llm` and budget/cache enabled, then compare local-KB vs LLM-evidence runs.
2. Add held-out perturbation split where test perturbations are unseen during training.
3. Evaluate on real perturbation data.
4. Compare against stronger baselines under the same no-leak split.
