# Output

Training outputs, predictions, and evaluation results are saved here. Only `.gitkeep` and this README are tracked by git.

## Structure

```
output/
├── stage1/
│   └── model.pt
├── stage2/
│   ├── model.pt
│   └── target_latent.pt
├── stage3/
│   └── model.pt
├── infer/
│   ├── pred.npz
│   └── pred.h5ad
└── eval/
    ├── metrics.json
    └── metrics.csv
```

## Notes

- Checkpoints (`.pt`, `.pth`, `.ckpt`) are git-ignored.
- Predictions (`.npz`, `.h5ad`) are git-ignored.
- Do NOT commit output files to GitHub.
