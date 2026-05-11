# Dataset

Place perturbation data files here. Only `.gitkeep` and this README are tracked by git.

## Supported Format

h5ad (AnnData) with:

| Component | Content |
|-----------|---------|
| `adata.X` | Expression matrix [N cells x G genes] |
| `adata.obs["perturbation"]` | Perturbation label (e.g., "TP53_KO", "control") |
| `adata.obs["cell_type"]` | Cell type annotation |
| `adata.obs["dose"]` | Dose (optional) |
| `adata.obs["time"]` | Time point (optional) |
| `adata.obs["batch"]` | Batch ID (optional) |
| `adata.obs["condition"]` | Experimental condition (optional) |

The control label defaults to `"control"`.

## Example

```
dataset/perturb.h5ad
```

## Notes

- Real data files are git-ignored.
- Do NOT commit `.h5ad`, `.h5`, `.npz`, `.csv`, or other large data files.
