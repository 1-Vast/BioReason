"""Single-cell perturbation data loading.

Supports h5ad (scanpy/anndata) format.
Outputs PyTorch Dataset returning: x, y, pert, cov, evidence, meta
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class PertDataset(Dataset):
    """Single-cell perturbation dataset.

    Reads from h5ad file:
      adata.X:   expression matrix [N, G]
      adata.obs: metadata (perturbation, cell_type, dose, time, batch)

    Returns dict:
      x, y, pert, pert_str, cov, evidence, meta
    """

    def __init__(self, adata, label_key="perturbation", control_label="control",
                 cell_type_key="cell_type", dose_key="dose", time_key="time",
                 batch_key="batch", use_hvg=True, n_hvg=2000,
                 evidence_key=None):
        super().__init__()
        self.adata = adata
        self.label_key = label_key
        self.control_label = control_label
        self.cell_type_key = cell_type_key
        self.dose_key = dose_key
        self.time_key = time_key
        self.batch_key = batch_key
        self.evidence_key = evidence_key

        pert_labels = adata.obs[label_key].astype(str)
        self.pert_cats = sorted(set(pert_labels))
        self.pert_to_id = {p: i for i, p in enumerate(self.pert_cats)}
        self.n_perts = len(self.pert_cats)

        if use_hvg and adata.n_vars > n_hvg:
            self._select_hvg(n_hvg)

        self.X = self._to_dense(adata.X)
        self.is_control = (adata.obs[label_key].astype(str) == control_label).values

        self.cov_maps = {}
        for key in [cell_type_key, dose_key, time_key, batch_key]:
            if key and key in adata.obs:
                cats = sorted(set(adata.obs[key].astype(str)))
                self.cov_maps[key] = {c: i for i, c in enumerate(cats)}
            else:
                self.cov_maps[key] = {"unknown": 0}

    def _select_hvg(self, n_hvg):
        from scipy.sparse import issparse
        X = self.adata.X
        if issparse(X):
            X = X.toarray()
        var = np.var(X, axis=0)
        top_idx = np.argsort(var)[-n_hvg:]
        self.adata = self.adata[:, top_idx].copy()

    def _to_dense(self, X):
        from scipy.sparse import issparse
        if issparse(X):
            return X.toarray()
        return np.asarray(X)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        obs = self.adata.obs.iloc[idx]
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        pert_str = str(obs[self.label_key])
        pert_id = self.pert_to_id.get(pert_str, 0)
        y = x.clone()

        cov = {}
        for key, vocab in self.cov_maps.items():
            if key and key in obs.index:
                val = str(obs[key])
                cov[key] = torch.tensor(vocab.get(val, 0), dtype=torch.long)

        evidence = None
        if self.evidence_key and self.evidence_key in self.adata.obsm:
            ev = self.adata.obsm[self.evidence_key][idx]
            evidence = torch.tensor(ev, dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "pert": pert_id,
            "pert_str": pert_str,
            "cov": cov,
            "evidence": evidence,
            "meta": {"idx": idx, "is_control": bool(self.is_control[idx])},
        }


def read_h5ad(path):
    """Load h5ad file using scanpy."""
    try:
        import scanpy as sc
        return sc.read_h5ad(path)
    except ImportError:
        raise ImportError("scanpy is required. Install: pip install scanpy")


def build_dataset(path, config):
    """Build PertDataset from h5ad path and config dict."""
    adata = read_h5ad(path)
    return PertDataset(
        adata,
        label_key=config.get("label_key", "perturbation"),
        control_label=config.get("control_label", "control"),
        cell_type_key=config.get("cell_type_key", "cell_type"),
        dose_key=config.get("dose_key", "dose"),
        time_key=config.get("time_key", "time"),
        batch_key=config.get("batch_key", "batch"),
        use_hvg=config.get("use_hvg", True),
        n_hvg=config.get("n_hvg", 2000),
    )


def build_loader(dataset, batch_size=128, shuffle=True, num_workers=2):
    """Build DataLoader from dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers)


def split_data(dataset, train_ratio=0.9, seed=42):
    """Split dataset into train/val. Returns (train_ds, val_ds)."""
    from torch.utils.data import random_split
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    return random_split(dataset, [n_train, n_val],
                         generator=torch.Generator().manual_seed(seed))


load_h5ad = read_h5ad  # alias
