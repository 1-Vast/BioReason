"""Single-cell perturbation data loading — sparse-safe.

Key design:
  - AnnData.X kept sparse; densify only row-wise or batch-wise.
  - group_means computed via sparse mean (safe).
  - HVG computed via sparse variance (no full .toarray()).
  - align_adata_to_genes returns sparse by default.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ── Sparse helpers ───────────────────────────────────────────────

def _issparse(X):
    from scipy.sparse import issparse
    return issparse(X)


def _row_dense(X, idx):
    """Return one row as 1D np.float32 dense array."""
    row = X[idx]
    if _issparse(row):
        row = row.toarray()
    return np.asarray(row, dtype=np.float32).reshape(-1)


def _select_hvg_sparse(adata, n_hvg):
    """Select HVG via sparse variance (no full .toarray())."""
    from scipy.sparse import issparse
    X = adata.X
    n, g = X.shape
    if not issparse(X):
        var = np.var(X.toarray() if hasattr(X, 'toarray') else np.asarray(X), axis=0)
    else:
        mean = np.asarray(X.mean(axis=0)).reshape(-1)
        mean_sq = np.asarray(X.power(2).mean(axis=0)).reshape(-1)
        var = mean_sq - mean ** 2
    top_idx = np.argsort(var)[-n_hvg:]
    return adata[:, top_idx].copy(), top_idx


def align_adata_to_genes(adata, target_genes, sparse_output=True):
    """Align adata to target_genes order. Returns new AnnData.

    Sparse-safe: uses column slicing + hstack, not dense zeros.
    Missing genes filled with sparse zero columns.
    """
    import anndata, pandas as pd
    from scipy.sparse import issparse, csr_matrix, hstack

    current = list(adata.var_names)
    target = list(target_genes)
    current_map = {g: i for i, g in enumerate(current)}

    cols = []
    missing = 0
    n_cells = adata.n_obs
    X = adata.X

    for gene in target:
        if gene in current_map:
            col_data = X[:, current_map[gene]]
            if issparse(col_data):
                coo = col_data.tocoo()
                col_2d = csr_matrix((coo.data, (coo.row, np.zeros_like(coo.row))),
                                     shape=(n_cells, 1), dtype=np.float32)
            else:
                col_2d = csr_matrix(np.asarray(col_data).reshape(n_cells, 1).astype(np.float32))
            cols.append(col_2d)
        else:
            missing += 1
            cols.append(csr_matrix((n_cells, 1), dtype=np.float32))

    if missing:
        logger.warning(f"align_adata: {missing}/{len(target)} genes not found, zero-filled")

    if sparse_output:
        aligned_X = hstack(cols, format="csr") if len(cols) > 1 else cols[0].tocsr()
    else:
        aligned_X = np.hstack([c.toarray() if issparse(c) else np.asarray(c).reshape(-1, 1)
                                for c in cols]).astype(np.float32)

    new_var = pd.DataFrame(index=target)
    result = anndata.AnnData(X=aligned_X, obs=adata.obs.copy(), var=new_var)
    for key in adata.obsm:
        result.obsm[key] = adata.obsm[key]
    return result


# ── Collate ──────────────────────────────────────────────────────

def bio_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "cov":
            cov_keys = vals[0].keys()
            collated[k] = {ck: torch.stack([v[ck] for v in vals]) for ck in cov_keys}
        elif k in ("meta", "pert_str"):
            collated[k] = vals
        elif k == "evidence":
            non_none = [v for v in vals if v is not None]
            collated[k] = torch.stack(non_none) if non_none else None
        elif k == "target_latent":
            non_none = [v for v in vals if v is not None]
            collated[k] = torch.stack(non_none) if non_none else None
        elif k == "target_latent_mask":
            collated[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else torch.tensor(vals)
        else:
            collated[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return collated


# ── Dataset ──────────────────────────────────────────────────────

class PertDataset(Dataset):
    """Single-cell perturbation dataset. Keeps X sparse; densify per-row."""

    def __init__(self, adata, label_key="perturbation", control_label="control",
                 cell_type_key="cell_type", dose_key="dose", time_key="time",
                 batch_key="batch", evidence_key="evidence",
                 target_mode="group_mean", pair_by=None,
                 use_control_as_input=True, use_hvg=True, n_hvg=2000,
                 cache_group_means=True):
        super().__init__()
        self.label_key = label_key
        self.control_label = control_label
        self.cell_type_key = cell_type_key
        self.dose_key = dose_key
        self.time_key = time_key
        self.batch_key = batch_key
        self.evidence_key = evidence_key
        self.target_mode = target_mode
        self.pair_by = pair_by
        self.use_control_as_input = use_control_as_input

        # Gene names
        self.var_names = list(adata.var_names)
        self.hvg_indices = None
        self.selected_var_names = list(adata.var_names)

        # HVG: sparse-safe
        if use_hvg and adata.n_vars > n_hvg:
            adata, hvg_idx = _select_hvg_sparse(adata, n_hvg)
            self.hvg_indices = hvg_idx
            self.selected_var_names = list(adata.var_names)

        self.adata = adata
        # Keep X sparse — never densify full matrix
        self.X = adata.X
        self.is_sparse = _issparse(self.X)
        self.input_dim = adata.n_vars
        self.n_obs = len(adata)

        # Perturbation vocab
        pert_arr = adata.obs[label_key].astype(str).values
        self.pert_cats = sorted(set(pert_arr))
        self.pert_to_id = {p: i for i, p in enumerate(self.pert_cats)}
        self.id_to_pert = {i: p for p, i in self.pert_to_id.items()}
        self.n_perts = len(self.pert_cats)
        self.is_control = (pert_arr == control_label)
        self.control_indices = np.where(self.is_control)[0].tolist()
        self.obs_indices = list(range(self.n_obs))

        # Group indices
        self.group_indices = {}
        for p in self.pert_cats:
            idx = np.where(pert_arr == p)[0]
            if len(idx) > 0:
                self.group_indices[p] = idx.tolist()

        self.cell_type_group_indices = {}
        if cell_type_key and cell_type_key in adata.obs:
            ct_arr = adata.obs[cell_type_key].astype(str).values
            for p in self.pert_cats:
                p_idx = np.where(pert_arr == p)[0]
                if len(p_idx) == 0: continue
                ct_in_group = ct_arr[p_idx]
                for ct in set(ct_in_group):
                    key = (p, ct)
                    idx = p_idx[ct_in_group == ct]
                    if len(idx) > 0:
                        self.cell_type_group_indices[key] = idx.tolist()

        # Evidence dim
        self.evidence_dim = None
        if evidence_key and evidence_key in adata.obsm:
            ev = adata.obsm[evidence_key]
            self.evidence_dim = ev.shape[1] if ev.ndim == 2 else 1

        # Group means (sparse-safe)
        self.cache_group_means = cache_group_means
        if cache_group_means:
            self._build_group_means()
        else:
            self.group_means = {}

        # Covariate vocabs
        self.cov_maps = {}
        self.cov_dims = {}
        for key in [cell_type_key, dose_key, time_key, batch_key]:
            if key and key in adata.obs:
                cats = sorted(set(adata.obs[key].astype(str)))
                self.cov_maps[key] = {c: i for i, c in enumerate(cats)}
            else:
                self.cov_maps[key] = {"unknown": 0}
            self.cov_dims[key] = max(1, len(self.cov_maps[key]))

        # Counterfactual
        self._target_pert = None; self._target_pert_id = None

        # Target latent
        self.target_latents = None; self.target_latent_mask = None

    # ── Row access ─────────────────────────────────────────────
    def _row(self, idx):
        return _row_dense(self.X, idx)

    def _mean_rows(self, indices):
        Xg = self.X[indices]
        m = np.asarray(Xg.mean(axis=0), dtype=np.float32).reshape(-1)
        return m

    # ── Group means (sparse-safe) ──────────────────────────────
    def _build_group_means(self):
        self.group_means = {}
        for p, indices in self.group_indices.items():
            self.group_means[p] = self._mean_rows(indices)

    # ── Vocabulary ─────────────────────────────────────────────
    def set_vocab(self, pert_to_id, id_to_pert=None):
        self.pert_to_id = pert_to_id
        self.id_to_pert = id_to_pert or {i: p for p, i in pert_to_id.items()}
        self.pert_cats = sorted(self.pert_to_id.keys())
        self.n_perts = len(self.pert_cats)

    def set_target_pert(self, pert_label_or_id):
        if isinstance(pert_label_or_id, str):
            if pert_label_or_id not in self.pert_to_id:
                raise ValueError(f"Unknown perturbation '{pert_label_or_id}'. Available: {list(self.pert_cats)[:20]}...")
            self._target_pert = pert_label_or_id
            self._target_pert_id = self.pert_to_id[pert_label_or_id]
        else:
            self._target_pert_id = pert_label_or_id
            self._target_pert = self.id_to_pert.get(pert_label_or_id, str(pert_label_or_id))

    def clear_target_pert(self):
        self._target_pert = None; self._target_pert_id = None

    def load_target_latent(self, path):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            latent = data["latent"]
            indices = data.get("indices", torch.arange(len(latent)))
            mask_in = data.get("mask", torch.ones(len(indices), dtype=torch.bool))
        else:
            latent = data; indices = torch.arange(len(latent))
            mask_in = torch.ones(len(indices), dtype=torch.bool)
        n = max(indices.max().item() + 1, len(self))
        D = latent.shape[1]
        self.target_latents = torch.zeros(n, D)
        self.target_latent_mask = torch.zeros(n, dtype=torch.bool)
        self.target_latents[indices] = latent
        self.target_latent_mask[indices] = mask_in

    # ── Target resolution ──────────────────────────────────────
    def _get_target(self, idx, pert_id):
        if self.target_mode == "identity":
            return self._row(idx)
        pert_str = self.id_to_pert.get(pert_id, self.pert_cats[0])
        if self.cache_group_means and pert_str in self.group_means:
            return self.group_means[pert_str].copy()
        # On-demand
        if pert_str in self.group_indices:
            return self._mean_rows(self.group_indices[pert_str])
        return self._row(idx)

    def _get_control_input(self, idx, pert_id):
        if self.target_mode == "identity":
            return self._row(idx)
        if not self.use_control_as_input:
            return self._row(idx)
        if self.is_control[idx]:
            return self._row(idx)
        # Cell-type matched control
        if self.pair_by and self.cell_type_key in self.adata.obs:
            ct = str(self.adata.obs.iloc[idx][self.cell_type_key])
            key_ctrl = (self.control_label, ct)
            if key_ctrl in self.cell_type_group_indices:
                ctrl_idx = np.random.choice(self.cell_type_group_indices[key_ctrl])
                return self._row(ctrl_idx)
        # Global control fallback
        if self.control_indices:
            ctrl_idx = np.random.choice(self.control_indices)
            return self._row(ctrl_idx)
        return self._row(idx)

    # ── __len__ / __getitem__ ──────────────────────────────────
    def __len__(self): return self.n_obs

    def __getitem__(self, idx):
        obs = self.adata.obs.iloc[idx]
        if self._target_pert_id is not None:
            pert_id = self._target_pert_id; pert_str = self._target_pert
        else:
            pert_str = str(obs[self.label_key])
            pert_id = self.pert_to_id.get(pert_str, 0)

        x = torch.tensor(self._get_control_input(idx, pert_id), dtype=torch.float32)
        y = torch.tensor(self._get_target(idx, pert_id), dtype=torch.float32)

        cov = {}
        for key, vocab in self.cov_maps.items():
            if key and key in obs.index:
                val = str(obs[key])
                cov[key] = torch.tensor(vocab.get(val, 0), dtype=torch.long)

        evidence = None
        if self.evidence_key and self.evidence_key in self.adata.obsm:
            ev = self.adata.obsm[self.evidence_key][idx]
            evidence = torch.tensor(np.atleast_1d(np.squeeze(ev)), dtype=torch.float32)

        target_latent = None; mask = False
        if self.target_latents is not None and idx < len(self.target_latents):
            if self.target_latent_mask is not None:
                mask = bool(self.target_latent_mask[idx].item())
            else:
                mask = bool(self.target_latents[idx].abs().sum() > 1e-10)
            if mask: target_latent = self.target_latents[idx]

        return {
            "x": x, "y": y,
            "pert": torch.tensor(pert_id, dtype=torch.long),
            "pert_str": pert_str, "cov": cov, "evidence": evidence,
            "target_latent": target_latent,
            "target_latent_mask": torch.tensor(mask, dtype=torch.bool),
            "meta": {"idx": idx, "source_idx": idx, "target_idx": -1,
                      "is_control": bool(self.is_control[idx]), "pert_str": pert_str},
        }


# ── Factory ──────────────────────────────────────────────────────

def read_h5ad(path):
    import scanpy as sc; return sc.read_h5ad(path)

def build_dataset(path, config):
    adata = read_h5ad(path)
    return PertDataset(
        adata,
        label_key=config.get("label_key", "perturbation"),
        control_label=config.get("control_label", "control"),
        cell_type_key=config.get("cell_type_key", "cell_type"),
        dose_key=config.get("dose_key", "dose"),
        time_key=config.get("time_key", "time"),
        batch_key=config.get("batch_key", "batch"),
        evidence_key=config.get("evidence_key", "evidence"),
        target_mode=config.get("target_mode", "group_mean"),
        pair_by=config.get("pair_by"),
        use_control_as_input=config.get("use_control_as_input", True),
        use_hvg=config.get("use_hvg", True),
        n_hvg=config.get("n_hvg", 2000),
        cache_group_means=config.get("cache_group_means", True),
    )

def build_loader(dataset, batch_size=128, shuffle=True, num_workers=0,
                 pin_memory=False, persistent_workers=False, prefetch_factor=2,
                 drop_last=False, avoid_single_batch=True):
    if avoid_single_batch and not drop_last and len(dataset) % batch_size == 1:
        logger.warning(f"avoid_single_batch: last batch would be size 1 → forcing drop_last=True")
        drop_last = True
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                  collate_fn=bio_collate_fn, num_workers=num_workers,
                  pin_memory=pin_memory)
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)

def split_data(dataset, train_ratio=0.9, seed=42):
    from torch.utils.data import random_split
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    return random_split(dataset, [n_train, n_val],
                         generator=torch.Generator().manual_seed(seed))

def batch_summary(batch):
    s = {"x": tuple(batch["x"].shape), "y": tuple(batch["y"].shape),
         "pert": tuple(batch["pert"].shape)}
    for k in ("evidence", "target_latent"):
        s[k] = tuple(batch[k].shape) if batch.get(k) is not None else None
    return s

load_h5ad = read_h5ad
