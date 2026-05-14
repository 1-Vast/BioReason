import logging
logger = logging.getLogger(__name__)
"""Single-cell perturbation data loading - sparse-safe.

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


# Sparse helpers

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


# Collate

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
        elif k in ("evidence", "evidence_conf", "trust"):
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


# Dataset

class PertDataset(Dataset):
    """Single-cell perturbation dataset. Keeps X sparse; densify per-row.

    split-aware: when split is set, only cells of that split are returned.
    group_means always computed from stats_split (default "train") only.
    """

    def __init__(self, adata, label_key="perturbation", control_label="control",
                 cell_type_key="cell_type", dose_key="dose", time_key="time",
                 batch_key="batch", evidence_key="evidence",
                 target_mode="group_mean", pair_by=None,
                 use_control_as_input=True, use_hvg=True, n_hvg=2000,
                 cache_group_means=True,
                 split=None,                   # "train" | "val" | "test" | None (all)
                 stats_split="train",          # which split to compute group_means from
                 control_input_mode="control_mean",  # "control_mean" | "random_control" | "matched_control"
                 warn_on_missing_stats=True):
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
        self.split = split
        self.stats_split = stats_split

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
        self.X = adata.X
        # CSC→CSR for fast row access (CSC row slicing is O(n_cols) per row)
        if _issparse(self.X):
            from scipy.sparse import csc_matrix
            if isinstance(self.X, csc_matrix):
                self.X = self.X.tocsr()
        self.is_sparse = _issparse(self.X)
        self.input_dim = adata.n_vars
        self.source_indices = np.asarray(
            adata.obs["_source_idx"].values if "_source_idx" in adata.obs else np.arange(adata.n_obs),
            dtype=np.int64,
        )

        # Split filtering
        pert_arr = adata.obs[label_key].astype(str).values

        if "split" in adata.obs and split is not None:
            split_mask = (adata.obs["split"].values == split)
            self.obs_indices = np.where(split_mask)[0].tolist()
        else:
            self.obs_indices = list(range(adata.n_obs))

        self.n_obs = len(self.obs_indices)

        # Stats indices: which cells to compute group_means from
        if "split" in adata.obs and stats_split is not None:
            stats_mask = (adata.obs["split"].values == stats_split)
            self.stats_indices = np.where(stats_mask)[0].tolist()
            self.train_indices = np.where(adata.obs["split"].values == "train")[0].tolist()
        else:
            self.stats_indices = list(range(adata.n_obs))
            self.train_indices = list(range(adata.n_obs))

        # Perturbation vocab (from all cells in adata)
        self.pert_cats = sorted(set(pert_arr))
        self.pert_to_id = {p: i for i, p in enumerate(self.pert_cats)}
        self.id_to_pert = {i: p for p, i in self.pert_to_id.items()}
        self.n_perts = len(self.pert_cats)

        # is_control and control_indices: from stats split only
        self.is_control = (pert_arr == control_label)
        stats_ctrl = [i for i in self.stats_indices if self.is_control[i]]
        self.control_indices = stats_ctrl if stats_ctrl else np.where(self.is_control)[0].tolist()

        # Group indices: only from stats_split
        self.group_indices = {}
        for p in self.pert_cats:
            idx = [i for i in self.stats_indices if pert_arr[i] == p]
            if len(idx) > 0:
                self.group_indices[p] = idx

        self.cell_type_group_indices = {}
        if cell_type_key and cell_type_key in adata.obs:
            ct_arr = adata.obs[cell_type_key].astype(str).values
            for p in self.pert_cats:
                p_idx = [i for i in self.stats_indices if pert_arr[i] == p]
                if len(p_idx) == 0: continue
                ct_in_group = ct_arr[p_idx]
                for ct in set(ct_in_group):
                    key = (p, ct)
                    idx = [p_idx[j] for j in range(len(p_idx)) if ct_in_group[j] == ct]
                    if len(idx) > 0:
                        self.cell_type_group_indices[key] = idx

        # Evidence dim
        self.evidence_dim = None
        if evidence_key and evidence_key in adata.obsm:
            ev = adata.obsm[evidence_key]
            self.evidence_dim = ev.shape[1] if ev.ndim == 2 else 1

        self.has_evidence_conf = "evidence_conf" in adata.obs

        # Group means: from stats_split only
        self.cache_group_means = cache_group_means
        self.group_means_source = stats_split if stats_split else "all"
        self.control_input_mode = control_input_mode
        
        # Compute train-only control_mean and cell_type-matched control_means
        self.control_mean = None
        self.control_mean_by_ct = {}
        if "split" in adata.obs and stats_split is not None:
            train_mask = adata.obs["split"].values == stats_split
            train_ctrl_mask = train_mask & (pert_arr == control_label)
            if train_ctrl_mask.any():
                ctrl_indices = np.where(train_ctrl_mask)[0].tolist()
                self.control_mean = self._mean_rows(ctrl_indices)
                # Cell-type matched control means
                if cell_type_key and cell_type_key in adata.obs:
                    ct_arr = adata.obs[cell_type_key].astype(str).values
                    for ct in set(ct_arr[train_ctrl_mask]):
                        ct_ctrl_idx = [i for i in ctrl_indices if ct_arr[i] == ct]
                        if ct_ctrl_idx:
                            self.control_mean_by_ct[ct] = self._mean_rows(ct_ctrl_idx)
            else:
                if warn_on_missing_stats:
                    logger.warning("No train control cells found for control_mean computation")
        else:
            if warn_on_missing_stats:
                logger.warning("No split info; control_mean computed from all control cells (may leak)")
            all_ctrl_mask = pert_arr == control_label
            if all_ctrl_mask.any():
                self.control_mean = self._mean_rows(np.where(all_ctrl_mask)[0].tolist())
        
        if cache_group_means:
            self._build_group_means()
        else:
            self.group_means = {}
            self.perturbation_effect = {}

        # Covariate vocabs from all cells; vocab coverage is model metadata, not target statistics.
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

    # Row access
    def _row(self, idx):
        return _row_dense(self.X, idx)

    def _mean_rows(self, indices):
        Xg = self.X[indices]
        m = np.asarray(Xg.mean(axis=0), dtype=np.float32).reshape(-1)
        return m

    # Group means (sparse-safe)
    def _build_group_means(self):
        self.group_means = {}
        for p, indices in self.group_indices.items():
            self.group_means[p] = self._mean_rows(indices)
        # Pre-compute perturbation effects: fixed delta per perturbation
        self.perturbation_effect = {}
        if self.control_label in self.group_means:
            ctrl_mean = self.group_means[self.control_label]
            for p, gm in self.group_means.items():
                if p != self.control_label:
                    self.perturbation_effect[p] = (gm - ctrl_mean).astype(np.float32)
            self.perturbation_effect[self.control_label] = np.zeros_like(ctrl_mean, dtype=np.float32)
        else:
            # Fallback: use zeros
            for p in self.group_means:
                self.perturbation_effect[p] = np.zeros(len(self.group_means[p]), dtype=np.float32)

    # Vocabulary
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
        self.target_latents[indices] = latent.float()
        # Only set mask=True for indices in train split
        train_set = set(self.source_indices[self.train_indices]) if self.train_indices else set(range(n))
        for i, idx in enumerate(indices):
            if mask_in[i] and idx.item() in train_set:
                self.target_latent_mask[idx] = True

    # Target resolution
    def _get_target(self, idx, pert_id):
        """Resolve training target.

        - "identity": own expression
        - "per_cell": own expression (per-cell counterfactual target)
        - "group_mean" (default): group mean of perturbation
        """
        if self.target_mode in ("identity", "per_cell"):
            return self._row(idx)
        pert_str = self.id_to_pert.get(pert_id, self.pert_cats[0])
        if self.cache_group_means and pert_str in self.group_means:
            return self.group_means[pert_str].copy()
        if pert_str in self.group_indices:
            return self._mean_rows(self.group_indices[pert_str])
        return self._row(idx)

    def _get_control_input(self, idx, pert_id):
        """Return control input based on control_input_mode.

        Modes:
        - "control_mean": train control mean (stable, fixed per perturbation)
        - "random_control": random control cell (current behavior)
        - "matched_control": same cell_type control cell
        - "identity": own expression (when use_control_as_input=False)
        """
        if self.target_mode == "identity":
            return self._row(idx)
        if not self.use_control_as_input:
            return self._row(idx)
        if self.is_control[idx]:
            return self._row(idx)
        
        mode = getattr(self, 'control_input_mode', 'random_control')
        
        if mode == "control_mean":
            # Use train control mean (stable, no per-sample noise)
            if self.pair_by and self.cell_type_key in self.adata.obs:
                ct = str(self.adata.obs.iloc[idx][self.cell_type_key])
                if ct in self.control_mean_by_ct:
                    return self.control_mean_by_ct[ct].copy()
            if self.control_mean is not None:
                return self.control_mean.copy()
            # Fallback to random control
            if self.control_indices:
                return self._row(np.random.choice(self.control_indices))
            return self._row(idx)
        
        elif mode == "matched_control":
            # Same cell-type control cell
            if self.pair_by and self.cell_type_key in self.adata.obs:
                ct = str(self.adata.obs.iloc[idx][self.cell_type_key])
                if ct in self.control_mean_by_ct:
                    # Use cell-type matched control mean for stability
                    return self.control_mean_by_ct[ct].copy()
            # Fallback to random control
            if self.control_indices:
                return self._row(np.random.choice(self.control_indices))
            return self._row(idx)
        
        else:  # "random_control" or default
            if self.pair_by and self.cell_type_key in self.adata.obs:
                ct = str(self.adata.obs.iloc[idx][self.cell_type_key])
                key_ctrl = (self.control_label, ct)
                if key_ctrl in self.cell_type_group_indices:
                    ctrl_idx = np.random.choice(self.cell_type_group_indices[key_ctrl])
                    return self._row(ctrl_idx)
            if self.control_indices:
                ctrl_idx = np.random.choice(self.control_indices)
                return self._row(ctrl_idx)
            return self._row(idx)

    # __len__ / __getitem__
    def __len__(self): return self.n_obs

    def __getitem__(self, idx):
        # Map DataLoader index to actual adata index (needed for split filtering)
        real_idx = self.obs_indices[idx] if idx < len(self.obs_indices) else idx
        source_idx = int(self.source_indices[real_idx])
        obs = self.adata.obs.iloc[real_idx]
        if self._target_pert_id is not None:
            pert_id = self._target_pert_id; pert_str = self._target_pert
        else:
            pert_str = str(obs[self.label_key])
            pert_id = self.pert_to_id.get(pert_str, 0)

        x = torch.tensor(self._get_control_input(real_idx, pert_id), dtype=torch.float32)
        y = torch.tensor(self._get_target(real_idx, pert_id), dtype=torch.float32)

        cov = {}
        for key, vocab in self.cov_maps.items():
            if key and key in obs.index:
                val = str(obs[key])
                cov[key] = torch.tensor(vocab.get(val, 0), dtype=torch.long)
            else:
                # Always include ALL registered cov keys for consistent CovEncoder input dim
                cov[key] = torch.tensor(0, dtype=torch.long)

        evidence = None
        evidence_conf = None
        if self.evidence_key and self.evidence_key in self.adata.obsm:
            ev = self.adata.obsm[self.evidence_key][real_idx]
            evidence = torch.tensor(np.atleast_1d(np.squeeze(ev)), dtype=torch.float32)
            conf = float(obs.get("evidence_conf", 1.0)) if self.has_evidence_conf else 1.0
            evidence_conf = torch.tensor(conf, dtype=torch.float32)

        target_latent = None; mask = False
        if self.target_latents is not None and source_idx < len(self.target_latents):
            if self.target_latent_mask is not None:
                mask = bool(self.target_latent_mask[source_idx].item())
            else:
                mask = bool(self.target_latents[source_idx].abs().sum() > 1e-10)
            if mask: target_latent = self.target_latents[source_idx]

        # Pre-computed perturbation effect (fixed per perturbation)
        pe = self.perturbation_effect.get(pert_str, np.zeros(self.input_dim, dtype=np.float32))
        
        return {
            "x": x, "y": y,
            "pert": torch.tensor(pert_id, dtype=torch.long),
            "pert_str": pert_str, "cov": cov, "evidence": evidence,
            "evidence_conf": evidence_conf,
            "target_latent": target_latent,
            "target_latent_mask": torch.tensor(mask, dtype=torch.bool),
            "perturbation_effect": torch.tensor(pe),
            "meta": {"idx": source_idx, "source_idx": source_idx, "target_idx": -1,
                      "is_control": bool(self.is_control[real_idx]), "pert_str": pert_str,
                      "control_input_mode": self.control_input_mode,
                      "control_source": "train_mean" if self.control_input_mode == "control_mean" else "random"}, 
        }


# Factory

def read_h5ad(path):
    try:
        import anndata as ad
        return ad.read_h5ad(path)
    except Exception:
        import scanpy as sc
        return sc.read_h5ad(path)

def _dataset_kwargs(config):
    return dict(
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
        split=config.get("split", None),
        stats_split=config.get("stats_split", "train"),
        control_input_mode=config.get("control_input_mode", "control_mean"),
        warn_on_missing_stats=config.get("warn_on_missing_stats", True),
    )

def build_dataset(path, config):
    adata = read_h5ad(path)
    return PertDataset(adata, **_dataset_kwargs(config))


def _ensure_source_idx(adata):
    if "_source_idx" not in adata.obs:
        adata.obs["_source_idx"] = np.arange(adata.n_obs, dtype=np.int64)
    return adata


def _train_val_masks(adata, train_ratio=0.9, seed=42):
    n = adata.n_obs
    if "split" in adata.obs:
        split = adata.obs["split"].astype(str).values
        train_mask = split == "train"
        val_mask = split == "val"
        if not val_mask.any():
            val_mask = split == "test"
        if train_mask.any() and val_mask.any():
            return train_mask, val_mask
        logger.warning("split column exists but lacks train/val cells; falling back to random split")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(n * train_ratio)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_mask[order[:n_train]] = True
    val_mask[order[n_train:]] = True
    return train_mask, val_mask


def _copy_aligned_metadata(dst, src):
    dst.pert_to_id = dict(src.pert_to_id)
    dst.id_to_pert = dict(src.id_to_pert)
    dst.pert_cats = list(src.pert_cats)
    dst.n_perts = src.n_perts
    dst.cov_maps = {k: dict(v) for k, v in src.cov_maps.items()}
    dst.cov_dims = dict(src.cov_dims)
    if src.cache_group_means:
        dst.group_means = {k: v.copy() for k, v in src.group_means.items()}
        dst.group_means_source = "train"
    dst.perturbation_effect = {k: v.copy() for k, v in getattr(src, "perturbation_effect", {}).items()}
    dst.control_mean = None if getattr(src, "control_mean", None) is None else src.control_mean.copy()
    dst.control_mean_by_ct = {k: v.copy() for k, v in getattr(src, "control_mean_by_ct", {}).items()}
    dst.control_indices = []


def build_train_val_datasets(path, config, train_ratio=0.9, seed=42):
    """Build train/val datasets without fitting statistics on validation cells."""
    adata = _ensure_source_idx(read_h5ad(path))
    train_mask, val_mask = _train_val_masks(adata, train_ratio=train_ratio, seed=seed)

    train_adata = adata[train_mask].copy()
    val_adata = adata[val_mask].copy()

    train_cfg = dict(config)
    train_cfg["split"] = None
    train_cfg["stats_split"] = "train"
    train_ds = PertDataset(train_adata, **_dataset_kwargs(train_cfg))

    val_adata = align_adata_to_genes(val_adata, train_ds.selected_var_names)
    val_cfg = dict(config)
    val_cfg["split"] = None
    val_cfg["stats_split"] = None
    val_cfg["use_hvg"] = False
    val_cfg["target_mode"] = "per_cell"
    val_cfg["cache_group_means"] = False
    val_cfg["warn_on_missing_stats"] = False
    val_ds = PertDataset(val_adata, **_dataset_kwargs(val_cfg))
    val_ds.selected_var_names = list(train_ds.selected_var_names)
    _copy_aligned_metadata(val_ds, train_ds)

    return train_ds, val_ds



def build_loader(dataset, batch_size=128, shuffle=True, num_workers=0,
                 pin_memory=False, persistent_workers=False, prefetch_factor=2,
                 drop_last=False, avoid_single_batch=True):
    if avoid_single_batch and not drop_last and len(dataset) % batch_size == 1:
        logger.warning(f"avoid_single_batch: last batch would be size 1; forcing drop_last=True")
        drop_last = True
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                  collate_fn=bio_collate_fn, num_workers=num_workers,
                  pin_memory=pin_memory)
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)

def batch_summary(batch):
    s = {"x": tuple(batch["x"].shape), "y": tuple(batch["y"].shape),
         "pert": tuple(batch["pert"].shape)}
    for k in ("evidence", "evidence_conf", "target_latent"):
        s[k] = tuple(batch[k].shape) if batch.get(k) is not None else None
    return s

load_h5ad = read_h5ad
