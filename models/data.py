"""Single-cell perturbation data loading.

Supports h5ad (scanpy/anndata) format with real target construction:
  - group_mean: y = mean expression of perturbation group
  - control_to_pert: x from control pool, y from perturbed pool
  - identity: y = x (debug only)

Counterfactual inference via set_target_pert():
  Overrides perturbation label for all cells to predict unseen perturbations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


def _to_dense(X):
    from scipy.sparse import issparse
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def _select_hvg(adata, n_hvg):
    from scipy.sparse import issparse
    X = adata.X
    if issparse(X):
        X = X.toarray()
    var = np.var(X, axis=0)
    top_idx = np.argsort(var)[-n_hvg:]
    return adata[:, top_idx].copy()


def bio_collate_fn(batch):
    """Collate function for PertDataset dict output.
    Handles evidence=None and cov dicts correctly.
    """
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "cov":
            # Merge dicts of tensors
            cov_keys = vals[0].keys()
            collated[k] = {}
            for ck in cov_keys:
                collated[k][ck] = torch.stack([v[ck] for v in vals])
        elif k == "evidence":
            non_none = [v for v in vals if v is not None]
            if len(non_none) == 0:
                collated[k] = None
            else:
                collated[k] = torch.stack(non_none)
        elif k == "meta":
            collated[k] = vals  # list of dicts
        elif k == "target_latent":
            non_none = [v for v in vals if v is not None]
            if len(non_none) == 0:
                collated[k] = None
            else:
                collated[k] = torch.stack(non_none)
        elif k in ("pert_str",):
            collated[k] = vals
        else:
            collated[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return collated


class PertDataset(Dataset):
    """Single-cell perturbation dataset with real target construction.

    target_mode:
      "group_mean" (default): y = group mean expression of the perturbation label
      "control_to_pert": x drawn from control pool, y from perturbation group
      "identity": y = x (debug only)

    counterfactual: set_target_pert("TP53_KO") overrides pert for all cells.
    """

    def __init__(self, adata, label_key="perturbation", control_label="control",
                 cell_type_key="cell_type", dose_key="dose", time_key="time",
                 batch_key="batch", evidence_key="evidence",
                 target_mode="group_mean", pair_by=None,
                 use_control_as_input=True, use_hvg=True, n_hvg=2000):
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

        # Perturbation vocabulary
        pert_labels = adata.obs[label_key].astype(str)
        self.pert_cats = sorted(set(pert_labels))
        self.pert_to_id = {p: i for i, p in enumerate(self.pert_cats)}
        self.id_to_pert = {i: p for p, i in self.pert_to_id.items()}
        self.n_perts = len(self.pert_cats)

        # HVG
        if use_hvg and adata.n_vars > n_hvg:
            adata = _select_hvg(adata, n_hvg)
        self.adata = adata

        # Expression
        self.X = _to_dense(adata.X)
        self.input_dim = self.X.shape[1]
        self.n_obs = len(adata)

        # Evidence dimension
        self.evidence_dim = None
        if evidence_key and evidence_key in adata.obsm:
            ev = adata.obsm[evidence_key]
            self.evidence_dim = ev.shape[1] if ev.ndim == 2 else 1

        # Indices
        pert_arr = adata.obs[label_key].astype(str).values
        self.is_control = (pert_arr == control_label)
        self.control_indices = np.where(self.is_control)[0].tolist()
        self.obs_indices = list(range(self.n_obs))

        # Group indices per perturbation label
        self.group_indices = {}
        for p in self.pert_cats:
            idx = np.where(pert_arr == p)[0]
            if len(idx) > 0:
                self.group_indices[p] = idx.tolist()

        # Cell-type group indices
        self.cell_type_group_indices = {}
        if cell_type_key and cell_type_key in adata.obs:
            ct_arr = adata.obs[cell_type_key].astype(str).values
            for p in self.pert_cats:
                p_idx = np.where(pert_arr == p)[0]
                if len(p_idx) == 0:
                    continue
                ct_in_group = ct_arr[p_idx]
                for ct in set(ct_in_group):
                    key = (p, ct)
                    idx = p_idx[ct_in_group == ct]
                    if len(idx) > 0:
                        self.cell_type_group_indices[key] = idx.tolist()

        # Build group means
        self._build_group_means()

        # Covariate vocabs
        self.cov_maps = {}
        for key in [cell_type_key, dose_key, time_key, batch_key]:
            if key and key in adata.obs:
                cats = sorted(set(adata.obs[key].astype(str)))
                self.cov_maps[key] = {c: i for i, c in enumerate(cats)}
            else:
                self.cov_maps[key] = {"unknown": 0}

        # Counterfactual override
        self._target_pert = None
        self._target_pert_id = None

        # Target latent (for stage 3 alignment)
        self.target_latents = None  # [N, D] tensor

    # ── Group mean construction ──────────────────────────────────
    def _build_group_means(self):
        """Precompute per-perturbation-group mean expression."""
        self.group_means = {}
        for p, indices in self.group_indices.items():
            self.group_means[p] = self.X[indices].mean(axis=0)

    # ── Counterfactual ──────────────────────────────────────────
    def set_target_pert(self, pert_label_or_id):
        """Set counterfactual perturbation for all cells."""
        if isinstance(pert_label_or_id, str):
            if pert_label_or_id not in self.pert_to_id:
                raise ValueError(
                    f"Unknown perturbation '{pert_label_or_id}'. "
                    f"Available: {list(self.pert_cats)}"
                )
            self._target_pert = pert_label_or_id
            self._target_pert_id = self.pert_to_id[pert_label_or_id]
        else:
            self._target_pert_id = pert_label_or_id
            self._target_pert = self.id_to_pert.get(pert_label_or_id, str(pert_label_or_id))

    def clear_target_pert(self):
        self._target_pert = None
        self._target_pert_id = None

    # ── Target latent ───────────────────────────────────────────
    def load_target_latent(self, path):
        """Load precomputed teacher latent from stage 2."""
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            latent = data["latent"]  # [N, D]
            indices = data.get("indices", torch.arange(len(latent)))
        else:
            latent = data
            indices = torch.arange(len(latent))
        # Build mapping: index -> latent
        n = max(indices.max().item() + 1, len(self))
        self.target_latents = torch.zeros(n, latent.shape[1])
        self.target_latents[indices] = latent

    # ── Target resolution ───────────────────────────────────────
    def _get_target(self, idx, pert_id):
        """Resolve target expression y for a given cell index and perturbation."""
        if self.target_mode == "identity":
            return self.X[idx].copy()

        pert_str = self.id_to_pert.get(pert_id, self.pert_cats[0])
        if pert_str not in self.group_means:
            return self.X[idx].copy()

        if self.target_mode == "group_mean":
            return self.group_means[pert_str].copy()

        if self.target_mode == "control_to_pert":
            # Try cell-type-matched control
            if self.pair_by and self.cell_type_key:
                ct = str(self.adata.obs.iloc[idx][self.cell_type_key])
                key_ctrl = (self.control_label, ct)
                key_pert = (pert_str, ct)
                if key_ctrl in self.cell_type_group_indices and key_pert in self.cell_type_group_indices:
                    ctrl_idx = np.random.choice(self.cell_type_group_indices[key_ctrl])
                    pert_idx = np.random.choice(self.cell_type_group_indices[key_pert])
                    return self.X[pert_idx].copy()
            # Fallback: random from group
            return self.group_means[pert_str].copy()

        # Default fallback
        return self.group_means.get(pert_str, self.X[idx].copy())

    def _get_control_input(self, idx, pert_id):
        """Resolve control input x for a given cell."""
        if self.use_control_as_input or self.is_control[idx]:
            return self.X[idx].copy()
        # Find a control cell
        if self.control_indices:
            ctrl_idx = np.random.choice(self.control_indices)
            return self.X[ctrl_idx].copy()
        return self.X[idx].copy()

    # ── __len__ / __getitem__ ────────────────────────────────────
    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        obs = self.adata.obs.iloc[idx]

        # Perturbation (counterfactual override)
        if self._target_pert_id is not None:
            pert_id = self._target_pert_id
            pert_str = self._target_pert
        else:
            pert_str = str(obs[self.label_key])
            pert_id = self.pert_to_id.get(pert_str, 0)

        # Input x
        x = torch.tensor(self._get_control_input(idx, pert_id), dtype=torch.float32)

        # Target y
        y = torch.tensor(self._get_target(idx, pert_id), dtype=torch.float32)

        # Covariates
        cov = {}
        for key, vocab in self.cov_maps.items():
            if key and key in obs.index:
                val = str(obs[key])
                cov[key] = torch.tensor(vocab.get(val, 0), dtype=torch.long)

        # Evidence
        evidence = None
        if self.evidence_key and self.evidence_key in self.adata.obsm:
            ev = self.adata.obsm[self.evidence_key][idx]
            evidence = torch.tensor(np.atleast_1d(np.squeeze(ev)), dtype=torch.float32)

        # Target latent
        target_latent = None
        if self.target_latents is not None and idx < len(self.target_latents):
            target_latent = self.target_latents[idx]

        return {
            "x": x,
            "y": y,
            "pert": torch.tensor(pert_id, dtype=torch.long),
            "pert_str": pert_str,
            "cov": cov,
            "evidence": evidence,
            "target_latent": target_latent,
            "meta": {
                "idx": idx,
                "source_idx": idx,
                "target_idx": -1,
                "is_control": bool(self.is_control[idx]),
                "pert_str": pert_str,
            },
        }


# ── Factory functions ───────────────────────────────────────────

def read_h5ad(path):
    try:
        import scanpy as sc
        return sc.read_h5ad(path)
    except ImportError:
        raise ImportError("scanpy required. pip install scanpy")


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
    )


def build_loader(dataset, batch_size=128, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, collate_fn=bio_collate_fn)


def split_data(dataset, train_ratio=0.9, seed=42):
    from torch.utils.data import random_split
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    return random_split(dataset, [n_train, n_val],
                         generator=torch.Generator().manual_seed(seed))


load_h5ad = read_h5ad
