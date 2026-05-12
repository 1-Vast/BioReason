"""Single-cell perturbation data loading.

Target construction: group_mean (default), control_to_pert, identity.
Counterfactual: set_target_pert() overrides perturbation for all cells.
Reproducibility: var_names, gene alignment, perturbation vocab, cov_dims.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────

def _to_dense(X):
    from scipy.sparse import issparse
    return X.toarray() if issparse(X) else np.asarray(X)


def _select_hvg(adata, n_hvg):
    from scipy.sparse import issparse
    X = adata.X
    if issparse(X):
        X = X.toarray()
    var = np.var(X, axis=0)
    top_idx = np.argsort(var)[-n_hvg:]
    return adata[:, top_idx].copy(), top_idx


def align_adata_to_genes(adata, target_genes):
    """Align adata to target_genes order.

    Missing genes are filled with zero columns.
    Extra genes are ignored. Returns new adata.
    """
    import pandas as pd
    current = list(adata.var_names)
    target = list(target_genes) if hasattr(target_genes, '__iter__') else target_genes

    # Build aligned matrix
    n_cells = adata.n_obs
    aligned = np.zeros((n_cells, len(target)), dtype=np.float32)
    missing = []
    for i, gene in enumerate(target):
        if gene in current:
            j = current.index(gene)
            aligned[:, i] = _to_dense(adata.X)[:, j]
        else:
            missing.append(gene)

    if missing:
        print(f"WARNING: {len(missing)} genes not found in adata, filled with 0")

    import anndata
    new_obs = adata.obs.copy()
    new_var = pd.DataFrame(index=target)
    result = anndata.AnnData(X=aligned, obs=new_obs, var=new_var)
    for key in adata.obsm:
        result.obsm[key] = adata.obsm[key]
    return result


# ── Collate ──────────────────────────────────────────────────────

def bio_collate_fn(batch):
    """Collate function for PertDataset.
    Handles: evidence=None, cov dicts, target_latent with mask.
    """
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
    """Single-cell perturbation dataset with reproducibility support.

    Stores: var_names, selected_var_names, cov_dims, perturbation vocab.
    Supports: set_vocab, set_target_pert, load_target_latent with mask.
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

        # ── Gene names ──────────────────────────────────────────
        self.var_names = list(adata.var_names)
        self.hvg_indices = None
        self.selected_var_names = list(adata.var_names)

        # HVG selection
        if use_hvg and adata.n_vars > n_hvg:
            adata, hvg_idx = _select_hvg(adata, n_hvg)
            self.hvg_indices = hvg_idx
            self.selected_var_names = list(adata.var_names)

        self.adata = adata
        self.X = _to_dense(adata.X)
        self.input_dim = self.X.shape[1]
        self.n_obs = len(adata)

        # ── Perturbation vocabulary ─────────────────────────────
        pert_arr = adata.obs[label_key].astype(str).values
        self.pert_cats = sorted(set(pert_arr))
        self.pert_to_id = {p: i for i, p in enumerate(self.pert_cats)}
        self.id_to_pert = {i: p for p, i in self.pert_to_id.items()}
        self.n_perts = len(self.pert_cats)
        self.is_control = (pert_arr == control_label)
        self.control_indices = np.where(self.is_control)[0].tolist()
        self.obs_indices = list(range(self.n_obs))

        # ── Group indices ───────────────────────────────────────
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

        # ── Evidence dimension ──────────────────────────────────
        self.evidence_dim = None
        if evidence_key and evidence_key in adata.obsm:
            ev = adata.obsm[evidence_key]
            self.evidence_dim = ev.shape[1] if ev.ndim == 2 else 1

        # ── Group means ─────────────────────────────────────────
        self._build_group_means()

        # ── Covariate vocabs + dims ─────────────────────────────
        self.cov_maps = {}
        self.cov_dims = {}
        for key in [cell_type_key, dose_key, time_key, batch_key]:
            if key and key in adata.obs:
                cats = sorted(set(adata.obs[key].astype(str)))
                self.cov_maps[key] = {c: i for i, c in enumerate(cats)}
            else:
                self.cov_maps[key] = {"unknown": 0}
            self.cov_dims[key] = max(1, len(self.cov_maps[key]))

        # ── Counterfactual override ─────────────────────────────
        self._target_pert = None
        self._target_pert_id = None

        # ── Target latent + mask ────────────────────────────────
        self.target_latents = None
        self.target_latent_mask = None

    # ── Group means ──────────────────────────────────────────────
    def _build_group_means(self):
        self.group_means = {}
        for p, indices in self.group_indices.items():
            self.group_means[p] = self.X[indices].mean(axis=0)

    # ── Vocabulary override ──────────────────────────────────────
    def set_vocab(self, pert_to_id, id_to_pert=None):
        """Override perturbation vocab (e.g., from checkpoint)."""
        self.pert_to_id = pert_to_id
        if id_to_pert is not None:
            self.id_to_pert = id_to_pert
        else:
            self.id_to_pert = {i: p for p, i in pert_to_id.items()}
        self.pert_cats = sorted(self.pert_to_id.keys())
        self.n_perts = len(self.pert_cats)

    # ── Counterfactual ───────────────────────────────────────────
    def set_target_pert(self, pert_label_or_id):
        """Set counterfactual perturbation. Uses current vocab (may be from ckpt)."""
        if isinstance(pert_label_or_id, str):
            if pert_label_or_id not in self.pert_to_id:
                raise ValueError(
                    f"Unknown perturbation '{pert_label_or_id}'. "
                    f"Available ({len(self.pert_cats)}): {list(self.pert_cats)[:20]}..."
                )
            self._target_pert = pert_label_or_id
            self._target_pert_id = self.pert_to_id[pert_label_or_id]
        else:
            self._target_pert_id = pert_label_or_id
            self._target_pert = self.id_to_pert.get(pert_label_or_id, str(pert_label_or_id))

    def clear_target_pert(self):
        self._target_pert = None
        self._target_pert_id = None

    # ── Target latent ────────────────────────────────────────────
    def load_target_latent(self, path):
        """Load precomputed teacher latent with mask."""
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            latent = data["latent"]
            indices = data.get("indices", torch.arange(len(latent)))
            mask_in = data.get("mask", torch.ones(len(indices), dtype=torch.bool))
        else:
            latent = data
            indices = torch.arange(len(latent))
            mask_in = torch.ones(len(indices), dtype=torch.bool)

        n = max(indices.max().item() + 1, len(self))
        D = latent.shape[1]
        self.target_latents = torch.zeros(n, D)
        self.target_latent_mask = torch.zeros(n, dtype=torch.bool)
        self.target_latents[indices] = latent
        self.target_latent_mask[indices] = mask_in

    # ── Target resolution ────────────────────────────────────────
    def _get_target(self, idx, pert_id):
        if self.target_mode == "identity":
            return self.X[idx].copy()
        pert_str = self.id_to_pert.get(pert_id, self.pert_cats[0])
        if pert_str not in self.group_means:
            return self.X[idx].copy()
        if self.target_mode == "group_mean":
            return self.group_means[pert_str].copy()
        if self.target_mode == "control_to_pert":
            if self.pair_by and self.cell_type_key:
                ct = str(self.adata.obs.iloc[idx][self.cell_type_key])
                key_pert = (pert_str, ct)
                if key_pert in self.cell_type_group_indices:
                    pert_idx = np.random.choice(self.cell_type_group_indices[key_pert])
                    return self.X[pert_idx].copy()
            return self.group_means[pert_str].copy()
        return self.group_means.get(pert_str, self.X[idx].copy())

    def _get_control_input(self, idx, pert_id):
        """Resolve control input. If use_control_as_input and not control,
        sample from control pool (cell-type-match first, then global, then self)."""
        if self.target_mode == "identity":
            return self.X[idx].copy()
        if not self.use_control_as_input:
            return self.X[idx].copy()
        if self.is_control[idx]:
            return self.X[idx].copy()

        # Try cell-type matched control
        if self.pair_by and self.cell_type_key in self.adata.obs:
            ct = str(self.adata.obs.iloc[idx][self.cell_type_key])
            key_ctrl = (self.control_label, ct)
            if key_ctrl in self.cell_type_group_indices:
                ctrl_idx = np.random.choice(self.cell_type_group_indices[key_ctrl])
                return self.X[ctrl_idx].copy()

        # Global control fallback
        if self.control_indices:
            ctrl_idx = np.random.choice(self.control_indices)
            return self.X[ctrl_idx].copy()

        # Last resort: self
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

        # Target latent with mask
        target_latent = None
        mask = False
        if self.target_latents is not None and idx < len(self.target_latents):
            if self.target_latent_mask is not None:
                mask = bool(self.target_latent_mask[idx].item())
            else:
                mask = bool(self.target_latents[idx].abs().sum() > 1e-10)
            if mask:
                target_latent = self.target_latents[idx]

        return {
            "x": x,
            "y": y,
            "pert": torch.tensor(pert_id, dtype=torch.long),
            "pert_str": pert_str,
            "cov": cov,
            "evidence": evidence,
            "target_latent": target_latent,
            "target_latent_mask": torch.tensor(mask, dtype=torch.bool),
            "meta": {
                "idx": idx,
                "source_idx": idx,
                "target_idx": -1,
                "is_control": bool(self.is_control[idx]),
                "pert_str": pert_str,
            },
        }


# ── Factory functions ────────────────────────────────────────────

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
