"""Tensor cache dataset for perturbation training.

The cache path moves h5ad/scipy/pandas work out of the training hot path.
Training-time ``__getitem__`` only indexes prebuilt tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedPertDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str = "train", preload_to_gpu: bool = False,
                 device: str = "cuda"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"cache meta not found: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.preload_to_gpu = bool(preload_to_gpu)
        self.device = torch.device(device if preload_to_gpu else "cpu")

        def load(name: str):
            path = self.cache_dir / f"{name}_{split}.pt"
            if not path.exists():
                raise FileNotFoundError(f"cache tensor not found: {path}")
            t = torch.load(path, map_location="cpu")
            return t.to(self.device, non_blocking=False) if self.preload_to_gpu else t

        self.x = load("x")
        self.y = load("y")
        self.pert = load("pert").long()
        self.evidence = load("evidence").float()
        self.evidence_conf = load("evidence_conf").float()
        self.idx = load("idx").long()
        self.perturbation_effect = load("perturbation_effect").float()
        self.target_latents = None
        self.target_latent_mask = None

        cov = torch.load(self.cache_dir / f"cov_{split}.pt", map_location="cpu")
        self.cov = {
            k: (v.long().to(self.device, non_blocking=False) if self.preload_to_gpu else v.long())
            for k, v in cov.items()
        }

        self.input_dim = int(self.meta["input_dim"])
        self.evidence_dim = int(self.meta.get("evidence_dim") or self.evidence.shape[1])
        self.pert_to_id = dict(self.meta["pert_to_id"])
        self.id_to_pert = {int(k): v for k, v in self.meta["id_to_pert"].items()}
        self.pert_cats = list(self.meta["pert_cats"])
        self.n_perts = len(self.pert_cats)
        self.selected_var_names = list(self.meta.get("selected_var_names", []))
        self.cov_dims = {k: int(v) for k, v in self.meta.get("cov_dims", {}).items()}
        self.source_indices = self.idx.cpu().numpy()
        self.train_indices = self.source_indices.tolist() if split == "train" else []

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def load_target_latent(self, path: str | Path) -> None:
        data = torch.load(path, map_location="cpu")
        latent = data["latent"] if isinstance(data, dict) else data
        indices = data.get("indices", torch.arange(len(latent))) if isinstance(data, dict) else torch.arange(len(latent))
        mask_in = data.get("mask", torch.ones(len(indices), dtype=torch.bool)) if isinstance(data, dict) else torch.ones(len(indices), dtype=torch.bool)
        by_idx = {int(idx): i for i, idx in enumerate(indices)}
        rows = []
        masks = []
        for idx in self.idx.cpu().tolist():
            pos = by_idx.get(int(idx))
            valid = pos is not None and bool(mask_in[pos])
            masks.append(valid)
            rows.append(latent[pos].float() if valid else torch.zeros(latent.shape[1], dtype=torch.float32))
        self.target_latents = torch.stack(rows)
        self.target_latent_mask = torch.tensor(masks, dtype=torch.bool)
        if self.preload_to_gpu:
            self.target_latents = self.target_latents.to(self.device)
            self.target_latent_mask = self.target_latent_mask.to(self.device)

    def __getitem__(self, i: int) -> dict:
        pert_id = int(self.pert[i].item())
        source_idx = int(self.idx[i].item())
        target_latent = None
        target_mask = torch.tensor(False, dtype=torch.bool, device=self.x.device)
        if self.target_latents is not None:
            target_mask = self.target_latent_mask[i]
            if bool(target_mask.item()):
                target_latent = self.target_latents[i]
        return {
            "x": self.x[i].float(),
            "y": self.y[i].float(),
            "target": self.y[i].float(),
            "pert": self.pert[i],
            "pert_str": self.id_to_pert.get(pert_id, str(pert_id)),
            "cov": {k: v[i] for k, v in self.cov.items()},
            "evidence": self.evidence[i].float(),
            "evidence_conf": self.evidence_conf[i].float(),
            "idx": self.idx[i],
            "split": self.split,
            "target_latent": target_latent,
            "target_latent_mask": target_mask,
            "perturbation_effect": self.perturbation_effect[i].float(),
            "meta": {"idx": source_idx, "source_idx": source_idx, "cache": True},
        }


def load_cached_train_val(cache_dir: str | Path, preload_to_gpu: bool = False, device: str = "cuda"):
    train = CachedPertDataset(cache_dir, "train", preload_to_gpu=preload_to_gpu, device=device)
    val_split = "val" if (Path(cache_dir) / "x_val.pt").exists() else "test"
    val = CachedPertDataset(cache_dir, val_split, preload_to_gpu=preload_to_gpu, device=device)
    return train, val


class CachedBatchLoader:
    """Vectorized GPU/CPU tensor batch loader for CachedPertDataset.

    This bypasses per-sample ``__getitem__`` and default collate overhead. It is
    intended for preloaded GPU caches, but works on CPU tensors too.
    """

    def __init__(self, dataset: CachedPertDataset, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = int(value)

    def __iter__(self):
        ds = self.dataset
        n = len(ds)
        device = ds.x.device
        order = torch.randperm(n, device=device) if self.shuffle else torch.arange(n, device=device)
        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            ix = order[start:start + self.batch_size]
            if ix.numel() < self.batch_size and self.drop_last:
                continue
            pert = ds.pert[ix]
            idx = ds.idx[ix]
            target_latent = None
            target_mask = torch.zeros(ix.numel(), dtype=torch.bool, device=device)
            if ds.target_latents is not None:
                target_mask = ds.target_latent_mask[ix]
                target_latent = ds.target_latents[ix]
            yield {
                "x": ds.x[ix].float(),
                "y": ds.y[ix].float(),
                "target": ds.y[ix].float(),
                "pert": pert,
                "pert_str": [ds.id_to_pert.get(int(p.item()), str(int(p.item()))) for p in pert],
                "cov": {k: v[ix] for k, v in ds.cov.items()},
                "evidence": ds.evidence[ix].float(),
                "evidence_conf": ds.evidence_conf[ix].float(),
                "idx": idx,
                "split": [ds.split] * ix.numel(),
                "target_latent": target_latent,
                "target_latent_mask": target_mask,
                "perturbation_effect": ds.perturbation_effect[ix].float(),
                "meta": [{"idx": int(i.item()), "source_idx": int(i.item()), "cache": True} for i in idx],
            }
