"""Local biological knowledge base query. No model dependencies."""

import json
import os
import re
from pathlib import Path


def load_kb(path=None):
    if path is None:
        path = Path(__file__).parent.parent / "dataset" / "kb" / "prior.json"
    if not path or not os.path.isfile(str(path)):
        return {}
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def normalize_pert(pert):
    s = str(pert).strip().upper()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\bKNOCKOUT\b", "KO", s)
    s = re.sub(r"\bKNOCKDOWN\b", "KD", s)
    s = re.sub(r"\bOVEREXPRESSION\b|\bOVEREXPRESS\b", "OE", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm(s):
    return normalize_pert(s)


def parse_pert_gene(pert):
    s = str(pert).strip()
    s = re.sub(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)(KO|KD|OE)$", r"\1+\2", s)
    s = normalize_pert(s)
    s = re.sub(r"\b(KO|KD|OE)\b", "", s)
    return [g.strip() for g in re.split(r"[+;,\s]+", s) if g.strip()]


def _dedupe(items, keys):
    out, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        marker = tuple(str(item.get(k, "")).upper() for k in keys)
        if marker not in seen:
            seen.add(marker)
            out.append(item)
    return out


def combine_prior(priors):
    priors = [p for p in priors if isinstance(p, dict)]
    if not priors:
        return None
    if len(priors) == 1:
        return priors[0]

    desc = "; ".join(p.get("description", "") for p in priors if p.get("description"))
    confs = []
    pathways, tfs, markers = [], [], []
    for prior in priors:
        try:
            confs.append(float(prior.get("confidence_score", 0.0)))
        except (TypeError, ValueError):
            confs.append(0.0)
        pathways.extend(prior.get("pathway_impact", []) or [])
        tfs.extend(prior.get("tf_activity", []) or [])
        markers.extend(prior.get("marker_genes", []) or [])

    return {
        "description": desc,
        "confidence_score": sum(confs) / max(len(confs), 1),
        "source": "local_combined",
        "pathway_impact": _dedupe(pathways, ("pathway", "direction")),
        "tf_activity": _dedupe(tfs, ("tf", "direction")),
        "marker_genes": _dedupe(markers, ("gene", "direction")),
    }


def query_kb(pert, kb):
    if not kb:
        return None
    if pert in kb:
        return kb[pert]

    norm_map = {normalize_pert(k): k for k in kb}
    norm = normalize_pert(pert)
    if norm in norm_map:
        return kb[norm_map[norm]]

    genes = parse_pert_gene(pert)
    if len(genes) > 1:
        parts = []
        for gene in genes:
            for suffix in ("KO", "KD", "OE", ""):
                candidate = normalize_pert(f"{gene} {suffix}".strip())
                if candidate in norm_map:
                    parts.append(kb[norm_map[candidate]])
                    break
            else:
                gene_norm = normalize_pert(gene)
                for key, prior in kb.items():
                    if normalize_pert(key).split(" ")[0] == gene_norm:
                        parts.append(prior)
                        break
        if parts:
            return combine_prior(parts)

    return None
