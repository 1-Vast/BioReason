"""Local biological knowledge base.

Prioritizes local lookup over LLM queries.
Supports exact, fuzzy, and combination perturbation matching.
"""

import json, os
from pathlib import Path


def load_kb(path=None):
    """Load knowledge base from JSON file. Returns dict or empty dict."""
    if path is None:
        path = Path(__file__).parent.parent / "dataset" / "kb" / "prior.json"
    if not path or not os.path.isfile(str(path)):
        return {}
    try:
        with open(str(path), "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _norm(s):
    """Normalize perturbation string for fuzzy matching."""
    return s.strip().upper().replace("_", " ").replace("-", " ").replace("  ", " ")


def parse_pert_gene(pert_str):
    """Extract gene names from perturbation string like 'TP53_KO' or 'TP53+MYC_KO'."""
    s = pert_str.replace("_KO", "").replace("_KD", "").replace("_OE", "")
    s = s.replace(" knockout", "").replace(" knockdown", "").replace(" overexpress", "")
    genes = [g.strip() for g in s.replace("+", " ").split()]
    return [g for g in genes if g]


def combine_priors(priors):
    """Combine multiple prior dicts into one."""
    if not priors:
        return None
    if len(priors) == 1:
        return priors[0]
    desc = "; ".join(p["description"] for p in priors if p.get("description"))
    conf = min(p.get("confidence_score", 0.5) for p in priors)
    pw = []; tf = []; mg = []
    seen_pathways = set()
    for p in priors:
        for pi in p.get("pathway_impact", []):
            key = pi["pathway"]
            if key not in seen_pathways: seen_pathways.add(key); pw.append(pi)
        for t in p.get("tf_activity", []):
            tf.append(t)
        for m in p.get("marker_genes", []):
            mg.append(m)
    return {"description": desc, "confidence_score": conf, "source": "local_combined",
            "pathway_impact": pw, "tf_activity": tf, "marker_genes": mg}


def query_kb(pert, kb=None):
    """Query KB for perturbation prior. Returns dict or None."""
    if kb is None:
        kb = load_kb()
    if not kb:
        return None

    # Exact match
    if pert in kb:
        return kb[pert]

    # Normalized match
    norm_pert = _norm(pert)
    norm_map = {_norm(k): k for k in kb}
    if norm_pert in norm_map:
        return kb[norm_map[norm_pert]]

    # Gene combination
    genes = parse_pert_gene(pert)
    if len(genes) > 1:
        parts = []
        for g in genes:
            for k, v in kb.items():
                if _norm(g) in _norm(k):
                    parts.append(v)
                    break
        if parts:
            return combine_priors(parts)

    return None
