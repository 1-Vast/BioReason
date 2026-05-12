"""Stage 0 evidence validation, construction, and AnnData writing."""

import json
import re
import warnings

import numpy as np
import pandas as pd


def zero_prior(pert, reason):
    return {
        "description": "",
        "confidence_score": 0.0,
        "source": "zero",
        "pathway_impact": [],
        "tf_activity": [],
        "marker_genes": [],
        "reason": reason,
    }


def validate_prior(obj, min_conf=0.5):
    if not isinstance(obj, dict):
        return False, zero_prior("unknown", "bad_prior"), "bad_prior"

    prior = dict(obj)
    prior["description"] = prior.get("description") or ""
    prior["source"] = prior.get("source") or "unknown"
    for key in ("pathway_impact", "tf_activity", "marker_genes"):
        if not isinstance(prior.get(key), list):
            prior[key] = []

    try:
        prior["confidence_score"] = float(prior.get("confidence_score", 0.0))
    except (TypeError, ValueError):
        prior["confidence_score"] = 0.0

    if prior["confidence_score"] < min_conf:
        return False, prior, "low_confidence"
    return True, prior, "ok"


def prior_to_text(obj):
    pathways = "; ".join(
        f"{p.get('pathway', '')} {p.get('direction', 'unknown')} confidence {p.get('confidence', '')}".strip()
        for p in obj.get("pathway_impact", [])
    )
    tfs = "; ".join(
        f"{t.get('tf', '')} {t.get('direction', 'unknown')}".strip()
        for t in obj.get("tf_activity", [])
    )
    markers = "; ".join(
        f"{m.get('gene', '')} {m.get('direction', 'unknown')}".strip()
        for m in obj.get("marker_genes", [])
    )
    return "\n".join([
        f"Description: {obj.get('description', '')}",
        f"Pathways: {pathways}.",
        f"TF activity: {tfs}.",
        f"Marker genes: {markers}.",
    ])


def _parse_llm_prior(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {"ok": False, "reason": "bad_prior"}
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"ok": False, "reason": "json_parse_failed"}


def build_evidence(adata, pert_key, control_label, kb_path, use_llm, min_conf,
                   evidence_dim, encoder, model_name=None):
    from .kb import load_kb, query_kb
    from .text import TextEncoder

    if pert_key not in adata.obs:
        raise KeyError(f"adata.obs['{pert_key}'] not found")

    kb = load_kb(kb_path)
    text_encoder = TextEncoder(dim=evidence_dim, mode=encoder, model_name=model_name)
    pert_per_cell = adata.obs[pert_key].astype(str).to_numpy()
    unique_perts = list(dict.fromkeys(pert_per_cell.tolist()))

    llm_available = use_llm
    if use_llm:
        from utils.llm import has_llm_key
        if not has_llm_key():
            warnings.warn("--use_llm set but no API key. Using KB and zero fallback only.")
            llm_available = False

    pert_to_vec = {}
    audit_rows = []
    zero = np.zeros(evidence_dim, dtype=np.float32)

    for pert in unique_perts:
        if pert == control_label:
            pert_to_vec[pert] = zero.copy()
            audit_rows.append({
                "pert": pert, "valid": True, "source": "control",
                "confidence": 1.0, "reason": "control",
                "local_hit": False, "llm_called": False, "zero_evidence": True,
            })
            continue

        prior = query_kb(pert, kb)
        local_hit = prior is not None
        llm_called = False
        if prior is None and llm_available:
            from utils.llm import build_perturbation_prior
            llm_called = True
            prior = _parse_llm_prior(build_perturbation_prior(pert))

        valid, cleaned, reason = validate_prior(prior, min_conf=min_conf)
        if not valid:
            pert_to_vec[pert] = zero.copy()
            audit_rows.append({
                "pert": pert, "valid": False, "source": cleaned.get("source", "zero"),
                "confidence": float(cleaned.get("confidence_score", 0.0)),
                "reason": reason, "local_hit": local_hit,
                "llm_called": llm_called, "zero_evidence": True,
            })
            continue

        pert_to_vec[pert] = text_encoder.encode([prior_to_text(cleaned)])[0].astype(np.float32)
        audit_rows.append({
            "pert": pert, "valid": True, "source": cleaned.get("source", "unknown"),
            "confidence": float(cleaned.get("confidence_score", 0.0)),
            "reason": "ok", "local_hit": local_hit,
            "llm_called": llm_called, "zero_evidence": False,
        })

    audit = pd.DataFrame(audit_rows)
    by_pert = audit.set_index("pert").to_dict(orient="index") if not audit.empty else {}
    evidence = np.vstack([pert_to_vec.get(p, zero) for p in pert_per_cell]).astype(np.float32)
    conf = np.array([by_pert.get(p, {}).get("confidence", 0.0) for p in pert_per_cell], dtype=np.float32)
    source = np.array([by_pert.get(p, {}).get("source", "zero") for p in pert_per_cell], dtype=object)
    return evidence, conf, source, audit


def write_evidence(adata, evidence, conf, source, audit):
    adata.obsm["evidence"] = evidence.astype(np.float32)
    adata.obs["evidence_conf"] = np.asarray(conf, dtype=np.float32)
    adata.obs["evidence_source"] = np.asarray(source, dtype=object)
    if audit is not None:
        if isinstance(audit, pd.DataFrame):
            adata.uns["evidence_audit"] = audit.to_dict(orient="list")
        else:
            adata.uns["evidence_audit"] = audit
