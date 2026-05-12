"""Stage 0 evidence validation, construction, and AnnData writing."""

import json
import re
import time
import warnings
from pathlib import Path

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
    if obj.get("ok") is False:
        reason = str(obj.get("reason", "bad_prior"))
        prior = zero_prior("unknown", reason)
        prior["source"] = obj.get("source", "llm")
        return False, prior, reason

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

    if prior.get("source") == "zero" and prior.get("reason"):
        return False, prior, str(prior["reason"])
    if prior["confidence_score"] < min_conf:
        return False, prior, "low_confidence"
    return True, prior, "ok"


def load_llm_cache(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_llm_cache(path, cache):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


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
                   evidence_dim, encoder, model_name=None,
                   max_llm_calls=50, max_llm_tokens=30000,
                   llm_max_tokens=512, llm_cache=None, llm_sleep=0.0,
                   dry_run=False):
    from .kb import load_kb, normalize_pert, query_kb
    from .text import TextEncoder

    if pert_key not in adata.obs:
        raise KeyError(f"adata.obs['{pert_key}'] not found")

    kb = load_kb(kb_path)
    text_encoder = TextEncoder(dim=evidence_dim, mode=encoder, model_name=model_name)
    pert_per_cell = adata.obs[pert_key].astype(str).to_numpy()
    unique_perts = list(dict.fromkeys(pert_per_cell.tolist()))
    max_llm_calls = max(0, int(max_llm_calls))
    max_llm_tokens = max(0, int(max_llm_tokens))
    llm_max_tokens = min(max(1, int(llm_max_tokens)), 1024)
    llm_sleep = max(0.0, float(llm_sleep))
    llm_calls = 0
    llm_tokens_budget_used = 0
    cache = load_llm_cache(llm_cache) if use_llm and llm_cache else {}

    llm_available = use_llm
    if use_llm and not dry_run:
        from utils.llm import has_llm_key
        if not has_llm_key():
            warnings.warn("--use_llm set but no API key. Using KB and zero fallback only.")
            llm_available = False

    pert_to_vec = {}
    audit_rows = []
    zero = np.zeros(evidence_dim, dtype=np.float32)

    def audit_base(pert):
        return {
            "pert": pert,
            "local_hit": False,
            "llm_called": False,
            "llm_cache_hit": False,
            "llm_skipped_reason": "",
            "llm_call_index": -1,
            "llm_max_tokens": llm_max_tokens,
            "llm_budget_used": llm_tokens_budget_used,
            "max_llm_calls": max_llm_calls,
            "max_llm_tokens": max_llm_tokens,
            "dry_run": bool(dry_run),
        }

    for pert in unique_perts:
        row = audit_base(pert)
        if pert == control_label:
            pert_to_vec[pert] = zero.copy()
            row.update({
                "pert": pert, "valid": True, "source": "control",
                "confidence": 1.0, "reason": "control",
                "llm_skipped_reason": "control",
                "zero_evidence": True,
            })
            audit_rows.append(row)
            continue

        prior = query_kb(pert, kb)
        local_hit = prior is not None
        row["local_hit"] = local_hit
        if local_hit:
            row["llm_skipped_reason"] = "local_hit"

        if prior is None and use_llm:
            key = normalize_pert(pert)
            cached = cache.get(key) if cache else None
            if isinstance(cached, dict) and "prior" in cached:
                prior = cached.get("prior")
                row["llm_cache_hit"] = True
                row["llm_skipped_reason"] = "cache_hit"
            elif dry_run:
                prior = zero_prior(pert, "dry_run")
                row["llm_skipped_reason"] = "dry_run"
            elif not llm_available:
                prior = zero_prior(pert, "missing_api_key")
                row["llm_skipped_reason"] = "missing_api_key"
            elif llm_calls >= max_llm_calls:
                prior = zero_prior(pert, "llm_call_limit")
                row["llm_skipped_reason"] = "llm_call_limit"
            elif llm_tokens_budget_used + llm_max_tokens > max_llm_tokens:
                prior = zero_prior(pert, "llm_token_budget")
                row["llm_skipped_reason"] = "llm_token_budget"
            else:
                from utils.llm import build_perturbation_prior
                prior = _parse_llm_prior(build_perturbation_prior(
                    pert,
                    max_tokens=llm_max_tokens,
                ))
                llm_calls += 1
                llm_tokens_budget_used += llm_max_tokens
                row["llm_called"] = True
                row["llm_call_index"] = llm_calls
                row["llm_budget_used"] = llm_tokens_budget_used
                if llm_cache:
                    cache[key] = {
                        "prior": prior,
                        "model": (prior.get("_llm_meta", {}) or {}).get("model", model_name),
                        "llm_max_tokens": llm_max_tokens,
                    }
                    save_llm_cache(llm_cache, cache)
                if llm_sleep > 0:
                    time.sleep(llm_sleep)
        elif prior is None:
            row["llm_skipped_reason"] = "not_requested"

        row["llm_budget_used"] = llm_tokens_budget_used

        if prior is None or not isinstance(prior, dict):
            prior = zero_prior(pert, "no_prior")

        valid, cleaned, reason = validate_prior(prior, min_conf=min_conf)
        if not valid:
            pert_to_vec[pert] = zero.copy()
            row.update({
                "pert": pert, "valid": False, "source": cleaned.get("source", "zero"),
                "confidence": float(cleaned.get("confidence_score", 0.0)),
                "reason": reason, "zero_evidence": True,
            })
            if not row["llm_skipped_reason"] and reason:
                row["llm_skipped_reason"] = reason
            audit_rows.append(row)
            continue

        pert_to_vec[pert] = text_encoder.encode([prior_to_text(cleaned)])[0].astype(np.float32)
        row.update({
            "pert": pert, "valid": True, "source": cleaned.get("source", "unknown"),
            "confidence": float(cleaned.get("confidence_score", 0.0)),
            "reason": "ok", "zero_evidence": False,
        })
        audit_rows.append(row)

    if use_llm and llm_cache:
        save_llm_cache(llm_cache, cache)

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
