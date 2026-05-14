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


GENERIC_PHRASES = [
    "gene expression",
    "cellular process",
    "transcription factor",
]


def qc_prior(prior, dataset_context=None, gene_vocab=None, pathway_vocab=None):
    """Run QC checks on a prior. Returns (passed, adjusted_prior, reason).

    QC checks:
      a. confidence_score < 0.35  →  FAIL  "low_confidence"
      b. description contains only generic phrases  →  generic_prior=True,
         confidence lowered by 0.2
      c. marker genes in gene_vocab  →  marker_in_vocab_ratio
      d. pathway names in pathway_vocab  →  pathway_match_ratio
      e. empty pathway_impact + tf_activity + marker_genes AND short
         description  →  FAIL  "empty_prior"
      f. on failure: evidence set to zeros, evidence_conf=0, source
         marked as source+"_filtered"
    """
    prior = dict(prior)
    reason = "ok"

    # ── a. confidence threshold ───────────────────────────────────────
    conf = float(prior.get("confidence_score", 0.0))
    if conf < 0.35:
        return False, prior, "low_confidence"

    # ── b. generic phrase detection ───────────────────────────────────
    desc = str(prior.get("description", "")).lower()
    generic_prior = False
    for phrase in GENERIC_PHRASES:
        if phrase in desc:
            # simplistic: flag if ANY generic phrase appears
            generic_prior = True
            break

    if generic_prior:
        prior["generic_prior"] = True
        prior["confidence_adjusted"] = max(0.0, conf - 0.2)
    else:
        prior["generic_prior"] = False
        prior["confidence_adjusted"] = conf

    # ── c. marker genes in vocab ──────────────────────────────────────
    marker_in_vocab_ratio = 0.0
    if gene_vocab is not None:
        markers = prior.get("marker_genes", [])
        if markers:
            in_vocab = sum(
                1 for m in markers
                if isinstance(m, dict) and m.get("gene", "") in gene_vocab
            )
            marker_in_vocab_ratio = in_vocab / max(len(markers), 1)
    prior["marker_in_vocab_ratio"] = marker_in_vocab_ratio

    # ── d. pathway match ratio ────────────────────────────────────────
    pathway_match_ratio = 0.0
    if pathway_vocab is not None:
        pathways = prior.get("pathway_impact", [])
        if pathways:
            matched = sum(
                1 for p in pathways
                if isinstance(p, dict) and p.get("pathway", "") in pathway_vocab
            )
            pathway_match_ratio = matched / max(len(pathways), 1)
    prior["pathway_match_ratio"] = pathway_match_ratio

    # ── e. empty prior check ──────────────────────────────────────────
    pathway_impact = prior.get("pathway_impact", [])
    tf_activity = prior.get("tf_activity", [])
    marker_genes = prior.get("marker_genes", [])
    desc_short = len(desc) < 20
    if not pathway_impact and not tf_activity and not marker_genes and desc_short:
        return False, prior, "empty_prior"

    return True, prior, reason


def prior_to_structured(prior):
    """Extract structured fields from a prior dict for structured encoding."""
    return {
        "description": prior.get("description", ""),
        "confidence_score": float(prior.get("confidence_score", 0.0)),
        "confidence_adjusted": float(prior.get("confidence_adjusted", prior.get("confidence_score", 0.0))),
        "source": prior.get("source", "unknown"),
        "pathway_impact": prior.get("pathway_impact", []),
        "tf_activity": prior.get("tf_activity", []),
        "marker_genes": prior.get("marker_genes", []),
        "response_programs": prior.get("response_programs", []),
        "perturbation_gene": prior.get("perturbation_gene", ""),
        "perturbation_type": prior.get("perturbation_type", ""),
        "expected_direction": prior.get("expected_direction", "unknown"),
        "generic_prior": prior.get("generic_prior", False),
        "marker_in_vocab_ratio": prior.get("marker_in_vocab_ratio", 0.0),
        "pathway_match_ratio": prior.get("pathway_match_ratio", 0.0),
        "caveats": prior.get("caveats", ""),
    }


def validate_prior(obj, min_conf=0.35):
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
                   evidence_dim, encoder="hash", model_name=None,
                   max_llm_calls=50, max_llm_tokens=30000,
                   llm_max_tokens=512, llm_cache=None, llm_sleep=0.0,
                   dry_run=False,
                   evidence_encoder_mode=None, evidence_schema="bio_v2",
                   dataset_context=None, gene_vocab=None, pathway_vocab=None):
    from .kb import load_kb, normalize_pert, query_kb
    from .text import TextEncoder

    if evidence_encoder_mode is None:
        evidence_encoder_mode = encoder

    if pert_key not in adata.obs:
        raise KeyError(f"adata.obs['{pert_key}'] not found")

    kb = load_kb(kb_path)
    text_encoder = TextEncoder(
        dim=evidence_dim,
        mode=evidence_encoder_mode,
        model_name=model_name,
        evidence_schema=evidence_schema,
        pathway_vocab=pathway_vocab,
        program_vocab=None,
    )
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
            "perturbation_gene": "",
            "source": "unknown",
            "evidence_encoder_mode": evidence_encoder_mode,
            "confidence_raw": 0.0,
            "confidence_adjusted": 0.0,
            "used_evidence": False,
            "filtered_reason": "",
            "generic_prior": False,
            "pathway_count": 0,
            "tf_count": 0,
            "marker_count": 0,
            "program_count": 0,
            "marker_in_vocab_ratio": 0.0,
            "evidence_norm": 0.0,
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
                "confidence_raw": 1.0, "confidence_adjusted": 1.0,
                "filtered_reason": "control",
                "llm_skipped_reason": "control",
                "used_evidence": False,
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

        # ── validate_prior (basic checks, min_conf=0.35) ──────────────
        valid, cleaned, reason = validate_prior(prior, min_conf=min_conf)
        if not valid:
            pert_to_vec[pert] = zero.copy()
            row.update({
                "pert": pert, "valid": False,
                "source": cleaned.get("source", "zero"),
                "confidence_raw": float(cleaned.get("confidence_score", 0.0)),
                "confidence_adjusted": 0.0,
                "filtered_reason": reason,
                "used_evidence": False,
                "evidence_norm": 0.0,
            })
            if not row["llm_skipped_reason"] and reason:
                row["llm_skipped_reason"] = reason
            audit_rows.append(row)
            continue

        # ── qc_prior (advanced QC) ────────────────────────────────────
        qc_passed, qc_prior_out, qc_reason = qc_prior(
            cleaned,
            dataset_context=dataset_context,
            gene_vocab=gene_vocab,
            pathway_vocab=pathway_vocab,
        )

        # fill audit fields from qc_prior output
        row["confidence_raw"] = float(cleaned.get("confidence_score", 0.0))
        row["confidence_adjusted"] = float(qc_prior_out.get("confidence_adjusted", row["confidence_raw"]))
        row["generic_prior"] = bool(qc_prior_out.get("generic_prior", False))
        row["pathway_count"] = len(qc_prior_out.get("pathway_impact", []))
        row["tf_count"] = len(qc_prior_out.get("tf_activity", []))
        row["marker_count"] = len(qc_prior_out.get("marker_genes", []))
        row["program_count"] = len(qc_prior_out.get("response_programs", []))
        row["marker_in_vocab_ratio"] = float(qc_prior_out.get("marker_in_vocab_ratio", 0.0))
        row["perturbation_gene"] = str(qc_prior_out.get("perturbation_gene", ""))

        if not qc_passed:
            pert_to_vec[pert] = zero.copy()
            row.update({
                "pert": pert, "valid": False,
                "source": str(qc_prior_out.get("source", "unknown")) + "_filtered",
                "filtered_reason": qc_reason,
                "used_evidence": False,
                "evidence_norm": 0.0,
            })
            audit_rows.append(row)
            continue

        # ── encode evidence ───────────────────────────────────────────
        row["used_evidence"] = True

        if evidence_encoder_mode in ("structured", "hybrid"):
            structured = prior_to_structured(qc_prior_out)
            vec = text_encoder.encode([structured])[0].astype(np.float32)
        else:
            vec = text_encoder.encode([prior_to_text(qc_prior_out)])[0].astype(np.float32)

        row["evidence_norm"] = float(np.linalg.norm(vec))
        row["source"] = str(qc_prior_out.get("source", "unknown"))
        row["valid"] = True
        row["filtered_reason"] = "ok"
        pert_to_vec[pert] = vec
        audit_rows.append(row)

    if use_llm and llm_cache:
        save_llm_cache(llm_cache, cache)

    audit = pd.DataFrame(audit_rows)
    by_pert = audit.set_index("pert").to_dict(orient="index") if not audit.empty else {}
    evidence = np.vstack([pert_to_vec.get(p, zero) for p in pert_per_cell]).astype(np.float32)
    conf = np.array([by_pert.get(p, {}).get("confidence_adjusted",
                   by_pert.get(p, {}).get("confidence_raw", 0.0)) for p in pert_per_cell], dtype=np.float32)
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
