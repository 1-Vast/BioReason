"""Biological evidence validation, conversion, and construction.

JSON validation with confidence filtering.
Prior-to-text conversion for vector encoding.
Evidence matrix construction for AnnData.
"""

import json, re
import numpy as np

_REQUIRED_KEYS = ["description", "confidence_score", "pathway_impact", "tf_activity", "marker_genes"]


def validate_prior(obj, min_conf=0.5):
    """Validate a prior dict. Returns (valid, reason, cleaned_obj)."""
    if not isinstance(obj, dict):
        return False, "not_a_dict", zero_prior("unknown", "not_a_dict")

    for k in _REQUIRED_KEYS:
        if k not in obj:
            obj[k] = [] if k in ("pathway_impact", "tf_activity", "marker_genes") else ("" if k == "description" else 0.0)

    conf = obj.get("confidence_score", 0.0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    obj["confidence_score"] = conf

    if conf < min_conf:
        return False, f"low_confidence ({conf:.2f} < {min_conf})", zero_prior(obj.get("description", "unknown"), f"low_confidence_{conf:.2f}")

    # Ensure list fields are lists
    for k in ("pathway_impact", "tf_activity", "marker_genes"):
        if not isinstance(obj.get(k), list):
            obj[k] = []

    obj.setdefault("source", "unknown")
    return True, "ok", obj


def parse_llm_response(text):
    """Extract JSON from LLM response text. Returns dict or error marker."""
    text = text.strip()
    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find outermost JSON block
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"ok": False, "reason": "json_parse_failed", "raw": text[:200]}


def prior_to_text(obj):
    """Convert structured prior to stable text for encoding."""
    parts = []
    if obj.get("description"):
        parts.append("Description: " + obj["description"])
    pw = obj.get("pathway_impact", [])
    if pw:
        pw_str = "; ".join(f"{p.get('pathway','')} {p.get('direction','')} conf {p.get('confidence','?')}" for p in pw)
        parts.append("Pathways: " + pw_str)
    tf = obj.get("tf_activity", [])
    if tf:
        parts.append("TF activity: " + "; ".join(f"{t.get('tf','')} {t.get('direction','')}" for t in tf))
    mg = obj.get("marker_genes", [])
    if mg:
        parts.append("Markers: " + "; ".join(f"{m.get('gene','')} {m.get('direction','')}" for m in mg))
    return ". ".join(parts) + "."


def zero_prior(pert, reason="unknown"):
    return {"description": "", "confidence_score": 0.0, "source": "zero",
            "pathway_impact": [], "tf_activity": [], "marker_genes": [],
            "reason": reason}


def build_evidence(pert_list, control_label="control", kb=None, use_llm=False,
                   min_conf=0.5, evidence_dim=128, encoder_mode="hash",
                   model_name=None, audit_out=None):
    """Build evidence matrix for a list of perturbation labels.

    Returns:
      evidence: np.ndarray [N, evidence_dim]
      confidence: np.ndarray [N]
      sources: list[str] of length N
      audit: list[dict] per perturbation
    """
    from .kb import query_kb, load_kb as _load_kb
    from .text import TextEncoder

    if kb is None:
        kb = _load_kb()
    encoder = TextEncoder(dim=evidence_dim, mode=encoder_mode, model_name=model_name)

    unique_perts = list(dict.fromkeys(pert_list))  # ordered unique
    pert_to_ev = {}
    audit = []
    local_hits, llm_calls, zeroed = 0, 0, 0

    for pert in unique_perts:
        if pert == control_label:
            ev = np.zeros(evidence_dim, dtype=np.float32)
            pert_to_ev[pert] = ev
            audit.append({"pert": pert, "source": "control", "confidence": 1.0, "valid": True, "reason": "control"})
            continue

        prior = query_kb(pert, kb)
        source = "local"
        if prior is not None:
            local_hits += 1
        elif use_llm:
            source = "llm"
            llm_calls += 1
            from .llm import build_perturbation_prior
            llm_result = build_perturbation_prior(pert)
            prior = parse_llm_response(llm_result) if isinstance(llm_result, str) else llm_result
        else:
            prior = None

        if prior is None or not isinstance(prior, dict):
            zeroed += 1
            ev = np.zeros(evidence_dim, dtype=np.float32)
            pert_to_ev[pert] = ev
            audit.append({"pert": pert, "source": "miss", "confidence": 0.0, "valid": False, "reason": "no_prior"})
            continue

        valid, reason, cleaned = validate_prior(prior, min_conf=min_conf)
        if not valid:
            zeroed += 1
            ev = np.zeros(evidence_dim, dtype=np.float32)
            pert_to_ev[pert] = ev
            audit.append({"pert": pert, "source": source, "confidence": cleaned["confidence_score"], "valid": False, "reason": reason})
            continue

        text = prior_to_text(cleaned)
        ev = encoder.encode([text])[0]
        pert_to_ev[pert] = ev.astype(np.float32)
        audit.append({"pert": pert, "source": source, "confidence": cleaned["confidence_score"], "valid": True, "reason": "ok"})

    evidence = np.array([pert_to_ev.get(p, np.zeros(evidence_dim, dtype=np.float32)) for p in pert_list])
    confidence = np.array([a["confidence"] for a in audit], dtype=np.float32)
    # Map back to per-observation
    pert_to_audit = {a["pert"]: a for a in audit}
    obs_confidence = np.array([pert_to_audit.get(p, {}).get("confidence", 0.0) for p in pert_list])
    obs_sources = [pert_to_audit.get(p, {}).get("source", "miss") for p in pert_list]

    return evidence, obs_confidence, obs_sources, audit


def write_evidence(adata, evidence, confidence, sources, audit):
    """Write evidence data into AnnData."""
    adata.obsm["evidence"] = evidence.astype(np.float32)
    adata.obs["evidence_conf"] = confidence.astype(np.float32)
    adata.obs["evidence_source"] = sources
    if audit:
        import pandas as pd
        adata.uns["evidence_audit"] = pd.DataFrame(audit).to_dict(orient="records")
