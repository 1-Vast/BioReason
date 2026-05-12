"""
LLM calls restricted to offline prior construction and API tests.
Never in model.forward or training loop.
"""

import os, json, re
from pathlib import Path


def get_llm_config():
    return {
        "api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "",
        "base_url": os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1",
        "model": os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-3.5-turbo",
        "provider": os.environ.get("LLM_PROVIDER", "openai"),
    }


def has_llm_key():
    return bool(get_llm_config()["api_key"])


def test_llm_connection(strict=False):
    cfg = get_llm_config()
    if not cfg["api_key"]:
        if strict: raise RuntimeError("missing_api_key")
        return {"ok": False, "reason": "missing_api_key", "model": cfg["model"]}
    try:
        from openai import OpenAI
    except ImportError:
        if strict: raise RuntimeError("openai_package_missing")
        return {"ok": False, "reason": "openai_package_missing", "model": cfg["model"]}
    try:
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": 'Return JSON only: {"status":"ok","domain":"single-cell perturbation"}'}],
            max_tokens=64, temperature=0)
        return {"ok": True, "reason": "connected", "model": cfg["model"],
                "response": resp.choices[0].message.content.strip()}
    except Exception as e:
        if strict: raise RuntimeError(str(e))
        return {"ok": False, "reason": str(e)[:200], "model": cfg["model"]}


def build_perturbation_prior(perturbation, genes=None, pathways=None):
    """Strict JSON-only LLM query for perturbation prior. Returns a dict."""
    cfg = get_llm_config()
    if not cfg["api_key"]:
        return {"ok": False, "reason": "missing_api_key"}
    try:
        from openai import OpenAI
    except ImportError:
        return {"ok": False, "reason": "openai_package_missing"}

    gene_str = ", ".join(genes[:30]) if genes else "N/A"
    path_str = ", ".join(pathways[:10]) if pathways else "N/A"

    prompt = (
        f"Perturbation: {perturbation}\nGenes: {gene_str}\nPathways: {path_str}\n\n"
        "Return ONLY valid JSON (no markdown, no explanation) with this schema:\n"
        '{"description":"brief 1-2 sentence biological mechanism",'
        '"confidence_score":0.0-1.0,'
        '"source":"llm",'
        '"pathway_impact":[{"pathway":"name","direction":"up|down|unknown","confidence":0.0-1.0}],'
        '"tf_activity":[{"tf":"name","direction":"up|down|unknown"}],'
        '"marker_genes":[{"gene":"name","direction":"up|down|unknown"}]}\n\n'
        "If biological mechanism is uncertain, set confidence_score below 0.5.\n"
        "Do not invent precise mechanisms. Do not include expression data."
    )

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, temperature=0)
    return parse_json_response(resp.choices[0].message.content)


def parse_json_response(text):
    """Extract JSON from text robustly. Returns dict."""
    if isinstance(text, dict):
        return text
    if text is None:
        return {"ok": False, "reason": "json_parse_failed"}
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text) or re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except json.JSONDecodeError: pass
    return {"ok": False, "reason": "json_parse_failed", "raw": text[:200]}
