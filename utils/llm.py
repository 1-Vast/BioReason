"""
LLM calls are restricted to offline biological prior construction
and API connectivity tests. They must not be used inside model.forward
or the training loop.
"""

import os
import json
from pathlib import Path


def get_llm_config():
    """Read LLM config from environment variables. Never hardcode keys."""
    return {
        "api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "",
        "base_url": os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1",
        "model": os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-3.5-turbo",
        "provider": os.environ.get("LLM_PROVIDER", "openai"),
    }


def has_llm_key():
    """Check if any LLM API key is configured. Returns bool."""
    cfg = get_llm_config()
    return bool(cfg["api_key"])


def test_llm_connection(strict=False):
    """Send a minimal API request to verify connectivity.

    Returns dict: {"ok": bool, "reason": str, "model": str}
    Never raises unless strict=True and the call actually fails.
    """
    cfg = get_llm_config()

    if not cfg["api_key"]:
        result = {"ok": False, "reason": "missing_api_key", "model": cfg["model"]}
        if strict:
            raise RuntimeError(result["reason"])
        return result

    try:
        from openai import OpenAI
    except ImportError:
        result = {"ok": False, "reason": "openai_package_missing", "model": cfg["model"]}
        if strict:
            raise RuntimeError(result["reason"])
        return result

    try:
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": 'Return JSON only: {"status": "ok", "domain": "single-cell perturbation"}'}],
            max_tokens=64,
            temperature=0,
        )
        return {"ok": True, "reason": "connected", "model": cfg["model"],
                "response": resp.choices[0].message.content.strip()}
    except Exception as e:
        result = {"ok": False, "reason": str(e)[:200], "model": cfg["model"]}
        if strict:
            raise RuntimeError(str(e)) from e
        return result


def build_perturbation_prior(perturbation, genes=None, pathways=None):
    """Offline biological prior builder. Restricted inputs only.

    Args:
      perturbation: str  perturbation name (e.g., "TP53_KO")
      genes: list[str] or None  small gene list (≤50 recommended)
      pathways: list[str] or None  small pathway list

    Returns: dict  JSON response from LLM

    IMPORTANT:
      - Do NOT pass expression matrices.
      - Do NOT pass patient/sample metadata.
      - max_tokens ≤ 512.
      - Not called during training.
    """
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
        f"Perturbation: {perturbation}\n"
        f"Genes: {gene_str}\n"
        f"Pathways: {path_str}\n\n"
        "Return a JSON object with estimated biological effects.\n"
        "Keys: 'pathway_impact' (list of {pathway, direction, confidence}), "
        "'tf_activity' (list of {tf, direction}), "
        "'description' (1-sentence summary).\n"
        "Keep concise, ≤ 512 tokens."
    )

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
    )
    return parse_json_response(resp.choices[0].message.content)


def parse_json_response(text):
    """Extract JSON from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON block
        import re
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw": text, "ok": False, "reason": "json_parse_failed"}
