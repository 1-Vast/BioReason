"""Test: LLM config and connectivity (no real key required)."""
print("--- test_llm_config ---")
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from utils.llm import get_llm_config, has_llm_key, test_llm_connection

cfg = get_llm_config()
assert "api_key" in cfg
assert "base_url" in cfg
assert "model" in cfg
print(f"  1) config OK: provider={cfg['provider']}, model={cfg['model']}")

has = has_llm_key()
print(f"  2) has_llm_key={has} (expected False without .env)")

result = test_llm_connection()
assert isinstance(result, dict)
assert "ok" in result and "reason" in result
print(f"  3) test_llm_connection ok={result['ok']}, reason={result['reason']}")

# Test missing key doesn't raise
assert result["ok"] is False or result["ok"] is True  # either is fine

print("ALL OK")
