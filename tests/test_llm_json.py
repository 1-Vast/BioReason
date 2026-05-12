"""Test: parse_json_response with markdown-fenced JSON and bad JSON."""
print("--- test_llm_json ---")
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from utils.llm import parse_json_response
from utils.evi import parse_llm_response

# 1) Clean JSON
r = parse_json_response('{"key": "value", "num": 42}')
assert r == {"key": "value", "num": 42}, f"Got {r}"
print("  1) clean JSON OK")

# 2) Markdown-fenced JSON
r = parse_json_response('```json\n{"key": "value"}\n```')
assert r == {"key": "value"}, f"Got {r}"
print("  2) markdown-fenced OK")

# 3) Markdown-fenced without json tag
r = parse_json_response('```\n{"key": "value"}\n```')
assert r == {"key": "value"}, f"Got {r}"
print("  3) markdown-fenced no tag OK")

# 4) JSON embedded in text (evi parse_llm_response)
r = parse_llm_response('Here is the result: {"key": "value"} thank you')
assert r == {"key": "value"}, f"Got {r}"
print("  4) embedded JSON via parse_llm_response OK")

# 5) Bad JSON → fallback
r = parse_json_response('not json at all')
assert isinstance(r, dict)
assert r.get("ok") is False or "reason" in r
print(f"  5) bad JSON fallback OK: {r}")

# 6) Bad JSON via parse_llm_response → dict with ok=False
r = parse_llm_response('just some text')
assert isinstance(r, dict)
assert r.get("ok") is False
print(f"  6) bad JSON parse_llm_response OK: {r}")

# 7) Empty string
r = parse_json_response('')
assert isinstance(r, dict)
print("  7) empty string OK")

# 8) JSON with nested objects
r = parse_json_response('{"a":{"b":1,"c":[1,2,3]}}')
assert r["a"]["b"] == 1
print("  8) nested JSON OK")

print("ALL OK")
