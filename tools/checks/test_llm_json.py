"""Test: parse_json_response with clean, fenced, embedded, and bad JSON."""
print("--- test_llm_json ---")
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from utils.llm import parse_json_response

r = parse_json_response('{"key": "value", "num": 42}')
assert r == {"key": "value", "num": 42}, f"Got {r}"
print("  1) clean JSON OK")

r = parse_json_response('```json\n{"key": "value"}\n```')
assert r == {"key": "value"}, f"Got {r}"
print("  2) markdown-fenced OK")

r = parse_json_response('```\n{"key": "value"}\n```')
assert r == {"key": "value"}, f"Got {r}"
print("  3) markdown-fenced no tag OK")

r = parse_json_response('Here is the result: {"key": "value"} thank you')
assert r == {"key": "value"}, f"Got {r}"
print("  4) embedded JSON OK")

r = parse_json_response("not json at all")
assert isinstance(r, dict)
assert r.get("ok") is False
assert r.get("reason") == "json_parse_failed"
print(f"  5) bad JSON fallback OK: {r}")

r = parse_json_response("")
assert isinstance(r, dict)
assert r.get("ok") is False
print("  6) empty string OK")

r = parse_json_response('{"a":{"b":1,"c":[1,2,3]}}')
assert r["a"]["b"] == 1
print("  7) nested JSON OK")

print("ALL OK")
