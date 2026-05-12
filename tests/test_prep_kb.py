"""Test tools/kb.py: query_kb with exact, fuzzy, combination matching."""
print("--- test_prep_kb ---")
import sys, json, tempfile, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from tools.kb import query_kb, load_kb, _norm, parse_pert_gene, combine_prior

# ── Toy KB ──
toy = {
    "TP53_KO": {
        "description": "TP53 knockout",
        "confidence_score": 0.95,
        "source": "local",
        "pathway_impact": [{"pathway": "DNA damage", "direction": "down", "confidence": 0.9}],
        "tf_activity": [{"tf": "TP53", "direction": "down"}],
        "marker_genes": [{"gene": "CDKN1A", "direction": "down"}],
    },
    "MYC_OE": {
        "description": "MYC overexpression",
        "confidence_score": 0.9,
        "source": "local",
        "pathway_impact": [{"pathway": "cell cycle", "direction": "up", "confidence": 0.85}],
        "tf_activity": [{"tf": "MYC", "direction": "up"}],
        "marker_genes": [{"gene": "CCND1", "direction": "up"}],
    },
}

# 1) Exact match
r = query_kb("TP53_KO", kb=toy)
assert r is not None, "Exact TP53_KO should hit"
assert r["confidence_score"] == 0.95
print("  1) exact match OK")

# 2) Non-existent
r = query_kb("NO_SUCH_PERT", kb=toy)
assert r is None, "Unknown pert should miss"
print("  2) miss OK")

# 3) Fuzzy match: underscores/spaces/case
r = query_kb("tp53 ko", kb=toy)
assert r is not None, "Fuzzy match should hit"
assert r["confidence_score"] == 0.95
print("  3) fuzzy match OK")

# 4) _norm function
assert _norm("TP53_KO") == _norm("tp53 ko")
assert _norm("MYC-OE") == _norm("myc oe")
print("  4) _norm consistent OK")

# 5) parse_pert_gene
assert parse_pert_gene("TP53_KO") == ["TP53"]
assert parse_pert_gene("TP53 knockdown") == ["TP53"]
assert parse_pert_gene("TP53+MYC_KO") == ["TP53", "MYC"]
print("  5) parse_pert_gene OK")

# 6) Gene combination
r = query_kb("TP53+MYC_KO", kb=toy)
assert r is not None, "Combo should combine"
assert r["source"] == "local_combined"
print("  6) gene combination OK")

# 7) combine_prior with single
r = combine_prior([toy["TP53_KO"]])
assert r["description"] == "TP53 knockout"
print("  7) combine single OK")

# 8) combine_prior with multiple
r = combine_prior([toy["TP53_KO"], toy["MYC_OE"]])
assert "TP53 knockout" in r["description"]
assert "MYC overexpression" in r["description"]
assert 0.8 <= r["confidence_score"] <= 1.0
print("  8) combine multiple OK")

# 9) load_kb with temp file
with tempfile.TemporaryDirectory() as tmpdir:
    kb_path = os.path.join(tmpdir, "kb.json")
    with open(kb_path, "w") as f:
        json.dump(toy, f)
    loaded = load_kb(kb_path)
    assert "TP53_KO" in loaded
    assert loaded["TP53_KO"]["confidence_score"] == 0.95
    print("  9) load_kb from file OK")

# 10) load_kb with nonexistent path → empty dict
loaded = load_kb("/nonexistent/path.json")
assert loaded == {}
print("  10) load_kb missing file OK")

print("ALL OK")
