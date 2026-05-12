"""Test: kb.py query_kb with toy dict, exact and fuzzy matching, gene combination."""
print("--- test_prior_kb ---")
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from utils.kb import query_kb, load_kb, _norm, parse_pert_gene, combine_priors

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

# 7) combine_priors with single
r = combine_priors([toy["TP53_KO"]])
assert r["description"] == "TP53 knockout"
print("  7) combine single OK")

# 8) combine_priors with multiple
r = combine_priors([toy["TP53_KO"], toy["MYC_OE"]])
assert "TP53 knockout" in r["description"]
assert "MYC overexpression" in r["description"]
assert r["confidence_score"] <= 0.95  # min of the two
print("  8) combine multiple OK")

print("ALL OK")
