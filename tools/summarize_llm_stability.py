"""Summarize remote LLM stability results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml


def fnum(row: dict, key: str) -> float | None:
    try:
        val = row.get(key, "")
        if val == "":
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def std(vals: list[float]) -> float:
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rows = list(csv.DictReader(open(args.results, newline="", encoding="utf-8"))) if Path(args.results).exists() else []
    passed = [r for r in rows if r.get("status") == "pass" and r.get("evidence") != "zero"]
    failed = [r for r in rows if r.get("status") not in ("pass", "", None)]

    by_config: dict[str, list[dict]] = defaultdict(list)
    by_seed: dict[str, list[dict]] = defaultdict(list)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in passed:
        by_config[r.get("config", "")].append(r)
        by_seed[r.get("seed", "")].append(r)
        by_split[r.get("split", "")].append(r)

    summary_rows = []
    for config, cr in sorted(by_config.items()):
        ddeg = [v for r in cr if (v := fnum(r, "delta_deg")) is not None]
        dtop = [v for r in cr if (v := fnum(r, "delta_top50")) is not None]
        ddel = [v for r in cr if (v := fnum(r, "delta_delta_pearson")) is not None]
        held = [v for r in cr if "heldout" in r.get("split", "") and (v := fnum(r, "delta_deg")) is not None]
        low = [v for r in cr if "lowcell" in r.get("split", "") and (v := fnum(r, "delta_deg")) is not None]
        seed_means = {}
        seed_pos = 0
        for seed in ["42", "123", "2024"]:
            vals = [v for r in cr if r.get("seed") == seed and (v := fnum(r, "delta_deg")) is not None]
            seed_means[f"seed{seed}_mean_delta_deg"] = round(mean(vals), 6) if vals else ""
            if vals and mean(vals) > 0:
                seed_pos += 1
        pos = sum(1 for v in ddeg if v > 0)
        summary_rows.append({
            "config": config,
            "n": len(ddeg),
            "mean_delta_deg": round(mean(ddeg), 6),
            "std_delta_deg": round(std(ddeg), 6),
            "positive_rate": round(pos / len(ddeg), 6) if ddeg else 0.0,
            "mean_delta_top50": round(mean(dtop), 6),
            "mean_delta_delta_pearson": round(mean(ddel), 6),
            "heldout_mean_delta_deg": round(mean(held), 6),
            "lowcell_mean_delta_deg": round(mean(low), 6),
            "seeds_positive": seed_pos,
            **seed_means,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        best = max(summary_rows, key=lambda r: (r["positive_rate"], r["mean_delta_deg"]))
        (out_dir / "best_config.yaml").write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")
    else:
        best = {}
        (out_dir / "summary.csv").write_text("", encoding="utf-8")
        (out_dir / "best_config.yaml").write_text("{}\n", encoding="utf-8")

    gpu_avg = ""
    gpu_path = out_dir / "gpu_summary.csv"
    if gpu_path.exists():
        gr = list(csv.DictReader(open(gpu_path, newline="", encoding="utf-8")))
        if gr:
            gpu_avg = gr[0].get("avg_gpu_util", "")

    stable = bool(best) and best["positive_rate"] >= 0.60 and best["mean_delta_deg"] > 0 and best["seeds_positive"] >= 2 and best["heldout_mean_delta_deg"] > 0 and best["lowcell_mean_delta_deg"] > 0
    judgment = "stable helpful" if stable else ("conditionally helpful" if best and best["mean_delta_deg"] > 0 else "neutral")
    model_judgment = "promising" if judgment in ("stable helpful", "conditionally helpful") else "uncertain"

    report = {
        "best_config": best,
        "judgment": judgment,
        "model_judgment": model_judgment,
        "failed_runs": failed,
        "gpu_average_utilization": gpu_avg,
        "metric_direction": {
            "loss/error": "improvement = ZERO_loss - LLM_loss",
            "correlation": "improvement = LLM_corr - ZERO_corr",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Remote LLM Stability Enhancement Report",
        "",
        "## 1. Background",
        "- Local result: LLM conditionally helpful, with seed instability.",
        "- This remote run expands seeds and split settings on the 4090 host.",
        "",
        "## 2. Environment",
        "- See `logs/env.txt` and `logs/gpu_before.txt`.",
        "",
        "## 3. Data",
        "- Adamson 10X010 benchmark splits generated under `dataset/processed`.",
        "- Leak audit outputs are under `audit/`.",
        "",
        "## 4. Evidence",
        "- Evidence modes: ZERO, STRUCT_LLM, HYBRID_LLM.",
        "- Evidence audit CSV files are under `evidence_audit/`.",
        "",
        "## 5. GPU Utilization",
        f"- Average GPU utilization: {gpu_avg}",
        "- Monitor log: `logs/gpu_monitor.csv`.",
        "",
        "## 6. Results",
        "- Metric direction: lower error/loss improves as `ZERO_loss - LLM_loss`; correlations improve as `LLM_corr - ZERO_corr`.",
        "- Summary table: `summary.csv`.",
        "",
        "## 7. LLM Judgment",
        f"- {judgment}",
        "",
        "## 8. Model Judgment",
        f"- {model_judgment}",
        "",
        "## 9. Failure Cases",
        f"- Failed runs: {len(failed)}",
        "",
        "## 10. Next Steps",
        "- If still unstable, run a larger curated-KB pass and strengthen the evidence encoder before paper claims.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    search_history = out_dir / "search_history.csv"
    if not search_history.exists():
        with search_history.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["config", "positive_rate", "mean_delta_deg"])
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k, "") for k in ["config", "positive_rate", "mean_delta_deg"]})


if __name__ == "__main__":
    main()
