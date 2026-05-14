"""Tune batch size on a tensor cache."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="output/llm_stability_remote/batch_tune.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--sizes", default="128,256,512,1024,2048")
    args = ap.parse_args()
    rows = []
    for size in [int(x) for x in args.sizes.split(",") if x.strip()]:
        out_tmp = Path(args.out).with_name(f".batch_{size}.json")
        cmd = [sys.executable, "tools/profile_pipeline.py", "--cache_dir", args.cache_dir,
               "--config", args.config, "--device", args.device, "--batches", str(args.batches),
               "--batch_size", str(size), "--out", str(out_tmp)]
        try:
            p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
            if p.returncode != 0:
                rows.append({"batch_size": size, "status": "failed", "log_tail": p.stdout[-2000:]})
                continue
            data = json.loads(out_tmp.read_text(encoding="utf-8"))
            data["status"] = "pass"
            rows.append(data)
        except RuntimeError as exc:
            rows.append({"batch_size": size, "status": "failed", "reason": str(exc)})
        except subprocess.TimeoutExpired:
            rows.append({"batch_size": size, "status": "timeout"})
    valid = [r for r in rows if r.get("status") == "pass"]
    best = max(valid, key=lambda r: (r.get("cells_per_sec", 0), r.get("gpu_util_mean") or 0)) if valid else None
    result = {"trials": rows, "best": best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
