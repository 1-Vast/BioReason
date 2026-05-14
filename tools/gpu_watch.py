"""Small nvidia-smi monitor used by remote stability experiments."""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path


QUERY = (
    "timestamp,name,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,power.draw"
)


class GpuWatch:
    def __init__(self, out_csv: str | Path, interval: int = 30, threshold: float = 30.0):
        self.out_csv = Path(out_csv)
        self.interval = int(interval)
        self.threshold = float(threshold)
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "nvidia-smi",
            f"--query-gpu={QUERY}",
            "--format=csv",
            "-l",
            str(self.interval),
        ]
        fh = self.out_csv.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        self._fh = fh

    def stop(self) -> dict:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if hasattr(self, "_fh"):
            self._fh.close()
        return summarize_gpu_csv(self.out_csv, self.threshold)


def _num(text: str) -> float | None:
    clean = "".join(ch for ch in str(text) if ch.isdigit() or ch in ".-")
    try:
        return float(clean)
    except ValueError:
        return None


def summarize_gpu_csv(path: str | Path, threshold: float = 30.0) -> dict:
    path = Path(path)
    vals, mems = [], []
    if path.exists():
        with path.open(newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                util = _num(row.get(" utilization.gpu [%]", row.get("utilization.gpu [%]", "")))
                mem = _num(row.get(" memory.used [MiB]", row.get("memory.used [MiB]", "")))
                if util is not None:
                    vals.append(util)
                if mem is not None:
                    mems.append(mem)
    avg = sum(vals) / len(vals) if vals else 0.0
    summary = {
        "samples": len(vals),
        "avg_gpu_util": round(avg, 3),
        "max_memory_mib": round(max(mems), 3) if mems else 0.0,
        "warning": avg < threshold if vals else True,
        "threshold": threshold,
    }
    warn_path = path.with_suffix(".warning.txt")
    if summary["warning"]:
        warn_path.write_text(
            f"Average GPU utilization {avg:.2f}% below threshold {threshold:.2f}%\n",
            encoding="utf-8",
        )
    return summary


def one_shot() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=30.0)
    args = ap.parse_args()
    watch = GpuWatch(args.csv, args.interval, args.threshold)
    watch.start()
    time.sleep(args.seconds)
    print(json.dumps(watch.stop(), indent=2))
