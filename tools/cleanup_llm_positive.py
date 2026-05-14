"""Clean up LLM-positive experiment outputs, keeping only whitelisted files.

Whitelist (retained):
  output/llm_positive/compare.csv
  output/llm_positive/search_history.csv
  output/llm_positive/best_config.yaml
  output/llm_positive/report.md
  output/llm_positive/report.json
  output/llm_positive/configs/
  output/llm_positive/logs/
  output/llm_positive/evidence/*audit.csv
  output/llm_positive/preprocess_report.*
  output/llm_positive/audit_*.json

Deleted:
  - tmp_run directory
  - runs directory
  - checkpoints (*.pt, *.pth, *.ckpt)
  - predictions (*.npz)
  - large temp files
  - API cache

Dataset retained:
  llm_*.h5ad in dataset/processed/
"""

import argparse, shutil, gc
from pathlib import Path

DELETE_EXTENSIONS = {".pt", ".pth", ".ckpt", ".npz"}
DELETE_PATTERNS = ["tmp_run", "runs", "llm_cache", "pred"]

RETAIN_DIRS = {"output/llm_positive/configs", "output/llm_positive/logs", "output/llm_positive/evidence"}

def cleanup(root=None):
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    deleted = []
    retained = []
    
    llm_pos = root / "output" / "llm_positive"
    if llm_pos.exists():
        for item in llm_pos.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(root)
            suffix = item.suffix.lower()

            # Always retain these
            if rel.name in ("compare.csv", "search_history.csv", "best_config.yaml",
                            "report.md", "report.json", "download_notes.md"):
                retained.append(str(rel))
                continue
            
            # Check in retained dirs
            in_rd = any(str(rel).replace("\\", "/").startswith(d) for d in RETAIN_DIRS)
            
            if suffix in DELETE_EXTENSIONS:
                item.unlink()
                deleted.append(str(rel))
            elif any(p in str(rel).lower() for p in DELETE_PATTERNS):
                item.unlink()
                deleted.append(str(rel))
            elif in_rd:
                retained.append(str(rel))
            elif "preprocess" in rel.name.lower() or "audit" in rel.name.lower():
                retained.append(str(rel))
            else:
                item.unlink()
                deleted.append(str(rel))

    # Remove directories
    for dname in ["tmp_run", "runs"]:
        d = llm_pos / dname
        if d.exists():
            shutil.rmtree(d)
            deleted.append(f"output/llm_positive/{dname}/ (directory)")

    print(f"LLM-positive cleanup complete.")
    print(f"  Deleted: {len(deleted)}")
    for d in deleted[:15]:
        print(f"    - {d}")
    print(f"  Retained: {len(retained)}")

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

if __name__ == "__main__":
    cleanup()
