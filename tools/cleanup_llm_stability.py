"""Clean up LLM stability experiment outputs."""
import shutil, gc
from pathlib import Path

DELETE_EXTENSIONS = {".pt", ".pth", ".ckpt", ".npz"}
DELETE_PATTERNS = ["tmp_run", "runs", "llm_cache", "pred"]
RETAIN_DIRS = {"output/llm_stability/configs", "output/llm_stability/logs", "output/llm_stability/evidence"}

def cleanup(root=None):
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    deleted = []
    retained = []
    
    sd = root / "output/llm_stability"
    if sd.exists():
        for item in sd.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(root)
            suffix = item.suffix.lower()
            
            if rel.name in ("results.csv", "summary.csv", "summary.json", "best_config.yaml",
                            "report.md", "report.json", "search_history.csv"):
                retained.append(str(rel))
                continue
            
            in_rd = any(str(rel).replace("\\", "/").startswith(d) for d in RETAIN_DIRS)
            
            if suffix in DELETE_EXTENSIONS:
                item.unlink(); deleted.append(str(rel))
            elif any(p in str(rel).lower() for p in DELETE_PATTERNS):
                item.unlink(); deleted.append(str(rel))
            elif in_rd:
                retained.append(str(rel))
            elif "preprocess" in rel.name.lower() or "audit" in rel.name.lower():
                retained.append(str(rel))
            else:
                item.unlink(); deleted.append(str(rel))
    
    for dname in ["tmp_run", "runs"]:
        d = sd / dname
        if d.exists():
            shutil.rmtree(d)
            deleted.append(f"output/llm_stability/{dname}/ (dir)")
    
    print(f"Cleanup: {len(deleted)} deleted, {len(retained)} retained")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass

if __name__ == "__main__":
    cleanup()
