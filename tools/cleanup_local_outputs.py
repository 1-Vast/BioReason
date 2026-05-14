"""Clean BioReason local outputs while preserving experiment conclusions.

The script is intentionally conservative: it only removes known transient
artifacts and writes a retained_files.txt manifest next to the summary.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path


KEEP_FILES = {
    "output/local_evi_fix/compare.csv",
    "output/local_evi_fix/report.md",
    "output/local_evi_fix/report.json",
    "output/llm_positive/compare.csv",
    "output/llm_positive/search_history.csv",
    "output/llm_positive/best_config.yaml",
    "output/llm_positive/report.md",
    "output/llm_positive/report.json",
    "dataset/processed/local_evi_fix_small.h5ad",
    "dataset/processed/llm_seen.h5ad",
    "dataset/processed/llm_heldout.h5ad",
}

KEEP_GLOBS = {
    "output/llm_positive/configs/**",
    "output/llm_positive/logs/*.summary.txt",
    "dataset/processed/llm_lowcell_*.h5ad",
}

DELETE_FILE_GLOBS = {
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
    "**/*.npz",
    "**/pred*.h5ad",
    "**/*llm_cache*.json",
    "dataset/processed/*_tmp*.h5ad",
    "output/**/*.h5ad",
}

DELETE_DIR_NAMES = {"tmp_run", "runs", "stage1", "stage2", "stage3"}


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_kept(rel: str) -> bool:
    if rel in KEEP_FILES:
        return True
    return any(fnmatch.fnmatch(rel, pat) for pat in KEEP_GLOBS)


def should_delete_file(rel: str) -> bool:
    if is_kept(rel):
        return False
    return any(fnmatch.fnmatch(rel, pat) for pat in DELETE_FILE_GLOBS)


def collect(root: Path) -> tuple[list[Path], list[Path], list[str]]:
    files_to_delete: list[Path] = []
    dirs_to_delete: list[Path] = []
    retained: list[str] = []

    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        rel = rel_posix(path, root)
        if path.is_dir() and path.name in DELETE_DIR_NAMES and rel.startswith("output/"):
            dirs_to_delete.append(path)
            continue
        if path.is_file():
            if is_kept(rel):
                retained.append(rel)
            if should_delete_file(rel):
                files_to_delete.append(path)

    dirs_to_delete = sorted(set(dirs_to_delete), key=lambda p: len(p.parts), reverse=True)
    files_to_delete = sorted(set(files_to_delete))
    retained = sorted(set(retained))
    return files_to_delete, dirs_to_delete, retained


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="preserve_results", choices=["preserve_results"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    files, dirs, retained = collect(root)

    lines = [
        "# Local Cleanup Summary",
        f"root: {root}",
        f"mode: {args.mode}",
        f"dry_run: {args.dry_run}",
        "",
        "## Delete candidates",
    ]
    for p in files:
        lines.append(f"FILE {rel_posix(p, root)}")
    for p in dirs:
        lines.append(f"DIR  {rel_posix(p, root)}")

    print("\n".join(lines))

    deleted: list[str] = []
    if not args.dry_run:
        for p in files:
            if p.exists() and p.is_file():
                p.unlink()
                deleted.append(f"FILE {rel_posix(p, root)}")
        for p in dirs:
            if p.exists() and p.is_dir():
                shutil.rmtree(p)
                deleted.append(f"DIR  {rel_posix(p, root)}")

    retained_manifest = out.parent / "retained_files.txt"
    retained_manifest.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")

    lines.extend(["", "## Deleted", *deleted, "", "## Retained", *retained])
    lines.append(f"\nretained_manifest: {retained_manifest}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDeleted: {len(deleted)}")
    print(f"Retained manifest: {retained_manifest}")
    print(f"Summary: {out}")


if __name__ == "__main__":
    main()
