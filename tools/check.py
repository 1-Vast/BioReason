"""Run BioReason verification checks from the tools entrypoint."""

from pathlib import Path
import runpy
import sys


def main():
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "tools" / "checks" / "check.py"), run_name="__main__")


if __name__ == "__main__":
    main()