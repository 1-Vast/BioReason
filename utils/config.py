"""Config loading utilities: YAML parse, merge, env loader."""

import os
import sys
from pathlib import Path


def load_env(dotenv_path=None):
    """Load .env file if python-dotenv is available."""
    if dotenv_path is None:
        dotenv_path = Path(__file__).parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
    except ImportError:
        pass


def load_yaml(path):
    """Load a YAML file safely. Returns dict or exits on missing dependency."""
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("PyYAML required. Install: pip install pyyaml")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file not found: {path}")
        return {}


def merge(*paths):
    """Merge multiple YAML configs. Later values override earlier ones."""
    merged = {}
    for path in paths:
        if os.path.isfile(path):
            cfg = load_yaml(path)
            for key, val in cfg.items():
                if key == "includes":
                    continue
                if isinstance(val, dict) and isinstance(merged.get(key), dict):
                    merged[key].update(val)
                else:
                    merged[key] = val
    return merged


def load(path, *overrides):
    """Load from path, optionally merged with overrides."""
    return merge(path, *overrides)
