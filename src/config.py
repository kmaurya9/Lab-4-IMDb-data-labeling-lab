"""
config.py — Reads config.yaml so paths and settings live in one place.
Author: Kshitij Maurya
"""

from pathlib import Path
from typing import Optional, Union

import yaml

LAB_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Optional[Union[str, Path]] = None) -> dict:
    """Load config.yaml (from the lab root unless another path is given)."""
    path = Path(path) if path else LAB_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def lab_path(relative: str) -> Path:
    """Turn a path from config.yaml (relative to the lab root) into an absolute path."""
    return LAB_ROOT / relative
