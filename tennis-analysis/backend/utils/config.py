from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else _DEFAULT_CONFIG
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = str(_PROJECT_ROOT)
    for key in ("data_raw", "data_processed", "data_reports", "weights"):
        rel = cfg["paths"][key]
        cfg["paths"][key] = str(_PROJECT_ROOT / rel)
    cfg["project_root"] = root
    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    return load_config()
