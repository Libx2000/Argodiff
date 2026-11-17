from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve project-relative artifact/output paths."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    project_root = config_path.parent.parent
    cfg = deepcopy(cfg)
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(project_root)

    for section, key in (("data", "stats_path"), ("train", "output_dir")):
        value = Path(cfg[section][key]).expanduser()
        if not value.is_absolute():
            value = project_root / value
        cfg[section][key] = str(value.resolve())

    cfg["data"]["root"] = str(Path(cfg["data"]["root"]).expanduser())
    return cfg


def inclusive_years(bounds: list[int] | tuple[int, int]) -> list[int]:
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError(f"Expected inclusive [start, end] year bounds, got {bounds}")
    return list(range(int(bounds[0]), int(bounds[1]) + 1))

