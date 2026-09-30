"""Access immutable resources bundled with the ``airwise`` package."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml


def resource(relative_path: str):
    """Return a traversable package resource."""
    return files(__package__).joinpath(relative_path)


def load_yaml_resource(relative_path: str) -> dict[str, Any]:
    """Load a mapping from a bundled YAML resource."""
    target = resource(relative_path)
    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Bundled resource {relative_path!r} must contain a mapping.")
    return payload
