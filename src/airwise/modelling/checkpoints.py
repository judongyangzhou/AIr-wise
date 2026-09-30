"""Checkpoint metadata and cache fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def file_fingerprint(path: str | Path) -> str:
    """Return a stable lightweight fingerprint for a model artifact."""
    target = Path(path)
    stat = target.stat()
    payload = f"{target.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inference_fingerprint(
    checkpoints: Mapping[str, str | Path],
    stats_paths: Mapping[str, str | Path],
    *,
    settings: Mapping[str, Any],
) -> str:
    """Fingerprint all inputs that affect a daily uncertainty product."""
    payload = {
        "checkpoints": {
            name: file_fingerprint(path)
            for name, path in sorted(checkpoints.items())
            if Path(path).is_file()
        },
        "stats": {
            name: file_fingerprint(path)
            for name, path in sorted(stats_paths.items())
            if Path(path).is_file()
        },
        "settings": dict(settings),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
