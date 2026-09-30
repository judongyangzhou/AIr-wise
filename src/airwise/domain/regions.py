"""Region definitions used by bulletin pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionDef:
    id: str
    name: str
    representative_city: str
    lat: float
    lon: float
    nuts_id: str | None = None
