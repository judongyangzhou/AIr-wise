"""Read cached CAMS Policy source-receptor inputs without network access."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from airwise.config import load_settings

DEFAULT_INVENTORY = "TNO"
DEFAULT_COUNTRY = "Germany"


def parse_date(value: str) -> date:
    for pattern in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"Invalid date {value!r}; use YYYYMMDD (or YYYY-MM-DD)"
    )


def default_out_dir() -> Path:
    return load_settings().paths.cams_policy_forecast


def _safe_filename(name: str) -> str:
    cleaned = "".join(
        character
        if character.isalnum() or character in "-_."
        else "_"
        for character in name.strip()
    )
    return cleaned.strip("._") or "city"


def city_json_path(
    out_dir: str | Path,
    report_date: date,
    city: str,
    *,
    inventory: str = DEFAULT_INVENTORY,
) -> Path:
    return (
        Path(out_dir)
        / f"{report_date:%Y%m%d}"
        / inventory
        / f"{_safe_filename(city)}.json"
    )


def cities_cache_path(out_dir: str | Path, year: int) -> Path:
    return Path(out_dir) / "cities" / f"City_{int(year)}.json"


def renaming_cache_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "config" / "renaming.json"


def load_cached_cities(year: int, *, out_dir: str | Path) -> list[dict[str, Any]]:
    path = cities_cache_path(out_dir, year)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cached CAMS Policy city catalogue not found: {path}. "
            "Run the acquisition pipeline first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Cached city catalogue must be an array: {path}")
    return payload


def _city_matches_country(city: dict[str, Any], country: str) -> bool:
    wanted = country.strip()
    if not wanted:
        raise ValueError("country must be a non-empty string")
    city_country = str(city.get("country") or "")
    if city_country.casefold() == wanted.casefold():
        return True
    code = str(city.get("code") or "")
    return len(wanted) == 2 and code.upper().startswith(wanted.upper())


def filter_cities(
    cities: Iterable[dict[str, Any]],
    country: str = DEFAULT_COUNTRY,
) -> list[dict[str, Any]]:
    matched = [city for city in cities if _city_matches_country(city, country)]
    matched.sort(key=lambda city: str(city.get("name") or ""))
    return matched
