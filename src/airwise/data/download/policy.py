"""Download CAMS Policy source-receptor city forecasts.

The CAMS Policy Tools API provides country contributions to city PM
concentrations. Given a date, this module:

1. Loads the city catalogue from ``/api/available_parameters/City/{year}``.
2. Caches ``/config/renaming.json`` for shortname labels.
3. Filters cities belonging to a country (default: Germany).
4. Downloads each city's daily forecast JSON from
   ``/api/daily_forecast/{YYYY-MM-DD}/{inventory}/{city}``.

Example::

    airwise-download-policy-forecast --date 2026-08-01
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from airwise.config import load_config, require_config, resolve_repo_path

logger = logging.getLogger(__name__)

SITE_ORIGIN = "https://policy.atmosphere.copernicus.eu"
API_BASE = f"{SITE_ORIGIN}/api"
RENAMING_URL = f"{SITE_ORIGIN}/config/renaming.json"
DEFAULT_INVENTORY = "TNO"
DEFAULT_COUNTRY = "Germany"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3
USER_AGENT = "airwise/0.1 (ECMWF Code for Earth 2026)"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def parse_date(value: str) -> date:
    """Parse YYYYMMDD (preferred) or YYYY-MM-DD."""
    text = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date {value!r}; use YYYYMMDD (or YYYY-MM-DD)"
    )


def default_out_dir() -> Path:
    return resolve_repo_path(require_config(load_config(), "paths", "cams_policy_forecast"))


def cities_url(year: int) -> str:
    return f"{API_BASE}/available_parameters/City/{int(year)}"


def daily_forecast_url(
    report_date: date,
    city: str,
    *,
    inventory: str = DEFAULT_INVENTORY,
    redistribution: bool = False,
) -> str:
    query = urlencode({"redistribution": str(bool(redistribution)).lower()})
    return (
        f"{API_BASE}/daily_forecast/{report_date.isoformat()}/"
        f"{quote(inventory, safe='')}/{quote(city, safe='')}?{query}"
    )


def city_json_path(
    out_dir: str | Path,
    report_date: date,
    city: str,
    *,
    inventory: str = DEFAULT_INVENTORY,
) -> Path:
    return Path(out_dir) / f"{report_date:%Y%m%d}" / inventory / f"{_safe_filename(city)}.json"


def cities_cache_path(out_dir: str | Path, year: int) -> Path:
    return Path(out_dir) / "cities" / f"City_{int(year)}.json"


def renaming_cache_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "config" / "renaming.json"


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.strip())
    cleaned = cleaned.strip("._") or "city"
    return cleaned


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


def fetch_json(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Any:
    """GET JSON from ``url`` with retries on transient failures."""
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    last_error: Exception | None = None
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_STATUS or attempt == attempts:
                raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == attempts:
                raise RuntimeError(f"Failed to fetch JSON from {url}") from exc
        sleep_s = 2 ** (attempt - 1)
        logger.warning(
            "Retry %d/%d for %s after %.0fs (%s)",
            attempt,
            attempts,
            url,
            sleep_s,
            last_error,
        )
        time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch JSON from {url}") from last_error


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)
    return path


def _load_cached_json(cache: Path) -> Any | None:
    if not cache.is_file():
        return None
    with cache.open(encoding="utf-8") as handle:
        return json.load(handle)


def fetch_cities(
    year: int,
    *,
    out_dir: str | Path | None = None,
    refresh: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> list[dict[str, Any]]:
    """Return the city catalogue, caching it under ``out_dir/cities``."""
    cache = cities_cache_path(out_dir or default_out_dir(), year)
    if not refresh:
        payload = _load_cached_json(cache)
        if payload is not None:
            if not isinstance(payload, list):
                raise ValueError(f"City catalogue cache is not a JSON array: {cache}")
            logger.info("Using cached city catalogue %s (%d cities)", cache, len(payload))
            return payload

    url = cities_url(year)
    logger.info("Fetching city catalogue %s", url)
    payload = fetch_json(url, timeout=timeout, retries=retries)
    if not isinstance(payload, list):
        raise ValueError(f"City catalogue is not a JSON array: {url}")
    _write_json(cache, payload)
    logger.info("Wrote city catalogue %s (%d cities)", cache, len(payload))
    return payload


def fetch_renaming(
    *,
    out_dir: str | Path | None = None,
    refresh: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """Return shortname labels, caching them under ``out_dir/config/renaming.json``."""
    cache = renaming_cache_path(out_dir or default_out_dir())
    if not refresh:
        payload = _load_cached_json(cache)
        if payload is not None:
            if not isinstance(payload, dict):
                raise ValueError(f"Renaming cache is not a JSON object: {cache}")
            logger.info("Using cached renaming table %s", cache)
            return payload

    logger.info("Fetching renaming table %s", RENAMING_URL)
    payload = fetch_json(RENAMING_URL, timeout=timeout, retries=retries)
    if not isinstance(payload, dict):
        raise ValueError(f"Renaming table is not a JSON object: {RENAMING_URL}")
    _write_json(cache, payload)
    logger.info("Wrote renaming table %s", cache)
    return payload


def download_city_forecast(
    report_date: date,
    city: str,
    *,
    out_dir: str | Path | None = None,
    inventory: str = DEFAULT_INVENTORY,
    redistribution: bool = False,
    skip_existing: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Path:
    out = city_json_path(
        out_dir or default_out_dir(),
        report_date,
        city,
        inventory=inventory,
    )
    if skip_existing and out.is_file():
        logger.info("Skip existing forecast: %s", out)
        return out

    url = daily_forecast_url(
        report_date,
        city,
        inventory=inventory,
        redistribution=redistribution,
    )
    logger.info("Downloading %s %s -> %s", report_date.isoformat(), city, out)
    payload = fetch_json(url, timeout=timeout, retries=retries)
    return _write_json(out, payload)


def download_country_city_forecasts(
    report_date: date,
    *,
    country: str = DEFAULT_COUNTRY,
    out_dir: str | Path | None = None,
    inventory: str = DEFAULT_INVENTORY,
    redistribution: bool = False,
    skip_existing: bool = True,
    refresh_cities: bool = False,
    refresh_renaming: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> list[Path]:
    """Download daily source-receptor forecasts for every city in ``country``."""
    dest = Path(out_dir or default_out_dir())
    fetch_renaming(
        out_dir=dest,
        refresh=refresh_renaming,
        timeout=timeout,
        retries=retries,
    )
    cities = filter_cities(
        fetch_cities(
            report_date.year,
            out_dir=dest,
            refresh=refresh_cities,
            timeout=timeout,
            retries=retries,
        ),
        country=country,
    )
    if not cities:
        raise ValueError(
            f"No cities found for country {country!r} in the {report_date.year} catalogue"
        )

    logger.info(
        "Downloading %d %s cities for %s (%s)",
        len(cities),
        country,
        report_date.isoformat(),
        inventory,
    )
    paths: list[Path] = []
    failures: list[str] = []
    for city in cities:
        name = str(city.get("name") or "").strip()
        if not name:
            failures.append(f"city record missing name: {city!r}")
            continue
        try:
            paths.append(
                download_city_forecast(
                    report_date,
                    name,
                    out_dir=dest,
                    inventory=inventory,
                    redistribution=redistribution,
                    skip_existing=skip_existing,
                    timeout=timeout,
                    retries=retries,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Failed to download %s: %s", name, exc)
            failures.append(f"{name}: {exc}")

    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(
            f"Downloaded {len(paths)}/{len(cities)} cities; failures: {detail}"
        )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download CAMS Policy daily source-receptor forecasts for all cities "
            "in a country (default: Germany)."
        ),
    )
    parser.add_argument(
        "--date",
        "-d",
        required=True,
        type=parse_date,
        help="Forecast date as YYYYMMDD (YYYY-MM-DD also accepted).",
    )
    parser.add_argument(
        "--country",
        default=DEFAULT_COUNTRY,
        help="Country name as used by the Policy API, or a 2-letter code (default: Germany).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: paths.cams_policy_forecast).",
    )
    parser.add_argument(
        "--inventory",
        default=DEFAULT_INVENTORY,
        help=f"Emission inventory in the Policy API path (default: {DEFAULT_INVENTORY}).",
    )
    parser.add_argument(
        "--redistribution",
        action="store_true",
        help="Request redistributed source-receptor results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing city JSON files.",
    )
    parser.add_argument(
        "--refresh-cities",
        action="store_true",
        help="Re-download the city catalogue even if a cache exists.",
    )
    parser.add_argument(
        "--refresh-renaming",
        action="store_true",
        help="Re-download config/renaming.json even if a cache exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT:.0f}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    paths = download_country_city_forecasts(
        args.date,
        country=args.country,
        out_dir=args.out_dir,
        inventory=args.inventory,
        redistribution=args.redistribution,
        skip_existing=not args.overwrite,
        refresh_cities=args.refresh_cities,
        refresh_renaming=args.refresh_renaming,
        timeout=args.timeout,
    )
    print(f"Downloaded {len(paths)} city forecasts under {paths[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
