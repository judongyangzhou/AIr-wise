"""Country/source contributions to city PM10 and PM2.5 concentrations."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from airwise.data.io.policy import (
    DEFAULT_COUNTRY,
    DEFAULT_INVENTORY,
    city_json_path,
    default_out_dir,
    filter_cities,
    load_cached_cities,
    renaming_cache_path,
)

logger = logging.getLogger(__name__)

LEAD_HOURS = 24
TOP_N = 3
COUNTRY_CONTRIBUTION_URL = (
    "https://policy.atmosphere.copernicus.eu/daily/country-contribution/"
)
POLLUTANTS: tuple[tuple[str, str, str], ...] = (
    ("pm10", "PM10", "PM10"),
    ("pm25", "PM25", "PM2.5"),
)
ISO3_DISPLAY_NAMES = {
    "ALB": "Albania",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BGR": "Bulgaria",
    "CHE": "Switzerland",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "GRC": "Greece",
    "HRV": "Croatia",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ITA": "Italy",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "LVA": "Latvia",
    "MKD": "North Macedonia",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "SRB": "Serbia",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SWE": "Sweden",
    "TUR": "Türkiye",
}
SOURCE_SHORTNAMES = {
    "BIC": "Hemispheric",
    "SHP": "Shipping",
    "RESTPPM": "Rest Primary",
    "RESTPPM_f": "Rest Primary",
    "DUST": "Dust",
    "DUST_f": "Dust",
    "SEASALT": "Sea Salt",
    "SEASALT_f": "Sea Salt",
    "NO3_f": "NO3",
    "POM_f": "POM",
    "EC_f": "EC",
    "FFIRE": "Forest Fire",
    "FFIRE_f": "Forest Fire",
}


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _city_id(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip()).strip("_")


def load_renaming_map(
    path: str | Path | None,
    *,
    inventory: str = DEFAULT_INVENTORY,
    timescale: str = "daily",
) -> dict[str, str]:
    if path is None or not Path(path).is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Renaming table is not a JSON object: {path}")
    mapping = payload.get(timescale, {}).get(inventory, {})
    if not isinstance(mapping, dict):
        return {}
    return {str(key): str(value) for key, value in mapping.items()}


def source_display_name(code: str, renaming: dict[str, str] | None = None) -> str:
    if code in ISO3_DISPLAY_NAMES:
        return ISO3_DISPLAY_NAMES[code]
    renaming = renaming or {}
    if code in renaming:
        return renaming[code]
    return SOURCE_SHORTNAMES.get(code, code)


def sector_label(code: str, renaming: dict[str, str]) -> str:
    return source_display_name(code, renaming)


def mean_lead_hours(values: list[float], hours: int = LEAD_HOURS) -> float:
    if hours <= 0:
        raise ValueError("hours must be positive")
    if len(values) < hours:
        raise ValueError(f"Expected at least {hours} hourly values, got {len(values)}")
    window = [float(item) for item in values[:hours]]
    return sum(window) / hours


def _records_for_date(payload: dict[str, Any], report_date: date) -> list[dict[str, Any]]:
    key = f"{report_date:%Y%m%d}"
    records = payload.get(key)
    if not isinstance(records, list):
        raise ValueError(f"Missing source-receptor records for {key}")
    return records


def summarise_pollutant(
    records: list[dict[str, Any]],
    poll: str,
    *,
    renaming: dict[str, str] | None = None,
    hours: int = LEAD_HOURS,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    renaming = renaming or {}
    totals: dict[str, float] = {}
    for record in records:
        if str(record.get("poll") or "") != poll:
            continue
        code = str(record.get("code") or record.get("name") or "").strip()
        if not code:
            raise ValueError(f"{poll} record is missing a source code")
        hout = record.get("hOut")
        if not isinstance(hout, list):
            raise ValueError(f"{poll}/{code} is missing hourly values")
        totals[code] = totals.get(code, 0.0) + mean_lead_hours(hout, hours=hours)

    if not totals:
        raise ValueError(f"No {poll} source records found")

    total = float(sum(totals.values()))
    sectors = []
    for code, mean in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
        share = 0.0 if total <= 0 else mean / total
        sectors.append(
            {
                "code": code,
                "label": source_display_name(code, renaming),
                "mean": _round(mean),
                "share": _round(share),
            }
        )
    top_sectors = sectors[:top_n]
    top_mean = sum(totals[item["code"]] for item in top_sectors)
    top_share = 0.0 if total <= 0 else top_mean / total
    return {
        "label": "PM2.5" if poll == "PM25" else poll,
        "unit": "ug/m3",
        "total": _round(total),
        "sectors": sectors,
        "top_sectors": top_sectors,
        "top_share": _round(top_share),
    }


def summarise_city(
    payload: dict[str, Any],
    report_date: date,
    *,
    renaming: dict[str, str] | None = None,
    hours: int = LEAD_HOURS,
    top_n: int = TOP_N,
) -> dict[str, dict[str, Any]]:
    records = _records_for_date(payload, report_date)
    return {
        pollutant_id: summarise_pollutant(
            records,
            poll,
            renaming=renaming,
            hours=hours,
            top_n=top_n,
        )
        for pollutant_id, poll, _label in POLLUTANTS
    }


def load_city_payload(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"City forecast is not a JSON object: {path}")
    return payload


def build_tbi_report(
    report_date: date,
    *,
    policy_dir: str | Path | None = None,
    country: str = DEFAULT_COUNTRY,
    inventory: str = DEFAULT_INVENTORY,
    hours: int = LEAD_HOURS,
    top_n: int = TOP_N,
    redistribution: bool = False,
) -> dict[str, Any]:
    dest = Path(policy_dir or default_out_dir())
    renaming = load_renaming_map(renaming_cache_path(dest), inventory=inventory)
    cities = filter_cities(
        load_cached_cities(report_date.year, out_dir=dest),
        country=country,
    )
    if not cities:
        raise ValueError(f"No cities found for country {country!r}")

    city_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for city in cities:
        name = str(city.get("name") or "").strip()
        path = city_json_path(dest, report_date, name, inventory=inventory)
        if not path.is_file():
            missing.append(name)
            continue
        logger.info("Summarising %s from %s", name, path)
        city_rows.append(
            {
                "id": _city_id(name),
                "name": name,
                "code": city.get("code"),
                "lat": city.get("lat"),
                "lon": city.get("lon"),
                "pollutants": summarise_city(
                    load_city_payload(path),
                    report_date,
                    renaming=renaming,
                    hours=hours,
                    top_n=top_n,
                ),
            }
        )

    if missing:
        raise FileNotFoundError(
            "Missing source-receptor JSON for "
            + ", ".join(missing)
            + f" under {dest / f'{report_date:%Y%m%d}' / inventory}"
        )
    if not city_rows:
        raise ValueError(f"No city forecasts found under {dest}")

    country_label = country.upper() if country.lower() == "germany" else country
    return {
        "schema_version": "1.0",
        "report_type": "transboundary_contribution",
        "metadata": {
            "title": f"Country contributions to PM10/PM2.5 for {country_label}",
            "country": country_label,
            "report_date": report_date.isoformat(),
            "inventory": inventory,
            "redistribution": redistribution,
            "lead_hours": [0, hours - 1],
            "aggregation": "mean",
            "top_n": top_n,
            "source_url": COUNTRY_CONTRIBUTION_URL,
            "project_ref": "AIr-wise / Code for Earth 2026",
            "project_title": "AI-Based Uncertainty-Aware Air Quality Assessment",
        },
        "section_title": "What are the contributions from countries to PM10/2.5 concentrations?",
        "interpretation": (
            f"Each source value is the mean contribution over lead hours 0-{hours - 1}. "
            "Share is that mean divided by the sum of all source means. "
            "top_share is the combined share of the three largest sources. "
            f"Source: CAMS Policy Tools country contribution ({COUNTRY_CONTRIBUTION_URL})."
        ),
        "pollutant_order": [pollutant_id for pollutant_id, _poll, _label in POLLUTANTS],
        "cities": city_rows,
    }


def slim_transboundary_section(
    tbi_report: dict[str, Any],
    *,
    section_title: str = "4 Transboundary pollution at city scale",
) -> dict[str, Any]:
    """Keep only the fields needed by the bulletin PDF table."""
    metadata = tbi_report.get("metadata") or {}
    hours = metadata.get("lead_hours") or [0, LEAD_HOURS - 1]
    cities = []
    for city in tbi_report.get("cities") or []:
        pollutants = {}
        for pollutant_id, _poll, _label in POLLUTANTS:
            block = (city.get("pollutants") or {}).get(pollutant_id) or {}
            pollutants[pollutant_id] = {
                "label": block.get("label") or ("PM2.5" if pollutant_id == "pm25" else "PM10"),
                "top_contributors": [
                    {
                        "code": item["code"],
                        "label": item["label"],
                        "share": item["share"],
                    }
                    for item in block.get("top_sectors") or []
                ],
            }
        cities.append(
            {
                "id": city["id"],
                "name": city["name"],
                "pollutants": pollutants,
            }
        )
    start_hour, end_hour = int(hours[0]), int(hours[-1])
    return {
        "section_title": section_title,
        "interpretation": (
            "Main contributors are the three largest sources of the city-mean "
            f"concentration over lead hours {start_hour}-{end_hour}. "
            f"Source: CAMS Policy Tools country contribution ({COUNTRY_CONTRIBUTION_URL})."
        ),
        "lead_hours": [start_hour, end_hour],
        "top_n": int(metadata.get("top_n") or TOP_N),
        "pollutant_order": [pollutant_id for pollutant_id, _poll, _label in POLLUTANTS],
        "cities": cities,
    }


def build_bulletin_transboundary_section(
    report_date: date,
    *,
    policy_dir: str | Path | None = None,
    country: str = DEFAULT_COUNTRY,
    inventory: str = DEFAULT_INVENTORY,
    hours: int = LEAD_HOURS,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    tbi_report = build_tbi_report(
        report_date,
        policy_dir=policy_dir,
        country=country,
        inventory=inventory,
        hours=hours,
        top_n=top_n,
    )
    write_tbi_report(
        default_output_path(
            report_date,
            country,
            policy_dir=policy_dir,
            inventory=inventory,
        ),
        tbi_report,
    )
    return slim_transboundary_section(tbi_report)


def attach_transboundary_pollution(
    report_data: dict[str, Any],
    report_date: date,
    *,
    policy_dir: str | Path | None = None,
    hours: int = LEAD_HOURS,
) -> dict[str, Any]:
    """Embed a slim transboundary table section into bulletin JSON."""
    report_data["transboundary_pollution"] = build_bulletin_transboundary_section(
        report_date,
        policy_dir=policy_dir,
        hours=hours,
    )
    time_series = report_data.get("forecast_time_series")
    if isinstance(time_series, dict):
        title = str(time_series.get("section_title") or "")
        if title.startswith("4 "):
            time_series["section_title"] = "5 " + title[2:]
    return report_data


def write_tbi_report(output_path: str | Path, report_data: dict[str, Any]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Wrote transboundary-contribution JSON to %s", output)
    return output


def default_output_path(
    report_date: date,
    country: str = DEFAULT_COUNTRY,
    *,
    policy_dir: str | Path | None = None,
    inventory: str = DEFAULT_INVENTORY,
) -> Path:
    slug = "germany" if country.strip().lower() in {"germany", "de"} else country.strip().lower()
    dest = Path(policy_dir or default_out_dir())
    return dest / f"{report_date:%Y%m%d}" / inventory / f"{slug}_TBC_{report_date.isoformat()}.json"


def generate_tbi_report(
    report_date: date,
    *,
    output_path: str | Path | None = None,
    policy_dir: str | Path | None = None,
    country: str = DEFAULT_COUNTRY,
    inventory: str = DEFAULT_INVENTORY,
    hours: int = LEAD_HOURS,
    top_n: int = TOP_N,
) -> Path:
    report = build_tbi_report(
        report_date,
        policy_dir=policy_dir,
        country=country,
        inventory=inventory,
        hours=hours,
        top_n=top_n,
    )
    return write_tbi_report(
        output_path
        or default_output_path(
            report_date,
            country,
            policy_dir=policy_dir,
            inventory=inventory,
        ),
        report,
    )
