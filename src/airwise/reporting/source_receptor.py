"""Country/source contributions to city PM10 and PM2.5 concentrations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
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
FALLBACK_LEAD_START = 24
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


class PolicyForecastUnavailable(FileNotFoundError):
    """Neither the report-date run nor the previous day's run can be summarised."""


@dataclass(frozen=True)
class PolicyForecastWindow:
    """Hourly slice of one Policy run used for a report date."""

    run_date: date
    lead_start: int
    hours: int

    @property
    def lead_end(self) -> int:
        return self.lead_start + self.hours - 1

    @property
    def uses_previous_run(self) -> bool:
        return self.lead_start != 0


def mean_lead_hours(
    values: list[float],
    hours: int = LEAD_HOURS,
    *,
    start: int = 0,
) -> float:
    if hours <= 0:
        raise ValueError("hours must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    end = start + hours
    if len(values) < end:
        raise ValueError(f"Expected at least {end} hourly values, got {len(values)}")
    window = [float(item) for item in values[start:end]]
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
    start: int = 0,
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
        totals[code] = totals.get(code, 0.0) + mean_lead_hours(hout, hours=hours, start=start)

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
    start: int = 0,
    top_n: int = TOP_N,
) -> dict[str, dict[str, Any]]:
    records = _records_for_date(payload, report_date)
    return {
        pollutant_id: summarise_pollutant(
            records,
            poll,
            renaming=renaming,
            hours=hours,
            start=start,
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


def _city_names(cities: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for city in cities:
        name = str(city.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _cities_for_year(dest: Path, year: int, country: str) -> list[dict[str, Any]]:
    try:
        catalogue = load_cached_cities(year, out_dir=dest)
    except FileNotFoundError:
        return []
    return filter_cities(catalogue, country=country)


def _all_city_files_present(
    dest: Path,
    run_date: date,
    cities: list[dict[str, Any]],
    inventory: str,
) -> bool:
    names = _city_names(cities)
    if not names:
        return False
    return all(
        city_json_path(dest, run_date, name, inventory=inventory).is_file() for name in names
    )


def _covers_lead_window(payload: dict[str, Any], run_date: date, start: int, hours: int) -> bool:
    records = payload.get(f"{run_date:%Y%m%d}")
    if not isinstance(records, list) or not records:
        return False
    end = start + hours
    for record in records:
        if not isinstance(record, dict):
            return False
        hout = record.get("hOut")
        if not isinstance(hout, list) or len(hout) < end:
            return False
    return True


def resolve_policy_window(
    report_date: date,
    *,
    policy_dir: str | Path | None = None,
    country: str = DEFAULT_COUNTRY,
    inventory: str = DEFAULT_INVENTORY,
    hours: int = LEAD_HOURS,
) -> PolicyForecastWindow | None:
    """Choose the report-date run, or the previous run's lead hours 24 onward.

    The report-date directory is used only when every city file is present.
    Otherwise the previous day's complete set is used when each series covers
    lead hours ``FALLBACK_LEAD_START`` through ``FALLBACK_LEAD_START + hours``.
    """
    dest = Path(policy_dir or default_out_dir())
    same_day_cities = _cities_for_year(dest, report_date.year, country)
    if _all_city_files_present(dest, report_date, same_day_cities, inventory):
        return PolicyForecastWindow(run_date=report_date, lead_start=0, hours=hours)

    previous = report_date - timedelta(days=1)
    if previous.year == report_date.year:
        previous_cities = same_day_cities
    else:
        previous_cities = _cities_for_year(dest, previous.year, country) or same_day_cities
    if not _all_city_files_present(dest, previous, previous_cities, inventory):
        return None
    for name in _city_names(previous_cities):
        path = city_json_path(dest, previous, name, inventory=inventory)
        if not _covers_lead_window(load_city_payload(path), previous, FALLBACK_LEAD_START, hours):
            return None
    return PolicyForecastWindow(
        run_date=previous,
        lead_start=FALLBACK_LEAD_START,
        hours=hours,
    )


def _pdf_interpretation(
    *,
    report_date: date,
    run_date: date,
    lead_start: int,
    lead_end: int,
    uses_previous_run: bool,
) -> str:
    source = f"Source: CAMS Policy Tools country contribution ({COUNTRY_CONTRIBUTION_URL})."
    if uses_previous_run:
        return (
            f"The CAMS Policy product for {report_date.isoformat()} was not yet available. "
            f"This table uses the forecast issued on {run_date.isoformat()}. "
            f"Each source value is the mean contribution over lead hours {lead_start}-{lead_end} "
            f"of that run, which are the hours valid on {report_date.isoformat()}. "
            "These shares can differ from the report-date run. "
            f"{source}"
        )
    return (
        "Main contributors are the three largest sources of the city-mean "
        f"concentration over lead hours {lead_start}-{lead_end}. "
        f"{source}"
    )


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
    window = resolve_policy_window(
        report_date,
        policy_dir=dest,
        country=country,
        inventory=inventory,
        hours=hours,
    )
    if window is None:
        previous = report_date - timedelta(days=1)
        raise PolicyForecastUnavailable(
            "No complete CAMS Policy city forecasts for "
            f"{report_date.isoformat()} or {previous.isoformat()} under {dest}"
        )
    renaming = load_renaming_map(renaming_cache_path(dest), inventory=inventory)
    cities = _cities_for_year(dest, window.run_date.year, country)
    if not cities:
        raise ValueError(f"No cities found for country {country!r}")

    city_rows: list[dict[str, Any]] = []
    for city in cities:
        name = str(city.get("name") or "").strip()
        if not name:
            continue
        path = city_json_path(dest, window.run_date, name, inventory=inventory)
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
                    window.run_date,
                    renaming=renaming,
                    hours=window.hours,
                    start=window.lead_start,
                    top_n=top_n,
                ),
            }
        )

    if not city_rows:
        raise ValueError(f"No city forecasts found under {dest}")

    country_label = country.upper() if country.lower() == "germany" else country
    if window.uses_previous_run:
        interpretation = _pdf_interpretation(
            report_date=report_date,
            run_date=window.run_date,
            lead_start=window.lead_start,
            lead_end=window.lead_end,
            uses_previous_run=True,
        )
    else:
        interpretation = (
            f"Each source value is the mean contribution over lead hours "
            f"{window.lead_start}-{window.lead_end}. "
            "Share is that mean divided by the sum of all source means. "
            "top_share is the combined share of the three largest sources. "
            f"Source: CAMS Policy Tools country contribution ({COUNTRY_CONTRIBUTION_URL})."
        )
    return {
        "schema_version": "1.0",
        "report_type": "transboundary_contribution",
        "metadata": {
            "title": f"Country contributions to PM10/PM2.5 for {country_label}",
            "country": country_label,
            "report_date": report_date.isoformat(),
            "forecast_run_date": window.run_date.isoformat(),
            "uses_previous_run": window.uses_previous_run,
            "inventory": inventory,
            "redistribution": redistribution,
            "lead_hours": [window.lead_start, window.lead_end],
            "aggregation": "mean",
            "top_n": top_n,
            "source_url": COUNTRY_CONTRIBUTION_URL,
            "project_ref": "AIr-wise / Code for Earth 2026",
            "project_title": "AI-Based Uncertainty-Aware Air Quality Assessment",
        },
        "section_title": "What are the contributions from countries to PM10/2.5 concentrations?",
        "interpretation": interpretation,
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
    report_day = date.fromisoformat(str(metadata.get("report_date")))
    run_day_text = metadata.get("forecast_run_date") or metadata.get("report_date")
    run_day = date.fromisoformat(str(run_day_text))
    return {
        "section_title": section_title,
        "interpretation": _pdf_interpretation(
            report_date=report_day,
            run_date=run_day,
            lead_start=start_hour,
            lead_end=end_hour,
            uses_previous_run=bool(metadata.get("uses_previous_run")),
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
    """Embed a slim transboundary table section into bulletin JSON.

    When neither the report-date run nor the previous day's run is complete,
    the section is left out and later bulletin sections keep their numbers.
    """
    try:
        report_data["transboundary_pollution"] = build_bulletin_transboundary_section(
            report_date,
            policy_dir=policy_dir,
            hours=hours,
        )
    except PolicyForecastUnavailable as exc:
        logger.warning("Omitting transboundary pollution table: %s", exc)
        return report_data
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
