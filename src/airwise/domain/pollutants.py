"""Stable pollutant identifiers and metadata shared across AIr-wise."""

from __future__ import annotations

from dataclasses import dataclass


POLLUTANT_IDS = ("no2", "o3", "pm25", "pm10")


@dataclass(frozen=True)
class Pollutant:
    id: str
    label: str
    unit: str
    variable: str


DEFAULT_POLLUTANTS: dict[str, Pollutant] = {
    "no2": Pollutant("no2", "NO2", "ug/m3", "no2_conc"),
    "o3": Pollutant("o3", "O3", "ug/m3", "o3_conc"),
    "pm25": Pollutant("pm25", "PM2.5", "ug/m3", "pm2p5_conc"),
    "pm10": Pollutant("pm10", "PM10", "ug/m3", "pm10_conc"),
}
