"""Download one 00Z IFS control forecast and write the project NetCDF layout.

The product is the single deterministic forecast published as ``oper/fc``
(not the 51-member ``enfo`` ensemble). Three channels fetch the same files:

- ``client``: the ``ecmwf-opendata`` Python client
- ``https``: ``curl`` or ``wget`` from ``https://data.ecmwf.int/forecasts``
- ``gsutil``: the public ``ecmwf-open-data`` bucket

Each lead is cropped to Europe and reduced to ``u10``, ``v10``, ``t2m``,
``d2m``, ``sp``, ``ssrd`` and ``tp``, then stored as ``number=0``.

Example::

    airwise-download-open-ifs --date 2026-08-01 --channel client
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import time
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import xarray as xr

from airwise.config import europe_bounds, load_config, require_config, resolve_repo_path
from airwise.data.io.openifs import OPEN_IFS_LEAD_HOURS, open_ifs_control_path

# ECMWF oper/fc product identifiers and the three ways to fetch the same files.
# Paths, the Europe crop, and CAMS variable names live in configs/default.yaml.
DEFAULT_BUCKET = "ecmwf-open-data"
DEFAULT_PORTAL = "https://data.ecmwf.int/forecasts"
DEFAULT_CHANNEL = "client"
DEFAULT_CLIENT_SOURCE = "google"
CONTROL_PARAMS = ("10u", "10v", "2t", "2d", "sp", "ssrd", "tp")
DOWNLOAD_CHANNELS = ("client", "https", "gsutil")

FILTER_GROUPS: list[dict] = [
    {"shortName": ["10u", "10v"], "typeOfLevel": "heightAboveGround", "level": 10},
    {"shortName": ["2t", "2d"], "typeOfLevel": "heightAboveGround", "level": 2},
    {"shortName": ["sp"], "typeOfLevel": "surface", "stepType": "instant"},
    {"shortName": ["ssrd", "tp"], "typeOfLevel": "surface", "stepType": "accum"},
]

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]
Sleeper = Callable[[float], None]


class OpenIFSDownloadError(RuntimeError):
    """One or more lead-time files could not be downloaded or extracted."""


def parse_open_ifs_date(value: date | str) -> date:
    """Parse ``YYYYMMDD`` or ``YYYY-MM-DD`` into a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {text}. Use YYYYMMDD or YYYY-MM-DD.")


def grib_object_name(init_date: date | str, lead_hour: int) -> str:
    """Return the oper/fc GRIB2 object name for one control-forecast lead."""
    day = parse_open_ifs_date(init_date)
    return f"{day:%Y%m%d}000000-{int(lead_hour)}h-oper-fc.grib2"


def portal_https_url(
    init_date: date | str,
    lead_hour: int,
    *,
    base_url: str = DEFAULT_PORTAL,
) -> str:
    """Return the ECMWF data-portal URL for one control-forecast lead."""
    day = parse_open_ifs_date(init_date)
    return (
        f"{base_url.rstrip('/')}/{day:%Y%m%d}/00z/ifs/0p25/oper/"
        f"{grib_object_name(day, lead_hour)}"
    )


def grib_uri(
    init_date: date | str,
    lead_hour: int,
    *,
    bucket: str = DEFAULT_BUCKET,
) -> str:
    """Return the GCS URI for one 00Z IFS control-forecast lead."""
    day = parse_open_ifs_date(init_date)
    return (
        f"gs://{bucket}/{day:%Y%m%d}/00z/ifs/0p25/oper/"
        f"{grib_object_name(day, lead_hour)}"
    )


def default_open_ifs_dir() -> Path:
    """Return the configured Open IFS NetCDF root."""
    return resolve_repo_path(require_config(load_config(), "paths", "open_ifs_data"))


def _crop_bounds(
    south: float | None,
    north: float | None,
    west: float | None,
    east: float | None,
) -> tuple[float, float, float, float]:
    """Fill omitted crop edges from ``era5_euro.area``."""
    if None not in (south, north, west, east):
        return float(south), float(north), float(west), float(east)
    configured = europe_bounds()
    return (
        configured["south"] if south is None else float(south),
        configured["north"] if north is None else float(north),
        configured["west"] if west is None else float(west),
        configured["east"] if east is None else float(east),
    )


def _download_grib(
    source: str,
    destination_dir: Path,
    *,
    retries: int,
    retry_delay: float,
    dry_run: bool,
    runner: Runner,
    sleeper: Sleeper,
) -> Path:
    """Download one GCS object. Existing files are left in place."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = source.rsplit("/", 1)[-1]
    local_path = destination_dir / filename
    if local_path.is_file():
        print(f"  [skip] {filename} already exists")
        return local_path

    command = ["gsutil", "cp", source, str(destination_dir)]
    for attempt in range(1, retries + 1):
        if dry_run:
            print(f"  [dry-run] {' '.join(command)}")
            return local_path

        print(f"  [{attempt}/{retries}] {filename}")
        result = runner(command)
        if result.returncode == 0 and local_path.is_file():
            return local_path
        if attempt < retries:
            print(
                f"    Transfer failed (rc={result.returncode}). "
                f"Retrying in {retry_delay}s..."
            )
            sleeper(retry_delay)

    raise OpenIFSDownloadError(
        f"{filename} could not be transferred after {retries} attempt(s): {source}"
    )


def _filter_signature(filter_keys: dict) -> str:
    pieces: list[str] = []
    for key in sorted(filter_keys):
        value = filter_keys[key]
        if isinstance(value, list):
            value_text = ",".join(str(item) for item in value)
        else:
            value_text = str(value)
        pieces.append(f"{key}={value_text}")
    digest = hashlib.md5("|".join(pieces).encode("utf-8")).hexdigest()
    return digest[:10]


def _crop(ds: xr.Dataset, south: float, north: float, west: float, east: float) -> xr.Dataset:
    lat_min, lat_max = min(south, north), max(south, north)
    lon_min, lon_max = min(west, east), max(west, east)
    return ds.where(
        (ds.latitude >= lat_min)
        & (ds.latitude <= lat_max)
        & (ds.longitude >= lon_min)
        & (ds.longitude <= lon_max),
        drop=True,
    )


def _open_one_group_once(
    path: Path,
    filter_keys: dict,
    index_dir: Path | None,
) -> xr.Dataset | None:
    try:
        if index_dir is None:
            indexpath = ""
        else:
            index_dir.mkdir(parents=True, exist_ok=True)
            signature = _filter_signature(filter_keys)
            indexpath = str(index_dir / f"{path.name}.{signature}.cfgrib.idx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filter_keys},
                indexpath=indexpath,
            )
    except Exception:
        return None


def _as_control_member(dataset: xr.Dataset) -> xr.Dataset:
    """Keep the deterministic forecast and label it ``number=0``."""
    if "number" in dataset.dims:
        numbers = [int(value) for value in np.atleast_1d(dataset["number"].values)]
        dataset = dataset.sel(number=0) if 0 in numbers else dataset.isel(number=0)
    if "number" in dataset.coords and "number" not in dataset.dims:
        dataset = dataset.drop_vars("number")
    if "number" not in dataset.dims:
        dataset = dataset.expand_dims(number=[0])
    return dataset


def _open_one_group(
    path: Path,
    filter_keys: dict,
    index_dir: Path | None,
) -> xr.Dataset | None:
    """Open one variable group from the oper/fc control forecast."""
    keyed = {**filter_keys, "dataType": "fc"}
    dataset = _open_one_group_once(path, keyed, index_dir)
    if dataset is None:
        dataset = _open_one_group_once(path, filter_keys, index_dir)
    if dataset is None:
        print(f"    Warning: could not open group {filter_keys.get('shortName')} from {path.name}")
        return None
    return _as_control_member(dataset)


def extract_grib_lead(
    grib_path: Path,
    output_path: Path,
    *,
    lead_hour: int,
    init_date: date,
    south: float | None = None,
    north: float | None = None,
    west: float | None = None,
    east: float | None = None,
    index_dir: Path | None = None,
) -> Path:
    """Crop one GRIB2 lead and write the NetCDF used by Open IFS inference."""
    south, north, west, east = _crop_bounds(south, north, west, east)
    parts: list[xr.Dataset] = []
    try:
        for filter_keys in FILTER_GROUPS:
            dataset = _open_one_group(grib_path, filter_keys, index_dir)
            if dataset is not None:
                parts.append(_crop(dataset, south, north, west, east))
        if not parts:
            raise OpenIFSDownloadError(f"No IFS variables extracted from {grib_path}")

        merged = xr.merge(parts, compat="override", join="inner")
        merged = merged.assign_coords(lead_hour=lead_hour)
        merged.attrs["date"] = init_date.strftime("%Y%m%d")
        merged.attrs["init_time"] = "00z"
        merged.attrs["source"] = "ECMWF IFS control forecast (oper/fc)"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_netcdf(output_path)
        return output_path
    finally:
        for part in parts:
            part.close()


def _require_cfgrib() -> None:
    try:
        import cfgrib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "cfgrib is not installed. Install it with 'pip install cfgrib' "
            "and the ECMWF ecCodes library before extracting Open IFS NetCDF."
        ) from exc


def _remove_grib_cache(grib_path: Path, index_dir: Path | None) -> None:
    grib_path.unlink(missing_ok=True)
    if index_dir is None or not index_dir.is_dir():
        return
    for cache_path in index_dir.glob(f"{grib_path.name}.*.cfgrib.idx"):
        cache_path.unlink(missing_ok=True)


def _http_tool(explicit: str | None = None) -> str:
    if explicit in {"curl", "wget"}:
        if shutil.which(explicit) is None:
            raise RuntimeError(f"{explicit} is not found in PATH.")
        return explicit
    if shutil.which("curl"):
        return "curl"
    if shutil.which("wget"):
        return "wget"
    raise RuntimeError("Neither curl nor wget is available for the https channel.")


def https_download_command(url: str, destination: Path, tool: str, *, retries: int, retry_delay: float) -> list[str]:
    if tool == "curl":
        return [
            "curl",
            "-fL",
            "--retry",
            str(retries),
            "--retry-delay",
            str(int(retry_delay)),
            "-o",
            str(destination),
            url,
        ]
    if tool == "wget":
        return ["wget", "-O", str(destination), url]
    raise ValueError(f"Unsupported HTTP tool: {tool}")


def _download_https(
    url: str,
    destination: Path,
    *,
    retries: int,
    retry_delay: float,
    dry_run: bool,
    runner: Runner,
    sleeper: Sleeper,
    http_tool: str | None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"  [skip] {destination.name} already exists")
        return destination
    tool = http_tool or "curl"
    command = https_download_command(
        url,
        destination,
        tool,
        retries=retries,
        retry_delay=retry_delay,
    )
    if dry_run:
        print(f"  [dry-run] {' '.join(command)}")
        return destination
    for attempt in range(1, retries + 1):
        print(f"  [{attempt}/{retries}] {destination.name}")
        result = runner(command)
        if result.returncode == 0 and destination.is_file():
            return destination
        destination.unlink(missing_ok=True)
        if attempt < retries:
            print(f"    Transfer failed (rc={result.returncode}). Retrying in {retry_delay}s...")
            sleeper(retry_delay)
    raise OpenIFSDownloadError(
        f"{destination.name} could not be transferred after {retries} attempt(s): {url}"
    )


def _download_with_client(
    destination: Path,
    *,
    init_date: date,
    lead_hour: int,
    client_source: str,
    retries: int,
    retry_delay: float,
    dry_run: bool,
    sleeper: Sleeper,
    retrieve: Callable[..., object] | None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"  [skip] {destination.name} already exists")
        return destination
    request = {
        "date": f"{init_date:%Y%m%d}",
        "time": 0,
        "stream": "oper",
        "type": "fc",
        "step": int(lead_hour),
        "param": list(CONTROL_PARAMS),
        "target": str(destination),
    }
    if dry_run:
        print(
            "  [dry-run] ecmwf.opendata.Client("
            f"source={client_source!r}).retrieve({request})"
        )
        return destination
    if retrieve is None:
        try:
            from ecmwf.opendata import Client
        except ImportError as exc:
            raise RuntimeError(
                "ecmwf-opendata is not installed. Install it with "
                "'pip install ecmwf-opendata', or use --channel https."
            ) from exc
        client = Client(
            source=client_source,
            model="ifs",
            resol="0p25",
            # The library otherwise waits 120s and retries up to 500 times.
            maximum_retries=2,
            retry_after=retry_delay,
            use_server_retry_after=False,
        )

        def retrieve(**kwargs: object) -> object:
            target = str(kwargs.pop("target"))
            return client.retrieve(kwargs, target)

    for attempt in range(1, retries + 1):
        print(f"  [{attempt}/{retries}] {destination.name}")
        try:
            retrieve(**request)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            if attempt >= retries:
                raise OpenIFSDownloadError(
                    f"{destination.name} could not be retrieved after {retries} attempt(s): {exc}"
                ) from exc
            print(f"    Retrieve failed ({exc}). Retrying in {retry_delay}s...")
            sleeper(retry_delay)
            continue
        if destination.is_file():
            return destination
        if attempt >= retries:
            break
        print(f"    Retrieve did not write {destination.name}. Retrying in {retry_delay}s...")
        sleeper(retry_delay)
    raise OpenIFSDownloadError(f"{destination.name} was not written by the ECMWF client.")


def download_open_ifs_day(
    init_date: date | str,
    *,
    output_dir: str | Path | None = None,
    grib_dir: str | Path | None = None,
    channel: str = DEFAULT_CHANNEL,
    bucket: str = DEFAULT_BUCKET,
    client_source: str = DEFAULT_CLIENT_SOURCE,
    portal_url: str = DEFAULT_PORTAL,
    http_tool: str | None = None,
    retries: int = 3,
    retry_delay: float = 10.0,
    dry_run: bool = False,
    grib_only: bool = False,
    keep_grib: bool = False,
    south: float | None = None,
    north: float | None = None,
    west: float | None = None,
    east: float | None = None,
    runner: Runner | None = None,
    sleeper: Sleeper | None = None,
    retrieve: Callable[..., object] | None = None,
    extract=extract_grib_lead,
) -> list[Path]:
    """Download one 00Z IFS control forecast.

    NetCDF files are written under ``output_dir/YYYYMMDD/`` with ``number=0``.
    GRIB2 files are staged under ``grib_dir`` and removed after a successful
    extract unless ``keep_grib`` or ``grib_only`` is set.
    """
    if channel not in DOWNLOAD_CHANNELS:
        raise ValueError(f"channel must be one of {DOWNLOAD_CHANNELS}, got {channel!r}")
    day = parse_open_ifs_date(init_date)
    south, north, west, east = _crop_bounds(south, north, west, east)
    output_root = Path(output_dir) if output_dir is not None else default_open_ifs_dir()
    staging_dir = (
        Path(grib_dir) if grib_dir is not None else output_root / "_grib" / f"{day:%Y%m%d}"
    )
    run = runner or subprocess.run
    pause = sleeper or time.sleep
    index_dir = staging_dir / ".cfgrib_index"
    resolved_http_tool = None
    if channel == "https":
        if dry_run or runner is not None:
            resolved_http_tool = http_tool or "curl"
        else:
            resolved_http_tool = _http_tool(http_tool)

    if not grib_only and not dry_run and extract is extract_grib_lead:
        pending = [
            lead
            for lead in OPEN_IFS_LEAD_HOURS
            if not open_ifs_control_path(output_root, day, lead).is_file()
        ]
        if pending:
            _require_cfgrib()

    if channel == "gsutil" and not dry_run and shutil.which("gsutil") is None and runner is None:
        needs_download = any(
            not open_ifs_control_path(output_root, day, lead).is_file()
            and not (staging_dir / grib_object_name(day, lead)).is_file()
            for lead in OPEN_IFS_LEAD_HOURS
        )
        if needs_download:
            raise RuntimeError("gsutil is not found in PATH. Install and configure it first.")

    written: list[Path] = []
    print(f"===== Download OpenIFS control: {day.isoformat()} (00z via {channel}) =====")
    for lead_hour in OPEN_IFS_LEAD_HOURS:
        netcdf_path = open_ifs_control_path(output_root, day, lead_hour)
        if not grib_only and netcdf_path.is_file():
            print(f"  [skip] {netcdf_path.name} already exists")
            written.append(netcdf_path)
            continue

        grib_path = staging_dir / grib_object_name(day, lead_hour)
        if channel == "gsutil":
            grib_path = _download_grib(
                grib_uri(day, lead_hour, bucket=bucket),
                staging_dir,
                retries=retries,
                retry_delay=retry_delay,
                dry_run=dry_run,
                runner=run,
                sleeper=pause,
            )
        elif channel == "https":
            grib_path = _download_https(
                portal_https_url(day, lead_hour, base_url=portal_url),
                grib_path,
                retries=retries,
                retry_delay=retry_delay,
                dry_run=dry_run,
                runner=run,
                sleeper=pause,
                http_tool=resolved_http_tool,
            )
        else:
            grib_path = _download_with_client(
                grib_path,
                init_date=day,
                lead_hour=int(lead_hour),
                client_source=client_source,
                retries=retries,
                retry_delay=retry_delay,
                dry_run=dry_run,
                sleeper=pause,
                retrieve=retrieve,
            )
        if dry_run or grib_only:
            written.append(grib_path)
            continue

        print(f"  extracting {grib_path.name} -> {netcdf_path.name}")
        extract(
            grib_path,
            netcdf_path,
            lead_hour=int(lead_hour),
            init_date=day,
            south=south,
            north=north,
            west=west,
            east=east,
            index_dir=index_dir,
        )
        written.append(netcdf_path)
        if not keep_grib:
            _remove_grib_cache(grib_path, index_dir)

    if not keep_grib and not grib_only and not dry_run:
        if index_dir.is_dir() and not any(index_dir.iterdir()):
            index_dir.rmdir()
        if staging_dir.is_dir() and not any(staging_dir.iterdir()):
            staging_dir.rmdir()
        default_parent = output_root / "_grib"
        if (
            grib_dir is None
            and default_parent.is_dir()
            and not any(default_parent.iterdir())
        ):
            default_parent.rmdir()
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download one 00Z IFS control forecast and write Open IFS NetCDF."
    )
    parser.add_argument("--date", required=True, help="Forecast date, YYYYMMDD or YYYY-MM-DD.")
    parser.add_argument(
        "--channel",
        choices=DOWNLOAD_CHANNELS,
        default=DEFAULT_CHANNEL,
        help="Download channel: ecmwf-opendata client, HTTPS curl/wget, or gsutil.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="NetCDF root. Defaults to paths.open_ifs_data in the project config.",
    )
    parser.add_argument(
        "--grib-dir",
        default=None,
        help="Temporary GRIB directory. Defaults to <output-dir>/_grib/YYYYMMDD.",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket name for --channel gsutil.")
    parser.add_argument(
        "--client-source",
        default=DEFAULT_CLIENT_SOURCE,
        help="ecmwf-opendata source. google avoids the data.ecmwf.int 429 limit; also aws, azure, or ecmwf.",
    )
    parser.add_argument(
        "--http-tool",
        choices=("curl", "wget"),
        default=None,
        help="HTTPS downloader. Defaults to curl, then wget.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Attempts per GRIB file.")
    parser.add_argument("--retry-delay", type=float, default=10.0, help="Seconds between retries.")
    parser.add_argument("--dry-run", action="store_true", help="Print the download request only.")
    parser.add_argument(
        "--grib-only",
        action="store_true",
        help="Download GRIB2 files and skip NetCDF extraction.",
    )
    parser.add_argument(
        "--keep-grib",
        action="store_true",
        help="Keep the staged GRIB2 files after NetCDF extraction.",
    )
    parser.add_argument("--south", type=float, default=None, help="Crop edge. Defaults to era5_euro.area.south.")
    parser.add_argument("--north", type=float, default=None, help="Crop edge. Defaults to era5_euro.area.north.")
    parser.add_argument("--west", type=float, default=None, help="Crop edge. Defaults to era5_euro.area.west.")
    parser.add_argument("--east", type=float, default=None, help="Crop edge. Defaults to era5_euro.area.east.")
    args = parser.parse_args(argv)

    try:
        download_open_ifs_day(
            args.date,
            output_dir=args.output_dir,
            grib_dir=args.grib_dir,
            channel=args.channel,
            bucket=args.bucket,
            client_source=args.client_source,
            http_tool=args.http_tool,
            retries=args.retries,
            retry_delay=args.retry_delay,
            dry_run=args.dry_run,
            grib_only=args.grib_only,
            keep_grib=args.keep_grib,
            south=args.south,
            north=args.north,
            west=args.west,
            east=args.east,
        )
    except (OpenIFSDownloadError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
