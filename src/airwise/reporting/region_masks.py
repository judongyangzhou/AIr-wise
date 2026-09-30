"""Rasterise NUTS polygons onto the CAMS lat/lon grid (land points only)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import contains_xy, wkb
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from airwise.config import load_config, require_config, resolve_repo_path

_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def default_nuts_shapefile() -> Path:
    return resolve_repo_path(require_config(load_config(), "paths", "nuts_shapefile"))


def gpkg_geometry_to_wkb(blob: bytes) -> bytes:
    """Strip the GeoPackageBinary header (envelope occupies flags bits 1-3)."""
    if len(blob) < 8 or blob[:2] != b"GP":
        raise ValueError("Not a GeoPackage geometry blob")
    envelope_indicator = (blob[3] >> 1) & 0x07
    try:
        envelope_bytes = _ENVELOPE_BYTES[envelope_indicator]
    except KeyError as exc:
        raise ValueError(f"Unsupported GeoPackage envelope type {envelope_indicator}") from exc
    return blob[8 + envelope_bytes :]


def _gpkg_layer(connection: sqlite3.Connection) -> tuple[str, str]:
    row = connection.execute(
        "SELECT table_name, column_name FROM gpkg_geometry_columns LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("GeoPackage has no vector layer in gpkg_geometry_columns")
    return str(row[0]), str(row[1])


def load_nuts_geometries(
    shapefile: str | Path,
    nuts_ids: list[str],
    *,
    target_crs: str = "EPSG:4326",
) -> dict[str, BaseGeometry]:
    """Load NUTS polygons and reproject them to WGS84 lon/lat."""
    path = Path(shapefile)
    if not path.is_file():
        raise FileNotFoundError(f"NUTS shapefile not found: {path}")
    wanted = [str(nuts_id) for nuts_id in nuts_ids]
    if not wanted:
        return {}

    placeholders = ",".join("?" for _ in wanted)
    connection = sqlite3.connect(path)
    try:
        table, geom_column = _gpkg_layer(connection)
        srs = connection.execute(
            "SELECT organization, organization_coordsys_id FROM gpkg_spatial_ref_sys "
            "WHERE srs_id = (SELECT srs_id FROM gpkg_contents WHERE table_name = ?)",
            (table,),
        ).fetchone()
        source_crs = "EPSG:3035"
        if srs is not None and srs[0] and srs[1]:
            source_crs = f"{srs[0].upper()}:{int(srs[1])}"
        quoted_geom = f'"{geom_column}"' if geom_column.lower() in {"shape", "geom", "geometry"} else geom_column
        rows = connection.execute(
            f'SELECT NUTS_ID, {quoted_geom} FROM "{table}" WHERE NUTS_ID IN ({placeholders})',
            wanted,
        ).fetchall()
    finally:
        connection.close()

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    geometries: dict[str, BaseGeometry] = {}
    for nuts_id, blob in rows:
        geom = wkb.loads(gpkg_geometry_to_wkb(blob))
        if source_crs.upper() != target_crs.upper():
            geom = shapely_transform(transformer.transform, geom)
        geometries[str(nuts_id)] = geom

    missing = [nuts_id for nuts_id in wanted if nuts_id not in geometries]
    if missing:
        raise KeyError(f"NUTS ids not found in {path}: {missing}")
    return geometries


def rasterize_polygon(
    geometry: BaseGeometry,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Return a [lat, lon] bool mask for CAMS cell centres inside ``geometry``."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon, lat)
    minx, miny, maxx, maxy = geometry.bounds
    bbox = (lon2d >= minx) & (lon2d <= maxx) & (lat2d >= miny) & (lat2d <= maxy)
    mask = np.zeros(lat2d.shape, dtype=bool)
    if not np.any(bbox):
        return mask
    rows, cols = np.where(bbox)
    mask[rows, cols] = contains_xy(geometry, lon2d[rows, cols], lat2d[rows, cols])
    return mask


def build_region_land_masks(
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    nuts_id_by_region: dict[str, str],
    shapefile: str | Path,
    land_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build land-only NUTS masks aligned to a CAMS grid.

    Each mask is True where the cell centre is inside the region polygon and,
    if ``land_mask`` is given, on land.
    """
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if land_mask is not None:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape != (lat.size, lon.size):
            raise ValueError(
                f"land_mask shape {land.shape} does not match grid "
                f"({lat.size}, {lon.size})"
            )
    else:
        land = None

    geometries = load_nuts_geometries(shapefile, list(nuts_id_by_region.values()))
    masks: dict[str, np.ndarray] = {}
    empty: list[str] = []
    for region_id, nuts_id in nuts_id_by_region.items():
        mask = rasterize_polygon(geometries[nuts_id], lat, lon)
        if land is not None:
            mask = mask & land
        if not np.any(mask):
            empty.append(f"{region_id} ({nuts_id})")
        masks[region_id] = mask
    if empty:
        raise ValueError(
            "No land grid points found inside NUTS polygons for: " + ", ".join(empty)
        )
    return masks
