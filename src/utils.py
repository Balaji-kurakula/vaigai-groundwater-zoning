"""Utility helpers used across the HydroSight TN project."""

from __future__ import annotations

import base64
import io
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from PIL import Image

from .config import (
    NODATA_FLOAT,
    NODATA_UINT8,
    PROJECT_ROOT,
    STUDY_BOUNDS_WGS84,
    TARGET_CRS,
)

LOGGER = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the project root directory."""

    return PROJECT_ROOT


def ensure_parent(path: Path) -> Path:
    """Create the parent directory for a path if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def study_bounds_in_target_crs() -> tuple[float, float, float, float]:
    """Transform WGS84 study bounds into the project target CRS."""

    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    minx, miny = transformer.transform(STUDY_BOUNDS_WGS84[0], STUDY_BOUNDS_WGS84[1])
    maxx, maxy = transformer.transform(STUDY_BOUNDS_WGS84[2], STUDY_BOUNDS_WGS84[3])
    return minx, miny, maxx, maxy


def create_target_grid(resolution: float) -> tuple[rasterio.Affine, int, int]:
    """Create a study-area grid for the provided output resolution."""

    left, bottom, right, top = study_bounds_in_target_crs()
    width = int(math.ceil((right - left) / resolution))
    height = int(math.ceil((top - bottom) / resolution))
    transform = from_origin(left, top, resolution, resolution)
    return transform, width, height


def build_raster_profile(
    width: int,
    height: int,
    transform: rasterio.Affine,
    dtype: str,
    crs: str = TARGET_CRS,
    nodata: float | int = NODATA_FLOAT,
) -> dict:
    """Construct a GeoTIFF profile for project rasters."""

    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }


def write_raster(
    array: np.ndarray,
    out_path: Path,
    profile: dict,
    nodata: float | int | None = None,
) -> Path:
    """Write a single-band raster to disk."""

    try:
        ensure_parent(out_path)
        updated_profile = profile.copy()
        if nodata is not None:
            updated_profile["nodata"] = nodata
        with rasterio.open(out_path, "w", **updated_profile) as dst:
            dst.write(array, 1)
    except OSError as exc:
        raise OSError(f"Failed to write raster to {out_path}") from exc
    return out_path


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Read a single-band raster array and profile."""

    try:
        with rasterio.open(path) as src:
            return src.read(1), src.profile.copy()
    except OSError as exc:
        raise OSError(f"Failed to read raster from {path}") from exc


def sample_raster_values(path: Path, xy_points: Iterable[tuple[float, float]]) -> np.ndarray:
    """Sample raster values for target-CRS XY coordinates."""

    try:
        with rasterio.open(path) as src:
            samples = np.array([value[0] for value in src.sample(list(xy_points))], dtype="float32")
        return samples
    except OSError as exc:
        raise OSError(f"Failed to sample raster {path}") from exc


def lonlat_to_target_xy(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 lon/lat coordinates into target CRS XY."""

    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    return transformer.transform(lon, lat)


def normalize_category(value: object) -> str:
    """Normalize a categorical value for robust lookup matching."""

    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    for token in (" ", "-", "/", "\\", ".", ","):
        text = text.replace(token, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def map_categories(series: pd.Series, value_map: dict[str | int, int]) -> pd.Series:
    """Map numeric or string categories to integer codes."""

    normalized_map = {normalize_category(key): val for key, val in value_map.items()}

    def mapper(value: object) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = int(value)
            if numeric in value_map:
                return float(value_map[numeric])
            return float(numeric)
        key = normalize_category(value)
        if key in normalized_map:
            return float(normalized_map[key])
        for candidate, code in normalized_map.items():
            if candidate in key or key in candidate:
                return float(code)
        return np.nan

    return series.apply(mapper)


def compute_area_stats(raster_path: Path) -> dict[int, float]:
    """Compute area statistics in km² for classes 1-5."""

    array, profile = read_raster(raster_path)
    transform = profile["transform"]
    pixel_area_km2 = abs(transform.a * transform.e) / 1_000_000
    nodata = profile.get("nodata", NODATA_UINT8)
    stats: dict[int, float] = {}
    for value in range(1, 6):
        stats[value] = float(np.sum(array == value) * pixel_area_km2)
    if np.all(array == nodata):
        LOGGER.warning("No valid pixels found in %s while computing area statistics.", raster_path)
    return stats


def png_to_data_uri(path: Path) -> str:
    """Convert a PNG file to a data URI for HTML embedding."""

    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def array_to_png_data_uri(rgba: np.ndarray) -> str:
    """Convert an RGBA array into a PNG data URI."""

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
