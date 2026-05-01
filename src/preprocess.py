"""Preprocessing utilities for HydroSight TN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import h5py
import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.model_selection import train_test_split
from skimage.morphology import skeletonize
from tqdm import tqdm
from xgboost import XGBRegressor

from .config import NODATA_FLOAT, NODATA_UINT8, TARGET_CRS, TARGET_RESOLUTION
from .utils import (
    build_raster_profile,
    create_target_grid,
    map_categories,
    read_raster,
    write_raster,
)

LOGGER = logging.getLogger(__name__)


def _load_dataset_array(src_path: Path) -> tuple[np.ndarray, dict]:
    """Load a raster or NetCDF dataset as an array/profile pair."""

    if src_path.suffix.lower() in {".nc", ".nc4"}:
        try:
            ds = xr.open_dataset(src_path)
        except OSError as exc:
            raise OSError(f"Failed to open NetCDF dataset {src_path}") from exc
        variable_name = next(
            (
                name
                for name, variable in ds.data_vars.items()
                if variable.ndim >= 2
            ),
            None,
        )
        if variable_name is None:
            raise ValueError(f"No 2D variable found in NetCDF file {src_path}")
        data = ds[variable_name].squeeze().values.astype("float32")
        lon_name = next((name for name in ds.coords if name.lower() in {"lon", "longitude", "x"}), None)
        lat_name = next((name for name in ds.coords if name.lower() in {"lat", "latitude", "y"}), None)
        if lon_name is None or lat_name is None:
            raise ValueError(f"Expected lat/lon coordinates in NetCDF file {src_path}")
        lon = ds[lon_name].values
        lat = ds[lat_name].values
        transform = from_bounds(
            float(np.min(lon)),
            float(np.min(lat)),
            float(np.max(lon)),
            float(np.max(lat)),
            data.shape[1],
            data.shape[0],
        )
        profile = build_raster_profile(
            width=data.shape[1],
            height=data.shape[0],
            transform=transform,
            dtype="float32",
            crs="EPSG:4326",
            nodata=NODATA_FLOAT,
        )
        return data, profile
    try:
        with rasterio.open(src_path) as src:
            array = src.read(1).astype("float32")
            profile = src.profile.copy()
        return array, profile
    except OSError as exc:
        raise OSError(f"Failed to open raster {src_path}") from exc


def _mask_nodata(array: np.ndarray, profile: dict) -> np.ndarray:
    """Replace nodata values with NaN for numeric processing."""

    nodata = profile.get("nodata", NODATA_FLOAT)
    masked = array.astype("float32").copy()
    masked[masked == nodata] = np.nan
    return masked


def standardize_layer(
    src_path: Path | str,
    out_path: Path | str,
    target_res: float = TARGET_RESOLUTION,
    resampling_method: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject, resample, and clip a layer to the project study area.

    Args:
        src_path: Input raster or NetCDF file path.
        out_path: Output GeoTIFF path.
        target_res: Output spatial resolution in meters.
        resampling_method: Rasterio resampling enum.

    Returns:
        The output raster path.
    """

    src_path = Path(src_path)
    out_path = Path(out_path)
    array, profile = _load_dataset_array(src_path)
    src_transform = profile["transform"]
    src_crs = profile["crs"] or "EPSG:4326"

    transform, width, height = create_target_grid(target_res)
    destination = np.full((height, width), NODATA_FLOAT, dtype="float32")

    reproject(
        source=array,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=profile.get("nodata", NODATA_FLOAT),
        dst_transform=transform,
        dst_crs=TARGET_CRS,
        dst_nodata=NODATA_FLOAT,
        resampling=resampling_method,
    )

    destination = destination.astype("float32")
    output_profile = build_raster_profile(
        width=width,
        height=height,
        transform=transform,
        dtype="float32",
        crs=TARGET_CRS,
        nodata=NODATA_FLOAT,
    )
    return write_raster(destination, out_path, output_profile, nodata=NODATA_FLOAT)


def rasterize_vector(
    shp_path: Path | str,
    ref_raster: Path | str,
    field: str,
    out_path: Path | str,
    value_map: dict[str | int, int] | None = None,
) -> Path:
    """Rasterize a vector layer to match a reference raster.

    Args:
        shp_path: Input shapefile path.
        ref_raster: Reference raster for grid alignment.
        field: Attribute field to burn.
        out_path: Output raster path.
        value_map: Optional mapping from category labels to integer codes.

    Returns:
        The output raster path.
    """

    shp_path = Path(shp_path)
    ref_raster = Path(ref_raster)
    out_path = Path(out_path)
    try:
        gdf = gpd.read_file(shp_path)
    except OSError as exc:
        raise OSError(f"Failed to read vector file {shp_path}") from exc
    if field not in gdf.columns:
        raise ValueError(f"Field '{field}' not found in {shp_path.name}. Available fields: {list(gdf.columns)}")

    _, ref_profile = read_raster(ref_raster)
    gdf = gdf.to_crs(ref_profile["crs"])
    values = gdf[field]
    if value_map is not None:
        values = map_categories(values, value_map)
    else:
        values = values.astype("float32")

    valid_mask = (~values.isna()) & (~gdf.geometry.is_empty) & gdf.geometry.notnull()
    shapes = list(zip(gdf.loc[valid_mask, "geometry"], values.loc[valid_mask]))
    burned = rasterize(
        shapes=shapes,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        fill=NODATA_FLOAT,
        dtype="float32",
    )
    output_profile = ref_profile.copy()
    output_profile.update(dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    return write_raster(burned.astype("float32"), out_path, output_profile, nodata=NODATA_FLOAT)


def extract_lineament_density(sentinel_b8_path: Path | str, out_path: Path | str) -> Path:
    """Estimate lineament density from Sentinel-2 NIR imagery.

    Args:
        sentinel_b8_path: Standardized Sentinel-2 Band 8 raster.
        out_path: Output density raster path.

    Returns:
        The output raster path.
    """

    array, profile = read_raster(Path(sentinel_b8_path))
    nodata = profile.get("nodata", NODATA_FLOAT)
    valid = array != nodata
    scaled = np.zeros_like(array, dtype="uint8")
    if np.any(valid):
        min_val = float(np.nanmin(array[valid]))
        max_val = float(np.nanmax(array[valid]))
        if max_val > min_val:
            scaled[valid] = ((array[valid] - min_val) / (max_val - min_val) * 255).astype("uint8")

    blurred = cv2.GaussianBlur(scaled, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=10)

    line_mask = np.zeros_like(edges, dtype="uint8")
    if lines is not None:
        for segment in tqdm(lines[:, 0], desc="Drawing lineaments", leave=False):
            x1, y1, x2, y2 = segment
            cv2.line(line_mask, (x1, y1), (x2, y2), color=1, thickness=1)

    window_size = 100
    mean_density = uniform_filter(line_mask.astype("float32"), size=window_size, mode="nearest")
    pixel_size_km = TARGET_RESOLUTION / 1000.0
    density = mean_density / max(pixel_size_km, 1e-6)
    density[~valid] = NODATA_FLOAT

    output_profile = profile.copy()
    output_profile.update(dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    return write_raster(density.astype("float32"), Path(out_path), output_profile, nodata=NODATA_FLOAT)


def compute_twi(dem_path: Path | str, out_path: Path | str) -> Path:
    """Compute a simplified topographic wetness index raster.

    Args:
        dem_path: Standardized DEM raster.
        out_path: Output TWI raster path.

    Returns:
        The output raster path.
    """

    dem, profile = read_raster(Path(dem_path))
    nodata = profile.get("nodata", NODATA_FLOAT)
    valid = dem != nodata
    filled = np.where(valid, dem, np.nan)
    filled = np.where(np.isnan(filled), np.nanmedian(filled), filled)

    y_grad, x_grad = np.gradient(filled, abs(profile["transform"].e), profile["transform"].a)
    slope_radians = np.arctan(np.sqrt(x_grad**2 + y_grad**2))
    slope_tan = np.maximum(np.tan(slope_radians), 0.001)

    lowland_proxy = np.maximum(0.0, np.nanmax(filled) - filled)
    flow_accum = uniform_filter(lowland_proxy + 1.0, size=3, mode="nearest") * 9.0
    twi = np.log(np.maximum(flow_accum, 0.001) / slope_tan)
    twi[~valid] = NODATA_FLOAT

    output_profile = profile.copy()
    output_profile.update(dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    return write_raster(twi.astype("float32"), Path(out_path), output_profile, nodata=NODATA_FLOAT)


def derive_drainage_density(
    dem_path: Path | str,
    out_path: Path | str,
    window_km: float = 5.0,
) -> Path:
    """Approximate drainage density from a DEM-derived channel proxy.

    Args:
        dem_path: Standardized DEM raster.
        out_path: Output drainage density raster path.
        window_km: Moving-window size in kilometers.

    Returns:
        The output raster path.
    """

    dem, profile = read_raster(Path(dem_path))
    nodata = profile.get("nodata", NODATA_FLOAT)
    valid = dem != nodata
    filled = np.where(valid, dem, np.nanmedian(np.where(valid, dem, np.nan)))

    lowland_proxy = np.maximum(0.0, np.nanmax(filled) - filled)
    flow_proxy = uniform_filter(lowland_proxy + 1.0, size=15, mode="nearest") * (15**2)
    threshold = float(np.nanpercentile(flow_proxy[valid], 90))
    channels = flow_proxy >= threshold
    skeleton = skeletonize(channels).astype("float32")

    window_size = max(3, int(round((window_km * 1000.0) / TARGET_RESOLUTION)))
    mean_density = uniform_filter(skeleton, size=window_size, mode="nearest")
    pixel_size_km = TARGET_RESOLUTION / 1000.0
    density = mean_density / max(pixel_size_km, 1e-6)
    density[~valid] = NODATA_FLOAT

    output_profile = profile.copy()
    output_profile.update(dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    return write_raster(density.astype("float32"), Path(out_path), output_profile, nodata=NODATA_FLOAT)


def _discover_hdf5_array(h5_file: h5py.File) -> np.ndarray:
    """Find the most likely SMAP soil-moisture dataset inside an HDF5 file."""

    candidates: list[tuple[str, tuple[int, ...], np.ndarray]] = []

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim < 2 or not np.issubdtype(obj.dtype, np.number):
            return
        array = np.array(obj)
        score = 1 if "soil" in name.lower() or "sm" in name.lower() else 0
        candidates.append((name, array.shape, array if score else array))

    h5_file.visititems(visitor)
    if not candidates:
        raise ValueError("No numeric 2D dataset found in SMAP HDF5 file.")
    prioritized = sorted(candidates, key=lambda item: ("soil" not in item[0].lower() and "sm" not in item[0].lower(), -np.prod(item[1])))
    chosen = prioritized[0][2]
    if chosen.ndim > 2:
        chosen = np.squeeze(chosen)
    if chosen.ndim != 2:
        raise ValueError("Unable to isolate a 2D SMAP soil moisture array.")
    return chosen.astype("float32")


def _resize_to_shape(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize an array to a target shape using bilinear interpolation."""

    return cv2.resize(array.astype("float32"), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def downscale_smap(
    smap_h5: Path | str,
    ndvi_1km: Path | str,
    lst_1km: Path | str,
    dem_1km: Path | str,
    slope_1km: Path | str,
    out_path: Path | str,
) -> Path:
    """Downscale SMAP soil moisture from 9 km to 1 km using XGBoost regression.

    Args:
        smap_h5: Input SMAP HDF5 file path.
        ndvi_1km: 1 km NDVI raster path.
        lst_1km: 1 km LST raster path.
        dem_1km: 1 km DEM raster path.
        slope_1km: 1 km slope raster path.
        out_path: Output 1 km soil moisture raster path.

    Returns:
        The output raster path.
    """

    try:
        with h5py.File(smap_h5, "r") as handle:
            smap = _discover_hdf5_array(handle)
    except OSError as exc:
        raise OSError(f"Failed to read SMAP HDF5 file {smap_h5}") from exc

    ndvi, ndvi_profile = read_raster(Path(ndvi_1km))
    lst, lst_profile = read_raster(Path(lst_1km))
    dem, dem_profile = read_raster(Path(dem_1km))
    slope, slope_profile = read_raster(Path(slope_1km))

    ndvi = _mask_nodata(ndvi, ndvi_profile)
    lst = _mask_nodata(lst, lst_profile)
    dem = _mask_nodata(dem, dem_profile)
    slope = _mask_nodata(slope, slope_profile)

    predictor_stack_1km = np.stack([ndvi, lst, dem, slope], axis=-1).astype("float32")
    predictor_stack_9km = np.stack(
        [_resize_to_shape(layer, smap.shape) for layer in (ndvi, lst, dem, slope)],
        axis=-1,
    ).astype("float32")

    train_mask = np.all(np.isfinite(predictor_stack_9km), axis=-1) & np.isfinite(smap)
    if np.sum(train_mask) < 20:
        raise ValueError("Insufficient valid samples to train SMAP downscaling model.")

    X_train = predictor_stack_9km[train_mask]
    y_train = smap[train_mask]
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    predict_mask = np.all(np.isfinite(predictor_stack_1km), axis=-1)
    prediction = np.full(ndvi.shape, NODATA_FLOAT, dtype="float32")
    prediction[predict_mask] = np.clip(model.predict(predictor_stack_1km[predict_mask]), 0.0, 1.0)

    output_profile = ndvi_profile.copy()
    output_profile.update(dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    return write_raster(prediction, Path(out_path), output_profile, nodata=NODATA_FLOAT)


def classify_layer(raster_path: Path | str, layer_name: str, out_path: Path | str) -> Path:
    """Classify a preprocessed layer into groundwater suitability ratings.

    Args:
        raster_path: Input raster path.
        layer_name: Canonical layer name.
        out_path: Output classified raster path.

    Returns:
        The output raster path.
    """

    array, profile = read_raster(Path(raster_path))
    nodata = profile.get("nodata", NODATA_FLOAT)
    valid = array != nodata
    classified = np.full(array.shape, NODATA_UINT8, dtype="uint8")
    layer = layer_name.lower()
    values = array.astype("float32")

    if layer == "slope":
        classified[valid & (values <= 3)] = 5
        classified[valid & (values > 3) & (values <= 5)] = 4
        classified[valid & (values > 5) & (values <= 10)] = 3
        classified[valid & (values > 10) & (values <= 20)] = 2
        classified[valid & (values > 20)] = 1
    elif layer == "lineament_density":
        classified[valid & (values > 2.0)] = 5
        classified[valid & (values >= 1.5) & (values <= 2.0)] = 4
        classified[valid & (values >= 1.0) & (values < 1.5)] = 3
        classified[valid & (values >= 0.5) & (values < 1.0)] = 2
        classified[valid & (values < 0.5)] = 1
    elif layer == "rainfall":
        classified[valid & (values > 1200)] = 5
        classified[valid & (values >= 900) & (values <= 1200)] = 4
        classified[valid & (values >= 700) & (values < 900)] = 3
        classified[valid & (values >= 500) & (values < 700)] = 2
        classified[valid & (values < 500)] = 1
    elif layer == "twi":
        classified[valid & (values > 10)] = 5
        classified[valid & (values >= 8) & (values <= 10)] = 4
        classified[valid & (values >= 6) & (values < 8)] = 3
        classified[valid & (values >= 4) & (values < 6)] = 2
        classified[valid & (values < 4)] = 1
    elif layer == "ndvi":
        classified[valid & (values > 0.5)] = 5
        classified[valid & (values >= 0.3) & (values <= 0.5)] = 4
        classified[valid & (values >= 0.1) & (values < 0.3)] = 3
        classified[valid & (values >= 0.0) & (values < 0.1)] = 2
        classified[valid & (values < 0.0)] = 1
    elif layer == "soil_moisture":
        classified[valid & (values > 0.4)] = 5
        classified[valid & (values >= 0.3) & (values <= 0.4)] = 4
        classified[valid & (values >= 0.2) & (values < 0.3)] = 3
        classified[valid & (values >= 0.1) & (values < 0.2)] = 2
        classified[valid & (values < 0.1)] = 1
    elif layer == "geology":
        rating_map = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        for code, rating in rating_map.items():
            classified[valid & (np.round(values).astype(int) == code)] = rating
    elif layer == "geomorphology":
        rating_map = {1: 5, 2: 4, 3: 4, 4: 3, 5: 2, 6: 1}
        for code, rating in rating_map.items():
            classified[valid & (np.round(values).astype(int) == code)] = rating
    elif layer == "soil":
        rating_map = {1: 5, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
        for code, rating in rating_map.items():
            classified[valid & (np.round(values).astype(int) == code)] = rating
    elif layer == "lulc":
        rating_map = {1: 5, 2: 4, 7: 4, 8: 2, 9: 1, 10: 3, 11: 3}
        for code, rating in rating_map.items():
            classified[valid & (np.round(values).astype(int) == code)] = rating
    elif layer == "drainage_density":
        q20, q40, q60, q80 = np.nanpercentile(values[valid], [20, 40, 60, 80])
        classified[valid & (values <= q20)] = 5
        classified[valid & (values > q20) & (values <= q40)] = 4
        classified[valid & (values > q40) & (values <= q60)] = 3
        classified[valid & (values > q60) & (values <= q80)] = 2
        classified[valid & (values > q80)] = 1
    else:
        raise ValueError(f"Unsupported layer name for classification: {layer_name}")

    output_profile = profile.copy()
    output_profile.update(dtype="uint8", nodata=NODATA_UINT8, compress="lzw")
    return write_raster(classified, Path(out_path), output_profile, nodata=NODATA_UINT8)


def fill_nodata_nearest(
    raster_path: Path | str,
    out_path: Path | str | None = None,
    nodata_value: float | int | None = None,
) -> Path:
    """Fill nodata cells using the nearest valid pixel value.

    This is useful for categorical rasters such as LULC when isolated nodata holes
    remain after clipping/reprojection.

    Args:
        raster_path: Input raster path.
        out_path: Optional output path. Overwrites the input if not provided.
        nodata_value: Optional nodata override.

    Returns:
        Path to the filled raster.
    """

    raster_path = Path(raster_path)
    out_path = Path(out_path) if out_path else raster_path
    array, profile = read_raster(raster_path)
    nodata = profile.get("nodata", NODATA_FLOAT) if nodata_value is None else nodata_value
    invalid = array == nodata
    if not np.any(invalid):
        if out_path != raster_path:
            write_raster(array, out_path, profile, nodata=nodata)
        return out_path

    valid = ~invalid
    if not np.any(valid):
        raise ValueError(f"Raster {raster_path} contains only nodata values and cannot be filled.")

    _, indices = distance_transform_edt(invalid, return_indices=True)
    filled = array.copy()
    filled[invalid] = array[tuple(indices[:, invalid])]
    return write_raster(filled.astype(profile["dtype"]), out_path, profile, nodata=nodata)
