"""FastAPI application for HydroSight TN."""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import geopandas as gpd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from folium.plugins import Fullscreen
from pyproj import Transformer
from rasterio.transform import xy as raster_xy

from src import CLASS_COLORS, CLASS_LABELS, FEATURE_ORDER, RECOMMENDATIONS
from src.config import CLASSIFIED_DIR, MODELS_DIR, OUTPUTS_DIR, RAW_DIR, TARGET_CRS
from src.utils import (
    array_to_png_data_uri,
    compute_area_stats,
    lonlat_to_target_xy,
    png_to_data_uri,
    read_raster,
    sample_raster_values,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MODEL_PATH = MODELS_DIR / "gwpz_xgboost.pkl"
AHP_MAP_PATH = OUTPUTS_DIR / "gwpz_ahp.tif"
ML_MAP_PATH = OUTPUTS_DIR / "gwpz_xgboost.tif"
ML_OVERLAY_PATH = OUTPUTS_DIR / "gwpz_xgboost_rgb.png"
AHP_OVERLAY_PATH = OUTPUTS_DIR / "gwpz_ahp_rgb.png"
BOREWELL_PATH = RAW_DIR / "borewells.shp"

app = FastAPI(title="HydroSight TN", version="1.0.0")
app.state.model = None


def _classified_layer_paths() -> dict[str, Path]:
    """Return the canonical classified-layer lookup."""

    return {feature: CLASSIFIED_DIR / f"{feature}.tif" for feature in FEATURE_ORDER}


def _load_compatible_model(path: Path):
    """Load a model only if its feature count matches the active feature stack."""

    if not path.exists():
        return None
    model = joblib.load(path)
    expected_features = len(FEATURE_ORDER)
    actual_features = getattr(model, "n_features_in_", expected_features)
    if actual_features != expected_features:
        LOGGER.warning(
            "Ignoring model at %s because it expects %s features while the active stack has %s.",
            path,
            actual_features,
            expected_features,
        )
        return None
    return model


def _require_file(path: Path, label: str) -> None:
    """Raise an HTTP error if an expected file is missing."""

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing {label}: {path}")


def _sample_features(lon: float, lat: float) -> tuple[dict[str, float], list[str]]:
    """Sample all classified feature rasters at a lon/lat location.

    Args:
        lon: Longitude in WGS84.
        lat: Latitude in WGS84.

    Returns:
        Tuple of sampled features and a list of layers that were nodata at the point.
    """

    x, y = lonlat_to_target_xy(lon, lat)
    samples: dict[str, float] = {}
    invalid_layers: list[str] = []
    for feature, path in _classified_layer_paths().items():
        _require_file(path, f"classified layer '{feature}'")
        value = float(sample_raster_values(path, [(x, y)])[0])
        if value == 255:
            invalid_layers.append(feature)
        else:
            samples[feature] = value
    return samples, invalid_layers


def _coverage_payload(lat: float, lon: float) -> dict[str, object]:
    """Build a coverage summary for a query point.

    Args:
        lat: Latitude in WGS84.
        lon: Longitude in WGS84.

    Returns:
        Coverage summary dictionary for the query point.
    """

    features, invalid_layers = _sample_features(lon=lon, lat=lat)
    ahp_available_at_point = False
    if AHP_MAP_PATH.exists():
        x, y = lonlat_to_target_xy(lon, lat)
        ahp_value = int(sample_raster_values(AHP_MAP_PATH, [(x, y)])[0])
        ahp_available_at_point = ahp_value != 255

    can_predict = len(invalid_layers) == 0 and (app.state.model is not None or ahp_available_at_point)
    return {
        "latitude": lat,
        "longitude": lon,
        "can_predict": can_predict,
        "missing_layers": invalid_layers,
        "available_layers": list(features.keys()),
        "available_layer_count": len(features),
        "required_layer_count": len(FEATURE_ORDER),
        "model_loaded": app.state.model is not None,
        "ahp_available_at_point": ahp_available_at_point,
        "message": (
            "Prediction is available at this location."
            if can_predict
            else "Prediction is not available at this location because one or more required layers are nodata."
        ),
    }


def _coverage_mask() -> tuple[np.ndarray, dict]:
    """Compute the common valid-data mask across all classified layers.

    Returns:
        Tuple of boolean coverage mask and raster profile from the last layer read.
    """

    mask = None
    profile = None
    for feature, path in _classified_layer_paths().items():
        _require_file(path, f"classified layer '{feature}'")
        array, profile = read_raster(path)
        current_valid = array != profile.get("nodata", 255)
        mask = current_valid if mask is None else (mask & current_valid)
    assert profile is not None
    return mask, profile


def _coverage_summary() -> dict[str, object]:
    """Summarize valid prediction coverage across the study area.

    Returns:
        Coverage statistics dictionary.
    """

    mask, profile = _coverage_mask()
    total_pixels = int(mask.size)
    valid_pixels = int(np.sum(mask))
    valid_fraction = float(valid_pixels / total_pixels) if total_pixels else 0.0
    pixel_area_km2 = abs(profile["transform"].a * profile["transform"].e) / 1_000_000
    return {
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_fraction": valid_fraction,
        "valid_percent": valid_fraction * 100.0,
        "valid_area_km2": valid_pixels * pixel_area_km2,
        "invalid_area_km2": (total_pixels - valid_pixels) * pixel_area_km2,
        "required_layer_count": len(FEATURE_ORDER),
    }


def _coverage_breakdown() -> dict[str, object]:
    """Explain per-layer data coverage and each layer's effect on common overlap.

    Returns:
        Dictionary with per-layer validity statistics and leave-one-out overlap gains.
    """

    masks: dict[str, np.ndarray] = {}
    profile = None
    per_layer: dict[str, dict[str, float | int]] = {}

    for feature, path in _classified_layer_paths().items():
        _require_file(path, f"classified layer '{feature}'")
        array, profile = read_raster(path)
        nodata = profile.get("nodata", 255)
        valid_mask = array != nodata
        masks[feature] = valid_mask
        per_layer[feature] = {
            "valid_pixels": int(np.sum(valid_mask)),
            "valid_percent": float(np.mean(valid_mask) * 100.0),
        }

    common_mask = np.logical_and.reduce([masks[feature] for feature in FEATURE_ORDER])
    common_valid_percent = float(np.mean(common_mask) * 100.0)

    leave_one_out: dict[str, dict[str, float | int]] = {}
    for feature in FEATURE_ORDER:
        remaining = [masks[name] for name in FEATURE_ORDER if name != feature]
        reduced_mask = np.logical_and.reduce(remaining)
        reduced_valid_percent = float(np.mean(reduced_mask) * 100.0)
        leave_one_out[feature] = {
            "valid_pixels": int(np.sum(reduced_mask)),
            "valid_percent": reduced_valid_percent,
            "gain_percent_points": reduced_valid_percent - common_valid_percent,
        }

    limiting_by_validity = min(per_layer.items(), key=lambda item: item[1]["valid_percent"])
    limiting_by_overlap = max(leave_one_out.items(), key=lambda item: item[1]["gain_percent_points"])

    return {
        "common_valid_percent": common_valid_percent,
        "per_layer": per_layer,
        "leave_one_out": leave_one_out,
        "most_limited_single_layer": {
            "layer": limiting_by_validity[0],
            **limiting_by_validity[1],
        },
        "largest_overlap_gain_if_ignored": {
            "layer": limiting_by_overlap[0],
            **limiting_by_overlap[1],
        },
    }


def _sample_valid_locations(count: int = 20) -> list[dict[str, float | int]]:
    """Return sample WGS84 coordinates from valid prediction coverage.

    Args:
        count: Number of valid sample points to return.

    Returns:
        List of dictionaries containing row/col and lat/lon coordinates.
    """

    mask, profile = _coverage_mask()
    valid_indices = np.argwhere(mask)
    if len(valid_indices) == 0:
        return []

    count = max(1, min(count, len(valid_indices), 200))
    sample_positions = np.linspace(0, len(valid_indices) - 1, num=count, dtype=int)
    transformer = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
    samples: list[dict[str, float | int]] = []
    for pos in sample_positions:
        row_idx, col_idx = valid_indices[pos]
        x, y = raster_xy(profile["transform"], int(row_idx), int(col_idx), offset="center")
        lon, lat = transformer.transform(x, y)
        samples.append(
            {
                "row": int(row_idx),
                "col": int(col_idx),
                "latitude": float(lat),
                "longitude": float(lon),
            }
        )
    return samples


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the root URL to the API docs."""

    return RedirectResponse(url="/docs")


@app.get("/coverage")
def coverage(lat: float, lon: float) -> JSONResponse:
    """Report whether a point has complete layer coverage for prediction."""

    return JSONResponse(_coverage_payload(lat=lat, lon=lon))


@app.get("/coverage-summary")
def coverage_summary() -> JSONResponse:
    """Return study-area prediction coverage statistics."""

    return JSONResponse(_coverage_summary())


@app.get("/coverage-breakdown")
def coverage_breakdown() -> JSONResponse:
    """Return per-layer coverage and overlap-loss diagnostics."""

    return JSONResponse(_coverage_breakdown())


@app.get("/coverage-samples")
def coverage_samples(count: int = 20) -> JSONResponse:
    """Return sample valid query locations from the coverage mask."""

    return JSONResponse(
        {
            "count": count,
            "samples": _sample_valid_locations(count=count),
        }
    )


@app.get("/coverage-map", response_class=HTMLResponse)
def coverage_map() -> HTMLResponse:
    """Return an interactive map showing where prediction coverage is valid."""

    mask, _ = _coverage_mask()
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask] = (26, 152, 80, 180)
    rgba[~mask] = (215, 48, 39, 70)
    overlay_uri = array_to_png_data_uri(rgba)
    summary = _coverage_summary()

    fmap = folium.Map(location=[10.0, 77.8], zoom_start=9, tiles="CartoDB positron")
    Fullscreen().add_to(fmap)
    folium.raster_layers.ImageOverlay(
        image=overlay_uri,
        bounds=[[9.5, 77.0], [11.0, 78.5]],
        opacity=0.7,
        name="Prediction Coverage",
        interactive=True,
        cross_origin=False,
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.add_child(folium.LatLngPopup())
    folium.Marker(
        [10.95, 77.05],
        tooltip=(
            f"Valid coverage: {summary['valid_percent']:.2f}% | "
            f"Valid area: {summary['valid_area_km2']:.2f} km2"
        ),
    ).add_to(fmap)
    return HTMLResponse(fmap.get_root().render())


@app.on_event("startup")
def load_model() -> None:
    """Load the trained model at API startup."""

    app.state.model = _load_compatible_model(MODEL_PATH)
    if app.state.model is not None:
        LOGGER.info("Loaded model from %s", MODEL_PATH)
    else:
        LOGGER.warning("No compatible model file available at %s", MODEL_PATH)


@app.get("/predict")
def predict(lat: float, lon: float) -> JSONResponse:
    """Predict groundwater potential for a given lat/lon."""
    features, invalid_layers = _sample_features(lon=lon, lat=lat)
    if invalid_layers:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Point falls outside valid coverage for one or more classified layers.",
                "invalid_layers": invalid_layers,
                "latitude": lat,
                "longitude": lon,
            },
        )

    ahp_class = None
    if AHP_MAP_PATH.exists():
        x, y = lonlat_to_target_xy(lon, lat)
        ahp_value = int(sample_raster_values(AHP_MAP_PATH, [(x, y)])[0])
        ahp_class = None if ahp_value == 255 else ahp_value

    if app.state.model is not None:
        ordered = np.array([[features[name] for name in FEATURE_ORDER]], dtype="float32")
        confidence = float(app.state.model.predict_proba(ordered)[0, 1])
        if confidence < 0.2:
            gwpz_class = 1
        elif confidence < 0.4:
            gwpz_class = 2
        elif confidence < 0.6:
            gwpz_class = 3
        elif confidence < 0.8:
            gwpz_class = 4
        else:
            gwpz_class = 5
        prediction_source = "ml"
    elif ahp_class is not None:
        gwpz_class = ahp_class
        confidence = float(gwpz_class) / 5.0
        prediction_source = "ahp_fallback"
    else:
        raise HTTPException(status_code=503, detail="No ML model or AHP map is available. Run the pipeline first.")

    response = {
        "latitude": lat,
        "longitude": lon,
        "gwpz_class": gwpz_class,
        "gwpz_label": CLASS_LABELS[gwpz_class],
        "confidence": confidence,
        "color": CLASS_COLORS[gwpz_class],
        "ahp_class": ahp_class,
        "recommendation": RECOMMENDATIONS[gwpz_class],
        "prediction_source": prediction_source,
    }
    return JSONResponse(response)


@app.get("/map", response_class=HTMLResponse)
def map_view() -> HTMLResponse:
    """Return an interactive Folium map with groundwater overlays."""

    overlay_path = ML_OVERLAY_PATH if ML_OVERLAY_PATH.exists() else AHP_OVERLAY_PATH
    overlay_label = "GWPZ ML Overlay" if ML_OVERLAY_PATH.exists() else "GWPZ AHP Overlay"
    _require_file(overlay_path, "classified overlay PNG")
    overlay_uri = png_to_data_uri(overlay_path)

    fmap = folium.Map(location=[10.0, 77.8], zoom_start=9, tiles="CartoDB positron")
    Fullscreen().add_to(fmap)
    folium.raster_layers.ImageOverlay(
        image=overlay_uri,
        bounds=[[9.5, 77.0], [11.0, 78.5]],
        opacity=0.65,
        name=overlay_label,
        interactive=True,
        cross_origin=False,
    ).add_to(fmap)

    if BOREWELL_PATH.exists():
        borewells = gpd.read_file(BOREWELL_PATH).to_crs("EPSG:4326")
        productive = folium.FeatureGroup(name="Productive Borewells", show=True)
        dry = folium.FeatureGroup(name="Dry/Low-yield Borewells", show=True)
        for _, row in borewells.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            group = productive if float(row.get("yield_lpm", 0)) > 500 else dry
            color = "blue" if group is productive else "red"
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.85,
                popup=f"yield_lpm: {row.get('yield_lpm', 'NA')}",
            ).add_to(group)
        productive.add_to(fmap)
        dry.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return HTMLResponse(fmap.get_root().render())


@app.get("/stats")
def stats() -> JSONResponse:
    """Return area statistics for AHP and ML outputs."""

    response = {
        "ml_area_km2": compute_area_stats(ML_MAP_PATH) if ML_MAP_PATH.exists() else None,
        "ahp_area_km2": compute_area_stats(AHP_MAP_PATH) if AHP_MAP_PATH.exists() else None,
    }
    return JSONResponse(response)


@app.get("/health")
def health() -> JSONResponse:
    """Health-check endpoint."""

    return JSONResponse({"status": "ok", "model_loaded": app.state.model is not None})
