"""Master pipeline for HydroSight TN."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import AHP_PAIRWISE_MATRIX, CLASS_COLORS, FEATURE_ORDER
from src.ahp import compute_ahp_weights, weighted_overlay
from src.config import CLASSIFIED_DIR, MODELS_DIR, NODATA_FLOAT, OUTPUTS_DIR, STANDARDIZED_DIR
from src.preprocess import (
    classify_layer,
    compute_twi,
    derive_drainage_density,
    downscale_smap,
    extract_lineament_density,
    rasterize_vector,
    standardize_layer,
)
from src.train import build_training_dataset, generate_shap_plot, predict_full_map, train_xgboost
from src.utils import read_raster, write_raster
from src.validate import validate_model
from src.visualize import plot_area_statistics, raster_to_rgb_png

LOGGER = logging.getLogger("hydrosight_tn")


def setup_logging() -> None:
    """Configure pipeline logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_raster_output(path: Path, label: str) -> None:
    """Log raster output path and shape.

    Args:
        path: Raster path to inspect.
        label: Human-readable label for logging.

    Returns:
        None.
    """

    with rasterio.open(path) as src:
        LOGGER.info("%s saved to %s with shape %s", label, path, src.shape)


def infer_vector_field(shp_path: Path, candidates: list[str]) -> str:
    """Infer the most likely attribute field from a shapefile.

    Args:
        shp_path: Shapefile path.
        candidates: Candidate column names ranked by preference.

    Returns:
        Selected field name.
    """

    gdf = gpd.read_file(shp_path)
    normalized = {column.lower(): column for column in gdf.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    string_columns = [col for col in gdf.columns if col.lower() != "geometry"]
    if string_columns:
        return string_columns[0]
    raise ValueError(f"Unable to infer attribute field for {shp_path.name}")


def prepare_train_test_csv(training_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    """Split the training data into train and holdout CSV files.

    Args:
        training_csv: Full training CSV path.
        output_dir: Output directory for the split CSV files.

    Returns:
        Tuple of train and test CSV paths.
    """

    data = pd.read_csv(training_csv)
    stratify = data["label"] if data["label"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        data,
        test_size=0.2 if len(data) >= 10 else 0.5,
        random_state=42,
        stratify=stratify,
    )
    train_csv = output_dir / "train_data.csv"
    test_csv = output_dir / "test_data.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    LOGGER.info("Prepared train split at %s and test split at %s", train_csv, test_csv)
    return train_csv, test_csv


def create_placeholder_raster(reference_raster: Path, out_path: Path, fill_value: float) -> Path:
    """Create a constant-value raster aligned to a reference raster.

    Args:
        reference_raster: Raster providing shape, transform, and valid mask.
        out_path: Output placeholder raster path.
        fill_value: Constant value written into valid pixels.

    Returns:
        Path to the saved placeholder raster.
    """

    array, profile = read_raster(reference_raster)
    nodata = profile.get("nodata", NODATA_FLOAT)
    valid_mask = array != nodata
    placeholder = np.full(array.shape, nodata, dtype="float32")
    placeholder[valid_mask] = float(fill_value)
    profile.update(dtype="float32", nodata=nodata, compress="lzw")
    write_raster(placeholder.astype("float32"), out_path, profile, nodata=nodata)
    return out_path


def try_use_preclassified_raster(raster_path: Path, out_path: Path) -> Path | None:
    """Preserve an already-rated 1-5 raster instead of reclassifying it again.

    Args:
        raster_path: Standardized thematic raster path.
        out_path: Output classified raster path.

    Returns:
        Classified raster path when the input already looks like a 1-5 suitability raster, else None.
    """

    array, profile = read_raster(raster_path)
    nodata = profile.get("nodata", NODATA_FLOAT)
    valid_mask = array != nodata
    if np.sum(valid_mask) == 0:
        return None

    rounded = np.rint(array[valid_mask])
    if np.allclose(array[valid_mask], rounded, atol=0.01) and np.nanmin(rounded) >= 1 and np.nanmax(rounded) <= 5:
        classified = np.full(array.shape, 255, dtype="uint8")
        classified[valid_mask] = rounded.astype("uint8")
        profile.update(dtype="uint8", nodata=255, compress="lzw")
        write_raster(classified, out_path, profile, nodata=255)
        return out_path
    return None


def prepare_thematic_layer(
    layer_name: str,
    raster_source: Path | None,
    vector_source: Path | None,
    reference_raster: Path,
    output_dir: Path,
    vector_mapping: dict[str, object],
    optional: bool = False,
    placeholder_value: float | None = None,
) -> Path:
    """Prepare a thematic layer from raster, vector, or placeholder input.

    Args:
        layer_name: Canonical layer name.
        raster_source: Optional raster source path.
        vector_source: Optional vector source path.
        reference_raster: Reference raster for rasterization and placeholders.
        output_dir: Standardized output directory.
        vector_mapping: Candidate field names and categorical mapping.
        optional: Whether the layer may be missing.
        placeholder_value: Optional constant placeholder code if no source exists.

    Returns:
        Path to the prepared standardized raster.
    """

    out_path = output_dir / f"{layer_name}.tif"
    if raster_source is not None and raster_source.exists():
        prepared = standardize_layer(raster_source, out_path, resampling_method=Resampling.nearest)
        log_raster_output(prepared, f"Standardized {layer_name}")
        return prepared

    if vector_source is not None and vector_source.exists():
        field = infer_vector_field(vector_source, vector_mapping["candidates"])
        prepared = rasterize_vector(
            vector_source,
            reference_raster,
            field=field,
            out_path=out_path,
            value_map=vector_mapping["value_map"],
        )
        log_raster_output(prepared, f"Rasterized {layer_name}")
        return prepared

    if placeholder_value is not None:
        prepared = create_placeholder_raster(reference_raster, out_path, fill_value=placeholder_value)
        LOGGER.warning("No %s source found. Created neutral placeholder raster with code %s.", layer_name, placeholder_value)
        log_raster_output(prepared, f"Placeholder {layer_name}")
        return prepared

    if optional:
        raise FileNotFoundError(f"Optional layer {layer_name} was requested but no source or placeholder is available.")
    raise FileNotFoundError(f"Missing required thematic source for {layer_name}.")


def run_pipeline(data_dir: Path, output_dir: Path) -> None:
    """Run the full HydroSight TN workflow.

    Args:
        data_dir: Input raw data directory.
        output_dir: Output directory for artifacts.

    Returns:
        None.
    """

    standardized_dir = STANDARDIZED_DIR
    classified_dir = CLASSIFIED_DIR
    models_dir = MODELS_DIR
    standardized_dir.mkdir(parents=True, exist_ok=True)
    classified_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rasters = {
        "slope": data_dir / "slope.tif",
        "dem": data_dir / "dem.tif",
        "sentinel2_b8": data_dir / "sentinel2_B8.tif",
        "rainfall": data_dir / "rainfall.tif",
        "ndvi": data_dir / "ndvi.tif",
        "ndvi_1km": data_dir / "ndvi_1km.tif",
        "lst_1km": data_dir / "lst_1km.tif",
        "geology": data_dir / "geology.tif",
        "geomorphology": data_dir / "geomorphology.tif",
        "soil": data_dir / "soil.tif",
        "twi_proxy": data_dir / "vaigai_twi.tif",
    }
    raw_vectors = {
        "geology": data_dir / "geology.shp",
        "geomorphology": data_dir / "geomorphology.shp",
        "soil": data_dir / "soil.shp",
    }
    observation_candidates = [
        data_dir / "borewells.shp",
        data_dir / "1_India_GWLs_2000_2024_wells_within_India.csv",
    ]
    smap_path = data_dir / "smap_sm.h5"

    required_inputs = [
        raw_rasters["slope"],
        raw_rasters["dem"],
        raw_rasters["sentinel2_b8"],
        raw_rasters["rainfall"],
        raw_rasters["ndvi"],
        raw_rasters["ndvi_1km"],
        raw_rasters["lst_1km"],
        smap_path,
    ]
    for required in required_inputs:
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")

    if not raw_rasters["geomorphology"].exists() and not raw_vectors["geomorphology"].exists():
        raise FileNotFoundError("Missing geomorphology source. Provide geomorphology.tif or geomorphology.shp.")
    if not raw_rasters["soil"].exists() and not raw_vectors["soil"].exists():
        raise FileNotFoundError("Missing soil source. Provide soil.tif or soil.shp.")

    observation_path = next((path for path in observation_candidates if path.exists()), None)
    if observation_path is None:
        LOGGER.warning("No borewell shapefile or point CSV found. Proxy training will be fully synthetic.")

    vector_mappings = {
        "geology": {
            "candidates": ["code", "class", "lithology", "geology", "rock_type", "unit_name"],
            "value_map": {
                "Alluvium": 1,
                "Laterite": 2,
                "Charnockite": 3,
                "Granite_Gneiss": 4,
                "Crystalline": 5,
            },
        },
        "geomorphology": {
            "candidates": ["code", "class", "geomorphology", "landform", "unit_name"],
            "value_map": {
                "Flood_Plain": 1,
                "Pediplain": 2,
                "Valley_Fill": 3,
                "Pediment": 4,
                "Denudational_Hill": 5,
                "Rocky": 6,
            },
        },
        "soil": {
            "candidates": ["code", "class", "soil", "soil_type", "texture", "unit_name"],
            "value_map": {
                "Sandy_Loam": 1,
                "Alluvial": 2,
                "Red_Loam": 3,
                "Black_Cotton": 4,
                "Sandy_Clay": 5,
                "Rocky": 6,
            },
        },
    }

    LOGGER.info("Step 1/9: Standardizing core raster inputs.")
    standardized_rasters: dict[str, Path] = {}
    for layer_name in tqdm(["slope", "dem", "sentinel2_b8", "rainfall", "ndvi"], desc="Standardizing 30m rasters"):
        out_path = standardized_dir / f"{layer_name}.tif"
        standardized_rasters[layer_name] = standardize_layer(raw_rasters[layer_name], out_path)
        log_raster_output(out_path, f"Standardized {layer_name}")

    dem_1km = standardize_layer(raw_rasters["dem"], output_dir / "dem_1km.tif", target_res=1000.0)
    slope_1km = standardize_layer(raw_rasters["slope"], output_dir / "slope_1km.tif", target_res=1000.0)
    ndvi_1km = standardize_layer(raw_rasters["ndvi_1km"], output_dir / "ndvi_1km_standardized.tif", target_res=1000.0)
    lst_1km = standardize_layer(raw_rasters["lst_1km"], output_dir / "lst_1km_standardized.tif", target_res=1000.0)
    for helper_path, label in [
        (dem_1km, "1km DEM"),
        (slope_1km, "1km slope"),
        (ndvi_1km, "1km NDVI"),
        (lst_1km, "1km LST"),
    ]:
        log_raster_output(helper_path, label)

    LOGGER.info("Step 2/9: Preparing thematic layers.")
    reference_raster = standardized_rasters["slope"]
    thematic_layers = {
        "geology": prepare_thematic_layer(
            "geology",
            raster_source=raw_rasters["geology"],
            vector_source=raw_vectors["geology"],
            reference_raster=reference_raster,
            output_dir=standardized_dir,
            vector_mapping=vector_mappings["geology"],
            optional=True,
            placeholder_value=3.0,
        ),
        "geomorphology": prepare_thematic_layer(
            "geomorphology",
            raster_source=raw_rasters["geomorphology"],
            vector_source=raw_vectors["geomorphology"],
            reference_raster=reference_raster,
            output_dir=standardized_dir,
            vector_mapping=vector_mappings["geomorphology"],
        ),
        "soil": prepare_thematic_layer(
            "soil",
            raster_source=raw_rasters["soil"],
            vector_source=raw_vectors["soil"],
            reference_raster=reference_raster,
            output_dir=standardized_dir,
            vector_mapping=vector_mappings["soil"],
        ),
    }

    LOGGER.info("Step 3/9: Deriving hydrologic and structural layers.")
    lineament_density = extract_lineament_density(standardized_rasters["sentinel2_b8"], standardized_dir / "lineament_density.tif")
    if raw_rasters["twi_proxy"].exists():
        twi = standardize_layer(raw_rasters["twi_proxy"], standardized_dir / "twi.tif")
        LOGGER.info("Using provided TWI proxy raster from %s", raw_rasters["twi_proxy"])
    else:
        twi = compute_twi(standardized_rasters["dem"], standardized_dir / "twi.tif")
    drainage_density = derive_drainage_density(standardized_rasters["dem"], standardized_dir / "drainage_density.tif")
    soil_moisture_1km = downscale_smap(smap_path, ndvi_1km, lst_1km, dem_1km, slope_1km, output_dir / "soil_moisture_1km.tif")
    soil_moisture = standardize_layer(soil_moisture_1km, standardized_dir / "soil_moisture.tif")
    for helper_path, label in [
        (lineament_density, "Lineament density"),
        (twi, "TWI"),
        (drainage_density, "Drainage density"),
        (soil_moisture_1km, "1km soil moisture"),
        (soil_moisture, "30m soil moisture"),
    ]:
        log_raster_output(helper_path, label)

    standardized_layers = {
        "geology": thematic_layers["geology"],
        "geomorphology": thematic_layers["geomorphology"],
        "lineament_density": lineament_density,
        "soil": thematic_layers["soil"],
        "slope": standardized_rasters["slope"],
        "rainfall": standardized_rasters["rainfall"],
        "drainage_density": drainage_density,
        "twi": twi,
        "ndvi": standardized_rasters["ndvi"],
        "soil_moisture": soil_moisture,
    }

    LOGGER.info("Step 4/9: Classifying layers on the 1-5 rating scale.")
    classified_layers: dict[str, Path] = {}
    for layer_name in tqdm(FEATURE_ORDER, desc="Classifying layers"):
        out_path = classified_dir / f"{layer_name}.tif"
        if layer_name in {"geomorphology", "soil"}:
            preclassified = try_use_preclassified_raster(standardized_layers[layer_name], out_path)
            if preclassified is not None:
                classified_layers[layer_name] = preclassified
                LOGGER.info("Detected preclassified %s raster and preserved its 1-5 ratings.", layer_name)
                log_raster_output(out_path, f"Classified {layer_name}")
                continue
        classified_layers[layer_name] = classify_layer(standardized_layers[layer_name], layer_name, out_path)
        log_raster_output(out_path, f"Classified {layer_name}")

    LOGGER.info("Step 5/9: Running AHP overlay.")
    ahp_weights = compute_ahp_weights(AHP_PAIRWISE_MATRIX)
    ahp_map_path, ahp_score_path, ahp_stats = weighted_overlay(
        {key: classified_layers[key] for key in ahp_weights.keys()},
        ahp_weights,
        output_dir / "gwpz_ahp.tif",
    )
    log_raster_output(ahp_map_path, "AHP GWPZ map")
    log_raster_output(ahp_score_path, "AHP score map")

    LOGGER.info("Step 6/9: Building the training dataset.")
    training_csv = build_training_dataset(
        observation_path,
        classified_layers,
        ahp_score_raster=ahp_score_path,
        output_csv=output_dir / "training_data.csv",
    )
    train_csv, test_csv = prepare_train_test_csv(training_csv, output_dir)

    LOGGER.info("Step 7/9: Training, validating, and finalizing XGBoost.")
    LOGGER.warning("ML training is running in reduced-data mode using groundwater-level-derived labels or synthetic proxies instead of observed borewell yield.")
    validation_model, _ = train_xgboost(train_csv, model_path=models_dir / "gwpz_xgboost_validation.pkl")
    test_df = pd.read_csv(test_csv)
    test_df["ml_score"] = validation_model.predict_proba(test_df[FEATURE_ORDER])[:, 1]
    test_df.to_csv(test_csv, index=False)
    validate_model(validation_model, test_csv, output_dir=output_dir)

    final_model, full_training_df = train_xgboost(training_csv, model_path=models_dir / "gwpz_xgboost.pkl")
    generate_shap_plot(final_model, full_training_df[FEATURE_ORDER], FEATURE_ORDER, out_path=output_dir / "shap_summary.png")

    LOGGER.info("Step 8/9: Generating full ML prediction map and visual outputs.")
    ml_map_path, ml_prob_path = predict_full_map(
        final_model,
        classified_layers,
        output_dir / "gwpz_xgboost.tif",
        probability_path=output_dir / "gwpz_xgboost_prob.tif",
    )
    raster_to_rgb_png(ml_map_path, CLASS_COLORS, output_dir / "gwpz_xgboost_rgb.png")
    raster_to_rgb_png(ahp_map_path, CLASS_COLORS, output_dir / "gwpz_ahp_rgb.png")
    plot_area_statistics(ml_map_path, output_dir / "gwpz_statistics_ml.png")
    plot_area_statistics(ahp_map_path, output_dir / "gwpz_statistics_ahp.png")
    for helper_path, label in [
        (ml_map_path, "ML GWPZ map"),
        (ml_prob_path, "ML probability map"),
    ]:
        log_raster_output(helper_path, label)

    LOGGER.info("Step 9/9: Summary report.")
    LOGGER.info("AHP area stats (km2): %s", ahp_stats)
    LOGGER.info("ML outputs saved to %s", output_dir)
    LOGGER.info("Deployment model saved to %s", models_dir / "gwpz_xgboost.pkl")
    LOGGER.info("Observation source used for training: %s", observation_path if observation_path else "synthetic AHP proxy points only")


def main() -> None:
    """Parse arguments and execute the pipeline."""

    parser = argparse.ArgumentParser(description="Run the HydroSight TN groundwater zoning pipeline.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"), help="Directory containing raw input datasets.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory for output rasters and reports.")
    args = parser.parse_args()

    setup_logging()
    run_pipeline(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
