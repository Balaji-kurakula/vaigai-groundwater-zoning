"""Training utilities for HydroSight TN machine learning workflow."""

from __future__ import annotations

import logging
from pathlib import Path
import re

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from xgboost import XGBClassifier

from .config import FEATURE_ORDER, NODATA_FLOAT, NODATA_UINT8, STUDY_BOUNDS_WGS84
from .utils import ensure_parent, read_raster, sample_raster_values, write_raster

LOGGER = logging.getLogger(__name__)

TIME_COLUMN_PATTERN = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})$")
MONTH_ORDER = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _sample_layers_for_points(
    gdf: gpd.GeoDataFrame,
    classified_layers_dict: dict[str, Path | str],
) -> pd.DataFrame:
    """Sample all classified rasters at point coordinates.

    Args:
        gdf: GeoDataFrame containing point geometries in raster CRS.
        classified_layers_dict: Mapping of feature names to classified raster paths.

    Returns:
        DataFrame containing sampled feature values.
    """

    sample_points = [(geom.x, geom.y) for geom in gdf.geometry]
    sampled_columns: dict[str, np.ndarray] = {}
    for feature in tqdm(FEATURE_ORDER, desc="Sampling classified layers"):
        if feature not in classified_layers_dict:
            raise KeyError(f"Missing classified layer '{feature}' in classified_layers_dict.")
        sampled_columns[feature] = sample_raster_values(Path(classified_layers_dict[feature]), sample_points)
    return pd.DataFrame(sampled_columns)


def _load_point_dataset(observation_path: Path, target_crs: str) -> gpd.GeoDataFrame:
    """Load a point dataset from a shapefile or CSV.

    Args:
        observation_path: Input point file path.
        target_crs: Output CRS matching the classified rasters.

    Returns:
        GeoDataFrame transformed into the target CRS.
    """

    if observation_path.suffix.lower() == ".shp":
        gdf = gpd.read_file(observation_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf.to_crs(target_crs)

    if observation_path.suffix.lower() == ".csv":
        table = pd.read_csv(observation_path)
        lat_col = next((col for col in table.columns if col.strip().lower() in {"latitude", "lat"}), None)
        lon_col = next((col for col in table.columns if col.strip().lower() in {"longitude", "lon", "long"}), None)
        if lat_col is None or lon_col is None:
            raise ValueError(f"CSV file {observation_path.name} must contain latitude/longitude columns.")

        table = table.dropna(subset=[lat_col, lon_col]).copy()
        table = table[
            table[lat_col].between(STUDY_BOUNDS_WGS84[1], STUDY_BOUNDS_WGS84[3])
            & table[lon_col].between(STUDY_BOUNDS_WGS84[0], STUDY_BOUNDS_WGS84[2])
        ].copy()
        gdf = gpd.GeoDataFrame(
            table,
            geometry=gpd.points_from_xy(table[lon_col], table[lat_col]),
            crs="EPSG:4326",
        )
        return gdf.to_crs(target_crs)

    raise ValueError(f"Unsupported observation file type: {observation_path.suffix}")


def _extract_temporal_columns(table: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Extract and sort monthly groundwater observation columns.

    Args:
        table: Input groundwater observation table.

    Returns:
        List of `(column_name, year, month_number)` tuples sorted chronologically.
    """

    temporal_columns: list[tuple[str, int, int]] = []
    for column in table.columns:
        match = TIME_COLUMN_PATTERN.match(column.strip())
        if match is None:
            continue
        month_name, year_suffix = match.groups()
        year_int = int(year_suffix)
        year = 2000 + year_int if year_int <= 30 else 1900 + year_int
        temporal_columns.append((column, year, MONTH_ORDER[month_name]))
    temporal_columns.sort(key=lambda item: (item[1], item[2]))
    return temporal_columns


def _derive_groundwater_labels(table: pd.DataFrame) -> pd.DataFrame:
    """Derive groundwater-condition labels from monitoring-well time series.

    Args:
        table: Monitoring-well table containing groundwater level time columns.

    Returns:
        DataFrame with derived depth metrics and binary favorability labels.
    """

    temporal_columns = _extract_temporal_columns(table)
    if not temporal_columns:
        raise ValueError("No monthly groundwater observation columns were found in the CSV dataset.")

    max_year = max(year for _, year, _ in temporal_columns)
    recent_columns = [name for name, year, _ in temporal_columns if year >= max_year - 9]
    if len(recent_columns) < 4:
        recent_columns = [name for name, _, _ in temporal_columns]

    pre_monsoon_columns = [name for name, year, month in temporal_columns if year >= max_year - 9 and month in {1, 5}]
    post_monsoon_columns = [name for name, year, month in temporal_columns if year >= max_year - 9 and month in {8, 11}]
    if len(pre_monsoon_columns) < 2:
        pre_monsoon_columns = [name for name, _, month in temporal_columns if month in {1, 5}]
    if len(post_monsoon_columns) < 2:
        post_monsoon_columns = [name for name, _, month in temporal_columns if month in {8, 11}]

    numeric_recent = table[recent_columns].apply(pd.to_numeric, errors="coerce")
    numeric_pre = table[pre_monsoon_columns].apply(pd.to_numeric, errors="coerce") if pre_monsoon_columns else numeric_recent
    numeric_post = table[post_monsoon_columns].apply(pd.to_numeric, errors="coerce") if post_monsoon_columns else numeric_recent

    metrics = pd.DataFrame(index=table.index)
    metrics["mean_depth_m"] = numeric_recent.mean(axis=1)
    metrics["median_depth_m"] = numeric_recent.median(axis=1)
    metrics["depth_variability_m"] = numeric_recent.std(axis=1)
    metrics["pre_monsoon_depth_m"] = numeric_pre.mean(axis=1)
    metrics["post_monsoon_depth_m"] = numeric_post.mean(axis=1)
    metrics["seasonal_fluctuation_m"] = (metrics["pre_monsoon_depth_m"] - metrics["post_monsoon_depth_m"]).abs()
    metrics["observation_count"] = numeric_recent.count(axis=1)

    valid_depth = metrics["mean_depth_m"].notna()
    depth_rank = metrics.loc[valid_depth, "mean_depth_m"].rank(pct=True, method="average")
    variability_rank = metrics.loc[valid_depth, "depth_variability_m"].fillna(metrics.loc[valid_depth, "depth_variability_m"].median()).rank(pct=True, method="average")
    fluctuation_rank = metrics.loc[valid_depth, "seasonal_fluctuation_m"].fillna(metrics.loc[valid_depth, "seasonal_fluctuation_m"].median()).rank(pct=True, method="average")
    availability_score = metrics.loc[valid_depth, "observation_count"] / max(metrics.loc[valid_depth, "observation_count"].max(), 1)

    metrics["groundwater_condition_score"] = np.nan
    metrics.loc[valid_depth, "groundwater_condition_score"] = (
        0.5 * (1.0 - depth_rank)
        + 0.2 * (1.0 - variability_rank)
        + 0.2 * (1.0 - fluctuation_rank)
        + 0.1 * availability_score
    )

    score_threshold = float(metrics["groundwater_condition_score"].median(skipna=True))
    metrics["label"] = (metrics["groundwater_condition_score"] >= score_threshold).astype("int64")
    metrics["source"] = "observed_groundwater_level"
    metrics["yield_lpm"] = np.nan

    valid_depth_values = metrics.loc[valid_depth, "mean_depth_m"]
    if len(valid_depth_values) >= 5:
        quantiles = np.nanpercentile(valid_depth_values, [20, 40, 60, 80])
        depth_class = np.full(len(metrics), np.nan)
        values = metrics["mean_depth_m"].to_numpy(dtype="float64")
        valid_idx = np.where(~np.isnan(values))[0]
        for idx in valid_idx:
            value = values[idx]
            if value <= quantiles[0]:
                depth_class[idx] = 5
            elif value <= quantiles[1]:
                depth_class[idx] = 4
            elif value <= quantiles[2]:
                depth_class[idx] = 3
            elif value <= quantiles[3]:
                depth_class[idx] = 2
            else:
                depth_class[idx] = 1
        metrics["depth_class"] = depth_class
    else:
        metrics["depth_class"] = np.nan

    return metrics


def _generate_synthetic_points(
    ahp_score_raster: Path,
    classified_layers_dict: dict[str, Path | str],
    required_points: int,
) -> pd.DataFrame:
    """Generate synthetic proxy samples using AHP score quintiles.

    Args:
        ahp_score_raster: Raw AHP score raster path.
        classified_layers_dict: Mapping of feature names to classified rasters.
        required_points: Target number of synthetic samples.

    Returns:
        DataFrame of proxy training samples.
    """

    ahp_score, profile = read_raster(ahp_score_raster)
    valid_mask = ahp_score != profile.get("nodata", NODATA_FLOAT)
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid pixels available for synthetic point generation.")

    valid_scores = ahp_score[valid_mask]
    quintile_edges = np.nanpercentile(valid_scores, [20, 40, 60, 80])
    rng = np.random.default_rng(42)
    picks_per_quintile = max(10, int(np.ceil(required_points / 5)))
    rows: list[dict[str, float | int | str]] = []

    with rasterio.open(ahp_score_raster) as src:
        for quintile in tqdm(range(5), desc="Generating synthetic samples"):
            if quintile == 0:
                mask = valid_mask & (ahp_score <= quintile_edges[0])
            elif quintile == 1:
                mask = valid_mask & (ahp_score > quintile_edges[0]) & (ahp_score <= quintile_edges[1])
            elif quintile == 2:
                mask = valid_mask & (ahp_score > quintile_edges[1]) & (ahp_score <= quintile_edges[2])
            elif quintile == 3:
                mask = valid_mask & (ahp_score > quintile_edges[2]) & (ahp_score <= quintile_edges[3])
            else:
                mask = valid_mask & (ahp_score > quintile_edges[3])

            candidates = np.argwhere(mask)
            if len(candidates) == 0:
                continue
            chosen_idx = rng.choice(len(candidates), size=min(picks_per_quintile, len(candidates)), replace=False)
            chosen_pixels = candidates[chosen_idx]

            for row_idx, col_idx in chosen_pixels:
                x, y = src.xy(int(row_idx), int(col_idx))
                rows.append(
                    {
                        "yield_lpm": float(ahp_score[row_idx, col_idx] * 200.0),
                        "label": int(quintile >= 3),
                        "source": "synthetic",
                        "ahp_score": float(ahp_score[row_idx, col_idx]),
                        "x": float(x),
                        "y": float(y),
                    }
                )

    synthetic_gdf = gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy([row["x"] for row in rows], [row["y"] for row in rows]),
        crs=profile["crs"],
    )
    sampled = _sample_layers_for_points(synthetic_gdf, classified_layers_dict)
    base = pd.DataFrame(rows).drop(columns=["x", "y"])
    return pd.concat([base.reset_index(drop=True), sampled.reset_index(drop=True)], axis=1)


def build_training_dataset(
    observation_path: Path | str | None,
    classified_layers_dict: dict[str, Path | str],
    ahp_score_raster: Path | str | None = None,
    output_csv: Path | str | None = None,
) -> Path:
    """Build a training dataset from observed or proxy point data.

    Args:
        observation_path: Point shapefile or CSV path. If unavailable, proxy samples are synthesized.
        classified_layers_dict: Mapping of canonical feature names to classified rasters.
        ahp_score_raster: Optional raw AHP score raster for proxy sampling.
        output_csv: Output CSV path.

    Returns:
        Path to the saved training CSV file.
    """

    output_csv = Path(output_csv) if output_csv else Path("outputs/training_data.csv")
    first_layer = Path(classified_layers_dict[FEATURE_ORDER[0]])
    _, profile = read_raster(first_layer)
    training_frames: list[pd.DataFrame] = []

    if observation_path is not None and Path(observation_path).exists():
        observation_path = Path(observation_path)
        try:
            points = _load_point_dataset(observation_path, str(profile["crs"]))
        except OSError as exc:
            raise OSError(f"Failed to read point observations from {observation_path}") from exc

        points = points[~points.geometry.is_empty & points.geometry.notnull()].copy()
        if not points.empty:
            sampled = _sample_layers_for_points(points, classified_layers_dict).reset_index(drop=True)
            base_df = pd.DataFrame(index=sampled.index)
            point_xy = [(geom.x, geom.y) for geom in points.geometry]
            if ahp_score_raster is not None:
                base_df["ahp_score"] = sample_raster_values(Path(ahp_score_raster), point_xy)

            if "yield_lpm" in points.columns:
                base_df["yield_lpm"] = pd.to_numeric(points["yield_lpm"], errors="coerce").reset_index(drop=True)
                base_df["label"] = (base_df["yield_lpm"] > 500).astype(int)
                base_df["source"] = "observed"
            elif observation_path.suffix.lower() == ".csv":
                derived = _derive_groundwater_labels(points.drop(columns="geometry")).reset_index(drop=True)
                base_df = pd.concat([base_df.reset_index(drop=True), derived], axis=1)
                LOGGER.info(
                    "Derived groundwater-condition labels from monitoring-well time series in %s.",
                    observation_path.name,
                )
            else:
                if "ahp_score" not in base_df.columns:
                    raise ValueError("Proxy label generation requires an AHP score raster.")
                proxy_scores = base_df["ahp_score"].replace({NODATA_FLOAT: np.nan})
                threshold = float(proxy_scores.quantile(0.6))
                base_df["yield_lpm"] = proxy_scores * 200.0
                base_df["label"] = (proxy_scores >= threshold).astype(int)
                base_df["source"] = f"proxy_from_{observation_path.suffix.lower().lstrip('.')}"
                LOGGER.warning(
                    "No 'yield_lpm' column found in %s. Using AHP-derived proxy labels at observation locations.",
                    observation_path.name,
                )

            point_df = pd.concat([base_df.reset_index(drop=True), sampled], axis=1)
            point_df.replace({profile.get("nodata", NODATA_UINT8): np.nan, NODATA_FLOAT: np.nan}, inplace=True)
            if "observation_count" in point_df.columns:
                point_df = point_df[point_df["observation_count"] >= 4].copy()
            point_df.dropna(inplace=True)
            if not point_df.empty:
                training_frames.append(point_df)
                LOGGER.info("Loaded %s usable point samples from %s.", len(point_df), observation_path.name)
    else:
        LOGGER.warning("No point observation dataset was found. Training will use only synthetic proxy points.")

    observed_count = sum(len(frame) for frame in training_frames)
    observed_labels = pd.concat(training_frames, ignore_index=True)["label"] if training_frames else pd.Series(dtype="int64")
    needs_synthetic = observed_count < 50 or observed_labels.nunique() < 2
    if needs_synthetic:
        if ahp_score_raster is None:
            raise ValueError("Synthetic point generation requires an AHP score raster.")
        LOGGER.warning(
            "Only %s usable training points with %s unique classes are available. Adding synthetic proxy samples.",
            observed_count,
            observed_labels.nunique(),
        )
        synthetic_df = _generate_synthetic_points(
            Path(ahp_score_raster),
            classified_layers_dict,
            required_points=max(100, 50 - observed_count),
        )
        training_frames.append(synthetic_df)

    if not training_frames:
        raise ValueError("Failed to construct any training data.")

    training_df = pd.concat(training_frames, ignore_index=True)
    ensure_parent(output_csv)
    training_df.to_csv(output_csv, index=False)
    class_balance = training_df["label"].value_counts(normalize=True).to_dict()
    LOGGER.info("Training dataset saved to %s with %s rows.", output_csv, len(training_df))
    LOGGER.info("Class balance: %s", class_balance)
    return output_csv


def train_xgboost(
    training_csv: Path | str,
    model_path: Path | str = Path("models/gwpz_xgboost.pkl"),
) -> tuple[XGBClassifier, pd.DataFrame]:
    """Train an XGBoost classifier using the project feature order.

    Args:
        training_csv: Input training CSV path.
        model_path: Output model path.

    Returns:
        The trained model and the loaded training dataframe.
    """

    training_csv = Path(training_csv)
    model_path = Path(model_path)
    data = pd.read_csv(training_csv)
    X = data[FEATURE_ORDER]
    y = data["label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training data must contain both positive and negative classes.")
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    scale_pos_weight = float(negatives / max(positives, 1))

    auc_scores: list[float] = []
    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in tqdm(list(cv.split(X, y)), desc="Cross-validating XGBoost"):
            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=1,
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            auc_scores.append(float(roc_auc_score(y.iloc[test_idx], y_prob)))
        LOGGER.info("Mean CV ROC-AUC: %.4f +/- %.4f", np.mean(auc_scores), np.std(auc_scores))
    else:
        LOGGER.warning("Skipping cross-validation because the smallest class has fewer than 2 samples.")

    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
    )
    final_model.fit(X, y)
    ensure_parent(model_path)
    joblib.dump(final_model, model_path)

    importances = dict(zip(FEATURE_ORDER, final_model.feature_importances_.tolist()))
    LOGGER.info("Feature importances: %s", importances)
    return final_model, data


def generate_shap_plot(
    model: XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    out_path: Path | str = Path("outputs/shap_summary.png"),
) -> Path:
    """Generate and save a SHAP summary plot.

    Args:
        model: Trained XGBoost classifier.
        X: Feature dataframe.
        feature_names: Ordered feature names.
        out_path: Output PNG path.

    Returns:
        The output PNG path.
    """

    out_path = Path(out_path)
    ensure_parent(out_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    LOGGER.info("Saved SHAP summary plot to %s", out_path)
    return out_path


def predict_full_map(
    model: XGBClassifier,
    classified_layers_dict: dict[str, Path | str],
    out_path: Path | str,
    probability_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Predict groundwater potential for every valid pixel in the study area.

    Args:
        model: Trained XGBoost classifier.
        classified_layers_dict: Mapping of feature names to classified rasters.
        out_path: Output classified GW potential raster path.
        probability_path: Optional output probability raster path.

    Returns:
        Tuple of classified raster path and probability raster path.
    """

    out_path = Path(out_path)
    probability_path = Path(probability_path) if probability_path else out_path.with_name(f"{out_path.stem}_prob.tif")

    arrays: list[np.ndarray] = []
    profile = None
    valid_mask = None
    for feature in FEATURE_ORDER:
        array, profile = read_raster(Path(classified_layers_dict[feature]))
        arrays.append(array.astype("float32"))
        current_valid = array != profile.get("nodata", NODATA_UINT8)
        valid_mask = current_valid if valid_mask is None else (valid_mask & current_valid)

    assert profile is not None
    stack = np.stack(arrays, axis=-1)
    flat_stack = stack.reshape(-1, len(FEATURE_ORDER))
    flat_valid = valid_mask.reshape(-1)

    probabilities = np.full(flat_valid.shape[0], NODATA_FLOAT, dtype="float32")
    valid_pixels = flat_stack[flat_valid]
    chunk_size = 100_000
    predicted_chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(valid_pixels), chunk_size), desc="Predicting full map"):
        chunk = valid_pixels[start:start + chunk_size]
        predicted_chunks.append(model.predict_proba(chunk)[:, 1].astype("float32"))
    if predicted_chunks:
        probabilities[flat_valid] = np.concatenate(predicted_chunks)

    probability_raster = probabilities.reshape(arrays[0].shape)
    class_raster = np.full(arrays[0].shape, NODATA_UINT8, dtype="uint8")
    valid_probs = probability_raster != NODATA_FLOAT
    class_raster[valid_probs & (probability_raster < 0.2)] = 1
    class_raster[valid_probs & (probability_raster >= 0.2) & (probability_raster < 0.4)] = 2
    class_raster[valid_probs & (probability_raster >= 0.4) & (probability_raster < 0.6)] = 3
    class_raster[valid_probs & (probability_raster >= 0.6) & (probability_raster < 0.8)] = 4
    class_raster[valid_probs & (probability_raster >= 0.8)] = 5

    prob_profile = profile.copy()
    prob_profile.update(dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    class_profile = profile.copy()
    class_profile.update(dtype="uint8", nodata=NODATA_UINT8, compress="lzw")

    write_raster(probability_raster.astype("float32"), probability_path, prob_profile, nodata=NODATA_FLOAT)
    write_raster(class_raster, out_path, class_profile, nodata=NODATA_UINT8)
    LOGGER.info("Saved ML probability raster to %s", probability_path)
    LOGGER.info("Saved ML classified raster to %s", out_path)
    return out_path, probability_path
