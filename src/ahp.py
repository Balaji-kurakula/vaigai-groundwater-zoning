"""AHP weighting and overlay routines."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import AHP_FACTORS, AHP_PAIRWISE_MATRIX, NODATA_FLOAT, NODATA_UINT8
from .utils import compute_area_stats, read_raster, write_raster

LOGGER = logging.getLogger(__name__)


def compute_ahp_weights(pairwise_matrix: list[list[float]] | np.ndarray) -> dict[str, float]:
    """Compute AHP weights and consistency statistics.

    Args:
        pairwise_matrix: Pairwise comparison matrix for the configured AHP factors.

    Returns:
        A mapping of factor names to normalized weights.
    """

    matrix = np.asarray(pairwise_matrix, dtype="float64")
    expected_size = len(AHP_FACTORS)
    if matrix.shape != (expected_size, expected_size):
        raise ValueError(f"AHP pairwise matrix must be {expected_size}x{expected_size}.")

    column_sums = matrix.sum(axis=0)
    normalized = matrix / column_sums
    weights = normalized.mean(axis=1)

    weighted_sum = matrix @ weights
    lambda_max = float(np.mean(weighted_sum / weights))
    ci = (lambda_max - matrix.shape[0]) / (matrix.shape[0] - 1)
    random_index_lookup = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }
    ri = random_index_lookup.get(matrix.shape[0], 1.49)
    cr = ci / ri if ri else 0.0
    if cr >= 0.10:
        raise ValueError(f"AHP consistency ratio is {cr:.4f}, which exceeds the 0.10 threshold.")

    return {factor: float(weight) for factor, weight in zip(AHP_FACTORS, weights)}


def weighted_overlay(
    classified_layers_dict: dict[str, Path | str],
    weights_dict: dict[str, float],
    out_path: Path | str,
) -> tuple[Path, Path, dict[int, float]]:
    """Generate a weighted AHP groundwater potential map.

    Args:
        classified_layers_dict: Mapping of factor names to classified raster paths.
        weights_dict: Factor weights from AHP.
        out_path: Output classified AHP raster path.

    Returns:
        Tuple of classified raster path, raw score raster path, and area statistics.
    """

    out_path = Path(out_path)
    score_path = out_path.with_name(f"{out_path.stem}_score.tif")

    arrays: list[np.ndarray] = []
    profile = None
    valid_mask = None
    for factor in AHP_FACTORS:
        if factor not in classified_layers_dict:
            raise KeyError(f"Missing classified layer for AHP factor '{factor}'")
        array, profile = read_raster(Path(classified_layers_dict[factor]))
        arrays.append(array.astype("float32"))
        current_valid = array != profile.get("nodata", NODATA_UINT8)
        valid_mask = current_valid if valid_mask is None else (valid_mask & current_valid)

    assert profile is not None
    weighted_score = np.full(arrays[0].shape, NODATA_FLOAT, dtype="float32")
    score_stack = np.zeros(arrays[0].shape, dtype="float32")
    for factor, array in zip(AHP_FACTORS, arrays):
        score_stack += np.where(valid_mask, array, 0.0) * float(weights_dict[factor])
    weighted_score[valid_mask] = score_stack[valid_mask]

    gwpz = np.full(arrays[0].shape, NODATA_UINT8, dtype="uint8")
    gwpz[valid_mask & (weighted_score >= 1.0) & (weighted_score < 1.8)] = 1
    gwpz[valid_mask & (weighted_score >= 1.8) & (weighted_score < 2.6)] = 2
    gwpz[valid_mask & (weighted_score >= 2.6) & (weighted_score < 3.4)] = 3
    gwpz[valid_mask & (weighted_score >= 3.4) & (weighted_score < 4.2)] = 4
    gwpz[valid_mask & (weighted_score >= 4.2)] = 5

    score_profile = profile.copy()
    score_profile.update(dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    class_profile = profile.copy()
    class_profile.update(dtype="uint8", nodata=NODATA_UINT8, compress="lzw")

    write_raster(weighted_score, score_path, score_profile, nodata=NODATA_FLOAT)
    write_raster(gwpz, out_path, class_profile, nodata=NODATA_UINT8)

    area_stats = compute_area_stats(out_path)
    for klass, area in area_stats.items():
        LOGGER.info("AHP class %s area: %.2f km2", klass, area)

    return out_path, score_path, area_stats


DEFAULT_AHP_WEIGHTS = compute_ahp_weights(AHP_PAIRWISE_MATRIX)
