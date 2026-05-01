"""Shared project configuration."""

from __future__ import annotations

from pathlib import Path

TARGET_CRS = "EPSG:32644"
TARGET_RESOLUTION = 30.0
STUDY_BOUNDS_WGS84 = (77.0, 9.5, 78.5, 11.0)
NODATA_FLOAT = -9999.0
NODATA_UINT8 = 255

FEATURE_ORDER = [
    "geology",
    "geomorphology",
    "lineament_density",
    "soil",
    "slope",
    "rainfall",
    "drainage_density",
    "twi",
    "ndvi",
    "soil_moisture",
]

AHP_FACTORS = [
    "geology",
    "geomorphology",
    "lineament_density",
    "soil",
    "slope",
    "rainfall",
    "drainage_density",
    "twi",
    "ndvi",
]

AHP_PAIRWISE_MATRIX = [
    [1, 2, 3, 4, 5, 3, 4, 4, 6],
    [1 / 2, 1, 2, 3, 4, 2, 3, 3, 5],
    [1 / 3, 1 / 2, 1, 2, 3, 2, 2, 2, 4],
    [1 / 4, 1 / 3, 1 / 2, 1, 2, 2, 2, 2, 3],
    [1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 1, 1, 1, 2],
    [1 / 3, 1 / 2, 1 / 2, 1 / 2, 1, 1, 1, 1, 2],
    [1 / 4, 1 / 3, 1 / 2, 1 / 2, 1, 1, 1, 1, 2],
    [1 / 4, 1 / 3, 1 / 2, 1 / 2, 1, 1, 1, 1, 2],
    [1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1],
]

CLASS_LABELS = {
    1: "Very Poor",
    2: "Poor",
    3: "Moderate",
    4: "Good",
    5: "Very Good",
}

CLASS_COLORS = {
    1: "#d73027",
    2: "#fc8d59",
    3: "#fee08b",
    4: "#91cf60",
    5: "#1a9850",
}

RECOMMENDATIONS = {
    1: "Very low potential. Avoid deep borewells. Consider rainwater harvesting.",
    2: "Low potential. Shallow dug wells with modest yield expected.",
    3: "Moderate potential. Suitable for domestic use; check seasonal variability.",
    4: "Good potential. Suitable for irrigation borewells (150-200m depth).",
    5: "Very high potential. Ideal recharge zone. Prioritize check dams & percolation ponds.",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STANDARDIZED_DIR = DATA_DIR / "standardized"
CLASSIFIED_DIR = DATA_DIR / "classified"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
