"""Visualization helpers for HydroSight TN outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import CLASS_COLORS, NODATA_UINT8
from .utils import compute_area_stats, ensure_parent, read_raster

LOGGER = logging.getLogger(__name__)


def _hex_to_rgba(hex_color: str, alpha: int = 200) -> tuple[int, int, int, int]:
    """Convert a hex color string to an RGBA tuple."""

    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def raster_to_rgb_png(
    raster_path: Path | str,
    color_map: dict[int, str],
    out_path: Path | str,
) -> Path:
    """Render a classified raster into an RGBA PNG for web overlays.

    Args:
        raster_path: Input classified raster path.
        color_map: Class-to-hex color mapping.
        out_path: Output PNG path.

    Returns:
        The output PNG path.
    """

    raster_path = Path(raster_path)
    out_path = Path(out_path)
    array, _ = read_raster(raster_path)
    rgba = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
    for value, hex_color in color_map.items():
        rgba[array == value] = _hex_to_rgba(hex_color)
    rgba[array == NODATA_UINT8] = (0, 0, 0, 0)

    ensure_parent(out_path)
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    LOGGER.info("Saved overlay PNG to %s", out_path)
    return out_path


def plot_area_statistics(raster_path: Path | str, out_path: Path | str) -> Path:
    """Plot area statistics for a classified GWPZ raster.

    Args:
        raster_path: Input classified raster path.
        out_path: Output PNG path.

    Returns:
        The output chart path.
    """

    raster_path = Path(raster_path)
    out_path = Path(out_path)
    stats = compute_area_stats(raster_path)
    classes = list(stats.keys())
    areas = [stats[key] for key in classes]
    labels = [f"Class {key}" for key in classes]
    colors = [CLASS_COLORS[key] for key in classes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, areas, color=colors)
    axes[0].set_ylabel("Area (km²)")
    axes[0].set_title("Groundwater Potential Area")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].pie(areas, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Area Share by Class")

    plt.tight_layout()
    ensure_parent(out_path)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved area statistics chart to %s", out_path)
    return out_path
