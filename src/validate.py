"""Validation utilities for groundwater potential zoning outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from .config import FEATURE_ORDER
from .utils import ensure_parent

LOGGER = logging.getLogger(__name__)


def validate_model(
    model: XGBClassifier,
    test_csv: Path | str,
    output_dir: Path | str = Path("outputs"),
) -> dict[str, object]:
    """Validate the trained model and generate diagnostic plots.

    Args:
        model: Trained XGBoost classifier.
        test_csv: Holdout test CSV path.
        output_dir: Directory for validation plots.

    Returns:
        Dictionary of validation metrics.
    """

    test_csv = Path(test_csv)
    output_dir = Path(output_dir)
    ensure_parent(output_dir / "placeholder.txt")

    data = pd.read_csv(test_csv)
    X_test = data[FEATURE_ORDER]
    y_true = data["label"].astype(int)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else float("nan")
    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=matrix).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    if y_true.nunique() > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", color="#1f78b4")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "ROC curve unavailable\n(single-class holdout)", ha="center", va="center")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close(fig)

    results: dict[str, object] = {
        "roc_auc": auc,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    if "yield_lpm" in data.columns and "ahp_score" in data.columns and data["yield_lpm"].notna().sum() > 2:
        ahp_corr = spearmanr(data["ahp_score"], data["yield_lpm"], nan_policy="omit")
        results["spearman_ahp_vs_yield"] = {
            "correlation": float(ahp_corr.correlation) if ahp_corr.correlation is not None else np.nan,
            "pvalue": float(ahp_corr.pvalue) if ahp_corr.pvalue is not None else np.nan,
        }
    if "yield_lpm" in data.columns and data["yield_lpm"].notna().sum() > 2:
        ml_corr = spearmanr(y_prob, data["yield_lpm"], nan_policy="omit")
        results["spearman_ml_vs_yield"] = {
            "correlation": float(ml_corr.correlation) if ml_corr.correlation is not None else np.nan,
            "pvalue": float(ml_corr.pvalue) if ml_corr.pvalue is not None else np.nan,
        }
    if "mean_depth_m" in data.columns and "ahp_score" in data.columns and data["mean_depth_m"].notna().sum() > 2:
        ahp_depth_corr = spearmanr(data["ahp_score"], data["mean_depth_m"], nan_policy="omit")
        results["spearman_ahp_vs_mean_depth_m"] = {
            "correlation": float(ahp_depth_corr.correlation) if ahp_depth_corr.correlation is not None else np.nan,
            "pvalue": float(ahp_depth_corr.pvalue) if ahp_depth_corr.pvalue is not None else np.nan,
        }
    if "mean_depth_m" in data.columns and data["mean_depth_m"].notna().sum() > 2:
        ml_depth_corr = spearmanr(y_prob, data["mean_depth_m"], nan_policy="omit")
        results["spearman_ml_vs_mean_depth_m"] = {
            "correlation": float(ml_depth_corr.correlation) if ml_depth_corr.correlation is not None else np.nan,
            "pvalue": float(ml_depth_corr.pvalue) if ml_depth_corr.pvalue is not None else np.nan,
        }

    metrics_path = output_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    LOGGER.info("Validation metrics saved to %s", metrics_path)
    LOGGER.info("ROC-AUC: %.4f", auc)
    LOGGER.info("Classification report:\n%s", classification_report(y_true, y_pred))
    return results
