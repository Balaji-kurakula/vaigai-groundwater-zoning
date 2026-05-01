"""Professional Streamlit frontend for the Vaigai groundwater project."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import streamlit as st
from folium.plugins import Fullscreen
from pyproj import Transformer
from streamlit_folium import st_folium

from src import CLASS_COLORS, CLASS_LABELS, FEATURE_ORDER, RECOMMENDATIONS
from src.config import CLASSIFIED_DIR, MODELS_DIR, OUTPUTS_DIR
from src.utils import array_to_png_data_uri, compute_area_stats, png_to_data_uri, read_raster, sample_raster_values

MODEL_PATH = MODELS_DIR / "gwpz_xgboost.pkl"
AHP_MAP_PATH = OUTPUTS_DIR / "gwpz_ahp.tif"
ML_MAP_PATH = OUTPUTS_DIR / "gwpz_xgboost.tif"
ML_OVERLAY_PATH = OUTPUTS_DIR / "gwpz_xgboost_rgb.png"
AHP_OVERLAY_PATH = OUTPUTS_DIR / "gwpz_ahp_rgb.png"
VALIDATION_PATH = OUTPUTS_DIR / "validation_metrics.json"
SHAP_PATH = OUTPUTS_DIR / "shap_summary.png"
CONFUSION_PATH = OUTPUTS_DIR / "confusion_matrix.png"
ROC_PATH = OUTPUTS_DIR / "roc_curve.png"


st.set_page_config(
    page_title="Geospatial Project Dashboard",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    """Load the trained model if available."""

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        expected_features = len(FEATURE_ORDER)
        actual_features = getattr(model, "n_features_in_", expected_features)
        if actual_features == expected_features:
            return model
    return None


@st.cache_data
def classified_layer_paths() -> dict[str, Path]:
    """Return classified layer paths."""

    return {feature: CLASSIFIED_DIR / f"{feature}.tif" for feature in FEATURE_ORDER}


@st.cache_data
def coverage_mask() -> tuple[np.ndarray, dict]:
    """Compute the common valid coverage mask across all classified rasters."""

    mask = None
    profile = None
    for feature, path in classified_layer_paths().items():
        array, profile = read_raster(path)
        current_valid = array != profile.get("nodata", 255)
        mask = current_valid if mask is None else (mask & current_valid)
    assert profile is not None
    return mask, profile


@st.cache_data
def coverage_summary() -> dict[str, float | int]:
    """Summarize valid prediction coverage."""

    mask, profile = coverage_mask()
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
    }


@st.cache_data
def sample_valid_locations(count: int = 24) -> list[dict[str, float | int]]:
    """Return representative valid query locations."""

    mask, profile = coverage_mask()
    valid_indices = np.argwhere(mask)
    if len(valid_indices) == 0:
        return []

    count = max(1, min(count, len(valid_indices), 200))
    positions = np.linspace(0, len(valid_indices) - 1, num=count, dtype=int)
    transformer = Transformer.from_crs(profile["crs"], "EPSG:4326", always_xy=True)
    samples: list[dict[str, float | int]] = []
    with rasterio.open(next(iter(classified_layer_paths().values()))) as src:
        for pos in positions:
            row_idx, col_idx = valid_indices[pos]
            x, y = src.xy(int(row_idx), int(col_idx))
            lon, lat = transformer.transform(x, y)
            samples.append(
                {
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lon), 6),
                    "row": int(row_idx),
                    "col": int(col_idx),
                }
            )
    return samples


@st.cache_data
def area_stats_payload() -> pd.DataFrame:
    """Build comparison table for AHP and ML class areas."""

    ahp_stats = compute_area_stats(AHP_MAP_PATH) if AHP_MAP_PATH.exists() else {}
    ml_stats = compute_area_stats(ML_MAP_PATH) if ML_MAP_PATH.exists() else {}
    rows = []
    for klass in range(1, 6):
        rows.append(
            {
                "class": klass,
                "label": CLASS_LABELS[klass],
                "ahp_area_km2": ahp_stats.get(klass, 0.0),
                "ml_area_km2": ml_stats.get(klass, 0.0),
                "color": CLASS_COLORS[klass],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def validation_metrics() -> dict[str, object]:
    """Load validation metrics if available."""

    if VALIDATION_PATH.exists():
        return json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    return {}


def predict_point(lat: float, lon: float) -> dict[str, object]:
    """Predict groundwater class for a single point."""

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    x, y = transformer.transform(lon, lat)
    invalid_layers: list[str] = []
    features: dict[str, float] = {}
    for feature, path in classified_layer_paths().items():
        value = float(sample_raster_values(path, [(x, y)])[0])
        if value == 255:
            invalid_layers.append(feature)
        else:
            features[feature] = value

    if invalid_layers:
        return {
            "can_predict": False,
            "invalid_layers": invalid_layers,
            "latitude": lat,
            "longitude": lon,
        }

    ahp_class = None
    if AHP_MAP_PATH.exists():
        ahp_value = int(sample_raster_values(AHP_MAP_PATH, [(x, y)])[0])
        ahp_class = None if ahp_value == 255 else ahp_value

    model = load_model()
    if model is not None:
        ordered = np.array([[features[name] for name in FEATURE_ORDER]], dtype="float32")
        confidence = float(model.predict_proba(ordered)[0, 1])
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
        source = "ML prediction"
    elif ahp_class is not None:
        gwpz_class = ahp_class
        confidence = gwpz_class / 5.0
        source = "AHP fallback"
    else:
        return {
            "can_predict": False,
            "invalid_layers": ["model_or_ahp_unavailable"],
            "latitude": lat,
            "longitude": lon,
        }

    return {
        "can_predict": True,
        "latitude": lat,
        "longitude": lon,
        "gwpz_class": gwpz_class,
        "gwpz_label": CLASS_LABELS[gwpz_class],
        "confidence": confidence,
        "color": CLASS_COLORS[gwpz_class],
        "ahp_class": ahp_class,
        "recommendation": RECOMMENDATIONS[gwpz_class],
        "source": source,
        "features": features,
    }


def make_overlay_map(mode: str = "ml") -> folium.Map:
    """Create folium map for ML, AHP, or coverage overlay."""

    fmap = folium.Map(location=[10.0, 77.8], zoom_start=9, tiles="CartoDB positron")
    Fullscreen().add_to(fmap)
    fmap.add_child(folium.LatLngPopup())

    if mode == "coverage":
        mask, _ = coverage_mask()
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask] = (26, 152, 80, 180)
        rgba[~mask] = (215, 48, 39, 70)
        overlay_uri = array_to_png_data_uri(rgba)
        overlay_name = "Prediction Coverage"
    else:
        overlay_path = ML_OVERLAY_PATH if mode == "ml" and ML_OVERLAY_PATH.exists() else AHP_OVERLAY_PATH
        overlay_uri = png_to_data_uri(overlay_path)
        overlay_name = "ML GWPZ" if mode == "ml" else "AHP GWPZ"

    folium.raster_layers.ImageOverlay(
        image=overlay_uri,
        bounds=[[9.5, 77.0], [11.0, 78.5]],
        opacity=0.72,
        name=overlay_name,
        interactive=True,
        cross_origin=False,
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def inject_css() -> None:
    """Inject custom styling for a polished Streamlit UI."""

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(33, 150, 243, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(46, 204, 113, 0.14), transparent 24%),
                linear-gradient(180deg, #f5f8fc 0%, #eef4fb 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .hero {
            padding: 1.9rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #073b4c 0%, #0b6e73 46%, #1b9aaa 100%);
            color: #f8fbff;
            box-shadow: 0 22px 45px rgba(7, 59, 76, 0.20);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.3rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0;
            max-width: 820px;
            opacity: 0.92;
            font-size: 1rem;
            line-height: 1.65;
        }
        .metric-card {
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 15px 35px rgba(15, 23, 42, 0.07);
        }
        .metric-card .label {
            color: #486581;
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }
        .metric-card .value {
            color: #0b172a;
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }
        .section-chip {
            display: inline-block;
            background: #dff6f1;
            color: #0f766e;
            font-weight: 600;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .result-good {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(26,152,80,0.18), rgba(145,207,96,0.18));
            border: 1px solid rgba(26,152,80,0.25);
        }
        .result-bad {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(215,48,39,0.12), rgba(252,141,89,0.16));
            border: 1px solid rgba(215,48,39,0.22);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render top hero panel."""

    st.markdown(
        """
        <div class="hero">
            <div class="section-chip">Geospatial Project</div>
            <h1>Groundwater Favorability and Depth Zoning Dashboard</h1>
            <p>
                Explore Vaigai Basin groundwater favorability, compare AHP and machine-learning
                zoning, inspect prediction coverage, and query basin coordinates through a clean
                professional interface built on the latest generated project outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics() -> None:
    """Render headline dashboard metrics."""

    summary = coverage_summary()
    metrics = validation_metrics()
    area_df = area_stats_payload()

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Valid Coverage", f"{summary['valid_percent']:.2f}%"),
        ("Predictable Area", f"{summary['valid_area_km2']:.0f} km^2"),
        ("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}" if metrics else "NA"),
        ("Model Classes", f"{len(area_df)} classes"),
    ]
    for col, (label, value) in zip((col1, col2, col3, col4), cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_performance_strip() -> None:
    """Render a dedicated model performance strip near the top of the dashboard."""

    metrics = validation_metrics()
    report = metrics.get("classification_report", {}) if metrics else {}
    accuracy = report.get("accuracy", None)
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", None) if report else None
    roc_auc = metrics.get("roc_auc", None) if metrics else None

    st.markdown('<div class="section-chip">Model Performance</div>', unsafe_allow_html=True)
    st.subheader("Latest Validation Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    perf_cards = [
        ("Accuracy", f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else "NA"),
        ("ROC-AUC", f"{roc_auc:.3f}" if isinstance(roc_auc, (int, float)) else "NA"),
        ("Weighted F1", f"{weighted_f1:.3f}" if isinstance(weighted_f1, (int, float)) else "NA"),
        ("Status", "Latest run"),
    ]
    for col, (label, value) in zip((c1, c2, c3, c4), perf_cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if CONFUSION_PATH.exists():
        with st.expander("Open confusion matrix preview", expanded=False):
            st.image(str(CONFUSION_PATH), use_container_width=True, caption="Confusion Matrix")


def render_sidebar() -> tuple[float, float]:
    """Render sidebar controls and return query coordinates."""

    st.sidebar.markdown("## Query Controls")
    st.sidebar.caption("Use sample valid points or enter your own coordinate pair inside the Vaigai Basin.")

    sample_points = sample_valid_locations(12)
    sample_options = {
        f"{item['latitude']:.4f}, {item['longitude']:.4f}": (item["latitude"], item["longitude"])
        for item in sample_points
    }
    selected = st.sidebar.selectbox("Sample valid locations", ["Custom"] + list(sample_options.keys())) if sample_options else "Custom"

    if selected != "Custom":
        default_lat, default_lon = sample_options[selected]
    else:
        default_lat, default_lon = 10.2, 78.1

    lat = st.sidebar.number_input("Latitude", min_value=9.5, max_value=11.0, value=float(default_lat), step=0.0001, format="%.4f")
    lon = st.sidebar.number_input("Longitude", min_value=77.0, max_value=78.5, value=float(default_lon), step=0.0001, format="%.4f")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Launch")
    st.sidebar.code("streamlit run streamlit_app.py", language="bash")
    return float(lat), float(lon)


def render_query_panel(lat: float, lon: float) -> None:
    """Render point query results."""

    st.markdown('<div class="section-chip">Point Query</div>', unsafe_allow_html=True)
    st.subheader("Groundwater Prediction at a Selected Coordinate")
    result = predict_point(lat, lon)

    if result["can_predict"]:
        color = result["color"]
        st.markdown(
            f"""
            <div class="result-good">
                <h3 style="margin:0; color:{color};">{result['gwpz_label']} ({result['gwpz_class']})</h3>
                <p style="margin:0.4rem 0 0 0;"><strong>Source:</strong> {result['source']} | <strong>Confidence:</strong> {result['confidence']:.3f}</p>
                <p style="margin:0.7rem 0 0 0;">{result['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        feature_df = pd.DataFrame([{"feature": key, "value": value} for key, value in result["features"].items()])
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            f"""
            <div class="result-bad">
                <h3 style="margin:0;">Prediction unavailable at this location</h3>
                <p style="margin:0.55rem 0 0 0;">Missing coverage in: {", ".join(result['invalid_layers'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_maps() -> None:
    """Render map explorer tabs."""

    st.markdown('<div class="section-chip">Explorer</div>', unsafe_allow_html=True)
    st.subheader("Map Explorer")
    map_tabs = st.tabs(["ML Overlay", "AHP Overlay", "Coverage"])
    with map_tabs[0]:
        st_folium(make_overlay_map("ml"), width=None, height=560, returned_objects=[])
    with map_tabs[1]:
        st_folium(make_overlay_map("ahp"), width=None, height=560, returned_objects=[])
    with map_tabs[2]:
        st.info("Green areas are valid for prediction. Red translucent areas are missing one or more required layers.")
        st_folium(make_overlay_map("coverage"), width=None, height=560, returned_objects=["last_clicked"])


def render_analytics() -> None:
    """Render coverage and area analytics."""

    st.markdown('<div class="section-chip">Analytics</div>', unsafe_allow_html=True)
    st.subheader("Coverage, Area, and Validation Analytics")
    summary = coverage_summary()
    area_df = area_stats_payload()
    metrics = validation_metrics()

    col_left, col_right = st.columns([1.1, 1.3])
    with col_left:
        coverage_df = pd.DataFrame(
            [
                {"metric": "Valid coverage (%)", "value": round(summary["valid_percent"], 2)},
                {"metric": "Valid area (km^2)", "value": round(summary["valid_area_km2"], 2)},
                {"metric": "Invalid area (km^2)", "value": round(summary["invalid_area_km2"], 2)},
                {"metric": "Valid pixels", "value": summary["valid_pixels"]},
                {"metric": "Total pixels", "value": summary["total_pixels"]},
            ]
        )
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)
        samples = sample_valid_locations(10)
        if samples:
            st.markdown("**Sample valid query coordinates**")
            st.dataframe(pd.DataFrame(samples), use_container_width=True, hide_index=True)

    with col_right:
        fig = px.bar(
            area_df.melt(
                id_vars=["class", "label", "color"],
                value_vars=["ahp_area_km2", "ml_area_km2"],
                var_name="method",
                value_name="area_km2",
            ),
            x="label",
            y="area_km2",
            color="method",
            barmode="group",
            color_discrete_map={"ahp_area_km2": "#0b6e73", "ml_area_km2": "#f4a261"},
            title="Groundwater Class Area Comparison",
        )
        fig.update_layout(height=420, xaxis_title="", yaxis_title="Area (km^2)", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    if metrics:
        roc_value = metrics.get("roc_auc", None)
        report = metrics.get("classification_report", {})
        st.markdown("### Validation Snapshot")
        c1, c2 = st.columns([1.1, 1.1])
        with c1:
            st.metric("ROC-AUC", f"{roc_value:.3f}" if roc_value is not None else "NA")
            if ROC_PATH.exists():
                st.image(str(ROC_PATH), use_container_width=True, caption="ROC Curve")
        with c2:
            if CONFUSION_PATH.exists():
                st.image(str(CONFUSION_PATH), use_container_width=True, caption="Confusion Matrix")
            if report:
                report_rows = []
                for key in ("0", "1", "macro avg", "weighted avg"):
                    if key in report and isinstance(report[key], dict):
                        row = {"label": key}
                        row.update({metric: round(value, 3) for metric, value in report[key].items()})
                        report_rows.append(row)
                if "accuracy" in report:
                    report_rows.append({"label": "accuracy", "score": round(float(report["accuracy"]), 3)})
                if report_rows:
                    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)


def render_model_insights() -> None:
    """Render SHAP and artifact previews."""

    st.markdown('<div class="section-chip">Model Insight</div>', unsafe_allow_html=True)
    st.subheader("Model Interpretation and Artifact Gallery")
    col1, col2 = st.columns([1.25, 1])
    with col1:
        if SHAP_PATH.exists():
            st.image(str(SHAP_PATH), use_container_width=True, caption="SHAP Summary")
    with col2:
        st.markdown("### Artifact Preview")
        preview_files = [
            ("ML Overlay", ML_OVERLAY_PATH),
            ("AHP Overlay", AHP_OVERLAY_PATH),
            ("AHP Statistics", OUTPUTS_DIR / "gwpz_statistics_ahp.png"),
            ("ML Statistics", OUTPUTS_DIR / "gwpz_statistics_ml.png"),
        ]
        for label, path in preview_files:
            if path.exists():
                with st.expander(label, expanded=False):
                    st.image(str(path), use_container_width=True)


def main() -> None:
    """Run the Streamlit dashboard."""

    inject_css()
    lat, lon = render_sidebar()
    render_hero()
    st.write("")
    render_metrics()
    st.write("")
    render_performance_strip()
    st.write("")

    tabs = st.tabs(["Dashboard", "Map Explorer", "Analytics", "Model Insight"])
    with tabs[0]:
        render_query_panel(lat, lon)
    with tabs[1]:
        render_maps()
    with tabs[2]:
        render_analytics()
    with tabs[3]:
        render_model_insights()


if __name__ == "__main__":
    main()
