# Groundwater Favorability and Depth Zoning

Groundwater Favorability and Depth Zoning is an end-to-end geospatial workflow for the Vaigai Basin in Tamil Nadu. It preprocesses multi-source geospatial layers, derives hydro-geomorphic indicators, generates an AHP suitability map, trains an XGBoost model from groundwater-condition labels, serves predictions through FastAPI, and includes a polished Streamlit dashboard for exploration.

## Use Case

This project is designed for hydrogeologists, watershed planners, irrigation teams, and researchers who need to:

- standardize heterogeneous geospatial layers to a common 30 m grid
- derive indicators such as TWI, drainage density, lineament density, and soil moisture
- compare AHP zoning against ML zoning
- inspect prediction coverage
- query groundwater favorability at specific coordinates
- present results through a professional interactive UI

## Folder Structure

```text
groundwater_tn/
|-- .streamlit/
|-- data/
|   |-- raw/
|   |-- classified/
|   `-- standardized/
|-- notebooks/
|   |-- 01_data_download.ipynb
|   |-- 02_preprocessing.ipynb
|   |-- 03_ahp_analysis.ipynb
|   |-- 04_xgboost_training.ipynb
|   `-- 05_validation.ipynb
|-- models/
|-- outputs/
|-- src/
|-- app.py
|-- pipeline.py
|-- requirements.txt
|-- streamlit_app.py
`-- README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Requirements

### Core inputs for the current workflow

Place these files in `data/raw/`:

- `slope.tif`
- `dem.tif`
- `smap_sm.h5`
- `sentinel2_B8.tif`
- `rainfall.tif`
- `ndvi.tif`
- `ndvi_1km.tif`
- `lst_1km.tif`
- `geomorphology.tif` or `geomorphology.shp`
- `soil.tif` or `soil.shp`

### Preferred additional inputs

- `geology.shp` or `geology.tif`
- `geomorphology.shp`
- `soil.shp`

### Current fallback inputs supported

- `1_India_GWLs_2000_2024_wells_within_India.csv`

If geology is missing, the pipeline creates a neutral geology placeholder automatically. If direct point labels are not available in a ready-to-use form, the workflow can derive groundwater-condition labels from the groundwater-level time series and augment them with synthetic samples when needed.

Expected study configuration:

- Target CRS: `EPSG:32644`
- Target resolution: `30 m`
- Study bounds: `[77.0, 9.5, 78.5, 11.0]`

## Run The Pipeline

```bash
python pipeline.py --data_dir data/raw --output_dir outputs
```

The pipeline runs:

1. preprocessing and standardization
2. thematic layer preparation
3. AHP weighted overlay
4. training-table generation from point observations
5. XGBoost training and validation
6. pixel-wise ML prediction over the study area
7. visualization export and API-ready assets

## Start The FastAPI Backend

```bash
python -m uvicorn app:app --reload --port 8000
```

Useful endpoints:

- `/docs`
- `/health`
- `/stats`
- `/predict?lat=10.2&lon=78.1`
- `/coverage?lat=10.2&lon=78.1`
- `/coverage-summary`
- `/coverage-samples?count=20`
- `/coverage-map`
- `/coverage-breakdown`
- `/map`

## Start The Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The Streamlit app includes:

- a professional dashboard landing view
- coordinate-based groundwater query
- ML and AHP map explorer
- prediction coverage viewer
- area comparison analytics
- validation and SHAP insight panels
- sample valid query coordinates

## Expected Outputs

After a successful run, expect at least:

- `data/standardized/*.tif`
- `data/classified/*.tif`
- `outputs/gwpz_ahp.tif`
- `outputs/gwpz_ahp_score.tif`
- `outputs/gwpz_ahp_rgb.png`
- `outputs/gwpz_xgboost.tif`
- `outputs/gwpz_xgboost_prob.tif`
- `outputs/gwpz_xgboost_rgb.png`
- `outputs/shap_summary.png`
- `outputs/gwpz_statistics_ml.png`
- `outputs/gwpz_statistics_ahp.png`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/training_data.csv`
- `outputs/train_data.csv`
- `outputs/test_data.csv`
- `models/gwpz_xgboost.pkl`

## Notes

- The current active modeling stack uses geology, geomorphology, lineament density, soil, slope, rainfall, drainage density, TWI, NDVI, and soil moisture.
- The workflow can build groundwater-condition labels from the groundwater-level point dataset available in `data/raw/`.
- If fewer than 50 usable point samples are available, synthetic proxy points are added automatically.
- Predictions are only returned where all required classified layers are available.
- The hydrologic derivatives use lightweight numerical approximations so the workflow stays runnable on CPU with the requested dependency stack.
