# T2D Onset Year Classifier (synthetic)

End-to-end pipeline that predicts the **onset year of Type-2 Diabetes (T2D)** on synthetic patient-year data.  
Includes data generation (Parts 1–3), feature engineering with temporal rolling (Part 4), model training with an explicit decision threshold (Part 5), and a **FastAPI** service for inference. Both training and serving can run in **Docker**. An EDA notebook provides sanity checks, ROC/PR plots, and slice analysis.

## Repo Layout

```
.
├─ app/                    # FastAPI service (main.py)
├─ src/                    # Data gen + feature eng + training (parts 1–5)
├─ notebooks/              # EDA / evaluation notebook(s)
├─ data/                   # Generated data & artifacts (gitignored)
├─ requirements-dev.txt
├─ requirements-training.txt
├─ requirements-serving.txt
├─ Dockerfile.training
├─ Dockerfile.serving
└─ docker-compose.yml
```

## Quick Start — Local (venv)

```bash
# 1) Create & activate venv
python -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements-dev.txt
pip install -r requirements-training.txt
pip install -r requirements-serving.txt
```

Run the pipeline (adjust names if yours differ):

```bash
# Parts 1–4: data & features
python -m src.part1_skeleton
python -m src.part2_signals
python -m src.part3_labels          # (or: src.part3_labels_min)
python -m src.part4_features

# Part 5: train HGB model, write artifacts under data/clinical_synth_v1/artifacts_v1/
python -m src.part5_train_hgb
```

Serve locally (without Docker):

```bash
export MODEL_ART_PATH="data/clinical_synth_v1/artifacts_v1/t2d_hgb_model_v1.joblib"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000/docs  (Swagger)
```

## Docker — Build & Run

### Serving only
```bash
# from repo root
docker compose build --no-cache serve
docker compose up -d serve

# Health & metadata
curl -s http://localhost:8000/health
curl -s http://localhost:8000/metadata
```

### Training (optional, in container)
```bash
docker compose build train
docker compose run --rm train
# Artifacts will be written under ./data/... (mounted into the container)
```

> If you retrain and want the serving image to pick up new artifacts, just run:
```bash
docker compose restart serve
```

## API

| Method | Path        | Description                  |
|-------:|-------------|------------------------------|
| GET    | `/health`   | Liveness check               |
| GET    | `/metadata` | Model metadata (features, τ) |
| POST   | `/predict`  | Batch predict (JSON)         |
| GET    | `/docs`     | Swagger UI                   |

**Example request body** (paste into `/docs` → `POST /predict` → *Try it out*):
```json
{
  "instances": [
    {
      "sex": "F", "region": "Center", "ses_quintile": 3, "cci": 1,
      "bmi_lag1": 30.1, "bmi_roll3_mean": 29.6, "bmi_delta_prev": 0.2,
      "sbp_lag1": 126, "sbp_roll3_mean": 125, "sbp_delta_prev": 1.0,
      "dbp_lag1": 78,  "dbp_roll3_mean": 77,  "dbp_delta_prev": 0.5,
      "ldl_lag1": 120, "ldl_roll3_mean": 118, "ldl_delta_prev": -2,
      "hdl_lag1": 52,  "hdl_roll3_mean": 51,  "hdl_delta_prev": 0.2,
      "triglycerides_lag1": 140, "triglycerides_roll3_mean": 138, "triglycerides_delta_prev": 3,
      "a1c_lag1": 7.8, "a1c_roll3_mean": 7.2, "a1c_delta_prev": 0.4,
      "egfr_lag1": 90, "egfr_roll3_mean": 88, "egfr_delta_prev": -1,
      "alt_lag1": 22,  "alt_roll3_mean": 23, "alt_delta_prev": 0.3,
      "ast_lag1": 24,  "ast_roll3_mean": 24, "ast_delta_prev": 0.1,
      "visits_lag1": 4, "visits_roll3_mean": 3.6, "visits_delta_prev": 1,
      "admits_lag1": 0, "admits_roll3_mean": 0.2, "admits_delta_prev": 0
    }
  ]
}
```

**Example curl**:
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"a1c_lag1":7.8,"a1c_roll3_mean":7.2,"a1c_delta_prev":0.4,"bmi_lag1":30.1,"bmi_roll3_mean":29.6,"bmi_delta_prev":0.2,"visits_lag1":4,"visits_roll3_mean":3.6,"visits_delta_prev":1}]}' | jq
```

## EDA Notebook

Launch Jupyter from your local venv:
```bash
jupyter lab
```

Open `notebooks/t2d_eda.ipynb`, select the venv kernel, and run all cells.  
If the kernel is missing:
```bash
python -m ipykernel install --user --name alzai-venv --display-name "Python (alzai venv)"
```
The notebook auto-detects BASE_DIR and loads the saved artifact from:

```bash
data/clinical_synth_v1/artifacts_v1/t2d_hgb_model_v1.joblib
```

What it does:

* Sanity checks: counts, patients, years; observed/true prevalence by year.

* Lag analysis: distribution of (observed_year - true_year).

* Missingness: top numeric missing-rates on engineered features (Part 4).

* ROC/PR plots on the validation year; prints AUC and AP.

* Permutation importance (AP scoring) for top features.

* Slice analysis by sex, region, ses_quintile for:

* precision/recall/F1, AUROC, AUPRC, prevalence, predicted positives,

* lift vs slice prevalence and normalized AP.

Saves slice metrics to:
```bash
data/clinical_synth_v1/artifacts_v1/slice_metrics_<label_mode>_<val_year>.csv
data/clinical_synth_v1/artifacts_v1/slice_metrics_with_lift_<label_mode>_<val_year>.csv
```

## Design Notes

- **Data volume**: yearly partitions (`year=YYYY`) in Parquet → scalable; adjust `N_PATIENTS` in Part 1.
- **Temporal features**: rolling stats (mean/std/min/max/count), lag(1), and deltas (Part 4).
- **Label uncertainty**: `label_true` (true onset) vs `label_observed` (small lag noise).
- **Class imbalance**: choose an explicit **threshold τ** on the validation year (optimize F1 or business objective).
- **Single artifact**: one Joblib with the full **pipeline + feature names + τ + val_year** → avoids train/serve drift.
- **Serving contract**: `/predict` accepts JSON with any subset of features; missing fields → `NaN` and imputed by the pipeline.

## Scalability / Tuning

- **Increase dataset size**: edit `N_PATIENTS` in `src/part1_skeleton.py` and re-run Parts **1→4**, then Part **5**.
- **Feature footprint**: run Part 4 with `--mode diet` (or set in code) for a leaner feature set.
- **Model**: `HistGradientBoostingClassifier` (scikit-learn) with early stopping; easy to swap to other classifiers.

