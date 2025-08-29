# app/main.py
import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---- Config ----
ART_PATH = os.environ.get(
    "MODEL_ART_PATH",
    "data/clinical_synth_v1/artifacts_v1/t2d_hgb_model_v1.joblib"
)

# ---- Load single-format artifact (pipeline-based) ----
# Expected keys: pipeline, feature_names, threshold, val_year
artifact = joblib.load(ART_PATH)

try:
    PIPELINE = artifact["pipeline"]                 # sklearn Pipeline (preproc + model)
    FEATURES = artifact["feature_names"]            # list[str], exact training order
except KeyError as e:
    raise RuntimeError(
        f"Artifact at {ART_PATH} missing required key: {e}. "
        "Expected a single-format artifact with 'pipeline' and 'feature_names'."
    )

THRESHOLD   = float(artifact.get("threshold", 0.5))
VAL_YEAR    = int(artifact.get("val_year", -1))
LABEL_TRAIN = artifact.get("label_train") or artifact.get("label_mode") or "label_observed"
SKL_VER     = artifact.get("sklearn_version")

def ensure_feature_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Ensure all training features exist; add missing as NaN and reorder columns."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df.loc[:, cols]

def predict_proba_df(df: pd.DataFrame) -> np.ndarray:
    """Predict probabilities using the trained pipeline."""
    X = ensure_feature_frame(df, FEATURES)
    return PIPELINE.predict_proba(X)[:, 1]

# ---- FastAPI App ----
app = FastAPI(title="T2D Onset Year Classifier", version="1.0")

class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(
        ..., description="List of feature dicts; keys should match training feature names"
    )

@app.get("/")
def root():
    return {"message": "OK. See /health, /metadata, /docs"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifact_path": ART_PATH,
        "n_features": len(FEATURES),
        "val_year": VAL_YEAR,
        "label_train": LABEL_TRAIN,
        "sklearn_version": SKL_VER,
        "uses_pipeline": True,  # single-format only
    }

@app.get("/metadata")
def metadata():
    return {
        "feature_names": FEATURES,
        "threshold": THRESHOLD,
        "val_year": VAL_YEAR,
        "label_train": LABEL_TRAIN,
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.instances:
        raise HTTPException(status_code=400, detail="Empty 'instances'.")
    df = pd.DataFrame(req.instances)
    proba = predict_proba_df(df)
    yhat = (proba >= THRESHOLD).astype(int)
    return {"proba": proba.tolist(), "yhat": yhat.tolist(), "threshold": THRESHOLD}
