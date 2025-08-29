# --- Part 5 (HGB): Train HistGradientBoosting, pick threshold (F1), eval, save artifact ---
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import joblib

# ===== Params =====
SEED        = 42
BASE_DIR    = "data/clinical_synth_v1"
IN_DATASET  = "patient_year_part4_nb"  # Part 4 full (no-leak)
LABEL_COL   = "label_observed"         # train on observed labels
ART_DIR     = "artifacts_v1"
ART_NAME    = "t2d_hgb_model_v1.joblib"

np.random.seed(SEED)

def read_partition(base_dir: str, year: int) -> pd.DataFrame:
    p = Path(base_dir) / f"year={year}"
    if (p / "part-0.parquet").exists():
        return pd.read_parquet(p / "part-0.parquet")
    if (p / "part-0.csv").exists():
        return pd.read_csv(p / "part-0.csv")
    raise FileNotFoundError(f"Missing partition for year={year} under {p}")

# ===== Load =====
in_base = Path(BASE_DIR) / IN_DATASET
years = sorted(int(d.split("=")[1]) for d in os.listdir(in_base) if d.startswith("year="))
dfs = [read_partition(in_base, y) for y in years]
df  = (pd.concat(dfs, ignore_index=True)
        .sort_values(["patient_id","year"])
        .reset_index(drop=True))

val_year = years[-1]
print("Observed positives per year:\n", df.groupby("year")[LABEL_COL].sum())
print(f"\nValidation year → {val_year}")

# ===== Features/target =====
drop_cols = ["patient_id","year","label_true","label_observed"]
feature_cols = [c for c in df.columns if c not in drop_cols]
cat_cols = [c for c in ["sex","region"] if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

train_df = df[df["year"] != val_year].copy()
val_df   = df[df["year"] == val_year].copy()

X_train = train_df[feature_cols]; y_train = train_df[LABEL_COL].astype(int).values
X_val   = val_df[feature_cols];   y_val   = val_df[LABEL_COL].astype(int).values

# ===== OHE compatibility (scikit-learn >=1.2 uses sparse_output, older uses sparse) =====
ohe_kwargs = {"handle_unknown": "ignore"}
if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
    ohe_kwargs["sparse_output"] = False   # new API
else:
    ohe_kwargs["sparse"] = False          # old API

# ===== Preprocess + HGB =====
pre = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(**ohe_kwargs)),
        ]), cat_cols),
    ],
    remainder="drop"
)

clf = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.06,
    max_depth=6,
    max_iter=500,
    random_state=SEED,
    early_stopping=False,
)

pipe = Pipeline(steps=[
    ("pre", pre),
    ("clf", clf),
])

# Class imbalance → weights
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
w_pos = (neg / max(pos, 1))
sample_w = np.where(y_train == 1, w_pos, 1.0).astype(float)

pipe.fit(X_train, y_train, clf__sample_weight=sample_w)

# ===== Predict & threshold selection (maximize F1) =====
val_prob = pipe.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_prob)

def best_f1_threshold(y_true, p):
    grid = np.unique(np.concatenate([thr, np.linspace(0, 1, 501)]))
    best_tau, best_f1 = 0.5, 0.0
    for t in grid:
        y_hat = (p >= t).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, t
    return float(best_tau), float(best_f1)

tau, f1_at_tau = best_f1_threshold(y_val, val_prob)

auroc = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else float("nan")
auprc = average_precision_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else float("nan")
y_hat = (val_prob >= tau).astype(int)
prec_at_tau = precision_score(y_val, y_hat, zero_division=0)
rec_at_tau  = recall_score(y_val, y_hat, zero_division=0)

print("\n== Observed labels (validation year) ==")
print(f"AUROC: {auroc:.3f}")
print(f"AUPRC: {auprc:.3f}")
print(f"F1@τ : {f1_at_tau:.3f} (τ={tau:.3f})")
print(f"Prec@τ: {prec_at_tau:.3f}")
print(f"Rec@τ : {rec_at_tau:.3f}")

# ===== Save artifact & metrics =====
art_base = Path(BASE_DIR) / ART_DIR
art_base.mkdir(parents=True, exist_ok=True)
art_path = art_base / ART_NAME

artifact = {
    "pipeline": pipe,
    "feature_names": feature_cols,
    "threshold": float(tau),
    "val_year": int(val_year),
    "label_mode": "observed",
    "model_type": "hist_gradient_boosting",
    "sklearn_version": __import__("sklearn").__version__,
}
joblib.dump(artifact, art_path)

metrics = {
    "val_year": val_year,
    "label_mode": "observed",
    "auroc": float(auroc),
    "auprc": float(auprc),
    "f1_at_tau": float(f1_at_tau),
    "precision_at_tau": float(prec_at_tau),
    "recall_at_tau": float(rec_at_tau),
    "threshold": float(tau),
    "n_val": int(len(y_val)),
    "positives_val": int(y_val.sum()),
    "prevalence_val": float(y_val.mean()),
    "n_features": len(feature_cols),
}
with open(art_base / "metrics_v1.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved model → {art_path}")
print(f"Saved metrics → {art_base/'metrics_v1.json'}")
print(f"Features used ({len(feature_cols)}): first 10 → {feature_cols[:10]}")
