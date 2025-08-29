# --- Part 1: Patient–year skeleton (2016–2023) ---
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

# ===== Params =====
SEED = 42
N_PATIENTS = 10_000
START_YEAR = 2016
END_YEAR   = 2023
OUT_DIR    = "data/clinical_synth_v1"
DATASET    = "patient_year_part1_nb"
SAVE_FORMAT = "parquet"   # 'parquet' or 'csv'  (Parquet requires: pip install pyarrow)

np.random.seed(SEED)
years = list(range(START_YEAR, END_YEAR + 1))
n_years = len(years)

# ===== Helpers =====
def sample_categories(n, probs, labels):
    return np.random.choice(labels, size=n, p=probs)

def bounded_normal(n, mean, sd, low, high):
    x = np.random.normal(mean, sd, size=n)
    return np.clip(x, low, high)

# ===== Patient-level (one row per patient) =====
pid = np.arange(1, N_PATIENTS + 1)
sex = sample_categories(N_PATIENTS, [0.5, 0.5], ["F", "M"])
region_labels = ["Center", "North", "South", "Jerusalem", "Sharon"]
region = sample_categories(N_PATIENTS, [0.35, 0.20, 0.20, 0.15, 0.10], region_labels)
ses_quintile = np.random.choice([1, 2, 3, 4, 5], size=N_PATIENTS)
age0 = bounded_normal(N_PATIENTS, 52, 16, 20, 85).round().astype(int)
cci  = np.clip((age0 - 50) / 15 + np.random.normal(0, 0.7, N_PATIENTS), 0, None)
cci  = np.round(cci, 0).astype(int)

patients = pd.DataFrame({
    "patient_id": pid,
    "sex": sex,
    "region": region,
    "ses_quintile": ses_quintile,
    "age0": age0,
    "cci": cci,
})

# ===== Expand to patient–year (one row per patient per year) =====
df = pd.DataFrame({
    "patient_id": np.repeat(patients["patient_id"].values, n_years),
    "year": np.tile(years, N_PATIENTS),
    "sex": np.repeat(patients["sex"].values, n_years),
    "region": np.repeat(patients["region"].values, n_years),
    "ses_quintile": np.repeat(patients["ses_quintile"].values, n_years),
    "cci": np.repeat(patients["cci"].values, n_years),
    "age": (np.repeat(patients["age0"].values, n_years) + np.tile(np.arange(n_years), N_PATIENTS)).astype(int),
})

# ===== Save partitioned by year (CSV/Parquet) =====
base_path = Path(OUT_DIR) / DATASET
base_path.mkdir(parents=True, exist_ok=True)

for y, g in df.groupby("year", sort=True, as_index=False):
    y_dir = base_path / f"year={int(y)}"
    y_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_FORMAT == "parquet":
        g.to_parquet(y_dir / "part-0.parquet", index=False)
    elif SAVE_FORMAT == "csv":
        g.to_csv(y_dir / "part-0.csv", index=False)
    else:
        raise ValueError("SAVE_FORMAT must be 'parquet' or 'csv'.")

# ===== Minimal metadata (useful for later steps) =====
meta = {
    "config": {
        "seed": SEED,
        "n_patients": N_PATIENTS,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "format": SAVE_FORMAT,
    },
    "n_rows": int(df.shape[0]),
    "n_patients": int(df["patient_id"].nunique()),
    "years": years,
    "features_static": ["sex", "region", "ses_quintile", "cci"],
    "features_temporal": ["age"],
}
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
with open(Path(OUT_DIR) / "metadata_part1_nb.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Part 1 ✔  Rows: {len(df):,}  Years: {START_YEAR}-{END_YEAR}  Out: {base_path}  Format: {SAVE_FORMAT}")
