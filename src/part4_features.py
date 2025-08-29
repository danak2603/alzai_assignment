# --- Part 4: Temporal feature engineering (FULL) with no leakage (uses delta_prev) ---
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

# ===== Params =====
SEED        = 42
BASE_DIR    = "data/clinical_synth_v1"
IN_DATASET  = "patient_year_part3_nb"
OUT_DATASET = "patient_year_part4_nb"
SAVE_FORMAT = "parquet"   # 'parquet' or 'csv'
MODE        = "full"      # keep "full" per your request

np.random.seed(SEED)

# ===== I/O helpers (Parquet/CSV) =====
def read_partition(base_dir: str, year: int) -> pd.DataFrame:
    p = Path(base_dir) / f"year={year}"
    if (p / "part-0.parquet").exists():
        return pd.read_parquet(p / "part-0.parquet")
    if (p / "part-0.csv").exists():
        return pd.read_csv(p / "part-0.csv")
    raise FileNotFoundError(f"Missing partition for year={year} under {p}")

def write_partition(df: pd.DataFrame, base_dir: str, year: int, fmt: str = "parquet") -> None:
    p = Path(base_dir) / f"year={year}"
    p.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(p / "part-0.parquet", index=False)
    elif fmt == "csv":
        df.to_csv(p / "part-0.csv", index=False)
    else:
        raise ValueError(f"Unsupported fmt={fmt}")

# ===== Load Part 3 (labels included) =====
in_base = os.path.join(BASE_DIR, IN_DATASET)
years = sorted(int(d.split("=")[1]) for d in os.listdir(in_base) if d.startswith("year="))
assert years, f"No year=YYYY partitions found under {in_base}"

df = (pd.concat([read_partition(in_base, y) for y in years], ignore_index=True)
        .sort_values(["patient_id","year"])
        .reset_index(drop=True))

# numeric signals to engineer
num_cols = [
    "bmi","sbp","dbp","ldl","hdl","triglycerides","a1c","egfr","alt","ast","visits","admits"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ===== Temporal features per patient =====
# Minimal set (past-only):
#   lag1 = x_{t-1}
#   roll3_mean = mean(x_{t-1}, x_{t-2}, x_{t-3})
#   delta_prev = x_{t-1} - x_{t-2}   (PAST ONLY → no leakage)
g = df.groupby("patient_id", sort=False)
feat_cols: list[str] = []

for c in num_cols:
    if c not in df.columns:
        continue
    lag1 = g[c].shift(1)
    lag2 = g[c].shift(2)

    df[f"{c}_lag1"]        = lag1
    df[f"{c}_roll3_mean"]  = g[c].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
    df[f"{c}_delta_prev"]  = lag1 - lag2

    feat_cols += [f"{c}_lag1", f"{c}_roll3_mean", f"{c}_delta_prev"]

# FULL extras (a bit richer but still past-only):
if MODE.lower() == "full":
    for c in num_cols:
        if c not in df.columns:
            continue
        s = g[c]
        df[f"{c}_lag2"]      = s.shift(2)
        df[f"{c}_roll3_std"] = s.apply(lambda x: x.shift(1).rolling(3, min_periods=2).std()).reset_index(level=0, drop=True)
        df[f"{c}_roll3_min"] = s.apply(lambda x: x.shift(1).rolling(3, min_periods=1).min()).reset_index(level=0, drop=True)
        df[f"{c}_roll3_max"] = s.apply(lambda x: x.shift(1).rolling(3, min_periods=1).max()).reset_index(level=0, drop=True)
        feat_cols += [f"{c}_lag2", f"{c}_roll3_std", f"{c}_roll3_min", f"{c}_roll3_max"]

# Keep essentials + engineered features
keep_cols = ["patient_id","year","sex","region","ses_quintile","cci","label_true","label_observed"]
out_cols  = [c for c in keep_cols if c in df.columns] + feat_cols
df_out    = df[out_cols].copy()

# ===== Save (partitioned by year) =====
out_base = os.path.join(BASE_DIR, OUT_DATASET)
os.makedirs(out_base, exist_ok=True)
for y, gdf in df_out.groupby("year", sort=True, as_index=False):
    write_partition(gdf, out_base, int(y), fmt=SAVE_FORMAT)

# ===== Metadata =====
with open(os.path.join(BASE_DIR, "metadata_part4_nb.json"), "w") as f:
    json.dump({
        "source": IN_DATASET,
        "output": OUT_DATASET,
        "years": years,
        "mode": MODE,
        "n_features": len(feat_cols),
        "features": feat_cols,
        "notes": "FULL no-leak: lag1, roll3_mean, delta_prev (past-only) + lag2/std/min/max."
    }, f, indent=2)

print(f"Part 4 ✔ mode={MODE}  Years: {years[0]}-{years[-1]}  Out: {out_base}  Format: {SAVE_FORMAT}")
print(f"Features generated: {len(feat_cols)} (e.g., {feat_cols[:10]})")
