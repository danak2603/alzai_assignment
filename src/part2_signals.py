# --- Part 2 (lean): Clinical temporal signals + simple utilization + missingness ---
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

# ===== Params =====
SEED        = 42
BASE_DIR    = "data/clinical_synth_v1"
IN_DATASET  = "patient_year_part1_nb"
OUT_DATASET = "patient_year_part2_nb"
SAVE_FORMAT = "parquet"   # 'parquet' or 'csv' (Parquet requires: pip install pyarrow)
NUM_MISS    = 0.03
CAT_MISS    = 0.01

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

# ===== Load Part 1 (patient–year skeleton) =====
in_base = os.path.join(BASE_DIR, IN_DATASET)
years = sorted(int(d.split("=")[1]) for d in os.listdir(in_base) if d.startswith("year="))
dfs = [read_partition(in_base, y) for y in years]
df = (pd.concat(dfs, ignore_index=True)
        .sort_values(["patient_id", "year"])
        .reset_index(drop=True))

n_patients = df["patient_id"].nunique()
n_years    = len(years)

# ===== AR(1) generators for panel time series =====
def ar1_series(n_steps: int, phi: float, sigma: float, start: float) -> np.ndarray:
    x = np.zeros(n_steps, dtype=float)
    x[0] = start + np.random.normal(0, sigma)
    for t in range(1, n_steps):
        x[t] = phi * x[t-1] + np.random.normal(0, sigma)
    return x

def gen_ar_for_panel(n_patients: int, n_years: int,
                     base_mean: float, base_sd: float,
                     phi: float, sigma: float,
                     low: float | None = None, high: float | None = None) -> np.ndarray:
    arr = np.empty(n_patients * n_years, dtype=float)
    idx = 0
    for _ in range(n_patients):
        start = np.random.normal(base_mean, base_sd)
        s = ar1_series(n_years, phi, sigma, start)
        if low  is not None: s = np.maximum(s, low)
        if high is not None: s = np.minimum(s, high)
        arr[idx:idx+n_years] = s
        idx += n_years
    return arr

# ===== Clinical numeric signals (AR(1)) =====
df["bmi"]           = gen_ar_for_panel(n_patients, n_years, 27.0,  4.0, 0.8, 0.9,  16, 55)
df["sbp"]           = gen_ar_for_panel(n_patients, n_years,125.0, 12.0, 0.7, 3.5,  90,200)
df["dbp"]           = gen_ar_for_panel(n_patients, n_years, 78.0,  8.0, 0.7, 2.5,  50,130)
df["ldl"]           = gen_ar_for_panel(n_patients, n_years,120.0, 25.0, 0.65,6.0,  40,250)
df["hdl"]           = gen_ar_for_panel(n_patients, n_years, 52.0, 10.0, 0.7, 2.5,  20,120)
df["triglycerides"] = gen_ar_for_panel(n_patients, n_years,140.0, 40.0, 0.65,10.0,  40,600)
df["a1c"]           = gen_ar_for_panel(n_patients, n_years,  5.6,  0.8, 0.75,0.2,   4.0,14.0)
df["egfr"]          = gen_ar_for_panel(n_patients, n_years, 90.0, 18.0, 0.7, 4.0,  10,140)
df["alt"]           = gen_ar_for_panel(n_patients, n_years, 22.0, 10.0, 0.6, 3.0,   5,200)
df["ast"]           = gen_ar_for_panel(n_patients, n_years, 24.0,  9.0, 0.6, 3.0,   5,200)

# Ensure numeric for downstream simple rules
for c in ["a1c","bmi","egfr","alt","ast"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ===== Simple utilization (depends on age/CCI/A1c thresholds) =====
base_visits = 2.0 + 0.4*(df["age"] > 65).astype(float) + 0.3*df["cci"].astype(float)
risk_bump  = 0.15*(df["a1c"] > 6.5).astype(float) + 0.25*(df["a1c"] > 8.0).astype(float)
lam_visits = np.clip(base_visits * (1.0 + risk_bump), 0.1, 30)
df["visits"] = np.random.poisson(lam_visits)

base_adm = 0.10 + 0.03*(df["age"] > 70).astype(float) + 0.05*(df["cci"] >= 2).astype(float)
sev_bump = 0.20*(df["a1c"] > 8.5).astype(float)
lam_adm  = np.clip(base_adm * (1.0 + sev_bump), 0.01, 5)
df["admits"] = np.random.poisson(lam_adm)

# ===== Medication category (coarse rule based on A1c/CCI) =====
choices = np.array(["none","oral","injectable"])
p_none = 1.0 - 0.06*df["cci"].astype(float) - 0.10*(df["a1c"] > 6.5).astype(float)
p_oral = 0.05 + 0.05*df["cci"].astype(float) + 0.08*(df["a1c"] > 6.5).astype(float)
p_inj  = 0.02 + 0.02*df["cci"].astype(float) + 0.05*(df["a1c"] > 8.0).astype(float)
probs  = np.clip(np.stack([p_none, p_oral, p_inj], axis=1), 0.001, None)
probs  = probs / probs.sum(axis=1, keepdims=True)
df["med_category"] = np.array([np.random.choice(choices, p=probs[i]) for i in range(len(df))])

# ===== Missingness =====
def apply_missingness(values: np.ndarray, miss_rate: float) -> np.ndarray:
    mask = np.random.rand(values.shape[0]) < miss_rate
    out = values.astype(object)
    out[mask] = None
    return out

num_cols = ["bmi","sbp","dbp","ldl","hdl","triglycerides","a1c","egfr","alt","ast","visits","admits"]
for c in num_cols:
    df[c] = apply_missingness(df[c].values, NUM_MISS)
df["med_category"] = apply_missingness(df["med_category"].astype(object).values, CAT_MISS)

# ===== Save (partitioned by year) =====
out_base = os.path.join(BASE_DIR, OUT_DATASET)
os.makedirs(out_base, exist_ok=True)
for y, g in df.groupby("year", sort=True, as_index=False):
    write_partition(g, out_base, int(y), fmt=SAVE_FORMAT)

# ===== Metadata =====
with open(os.path.join(BASE_DIR, "metadata_part2_nb.json"), "w") as f:
    json.dump({
        "source": IN_DATASET, "output": OUT_DATASET, "years": years,
        "features_numeric": num_cols,
        "features_categorical": ["sex","region","ses_quintile","med_category","cci"],
        "notes": "Part 2 adds AR(1) clinical signals, simple utilization, and missingness."
    }, f, indent=2)

print(f"Part 2 ✔  Rows: {len(df):,}  Years: {years[0]}-{years[-1]}  Out: {out_base}  Format: {SAVE_FORMAT}")
