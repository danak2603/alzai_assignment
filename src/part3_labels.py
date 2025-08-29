# --- Part 3 (v2, T2D): label_true / label_observed with time-weighted hazard + pre-onset drift ---
import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd

# ===== Params =====
SEED        = 42
POS_PREV    = 0.08  # target positive prevalence at patient level
BASE_DIR    = "data/clinical_synth_v1"
IN_DATASET  = "patient_year_part2_nb"
OUT_DATASET = "patient_year_part3_nb"
SAVE_FORMAT = "parquet"   # 'parquet' or 'csv' (Parquet requires: pip install pyarrow)

# Hazard shaping (selecting true diagnosis year)
TIME_SKEW   = 2.0   # >1 biases toward later years (1.0 = no time skew)
SCALE_SCORE = 0.7   # how strongly clinical signal affects hazard weights

# Delay (observed vs true) for diagnosis year
LAG_CHOICES = np.array([0, 1, 2, -1])
LAG_PROBS   = np.array([0.70, 0.25, 0.04, 0.01])  # T2D: documentation near onset is common

np.random.seed(SEED)

# ===== I/O helpers (support Parquet/CSV seamlessly) =====
def read_partition(base_dir: str, year: int) -> pd.DataFrame:
    p = Path(base_dir) / f"year={year}"
    p_par = p / "part-0.parquet"
    p_csv = p / "part-0.csv"
    if p_par.exists():
        return pd.read_parquet(p_par)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Missing partition for year={year} under {p}")

def write_partition(df: pd.DataFrame, base_dir: str, year: int, fmt: str = "parquet") -> None:
    p = Path(base_dir) / f"year={year}"
    p.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        (p / "part-0.parquet").unlink(missing_ok=True)
        df.to_parquet(p / "part-0.parquet", index=False)
    elif fmt == "csv":
        (p / "part-0.csv").unlink(missing_ok=True)
        df.to_csv(p / "part-0.csv", index=False)
    else:
        raise ValueError(f"Unsupported fmt={fmt}")

# ===== Load Part 2 =====
in_base = os.path.join(BASE_DIR, IN_DATASET)
years = sorted(int(d.split("=")[1]) for d in os.listdir(in_base) if d.startswith("year="))
assert years, f"No year=YYYY partitions found under {in_base}"

dfs = [read_partition(in_base, y) for y in years]
df = (pd.concat(dfs, ignore_index=True)
        .sort_values(["patient_id", "year"])
        .reset_index(drop=True))

y_min, y_max = years[0], years[-1]

# Ensure numeric dtypes (avoid downcast warnings)
for c in ["a1c","bmi","cci","age","visits"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ===== 1) Choose positive patients by strongest T2D signal (A1c mean) =====
per_pat = df.groupby("patient_id", as_index=True)["a1c"].mean()
n_patients = per_pat.shape[0]
n_pos = max(1, int(round(n_patients * POS_PREV)))
pos_ids = per_pat.sort_values(ascending=False).head(n_pos).index.values

# ===== 2) true_dx_year for positives: time-weighted hazard + clinical score =====
true_dx: dict[int, float | int] = {}
for pid in pos_ids:
    sub = df.loc[df["patient_id"] == pid, ["year","a1c","bmi","cci","age"]].copy()
    # Clinical score (T2D): A1c dominant; BMI/CCI/Age supporting
    score = (
        1.2 * (sub["a1c"].astype(float).fillna(5.6) - 6.2) +
        0.3 * ((sub["bmi"].astype(float).fillna(27.0) - 28.0) / 3.0) +
        0.5 * sub["cci"].astype(float).fillna(0.0) +
        0.2 * ((sub["age"].astype(float).fillna(50.0) - 50.0) / 10.0)
    )

    # More weight to later years (time skew)
    t_weight = ((sub["year"] - y_min + 1) / len(years)) ** TIME_SKEW

    # Positive normalized weights
    w = np.exp(SCALE_SCORE * score.fillna(0.0)) * t_weight
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = t_weight.clip(lower=1e-6)
    p = (w / w.sum()).values

    chosen_year = int(np.random.choice(sub["year"].values, p=p))
    true_dx[int(pid)] = chosen_year

# Non-positives: no event
for pid in df["patient_id"].unique():
    pid = int(pid)
    if pid not in true_dx:
        true_dx[pid] = math.nan

# ===== 3) observed_dx_year: apply small delay w.r.t. true_dx_year =====
observed_dx: dict[int, float | int] = {}
for pid, tdy in true_dx.items():
    if isinstance(tdy, float) and math.isnan(tdy):
        observed_dx[pid] = math.nan
        continue
    lag = int(np.random.choice(LAG_CHOICES, p=LAG_PROBS))
    observed_dx[pid] = int(np.clip(int(tdy) + lag, y_min, y_max))

# ===== 4) Inject pre-onset drift (helps rolling features) =====
# A1c rises in years before diagnosis; BMI and visits rise slightly
for pid, tyear in true_dx.items():
    if isinstance(tyear, float) and math.isnan(tyear):
        continue
    for offset, a1c_bump, bmi_bump, vis_mult in [
        (-2, 0.30, 0.10, 1.05),
        (-1, 0.60, 0.20, 1.10),
        ( 0, 1.00, 0.30, 1.15),
    ]:
        y = int(np.clip(int(tyear) + offset, y_min, y_max))
        m = (df["patient_id"] == pid) & (df["year"] == y)
        if m.any():
            df.loc[m, "a1c"]   = pd.to_numeric(df.loc[m, "a1c"], errors="coerce").fillna(6.0) + a1c_bump
            df.loc[m, "bmi"]   = pd.to_numeric(df.loc[m, "bmi"], errors="coerce").fillna(27.0) + bmi_bump
            df.loc[m, "visits"] = (pd.to_numeric(df.loc[m, "visits"], errors="coerce").fillna(2.0) * vis_mult).round()

# ===== 5) Row-level labels =====
map_true = pd.Series(true_dx)
map_obs  = pd.Series(observed_dx)
df["label_true"]     = (df["year"].values == map_true.reindex(df["patient_id"].values).values).astype(int)
df["label_observed"] = (df["year"].values == map_obs.reindex(df["patient_id"].values).values).astype(int)

# ===== Save (partitioned by year) =====
out_base = os.path.join(BASE_DIR, OUT_DATASET)
os.makedirs(out_base, exist_ok=True)
for y, g in df.groupby("year", sort=True, as_index=False):
    write_partition(g, out_base, int(y), fmt=SAVE_FORMAT)

# ===== Metadata =====
with open(os.path.join(BASE_DIR, "metadata_part3_nb.json"), "w") as f:
    json.dump({
        "source": IN_DATASET, "output": OUT_DATASET, "years": years,
        "positive_prevalence": POS_PREV,
        "hazard_time_skew": TIME_SKEW, "hazard_scale": SCALE_SCORE,
        "lag_probs": {"0":0.70, "+1":0.25, "+2":0.04, "-1":0.01},
        "pre_onset_drift": {"a1c":[-2,-1,0], "bmi":[-2,-1,0], "visits":[-2,-1,0]}
    }, f, indent=2)

# ===== Summary =====
pos_count = int(df.groupby("patient_id")["label_true"].max().sum())
print(f"Part 3 ✔  Positives: {pos_count}/{n_patients} (~{pos_count/n_patients:.1%})  Out: {out_base}")
print("Observed positives per year:\n", df.groupby("year")["label_observed"].sum())
