# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
# -------- Install if needed (run once) --------
# %pip install lifelines

# -------- Core Imports --------
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Spark (for percentile step) --------
from pyspark.sql import functions as F

# -------- Survival Library --------
from lifelines import CoxPHFitter

# -------- Base Output Directory --------
BASE_DIR = "/Workspace/Shared/DAMO_699-4-Capstone-Project/output"

# -------- Model Ready --------
TORONTO_TABLE = "workspace.capstone_project.toronto_model_ready"
NYC_TABLE     = "workspace.capstone_project.nyc_model_ready"

# -------- Cox Model Paths --------
COX_META_TORONTO = f"{BASE_DIR}/models/cox_meta_Toronto.json"
COX_META_NYC     = f"{BASE_DIR}/models/cox_meta_NYC.json"

# -------- Model Paths --------
CPH_TORONTO = f"{BASE_DIR}/models/cph_Toronto.pkl"
CPH_NYC     = f"{BASE_DIR}/models/cph_NYC.pkl"

META_TORONTO = f"{BASE_DIR}/models/cox_meta_Toronto.json"
META_NYC     = f"{BASE_DIR}/models/cox_meta_NYC.json"

# -------- HR Tables --------
HR_TORONTO = f"{BASE_DIR}/tables/cox_hr_Toronto.csv"
HR_NYC     = f"{BASE_DIR}/tables/cox_hr_NYC.csv"

# -------- Stats Tables --------
STATS_TORONTO = f"{BASE_DIR}/tables/cox_stats_Toronto.csv"
STATS_NYC     = f"{BASE_DIR}/tables/cox_stats_NYC.csv"

# -------- Reference Rows --------
REF_TORONTO = f"{BASE_DIR}/tables/cox_reference_row_Toronto.csv"
REF_NYC     = f"{BASE_DIR}/tables/cox_reference_row_NYC.csv"

# -------- SHAP Files --------
SHAP_TORONTO = f"{BASE_DIR}/shap/toronto/toronto_shap_importance.csv"
SHAP_NYC     = f"{BASE_DIR}/shap/nyc/nyc_shap_importance.csv"

# -------- Figures + Tables --------
FIG_DIR = f"{BASE_DIR}/graphs"
TABLE_DIR = f"{BASE_DIR}/tables"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# -------- Demand Features --------
DEMAND_FEATURES = ["calls_past_30min", "calls_past_60min"]

print("Common setup loaded successfully.")


# %%
# ---------- Extracting feature list from Cox meta ----------
def extract_features(meta):
    possible_keys = [
        "features", "feature_list", "covariates",
        "columns", "model_features", "x_cols"
    ]
    for key in possible_keys:
        if key in meta and isinstance(meta[key], list):
            return meta[key]

    # check nested structures
    for key in ["spec", "model_spec", "meta"]:
        if key in meta and isinstance(meta[key], dict):
            if "features" in meta[key] and isinstance(meta[key]["features"], list):
                return meta[key]["features"]

    return []


# %%
# ---------- Survival confirmation ----------
def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def check_survival(meta_path):
    meta = read_json(meta_path)
    features = set(extract_features(meta))

    return {
        "calls_past_30min_in_survival": "Yes" if "calls_past_30min" in features else "No",
        "calls_past_60min_in_survival": "Yes" if "calls_past_60min" in features else "No"
    }



# %%
# ---------- SHAP confirmation ----------
def check_shap(shap_path):
    df = pd.read_csv(shap_path)

    # try common feature column names
    possible_cols = ["feature", "Feature", "variable", "Variable", "name"]
    feature_col = None

    for col in possible_cols:
        if col in df.columns:
            feature_col = col
            break

    if feature_col is None:
        return {
            "calls_past_30min_in_predictive": "Unknown",
            "calls_past_60min_in_predictive": "Unknown"
        }

    features = set(df[feature_col].astype(str))

    return {
        "calls_past_30min_in_predictive": "Yes" if "calls_past_30min" in features else "No",
        "calls_past_60min_in_predictive": "Yes" if "calls_past_60min" in features else "No"
    }


# %%
# ---------- Running checks ----------
toronto_surv = check_survival(COX_META_TORONTO)
nyc_surv     = check_survival(COX_META_NYC)

toronto_shap = check_shap(SHAP_TORONTO)
nyc_shap     = check_shap(SHAP_NYC)


# %%
# ---------- Building confirmation table ----------
confirmation_df = pd.DataFrame([
    {
        "city": "Toronto",
        **toronto_surv,
        **toronto_shap
    },
    {
        "city": "NYC",
        **nyc_surv,
        **nyc_shap
    }
])
display(confirmation_df)

# %%
# ---------- Save confirmation table ----------
out_path = f"{TABLE_DIR}/demand_feature_confirmation.csv"
confirmation_df.to_csv(out_path, index=False)
print("Saved confirmation table:", out_path)


# %%
# ---------- helpers ----------
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_censor_time(meta: dict):
    # try common keys
    for k in ["censor_time", "censor_minutes", "censor_threshold", "time_horizon", "t_max"]:
        if k in meta:
            return meta[k]
    # nested possibility
    for a in ["spec", "model_spec", "meta"]:
        if a in meta and isinstance(meta[a], dict):
            for k in ["censor_time", "censor_minutes", "censor_threshold", "time_horizon", "t_max"]:
                if k in meta[a]:
                    return meta[a][k]
    return None

def read_stats(path):
    """
    Flexible reader: supports stats as 1-row CSV or key/value layout.
    Attempts to extract:
      - n (sample size)
      - concordance_index (c-index)
    """
    df = pd.read_csv(path)

    # Case A: 1-row wide table with columns
    lower_cols = [c.lower() for c in df.columns]
    if df.shape[0] >= 1:
        row = df.iloc[0].to_dict()
        # find n
        n = None
        for key in row.keys():
            if str(key).lower() in ["n", "sample_size", "samples", "num_rows", "n_samples"]:
                n = row[key]
                break
        # find c-index
        c_index = None
        for key in row.keys():
            if str(key).lower() in ["concordance_index", "c_index", "cindex", "concordance"]:
                c_index = row[key]
                break

        # Case B: key/value table (e.g. metric,value)
        if n is None or c_index is None:
            # try to detect metric/value cols
            metric_col = None
            value_col = None
            for c in df.columns:
                if c.lower() in ["metric", "name", "stat", "key"]:
                    metric_col = c
                if c.lower() in ["value", "val", "stat_value"]:
                    value_col = c
            if metric_col and value_col:
                metric_map = dict(zip(df[metric_col].astype(str).str.lower(), df[value_col]))
                n = n if n is not None else metric_map.get("n", metric_map.get("sample_size"))
                c_index = c_index if c_index is not None else metric_map.get("concordance_index", metric_map.get("c_index"))

        return n, c_index

    return None, None

def standardize_hr_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common Cox HR table column names from lifelines or custom exports.
    """
    # normalize columns to lower for matching
    cols = {c: c.lower() for c in df.columns}
    df2 = df.rename(columns=cols)

    # Try to identify the variable/feature column
    var_col_candidates = ["covariate", "variable", "feature", "term", "index", "name"]
    var_col = None
    for c in var_col_candidates:
        if c in df2.columns:
            var_col = c
            break

    if var_col is None:
        # sometimes the feature is the index
        df2 = df2.reset_index().rename(columns={"index": "feature"})
        var_col = "feature"

    # lifelines often uses: coef, exp(coef), p, exp(coef) lower 95%, exp(coef) upper 95%
    # normalize to: coef, hr, ci_lower, ci_upper, p_value
    rename_map = {}

    # coef
    if "coef" in df2.columns:
        rename_map["coef"] = "coef"

    # HR
    for c in ["exp(coef)", "hr", "hazard_ratio", "exp_coef"]:
        if c in df2.columns:
            rename_map[c] = "hr"
            break

    # p-value
    for c in ["p", "p_value", "p-value"]:
        if c in df2.columns:
            rename_map[c] = "p_value"
            break

    # CI lower/upper
    for c in ["exp(coef) lower 95%", "hr_lower_95", "ci_lower", "lower_ci", "lower 95%", "lower"]:
        if c in df2.columns:
            rename_map[c] = "ci_lower"
            break

    for c in ["exp(coef) upper 95%", "hr_upper_95", "ci_upper", "upper_ci", "upper 95%", "upper"]:
        if c in df2.columns:
            rename_map[c] = "ci_upper"
            break

    df2 = df2.rename(columns=rename_map)

    # Feature column
    df2 = df2.rename(columns={var_col: "feature"})

    # Keep only necessary columns if they exist
    keep = ["feature", "coef", "hr", "ci_lower", "ci_upper", "p_value"]
    existing = [c for c in keep if c in df2.columns]
    return df2[existing]

def build_city_demand_hr(city, hr_path, stats_path, meta_path):
    hr_df_raw = pd.read_csv(hr_path)
    hr_df = standardize_hr_table(hr_df_raw)

    # filter demand features only
    hr_df["feature"] = hr_df["feature"].astype(str)
    hr_df = hr_df[hr_df["feature"].isin(DEMAND_FEATURES)].copy()

    # add city
    hr_df.insert(0, "city", city)

    # add meta context
    meta = read_json(meta_path)
    censor_time = get_censor_time(meta)
    hr_df["censor_time"] = censor_time

    # add stats context
    n, c_index = read_stats(stats_path)
    hr_df["n"] = n
    hr_df["concordance_index"] = c_index

    # interpretation indicator
    if "hr" in hr_df.columns:
        hr_df["interpretation"] = hr_df["hr"].apply(
            lambda x: "HR < 1 → slower arrival (higher delay risk)" if pd.notna(x) and x < 1
            else ("HR > 1 → faster arrival (lower delay risk)" if pd.notna(x) and x > 1 else "HR = 1 → no change")
        )
    else:
        hr_df["interpretation"] = "HR not found in table"

    return hr_df

# ---------- Build combined table ----------
toronto_demand_hr = build_city_demand_hr("Toronto", HR_TORONTO, STATS_TORONTO, META_TORONTO)
nyc_demand_hr     = build_city_demand_hr("NYC",     HR_NYC,     STATS_NYC,     META_NYC)

combined_demand_hr = pd.concat([toronto_demand_hr, nyc_demand_hr], ignore_index=True)

# nice ordering
preferred_order = ["city", "feature", "coef", "hr", "ci_lower", "ci_upper", "p_value", "censor_time", "n", "concordance_index", "interpretation"]
combined_demand_hr = combined_demand_hr[[c for c in preferred_order if c in combined_demand_hr.columns]]

display(combined_demand_hr)

# ---------- Save ----------
out_path = f"{TABLE_DIR}/demand_hr_table_toronto_nyc.csv"
combined_demand_hr.to_csv(out_path, index=False)
print("Saved combined demand HR table:", out_path)


# %%
# Function to compute percentiles
def compute_percentiles(table_name, city_name):
    
    df = spark.table(table_name)
    
    results = []
    
    for feature in DEMAND_FEATURES:
        
        percentiles = df.select(
            F.expr(f"percentile_approx({feature}, array(0.1, 0.9)) as pct")
        ).collect()[0]["pct"]
        
        results.append({
            "city": city_name,
            "feature": feature,
            "low_demand_p10": percentiles[0],
            "high_demand_p90": percentiles[1]
        })
    
    return results

# Compute for both cities
toronto_results = compute_percentiles(TORONTO_TABLE, "Toronto")
nyc_results     = compute_percentiles(NYC_TABLE, "NYC")

scenario_df = pd.DataFrame(toronto_results + nyc_results)

display(scenario_df)

# Save output
OUTPUT_DIR = "/Workspace/Shared/DAMO_699-4-Capstone-Project/output/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

out_path = f"{OUTPUT_DIR}/demand_scenario_definition.csv"
scenario_df.to_csv(out_path, index=False)

print("Saved scenario table:", out_path)

# %%
# ===== Demand scenario values from your Step 3 output =====
TOR_LOW_60, TOR_HIGH_60 = 0, 1
NYC_LOW_60, NYC_HIGH_60 = 45, 107


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_censor_time(meta: dict, default=60):
    for k in ["censor_time", "censor_minutes", "censor_threshold", "time_horizon", "t_max"]:
        if k in meta:
            return int(meta[k])
    for a in ["spec", "model_spec", "meta"]:
        if a in meta and isinstance(meta[a], dict):
            for k in ["censor_time", "censor_minutes", "censor_threshold", "time_horizon", "t_max"]:
                if k in meta[a]:
                    return int(meta[a][k])
    return default

def load_cph(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_reference_row(path):
    df = pd.read_csv(path)
    # Ensure single-row dataframe
    if df.shape[0] != 1:
        df = df.head(1)
    return df

def align_reference_to_model(cph, ref_df):
    """
    Ensures reference row has exactly the columns expected by the Cox model.
    If extra cols exist, drop them.
    If missing cols exist, fill with 0 (rare, but safe).
    """
    model_cols = list(getattr(cph, "params_", pd.Series()).index)
    if not model_cols:
        # fallback: sometimes model stores a training column list differently
        model_cols = list(ref_df.columns)

    aligned = ref_df.copy()

    # Drop extras
    extras = [c for c in aligned.columns if c not in model_cols]
    if extras:
        aligned = aligned.drop(columns=extras)

    # Add missing
    missing = [c for c in model_cols if c not in aligned.columns]
    for c in missing:
        aligned[c] = 0

    # Reorder to match model
    aligned = aligned[model_cols]
    return aligned

def risk_curve_from_cph(cph, base_row, demand_value_60, times):
    row = base_row.copy()
    if "calls_past_60min" not in row.columns:
        raise ValueError("Reference row does not contain 'calls_past_60min'. Check cox_reference_row_*.csv.")
    row.loc[:, "calls_past_60min"] = demand_value_60

    # lifelines returns survival function DataFrame: index=times, col per row
    sf = cph.predict_survival_function(row, times=times)
    # Convert to delay risk: 1 - S(t)
    risk = 1 - sf.iloc[:, 0].values
    return risk

def plot_city_low_high(city, cph_path, ref_path, meta_path, low_60, high_60, out_name):
    cph = load_cph(cph_path)
    meta = read_json(meta_path)
    censor_time = get_censor_time(meta, default=60)

    times = np.arange(0, censor_time + 1, 1)  # 0..60 by 1 minute

    ref = load_reference_row(ref_path)
    ref = align_reference_to_model(cph, ref)

    risk_low  = risk_curve_from_cph(cph, ref, low_60, times)
    risk_high = risk_curve_from_cph(cph, ref, high_60, times)

    plt.figure(figsize=(8, 6))
    plt.plot(times, risk_low,  label=f"Low demand (P10): {low_60}")
    plt.plot(times, risk_high, label=f"High demand (P90): {high_60}")
    plt.title(f"{city} – Delay Risk Curves (1 − S(t))\nVarying calls_past_60min, others fixed")
    plt.xlabel("Response time threshold t (minutes)")
    plt.ylabel("Delay risk = P(Response time > t)")
    plt.xlim(0, censor_time)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = f"{FIG_DIR}/{out_name}"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", out_path)

# --- Generate both city plots ---
plot_city_low_high(
    city="Toronto",
    cph_path=CPH_TORONTO,
    ref_path=REF_TORONTO,
    meta_path=META_TORONTO,
    low_60=TOR_LOW_60,
    high_60=TOR_HIGH_60,
    out_name="risk_curve_toronto_low_vs_high_demand.png"
)

plot_city_low_high(
    city="NYC",
    cph_path=CPH_NYC,
    ref_path=REF_NYC,
    meta_path=META_NYC,
    low_60=NYC_LOW_60,
    high_60=NYC_HIGH_60,
    out_name="risk_curve_nyc_low_vs_high_demand.png"
)


# %%
# =========================================
# STEP 5: SHAP Demand Alignment 
# Output: a small table + bullet notes
# =========================================

def read_shap_importance(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Detect feature column
    feat_cols = ["feature", "Feature", "variable", "Variable", "name", "Name"]
    feat_col = next((c for c in feat_cols if c in df.columns), None)
    if feat_col is None:
        raise ValueError(f"No feature column found in SHAP file: {path}. Columns: {df.columns.tolist()}")

    # Detect importance column
    # Common: mean_abs_shap, mean_abs_shap_value, importance, mean(|shap|), etc.
    imp_cols = ["mean_abs_shap", "mean_abs_shap_value", "importance", "mean_abs", "mean_shap", "shap_importance"]
    imp_col = next((c for c in imp_cols if c in df.columns), None)

    # If we can't find an importance column, we still can rank by existing order.
    df = df.copy()
    df.rename(columns={feat_col: "feature"}, inplace=True)
    if imp_col:
        df.rename(columns={imp_col: "importance"}, inplace=True)
        df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
        df = df.sort_values("importance", ascending=False)
    else:
        df["importance"] = np.nan  # unknown

    df["rank"] = range(1, len(df) + 1)
    return df[["rank", "feature", "importance"]]

def demand_shap_summary(city: str, shap_path: str):
    df = read_shap_importance(shap_path)

    rows = []
    for feat in DEMAND_FEATURES:
        hit = df[df["feature"].astype(str) == feat]
        if hit.empty:
            rows.append({"city": city, "feature": feat, "shap_rank": None, "shap_importance": None, "present": "No"})
        else:
            rows.append({
                "city": city,
                "feature": feat,
                "shap_rank": int(hit["rank"].iloc[0]),
                "shap_importance": hit["importance"].iloc[0] if pd.notna(hit["importance"].iloc[0]) else None,
                "present": "Yes"
            })
    return pd.DataFrame(rows)

shap_to = demand_shap_summary("Toronto", SHAP_TORONTO)
shap_ny = demand_shap_summary("NYC", SHAP_NYC)

shap_demand_df = pd.concat([shap_to, shap_ny], ignore_index=True)
display(shap_demand_df)

# ---- Bullet notes (2–4 bullets total) ----
print("\n----- Step 5: SHAP Alignment Notes -----\n")

for city in ["Toronto", "NYC"]:
    sub = shap_demand_df[shap_demand_df["city"] == city]
    present_feats = sub[sub["present"] == "Yes"]

    if present_feats.empty:
        print(f"{city}: Demand features do NOT appear in the SHAP importance file (check feature naming).")
        continue

    lines = []
    for _, r in present_feats.iterrows():
        if r["shap_rank"] is not None:
            lines.append(f"- {city}: {r['feature']} appears in SHAP feature list (rank #{r['shap_rank']}).")
        else:
            lines.append(f"- {city}: {r['feature']} appears in SHAP feature list.")

    # Keep it short: print up to 2 bullets per city
    for line in lines[:2]:
        print(line)

print("\n(Use these bullets in your report as 'SHAP supports demand as a predictive driver' evidence.)")

# %% [markdown]
# ## Demand Surge → Delay Risk (Toronto vs NYC)
#
# - **Toronto:** Delay risk is most sensitive to **short-term demand spikes (last 30 minutes)**.  
#   Hazard ratio (HR = 0.887) indicates higher recent call volume increases delay risk.
#
# - **Toronto (60-min load):** Minimal practical impact due to narrow demand range  
#   (P10 = 0, P90 = 1); risk curves nearly overlap.
#
# - **NYC:** Per-unit hazard effects are small, but **large demand swings**  
#   (P10 = 45, P90 = 107) create visible delay-risk separation under sustained load.
#
# - **Risk curves confirm:** NYC shows consistent delay-risk increase under high rolling-hour demand; Toronto shows limited 60-min effect.
#
# - **SHAP alignment:** Demand variables rank among top predictive drivers in both cities  
#   (Toronto rank #5 for 30-min; NYC ranks #5/#6).
#
# - **Operational insight:** Toronto is **spike-sensitive** (short-term congestion),  
#   while NYC is **load-sensitive** (sustained high demand).
#
# - **Capacity implication:** NYC appears more vulnerable to rolling congestion;  
#   Toronto requires rapid surge handling.
#   
