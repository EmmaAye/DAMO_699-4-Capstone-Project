# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # US5.3: Toronto vs NYC Baseline Survival + Hazard Comparison (Cox)
#
# This notebook/script compares baseline survival structure between Toronto and NYC using previously fitted Cox models.
# It focuses on structural timing differences only (no demand-scenario analysis; that is US5.4).

# %%
# !pip install lifelines

# %%
# %%
# Databricks sometimes needs this after installs
dbutils.library.restartPython()

# %% [markdown]
# ## 0. Environment Setup
#
# This section imports required libraries and prepares the analysis environment. Standard scientific Python packages are loaded for data handling, survival analysis outputs, and visualization. Paths for model artifacts and output figures are also initialized to ensure consistent file management within Databricks.

# %%
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Tuple

# %%
# Repo root (Databricks workspace path)
REPO_ROOT = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
#---------- Import Libraries from Project Code -----------#
from src.models.survival_analysis import (
# hazard utilities
get_baseline_hazard_series,
align_and_smooth_hazard,

# survival metrics
median_survival_time_interp,
survival_at_times,
hazard_peak,
get_concordance,

# plotting
plot_survival_overlay,
plot_hazard_overlay,
)

# %% [markdown]
# ## 1. Paths + Inputs

# %%
# ---- INPUTS (edit these) ----
BASE_PATH = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project/output"          # example root
MODEL_DIR  = f"{BASE_PATH}/models"
META_DIR   = f"{BASE_PATH}/models"
TABLE_DIR  = f"{BASE_PATH}/tables"

TOR_MODEL_PATH = f"{MODEL_DIR}/cph_Toronto.pkl"
NYC_MODEL_PATH = f"{MODEL_DIR}/cph_NYC.pkl"

TOR_META_PATH  = f"{META_DIR}/cox_meta_Toronto.json"
NYC_META_PATH  = f"{META_DIR}/cox_meta_NYC.json"

TOR_REFROW_PATH = f"{TABLE_DIR}/cox_reference_row_Toronto.csv"
NYC_REFROW_PATH = f"{TABLE_DIR}/cox_reference_row_NYC.csv"

# ---- OUTPUTS ----
FIG_DIR = f"{BASE_PATH}/graphs"                  # required output folder

# local driver path for matplotlib saving

print("Figure output dir:", FIG_DIR)


# %% [markdown]
# ## 2. Load Fitted Cox Models and Metadata
#
# Previously trained Cox proportional hazards models for Toronto and NYC are loaded from Databricks storage along with their associated metadata. The metadata contains model configuration details such as reference covariates, censoring horizon, and evaluation metrics.
#
# No model refitting is performed in this notebook; the loaded models are used solely for generating comparable survival and hazard estimates.

# %%
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def load_csv_optional(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

cph_tor = load_pickle(TOR_MODEL_PATH)
cph_nyc = load_pickle(NYC_MODEL_PATH)

meta_tor = load_json(TOR_META_PATH)
meta_nyc = load_json(NYC_META_PATH)

refrow_tor = load_csv_optional(TOR_REFROW_PATH)
refrow_nyc = load_csv_optional(NYC_REFROW_PATH)

print("Loaded models + metadata.")
print("Toronto meta keys:", sorted(meta_tor.keys()))
print("NYC meta keys:", sorted(meta_nyc.keys()))
print("Toronto ref row:", None if refrow_tor is None else refrow_tor.shape)
print("NYC ref row:", None if refrow_nyc is None else refrow_nyc.shape)

# %%
print(len(cph_tor.params_), len(cph_nyc.params_))
print(refrow_tor.shape[1], refrow_nyc.shape[1])

# %% [markdown]
# ## 3. Construct Reference Covariate Profiles
#
# Baseline survival curves require a standardized covariate configuration representing a typical system state. This section constructs a single reference observation for each city using:
#
# 1. Saved reference rows (if available),
# 2. Reference covariates defined in metadata, or
# 3. A neutral fallback configuration.
#
# This ensures survival estimates are comparable across cities under consistent baseline conditions.

# %%
# --------------------------------------------------
# Align baseline demand covariates across cities
# (Required for structural comparison in US5.3)
# --------------------------------------------------
DEMAND_COLS = ["calls_past_30min", "calls_past_60min"]

for df, name in [(refrow_tor, "Toronto"), (refrow_nyc, "NYC")]:
    if df is not None:
        for col in DEMAND_COLS:
            if col in df.columns:
                df.loc[:, col] = 0
        print(f"{name} demand covariates aligned to baseline (0).")


# %%
def make_reference_row(
    cph,
    meta: Dict,
    refrow_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Returns a 1-row DataFrame with columns aligned to cph.params_.index.
    Priority:
      1) refrow_df (first row)
      2) meta["reference_covariates"] dict
      3) zeros
    """
    cols = list(cph.params_.index)

    if refrow_df is not None and len(refrow_df) > 0:
        row = refrow_df.iloc[[0]].copy()
        # align columns
        for c in cols:
            if c not in row.columns:
                row[c] = 0.0
        row = row[cols]
        return row

    ref_dict = meta.get("reference_covariates") or meta.get("reference_covariate_values")
    if isinstance(ref_dict, dict) and len(ref_dict) > 0:
        row = pd.DataFrame([{c: ref_dict.get(c, 0.0) for c in cols}])
        return row

    # last resort
    return pd.DataFrame([np.zeros(len(cols))], columns=cols)

x_tor = make_reference_row(cph_tor, meta_tor, refrow_tor)
x_nyc = make_reference_row(cph_nyc, meta_nyc, refrow_nyc)

display(x_tor.head())
display(x_nyc.head())

# %%
meta_tor["reference_categories"]


# %% [markdown]
# ## 4. Align Analysis Time Horizon
#
# To enable direct comparison, both cities are evaluated over a shared time range defined by the censoring horizon stored in model metadata. The analysis timeline spans from time zero to the minimum censor time across cities, ensuring survival and hazard estimates are computed on identical axes.
#
# We’ll align both cities to the same horizon:
#
# - start at 0
# - end at `min(censor_time_toronto, censor_time_nyc)` (or a fallback like 60)

# %%
def get_censor_time(meta: Dict, fallback: float = 60.0) -> float:
    # support a few common key names
    for k in ["censor_time", "censor_minutes", "time_horizon", "max_time"]:
        if k in meta and meta[k] is not None:
            return float(meta[k])
    return float(fallback)

censor_tor = get_censor_time(meta_tor, fallback=60.0)
censor_nyc = get_censor_time(meta_nyc, fallback=60.0)

T_MAX = min(censor_tor, censor_nyc)
timeline = np.linspace(0, T_MAX, int(T_MAX) + 1)  # 1-minute increments

print("censor_tor:", censor_tor, "censor_nyc:", censor_nyc, "T_MAX:", T_MAX, "len(timeline):", len(timeline))

# %%
timeline


# %% [markdown]
# ## 5. Generate Baseline Survival Curves
#
# Baseline survival functions are generated using the fitted Cox models and the reference covariate profiles. These curves represent the probability that an incident has not yet reached completion (or arrival) by a given time under typical system conditions.
#
# The resulting survival series form the foundation for cross-city structural comparison.

# %%
def baseline_survival(cph, x_ref: pd.DataFrame, timeline: np.ndarray) -> pd.Series:
    sf = cph.predict_survival_function(x_ref, times=timeline)
    # lifelines returns a DF with one column per row in x_ref; we have 1 row
    s = sf.iloc[:, 0]
    s.index.name = "time"
    s.name = "S(t)"
    return s

S_tor = baseline_survival(cph_tor, x_tor, timeline)
S_nyc = baseline_survival(cph_nyc, x_nyc, timeline)

display(S_tor.head())
display(S_nyc.head())


# %%
def survival_sanity_check(S, name=""):
    print(f"\n--- {name} ---")

    print("Start value S(0):", S.iloc[0])
    print("End value S(T):", S.iloc[-1])

    # Survival must be non-increasing
    monotonic = (S.diff().dropna() <= 1e-10).all()
    print("Monotonic decreasing:", monotonic)

    # Survival must stay within [0,1]
    in_range = ((S >= 0) & (S <= 1)).all()
    print("Within [0,1]:", in_range)

    # Any NaNs?
    print("Contains NaN:", S.isna().any())

survival_sanity_check(S_tor, "Toronto")
survival_sanity_check(S_nyc, "NYC")

# %% [markdown]
# ## 6. Cross-City Survival Curve Overlay Plot
#
# Survival curves for Toronto and NYC are plotted on a shared axis to visualize differences in response-time decay patterns. Overlaying the curves highlights relative timing behavior, including how quickly events progress toward completion in each system.
#
# The resulting figure is saved for inclusion in reports and dashboard visualizations.

# %%
surv_fig = os.path.join(FIG_DIR, "final_cox_baseline_toronto_vs_nyc.png")
plot_survival_overlay(S_tor, S_nyc, "Toronto", "NYC", surv_fig)

print("Saved:", surv_fig)

# %%
print("Toronto final survival:", S_tor.iloc[-1])
print("NYC final survival:", S_nyc.iloc[-1])

# %% [markdown]
# ## 6. Estimate Baseline Hazard Functions
#
# Baseline hazard functions are estimated from each fitted Cox proportional hazards model to quantify the instantaneous event (completion) rate over time. The hazard series are aligned to a shared timeline to enable direct cross-city comparison, and an optional rolling-window smoothing step is applied to improve interpretability while preserving overall structural patterns.
#
# These curves describe how response intensity evolves throughout the observation horizon and complement the baseline survival curves by highlighting when completion activity is most concentrated.
#
# In `lifelines` (`CoxPHFitter`), baseline hazards are obtained from model outputs such as:
#
# - `cph.baseline_hazard_` (hazard evaluated at observed event times), or  
# - `cph.baseline_cumulative_hazard_`.
#
# The implementation therefore:
# 1. extracts the estimated baseline hazard from each fitted model,
# 2. reindexes the hazard onto a shared timeline (0–T_MAX),
# 3. optionally applies smoothing for visualization purposes.

# %%
h0_tor_raw = get_baseline_hazard_series(cph_tor)
h0_nyc_raw = get_baseline_hazard_series(cph_nyc)

h0_tor = align_and_smooth_hazard(h0_tor_raw, timeline, smooth_window=5)
h0_nyc = align_and_smooth_hazard(h0_nyc_raw, timeline, smooth_window=5)

print("Tortonto Hazard Series:")
display(h0_tor.head(10))
print("\n NYC Hazard Series:")
display(h0_nyc.head(10))

# %%
print("Toronto peak hazard:", float(h0_tor.max()), "at time:", float(h0_tor.idxmax()))
print("NYC peak hazard:", float(h0_nyc.max()), "at time:", float(h0_nyc.idxmax()))

# %% [markdown]
# ### Hazard Function Comparison
#
# Baseline hazard functions further clarify structural timing differences between cities. Toronto exhibits a rapid increase in hazard during the early response window, reaching a higher peak magnitude around 7 minutes before declining quickly. This pattern indicates a concentrated period of incident completion shortly after initiation, consistent with the steep early decline observed in the survival curve.
#
# In contrast, NYC shows a lower and more gradually evolving hazard profile, peaking earlier but at substantially smaller magnitude and decaying more slowly over time. This suggests that incident completions are distributed across a wider time range rather than concentrated in an early response window.
#
# Together, the hazard patterns explain the survival curve differences: Toronto’s higher early hazard produces rapid completion clustering, whereas NYC’s lower and broader hazard structure results in a longer response-time tail.

# %% [markdown]
# ## 8. Cross-City Hazard Comparison
#
# Baseline hazard functions for both cities are overlaid to compare:
#
# - timing of peak response intensity,
# - magnitude of instantaneous response rates, and
# - overall decay behavior.
#
# This visualization complements survival curves by revealing differences in when system activity is most concentrated.

# %%
haz_fig = os.path.join(FIG_DIR, "final_hr_cox_overlay.png")
plot_hazard_overlay(h0_tor, h0_nyc, "Toronto", "NYC", haz_fig)
print("Saved:", haz_fig)

# %% [markdown]
# ### Cross-City Hazard Comparison Summary 
# Baseline hazard functions derived from fitted Cox proportional hazards models under standardized reference covariate conditions. Toronto exhibits a higher and sharper hazard peak (0.0078 at 7 minutes), indicating a strongly front-loaded response structure with rapid early incident completion. NYC shows a lower and broader peak (0.0033 at 6 minutes), suggesting a more distributed response process and longer completion tail. Differences reflect structural timing dynamics rather than demand-driven effects.

# %% [markdown]
# ## 9. Quantitative Survival and Hazard Metrics
#
# Key quantitative indicators are computed to support objective cross-city comparison, including:
#
# - **Median survival time** (first time at which S(t) ≤ 0.5; reported as NaN if not reached within the observation horizon)
# - **Survival probabilities S(t)** at selected time points (e.g., 5, 10, 15, 30, 45, and 60 minutes, clipped to T_MAX)
# - **Hazard peak timing and magnitude**
# - **Concordance index**, obtained from model metadata (or from the fitted model when available)
#
# A summary comparison table is generated and saved for downstream reporting and dashboard integration.

# %%
selected_times = [5, 10, 15, 30, 45, 60]
selected_times = [t for t in selected_times if t <= T_MAX]

tor_med = median_survival_time_interp(S_tor)
nyc_med = median_survival_time_interp(S_nyc)

tor_peak_t, tor_peak_h = hazard_peak(h0_tor)
nyc_peak_t, nyc_peak_h = hazard_peak(h0_nyc)

hazard_peak_ratio = tor_peak_h / nyc_peak_h if nyc_peak_h != 0 else float("inf")

row_tor = {
    "city": "Toronto",
    "median_survival_min": tor_med,
    "hazard_peak_time_min": tor_peak_t,
    "hazard_peak_magnitude": tor_peak_h,
    "concordance_index": get_concordance(meta_tor, cph_tor),
    **survival_at_times(S_tor, selected_times)
}

row_nyc = {
    "city": "NYC",
    "median_survival_min": nyc_med,
    "hazard_peak_time_min": nyc_peak_t,
    "hazard_peak_magnitude": nyc_peak_h,
    "concordance_index": get_concordance(meta_nyc, cph_nyc),
    **survival_at_times(S_nyc, selected_times)
}

row_tor["S(T_MAX)"] = float(S_tor.iloc[-1])
row_nyc["S(T_MAX)"] = float(S_nyc.iloc[-1])

summary_df_base = pd.DataFrame([row_tor, row_nyc]).set_index("city")
diff_row = (summary_df_base.loc["NYC"] - summary_df_base.loc["Toronto"]).to_frame().T
diff_row.index = ["Difference (NYC − Toronto)"]
summary_df_base["hazard_peak_ratio_Toronto_over_NYC"] = np.nan
diff_row["hazard_peak_ratio_Toronto_over_NYC"] = hazard_peak_ratio

summary_df = pd.concat([summary_df_base, diff_row]).reset_index().rename(columns={"index":"city"})

summary_df

# %%
OUT_TABLE_PATH = f"{TABLE_DIR}/cox_cross_city_survival_hazard_summary.csv"

summary_df.to_csv(OUT_TABLE_PATH, index=False)

print("Saved summary table:", OUT_TABLE_PATH)

# %%
# sanity:
print("Hazard peak ratio:", hazard_peak_ratio)
print("Toronto median:", tor_med, "NYC median:", nyc_med)
print("Toronto S(T_MAX):", row_tor["S(T_MAX)"], "NYC S(T_MAX):", row_nyc["S(T_MAX)"])


# %% [markdown]
# ### Quantitative Comparison of Baseline Survival and Hazard Metrics
#
# This summarizes key quantitative metrics derived from the Cox baseline survival and hazard functions for Toronto and NYC under standardized reference conditions. Toronto demonstrates substantially faster structural response timing, with a median baseline survival time of 4.9 minutes compared with 10.4 minutes in NYC. Survival probabilities decline much more rapidly in Toronto across all evaluated time points, while NYC maintains higher survival levels throughout the observation horizon (S(T_MAX)=0.457 vs 0.012), indicating a longer response-time tail.
#
# Hazard-based metrics show that Toronto reaches a higher peak hazard magnitude (0.0078 at 7 minutes) than NYC (0.0033 at 6 minutes). The peak hazard ratio (≈2.38) indicates that the instantaneous completion intensity in Toronto is more than twice that of NYC under comparable baseline conditions. Model concordance values provide contextual performance information, with higher discrimination observed for NYC (0.660) relative to Toronto (0.567).
#
# Overall, the quantitative metrics confirm substantial structural differences in timing dynamics between the two cities, with Toronto exhibiting a more front-loaded completion pattern and NYC showing a more gradual response distribution.

# %% [markdown]
# ## 10. Structural Interpretation
#
# A concise interpretation is generated based on survival and hazard characteristics. The discussion focuses on structural timing differences between cities, including response concentration and delay distribution patterns.
#
# This interpretation reflects baseline system behavior only and intentionally excludes demand-driven scenario analysis.

# %%
def structural_interpretation(summary_df: pd.DataFrame, selected_times: List[float]) -> str:
    tor = summary_df[summary_df["city"]=="Toronto"].iloc[0].to_dict()
    nyc = summary_df[summary_df["city"]=="NYC"].iloc[0].to_dict()

    # helper
    def fmt(x, nd=2):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{x:.{nd}f}"

    lines = []
    lines.append("Structural baseline comparison (Toronto vs NYC)")
    lines.append("")
    lines.append(f"- Median baseline survival time (time until S(t) drops to 0.5): Toronto={fmt(tor['median_survival_min'],1)} min, NYC={fmt(nyc['median_survival_min'],1)} min.")
    lines.append(f"- Baseline hazard peak (highest instantaneous completion/arrival rate): Toronto peaks at {fmt(tor['hazard_peak_time_min'],1)} min (h0={fmt(tor['hazard_peak_magnitude'],6)}), NYC peaks at {fmt(nyc['hazard_peak_time_min'],1)} min (h0={fmt(nyc['hazard_peak_magnitude'],6)}).")

    # survival levels at fixed times
    st_bits = []
    for t in selected_times:
        st_bits.append(f"S({int(t)}m): Toronto={fmt(tor[f'S({int(t)}m)'],3)}, NYC={fmt(nyc[f'S({int(t)}m)'],3)}")
    lines.append("- Survival levels at fixed time points: " + "; ".join(st_bits) + ".")

    # interpret directionally
    # Note: In survival-of-response-time framing, lower S(t) earlier typically implies faster completion/arrival.
    lines.append("")
    lines.append("Interpretation:")

    lines.append(
        f"- Toronto exhibits a substantially faster early decline in survival probability than NYC "
        f"(median survival: {fmt(tor['median_survival_min'],1)} vs {fmt(nyc['median_survival_min'],1)} minutes), "
        "indicating a more front-loaded response structure under baseline conditions."
    )

    lines.append(
        f"- The baseline hazard peak is notably higher in Toronto "
        f"(h₀={fmt(tor['hazard_peak_magnitude'],6)}) compared with NYC "
        f"(h₀={fmt(nyc['hazard_peak_magnitude'],6)}), corresponding to approximately "
        f"{fmt(tor['hazard_peak_magnitude']/nyc['hazard_peak_magnitude'],2)}× greater peak intensity. "
        "This suggests stronger early event completion dynamics in Toronto, whereas NYC responses are "
        "distributed across a longer time horizon."
    )

    lines.append(
        f"- By the end of the observation window (T_MAX), survival probability remains substantially higher "
        f"in NYC (S={fmt(nyc['S(T_MAX)'],3)}) than in Toronto (S={fmt(tor['S(T_MAX)'],3)}), "
        "indicating a heavier long-duration tail in NYC response times."
    )

    lines.append(
        "- Because survival estimates are evaluated under standardized reference-category covariates, "
        "these differences reflect structural timing characteristics of the response systems rather than "
        "demand-driven scenario effects (examined separately in US5.4)."
    )

    return "\n".join(lines)

writeup = structural_interpretation(summary_df, selected_times)
print(writeup)

# %%
# Databricks notebook cell

WRITEUP_PATH = f"{TABLE_DIR}/cox_structural_baseline_comparison_writeup.txt"
with open(WRITEUP_PATH, "w") as f:
    f.write(writeup)

print("Saved write-up:", WRITEUP_PATH)
