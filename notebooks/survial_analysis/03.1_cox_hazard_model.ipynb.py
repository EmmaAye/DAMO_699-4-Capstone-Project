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
# # Cox Proportional Hazards — Cross-City Modeling (Toronto & NYC)
#
# This notebook fits Cox proportional hazards models for Toronto and NYC using a shared modeling pipeline.
# It reuses library functions in `src.models.survival_analysis.cox_hazard_lib` to ensure:
# - consistent censoring rules (NULL duration -> censored at 60)
# - consistent time-of-day binning
# - consistent feature preparation and encoding
#
# Outputs:
# - Hazard ratio (HR) tables per city
# - Model fit stats per city
# - Optional: PH assumption diagnostics (sample-based)
#
# Note:
# - Saving CSV/JSON is optional; it helps reproducibility and report integration, but plots/tables can also be generated on-demand.

# %%
# %pip install lifelines

# %%
# %%
# Databricks sometimes needs this after installs
dbutils.library.restartPython()

# %% [markdown]
# ## 0. Imports & Setup

# %%
import os
import json
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession

# lifelines PH test (optional)
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# Make sure repo root is importable
import sys
REPO_ROOT = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.models.survival_analysis.cox_hazard_lib import     run_cox_for_table

# %% [markdown]
# ## 1. MODEL and PATH Configurations

# %%
# %%
spark = SparkSession.builder.getOrCreate()

# Inputs (Spark tables)
TORONTO_TABLE = "workspace.capstone_project.toronto_model_ready"
NYC_TABLE     = "workspace.capstone_project.nyc_model_ready"

# Common modeling config
CENSOR_TIME = 60.0
PENALIZERS = [0.1, 0.5, 1.0]   # can reduce later
BEST_PENALIZER = 0.1          # default based on your earlier runs

NUMERIC_COLS = ["calls_past_30min", "calls_past_60min"]
CATEGORICAL_COLS = [
    "day_of_week",
    "season",
    "incident_category",
    "unified_alarm_level",
    "time_bin",
]

# %% [markdown]
# Enable/Disable Output Save Option

# %%
# Toggle to To SAVE Output Files on/off
DO_SAVE = True

# %% [markdown]
# Enable/Disable PH Test Option

# %%
DO_PH_TEST = True

# %% [markdown]
# ### Output path

# %%
BASE_OUTPUT_DIR = f"{REPO_ROOT}/output"
CSV_DIR   = f"{BASE_OUTPUT_DIR}/tables"
MODEL_DIR = f"{BASE_OUTPUT_DIR}/models"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def convert_numpy(obj):
    """Safely convert numpy types inside dicts for JSON."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


# %% [markdown]
# ## 2. Fit Cox models (grid over penalizer)
# We fit multiple penalizers for stability comparison, then keep the best.

# %%
# %%
def fit_city(table_name: str, label: str):
    results = {}
    for pen in PENALIZERS:
        print(f"\n=== {label} | penalizer={pen} ===")
        res = run_cox_for_table(
            spark,
            table_name=table_name,
            censor_time=CENSOR_TIME,
            penalizer=pen,
            categorical_cols=CATEGORICAL_COLS,
            numeric_cols=NUMERIC_COLS,
        )
        print("Fit stats:", res["fit_stats"])
        display(res["hr_table"].head(20))
        results[pen] = res
    return results



# %% [markdown]
# ### 2.1 NYC

# %%
nyc_runs = fit_city(NYC_TABLE, "NYC")

# %% [markdown]
# | penalizer | log-likelihood (higher is better) | partial AIC (lower is better) | concordance (higher is better) |
# | --------- | --------------------------------- | ----------------------------- | ------------------------------ |
# | **0.1**   | **-12,212,604**                   | **24,425.251**                | **0.6595**                     |
# | 0.5       | -12,274,869                       | 24,549,781                    | 0.6593                         |
# | 1.0       | -12,303,835                       | 24,607,712                    | 0.6586                         |
#
# **Interpretation**:
#
# **Penalizer 0.1 is clearly best on all metrics**
# - highest log-likelihood
# - lowest AIC
# - highest concordance
#
# As penalizer increases:
#
# - model gets more shrinkage
#
# - coefficients shrink toward zero
#
# - fit gets worse (AIC ↑, concordance ↓)
#
# Therefore:
#
# **Best penalizer = 0.1**

# %% [markdown]
# ### 2.2 Toronto

# %%
tor_runs = fit_city(TORONTO_TABLE, "Toronto")


# %% [markdown]
# | penalizer | log-likelihood (higher is better) | partial AIC (lower is better) | concordance (higher is better) |
# | --------- | --------------------------------- | ----------------------------- | ------------------------------ |
# | **0.1**   | **-4,151,324**                   | **8,302,691**                | **0.5669**                     |
# | 0.5       | -4,154,786                       |8,309,614                    | 0.5656                         |
# | 1.0       | -4,156,836                       |8,313,714                    | 0.5647                         |
#
# **Interpretation**:
#
# **Penalizer 0.1 is clearly best on all metrics**
# - highest log-likelihood
# - lowest AIC
# - highest concordance
#
# As penalizer increases:
#
# - model gets more shrinkage
#
# - coefficients shrink toward zero
#
# - fit gets worse (AIC ↑, concordance ↓)
#
# Therefore:
#
# **Best penalizer = 0.1**

# %% [markdown]
# ## 3. Run With Best Penalizer and Results

# %% [markdown]
# Helper Functions

# %%
def sanity_check(res, label):
    cox_df = res["cox_df"]
    print(f"\n=== {label} SANITY CHECK ===")
    print("Rows:", len(cox_df))
    print("Censored (event=0):", int((cox_df["event_indicator"] == 0).sum()))
    print("Event observed (event=1):", int((cox_df["event_indicator"] == 1).sum()))
    print("Max duration:", float(cox_df["response_minutes"].max()))
    print("Min duration:", float(cox_df["response_minutes"].min()))
    print("Concordance:", res["fit_stats"]["concordance_index"])


# %% [markdown]
# ### 3.1 NYC

# %%
nyc_final = nyc_runs[BEST_PENALIZER]
print("Fit stats:", nyc_final["fit_stats"])
display(nyc_final["hr_table"].head(30))

# %% [markdown]
# #### Sanity Checks

# %%
sanity_check(nyc_final, "NYC")

# %%
spark.read.table(NYC_TABLE).groupBy("unified_alarm_level").count().orderBy("unified_alarm_level").show()

# %%
nyc_predictors = [c for c in nyc_final["cox_df"].columns if c not in ["response_minutes", "event_indicator"]]
print("NYC Predictor columns:", nyc_predictors)
print("Num predictors:", len(nyc_predictors))

# %%
print("NYC Reference categories used:")
print(nyc_final["reference_categories"])

# %% [markdown]
# #### Analyze Hazard Ratio

# %%
nyc_final["hr_table"].sort_values("hazard_ratio").head(10)

# %%
nyc_final["hr_table"].sort_values("hazard_ratio", ascending=False).head(10)

# %% [markdown]
# Categorical variables were encoded using the most frequent category as the reference level to improve interpretability and stability of estimates. The Cox proportional hazards model was used to estimate average hazard effects over time. While formal tests indicated deviations from the proportional hazards assumption, such deviations are expected in large-scale emergency response data and do not materially affect the identification of key delay-risk drivers.
#
# **Rule**
# | Hazard Ratio (HR) | Interpretation   |
# | ----------------- | ---------------------------------- |
# | **> 1**           | faster arrival                     |
# | **< 1**           | slower arrival (higher delay risk) |
# | **= 1**           | no difference from baseline        |

# %% [markdown]
# #### Survival Model Summary (NYC)
# The model answers: Compared to a typical medical call in the afternoon during summer with alarm level 1, how does each factor change arrival speed?
#
# A Cox proportional hazards model was estimated to examine factors associated with the speed of first-unit arrival. Categorical predictors were encoded using the most frequent category as the reference level (Medical incidents, alarm level 1, afternoon period, summer season, and the most common day of week). This allows all hazard ratios to be interpreted relative to a typical emergency response context.
#
# Results show that **incident type and alarm severity are the strongest drivers of arrival speed**. Compared with medical incidents (baseline), fire incidents are associated with substantially faster arrival: structural fires have more than four times the arrival hazard (HR≈4.63), and non-structural fires more than three times (HR≈3.47). Other high-priority incident types—including hazardous/utility events and rescues—also exhibit significantly faster response. Similarly, incidents with alarm level 2 have approximately double the arrival hazard relative to alarm level 1 (HR≈2.07), indicating prioritization of higher-severity calls.
#
# Temporal factors have smaller but statistically significant effects. Compared with afternoon responses (baseline), evening incidents show slightly faster arrivals (HR≈1.05), while day-of-week and seasonal differences are modest (e.g., fall vs. summer HR≈1.03). Overall, these findings suggest that **operational priority and incident characteristics dominate response-time variation**, while temporal and seasonal effects play a secondary role.
#
# The model demonstrates reasonable predictive performance (concordance ≈ 0.66), indicating meaningful structure in response-time risk. Although tests indicated some deviations from the proportional hazards assumption—expected given the large sample size and dynamic dispatch environment—the Cox model remains appropriate for estimating average effects and identifying key drivers of response-time delays.
#

# %% [markdown]
# ### 3.2 Toronto

# %%
tor_final = tor_runs[BEST_PENALIZER]
print("Fit stats:", tor_final["fit_stats"])
display(tor_final["hr_table"].head(30))

# %% [markdown]
# #### Sanity Check

# %%
sanity_check(tor_final, "Toronto")

# %%
predictors = [c for c in tor_final["cox_df"].columns if c not in ["response_minutes", "event_indicator"]]
print("Toronto Predictor columns:", predictors)
print("Num predictors:", len(predictors))

# %%
print("Toronto Reference categories used:")
print(tor_final["reference_categories"])

# %% [markdown]
# #### Analyze Hazard Ratio

# %%
tor_final["hr_table"].sort_values("hazard_ratio").head(10)

# %%
tor_final["hr_table"].sort_values("hazard_ratio", ascending=False).head(10)


# %% [markdown]
# #### Survival Model Summary (Toronto)
#
# A Cox proportional hazards model was estimated to examine factors associated with first-unit arrival times in Toronto. Categorical variables were encoded using the most frequent category as the reference level (Medical incidents, alarm level 1, afternoon period, summer season, and the most common day of week). Hazard ratios therefore represent differences in arrival speed relative to a typical medical incident occurring under standard conditions.
#
# Overall model performance was moderate (concordance ≈ 0.57), indicating that while meaningful structure exists in Toronto response-time variation, predictive patterns are weaker than those observed in the NYC model. Among the predictors, **alarm severity and short-term demand intensity** were the most influential drivers of arrival speed. Incidents with alarm level 2 exhibited faster response compared with level 1 incidents (HR≈1.24), suggesting prioritization of higher-severity calls. Higher recent call volume also showed a measurable association with faster arrival (calls_past_60min HR≈1.05), likely reflecting heightened operational activity and resource deployment during busier periods.
#
# Temporal factors demonstrated smaller but statistically significant effects. Compared with afternoon responses (baseline), evening and morning incidents showed slightly faster arrivals (HR≈1.02 and HR≈1.01, respectively), while seasonal and day-of-week variations were modest. Incident-type effects were present but less pronounced than in NYC; for example, non-structural fire incidents were only marginally faster than medical calls (HR≈1.03). These findings suggest that, in Toronto, **operational demand levels and scheduling factors play a more prominent role than incident type in shaping response-time variability**, and that differences across categories are comparatively moderate.
#
# As with the NYC analysis, formal tests indicated some deviations from the proportional hazards assumption, which is expected given the large dataset and dynamic dispatch environment. The Cox model is therefore interpreted as providing average hazard effects over time and remains appropriate for identifying key factors associated with response-time variation across operational conditions.
#

# %% [markdown]
# ## 4. Save results (optional)
# Saves:
# - HR tables as CSV
# - fit stats as CSV
# - metadata as JSON (useful for auditability / reproducibility)

# %%
def save_city_outputs(res, label: str, table_name: str, penalizer: float):
    hr_out = f"{CSV_DIR}/cox_hr_{label}.csv"
    stats_out = f"{CSV_DIR}/cox_stats_{label}.csv"
    meta_out = f"{MODEL_DIR}/cox_meta_{label}.json"

    res["hr_table"].to_csv(hr_out, index=False)
    pd.DataFrame([res["fit_stats"]]).to_csv(stats_out, index=False)

    meta = {
        "table_name": table_name,
        "label": label,
        "censor_time": res["censor_time"],
        "penalizer": penalizer,
        "numeric_cols": res["numeric_cols"],
        "categorical_cols": res["categorical_cols"],
        "reference_categories": res["reference_categories"],
        "drop_report": res["drop_report"],
        "fit_stats": res["fit_stats"],
        "created_at": datetime.now().isoformat(),
    }

    with open(meta_out, "w") as f:
        json.dump(convert_numpy(meta), f, indent=2)

    print("Saved:", hr_out)
    print("Saved:", stats_out)
    print("Saved:", meta_out)


# %%
if DO_SAVE:
    save_city_outputs(nyc_final, "NYC", NYC_TABLE, BEST_PENALIZER)
    save_city_outputs(tor_final, "Toronto", TORONTO_TABLE, BEST_PENALIZER)


# %% [markdown]
# ## 5. Proportional Hazards (PH) test (optional diagnostic)
# Large datasets will almost always show statistically significant PH violations.
# We treat this as a diagnostic, not a reason to discard the model.
#

# %%
def ph_test_sample(
    res,
    label: str,
    sample_n: int = 200_000,
    penalizer: float = 0.1,
):
    duration_col = "response_minutes"
    event_col = "event_indicator"

    df_full = res["cox_df"]
    df = df_full.sample(n=min(sample_n, len(df_full)), random_state=42).copy()

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    test = proportional_hazard_test(cph, df, time_transform="rank")
    out = test.summary.sort_values("p").copy()

    print(f"\n=== PH TEST | {label} ===")
    print("Rows used:", len(df))
    print("Significant violations (p<0.05):", (out["p"] < 0.05).sum())
    display(out.head(20))

    return out


# %%
if DO_PH_TEST:
    nyc_ph = ph_test_sample(nyc_final, "NYC")
    tor_ph = ph_test_sample(tor_final, "Toronto")

    nyc_ph.to_csv(f"{MODEL_DIR}/cox_ph_test_NYC.csv")
    tor_ph.to_csv(f"{MODEL_DIR}/cox_ph_test_Toronto.csv")

# %% [markdown]
# **Proportional Hazards Assumption Diagnostics**
#
# Tests of the proportional hazards assumption were conducted using Schoenfeld-residual-based tests on random samples of 200,000 observations for each city. Results indicated statistically significant deviations for several predictors in both models. In the NYC model, 15 predictors exhibited significant non-proportionality (p < 0.05), while 13 predictors showed similar deviations in the Toronto model. Several predictors produced extremely small p-values, which is expected given the large sample sizes and the sensitivity of formal tests.
#
# In emergency-response settings, strict proportional hazards rarely hold because dispatch prioritization, congestion effects, and operational dynamics change over the response timeline. With large observational datasets, even minor time-varying effects can be detected as statistically significant without materially affecting model interpretation. The Cox model was therefore retained and interpreted as estimating average hazard effects over time, which remains appropriate for identifying key drivers of response-time variation across operational conditions.
