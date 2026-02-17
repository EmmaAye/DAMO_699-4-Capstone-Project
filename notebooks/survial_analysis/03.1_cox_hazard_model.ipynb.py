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
# %pip install lifelines

# %% [markdown]
# ## cox_hazard_lib.py

# %%
# ------------------------------------------------------------
# Generic Cox Proportional Hazards library for emergency response time.
#
# Project rules:
# 1) If response_minutes is NULL -> treat as RIGHT-CENSORED at censor_time (default 60):
#       response_minutes = censor_time
#       event_indicator  = 0
# 2) hour should be modeled as CATEGORICAL via time-of-day bins:
#       time_bin in {Night, Morning, Afternoon, Evening}
#
# Supports: separate tables for NYC/Toronto (no city column needed).
#
# Requires: lifelines, pandas, numpy, pyspark
# ------------------------------------------------------------

# %%
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from pyspark.sql.functions import col, when


# %% [markdown]
# ### Data Prepartion

# %%
# ----------------------------
# Spark helpers
# ----------------------------

def add_time_of_day_bin(
    df_spark,
    hour_col: str = "hour",
    out_col: str = "time_bin",
):
    """
    Adds time-of-day bins:
      Night (0–5), Morning (6–11), Afternoon (12–17), Evening (18–23)
    """
    return df_spark.withColumn(
        out_col,
        when((col(hour_col) >= 0) & (col(hour_col) <= 5), "Night")
        .when((col(hour_col) >= 6) & (col(hour_col) <= 11), "Morning")
        .when((col(hour_col) >= 12) & (col(hour_col) <= 17), "Afternoon")
        .otherwise("Evening")
    )


# %% [markdown]
# ### Cox Design

# %% [markdown]
# For categorical predictors, the most frequent category was used as the reference level to improve interpretability and stability of coefficient estimates. This choice does not affect model predictions or overall fit but provides clearer comparisons across incident types and temporal factors.

# %% [markdown]
# **Helper Functions**

# %%
# ----------------------------
# Spark: load + base filtering
# ----------------------------

def load_cox_base_spark(
    spark,
    table_name: str,
    duration_col: str = "response_minutes",
    event_col: str = "event_indicator",
    hour_col: str = "hour",
    numeric_cols: list | None = None,
    categorical_cols: list | None = None,
    add_time_bin: bool = True,
    timebin_col: str = "time_bin",
):
    """
    Loads Spark table and selects needed columns.
    Keeps rows where duration is NULL (to be censored later).

    Drops:
      - rows where event is NULL
      - rows where event not in {0,1}
      - rows where duration <= 0 (when duration is not NULL)
      - optionally rows where hour is NULL if add_time_bin=True (since time_bin needs hour)
    """
    if numeric_cols is None:
        numeric_cols = []
    if categorical_cols is None:
        categorical_cols = []

    # If we will create time_bin, we need hour_col present
    base_cols = [duration_col, event_col]
    if add_time_bin:
        base_cols.append(hour_col)

    cols = base_cols + numeric_cols + categorical_cols

    df = spark.read.table(table_name).select(*cols)

    df = (
        df.filter(col(event_col).isNotNull())
          .filter(col(event_col).isin([0, 1]))
          .filter((col(duration_col).isNull()) | (col(duration_col) > 0))
    )

    if add_time_bin:
        df = df.filter(col(hour_col).isNotNull())
        df = add_time_of_day_bin(df, hour_col=hour_col, out_col=timebin_col)
        df = df.drop(hour_col) # drop hour_col since time_bin is used instead

    return df

def determine_reference_levels(pdf: pd.DataFrame, categorical_cols: list[str]) -> dict:
    """
    Determine most frequent category for each categorical column.
    Returns dict: {column: reference_category}
    """
    ref_map = {}

    for col in categorical_cols:
        counts = pdf[col].value_counts(dropna=True)

        if len(counts) == 0:
            continue

        ref_value = counts.idxmax()
        ref_map[col] = ref_value

    return ref_map



# %%
def identify_dummy_like_columns(pdf: pd.DataFrame, exclude: list[str]) -> list[str]:
    """
    Identify dummy-like 0/1 columns (bool or integer with values in {0,1}).
    We only drop low-frequency/low-variance among these, to avoid touching continuous predictors.
    """
    dummy_cols = []
    for c in pdf.columns:
        if c in exclude:
            continue

        s = pdf[c]
        if pd.api.types.is_bool_dtype(s):
            dummy_cols.append(c)
            continue

        if pd.api.types.is_integer_dtype(s):
            uniq = s.dropna().unique()
            if len(uniq) <= 2 and set(uniq).issubset({0, 1}):
                dummy_cols.append(c)

    return dummy_cols


def drop_low_freq_low_var_dummies(
    pdf: pd.DataFrame,
    dummy_cols: list[str],
    min_freq_rate: float = 0.001,   # 0.1%
    min_freq_abs: int = 50,
    min_var: float = 1e-8,
) -> tuple[pd.DataFrame, dict]:
    """
    Drops dummy-like columns that are:
      - ultra-rare or ultra-common (frequency based)
      - near-constant (variance based)

    Returns:
      pdf_out, details dict
    """
    n = len(pdf)
    min_count = max(min_freq_abs, int(min_freq_rate * n))

    low_freq_cols = []
    for c in dummy_cols:
        ones = int(pdf[c].sum())
        if ones < min_count or ones > (n - min_count):
            low_freq_cols.append(c)

    low_var_cols = []
    if dummy_cols:
        var = pdf[dummy_cols].var()
        low_var_cols = var[var < min_var].index.tolist()

    drop_cols = sorted(set(low_freq_cols + low_var_cols))
    pdf_out = pdf.drop(columns=drop_cols, errors="ignore") if drop_cols else pdf

    details = {
        "min_count_threshold": int(min_count),
        "min_var_threshold": float(min_var),
        "dropped_low_frequency": sorted(set(low_freq_cols)),
        "dropped_low_variance": sorted(set(low_var_cols)),
        "dropped_total": drop_cols,
        "n_features_before_drop": int(len(pdf.columns)),
        "n_features_after_drop": int(len(pdf_out.columns)),
    }
    return pdf_out, details


# %%
def build_cox_design(
    df_spark,
    duration_col: str,
    event_col: str,
    numeric_cols: list,
    categorical_cols: list,
    censor_time: float = 60.0,
    drop_first: bool = True,
    # dummy-drop configs
    min_freq_rate: float = 0.001,   # 0.1%
    min_freq_abs: int = 50,
    min_var: float = 1e-8,
) -> tuple[pd.DataFrame, dict]:
    """
    Spark -> pandas Cox design matrix with:
    - NULL duration censored at censor_time (duration=censor_time, event=0)
    - Validity filters: duration>0, event in {0,1}
    - Drop missing predictors (numeric + categorical)
    - One-hot encoding
    - Drop low-frequency & low-variance dummy-like columns
    Returns:
      (cox_df, drop_report, reference_map)
    """
    # -----------------------------
    # 1) Spark -> pandas
    # -----------------------------
    keep_cols = [duration_col, event_col] + list(numeric_cols) + list(categorical_cols)
    pdf = df_spark.select(*keep_cols).toPandas()

    # -----------------------------
    # 2) Type coercion
    # -----------------------------
    pdf[duration_col] = pd.to_numeric(pdf[duration_col], errors="coerce")
    pdf[event_col] = pd.to_numeric(pdf[event_col], errors="coerce")
    for c in numeric_cols:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

    # -----------------------------
    # 3) Censor NULL duration (core survival rule)
    # -----------------------------
    null_dur = pdf[duration_col].isna()
    n_censored_from_null = int(null_dur.sum())
    if n_censored_from_null > 0:
        pdf.loc[null_dur, duration_col] = float(censor_time)
        pdf.loc[null_dur, event_col] = 0

    # -----------------------------
    # 4) Validity filters (core survival rule)
    # -----------------------------
    pdf = pdf[pdf[duration_col] > 0].copy()
    pdf = pdf[pdf[event_col].isin([0, 1])].copy()

    # -----------------------------
    # 5) Drop missing predictors only (keep censored rows!)
    # -----------------------------
    pred_cols = list(numeric_cols) + list(categorical_cols)
    n_before_pred_drop = len(pdf)
    if pred_cols:
        pdf = pdf.dropna(subset=pred_cols).copy()
    n_rows_dropped_missing_predictors = n_before_pred_drop - len(pdf)

    # -------------------------------------------------
    # 6) DETERMINE BASELINES FROM FREQUENCY
    # -------------------------------------------------
    reference_map = determine_reference_levels(pdf, categorical_cols)

    # Apply categorical ordering so most frequent becomes baseline
    for col, ref in reference_map.items():
        categories = pdf[col].value_counts().index.tolist()

        # put reference first
        ordered = [ref] + [c for c in categories if c != ref]

        pdf[col] = pd.Categorical(pdf[col], categories=ordered)

    # -----------------------------
    # 7) One-hot encode categoricals
    # -----------------------------
    if categorical_cols:
        pdf = pd.get_dummies(pdf, columns=categorical_cols, drop_first=drop_first)

    # -----------------------------
    # 8) Drop rare/constant dummy columns (helpers)
    # -----------------------------
    exclude = [duration_col, event_col]
    dummy_cols = identify_dummy_like_columns(pdf, exclude=exclude)

    pdf2, drop_details = drop_low_freq_low_var_dummies(
        pdf,
        dummy_cols=dummy_cols,
        min_freq_rate=min_freq_rate,
        min_freq_abs=min_freq_abs,
        min_var=min_var,
    )

    # -------------------------
    # 9) Build drop report
    # -------------------------
    drop_report = {
        "n_rows_final": int(len(pdf2)),
        "n_features_final": int(len(pdf2.columns)),
        "n_censored_from_null_duration": int(n_censored_from_null),
        "n_rows_dropped_missing_predictors": int(n_rows_dropped_missing_predictors),
        "reference_categories": reference_map,
        **drop_details,
    }

    return pdf2, drop_report, reference_map


# %% [markdown]
# ### Cox Model (Fit + Outputs)

# %%
# ----------------------------
# Fit Cox + outputs
# ----------------------------

def fit_cox_model(
    cox_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    penalizer: float = 0.1,
):
    """
    Fits CoxPH model with L2 penalization for stability.
    """
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
    return cph


def hr_table(cph: CoxPHFitter, sort_by: str = "p") -> pd.DataFrame:
    """
    Hazard ratio table with 95% CI + p-values.
    HR > 1 => faster arrival
    HR < 1 => slower arrival (higher delay risk)
    """
    s = cph.summary.copy()

    # robustly find the CI columns (handles lifelines version differences)
    lower_col = [c for c in s.columns if ("lower" in c and "95" in c)][0]
    upper_col = [c for c in s.columns if ("upper" in c and "95" in c)][0]

    s["hazard_ratio"] = np.exp(s["coef"])
    s["hr_lower_95"] = np.exp(s[lower_col])
    s["hr_upper_95"] = np.exp(s[upper_col])

    out = (
        s[["hazard_ratio", "hr_lower_95", "hr_upper_95", "p", "coef", "se(coef)"]]
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=True).reset_index(drop=True)

    return out


def fit_stats(cph: CoxPHFitter) -> dict:
    return {
        "log_likelihood": float(cph.log_likelihood_),
        "partial_aic": float(cph.AIC_partial_),
        "concordance_index": float(cph.concordance_index_),
        "n": int(cph._n_examples),
    }


# %% [markdown]
# ### Run Cox Model

# %%
# ----------------------------
# One-call runner per table
# ----------------------------

def run_cox_for_table(
    spark,
    table_name: str,
    duration_col: str = "response_minutes",
    event_col: str = "event_indicator",
    hour_col: str = "hour",
    timebin_col: str = "time_bin",
    numeric_cols: list | None = None,
    categorical_cols: list | None = None,
    censor_time: float = 60.0,
    penalizer: float = 0.1,
    # dummy drop configs (passed into build_cox_design)
    min_freq_rate: float = 0.001,
    min_freq_abs: int = 50,
    min_var: float = 1e-8,
):
    """
    Full pipeline:
      Spark load -> create time_bin -> build design (NULL duration censored @ censor_time)
      -> drop low-freq/low-var dummy columns -> fit Cox -> HR + stats

    Default predictors:
      numeric: calls_past_30min, calls_past_60min
      categorical: time_bin, day_of_week, season, incident_category, unified_alarm_level
    """
    if numeric_cols is None:
        numeric_cols = ["calls_past_30min", "calls_past_60min"]

    if categorical_cols is None:
        categorical_cols = [
            "day_of_week",
            "season",
            "incident_category",
            "unified_alarm_level",
            timebin_col,
        ]

    # Ensure timebin_col appears only once
    categorical_cols = [c for c in categorical_cols if c != timebin_col] + [timebin_col]

    # Spark base: load columns (time_bin is created, not loaded)
    df_base = load_cox_base_spark(
        spark,
        table_name=table_name,
        duration_col=duration_col,
        event_col=event_col,
        hour_col=hour_col,
        numeric_cols=numeric_cols,
        categorical_cols=[c for c in categorical_cols if c != timebin_col],
        add_time_bin=True,
        timebin_col=timebin_col,
    )

    # Pandas design matrix + drop report
    cox_df, drop_report, reference_map= build_cox_design(
        df_base,
        duration_col=duration_col,
        event_col=event_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        censor_time=censor_time,
        drop_first=True,
        min_freq_rate=min_freq_rate,
        min_freq_abs=min_freq_abs,
        min_var=min_var,
    )

    # Fit Cox
    cph = fit_cox_model(cox_df, duration_col, event_col, penalizer=penalizer)

    return {
        "df_base_spark": df_base,
        "cox_df": cox_df,
        "drop_report": drop_report,
        "model": cph,
        "hr_table": hr_table(cph),
        "fit_stats": fit_stats(cph),
        "censor_time": float(censor_time),
        "timebin_col": timebin_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "reference_categories": reference_map,
    }


# %% [markdown]
# ## survival_main.py

# %%
from pyspark.sql import SparkSession
import pandas as pd
# from cox_hazard_lib import run_cox_for_table

# %%
spark = SparkSession.builder.getOrCreate()

# %% [markdown]
# ### NYC

# %%
TABLE_NAME = "workspace.capstone_project.nyc_model_ready"
LABEL = "NYC"

OUTPUT_DIR = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project/output/csv/"
HR_OUT = f"{OUTPUT_DIR}cox_hr_{LABEL}.csv"
STATS_OUT = f"{OUTPUT_DIR}cox_stats_{LABEL}.csv"

# %% [markdown]
# #### Run For Different Penalizers (0.1, 0.5, 1.0)

# %% [markdown]
# **penalizer = 0.1**

# %%
res = run_cox_for_table(
    spark,
    table_name=TABLE_NAME,
    censor_time=60.0,
    penalizer=0.1,
    categorical_cols=[
        "day_of_week",
        "season",
        "incident_category",
        "unified_alarm_level",
        "time_bin"
    ],
    numeric_cols=["calls_past_30min", "calls_past_60min"]
)

print("Fit stats:", res["fit_stats"])
display(res["hr_table"].head(30))

# %% [markdown]
# #### penalizer = 0.5

# %%
res = run_cox_for_table(
    spark,
    table_name=TABLE_NAME,
    censor_time=60.0,
    penalizer=0.5,
    categorical_cols=[
        "day_of_week",
        "season",
        "incident_category",
        "unified_alarm_level",
        "time_bin"
    ],
    numeric_cols=["calls_past_30min", "calls_past_60min"]
)

print("Fit stats:", res["fit_stats"])
display(res["hr_table"].head(30))

# %% [markdown]
# #### penalizer = 1

# %%
res = run_cox_for_table(
    spark,
    table_name=TABLE_NAME,
    censor_time=60.0,
    penalizer=1.0,
    categorical_cols=[
        "day_of_week",
        "season",
        "incident_category",
        "unified_alarm_level",
        "time_bin"
    ],
    numeric_cols=["calls_past_30min", "calls_past_60min"]
)

print("Fit stats:", res["fit_stats"])
display(res["hr_table"].head(30))

# %% [markdown]
# | penalizer | log-likelihood (higher is better) | partial AIC (lower is better) | concordance (higher is better) |
# | --------- | --------------------------------- | ----------------------------- | ------------------------------ |
# | **0.1**   | **-12,201,604**                   | **24,425,251**                | **0.6595**                     |
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
# #### Run With Best Penalizer and Results

# %%
res = run_cox_for_table(
    spark,
    table_name=TABLE_NAME,
    censor_time=60.0,
    penalizer=0.1,
    categorical_cols=[
        "day_of_week",
        "season",
        "incident_category",
        "unified_alarm_level",
        "time_bin"
    ],
    numeric_cols=["calls_past_30min", "calls_past_60min"]
)

print("Fit stats:", res["fit_stats"])
display(res["hr_table"].head(30))

# %%
print("Dropped columns:", res["drop_report"]["dropped_total"])
print("Alarm dummies left:", [c for c in res["cox_df"].columns if "unified_alarm_level" in c])
print("unified_alarm_level_3 present?", "unified_alarm_level_3" in res["cox_df"].columns)

# %%
spark.read.table(TABLE_NAME).groupBy("unified_alarm_level").count().orderBy("unified_alarm_level").show()

# %%
predictors = [c for c in res["cox_df"].columns if c not in ["response_minutes", "event_indicator"]]
print("Predictor columns:", predictors)
print("Num predictors:", len(predictors))

# %%
print("Reference categories used:")
print(res["reference_categories"])

# %%
print("unified_alarm_level_3 in cox_df?",
      "unified_alarm_level_3" in res["cox_df"].columns)

# optional: see counts if it exists
if "unified_alarm_level_3" in res["cox_df"].columns:
    print(res["cox_df"]["unified_alarm_level_3"].value_counts().head())


# %%
res["hr_table"].sort_values("hazard_ratio").head(10)

# %%
res["hr_table"].sort_values("hazard_ratio", ascending=False).head(10)

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
# #### Save Results

# %%
res["hr_table"].to_csv(HR_OUT, index=False)
pd.DataFrame([res["fit_stats"]]).to_csv(STATS_OUT, index=False)
print("Saved:", HR_OUT)
print("Saved:", STATS_OUT)

# %% [markdown]
# ### PH Test

# %%
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

duration_col = "response_minutes"
event_col = "event_indicator"

# 1) Sample for diagnostics
df_test = res["cox_df"].sample(n=200_000, random_state=42).copy()

# 2) Fit Cox on the SAME sample
cph_test = CoxPHFitter(penalizer=0.1)
cph_test.fit(df_test, duration_col=duration_col, event_col=event_col)

# 3) PH test on the SAME sample + model
ph_test = proportional_hazard_test(cph_test, df_test, time_transform="rank")

# 4) View most significant violations
ph_test.summary.sort_values("p").head(20)


# %% [markdown]
# Tests of the proportional hazards assumption indicated statistically significant deviations for several predictors, particularly incident type and time-of-day. Given the very large sample size, even minor time-varying effects were detected as statistically significant. In emergency-response settings, such deviations are expected due to dynamic dispatch prioritization and varying operational phases over the response timeline.
#
# The Cox model was therefore retained and interpreted as providing average hazard effects over time, consistent with prior emergency-service survival analyses. This approach allows meaningful comparison of delay-risk drivers across cities while maintaining interpretability.
