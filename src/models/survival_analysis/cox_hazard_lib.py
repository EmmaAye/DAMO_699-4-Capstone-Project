"""
cox_hazard_lib.py

Reusable Cox Proportional Hazards utilities for emergency response-time survival modeling.

Project rules:
1) If response_minutes is NULL -> RIGHT-CENSORED at censor_time (default 60):
      response_minutes = censor_time
      event_indicator  = 0
2) hour should be modeled as CATEGORICAL via time-of-day bins:
      time_bin in {Night, Morning, Afternoon, Evening}

Dependencies:
  - lifelines
  - pandas, numpy
  - pyspark
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, when


# ----------------------------
# Spark helpers
# ----------------------------

def add_time_of_day_bin(
    df_spark: SparkDataFrame,
    hour_col: str = "hour",
    out_col: str = "time_bin",
) -> SparkDataFrame:
    """
    Adds time-of-day bins:
      Night (0–5), Morning (6–11), Afternoon (12–17), Evening (18–23)
    """
    return df_spark.withColumn(
        out_col,
        when((col(hour_col) >= 0) & (col(hour_col) <= 5), "Night")
        .when((col(hour_col) >= 6) & (col(hour_col) <= 11), "Morning")
        .when((col(hour_col) >= 12) & (col(hour_col) <= 17), "Afternoon")
        .otherwise("Evening"),
    )


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
) -> SparkDataFrame:
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
        df = df.drop(hour_col)

    return df


# ----------------------------
# Pandas helpers (design matrix)
# ----------------------------

def determine_reference_levels(pdf: pd.DataFrame, categorical_cols: list[str]) -> dict:
    """
    Determine most frequent category for each categorical column.
    Returns dict: {column: reference_category}
    """
    ref_map = {}
    for c in categorical_cols:
        counts = pdf[c].value_counts(dropna=True)
        if len(counts) == 0:
            continue
        ref_map[c] = counts.idxmax()
    return ref_map


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


def build_cox_design(
    df_spark: SparkDataFrame,
    duration_col: str,
    event_col: str,
    numeric_cols: list,
    categorical_cols: list,
    censor_time: float = 60.0,
    drop_first: bool = True,
    min_freq_rate: float = 0.001,
    min_freq_abs: int = 50,
    min_var: float = 1e-8,
) -> tuple[pd.DataFrame, dict, dict]:
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
    keep_cols = [duration_col, event_col] + list(numeric_cols) + list(categorical_cols)
    pdf = df_spark.select(*keep_cols).toPandas()

    # Type coercion
    pdf[duration_col] = pd.to_numeric(pdf[duration_col], errors="coerce")
    pdf[event_col] = pd.to_numeric(pdf[event_col], errors="coerce")
    for c in numeric_cols:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

    # Censor NULL duration
    null_dur = pdf[duration_col].isna()
    n_censored_from_null = int(null_dur.sum())
    if n_censored_from_null > 0:
        pdf.loc[null_dur, duration_col] = float(censor_time)
        pdf.loc[null_dur, event_col] = 0

    # Validity filters
    pdf = pdf[pdf[duration_col] > 0].copy()
    pdf = pdf[pdf[event_col].isin([0, 1])].copy()

    # Drop missing predictors
    pred_cols = list(numeric_cols) + list(categorical_cols)
    n_before_pred_drop = len(pdf)
    if pred_cols:
        pdf = pdf.dropna(subset=pred_cols).copy()
    n_rows_dropped_missing_predictors = n_before_pred_drop - len(pdf)

    # Reference categories (most frequent)
    reference_map = determine_reference_levels(pdf, categorical_cols)

    # Force category ordering so baseline is most frequent
    for c, ref in reference_map.items():
        categories = pdf[c].value_counts().index.tolist()
        ordered = [ref] + [x for x in categories if x != ref]
        pdf[c] = pd.Categorical(pdf[c], categories=ordered)

    # One-hot
    if categorical_cols:
        pdf = pd.get_dummies(pdf, columns=categorical_cols, drop_first=drop_first)

    # Drop rare/constant dummies
    exclude = [duration_col, event_col]
    dummy_cols = identify_dummy_like_columns(pdf, exclude=exclude)

    pdf2, drop_details = drop_low_freq_low_var_dummies(
        pdf,
        dummy_cols=dummy_cols,
        min_freq_rate=min_freq_rate,
        min_freq_abs=min_freq_abs,
        min_var=min_var,
    )

    drop_report = {
        "n_rows_final": int(len(pdf2)),
        "n_features_final": int(len(pdf2.columns)),
        "n_censored_from_null_duration": int(n_censored_from_null),
        "n_rows_dropped_missing_predictors": int(n_rows_dropped_missing_predictors),
        "reference_categories": reference_map,
        **drop_details,
    }

    return pdf2, drop_report, reference_map


# ----------------------------
# Cox model + outputs
# ----------------------------

def fit_cox_model(
    cox_df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    penalizer: float = 0.1,
) -> CoxPHFitter:
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
    min_freq_rate: float = 0.001,
    min_freq_abs: int = 50,
    min_var: float = 1e-8,
) -> dict:
    """
    Full pipeline per table:
      Spark load -> create time_bin -> build design (NULL duration censored @ censor_time)
      -> drop low-freq/low-var dummy columns -> fit Cox -> HR + stats
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

    categorical_cols = [c for c in categorical_cols if c != timebin_col] + [timebin_col]

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

    cox_df, drop_report, reference_map = build_cox_design(
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