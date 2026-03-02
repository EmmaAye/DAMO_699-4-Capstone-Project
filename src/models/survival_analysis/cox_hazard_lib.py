"""
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
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

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
    '''For categorical predictors, the most frequent category was used as the reference level to improve interpretability and stability of coefficient estimates. This choice does not affect model predictions or overall fit but provides clearer comparisons across incident types and temporal factors.'''
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
    # 3.5) Administrative censoring beyond censor_time (core survival rule)
    # -----------------------------
    over = pdf[duration_col] > float(censor_time)
    n_censored_from_over = int(over.sum())
    if n_censored_from_over > 0:
        pdf.loc[over, duration_col] = float(censor_time)
        pdf.loc[over, event_col] = 0

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

    # Force category ordering so baseline is most frequent
    for c, ref in reference_map.items():
        categories = pdf[c].value_counts().index.tolist()
        ordered = [ref] + [x for x in categories if x != ref]
        pdf[c] = pd.Categorical(pdf[c], categories=ordered)

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

    # -----------------------------
    # 8.5) Sanity checks (admin censoring consistency)
    # -----------------------------
    # 1) Duration must be present and positive
    assert pdf2[duration_col].notna().all(), "Cox design has NULL durations after preprocessing."
    assert (pdf2[duration_col] > 0).all(), "Cox design has non-positive durations after preprocessing."

    # 2) Event must be binary
    bad_events = set(pd.unique(pdf2[event_col])) - {0, 1}
    assert len(bad_events) == 0, f"Cox design has invalid event values: {bad_events}"

    # 3) Administrative censoring at censor_time must hold
    mx = float(pdf2[duration_col].max())
    assert mx <= float(censor_time) + 1e-9, f"Max duration {mx} exceeds censor_time {censor_time}."

    # 4) No 'fake events' at the censoring boundary due to clipping
    n_fake = int(((pdf2[duration_col] == float(censor_time)) & (pdf2[event_col] == 1)).sum())
    assert n_fake == 0, f"Found {n_fake} rows with duration==censor_time but event==1. Should be censored (event=0)."
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
    # dummy drop configs (passed into build_cox_design)
    min_freq_rate: float = 0.001,
    min_freq_abs: int = 50,
    min_var: float = 1e-8,
)-> dict:
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

    import numpy as np
import pandas as pd


def summarize_driver_strength(
    hr_df: pd.DataFrame,
    city: str,
    alpha: float = 0.05,
    top_k: int = 5,
):
    """
    Summarize Cox driver strength by feature group using HR distance from 1.

    Parameters
    ----------
    hr_df : DataFrame
        Cox hazard-ratio table (hr_to / hr_nyc)
        Must contain:
            - covariate
            - hazard_ratio
        Optional:
            - p

    city : str
        City label ("Toronto", "NYC")

    alpha : float
        Optional significance filter. If None → no filtering.

    top_k : int
        Number of strongest effects used for stable average.

    Returns
    -------
    DataFrame summary per bucket:
        Demand
        Temporal
        Incident/Severity
    """

    def bucket(name: str) -> str:
        if name.startswith("calls_past"):
            return "Demand"
        if name.startswith(("time_bin", "hour_group", "day_of_week", "season")):
            return "Temporal"
        if name.startswith(("incident_category", "unified_alarm_level")):
            return "Incident/Severity"
        return "Other"

    df = hr_df.copy()

    if alpha is not None and "p" in df.columns:
        df = df[df["p"] < alpha].copy()

    df["covariate"] = df["covariate"].astype(str)
    df["hazard_ratio"] = pd.to_numeric(df["hazard_ratio"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["hazard_ratio"])
    df = df[df["hazard_ratio"] > 0]

    df["bucket"] = df["covariate"].apply(bucket)

    # distance from 1
    hr = df["hazard_ratio"].to_numpy()
    df["effect_size"] = np.maximum(hr, 1.0 / hr)

    base = (
        df.groupby("bucket")["effect_size"]
          .agg(count="count", max_effect="max", median_effect="median")
          .reset_index()
    )

    topk_df = (
        df.sort_values("effect_size", ascending=False)
          .groupby("bucket")
          .head(top_k)
    )

    topk = (
        topk_df.groupby("bucket")["effect_size"]
               .mean()
               .reset_index()
               .rename(columns={"effect_size": f"top{top_k}_mean"})
    )

    out = base.merge(topk, on="bucket", how="left")
    out["city"] = city
    out = out.sort_values("max_effect", ascending=False)

    return out

#--------------------------------------------
# Cross-City Hazard Structure Comparison
#---------------------------------------------

def get_baseline_hazard_series(cph) -> pd.Series:
    """
    Returns baseline hazard as a Series indexed by time.
    """
    if hasattr(cph, "baseline_hazard_") and cph.baseline_hazard_ is not None:
        bh = cph.baseline_hazard_.copy()
        # bh is typically a DataFrame with 1 column
        if isinstance(bh, pd.DataFrame):
            s = bh.iloc[:, 0]
        else:
            s = pd.Series(bh)
        s.index = s.index.astype(float)
        s.index.name = "time"
        s.name = "h0(t)"
        return s.sort_index()

    raise ValueError("Model does not have baseline_hazard_. Did you fit with lifelines CoxPHFitter?")

def align_and_smooth_hazard(h0: pd.Series, timeline: np.ndarray, smooth_window: int = 5) -> pd.Series:
    # Align to common timeline
    aligned = h0.reindex(timeline, method="ffill")

    # Replace deprecated fillna(method=...) with bfill()/ffill()
    aligned = aligned.bfill().fillna(0.0)

    # Optional smoothing for readability
    if smooth_window and smooth_window > 1:
        aligned = aligned.rolling(window=smooth_window, min_periods=1, center=True).mean()

    aligned.name = h0.name
    return aligned

def median_survival_time_interp(S: pd.Series) -> float:
    # If never drops below 0.5
    if (S <= 0.5).sum() == 0:
        return float("nan")

    t2 = float(S[S <= 0.5].index[0])  # first time below
    # if exactly at t2
    if S.loc[t2] == 0.5 or t2 == float(S.index.min()):
        return t2

    # previous time above 0.5
    idx_pos = S.index.get_loc(t2)
    t1 = float(S.index[idx_pos - 1])
    s1 = float(S.loc[t1])
    s2 = float(S.loc[t2])

    # linear interpolate time where S(t)=0.5
    if s1 == s2:
        return t2
    return float(t1 + (0.5 - s1) * (t2 - t1) / (s2 - s1))

def survival_at_times(S: pd.Series, times: List[float]) -> Dict[str, float]:
    # nearest time on index (index is minute timeline)
    out = {}
    for t in times:
        t2 = float(min(max(t, S.index.min()), S.index.max()))
        out[f"S({int(t2)}m)"] = float(S.loc[t2])
    return out

def hazard_peak(h: pd.Series) -> Tuple[float, float]:
    idx = float(h.idxmax())
    val = float(h.max())
    return idx, val

def get_concordance(meta: Dict, cph=None) -> float:
    for k in ["concordance", "concordance_index", "c_index", "cindex"]:
        if k in meta and meta[k] is not None:
            return float(meta[k])
    # fallback if lifelines stored it
    if cph is not None and hasattr(cph, "concordance_index_"):
        return float(cph.concordance_index_)
    return float("nan")

def plot_survival_overlay(S_a: pd.Series, S_b: pd.Series, label_a: str, label_b: str, outpath: str):
    plt.figure(figsize=(8, 5))
    plt.plot(S_a.index, S_a.values, label=label_a)
    plt.plot(S_b.index, S_b.values, label=label_b)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Baseline survival probability S(t | reference covariates)")
    plt.title("Cox Model Baseline Survival Toronto vs NYC (Reference Profile)")
    plt.ylim(0, 1.01)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()

def plot_hazard_overlay(h_a: pd.Series, h_b: pd.Series,
                        label_a: str, label_b: str,
                        outpath: str):

    plt.figure(figsize=(10, 6))

    color_a = "#1f77b4"   # Toronto
    color_b = "#d62728"   # NYC

    plt.plot(h_a.index, h_a.values, label=label_a)
    plt.plot(h_b.index, h_b.values, label=label_b)

    # ---- peak locations ----
    peak_t_a = float(h_a.idxmax())
    peak_v_a = float(h_a.max())

    peak_t_b = float(h_b.idxmax())
    peak_v_b = float(h_b.max())

    # ---- peak lines ----
    plt.axvline(peak_t_a, color=color_a, linestyle="--", linewidth=2, alpha=0.5)
    plt.axvline(peak_t_b, color=color_b, linestyle="--", linewidth=2, alpha=0.5)

    # ---- peak markers ----
    plt.scatter(peak_t_a, peak_v_a, color=color_a, zorder=5)
    plt.scatter(peak_t_b, peak_v_b, color=color_b, zorder=5)

    plt.text(
        peak_t_a + 0.6,
        peak_v_a * 0.95,
        f"{label_a}\npeak={peak_v_a:.4f}\nat {peak_t_a:.0f} min",
        color=color_a,
        fontsize=9,
        weight="bold"
    )

    plt.text(
        peak_t_b + 0.6,
        peak_v_b * 0.85,
        f"{label_b}\npeak={peak_v_b:.4f}\nat {peak_t_b:.0f} min",
        color=color_b,
        fontsize=9,
        weight="bold"
    )

    # ---- formatting ----
    plt.xlabel("Time (minutes)")
    plt.ylabel("Baseline hazard h₀(t)")
    plt.title("Baseline Hazard from Cox Model (Reference Covariates)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # SAVE FIGURE
    plt.savefig(outpath, dpi=200, bbox_inches="tight")

    # display
    plt.show()

    # prevent memory accumulation in notebooks
    plt.close()
