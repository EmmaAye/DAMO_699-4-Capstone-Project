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
# # Survival Analysis — RQ1–RQ5 (Toronto vs NYC)
#
# This notebook implements a **single, consistent survival-analysis framework** to answer the thesis research questions:
#
# - **RQ1 (Temporal):** hour / day-of-week / season effects → stratified KM + log-rank (**+ Cox read-in if available**)
# - **RQ2 (Demand):** calls\_past\_(30, 60)min effects → **Cox read-in**
# - **RQ3 (Cross-city):** Toronto vs NYC → KM overlay + log-rank + hazard comparison
# - **RQ4 (Drivers):** compare whether temporal+demand explain delay risk more than incident type → **Cox read-in + magnitude comparison** (optional reduced-model comparison)
# - **RQ5 (Tail risk):** survival probabilities at thresholds (10/15/30/60) vs averages
#
# **Administrative censoring rule (for comparability):** analysis uses a common window **0–60 minutes**:
# - `response_minutes` is right-censored at 60
# - `event_indicator = 1` only if arrival observed within 60; otherwise `event_indicator = 0`
#
# > Why this matters: NYC is capped at 60 in the raw data; applying the same rule to Toronto ensures fair cross-city comparison.
#
# _Last updated: 2026-02-23_

# %%
# %pip install lifelines

# %%
# %%
# Databricks sometimes needs this after installs
dbutils.library.restartPython()

# %% [markdown]
# ## 0. Setup

# %% [markdown]
# ### Import Libraries

# %%
try:
    spark
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import functions as F

# Repo root (Databricks workspace path)
REPO_ROOT = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
#---------- Import Libraries from Project Code -----------#
from src.models.survival_analysis import (
    load_city_survival_spark,
    fit_km,
    km_plot_single_city,
    km_overlay_plot,
    validate_km,
    baseline_report_text,
    cross_city_logrank,
    binned_hazard,
    hazard_overlay_plot,
    threshold_summary,
    DEFAULT_CENSOR_TIME,
    DEFAULT_THRESHOLDS,
    add_strata_columns,
    HOUR_ORDER,
    SEASON_ORDER,
    DOW_ORDER,
    km_plot_stratified,
    within_city_multivariate_logrank,
    summarize_driver_strength,
)


# %% [markdown]
# ### Config Paths

# %%
# Project tables
TORONTO_TABLE = "workspace.capstone_project.toronto_model_ready"
NYC_TABLE     = "workspace.capstone_project.nyc_model_ready"

# Censoring + thresholds
THRESHOLDS = DEFAULT_THRESHOLDS
CENSOR_TIME =DEFAULT_CENSOR_TIME

# Output folders (repo)
OUT_GRAPH_DIR = f"{REPO_ROOT}/output/graphs"
OUT_TABLE_DIR = f"{REPO_ROOT}/output/tables"
os.makedirs(OUT_GRAPH_DIR, exist_ok=True)
os.makedirs(OUT_TABLE_DIR, exist_ok=True)

print("Graphs:", OUT_GRAPH_DIR)
print("Tables:", OUT_TABLE_DIR)

# %% [markdown]
# ## 1. Load censored survival datasets (Toronto + NYC)

# %%
# Load survival datasets with uniform 60-min administrative censoring
extra_cols = [
    "hour", "day_of_week", "season",
    "calls_past_30min", "calls_past_60min",
    "incident_category", "unified_alarm_level"
]

df_to_base  = load_city_survival_spark(spark, TORONTO_TABLE, censor_threshold=CENSOR_TIME, extra_cols=extra_cols)
df_nyc_base = load_city_survival_spark(spark, NYC_TABLE,     censor_threshold=CENSOR_TIME, extra_cols=extra_cols)

print("Toronto rows:", df_to_base.count())
print("NYC rows:", df_nyc_base.count())


# %%
# Quick invariants (should be <=60 and non-null)
df_to_base.selectExpr(
    "max(response_minutes) as max_t",
    "min(response_minutes) as min_t",
    "sum(case when response_minutes is null then 1 else 0 end) as null_t",
    "sum(case when event_indicator not in (0,1) or event_indicator is null then 1 else 0 end) as bad_event"
).show()

# %%
df_nyc_base.selectExpr(
    "max(response_minutes) as max_t",
    "min(response_minutes) as min_t",
    "sum(case when response_minutes is null then 1 else 0 end) as null_t",
    "sum(case when event_indicator not in (0,1) or event_indicator is null then 1 else 0 end) as bad_event"
).show()

# %% [markdown]
# ## 2. RQ5: Tail risk vs averages (threshold survival probabilities)

# %%
to_thr  = threshold_summary(df_to_base,  city_label="Toronto")
nyc_thr = threshold_summary(df_nyc_base, city_label="NYC")

thr_df = pd.concat([to_thr, nyc_thr], ignore_index=True)
display(thr_df)

thr_path = f"{OUT_TABLE_DIR}/km_survival_thresholds_cross_city.csv"
thr_df.to_csv(thr_path, index=False)
print("Saved:", thr_path)

# %% [markdown]
# ## 3. Baseline KM per city (supports RQ5 + context for RQ3)

# %%
# -------- convert to Pandas and Fit Km ----------#
to_pd_base  = df_to_base.select("response_minutes","event_indicator").toPandas()
nyc_pd_base = df_nyc_base.select("response_minutes","event_indicator").toPandas()

km_to  = fit_km(to_pd_base,  label="Toronto", duration_col="response_minutes", event_col="event_indicator")
km_nyc = fit_km(nyc_pd_base, label="NYC",     duration_col="response_minutes", event_col="event_indicator")


# --- validate KM Fit for each City ---------#


display(validate_km(to_pd_base,  km_to,  "Toronto"))
display(validate_km(nyc_pd_base, km_nyc, "Kaplan–Meier Baseline — NYC"))

#-------- Plot KM curves for each city -------------#
toronto_graph_path = f"{OUT_GRAPH_DIR}/final_km_toronto.png"
nyc_graph_path = f"{OUT_GRAPH_DIR}/final_km_nyc.png"

km_plot_single_city(km_to,  title="Kaplan–Meier Baseline — Toronto")
plt.savefig(toronto_graph_path, dpi=200)

km_plot_single_city(km_nyc, title="Kaplan–Meier Baseline — NYC")
plt.savefig(nyc_graph_path, dpi=200)
plt.show()

print("Saved:", toronto_graph_path)
print("Saved:", nyc_graph_path)

print("\n--- Baseline narrative (Toronto vs NYC) ---")
print(baseline_report_text(km_to, km_nyc))


# %% [markdown]
# ## 4. RQ3: Cross-city survival comparison (KM overlay + log-rank + hazard)

# %%
# KM overlay
graph_path = f"{OUT_GRAPH_DIR}/final_km_baseline_toronto_vs_nyc.png"
km_overlay_plot(
    km_to,
    km_nyc,
    censor_threshold=60,
    title="Kaplan–Meier Survival Toronto vs NYC (Observed Response Times)",
)

plt.savefig(graph_path, dpi=200)
plt.show()
print("Saved:", graph_path)


# %%
# Formal cross-city comparison (log-rank)
lr = cross_city_logrank(to_pd_base, nyc_pd_base)
print("Cross-city log-rank:", lr)

# %% [markdown]
# The Kaplan–Meier survival curves indicate that the probability of incidents remaining unresolved declines rapidly during the early minutes for both cities. The steep initial drop suggests that a large proportion of incidents are resolved shortly after initiation.
#
# Toronto’s survival curve lies consistently below that of NYC during the early time period, indicating that incidents in Toronto tend to be resolved more quickly. This difference is most pronounced within the first 10–15 minutes, after which both curves flatten, reflecting a smaller subset of longer-duration cases.
#
# The log-rank test confirms that the difference in time-to-resolution distributions between the two cities is statistically significant (if applicable — include only if you tested this).
#
# **Interpretation:**
# Overall, incidents in Toronto resolve faster than those in NYC, particularly in the early response window.

# %% [markdown]
# A log-rank test comparing Toronto and NYC Kaplan–Meier survival curves strongly rejects the null hypothesis of equal response-time distributions (χ² ≈ 165,947, p < 0.001). This indicates statistically significant systemic differences in response-time completion patterns between the two cities under the shared 0–60 minute administrative censoring window.
#
# Visual inspection shows Toronto exhibits faster early-time arrival and a lighter long-delay tail compared with NYC.
#
# What the log-rank test is testing:
#
# Null hypothesis (H₀):
# Toronto and NYC have the same survival distribution over time (0–60 min).
# (The probability that the first unit has not yet arrived evolves the same way in both cities.)
#
# - test_statistic: 165,946.81
# - p_value: < 1e-300
# - n_A (Toronto): 361,710
# - n_B (NYC): 1,283,156
# - events_A: 349,214
# - events_B: 910,612
#
# **Interpretation:**
# - The test statistic is extremely large.
# - The p-value is effectively zero (machine underflow).
# - The data is very large sample sizes.
# - Both cities have many observed arrival events.
#
# **Conclusion**:
#
# **The null hypothesis is decisively rejected.**
# Toronto and NYC survival curves are statistically different.

# %%
# Hazard comparison (binned)
hz_to = binned_hazard(to_pd_base, censor_threshold=60.0, bin_width=2.0)
hz_ny = binned_hazard(nyc_pd_base, censor_threshold=60.0, bin_width=2.0)
hazard_overlay_plot(hz_to, hz_ny, title="Observed Hazard Rate (Binned Kaplan–Meier Estimate)")
plt.savefig(f"{OUT_GRAPH_DIR}/final_hr_km_binned_overlay.png", dpi=300, bbox_inches="tight")

hz_to["city"] = "Toronto"
hz_ny["city"] = "NYC"
haz_df = pd.concat([hz_to, hz_ny], ignore_index=True)
haz_path = f"{OUT_TABLE_DIR}/km_hazard_binned_cross_city.csv"
haz_df.to_csv(haz_path, index=False)
print("Saved:", haz_path)

# %% [markdown]
# Both cities exhibit the highest completion hazard within the first 5–8 minutes, indicating that incidents are most likely to be resolved during this early period. This suggests that the majority of incidents are handled quickly after initiation, while a smaller subset persists into longer durations.
#
# The hazard declines after the initial peak, reflecting that incidents remaining unresolved beyond approximately 15 minutes have a lower conditional probability of resolution within any given time interval. This decline does not imply that resolution ceases after 20 minutes; rather, it indicates that only a small number of more complex cases remain in the risk set, leading to a substantially reduced per-interval likelihood of completion.

# %% [markdown]
# ### RQ3: Cross-city survival comparison
#
# Kaplan–Meier curves indicate that incidents in both cities are resolved rapidly in the early minutes, with Toronto’s survival curve dropping more steeply than NYC’s, suggesting faster resolution overall. The log-rank test confirms a statistically significant difference in time-to-resolution distributions between the two cities.
#
# The binned hazard comparison shows that both cities experience peak resolution probability within the first 5–8 minutes; however, Toronto exhibits a consistently higher early hazard, indicating a greater conditional likelihood of resolution during this initial period. After approximately 15 minutes, the hazard declines for both cities, reflecting that only a small number of longer-duration cases remain and that the per-interval probability of completion becomes low.
#
# Together, these results suggest that while most incidents are resolved quickly in both cities, Toronto demonstrates faster early-stage resolution dynamics than NYC, with differences most pronounced in the first 10–15 minutes.
#

# %% [markdown]
# ## 5. RQ1: Temporal Effects on Response Time Within Each City
#
# This section examines whether incident response times vary across temporal contexts within each city.  
# We analyze time-of-day, day-of-week, and seasonal effects using stratified Kaplan–Meier survival curves and within-city multivariate log-rank tests.
#
# All analyses use a uniform right-censoring threshold of 60 minutes to ensure comparability.

# %%
# Add temporal strata (hour_group + day_of_week_name)
df_to_temporal  = add_strata_columns(df_to_base)
df_nyc_temporal = add_strata_columns(df_nyc_base)

to_pd  = df_to_temporal.toPandas()
nyc_pd = df_nyc_temporal.toPandas()


# %% [markdown]
# Helper For Plot and Test

# %%
def plot_and_test(pdf, city, group_col, order, title, save_path):
    km_plot_stratified(
        pdf,
        group_col,
        title,
        censor_threshold=CENSOR_TIME,
        group_order=order
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    res = within_city_multivariate_logrank(
        pdf,
        group_col,
        censor_threshold=CENSOR_TIME
    )
    return res["p_value"]


# %% [markdown]
# ### 5.1 Toronto Temporal Stratified Survival

# %%
p_to_time = plot_and_test(
    to_pd, "Toronto", "hour_group", HOUR_ORDER,
    "Toronto KM by time of day",
    f"{OUT_GRAPH_DIR}/final_km_toronto_by_hour.png"
)

p_to_dow = plot_and_test(
    to_pd, "Toronto", "day_of_week_name", DOW_ORDER,
    "Toronto KM by day of week",
    f"{OUT_GRAPH_DIR}/Final_km_toronto_by_dow.png"
)

p_to_season = plot_and_test(
    to_pd, "Toronto", "season", SEASON_ORDER,
    "Toronto KM by season",
    f"{OUT_GRAPH_DIR}/Final_km_toronto_by_season.png"
)

# %% [markdown]
# ### 5.2 NYC Temporal Stratified Survival

# %%
p_ny_time = plot_and_test(
    nyc_pd, "NYC", "hour_group", HOUR_ORDER,
    "NYC KM by time of day",
    f"{OUT_GRAPH_DIR}/final_km_nyc_by_hour.png"
)

p_ny_dow = plot_and_test(
    nyc_pd, "NYC", "day_of_week_name", DOW_ORDER,
    "NYC KM by day of week",
    f"{OUT_GRAPH_DIR}/final_km_nyc_by_dow.png"
)

p_ny_season = plot_and_test(
    nyc_pd, "NYC", "season", SEASON_ORDER,
    "NYC KM by season",
    f"{OUT_GRAPH_DIR}/final_km_nyc_by_season.png"
)

# %% [markdown]
# ### 5.3 Log-rank Test Summary

# %%
temporal_tests = pd.DataFrame([
    {"city":"Toronto","strat":"hour_group","p_value":p_to_time},
    {"city":"Toronto","strat":"day_of_week","p_value":p_to_dow},
    {"city":"Toronto","strat":"season","p_value":p_to_season},
    {"city":"NYC","strat":"hour_group","p_value":p_ny_time},
    {"city":"NYC","strat":"day_of_week","p_value":p_ny_dow},
    {"city":"NYC","strat":"season","p_value":p_ny_season},
])

display(temporal_tests)

temporal_tests.to_csv(
    f"{OUT_TABLE_DIR}/logrank_within_city_temporal.csv",
    index=False
)
print("Saved:", f"{OUT_TABLE_DIR}/logrank_within_city_temporal.csv")

# %% [markdown]
# ### 5.4 Summary of RQ1: Temporal Effects on Response Time Within Cities
#
# Stratified Kaplan–Meier curves and within-city multivariate log-rank tests were used to evaluate whether response times vary by time of day, day of week, and season within each city. All analyses applied a uniform 60-minute right-censoring threshold.
#
# Across both Toronto and NYC, the log-rank tests indicate **statistically significant differences** in response-time distributions across all temporal dimensions (all p < 0.001). Time of day shows the strongest separation in survival curves, with the most noticeable differences occurring in the early response window. In both cities, the night period generally exhibits slightly higher survival probabilities at later time points, indicating comparatively slower resolution for incidents occurring overnight.
#
# Day-of-week effects are statistically significant but visually modest, suggesting that while distributions differ across days, the magnitude of variation is relatively small. Seasonal effects are also significant in both cities, though the survival curves remain closely clustered, indicating that seasonal variation exists but does not substantially alter overall response-time patterns.
#
# Overall, these results suggest that **temporal context is associated with measurable differences in response-time dynamics within each city**, with time-of-day effects being the most practically meaningful. However, despite statistical significance across all strata, the overall shapes of the survival curves remain similar, indicating that most incidents are resolved rapidly regardless of temporal conditions, with differences primarily affecting the tail of the distribution.

# %% [markdown]
# ## 6. Cox results read-in (RQ1/RQ2/RQ4)
# Cox is **not refit here**; we load the saved outputs if present.

# %%
cox_paths = {
    "hr_nyc":   f"{OUT_TABLE_DIR}/cox_hr_NYC.csv",
    "hr_to":    f"{OUT_TABLE_DIR}/cox_hr_Toronto.csv",
    "stats_nyc":f"{OUT_TABLE_DIR}/cox_stats_NYC.csv",
    "stats_to": f"{OUT_TABLE_DIR}/cox_stats_Toronto.csv",
    "meta_nyc": f"{REPO_ROOT}/output/models/cox_meta_NYC.json",
    "meta_to":  f"{REPO_ROOT}/output/models/cox_meta_Toronto.json",
}

missing = [k for k,p in cox_paths.items() if not os.path.exists(p)]
if missing:
    print("Cox outputs missing:", missing)
    print("Run your Cox notebook once with DO_SAVE=True to generate the CSV/JSON outputs.")
else:
    hr_nyc = pd.read_csv(cox_paths["hr_nyc"])
    hr_to  = pd.read_csv(cox_paths["hr_to"])
    stats_nyc = pd.read_csv(cox_paths["stats_nyc"])
    stats_to  = pd.read_csv(cox_paths["stats_to"])
    with open(cox_paths["meta_nyc"], "r") as f:
        meta_nyc = json.load(f)
    with open(cox_paths["meta_to"], "r") as f:
        meta_to = json.load(f)

    print("Cox stats (NYC):")
    display(stats_nyc)
    print("Cox stats (Toronto):")
    display(stats_to)

    print("Reference categories (NYC):", meta_nyc.get("reference_categories"))
    print("Reference categories (Toronto):", meta_to.get("reference_categories"))

    print("Top HR (NYC):")
    display(hr_nyc.sort_values("hazard_ratio", ascending=False).head(10))
    print("Bottom HR (NYC):")
    display(hr_nyc.sort_values("hazard_ratio", ascending=True).head(10))

    print("Top HR (Toronto):")
    display(hr_to.sort_values("hazard_ratio", ascending=False).head(10))
    print("Bottom HR (Toronto):")
    display(hr_to.sort_values("hazard_ratio", ascending=True).head(10))

# %% [markdown]
# ### Cox Proportional Hazards Analysis (RQ1, RQ2, RQ4)
#
# Separate Cox proportional hazards models were fitted for Toronto and NYC using a 60-minute administrative censoring threshold. Both models used identical reference categories (Friday, summer, Medical incidents, alarm level 1, and Afternoon time period). Model discrimination was moderate, with concordance indices of 0.66 (NYC) and 0.57 (Toronto), indicating stronger explanatory power in NYC.
#
#
#
# #### RQ1: Temporal Drivers of Delay Risk
#
# **Research Question:**  
# Do hour of day, day of week, and season significantly influence delay risk?
#
# Temporal variables are statistically significant in both cities. Night periods are associated with lower hazard ratios (HR < 1), indicating slower response and elevated delay risk relative to the afternoon reference period. Evening periods show slightly higher hazards (HR > 1), suggesting faster resolution. Day-of-week and seasonal indicators are statistically significant but exhibit modest effect sizes.
#
# Overall, while temporal factors significantly influence hazard rates, their magnitude is moderate compared to incident-related variables. This supports the conclusion that temporal context matters, but does not dominate delay risk.
#
#
#
# #### RQ2: Demand Intensity Effects
#
# **Research Question:**  
# Do short-term demand surges increase delay risk?
#
# Demand intensity was captured using short-term call volume variables (`calls_past_30min`, `calls_past_60min`).
#
# In Toronto, higher recent call volume is associated with hazard ratios below 1, indicating slower resolution and increased delay risk during demand surges. In contrast, NYC shows hazard ratios slightly above 1 for workload variables, suggesting that higher call activity does not slow response and may coincide with system scaling or operational prioritization.
#
# These results indicate that demand intensity affects delay risk, but the operational response to demand differs structurally between the two cities.
#
#
#
# ### RQ4: Relative Importance of Temporal and Demand Factors
#
# **Research Question:**  
# Do temporal and demand factors explain delay risk more effectively than incident classification alone?
#
# Across both cities, incident category and alarm severity produce the largest hazard ratios. In NYC, structural fire incidents exhibit hazard ratios exceeding 4.0, indicating substantially faster response relative to medical incidents. In Toronto, incident-type effects are smaller but remain stronger than temporal or workload effects.
#
# Temporal and demand variables are statistically significant but show comparatively modest hazard ratio magnitudes. This indicates that incident characteristics and operational severity are the dominant drivers of delay risk, while temporal and demand factors contribute incremental explanatory power.
#
#
#
# #### Overall Interpretation
#
# The Cox models demonstrate that:
#
# - Temporal factors significantly affect hazard rates (RQ1), though with moderate magnitude.
# - Demand intensity influences delay risk, with city-specific differences in direction (RQ2).
# - Incident classification and alarm severity are the strongest predictors of response-time hazard (RQ4).
#
# Together, these findings suggest that delay risk is primarily driven by incident context, with temporal and workload factors acting as secondary modifiers.

# %% [markdown]
# ## 9. RQ4: Drivers summary helper (temporal+demand vs incident type)
# This section provides a pragmatic comparison using the **magnitude of HRs**.
# Optional extension: fit reduced Cox models (A/B/C) in the Cox notebook.

# %%
strength = pd.concat([
    summarize_driver_strength(hr_nyc, "NYC"),
    summarize_driver_strength(hr_to, "Toronto"),
], ignore_index=True)

display(strength)

strength_path = f"{OUT_TABLE_DIR}/cox_driver_strength_summary.csv"
strength.to_csv(strength_path, index=False)
print("Saved:", strength_path)

# %% [markdown]
# ### RQ4: Relative Importance of Temporal, Demand, and Incident Drivers
#
# To compare the relative influence of feature groups, hazard-ratio magnitudes were summarized using a distance-from-1 effect size metric. Variables were grouped into three buckets: Incident/Severity, Temporal, and Demand.
#
# Across both cities, **incident and severity variables clearly dominate in magnitude**.  
# In NYC, the largest effect sizes occur in the Incident/Severity bucket (max ≈ 4.63, median ≈ 3.00), substantially exceeding Temporal effects (max ≈ 1.15) and Demand effects (≈ 1.00). This indicates that incident classification and alarm level are the primary drivers of response-time hazard in NYC, with temporal and demand variables acting as secondary modifiers.
#
# Toronto shows the same overall pattern but with smaller magnitudes. Incident/Severity effects remain the largest (max ≈ 1.69), followed by Temporal effects (max ≈ 1.27) and Demand effects (max ≈ 1.13). While temporal and demand variables are statistically significant, their effect sizes are notably smaller than incident-related predictors.
#
# Overall, these results indicate that **incident type and severity are the strongest determinants of delay risk in both cities**, while temporal conditions and short-term demand intensity provide incremental explanatory power. This supports the conclusion that temporal and demand variables influence delay risk but do not outweigh incident characteristics as primary predictors.

# %% [markdown]
# ## 8. Survival & Cox Model Findings by Research Question
# This module applied survival analysis and Cox proportional hazards modeling to investigate drivers of emergency response-time delay risk across Toronto and New York City. Kaplan–Meier curves, log-rank tests, and multivariate Cox models were used to examine temporal effects, demand intensity, cross-city structural patterns, and key predictive drivers of delay.
#
# Across both cities, incident type and operational severity emerged as the strongest determinants of response-time hazard, with high-priority incidents resolving substantially faster than lower-priority calls. Temporal factors (hour, day, season) and short-term demand intensity were statistically significant but exhibited comparatively smaller effect sizes, indicating that they modify—but do not dominate—delay risk. Demand effects differed across cities, suggesting distinct operational responses to workload conditions.
#
# Cross-city comparisons reveal similar early-resolution structures but meaningful differences in tail behavior and prioritization strength. Finally, survival-based tail metrics demonstrate that meaningful delay risk exists beyond commonly reported average response times, highlighting the importance of examining distributional and tail-risk dynamics rather than relying solely on mean-based performance metrics.
#
# Overall, the results indicate that delay risk is primarily driven by incident context and operational severity, with temporal and demand factors acting as secondary but meaningful modifiers. Survival analysis provides a robust framework for understanding both central tendencies and tail-risk behavior in emergency response systems.
#
# **All survival and Cox analyses use a uniform 60-minute administrative censoring threshold to ensure comparability across cities.**
#
#
#

# %% [markdown]
# ### RQ1: Temporal Drivers of Delay Risk
#
# **Research Question:**
# Do hour, day, and season significantly influence emergency response-time delay risk?
#
# **Kaplan–Meier Evidence**
#
# Stratified Kaplan–Meier curves show statistically significant differences across hour-of-day, day-of-week, and season in both Toronto and NYC (all p < 0.001 via log-rank tests). The largest separation appears across time-of-day strata, particularly between Night and Afternoon/Evening periods.
#
# However, although statistically significant, the survival curves remain closely clustered, indicating that temporal variation exists but is modest in magnitude relative to overall response patterns.
#
# **Cox Model Evidence**
#
# In both cities:
#
# * Night periods are associated with lower hazard (HR < 1), indicating slower resolution and higher delay risk.
# * Evening periods show slightly higher hazard (HR > 1).
# * Day-of-week and seasonal effects are statistically significant but small in magnitude.
#
# **Conclusion for RQ1:**
# Temporal factors significantly influence delay risk, but effect sizes are modest. Time-of-day has the strongest temporal impact, while day-of-week and season contribute smaller variations.
#
#
#

# %% [markdown]
# ### RQ2: Demand Intensity Effects
#
# **Research Question:**
# Do short-term demand surges increase delay risk?
#
# **Cox Model Evidence**
#
# Workload variables (`calls_past_30min`, `calls_past_60min`) directly test this hypothesis.
#
# * In Toronto:
#
#   * `calls_past_30min` has HR < 1.
#   * Higher short-term demand is associated with reduced hazard (slower response), indicating increased delay risk during demand surges.
# * In NYC:
#
#   * Workload HRs are slightly > 1.
#   * This suggests higher activity coincides with slightly faster resolution, possibly reflecting system scaling or triage dynamics.
#
# **Conclusion for RQ2:**
# Demand intensity affects delay risk, but the direction differs by city. Toronto shows evidence of strain under short-term surges, while NYC appears more resilient to demand fluctuations.

# %% [markdown]
# ### RQ3: Cross-City Structural Comparison
#
# **Research Question:**
# Are Toronto and NYC delay patterns structurally similar?
#
# **KM + Log-Rank**
#
# Cross-city Kaplan–Meier overlays show broadly similar rapid early resolution patterns but different tail behavior. Log-rank tests confirm statistically significant distributional differences.
#
# **Cox Comparison**
#
# * NYC shows much stronger hazard differentiation by incident category (e.g., structural fire HR ≈ 4.6).
# * Toronto shows more moderate differentiation.
# * Model concordance is higher in NYC (0.66) than Toronto (0.57).
#
# **Conclusion for RQ3:**
# Both cities share similar early-resolution structures, but differ in the strength of prioritization effects and workload dynamics. NYC exhibits stronger structural differentiation across incident types.

# %% [markdown]
# ### RQ4: Key Predictive Drivers
#
# **Research Question:**
# Do temporal and demand factors explain delay risk more effectively than incident classification?
#
# **Cox Findings**
#
# Incident category and alarm level produce the largest hazard ratios in both cities. Temporal and demand factors are statistically significant but much smaller in magnitude.
#
# This suggests:
#
# * Incident type and severity are dominant predictors.
# * Temporal and workload factors contribute incremental explanatory power but do not exceed incident-level influence.
#
# **Conclusion for RQ4:**
# Incident classification explains delay risk more strongly than temporal or demand factors alone. Temporal and demand variables enhance model performance but are secondary drivers.

# %% [markdown]
# ### RQ5: Tail Risk vs Averages
#
# **Research Question:**
# Do survival-based tail metrics reveal risks hidden by averages?
#
# **Evidence**
#
# Mean response times suggest rapid overall service. However:
#
# * Kaplan–Meier curves reveal non-trivial survival probabilities beyond 10–15 minutes.
# * P90 and delay percentages highlight tail exposure.
# * Cross-city differences are more visible in tail survival than in averages.
#
# **Conclusion for RQ5:**
# Tail-delay metrics and survival-based probabilities reveal structural delay risk not fully captured by average response times. Survival analysis provides a clearer view of risk accumulation over time.

# %% [markdown]
# ### Overall Synthesis
#
# Across both cities:
#
# * Most incidents are resolved rapidly.
# * Incident severity is the dominant determinant of delay risk.
# * Temporal factors are statistically significant but modest.
# * Demand effects differ structurally between cities.
# * Tail-risk analysis provides insight beyond average-based reporting.
