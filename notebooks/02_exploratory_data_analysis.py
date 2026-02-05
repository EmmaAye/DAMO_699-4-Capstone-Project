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
# # Exploratory Data Analysis (EDA)
#
# This notebook conducts exploratory data analysis on the harmonized and model-ready emergency incident datasets for Toronto and New York City. The objective of this analysis is to examine the distribution and variability of emergency response times, identify temporal and operational patterns associated with peak demand and delayed responses, and assess service-level performance beyond simple averages. Particular attention is given to tail delays and response-time threshold breaches, which are critical for understanding operational risk in emergency response systems. The findings from this EDA are used to guide feature engineering, model selection, and comparative analysis in subsequent stages of the project.
#
#

# %% [markdown]
# ## 0. Import Libraries

# %%
# PySpark core
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sum as spark_sum,
    count,
    when,
    hour,
    dayofweek,
    date_format
)
from pyspark.sql import functions as F
from pyspark.sql.functions import countDistinct
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
# Optional: for local conversion & plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


sns.set_theme(
    style="whitegrid",   # grid theme
    context="notebook"   # scales text appropriately
)
# Plot settings
plt.style.use("default")
sns.set_context("notebook")

# %% [markdown]
# ## 1. Sanity & Structure Check
# Goal: Make sure the tables are truly “model-ready”.
#
# - Row count (compare Toronto vs NYC scale)
# - Column list & data types
# - Missing values per column
# - Duplicate incidents (by incident ID + timestamp)

# %% [markdown]
# ### 1.1 Load Tables

# %%
toronto_df = spark.table("workspace.capstone_project.toronto_model_ready")
nyc_df = spark.table("workspace.capstone_project.nyc_model_ready")

# %% [markdown]
# ### 1.2 Row Count (Scale Comparison)

# %%
toronto_count = toronto_df.count()
nyc_count = nyc_df.count()

toronto_count, nyc_count

# %% [markdown]
# **Row Count Summary**
#
# The Toronto dataset contains 349,198 emergency incidents, while the New York City dataset contains 1,060,771 incidents. The substantially larger volume of incidents in New York City is expected due to differences in population size, urban density, and emergency service demand. These scale differences are taken into account during exploratory analysis and modeling, particularly when comparing response-time distributions and service-level risk across cities.

# %% [markdown]
# ### 1.3 Column List & Data Types

# %%
toronto_df.printSchema()

# %%
nyc_df.printSchema()

# %%
set(toronto_df.columns) - set(nyc_df.columns), set(nyc_df.columns) - set(toronto_df.columns)


# %% [markdown]
# **Schema Consistency Check**
#
# A comparison of column names across the Toronto and New York City datasets shows no differences in schema. Both datasets contain identical sets of analytical features, confirming that the data harmonization process successfully aligned the structure of the two datasets and enables direct cross-city comparison.

# %% [markdown]
# ### 1.4 Missing Value per Column

# %%
def missing_value_summary(df):
    return df.select([
        spark_sum(col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])


# %%
def missing_table(df):
    total = df.count()
    m = missing_value_summary(df).toPandas().T.reset_index()
    m.columns = ["column_name", "missing_count"]
    m["missing_pct"] = m["missing_count"] / total * 100
    return m.sort_values("missing_count", ascending=False)

display(missing_table(toronto_df))
display(missing_table(nyc_df))

# %% [markdown]
# ### 1.5 Duplicate Values Check

# %%
toronto_dupes = (
    toronto_df
    .groupBy("incident_id")
    .count()
    .filter(F.col("count") > 1)
)

print("Toronto duplicate incident_id count:", toronto_dupes.count())
display(toronto_dupes.orderBy(F.desc("count")).limit(20))

# %%
nyc_dupes = (
    nyc_df
    .groupBy("incident_id")
    .count()
    .filter(F.col("count") > 1)
)

print("NYC duplicate incident_id count:", nyc_dupes.count())
display(nyc_dupes.orderBy(F.desc("count")).limit(20))


# %% [markdown]
# ### 1.6 Summary of Data Sanity & Structure
#
# A series of sanity and structural checks were performed on the model-ready datasets for Toronto and New York City to ensure suitability for exploratory analysis and downstream modeling.
#
# **Schema and Duplicates**  
# Both datasets share an identical schema with consistent data types across all analytical fields. No duplicate records were detected in either dataset when grouped by `incident_id`, confirming one-to-one representation of emergency incidents.
#
# **Missing Values**  
# All feature variables are fully populated in both datasets. Missing values are observed only in the target variable `response_minutes`:
#
# - **Toronto:** 12,469 missing values (3.45%)
# - **New York City:** 422,625 missing values (28.49%)
#
# The higher proportion of missing response times in the NYC dataset reflects a substantial number of incidents without an observed response completion time, while Toronto exhibits a much smaller fraction of such cases. These missing values are retained intentionally and are interpreted as censored observations, enabling subsequent survival analysis.
#
# **Data Readiness**  
# No unintended row filtering, duplication, or imputation was identified during data preparation. The datasets are therefore confirmed to be structurally sound and analytically ready for:
# - distributional and tail-risk analysis using completed incidents, and  
# - censor-aware survival modeling using the `event_indicator` field.
#
# Overall, the model-ready datasets provide a reliable and consistent foundation for comparative analysis of emergency response performance across Toronto and New York City.

# %% [markdown]
# ## 2. Target Variable Exploration
# (Assuming response time or delay-based target)
#
# - Distribution (histogram / KDE)
# - Summary stats (mean, median, P90, P95)
# - Skewness & outliers
# - % of incidents breaching SLA thresholds (e.g. > X minutes)
#
# Distributions and summary statistics below use completed incidents only
# i.e. response_minutes IS NOT NULL.
# <br>Censored cases are handled separately in survival analysis

# %% [markdown]
# ### 2.1 Define Completed Incidents Subsets

# %%
toronto_complete = toronto_df.filter(F.col("response_minutes").isNotNull())
nyc_complete     = nyc_df.filter(F.col("response_minutes").isNotNull())

# %%
print("Toronto completed:", toronto_complete.count(), "/", toronto_df.count())
print("NYC completed:", nyc_complete.count(), "/", nyc_df.count())

# %% [markdown]
# ### 2.2 Distribution: Histogram (KDE)

# %%
toronto_pd = toronto_complete.select("response_minutes").sample(fraction=0.2, seed=42).toPandas()
nyc_pd = nyc_complete.select("response_minutes").sample(fraction=0.2, seed=42).toPandas()
fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

sns.histplot(
    toronto_pd["response_minutes"],
    bins=50, kde=True, ax=axes[0]
)
axes[0].set_title("Toronto Response Time Distribution (Completed Incidents)")
axes[0].set_ylabel("Count")

sns.histplot(
    nyc_pd["response_minutes"],
    bins=50, kde=True, ax=axes[1]
)
axes[1].set_title("NYC Response Time Distribution (Completed Incidents)")
axes[1].set_xlabel("Response Minutes")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


# %% [markdown]
# **Response Time Distributions (Completed Incidents)**
#
# The response time distributions for both Toronto and New York City are strongly right-skewed, indicating that while most incidents are handled within a relatively short time window, a non-trivial fraction experience substantially longer delays.
#
# Toronto’s distribution exhibits a pronounced peak around the central response range, followed by a long and heavy right tail. This aligns with the high skewness value observed earlier and reflects the presence of extreme delayed responses that are not captured by average response times.
#
# NYC shows a similarly right-skewed pattern but with a broader spread and a longer tail extending to higher response times. Compared to Toronto, NYC displays greater dispersion and a higher frequency of longer delays, consistent with its higher SLA breach rates and outlier prevalence.
#
# Overall, these distributions reinforce that response time behavior in both cities is dominated by tail risk, motivating the use of percentile-based metrics, outlier analysis, and survival-based modeling rather than reliance on mean response times alone.

# %% [markdown]
# ### 2.3 Summary Statistics (Mean, Median, P90, P95)

# %% [markdown]
# Helper Function

# %%
def response_summary(df):
    return df.select(
        F.round(F.mean("response_minutes"),4).alias("mean"),
        F.round(F.expr("percentile_approx(response_minutes, 0.5)"),4).alias("median"),
        F.round(F.expr("percentile_approx(response_minutes, 0.9)"),4).alias("p90"),
        F.round(F.expr("percentile_approx(response_minutes, 0.95)"),4).alias("p95"),
    )


# %%
# Compute summaries
toronto_stats = response_summary(toronto_complete).first()
nyc_stats     = response_summary(nyc_complete).first()

# Create Spark DataFrame
summary_df = spark.createDataFrame(
    [
        ("Toronto", toronto_stats["mean"], toronto_stats["median"],
         toronto_stats["p90"], toronto_stats["p95"]),
        ("NYC", nyc_stats["mean"], nyc_stats["median"],
         nyc_stats["p90"], nyc_stats["p95"]),
    ],
    ["city", "mean", "median", "p90", "p95"]
)

# Round for readability
summary_df = (
    summary_df
    .withColumn("mean", F.round("mean", 2))
    .withColumn("median", F.round("median", 2))
    .withColumn("p90", F.round("p90", 2))
    .withColumn("p95", F.round("p95", 2))
)

display(summary_df)

# %% [markdown]
# **Response Time Summary Statistics (Completed Incidents)**
#
# Summary statistics further highlight the right-skewed nature of response-time distributions in both cities. In Toronto, the mean response time (5.33 minutes) exceeds the median (5.10 minutes), with high-percentile values reaching 7.67 minutes at P90 and 8.80 minutes at P95. New York City exhibits consistently higher values across all metrics, with a mean of 5.88 minutes, a median of 5.50 minutes, and substantially higher tail percentiles (P90 = 8.68 minutes, P95 = 10.22 minutes).
#
# The divergence between median and high-percentile response times indicates that a relatively small fraction of delayed incidents disproportionately influences overall performance. The higher P90 and P95 values observed in NYC align with its greater skewness, higher outlier prevalence, and elevated SLA breach rates, underscoring more pronounced tail risk compared to Toronto.

# %% [markdown]
# ### 2.4 Skewness & Outliers

# %% [markdown]
# #### 2.4.1 Skewness

# %%
# Compute skewness values
toronto_skew = toronto_complete.select(F.skewness("response_minutes")).first()[0]
nyc_skew     = nyc_complete.select(F.skewness("response_minutes")).first()[0]

# Create Spark DataFrame
skewness_df = spark.createDataFrame(
    [
        ("Toronto", toronto_skew),
        ("NYC", nyc_skew),
    ],
    ["city", "response_minutes_skewness"]
)

# Round for readability
skewness_df = skewness_df.withColumn(
    "response_minutes_skewness",
    F.round("response_minutes_skewness", 4)
)

display(skewness_df)

# %% [markdown]
# **Skewness of Response Time Distributions**
#
# Both cities exhibit positively skewed response-time distributions, confirming the presence of long right tails. Toronto shows markedly higher skewness (4.78), indicating a heavier concentration of extreme delayed responses relative to its central tendency. New York City displays more moderate skewness (1.27), suggesting less extreme but still asymmetric response-time behavior.
#
# Despite NYC exhibiting higher mean and high-percentile response times, Toronto’s stronger skewness indicates that its distribution is more sharply peaked with rarer but more extreme delay events. Together with the outlier and SLA breach analyses, these results demonstrate that response-time performance in both cities is driven by tail behavior rather than average outcomes.

# %% [markdown]
# #### 2.4.2 Outlier Inspection (IQR-based, diagnostic only)

# %%
# Function to compute outlier bounds


def outlier_profile(df):
    # bounds from your existing function
    q1, q3, lower, upper = outlier_bounds(df)

    stats = (
        df.select(
            F.count("*").alias("n_total"),
            F.sum((F.col("response_minutes") < lower).cast("int")).alias("n_lower_outliers"),
            F.sum((F.col("response_minutes") > upper).cast("int")).alias("n_upper_outliers"),
        )
        .withColumn("n_outliers", F.col("n_lower_outliers") + F.col("n_upper_outliers"))
        .withColumn("pct_outliers", F.col("n_outliers") / F.col("n_total") * 100)
    ).first()

    return {
        "Q1": q1, "Q3": q3, "lower_bound": lower, "upper_bound": upper,
        "n_total": stats["n_total"],
        "n_lower_outliers": stats["n_lower_outliers"],
        "n_upper_outliers": stats["n_upper_outliers"],
        "n_outliers": stats["n_outliers"],
        "pct_outliers": stats["pct_outliers"],
    }

tor = outlier_profile(toronto_complete)
nyc = outlier_profile(nyc_complete)

outlier_full_df = spark.createDataFrame(
    [
        ("Toronto", tor["n_outliers"], tor["pct_outliers"], tor["Q1"], tor["Q3"], tor["lower_bound"], tor["upper_bound"], tor["n_total"], tor["n_lower_outliers"], tor["n_upper_outliers"] ),
        ("NYC", nyc["n_outliers"], nyc["pct_outliers"], nyc["Q1"], nyc["Q3"], nyc["lower_bound"], nyc["upper_bound"],
         nyc["n_total"], nyc["n_lower_outliers"], nyc["n_upper_outliers"]),
    ],
    ["city",  "n_outliers", "pct_outliers", "Q1", "Q3", "lower_bound", "upper_bound",
     "n_total", "n_lower_outliers", "n_upper_outliers"]
)

outlier_full_df = (
    outlier_full_df
    .withColumn("Q1", F.round("Q1", 2))
    .withColumn("Q3", F.round("Q3", 2))
    .withColumn("lower_bound", F.round("lower_bound", 2))
    .withColumn("upper_bound", F.round("upper_bound", 2))
    .withColumn("pct_outliers", F.round("pct_outliers", 2))
)

display(outlier_full_df)



# %% [markdown]
# **Outlier Analysis (IQR Method, Completed Incidents)**
#
# Outlier thresholds were defined using the interquartile range (IQR) method. Both Toronto and NYC exhibit a small but non-negligible proportion of response times exceeding the upper outlier bound, reflecting heavy right-tailed delay behavior.
#
# - **Toronto:**  
#   The upper outlier threshold is 9.58 minutes. Approximately 11,249 incidents exceed this threshold, representing 4.07% of completed incidents. Lower-bound outliers are rare and likely reflect minor timestamp irregularities rather than meaningful early responses.
#
# - **New York City:**  
#   The upper outlier threshold is higher at 10.68 minutes, with 43,467 incidents classified as upper outliers (4.20%). Similar to Toronto, lower-bound outliers are minimal.
#
# Despite differences in absolute thresholds and incident volume, both cities show a comparable proportion of extreme delays. These results reinforce the presence of substantial tail risk in emergency response times and motivate the use of tail-sensitive metrics and censor-aware modeling rather than reliance on average response times alone.
#
# **Note: Do not remove outliers — long delays are operationally meaningful**.

# %% [markdown]
# ### 2.5 Service Level Agreement(SLA) Breach Analysis
# SLA breach analysis measures the share of incidents for which response times exceed selected time thresholds. These thresholds represent practical performance benchmarks rather than strict policy guarantees. 
#
# By examining breach rates, the analysis highlights delayed responses that are masked by average response times and provides a clearer view of operational risk during high-demand or constrained conditions.
#
#

# %%
SLA_1 = 5    # minutes
SLA_2 = 8    # minutes


# %% [markdown]
# Helper Function

# %%
def sla_breach_pct(df, threshold):
    return (
        df.select(
            (F.sum((F.col("response_minutes") > threshold).cast("int")) / F.count("*") * 100)
            .alias("pct")
        )
        .first()["pct"]
    )


# %% [markdown]
# Compute

# %%
tor_5 = sla_breach_pct(toronto_complete, SLA_1)
tor_8 = sla_breach_pct(toronto_complete, SLA_2)

nyc_5 = sla_breach_pct(nyc_complete, SLA_1)
nyc_8 = sla_breach_pct(nyc_complete, SLA_2)

sla_df = spark.createDataFrame(
    [
        ("Toronto", SLA_1, tor_5),
        ("Toronto", SLA_2, tor_8),
        ("NYC", SLA_1, nyc_5),
        ("NYC", SLA_2, nyc_8),
    ],
    ["city", "sla_threshold_minutes", "pct_breach"]
)

sla_df = sla_df.withColumn(
    "pct_breach", F.round("pct_breach", 2)
)
sla_pivot_df = (
    sla_df
    .groupBy("city")
    .pivot("sla_threshold_minutes")
    .agg(F.first("pct_breach"))
    .orderBy("city")
)

sla_pivot_df = sla_pivot_df.selectExpr(
    "city",
    "`5` as pct_over_5min",
    "`8` as pct_over_8min"
)

display(sla_pivot_df)

# %% [markdown]
# **SLA Breach Results**
#
# SLA breach analysis reveals substantial differences in response-time reliability between Toronto and New York City. Using benchmark thresholds of 5 and 8 minutes, both cities exhibit high breach rates at stricter thresholds, indicating that delayed responses are common rather than exceptional.
#
# - At the **5-minute threshold**, approximately **61.05%** of NYC incidents exceed the benchmark, compared to **52.25%** in Toronto.
# - At the **8-minute threshold**, breach rates drop substantially but remain non-trivial, with **13.99%** of NYC incidents and **8.03%** of Toronto incidents exceeding this level.
#
# Across both thresholds, NYC consistently exhibits higher breach rates, suggesting greater tail risk in response times. These findings reinforce earlier evidence from skewness and outlier analyses that average response times mask meaningful operational delays, and that tail-sensitive metrics are essential for evaluating emergency response performance.

# %% [markdown]
# ### 2.6 Censoring Awareness (For Survival Analysis)
# Censoring Validation

# %%
# Toronto
toronto_censoring = (
    toronto_df
    .agg(
        F.count("*").alias("n_total"),
        F.sum((F.col("event_indicator") == 1).cast("int")).alias("n_completed"),
        F.sum((F.col("event_indicator") == 0).cast("int")).alias("n_censored")
    )
    .withColumn("pct_censored", F.round(F.col("n_censored") / F.col("n_total") * 100, 2))
    .withColumn("city", F.lit("Toronto"))
)

# NYC
nyc_censoring = (
    nyc_df
    .agg(
        F.count("*").alias("n_total"),
        F.sum((F.col("event_indicator") == 1).cast("int")).alias("n_completed"),
        F.sum((F.col("event_indicator") == 0).cast("int")).alias("n_censored")
    )
    .withColumn("pct_censored", F.round(F.col("n_censored") / F.col("n_total") * 100, 2))
    .withColumn("city", F.lit("NYC"))
)

censoring_summary = toronto_censoring.unionByName(nyc_censoring)

display(censoring_summary.select(
    "city", "n_total", "n_completed", "n_censored", "pct_censored"
))



# %% [markdown]
# **Censoring Summary**
#
# The censoring structure differs substantially between Toronto and New York City. In Toronto, 12,469 incidents (3.45%) are censored, indicating that the vast majority of incidents have an observed response completion time. In contrast, NYC exhibits a much higher degree of censoring, with 422,625 incidents (28.49%) lacking an observed response time.
#
# This divergence reflects structural and operational differences in data recording and incident resolution across the two cities. For descriptive and distributional analyses, only completed incidents were used to ensure accurate characterization of observed response-time behavior. However, censored incidents are intentionally retained in the model-ready datasets and explicitly modeled using the `event_indicator` variable in subsequent survival analysis.
#
# Accounting for censoring is therefore essential for valid cross-city comparison and for avoiding bias that would arise from analyzing completed incidents alone, particularly in the NYC dataset.

# %% [markdown]
# ### 2.7 Summary of Target Variable Exploration
#
# Exploratory analysis of the response time target reveals strongly right-skewed distributions in both Toronto and New York City, with long tails driven by a minority of substantially delayed incidents. Mean response times exceed median values, and high-percentile metrics (P90 and P95) indicate pronounced tail risk that is not captured by average performance measures alone. Outlier and SLA breach analyses further confirm that delayed responses are operationally meaningful and occur with non-trivial frequency in both cities, particularly in NYC.
#
# Censoring is an important feature of the data, with a small proportion of censored incidents in Toronto (3.45%) and a substantially larger share in NYC (28.49%). To ensure valid interpretation, distributional analyses were conducted using completed incidents only, while censored cases are retained for subsequent survival-based modeling. Together, these findings motivate the use of tail-sensitive and censor-aware analytical approaches in the modeling stages that follow.

# %% [markdown]
#

# %% [markdown]
# ## 3. Temporal Patterns
# Create / validate:
# - Hour of day
# - Day of week
# - Month / season
# - Weekend vs weekday
#
# Explore:
# - Avg & P90 response time by hour
# - Incident volume by hour
# - Heatmap: hour × day_of_week
#
# **Important scope note (keep this logic consistent):**
#
# - Response-time statistics → completed incidents only
# - Incident volume → all incidents (completed + censored)
#
# **P90** is the response time within which 90% of incidents are completed, highlighting delays in the slowest 10% of cases.
#
# P90 refers to the response time value below which 90% of incidents fall, rather than the average response time of the fastest 90% of incidents. As a percentile-based metric, P90 captures the boundary of slower response behavior, whereas trimmed means summarize typical performance after excluding extreme delays.
#

# %% [markdown]
# ### 3.1 Validate Temporal Features
# - hour (0–23)
# - day_of_week (1=Sunday … 7=Saturday)
# - month (1–12)
# - season

# %%
toronto_temporal = (
    toronto_df.select(
        F.lit("Toronto").alias("city"),
        F.min("hour").alias("min_hour"),
        F.max("hour").alias("max_hour"),
        F.min("day_of_week").alias("min_dow"),
        F.max("day_of_week").alias("max_dow"),
        F.min("month").alias("min_month"),
        F.max("month").alias("max_month")
    )
)

nyc_temporal = (
    nyc_df.select(
        F.lit("NYC").alias("city"),
        F.min("hour").alias("min_hour"),
        F.max("hour").alias("max_hour"),
        F.min("day_of_week").alias("min_dow"),
        F.max("day_of_week").alias("max_dow"),
        F.min("month").alias("min_month"),
        F.max("month").alias("max_month")
    )
)

temporal_validation_df = (
    toronto_temporal
    .unionByName(nyc_temporal)
    .toPandas()
    .set_index("city")   # make city the column header anchor
    .T                   # transpose
    .reset_index()
)

temporal_validation_df.rename(columns={"index": "metric"}, inplace=True)

display(temporal_validation_df)


# %% [markdown]
# Temporal validation confirms that hour, day-of-week, and month variables fall within expected ranges for both Toronto and NYC, indicating correct temporal encoding and readiness for downstream temporal pattern analysis.

# %% [markdown]
# ### 3.2 Average & P90 Response Time by Hour
# Use completed incidents only. 

# %% [markdown]
# Helper Function

# %%
def hourly_response_stats(df):
    return (
        df.filter(F.col("response_minutes").isNotNull())
          .groupBy("hour")
          .agg(
              F.round(F.mean("response_minutes"), 2).alias("avg_response"),
              F.round(F.expr("percentile_approx(response_minutes, 0.9)"), 2).alias("p90_response")
          )
          .orderBy("hour")
    )


# %% [markdown]
# #### 3.2.1 Average Respone by hour (Toronto vs NYC)

# %%
tor = hourly_response_stats(toronto_df).withColumn("city", F.lit("Toronto"))
nyc = hourly_response_stats(nyc_df).withColumn("city", F.lit("NYC"))

hourly_combined = tor.unionByName(nyc)

avg_pivot = (
    hourly_combined
    .groupBy("hour")
    .pivot("city", ["Toronto", "NYC"])
    .agg(F.first("avg_response"))
    .orderBy("hour")
)

display(avg_pivot)



# %% [markdown]
# #### 3.2.1 P90 response by hour (Toronto vs NYC)
# **P90** is the response time within which 90% of incidents are completed, highlighting delays in the slowest 10% of cases.
#

# %%
p90_pivot = (
    hourly_combined
    .groupBy("hour")
    .pivot("city", ["Toronto", "NYC"])
    .agg(F.first("p90_response"))
    .orderBy("hour")
)

display(p90_pivot)


# %% [markdown]
# #### Hourly Response-Time Patterns Summary
#
# Hourly analysis reveals clear diurnal patterns in both cities. Average response times are lowest during daytime and evening hours and increase during late-night and early-morning periods, with NYC consistently exhibiting higher average response times than Toronto across all hours. While average differences are modest (approximately 0.3–0.6 minutes), tail behavior differs more substantially.
#
# P90 response times show pronounced overnight and early-morning delays, particularly in NYC, where the slowest 10% of responses exceed Toronto’s P90 by more than one minute during several off-peak hours. These patterns indicate that response-time risk is driven less by typical daytime operations and more by reduced overnight capacity and elevated tail delays, especially in NYC.
#

# %% [markdown]
# ### 3.3 Incident Volumne by Hour
# Volume includes all incidents, regardless of completion.

# %%
def hourly_volume(df):
    return (
        df.groupBy("hour")
          .count()
          .withColumnRenamed("count", "incident_volume")
          .orderBy("hour")
    )



# %%
toronto_hourly_vol = (
    toronto_df
    .groupBy("hour")
    .count()
    .withColumn("city", F.lit("Toronto"))
)

nyc_hourly_vol = (
    nyc_df
    .groupBy("hour")
    .count()
    .withColumn("city", F.lit("NYC"))
)

hourly_volume_combined = toronto_hourly_vol.unionByName(nyc_hourly_vol)

hourly_volume_pivot = (
    hourly_volume_combined
    .groupBy("hour")
    .pivot("city", ["Toronto", "NYC"])
    .agg(F.first("count"))
    .orderBy("hour")
)

display(hourly_volume_pivot)


# %% [markdown]
# **Incident Volume by Hour Summary**
#
# Incident volume exhibits a strong diurnal pattern in both Toronto and New York City. Call volumes are lowest during late-night and early-morning hours (approximately 02:00–05:00) and increase steadily throughout the day, peaking during late afternoon and early evening. NYC consistently experiences substantially higher incident volumes than Toronto at every hour, often by a factor of three to four.
#
# The temporal alignment between peak incident volume and elevated response-time levels suggests that demand intensity is an important driver of response-time variation. However, the presence of higher response-time tail risk during overnight hours—despite lower call volumes—indicates that capacity constraints and staffing availability likely play a more significant role during off-peak periods.
#

# %% [markdown]
# ### 3.4 Heatmap for  Hours x Day of Week

# %% [markdown]
# Prepare heatmap data (only for completed incidents)

# %%
def heatmap_data(df):
    return (
        df.filter(F.col("response_minutes").isNotNull())
          .groupBy("day_of_week", "hour")
          .agg(F.round(F.mean("response_minutes"), 2).alias("avg_response"))
          .orderBy("day_of_week", "hour")
    )


# %%
# Convert to Pandas for plotting:
toronto_heat_pd = heatmap_data(toronto_df).toPandas()
nyc_heat_pd     = heatmap_data(nyc_df).toPandas()

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- Toronto (top) ---
sns.heatmap(
    toronto_heat_pd.pivot(
        index="day_of_week",
        columns="hour",
        values="avg_response"
    ),
    ax=axes[0],
    cmap="YlOrRd"
)
axes[0].set_title("Toronto: Avg Response Time by Hour × Day of Week")
axes[0].set_ylabel("Day of Week")

# --- NYC (bottom) ---
sns.heatmap(
    nyc_heat_pd.pivot(
        index="day_of_week",
        columns="hour",
        values="avg_response"
    ),
    ax=axes[1],
    cmap="YlOrRd"
)
axes[1].set_title("NYC: Avg Response Time by Hour × Day of Week")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Day of Week")

plt.tight_layout()
plt.show()


# %% [markdown]
# **Hour × Day-of-Week Response-Time Patterns**
#
# The heatmaps reveal clear and consistent temporal structure in both cities. Average response times are highest during overnight and early-morning hours (approximately 00:00–06:00) across most days of the week, with gradual improvement during daytime hours. Toronto exhibits relatively stable response times throughout the week, with modest weekday–weekend variation.
#
# In contrast, NYC shows uniformly higher response times across all hours, with more pronounced overnight delays and less recovery during daytime periods. The persistence of elevated response times during low-demand overnight hours suggests that capacity constraints, staffing levels, or operational coverage—rather than demand alone—play a key role in shaping response-time performance, particularly in NYC.
#

# %% [markdown]
# ### 3.5 Weekend vs Weekday Analysis

# %%
def add_weekend_flag(df):
    return df.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin(1, 7), "Weekend").otherwise("Weekday")
    )

toronto_wd = add_weekend_flag(toronto_df)
nyc_wd     = add_weekend_flag(nyc_df)

# %%
# Toronto
toronto_weekend_stats = (
    toronto_wd
    .filter(F.col("response_minutes").isNotNull())
    .groupBy("is_weekend")
    .agg(
        F.round(F.mean("response_minutes"), 2).alias("avg_response"),
        F.round(F.expr("percentile_approx(response_minutes, 0.9)"), 2).alias("p90_response")
    )
    .withColumn("city", F.lit("Toronto"))
)

# NYC
nyc_weekend_stats = (
    nyc_wd
    .filter(F.col("response_minutes").isNotNull())
    .groupBy("is_weekend")
    .agg(
        F.round(F.mean("response_minutes"), 2).alias("avg_response"),
        F.round(F.expr("percentile_approx(response_minutes, 0.9)"), 2).alias("p90_response")
    )
    .withColumn("city", F.lit("NYC"))
)

# Combine
weekend_comparison = (
    toronto_weekend_stats
    .unionByName(nyc_weekend_stats)
    .select("city", "is_weekend", "avg_response", "p90_response")
    .orderBy("city", "is_weekend")
)

display(weekend_comparison)


# %%
weekend_pd = weekend_comparison.toPandas()

plt.figure(figsize=(8, 5))
sns.set_theme(style="whitegrid")
ax = sns.barplot(
    data=weekend_pd,
    x="is_weekend",
    y="p90_response",
    hue="city"
)

ax.set_title("P90 Response Time: Weekday vs Weekend", fontsize=12)
ax.set_xlabel("Day Type")
ax.set_ylabel("P90 Response Time (minutes)")

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=0.5)

# Move legend outside
ax.legend(
    title="City",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(8, 5))
sns.set_theme(style="whitegrid")
ax = sns.barplot(
    data=weekend_pd,
    x="is_weekend",
    y="avg_response",
    hue="city"
)

ax.set_title("Average Response Time: Weekday vs Weekend", fontsize=12)
ax.set_xlabel("Day Type")
ax.set_ylabel("Average Response Time (minutes)")

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=0.5)

# Move legend outside
ax.legend(
    title="City",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Weekday vs Weekend Response-Time Comparison**
#
# Both Toronto and New York City exhibit slightly lower average response times on weekends compared to weekdays, consistent with reduced traffic congestion and lower overall incident demand during non-working days. This improvement is reflected not only in average response times but also in high-percentile (P90) values, indicating that weekend conditions influence response performance broadly across the distribution.
#
# The gap between average and P90 response times in both cities reflects a right-skewed response-time distribution, where a small proportion of incidents experience substantially longer delays. However, the similarity in weekday–weekend patterns across both metrics suggests that weekend effects do not disproportionately alter extreme response delays, but instead provide modest, uniform improvements across typical and slower response cases.
#
#

# %% [markdown]
# ### 3.6 Summary of Temporal Patterns
# Temporal analysis reveals strong diurnal and weekly structure in emergency response performance across both cities. Response times and incident volumes vary systematically by hour of day, with slower responses occurring during overnight and early-morning periods and higher volumes during daytime and evening hours. Weekend response times are slightly lower than weekday levels, consistent with reduced traffic and demand, and this pattern is observed uniformly across both average and high-percentile metrics.
#
# Overall, temporal variation influences response performance across the entire distribution, highlighting the importance of accounting for time-of-day and day-of-week effects in subsequent modeling and risk analysis.
#

# %% [markdown]
# ## 4. Spatial/ Operational Signals
# **Toronto**
# - Ward / Station Area
# - Alarm level
# - Call source
#
# **NYC**
# - Borough
# - Incident type
# - Alarm level
#
# Explore:
# - Response time by area (mean + tail)
# - Volume vs delay by area
# - High-volume ≠ fast response (important insight)

# %% [markdown]
# **Toronto Station Area (`location_area`)**

# %%
display(
  toronto_df
    .select(col("location_area").alias("station_area"))
    .where(col("location_area").isNotNull())
    .distinct()
    .orderBy("station_area")
)

# %% [markdown]
# **NYC Borough (`location_area`)**

# %% [markdown]
# **Appendix A: Toronto Fire Station Code Reference (Contextual)**
#
# **Purpose:**
#
# - Help readers understand what the codes refer to
# - Provide organizational context
# - Not used in modeling logic
#
#
# Toronto Fire Services operates approximately 84 fire stations, organized into multiple divisions. Publicly available sources document that station numbering reflects organizational structure and home station assignment (e.g., Station 312 serving the Yorkville neighbourhood). The table below provides a contextual reference for selected station codes based on secondary sources. This mapping is provided for interpretability only and is not used in model construction.
#
#
#
# | Station Code | Referenced Area / Notes      | Source             |
# | ------------ | ---------------------------- | ------------------ |
# | 312          | Yorkville neighbourhood      | Wikipedia          |
# | 111–116      | Central / Downtown (general) | torontofirefan.com |
# | 424–431      | Division 4 (general)         | torontofirefan.com |
#
# **Note: Sources are secondary and informational; boundaries may not reflect official administrative definitions.**
#

# %%
display(
  nyc_df
    .select(col("location_area").alias("Borough"))
    .where(col("location_area").isNotNull())
    .distinct()
    .orderBy("Borough")
)


# %% [markdown]
# ### 4.1 Response time(min) by area (mean + tail)

# %% [markdown]
# Helper Functions

# %%

def area_response_perf(df, area_col="location_area", p=0.9):
    """
    4.1 Response time by area (mean + tail), computed on completed incidents only.
    Returns: area, n_completed, mean_response, pXX_response
    """
    p_label = f"p{int(p*100)}_response"

    return (
        df.filter(F.col("response_minutes").isNotNull())
          .groupBy(area_col)
          .agg(
              F.count("*").alias("n_completed"),
              F.round(F.mean("response_minutes"), 2).alias("mean_response"),
              F.round(F.expr(f"percentile_approx(response_minutes, {p})"), 2).alias(p_label)
          )
          .orderBy(F.desc(p_label))  # slowest by tail risk (P90) on top
    )

# Toronto (Station area)
toronto_area_perf = (
    area_response_perf(toronto_df, area_col="location_area", p=0.9)
      .withColumn("city", F.lit("Toronto"))
)


# %% [markdown]
# **Calculated Metrics in This Section**
# <br>Following Columns are calculated in this section
# - n_total: Total Incidents
# - n_censored: Incidents with NULL response time (This will be treated as censored flag in survival Analysis)
# - pct_censored: Percentage of Censored Response Time
# - n_completed: Incidents with Response Time Available
# - mean_response: Mean value of response minutes
# - p90_response: 90th percentile response

# %% [markdown]
# **Rationale for Using P90 Response Time**
# <br>Mean response times provide a useful summary of typical performance, but they are insufficient for evaluating emergency response operational risk, which is driven by delays in the upper tail of the distribution rather than average cases. Emergency response time distributions are right-skewed, with a small but critical proportion of incidents experiencing substantial delays.
# <br>To capture this tail risk, we report the 90th percentile (P90) response time by area. P90 represents the response time within which 90% of completed incidents are handled, directly reflecting worst-case conditions affecting a meaningful fraction of incidents. This metric is therefore more appropriate than the mean alone for identifying high-risk station areas and boroughs, particularly in the context of capacity constraints, surge conditions, and service-level performance evaluation.
#

# %% [markdown]
# #### 4.1.1 Toronto Response Time(min) by Station_area (`location_area`)

# %%
# Toronto (Station area)
toronto_area_perf = (
    area_response_perf(toronto_df, area_col="location_area", p=0.9)
      .withColumn("city", F.lit("Toronto"))
)
display(toronto_area_perf)

# %% [markdown]
# Slowest Area by P90

# %%
display(toronto_area_perf.orderBy(F.desc("p90_response")).limit(15))

# %% [markdown]
# #### 4.1.2 NYC Response Time(min) Borough (`location_area`)

# %%
# NYC (Borough)
nyc_area_perf = (
    area_response_perf(nyc_df, area_col="location_area", p=0.9)
      .withColumn("city", F.lit("NYC"))
)

display(nyc_area_perf)

# %%
display(nyc_area_perf.orderBy(F.desc("p90_response")).limit(15))


# %% [markdown]
# ### 4.2 Volume by Area + Censoring (Data Completeness)

# %% [markdown]
# Helper Function

# %%
def area_volume_quality(df, area_col="location_area"):
    """
    4.2 Volume by area + censoring (missing response time).
    Returns: area, n_total, n_censored, pct_censored, n_completed
    """
    return (
        df.groupBy(area_col)
          .agg(
              F.count("*").alias("n_total"),
              F.sum(F.col("response_minutes").isNull().cast("int")).alias("n_censored"),
              F.sum(F.col("response_minutes").isNotNull().cast("int")).alias("n_completed"),
          )
          .withColumn(
              "pct_censored",
              F.round(F.col("n_censored") / F.col("n_total") * 100, 2)
          )
          .orderBy(F.desc("n_total"))  # highest volume on top
    )


# %% [markdown]
# #### 4.2.1 Toronto Incident Volumne by Area

# %%
toronto_area_vol = (
    area_volume_quality(toronto_df, area_col="location_area")
      .withColumn("city", F.lit("Toronto"))
)
display(toronto_area_vol)

# %% [markdown]
# Most Censored Area

# %%
# most censored areas (highest pct missing response time)
display(toronto_area_vol.orderBy(F.desc("pct_censored")).limit(15))

# %% [markdown]
# #### 4.2.2 NYC Incident Volumne by Area

# %%
nyc_area_vol = (
    area_volume_quality(nyc_df, area_col="location_area")
      .withColumn("city", F.lit("NYC"))
)
display(nyc_area_vol)

# %% [markdown]
# Most Censored Area (higest percent of missing response time)

# %%
display(nyc_area_vol.orderBy(F.desc("pct_censored")).limit(15))

# %% [markdown]
# ### 4.3 Demand–Performance Relationship by Area (Exploratory)
# Purpose:
# - Explore whether higher demand correlates with slower response
#
# This subsection explores the relationship between incident volume and average response-time performance at the area level. By plotting total incident volume against mean response time, we assess whether higher demand is associated with systematically slower responses, and whether this relationship differs between Toronto and NYC.

# %%
area_pd = (
    area_compare
        .filter(F.col("mean_response").isNotNull())
        .select("city", "location_area", "n_total", "mean_response")
        .toPandas()
)

# %% [markdown]
# #### 4.3.1 Toronto: Volume vs Mean Response Time (Typical Performance)
# Mean Response Time is used to observed Typical Performance

# %%
TOP_N = 15  # or 20
toronto_top_areas = (
    area_compare
        .filter(F.col("city") == "Toronto")
        .orderBy(F.desc("n_total"))
        .limit(TOP_N)
        .select("location_area")
)

toronto_pd_top = (
    area_compare
        .filter(F.col("city") == "Toronto")
        .join(toronto_top_areas, on="location_area", how="inner")
        .select("location_area", "n_total", "mean_response")
        .toPandas()
)

# %%
city_mean = toronto_pd_top["mean_response"].mean()
plt.figure(figsize=(10,5))

ax = sns.scatterplot(
    data=toronto_pd_top,
    x="location_area",
    y="mean_response",
    size="n_total",
    hue="n_total",                 # color by volume
    palette="viridis",             # perceptually uniform
    sizes=(80, 600),
    alpha=0.8,
    legend="brief"
)
ax.axhline(city_mean, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
ax.text(
    0.7, city_mean + 0.03,
    f"Citywide mean = {city_mean:.2f} min",
    transform=ax.get_yaxis_transform(),
    color="blue",
    fontsize=9,
    va="bottom"
)
ax.set_title(
    f"Toronto: Mean Response Time by Station Area (Top {TOP_N} by Volume, Colored by Volume)",
    fontsize=14,            
    fontweight="bold"

)
ax.set_xlabel("Station Area", fontsize=12 )
ax.set_ylabel("Mean Response Time (Completed Incidents)", fontsize=12 )
ax.tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()


# %% [markdown]
# **Top 15 Stattion by Volumne Analysis**
#
# Among the top 15 station areas by incident volume in Toronto, mean response times are tightly clustered, with most areas exhibiting typical response times between approximately 4.2 and 5.6 minutes. High-volume station areas do not consistently exhibit slower average response times, indicating that demand volume alone does not explain differences in typical performance. This suggests that operational capacity and deployment are generally effective at accommodating high demand, while residual variation in mean response time may reflect localized structural or geographic factors.

# %%
plt.figure(figsize=(10,5))

ax = sns.scatterplot(
    data=toronto_pd_top,
    x="location_area",
    y="mean_response",
    size="n_total",
    hue="location_area",           # categorical colors
    palette="tab20",               # max 20 distinct colors
    sizes=(80, 600),
    alpha=0.8,
    legend=False                   # turn off legend (too crowded)
)

ax.set_title(
    f"Toronto: Mean Response Time by Station Area (Top {TOP_N} by Volume)"
)
ax.set_xlabel("Station Area")
ax.set_ylabel("Mean Response Time (Completed Incidents)")
ax.tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()

# %% [markdown]
# #### 4.3.2 NYC: 

# %%
# NYC data to pandas
nyc_pd = (
    area_compare
        .filter((F.col("city") == "NYC") & F.col("mean_response").isNotNull())
        .select("location_area", "n_total", "mean_response")
        .toPandas()
)
nyc_pd

# %%
city_mean = nyc_pd["mean_response"].mean()
plt.figure(figsize=(8,4))
ax = sns.scatterplot(
    data=nyc_pd,
    x="location_area",
    y="mean_response",
    size="n_total",
    hue="n_total",
    palette="viridis",
    sizes=(120, 900),
    alpha=0.85
)

ax.set_title("NYC: Mean Response Time by Borough",
             fontsize=14,
             fontweight="bold")
ax.set_xlabel("Borough", fontsize=11)
ax.set_ylabel("Mean Response Time (Completed Incidents)", fontsize=11)
ax.tick_params(axis="x", rotation=30,labelsize=9)
ax.axhline(city_mean, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
ax.text(
    0.3, city_mean -0.15,                      # move right a bit; adjust 0.55–0.75
    f"Citywide mean = {city_mean:.2f} min",
    transform=ax.get_yaxis_transform(),          # x in [0,1], y in data coords
    color="blue",
    fontsize=9,
    ha="left",
    va="top",
)
# Fix Y-axis
ax.set_ylim(0, 10)
# Make lengend size smaller
ax.legend(
    title="Incident Volume",
    loc="lower right",
    bbox_to_anchor=(0.92, 0.08),
    frameon=False,
    fontsize=8,
    title_fontsize=9
)
# Shrink bubble sizes in legend
leg = ax.get_legend()
for h in leg.legend_handles:
    try:
        h.set_sizes([50])   # shrink legend bubbles only
    except:
        pass
# Fix clipping of Y-axis label
plt.subplots_adjust(left=0.16, right=0.78)

plt.show()


# %% [markdown]
# **Summary**
#
# Mean response times across NYC boroughs are tightly clustered, ranging from roughly 5.3 to 6.5 minutes, despite large differences in incident volume. Brooklyn, the highest-volume borough, performs below the citywide mean, indicating that higher demand does not necessarily result in slower typical response. <br>Overall, borough-level differences in mean response time are modest, suggesting that factors beyond volume such as geography or traffic conditions—likely drive residual variation.

# %% [markdown]
# ### 4.4 Operational Signals Effects

# %% [markdown]
# Helper Function

# %%
def response_by_category(df, colname, p=0.9):
    """
    Categorical breakdown on completed incidents only.
    """
    return (
        df.filter(F.col("response_minutes").isNotNull())
          .groupBy(colname)
          .agg(
              F.count("*").alias("n_completed"),
              F.round(F.mean("response_minutes"), 2).alias("mean_response"),
              F.round(F.expr(f"percentile_approx(response_minutes, {p})"), 2).alias(f"p{int(p*100)}_response")
          )
          .orderBy(F.desc("n_completed"))
    )


# %% [markdown]
# #### 4.4.1 Alarm Level Effect
# Based on p90

# %% [markdown]
# ##### 4.4.1.1 Toronto Response Time by Alarm Level

# %%
display(response_by_category(toronto_df, "unified_alarm_level", p=0.9))

# %% [markdown]
# ##### 4.4.1.2 NYC Response Time Table by Alaram Level

# %%
display(response_by_category(toronto_df, "unified_alarm_level", p=0.9))

# %% [markdown]
# #### 4.4.2 Call Source Effects
# Based on P90

# %% [markdown]
# ##### 4.4.2.1 Torontal Response Time by Call Source

# %%
display(response_by_category(toronto_df, "unified_call_source", p=0.9))


# %% [markdown]
# ##### 4.4.2.2 NYC Reponse Time Call Source

# %%
display(response_by_category(nyc_df, "unified_call_source", p=0.9))

# %% [markdown]
# #### 4.4.3 Summary of Operation Singals and Call Source Effects
# Across both Toronto and NYC, **alarm level is strongly associated with response prioritization**, with higher alarm levels exhibiting **shorter mean and P90 response times**, despite representing a very small fraction of total incidents. This pattern suggests that escalated incidents are dispatched and responded to more rapidly, while level-1 alarms dominate overall system performance.
#
# When stratified by call source, **differences in mean response times are modest**, but **tail delays (P90)** vary more noticeably. Public and EMS-initiated calls tend to exhibit **higher P90 response times** than alarm-system-initiated incidents in both cities, indicating that extreme delays are more sensitive to call origin than typical performance. Small-volume categories are reported for completeness but are not emphasized in interpretation.

# %% [markdown]
# ### 4.5 High-Volume Areas and Response Performance (Ranked Comparison)

# %% [markdown]
# #### 4.5.1 Toroton Volume Area and Response Performance

# %%
display(
    toronto_area_stats
    .select("location_area", "n_total", "mean_response", "p90_response")
    .orderBy(F.desc("n_total"))
    .limit(15)
)

# %% [markdown]
# #### 4.5.2 NYC Volume Area and Response Performance

# %%
display(
    nyc_area_stats
    .select("location_area", "n_total", "mean_response", "p90_response")
    .orderBy(F.desc("n_total"))
    .limit(15)
)


# %% [markdown]
# #### 4.5.3 Summary of High-Volume Areas and Response Performance.
# Among Toronto’s highest-volume station areas, both mean and P90 response times vary substantially, ranging from approximately **4.1 to 5.6 minutes** for the mean and **5.8 to 7.9 minutes** for the P90, despite similar incident volumes. Several high-demand areas maintain comparatively fast response times, while others exhibit elevated tail delays, indicating that demand volume alone does not explain performance differences at the station level.
#
# In NYC, borough-level response performance also varies across high-volume areas, with the **Bronx and Manhattan** exhibiting higher mean and P90 response times than **Brooklyn** and **Queens**, despite comparable demand. 
#
# Overall, these ranked comparisons reinforce that high incident volume does not systematically correspond to faster or slower response, and that additional operational and spatial factors likely drive observed performance variation.

# %% [markdown]
# ### 4.6 Exploratory Clustering (Toronto) and NYC Spatial Analysis
#
# To better understand spatial variation in emergency response performance, exploratory clustering was conducted for Toronto location areas using aggregated demand and response-time characteristics. Toronto contains a sufficiently large number of unique location areas to allow meaningful segmentation of operational environments. K-means clustering was therefore applied to identify groups of locations with similar workload intensity, response-time patterns, and incident severity characteristics. This segmentation helps reveal whether distinct operational risk profiles exist across different parts of the city.
#
# The NYC dataset, however, contains only five unique location areas. With such limited spatial granularity, unsupervised clustering would not produce stable or interpretable groupings, as the number of clusters would approach the number of observations. Instead of forcing clustering, spatial patterns in NYC are examined through descriptive comparison of key metrics across locations. These include mean response time, tail-response time (P90), call volume, and short-term demand intensity. 
#
# This combined approach ensures that spatial segmentation is applied where statistically appropriate (Toronto), while NYC spatial variation is analyzed using direct comparison methods that remain analytically rigorous and interpretable. Together, these analyses provide a comprehensive view of how response-time risk and demand characteristics vary across locations in both cities, supporting subsequent modeling and cross-city comparison.

# %% [markdown]
# | Feature          | Interpretation                  |
# | ---------------- | ------------------------------- |
# | call_volume      | workload                        |
# | mean_response    | average delay                   |
# | p90_response     | tail-delay risk                 |
# | delay_rate       | % incidents exceeding threshold |
# | mean_calls_30    | short-term demand               |
# | mean_calls_60    | sustained demand                |
# | mean_alarm_level | incident severity               |
#

# %% [markdown]
# #### 4.6.1 Clustering for Toronto Data

# %% [markdown]
# **Helper Function: Build location-level features**

# %%
def build_location_features(df):
    """
    Aggregates incident-level data into location_area-level risk/demand features.
    Uses event_indicator ONLY to filter valid response times (NOT as delay rate).
    """

    df_valid = (
        df.filter(F.col("response_minutes").isNotNull())
          .filter(F.col("location_area").isNotNull())
    )

    location_df = (
        df_valid.groupBy("location_area")
        .agg(
            F.count("*").alias("call_volume"),
            F.avg("response_minutes").alias("mean_response"),
            F.expr("percentile(response_minutes, 0.9)").alias("p90_response"),
            F.expr("percentile(response_minutes, 0.5)").alias("median_response"),
            F.avg("calls_past_30min").alias("mean_calls_30"),
            F.avg("calls_past_60min").alias("mean_calls_60"),
            F.avg("unified_alarm_level").alias("mean_alarm_level"),
            # Optional: share of high-alarm incidents (define high alarm as >= 3; adjust if needed)
            F.avg(F.when(F.col("unified_alarm_level") >= 3, 1).otherwise(0)).alias("high_alarm_share"),
        )
        # Optional: avoid tiny groups that can distort clustering
        .filter(F.col("call_volume") >= 50)
    )

    return location_df


# %% [markdown]
# **Helper Function: Main Funtion to run K-Mean and Plot**

# %%
def cluster_locations(location_df, city_name="City", k=4):
    """
    Runs KMeans clustering on location_area-level features and produces:
    - Elbow plot
    - PCA scatter plot
    - Cluster summary table
    """

    pdf = location_df.toPandas().set_index("location_area")

    # Keep only numeric columns (safe)
    X = pdf.select_dtypes(include=[np.number]).copy()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Elbow plot (2..7) ---
    inertia = []
    k_range = range(2, 8)
    for kk in k_range:
        km = KMeans(n_clusters=kk, random_state=42, n_init="auto")
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    plt.figure(figsize=(7,5))
    plt.plot(list(k_range), inertia, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title(f"{city_name}: Elbow Method for Location Clustering")
    plt.show()

    # --- Fit final model ---
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    pdf["cluster"] = kmeans.fit_predict(X_scaled)

    # --- PCA plot ---
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    # --- PCA loadings (feature contributions to PC1 & PC2) ---
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=X.columns
    )

    print(f"\n{city_name} PCA Feature Loadings:")
    print(loadings.round(3))

    plt.figure(figsize=(7,5))
    plt.scatter(coords[:, 0], coords[:, 1], c=pdf["cluster"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{city_name}: Location Risk Clusters (PCA projection)")
    plt.show()

    # --- Summary table ---
    summary = pdf.groupby("cluster")[X.columns].mean().round(2).sort_index()

    # Top locations per cluster (by tail risk)
    top_by_tail = (
        pdf.sort_values(["cluster", "p90_response"], ascending=[True, False])
           .groupby("cluster")
           .head(5)[["p90_response", "mean_response", "call_volume"]]
    )

    return pdf, summary, top_by_tail



# %%
# Toronto
tor_loc = build_location_features(toronto_df)
tor_pdf, tor_summary, tor_top = cluster_locations(tor_loc, city_name="Toronto", k=4)

# %% [markdown]
# ##### Summary: Exploratory Clustering of Toronto Location Risk Profiles 
# _**Note: (This draft is to be placed in report. The shorter version will be replaced with this section later)**_
#
# To explore spatial variation in emergency response performance across Toronto, an unsupervised clustering analysis was conducted at the location-area level. Each location was characterized using aggregated operational metrics, including call volume, mean and P90 response time, short-term demand intensity (calls in the past 30 and 60 minutes), and alarm severity indicators. Features were standardized prior to clustering to ensure comparability across variables with different scales.
#
# An elbow-method assessment indicated that a four-cluster solution provided a reasonable balance between model simplicity and explanatory power, as inertia declined sharply between two and four clusters before stabilizing. K-means clustering with four clusters was therefore selected to identify distinct operational risk profiles across locations.
#
# To visualize cluster separation, Principal Component Analysis (PCA) was applied to project the multi-dimensional feature space into two dimensions. The PCA scatter plot reveals clear grouping of location areas, indicating that Toronto emergency response environments can be segmented into several distinct operational profiles.
#
# PCA loadings suggest that the first principal component (PC1) primarily reflects overall demand intensity and workload. Variables such as call volume and short-term demand indicators (calls in the past 30 and 60 minutes) load positively on PC1, while response-time metrics load negatively. As a result, locations positioned on the positive side of PC1 tend to experience higher incident volumes and demand pressure, whereas locations on the negative side are characterized by lower demand and comparatively faster response times.
#
# The second principal component (PC2) is primarily associated with incident severity. Alarm-level variables, particularly the proportion of high-alarm incidents, load strongly on PC2, indicating that vertical separation in the PCA plot reflects differences in incident severity rather than workload alone. Locations with higher PC2 values tend to handle more severe incidents, even if their call volumes are not the highest.
#
# Cluster interpretation suggests the presence of four operational profiles across Toronto:
#
# * **High-demand locations:** Areas with elevated call volumes and demand intensity, forming a distinct cluster along the positive PC1 axis. These represent busy operational zones requiring sustained resource allocation.
# * **Moderate-demand baseline locations:** A central cluster representing typical operational environments with balanced demand and response performance.
# * **Low-demand, fast-response locations:** Areas characterized by lower incident volume and relatively faster response times, indicating stable operational conditions.
# * **Severity-influenced locations:** Locations with relatively higher alarm-level characteristics, separated along the PC2 axis, suggesting environments where incident severity may contribute to response-time variation.
#
# Overall, the clustering analysis highlights meaningful spatial heterogeneity in Toronto’s emergency response system. While many locations operate under similar conditions, a subset of areas exhibits distinct demand and severity profiles that may contribute to elevated response-time risk. These findings support subsequent predictive and survival modeling by identifying underlying structural differences in operational environments across the city.
#

# %% [markdown]
# #### 4.6.2 NYC Spatial Analysis (Alternative to Clustering)
#
# The NYC dataset contains only five unique location areas.  
# Given this limited spatial granularity, unsupervised clustering would not produce stable or interpretable groupings. Instead, spatial variation in NYC is examined using descriptive comparison methods that directly evaluate response-time performance and demand patterns across locations.
#
# The following analyses are used in place of clustering:
#
# 1. Location-level descriptive statistics  
# 2. Bar-chart comparison of response-time metrics  
# 3. Tail-delay comparison across locations  
# 4. Demand vs response relationship  
# 5. Cross-city comparison with Toronto clusters
#

# %% [markdown]
# ##### 4.6.2.1 Location Summary Table

# %%
nyc_loc_summary = (
    nyc_df.groupBy("location_area")
    .agg(
        F.count("*").alias("call_volume"),
        F.round(F.mean("response_minutes"),4).alias("mean_response"),
        F.round(F.expr("percentile(response_minutes, 0.9)"),4).alias("p90_response"),
        F.mean("calls_past_30min").alias("mean_calls_30"),
        F.mean("calls_past_60min").alias("mean_calls_60"),
        F.mean("unified_alarm_level").alias("mean_alarm")
    )
    .orderBy("p90_response", ascending=False)
)

display(nyc_loc_summary)

# %%
pdf = nyc_loc_summary.toPandas()

# reshape for grouped bars
plot_df = pdf.melt(
    id_vars="location_area",
    value_vars=["mean_response", "p90_response"],
    var_name="metric",
    value_name="minutes"
)

plt.figure(figsize=(9,5))
ax = sns.barplot(
    data=plot_df,
    x="location_area",
    y="minutes",
    hue="metric"
)

# Add headroom
y_max = plot_df["minutes"].max()
ax.set_ylim(0, y_max * 1.15)

# Grid
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Smaller tick labels
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)

plt.xticks(rotation=45)

plt.title("NYC Mean vs P90 Response Time by Location", fontsize=13, fontweight="bold")
plt.xlabel("Location Area", fontsize=10)
plt.ylabel("Response Time (minutes)", fontsize=10)

# Data labels for grouped bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height:.2f}",
        (p.get_x() + p.get_width()/2., height),
        ha="center",
        va="bottom",
        fontsize=7,
        xytext=(0, 3),
        textcoords="offset points"
    )

plt.legend(title="Metric", fontsize=8, title_fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ##### 4.6.2.2 Mean and P90 Response Time Bar Charts

# %% [markdown]
# ##### 4.6.2.3 Call Volume Bar Chart

# %%
pdf = nyc_loc_summary.toPandas()

plt.figure(figsize=(8,5))
ax = sns.barplot(
    data=pdf,
    x="location_area",
    y="call_volume"
)

# Grid
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Smaller ticks
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)

plt.xticks(rotation=45)

plt.title("NYC Incident Volume by Location", fontsize=13, fontweight="bold")
plt.xlabel("Location Area", fontsize=10)
plt.ylabel("Incident Count", fontsize=10)

# Headroom
y_max = pdf["call_volume"].max()
ax.set_ylim(0, y_max * 1.15)

# Data labels
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width()/2., p.get_height()),
        ha="center",
        va="bottom",
        fontsize=8,
        xytext=(0, 3),
        textcoords="offset points"
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ##### Summary of NYC Spatial Patterns and Demand–Performance Comparison
#
# Spatial analysis of NYC incident data reveals clear variation in response-time performance and incident volume across boroughs. Brooklyn and Manhattan account for the highest incident volumes, with approximately 421,772 and 374,377 incidents respectively, followed by the Bronx and Queens. Richmond/Staten Island has substantially lower call volume (68,814), reflecting its smaller population and geographic scale.
#
# Despite these differences in workload, response-time patterns do not strictly follow call volume. The Bronx exhibits the highest mean response time (6.46 minutes) and the highest P90 response time (9.45 minutes), indicating comparatively greater tail-delay risk. Manhattan and Queens show moderately high mean response times and elevated P90 values (8.97 and 8.57 minutes respectively), suggesting that peak-period delays occur even in high-capacity urban areas. In contrast, Brooklyn records the highest incident volume but one of the lowest mean response times (5.30 minutes) and relatively lower tail-delay risk (P90 = 7.80 minutes). Richmond/Staten Island, while lower in demand, shows slightly faster average response performance overall.
#
# Short-term demand indicators (calls in the past 30 and 60 minutes) are relatively consistent across boroughs, suggesting that baseline demand intensity is broadly similar despite differences in total volume. Alarm-level averages are also highly stable across locations, indicating that incident severity distribution does not vary substantially by borough.
#
# Overall, these findings suggest that higher call volume does not always correspond to slower response performance. Some high-demand boroughs maintain relatively efficient response times, while others experience elevated tail delays. This highlights the importance of examining both average and tail-response metrics when evaluating operational risk. The NYC spatial analysis complements the Toronto clustering results by providing a descriptive comparison of demand and response characteristics across boroughs without applying clustering to a dataset with limited spatial granularity.
#

# %% [markdown]
# ### 4.7 Composite Tail-Risk Prioritization Score
#
# To identify areas with the greatest exposure to extreme response delays, a composite risk score is defined as:
#
# **Risk Score = Incident Volume × P90 Response Time**
#
# where incident volume represents total demand in an area and the P90 response time captures tail delay risk. This heuristic metric prioritizes areas where a large number of incidents are affected by longer extreme response times, complementing analyses based on mean performance alone. The risk score is used for **within-city prioritization** and is interpreted as an operational indicator rather than a formal probabilistic measure of risk.
#

# %% [markdown]
# #### 4.7.1 Toronto Risk Score Table

# %%
toronto_risk = (
    toronto_area_stats
    .withColumn("risk_score", F.round(F.col("n_total") * F.col("p90_response"), 2))
    .orderBy(F.desc("risk_score"))
)
display(toronto_risk.select("location_area","n_total","p90_response","risk_score").limit(15))

# %% [markdown]
# #### 4.7.2 NYC Risk Score Table

# %%
nyc_risk = (
    nyc_area_stats
    .withColumn("risk_score", F.round(F.col("n_total") * F.col("p90_response"), 2))
    .orderBy(F.desc("risk_score"))
)
display(nyc_risk.select("location_area","n_total","p90_response","risk_score").limit(15))

# %% [markdown]
# #### 4.7.3 Summary of Composite Tail-Risk Prioritization
# The composite risk score highlights areas where **high incident volume coincides with elevated tail response times**, identifying locations with the greatest exposure to extreme delays. In Toronto, the highest-risk station areas combine moderate-to-high demand with relatively large P90 response times, indicating that tail delays, not just volume, drive prioritization. In NYC, **Manhattan, Brooklyn, and the Bronx** dominate the risk rankings due to their very high incident volumes coupled with elevated P90 response times, while Staten Island exhibits substantially lower overall exposure.
#
# Overall, the rankings reinforce that operational risk is shaped by the **interaction of demand and tail performance**, rather than by response speed or volume alone, supporting the use of this composite metric for within-city prioritization.

# %% [markdown]
# ## 5. Incident Characteristic
# This section examines how response performance varies by incident characteristics (incident type and alarm level) and highlights rare categories with elevated tail risk.
# - Incident type vs response time
# - Alarm level vs response time
# - Rare but high-risk categories

# %%
# ---- Set Column Names ----
INCIDENT_COL = "incident_category"     
ALARM_COL = "unified_alarm_level"      # you already have this
P = 0.90                               # percentile for tail metric (P90)

# ---- Utility to attach city label ----
def add_city(df, city_name):
    return df.withColumn("city", F.lit(city_name))


# %% [markdown]
# ### 5.1 Incident Type vs Response Time (Mean + Tail)
#
# We compare typical performance (mean) and tail delays (P90) across incident types using completed incidents only.

# %% [markdown]
# #### 5.1.1 Toronto Incident Characteristics

# %%
# Toronto: Incident type vs response time
toronto_incident_stats = add_city(
    response_by_category(toronto_df, INCIDENT_COL, p=P), "Toronto"
)

display(toronto_incident_stats)

# %% [markdown]
# #### 5.1.2 NYC Incident Characteristics

# %%
# NYC: Incident type vs response time
nyc_incident_stats = add_city(
    response_by_category(nyc_df, INCIDENT_COL, p=P), "NYC"
)

display(nyc_incident_stats)

# %% [markdown]
# #### 5.1.3 Summary fo Incident Type vs Response Time Characteristics
# Across both Toronto and NYC,** medical incidents account for the largest share of completed responses** and exhibit moderate mean response times, though tail delays (P90) remain elevated.** Fire-related incidents**, particularly structural fires, tend to receive **faster typical responses** and lower P90 values, reflecting prioritization of high-severity events. In contrast, **Other / Assistance, Rescue / Entrapment, and Hazardous / Utility** incidents show **higher tail response times** in both cities, indicating greater exposure to extreme delays despite reasonable average performance.
#

# %% [markdown]
# ### 5.2 Alarm Level vs Response Time
#
# Alarm level is treated here as an incident-severity characteristic. Results for levels above 1 are interpreted descriptively due to small sample sizes.

# %% [markdown]
# #### 5.2.1 Toronto Alarm Level vs Response Time Performance

# %%
toronto_alarm_stats = add_city(
    response_by_category(toronto_df, ALARM_COL, p=P), "Toronto"
)

display(toronto_alarm_stats)

# %% [markdown]
# #### 5.2.2 NYC Alarm Level vs Response Time Performance

# %%
nyc_alarm_stats = add_city(
    response_by_category(nyc_df, ALARM_COL, p=P), "NYC"
)

display(nyc_alarm_stats)

# %% [markdown]
# #### 5.2.3 Summary of Alarm Level vs Response Time Performance
# In both Toronto and NYC, higher alarm levels are associated with shorter mean and P90 response times, indicating that escalated incidents receive faster operational prioritization. Level-1 alarms dominate overall incident volume and therefore largely determine system-wide response performance. Results for alarm levels above 1 are based on small sample sizes and are interpreted descriptively rather than as statistically robust differences.

# %% [markdown]
# ### 5.3 Rare but High-Risk Categories (High P90)
#
# To highlight “rare but operationally risky” incident types, we filter to low-volume categories and rank by P90 response time.
# - **Rare** is defined as categories with `n_completed` between `MIN_N` and `RARE_MAX_N`
# - This avoids over-interpreting extremely tiny groups while still surfacing tail-risk patterns.
#

# %%
MIN_N = 100          # ignore extremely tiny categories (unstable)
RARE_MAX_N = 1000    # "rare" threshold; tune if needed (e.g., 500 / 2000)

toronto_rare_highrisk = (
    toronto_incident_stats
    .filter((F.col("n_completed") >= MIN_N) & (F.col("n_completed") <= RARE_MAX_N))
    .orderBy(F.desc(f"p{int(P*100)}_response"), F.desc("n_completed"))
)

nyc_rare_highrisk = (
    nyc_incident_stats
    .filter((F.col("n_completed") >= MIN_N) & (F.col("n_completed") <= RARE_MAX_N))
    .orderBy(F.desc(f"p{int(P*100)}_response"), F.desc("n_completed"))
)

display(toronto_rare_highrisk.limit(20))
display(nyc_rare_highrisk.limit(20))


# %% [markdown]
# **Summary of Rare but High-Risk Incident Categories**
#
# All incident categories in Toronto exhibit high volumes, with no category occurring infrequently enough to be considered rare under reasonable stability thresholds. Consequently, elevated tail response times are observed within common incident types rather than being driven by low-frequency categories, indicating that response-time risk is systemic rather than category-specific.
#

# %% [markdown]
# ## 6. Cross-City Comparability Check
#
# Before developing predictive and prescriptive models, we verify that response-time definitions, units, and feature engineering logic are aligned across Toronto and NYC. This section ensures that observed differences reflect operational realities rather than data construction artifacts.
#
# Before modeling:
# - Are response-time definitions aligned?
# - Same units? (seconds vs minutes)
# - Similar feature engineering logic?
#
# Create:
# - Percentile comparison (Toronto vs NYC)

# %% [markdown]
# ### 6.1 Response-Time Definition and Unit Consistency
#
# Response time is defined consistently in both datasets as the elapsed time between alarm receipt and first unit arrival. All response times are expressed in **minutes**, and incidents with missing response times are treated as censored observations and excluded from distributional comparisons.

# %%
# Sanity check: response time units
display(
    toronto_df.select("response_minutes")
              .summary("min", "mean", "max")
)

display(
    nyc_df.select("response_minutes")
          .summary("min", "mean", "max")
)


# %% [markdown]
# **Summary**
#
# Response times in both Toronto and NYC are expressed in minutes and exhibit comparable central tendencies, with mean response times of approximately 5.3 minutes for Toronto and 5.9 minutes for NYC. Minimum values are near zero in both datasets, consistent with immediate arrivals or rounding effects. Toronto exhibits a substantially larger maximum response time than NYC, indicating heavier tail behavior and reinforcing the importance of tail-focused metrics in subsequent analysis.
#

# %% [markdown]
# ### 6.2 Feature Engineering Alignment
#
# Both datasets apply consistent feature engineering logic, including:
# - Timestamp normalization
# - Response time calculation in minutes
# - Treatment of missing response times
# - Harmonized categorical mappings (incident type, call source, alarm level)
#
# This alignment ensures that downstream comparisons and models operate on equivalent representations.
# (No code needed here — this is methodological assurance.)

# %% [markdown]
# ### 6.3 Percentile-Based Cross-City Comparison
#
# Percentile comparisons provide a scale-invariant way to compare response-time behavior across cities, particularly in the distribution tail.

# %%
def percentile_summary(df, city):
    return (
        df.filter(F.col("response_minutes").isNotNull())
          .selectExpr(
              "percentile_approx(response_minutes, 0.50) as p50",
              "percentile_approx(response_minutes, 0.75) as p75",
              "percentile_approx(response_minutes, 0.90) as p90",
              "percentile_approx(response_minutes, 0.95) as p95"
          )
          .withColumn("city", F.lit(city))
    )

percentiles = (
    percentile_summary(toronto_df, "Toronto")
    .unionByName(percentile_summary(nyc_df, "NYC"))
)

display(percentiles)

# %% [markdown]
# **Summary of Percentile-Based Cross-City Comparison **
#
# Toronto and NYC exhibit similar median response times (P50), indicating broadly comparable typical performance. However, NYC shows consistently higher upper-tail percentiles (P75–P95), with notably larger differences at P90 and P95. This indicates heavier tail behavior in NYC, where extreme response delays occur more frequently than in Toronto, reinforcing the importance of tail-focused metrics in cross-city comparison and subsequent modeling.
#

# %% [markdown]
# ## 7. Correlation Analysis
#
# This chapter explores linear relationships among numeric incident-level features and examines how categorical incident characteristics relate to response time. The objective is to gain exploratory insight into feature relationships and potential redundancy prior to predictive modeling.
#

# %% [markdown]
# ### 7.1 Numeric Feature Correlation Analysis
#
# Correlation analysis is restricted to numeric, non-binary features with continuous or count-based meaning. Identifier fields, binary indicators, and categorical variables are excluded, as Pearson correlation is not appropriate for such data types.

# %%
# Numeric, non-binary features selected for correlation analysis
NUMERIC_COLS = [
    "response_minutes",
    "hour",
    "day_of_week",
    "month",
    "calls_past_30min",
    "calls_past_60min"
]

# %% [markdown]
# ##### 7.1.1 Toronto Data

# %%

# Prepare Toronto data
toronto_corr_pd = (
    toronto_df
    .select(NUMERIC_COLS)
    .dropna()            # drop rows with missing values
    .toPandas()
)

# Compute Pearson correlation
toronto_corr = toronto_corr_pd.corr(method="pearson")

toronto_corr


# %%
plt.figure(figsize=(8, 6))

ax = sns.heatmap(
    toronto_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

# Bold title, larger than labels
ax.set_title(
    "Toronto: Correlation Matrix of Numeric Features",
    fontsize=14,
    fontweight="bold"
)

# Smaller tick labels
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# #### 7.1.2 NYC Data

# %%
# Prepare NYC data
nyc_corr_pd = (
    nyc_df
    .select(NUMERIC_COLS)
    .dropna()
    .toPandas()
)

# Compute Pearson correlation
nyc_corr = nyc_corr_pd.corr(method="pearson")

nyc_corr


# %%
plt.figure(figsize=(8, 6))

ax = sns.heatmap(
    nyc_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

ax.set_title(
    "NYC: Correlation Matrix of Numeric Features",
    fontsize=14,
    fontweight="bold"
)

ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.show()


# %% [markdown]
# ### 7.2 Categorical Feature Analysis
#
# Pearson correlation is not appropriate for categorical variables. Instead, categorical incident characteristics are analyzed using distributional comparisons to assess how response time varies across categories. This approach allows identification of systematic differences in typical and tail response behavior associated with incident type, alarm level, and call source.

# %% [markdown]
# #### 7.2.1 Incident Category vs Response Time

# %% [markdown]
# ##### Toronto

# %%
toronto_cat_pd = (
    toronto_df
    .select("incident_category", "response_minutes")
    .dropna()
    .toPandas()
)

cat_pd = toronto_cat_pd
plt.figure(figsize=(10, 8))  # increased height

ax = sns.boxplot(
    data=cat_pd,
    x="incident_category",
    y="response_minutes",
    showfliers=True   # keep outliers, but no overlay
)

ax.set_ylim(0, 60)

ax.set_title(
    "Toronto Response Time by Incident Category",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Incident Category", fontsize=10)
ax.set_ylabel("Response Time (minutes)", fontsize=10)
ax.tick_params(axis="x", labelsize=9, rotation=45)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.show()



# %% [markdown]
# Response time distributions vary modestly across incident categories, with most medians clustered between four and six minutes. Categories such as Hazardous / Utility and Rescue / Entrapment exhibit slightly higher variability and longer tails, though extreme delays are present across all categories. This suggests that while incident type influences response characteristics, tail delays are largely systemic rather than driven by a single category.

# %% [markdown]
# ##### NYC

# %%
nyc_cat_pd = (
    nyc_df
    .select("incident_category", "response_minutes")
    .dropna()
    .toPandas()
)

cat_pd = nyc_cat_pd
plt.figure(figsize=(10, 8))  # increased height

ax = sns.boxplot(
    data=cat_pd,
    x="incident_category",
    y="response_minutes",
    showfliers=True   # keep outliers, but no overlay
)

ax.set_ylim(0, 20)

ax.set_title(
    "NYC Response Time by Incident Category",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Incident Category", fontsize=10)
ax.set_ylabel("Response Time (minutes)", fontsize=10)
ax.tick_params(axis="x", labelsize=9, rotation=45)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# In NYC, response time distributions vary across incident categories, with Fire, Structural incidents exhibiting the lowest typical response times and categories such as Rescue / Entrapment and Hazardous / Utility showing higher medians and wider spreads. Despite these differences, extreme delays occur across all incident types, indicating that tail risk is not confined to specific categories. Overall patterns are broadly consistent with Toronto, supporting cross-city comparability.
#

# %% [markdown]
# #### 7.2.2 Alarm Level vs Response Time

# %% [markdown]
# ##### Toronto

# %%
toronto_alarm_pd = (
    toronto_df
    .select("unified_alarm_level", "response_minutes")
    .dropna()
    .toPandas()
)

plt.figure(figsize=(10, 8))
ax = sns.boxplot(
    data=toronto_alarm_pd,
    x="unified_alarm_level",
    y="response_minutes"
)
ax.set_ylim(0, 80)
ax.set_title(
    "Toronto Response Time by Alarm Level",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Alarm Level", fontsize=10)
ax.set_ylabel("Response Time (minutes)", fontsize=10)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.show()


# %% [markdown]
# ##### NYC

# %%
nyc_alarm_pd = (
    nyc_df
    .select("unified_alarm_level", "response_minutes")
    .dropna()
    .toPandas()
)

plt.figure(figsize=(10, 8))
ax = sns.boxplot(
    data=nyc_alarm_pd,
    x="unified_alarm_level",
    y="response_minutes"
)
ax.set_ylim(0,20)
ax.set_title(
    "NYC Response Time by Alarm Level",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Alarm Level", fontsize=10)
ax.set_ylabel("Response Time (minutes)", fontsize=10)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ##### Insight
# Alarm level does not exhibit a monotonic relationship with response time in either city. In both Toronto and NYC, Alarm Level 1 incidents show the greatest variability and the longest tail delays, reflecting their high volume and diverse operational contexts. Higher alarm levels are comparatively rare and display tighter response distributions, suggesting that tail risk is driven more by system-wide demand than by incident escalation level.

# %% [markdown]
# #### 7.2.3 Call Source vs Response Time

# %% [markdown]
# ##### Toronto

# %%
toronto_call_pd = (
    toronto_df
    .select("unified_call_source", "response_minutes")
    .dropna()
    .toPandas()
)

plt.figure(figsize=(10, 8))
ax = sns.boxplot(
    data=toronto_call_pd,
    x="unified_call_source",
    y="response_minutes"
)
ax.set_ylim(0, 80)
ax.set_title(
    "Toronto Response Time by Call Source",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Call Source", fontsize=10)
ax.set_ylabel("Response Time (minutes)", fontsize=10)
ax.tick_params(axis="x", labelsize=9, rotation=45)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.show()


# %% [markdown]
# ##### NYC

# %%
nyc_call_pd = (
    nyc_df
    .select("unified_call_source", "response_minutes")
    .dropna()
    .toPandas()
)

plt.figure(figsize=(10, 8))
ax = sns.boxplot(
    data=nyc_call_pd,
    x="unified_call_source",
    y="response_minutes"
)
ax.set_ylim(0, 20)
ax.set_title(
    "NYC Response Time by Call Source",
    fontsize=14,
    fontweight="bold"
)
ax.set_xlabel("Call Source", fontsize=10)
ax.set_ylabel("Response Time (minutes)", fontsize=10)
ax.tick_params(axis="x", labelsize=9, rotation=45)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ##### Insight
# Response time distributions vary by call source in both Toronto and NYC, with Public and EMS / Medical calls exhibiting greater variability and longer tail delays than Alarm System calls. Alarm-originated incidents show more consistent response times, likely reflecting automated detection and standardized dispatch workflows. Overall patterns are consistent across cities, indicating that call source influences response-time variability rather than typical response speed.
#

# %% [markdown]
# ### Summary
# Overall, correlation and categorical analyses reveal expected relationships and variability patterns without indicating excessive redundancy or anomalous behavior, supporting the selected feature set for downstream predictive modeling.

# %% [markdown]
# ## 9. EDA-Driven Insights

# %% [markdown]
# The exploratory data analysis reveals several consistent operational patterns across Toronto and NYC that directly inform subsequent modeling and interpretation.
#
# First, response times exhibit strong right-skewness with long tails across cities, incident types, and locations. Average response times mask meaningful operational risk, while percentile-based metrics (P90 and above) more effectively capture worst-case delays. This supports the use of tail-aware evaluation metrics and motivates modeling approaches that prioritize extreme delays rather than mean performance alone.
#
# Second, incident volume is not strongly correlated with faster response. High-volume station areas and boroughs frequently exhibit elevated P90 response times, indicating that workload concentration does not translate into proportional operational efficiency. This finding motivates the inclusion of demand intensity features (e.g., calls in the past 30–60 minutes) and supports prescriptive and risk-weighted analyses rather than volume-only prioritization.
#
# Third, temporal features (hour, day of week, month) show weak linear correlation with response time, while recent call volume features demonstrate stronger relationships. This suggests that short-term demand pressure is more informative than calendar effects for predicting response delays, guiding feature selection toward operational load indicators.
#
# Finally, categorical factors—including incident category, alarm level, and call source—exhibit distinct response-time distributions with consistent tail behavior. Elevated tail risk appears across common incident types rather than being driven by rare categories, indicating that response-time risk is systemic rather than isolated to niche scenarios. These categorical variables are therefore retained for downstream modeling via appropriate encoding strategies.
#
# Overall, insights derived from this EDA directly inform feature engineering decisions, model selection strategies focused on tail-risk prediction, and the interpretation of operational risk in emergency response systems. Key findings and visualizations from this analysis are carried forward into subsequent modeling, comparative evaluation, and dashboard reporting.
