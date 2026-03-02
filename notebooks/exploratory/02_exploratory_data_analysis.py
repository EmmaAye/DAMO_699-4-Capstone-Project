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

# %%
print("Toronto years:")
display(toronto_df.select("year").distinct())

# %%
print("NYC years:")
display(nyc_df.select("year").distinct())


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
    m["missing_pct"] = (m["missing_count"] / total * 100).round(2)
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
# - **New York City:** 372,544 missing values (29.03%)
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
# *(Response time and delay-based targets)*
#
# This study uses two complementary target representations:
#
# - **Continuous target:** `response_minutes` (time from alarm to first-unit arrival, in minutes)
# - **Binary delay target:** `delay_indicator` (1 if `response_minutes` > **8 minutes**, else 0)
#
# The continuous target supports distributional, tail-risk, and survival-based analysis, while the binary delay target supports classification-based delay-risk modeling.
#
# **Scope and data handling**
# - **Distributional plots and summary statistics** (histogram/KDE, mean/median/P90/P95, skewness, outliers) are computed using **completed incidents only**, where `response_minutes IS NOT NULL`.
# - **SLA breach rates** (e.g., % of incidents exceeding 5 or 8 minutes) are also reported using **completed incidents only** to reflect observed response-time performance.
# - **Delay indicator exploration** (count and % delayed) is computed on rows where `delay_indicator` is defined; if `response_minutes` is missing, `delay_indicator` is treated as **NULL** and excluded from delay-rate calculation.
# - **Censored incidents** (missing `response_minutes`) are retained in the model-ready tables and are handled explicitly in **survival analysis** using `event_indicator`.
#
# **Analyses included in this section**
# - Distribution of `response_minutes` (Histogram / KDE)
# - Summary statistics (Mean, Median, P90, P95)
# - Skewness and outlier diagnostics (IQR-based; outliers are retained)
# - Service-level threshold breach rates (e.g., > 5 minutes, > 8 minutes)
# - Delay indicator prevalence (delayed count and percentage by city)
#

# %% [markdown]
# ### 2.1 Define Completed Incidents Subsets
# Distributional and summary-statistic analyses of response time are conducted using **completed incidents only**, where `response_minutes IS NOT NULL`.  
# Incidents without an observed response time are retained and treated as censored observations for survival analysis.

# %%
toronto_complete = toronto_df.filter(F.col("response_minutes").isNotNull())
nyc_complete     = nyc_df.filter(F.col("response_minutes").isNotNull())

# %%
print("Toronto completed:", toronto_complete.count(), "/", toronto_df.count())
print("NYC completed:", nyc_complete.count(), "/", nyc_df.count())

# %% [markdown]
# ### 2.2 Response Time Analysis
# This section examines the distribution and variability of the continuous response-time target.

# %% [markdown]
# #### 2.2.1 Response Time Distribution (Completed Incidents Only)
#
# Response-time distributions are visualized using histograms and kernel density estimates (KDE) based on completed incidents only.  
# These plots characterize the overall shape of response-time behavior and highlight the presence of tail delays.

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
# Both Toronto and NYC show right-skewed response time distributions, meaning most incidents are handled quickly with a smaller number of longer delays.
#
# - **Toronto:** Most responses fall around 4–8 minutes, but the longer right tail indicates more extreme delay cases and higher variability.
# - **NYC:** Response times are tightly concentrated around 3–7 minutes, with fewer extreme delays and more consistent performance.
#
# Overall, central response times are similar, but Toronto shows greater tail-delay risk, highlighting the importance of examining high-percentile metrics (e.g., P90) in addition to averages.

# %% [markdown]
# #### 2.2.2 Summary Statistics (Mean, Median, P90, P95)
#
# Summary statistics are computed for the continuous response-time variable (`response_minutes`) using **completed incidents only** (i.e., where `response_minutes IS NOT NULL`). These statistics describe central tendency and tail behavior in response-time performance across Toronto and New York City.
#
# Mean, median, P90, and P95 statistics are computed for completed incidents.  
# While mean and median describe typical response performance, high-percentile metrics capture extreme delays and operational risk in the upper tail of the distribution.

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
# Toronto and NYC have similar central response times, with medians around 5–5.5 minutes.  
# Toronto performs slightly better overall, showing lower mean and tail metrics (P90 and P95), indicating fewer extreme delays.  
#
# NYC has a slightly higher average and noticeably higher P90/P95 values, suggesting greater variability and more frequent longer response times.  
# Overall, while typical response performance is comparable, NYC exhibits higher tail-delay risk than Toronto.

# %% [markdown]
# #### 2.2.3 Skewness & Outliers
#
# Skewness and outlier analysis are conducted on the continuous response-time variable (`response_minutes`) using completed incidents only. These diagnostics assess the degree of asymmetry and the presence of extreme delays in response-time distributions across Toronto and New York City.
#
# Both cities exhibit **positive skewness**, indicating that while most incidents are handled within a typical response window, a smaller subset experiences substantially longer delays. This right-skewed structure is characteristic of emergency response systems, where extreme delays are infrequent but operationally significant.
#
# Outliers are identified using the interquartile range (IQR) method as a diagnostic tool rather than as a basis for data removal. Incidents exceeding the upper outlier threshold represent genuine extreme delays and are retained for analysis, as they reflect meaningful operational conditions such as demand surges, congestion, or resource constraints.
#
# Because the delay indicator is derived from response time using a fixed threshold, skewness and outlier diagnostics are performed on the continuous response-time variable only. The binary delay indicator is analyzed separately through prevalence and threshold-breach metrics.
#
# Overall, the presence of strong right-skewness and a non-trivial share of extreme delays reinforces the importance of tail-sensitive metrics and motivates modeling approaches that explicitly account for delay risk rather than relying solely on average response times.
#

# %% [markdown]
# ##### 2.2.3.1 Skewness

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
# Both cities show positive skewness, meaning most response times are short with a tail of longer delays.  
# Toronto has much higher skewness (4.78) than NYC (1.28), indicating a heavier right tail and more extreme delay outliers.  
#
# This suggests Toronto’s typical response times are similar to NYC’s, but it experiences more occasional long delays, increasing tail-risk variability.

# %% [markdown]
# ##### 2.2.3.2 Outlier Inspection (IQR-based, diagnostic only)

# %%
# Function to compute outlier bounds

def outlier_bounds(df):
    """
    Computes IQR bounds for response_minutes
    using completed incidents only.
    """
    stats = df.selectExpr(
        "percentile_approx(response_minutes, 0.25) as q1",
        "percentile_approx(response_minutes, 0.75) as q3"
    ).first()

    q1 = stats["q1"]
    q3 = stats["q3"]
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return q1, q3, lower, upper

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
# Both cities have a similar proportion of outliers (about 4–4.3%), indicating that extreme response times are present but not dominant in the data.
#
# Toronto has fewer total incidents but a higher share of upper outliers relative to its dataset size, with most outliers occurring on the high-delay side.  
# NYC has a much larger dataset and a higher number of upper outliers overall, reflecting more frequent long-delay cases in absolute terms.
#
# In both cities, the majority of outliers are upper-bound (long response times), reinforcing that operational risk is driven mainly by occasional extended delays rather than unusually fast responses.
#
# **Note: Do not remove outliers — long delays are operationally meaningful**.

# %% [markdown]
# #### 2.2.4 Service Level Agreement (SLA) Breach Analysis
#
# SLA breach analysis evaluates the share of **completed incidents** whose response times exceed selected time thresholds. These thresholds represent practical performance benchmarks rather than strict policy guarantees and are used to assess service-level reliability.
#
# This analysis is conducted using the continuous response-time variable (`response_minutes`) for completed incidents only (i.e., where `response_minutes IS NOT NULL`). By measuring the proportion of incidents exceeding specified thresholds (e.g., 5 and 8 minutes), SLA breach rates provide a threshold-based view of performance that complements distributional and percentile metrics.
#
# While summary statistics describe typical response performance, breach rates highlight delayed responses that are masked by averages and reveal operational risk under high-demand or constrained conditions. Because the binary delay indicator is derived directly from the 8-minute threshold, SLA breach analysis provides the continuous-response context needed to interpret delay prevalence and supports subsequent predictive modeling of delay risk.
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
# NYC shows a higher proportion of incidents exceeding key response thresholds compared to Toronto.  
# About 61% of NYC responses exceed 5 minutes and 14% exceed 8 minutes, while Toronto records lower breach rates at roughly 52% over 5 minutes and 8% over 8 minutes.
#
# This indicates that although typical response times are similar between the two cities, NYC experiences more frequent SLA breaches and greater tail-delay pressure, suggesting higher operational risk at stricter service thresholds.

# %% [markdown]
# #### 2.2.5 Threshold Risk Analysis (Service-Level Exceedance)
#
# To support operational risk assessment, we compute the probability that response times exceed key service thresholds.  
# These thresholds represent service-level expectations and help quantify delay risk.
#
# We evaluate exceedance probabilities at:
# - **8 minutes** — official service-level delay threshold used in modeling
# - **10 minutes** — severe-delay threshold to assess tail-risk conditions
#
# Outputs:
# - Exceedance probability by city
# - Risk comparison tables
# - Visualization-ready summaries
#

# %% [markdown]
# Threshold Exceedance Analysis

# %%
thresholds = [8, 10]

def compute_threshold_risk(df, city_name):
    total = df.filter(F.col("response_minutes").isNotNull()).count()
    rows = []

    for t in thresholds:
        exceed = df.filter(F.col("response_minutes") > t).count()
        prob = round(exceed / total * 100, 2)
        rows.append((city_name, t, exceed, total, prob))

    return spark.createDataFrame(
        rows,
        ["city","threshold_min","n_exceed","n_total","exceed_prob_%"]
    )

toronto_risk = compute_threshold_risk(toronto_df, "Toronto")
nyc_risk = compute_threshold_risk(nyc_df, "NYC")

threshold_risk_df = toronto_risk.unionByName(nyc_risk)
display(threshold_risk_df)


# %%
risk_pdf = threshold_risk_df.toPandas()

plt.figure(figsize=(7,5))
sns.barplot(data=risk_pdf, x="threshold_min", y="exceed_prob_%", hue="city")

plt.title("Probability of Exceeding Response-Time Thresholds",fontsize=13, fontweight="bold")
plt.xlabel("Threshold (minutes)", fontsize=11)
plt.ylabel("Exceedance Probability (%)", fontsize=11)

for p in plt.gca().patches:
    h = p.get_height()
    plt.gca().annotate(f"{h:.1f}%",
        (p.get_x()+p.get_width()/2, h),
        ha="center", va="bottom", fontsize=9)

plt.legend(title="City", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ##### Threshold Risk Summary
#
# The probability of exceeding service-level thresholds is consistently higher in NYC than in Toronto.  
# At the 8-minute benchmark, about **14.1%** of NYC incidents exceed the target compared with **8.0%** in Toronto, indicating a greater likelihood of moderate delays in NYC.
#
# At the more severe 10-minute threshold, exceedance rates decline in both cities but remain higher in NYC (**5.6%**) than in Toronto (**2.6%**).  
# This suggests that while most incidents meet expected response times, NYC faces a higher level of tail-delay risk.
#
# Overall, differences between cities are most visible in the upper tail of the distribution, reinforcing the importance of threshold- and percentile-based metrics rather than relying solely on average response times.
#

# %% [markdown]
# ### 2.3 Delay Indicator Analysis
#
# To support predictive modeling of response-time delays, a binary delay indicator is used.  
# An incident is classified as **delayed** if the response time exceeds **8 minutes**.
#
# This section summarizes:
# - number of delayed incidents  
# - percentage of delayed incidents  
# - comparison across cities  
#
# This provides an overview of class balance and establishes the modeling target.

# %% [markdown]
# #### 2.3.1 Delay Prevalence
#
# The number and percentage of delayed incidents are computed for each city.  
# This provides an overview of class balance and establishes the modeling target distribution.

# %% [markdown]
# Count and Percentage of Delay Incidents

# %%
# Add city labels for comparison
toronto_delay = toronto_df.withColumn("city", F.lit("Toronto"))
nyc_delay     = nyc_df.withColumn("city", F.lit("NYC"))

combined_delay = toronto_delay.unionByName(nyc_delay)

delay_summary = (
    combined_delay
    .groupBy("city")
    .agg(
        F.count("*").alias("total_incidents"),
        F.sum("delay_indicator").alias("delayed_incidents"),
        F.round(F.mean("delay_indicator") * 100, 2).alias("delay_percent")
    )
    .orderBy("city")
)

display(delay_summary)

# %% [markdown]
# Overall(Bot cities combined)

# %%
combined_delay.select(
    F.count("*").alias("total_incidents"),
    F.sum("delay_indicator").alias("delayed_incidents"),
    F.round(F.mean("delay_indicator") * 100, 2).alias("delay_percent")
).show()

# %%
delay_pd = delay_summary.toPandas()

plt.figure(figsize=(6,4))
ax = sns.barplot(
    data=delay_pd,
    x="city",
    y="delay_percent"
)

# Grid
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Smaller ticks
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)

plt.xticks(rotation=45)

plt.title("Percentage of Incidents Exceeding 8-Minute Threshold", fontsize=13, fontweight="bold")
plt.xlabel("City", fontsize=11)
plt.ylabel("Delay %", fontsize=11)

# Headroom
y_max = delay_pd["delay_percent"].max()
ax.set_ylim(0, y_max * 1.15)

# for i, v in enumerate(delay_pd["delay_percent"]):
#     ax.text(i, v, f"{v:.2f}%")
# Data labels
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height:.2f}%",                     # show percent
        (p.get_x() + p.get_width()/2., height),
        ha="center",
        va="bottom",
        fontsize=9,
        xytext=(0, 3),
        textcoords="offset points"
    )


plt.tight_layout()
plt.show()

# %% [markdown]
# #### 2.3.2 Delay Indicator Analysis (8-Minute Threshold)
#
# NYC has a higher proportion of incidents exceeding the 8-minute threshold compared to Toronto.  
# About 14.13% of NYC incidents are delayed beyond 8 minutes (128,689 cases), while Toronto records 8.03% (28,046 cases).
#
# Although NYC has a larger total volume of incidents, the higher delay percentage indicates greater pressure on response performance and more frequent threshold breaches.  
# Overall, Toronto demonstrates stronger adherence to the 8-minute service target, while NYC shows higher operational delay risk at this threshold.

# %% [markdown]
# ### 2.4 Censoring Awareness (For Survival Analysis)
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
# Toronto has a very low censoring rate, with about 3.45% of incidents not reaching a completed response time.  
# In contrast, NYC shows a much higher censoring rate at 29.03%, meaning a large share of incidents do not have finalized response times in the dataset.
#
# This suggests Toronto’s data is more complete and directly comparable for response-time analysis, while NYC’s higher censoring may reflect ongoing incidents, data recording differences, or operational complexities that should be considered when interpreting comparative results.

# %% [markdown]
# ### 2.5 Summary of Target Variable Exploration
#
# Exploratory analysis of response times shows right-skewed distributions in both Toronto and NYC, with most incidents handled quickly but a minority experiencing longer delays. Mean values exceed medians, and higher-percentile metrics (P90/P95) highlight tail-risk that averages alone do not capture. SLA and outlier analyses confirm that extended delays occur with meaningful frequency in both cities, with NYC showing higher rates of threshold breaches and long-delay cases.
#
# Censoring is also present in the data. Toronto has a low censoring rate (3.45%), while NYC’s is substantially higher (29.03%). To maintain valid comparisons, distributional analyses were conducted on completed incidents only, while censored cases are retained for survival-based modeling. Overall, the target variable exhibits skewness, tail-risk, and censoring effects that justify the use of tail-sensitive and censor-aware methods in subsequent modeling stages.

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
# Both cities show relatively stable average response times throughout the day, with modest variation by hour. Toronto maintains slightly lower average response times overall, particularly during daytime and evening hours, while NYC shows higher averages during overnight and early-morning periods (approximately 1–7 AM).
#
# P90 patterns reveal clearer differences in tail risk. NYC consistently records higher P90 values across nearly all hours, with the largest gaps occurring in early-morning periods when delays are most pronounced. Toronto’s P90 remains more stable and lower throughout the day, indicating fewer extreme delays.
#
# Overall, while typical response times are broadly comparable, NYC experiences greater hourly variability and higher tail-delay pressure, especially overnight, whereas Toronto demonstrates more consistent performance across the daily cycle.
#
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
# Both cities show clear daily demand cycles, with lower incident volumes overnight and steady increases beginning in the morning. Volumes rise sharply from around 7–9 AM, peak in the afternoon and early evening (approximately 14:00–19:00), and then gradually decline into the night.
#
# NYC consistently handles a much higher number of incidents per hour than Toronto across the entire day, reflecting its larger operational scale. Despite this higher volume, peak demand periods in both cities follow similar timing patterns, suggesting comparable urban activity cycles.
#
# These hourly demand trends provide important context for interpreting response-time patterns and highlight the need to consider staffing and resource allocation during afternoon and early evening peak periods.
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
# Both cities show consistent temporal patterns across the week. Response times tend to be higher during overnight and early-morning hours (roughly 1–7 AM) and improve during daytime and evening periods. This pattern is visible across most days of the week, suggesting that time-of-day effects are stronger than day-of-week differences.
#
# NYC generally records higher response times across nearly all hour–day combinations, with the most pronounced delays occurring in early-morning periods. Toronto shows more stable and slightly lower response times overall, with less extreme variation across the weekly cycle.
#
# Overall, temporal patterns are broadly similar between the two cities, but NYC experiences consistently higher delays and greater overnight pressure, while Toronto maintains more stable performance throughout the week.
#
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
        F.round(F.expr("percentile_approx(response_minutes, 0.5)"), 2).alias("median_response"),
        F.round(F.expr("percentile_approx(response_minutes, 0.9)"), 2).alias("p90_response")
    )
    .withColumn("city", F.lit("NYC"))
)

# Toronto (make sure yours matches this structure)
toronto_weekend_stats = (
    toronto_wd
    .filter(F.col("response_minutes").isNotNull())
    .groupBy("is_weekend")
    .agg(
        F.round(F.mean("response_minutes"), 2).alias("avg_response"),
        F.round(F.expr("percentile_approx(response_minutes, 0.5)"), 2).alias("median_response"),
        F.round(F.expr("percentile_approx(response_minutes, 0.9)"), 2).alias("p90_response")
    )
    .withColumn("city", F.lit("Toronto"))
)

# Combine
weekend_comparison = (
    toronto_weekend_stats
    .unionByName(nyc_weekend_stats)
    .select("city", "is_weekend", "avg_response", "median_response", "p90_response")
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

ax.set_title("P90 Response Time: Weekday vs Weekend", fontsize=13, fontweight="bold")
ax.set_xlabel("Day Type", fontsize=11)
ax.set_ylabel("P90 Response Time (minutes)", fontsize=11)
# Smaller ticks
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
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

ax.set_title("Average Response Time: Weekday vs Weekend", fontsize=13, fontweight="bold")
ax.set_xlabel("Day Type", fontsize=11)
ax.set_ylabel("Average Response Time (minutes)", fontsize = 11)
# Smaller ticks
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

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
# Response-time differences between weekdays and weekends are modest in both cities.  
# NYC shows slightly faster performance on weekends, with lower average, median, and P90 response times compared to weekdays. Toronto follows a similar pattern, with marginally lower response times on weekends across all metrics.
#
# Despite these small improvements, the overall weekday–weekend gap is minimal, suggesting that time-of-day and demand patterns likely have a stronger influence on response performance than day type alone. Across both periods, Toronto maintains consistently lower average and tail response times than NYC.
#
#

# %% [markdown]
# ### 3.7 Seasonal Response Time Analysis
# This section evaluates whether emergency response performance varies by season, which can inform staffing and operational readiness planning.
#
# We compare response-time performance across seasons using:
# - Average response time (mean)
# - Median response time (P50)
# - Tail performance (P90)
#
# Results are reported for both Toronto and NYC to support cross-city benchmarking of seasonal patterns.
#
#

# %% [markdown]
# #### 3.7.1 Seasonal statistics

# %%
def seasonal_stats(df, city_name):
    return (
        df.filter(F.col("response_minutes").isNotNull())
          .groupBy("season")
          .agg(
              F.round(F.mean("response_minutes"), 2).alias("avg_response"),
              F.round(F.expr("percentile_approx(response_minutes, 0.5)"), 2).alias("median_response"),
              F.round(F.expr("percentile_approx(response_minutes, 0.9)"), 2).alias("p90_response"),
              F.count("*").alias("n_incidents")
          )
          .withColumn("city", F.lit(city_name))
    )


# %%
toronto_season_stats = seasonal_stats(toronto_df, "Toronto")
nyc_season_stats = seasonal_stats(nyc_df, "NYC")

season_comparison = (
    toronto_season_stats
    .unionByName(nyc_season_stats)
    .select("city", "season", "n_incidents", "avg_response", "median_response", "p90_response")
)

# order seasons in a logical order (Spring, Summer, Fall, Winter)
season_clean = season_comparison.withColumn(
    "season",
    F.initcap("season")
)
season_ordered = season_clean.withColumn(
    "season_order",
    F.when(F.col("season") == "Spring", 1)
     .when(F.col("season") == "Summer", 2)
     .when(F.col("season") == "Fall", 3)
     .when(F.col("season") == "Winter", 4)
)
season_final = (
    season_ordered
    .orderBy("city", "season_order")
    .drop("season_order")
)

display(season_final)


# %% [markdown]
# #### 3.7.2 Average Response Time by Season Plot

# %%
pdf_season = season_final.toPandas()

plt.figure(figsize=(8,5))
ax = sns.barplot(data=pdf_season, x="season", y="avg_response", hue="city")

# Add data labels
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.2f}",
                (p.get_x() + p.get_width()/2, height),
                ha="center", va="bottom",
                fontsize=9,
                xytext=(0,3),
                textcoords="offset points")

plt.title("Average Response Time by Season", fontsize=13, fontweight="bold")
plt.xlabel("Season", fontsize=11)
plt.ylabel("Average Response Time (minutes)", fontsize=11)
# Smaller ticks
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
# Legend outside
plt.legend(title="City", bbox_to_anchor=(1.02,1), loc="upper left")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### 3.7.2 P90 Response Time by Season

# %%
plt.figure(figsize=(8,5))
ax = sns.barplot(data=pdf_season, x="season", y="p90_response", hue="city")

# Data labels
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.2f}",
                (p.get_x() + p.get_width()/2, height),
                ha="center", va="bottom",
                fontsize=9,
                xytext=(0,3),
                textcoords="offset points")

plt.title("P90 Response Time by Season (Tail Performance)", fontsize=13, fontweight="bold")
plt.xlabel("Season", fontsize=11)
plt.ylabel("P90 Response Time (minutes)", fontsize =11)
# Smaller ticks
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)

plt.legend(title="City", bbox_to_anchor=(1.02,1), loc="upper left")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### 3.7.3 Response Time Distribution by Season

# %%
combined = (
    toronto_df.select("season", "response_minutes").withColumn("city", F.lit("Toronto"))
    .unionByName(nyc_df.select("season", "response_minutes").withColumn("city", F.lit("NYC")))
    .filter(F.col("response_minutes").isNotNull())
    .withColumn("season", F.initcap("season"))  # <-- FIX
    .sample(False, 0.05, seed=42)
)

box_pdf = combined.toPandas()

season_cat = ["Spring", "Summer", "Fall", "Winter"]
box_pdf["season"] = pd.Categorical(box_pdf["season"], categories=season_cat, ordered=True)

plt.figure(figsize=(9, 5))
sns.boxplot(data=box_pdf, x="season", y="response_minutes", hue="city", showfliers=False)
plt.title("Response Time Distribution by Season", fontsize=13, fontweight="bold")
plt.xlabel("Season", fontsize=11, fontweight="bold")
plt.ylabel("Response Time (minutes)", fontsize = 11, fontweight="bold")
plt.legend(title="City", bbox_to_anchor=(1.02, 1), loc="upper left")  # optional: legend outside
plt.tight_layout()
plt.show()



# %% [markdown]
# #### Summary of Seasonal Response-Time Patterns
#
# Seasonal differences in response times are relatively small in both cities. Average and median response times remain stable across spring, summer, fall, and winter, indicating consistent year-round operational performance.
#
# NYC records slightly higher response times than Toronto in every season, with P90 values consistently around 8.7–8.8 minutes compared to Toronto’s 7.5–7.9 minutes. Summer shows a mild increase in both average and tail response times for each city, suggesting slightly higher demand or operational pressure during this period.
#
# Overall, seasonal effects are modest compared to hourly or tail-risk variations. Response performance is generally stable throughout the year, with Toronto maintaining lower average and tail response times than NYC across all seasons.
#

# %% [markdown]
# ### 3.8 Exploratory Delay Rate by Hour
#
# To complement response-time and volume analysis, the share of incidents exceeding the 8-minute threshold was examined across hours of the day. Delay rates broadly follow the same temporal structure observed in response-time percentiles, with elevated delay prevalence during overnight and early-morning hours. NYC exhibits consistently higher delay rates across most hours, reflecting greater tail-delay exposure. These exploratory patterns motivate formal modeling of temporal delay drivers in subsequent analysis sections.

# %% [markdown]
# #### 3.8.1 Toronto

# %%
tor_delay = (
    toronto_df
    .filter(F.col("delay_indicator").isNotNull())
    .groupBy("hour")
    .agg(F.round(F.mean("delay_indicator")*100,2).alias("delay_pct"))
    .withColumn("city", F.lit("Toronto"))
    .orderBy(F.col("hour").asc())
)
display(tor_delay)

# %%
# Convert to pandas (already sorted by hour)
delay_pd = tor_delay.toPandas()

plt.figure(figsize=(10,5))

ax = sns.barplot(
    data=delay_pd,
    x="hour",
    y="delay_pct"
)

ax.set_title("Toronto: Delay Rate by Hour of Day (>8 minutes)", fontsize=13, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Delayed Incidents (%)")

# show every hour
ax.set_xticks(range(0,24))

# grid
ax.grid(axis="y", linestyle="--", alpha=0.5)

# ---- labels on bars ----
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height:.1f}",
        (p.get_x() + p.get_width()/2., height),
        ha="center",
        va="bottom",
        fontsize=8,
        xytext=(0,2),
        textcoords="offset points"
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# #### 3.8.2 NYC

# %%
nyc_delay = (
    nyc_df
    .filter(F.col("delay_indicator").isNotNull())
    .groupBy("hour")
    .agg(F.round(F.mean("delay_indicator")*100,2).alias("delay_pct"))
    .withColumn("city", F.lit("NYC"))
    .orderBy(F.col("hour").asc())
)
display(nyc_delay)

# %%
# Convert to pandas (already sorted by hour)
delay_pd = nyc_delay.toPandas()

plt.figure(figsize=(10,5))

ax = sns.barplot(
    data=delay_pd,
    x="hour",
    y="delay_pct"
)

ax.set_title("NYC: Delay Rate by Hour of Day (>8 minutes)", fontsize=13, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Delayed Incidents (%)")

# show every hour
ax.set_xticks(range(0,24))

# grid
ax.grid(axis="y", linestyle="--", alpha=0.5)

# ---- labels on bars ----
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height:.1f}",
        (p.get_x() + p.get_width()/2., height),
        ha="center",
        va="bottom",
        fontsize=8,
        xytext=(0,2),
        textcoords="offset points"
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.9 Summary of Temporal Delay Patterns (>8 minutes)
#
# Delay rates vary clearly by hour of day in both cities, with the highest risk occurring during overnight and early-morning periods. Toronto’s delay rate peaks around 2–5 AM (about 10–11%) and declines to its lowest levels in the early evening (around 6–7%), before rising slightly again late at night.  
#
# NYC shows a similar daily pattern but at consistently higher levels. Delay rates climb sharply overnight and peak between roughly 5–7 AM (about 20–21%), then gradually decline through the day, reaching their lowest levels late evening (around 10%).  
#
# Overall, both cities experience the greatest delay risk during overnight hours, but NYC’s rates are substantially higher across nearly all time periods. This indicates stronger overnight operational pressure and greater temporal variability in NYC, while Toronto maintains lower and more stable delay rates throughout the day.
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
tor_area = (
    toronto_df
    .filter(F.col("response_minutes").isNotNull())
    .groupBy("location_area")
    .agg(
        F.count("*").alias("n_total"),
        F.round(F.mean("response_minutes"), 2).alias("mean_response")
    )
    .withColumn("city", F.lit("Toronto"))
)
nyc_area = (
    nyc_df
    .filter(F.col("response_minutes").isNotNull())
    .groupBy("location_area")
    .agg(
        F.count("*").alias("n_total"),
        F.round(F.mean("response_minutes"), 2).alias("mean_response")
    )
    .withColumn("city", F.lit("NYC"))
)

area_compare = tor_area.unionByName(nyc_area)

display(area_compare.orderBy("city", F.desc("n_total")))

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
# **Toronto Spatial Response-Time Patterns (Station Areas)**
#
# Average response times across Toronto station areas are generally close to the citywide mean of about 4.83 minutes, with most high-volume areas performing near or below this benchmark. A few station areas show higher mean response times (above 5 minutes), indicating localized pockets of slower performance.
#
# Higher call volumes do not consistently correspond to slower response times. Some of the busiest station areas maintain relatively fast averages, suggesting efficient resource allocation, while a few lower-volume areas exhibit higher delays. Overall, spatial variation exists but is moderate, with most station areas demonstrating stable and comparable response performance across the city.

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
# **NYC Spatial Response-Time Patterns (Borough)**
#
# Average response times vary modestly across NYC boroughs but remain close to the citywide mean of about 5.85 minutes. The Bronx records the highest mean response time, followed by Manhattan and Queens, while Brooklyn performs slightly better and Staten Island shows the lowest average response time.
#
# Incident volume differs substantially by borough, with Brooklyn handling the largest share of calls and Staten Island the smallest. Despite these volume differences, response-time variation across boroughs is relatively moderate, suggesting broadly consistent performance citywide with some localized pressure in higher-delay areas such as the Bronx.
#

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
#
# Most incidents in both cities occur at alarm level 1, and response times are broadly similar across alarm levels. Higher alarm levels (2 and 3) are relatively rare and show slightly faster average and P90 response times, likely reflecting prioritization and rapid dispatch for more severe events.
#
# Call source shows clearer performance differences. In Toronto, public-initiated calls have the highest mean and P90 response times, while alarm-system calls tend to be handled more quickly and consistently. EMS/medical calls dominate total volume and exhibit performance close to the citywide average.  
#
# In NYC, EMS/medical calls account for the largest share of incidents and show the highest mean and P90 response times, indicating greater complexity or operational demand. Alarm-system calls again show comparatively faster and more stable performance, while public calls fall between these extremes.
#
# Overall, alarm level has limited impact on average response performance, whereas call source—particularly EMS/medical and public calls—shows stronger associations with response-time variability and tail risk.
#

# %% [markdown]
# ### 4.5 High-Volume Areas and Response Performance (Ranked Comparison)

# %% [markdown]
# #### 4.5.1 Toroton Volume Area and Response Performance

# %%
def area_response_summary(df, area_col="location_area", p=0.9):
    """
    Area-level volume + response-time metrics.
    - Volume counts all rows
    - Response metrics computed on completed incidents only (response_minutes not null)
    """
    base = (
        df.groupBy(area_col)
          .agg(
              F.count("*").alias("n_total"),
              F.sum(F.col("response_minutes").isNull().cast("int")).alias("n_censored")
          )
          .withColumn("pct_censored", F.round(F.col("n_censored") / F.col("n_total") * 100, 2))
    )

    resp = (
        df.filter(F.col("response_minutes").isNotNull())
          .groupBy(area_col)
          .agg(
              F.count("*").alias("n_completed"),
              F.round(F.mean("response_minutes"), 2).alias("mean_response"),
              F.round(F.expr(f"percentile_approx(response_minutes, {p})"), 2).alias(f"p{int(p*100)}_response")
          )
    )

    return (
        base.join(resp, on=area_col, how="left")
            .fillna({"n_completed": 0})
            .orderBy(F.desc("n_total"))
    )


toronto_area_stats = (
    area_response_summary(toronto_df, area_col="location_area", p=0.9)
    .withColumn("city", F.lit("Toronto"))
)

display(
    toronto_area_stats
    .select("location_area", "n_total", "mean_response", "p90_response")
    .orderBy(F.desc("n_total"))
    .limit(15)
)

# %% [markdown]
# #### 4.5.2 NYC Volume Area and Response Performance

# %%
nyc_area_stats = (
    area_response_summary(nyc_df, area_col="location_area", p=0.9)
    .withColumn("city", F.lit("NYC"))
)

display(
    nyc_area_stats
    .select("location_area", "n_total", "mean_response", "p90_response")
    .orderBy(F.desc("n_total"))
    .limit(15)
)


# %% [markdown]
# #### 4.5.3 Summary of High-Volume Areas and Response Performance
#
# High-volume areas in both cities generally maintain response times close to their respective citywide averages, indicating that heavy demand does not automatically translate into slower performance. In Toronto, several of the busiest station areas (e.g., 314, 325, 332) show relatively fast mean and P90 response times, suggesting efficient resource deployment in core service zones. However, a few high- or mid-volume areas (such as 442 and 231) exhibit higher averages and tail delays, pointing to localized operational pressure.
#
# In NYC, Brooklyn handles the largest incident volume while maintaining comparatively lower mean response times than the Bronx and Manhattan, indicating stronger efficiency under high demand. The Bronx and Manhattan show higher mean and P90 values despite slightly lower volumes, suggesting greater congestion or operational complexity in those boroughs.
#
# Overall, high incident volume alone does not determine slower response performance. Some high-demand areas demonstrate strong operational efficiency, while certain lower- or mid-volume areas show elevated delays, highlighting the importance of localized resource allocation and operational conditions.
#
#

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
    palette = sns.color_palette("tab10", n_colors=k)   # distinct colors

    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=pdf["cluster"],
        palette=palette,
        s=80,
        edgecolor="black"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{city_name}: Location Risk Clusters (PCA projection)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")

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
#
# Exploratory clustering was conducted to group Toronto station areas based on response-time characteristics and incident volume. The elbow method suggests that a 3–4 cluster solution provides a reasonable balance between model simplicity and explanatory power, with diminishing improvements in inertia beyond this range. A 4-cluster structure was selected to capture meaningful variation in operational risk profiles.
#
# The resulting clusters reveal distinct location patterns. Some areas combine high call volume with relatively fast response times, indicating efficient resource deployment. Others show moderate volume but higher mean and P90 response times, suggesting localized delay risk. A smaller set of locations exhibits low volume yet elevated response times, pointing to potential coverage or resource-allocation challenges.
#
# Overall, clustering highlights that response-time performance varies across locations in structured ways rather than randomly. These groupings provide a useful foundation for identifying high-risk areas, informing staffing considerations, and incorporating location risk profiles as features in subsequent modeling.
#
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
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)

plt.xticks(rotation=45)

plt.title("NYC Mean vs P90 Response Time by Location", fontsize=13, fontweight="bold")
plt.xlabel("Location Area", fontsize=11, fontweight = "bold")
plt.ylabel("Response Time (minutes)", fontsize=11, fontweight = "bold")

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
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)

plt.xticks(rotation=45)

plt.title("NYC Incident Volume by Location", fontsize=13, fontweight="bold")
plt.xlabel("Location Area", fontsize=11, fontweight = "bold")
plt.ylabel("Incident Count", fontsize=11, fontweight = "bold")

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
# Response-time performance varies across NYC boroughs, with the Bronx and Manhattan showing the highest mean and P90 response times, indicating greater delay risk and operational pressure. Queens sits near the middle of the distribution, while Brooklyn and Staten Island demonstrate comparatively lower average and tail response times.
#
# Incident volume differs substantially by borough. Brooklyn handles the largest share of calls but maintains relatively lower mean and P90 response times, suggesting more efficient performance under high demand. In contrast, the Bronx and Manhattan show higher delays despite lower or comparable volumes, pointing to potential congestion, travel complexity, or resource constraints.
#
# Overall, the comparison shows that higher call volume does not necessarily lead to slower response times. Some high-demand areas maintain strong performance, while others exhibit elevated delays, highlighting the importance of localized operational conditions and resource allocation in shaping response outcomes.
#
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
#
# Composite risk scores, combining incident volume and P90 response time, highlight priority areas where both demand and tail-delay risk are elevated.  
#
# In Toronto, several high-volume station areas (e.g., 314, 332, 325) rank highest in composite risk due to their combination of substantial call volume and moderate tail delays. Some mid-volume areas (such as 132, 234, and 433) also emerge as risk priorities because of higher P90 response times, indicating localized delay pressure despite smaller workloads.
#
# In NYC, borough-level risk is dominated by Manhattan, Brooklyn, and the Bronx. Manhattan and the Bronx show particularly high risk due to elevated P90 response times combined with large incident volumes, while Brooklyn’s high call volume keeps its composite risk high even with somewhat lower tail delays. Staten Island shows the lowest overall risk due to substantially lower incident volume.
#
# Overall, composite tail-risk scoring provides a practical way to prioritize areas for operational attention by identifying locations where high demand and elevated delay risk intersect. This approach supports targeted resource planning and highlights areas where improvements could have the greatest impact on reducing extreme delays.
#

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
#
# Response times vary by incident category in both cities, with medical-related calls accounting for the largest share of incidents and showing performance close to overall city averages. In Toronto, medical and non-structural fire incidents tend to have relatively fast and stable response times, while categories such as hazardous/utility and other assistance show higher mean and P90 values, indicating greater operational complexity and tail-delay risk.
#
# In NYC, similar patterns emerge but at generally higher response-time levels. Rescue/entrapment and other assistance calls show the highest mean and P90 response times, suggesting more complex or resource-intensive incidents. Structural fire incidents, although critical, display comparatively faster and more consistent response times, likely reflecting prioritization and rapid dispatch protocols.
#
# Overall, incident type plays a meaningful role in response-time variability. High-volume medical calls shape baseline performance, while specialized or complex incident categories contribute disproportionately to longer delays and tail-risk in both cities, particularly in NYC.
#
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
# Most incidents in both cities occur at alarm level 1 and largely determine overall response-time performance. These routine incidents show average response times around 5–6 minutes, with NYC consistently higher than Toronto in both mean and P90 values.
#
# Higher alarm levels (2 and 3) are relatively rare but exhibit faster average and tail response times in both cities. This likely reflects prioritization and rapid dispatch for more severe incidents, where additional resources and urgency reduce delays.
#
# Overall, alarm level has a limited impact on overall averages due to the dominance of level-1 calls, but higher-severity incidents tend to be handled more quickly and with lower tail-delay risk, indicating effective prioritization in emergency response operations.
#

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
# **Response-Time Definition and Unit Consistency Summary**
#
# Response time is measured in minutes for both Toronto and NYC, representing the interval between alarm initiation and unit arrival at the scene. Average response times are similar across cities, with Toronto at approximately 5.33 minutes and NYC at 5.88 minutes.
#
# Toronto shows a wider overall range, with a minimum of 0.08 minutes and a maximum exceeding 113 minutes, indicating the presence of extreme outliers or exceptional delays. NYC’s range is narrower, from 0.15 to 16.65 minutes, suggesting fewer extreme cases in the dataset used for analysis.
#
# Despite these differences in range, the response-time metric is consistently defined and comparable across both datasets, allowing for valid cross-city analysis of averages, percentiles, and delay patterns.
#
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
# **Summary of Percentile-Based Cross-City Comparison**
#
# Percentile comparisons show that typical response times are similar across the two cities, with medians around 5–5.5 minutes. However, differences become more pronounced at higher percentiles. Toronto maintains lower P75, P90, and P95 values than NYC, indicating fewer extreme delays and more stable tail performance.
#
# NYC’s higher P90 and P95 values suggest greater variability and a higher likelihood of longer response times in the upper tail of the distribution. Overall, while central performance is comparable, Toronto demonstrates stronger tail performance and lower extreme-delay risk than NYC.
#
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

# %% [markdown]
#
# **Note: Day-of-week and month were excluded from the Pearson correlation matrix because they are categorical/cyclical variables. Treating them as numeric values can produce misleading correlation coefficients. Their effects are instead evaluated through grouped summaries and temporal pattern analysis.**

# %%
# Numeric, non-binary features selected for correlation analysis
NUMERIC_COLS = [
    "response_minutes",
    "hour",
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
# **Correlation Summary of Numeric Features (Toronto)**
#
# Pearson correlation analysis shows that response time has very weak linear relationships with the available numeric predictors. Hour of day has a small negative correlation with response time (r ≈ −0.07), suggesting slightly faster responses during certain daytime periods, but the effect is minimal. Short-term demand indicators (`calls_past_30min` and `calls_past_60min`) also show near-zero correlation with response time, indicating that immediate workload alone does not strongly explain delay variation in a linear sense.
#
# The two demand variables are strongly correlated with each other (r ≈ 0.74), reflecting that they capture similar short-term call volume conditions. This suggests potential redundancy between these features in modeling.
#
# Overall, the weak correlations indicate that response-time variability is likely driven by nonlinear, spatial, and operational factors rather than simple linear relationships with individual temporal or demand variables.
#

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
# **Correlation Analysis of Numeric Features (NYC)**
#
# Pearson correlation results show that response time has very weak linear relationships with hour of day and short-term call volume indicators. Correlations between response time and hour (r ≈ −0.08), calls in the past 30 minutes (r ≈ −0.04), and calls in the past 60 minutes (r ≈ −0.05) are all close to zero, suggesting that simple linear relationships do not explain response-time variability well.
#
# However, strong relationships exist among the workload variables themselves. Hour of day is moderately correlated with recent call volume (r ≈ 0.51–0.57), indicating predictable daily demand cycles. Calls in the past 30 and 60 minutes are highly correlated (r ≈ 0.93), showing that they capture similar short-term demand conditions and may be redundant in modeling.
#
# Overall, the weak direct correlations with response time suggest that delays are likely influenced by nonlinear effects, spatial factors, and operational conditions rather than simple linear relationships with individual temporal or workload variables.
#

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
# **Response Time by Incident Category (Toronto)**
#
# Response times across incident categories are broadly similar in their central ranges, with most medians clustered around 4–6 minutes. However, variability differs by category. Hazardous/utility and rescue/entrapment incidents show slightly higher median and upper-tail response times, suggesting greater operational complexity and longer handling times. Medical and non-structural fire incidents tend to have more stable and slightly lower central response times.
#
# All categories exhibit substantial right-tail outliers, indicating that extended delays occur across incident types rather than being confined to a single category. Overall, while typical response performance is comparable across categories, specialized or complex incidents contribute more to tail-risk variability.
#

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
# **Response Time by Incident Category (NYC)**
#
# Response-time distributions vary by incident category but share similar central ranges, with most medians falling between roughly 4–6 minutes. Structural fire incidents tend to have the fastest and most consistent response times, reflecting prioritization and rapid dispatch for critical events. In contrast, rescue/entrapment and other assistance incidents show higher medians and wider upper tails, indicating greater operational complexity and increased likelihood of longer delays.
#
# Medical incidents account for a large share of calls and display moderate variability around the citywide average. All categories exhibit right-skewed distributions with noticeable upper-tail outliers, suggesting that extended delays occur across multiple incident types rather than being confined to a single category. Overall, specialized and complex incidents contribute more to response-time variability and tai
#
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
# **Response Time by Alarm Level (Toronto)**
#
# Response-time distributions are broadly similar across alarm levels, with most incidents concentrated around 4–6 minutes. Alarm level 1 accounts for the vast majority of calls and therefore drives overall performance patterns. Higher alarm levels (2 and 3), though far less frequent, show slightly lower median and P90 response times, suggesting prioritization and faster dispatch for more severe incidents.
#
# All alarm levels exhibit right-skewed distributions with upper-tail outliers, but extreme delays are most visible in level-1 incidents due to their large volume. Overall, alarm severity does not substantially increase average response times; instead, higher-severity incidents tend to be handled slightly faster, indicating effective prioritization in eme
#

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
# **Response Time by Alarm Level (NYC)**
#
# Most NYC incidents occur at alarm level 1 and show median response times around 5–6 minutes, with a wider spread and more upper-tail outliers compared to higher alarm levels. Alarm levels 2 and 3 are much less frequent but display lower median and P90 response times, indicating faster dispatch and prioritization for more severe incidents.
#
# Level-1 incidents account for the majority of extreme delays due to their high volume, while higher alarm levels exhibit tighter distributions and fewer prolonged response times. Overall, response performance does not worsen with increasing alarm severity; instead, higher-severity incidents tend to be handled more quickly, suggesting effective prioritization and resource mobilization in NYC’s response system.
#

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
# **Response Time by Call Source (Toronto)**
#
# Response-time distributions vary slightly by call source but share similar central ranges, with most medians around 4–6 minutes. Public-initiated calls and EMS/medical calls account for the largest share of incidents and show moderate variability around the citywide average. Alarm-system calls tend to have slightly lower and more stable response times, suggesting quicker dispatch for automatically triggered alerts.
#
# Calls categorized as “Other/System” display somewhat wider variability and higher upper-tail delays, indicating greater operational complexity or coordination requirements. All call sources exhibit right-skewed distributions with occasional extreme delays, though most incidents are handled within a relatively consistent time range. Overall, call source influences variability and tail risk more than typical response-t
#

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
# **Response Time by Call Source (NYC)**
#
# Response-time distributions are broadly similar across call sources, with most medians clustered around 4–6 minutes. Alarm-system calls tend to show slightly lower and more consistent response times, reflecting faster dispatch for automatically triggered incidents. EMS/medical and public-initiated calls account for the largest share of incidents and display moderate variability around the citywide average.
#
# Calls categorized as “Other/System” exhibit somewhat wider upper tails, indicating a higher likelihood of longer delays in more complex or less standardized situations. Across all call sources, distributions are right-skewed with visible upper-tail outliers, suggesting that extended delays occur across multiple call types. Overall, call source influences variability and tail risk more than typical response-time levels in NYC.
#
#

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
