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
dbutils.library.restartPython()

# %%
# ---
# Baseline Kaplan–Meier Survival Curves (Toronto vs NYC)
#
# Purpose
# - Fit baseline KM models for each city using the same duration/event definition
# - Visualize Toronto and NYC survival curves
# - Validate KM medians vs empirical medians (events-only)
# - Output a short text summary for reporting
# ---

# Install dependency if needed (run once per cluster)
try:
    import lifelines
except ImportError:
    # %pip install lifelines
    dbutils.library.restartPython()

# %%
import sys
import os
import matplotlib.pyplot as plt

# Make repo importable
REPO_ROOT = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.models.survival_analysis import (
    load_city_survival_spark,
    fit_km,
    km_plot_single_city,
    km_overlay_plot,
    validate_km,
    baseline_report_text
)

# %%
TORONTO_TABLE = "workspace.capstone_project.toronto_model_ready"
NYC_TABLE     = "workspace.capstone_project.nyc_model_ready"

OUT_DIR = "/Workspace/Shared/DAMO_699-4-Capstone-Project/output/graphs"
print("Saving plots to:", OUT_DIR)


# %%
df_to = load_city_survival_spark(spark, TORONTO_TABLE)
df_nyc = load_city_survival_spark(spark, NYC_TABLE)

print("Toronto rows:", df_to.count())
print("NYC rows:", df_nyc.count())

# %% [markdown]
# **_KM curve Toronto_**

# %%
to_pd = df_to.toPandas()

km_to = fit_km(to_pd, label="Toronto")

to_path = f"{OUT_DIR}/km_baseline_toronto.png"
ax = km_plot_single_city(
    km_to,
    title="Kaplan–Meier Baseline — Toronto",
    censor_threshold=60,
)
plt.savefig(to_path, dpi=200)
plt.show()

print("Saved:", to_path)
print("KM median (Toronto):", km_to.median_survival_time_)


# %%
nyc_pd = df_nyc.toPandas()

km_nyc = fit_km(nyc_pd, label="NYC")

nyc_path = f"{OUT_DIR}/km_baseline_nyc.png"
ax = km_plot_single_city(
    km_nyc,
    title="Kaplan–Meier Baseline — NYC",
    censor_threshold=60,
)
plt.savefig(nyc_path, dpi=200)
plt.show()

print("Saved:", nyc_path)
print("KM median (NYC):", km_nyc.median_survival_time_)


# %%
both_path = f"{OUT_DIR}/km_baseline_toronto_vs_nyc.png"
ax = km_overlay_plot(
    km_to,
    km_nyc,
    censor_threshold=60,
    title="Toronto vs NYC",
)
plt.savefig(both_path, dpi=200)
plt.show()

print("Saved:", both_path)


# %%
# run validations
validate_km(to_pd, km_to, "Toronto")
validate_km(nyc_pd, km_nyc, "NYC")


# %%
print(baseline_report(km_to, km_nyc))

