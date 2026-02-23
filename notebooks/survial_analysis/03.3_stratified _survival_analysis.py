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
# # US4.2 — Stratified Kaplan–Meier Survival Analysis
#
# This notebook performs within-city stratified survival analysis using:
# - Uniform censoring at 60 minutes
# - Kaplan–Meier curves stratified by:
#     - Time of Day
#     - Season
#     - Day of Week
# - Multivariate log-rank tests to assess statistical differences
#
# Interpretation:
# The survival curve represents:
# **Probability that the unit has not arrived yet by time t.**
#
# Lower curves → faster arrival.
# Higher curves → higher delay risk.
#

# %%
# %pip install lifelines


# %%
dbutils.library.restartPython()

# %% [markdown]
# ## 0. Import Libraries

# %%
import sys, os
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# Set Root Directories and Import Survival Analysis Libraries

# %%
# Make repo importable (adjust if your repo root differs)
REPO_ROOT = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.models.survival_analysis import (
    prepare_city_df,
    km_plot_stratified,
    run_city_logrank_tests,
    print_within_city_summary,
    STRATA_SPECS,
)

# %% [markdown]
# ## 1. Configuration

# %% [markdown]
# Set Input Table and Output Paths

# %%
TORONTO_TABLE = "workspace.capstone_project.toronto_model_ready"
NYC_TABLE     = "workspace.capstone_project.nyc_model_ready"

CENSOR_THRESHOLD = 60.0
ALPHA = 0.05

SAVE_DIR = "/Workspace/Shared/DAMO_699-4-Capstone-Project/output/graphs"
TABLE_DIR = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project/output/tables"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

print("Saving plots to:", SAVE_DIR)
print("Saving tables to:", TABLE_DIR)


# %% [markdown]
# Arrange Columns

# %% [markdown]
# ## 2. Stratified Kaplan–Meier Plots

# %%

def plot_city_stratified_km(
    pdf: pd.DataFrame,
    city_name: str,
    censor_threshold: float,
):
    """
    Plot stratified KM curves only (no statistics).
    """
    for group_col, label, order in STRATA_SPECS:
        title = f"{city_name} — Kaplan–Meier by {label}"

        ax = km_plot_stratified(
            df_pd=pdf,
            group_col=group_col,
            title=title,
            censor_threshold=censor_threshold,
            group_order=order,
        )

        out_path = os.path.join(
            SAVE_DIR,
            f"{city_name.lower()}_km_by_{label.lower().replace(' ', '_')}.png",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()
        print("Saved:", out_path)





# %%
toronto_pd = prepare_city_df(spark, TORONTO_TABLE, CENSOR_THRESHOLD)
nyc_pd     = prepare_city_df(spark, NYC_TABLE, CENSOR_THRESHOLD)

print("Toronto rows:", len(toronto_pd))
print("NYC rows:", len(nyc_pd))

# %%
plot_city_stratified_km(toronto_pd, "Toronto", CENSOR_THRESHOLD)
plot_city_stratified_km(nyc_pd, "NYC", CENSOR_THRESHOLD)

# %% [markdown]
# ## 3. Multivariate Log-Rank Tests

# %%
summary_to  = run_city_logrank_tests(toronto_pd, "Toronto", CENSOR_THRESHOLD, ALPHA)
summary_nyc = run_city_logrank_tests(nyc_pd,     "NYC",     CENSOR_THRESHOLD, ALPHA)

summary_df = pd.concat([summary_to, summary_nyc], ignore_index=True)
display(summary_df)

# %%
summary_path = os.path.join(TABLE_DIR, "logrank_summary_within_city.csv")
summary_df.to_csv(summary_path, index=False)
print("Saved summary:", summary_path)

# %% [markdown]
# ### Interpretation Summary

# %%

print_within_city_summary(summary_df, alpha=ALPHA)
