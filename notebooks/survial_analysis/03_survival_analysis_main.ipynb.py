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
import sys

SRC_PATH = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project/src"
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# %%
from src.models.survival_analysis.censoring import apply_uniform_censoring_pandas

# %%
from src.models.survival_analysis import (
    load_city_survival_spark,
    fit_km,
    km_overlay_plot,
    survival_at_thresholds,
    binned_hazard,
    hazard_overlay_plot,
    cross_city_logrank,
    cross_city_summary_text,
    apply_uniform_censoring_pandas,
)

# %%
# %pip install lifelines

# %% [markdown]
# # US4.5: Cross-City Survival Comparison
#
# This notebook compares response-time survival patterns between **Toronto** and **NYC** to identify differences in delay risk, service reliability, and time-to-completion behaviour.
#
# The analysis:
# - Applies a consistent censoring threshold and event definition across both cities  
# - Builds Kaplan–Meier survival curves for direct comparison  
# - Compares hazard patterns to understand where delay risk differs  
# - Uses a statistical test (log-rank) to assess whether differences are meaningful  
#
# Outputs include cross-city survival plots, hazard comparisons, statistical results, and a short interpretation for reporting and dashboard use.

# %% [markdown]
# ## Setup and Config

# %%
import os, json
import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter

# %% [markdown]
# ## 1. Data Preparation & Survival Modeling
#
# ### Goal
# Prepare aligned survival datasets for Toronto and NYC so curves can be compared directly.
#
# ### What this Section does
# - Loads Toronto + NYC survival CSVs
# - Checks required columns exist
# - Applies the SAME censoring threshold and event definition to both cities
# - Combines into one analysis-ready dataset with a `city` column
# - Fits Kaplan–Meier models for each city
# - Exports combined dataset + KM curve points for dashboarding
#
# ## Inputs needed
# - Toronto survival CSV (must include duration + event columns)
# - NYC survival CSV (must include duration + event columns)
#
# ## Outputs
# - `outputs/us45/combined_survival_us45.csv`
# - `outputs/us45/km_curves.csv`

# %% [markdown]
# Get User Config
