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
# This notebook uses **one consistent survival framework** (uniform 60-minute administrative censoring) to answer the research questions:
#
# - **RQ1 (Temporal):** hour / day-of-week / season effects → stratified KM + log-rank (+ Cox results read-in, if available)
# - **RQ2 (Demand):** calls_past_{30,60}min effects → Cox results read-in
# - **RQ3 (Cross-city):** Toronto vs NYC KM overlay + log-rank + hazard comparison
# - **RQ4 (Drivers):** compare whether temporal+demand explain delay risk more than incident type → Cox results read-in (optional)
# - **RQ5 (Tail risk):** survival-based probabilities at thresholds (e.g., 10/15/30/60) vs averages
#
# **Important consistency rule:** because NYC is capped at 60 minutes, all cross-city survival analyses are interpreted **within 0–60 minutes** using:
# - duration = `response_minutes` (NULL → 60; >60 → 60)
# - event = `event_indicator` (event=1 only if observed within 60; otherwise 0)
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

# %%
try:
    spark
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import importlib

REPO_ROOT = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PACKAGE_NAME = "src.models.survival_analysis"   # this must be the folder name in your repo

survival_lib = importlib.import_module(PACKAGE_NAME)

from survival_lib import (
    prepare_city_df,
    fit_km,
    km_overlay_plot,
    survival_at_thresholds,
    validate_km,
    binned_hazard,
    hazard_overlay_plot,
    cross_city_logrank,
    cross_city_summary_text,
)
