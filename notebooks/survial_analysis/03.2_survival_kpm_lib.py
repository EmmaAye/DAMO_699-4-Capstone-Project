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
dbutils.library.restartPython()


# %%
from pyspark.sql.functions import col
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import os


# %%
TORONTO_TABLE = "workspace.capstone_project.toronto_model_ready"
NYC_TABLE     = "workspace.capstone_project.nyc_model_ready"

OUT_DIR = "/Workspace/Shared/DAMO_699-4-Capstone-Project/output/graphs"
print("Saving plots to:", OUT_DIR)


# %%
# Toronto: seconds -> minutes
df_to = spark.read.table(TORONTO_TABLE).select(
     col("response_minutes"),
    col("event_indicator")
).where("response_minutes is not null and response_minutes > 0 and event_indicator is not null")

# NYC: already minutes
df_nyc = spark.read.table(NYC_TABLE).select(
    col("response_minutes"),
    col("event_indicator")
).where("response_minutes is not null and response_minutes > 0 and event_indicator is not null")

print("Toronto rows:", df_to.count())
print("NYC rows:", df_nyc.count())


# %% [markdown]
# **_KM curve Toronto_**

# %%
to_pd = df_to.toPandas()

km_to = KaplanMeierFitter()
km_to.fit(durations=to_pd["response_minutes"], event_observed=to_pd["event_indicator"], label="Toronto")

plt.figure(figsize=(8,6))
km_to.plot_survival_function()
plt.title("Kaplan–Meier Baseline Survival Curve — Toronto")
plt.xlabel("Minutes")
plt.ylabel("Probability Unit Not Yet Arrived")
plt.grid(True)
plt.tight_layout()

to_path = f"{OUT_DIR}/km_baseline_toronto.png"
plt.savefig(to_path, dpi=200)
plt.show()

print("Saved:", to_path)
print("KM median (Toronto):", km_to.median_survival_time_)


# %%
nyc_pd = df_nyc.toPandas()

km_nyc = KaplanMeierFitter()
km_nyc.fit(durations=nyc_pd["response_minutes"], event_observed=nyc_pd["event_indicator"], label="NYC")

plt.figure(figsize=(8,6))
km_nyc.plot_survival_function()
plt.title("Kaplan–Meier Baseline Survival Curve — NYC")
plt.xlabel("Minutes")
plt.ylabel("Probability Unit Not Yet Arrived")
plt.grid(True)
plt.tight_layout()

nyc_path = f"{OUT_DIR}/km_baseline_nyc.png"
plt.savefig(nyc_path, dpi=200)
plt.show()

print("Saved:", nyc_path)
print("KM median (NYC):", km_nyc.median_survival_time_)


# %%
plt.figure(figsize=(8,6))
km_to.plot_survival_function()
km_nyc.plot_survival_function()
plt.title("Kaplan–Meier Baseline Survival — Toronto vs NYC")
plt.xlabel("Minutes")
plt.ylabel("Probability Unit Not Yet Arrived")
plt.grid(True)
plt.tight_layout()

both_path = f"{OUT_DIR}/km_baseline_toronto_vs_nyc.png"
plt.savefig(both_path, dpi=200)
plt.show()

print("Saved:", both_path)


# %%
def validate_km(df_pd, kmf, city_name, t_values=(5,10,15)):
    # Use observed events only for empirical comparison
    observed = df_pd[df_pd["event_indicator"] == 1].copy()

    raw_median = float(np.median(observed["response_minutes"])) if len(observed) > 0 else np.nan
    km_median  = float(kmf.median_survival_time_) if kmf.median_survival_time_ is not None else np.nan

    print(f"\n=== Validation: {city_name} ===")
    print("Raw median (events only):", raw_median)
    print("KM median:", km_median)

    for t in t_values:
        km_s = float(kmf.predict(t))
        emp_s = float(np.mean(observed["response_minutes"] > t)) if len(observed) > 0 else np.nan
        print(f"S({t}) KM={km_s:.3f} | Empirical={emp_s:.3f}  (events only)")

# run validations
validate_km(to_pd, km_to, "Toronto")
validate_km(nyc_pd, km_nyc, "NYC")


# %%
def baseline_report(km_to, km_nyc):
    tor_m = km_to.median_survival_time_
    nyc_m = km_nyc.median_survival_time_

    text = f"""
Baseline Survival (Kaplan–Meier)

Toronto:
The baseline Kaplan–Meier curve quantifies the probability that the first unit has not yet arrived as a function of time (minutes). The estimated median time-to-arrival is approximately {tor_m:.2f} minutes (KM median).

NYC:
The baseline Kaplan–Meier curve for NYC is constructed using the same survival definition and time units (minutes). The estimated median time-to-arrival is approximately {nyc_m:.2f} minutes (KM median).

Cross-city comparison:
A faster-dropping survival curve indicates quicker arrivals (lower probability of still waiting). Comparing Toronto vs NYC curves provides a direct baseline view of relative delay risk over time.
"""
    return text.strip()

print(baseline_report(km_to, km_nyc))

