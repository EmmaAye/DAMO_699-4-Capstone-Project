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

# ============================================
# Sprint 5 / RQ5 (Predictive Only): NYC Quantile Models (P90 / P95)
# Target: response_minutes (tail risk)
# Models: LightGBM Quantile Regression (q=0.90, q=0.95)
# Outputs:
# - Model artifacts saved to /Volumes/.../models/
# - Pinball loss computed
# - Validation plots saved (PNG)
# ============================================

# 1) Setup and Imports
import os, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from pyspark.sql.functions import col
from pyspark.sql import functions as F

# 2) Config
CITY = "NYC"
TABLE = "workspace.capstone_project.nyc_model_ready"
TARGET = "response_minutes"   # keep consistent with your model_ready tables

CATEGORICAL = ["incident_category", "season", "unified_call_source", "location_area"]
NUMERIC     = ["hour", "day_of_week", "month", "year", "unified_alarm_level",
               "calls_past_30min", "calls_past_60min"]

SEED = 42
CAP_MAX = 180.0   # minutes (cap extreme outliers)
TRAIN_FRACTION = 0.35
MAX_TRAIN_ROWS = 250_000
MAX_TEST_ROWS  = 120_000

MODEL_DIR = "/tmp/models"
os.makedirs(MODEL_DIR, exist_ok=True)

SAVE_Q90 = f"{MODEL_DIR}/quantile_q90_{CITY.lower()}.txt"
SAVE_Q95 = f"{MODEL_DIR}/quantile_q95_{CITY.lower()}.txt"
PLOT_Q90 = f"{MODEL_DIR}/quantile_q90_{CITY.lower()}_validation.png"
PLOT_Q95 = f"{MODEL_DIR}/quantile_q95_{CITY.lower()}_validation.png"

LGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED
)

# 3) Helpers
def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

def safe_to_pandas(df, limit_rows):
    return df.limit(limit_rows).toPandas()

def make_validation_plot(y_true, y_pred, q, save_path, city):
    # Scatter + y=x line
    plt.figure()
    n = min(len(y_true), 8000)  # keep plot fast
    idx = np.random.RandomState(SEED).choice(len(y_true), size=n, replace=False) if len(y_true) > n else np.arange(len(y_true))

    yt = np.array(y_true)[idx]
    yp = np.array(y_pred)[idx]

    plt.scatter(yt, yp, s=6)
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    plt.plot([lo, hi], [lo, hi])

    plt.title(f"{city} Quantile Validation (q={q})")
    plt.xlabel("Actual response_minutes")
    plt.ylabel(f"Predicted q{int(q*100)} response_minutes")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 4) Load + Clean
print(f"Loading {CITY} data: {TABLE}")
df = spark.table(TABLE)

if TARGET not in df.columns:
    raise ValueError(f"{CITY}: Missing target {TARGET}. Available columns: {df.columns}")

df = (
    df.filter(col(TARGET).isNotNull())
      .filter(col(TARGET) > 0)
      .filter(col(TARGET) <= CAP_MAX)
)

existing = set(df.columns)
cat_cols = [c for c in CATEGORICAL if c in existing]
num_cols = [c for c in NUMERIC if c in existing]

if len(cat_cols) + len(num_cols) == 0:
    raise ValueError(f"{CITY}: No feature columns found. Available: {df.columns}")

print("Using numeric cols:", num_cols)
print("Using categorical cols:", cat_cols)

df_model = df.select(*(num_cols + cat_cols + [TARGET]))

# 5) Split + Sample
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=SEED)
train_df = train_df.sample(withReplacement=False, fraction=TRAIN_FRACTION, seed=SEED)

train_pdf = safe_to_pandas(train_df, MAX_TRAIN_ROWS)
test_pdf  = safe_to_pandas(test_df,  MAX_TEST_ROWS)

for c in cat_cols:
    train_pdf[c] = train_pdf[c].astype("category")
    test_pdf[c] = test_pdf[c].astype("category")

X_train = train_pdf[num_cols + cat_cols]
y_train = train_pdf[TARGET].astype(float)

X_test  = test_pdf[num_cols + cat_cols]
y_test  = test_pdf[TARGET].astype(float)

# 6) Train q90
print("Training q=0.90...")
m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **LGB_PARAMS)
m90.fit(X_train, y_train)
pred90 = m90.predict(X_test)

loss90 = pinball_loss(y_test.values, pred90, 0.90)
mae90  = float(np.mean(np.abs(y_test.values - pred90)))

print(f"{CITY} Pinball Loss q90: {loss90}")
print(f"{CITY} MAE q90: {mae90}")

# Save artifact
m90.booster_.save_model(SAVE_Q90)
print("Saved:", SAVE_Q90)

# Plot
make_validation_plot(y_test.values, pred90, 0.90, PLOT_Q90, CITY)
print("Saved:", PLOT_Q90)

# 7) Train q95
print("Training q=0.95...")
m95 = lgb.LGBMRegressor(objective="quantile", alpha=0.95, **LGB_PARAMS)
m95.fit(X_train, y_train)
pred95 = m95.predict(X_test)

loss95 = pinball_loss(y_test.values, pred95, 0.95)
mae95  = float(np.mean(np.abs(y_test.values - pred95)))

print(f"{CITY} Pinball Loss q95: {loss95}")
print(f"{CITY} MAE q95: {mae95}")

m95.booster_.save_model(SAVE_Q95)
print("Saved:", SAVE_Q95)

make_validation_plot(y_test.values, pred95, 0.95, PLOT_Q95, CITY)
print("Saved:", PLOT_Q95)

# 8) Quick tail summary (mean vs P90/P95 observed vs predicted)
obs_mean = float(np.mean(y_test.values))
obs_p90  = float(np.percentile(y_test.values, 90))
obs_p95  = float(np.percentile(y_test.values, 95))

pred_p90_avg = float(np.mean(pred90))
pred_p95_avg = float(np.mean(pred95))

print("\n--- Tail Summary (minutes) ---")
print("Observed mean:", obs_mean)
print("Observed p90 :", obs_p90)
print("Observed p95 :", obs_p95)
print("Pred avg p90 :", pred_p90_avg)
print("Pred avg p95 :", pred_p95_avg)

# Cleanup
del train_pdf, test_pdf, X_train, X_test, y_train, y_test, pred90, pred95, m90, m95
gc.collect()
print("Done.")
     
