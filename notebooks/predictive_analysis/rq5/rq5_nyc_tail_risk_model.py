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
# !pip install lightgbm

# ============================================
# Sprint 5 / RQ5 (Predictive Only): NYC Quantile Models (P90 / P95)
# Target: response_minutes (tail risk)
# Models:
# - Simple Baseline Quantile
# - Random Forest Regressor
# - LightGBM Quantile Regressor
# Outputs:
# - LightGBM model artifacts saved to /tmp/models/
# - Pinball loss computed
# - Validation plots saved (PNG)
# - ROC comparison plots saved to output/graphs
# ============================================

# 1) Setup and Imports
import os, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from pyspark.sql.functions import col
from pyspark.sql import functions as F

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc

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

# OUTPUT_GRAPH_DIR = "/Workspace/Users/pratiksha.pawar18@gmail.com/DAMO_699-4-Capstone-Project/output/graphs"
OUTPUT_GRAPH_DIR = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project/output/graphs"
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

SAVE_Q90 = f"{MODEL_DIR}/quantile_q90_{CITY.lower()}.txt"
SAVE_Q95 = f"{MODEL_DIR}/quantile_q95_{CITY.lower()}.txt"

PLOT_Q90 = f"{OUTPUT_GRAPH_DIR}/rq5_{CITY.lower()}_q90_validation_comparison.png"
PLOT_Q95 = f"{OUTPUT_GRAPH_DIR}/rq5_{CITY.lower()}_q95_validation_comparison.png"

ROC_Q90  = f"{OUTPUT_GRAPH_DIR}/rq5_{CITY.lower()}_q90_roc_comparison.png"
ROC_Q95  = f"{OUTPUT_GRAPH_DIR}/rq5_{CITY.lower()}_q95_roc_comparison.png"

LGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED
)

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)

# 3) Helpers
def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

def safe_to_pandas(df, limit_rows):
    return df.limit(limit_rows).toPandas()

def make_validation_plot_comparison(y_true, pred_dict, q, save_path, city):
    plt.figure()
    n = min(len(y_true), 8000)
    idx = np.random.RandomState(SEED).choice(len(y_true), size=n, replace=False) if len(y_true) > n else np.arange(len(y_true))

    yt = np.array(y_true)[idx]

    all_preds = [np.array(v)[idx] for v in pred_dict.values()]
    lo = float(min([yt.min()] + [p.min() for p in all_preds]))
    hi = float(max([yt.max()] + [p.max() for p in all_preds]))

    for name, preds in pred_dict.items():
        yp = np.array(preds)[idx]
        plt.scatter(yt, yp, s=6, alpha=0.5, label=name)

    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title(f"{city} Validation Comparison (q={q})")
    plt.xlabel("Actual response_minutes")
    plt.ylabel(f"Predicted q{int(q*100)} response_minutes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

def make_roc_plot_comparison(y_true_binary, score_dict, save_path, city, label_name):
    plt.figure()
    auc_results = {}

    valid_plot = False

    for model_name, y_score in score_dict.items():
        if len(np.unique(y_true_binary)) < 2:
            print(f"Skipping ROC for {model_name}: only one class present.")
            continue

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        auc_results[model_name] = float(roc_auc)

        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.4f})")
        valid_plot = True

    if valid_plot:
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{city} ROC Comparison ({label_name})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        print("Saved ROC:", save_path)
        plt.show()
    else:
        print("ROC plot not generated due to insufficient class variation.")

    plt.close()
    return auc_results

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

# 6) Observed tail thresholds on test
obs_mean = float(np.mean(y_test.values))
obs_p90  = float(np.percentile(y_test.values, 90))
obs_p95  = float(np.percentile(y_test.values, 95))

print("\n--- Observed Tail Summary (minutes) ---")
print("Observed mean:", obs_mean)
print("Observed p90 :", obs_p90)
print("Observed p95 :", obs_p95)

# 7) Baseline predictions
baseline_q90_value = float(np.percentile(y_train.values, 90))
baseline_q95_value = float(np.percentile(y_train.values, 95))

baseline_pred90 = np.full(len(y_test), baseline_q90_value)
baseline_pred95 = np.full(len(y_test), baseline_q95_value)

baseline_loss90 = pinball_loss(y_test.values, baseline_pred90, 0.90)
baseline_loss95 = pinball_loss(y_test.values, baseline_pred95, 0.95)

baseline_mae90 = float(np.mean(np.abs(y_test.values - baseline_pred90)))
baseline_mae95 = float(np.mean(np.abs(y_test.values - baseline_pred95)))

print("\nBaseline model done.")
print(f"{CITY} Baseline Pinball Loss q90: {baseline_loss90}")
print(f"{CITY} Baseline MAE q90: {baseline_mae90}")
print(f"{CITY} Baseline Pinball Loss q95: {baseline_loss95}")
print(f"{CITY} Baseline MAE q95: {baseline_mae95}")

# 8) Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rf = RandomForestRegressor(**RF_PARAMS)
rf.fit(pd.get_dummies(X_train, drop_first=False), y_train)

X_test_rf = pd.get_dummies(X_test, drop_first=False)
X_train_rf = pd.get_dummies(X_train, drop_first=False)
X_test_rf = X_test_rf.reindex(columns=X_train_rf.columns, fill_value=0)

rf_pred = rf.predict(X_test_rf)

rf_loss90 = pinball_loss(y_test.values, rf_pred, 0.90)
rf_loss95 = pinball_loss(y_test.values, rf_pred, 0.95)

rf_mae90 = float(np.mean(np.abs(y_test.values - rf_pred)))
rf_mae95 = float(np.mean(np.abs(y_test.values - rf_pred)))

print(f"{CITY} RF Pinball Loss q90: {rf_loss90}")
print(f"{CITY} RF MAE q90: {rf_mae90}")
print(f"{CITY} RF Pinball Loss q95: {rf_loss95}")
print(f"{CITY} RF MAE q95: {rf_mae95}")

# 9) LightGBM Quantile q90
print("\nTraining LightGBM q=0.90...")
m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **LGB_PARAMS)
m90.fit(X_train, y_train)
pred90 = m90.predict(X_test)

loss90 = pinball_loss(y_test.values, pred90, 0.90)
mae90  = float(np.mean(np.abs(y_test.values - pred90)))

print(f"{CITY} LightGBM Pinball Loss q90: {loss90}")
print(f"{CITY} LightGBM MAE q90: {mae90}")

m90.booster_.save_model(SAVE_Q90)
print("Saved:", SAVE_Q90)

# 10) LightGBM Quantile q95
print("\nTraining LightGBM q=0.95...")
m95 = lgb.LGBMRegressor(objective="quantile", alpha=0.95, **LGB_PARAMS)
m95.fit(X_train, y_train)
pred95 = m95.predict(X_test)

loss95 = pinball_loss(y_test.values, pred95, 0.95)
mae95  = float(np.mean(np.abs(y_test.values - pred95)))

print(f"{CITY} LightGBM Pinball Loss q95: {loss95}")
print(f"{CITY} LightGBM MAE q95: {mae95}")

m95.booster_.save_model(SAVE_Q95)
print("Saved:", SAVE_Q95)

# 11) Predicted averages
pred_p90_avg = float(np.mean(pred90))
pred_p95_avg = float(np.mean(pred95))
rf_pred_avg  = float(np.mean(rf_pred))

print("\n--- Tail Summary (minutes) ---")
print("Observed mean:", obs_mean)
print("Observed p90 :", obs_p90)
print("Observed p95 :", obs_p95)
print("Baseline q90 constant:", baseline_q90_value)
print("Baseline q95 constant:", baseline_q95_value)
print("RF predicted avg:", rf_pred_avg)
print("LightGBM pred avg p90:", pred_p90_avg)
print("LightGBM pred avg p95:", pred_p95_avg)

# 12) Validation plots
make_validation_plot_comparison(
    y_true=y_test.values,
    pred_dict={
        "Baseline q90": baseline_pred90,
        "RF Regressor": rf_pred,
        "LightGBM q90": pred90
    },
    q=0.90,
    save_path=PLOT_Q90,
    city=CITY
)
print("Saved:", PLOT_Q90)

make_validation_plot_comparison(
    y_true=y_test.values,
    pred_dict={
        "Baseline q95": baseline_pred95,
        "RF Regressor": rf_pred,
        "LightGBM q95": pred95
    },
    q=0.95,
    save_path=PLOT_Q95,
    city=CITY
)
print("Saved:", PLOT_Q95)

# 13) ROC comparisons
# Tail-event discrimination:
# q90 positive class = actual >= observed p90
# q95 positive class = actual >= observed p95

y_test_arr = y_test.values
y_true_p90 = (y_test_arr >= obs_p90).astype(int)
y_true_p95 = (y_test_arr >= obs_p95).astype(int)

print("\n--- ROC AUC Comparison: q90 ---")
auc_q90 = make_roc_plot_comparison(
    y_true_binary=y_true_p90,
    score_dict={
        "Baseline q90": baseline_pred90,
        "RF Regressor": rf_pred,
        "LightGBM q90": pred90
    },
    save_path=ROC_Q90,
    city=CITY,
    label_name="Tail Event >= Observed P90"
)
for k, v in auc_q90.items():
    print(f"{CITY} {k} AUC q90: {v:.4f}")

print("\n--- ROC AUC Comparison: q95 ---")
auc_q95 = make_roc_plot_comparison(
    y_true_binary=y_true_p95,
    score_dict={
        "Baseline q95": baseline_pred95,
        "RF Regressor": rf_pred,
        "LightGBM q95": pred95
    },
    save_path=ROC_Q95,
    city=CITY,
    label_name="Tail Event >= Observed P95"
)
for k, v in auc_q95.items():
    print(f"{CITY} {k} AUC q95: {v:.4f}")

# 14) Final comparison table
results_df = pd.DataFrame([
    ["Baseline", "q90", baseline_loss90, baseline_mae90],
    ["RF Regressor", "q90", rf_loss90, rf_mae90],
    ["LightGBM", "q90", loss90, mae90],
    ["Baseline", "q95", baseline_loss95, baseline_mae95],
    ["RF Regressor", "q95", rf_loss95, rf_mae95],
    ["LightGBM", "q95", loss95, mae95],
], columns=["Model", "Quantile", "Pinball_Loss", "MAE"])

print("\n--- RQ5 Model Comparison ---")
display(results_df)

# Cleanup
del train_pdf, test_pdf, X_train, X_test, y_train, y_test
del baseline_pred90, baseline_pred95, rf_pred, pred90, pred95
del m90, m95, rf, X_train_rf, X_test_rf
gc.collect()
print("Done.")
