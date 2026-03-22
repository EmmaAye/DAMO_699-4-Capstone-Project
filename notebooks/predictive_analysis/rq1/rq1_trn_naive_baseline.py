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
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, sum as F_sum

spark = SparkSession.builder.getOrCreate()
print(f"Spark version: {spark.version}")

# =========================================================
# 1. SETUP
# =========================================================
output_dir = "../../../output"
os.makedirs(output_dir, exist_ok=True)

metrics_csv_path = f"{output_dir}/tables/rq1_trn_naive_baseline_metrics.csv"
pred_dist_csv_path = f"{output_dir}/graphs/rq1_trn_naive_baseline_prediction_distribution.csv"

label_col = "delay_indicator"

# =========================================================
# 2. LOAD DATA
# =========================================================
df = spark.table("workspace.capstone_project.toronto_model_ready").filter(col(label_col).isNotNull())

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("Train distribution:")
train_df.groupBy(label_col).count().orderBy(label_col).show()

print("Test distribution:")
test_df.groupBy(label_col).count().orderBy(label_col).show()

# =========================================================
# 3. FIND MAJORITY CLASS FROM TRAIN
# =========================================================
majority_row = (
    train_df.groupBy(label_col)
    .count()
    .orderBy(col("count").desc())
    .first()
)

majority_class = float(majority_row[label_col])
print(f"Majority class: {majority_class}")

# =========================================================
# 4. CREATE BASELINE PREDICTIONS
# =========================================================
predictions = test_df.withColumn("prediction", lit(majority_class))

# =========================================================
# 5. METRICS
# =========================================================
tp = predictions.select(
    F_sum(when((col(label_col) == 1) & (col("prediction") == 1), 1).otherwise(0)).alias("tp"),
    F_sum(when((col(label_col) == 0) & (col("prediction") == 0), 1).otherwise(0)).alias("tn"),
    F_sum(when((col(label_col) == 0) & (col("prediction") == 1), 1).otherwise(0)).alias("fp"),
    F_sum(when((col(label_col) == 1) & (col("prediction") == 0), 1).otherwise(0)).alias("fn")
).collect()[0]

tp = tp["tp"]
tn = tp if False else predictions.select(
    F_sum(when((col(label_col) == 0) & (col("prediction") == 0), 1).otherwise(0)).alias("tn")
).collect()[0]["tn"]
fp = predictions.select(
    F_sum(when((col(label_col) == 0) & (col("prediction") == 1), 1).otherwise(0)).alias("fp")
).collect()[0]["fp"]
fn = predictions.select(
    F_sum(when((col(label_col) == 1) & (col("prediction") == 0), 1).otherwise(0)).alias("fn")
).collect()[0]["fn"]

total = tp + tn + fp + fn

accuracy = (tp + tn) / total if total > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

# AUC-ROC for naive constant classifier is effectively 0.5
auc_roc = 0.5

metrics_df = pd.DataFrame([{
    "Model": "Naive Majority Baseline",
    "AUC-ROC": round(auc_roc, 3),
    "Precision": round(precision, 3),
    "Recall": round(recall, 3),
    "F1-Score": round(f1, 3),
    "Accuracy": round(accuracy, 3)
}])

metrics_df.to_csv(metrics_csv_path, index=False)

print("\nNaive Baseline Metrics:")
print(metrics_df.to_string(index=False))

# =========================================================
# 6. SAVE PREDICTION DISTRIBUTION
# =========================================================
pred_dist = predictions.groupBy(label_col, "prediction").count().orderBy(label_col, "prediction")
pred_dist_pd = pred_dist.toPandas()
pred_dist_pd.to_csv(pred_dist_csv_path, index=False)

print("\nPrediction distribution:")
print(pred_dist_pd)

print("\nNaive baseline complete.")
