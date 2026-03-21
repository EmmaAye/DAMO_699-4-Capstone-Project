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
import gc
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT

print("Starting RQ4 TORONTO Naive Baseline...")

# ============================================================
# 1. PATHS
# ============================================================
base_output_dir = os.path.abspath("../../../output")
tables_dir = os.path.join(base_output_dir, "tables")
graphs_dir = os.path.join(base_output_dir, "graphs")

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================
label_col = "delay_indicator"

df = spark.table("workspace.capstone_project.toronto_model_ready")
df = df.filter(F.col(label_col).isNotNull())

categorical_cols = [
    "incident_category",
    "season",
    "unified_call_source",
    "location_area"
]

numeric_cols = [
    "hour",
    "day_of_week",
    "month",
    "year",
    "unified_alarm_level",
    "calls_past_30min",
    "calls_past_60min"
]

required_cols = ["incident_datetime"] + categorical_cols + numeric_cols + [label_col]

base_df = (
    df.select(*required_cols)
    .dropna(subset=numeric_cols + [label_col])
)

train_df, test_df = base_df.randomSplit([0.8, 0.2], seed=42)
train_count = train_df.count()
test_count = test_df.count()

print("Train size:", train_count)
print("Test size :", test_count)

# ============================================================
# 3. EVALUATORS
# ============================================================
roc_eval = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_eval = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

precision_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="f1"
)

# ============================================================
# 4. UDFS
# ============================================================
to_prob_vector = udf(
    lambda p: Vectors.dense([float(1.0 - p), float(p)]),
    VectorUDT()
)

to_raw_vector = udf(
    lambda p: Vectors.dense([float(1.0 - p), float(p)]),
    VectorUDT()
)

# ============================================================
# 5. NAIVE BASELINE
# ============================================================
class_counts = (
    train_df.groupBy(label_col)
    .count()
    .orderBy(F.desc("count"))
    .collect()
)

majority_class = float(class_counts[0][label_col])
majority_count = class_counts[0]["count"]

positive_rate = train_df.agg(
    F.avg(F.col(label_col).cast("double")).alias("positive_rate")
).collect()[0]["positive_rate"]

positive_rate = float(positive_rate) if positive_rate is not None else 0.0

print(f"Majority class: {majority_class}")
print(f"Training majority count: {majority_count} / {train_count}")
print(f"Training positive rate: {positive_rate:.6f}")

baseline_predictions = (
    test_df
    .withColumn("prediction", F.lit(majority_class).cast(DoubleType()))
    .withColumn("baseline_positive_rate", F.lit(float(positive_rate)))
    .withColumn("rawPrediction", to_raw_vector(F.col("baseline_positive_rate")))
    .withColumn("probability", to_prob_vector(F.col("baseline_positive_rate")))
    .drop("baseline_positive_rate")
)

baseline_auc = float(roc_eval.evaluate(baseline_predictions))
baseline_pr = float(pr_eval.evaluate(baseline_predictions))
baseline_precision = float(precision_eval.evaluate(baseline_predictions))
baseline_recall = float(recall_eval.evaluate(baseline_predictions))
baseline_f1 = float(f1_eval.evaluate(baseline_predictions))
baseline_accuracy = (
    baseline_predictions.filter(F.col(label_col) == F.col("prediction")).count() / test_count
)

metrics_df = pd.DataFrame([{
    "Model": "Naive Majority Class",
    "AUC-ROC": baseline_auc,
    "PR-AUC": baseline_pr,
    "Precision": baseline_precision,
    "Recall": baseline_recall,
    "F1-Score": baseline_f1,
    "Accuracy": baseline_accuracy
}])

metrics_path = os.path.join(tables_dir, "rq4_trn_naive_majority_class_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

display(metrics_df.round(3))

pred_dist = (
    baseline_predictions.groupBy(label_col, "prediction")
    .count()
    .orderBy(label_col, "prediction")
    .toPandas()
)

pred_dist.to_csv(
    os.path.join(graphs_dir, "rq4_trn_naive_majority_class_prediction_distribution.csv"),
    index=False
)

del baseline_predictions
gc.collect()

print("RQ4 TORONTO Naive Baseline complete.")
