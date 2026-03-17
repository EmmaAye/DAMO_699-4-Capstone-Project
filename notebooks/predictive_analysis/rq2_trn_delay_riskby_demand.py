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
# ============================================================
# RQ2: TORONTO Demand Effect on Delay Risk
# Fully updated version with threshold-based classification
#
# Includes:
# - String categorical handling
# - Naive Majority baseline
# - Logistic Regression, Random Forest, GBT
# - ROC point generation
# - ROC curve plot saved to output/graphs
# - Custom threshold for Precision / Recall / F1
# ============================================================

import gc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

# =========================
# 0. Config
# =========================
threshold = 0.30

# =========================
# 1. Load Toronto model-ready data
# =========================
city_name = "toronto"
source_table = "workspace.capstone_project.toronto_model_ready"

print(f"Loading {city_name.upper()} model-ready data from {source_table}...")
print(f"Using classification threshold: {threshold}")

df = spark.table(source_table)

categorical_cols = ["incident_category", "location_area"]
numeric_cols = ["calls_past_30min", "calls_past_60min"]
feature_cols = categorical_cols + numeric_cols
label_col = "delay_indicator"

df = (
    df.filter(F.col(label_col).isNotNull())
      .dropna(subset=feature_cols + [label_col])
      .withColumn(label_col, F.col(label_col).cast("double"))
)

print("Overall class distribution:")
df.groupBy(label_col).count().orderBy(label_col).show()

# =========================
# 2. Train/test split
# =========================
train_raw, test_raw = df.randomSplit([0.8, 0.2], seed=42)

train_count = train_raw.count()
test_count = test_raw.count()

print(f"Training rows: {train_count}")
print(f"Testing rows : {test_count}")

print("Training label distribution:")
train_raw.groupBy(label_col).count().orderBy(label_col).show()

print("Testing label distribution:")
test_raw.groupBy(label_col).count().orderBy(label_col).show()

# =========================
# 3. Encode categorical features
# =========================
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(
        inputCol=f"{c}_idx",
        outputCol=f"{c}_ohe"
    )
    for c in categorical_cols
]

assembled_inputs = [f"{c}_ohe" for c in categorical_cols] + numeric_cols

assembler = VectorAssembler(
    inputCols=assembled_inputs,
    outputCol="features",
    handleInvalid="skip"
)

feature_pipeline = Pipeline(stages=indexers + encoders + [assembler])

feature_model = feature_pipeline.fit(train_raw)

train_df = feature_model.transform(train_raw).select(
    *categorical_cols, *numeric_cols, label_col, "features"
)

test_df = feature_model.transform(test_raw).select(
    *categorical_cols, *numeric_cols, label_col, "features"
)

print("Encoded feature sample:")
train_df.show(5, truncate=False)

# =========================
# 4. Precompute demand thresholds
# =========================
q25, q75 = test_df.approxQuantile("calls_past_60min", [0.25, 0.75], 0.01)
print(f"Demand thresholds from test set -> Q25: {q25}, Q75: {q75}")

# =========================
# 5. Helper UDFs for baseline vectors
# =========================
to_prob_vector = udf(
    lambda p: Vectors.dense([float(1.0 - p), float(p)]),
    VectorUDT()
)

to_raw_vector = udf(
    lambda p: Vectors.dense([float(1.0 - p), float(p)]),
    VectorUDT()
)

# =========================
# 6. Model definitions
# =========================
models = [
    (
        "Logistic Regression",
        LogisticRegression(
            featuresCol="features",
            labelCol=label_col,
            maxIter=100
        )
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            featuresCol="features",
            labelCol=label_col,
            numTrees=50,
            maxDepth=5,
            seed=42
        )
    ),
    (
        "GBT Classifier",
        GBTClassifier(
            featuresCol="features",
            labelCol=label_col,
            maxIter=12,
            maxDepth=5,
            stepSize=0.1,
            seed=42
        )
    )
]

# =========================
# 7. Storage containers
# =========================
all_results = []
roc_points_dfs = []

# =========================
# 8. Naive Majority Class baseline
# =========================
print("\n" + "=" * 90)
print("Evaluating Naive Majority Class baseline...")

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

print(f"Majority class in training data: {majority_class}")
print(f"Training majority count: {majority_count} / {train_count}")
print(f"Training positive rate: {positive_rate:.6f}")

baseline_predictions = (
    test_df
    .withColumn("prediction", F.lit(majority_class).cast(DoubleType()))
    .withColumn("baseline_positive_rate", F.lit(float(positive_rate)))
    .withColumn("rawPrediction", to_raw_vector(F.col("baseline_positive_rate")))
    .withColumn("probability", to_prob_vector(F.col("baseline_positive_rate")))
    .drop("baseline_positive_rate")
    .withColumn("prob_array", vector_to_array(F.col("probability")))
    .withColumn("delay_risk_probability", F.col("prob_array").getItem(1))
    .withColumn(
        "custom_prediction",
        F.when(F.col("delay_risk_probability") >= threshold, F.lit(1.0)).otherwise(F.lit(0.0))
    )
    .withColumn(
        "demand_group",
        F.when(F.col("calls_past_60min") <= q25, F.lit("Low Demand"))
         .when(F.col("calls_past_60min") <= q75, F.lit("Medium Demand"))
         .otherwise(F.lit("High Demand"))
    )
)

print(f"\nPrediction distribution for Naive Majority Class at threshold {threshold}:")
baseline_predictions.groupBy("custom_prediction").count().orderBy("custom_prediction").show()

print("\nActual label distribution in baseline test set:")
baseline_predictions.groupBy(label_col).count().orderBy(label_col).show()

baseline_eval_pd = baseline_predictions.select(
    F.col(label_col).cast("double").alias("label"),
    F.col("custom_prediction").cast("double").alias("prediction"),
    F.col("delay_risk_probability").cast("double").alias("score")
).toPandas()

y_true = baseline_eval_pd["label"].values
y_pred = baseline_eval_pd["prediction"].values
y_score = baseline_eval_pd["score"].values

baseline_auc = roc_auc_score(y_true, y_score)
baseline_pr = average_precision_score(y_true, y_score)
baseline_precision = precision_score(y_true, y_pred, zero_division=0)
baseline_recall = recall_score(y_true, y_pred, zero_division=0)
baseline_f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Naive Majority Class AUC-ROC   : {baseline_auc:.6f}")
print(f"Naive Majority Class PR-AUC    : {baseline_pr:.6f}")
print(f"Naive Majority Class Precision : {baseline_precision:.6f}")
print(f"Naive Majority Class Recall    : {baseline_recall:.6f}")
print(f"Naive Majority Class F1 Score  : {baseline_f1:.6f}")

print("\nConfusion Matrix for Naive Majority Class:")
baseline_predictions.groupBy(label_col, "custom_prediction") \
    .count() \
    .orderBy(label_col, "custom_prediction") \
    .show()

print("\nDemand-group summary for Naive Majority Class:")
baseline_summary = baseline_predictions.groupBy("demand_group").agg(
    F.avg("delay_risk_probability").alias("avg_predicted_delay_risk"),
    F.avg(F.col(label_col).cast("double")).alias("observed_delay_rate"),
    F.avg("calls_past_30min").alias("avg_calls_past_30min"),
    F.avg("calls_past_60min").alias("avg_calls_past_60min"),
    F.count("*").alias("incident_count")
).orderBy(
    F.when(F.col("demand_group") == "Low Demand", 1)
     .when(F.col("demand_group") == "Medium Demand", 2)
     .otherwise(3)
)
baseline_summary.show(truncate=False)

fpr, tpr, _ = roc_curve(y_true, y_score)
baseline_roc_pd = pd.DataFrame({
    "model": ["Naive Majority Class"] * len(fpr),
    "fpr": fpr,
    "tpr": tpr
})
roc_points_dfs.append(baseline_roc_pd)

all_results.append((
    "Naive Majority Class",
    baseline_auc,
    baseline_pr,
    baseline_precision,
    baseline_recall,
    baseline_f1
))

del baseline_predictions, baseline_eval_pd, baseline_summary
gc.collect()

# =========================
# 9. Train and evaluate ML models
# =========================
for model_name, clf in models:
    print("\n" + "=" * 90)
    print(f"Training {model_name}...")

    model = None
    predictions = None

    try:
        model = clf.fit(train_df)
        predictions = model.transform(test_df)

        pred_cols = predictions.columns

        if "probability" in pred_cols:
            predictions = (
                predictions
                .withColumn("prob_array", vector_to_array(F.col("probability")))
                .withColumn("delay_risk_probability", F.col("prob_array").getItem(1))
            )
        elif "rawPrediction" in pred_cols:
            predictions = (
                predictions
                .withColumn("raw_array", vector_to_array(F.col("rawPrediction")))
                .withColumn(
                    "delay_risk_probability",
                    F.when(
                        (F.col("raw_array").getItem(0) + F.col("raw_array").getItem(1)) != 0,
                        F.col("raw_array").getItem(1) /
                        (F.col("raw_array").getItem(0) + F.col("raw_array").getItem(1))
                    ).otherwise(F.lit(0.0))
                )
            )
        else:
            predictions = predictions.withColumn(
                "delay_risk_probability",
                F.col("prediction").cast("double")
            )

        predictions = (
            predictions
            .withColumn(
                "custom_prediction",
                F.when(F.col("delay_risk_probability") >= threshold, F.lit(1.0)).otherwise(F.lit(0.0))
            )
            .withColumn(
                "demand_group",
                F.when(F.col("calls_past_60min") <= q25, F.lit("Low Demand"))
                 .when(F.col("calls_past_60min") <= q75, F.lit("Medium Demand"))
                 .otherwise(F.lit("High Demand"))
            )
        )

        print(f"\nPrediction distribution for {model_name} at threshold {threshold}:")
        predictions.groupBy("custom_prediction").count().orderBy("custom_prediction").show()

        print(f"\nActual label distribution in test set for {model_name}:")
        predictions.groupBy(label_col).count().orderBy(label_col).show()

        eval_pd = predictions.select(
            F.col(label_col).cast("double").alias("label"),
            F.col("custom_prediction").cast("double").alias("prediction"),
            F.col("delay_risk_probability").cast("double").alias("score")
        ).toPandas()

        y_true = eval_pd["label"].values
        y_pred = eval_pd["prediction"].values
        y_score = eval_pd["score"].values

        if len(np.unique(y_true)) < 2:
            print(f"{model_name} skipped metric calculation because test labels contain only one class.")
            continue

        auc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"{model_name} AUC-ROC   : {auc:.6f}")
        print(f"{model_name} PR-AUC    : {auc_pr:.6f}")
        print(f"{model_name} Precision : {precision:.6f}")
        print(f"{model_name} Recall    : {recall:.6f}")
        print(f"{model_name} F1 Score  : {f1:.6f}")

        print(f"\nConfusion Matrix for {model_name}:")
        predictions.groupBy(label_col, "custom_prediction") \
            .count() \
            .orderBy(label_col, "custom_prediction") \
            .show()

        print(f"\nDemand-group summary for {model_name}:")
        rq2_summary = predictions.groupBy("demand_group").agg(
            F.avg("delay_risk_probability").alias("avg_predicted_delay_risk"),
            F.avg(F.col(label_col).cast("double")).alias("observed_delay_rate"),
            F.avg("calls_past_30min").alias("avg_calls_past_30min"),
            F.avg("calls_past_60min").alias("avg_calls_past_60min"),
            F.count("*").alias("incident_count")
        ).orderBy(
            F.when(F.col("demand_group") == "Low Demand", 1)
             .when(F.col("demand_group") == "Medium Demand", 2)
             .otherwise(3)
        )
        rq2_summary.show(truncate=False)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_pd = pd.DataFrame({
            "model": [model_name] * len(fpr),
            "fpr": fpr,
            "tpr": tpr
        })
        roc_points_dfs.append(roc_pd)

        all_results.append((model_name, auc, auc_pr, precision, recall, f1))

        del eval_pd, rq2_summary
        gc.collect()

    except Exception as e:
        print(f"{model_name} failed with error: {e}")

    finally:
        model = None
        predictions = None
        gc.collect()

# =========================
# 10. Final summary table
# =========================
print("\n" + "=" * 90)
print(f"FINAL MODEL PERFORMANCE SUMMARY - TORONTO RQ2 (Threshold = {threshold})")
print("=" * 90)
print(f"{'Model':<24} {'AUC-ROC':<12} {'PR-AUC':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-" * 90)

for row in all_results:
    model_name, auc, auc_pr, precision, recall, f1 = row
    print(f"{model_name:<24} {auc:<12.6f} {auc_pr:<12.6f} {precision:<12.6f} {recall:<12.6f} {f1:<12.6f}")

# =========================
# 11. Save ROC points table
# =========================
roc_all_pd = None
roc_all_spark = None

if len(roc_points_dfs) > 0:
    roc_all_pd = pd.concat(roc_points_dfs, ignore_index=True)
    roc_all_spark = spark.createDataFrame(roc_all_pd)

    roc_output_table = "workspace.capstone_project.rq2_toronto_roc_points_all_models"
    roc_all_spark.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(roc_output_table)

    print(f"\nROC points saved to table: {roc_output_table}")
else:
    print("No ROC points were generated.")

# =========================
# 12. Plot ROC curve and save image
# =========================
if roc_all_pd is not None and len(roc_all_pd) > 0:
    graph_dir = "/Workspace/Users/pratiksha.pawar18@gmail.com/DAMO_699-4-Capstone-Project/output/graphs"
    os.makedirs(graph_dir, exist_ok=True)

    roc_plot_path = os.path.join(graph_dir, "rq2_toronto_roc_curve_all_models.png")

    plt.figure(figsize=(9, 7))

    for model_name in roc_all_pd["model"].unique():
        model_subset = (
            roc_all_pd[roc_all_pd["model"] == model_name]
            .sort_values(["fpr", "tpr"])
        )
        plt.plot(
            model_subset["fpr"],
            model_subset["tpr"],
            linewidth=2,
            label=model_name
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison - Toronto RQ2")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"ROC curve plot saved to: {roc_plot_path}")
else:
    print("ROC plot not created because no ROC points were available.")
