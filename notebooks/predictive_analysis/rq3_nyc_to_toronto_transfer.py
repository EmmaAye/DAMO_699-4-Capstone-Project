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
# Sprint 7 / RQ3 Subtask: Toronto -> NYC Cross-City Transfer
# Updated version:
# - Adds Naive Majority Class baseline
# - Adds GBT Classifier
# - Plots ROC curve for all 4 models
# - Saves ROC curve image to output/graphs
# - Does NOT save ROC points
# - Serverless-safe memory handling
# - Fixes GBT init error
# ============================================

import gc
import os
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    roc_curve
)

# =========================================================
# REVERSE TRANSFER BLOCK
# =========================================================
TRAIN_TABLE = "workspace.capstone_project.toronto_model_ready"
TEST_TABLE  = "workspace.capstone_project.nyc_model_ready"

TRAIN_CITY = "Toronto"
TEST_CITY  = "NYC"

SAVE_RESULTS_TABLE = "workspace.capstone_project.transfer_test_toronto_to_nyc_sprint7"

LABEL_COL = "delay_indicator"

DESIRED_CATEGORICAL = ["incident_category", "season", "unified_call_source"]
DESIRED_NUMERIC = [
    "hour", "day_of_week", "month", "year", "unified_alarm_level",
    "calls_past_30min", "calls_past_60min"
]

HASH_DIM = 512
SEED = 42
MAX_PER_CLASS = 200_000

RF_PARAMS = dict(
    numTrees=40,
    maxDepth=5,
    seed=SEED
)

GBT_PARAMS = dict(
    maxIter=12,
    maxDepth=5,
    stepSize=0.1,
    seed=SEED
)

# =========================================================
# Output graph path
# =========================================================
GRAPH_DIR = "/Workspace/Users/pratiksha.pawar18@gmail.com/DAMO_699-4-Capstone-Project/output/graphs"
ROC_FILENAME = f"rq3_{TRAIN_CITY.lower()}_to_{TEST_CITY.lower()}_roc_curve.png"
ROC_PATH = f"{GRAPH_DIR}/{ROC_FILENAME}"

# =========================================================
# Helper UDFs for Naive Baseline
# =========================================================
to_prob_vector = udf(
    lambda p: Vectors.dense([float(1.0 - p), float(p)]),
    VectorUDT()
)

to_raw_vector = udf(
    lambda p: Vectors.dense([float(1.0 - p), float(p)]),
    VectorUDT()
)

# =========================================================
# Load + Prepare
# =========================================================
def load_and_prepare(table_name: str, city_name: str):
    df = spark.table(table_name).filter(col(LABEL_COL).isNotNull())
    df = df.withColumn(LABEL_COL, col(LABEL_COL).cast("int"))

    existing = set(df.columns)
    cat_cols = [c for c in DESIRED_CATEGORICAL if c in existing]
    num_cols = [c for c in DESIRED_NUMERIC if c in existing]

    if len(cat_cols) + len(num_cols) == 0:
        raise ValueError(f"{city_name}: No usable feature columns found in {table_name}")

    dist = df.groupBy(LABEL_COL).count().orderBy(LABEL_COL)
    print(f"\n{city_name} label distribution:")
    dist.show()

    labels = [r[LABEL_COL] for r in dist.select(LABEL_COL).collect()]
    if len(labels) < 2:
        raise ValueError(f"{city_name} has only one class in {LABEL_COL}: {labels}. Fix label creation upstream.")

    keep_cols = list(dict.fromkeys(cat_cols + num_cols + [LABEL_COL]))
    return df.select(*keep_cols), cat_cols, num_cols


def balanced_train_sample(df, label_col, max_per_class=200_000, seed=42):
    df0 = df.filter(col(label_col) == 0).limit(max_per_class)
    df1 = df.filter(col(label_col) == 1).limit(max_per_class)

    n0 = df0.count()
    n1 = df1.count()

    if n0 == 0 or n1 == 0:
        raise ValueError(f"Train set has one class only (n0={n0}, n1={n1}).")

    n_min = min(n0, n1)

    frac0 = min(1.0, n_min / n0)
    frac1 = min(1.0, n_min / n1)

    df0s = df0.sample(withReplacement=False, fraction=frac0, seed=seed)
    df1s = df1.sample(withReplacement=False, fraction=frac1, seed=seed)

    balanced = df0s.unionByName(df1s)

    print(f"Balanced train sample -> class 0 fraction: {frac0:.4f}, class 1 fraction: {frac1:.4f}")
    balanced.groupBy(label_col).count().orderBy(label_col).show()

    return balanced


def build_pipeline(cat_cols, num_cols, model_name="rf"):
    stages = []

    if len(cat_cols) > 0:
        hasher = FeatureHasher(
            inputCols=cat_cols,
            outputCol="categorical_features",
            numFeatures=HASH_DIM
        )
        stages.append(hasher)
        assembler_inputs = num_cols + ["categorical_features"]
    else:
        assembler_inputs = num_cols

    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep"
    )
    stages.append(assembler)

    model_name = model_name.lower()

    if model_name == "rf":
        clf = RandomForestClassifier(
            featuresCol="features",
            labelCol=LABEL_COL,
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            predictionCol="prediction",
            **RF_PARAMS
        )
    elif model_name == "lr":
        clf = LogisticRegression(
            featuresCol="features",
            labelCol=LABEL_COL,
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            predictionCol="prediction",
            maxIter=100
        )
    elif model_name == "gbt":
        clf = GBTClassifier(
            featuresCol="features",
            labelCol=LABEL_COL,
            predictionCol="prediction",
            **GBT_PARAMS
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    stages.append(clf)
    return Pipeline(stages=stages)


def evaluate_from_pandas(eval_pd, model_label):
    y_true = eval_pd["label"].values
    y_pred = eval_pd["prediction"].values
    y_score = eval_pd["score"].values

    metrics = {
        "train_city": TRAIN_CITY,
        "test_city": TEST_CITY,
        "model_name": model_label,
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_positive_rate": float(eval_pd["label"].mean())
    }

    fpr, tpr, _ = roc_curve(y_true, y_score)

    roc_pd = pd.DataFrame({
        "model_name": [model_label] * len(fpr),
        "fpr": fpr,
        "tpr": tpr
    })

    return metrics, roc_pd


# =========================================================
# Load data
# =========================================================
train_df, train_cat, train_num = load_and_prepare(TRAIN_TABLE, TRAIN_CITY)
test_df, test_cat, test_num = load_and_prepare(TEST_TABLE, TEST_CITY)

common_cat = sorted(list(set(train_cat).intersection(set(test_cat))))
common_num = sorted(list(set(train_num).intersection(set(test_num))))

print("\nCommon numeric cols     :", common_num)
print("Common categorical cols :", common_cat)

if len(common_cat) + len(common_num) == 0:
    raise ValueError("No common feature columns between train and test tables.")

train_common = train_df.select(*(common_num + common_cat + [LABEL_COL]))
test_common  = test_df.select(*(common_num + common_cat + [LABEL_COL]))

train_sample = balanced_train_sample(train_common, LABEL_COL, MAX_PER_CLASS, SEED)

all_metrics = []
roc_points_dfs = []

# =========================================================
# 1) Naive Majority Class baseline
# =========================================================
print("\n" + "=" * 90)
print(f"Evaluating Naive Majority Class baseline ({TRAIN_CITY} -> {TEST_CITY})...")

class_counts = (
    train_sample.groupBy(LABEL_COL)
    .count()
    .orderBy(F.desc("count"))
    .collect()
)

majority_class = float(class_counts[0][LABEL_COL])
majority_count = class_counts[0]["count"]

positive_rate = train_sample.agg(
    F.avg(F.col(LABEL_COL).cast("double")).alias("positive_rate")
).collect()[0]["positive_rate"]
positive_rate = float(positive_rate) if positive_rate is not None else 0.0

print(f"Majority class in training sample: {majority_class}")
print(f"Training majority count         : {majority_count}")
print(f"Training positive rate          : {positive_rate:.6f}")

baseline_pred = (
    test_common
    .withColumn("prediction", F.lit(majority_class).cast(DoubleType()))
    .withColumn("baseline_positive_rate", F.lit(float(positive_rate)))
    .withColumn("rawPrediction", to_raw_vector(F.col("baseline_positive_rate")))
    .withColumn("probability", to_prob_vector(F.col("baseline_positive_rate")))
    .drop("baseline_positive_rate")
    .withColumn("prob_array", vector_to_array(F.col("probability")))
    .withColumn("score", F.col("prob_array").getItem(1))
    .select(
        F.col(LABEL_COL).cast("double").alias("label"),
        F.col("prediction").cast("double").alias("prediction"),
        "score",
        "rawPrediction",
        "probability"
    )
)

baseline_eval_pd = baseline_pred.select("label", "prediction", "score").toPandas()
baseline_metrics, baseline_roc_pd = evaluate_from_pandas(baseline_eval_pd, "Naive Majority Class")

all_metrics.append(baseline_metrics)
roc_points_dfs.append(baseline_roc_pd)

print(f"\nResults for Naive Majority Class ({TRAIN_CITY} -> {TEST_CITY})")
print(f"AUC-ROC   : {baseline_metrics['auc_roc']:.6f}")
print(f"PR-AUC    : {baseline_metrics['auc_pr']:.6f}")
print(f"Accuracy  : {baseline_metrics['accuracy']:.6f}")
print(f"F1 Score  : {baseline_metrics['f1']:.6f}")
print(f"Pos Rate  : {baseline_metrics['test_positive_rate']:.6f}")

print(f"\nConfusion Matrix for Naive Majority Class ({TRAIN_CITY} -> {TEST_CITY}):")
baseline_pred.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

del baseline_pred
del baseline_eval_pd
gc.collect()

# =========================================================
# 2) LR, RF, GBT transfer evaluation
# =========================================================
for model_name in ["lr", "rf", "gbt"]:
    pretty_name = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "gbt": "GBT Classifier"
    }[model_name]

    print("\n" + "=" * 90)
    print(f"Training on {TRAIN_CITY} (balanced sample) using {pretty_name} ...")

    model = None
    pred = None
    pipeline = None

    try:
        pipeline = build_pipeline(common_cat, common_num, model_name=model_name)
        model = pipeline.fit(train_sample)
        print("Training complete.")

        print(f"Testing on {TEST_CITY} ...")
        pred = model.transform(test_common).withColumn(
            "prob_array",
            vector_to_array(F.col("probability"))
        ).withColumn(
            "score",
            F.col("prob_array").getItem(1)
        ).select(
            F.col(LABEL_COL).cast("double").alias("label"),
            F.col("prediction").cast("double").alias("prediction"),
            "rawPrediction",
            "probability",
            "score"
        )

        eval_pd = pred.select("label", "prediction", "score").toPandas()
        metrics, roc_pd = evaluate_from_pandas(eval_pd, pretty_name)

        all_metrics.append(metrics)
        roc_points_dfs.append(roc_pd)

        print(f"\nResults for {pretty_name} ({TRAIN_CITY} -> {TEST_CITY})")
        print(f"AUC-ROC   : {metrics['auc_roc']:.6f}")
        print(f"PR-AUC    : {metrics['auc_pr']:.6f}")
        print(f"Accuracy  : {metrics['accuracy']:.6f}")
        print(f"F1 Score  : {metrics['f1']:.6f}")
        print(f"Pos Rate  : {metrics['test_positive_rate']:.6f}")

        print(f"\nConfusion Matrix for {pretty_name} ({TRAIN_CITY} -> {TEST_CITY}):")
        pred.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

        del eval_pd
        gc.collect()

    except Exception as e:
        print(f"{pretty_name} failed with error: {e}")

    finally:
        del pipeline
        del model
        del pred
        gc.collect()

# =========================================================
# 3) Final summary
# =========================================================
results_df = spark.createDataFrame(all_metrics).withColumn("run_ts", F.current_timestamp())

print("\nFinal summary table:")
display(results_df)

results_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(SAVE_RESULTS_TABLE)

print(f" Results saved to: {SAVE_RESULTS_TABLE}")

# =========================================================
# 4) Plot ROC curve for all models
# Save first, then show
# =========================================================
roc_plot_df = pd.concat(roc_points_dfs, ignore_index=True)

# make sure folder exists
os.makedirs(GRAPH_DIR, exist_ok=True)

plt.figure(figsize=(8, 6))

for model_name in roc_plot_df["model_name"].unique():
    subset = roc_plot_df[roc_plot_df["model_name"] == model_name]
    auc_val = next(
        m["auc_roc"] for m in all_metrics if m["model_name"] == model_name
    )
    plt.plot(subset["fpr"], subset["tpr"], label=f"{model_name} (AUC={auc_val:.6f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Chance Line")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"RQ3 Cross-City Transfer ROC Curve ({TRAIN_CITY} → {TEST_CITY})")
plt.legend(loc="lower right")
plt.grid(True)

# save first
plt.savefig(ROC_PATH, dpi=300, bbox_inches="tight")
print(f" ROC curve saved to: {ROC_PATH}")

# then show
plt.show()
plt.close()
