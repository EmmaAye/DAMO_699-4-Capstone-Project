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
# RQ4: TORONTO Short-Term Delay Risk Forecasting (Combined)
# Preserves earlier Toronto classifier feature space
# Includes:
# - Naive Majority Class baseline
# - LR / RF / GBT comparison
# - best model selection
# - retrain on full Toronto data
# - next 24-hour delay-risk forecast
# - Delta overwriteSchema fix
# ============================================================

import datetime
import gc

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT

print("Starting RQ4 TORONTO Combined Training + Forecast Pipeline...")

# ============================================================
# 1. Load Toronto data
# ============================================================

df = spark.table("workspace.capstone_project.toronto_model_ready")
df = df.filter(F.col("delay_indicator").isNotNull())

print("Toronto dataset loaded.")
df.groupBy("delay_indicator").count().show()

# ============================================================
# 2. Columns
# ============================================================

label_col = "delay_indicator"

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

print("Prepared Toronto base dataset.")
print("Row count:", base_df.count())

# ============================================================
# 3. Feature Engineering
# Same as your earlier Toronto classifier code
# ============================================================

hasher = FeatureHasher(
    inputCols=categorical_cols,
    outputCol="categorical_features",
    numFeatures=512
)

assembler = VectorAssembler(
    inputCols=numeric_cols + ["categorical_features"],
    outputCol="features",
    handleInvalid="keep"
)

# ============================================================
# 4. Train/Test Split
# ============================================================

train_df, test_df = base_df.randomSplit([0.8, 0.2], seed=42)

train_count = train_df.count()
test_count = test_df.count()

print("Train size:", train_count)
print("Test size :", test_count)

# ============================================================
# 5. Evaluators
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

results = []

# ============================================================
# 6. Helper UDFs for Naive Baseline
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
# 7. Naive Majority Class Baseline
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

print("\n" + "=" * 80)
print("Evaluating Naive Majority Class Baseline...")
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
)

baseline_auc = roc_eval.evaluate(baseline_predictions)
baseline_pr = pr_eval.evaluate(baseline_predictions)
baseline_precision = precision_eval.evaluate(baseline_predictions)
baseline_recall = recall_eval.evaluate(baseline_predictions)
baseline_f1 = f1_eval.evaluate(baseline_predictions)

print(f"Naive Majority Class AUC-ROC   : {baseline_auc:.6f}")
print(f"Naive Majority Class PR-AUC    : {baseline_pr:.6f}")
print(f"Naive Majority Class Precision : {baseline_precision:.6f}")
print(f"Naive Majority Class Recall    : {baseline_recall:.6f}")
print(f"Naive Majority Class F1 Score  : {baseline_f1:.6f}")

print("Confusion Matrix: Naive Majority Class")
baseline_predictions.groupBy(label_col, "prediction") \
    .count() \
    .orderBy(label_col, "prediction") \
    .show()

results.append((
    "Naive Majority Class",
    baseline_auc,
    baseline_pr,
    baseline_precision,
    baseline_recall,
    baseline_f1
))

del baseline_predictions
gc.collect()

# ============================================================
# 8. Trainable Models
# Preserves your earlier Toronto model settings
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=100,
        regParam=0.0,
        elasticNetParam=0.0
    ),
    "Random Forest": RandomForestClassifier(
        featuresCol="features",
        labelCol=label_col,
        numTrees=150,
        maxDepth=10,
        minInstancesPerNode=5,
        seed=42
    ),
    "GBT Classifier": GBTClassifier(
        featuresCol="features",
        labelCol=label_col,
        maxIter=80,
        maxDepth=7,
        stepSize=0.03,
        minInstancesPerNode=10,
        subsamplingRate=0.8,
        seed=42
    )
}

best_model_name = None
best_auc = -1.0

# ============================================================
# 9. Train + Evaluate Models
# ============================================================

for model_name, classifier in models.items():
    print("\n" + "=" * 80)
    print(f"Training {model_name} on Toronto...")

    try:
        pipeline = Pipeline(stages=[hasher, assembler, classifier])
        fitted_model = pipeline.fit(train_df)
        predictions = fitted_model.transform(test_df)

        auc = roc_eval.evaluate(predictions)
        auc_pr = pr_eval.evaluate(predictions)
        precision = precision_eval.evaluate(predictions)
        recall = recall_eval.evaluate(predictions)
        f1 = f1_eval.evaluate(predictions)

        print(f"{model_name} AUC-ROC   : {auc:.6f}")
        print(f"{model_name} PR-AUC    : {auc_pr:.6f}")
        print(f"{model_name} Precision : {precision:.6f}")
        print(f"{model_name} Recall    : {recall:.6f}")
        print(f"{model_name} F1 Score  : {f1:.6f}")

        print(f"Confusion Matrix for {model_name} (Actual vs Predicted):")
        predictions.groupBy(label_col, "prediction") \
            .count() \
            .orderBy(label_col, "prediction") \
            .show()

        safe_name = model_name.lower().replace(" ", "_")
        save_path = f"/Volumes/workspace/capstone_project/models/delay_classifier_toronto_{safe_name}"

        try:
            dbutils.fs.rm(save_path, True)
        except:
            pass

        fitted_model.write().overwrite().save(save_path)
        print(f"Saved {model_name} model to: {save_path}")

        results.append((model_name, auc, auc_pr, precision, recall, f1))

        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name

    except Exception as e:
        print(f"{model_name} failed with error: {e}")

    finally:
        gc.collect()

# ============================================================
# 10. Final Summary Table
# ============================================================

print("\n" + "=" * 88)
print("FINAL MODEL PERFORMANCE SUMMARY - TORONTO")
print("=" * 88)

print(f"{'Model':<24} {'AUC-ROC':<12} {'PR-AUC':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-" * 88)

for row in results:
    model_name, auc, auc_pr, precision, recall, f1 = row
    print(f"{model_name:<24} {auc:<12.6f} {auc_pr:<12.6f} {precision:<12.6f} {recall:<12.6f} {f1:<12.6f}")

if best_model_name is None:
    raise ValueError("No trainable model completed successfully.")

print(f"\nBest trainable model selected for forecasting: {best_model_name} (AUC-ROC={best_auc:.6f})")

# ============================================================
# 11. Rebuild Best Model and Train on Full Toronto Data
# ============================================================

if best_model_name == "Logistic Regression":
    best_classifier = LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=100,
        regParam=0.0,
        elasticNetParam=0.0
    )
elif best_model_name == "Random Forest":
    best_classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol=label_col,
        numTrees=150,
        maxDepth=10,
        minInstancesPerNode=5,
        seed=42
    )
else:
    best_classifier = GBTClassifier(
        featuresCol="features",
        labelCol=label_col,
        maxIter=80,
        maxDepth=7,
        stepSize=0.03,
        minInstancesPerNode=10,
        subsamplingRate=0.8,
        seed=42
    )

final_pipeline = Pipeline(stages=[hasher, assembler, best_classifier])
final_model = final_pipeline.fit(base_df)

print(f"Final {best_model_name} model trained on full Toronto data.")

# ============================================================
# 12. Save final best model for forecasting
# ============================================================

best_safe_name = best_model_name.lower().replace(" ", "_")
best_model_path = f"/Volumes/workspace/capstone_project/models/rq4_toronto_best_{best_safe_name}"

try:
    dbutils.fs.rm(best_model_path, True)
except:
    pass

final_model.write().overwrite().save(best_model_path)
print("Best forecasting model saved to:", best_model_path)

# ============================================================
# 13. Build future forecast inputs
# IMPORTANT: to preserve earlier Toronto training feature space,
# we create future rows with:
# - future hour/day/month/year
# - historical avg demand by hour/day_of_week
# - most frequent categorical values
# ============================================================

last_ts_row = df.select(F.max("incident_datetime")).collect()[0][0]
last_ts = last_ts_row if last_ts_row else datetime.datetime.now()

print("Last Toronto incident timestamp:", last_ts)

# most frequent alarm level
alarm_row = (
    base_df.groupBy("unified_alarm_level")
    .count()
    .orderBy(F.desc("count"))
    .first()
)
overall_alarm = alarm_row[0] if alarm_row is not None else 1

# most frequent category values
mode_values = {}
for c in categorical_cols:
    mode_row = (
        base_df.groupBy(c)
        .count()
        .orderBy(F.desc("count"))
        .first()
    )
    mode_values[c] = mode_row[0] if mode_row is not None else "UNKNOWN"

print("Forecast categorical defaults:", mode_values)

# historical averages by hour/day_of_week
historical_stats = base_df.groupBy("hour", "day_of_week").agg(
    F.avg("calls_past_30min").alias("calls_past_30min"),
    F.avg("calls_past_60min").alias("calls_past_60min")
)

forecast_slots = []
for i in range(1, 25):
    next_time = last_ts + datetime.timedelta(hours=i)
    spark_day_of_week = (next_time.weekday() + 1) % 7 + 1

    # derive season from month
    month_val = next_time.month
    if month_val in [12, 1, 2]:
        season_val = "Winter"
    elif month_val in [3, 4, 5]:
        season_val = "Spring"
    elif month_val in [6, 7, 8]:
        season_val = "Summer"
    else:
        season_val = "Fall"

    forecast_slots.append((
        next_time,
        next_time.hour,
        spark_day_of_week,
        next_time.month,
        next_time.year,
        overall_alarm,
        mode_values["incident_category"],
        season_val if "season" in categorical_cols else mode_values["season"],
        mode_values["unified_call_source"],
        mode_values["location_area"]
    ))

forecast_base_df = spark.createDataFrame(
    forecast_slots,
    [
        "forecast_timestamp",
        "hour",
        "day_of_week",
        "month",
        "year",
        "unified_alarm_level",
        "incident_category",
        "season",
        "unified_call_source",
        "location_area"
    ]
)

forecast_enriched_df = forecast_base_df.join(
    historical_stats,
    ["hour", "day_of_week"],
    "left"
).select(
    "forecast_timestamp",
    "incident_category",
    "season",
    "unified_call_source",
    "location_area",
    "hour",
    "day_of_week",
    "month",
    "year",
    "unified_alarm_level",
    "calls_past_30min",
    "calls_past_60min"
)

forecast_enriched_df = forecast_enriched_df.fillna({
    "calls_past_30min": 0.0,
    "calls_past_60min": 0.0,
    "unified_alarm_level": 1,
    "incident_category": mode_values["incident_category"],
    "season": mode_values["season"],
    "unified_call_source": mode_values["unified_call_source"],
    "location_area": mode_values["location_area"]
})

print("Forecast input preview:")
forecast_enriched_df.show(24, truncate=False)

# ============================================================
# 14. Generate forecast
# ============================================================

forecast_pred = final_model.transform(forecast_enriched_df)

forecast_output = (
    forecast_pred
    .withColumn("prob_array", vector_to_array(F.col("probability")))
    .withColumn("delay_risk_probability", F.col("prob_array").getItem(1))
    .withColumn("model_version", F.lit(f"{best_safe_name}_v1.0"))
    .withColumn("forecast_generated_at", F.current_timestamp())
    .withColumn("last_training_timestamp", F.lit(last_ts))
    .select(
        "forecast_timestamp",
        "hour",
        "day_of_week",
        "month",
        "year",
        "incident_category",
        "season",
        "unified_call_source",
        "location_area",
        "unified_alarm_level",
        "calls_past_30min",
        "calls_past_60min",
        "delay_risk_probability",
        "model_version",
        "forecast_generated_at",
        "last_training_timestamp"
    )
    .orderBy("forecast_timestamp")
)

print("\nNext 24-hour Toronto delay risk forecast:")
forecast_output.show(24, truncate=False)

display(forecast_output)

# ============================================================
# 15. Save forecast output
# ============================================================

output_table = "workspace.capstone_project.toronto_risk_forecast_output"

forecast_output.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(output_table)

print("\nForecast saved to table:", output_table)
print("RQ4 TORONTO Combined Pipeline Complete.")
