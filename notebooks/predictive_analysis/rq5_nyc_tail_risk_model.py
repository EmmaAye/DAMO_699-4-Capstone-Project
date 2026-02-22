# ============================================
# RQ5 Subtask (Baseline): NYC Tail-Risk Model (GBT Regression)
# Same approach as teammate:
# - Train GBTRegressor on response_minutes
# - Compare actual_mean vs predicted_p90 / predicted_p95
# ============================================

# 1) Setup and Imports
import gc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.functions import col

# 2) Config
CITY = "NYC"
TABLE = "workspace.capstone_project.nyc_model_ready"
TARGET = "response_minutes"

# Keep features similar to teammate; auto-drop missing columns to avoid unresolved column errors
DESIRED_FEATURES = ["hour", "day_of_week", "calls_past_30min", "unified_alarm_level"]

SEED = 42
MAX_ITERS = 30
MAX_DEPTH = 6

SAVE_MODEL_PATH = "/Volumes/workspace/capstone_project/models/tail_risk_model_nyc"
SAVE_RESULTS_TABLE = "workspace.capstone_project.rq5_tailrisk_baseline_nyc"

# 3) Load + Clean
df = spark.table(TABLE).filter(col(TARGET).isNotNull())
df = df.filter(col(TARGET) > 0)

# Ensure feature cols exist
existing = set(df.columns)
feature_cols = [c for c in DESIRED_FEATURES if c in existing]
if len(feature_cols) == 0:
    raise ValueError(f"{CITY}: None of the desired features exist. Available: {df.columns}")

print(f"{CITY}: Using features = {feature_cols}")

df_model = df.select(*(feature_cols + [TARGET]))

# 4) Assemble features + Train
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

gbt = GBTRegressor(
    labelCol=TARGET,
    featuresCol="features",
    maxIter=MAX_ITERS,
    maxDepth=MAX_DEPTH,
    seed=SEED
)

pipeline = Pipeline(stages=[assembler, gbt])

train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=SEED)

print(f"Training {CITY} GBT tail-risk baseline...")
model = pipeline.fit(train_df)
print("Training complete!")

# 5) Predict + Compare mean vs tail of predictions
pred = model.transform(test_df).select(TARGET, "prediction")

summary = (
    pred.agg(
        F.avg(col(TARGET)).alias("actual_mean"),
        F.expr("percentile_approx(prediction, 0.90)").alias("predicted_p90"),
        F.expr("percentile_approx(prediction, 0.95)").alias("predicted_p95"),
        F.expr("percentile_approx(response_minutes, 0.90)").alias("actual_p90"),
        F.expr("percentile_approx(response_minutes, 0.95)").alias("actual_p95"),
    )
    .withColumn("city", F.lit(CITY))
    .withColumn("tail_gap_p90_minus_mean", col("predicted_p90") - col("actual_mean"))
    .withColumn("tail_gap_p95_minus_mean", col("predicted_p95") - col("actual_mean"))
)

print(" NYC tail-risk baseline summary:")
display(summary)

# Optional: RMSE for sanity
rmse_eval = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="rmse")
rmse = rmse_eval.evaluate(pred)
print(f"{CITY} RMSE: {rmse}")

# 6) Save model (Spark-friendly)
print(f"Saving model to: {SAVE_MODEL_PATH}")
model.write().overwrite().save(SAVE_MODEL_PATH)

# 7) Save results table (for JIRA documentation)
(
    summary
    .withColumn("rmse", F.lit(float(rmse)))
    .withColumn("run_ts", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SAVE_RESULTS_TABLE)
)

print(f" Results saved to: {SAVE_RESULTS_TABLE}")

del model
gc.collect()