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
# =========================================================
# SERVERLESS-SAFE SPARK ML CODE
# RUN ONLY ONE MODEL PER SESSION
# Change CITY and MODEL_TO_RUN before each execution
# =========================================================

import gc
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ---------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------
CITY = "Toronto"          # "Toronto" or "NYC"
MODEL_TO_RUN = "baseline" # "baseline", "lr", "rf"
SEED = 42
LABEL = "delay_indicator"

# Save one result row per run here
RESULTS_PATH = "/Workspace/Users/pratiksha.pawar18@gmail.com/DAMO_699-4-Capstone-Project/output/tables/lr_rf_baseline_single_runs"

# ---------------------------------------------------------
# OPTIONAL CLEANUP AT START
# ---------------------------------------------------------
for v in [
    "df", "train_df", "test_df", "pred_df", "pipeline", "model",
    "lr", "rf", "baseline_pred", "results_df"
]:
    if v in globals():
        del globals()[v]
gc.collect()

try:
    spark.catalog.clearCache()
except:
    pass

# ---------------------------------------------------------
# LOAD CITY DATA
# ---------------------------------------------------------
table_name = (
    "workspace.capstone_project.toronto_model_ready"
    if CITY == "Toronto"
    else "workspace.capstone_project.nyc_model_ready"
)

df = spark.table(table_name)

# Keep only required columns
candidate_cols = [
    LABEL,
    "hour", "day_of_week", "month", "year",
    "unified_alarm_level", "calls_past_30min", "calls_past_60min",
    "season", "incident_category", "unified_call_source", "location_area"
]

keep_cols = [c for c in candidate_cols if c in df.columns]
df = df.select(*keep_cols).filter(col(LABEL).isNotNull())

print(f"City: {CITY}")
print(f"Model: {MODEL_TO_RUN}")
print(f"Rows: {df.count()}")
df.groupBy(LABEL).count().orderBy(LABEL).show()

# ---------------------------------------------------------
# SPLIT
# ---------------------------------------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)

print("Train count:", train_df.count())
print("Test count :", test_df.count())

# ---------------------------------------------------------
# EVALUATION HELPER
# ---------------------------------------------------------
def evaluate_predictions(pred_df, city_name, model_name):
    total = pred_df.count()

    accuracy = (
        pred_df.filter(col(LABEL) == col("prediction")).count() / total
        if total > 0 else None
    )

    precision = MulticlassClassificationEvaluator(
        labelCol=LABEL,
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(pred_df)

    recall = MulticlassClassificationEvaluator(
        labelCol=LABEL,
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(pred_df)

    f1 = MulticlassClassificationEvaluator(
        labelCol=LABEL,
        predictionCol="prediction",
        metricName="f1"
    ).evaluate(pred_df)

    if "rawPrediction" in pred_df.columns:
        auc = BinaryClassificationEvaluator(
            labelCol=LABEL,
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        ).evaluate(pred_df)
    else:
        auc = 0.5

    print("=====================================================")
    print(f"{city_name} - {model_name}")
    print("=====================================================")
    print(f"Accuracy : {accuracy:.6f}")
    print(f"AUC-ROC  : {auc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1 Score : {f1:.6f}")

    pred_df.groupBy(LABEL, "prediction").count().orderBy(LABEL, "prediction").show()

    return [(city_name, model_name, float(accuracy), float(auc), float(precision), float(recall), float(f1))]

# ---------------------------------------------------------
# BASELINE
# ---------------------------------------------------------
if MODEL_TO_RUN == "baseline":
    majority = (
        train_df.groupBy(LABEL)
        .count()
        .orderBy(F.desc("count"))
        .first()[LABEL]
    )

    pred_df = test_df.withColumn("prediction", F.lit(float(majority)))
    result_rows = evaluate_predictions(pred_df, CITY, "Naive Majority Baseline")

# ---------------------------------------------------------
# PIPELINE HELPERS FOR LR / RF
# ---------------------------------------------------------
else:
    numeric_cols = [c for c in [
        "hour", "day_of_week", "month", "year",
        "unified_alarm_level", "calls_past_30min", "calls_past_60min"
    ] if c in df.columns]

    categorical_cols = [c for c in [
        "season", "incident_category", "unified_call_source", "location_area"
    ] if c in df.columns]

    stages = []
    encoded_cols = []

    for c in categorical_cols:
        idx = f"{c}_idx"
        ohe = f"{c}_ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx], outputCols=[ohe]))
        encoded_cols.append(ohe)

    assembler = VectorAssembler(
        inputCols=numeric_cols + encoded_cols,
        outputCol="assembled_features",
        handleInvalid="skip"
    )

    scaler = StandardScaler(
        inputCol="assembled_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    if MODEL_TO_RUN == "lr":
        estimator = LogisticRegression(
            labelCol=LABEL,
            featuresCol="features",
            predictionCol="prediction",
            rawPredictionCol="rawPrediction",
            probabilityCol="probability",
            maxIter=30,
            regParam=0.01,
            elasticNetParam=0.0
        )
        model_name = "Logistic Regression"

    elif MODEL_TO_RUN == "rf":
        estimator = RandomForestClassifier(
            labelCol=LABEL,
            featuresCol="features",
            predictionCol="prediction",
            rawPredictionCol="rawPrediction",
            probabilityCol="probability",
            numTrees=40,
            maxDepth=6,
            seed=SEED
        )
        model_name = "Random Forest"

    else:
        raise ValueError("MODEL_TO_RUN must be one of: baseline, lr, rf")

    pipeline = Pipeline(stages=stages + [assembler, scaler, estimator])
    model = pipeline.fit(train_df)

    pred_df = model.transform(test_df).select(LABEL, "prediction", "rawPrediction")
    result_rows = evaluate_predictions(pred_df, CITY, model_name)

# ---------------------------------------------------------
# SAVE THIS SINGLE RUN RESULT
# ---------------------------------------------------------
results_df = spark.createDataFrame(
    result_rows,
    ["City", "Model", "Accuracy", "AUC_ROC", "Precision", "Recall", "F1_Score"]
)

results_df.write.mode("append").format("delta").save(RESULTS_PATH)

print("Saved result row to:")
print(RESULTS_PATH)

# ---------------------------------------------------------
# FINAL CLEANUP
# ---------------------------------------------------------
for v in [
    "pred_df", "model", "pipeline", "estimator", "assembler", "scaler",
    "train_df", "test_df", "df", "results_df"
]:
    if v in globals():
        del globals()[v]

gc.collect()

try:
    spark.catalog.clearCache()
except:
    pass

print("Run completed. Reattach / restart session before the next model.")

# %%
from pyspark.sql import functions as F

RESULTS_PATH = "/Workspace/Users/pratiksha.pawar18@gmail.com/DAMO_699-4-Capstone-Project/output/tables/lr_rf_baseline_single_runs"

results_df = spark.read.format("delta").load(RESULTS_PATH).dropDuplicates()

results_matrix = (
    results_df
    .withColumn("Accuracy", F.round("Accuracy", 3))
    .withColumn("AUC_ROC", F.round("AUC_ROC", 3))
    .withColumn("Precision", F.round("Precision", 3))
    .withColumn("Recall", F.round("Recall", 3))
    .withColumn("F1_Score", F.round("F1_Score", 3))
    .orderBy("City", "Model")
)

print("===== FULL COMPARISON MATRIX =====")
results_matrix.show(truncate=False)

print("===== TORONTO =====")
results_matrix.filter(F.col("City") == "Toronto").show(truncate=False)

print("===== NYC =====")
results_matrix.filter(F.col("City") == "NYC").show(truncate=False)
