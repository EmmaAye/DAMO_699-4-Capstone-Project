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
# SERVERLESS-SAFE CITY RUN
# Baseline vs Logistic Regression
# Run once for Toronto, then again for NYC in a fresh session
# =========================================================

import gc
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ---------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------
CITY = "Toronto"   # Change to "NYC" in fresh session
SEED = 42
LABEL = "delay_indicator"
SAVE_TABLE = "workspace.capstone_project.baseline_lr_city_results"

# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------
for v in [
    "df", "train_df", "test_df", "baseline_pred", "pred_df",
    "pipeline", "model", "assembler", "results_df", "results_matrix"
]:
    if v in globals():
        del globals()[v]
gc.collect()

try:
    spark.catalog.clearCache()
except:
    pass

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
table_name = (
    "workspace.capstone_project.toronto_model_ready"
    if CITY == "Toronto"
    else "workspace.capstone_project.nyc_model_ready"
)

candidate_cols = [
    LABEL,
    "hour", "day_of_week", "month", "year",
    "unified_alarm_level", "calls_past_30min", "calls_past_60min",
    "season", "incident_category", "unified_call_source", "location_area"
]

df = spark.table(table_name)
keep_cols = [c for c in candidate_cols if c in df.columns]
df = df.select(*keep_cols).filter(col(LABEL).isNotNull())

print(f"City: {CITY}")
print(f"Rows: {df.count()}")
df.groupBy(LABEL).count().orderBy(LABEL).show()

# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)

print("Train count:", train_df.count())
print("Test count :", test_df.count())

# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate_predictions(pred_df, city_name, model_name):
    total = pred_df.count()

    accuracy = pred_df.filter(col(LABEL) == col("prediction")).count() / total

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

    return (city_name, model_name, float(accuracy), float(auc), float(precision), float(recall), float(f1))

# ---------------------------------------------------------
# 1. BASELINE
# ---------------------------------------------------------
majority = (
    train_df.groupBy(LABEL)
    .count()
    .orderBy(F.desc("count"))
    .first()[LABEL]
)

baseline_pred = test_df.withColumn("prediction", F.lit(float(majority)))
baseline_result = evaluate_predictions(baseline_pred, CITY, "Naive Majority Baseline")

del baseline_pred
gc.collect()

# ---------------------------------------------------------
# 2. LOGISTIC REGRESSION
# ---------------------------------------------------------
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
    outputCol="features",
    handleInvalid="skip"
)

lr = LogisticRegression(
    labelCol=LABEL,
    featuresCol="features",
    predictionCol="prediction",
    rawPredictionCol="rawPrediction",
    probabilityCol="probability",
    maxIter=20,
    regParam=0.01,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=stages + [assembler, lr])
model = pipeline.fit(train_df)

pred_df = model.transform(test_df).select(LABEL, "prediction", "rawPrediction")
lr_result = evaluate_predictions(pred_df, CITY, "Logistic Regression")

# ---------------------------------------------------------
# 3. SAVE CITY RESULTS TO DELTA TABLE
# ---------------------------------------------------------
results_df = spark.createDataFrame(
    [baseline_result, lr_result],
    ["City", "Model", "Accuracy", "AUC_ROC", "Precision", "Recall", "F1_Score"]
)

results_df.write.mode("append").format("delta").saveAsTable(SAVE_TABLE)

print("Saved city results to table:")
print(SAVE_TABLE)

# ---------------------------------------------------------
# 4. OPTIONAL DISPLAY
# ---------------------------------------------------------
results_matrix = (
    results_df
    .withColumn("Accuracy", F.round("Accuracy", 3))
    .withColumn("AUC_ROC", F.round("AUC_ROC", 3))
    .withColumn("Precision", F.round("Precision", 3))
    .withColumn("Recall", F.round("Recall", 3))
    .withColumn("F1_Score", F.round("F1_Score", 3))
)

print("=====================================================")
print(f"{CITY} - Baseline vs Logistic Regression")
print("=====================================================")
results_matrix.show(truncate=False)

# ---------------------------------------------------------
# 5. CLEANUP
# ---------------------------------------------------------
for v in [
    "pred_df", "model", "pipeline", "assembler", "lr",
    "train_df", "test_df", "df", "results_df", "results_matrix"
]:
    if v in globals():
        del globals()[v]

gc.collect()

try:
    spark.catalog.clearCache()
except:
    pass

print("Done. Restart/reattach session before running the next city.")

# %%
# =========================================================
# ONE-CELL SERVERLESS-SAFE COMPARISON
# Toronto vs NYC
# Baseline vs Logistic Regression
# No file writes, no /Workspace path usage
# =========================================================

import gc
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

SEED = 42
LABEL = "delay_indicator"

# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------
for v in [
    "df", "train_df", "test_df", "baseline_pred", "pred_df",
    "pipeline", "model", "assembler", "results_df", "results_matrix",
    "toronto_results", "nyc_results"
]:
    if v in globals():
        del globals()[v]
gc.collect()

try:
    spark.catalog.clearCache()
except:
    pass

# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate_predictions(pred_df, city_name, model_name):
    total = pred_df.count()

    accuracy = pred_df.filter(col(LABEL) == col("prediction")).count() / total

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

    return (city_name, model_name, float(accuracy), float(auc), float(precision), float(recall), float(f1))

# ---------------------------------------------------------
# RUN FUNCTION FOR ONE CITY
# ---------------------------------------------------------
def run_city(city_name, table_name):
    print("\n")
    print("#####################################################")
    print(f"RUNNING CITY: {city_name}")
    print("#####################################################")

    candidate_cols = [
        LABEL,
        "hour", "day_of_week", "month", "year",
        "unified_alarm_level", "calls_past_30min", "calls_past_60min",
        "season", "incident_category", "unified_call_source", "location_area"
    ]

    df = spark.table(table_name)
    keep_cols = [c for c in candidate_cols if c in df.columns]
    df = df.select(*keep_cols).filter(col(LABEL).isNotNull())

    print(f"{city_name} rows: {df.count()}")
    df.groupBy(LABEL).count().orderBy(LABEL).show()

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)

    print(f"{city_name} train count: {train_df.count()}")
    print(f"{city_name} test count : {test_df.count()}")

    # -----------------------------------------------------
    # 1. Baseline
    # -----------------------------------------------------
    majority = (
        train_df.groupBy(LABEL)
        .count()
        .orderBy(F.desc("count"))
        .first()[LABEL]
    )

    baseline_pred = test_df.withColumn("prediction", F.lit(float(majority)))
    baseline_result = evaluate_predictions(
        baseline_pred,
        city_name,
        "Naive Majority Baseline"
    )

    del baseline_pred
    gc.collect()

    # -----------------------------------------------------
    # 2. Logistic Regression
    # -----------------------------------------------------
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
        outputCol="features",
        handleInvalid="skip"
    )

    lr = LogisticRegression(
        labelCol=LABEL,
        featuresCol="features",
        predictionCol="prediction",
        rawPredictionCol="rawPrediction",
        probabilityCol="probability",
        maxIter=20,
        regParam=0.01,
        elasticNetParam=0.0
    )

    pipeline = Pipeline(stages=stages + [assembler, lr])
    model = pipeline.fit(train_df)

    pred_df = model.transform(test_df).select(LABEL, "prediction", "rawPrediction")
    lr_result = evaluate_predictions(
        pred_df,
        city_name,
        "Logistic Regression"
    )

    # Explicit cleanup to reduce Spark Connect cache pressure
    del pred_df
    del model
    del pipeline
    del assembler
    del lr
    del stages
    del encoded_cols
    del train_df
    del test_df
    del df
    gc.collect()

    try:
        spark.catalog.clearCache()
    except:
        pass

    return [baseline_result, lr_result]

# ---------------------------------------------------------
# RUN BOTH CITIES
# ---------------------------------------------------------
toronto_results = run_city(
    "Toronto",
    "workspace.capstone_project.toronto_model_ready"
)

# Important cleanup between cities
gc.collect()
try:
    spark.catalog.clearCache()
except:
    pass

nyc_results = run_city(
    "NYC",
    "workspace.capstone_project.nyc_model_ready"
)

# ---------------------------------------------------------
# FINAL COMPARISON TABLE
# ---------------------------------------------------------
all_results = toronto_results + nyc_results

results_df = spark.createDataFrame(
    all_results,
    ["City", "Model", "Accuracy", "AUC_ROC", "Precision", "Recall", "F1_Score"]
)

results_matrix = (
    results_df
    .withColumn("Accuracy", F.round("Accuracy", 3))
    .withColumn("AUC_ROC", F.round("AUC_ROC", 3))
    .withColumn("Precision", F.round("Precision", 3))
    .withColumn("Recall", F.round("Recall", 3))
    .withColumn("F1_Score", F.round("F1_Score", 3))
)

print("\n=====================================================")
print("FINAL COMPARISON: TORONTO vs NYC")
print("=====================================================")
results_matrix.orderBy("City", "Model").show(truncate=False)

print("\n=====================================================")
print("TORONTO")
print("=====================================================")
results_matrix.filter(col("City") == "Toronto").orderBy("Model").show(truncate=False)

print("\n=====================================================")
print("NYC")
print("=====================================================")
results_matrix.filter(col("City") == "NYC").orderBy("Model").show(truncate=False)

print("Done.")

# %%
from pyspark.sql import functions as F

SAVE_PATH = "/Workspace/Users/pratiksha.pawar18@gmail.com/DAMO_699-4-Capstone-Project/output/tables/baseline_lr_city_results"

results_df = (
    spark.read.format("delta").load(SAVE_PATH)
    .dropDuplicates(["City", "Model"])
)

results_matrix = (
    results_df
    .withColumn("Accuracy", F.round("Accuracy", 3))
    .withColumn("AUC_ROC", F.round("AUC_ROC", 3))
    .withColumn("Precision", F.round("Precision", 3))
    .withColumn("Recall", F.round("Recall", 3))
    .withColumn("F1_Score", F.round("F1_Score", 3))
)

print("=====================================================")
print("Baseline vs Logistic Regression - Toronto and NYC")
print("=====================================================")
results_matrix.orderBy("City", "Model").show(truncate=False)

print("=====================================================")
print("Toronto")
print("=====================================================")
results_matrix.filter(F.col("City") == "Toronto").orderBy("Model").show(truncate=False)

print("=====================================================")
print("NYC")
print("=====================================================")
results_matrix.filter(F.col("City") == "NYC").orderBy("Model").show(truncate=False)
