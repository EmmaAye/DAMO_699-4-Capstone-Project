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
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

# =========================
# 1. Load NYC model-ready data
# =========================
df = spark.table("workspace.capstone_project.nyc_model_ready")

feature_cols = [
    "hour",
    "day_of_week",
    "unified_alarm_level",
    "calls_past_30min",
    "calls_past_60min"
]
label_col = "delay_indicator"

# Keep valid rows only
df = df.filter(F.col(label_col).isNotNull())

print("Class distribution:")
df.groupBy(label_col).count().orderBy(label_col).show()

# =========================
# 2. Assemble features
# =========================
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)
df_ml = assembler.transform(df).select(*feature_cols, label_col, "features")

# Train/test split
train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)

print(f"Training rows: {train_df.count()}")
print(f"Testing rows : {test_df.count()}")

# =========================
# 3. Evaluators
# =========================
roc_evaluator = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_evaluator = BinaryClassificationEvaluator(
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

# =========================
# 4. Define models
# =========================
models = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxIter=100
    ),
    "Random Forest": RandomForestClassifier(
        featuresCol="features",
        labelCol=label_col,
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=50,
        maxDepth=5,
        seed=42
    )
}

# =========================
# 5. Train, evaluate, and summarize demand effect
# =========================
all_results = []

for model_name, clf in models.items():
    print("\n" + "=" * 90)
    print(f"Training {model_name}...")
    
    model = clf.fit(train_df)
    predictions = model.transform(test_df)

    # --- Model performance ---
    auc = roc_evaluator.evaluate(predictions)
    auc_pr = pr_evaluator.evaluate(predictions)
    precision = precision_eval.evaluate(predictions)
    recall = recall_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    print(f"{model_name} AUC-ROC   : {auc:.6f}")
    print(f"{model_name} PR-AUC    : {auc_pr:.6f}")
    print(f"{model_name} Precision : {precision:.6f}")
    print(f"{model_name} Recall    : {recall:.6f}")
    print(f"{model_name} F1 Score  : {f1:.6f}")

    print(f"\nConfusion Matrix for {model_name}:")
    predictions.groupBy(label_col, "prediction") \
        .count() \
        .orderBy(label_col, "prediction") \
        .show()

    # --- Extract positive-class probability ---
    predictions = predictions.withColumn(
        "prob_array",
        vector_to_array(F.col("probability"))
    ).withColumn(
        "delay_risk_probability",
        F.col("prob_array").getItem(1)
    )

    # --- Create demand groups using calls_past_60min ---
    q25, q75 = test_df.approxQuantile("calls_past_60min", [0.25, 0.75], 0.01)

    predictions = predictions.withColumn(
        "demand_group",
        F.when(F.col("calls_past_60min") <= q25, F.lit("Low Demand"))
         .when(F.col("calls_past_60min") <= q75, F.lit("Medium Demand"))
         .otherwise(F.lit("High Demand"))
    )

    # --- Summarize demand effect ---
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

    print(f"\nDemand-group summary for {model_name}:")
    rq2_summary.show(truncate=False)

    # Save summary results in Python list for final comparison print
    all_results.append((model_name, auc, auc_pr, precision, recall, f1))

# =========================
# 6. Final model comparison table
# =========================
print("\n" + "=" * 90)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 90)
print(f"{'Model':<22} {'AUC-ROC':<12} {'PR-AUC':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-" * 90)

for row in all_results:
    model_name, auc, auc_pr, precision, recall, f1 = row
    print(f"{model_name:<22} {auc:<12.6f} {auc_pr:<12.6f} {precision:<12.6f} {recall:<12.6f} {f1:<12.6f}")
