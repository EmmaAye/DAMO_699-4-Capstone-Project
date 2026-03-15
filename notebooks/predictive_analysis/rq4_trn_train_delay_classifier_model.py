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
# 1: Setup and Imports
import gc
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# 2: Data Loading & Preprocessing
print("Loading Toronto dataset...")
df = spark.table("workspace.capstone_project.toronto_model_ready")
df = df.filter(col("delay_indicator").isNotNull())
df.groupBy("delay_indicator").count().show()

# Categorical columns
categorical_cols = ['incident_category', 'season', 'unified_call_source', 'location_area']

# Numeric columns
numeric_cols = [
    'hour', 'day_of_week', 'month', 'year',
    'unified_alarm_level', 'calls_past_30min', 'calls_past_60min'
]

# Feature hashing
hasher = FeatureHasher(
    inputCols=categorical_cols,
    outputCol="categorical_features",
    numFeatures=512
)

# Feature assembly
assembler = VectorAssembler(
    inputCols=numeric_cols + ["categorical_features"],
    outputCol="features",
    handleInvalid="keep"
)

# 3: Train-Test Split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 4: Evaluators
roc_evaluator = BinaryClassificationEvaluator(
    labelCol="delay_indicator",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="delay_indicator",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

precision_eval = MulticlassClassificationEvaluator(
    labelCol="delay_indicator",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_eval = MulticlassClassificationEvaluator(
    labelCol="delay_indicator",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="delay_indicator",
    predictionCol="prediction",
    metricName="f1"
)

# 5: Model Definitions (Baseline + Proposed)
models = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features",
        labelCol="delay_indicator",
        maxIter=100,
        regParam=0.0,
        elasticNetParam=0.0
    ),
    "Random Forest": RandomForestClassifier(
    featuresCol="features",
    labelCol="delay_indicator",
    numTrees=150,       # increase trees
    maxDepth=10,        # allow deeper splits
    minInstancesPerNode=5,
    seed=42
),
    "GBT Classifier": GBTClassifier(
    featuresCol="features",
    labelCol="delay_indicator",
    maxIter=80,              # more boosting rounds
    maxDepth=7,              # allow interactions
    stepSize=0.03,           # smaller learning rate
    minInstancesPerNode=10,
    subsamplingRate=0.8,
    seed=42
)

}

# 6: Train, Evaluate, and Save Each Model
results = []

for model_name, classifier in models.items():
    print("\n" + "=" * 80)
    print(f"Training {model_name}...")
    
    pipeline = Pipeline(stages=[hasher, assembler, classifier])
    model = pipeline.fit(train_df)
    print(f"{model_name} training complete!")

    predictions = model.transform(test_df)

    # Metrics
    auc = roc_evaluator.evaluate(predictions)
    auc_pr = pr_evaluator.evaluate(predictions)
    precision = precision_eval.evaluate(predictions)
    recall = recall_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    print(f"{model_name} AUC-ROC: {auc}")
    print(f"{model_name} PR-AUC: {auc_pr}")
    print(f"{model_name} Precision: {precision}")
    print(f"{model_name} Recall: {recall}")
    print(f"{model_name} F1 Score: {f1}")

    # Confusion Matrix
    print(f"Confusion Matrix for {model_name} (Actual vs Predicted):")
    predictions.groupBy("delay_indicator", "prediction") \
        .count() \
        .orderBy("delay_indicator", "prediction") \
        .show()

    # Save model
    safe_name = model_name.lower().replace(" ", "_")
    save_path = f"/Volumes/workspace/capstone_project/models/delay_classifier_toronto_{safe_name}"
    print(f"Saving {model_name} model to: {save_path}")
    model.write().overwrite().save(save_path)

    results.append((model_name, auc, auc_pr, precision, recall, f1))

    # cleanup
    del model
    del predictions
    del pipeline
    gc.collect()

# 7: Final Summary Table
print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE SUMMARY - TORONTO")
print("=" * 80)

print(f"{'Model':<22} {'AUC-ROC':<12} {'PR-AUC':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-" * 82)

for row in results:
    model_name, auc, auc_pr, precision, recall, f1 = row
    print(f"{model_name:<22} {auc:<12.6f} {auc_pr:<12.6f} {precision:<12.6f} {recall:<12.6f} {f1:<12.6f}")
