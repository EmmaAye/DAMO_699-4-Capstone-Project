# 1: Setup and Imports
import gc
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# 2: Data Loading & Preprocessing
print("Loading NYC dataset...")
# Using the NYC table
df = spark.table("workspace.capstone_project.nyc_model_ready")
df = df.filter(col("delay_indicator").isNotNull())
df.groupBy("delay_indicator").count().show()

# Preprocessing with Feature Hashing to prevent the 1GB overflow error
categorical_cols = ['incident_category', 'season', 'unified_call_source', 'location_area']
hasher = FeatureHasher(inputCols=categorical_cols, outputCol="categorical_features", numFeatures=512)

# Feature Assembly
numeric_cols = ['hour', 'day_of_week', 'month', 'year', 'unified_alarm_level', 
                'calls_past_30min', 'calls_past_60min']
assembler = VectorAssembler(inputCols=numeric_cols + ["categorical_features"], outputCol="features")

# 3: Training with Downsampling
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="delay_indicator",
    numTrees=50,
    maxDepth=5,
    seed=42
)

pipeline = Pipeline(stages=[hasher, assembler, rf])

print("Starting fit on NYC sample...")
model = pipeline.fit(train_df)
print("Training complete!")

# 4: Evaluation and Saving
# Generate predictions
predictions = model.transform(test_df)

# AUC-ROC & PR-AUC (threshold-independent)
roc_evaluator = BinaryClassificationEvaluator(
    labelCol="delay_indicator",
    metricName="areaUnderROC"
)
auc = roc_evaluator.evaluate(predictions)
print(f"NYC Model AUC-ROC: {auc}")

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="delay_indicator",
    metricName="areaUnderPR"
)
auc_pr = pr_evaluator.evaluate(predictions)
print(f"NYC Model PR-AUC: {auc_pr}")

# Confusion Matrix (label-based)
print("Confusion Matrix (Actual vs Predicted):")
predictions.groupBy("delay_indicator", "prediction") \
    .count() \
    .orderBy("delay_indicator", "prediction") \
    .show()

# Precision / Recall / F1 (label-based)
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

precision = precision_eval.evaluate(predictions)
recall = recall_eval.evaluate(predictions)
f1 = f1_eval.evaluate(predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save model
save_path = "/Volumes/workspace/capstone_project/models/delay_classifier_nyc"
print(f"Saving model to: {save_path}")
model.write().overwrite().save(save_path)

# Clean up memory
del model
gc.collect()
