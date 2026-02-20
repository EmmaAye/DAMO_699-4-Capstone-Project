%python
#1: Setup and Imports
import gc
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

# 2: Data Loading & Preprocessing
print("Loading Toronto dataset...")
# Using the toronto table
df = spark.table("workspace.capstone_project.toronto_model_ready")
df = df.filter(col("delay_indicator").isNotNull())

# Preprocessing with Feature Hashing to prevent the 1GB overflow error
categorical_cols = ['incident_category', 'season', 'unified_call_source', 'location_area']
hasher = FeatureHasher(inputCols=categorical_cols, outputCol="categorical_features", numFeatures=512)

# Feature Assembly
numeric_cols = ['hour', 'day_of_week', 'month', 'year', 'unified_alarm_level', 
                'calls_past_30min', 'calls_past_60min']
assembler = VectorAssembler(inputCols=numeric_cols + ["categorical_features"], outputCol="features")

# 3: Training with Downsampling
# We start with 50% to ensure the session stays alive
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df_sample = train_df.sample(withReplacement=False, fraction=0.5, seed=42)

rf = RandomForestClassifier(
    featuresCol="features", 
    labelCol="delay_indicator", 
    numTrees=50, 
    maxDepth=5
)

pipeline = Pipeline(stages=[hasher, assembler, rf])

print("Starting fit on Toronto sample...")
model = pipeline.fit(train_df_sample)
print("Training complete!")

# 4: Evaluation and Saving
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="delay_indicator", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Toronto Model AUC: {auc}")

save_path = "/Volumes/workspace/capstone_project/models/delay_classifier_trn"
print(f"Saving model to: {save_path}")
model.write().overwrite().save(save_path)

# Clean up memory
del model
gc.collect()
