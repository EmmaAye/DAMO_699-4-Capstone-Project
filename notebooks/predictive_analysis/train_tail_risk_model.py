from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("TailRiskModel").getOrCreate()

# 1. Load Data
df = spark.table("workspace.capstone_project.nyc_model_ready")
df = df.filter(col("response_minutes").isNotNull())

# 2. Features for Tail Risk
feature_cols = ['hour', 'day_of_week', 'calls_past_30min', 'unified_alarm_level']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 3. Baseline regression model for response time.
# Tail-risk proxy evaluated via predicted P90 comparison.
gbt = GBTRegressor(labelCol="response_minutes", featuresCol="features", maxIter=20)

# 4. Training
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
pipeline_tail = Pipeline(stages=[assembler, gbt])
model_tail = pipeline_tail.fit(train_df)

# 5. Evaluation for RQ5
predictions = model_tail.transform(test_df)
evaluator_rmse = RegressionEvaluator(
    labelCol="response_minutes",
    predictionCol="prediction",
    metricName="rmse"
)
print("RMSE:", evaluator_rmse.evaluate(predictions))

evaluator_mae = RegressionEvaluator(
    labelCol="response_minutes",
    predictionCol="prediction",
    metricName="mae"
)
print("MAE:", evaluator_mae.evaluate(predictions))

evaluator_r2 = RegressionEvaluator(
    labelCol="response_minutes",
    predictionCol="prediction",
    metricName="r2"
)
print("R2:", evaluator_r2.evaluate(predictions))
# We look for high predictions (P90) vs the actual mean
predictions.createOrReplaceTempView("results")
comparison = spark.sql("""
    SELECT 
        AVG(response_minutes) as actual_mean,
        percentile_approx(response_minutes, 0.90) as actual_p90,
        percentile_approx(prediction, 0.90) as predicted_p90
    FROM results
""")
comparison.show()

# Save Model
model_tail.write().overwrite().save("/Volumes/workspace/capstone_project/models/tail_risk_model")