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
import os
import datetime
import pandas as pd

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.functions import vector_to_array

print("Starting RQ4 NYC forecast pipeline...")

# ============================================================
# 1. PATHS
# ============================================================
base_output_dir = os.path.abspath("../../../output")
tables_dir = os.path.join(base_output_dir, "tables")

# ============================================================
# 2. LOAD BEST MODEL FROM COMPARISON
# ============================================================
comparison_df = pd.read_csv(os.path.join(tables_dir, "rq4_nyc_model_comparison.csv"))
display(comparison_df.round(3))

best_model_name = comparison_df.iloc[0]["Model"]
print("Best trainable model selected for forecasting:", best_model_name)

if best_model_name == "Naive Majority Class":
    raise ValueError("Naive baseline should not be selected for forecasting.")

# ============================================================
# 3. LOAD NYC DATA
# ============================================================
label_col = "delay_indicator"

df = spark.table("workspace.capstone_project.nyc_model_ready")
df = df.filter(F.col(label_col).isNotNull())

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
import os
import datetime
import pandas as pd

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.functions import vector_to_array

print("Starting RQ4 NYC forecast pipeline...")

# ============================================================
# 1. PATHS
# ============================================================
base_output_dir = os.path.abspath("../../../output")
tables_dir = os.path.join(base_output_dir, "tables")

# ============================================================
# 2. LOAD BEST MODEL FROM COMPARISON
# ============================================================
comparison_df = pd.read_csv(os.path.join(tables_dir, "rq4_nyc_model_comparison.csv"))
display(comparison_df.round(3))

best_model_name = comparison_df.iloc[0]["Model"]
print("Best trainable model selected for forecasting:", best_model_name)

if best_model_name == "Naive Majority Class":
    raise ValueError("Naive baseline should not be selected for forecasting.")

# ============================================================
# 3. LOAD NYC DATA
# ============================================================
label_col = "delay_indicator"

df = spark.table("workspace.capstone_project.nyc_model_ready")
df = df.filter(F.col(label_col).isNotNull())

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

required_cols = ["incident_datetime", "response_minutes"] + categorical_cols + numeric_cols + [label_col]


base_df = (
    df.select(*required_cols)
    .dropna(subset=numeric_cols + [label_col])
)

# ============================================================
# 4. FEATURE PIPELINE
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
        numTrees=50,
        maxDepth=5,
        seed=42
    )
else:
    best_classifier = GBTClassifier(
        featuresCol="features",
        labelCol=label_col,
        maxIter=20,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )

final_pipeline = Pipeline(stages=[hasher, assembler, best_classifier])
final_model = final_pipeline.fit(base_df)

print(f"Final {best_model_name} model trained on full NYC data.")

# ============================================================
# 5. BUILD FUTURE INPUTS
# ============================================================
last_ts_row = df.select(F.max("incident_datetime")).collect()[0][0]
last_ts = last_ts_row if last_ts_row else datetime.datetime.now()

print("Last NYC incident timestamp:", last_ts)

# most frequent alarm level
alarm_row = (
    base_df.groupBy("unified_alarm_level")
    .count()
    .orderBy(F.desc("count"))
    .first()
)
overall_alarm = alarm_row[0] if alarm_row is not None else 1

# most frequent categorical values
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
    F.avg("calls_past_60min").alias("calls_past_60min"),
    F.avg("response_minutes").alias("response_minutes")
)

forecast_slots = []
for i in range(1, 25):
    next_time = last_ts + datetime.timedelta(hours=i)
    spark_day_of_week = (next_time.weekday() + 1) % 7 + 1

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
    "calls_past_60min",
    "response_minutes" 
)

forecast_enriched_df = forecast_enriched_df.fillna({
    "calls_past_30min": 0.0,
    "calls_past_60min": 0.0,
    "response_minutes": 0.0,
    "unified_alarm_level": 1,
    "incident_category": mode_values["incident_category"],
    "season": mode_values["season"],
    "unified_call_source": mode_values["unified_call_source"],
    "location_area": mode_values["location_area"]
})

print("Forecast input preview:")
forecast_enriched_df.show(24, truncate=False)

# ============================================================
# 6. GENERATE FORECAST
# ============================================================
forecast_pred = final_model.transform(forecast_enriched_df)

forecast_output = (
    forecast_pred
    .withColumn("prob_array", vector_to_array(F.col("probability")))
    .withColumn("delay_risk_probability", F.col("prob_array").getItem(1))
    .withColumn("model_version", F.lit(best_model_name.lower().replace(" ", "_") + "_v1.0"))
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
        "response_minutes",
        "delay_risk_probability",
        "model_version",
        "forecast_generated_at",
        "last_training_timestamp"
    )
    .orderBy("forecast_timestamp")
)

print("\nNext 24-hour NYC delay risk forecast:")
forecast_output.show(24, truncate=False)

display(forecast_output)

# ============================================================
# 7. SAVE FORECAST OUTPUT
# ============================================================
output_table = "workspace.capstone_project.nyc_risk_forecast_output"

forecast_output.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(output_table)

print("\nForecast saved to table:", output_table)
print("RQ4 NYC forecast complete.")

base_df = (
    df.select(*required_cols)
    .dropna(subset=numeric_cols + [label_col])
)

# ============================================================
# 4. FEATURE PIPELINE
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
        numTrees=50,
        maxDepth=5,
        seed=42
    )
else:
    best_classifier = GBTClassifier(
        featuresCol="features",
        labelCol=label_col,
        maxIter=20,
        maxDepth=5,
        stepSize=0.1,
        seed=42
    )

final_pipeline = Pipeline(stages=[hasher, assembler, best_classifier])
final_model = final_pipeline.fit(base_df)

print(f"Final {best_model_name} model trained on full NYC data.")

# ============================================================
# 5. BUILD FUTURE INPUTS
# ============================================================
last_ts_row = df.select(F.max("incident_datetime")).collect()[0][0]
last_ts = last_ts_row if last_ts_row else datetime.datetime.now()

print("Last NYC incident timestamp:", last_ts)

# most frequent alarm level
alarm_row = (
    base_df.groupBy("unified_alarm_level")
    .count()
    .orderBy(F.desc("count"))
    .first()
)
overall_alarm = alarm_row[0] if alarm_row is not None else 1

# most frequent categorical values
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
# 6. GENERATE FORECAST
# ============================================================
forecast_pred = final_model.transform(forecast_enriched_df)

forecast_output = (
    forecast_pred
    .withColumn("prob_array", vector_to_array(F.col("probability")))
    .withColumn("delay_risk_probability", F.col("prob_array").getItem(1))
    .withColumn("model_version", F.lit(best_model_name.lower().replace(" ", "_") + "_v1.0"))
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

print("\nNext 24-hour NYC delay risk forecast:")
forecast_output.show(24, truncate=False)

display(forecast_output)

# ============================================================
# 7. SAVE FORECAST OUTPUT
# ============================================================
output_table = "workspace.capstone_project.nyc_risk_forecast_output"

forecast_output.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(output_table)

print("\nForecast saved to table:", output_table)
print("RQ4 NYC forecast complete.")
