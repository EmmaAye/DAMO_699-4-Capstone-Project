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
from pyspark.sql.window import Window
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
import datetime

# 1. Load the NYC Operational Data
df = spark.table("workspace.capstone_project.nyc_model_ready")

# 2. Feature Selection & Engineering
feature_cols = [
    "hour", "day_of_week", "unified_alarm_level", 
    "calls_past_30min", "calls_past_60min"
]
label_col = "delay_indicator"

# Assemble features for MLlib
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
df_ml = assembler.transform(df)

# Filter out rows with null labels
df_ml = df_ml.filter(df_ml[label_col].isNotNull())

# 3. Train the Model
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol=label_col,
    predictionCol="prediction",
    probabilityCol="risk_probability_vec",
    seed=42
)
model = rf.fit(df_ml)

row = (
    df_ml.groupBy("unified_alarm_level").count()
        .orderBy(F.desc("count"))
        .first()
)
overall_alarm = row[0] if row is not None else 1

historical_stats = df_ml.groupBy("hour", "day_of_week").agg(
    F.avg("calls_past_30min").alias("calls_past_30min"),
    F.avg("calls_past_60min").alias("calls_past_60min"),
    F.lit(overall_alarm).alias("unified_alarm_level")
)

# 4. Generate "Next-Day" Forecast Slots
last_ts_row = df.select(F.max("incident_datetime")).collect()[0][0]
last_ts = last_ts_row if last_ts_row else datetime.datetime.now()

forecast_slots = []
for i in range(1, 25):
    next_time = last_ts + datetime.timedelta(hours=i)
    forecast_slots.append((
        next_time, 
        next_time.hour, 
        (next_time.weekday() + 1) % 7 + 1 # Aligning with Spark day_of_week (1=Sun)
    ))

forecast_base_df = spark.createDataFrame(
    forecast_slots, 
    ["forecast_timestamp", "hour", "day_of_week"]
)

# Join historical averages into the forecast slots
forecast_enriched_df = forecast_base_df.join(
    historical_stats, 
    ["hour", "day_of_week"], 
    "left"
).select(
    "forecast_timestamp",
    "hour",
    "day_of_week",
    "unified_alarm_level",
    "calls_past_30min",
    "calls_past_60min"
)

# Assemble features for forecast data
forecast_final_ml = assembler.transform(forecast_enriched_df)

# 5. Run Prediction (Risk Probabilities)
predictions = model.transform(forecast_final_ml)

final_forecast_df = predictions.withColumn(
    "prob_array", vector_to_array(F.col("risk_probability_vec"))
).withColumn(
    "delay_risk_probability", F.col("prob_array").getItem(1)
)

final_forecast_df = final_forecast_df.withColumn(
    "model_version", F.lit("RF_v1.0")
).withColumn(
    "forecast_generated_at", F.current_timestamp()
).withColumn(
    "last_training_timestamp", F.lit(last_ts)
).select(
    "forecast_timestamp", 
    "hour", 
    "delay_risk_probability",
    "model_version",
    "forecast_generated_at",
    "last_training_timestamp"
)

# 6. Store for Dashboard Integration
toronto_risk_forecast_output = "workspace.capstone_project.toronto_risk_forecast_output"
final_forecast_df.write.mode("overwrite").saveAsTable(toronto_risk_forecast_output)

print(f"Operational Risk Forecast complete. Table saved: {toronto_risk_forecast_output}")

# Print top 24 rows
final_forecast_df.show(24, truncate=False)

# Optional Databricks notebook display
display(final_forecast_df)
