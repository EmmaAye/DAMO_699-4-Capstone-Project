from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
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
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_ml = assembler.transform(df)
# filter out rows with null labels
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

# Compute historical averages for missing columns
historical_stats = df_ml.groupBy("hour", "day_of_week").agg(
    F.avg("calls_past_30min").alias("calls_past_30min"),
    F.avg("calls_past_60min").alias("calls_past_60min"),
    F.first("unified_alarm_level").alias("unified_alarm_level")
)

# 4. Generate "Next-Day" Forecast Slots
last_ts = df.select(F.max("incident_datetime")).collect()[0][0]
forecast_data = []
for i in range(1, 25):
    next_time = last_ts + datetime.timedelta(hours=i)
    forecast_data.append((
        next_time, 
        next_time.hour, 
        (next_time.weekday() + 1) % 7 + 1
    ))
forecast_base_df = spark.createDataFrame(
    forecast_data, 
    ["forecast_timestamp", "hour", "day_of_week"]
)

# Join historical averages to supply required columns
forecast_base_df = forecast_base_df.join(historical_stats, ["hour", "day_of_week"], "left")

# Assemble features for forecast data
forecast_base_ml = assembler.transform(forecast_base_df)

# 5. Run Prediction (Risk Probabilities)
predictions = model.transform(forecast_base_ml)

# Extract probability of 'Delay' (class 1)
extract_prob_udf = F.udf(lambda v: float(v[1]))
final_forecast_df = predictions.withColumn(
    "delay_risk_probability", 
    extract_prob_udf("risk_probability_vec")
).select(
    "forecast_timestamp", 
    "hour", 
    "delay_risk_probability"
)

# 6. Store for Dashboard Integration
nyc_risk_forecast_output = "workspace.capstone_project.nyc_risk_forecast_output"
final_forecast_df.write.mode("overwrite").saveAsTable(nyc_risk_forecast_output)

print(f"Operational Risk Forecast complete. Table saved: {nyc_risk_forecast_output}")