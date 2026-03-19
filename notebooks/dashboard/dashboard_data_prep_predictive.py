import pandas as pd
import os
import pickle
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

PATH_CPH_NYC = "/Workspace/Users/thanda.aye03@gmail.com/DAMO_699-4-Capstone-Project/output/models/cph_NYC.pkl"
PATH_CPH_TORONTO = "/Workspace/Users/thanda.aye03@gmail.com/DAMO_699-4-Capstone-Project/output/models/cph_Toronto.pkl"

def get_hazard_ratios(path, city_name):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        df = pd.DataFrame({
            'Feature': model.params_.index,
            'Hazard_Ratio': model.hazard_ratios_,
            'City': city_name
        })
        return df
    except Exception as e:
        print(f"Failed to load {city_name} model: {e}")
        return pd.DataFrame()

# 1. HARVEST HAZARD RATIOS (Feature Importance)
nyc_hr = get_hazard_ratios(PATH_CPH_NYC, "NYC")
toronto_hr = get_hazard_ratios(PATH_CPH_TORONTO, "Toronto")

all_hazard_ratios = pd.concat([nyc_hr, toronto_hr])
if not all_hazard_ratios.empty:
    spark.createDataFrame(all_hazard_ratios).write.mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("capstone_project.dashboard_hazard_ratios")
    print("Combined Hazard Ratios saved.")

# 2. GENERATE PREDICTIVE METRICS AND DISTRIBUTIONS
base_output_dir = os.path.abspath("../../output")
tables_dir = os.path.join(base_output_dir, "tables")

try:
    # Load Data
    nyc_pd = pd.read_csv(os.path.join(tables_dir, "rq4_nyc_forecast_best_model.csv"))
    nyc_forecast = spark.createDataFrame(nyc_pd).withColumn("City", F.lit("NYC"))
    
    toronto_pd = pd.read_csv(os.path.join(tables_dir, "rq4_toronto_forecast_best_model.csv"))
    toronto_forecast = spark.createDataFrame(toronto_pd).withColumn("City", F.lit("Toronto"))

    def align_schema(df):
        numeric_targets = ["delay_risk_probability", "response_minutes"]
        for col_name in df.columns:
            if col_name in numeric_targets:
                df = df.withColumn(col_name, F.col(col_name).cast("double"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        return df

    nyc_forecast = align_schema(nyc_forecast)
    toronto_forecast = align_schema(toronto_forecast)
    combined_forecast = nyc_forecast.unionByName(toronto_forecast, allowMissingColumns=True)

    delay_threshold = 5.7  
    prob_threshold = 0.07  

    combined_forecast = combined_forecast.fillna({"response_minutes": 0})

    combined_forecast = combined_forecast \
        .withColumn("prediction", F.when(F.col("delay_risk_probability") >= prob_threshold, 1).otherwise(0)) \
        .withColumn("actual_label", F.when(F.col("response_minutes") > delay_threshold, 1).otherwise(0))

    # Save for Confusion Matrix
    combined_forecast.select("City", "prediction", "actual_label", "delay_risk_probability", "response_minutes") \
        .write.mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("capstone_project.dashboard_predictions_all")

    # Save for Risk Distribution
    combined_forecast.select("City", "delay_risk_probability") \
        .write.mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("capstone_project.dashboard_risk_distribution")
    
    print("Dashboard tables updated successfully!")

except Exception as e:
    print(f"Error processing forecast data: {e}")