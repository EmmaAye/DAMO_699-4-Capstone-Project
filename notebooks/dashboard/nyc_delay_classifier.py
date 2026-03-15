import pandas as pd
import pickle
from pyspark.sql import functions as F

# 1. HARVEST HAZARD RATIOS (Survival Analysis)
PATH_CPH_NYC = "/Workspace/Users/thanda.aye03@gmail.com/DAMO_699-4-Capstone-Project/output/models/cph_NYC.pkl"

try:
    with open(PATH_CPH_NYC, 'rb') as f:
        cph_model = pickle.load(f)
    
    hazard_data = pd.DataFrame({
        'Feature': cph_model.params_.index,
        'Hazard_Ratio': cph_model.hazard_ratios_
    }).sort_values(by='Hazard_Ratio')
    
    spark.createDataFrame(hazard_data).write.mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("capstone_project.dashboard_hazard_ratios")
    print("Hazard Ratios saved successfully.")
except Exception as e:
    print(f"CPH Load failed: {e}")

# 2. GENERATE PREDICTIVE METRICS FROM FORECAST TABLE
try:
    forecast_df = spark.table("capstone_project.nyc_risk_forecast_output")
    
    # Requirement 3: Risk Distribution Histogram
    forecast_df.select("delay_risk_probability") \
        .write.mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("capstone_project.dashboard_risk_distribution")

    # Requirement 2: Confusion Matrix
    forecast_df.withColumn("prediction", F.when(F.col("delay_risk_probability") > 0.5, 1).otherwise(0)) \
        .select("prediction", "delay_risk_probability") \
        .write.mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("capstone_project.dashboard_predictions_nyc")
        
    # Requirement 1: Static Metrics Table
    metrics_data = [
        ("NYC Delay Classifier", 0.84, 0.88, "XGBoost"),
        ("Cox Proportional Hazards", None, None, "Survival Analysis")
    ]
    
    spark.createDataFrame(metrics_data, ["Model", "Accuracy", "AUC_ROC", "Type"]) \
        .write.mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable("capstone_project.dashboard_metrics")

    print("Dashboard tables are now live and schema-synced!")

except Exception as e:
    print(f"Error: {e}")