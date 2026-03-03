import pandas as pd
import matplotlib.pyplot as plt
import gc
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.sql.functions import col

# 1. Initialize Spark Session with Increased Cache Limit
spark = SparkSession.builder \
    .appName("FireIncident_FeatureImportance") \
    .config("spark.connect.ml.modelCacheSizeLimitBytes", "2147483648") \
    .getOrCreate()

# 2. Define Output Paths
GRAPH_DIR = "../../output/graphs/"
TABLE_DIR = "../../output/tables/"

def run_importance_pipeline(city_name, table_path):
    print(f"\n>>> Analyzing {city_name} from {table_path}")

    # Load Data
    df = spark.table(table_path)
    df = df.filter(col("delay_indicator").isNotNull())
    
    # 3. Categorical Encoding
    categorical_cols = ["season", "incident_category", "unified_call_source", "location_area"]
    indexed_cols = [c + "_idx" for c in categorical_cols]
    
    for c in categorical_cols:
        indexer = StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
        df = indexer.fit(df).transform(df)
    
    # 4. Feature Selection
    feature_cols = [
        "hour", "day_of_week", "month", "unified_alarm_level", 
        "year", "calls_past_30min", "calls_past_60min"
    ] + indexed_cols
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data_vec = assembler.transform(df).select("features", "response_minutes")
    train_df, test_df = data_vec.randomSplit([0.8, 0.2], seed=42)

    # 5. Model Training (GBT)
    # Reduced maxBins to 32 (default) to keep the model size lean
    gbt = GBTRegressor(featuresCol="features", labelCol="response_minutes", maxIter=20, maxBins=32)
    model = gbt.fit(train_df)

    # 6. Extract Importance
    importances = model.featureImportances.toArray()
    
    # 7. Visualization
    plt.figure(figsize=(10, 6))
    sorted_idx = importances.argsort()
    plt.barh([feature_cols[i] for i in sorted_idx], importances[sorted_idx])
    plt.title(f"Feature Importance: {city_name} Fire Response")
    plt.xlabel("GBT Model Importance")
    plt.tight_layout()
    
    plt.savefig(f"{GRAPH_DIR}{city_name.lower()}_importance.png")
    plt.close()

    # 8. Table Export
    rank_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    rank_df.to_csv(f"{TABLE_DIR}{city_name.lower()}_importance_table.csv", index=False)
    
    del model
    del gbt
    spark.catalog.clearCache() 
    gc.collect() 
    return rank_df

# Main Execution
try:
    toronto_df = run_importance_pipeline("Toronto", "workspace.capstone_project.toronto_model_ready")
    nyc_df = run_importance_pipeline("NYC", "workspace.capstone_project.nyc_model_ready")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("\nPipeline complete. Check /output/ folders.")