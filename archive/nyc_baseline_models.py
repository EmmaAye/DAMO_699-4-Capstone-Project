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
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 1. SETUP & DATA LOADING
output_dir = "../../output/graphs"
os.makedirs(output_dir, exist_ok=True)

# Load the NYC model-ready table
df = spark.table("workspace.capstone_project.nyc_model_ready")
df = df.filter(df.delay_indicator.isNotNull())

# Define feature groups based on schema
numeric_features = ['hour', 'day_of_week', 'month', 'year', 'unified_alarm_level', 'calls_past_30min', 'calls_past_60min']
categorical_features = ['season', 'incident_category', 'unified_call_source', 'location_area']

# 2. PREPROCESSING PIPELINE
stages = []
for col in categorical_features:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=[f"{col}_index"], outputCols=[f"{col}_vec"])
    stages += [indexer, encoder]

assembler_inputs = [f"{col}_vec" for col in categorical_features] + numeric_features
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="unscaled_features")
stages.append(assembler)

scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
stages.append(scaler)

# 3. MODEL TRAINING
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Logistic Regression
lr = LogisticRegression(labelCol="delay_indicator", featuresCol="features", maxIter=100)
pipeline_lr = Pipeline(stages=stages + [lr])
model_lr = pipeline_lr.fit(train_df)
preds_lr = model_lr.transform(test_df)

# Random Forest
rf = RandomForestClassifier(labelCol="delay_indicator", featuresCol="features", numTrees=100, seed=42)
pipeline_rf = Pipeline(stages=stages + [rf])
model_rf = pipeline_rf.fit(train_df)
preds_rf = model_rf.transform(test_df)

# 4. EVALUATION & COMPARISON TABLE
def get_metrics(predictions, model_name):
    auc_eval = BinaryClassificationEvaluator(labelCol="delay_indicator", metricName="areaUnderROC")
    multi_eval = MulticlassClassificationEvaluator(labelCol="delay_indicator", predictionCol="prediction")
    
    return {
        "Model": model_name,
        "AUC-ROC": round(auc_eval.evaluate(predictions), 3),
        "Precision": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}), 3),
        "Recall": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}), 3)
    }

results = [get_metrics(preds_lr, "Logistic Regression"), get_metrics(preds_rf, "Random Forest")]
comparison_df = pd.DataFrame(results)

print("\n--- Baseline vs Proposed Comparison - NYC ---")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(f"{output_dir}/model_comparison_nyc.csv", index=False)

# 5. VISUALIZATIONS

# Chart 1: Combined ROC Curve
plt.figure(figsize=(10, 6))
# LR Curve
lr_summary = model_lr.stages[-1].summary
lr_roc = lr_summary.roc.toPandas()
plt.plot(lr_roc['FPR'], lr_roc['TPR'], label=f"Logistic Regression (AUC = {results[0]['AUC-ROC']})")

# RF Curve
rf_summary = model_rf.stages[-1].summary
rf_roc = rf_summary.roc.toPandas()
plt.plot(rf_roc['FPR'], rf_roc['TPR'], label=f"Random Forest (AUC = {results[1]['AUC-ROC']})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - NYC Data')
plt.legend()
plt.savefig(f"{output_dir}/combined_roc_comparison.png")
plt.close()

# Chart 2: Random Forest Feature Importance
importances = model_rf.stages[-1].featureImportances.toArray()
# For simplicity, using assembler input names (indices might vary if using OHE)
feat_importance_df = pd.DataFrame({'Feature': assembler_inputs, 'Importance': importances[:len(assembler_inputs)]})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feat_importance_df['Feature'], feat_importance_df['Importance'], color='salmon')
plt.title('Random Forest Feature Importance - NYC')
plt.gca().invert_yaxis()
plt.savefig(f"{output_dir}/rf_feature_importance.png")
plt.close()
