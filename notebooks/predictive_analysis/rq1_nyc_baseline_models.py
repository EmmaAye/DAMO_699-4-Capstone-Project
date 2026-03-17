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
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

from sklearn.metrics import roc_curve, auc

# =========================================================
# 1. SETUP & DATA LOADING
# =========================================================
output_dir = "../../output/graphs"
os.makedirs(output_dir, exist_ok=True)

# Load NYC model-ready table
df = spark.table("workspace.capstone_project.nyc_model_ready")
df = df.filter(col("delay_indicator").isNotNull())

# Feature groups
numeric_features = ['hour', 'day_of_week', 'month', 'year', 'unified_alarm_level']
categorical_features = ['season', 'incident_category', 'unified_call_source', 'location_area']

# =========================================================
# 2. PREPROCESSING PIPELINE
# =========================================================
stages = []

for c in categorical_features:
    indexer = StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=[f"{c}_index"], outputCols=[f"{c}_vec"])
    stages += [indexer, encoder]

assembler_inputs = [f"{c}_vec" for c in categorical_features] + numeric_features
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="unscaled_features")
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withMean=False, withStd=True)

stages += [assembler, scaler]

# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# =========================================================
# 4. MODEL DEFINITIONS
# =========================================================
lr = LogisticRegression(
    labelCol="delay_indicator",
    featuresCol="features",
    maxIter=100
)

rf = RandomForestClassifier(
    labelCol="delay_indicator",
    featuresCol="features",
    numTrees=100,
    seed=42
)

gbt = GBTClassifier(
    labelCol="delay_indicator",
    featuresCol="features",
    maxIter=100,
    seed=42
)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "GBT": gbt
}

# =========================================================
# 5. TRAIN MODELS
# =========================================================
fitted_models = {}
predictions_dict = {}

for model_name, clf in models.items():
    pipeline = Pipeline(stages=stages + [clf])
    fitted_model = pipeline.fit(train_df)
    preds = fitted_model.transform(test_df)

    fitted_models[model_name] = fitted_model
    predictions_dict[model_name] = preds

# =========================================================
# 6. EVALUATION FUNCTION
# =========================================================
auc_eval = BinaryClassificationEvaluator(
    labelCol="delay_indicator",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

multi_eval = MulticlassClassificationEvaluator(
    labelCol="delay_indicator",
    predictionCol="prediction"
)

def get_metrics(predictions, model_name):
    return {
        "Model": model_name,
        "AUC-ROC": round(auc_eval.evaluate(predictions), 3),
        "Precision": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}), 3),
        "Recall": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}), 3),
        "F1-Score": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"}), 3)
    }

results = []
for model_name, preds in predictions_dict.items():
    results.append(get_metrics(preds, model_name))

comparison_df = pd.DataFrame(results).sort_values(by="AUC-ROC", ascending=False)

print("\n--- RQ1 NYC: Model Comparison ---")
print(comparison_df.to_string(index=False))

comparison_csv_path = f"{output_dir}/rq1_nyc_model_comparison.csv"
comparison_df.to_csv(comparison_csv_path, index=False)

# =========================================================
# 7. ROC CURVE FOR ALL 3 MODELS
# =========================================================
plt.figure(figsize=(10, 6))

roc_auc_records = []

for model_name, preds in predictions_dict.items():
    # Convert probability vector to array and extract positive-class probability
    roc_df = preds.select(
        col("delay_indicator").cast("double").alias("label"),
        vector_to_array("probability")[1].alias("score")
    ).toPandas()

    y_true = roc_df["label"]
    y_score = roc_df["score"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    model_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_auc:.3f})")

    roc_auc_records.append({
        "Model": model_name,
        "ROC_AUC_from_curve": round(model_auc, 3)
    })

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("RQ1 NYC - ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()

roc_plot_path = f"{output_dir}/rq1_nyc_combined_roc_curve.png"
plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# Save ROC AUC values separately too
roc_auc_df = pd.DataFrame(roc_auc_records).sort_values(by="ROC_AUC_from_curve", ascending=False)
roc_auc_csv_path = f"{output_dir}/rq1_nyc_roc_auc_values.csv"
roc_auc_df.to_csv(roc_auc_csv_path, index=False)

print(f"\nSaved comparison table: {comparison_csv_path}")
print(f"Saved ROC AUC values: {roc_auc_csv_path}")
print(f"Saved ROC curve plot: {roc_plot_path}")

# =========================================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================
rf_model = fitted_models["Random Forest"].stages[-1]
importances = rf_model.featureImportances.toArray()

# NOTE:
# Because one-hot encoding expands categorical features into many columns,
# these importances do not map perfectly back to original feature names.
# This simple version keeps only the first len(assembler_inputs) importances.
feat_importance_df = pd.DataFrame({
    "Feature": assembler_inputs[:len(importances)],
    "Importance": importances[:len(assembler_inputs)]
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feat_importance_df["Feature"], feat_importance_df["Importance"])
plt.title("RQ1 NYC - Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

rf_imp_path = f"{output_dir}/rq1_nyc_rf_feature_importance.png"
plt.savefig(rf_imp_path, dpi=300, bbox_inches="tight")
plt.close()

feat_imp_csv_path = f"{output_dir}/rq1_nyc_rf_feature_importance.csv"
feat_importance_df.to_csv(feat_imp_csv_path, index=False)

print(f"Saved RF feature importance plot: {rf_imp_path}")
print(f"Saved RF feature importance CSV: {feat_imp_csv_path}")
