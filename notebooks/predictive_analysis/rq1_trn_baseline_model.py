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
import gc

for name in list(globals().keys()):
    if "model" in name or "pipeline" in name or "preds" in name:
        del globals()[name]

gc.collect()

# %%
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# =========================================================
# 0. SERVERLESS / SPARK CONNECT CLEANUP
# =========================================================
for var_name in [
    "model_lr", "model_rf",
    "pipeline_lr", "pipeline_rf",
    "preds_lr", "preds_rf",
    "train_df", "test_df",
    "comparison_df", "results"
]:
    if var_name in globals():
        del globals()[var_name]

gc.collect()

# =========================================================
# 1. SETUP & DATA LOADING
# =========================================================
output_dir = "../../output/graphs"
os.makedirs(output_dir, exist_ok=True)

# Load the TORONTO model-ready table
df = spark.table("workspace.capstone_project.toronto_model_ready")
df = df.filter(F.col("delay_indicator").isNotNull())

# Define feature groups
numeric_features = [
    'hour',
    'day_of_week',
    'month',
    'year',
    'unified_alarm_level'
]

categorical_features = [
    'season',
    'incident_category',
    'unified_call_source',
    'location_area'
]

label_col = "delay_indicator"

# =========================================================
# 2. PREPROCESSING PIPELINE
# =========================================================
stages = []

for col in categorical_features:
    indexer = StringIndexer(
        inputCol=col,
        outputCol=f"{col}_index",
        handleInvalid="keep"
    )
    encoder = OneHotEncoder(
        inputCols=[f"{col}_index"],
        outputCols=[f"{col}_vec"]
    )
    stages += [indexer, encoder]

assembler_inputs = [f"{col}_vec" for col in categorical_features] + numeric_features

assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="unscaled_features",
    handleInvalid="skip"
)
stages.append(assembler)

scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features",
    withMean=False,
    withStd=True
)
stages.append(scaler)

# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# =========================================================
# 4. LOGISTIC REGRESSION (BASELINE)
# =========================================================
lr = LogisticRegression(
    labelCol=label_col,
    featuresCol="features",
    maxIter=100
)

pipeline_lr = Pipeline(stages=stages + [lr])
model_lr = pipeline_lr.fit(train_df)
preds_lr = model_lr.transform(test_df)

# =========================================================
# 5. EVALUATION FUNCTION
# =========================================================
auc_eval = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

acc_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="accuracy"
)

prec_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="weightedPrecision"
)

rec_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="f1"
)

def get_metrics(predictions, model_name):
    return {
        "Model": model_name,
        "AUC_ROC": round(auc_eval.evaluate(predictions), 3),
        "Accuracy": round(acc_eval.evaluate(predictions), 3),
        "Precision": round(prec_eval.evaluate(predictions), 3),
        "Recall": round(rec_eval.evaluate(predictions), 3),
        "F1_Score": round(f1_eval.evaluate(predictions), 3)
    }

lr_result = get_metrics(preds_lr, "Logistic Regression")

# =========================================================
# 6. EXTRACT LR ROC DATA BEFORE CLEANUP
# =========================================================
lr_pdf = (
    preds_lr
    .select(
        F.col(label_col).cast("double").alias("label"),
        F.col("probability")
    )
    .toPandas()
)

lr_y_true = lr_pdf["label"].values
lr_y_score = lr_pdf["probability"].apply(lambda x: float(x[1])).values
lr_fpr, lr_tpr, _ = roc_curve(lr_y_true, lr_y_score)
lr_auc = auc(lr_fpr, lr_tpr)

# =========================================================
# 7. CLEANUP LR MODEL BEFORE RF FIT
# =========================================================
for var_name in ["model_lr", "pipeline_lr", "preds_lr"]:
    if var_name in globals():
        del globals()[var_name]

gc.collect()

# =========================================================
# 8. RANDOM FOREST (PROPOSED MODEL)
# =========================================================
rf = RandomForestClassifier(
    labelCol=label_col,
    featuresCol="features",
    numTrees=100,
    seed=42
)

pipeline_rf = Pipeline(stages=stages + [rf])
model_rf = pipeline_rf.fit(train_df)
preds_rf = model_rf.transform(test_df)

rf_result = get_metrics(preds_rf, "Random Forest")

# =========================================================
# 9. EXTRACT RF ROC DATA
# =========================================================
rf_pdf = (
    preds_rf
    .select(
        F.col(label_col).cast("double").alias("label"),
        F.col("probability")
    )
    .toPandas()
)

rf_y_true = rf_pdf["label"].values
rf_y_score = rf_pdf["probability"].apply(lambda x: float(x[1])).values
rf_fpr, rf_tpr, _ = roc_curve(rf_y_true, rf_y_score)
rf_auc = auc(rf_fpr, rf_tpr)

# =========================================================
# 10. COMPARISON TABLE
# =========================================================
results = [lr_result, rf_result]
comparison_df = pd.DataFrame(results)

print("\n--- Baseline vs Proposed Comparison - TORONTO ---")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(f"{output_dir}/model_comparison_toronto.csv", index=False)

# =========================================================
# 11. VISUALIZATION: COMBINED ROC CURVE
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.3f})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Toronto Data")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/combined_roc_comparison_toronto.png")
plt.show()
plt.close()

# =========================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================
preprocess_pipeline = Pipeline(stages=stages)
preprocess_model = preprocess_pipeline.fit(train_df)
sample_transformed = preprocess_model.transform(train_df.limit(1))

unscaled_metadata = sample_transformed.schema["unscaled_features"].metadata
attrs = unscaled_metadata["ml_attr"]["attrs"]

ordered_features = []
for attr_type in attrs:
    ordered_features.extend(attrs[attr_type])

expanded_feature_names = [x["name"] for x in ordered_features]

importances = model_rf.stages[-1].featureImportances.toArray()

importance_df = pd.DataFrame({
    "expanded_feature": expanded_feature_names,
    "importance": importances[:len(expanded_feature_names)]
})

def map_to_original_feature(expanded_name):
    for cat in categorical_features:
        if expanded_name.startswith(f"{cat}_vec"):
            return cat
    for num in numeric_features:
        if expanded_name == num:
            return num
    return expanded_name

importance_df["Feature"] = importance_df["expanded_feature"].apply(map_to_original_feature)

grouped_importance_df = (
    importance_df
    .groupby("Feature", as_index=False)["importance"]
    .sum()
    .sort_values(by="importance", ascending=False)
)

print("\n--- Random Forest Feature Importance - TORONTO ---")
print(grouped_importance_df.to_string(index=False))

grouped_importance_df.to_csv(
    f"{output_dir}/rf_feature_importance_toronto.csv",
    index=False
)

plt.figure(figsize=(10, 8))
plt.barh(grouped_importance_df["Feature"], grouped_importance_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance - Toronto")
plt.tight_layout()
plt.savefig(f"{output_dir}/rf_feature_importance_toronto.png")
plt.show()
plt.close()

# =========================================================
# 13. FINAL CLEANUP
# =========================================================
for var_name in [
    "model_rf", "pipeline_rf", "preds_rf",
    "preprocess_model", "preprocess_pipeline",
    "lr_pdf", "rf_pdf"
]:
    if var_name in globals():
        del globals()[var_name]

gc.collect()
