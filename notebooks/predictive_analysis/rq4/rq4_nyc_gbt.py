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
import gc
import json
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import roc_curve, auc

print("Starting RQ4 NYC GBT Classifier...")

# =========================================================
# 1. PATHS
# =========================================================
base_output_dir = os.path.abspath("../../../output")
tables_dir = os.path.join(base_output_dir, "tables")
graphs_dir = os.path.join(base_output_dir, "graphs")
models_dir = os.path.join(base_output_dir, "models")

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

label_col = "delay_indicator"

# =========================================================
# 2. LOAD DATA
# =========================================================
df = spark.table("workspace.capstone_project.nyc_model_ready").filter(F.col(label_col).isNotNull())

categorical_cols = ["incident_category", "season", "unified_call_source", "location_area"]
numeric_cols = ["hour", "day_of_week", "month", "year", "unified_alarm_level", "calls_past_30min", "calls_past_60min"]
required_cols = ["incident_datetime"] + categorical_cols + numeric_cols + [label_col]

base_df = df.select(*required_cols).dropna(subset=numeric_cols + [label_col])
train_df, test_df = base_df.randomSplit([0.8, 0.2], seed=42)

# =========================================================
# 3. PIPELINE
# =========================================================
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

clf = GBTClassifier(
    featuresCol="features",
    labelCol=label_col,
    maxIter=20,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

pipeline = Pipeline(stages=[hasher, assembler, clf])

# =========================================================
# 4. EVALUATORS
# =========================================================
roc_eval = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_eval = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

precision_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="f1"
)

fitted_model = None
predictions = None
roc_pdf = None

try:
    # =====================================================
    # 5. TRAIN + PREDICT
    # =====================================================
    fitted_model = pipeline.fit(train_df)
    predictions = fitted_model.transform(test_df)

    # =====================================================
    # 6. METRICS
    # =====================================================
    m_auc = float(roc_eval.evaluate(predictions))
    m_pr = float(pr_eval.evaluate(predictions))
    m_prec = float(precision_eval.evaluate(predictions))
    m_rec = float(recall_eval.evaluate(predictions))
    m_f1 = float(f1_eval.evaluate(predictions))
    m_acc = predictions.filter(F.col(label_col) == F.col("prediction")).count() / predictions.count()

    metrics_df = pd.DataFrame([{
        "Model": "GBT Classifier",
        "AUC-ROC": round(m_auc, 3),
        "PR-AUC": round(m_pr, 3),
        "Precision": round(m_prec, 3),
        "Recall": round(m_rec, 3),
        "F1-Score": round(m_f1, 3),
        "Accuracy": round(float(m_acc), 3)
    }])

    metric_path = os.path.join(tables_dir, "rq4_nyc_gbt_classifier_metrics.csv")
    metrics_df.to_csv(metric_path, index=False)
    print(f"Metrics saved to {metric_path}")
    display(metrics_df)

    # =====================================================
    # 7. SAVE LIGHTWEIGHT MODEL ARTIFACTS
    # =====================================================
    gbt_stage = fitted_model.stages[-1]

    gbt_artifact = {
        "model_type": "GBT Classifier",
        "city": "NYC",
        "rq": "RQ4",
        "label_col": label_col,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "feature_hasher_num_features": 512,
        "max_iter": 20,
        "max_depth": 5,
        "step_size": 0.1,
        "num_trees": int(len(gbt_stage.trees))
    }

    # Some Spark versions expose featureImportances on GBTClassificationModel, some do not.
    try:
        gbt_artifact["feature_importances"] = gbt_stage.featureImportances.toArray().tolist()
    except Exception:
        gbt_artifact["feature_importances"] = None

    artifact_path = os.path.join(models_dir, "rq4_nyc_gbt_classifier_artifact.json")
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(gbt_artifact, f, indent=2)

    print(f"Lightweight artifact saved to {artifact_path}")

    if gbt_artifact["feature_importances"] is not None:
        feat_imp_df = pd.DataFrame({
            "Feature_Index": list(range(len(gbt_artifact["feature_importances"]))),
            "Importance": gbt_artifact["feature_importances"]
        }).sort_values(by="Importance", ascending=False)

        feat_imp_csv_path = os.path.join(tables_dir, "rq4_nyc_gbt_classifier_feature_importance.csv")
        feat_imp_df.to_csv(feat_imp_csv_path, index=False)
        print(f"Feature importance saved to {feat_imp_csv_path}")

    # =====================================================
    # 8. ROC DATA + PLOT
    # =====================================================
    roc_pdf = predictions.select(
        F.col(label_col).cast("double").alias("label"),
        vector_to_array("probability")[1].alias("score")
    ).toPandas()

    fpr, tpr, _ = roc_curve(roc_pdf["label"], roc_pdf["score"])

    roc_values_path = os.path.join(graphs_dir, "rq4_nyc_gbt_classifier_roc_points.csv")
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(roc_values_path, index=False)
    print(f"ROC values saved to {roc_values_path}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"GBT Classifier (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RQ4 NYC ROC Curve - GBT Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_curve_path = os.path.join(graphs_dir, "rq4_nyc_gbt_classifier_roc_curve.png")
    plt.savefig(roc_curve_path, dpi=300)
    plt.close()

    print(f"ROC curve saved to {roc_curve_path}")

finally:
    if roc_pdf is not None:
        del roc_pdf
    if predictions is not None:
        del predictions
    if fitted_model is not None:
        del fitted_model
    del pipeline
    gc.collect()

print("RQ4 NYC GBT Classifier complete.")
