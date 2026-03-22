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

print("Starting RQ4 TORONTO GBT Classifier...")

base_output_dir = os.path.abspath("../../../output")
tables_dir = os.path.join(base_output_dir, "tables")
graphs_dir = os.path.join(base_output_dir, "graphs")
models_dir = os.path.join(base_output_dir, "models")

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

label_col = "delay_indicator"

df = spark.table("workspace.capstone_project.toronto_model_ready")
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

base_df = (
    df.select(*required_cols)
    .dropna(subset=numeric_cols + [label_col])
)

train_df, test_df = base_df.randomSplit([0.8, 0.2], seed=42)

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
    maxIter=80,
    maxDepth=7,
    stepSize=0.03,
    minInstancesPerNode=10,
    subsamplingRate=0.8,
    seed=42
)

pipeline = Pipeline(stages=[hasher, assembler, clf])

roc_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
pr_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
precision_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision")
recall_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedRecall")
f1_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")

fitted_model = None
predictions = None
roc_pdf = None

try:
    fitted_model = pipeline.fit(train_df)
    predictions = fitted_model.transform(test_df)

    m_auc = float(roc_eval.evaluate(predictions))
    m_pr = float(pr_eval.evaluate(predictions))
    m_prec = float(precision_eval.evaluate(predictions))
    m_rec = float(recall_eval.evaluate(predictions))
    m_f1 = float(f1_eval.evaluate(predictions))
    m_acc = predictions.filter(F.col(label_col) == F.col("prediction")).count() / predictions.count()

    metrics_df = pd.DataFrame([{
        "Model": "GBT Classifier",
        "AUC-ROC": m_auc,
        "PR-AUC": m_pr,
        "Precision": m_prec,
        "Recall": m_rec,
        "F1-Score": m_f1,
        "Accuracy": m_acc
    }])

    metrics_df.to_csv(os.path.join(tables_dir, "rq4_trn_gbt_classifier_metrics.csv"), index=False)
    display(metrics_df.round(3))

    gbt_stage = fitted_model.stages[-1]

    gbt_artifact = {
        "model_type": "GBT Classifier",
        "city": "Toronto",
        "rq": "RQ4",
        "label_col": label_col,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "feature_hasher_num_features": 512,
        "max_iter": 80,
        "max_depth": 7,
        "step_size": 0.03,
        "min_instances_per_node": 10,
        "subsampling_rate": 0.8,
        "num_trees": int(len(gbt_stage.trees))
    }

    try:
        gbt_artifact["feature_importances"] = gbt_stage.featureImportances.toArray().tolist()
    except Exception:
        gbt_artifact["feature_importances"] = None

    with open(os.path.join(models_dir, "rq4_trn_gbt_classifier_artifact.json"), "w", encoding="utf-8") as f:
        json.dump(gbt_artifact, f, indent=2)

    if gbt_artifact["feature_importances"] is not None:
        pd.DataFrame({
            "Feature_Index": list(range(len(gbt_artifact["feature_importances"]))),
            "Importance": gbt_artifact["feature_importances"]
        }).sort_values(by="Importance", ascending=False).to_csv(
            os.path.join(tables_dir, "rq4_trn_gbt_classifier_feature_importance.csv"),
            index=False
        )

    roc_pdf = predictions.select(
        F.col(label_col).cast("double").alias("label"),
        vector_to_array("probability")[1].alias("score")
    ).toPandas()

    fpr, tpr, _ = roc_curve(roc_pdf["label"], roc_pdf["score"])

    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        os.path.join(graphs_dir, "rq4_trn_gbt_classifier_roc_points.csv"),
        index=False
    )

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"GBT Classifier (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Delay Classification Model Performance (Toronto)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "rq4_trn_gbt_classifier_roc_curve.png"), dpi=300)
    plt.close()

finally:
    if roc_pdf is not None:
        del roc_pdf
    if predictions is not None:
        del predictions
    if fitted_model is not None:
        del fitted_model
    del pipeline
    gc.collect()

print("RQ4 TORONTO GBT Classifier complete.")
