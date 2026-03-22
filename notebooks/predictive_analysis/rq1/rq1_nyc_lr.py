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
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

from sklearn.metrics import roc_curve, auc

spark = SparkSession.builder.getOrCreate()
print(f"Spark version: {spark.version}")

# =========================================================
# 1. SETUP
# =========================================================
output_dir = "../../../output"
os.makedirs(output_dir, exist_ok=True)

df = spark.table("workspace.capstone_project.nyc_model_ready").filter(col("delay_indicator").isNotNull())

numeric_features = ['hour', 'day_of_week', 'month', 'year', 'unified_alarm_level']
categorical_features = ['season', 'incident_category', 'unified_call_source', 'location_area']

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

auc_eval = BinaryClassificationEvaluator(
    labelCol="delay_indicator",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

multi_eval = MulticlassClassificationEvaluator(
    labelCol="delay_indicator",
    predictionCol="prediction"
)

# =========================================================
# 2. PIPELINE
# =========================================================
stages = []

for c in categorical_features:
    stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep"))
    stages.append(OneHotEncoder(inputCols=[f"{c}_index"], outputCols=[f"{c}_vec"]))

assembler_inputs = [f"{c}_vec" for c in categorical_features] + numeric_features

assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="unscaled_features"
)

scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features",
    withMean=False,
    withStd=True
)

clf = LogisticRegression(
    labelCol="delay_indicator",
    featuresCol="features",
    maxIter=30,
    regParam=0.01
)

pipeline = Pipeline(stages=stages + [assembler, scaler, clf])

# =========================================================
# 3. FIT + EVALUATE
# =========================================================
fitted_model = None
predictions = None
roc_pdf = None

try:
    fitted_model = pipeline.fit(train_df)
    predictions = fitted_model.transform(test_df)

    m_auc = float(auc_eval.evaluate(predictions))
    m_prec = float(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}))
    m_rec = float(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}))
    m_f1 = float(multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"}))

    accuracy = (
        predictions.filter(col("delay_indicator") == col("prediction")).count() / predictions.count()
    )

    metrics_df = pd.DataFrame([{
        "Model": "Logistic Regression",
        "AUC-ROC": round(m_auc, 3),
        "Precision": round(m_prec, 3),
        "Recall": round(m_rec, 3),
        "F1-Score": round(m_f1, 3),
        "Accuracy": round(float(accuracy), 3)
    }])
    metrics_path = f"{output_dir}/tables/rq1_nyc_logistic_regression_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    # print(metrics_df.to_string(index=False))
    display(metrics_df)

    roc_pdf = predictions.select(
        col("delay_indicator").cast("double").alias("label"),
        vector_to_array("probability")[1].alias("score")
    ).toPandas()

    fpr, tpr, _ = roc_curve(roc_pdf["label"], roc_pdf["score"])
    calc_auc = auc(fpr, tpr)

    roc_values_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    csv_path = f"{output_dir}/graphs/rq1_nyc_logistic_regression_roc_points.csv"
    roc_values_df.to_csv(csv_path, index=False)
    print(f"ROC curve points saved to {csv_path}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {calc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("NYC ROC Curve - Logistic Regression")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rq1_nyc_logistic_regression_roc_curve.png", dpi=300)
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
    
print("Logistic Regression complete.")
