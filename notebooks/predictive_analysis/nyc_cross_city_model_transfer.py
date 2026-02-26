# ============================================
# Sprint 7 / RQ3 Subtask: Toronto -> NYC Cross-City Transfer (Predictive Only)
# Uses teammate delay-classifier pipeline (Hasher + Assembler + RF)
# ============================================

import gc
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# -----------------------
# Config
# -----------------------
TRAIN_TABLE = "workspace.capstone_project.toronto_model_ready"
TEST_TABLE  = "workspace.capstone_project.nyc_model_ready"

TRAIN_CITY = "Toronto"
TEST_CITY  = "NYC"

LABEL_COL = "delay_indicator"

# Match teammate feature family
DESIRED_CATEGORICAL = ["incident_category", "season", "unified_call_source", "location_area"]
DESIRED_NUMERIC     = ["hour", "day_of_week", "month", "year", "unified_alarm_level",
                       "calls_past_30min", "calls_past_60min"]

HASH_DIM = 512
SEED = 42

# Match teammate RF settings
RF_PARAMS = dict(numTrees=50, maxDepth=5, seed=SEED)

# Balanced sample size (keeps session stable + prevents one-class in train)
MAX_PER_CLASS = 200_000

SAVE_RESULTS_TABLE = "workspace.capstone_project.transfer_test_toronto_to_nyc_sprint7"

# -----------------------
# Helpers
# -----------------------
def load_and_prepare(table_name: str, city_name: str):
    df = spark.table(table_name).filter(col(LABEL_COL).isNotNull())
    df = df.withColumn(LABEL_COL, col(LABEL_COL).cast("int"))

    existing = set(df.columns)
    cat_cols = [c for c in DESIRED_CATEGORICAL if c in existing]
    num_cols = [c for c in DESIRED_NUMERIC if c in existing]

    if len(cat_cols) + len(num_cols) == 0:
        raise ValueError(f"{city_name}: No usable feature columns found in {table_name}")

    # label distribution
    dist = df.groupBy(LABEL_COL).count().orderBy(LABEL_COL)
    print(f"\n{city_name} label distribution:")
    dist.show()

    labels = [r[LABEL_COL] for r in dist.select(LABEL_COL).collect()]
    if len(labels) < 2:
        raise ValueError(f"{city_name} has only one class in {LABEL_COL}: {labels}. Fix label creation upstream.")

    keep_cols = list(dict.fromkeys(cat_cols + num_cols + [LABEL_COL]))
    return df.select(*keep_cols), cat_cols, num_cols

def balanced_train_sample(df, label_col, max_per_class=200_000, seed=42):
    """
    Ensures both classes are present & reduces imbalance by sampling each class up to max_per_class.
    """
    df0 = df.filter(col(label_col) == 0).limit(max_per_class)
    df1 = df.filter(col(label_col) == 1).limit(max_per_class)

    n0 = df0.count()
    n1 = df1.count()
    if n0 == 0 or n1 == 0:
        raise ValueError(f"Train set has one class only (n0={n0}, n1={n1}).")

    # If one class is still huge, downsample to match the smaller class
    n_min = min(n0, n1)
    df0s = df0.sample(withReplacement=False, fraction=min(1.0, n_min / n0), seed=seed)
    df1s = df1.sample(withReplacement=False, fraction=min(1.0, n_min / n1), seed=seed)

    return df0s.unionByName(df1s)

def build_pipeline(cat_cols, num_cols):
    stages = []

    if len(cat_cols) > 0:
        hasher = FeatureHasher(inputCols=cat_cols, outputCol="categorical_features", numFeatures=HASH_DIM)
        stages.append(hasher)
        assembler_inputs = num_cols + ["categorical_features"]
    else:
        assembler_inputs = num_cols

    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=LABEL_COL,
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        **RF_PARAMS
    )
    stages.append(rf)

    return Pipeline(stages=stages)

def evaluate(pred_df):
    roc_eval = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr_eval  = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    acc_eval = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy")
    f1_eval  = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="f1")

    return {
        "train_city": TRAIN_CITY,
        "test_city": TEST_CITY,
        "auc_roc": float(roc_eval.evaluate(pred_df)),
        "auc_pr": float(pr_eval.evaluate(pred_df)),
        "accuracy": float(acc_eval.evaluate(pred_df)),
        "f1": float(f1_eval.evaluate(pred_df)),
        "test_positive_rate": float(pred_df.agg(F.avg(col(LABEL_COL))).first()[0])
    }

# -----------------------
# Load + align common columns
# -----------------------
train_df, train_cat, train_num = load_and_prepare(TRAIN_TABLE, TRAIN_CITY)
test_df,  test_cat,  test_num  = load_and_prepare(TEST_TABLE, TEST_CITY)

common_cat = sorted(list(set(train_cat).intersection(set(test_cat))))
common_num = sorted(list(set(train_num).intersection(set(test_num))))

print("\nCommon numeric cols     :", common_num)
print("Common categorical cols :", common_cat)

if len(common_cat) + len(common_num) == 0:
    raise ValueError("No common feature columns between train and test tables.")

train_common = train_df.select(*(common_num + common_cat + [LABEL_COL]))
test_common  = test_df.select(*(common_num + common_cat + [LABEL_COL]))

# -----------------------
# Train on Toronto -> Test on NYC
# -----------------------
train_sample = balanced_train_sample(train_common, LABEL_COL, MAX_PER_CLASS, SEED)
pipeline = build_pipeline(common_cat, common_num)

print(f"\nTraining on {TRAIN_CITY} (balanced sample) ...")
model = pipeline.fit(train_sample)
print("Training complete.")

print(f"Testing on {TEST_CITY} ...")
pred = model.transform(test_common).select(LABEL_COL, "rawPrediction", "probability", "prediction")

metrics = evaluate(pred)

results_df = spark.createDataFrame([metrics]).withColumn("run_ts", F.current_timestamp())
display(results_df)

results_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SAVE_RESULTS_TABLE)
print(f" Results saved to: {SAVE_RESULTS_TABLE}")

del model
gc.collect()