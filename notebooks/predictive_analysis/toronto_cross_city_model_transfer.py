# ============================================
# Sprint 7 / RQ3 Subtask: Toronto Cross-City Model Transfer
# Goal: Train on Toronto -> Test on NYC (generalizability)
# Output: Metrics table + saved Delta results
# ============================================

# 1) Setup and Imports
import gc
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, FeatureHasher
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 2) Config
TRAIN_TABLE = "workspace.capstone_project.toronto_model_ready"
TEST_TABLE  = "workspace.capstone_project.nyc_model_ready"

TRAIN_CITY = "Toronto"
TEST_CITY  = "NYC"

LABEL_COL = "delay_indicator"

DESIRED_CATEGORICAL = ["incident_category", "season", "unified_call_source", "location_area"]
DESIRED_NUMERIC     = ["hour", "day_of_week", "month", "year", "unified_alarm_level",
                       "calls_past_30min", "calls_past_60min"]

HASH_DIM = 512
SEED = 42
TRAIN_FRACTION = 0.5
MODEL_TYPE = "rf"  # "rf" or "lr"

RF_PARAMS = dict(numTrees=80, maxDepth=6)
LR_PARAMS = dict(maxIter=50, regParam=0.05, elasticNetParam=0.0)

SAVE_RESULTS_TABLE = "workspace.capstone_project.transfer_test_toronto_to_nyc_sprint7"

# 3) Helpers
def load_and_prepare(table_name: str, city_name: str):
    df = spark.table(table_name).filter(col(LABEL_COL).isNotNull())
    df = df.withColumn(LABEL_COL, col(LABEL_COL).cast("int"))

    existing = set(df.columns)
    cat_cols = [c for c in DESIRED_CATEGORICAL if c in existing]
    num_cols = [c for c in DESIRED_NUMERIC if c in existing]

    if len(cat_cols) + len(num_cols) == 0:
        raise ValueError(f"No expected feature columns found in {table_name}. Columns: {df.columns}")

    dist = df.groupBy(LABEL_COL).count().orderBy(LABEL_COL)
    print(f"\n{city_name} label distribution:")
    dist.show()

    labels = [r[LABEL_COL] for r in dist.select(LABEL_COL).collect()]
    if len(labels) < 2:
        raise ValueError(f"{city_name} has only one class in {LABEL_COL}: {labels}")

    keep_cols = list(set(cat_cols + num_cols + [LABEL_COL]))
    return df.select(*keep_cols), cat_cols, num_cols

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

    if MODEL_TYPE == "rf":
        clf = RandomForestClassifier(
            featuresCol="features",
            labelCol=LABEL_COL,
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            **RF_PARAMS
        )
    else:
        clf = LogisticRegression(
            featuresCol="features",
            labelCol=LABEL_COL,
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            **LR_PARAMS
        )

    stages.append(clf)
    return Pipeline(stages=stages)

def evaluate(pred_df):
    roc_eval = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr_eval  = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    acc_eval = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy")
    f1_eval  = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="f1")

    auc_roc = float(roc_eval.evaluate(pred_df))
    auc_pr  = float(pr_eval.evaluate(pred_df))
    acc     = float(acc_eval.evaluate(pred_df))
    f1      = float(f1_eval.evaluate(pred_df))
    pos_rate = float(pred_df.agg(F.avg(col(LABEL_COL))).first()[0])

    return {
        "train_city": TRAIN_CITY,
        "test_city": TEST_CITY,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": acc,
        "f1": f1,
        "test_positive_rate": pos_rate
    }

# 4) Load and align common columns
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

# 5) Train on Toronto -> Test on NYC
train_sample = train_common.sample(withReplacement=False, fraction=TRAIN_FRACTION, seed=SEED)
pipeline = build_pipeline(common_cat, common_num)

print(f"\nTraining on {TRAIN_CITY} (fraction={TRAIN_FRACTION}) ...")
model = pipeline.fit(train_sample)
print("Training complete.")

print(f"Testing on {TEST_CITY} ...")
pred = model.transform(test_common).select(LABEL_COL, "rawPrediction", "probability", "prediction")

metrics = evaluate(pred)

# 6) Results table
results_df = spark.createDataFrame([metrics])
print("\n Toronto -> NYC Transfer Results:")
display(results_df)

# 7) Save results (DoD)
(
    results_df
    .withColumn("model_type", F.lit(MODEL_TYPE))
    .withColumn("train_fraction", F.lit(TRAIN_FRACTION))
    .withColumn("run_ts", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SAVE_RESULTS_TABLE)
)

print(f"✅ Results saved to: {SAVE_RESULTS_TABLE}")

# Cleanup
del model
gc.collect()

