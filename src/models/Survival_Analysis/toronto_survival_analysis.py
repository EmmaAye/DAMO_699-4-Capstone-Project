from pyspark.sql.functions import col, when
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

def load_and_clean_base(spark, table_name: str):
    """
    Loads table and returns cleaned Spark DF with required columns.
    Requires columns: response_minutes, event_indicator, hour
    """
    df = spark.read.table(table_name)

    df_clean = (
        df.select("response_minutes", "event_indicator", "hour")
          .filter(col("response_minutes").isNotNull())
          .filter(col("event_indicator").isNotNull())
          .filter(col("response_minutes") > 0)
    )
    return df_clean


def km_overall(df_clean_spark, city_label="Toronto", save_path="km_overall.png"):
    """
    Plots overall KM curve and saves PNG.
    """
    df_pd = df_clean_spark.select("response_minutes", "event_indicator").toPandas()

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df_pd["response_minutes"],
        event_observed=df_pd["event_indicator"],
        label=city_label
    )

    plt.figure(figsize=(8, 6))
    kmf.plot_survival_function()
    plt.title(f"Kaplan–Meier Survival Curve - {city_label}")
    plt.xlabel("Minutes")
    plt.ylabel("Probability Unit Not Yet Arrived")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved:", save_path)
    print("Median time-to-arrival (KM):", kmf.median_survival_time_)

    return kmf


def km_by_time_of_day(df_clean_spark, city_label="Toronto", save_path="km_by_time.png"):
    """
    Plots KM curves by time-of-day bins and saves PNG.
    """
    df_time = df_clean_spark.withColumn(
        "time_bin",
        when((col("hour") >= 0) & (col("hour") <= 5), "Night")
        .when((col("hour") >= 6) & (col("hour") <= 11), "Morning")
        .when((col("hour") >= 12) & (col("hour") <= 17), "Afternoon")
        .otherwise("Evening")
    )

    df_pd = df_time.select("response_minutes", "event_indicator", "time_bin").toPandas()

    plt.figure(figsize=(8, 6))

    for group in ["Night", "Morning", "Afternoon", "Evening"]:
        sub = df_pd[df_pd["time_bin"] == group]
        if len(sub) == 0:
            print(f"No rows for group: {group}")
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=sub["response_minutes"],
            event_observed=sub["event_indicator"],
            label=group
        )
        kmf.plot_survival_function()

    plt.title(f"Kaplan–Meier by Time of Day - {city_label}")
    plt.xlabel("Minutes")
    plt.ylabel("Probability Unit Not Yet Arrived")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved:", save_path)
