from __future__ import annotations

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def load_city_survival_spark(
    spark,
    table_name: str,
    censor_threshold: float = 60.0,
    duration_col: str = "response_minutes",
    event_col: str = "event_indicator",
) -> SparkDataFrame:
    """
    Load survival columns from a Spark table and apply uniform right-censoring at censor_threshold.

    Rules:
      - keep rows where duration_original is not null and > 0
      - event must be not null
      - response_minutes = min(duration_original, censor_threshold)
      - event_indicator = 1 only if duration_original <= threshold and event_original == 1 else 0

    Returns Spark DF with:
      - response_minutes
      - event_indicator
    """
    df = (
        spark.read.table(table_name)
        .select(
            F.col(duration_col).alias("duration_original"),
            F.col(event_col).alias("event_original"),
        )
        .where("duration_original is not null AND duration_original > 0 AND event_original is not null")
        .withColumn("response_minutes", F.least(F.col("duration_original"), F.lit(float(censor_threshold))))
        .withColumn(
            "event_indicator",
            F.when(
                (F.col("duration_original") <= censor_threshold) & (F.col("event_original") == 1),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .select("response_minutes", "event_indicator")
    )
    return df


def add_strata_columns(
    df_spark: SparkDataFrame,
    hour_col: str = "hour",
    season_col: str = "season",
    dow_col: str = "day_of_week",
) -> SparkDataFrame:
    """
    Adds stratification columns used in your US4.2 stratified KM notebook:
      - hour_group: Night/Morning/Afternoon/Evening
      - day_of_week_name: Sunday..Saturday

    Assumes df_spark contains hour, season, day_of_week columns already.
    """
    df = df_spark

    df = df.withColumn(
        "hour_group",
        F.when((F.col(hour_col) >= 0) & (F.col(hour_col) < 6), "Night")
        .when((F.col(hour_col) >= 6) & (F.col(hour_col) < 12), "Morning")
        .when((F.col(hour_col) >= 12) & (F.col(hour_col) < 18), "Afternoon")
        .otherwise("Evening"),
    )

    df = df.withColumn(
        "day_of_week_name",
        F.when(F.col(dow_col) == 1, "Sunday")
        .when(F.col(dow_col) == 2, "Monday")
        .when(F.col(dow_col) == 3, "Tuesday")
        .when(F.col(dow_col) == 4, "Wednesday")
        .when(F.col(dow_col) == 5, "Thursday")
        .when(F.col(dow_col) == 6, "Friday")
        .when(F.col(dow_col) == 7, "Saturday"),
    )

    # Keep season as-is (your data already uses winter/spring/summer/fall)
    return df