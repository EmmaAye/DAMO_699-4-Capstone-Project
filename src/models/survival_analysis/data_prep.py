from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def load_city_survival_spark(
    spark,
    table_name: str,
    censor_threshold: float = 60.0,
    duration_col: str = "response_minutes",
    event_col: str = "event_indicator",
    extra_cols: Optional[Iterable[str]] = None,
) -> SparkDataFrame:
    """
    Load survival columns from a Spark table and apply uniform right-censoring at censor_threshold.

    Returns Spark DF with:
      - response_minutes
      - event_indicator
      - plus any extra_cols requested (e.g., hour, season, day_of_week)
    """
    extra_cols = list(extra_cols) if extra_cols else []
    # avoid duplicates if caller accidentally includes these
    extra_cols = [c for c in extra_cols if c not in [duration_col, event_col]]
    thr = F.lit(float(censor_threshold))

    df = (
        spark.read.table(table_name)
        .select(
            F.col(duration_col).alias("duration_original"),
            F.col(event_col).alias("event_original"),
            *[F.col(c) for c in extra_cols],
        )
        # event is always explicit 0/1 in model ready data
        .where("event_original in (0,1)")
        # duration must be > 0 when present; NULL means censored (keep it)
        .where("duration_original is null OR duration_original > 0")
        # duration: NULL => 60, else clip at 60
        .withColumn(
            "response_minutes",
            F.when(F.col("duration_original").isNull(), thr)
             .otherwise(F.least(F.col("duration_original"), thr))
        )
        # event: only true events when observed duration <= 60 and event_original==1
        .withColumn(
            "event_indicator",
            F.when(
                F.col("duration_original").isNotNull()
                & (F.col("duration_original") <= thr)
                & (F.col("event_original") == 1),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .drop("duration_original", "event_original")
    )
    return df


def add_strata_columns(
    df_spark: SparkDataFrame,
    hour_col: str = "hour",
    dow_col: str = "day_of_week",
) -> SparkDataFrame:
    """
    Adds stratification columns used in stratified KM:
      - hour_group: Night/Morning/Afternoon/Evening
      - day_of_week_name: Sunday..Saturday

    Assumes df_spark contains: hour, day_of_week.
    (Season is kept as-is from the input DF.)
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
    return df


def prepare_city_df(
    spark,
    table_name: str,
    censor_threshold: float = 60.0,
) -> pd.DataFrame:
    """
    Convenience wrapper:
      1) load + censor survival fields
      2) bring in hour/season/day_of_week for stratification
      3) add derived strata columns
      4) return pandas for lifelines
    """
    df = load_city_survival_spark(
        spark,
        table_name=table_name,
        censor_threshold=censor_threshold,
        extra_cols=["hour", "season", "day_of_week"],
    )
    df = add_strata_columns(df, hour_col="hour", dow_col="day_of_week")
    return df.toPandas()
