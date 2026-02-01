import dlt
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, hour, date_format, when

@dlt.table(
    name="tfs_incidents_gold",
    comment="Incident-level enriched dataset for analytics and ML"
)
def tfs_incidents_gold():
    w30 = (
        Window
        .partitionBy("Incident_Station_Area")
        .orderBy(F.col("alarm_time").cast("long"))
        .rangeBetween(-30 * 60, 0)
    )
    w60 = (
        Window
        .partitionBy("Incident_Station_Area")
        .orderBy(F.col("alarm_time").cast("long"))
        .rangeBetween(-60 * 60, 0)
    )
    return (dlt.read("tfs_incidents_silver")
        .filter(F.col("alarm_time").isNotNull())
        .withColumn("incident_hour", hour(col("alarm_time")))
        .withColumn("day_of_week", date_format(col("alarm_time"), "EEEE"))
        .withColumn("month_name", date_format(col("alarm_time"), "MMMM"))
        .withColumn("time_of_day",
            when(col("incident_hour").between(0, 5), "Night (12AM-6AM)")
            .when(col("incident_hour").between(6, 11), "Morning (6AM-12PM)")
            .when(col("incident_hour").between(12, 17), "Afternoon (12PM-6PM)")
            .otherwise("Evening (6PM-12AM)")
        )
        .withColumn("is_weekend",
            when(col("day_of_week").isin(["Saturday", "Sunday"]), True).otherwise(False)
        )
        .withColumn("calls_past_30m", F.count("*").over(w30) - 1)
        .withColumn("calls_past_60m", F.count("*").over(w60) - 1)

        # safety: avoid negative values
        .withColumn(
            "calls_past_30m",
            F.when(F.col("calls_past_30m") < 0, 0).otherwise(F.col("calls_past_30m"))
        )
        .withColumn(
            "calls_past_60m",
            F.when(F.col("calls_past_60m") < 0, 0).otherwise(F.col("calls_past_60m"))
        )
         .withColumn("season",
            when(col("month_name").isin(["December", "January", "February"]), "Winter")
            .when(col("month_name").isin(["March", "April", "May"]), "Spring")
            .when(col("month_name").isin(["June", "July", "August"]), "Summer")
            .otherwise("Fall")  # September, October, November
        )
        # Dropping of duplicate columns
        .drop("TFS_Alarm_Time", "TFS_Arrival_Time", "Last_TFS_Unit_Clear_Time")
    )