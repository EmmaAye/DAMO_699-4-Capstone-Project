import dlt
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, hour, date_format,lit,month,dayofweek, when

@dlt.table(
    name="tfs_incidents_gold",
    comment="Incident-level enriched dataset for analytics and ML"
)
def tfs_incidents_gold():
    w30 = ( Window .partitionBy("Incident_Station_Area") 
           .orderBy(F.col("alarm_time").cast("long")) 
           .rangeBetween(-30 * 60, 0) 
        )
    w60 = ( Window .partitionBy("Incident_Station_Area")
           .orderBy(F.col("alarm_time")
            .cast("long")) 
           .rangeBetween(-60 * 60, 0) 
           )
    return (dlt.read("tfs_incidents_silver")
        #.filter(F.col("alarm_time").isNotNull())
        .withColumn("hour", hour(col("alarm_time")))
        .withColumn("day_of_week", dayofweek(col("alarm_time")))
        .withColumn("month", month(col("alarm_time")))
        .withColumn("time_of_day",
            when(col("hour").between(0, 5), "Night (12AM-6AM)")
            .when(col("hour").between(6, 11), "Morning (6AM-12PM)")
            .when(col("hour").between(12, 17), "Afternoon (12PM-6PM)")
            .otherwise("Evening (6PM-12AM)")
        )
        .withColumn("is_weekend",
            when(col("day_of_week").isin([1, 7]), True).otherwise(False)
        )
        .withColumn("calls_past_30m", F.count("*").over(w30) - 1)
        .withColumn("calls_past_60m", F.count("*").over(w60) - 1)

        # safety: avoid negative values
        .withColumn(
            "calls_past_30min",
            F.when(F.col("calls_past_30m") < 0, 0).otherwise(F.col("calls_past_30m"))
        )
        .withColumn(
        "calls_past_60min",
        F.when(F.col("calls_past_60m") < 0, 0).otherwise(F.col("calls_past_60m"))
)
        .withColumn("season",
             when(col("month").isin([12, 1, 2]), "Winter")
             .when(col("month").isin([3, 4, 5]), "Spring")
             .when(col("month").isin([6, 7, 8]), "Summer")
             .otherwise("Fall")
        )
        .withColumn(
            "event_indicator",
            when(col("response_time_minutes").isNotNull(), lit(1)).otherwise(lit(0)).cast("int")
        )
        # Rename to match NYC schema
        .withColumnRenamed("response_time_minutes", "response_minutes")
        #.withColumnRenamed("_id", "incident_id")
        .withColumnRenamed("INCIDENT_NUMBER", "incident_id")
        # ADD: Delay indicator (8-minute threshold)
        .withColumn(
            "delay_indicator",
            when(col("response_minutes").isNull(), lit(None))
            .when(col("response_minutes") > lit(8.0), lit(1))
            .otherwise(lit(0))
            .cast("int")
        )
        .drop("calls_past_30m", "calls_past_60m")
        .select(
            "incident_id",
            "response_minutes",
            "delay_indicator",
            "event_indicator",
            "hour",
            "day_of_week",
            #"day",
            "month",
            "season",
            "Final_Incident_Type",
            "Event_Alarm_Level",
            "Call_Source",
            "Incident_Station_Area",
            "calls_past_30min",
            "calls_past_60min"
        )
    )