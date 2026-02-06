import dlt
from pyspark.sql.functions import col, when, hour, dayofweek, month, count, unix_timestamp, lit, round
from pyspark.sql.window import Window

@dlt.table(
    name="nyc_fire_incidents_gold",
    comment="Gold modeling feature table for NYC Fire analytics (incident-level)"
)
# Ensuring data quality for the final features
def nyc_fire_incidents_gold():
    # Citywide demand (remove partitionBy for system load)
    window_30 = Window.orderBy(unix_timestamp("INCIDENT_DATETIME")).rangeBetween(-1800, -1)
    window_60 = Window.orderBy(unix_timestamp("INCIDENT_DATETIME")).rangeBetween(-3600, -1)

    return (
        dlt.read("nyc_fire_incidents_silver")
        
        # 1. Map and Calculate Base Features
        .withColumn("incident_id", col("STARFIRE_INCIDENT_ID"))
        .withColumn("response_minutes", round((col("INCIDENT_RESPONSE_SECONDS_QY") / 60).cast("double"), 2))
        # Delay indicator (8-minute threshold)
        .withColumn(
            "delay_indicator",
            when(col("response_minutes").isNull(), lit(None))
            .when(col("response_minutes") > lit(8.0), lit(1))
            .otherwise(lit(0))
            .cast("int")
        )
        
        # 2. Survival event indicator: 1 if response time observed else 0
        .withColumn("event_indicator", when(col("INCIDENT_RESPONSE_SECONDS_QY").isNotNull(), lit(1)).otherwise(lit(0)).cast("int"))
        
        # 3. Extract Time Components
        .withColumn("hour", hour(col("INCIDENT_DATETIME")))
        .withColumn("day_of_week", dayofweek(col("INCIDENT_DATETIME")))
        .withColumn("month", month(col("INCIDENT_DATETIME")))
        
        # 4. Season Logic
        .withColumn("season", 
            when(col("month").isin(12, 1, 2), "winter")
            .when(col("month").isin(3, 4, 5), "spring")
            .when(col("month").isin(6, 7, 8), "summer")
            .otherwise("fall"))
        
        # 5. Window Functions for Call Volume
        .withColumn("calls_past_30min", count("*").over(window_30))
        .withColumn("calls_past_60min", count("*").over(window_60))
        
        # 6. Select and Lowercase all final columns
        .select(
            col("incident_id"),
            col("response_minutes"),
            col("delay_indicator"),
            col("event_indicator"),
            col("hour"),
            col("day_of_week"),
            col("month"),
            col("season"),
            col("INCIDENT_CLASSIFICATION").alias("incident_classification"),
            col("INCIDENT_CLASSIFICATION_GROUP").alias("incident_classification_group"),
            col("ALARM_LEVEL_INDEX_DESCRIPTION").alias("alarm_level_index_description"),
            col("ALARM_SOURCE_DESCRIPTION_TX").alias("alarm_source_description_tx"),
            col("INCIDENT_BOROUGH").alias("incident_borough"),
            col("calls_past_30min"),
            col("calls_past_60min")
        )
    )
