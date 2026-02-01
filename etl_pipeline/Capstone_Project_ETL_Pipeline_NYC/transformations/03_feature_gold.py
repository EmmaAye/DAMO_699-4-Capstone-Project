import dlt
from pyspark.sql.functions import col, when, hour, dayofweek, month, count, unix_timestamp
from pyspark.sql.window import Window

@dlt.table(
    name="nyc_fire_incidents_gold",
    comment="Gold feature table for NYC Fire analytics with lowercase column names"
)
# Ensuring data quality for the final features
@dlt.expect_or_drop("valid_response", "response_minutes IS NOT NULL")
def nyc_fire_incidents_gold():
    # Define the windows for past call counts (in seconds: 30m = 1800s, 60m = 3600s)
    # We partition by Borough to get localized traffic/demand
    window_30 = Window.partitionBy("INCIDENT_BOROUGH").orderBy(unix_timestamp("INCIDENT_DATETIME")).rangeBetween(-1800, -1)
    window_60 = Window.partitionBy("INCIDENT_BOROUGH").orderBy(unix_timestamp("INCIDENT_DATETIME")).rangeBetween(-3600, -1)

    return (
        dlt.read("nyc_fire_incidents_silver")
        
        # 1. Map and Calculate Base Features
        .withColumn("incident_id", col("STARFIRE_INCIDENT_ID"))
        .withColumn("response_minutes", col("INCIDENT_RESPONSE_SECONDS_QY") / 60)
        .withColumn("event_indicator", col("VALID_INCIDENT_RSPNS_TIME_INDC"))
        
        # 2. Extract Time Components
        .withColumn("hour", hour(col("INCIDENT_DATETIME")))
        .withColumn("day_of_week", dayofweek(col("INCIDENT_DATETIME")))
        .withColumn("month", month(col("INCIDENT_DATETIME")))
        
        # 3. Season Logic
        .withColumn("season", 
            when(col("month").isin(12, 1, 2), "winter")
            .when(col("month").isin(3, 4, 5), "spring")
            .when(col("month").isin(6, 7, 8), "summer")
            .otherwise("fall"))
        
        # 4. Window Functions for Call Volume
        .withColumn("calls_past_30min", count("STARFIRE_INCIDENT_ID").over(window_30))
        .withColumn("calls_past_60min", count("STARFIRE_INCIDENT_ID").over(window_60))
        
        # 5. Select and Lowercase all final columns
        .select(
            col("incident_id"),
            col("response_minutes"),
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