import dlt
from pyspark.sql.functions import col,round, when

# SIMPLE SILVER LAYER
@dlt.table(
    name="tfs_incidents_silver",
    comment="Cleaned and cleaned up the raw data of Toronto Fire Incidents"
)

@dlt.expect_or_drop("valid_incident_number", "INCIDENT_NUMBER IS NOT NULL")
@dlt.expect_or_drop("valid_alarm_level", "Event_Alarm_Level BETWEEN 0 AND 5")
@dlt.expect_or_drop("non_negative_rescues", "Persons_Rescued >= 0")
@dlt.expect_or_drop("has_geometry", "geometry IS NOT NULL")

def tfs_incidents_silver():
    df = (
        dlt.read("tfs_incidents_bronze")
        
        # 1. Simple timestamp conversion
        .withColumn("alarm_time", col("TFS_Alarm_Time").cast("timestamp"))
        .withColumn("arrival_time", col("TFS_Arrival_Time").cast("timestamp"))
        .withColumn("clear_time", col("Last_TFS_Unit_Clear_Time").cast("timestamp"))
                
        # 7. Remove duplicates
        .dropDuplicates(["INCIDENT_NUMBER", "alarm_time"])

        # ========== RESPONSE TIME CALCULATIONS ==========
        # Calculate response time in SECONDS
        .withColumn("response_time_seconds",
            (col("arrival_time").cast("long") - col("alarm_time").cast("long")))
        
        # Calculate response time in MINUTES
        .withColumn("response_time_minutes",
            round(col("response_time_seconds") / 60.0, 2))
        
        # Calculate total incident duration in minutes
        .withColumn("incident_duration_minutes",
            round((col("clear_time").cast("long") - col("alarm_time").cast("long")) / 60.0, 2))
        
        # Response time categories
        .withColumn("response_time_category",
            when(col("response_time_minutes") < 5, "Excellent (<5 min)")
            .when(col("response_time_minutes") < 10, "Good (5-10 min)")
            .when(col("response_time_minutes") < 15, "Fair (10-15 min)")
            .otherwise("Poor (>15 min)"))
        .drop("Ward_At_Event_Dispatch")
    )
    return df