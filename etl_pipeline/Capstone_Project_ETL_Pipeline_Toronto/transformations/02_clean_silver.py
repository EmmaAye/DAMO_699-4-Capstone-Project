import dlt
from pyspark.sql.functions import col,round, when,to_timestamp,coalesce

# SIMPLE SILVER LAYER
@dlt.table(
    name="tfs_incidents_silver",
    comment="Cleaned and cleaned up the raw data of Toronto Fire Incidents"
)

@dlt.expect_or_drop("valid_incident_number", "INCIDENT_NUMBER IS NOT NULL")
@dlt.expect_or_drop("valid_alarm_level", "Event_Alarm_Level BETWEEN 0 AND 5")
#@dlt.expect_or_drop("non_negative_rescues", "Persons_Rescued >= 0")
@dlt.expect_or_drop("has_geometry", "geometry IS NOT NULL")
@dlt.expect_or_drop(
"non_negative_response_time",
"response_time_seconds IS NULL OR response_time_seconds >= 0"
)
@dlt.expect("alarm_time_parsed", "alarm_time IS NOT NULL")

def tfs_incidents_silver():
    ts_formats = [
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd'T'HH:mm:ss",
    "MM/dd/yyyy HH:mm:ss",
    "MM/dd/yyyy h:mm a"
]
    df = (
        dlt.read("tfs_incidents_bronze")
        
        # 1. Simple timestamp conversion
    .withColumn
    (
        "alarm_time",
         coalesce(*[to_timestamp(col("TFS_Alarm_Time"), f) for f in ts_formats]
        )
    )
    .withColumn(
        "arrival_time",
         coalesce(*[to_timestamp(col("TFS_Arrival_Time"), f) for f in ts_formats]
        )
    )
    .withColumn(
        "clear_time",
        coalesce(
        *[to_timestamp(col("Last_TFS_Unit_Clear_Time"), f) for f in ts_formats]
         )
    )          
        # 7. Remove duplicates
        .dropDuplicates(["INCIDENT_NUMBER", "alarm_time"])

        # ========== RESPONSE TIME CALCULATIONS ==========
        # Calculate response time in SECONDS
        .withColumn(
            "response_time_seconds",
            when(
                col("arrival_time").isNull() | col("alarm_time").isNull(),
                None
            ).otherwise(
                col("arrival_time").cast("long") - col("alarm_time").cast("long")
            )
        )
        
        # Calculate response time in MINUTES
        .withColumn("response_time_minutes",
            round(col("response_time_seconds") / 60.0, 2))
        
        # Calculate total incident duration in minutes
        .withColumn(
            "response_time_minutes",
            when(
                col("response_time_seconds").isNull(),
                None
            ).otherwise(
                round(col("response_time_seconds") / 60.0, 2)
            )
        )
        
        # Response time categories
        .withColumn("response_time_category",
            when(col("response_time_minutes") < 5, "Excellent (<5 min)")
            .when(col("response_time_minutes") < 10, "Good (5-10 min)")
            .when(col("response_time_minutes") < 15, "Fair (10-15 min)")
            .otherwise("Poor (>15 min)"))
       
        # Dropping of unnecessary columns
        .drop("Ward_At_Event_Dispatch")

        # Dropping duplicate raw columns 
        .drop("TFS_Alarm_Time", "TFS_Arrival_Time", "Last_TFS_Unit_Clear_Time")    
    )
    return df