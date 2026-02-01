import dlt
from pyspark.sql.functions import col, hour, date_format, count, sum, avg, round, expr,sum, when

@dlt.table(
    name="tfs_aggregate_gold",
    comment="Aggregated summary of Toronto fire incidents for analytics"
)
def tfs_aggregate_gold():
    return (
        dlt.read("tfs_incidents_silver")
        
        # 1. Create time-based features
        .withColumn("incident_hour", hour(col("alarm_time")))
        .withColumn("day_of_week", date_format(col("alarm_time"), "EEEE"))
        .withColumn("month_name", date_format(col("alarm_time"), "MMMM"))
        .withColumn("is_weekend", 
                    when(date_format(col("alarm_time"), "EEEE").isin(["Saturday", "Sunday"]), True)
                    .otherwise(False))
        
         # 3. Time of day categories
        .withColumn("time_of_day",
            when(col("incident_hour").between(0, 5), "Night (12AM-6AM)")
            .when(col("incident_hour").between(6, 11), "Morning (6AM-12PM)")
            .when(col("incident_hour").between(12, 17), "Afternoon (12PM-6PM)")
            .otherwise("Evening (6PM-12AM)"))
        
        # 4. Weekend indicator
        .withColumn("is_weekend", 
            when(col("day_of_week").isin(["Saturday", "Sunday"]), True)
            .otherwise(False))
        
        # AGGREGATION (this is what makes it Gold)
        .groupBy("day_of_week", "time_of_day", "is_weekend", "response_time_category")
        .agg(
            count("*").alias("total_incidents"),
            round(avg(col("response_time_minutes")), 2).alias("avg_response_time_minutes"),
            round(avg(col("incident_duration_minutes")), 2).alias("avg_incident_duration_minutes"),
            sum(when(col("Persons_Rescued") > 0, 1).otherwise(0)).alias("incidents_with_rescues"),
            sum(col("Persons_Rescued")).alias("total_persons_rescued")
        )

        # Dropping of duplicate columns
        .drop("TFS_Alarm_Time", "TFS_Arrival_Time", "Last_TFS_Unit_Clear_Time")
    )
