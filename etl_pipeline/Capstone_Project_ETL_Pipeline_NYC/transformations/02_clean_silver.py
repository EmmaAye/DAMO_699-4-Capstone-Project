import dlt
from pyspark.sql.functions import col, when, upper, trim

@dlt.table(
    name="nyc_fire_incidents_silver",
    comment="Cleaned NYC Fire data with standardized types and quality rules"
)
@dlt.expect_or_drop("valid_incident_date", "INCIDENT_DATETIME IS NOT NULL")
@dlt.expect_or_drop("valid_zipcode", "ZIPCODE IS NOT NULL")
@dlt.expect_or_drop(
    "positive_response_time", 
    "INCIDENT_RESPONSE_SECONDS_QY >= 0 OR INCIDENT_RESPONSE_SECONDS_QY IS NULL"
)
@dlt.expect_or_drop(
    "valid_response_and_travel_times", 
    "INCIDENT_RESPONSE_SECONDS_QY IS NOT NULL AND INCIDENT_TRAVEL_TM_SECONDS_QY IS NOT NULL"
)
@dlt.expect_or_drop(
    "logical_time_flow", 
    "INCIDENT_CLOSE_DATETIME IS NULL OR INCIDENT_CLOSE_DATETIME >= INCIDENT_DATETIME"
)
def nyc_fire_incidents_silver():
    return (
        dlt.read("nyc_fire_incidents_bronze")
        
        # 1. Convert 'Y/N' Strings to BooleanType
        .withColumn("VALID_DISPATCH_RSPNS_TIME_INDC", 
            when(col("VALID_DISPATCH_RSPNS_TIME_INDC") == "Y", True) 
            .when(col("VALID_DISPATCH_RSPNS_TIME_INDC") == "N", False))
        .withColumn("VALID_INCIDENT_RSPNS_TIME_INDC", 
            when(col("VALID_INCIDENT_RSPNS_TIME_INDC") == "Y", True) 
            .when(col("VALID_INCIDENT_RSPNS_TIME_INDC") == "N", False))

        # 2. Handle Missing Values for critical columns
        .fillna({"ALARM_SOURCE_DESCRIPTION_TX": "UNKNOWN", "HIGHEST_ALARM_LEVEL": "0"})
        
        # 3. Standardize Borough names to uppercase for consistent grouping
        .withColumn("INCIDENT_BOROUGH", upper(trim(col("INCIDENT_BOROUGH"))))

        # 4. Deduplication
        .dropDuplicates(["STARFIRE_INCIDENT_ID"])
        .dropDuplicates(["INCIDENT_DATETIME", "ALARM_BOX_NUMBER", "INCIDENT_BOROUGH"])
    )