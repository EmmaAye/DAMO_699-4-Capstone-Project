import dlt
from pyspark.sql.types import *

# 1. Define the Schema based on your requirements
nyc_schema = StructType([
    StructField("STARFIRE_INCIDENT_ID", StringType(), True),
    StructField("INCIDENT_DATETIME", TimestampType(), True),
    StructField("ALARM_BOX_BOROUGH", StringType(), True),
    StructField("ALARM_BOX_NUMBER", IntegerType(), True),
    StructField("ALARM_BOX_LOCATION", StringType(), True),
    StructField("INCIDENT_BOROUGH", StringType(), True),
    StructField("ZIPCODE", StringType(), True),
    StructField("POLICEPRECINCT", IntegerType(), True),
    StructField("CITYCOUNCILDISTRICT", IntegerType(), True),
    StructField("COMMUNITYDISTRICT", IntegerType(), True),
    StructField("COMMUNITYSCHOOLDISTRICT", IntegerType(), True),
    StructField("CONGRESSIONALDISTRICT", IntegerType(), True),
    StructField("ALARM_SOURCE_DESCRIPTION_TX", StringType(), True),
    StructField("ALARM_LEVEL_INDEX_DESCRIPTION", StringType(), True),
    StructField("HIGHEST_ALARM_LEVEL", StringType(), True),
    StructField("INCIDENT_CLASSIFICATION", StringType(), True),
    StructField("INCIDENT_CLASSIFICATION_GROUP", StringType(), True),
    StructField("DISPATCH_RESPONSE_SECONDS_QY", IntegerType(), True),
    StructField("FIRST_ASSIGNMENT_DATETIME", TimestampType(), True),
    StructField("FIRST_ACTIVATION_DATETIME", TimestampType(), True),
    StructField("FIRST_ON_SCENE_DATETIME", TimestampType(), True),
    StructField("INCIDENT_CLOSE_DATETIME", TimestampType(), True),
    StructField("VALID_DISPATCH_RSPNS_TIME_INDC", StringType(), True),
    StructField("VALID_INCIDENT_RSPNS_TIME_INDC", StringType(), True),
    StructField("INCIDENT_RESPONSE_SECONDS_QY", IntegerType(), True),
    StructField("INCIDENT_TRAVEL_TM_SECONDS_QY", IntegerType(), True),
    StructField("ENGINES_ASSIGNED_QUANTITY", IntegerType(), True),
    StructField("LADDERS_ASSIGNED_QUANTITY", IntegerType(), True),
    StructField("OTHERS_UNITS_ASSIGNED_QUANTITY", IntegerType(), True)
])

# 2. Define the Bronze Streaming Table
@dlt.table(
    name="nyc_fire_incidents_bronze",
    comment="Raw ingestion of NYC Fire Incident Dispatch Data"
)
def nyc_fire_incidents_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("timestampFormat", "MM/dd/yyyy hh:mm:ss a") 
        .schema(nyc_schema)
        .load("/Volumes/workspace/capstone_project/bronze_layer/")
    )