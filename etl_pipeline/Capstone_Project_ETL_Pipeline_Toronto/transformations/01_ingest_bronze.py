import dlt
from pyspark.sql.types import *

# 1. Define the Schema for Toronto Fire Data
tfs_schema = StructType([
    StructField("_id", IntegerType(), True),
    StructField("INCIDENT_NUMBER", StringType(), True),
    StructField("Initial_CAD_Event_Type", StringType(), True),
    StructField("Initial_CAD_Event_Call_Type", StringType(), True),
    StructField("Final_Incident_Type", StringType(), True),
    StructField("Event_Alarm_Level", IntegerType(), True),
    StructField("Call_Source", StringType(), True),
    StructField("Incident_Station_Area", StringType(), True),
    StructField("Incident_Ward", IntegerType(), True),
    StructField("Ward_At_Event_Dispatch", StringType(), True), # Some values are null
    StructField("Intersection", StringType(), True),
    StructField("TFS_Alarm_Time", StringType(), True),   # Keep as string for Bronze
    StructField("TFS_Arrival_Time", StringType(), True), # Keep as string for Bronze
    StructField("Last_TFS_Unit_Clear_Time", StringType(), True),
    StructField("Persons_Rescued", IntegerType(), True),
    StructField("geometry", StringType(), True)          # JSON geometry string
])

# 2. Define the Bronze Streaming Table
@dlt.table(
    name="tfs_incidents_bronze",
    comment="Raw ingestion of Toronto Fire Incident Data from Volumes"
)
def tfs_incidents_bronze():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        # Schema location for Auto Loader
        .option("cloudFiles.schemaLocation", "dbfs:/Volumes/workspace/capstone_project/trt_bronze_layer")
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .option("quote", '"')
        .option("escape", '"')
        .schema(tfs_schema)
        .load("dbfs:/Volumes/workspace/capstone_project/trt_bronze_layer/")
    )