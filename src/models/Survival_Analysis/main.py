# main.py
%pip install lifelines
from pyspark.sql import SparkSession
from toronto_survival_analysis import load_and_clean_base, km_overall, km_by_time_of_day


TABLE_NAME = "workspace.capstone_project.tfs_incidents_gold"


def main():
    spark = SparkSession.builder.getOrCreate()

    # Load + clean
    df_clean = load_and_clean_base(spark, TABLE_NAME)
    print("Rows after cleaning:", df_clean.count())
    df_clean.show(5)

    # Overall KM
    km_overall(
        df_clean_spark=df_clean,
        city_label="Toronto",
        save_path="/Workspace/Shared/DAMO_699-4-Capstone-Project/output/graph/km_toronto_overall.png"
    )

    # Time-of-day KM
    km_by_time_of_day(
        df_clean_spark=df_clean,
        city_label="Toronto",
        save_path="/Workspace/Shared/DAMO_699-4-Capstone-Project/output/graph/km_toronto_by_time.png"
    )


if __name__ == "__main__":
    main()
