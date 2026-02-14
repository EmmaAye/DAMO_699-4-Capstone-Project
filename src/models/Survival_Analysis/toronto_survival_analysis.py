from pyspark.sql.functions import col, when
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from lifelines import CoxPHFitter
import pandas as pd

def load_and_clean_base(spark, table_name: str):
    """
    Loads table and returns cleaned Spark DF with required columns.
    Requires columns: response_minutes, event_indicator, hour
    """
    df = spark.read.table(table_name)

    df_clean = (
        df.select("response_minutes", "event_indicator", "hour", "day_of_week", "calls_past_30min", "calls_past_60min", "Final_Incident_Type", "Event_Alarm_Level" )
          .filter(col("response_minutes").isNotNull())
          .filter(col("event_indicator").isNotNull())
          .filter(col("response_minutes") > 0)
    )
    return df_clean


def km_overall(df_clean_spark, city_label="Toronto", save_path="km_overall.png"):
    """
    Plots overall KM curve and saves PNG.
    """
    df_pd = df_clean_spark.select("response_minutes", "event_indicator").toPandas()

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df_pd["response_minutes"],
        event_observed=df_pd["event_indicator"],
        label=city_label
    )

    plt.figure(figsize=(8, 6))
    kmf.plot_survival_function()
    plt.title(f"Kaplan–Meier Survival Curve - {city_label}")
    plt.xlabel("Minutes")
    plt.ylabel("Probability Unit Not Yet Arrived")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved:", save_path)
    print("Median time-to-arrival (KM):", kmf.median_survival_time_)

    return kmf


def km_by_time_of_day(df_clean_spark, city_label="Toronto", save_path="km_by_time.png"):
    """
    Plots KM curves by time-of-day bins and saves PNG.
    """
    df_time = df_clean_spark.withColumn(
        "time_bin",
        when((col("hour") >= 0) & (col("hour") <= 5), "Night")
        .when((col("hour") >= 6) & (col("hour") <= 11), "Morning")
        .when((col("hour") >= 12) & (col("hour") <= 17), "Afternoon")
        .otherwise("Evening")
    )

    df_pd = df_time.select("response_minutes", "event_indicator", "time_bin").toPandas()

    plt.figure(figsize=(8, 6))

    for group in ["Night", "Morning", "Afternoon", "Evening"]:
        sub = df_pd[df_pd["time_bin"] == group]
        if len(sub) == 0:
            print(f"No rows for group: {group}")
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=sub["response_minutes"],
            event_observed=sub["event_indicator"],
            label=group
        )
        kmf.plot_survival_function()

    plt.title(f"Kaplan–Meier by Time of Day - {city_label}")
    plt.xlabel("Minutes")
    plt.ylabel("Probability Unit Not Yet Arrived")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("Saved:", save_path)

def run_cox_model(df_clean_spark, city_label="Toronto", save_path="cox_summary.csv"):

    #Running Cox Proportional Hazards model.

    # Select required columns
    df_model = df_clean_spark.select(
        "response_minutes",
        "event_indicator",
        "hour",
        "day_of_week",
        "calls_past_30min",
        "calls_past_60min",
        #"Final_Incident_Type",
        "Event_Alarm_Level"
    )

    df_pd = df_model.toPandas()

    # Initialize Cox model
    cph = CoxPHFitter()

    # Fit model
    cph.fit(
        df_pd,
        duration_col="response_minutes",
        event_col="event_indicator"
    )
    cph.check_assumptions(df_pd, p_value_threshold=0.05, show_plots=False)
    
    # Print summary
    print(f"\nCox Model Summary - {city_label}")
    cph.print_summary()

    # Save results
    summary_df = cph.summary
    results_table = summary_df[[
    "coef",
    "exp(coef)",          # Hazard Ratio
    "p",
    "exp(coef) lower 95%",
    "exp(coef) upper 95%"
]]
    results_table = results_table.rename(columns={
    "coef": "Coefficient",
    "exp(coef)": "Hazard_Ratio",
    "p": "p_value",
    "exp(coef) lower 95%": "HR_lower_95CI",
    "exp(coef) upper 95%": "HR_upper_95CI"
})
    print("\nCox Results Table - Toronto\n")
    print(results_table)
    results_table.to_csv(save_path)

    print("Cox summary saved:", save_path)

    top = results_table.sort_values("p_value").head(10)
    print(top[["Hazard_Ratio", "p_value", "HR_lower_95CI", "HR_upper_95CI"]])
    return cph