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