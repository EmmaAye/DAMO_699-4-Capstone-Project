# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
# %pip install lifelines

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, hour, month, when, lit, least
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


# %%
TORONTO_TABLE = "workspace.capstone_project.toronto_model_ready"
NYC_TABLE     = "workspace.capstone_project.nyc_model_ready"

CENSOR_THRESHOLD = 60   # minutes
ALPHA = 0.05

SAVE_DIR = "/Workspace/Shared/DAMO_699-4-Capstone-Project/output/graphs"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Saving outputs to:", SAVE_DIR)


# %%
# -----------------------
# Prepare Toronto
# -----------------------
df_to = spark.read.table(TORONTO_TABLE).select(
    col("response_minutes").alias("duration_original"),
    col("event_indicator").alias("event_original"),
    col("hour"),
    col("season"),
    col("day_of_week")
)

df_to = df_to.where(
"duration_original is not null AND duration_original > 0 AND event_original is not null"
)

# Apply censoring: duration = min(duration_original, 60)
df_to = df_to.withColumn(
    "response_minutes",
    least(col("duration_original"), lit(float(CENSOR_THRESHOLD)))
)

# Apply censoring event rule:
# event = 1 if duration_original <= 60 AND event_original==1 else 0
df_to = df_to.withColumn(
    "event_indicator",
    when((col("duration_original") <= CENSOR_THRESHOLD) & (col("event_original") == 1), 1).otherwise(0)
)

# hour_group bins: Night 00–06, Morning 06–12, Afternoon 12–18, Evening 18–24
df_to = df_to.withColumn(
    "hour_group",
    when((col("hour") >= 0) & (col("hour") < 6), "Night")
    .when((col("hour") >= 6) & (col("hour") < 12), "Morning")
    .when((col("hour") >= 12) & (col("hour") < 18), "Afternoon")
    .otherwise("Evening")
)

df_to = df_to.withColumn(
    "day_of_week_name",
    when(col("day_of_week") == 1, "Sunday")
    .when(col("day_of_week") == 2, "Monday")
    .when(col("day_of_week") == 3, "Tuesday")
    .when(col("day_of_week") == 4, "Wednesday")
    .when(col("day_of_week") == 5, "Thursday")
    .when(col("day_of_week") == 6, "Friday")
    .when(col("day_of_week") == 7, "Saturday")
)

# -----------------------
# Prepare NYC
# -----------------------
df_nyc = spark.read.table(NYC_TABLE).select(
    col("response_minutes").alias("duration_original"),
    col("event_indicator").alias("event_original"),
    col("hour"),
    col("season"),
    col("day_of_week")
)

df_nyc = df_nyc.where(
"duration_original is not null AND duration_original > 0 AND event_original is not null"
)

# Apply censoring duration
df_nyc = df_nyc.withColumn(
    "response_minutes",
    least(col("duration_original"), lit(float(CENSOR_THRESHOLD)))
)

# Apply censoring event rule
df_nyc = df_nyc.withColumn(
    "event_indicator",
    when((col("duration_original") <= CENSOR_THRESHOLD) & (col("event_original") == 1), 1).otherwise(0)
)

# hour_group bins
df_nyc = df_nyc.withColumn(
    "hour_group",
    when((col("hour") >= 0) & (col("hour") < 6), "Night")
    .when((col("hour") >= 6) & (col("hour") < 12), "Morning")
    .when((col("hour") >= 12) & (col("hour") < 18), "Afternoon")
    .otherwise("Evening")
)
df_nyc = df_nyc.withColumn(
    "day_of_week_name",
    when(col("day_of_week") == 1, "Sunday")
    .when(col("day_of_week") == 2, "Monday")
    .when(col("day_of_week") == 3, "Tuesday")
    .when(col("day_of_week") == 4, "Wednesday")
    .when(col("day_of_week") == 5, "Thursday")
    .when(col("day_of_week") == 6, "Friday")
    .when(col("day_of_week") == 7, "Saturday")
)

print("Toronto rows:", df_to.count())
print("NYC rows:", df_nyc.count())


# %%
def plot_km_and_test(df_spark, city_name, group_col, strat_label, group_order=None):
    
    df_pd = df_spark.select(
        "response_minutes",
        "event_indicator",
        group_col
    ).toPandas()
    
    plt.figure(figsize=(8, 6))
    
    if group_order is None:
        groups = sorted(df_pd[group_col].dropna().astype(str).unique())
    else:
        groups = group_order
    
    for g in groups:
        sub = df_pd[df_pd[group_col].astype(str) == str(g)]
        if len(sub) == 0:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(sub["response_minutes"], sub["event_indicator"], label=str(g))
        kmf.plot_survival_function()
    
    plt.title(f"{city_name} – KM by {strat_label}")
    plt.xlabel("Response Time (minutes)")
    plt.ylabel("Survival Probability")
    plt.xlim(0, CENSOR_THRESHOLD)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    save_path = f"{SAVE_DIR}/{city_name.lower()}_km_by_{strat_label.lower().replace(' ','_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print("Saved:", save_path)
    
    # Log-rank test
    result = multivariate_logrank_test(
        df_pd["response_minutes"],
        df_pd[group_col],
        df_pd["event_indicator"]
    )
    
    pval = float(result.p_value)
    significant = pval < ALPHA
    
    print(f"{city_name} – Log-rank p-value ({strat_label}):", pval)
    
    return {
        "City": city_name,
        "Stratification": strat_label,
        "Test": "Log-rank",
        "p_value": pval,
        "Significant?": significant
    }



# %%
results = []

hour_order = ["Night","Morning","Afternoon","Evening"]
season_order = ["winter","spring","summer","fall"]
dow_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# Toronto
results.append(plot_km_and_test(df_to, "Toronto", "hour_group", "Hour", hour_order))
results.append(plot_km_and_test(df_to, "Toronto", "season", "Season", season_order))
results.append(plot_km_and_test(df_to, "Toronto", "day_of_week_name", "Day of Week", dow_order))

# NYC
results.append(plot_km_and_test(df_nyc, "NYC", "hour_group", "Hour", hour_order))
results.append(plot_km_and_test(df_nyc, "NYC", "season", "Season", season_order))
results.append(plot_km_and_test(df_nyc, "NYC", "day_of_week_name", "Day of Week", dow_order))


# %%
def logrank_summary(df_spark, city, strat_col):
    df_pd = df_spark.select("response_minutes", "event_indicator", strat_col).toPandas()

    res = multivariate_logrank_test(
        df_pd["response_minutes"],
        df_pd[strat_col],
        df_pd["event_indicator"]
    )

    pval = float(res.p_value)
    significant = pval < ALPHA

    # Higher-risk group (tail @60): highest S(60)
    risk_group = None
    if significant:
        s_at_60 = {}
        for g in sorted(df_pd[strat_col].dropna().astype(str).unique()):
            sub = df_pd[df_pd[strat_col].astype(str) == g]
            if len(sub) == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["response_minutes"], sub["event_indicator"])
            s_at_60[g] = float(kmf.predict(CENSOR_THRESHOLD))

        if len(s_at_60) > 0:
            risk_group = max(s_at_60, key=s_at_60.get)

    # Safe label mapping
    if strat_col == "hour_group":
        label = "hour"
    elif strat_col == "season":
        label = "season"
    elif strat_col in ["day_of_week", "day_of_week_norm", "day_of_week_name"]:
        label = "day of week"
    else:
        label = strat_col

    return {
        "city": city,
        "stratification": label,
        "test": "Log-rank",
        "p_value": pval,
        "significant": significant,
        "higher-risk group (tail @60)": risk_group
    }



# %%
results = []

results.append(logrank_summary(df_to,  "Toronto", "hour_group"))
results.append(logrank_summary(df_to,  "Toronto", "season"))
results.append(logrank_summary(df_to,  "Toronto", "day_of_week"))

results.append(logrank_summary(df_nyc, "NYC",     "hour_group"))
results.append(logrank_summary(df_nyc, "NYC",     "season"))
results.append(logrank_summary(df_nyc, "NYC",     "day_of_week"))

summary_df = pd.DataFrame(results)
display(summary_df)
summary_path = f"/Workspace/Shared/DAMO_699-4-Capstone-Project/output/tables/logrank_summary_within_city.csv"
summary_df.to_csv(summary_path, index=False)
print("Saved summary:", summary_path)


# %%
print(summary_df.columns.tolist())

# %%
print("----- US4.2 Interpretation Summary -----\n")

risk_col = "Higher-risk group (tail @60)"
has_risk = risk_col in summary_df.columns

for _, row in summary_df.iterrows():
    city = row["city"]
    strat = row["stratification"]
    pval = row["p_value"]
    sig  = row["significant"]
    risk = row["higher-risk group (tail @60)"]

    risk = row[risk_col] if has_risk else None

    if sig:
        print(f"{city} – {strat}:")
        print("Log-rank test indicates significant differences (p < 0.05).")

        if has_risk and pd.notna(risk):
            print(f"Higher delay-risk group (longer tail at 60 min): {risk}.")

        print("Operational insight: Differences suggest temporal variation in response performance.")
        print("Check whether curves diverge early (general speed) or mainly in the tail (extreme delays).")
        print("")
    else:
        print(f"{city} – {strat}: No statistically significant differences detected.\n")

