# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Toronto–NYC Gold Merge & EDA (Unified Dataset)
#
# This notebook loads Gold-layer outputs for Toronto and NYC fire incidents, harmonizes them to a shared schema, merges them into a unified dataset with a `city` indicator, and performs comparative EDA (distribution, missingness, categorical profiles, and tail-risk percentiles).
#
#

# %% [markdown]
# ## Run Order (Required)
#
# 1. Run Toronto pipeline `Capstone_ETL_TRT` to refresh Gold tables.
# 2. Run NYC pipeline `Capstone_ETL_NYC` to refresh Gold tables.
# 3. Run this notebook top-to-bottom to generate:
#    - `toronto_gold_feat`, `nyc_gold_feat`
#    - `toronto_model_ready`, `nyc_model_ready`
# 4. Proceed to EDA / modeling notebooks using the model-ready tables.

# %% [markdown]
# ## Inputs
# - Toronto Gold: `workspace.capstone_project.tfs_incidents_gold`
# - NYC Gold: `workspace.capstone_project.nyc_fire_incidents_gold`
#
# ## Outputs
# - `workspace.capstone_project.toronto_model_ready`
# - `workspace.capstone_project.nyc_model_ready`
#

# %% [markdown]
# ## 1. Import and Load Tables

# %%
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import DataFrame

# --- DATA TABLE DIRECTORIES ---
TORONTO_GOLD_TABLE = "workspace.capstone_project.tfs_incidents_gold"   
NYC_GOLD_TABLE     = "workspace.capstone_project.nyc_fire_incidents_gold"



# %% [markdown]
# ### 1.1 Load Toronto Table

# %%
toronto_gold = spark.table(TORONTO_GOLD_TABLE)

print("Toronto Gold count:", toronto_gold.count())

display(toronto_gold.limit(5))

toronto_gold.printSchema()

# %% [markdown]
# ### 1.2 Load NYC Table

# %%
nyc_gold     = spark.table(NYC_GOLD_TABLE)

print("NYC Gold count:", nyc_gold.count())

display(nyc_gold.limit(5))

nyc_gold.printSchema()


# %% [markdown]
# Copy Spark Data Frame before Changing

# %%
toronto_gold_feat = toronto_gold
nyc_gold_feat     = nyc_gold


# %%
print("NYC years")
display(nyc_gold_feat.select("year").distinct().orderBy("year"))
print("Toronto years")
display(toronto_gold_feat.select("year").distinct().orderBy("year"))

# %% [markdown]
# #### Filter NYC Data For Year 2023, 2024 to match Toronto Data

# %%
nyc_gold_feat = nyc_gold_feat.filter(F.col("year").isin(2023, 2024))

# %%
print("NYC years")
display(nyc_gold_feat.select("year").distinct().orderBy("year"))

# %%
print("NYC Filtered Gold count:", nyc_gold_feat.count())

# %%
print("Toronto Gold count:", toronto_gold_feat.count())

# %% [markdown]
# ## 2. Incident Type Harmonization
#
# Incident type definitions differ substantially between Toronto and NYC. Toronto reports fine-grained incident types, while NYC reports a small number of aggregated categories. To enable valid cross-city comparison, incident types are harmonized into shared high-level categories.
#
# **Toronto**
# <br>Input Col : `Final_Incident_Type`
# <br>Output Col: `incident_category`
#
# **NYC**
# <br>Input col: `INCIDENT_CLASSIFICATION`
# <br>Output col: `incident_category`

# %% [markdown]
# ### 2.1 Pre-Inspection: Raw Incident Type Distributions
#
# Before harmonization, we inspect all distinct incident type values and their frequencies in each city. This ensures the harmonization logic is grounded in observed data and covers dominant categories.
#
#

# %% [markdown]
# #### 2.1.1 Toroto Data Inspection

# %% [markdown]
# Check NULL or Empty Incident Types

# %%
total_rows = toronto_gold_feat.count()

null_or_empty = (
    toronto_gold_feat
    .filter(
        F.col("Final_Incident_Type").isNull() |
        (F.trim(F.col("Final_Incident_Type")) == "")
    )
    .count()
)

print(f"Toronto Null or empty Final_Incident_Type: {null_or_empty} ({null_or_empty/total_rows:.2%})")

# %% [markdown]
# Get Distinct Count

# %%
toronto_distinct_count = (
    toronto_gold_feat
    .select("Final_Incident_Type")
    .distinct()
    .count()
)

print("Toronto distinct Final_Incident_Type count:", toronto_distinct_count)

# %% [markdown]
# List Distinct Values

# %%
# Full distribution (all categories, sorted by frequency)
toronto_incident_dist = (
    toronto_gold_feat
    .groupBy("Final_Incident_Type")
    .count()
    .orderBy(F.desc("count"))
)

display(toronto_incident_dist)

# %% [markdown]
# #### 2.1.2 NYC Data Inspection

# %% [markdown]
# Check NULL or Empty Incidents Types

# %%
total_rows_nyc = nyc_gold_feat.count()

null_or_empty_nyc = (
    nyc_gold_feat
    .filter(
        F.col("INCIDENT_CLASSIFICATION").isNull() |
        (F.trim(F.col("INCIDENT_CLASSIFICATION")) == "")
    )
    .count()
)

print(
    "NYC null or empty INCIDENT_CLASSIFICATION:",
    null_or_empty_nyc,
    "(",
    null_or_empty_nyc / total_rows_nyc,
    ")"
)

# %% [markdown]
# Get Distinct Count

# %%
nyc_distinct_count = (
    nyc_gold_feat
    .select("INCIDENT_CLASSIFICATION")
    .distinct()
    .count()
)

print("NYC distinct INCIDENT_CLASSIFICATION count:", nyc_distinct_count)

# %% [markdown]
# List Distinct Values

# %%
nyc_incident_dist = (
    nyc_gold_feat
    .groupBy("INCIDENT_CLASSIFICATION")
    .count()
    .orderBy(F.desc("count"))
)

display(nyc_incident_dist)

print(
    "NYC distinct INCIDENT_CLASSIFICATION count:",
    nyc_gold_feat.select("INCIDENT_CLASSIFICATION").distinct().count()
)


# %% [markdown]
# ### 2.2 Incident Coding: Unified Category Design
#
# Based on the pre-inspection, raw incident types are grouped into the following unified categories:
#
# - Medical  
# - Fire – Structural  
# - Fire – Non-Structural  
# - Rescue / Entrapment  
# - Hazardous / Utility  
# - False Alarm / No Action  
# - Other / Assistance  
#
# This taxonomy preserves operational meaning while accommodating differences in reporting granularity between cities.
#

# %% [markdown]
# #### 2.2.1 Define Mapping Function
#
# This function standardizes raw incident-type labels from Toronto and NYC into a common set of high-level incident categories. It applies explicit overrides for known edge cases first, followed by general pattern-matching rules, ensuring consistent, defensible classification across both datasets for downstream modeling and cross-city comparison.

# %%
# 1. Define a function to map the incident category to the unified incident category
# Toronto Data Mapping Function
def map_toronto_incident_category(col_expr):
    """
    Unified 7-category incident taxonomy for Toronto + NYC.

    Categories:
    - Medical
    - Fire – Structural
    - Fire – Non-Structural
    - Rescue / Entrapment
    - Hazardous / Utility
    - False Alarm / No Action
    - Other / Assistance

    Explicit overrides are included to match the user's "Correct Classification" table.
    """
    s = F.upper(F.trim(col_expr))

    return (
        # =========================================================
        # 0) NYC explicit mapping (since NYC is already aggregated)
        # =========================================================
        F.when(s.rlike(r"^MEDICAL EMERGENCIES$|^MEDICAL MFAS$"), F.lit("Medical"))
         .when(s.rlike(r"^STRUCTURAL FIRES$"), F.lit("Fire – Structural"))
         .when(s.rlike(r"^NONSTRUCTURAL FIRES$"), F.lit("Fire – Non-Structural"))
         .when(s.rlike(r"^NONMEDICAL MFAS$"), F.lit("Rescue / Entrapment"))
         .when(s.rlike(r"^NONMEDICAL EMERGENCIES$"), F.lit("Other / Assistance"))

        # =========================================================
        # 1) Toronto explicit overrides (to match Correct Classification)
        # =========================================================

        # Must be Other / Assistance (NOT false alarm)
         .when(s.rlike(r"^\s*98\s*-\s*ASSISTANCE NOT REQUIRED"), F.lit("Other / Assistance"))

        # Must be Other / Assistance (NOT Fire – Non-Structural)
         .when(s.rlike(r"^\s*23\s*-\s*OPEN AIR BURNING/UNAUTHORIZED CONTROLLED BURNING"), F.lit("Other / Assistance"))
         .when(s.rlike(r"^\s*21\s*-\s*OVERHEAT"), F.lit("Other / Assistance"))
         .when(s.rlike(r"^\s*29\s*-\s*OTHER PRE FIRE CONDITIONS"), F.lit("Other / Assistance"))

        # Must be Hazardous / Utility
         .when(s.rlike(r"^\s*49\s*-\s*RUPTURED WATER,\s*STEAM PIPE"), F.lit("Hazardous / Utility"))
         .when(s.rlike(r"^\s*11\s*-\s*OVERPRESSURE RUPTURE"), F.lit("Hazardous / Utility"))
         .when(s.rlike(r"^\s*13\s*-\s*OVERPRESSURE RUPTURE\s*-\s*GAS PIPE"), F.lit("Hazardous / Utility"))
         .when(s.rlike(r"^\s*53\s*-\s*CO INCIDENT,\s*CO PRESENT"), F.lit("Hazardous / Utility"))
         .when(s.rlike(r"^\s*48\s*-\s*RADIO-?ACTIVE MATERIAL PROBLEM"), F.lit("Hazardous / Utility"))

        # Must be Rescue / Entrapment
         .when(s.rlike(r"^\s*69\s*-\s*OTHER RESCUE"), F.lit("Rescue / Entrapment"))
         .when(s.rlike(r"^\s*605\s*-\s*ANIMAL RESCUE"), F.lit("Rescue / Entrapment"))
         .when(s.rlike(r"^\s*68\s*-\s*WATER ICE RESCUE"), F.lit("Rescue / Entrapment"))

        # Must be Other / Assistance
         .when(s.rlike(r"^\s*54\s*-\s*SUSPICIOUS SUBSTANCE"), F.lit("Other / Assistance"))
         .when(s.rlike(r"^\s*26\s*-\s*FIREWORKS\s*\(NO FIRE\)"), F.lit("Other / Assistance"))

        # =========================================================
        # 2) General rules (Toronto + any remaining strings)
        # =========================================================

        # Medical
         .when(s.rlike(r"\bMEDICAL\b|\bEMS\b"), F.lit("Medical"))

        # False Alarm / No Action (alarm/cancelled/not found/CO false alarm/prank)
         .when(
             s.rlike(
                 r"\bALARM\b|ALARM SYSTEM|ALARM EQUIPMENT|MALFUNCTION|ACCIDENTAL ACTIVATION|"
                 r"PERCEIVED EMERGENCY|PRANK|MALICIOUS|"
                 r"CO FALSE ALARM|NO CO PRESENT|"
                 r"INCIDENT NOT FOUND|CANCELLED ON ROUTE|CANCELLED|"
                 r"PUBLIC HAZARD CALL FALSE ALARM|PUBLIC HAZARD NO ACTION REQUIRED|"
                 r"RESCUE FALSE ALARM|RESCUE NO ACTION REQUIRED|"
                 r"NO ACTION REQUIRED"
             ),
             F.lit("False Alarm / No Action")
         )

        # Fire – Structural
         .when(s.rlike(r"\b01\s*-\s*FIRE\b|STRUCTURAL FIRE|STRUCTURE FIRE"), F.lit("Fire – Structural"))

        # Fire – Non-Structural (keep cooking/smoke/pot on stove/outdoor fire codes; exclude the overridden ones above)
         .when(
             s.rlike(
                 r"NONSTRUCTURAL FIRE|NO LOSS OUTDOOR FIRE|OUTDOOR FIRE|"
                 r"COOKING|TOASTING|SMOKE|STEAM|"
                 r"POT ON STOVE|STOVE"
             ),
             F.lit("Fire – Non-Structural")
         )

        # Rescue / Entrapment
         .when(
             s.rlike(
                 r"PERSONS TRAPPED|ENTRAPMENT|ELEVATOR|EXTRICATION|"
                 r"WATER RESCUE|HIGH ANGLE|LOW ANGLE|CONFINED SPACE|TRENCH|"
                 r"\b691\b\s*-\s*PERSONAL/INDUSTRIAL ENTRAPMENT"
             ),
             F.lit("Rescue / Entrapment")
         )

        # Hazardous / Utility (gas leaks, spills, power lines, CO present, radiation, etc.)
         .when(
             s.rlike(
                 r"GAS LEAK|NATURAL GAS|PROPANE|REFRIGERATION|"
                 r"\bCO INCIDENT\b|CO PRESENT|CARBON MONOXIDE|"
                 r"HAZMAT|HAZARDOUS|"
                 r"SPILL|TOXIC CHEMICAL|GASOLINE|FUEL|"
                 r"POWER LINES|ARCI?NG|"
                 r"RADIO-?ACTIVE|RADIATION|"
                 r"OVERPRESSURE RUPTURE|GAS PIPE|"
                 r"RUPTURED WATER|STEAM PIPE"
             ),
             F.lit("Hazardous / Utility")
         )

        # Fallback
         .otherwise(F.lit("Other / Assistance"))
    )


# %%
# NYC Data Mapping Function
def map_nyc_incident_category(incident_classification_col):
    s = F.upper(F.trim(incident_classification_col))

    # ---------- 1) Medical ----------
    is_medical = s.rlike(r"^MEDICAL\s*-|^MEDICAL MFA\s*-")

    # ---------- 2) False Alarm / No Action ----------
    # Alarm / sprinkler / unnecessary/testing/defective categories + private fire alarm MFA
    is_false_alarm = (
        s.rlike(
            r"^ALARM SYSTEM\s*-|"
            r"^SPRINKLER SYSTEM\s*-|"
            r"NON-MEDICAL 10-91\s*\(UNNECESSARY ALARM\)"
        )
        | s.rlike(r"NON-MEDICAL MFA\s*-\s*PRIVATE FIRE ALARM")
    )

    # ---------- 3) Fire – Structural ----------
    # Building / structure fires + special structural escalation cases
    is_fire_structural = s.rlike(
        r"PRIVATE DWELLING FIRE|"
        r"MULTIPLE DWELLING\s*'A'\s*-.*FIRE|"
        r"MULTIPLE DWELLING\s*'B'\s*FIRE|"
        r"OTHER COMMERCIAL BUILDING FIRE|"
        r"STORE FIRE|SCHOOL FIRE|HOSPITAL FIRE|CHURCH FIRE|FACTORY FIRE|"
        r"OTHER PUBLIC BUILDING FIRE|THEATER OR TV STUDIO FIRE|"
        r"TRANSIT SYSTEM\s*-\s*STRUCTURAL|"
        r"UNDER CONTRUCTION\s*/\s*VACANT FIRE|UNDER CONSTRUCTION\s*/\s*VACANT FIRE|"
        r"MANHOLE FIRE\s*-\s*EXTENDED TO BUILDING"
    )

    # ---------- 4) Fire – Non-Structural ----------
    # Rubbish/brush/auto/transport/maritime + manhole fire (except extended-to-building)
    is_fire_nonstructural = (
        s.rlike(
            r"DEMOLITION DEBRIS OR RUBBISH FIRE|"
            r"BRUSH FIRE|"
            r"AUTOMOBILE FIRE|"
            r"ABANDONED DERELICT VEHICLE FIRE|"
            r"OTHER TRANSPORTATION FIRE|"
            r"TRANSIT SYSTEM\s*-\s*NONSTRUCTURAL|"
            r"UNDEFINED NONSTRUCTURAL FIRE|"
            r"MARITIME FIRE"
        )
        | (
            s.rlike(r"^MANHOLE FIRE\s*-")
            & (~s.rlike(r"EXTENDED TO BUILDING"))
        )
    )

    # ---------- 5) Hazardous / Utility ----------
    # Utilities + CO + odors + defective oil burner + downed tree (kept as Other per your table)
    is_hazardous_utility = s.rlike(
        r"^UTILITY EMERGENCY\s*-|"
        r"^CARBON MONOXIDE\s*-|"
        r"^ODOR\s*-|"
        r"DEFECTIVE OIL BURNER"
    )

    # ---------- 6) Rescue / Entrapment ----------
    # Elevator occupied, extrication, remove civilian, maritime emergency, MFA ERS/BARS
    is_rescue = s.rlike(
        r"^ELEVATOR EMERGENCY\s*-\s*OCCUPIED|"
        r"VEHICLE ACCIDENT\s*-\s*WITH EXTRICATION|"
        r"REMOVE CIVILIAN\s*-\s*NON-FIRE|"
        r"MARITIME EMERGENCY|"
        r"NON-MEDICAL MFA\s*-\s*(ERS|BARS)"
    )

    # ---------- Apply in correct priority order ----------
    return (
        F.when(is_medical, F.lit("Medical"))
         .when(is_false_alarm, F.lit("False Alarm / No Action"))
         .when(is_fire_structural, F.lit("Fire – Structural"))
         .when(is_fire_nonstructural, F.lit("Fire – Non-Structural"))
         .when(is_hazardous_utility, F.lit("Hazardous / Utility"))
         .when(is_rescue, F.lit("Rescue / Entrapment"))
         .otherwise(F.lit("Other / Assistance"))
    )



# %% [markdown]
# #### 2.2.2 Apply the Mapping

# %% [markdown]
# Toronto Data

# %%
# 2. Apply Mapping
toronto_gold_feat = toronto_gold_feat.withColumn(
    "incident_category",
    map_toronto_incident_category(F.col("Final_Incident_Type"))
)

# %% [markdown]
# 3. Mapping Validation

# %%
# Validate Mapping for Toronto
display(
    toronto_gold_feat
    .select(
        F.col("Final_Incident_Type").alias("raw_incident_type"),
        F.col("incident_category")
    )
    .groupBy("raw_incident_type", "incident_category")
    .count()
    .orderBy(F.desc("count"))
)


# %% [markdown]
# NYC Data

# %%
nyc_gold_feat = nyc_gold_feat.withColumn(
    "incident_category",
    map_nyc_incident_category(F.col("INCIDENT_CLASSIFICATION"))
)

# %%
# Validate Mapping for NYC
display(
    nyc_gold_feat
    .select(
        F.col("INCIDENT_CLASSIFICATION").alias("raw_incident_type"),
        F.col("incident_category")
    )
    .groupBy("raw_incident_type", "incident_category")
    .count()
    .orderBy(F.desc("count"))
)

# %% [markdown]
# ### 2.3 Category Mapping Sanity Check

# %% [markdown]
# Unified Categorization of Incident Types for Toronto and NYC Data 

# %%
# Add city labels if not already present
toronto_counts = (
    toronto_gold_feat
    .withColumn("city", F.lit("Toronto"))
    .groupBy("city", "incident_category")
    .count()
)

nyc_counts = (
    nyc_gold_feat
    .withColumn("city", F.lit("NYC"))
    .groupBy("city", "incident_category")
    .count()
)

combined_counts = toronto_counts.unionByName(nyc_counts)
category_city_table = (
    combined_counts
    .groupBy("incident_category")
    .pivot("city", ["Toronto", "NYC"])
    .agg(F.first("count"))
    .fillna(0)
    .orderBy("incident_category")
)

display(category_city_table)

# %% [markdown]
# Check if New Inciddent Categories are assigned

# %%
display(toronto_gold_feat.limit(5))
display(nyc_gold_feat.limit(5))

# %% [markdown]
# Verify total row counts match

# %%
tor_total = toronto_gold_feat.count()
nyc_total = nyc_gold_feat.count()

tor_sum = toronto_gold_feat.groupBy("incident_category").count().agg(F.sum("count")).first()[0]
nyc_sum = nyc_gold_feat.groupBy("incident_category").count().agg(F.sum("count")).first()[0]

print("Toronto total:", tor_total, " | sum by category:", tor_sum)
print("NYC total:", nyc_total, " | sum by category:", nyc_sum)


# %% [markdown]
# ## 3. Alarm Level Harmonization
# **Objective**
#
# To harmonize city-specific alarm level definitions from Toronto Fire Services (TFS) and the Fire Department of New York (FDNY) into a unified escalation framework that enables consistent cross-city analysis and modeling.
#
# Alarm level harmonization follows the same structured approach used for incident type harmonization:
#
# 1. Pre-inspection
# 2. Data mapping
# 3. Validation
#
# **Toronto**
# <br>Input Col : `Event_Alarm_Level`
# <br>Output Col: `unified_alaram_level`
#
# **NYC**
# <br>Input Col: `alarm_level_index_description`
# <br>Output Col: `unified_alaram_level`

# %% [markdown]
# ### 3.1 Pre-Inspection
# Understand how alarm levels are represented, labeled, and distributed in each city before harmonization.

# %% [markdown]
# #### 3.1.1 Toronto Alarm Level Distribution
# Input Col: 

# %%
# Toronto raw alarm column is Event_Alarm_Level (integer)
(
    toronto_gold_feat
    .groupBy(F.col("Event_Alarm_Level").alias("alarm_raw"))
    .count()
    .orderBy(F.desc("count"))
    .show(200, truncate=False)
)

# %% [markdown]
# #### 3.1.2 NYC Alarm Level Distribution

# %%
# NYC raw alarm column is alarm_level_index_description (string)
(
  nyc_gold_feat
  .groupBy(F.col("alarm_level_index_description").alias("alarm_raw"))
  .count()
  .orderBy(F.desc("count"))
  .show(200, truncate=False)
)

# %%
toronto_gold_feat.printSchema()

# %% [markdown]
# ### 3.2 Alarm Level Coding amd Data Mapping
# Map city-specific alarm levels into a common escalation scale while preserving operational meaning.
# - Level 1: Initial response (first alarm)
# - Level 2: Esclated response
# - Level 3: Major/ Mulit-alarm incident

# %% [markdown]
# #### 3.2.1 Alarm Level Mapping and Harmoinzation
# | Unified Level     | Meaning               | Justification            |
# | ----------------- | --------------------- | ------------------------ |
# | **1 – Initial**   | Routine / first alarm | Dominates both cities    |
# | **2 – Escalated** | Additional resources  | Signals / Second Alarm   |
# | **3 – Major**     | Multi-alarm / extreme | Very rare, high severity |
#

# %% [markdown]
# ##### 3.2.1.1 Toronto Alarm Harmonization
# | Toronto `alarm_raw` | Unified Level |
# | ------------------- | ------------- |
# | 0                   | 1             |
# | 1, 2                | 2             |
# | ≥ 3                 | 3             |
#

# %%
toronto_gold_feat = (
    toronto_gold_feat
    .withColumn(
        "unified_alarm_level",
        F.when(F.col("Event_Alarm_Level").isNull(), F.lit(None).cast("int"))
         .when(F.col("Event_Alarm_Level") == 0, F.lit(1))
         .when(F.col("Event_Alarm_Level").isin(1, 2), F.lit(2))
         .when(F.col("Event_Alarm_Level") >= 3, F.lit(3))
         .otherwise(F.lit(None).cast("int"))
    )
)

# %%
(
    toronto_gold_feat
    .groupBy("Event_Alarm_Level", "unified_alarm_level")
    .count()
    .orderBy("Event_Alarm_Level", "unified_alarm_level")
    .show(200, truncate=False)
)


# %% [markdown]
# ##### 3.2.1.2 NYC Alaram Harmonization
# | FDNY `alarm_raw`         | Unified Level |
# | ------------------------ | ------------- |
# | Initial Alarm            | 1             |
# | DEFAULT RECORD           | 1             |
# | 7-5, 10-75, 10-76, 10-77 | 2             |
# | Second Alarm             | 2             |
# | Third Alarm and above    | 3             |
#

# %% [markdown]
# Helper: Normalized Alarm Text For NYC Alarm Levels

# %%
def normalize_alarm_text(col_):
    return F.trim(
        F.regexp_replace(
            F.regexp_replace(F.lower(col_), r"[^a-z0-9]+", " "),
            r"\s+", " "
        )
    )


# %% [markdown]
# NYC Alarm Level Mapping

# %%
nyc_gold_feat = (
    nyc_gold_feat
    .withColumn(
        "unified_alarm_level",
        F.when(F.col("alarm_level_index_description").isNull(), F.lit(None).cast("int"))
         .when(
             normalize_alarm_text(F.col("alarm_level_index_description"))
                 .isin("initial alarm", "default record"),
             F.lit(1)
         )
         .when(
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("second alarm") |
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("7 5") |
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("10 75") |
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("10 76") |
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("10 77"),
             F.lit(2)
         )
         .when(
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("third alarm") |
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("fourth alarm") |
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("fifth alarm") |
             normalize_alarm_text(F.col("alarm_level_index_description")).contains("sixth alarm"),
             F.lit(3)
         )
         .otherwise(F.lit(None).cast("int"))
    )
)

# %%
(
    nyc_gold_feat
    .groupBy("alarm_level_index_description", "unified_alarm_level")
    .count()
    .orderBy(F.desc("count"))
    .show(200, truncate=False)
)

# %% [markdown]
# ### 3.3 Alaram Mapping Sanity Check

# %%
toronto_counts = (
    toronto_gold_feat
    .withColumn("city", F.lit("Toronto"))
    .groupBy("city", "unified_alarm_level")
    .count()
)

nyc_counts = (
    nyc_gold_feat
    .withColumn("city", F.lit("NYC"))
    .groupBy("city", "unified_alarm_level")
    .count()
)

combined_counts = toronto_counts.unionByName(nyc_counts)
alarm_level_city_table = (
    combined_counts
    .groupBy("unified_alarm_level")
    .pivot("city", ["Toronto", "NYC"])
    .agg(F.first("count"))
    .fillna(0)
    .orderBy("unified_alarm_level")
)

alarm_city_table_labeled = (
    alarm_level_city_table
    .withColumn(
        "alarm_level",
        F.when(F.col("unified_alarm_level") == 1, F.lit("Level 1 – Initial"))
         .when(F.col("unified_alarm_level") == 2, F.lit("Level 2 – Escalated"))
         .when(F.col("unified_alarm_level") == 3, F.lit("Level 3 – Major"))
         .otherwise(F.concat(F.lit("Unknown ("), F.col("unified_alarm_level").cast("string"), F.lit(")")))
    )
    .drop("unified_alarm_level")
    .select("alarm_level", "Toronto", "NYC")
)

display(alarm_city_table_labeled)

# %%
# Toronto unmapped
toronto_gold_feat.select(
    F.count("*").alias("rows_total"),
    F.sum(F.col("unified_alarm_level").isNull().cast("int")).alias("unmapped_rows")
).show()

# NYC unmapped
nyc_gold_feat.select(
    F.count("*").alias("rows_total"),
    F.sum(F.col("unified_alarm_level").isNull().cast("int")).alias("unmapped_rows")
).show()


# %%
# NYC: what values failed mapping?
(
  nyc_gold_feat
  .filter(F.col("unified_alarm_level").isNull())
  .groupBy("alarm_level_index_description")
  .count()
  .orderBy(F.desc("count"))
  .show(200, truncate=False)
)


# %% [markdown]
# ## 4. Call Source Harmonization
# To harmonize call origin information across Toronto Fire Services and FDNY into a small set of unified call-source categories, enabling cross-city comparison and modeling while preserving operational meaning.
#
# **Toronto**
# <br>Input Col: `Call_Source`
# <br>Output Col: `unified_call_source`
#
# **NYC**
# <br>Input Col: `alarm_source_description_tx`
# <br>Output Col: `unified_call_source`

# %% [markdown]
# ### 4.1 Pre-inpsection
# Understanding of raw call sources

# %% [markdown]
# #### 4.1.1 Toronto Call Source Inspection

# %%
(
    toronto_gold_feat
    .groupBy("Call_Source")
    .count()
    .orderBy(F.desc("count"))
    .show(200, truncate=False)
)


# %% [markdown]
# The Toronto dataset records call origin using structured numeric codes with descriptive labels. These sources represent a mix of public callers, emergency services, alarm systems, and internal discoveries.
# | Raw Value                                              | Interpretation                      |
# | ------------------------------------------------------ | ----------------------------------- |
# | **01 – 911**                                           | Public emergency call               |
# | **02 – Telephone from Civilian (other than 911)**      | Public, non-911 call                |
# | **03 – From Ambulance**                                | EMS / medical referral              |
# | **04 – From Police Services**                          | Police referral                     |
# | **05 – Telephone from Monitoring Agency**              | Alarm monitoring company            |
# | **06 – Direct Connection**                             | Automated alarm system              |
# | **07 – Verbal Report to Station (in person)**          | Walk-in / in-person report          |
# | **08 – Two-Way Radio (Fire Department)**               | Internal fire department report     |
# | **09 – Other Alarm**                                   | Alarm system                        |
# | **11 – No alarm received – incident discovered by FD** | Internal discovery by fire services |
# | **10 – No alarm received – no response**               | Rare / edge case                    |
# | **NULL**                                               | Missing                             |
#              |
#

# %% [markdown]
# #### 4.1.1 NYC Call Source Inspection

# %%
(
    nyc_gold_feat
    .groupBy("alarm_source_description_tx")
    .count()
    .orderBy(F.desc("count"))
    .show(200, truncate=False)
)


# %% [markdown]
# The NYC dataset uses short categorical codes to indicate call origin. Unlike Toronto, it does not explicitly identify police or fire department referrals in this field.
# | Raw Value                       | Interpretation                           |
# | ------------------------------- | ---------------------------------------- |
# | **PHONE, 911, 911TEXT, VERBAL** | Public-initiated calls                   |
# | **EMS, EMS-911**                | EMS / medical referrals                  |
# | **CLASS-3**                     | Alarm system (monitored building alarms) |
# | **ERS, BARS, SOL**              | System- or agency-generated sources      |
#
#

# %% [markdown]
# ### 4.2 Call Source Mapping
# Based on the pre-inspection results and structural differences in how call sources are recorded across cities, call origins were harmonized into the following unified categories. Police/fire referrals (Toronto) and system/agency-generated sources (NYC) were consolidated to improve cross-city comparability and reduce sparsity in downstream modeling.
# | Unified Call Source | Includes                                                                                                                              |
# | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
# | **Public**          | 911 calls, phone calls, walk-in/verbal reports, text messages                                                                         |
# | **EMS / Medical**   | Ambulance referrals, EMS, EMS-911                                                                                                     |
# | **Alarm System**    | Monitoring agencies, CLASS-3 alarms, direct/automated alarm connections                                                               |
# | **Other / System**  | Non-public institutional sources, including police/fire referrals (Toronto) and system/agency-generated sources (NYC: ERS, BARS, SOL) |
# | **Unknown**         | Null or truly missing values                                                                                                          |
#

# %% [markdown]
# Helper Function for Text Normalization

# %%
def normalize_text(col_):
    return F.trim(
        F.regexp_replace(
            F.regexp_replace(F.lower(col_), r"[^a-z0-9]+", " "),
            r"\s+", " "
        )
    )


# %% [markdown]
# #### 4.2.1 Toronto Call Source Mapping
#

# %%
toronto_gold_feat = (
    toronto_gold_feat
    .withColumn(
        "unified_call_source",
        F.when(F.col("Call_Source").isNull(), F.lit("Unknown"))

        # Public-origin calls
        .when(
            normalize_text(F.col("Call_Source")).contains("911") |
            normalize_text(F.col("Call_Source")).contains("civilian") |
            normalize_text(F.col("Call_Source")).contains("verbal report"),
            F.lit("Public")
        )

        # EMS / Medical referrals
        .when(
            normalize_text(F.col("Call_Source")).contains("ambulance"),
            F.lit("EMS / Medical")
        )

        # Alarm system sources
        .when(
            normalize_text(F.col("Call_Source")).contains("monitoring agency") |
            normalize_text(F.col("Call_Source")).contains("direct connection") |
            normalize_text(F.col("Call_Source")).contains("other alarm"),
            F.lit("Alarm System")
        )

        # Non-public institutional & system-originated sources
        # (includes police/fire referrals and rare system records)
        .when(
            normalize_text(F.col("Call_Source")).contains("police") |
            normalize_text(F.col("Call_Source")).contains("fire department") |
            normalize_text(F.col("Call_Source")).contains("incident discovered") |
            normalize_text(F.col("Call_Source")).contains("two way radio") |
            normalize_text(F.col("Call_Source")).contains("no alarm"),
            F.lit("Other / System")
        )

        # Fallback
        .otherwise(F.lit("Other / System"))
    )
)

# %%
(
    toronto_gold_feat
    .groupBy("Call_Source", "unified_call_source")
    .count()
    .orderBy(F.desc("count"))
    .show(200, truncate=False)
)

# %% [markdown]
# #### 4.2.2 NYC Call Source Mapping

# %%
nyc_src_norm = normalize_text(F.col("alarm_source_description_tx"))

nyc_gold_feat = (
    nyc_gold_feat
    .withColumn(
        "unified_call_source",
        F.when(F.col("alarm_source_description_tx").isNull(), F.lit("Unknown"))

        # Public-origin calls
        .when(
            nyc_src_norm.isin("phone", "911", "911text", "verbal"),
            F.lit("Public")
        )

        # EMS / Medical referrals
        .when(
            nyc_src_norm.contains("ems"),
            F.lit("EMS / Medical")
        )

        # Alarm system sources
        .when(
            nyc_src_norm.contains("class 3"),
            F.lit("Alarm System")
        )

        # Other / System sources (system/agency-coded)
        .when(
            nyc_src_norm.isin("ers", "bars", "sol"),
            F.lit("Other / System")
        )

        # Fallback
        .otherwise(F.lit("Other / System"))
    )
)


# %%
(
    nyc_gold_feat
    .groupBy("alarm_source_description_tx", "unified_call_source")
    .count()
    .orderBy(F.desc("count"))
    .show(200, truncate=False)
)

# %% [markdown]
# #### 4.3 Call Soucre Mapping Sanity Check

# %%
toronto_counts = (
    toronto_gold_feat
    .withColumn("city", F.lit("Toronto"))
    .groupBy("city", "unified_call_source")
    .count()
)

nyc_counts = (
    nyc_gold_feat
    .withColumn("city", F.lit("NYC"))
    .groupBy("city", "unified_call_source")
    .count()
)

combined_counts = toronto_counts.unionByName(nyc_counts)

call_source_city_table = (
    combined_counts
    .groupBy("unified_call_source")
    .pivot("city", ["Toronto", "NYC"])
    .agg(F.first("count"))
    .fillna(0)
    .orderBy("unified_call_source")
)

preferred_order = ["Public", "EMS / Medical", "Alarm System", "Police / Fire", "Other / System", "Unknown"]

call_source_city_table_ordered = (
    call_source_city_table
    .withColumn(
        "sort_key",
        F.array_position(F.array([F.lit(x) for x in preferred_order]), F.col("unified_call_source"))
    )
    .orderBy("sort_key")
    .drop("sort_key")
)

display(call_source_city_table_ordered)

# %% [markdown]
# Across both cities, **EMS / Medical** is the most common call source, followed by **public-initiated calls**, highlighting the central role of fire services in medical response and civilian-reported incidents.** Alarm system-generated** calls account for a substantial share in both Toronto and NYC, with higher volumes in NYC. **Other / System **sources represent a small fraction of incidents, and **Unknown** values are negligible, indicating good data completeness.

# %%
tor_total = toronto_gold_feat.count()
nyc_total = nyc_gold_feat.count()

tor_sum = toronto_gold_feat.groupBy("unified_call_source").count().agg(F.sum("count")).first()[0]
nyc_sum = nyc_gold_feat.groupBy("unified_call_source").count().agg(F.sum("count")).first()[0]

print("Toronto total:", tor_total, " | sum by unified_call_source:", tor_sum)
print("NYC total:", nyc_total, " | sum by unified_call_source:", nyc_sum)

# %% [markdown]
# ## 5. Data Preparation For Modeling
# To construct consistent, model-ready datasets for Toronto and NYC fire incident data by defining unified targets, harmonizing conceptually comparable features, standardizing temporal and demand-related variables, and retaining city-specific attributes where appropriate, in order to support robust predictive and survival modeling.

# %% [markdown]
# ### 5.1 Toronto Model-Ready DataFrame
#
# This section constructs the Toronto model-ready dataset by selecting the finalized target and feature variables from the harmonized feature table. No additional transformations are applied beyond renaming and selection.
#
# **Target Variables**
#
# * `response_minutes`: response time in minutes (derived from `response_time_minutes`)
# * `event_time`: equivalent to `response_minutes` for survival analysis
# * `event_indicator`: binary indicator set to 1 when a valid arrival timestamp exists
#
# **Feature Variables**
#
# * **Temporal:** hour, day_of_week, month, season
# * **Categorical:** incident_category, unified_alarm_level, unified_call_source
# * **Location:** Incident_Station_Area
# * **Demand Intensity:** calls_past_30m, calls_past_60m
#
# The resulting DataFrame is ready for exploratory analysis and downstream modeling.

# %%
# Inspect gold level schema 
toronto_gold_feat.printSchema()

# %%
# Toronto Model-Ready Data Frame with features and Targets

toronto_model_df = (
    toronto_gold_feat
    .select(
        # --- IDs & targets ---
        "incident_id",
        "incident_datetime",
        "response_minutes",
        "delay_indicator",
        "event_indicator",

        # --- temporal ---
        "hour",
        "day_of_week",
        "month",
        "season",
        "year",

        # --- categorical ---
        "incident_category",
        "unified_alarm_level",
        "unified_call_source",

        # --- location ---
        F.col("Incident_Station_Area").alias("location_area"),

        # --- demand intensity ---
        "calls_past_30min",
        "calls_past_60min",
    )
)


# %%
# Check target and features columns
toronto_model_df.printSchema()

# %%
display(toronto_model_df.limit(5))

# %% [markdown]
# ### 5.2 NYC Model-Ready DataFrame
#
# This section constructs the NYC model-ready dataset by selecting the finalized target and feature variables from the harmonized feature table. No additional transformations are applied beyond renaming and selection.
#
# **Target Variables**
#
# * `response_minutes`: response time in minutes (converted from response time in seconds)
# * `event_indicator`: binary indicator set to 1 when a valid arrival timestamp exists
#
# **Feature Variables**
#
# * **Temporal:** hour, day_of_week, month, season
# * **Categorical:** incident_category, unified_alarm_level, unified_call_source
# * **Location:** incident_borough
# * **Demand Intensity:** calls_past_30min, calls_past_60min
#
# The resulting DataFrame is ready for exploratory analysis and downstream modeling.

# %%
# check gold level schema
nyc_gold_feat.printSchema()

# %%
# Check Target and Features Column
nyc_model_df = (
    nyc_gold_feat
    .withColumn("incident_id", F.col("incident_id").cast("string"))
    .select(
        # --- IDs & targets ---
        "incident_id",
        "incident_datetime",
        "response_minutes",
        "delay_indicator",
        "event_indicator",

        # --- temporal ---
        "hour",
        "day_of_week",
        "month",
        "season",
        "year",

        # --- categorical ---
        "incident_category",
        "unified_alarm_level",
        "unified_call_source",

        # --- location ---
        F.col("incident_borough").alias("location_area"),

        # --- demand intensity ---
        "calls_past_30min",
        "calls_past_60min",
    )
)

# %%
# Check Target and Features Column
nyc_model_df.printSchema()
display(nyc_model_df.limit(5))

# %% [markdown]
# ### 5.3 Sanity Check

# %%
toronto_model_df.select(
    F.count("*").alias("n"),
    F.sum(F.col("response_minutes").isNull().cast("int")).alias("missing_response")
).show()

nyc_model_df.select(
    F.count("*").alias("n"),
    F.sum(F.col("response_minutes").isNull().cast("int")).alias("missing_response")
).show()

# %% [markdown]
# ### 5.4 Save Dataframes

# %% [markdown]
# ##### Save as tables
# Model-ready datasets are written as Delta tables to provide a stable, reproducible snapshot for exploratory analysis and modeling. Persisting these tables avoids repeated recomputation, ensures consistency across team members, and supports reliable downstream analysis without additional pipeline orchestration.
#
#

# %%
nyc_model_df.printSchema()

# %%
# Toronto
# Overwrite Toronto table and update schema
(
    toronto_model_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.capstone_project.toronto_model_ready")
)

# %%
# NYC
# Overwrite NYC table and update schema
(
    nyc_model_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.capstone_project.nyc_model_ready")
)

# %% [markdown]
# #### Missing Values Check

# %% [markdown]
# **Toronto**

# %%
n = toronto_gold.count()

missing = toronto_gold.select([
    F.round(
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.lit(n) * 100),
        2   # <-- number of decimal places
    ).alias(c)
    for c in toronto_gold.columns
])
display(missing.toPandas().T.reset_index()
        .rename(columns={"index":"column", 0:"percent_missing"}))
print("Column Level Missing Count Toronto Gold Table:", missing.count())

# %%
n = toronto_gold_feat.count()

missing = toronto_gold_feat.select([
    F.round(
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.lit(n) * 100),
        2   # <-- number of decimal places
    ).alias(c)
    for c in toronto_gold_feat.columns
])
display(missing.toPandas().T.reset_index()
        .rename(columns={"index":"column", 0:"percent_missing"}))
print("Column Level Missing Count Toronto Feature Table:", missing.count())

# %%
n = toronto_model_df.count()

missing = toronto_model_df.select([
    F.round(
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.lit(n) * 100),
        2   # <-- number of decimal places
    ).alias(c)
    for c in toronto_model_df.columns
])
display(missing.toPandas().T.reset_index()
        .rename(columns={"index":"column", 0:"percent_missing"}))
print("Column Level Missing Count Toronto Model-ready Table:", missing.count())

# %% [markdown]
# **NYC**

# %%
n = nyc_gold.count()

missing = nyc_gold.select([
    F.round(
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.lit(n) * 100),
        2   # <-- number of decimal places
    ).alias(c)
    for c in nyc_gold.columns
])
display(missing.toPandas().T.reset_index()
        .rename(columns={"index":"column", 0:"percent_missing"}))
print("Column Level Missing Count NYC Gold Table:", missing.count())

# %%
n = nyc_gold_feat.count()

missing = nyc_gold_feat.select([
    F.round(
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.lit(n) * 100),
        2   # <-- number of decimal places
    ).alias(c)
    for c in nyc_gold_feat.columns
])
display(missing.toPandas().T.reset_index()
        .rename(columns={"index":"column", 0:"percent_missing"}))
print("Column Level Missing Count NYC Feature Table:", missing.count())

# %%
n = nyc_model_df.count()

missing = nyc_model_df.select([
    F.round(
        (F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.lit(n) * 100),
        2   # <-- number of decimal places
    ).alias(c)
    for c in nyc_model_df.columns
])
display(missing.toPandas().T.reset_index()
        .rename(columns={"index":"column", 0:"percent_missing"}))
print("Column Level Missing Count NYC Model-ready Table:", missing.count())

# %% [markdown]
# ### 5.5 Merge Data Frames Code

# %%
toronto_model_df.printSchema()

# %%
nyc_model_df.printSchema()

# %%
# Ensure we start from the model-ready tables (recommended)
toronto_df = spark.table("workspace.capstone_project.toronto_model_ready")
nyc_df     = spark.table("workspace.capstone_project.nyc_model_ready")

# Columns to keep in the unified dataset (explicit = safer)
common_cols = [
    "incident_id",
    "incident_datetime",
    "response_minutes",
    "event_indicator",
    "hour",
    "day_of_week",
    "month",
    "season",
    "year",
    "incident_category",
    "unified_alarm_level",
    "unified_call_source",
    "location_area",
    "calls_past_30min",
    "calls_past_60min",
]

toronto_for_merge = (
    toronto_df
    .select(*common_cols)
    .withColumn("city", F.lit("Toronto"))
)

nyc_for_merge = (
    nyc_df
    .select(*common_cols)
    .withColumn("city", F.lit("NYC"))
)

combined_model_df = toronto_for_merge.unionByName(nyc_for_merge)

combined_model_df.printSchema()
display(combined_model_df.limit(5))
