## Data Preparation – ETL & Harmonization

This project uses a two-stage pipeline in Databricks:

1. **ETL pipelines** → generate cleaned city tables  
2. **Harmonization step** → generate model-ready tables for analysis  

Run these steps in order.

---

## Step 1: Prepare ETL Pipelines in Databricks

Navigate to:

```

etl_pipelines/
├── Capstone_Project_ETL_Pipeline_Toronto
└── Capstone_Project_ETL_Pipeline_NYC

```

1. Open the `transformation` notebook for each city.
2. Confirm:
   - input file paths
   - target schema/database
   - output table names
3. Ensure both pipelines write to the same workspace schema.

---

## Step 2: Set Up Schema & Upload Raw Data

1. Create or confirm the Databricks schema.
2. Open the schema **Volume**.
3. Upload raw CSV files for:
   - Toronto
   - NYC
4. Verify file paths match those used in the ETL notebooks.

---

## Step 3: Run ETL Pipelines

1. Run the Toronto ETL pipeline notebook.
2. Run the NYC ETL pipeline notebook.
3. Ensure all cells execute successfully.
4. Check for:
   - schema errors
   - missing columns
   - datatype mismatches

---

## Step 4: Confirm Cleaned Tables

After ETL completes, verify the following tables exist:

- `nyc_fire_incidents_bronze`
- `nyc_fire_incidents_silver`
- `nyc_fire_incidents_gold`
- `tfs_incidents_bronze`
- `tfs_incidents_silver`
- `tfs_incidents_gold`

Checks:
- Row counts look reasonable  
- `response_minutes` exists  
- timestamps parsed correctly  
- no null values in key fields  

These tables are now ready for harmonization.

---

## Step 5: Run Data Harmonization

The harmonization step aligns both cities into a consistent modeling schema.

Navigate to:

```

notebooks/data_prepartion/

```

Run the `01_data_harmonization_and_validation` notebook to:

- standardize time units (minutes)
- create `event_indicator`
- apply censoring rules
- align column names
- generate temporal features
- produce final modeling tables

---

## Step 6: Confirm Model-Ready Tables

Expected outputs:

- `toronto_model_ready`
- `nyc_model_ready`

These tables should include:
- `response_minutes`
- `event_indicator`
- temporal features (hour, day_of_week, season)
- demand features
- consistent schema across cities

---

## Final Check

Before running modeling notebooks, confirm:

- Both cities use identical units (minutes)
- Censoring rules match
- Row counts are valid
- No missing critical columns

These model-ready tables are used by:
- survival analysis notebooks
- predictive modeling notebooks
- cross-city comparison workflows
```

