# Capstone Project: Emergency Response Delay Risk Analytics  
## Survival Analysis & Predictive Modeling for Urban Fire Services

## Overview

This capstone project develops a data-driven framework to analyze and predict emergency response-time delay risk in large metropolitan fire services. Using dispatch and incident data from **Toronto** and **New York City**, the project applies survival analysis and predictive modeling to quantify delay risk, identify key drivers, and compare response-time structures across cities.

Rather than focusing solely on average response times, this project emphasizes **tail-risk behavior**—the probability that response times exceed critical service thresholds. This provides a more realistic and operationally meaningful measure of service reliability.

---

## Objectives

- Quantify emergency response delay risk using survival analysis  
- Identify temporal and demand-related drivers of delays  
- Compare response-time survival patterns across Toronto and NYC  
- Evaluate predictive drivers of delay risk  
- Reveal tail-risk patterns not visible in average-based reporting  

---

## Research Questions

**RQ1 – Temporal Drivers**  
Do time-of-day, day-of-week, and seasonal factors influence delay risk?

**RQ2 – Demand Intensity Effects**  
Do short-term demand surges increase the probability of response delays?

**RQ3 – Cross-City Structure**  
Are delay-risk patterns similar between Toronto and NYC?

**RQ4 – Key Predictive Drivers**  
Do temporal and demand-related factors explain delay risk more effectively than incident type alone?

**RQ5 – Tail Risk vs Averages**  
Do survival-based delay probabilities reveal risks not captured by average response-time metrics?

---

## Methodology

### Survival Analysis
- Kaplan–Meier survival curves (baseline and stratified)
- Log-rank tests for group comparison
- Cox proportional hazards modeling

### Feature Engineering
- Temporal features (hour, day-of-week, season)
- Demand intensity metrics (calls in past 30/60 minutes)
- Incident-level characteristics

### Cross-City Comparison
- Survival curve overlays
- Hazard pattern comparison
- Threshold-based delay probability analysis

---

## Key Outputs

- Baseline and stratified survival curves
- Cross-city delay-risk comparison
- Hazard ratio estimates for predictive factors
- Tail-risk probability metrics
- Visualizations for reporting and dashboards

---

## Tools & Technologies

- Python (Pandas, NumPy, Matplotlib)
- PySpark / Databricks
- Lifelines (survival analysis)
- SQL
- Git & GitHub

---

## Project Structure

```

data/
notebooks/
models/
output/
├── graphs/
├── models/
└── tables/
docs/
tools/

```



---

## Impact

This project demonstrates how survival analysis and predictive analytics can enhance understanding of emergency service performance by focusing on **risk, reliability, and long-tail delays**, rather than averages alone. The framework supports operational decision-making, resource allocation, and performance benchmarking across cities.
