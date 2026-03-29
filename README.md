# Capstone Project: Emergency Response Delay Risk Analytics  
## Survival Analysis & Predictive Modeling for Urban Fire Services

## Overview

This capstone project develops a data-driven framework to analyze, compare, and predict emergency response-time delay risk in large metropolitan fire services. Using dispatch and incident data from Toronto and New York City, the project combines survival analysis and predictive modeling to quantify delay risk, identify key operational drivers, and examine whether response-time structures are consistent across cities.

Rather than focusing only on average response times, the project emphasizes tail-risk behavior,the probability that response times exceed critical service thresholds. In addition, predictive models are used to estimate the likelihood of delays based on temporal, demand-related, and incident-level factors. Together, these approaches provide a more realistic and operationally meaningful view of service reliability.y.

---

## Objectives

- Quantify emergency response delay risk using survival analysis
- Build predictive models to estimate the likelihood of response delays
- Identify temporal and demand-related drivers of delays  
- Compare response-time survival patterns across Toronto and NYC  
- Evaluate the relative importance of predictive drivers of delay risk
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

### Predictive Modeling

* Classification models (Logistic Regression, Random Forest, Gradient Boosted Trees (GBTClassifier))to estimate delay likelihood
* Model evaluation using performance metrics such as AUC, precision, recall, and F1-score
* Comparative analysis of predictive drivers across cities

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

- Predictive models for delay classification
- Baseline and stratified survival curves
- Cross-city delay-risk comparison
- Hazard ratio estimates for predictive factors
- Tail-risk probability metrics
- Visualizations for reporting and dashboards

---

## Tools & Technologies

- Python (Pandas, NumPy, Matplotlib, Scikit-learn)
- PySpark / Databricks
- Lifelines (survival analysis)
- SQL
- Git & GitHub

---

## Project Structure

```

etl_pipelines/
├── NYC/
├── toronto/
notebooks/
src/
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
