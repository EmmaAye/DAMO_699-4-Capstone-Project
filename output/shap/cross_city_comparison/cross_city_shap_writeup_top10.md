# US5.3 — Cross-City SHAP Explainability Comparison (Predictive)

## Inputs (from US5.2)
- toronto_shap_importance.csv
- nyc_shap_importance.csv  
(No new SHAP computation performed.)

## Summary (Top-10)
 All Top-10 drivers are shared across Toronto and NYC (same feature set; order/magnitude may differ).

## Top-10 Drivers (Toronto)
location_area, hour, unified_call_source, incident_category, calls_past_30min, month, day_of_week, year, season, calls_past_60min

## Top-10 Drivers (NYC)
incident_category, location_area, hour, unified_call_source, calls_past_60min, calls_past_30min, month, day_of_week, year, season

## Shared vs City-Specific Drivers (Top-10)
**Common Drivers:** calls_past_30min, calls_past_60min, day_of_week, hour, incident_category, location_area, month, season, unified_call_source, year  
**Toronto-Specific Drivers:** None  
**NYC-Specific Drivers:** None  

## Largest Cross-City Magnitude Differences (Mean |SHAP|)
Story (technical):
Even when the same drivers appear in both cities, models may differ in how *strongly* they rely on each feature.
Below are the top differences in mean |SHAP| magnitude:

- location_area: Toronto=0.6935, NYC=0.3489, |Δ|=0.3446, Gap% (Tor vs NYC)=98.8%
- incident_category: Toronto=0.2167, NYC=0.4691, |Δ|=0.2524, Gap% (Tor vs NYC)=-53.8%
- unified_call_source: Toronto=0.2318, NYC=0.0829, |Δ|=0.1488, Gap% (Tor vs NYC)=179.4%

## Technical Notes
- Shared drivers suggest structural similarity in predictive patterns across cities.
- Rank and magnitude differences are captured in `cross_city_shap_driver_table.csv`.
- Operational/policy interpretation is intentionally deferred to US6.3.

## Outputs Produced (US5.3)
- cross_city_shap_driver_table.csv
- cross_city_shap_comparison_clean_top10.png
- cross_city_shap_writeup_top10.md
