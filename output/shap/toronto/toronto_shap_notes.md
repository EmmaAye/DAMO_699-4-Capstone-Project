# Toronto — SHAP Interpretation Notes

## Top drivers

- **location_area** (mean|SHAP|=0.693520)
- **hour** (mean|SHAP|=0.374433)
- **unified_call_source** (mean|SHAP|=0.231754)
- **incident_category** (mean|SHAP|=0.216704)
- **calls_past_30min** (mean|SHAP|=0.144152)

## Key patterns

- Summary plot: direction + spread of feature impact.
- Bar plot: global importance ranking.
- Dependence plots: feature behavior and thresholds.
- Waterfall: one incident explanation for storytelling.
- Positive SHAP increases delay risk; negative SHAP decreases delay risk.