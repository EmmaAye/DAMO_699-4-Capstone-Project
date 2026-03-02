# Toronto — SHAP Interpretation Notes

## Top drivers

- **location_area** (mean|SHAP|=0.693520)
- **hour** (mean|SHAP|=0.374433)
- **unified_call_source** (mean|SHAP|=0.231754)
- **incident_category** (mean|SHAP|=0.216704)
- **calls_past_30min** (mean|SHAP|=0.144152)

## Key patterns

- Beeswarm: direction + spread of feature impact.
- Bar chart: global importance ranking.
- Waterfall: one incident explanation (best for storytelling).
- Positive SHAP → higher delay risk; negative SHAP → lower delay risk.