# NYC — SHAP Interpretation Notes

## Top drivers

- **incident_category** (mean|SHAP|=0.469148)
- **location_area** (mean|SHAP|=0.348915)
- **hour** (mean|SHAP|=0.281041)
- **unified_call_source** (mean|SHAP|=0.082936)
- **calls_past_60min** (mean|SHAP|=0.073721)

## Key patterns

- Beeswarm: direction + spread of feature impact.
- Bar chart: global importance ranking.
- Waterfall: one incident explanation (best for storytelling).
- Positive SHAP → higher delay risk; negative SHAP → lower delay risk.