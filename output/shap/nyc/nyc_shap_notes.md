# NYC — SHAP Interpretation Notes

## Top drivers (mean |SHAP|)

- **incident_category** (mean|SHAP|=0.469148)
- **location_area** (mean|SHAP|=0.348915)
- **hour** (mean|SHAP|=0.281041)
- **unified_call_source** (mean|SHAP|=0.082936)
- **calls_past_60min** (mean|SHAP|=0.073721)

## Key technical patterns

- Beeswarm: direction + spread of feature impact across incidents.
- Bar chart: global importance ranking by mean(|SHAP|).
- Dependence: how each top feature changes risk (thresholds/nonlinearity).
- Waterfall: one high-risk case explanation (best for storytelling).
- Interpretation rule: **positive SHAP → higher delay risk**, **negative SHAP → lower delay risk**.