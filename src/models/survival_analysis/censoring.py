import numpy as np
import pandas as pd

def apply_uniform_censoring_pandas(
    df: pd.DataFrame,
    censor_threshold: float,
    duration_col="response_minutes",
    event_col="event_indicator",
) -> pd.DataFrame:
    out = df.copy()

    orig_t = pd.to_numeric(out[duration_col], errors="coerce")
    orig_e = pd.to_numeric(out[event_col], errors="coerce").fillna(0).astype(int)

    # NULL duration => censored at threshold
    t_filled = orig_t.fillna(float(censor_threshold))

    out[duration_col] = t_filled.clip(upper=float(censor_threshold))
    out[event_col] = np.where((orig_t.notna()) & (orig_t <= censor_threshold) & (orig_e == 1), 1, 0).astype(int)

    return out