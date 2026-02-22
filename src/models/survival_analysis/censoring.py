import numpy as np
import pandas as pd

def apply_uniform_censoring_pandas(df: pd.DataFrame, censor_threshold: float,
                                  duration_col="response_minutes", event_col="event_indicator") -> pd.DataFrame:
    """
    Enforce: time = min(time, censor_threshold)
             event = 1 only if original event==1 and original time<=threshold else 0
    """
    out = df.copy()
    orig_t = out[duration_col].astype(float)
    orig_e = out[event_col].astype(int)

    out[duration_col] = orig_t.clip(upper=censor_threshold)
    out[event_col] = np.where((orig_t <= censor_threshold) & (orig_e == 1), 1, 0).astype(int)
    return out