from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def binned_hazard(
    df,
    censor_threshold: float = 60.0,
    bin_width: float = 2.0,
    duration_col="response_minutes",
    event_col="event_indicator",
):
    """
    Binned hazard estimator:
      hazard(bin) = events_in_bin / at_risk_at_bin_start
    """
    times = df[duration_col].to_numpy(dtype=float)
    events = df[event_col].to_numpy(dtype=int)

    edges = np.arange(0, censor_threshold + bin_width, bin_width)
    rows = []

    for start in edges[:-1]:
        end = start + bin_width
        at_risk = int(np.sum(times >= start))
        d = int(np.sum((times >= start) & (times < end) & (events == 1)))
        hz = (d / at_risk) if at_risk > 0 else np.nan
        rows.append(
            {"bin_start": start, "bin_end": end, "at_risk": at_risk, "events": d, "hazard": hz}
        )

    return pd.DataFrame(rows)


def hazard_overlay_plot(
    hz_a,
    hz_b,
    label_a="Toronto",
    label_b="NYC",
    censor_threshold: float = 60.0,
    title="Hazard Comparison (Binned)",
):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.plot(hz_a["bin_start"], hz_a["hazard"], marker="o", linewidth=1, label=label_a)
    ax.plot(hz_b["bin_start"], hz_b["hazard"], marker="o", linewidth=1, label=label_b)

    ax.set_xlim(0, censor_threshold)
    ax.set_xlabel("Time bin start (minutes)")
    ax.set_ylabel("Estimated hazard (per bin)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    return ax