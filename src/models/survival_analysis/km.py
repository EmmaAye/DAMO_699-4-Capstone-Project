from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def fit_km(df, label: str, duration_col="response_minutes", event_col="event_indicator"):
    km = KaplanMeierFitter()
    km.fit(df[duration_col], df[event_col], label=label)
    return km


def km_plot_single_city(
    km,
    title: str,
    censor_threshold: float | None = None,
):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    km.plot_survival_function(ax=ax, ci_show=False)

    ax.set_title(title)
    ax.set_xlabel("Response time (minutes)")
    ax.set_ylabel("Probability that the unit has not arrived yet")
    ax.grid(True)

    if censor_threshold is not None:
        ax.set_xlim(0, censor_threshold)
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    return ax


def km_overlay_plot(
    km_a,
    km_b,
    censor_threshold: float,
    thresholds=(10, 30, 60),
    title="Kaplan–Meier Survival — Cross-City",
):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    km_a.plot_survival_function(ax=ax, ci_show=False)
    km_b.plot_survival_function(ax=ax, ci_show=False)

    for t in thresholds:
        ax.axvline(t, linestyle="--", linewidth=1)
        ax.text(t, 0.02, f"{t}m", rotation=90, va="bottom")

    ax.set_xlim(0, censor_threshold)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Response time (minutes)")
    ax.set_ylabel("Probability that the unit has not arrived yet")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    return ax


def survival_at_thresholds(km, thresholds=(10, 30, 60)):
    return {t: float(km.predict(t)) for t in thresholds}


def validate_km(df_pd, kmf, city_name, t_values=(5, 10, 15), duration_col="response_minutes", event_col="event_indicator"):
    """
    Your validation block from baseline notebook:
    - compares KM S(t) vs empirical survival among observed events
    """
    observed = df_pd[df_pd[event_col] == 1].copy()

    raw_median = float(np.median(observed[duration_col])) if len(observed) > 0 else np.nan
    km_median = float(kmf.median_survival_time_) if kmf.median_survival_time_ is not None else np.nan

    lines = []
    lines.append(f"=== Validation: {city_name} ===")
    lines.append(f"Raw median (events only): {raw_median}")
    lines.append(f"KM median: {km_median}")

    for t in t_values:
        km_s = float(kmf.predict(t))
        emp_s = float(np.mean(observed[duration_col] > t)) if len(observed) > 0 else np.nan
        lines.append(f"S({t}) KM={km_s:.3f} | Empirical={emp_s:.3f}  (events only)")

    return "\n".join(lines)


def km_plot_stratified(
    df_pd,
    group_col: str,
    title: str,
    censor_threshold: float = 60.0,
    group_order=None,
    duration_col="response_minutes",
    event_col="event_indicator",
):
    """
    Stratified KM plotting from your stratified notebook.
    (This function only plots; testing lives in stats.py)
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    if group_order is None:
        groups = sorted(df_pd[group_col].dropna().astype(str).unique())
    else:
        groups = group_order

    for g in groups:
        sub = df_pd[df_pd[group_col].astype(str) == str(g)]
        if len(sub) == 0:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(sub[duration_col], sub[event_col], label=str(g))
        kmf.plot_survival_function(ax=ax, ci_show=False)

    ax.set_title(title)
    ax.set_xlabel("Response Time (minutes)")
    ax.set_ylabel("Survival Probability  P(T > t)")
    ax.set_xlim(0, censor_threshold)
    ax.set_ylim(0, 1.0)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return ax