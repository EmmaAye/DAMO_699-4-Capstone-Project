from __future__ import annotations


def cross_city_summary_text(lr: dict, s_to: dict, s_nyc: dict, alpha: float = 0.05):
    sig = lr["p_value"] < alpha
    lines = []
    lines.append("Cross-city survival comparison (Toronto vs NYC):")
    lines.append(f"- Log-rank p-value = {lr['p_value']:.3g} (significant={sig})")
    lines.append("- Threshold survival S(t)=P(response time > t):")
    for t in sorted(s_to.keys()):
        lines.append(f"  t={t}m: Toronto S(t)={s_to[t]:.3f} | NYC S(t)={s_nyc[t]:.3f}")
    lines.append("Interpretation: lower S(t) at early times implies lower early delay risk (faster service).")
    return "\n".join(lines)


def within_city_summary_text(city: str, stat: dict, alpha: float = 0.05):
    sig = stat["p_value"] < alpha
    p_txt = "< 1e-300" if stat["p_value"] == 0 else f"{stat['p_value']:.3g}"

    lines = []
    lines.append(f"{city} – {stat['group_col']}:")
    lines.append(f"- Log-rank p-value = {p_txt} (significant={sig})")

    if sig and stat.get("higher_risk_group_tail"):
        lines.append(f"- Higher delay-risk stratum (tail @ threshold): {stat['higher_risk_group_tail']}")

    return "\n".join(lines)


def baseline_report_text(km_to, km_nyc) -> str:
    """
    Generate a short baseline KM comparison paragraph for the report.
    Expects fitted lifelines KaplanMeierFitter objects.
    """
    tor_m = km_to.median_survival_time_
    nyc_m = km_nyc.median_survival_time_

    text = f"""
Baseline Survival (Kaplan–Meier)

Toronto:
The baseline Kaplan–Meier curve quantifies the probability that the first unit has not yet arrived as a function of time (minutes). The estimated median time-to-arrival is approximately {tor_m:.2f} minutes (KM median).

NYC:
The baseline Kaplan–Meier curve for NYC is constructed using the same survival definition and time units (minutes). The estimated median time-to-arrival is approximately {nyc_m:.2f} minutes (KM median).

Cross-city comparison:
A faster-dropping survival curve indicates quicker arrivals (lower probability of still waiting). Comparing Toronto vs NYC curves provides a direct baseline view of relative delay risk over time.
"""
    return text.strip()