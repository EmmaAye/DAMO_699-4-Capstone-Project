from __future__ import annotations


def cross_city_summary_text(lr: dict, s_to: dict, s_nyc: dict, alpha: float = 0.05) -> str:
    """
    Cross-city KM comparison summary.
    Keeps plain English + optional technical clarification.
    """
    p = float(lr["p_value"])
    sig = p < alpha
    p_txt = "< 1e-300" if p == 0 else f"{p:.3g}"

    lines = []
    lines.append("Cross-city survival comparison (Toronto vs NYC)")
    lines.append(f"- Log-rank test p-value = {p_txt} (α = {alpha}; significant = {sig})")

    lines.append(
        "- Probability that the unit has not arrived yet by time t (minutes):"
    )

    for t in sorted(s_to.keys()):
        lines.append(
            f"  • t={t} min → Toronto={s_to[t]:.3f} | NYC={s_nyc[t]:.3f}"
        )

    lines.append(
        "Interpretation: lower probability at earlier times means faster arrivals "
        "(lower delay risk). Differences later in time reflect tail-risk or extreme delays."
    )

    # optional technical note
    lines.append(
        "Technical note: This probability corresponds to the Kaplan–Meier survival function "
        "(the probability response time exceeds t minutes)."
    )

    return "\n".join(lines)


# def within_city_summary_text(
#     city: str,
#     stat: dict,
#     alpha: float = 0.05,
#     censor_threshold: float = 60.0,
#     include_operational_lines: bool = True,
# ) -> str:
#     """
#     Within-city stratified KM log-rank summary (matches your original notebook format).

#     Expected keys in `stat` (supports multiple variants):
#       - stratification OR group_col
#       - p_value
#       - significant
#       - risk group key can be either:
#           * "higher-risk group (tail @60)"   (notebook summary_df column)
#           * "higher_risk_group_tail"         (library stats output)
#     """
#     # --- Required fields ---
#     pval = float(stat.get("p_value", float("nan")))
#     sig = bool(stat.get("significant", pval < alpha))

#     # --- Strat label (support both styles) ---
#     strat = stat.get("stratification") or stat.get("group_col") or "Stratification"

#     # --- Risk group (support both key names) ---
#     risk = None
#     if "higher-risk group (tail @60)" in stat:
#         risk = stat.get("higher-risk group (tail @60)")
#     elif "higher_risk_group_tail" in stat:
#         risk = stat.get("higher_risk_group_tail")

#     # normalize empty/NaN
#     if risk is not None and (pd.isna(risk) or str(risk).strip() == ""):
#         risk = None

#     # --- p formatting (your old logic) ---
#     p_txt = "< 1e-300" if pval == 0 else f"{pval:.3g}"

#     # --- Build output in your exact old style ---
#     lines = []
#     if sig:
#         lines.append(f"{city} – {strat}:")
#         lines.append(f"Log-rank test indicates significant differences (p = {p_txt}, α = {alpha}).")
#         if risk is not None:
#             lines.append(f"Higher delay-risk group (longer tail at {int(censor_threshold)} min): {risk}.")
#         if include_operational_lines:
#             lines.append("Operational insight: Differences suggest temporal variation in response performance.")
#             lines.append("Check whether curves diverge early (general speed) or mainly in the tail (extreme delays).")
#         lines.append("")  # blank line like your notebook
#     else:
#         lines.append(f"{city} – {strat}: No statistically significant differences detected.\n")

#     return "\n".join(lines)

def within_city_summary_text(stat: dict, alpha: float = 0.05) -> str:
    """
    Format ONE row of within-city logrank results in the same style as the old notebook printing.
    Expects keys like:
      - city
      - stratification
      - p_value
      - significant
      - higher_risk_group_tail (may be None)
    """
    city = stat["city"]
    strat = stat["stratification"]
    pval = float(stat["p_value"])
    sig = bool(stat["significant"])

    p_txt = "< 1e-300" if pval == 0 else f"{pval:.3g}"

    lines = []
    if sig:
        lines.append(f"{city} – {strat}:")
        lines.append(f"Log-rank test indicates significant differences (p = {p_txt}, α = {alpha}).")

        risk = stat.get("higher_risk_group_tail", None)
        if risk is not None and str(risk).strip() != "":
            lines.append(f"Higher delay-risk group (longer tail at 60 min): {risk}.")

        lines.append("Operational insight: Differences suggest temporal variation in response performance.")
        lines.append("Check whether curves diverge early (general speed) or mainly in the tail (extreme delays).")
        lines.append("")  # blank line after each block
    else:
        lines.append(f"{city} – {strat}: No statistically significant differences detected.")
        lines.append("")  # blank line

    return "\n".join(lines)
    
def baseline_report_text(km_to, km_nyc) -> str:
    """
    Baseline KM narrative for report section.
    Plain English first, technical note after.
    """
    tor_m = km_to.median_survival_time_
    nyc_m = km_nyc.median_survival_time_

    text = f"""
Baseline survival (Kaplan–Meier)

Toronto:
The curve shows the probability that the first unit has not arrived yet as time increases. 
The estimated median arrival time is {tor_m:.2f} minutes.

NYC:
The NYC curve uses the same definition and time units. 
The estimated median arrival time is {nyc_m:.2f} minutes.

Cross-city interpretation:
A curve that drops faster indicates quicker arrivals (lower probability of still waiting). 
Comparing both cities shows relative delay risk over time, including early response speed 
and the likelihood of longer delays.

Technical note:
These curves are Kaplan–Meier survival estimates representing the probability that 
response time exceeds a given number of minutes.
"""
    return text.strip()