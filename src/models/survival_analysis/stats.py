# Ensure lifelines is installed (safe install pattern)
from lifelines.statistics import logrank_test, multivariate_logrank_test
import pandas as pd
from .constants import STRATA_SPECS


def cross_city_logrank(to_df, nyc_df,
                       duration_col="response_minutes",
                       event_col="event_indicator",
                       label_a="Toronto",
                       label_b="NYC"):
    # assumes to_df and nyc_df are pandas DataFrames
    a = to_df[[duration_col, event_col]].dropna()
    b = nyc_df[[duration_col, event_col]].dropna()

    # ensure numeric/binary
    a = a[a[event_col].isin([0,1])]
    b = b[b[event_col].isin([0,1])]

    res = logrank_test(
        a[duration_col].values,
        b[duration_col].values,
        event_observed_A=a[event_col].values,
        event_observed_B=b[event_col].values,
    )

    p = float(res.p_value)
    p_txt = "< 1e-300" if p == 0.0 else f"{p:.3g}"

    return {
        "test": "log-rank",
        "group_A": label_a,
        "group_B": label_b,
        "test_statistic": float(res.test_statistic),
        "p_value": p,
        "p_value_text": p_txt,
        "n_A": int(len(a)),
        "n_B": int(len(b)),
        "events_A": int(a[event_col].sum()),
        "events_B": int(b[event_col].sum()),
    }

def within_city_multivariate_logrank(
    df_pd,
    group_col: str,
    alpha: float = 0.05,
    censor_threshold: float = 60.0,
    duration_col="response_minutes",
    event_col="event_indicator",
    group_order=None,  # <-- FIX: accept but do not use (ordering is for plotting, not the test)
):
    """
    Multivariate log-rank across strata within a city.
    Returns p-value, significance, and (optionally) which stratum has highest tail risk at censor_threshold.
    """
    res = multivariate_logrank_test(
        df_pd[duration_col],
        df_pd[group_col],
        df_pd[event_col],
    )
    pval = float(res.p_value)
    significant = pval < alpha

    risk_group = None
    if significant:
        # Tail-risk at threshold: highest S(threshold)
        from lifelines import KaplanMeierFitter

        s_at_t = {}
        for g in sorted(df_pd[group_col].dropna().astype(str).unique()):
            sub = df_pd[df_pd[group_col].astype(str) == g]
            if len(sub) == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub[duration_col], sub[event_col])
            s_at_t[g] = float(kmf.predict(censor_threshold))

        if len(s_at_t) > 0:
            risk_group = max(s_at_t, key=s_at_t.get)

    return {
        "group_col": group_col,
        "p_value": pval,
        "significant": significant,
        "higher_risk_group_tail": risk_group,
    }


def run_city_logrank_tests(
    pdf: pd.DataFrame,
    city_name: str,
    censor_threshold: float = 60.0,
    alpha: float = 0.05,
    strata_specs=None,
) -> pd.DataFrame:
    """
    Run within-city multivariate log-rank tests only (no plots).
    """
    specs = STRATA_SPECS if strata_specs is None else strata_specs

    rows = []
    for group_col, label, group_order in specs:
        res = within_city_multivariate_logrank(
            df_pd=pdf,
            group_col=group_col,
            censor_threshold=censor_threshold,
            group_order=group_order,
            alpha=alpha,
        )
        rows.append({"city": city_name, "stratification": label, **res})

    return pd.DataFrame(rows)
