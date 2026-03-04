# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# ## 0. Import Libraries

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Set Paths

# %%
BASE_DIR  = "/Workspace/Repos/jihirosan@gmail.com/damo_699-4-capstone-project/output"
TABLE_DIR = f"{BASE_DIR}/tables"
FIG_DIR   = f"{BASE_DIR}/graphs"   # or /figures if that’s your convention
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TOP_N = 15

HR_TORONTO = f"{TABLE_DIR}/cox_hr_Toronto.csv"
HR_NYC     = f"{TABLE_DIR}/cox_hr_NYC.csv"


# %% [markdown]
# ## 2. Helper Functions

# %%
def standardize_cox_hr(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # detect feature column
    feat_candidates = ["feature", "covariate", "variable", "term", "name"]
    feat_col = next((c for c in feat_candidates if c in df.columns), None)
    if feat_col is None:
        df = df.reset_index().rename(columns={"index": "feature"})
        feat_col = "feature"

    # detect coef
    coef_col = "coef" if "coef" in df.columns else None

    # detect hr
    hr_candidates = ["hr", "hazard_ratio", "exp(coef)", "exp_coef"]
    hr_col = next((c for c in hr_candidates if c in df.columns), None)

    # detect p
    p_candidates = ["p", "p_value", "p-value"]
    p_col = next((c for c in p_candidates if c in df.columns), None)

    # detect CI
    lo_candidates = ["ci_lower", "hr_lower_95", "exp(coef) lower 95%", "lower 95%"]
    hi_candidates = ["ci_upper", "hr_upper_95", "exp(coef) upper 95%", "upper 95%"]
    lo_col = next((c for c in lo_candidates if c in df.columns), None)
    hi_col = next((c for c in hi_candidates if c in df.columns), None)

    out = pd.DataFrame({
        "feature": df[feat_col].astype(str)
    })

    if coef_col:
        out["coef"] = pd.to_numeric(df[coef_col], errors="coerce")
    if hr_col:
        out["hr"] = pd.to_numeric(df[hr_col], errors="coerce")
    if p_col:
        out["p_value"] = pd.to_numeric(df[p_col], errors="coerce")
    if lo_col:
        out["ci_lower"] = pd.to_numeric(df[lo_col], errors="coerce")
    if hi_col:
        out["ci_upper"] = pd.to_numeric(df[hi_col], errors="coerce")

    return out

def rank_and_plot(city: str, hr_path: str, top_n: int = 15):
    df_raw = pd.read_csv(hr_path)
    df = standardize_cox_hr(df_raw)

    # choose ranking metric
    # Prefer abs(coef) if present, else abs(log(HR))
    if "coef" in df.columns and df["coef"].notna().any():
        df["importance"] = df["coef"].abs()
        metric_name = "|coef|"
    elif "hr" in df.columns and df["hr"].notna().any():
        df["importance"] = np.abs(np.log(df["hr"].clip(lower=1e-9)))
        metric_name = "|log(HR)|"
    else:
        raise ValueError(f"{city}: cannot compute importance (missing coef/hr columns). Columns={df.columns.tolist()}")

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # save full table + topN
    full_path = f"{TABLE_DIR}/cox_{city.lower()}_feature_importance_full.csv"
    top_path  = f"{TABLE_DIR}/cox_{city.lower()}_feature_importance_top{top_n}.csv"
    df.to_csv(full_path, index=False)
    df.head(top_n).to_csv(top_path, index=False)

    # plot topN
    top_df = df.head(top_n).iloc[::-1]  # reverse for barh
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title(f"{city} — Survival Feature Importance (Cox) by {metric_name} (Top {top_n})")
    plt.xlabel(f"Importance = {metric_name}")
    plt.tight_layout()

    fig_path = f"{FIG_DIR}/cox_{city.lower()}_feature_importance_top{top_n}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"[{city}] Saved tables:")
    print(" -", full_path)
    print(" -", top_path)
    print(f"[{city}] Saved plot:")
    print(" -", fig_path)



# %% [markdown]
# ## 3. Feature Importance for Toronto

# %%
rank_and_plot("Toronto", HR_TORONTO, top_n=TOP_N)

# %% [markdown]
# ## 4. Feature Importance for NYC

# %%
rank_and_plot("NYC", HR_NYC, top_n=TOP_N)

# %% [markdown]
# ## 5. Survival Feature Importance Summary (Cox Models)
#
# The Cox survival models identify **incident type as the strongest driver of response-time outcomes in both cities**. In Toronto, the highest importance features include *False Alarm / No Action*, *Hazardous / Utility incidents*, and *Other Assistance*, indicating that operational context and incident classification strongly influence response dynamics. Temporal factors such as **night-time incidents** and operational characteristics like **alarm level** also contribute meaningfully.
#
# In NYC, the effect of incident type is even more pronounced. **Structural and Non-Structural Fire incidents dominate the feature importance rankings**, followed by other incident categories such as *False Alarm*, *Hazardous / Utility*, and *Rescue / Entrapment*. This suggests that **incident severity and type play a larger role in response-time variability in NYC compared with Toronto**.
#
# Demand indicators such as **recent call volume (calls_past_30min and calls_past_60min)** appear with moderate importance in Toronto but are less dominant in NYC, indicating that **incident characteristics rather than short-term demand fluctuations are the primary drivers of response-time differences in the survival models**.
#
# Overall, the survival analysis highlights that **incident category, operational priority (alarm level), and time-of-day effects are key structural determinants of response-time performance**, with demand effects playing a secondary but still observable role.
