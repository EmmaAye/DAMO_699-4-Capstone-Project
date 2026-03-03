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
# #### 1. Configuration & Path Set up

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# CONFIG + PATH
# -----------------------
TOP_N = 10  # agreed threshold

PROJECT_FOLDER = "DAMO_699-4-Capstone-Project"
SHAP_ROOT = "output"

cwd = os.getcwd()
print("Current working directory:", cwd)

if PROJECT_FOLDER not in cwd:
    raise ValueError(
        f"Run this notebook from inside '{PROJECT_FOLDER}'. Current dir = {cwd}"
    )

project_root = cwd[:cwd.index(PROJECT_FOLDER) + len(PROJECT_FOLDER)]
base_dir = os.path.join(project_root, SHAP_ROOT)

# Inputs from US5.2
TOR_IMP_PATH = os.path.join(base_dir, "shap", "toronto", "toronto_shap_importance.csv")
NYC_IMP_PATH = os.path.join(base_dir, "shap", "nyc", "nyc_shap_importance.csv")

# Outputs for US5.3
OUT_DIR = os.path.join(base_dir, "shap", "cross_city_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

print("\nUS5.3 Setup")
print("Toronto importance:", TOR_IMP_PATH)
print("NYC importance    :", NYC_IMP_PATH)
print("Outputs directory :", OUT_DIR)

# %% [markdown]
# ### 2. SUBTASK 1: Cross-City Driver Identification & Table Generation

# %% [markdown]
# #### 2.1. Load & Validate Schema

# %%
required_cols = {"feature", "mean_abs_shap", "rank"}

tor = pd.read_csv(TOR_IMP_PATH)
nyc = pd.read_csv(NYC_IMP_PATH)

missing_tor = required_cols - set(tor.columns)
missing_nyc = required_cols - set(nyc.columns)

if missing_tor:
    raise ValueError(f"Toronto CSV missing columns: {missing_tor}")
if missing_nyc:
    raise ValueError(f"NYC CSV missing columns: {missing_nyc}")

# Normalize feature naming consistency (required by JIRA)
tor["feature"] = tor["feature"].astype(str).str.strip().str.lower()
nyc["feature"] = nyc["feature"].astype(str).str.strip().str.lower()

# Type safety
tor["mean_abs_shap"] = pd.to_numeric(tor["mean_abs_shap"], errors="coerce")
nyc["mean_abs_shap"] = pd.to_numeric(nyc["mean_abs_shap"], errors="coerce")
tor["rank"] = pd.to_numeric(tor["rank"], errors="coerce")
nyc["rank"] = pd.to_numeric(nyc["rank"], errors="coerce")

tor = tor.dropna(subset=["feature", "mean_abs_shap", "rank"])
nyc = nyc.dropna(subset=["feature", "mean_abs_shap", "rank"])

print(" Loaded + validated schema.")
print("Toronto rows:", tor.shape[0], "| NYC rows:", nyc.shape[0])

# %% [markdown]
# #### 2.2 Select Top-N Per City + Identify Common/Unique

# %%
tor_top = tor.sort_values("rank").head(TOP_N).copy()
nyc_top = nyc.sort_values("rank").head(TOP_N).copy()

tor_set = set(tor_top["feature"])
nyc_set = set(nyc_top["feature"])

common = sorted(list(tor_set & nyc_set))
tor_only = sorted(list(tor_set - nyc_set))
nyc_only = sorted(list(nyc_set - tor_set))

print(f"Top-{TOP_N} Toronto drivers:", list(tor_top["feature"]))
print(f"Top-{TOP_N} NYC drivers    :", list(nyc_top["feature"]))

print("\nCommon drivers:", common)
print("Toronto-specific:", tor_only)
print("NYC-specific:", nyc_only)

# %% [markdown]
# #### 2.3 Create Required Driver Table (UPGRADED for US6.3)
#
# Output: cross_city_shap_driver_table.csv

# %%
# ==========================================
# SUBTASK 1 — CELL 4
# Cross-City Driver Table (Required Output)
#
# Story:
# We combine Toronto + NYC SHAP rankings into ONE structured driver table:
# - Common drivers
# - Toronto-only drivers
# - NYC-only drivers
#
# This table becomes the key input for:
# - US5.3 visualization
# - US6.3 operational interpretation
# ==========================================

print(f"""
Step — Build Cross-City SHAP Driver Table (Top-{TOP_N})

Story:
We are building a single comparison table that clearly answers:
1) Which drivers are common across both cities?
2) Which are city-specific (if any)?
3) For common drivers, how do rank and importance differ?

Output:
cross_city_shap_driver_table.csv
""")

# ----------------------------
# 1) Create easy lookup maps
# ----------------------------
tor_rank = tor.set_index("feature")["rank"].to_dict()
nyc_rank = nyc.set_index("feature")["rank"].to_dict()

tor_imp  = tor.set_index("feature")["mean_abs_shap"].to_dict()
nyc_imp  = nyc.set_index("feature")["mean_abs_shap"].to_dict()

# ----------------------------
# 2) Helper to build one row
# ----------------------------
def make_row(category: str, feature: str):
    tr = tor_rank.get(feature)
    nr = nyc_rank.get(feature)

    tv = float(tor_imp.get(feature, 0.0))
    nv = float(nyc_imp.get(feature, 0.0))

    # Differences (useful for US6.3)
    rank_diff = (tr - nr) if (tr is not None and nr is not None) else None
    imp_diff  = tv - nv

    return {
        "category": category,
        "feature": feature,
        "toronto_rank": tr,
        "nyc_rank": nr,
        "toronto_mean_abs_shap": tv,
        "nyc_mean_abs_shap": nv,
        "rank_diff_tor_minus_nyc": rank_diff,
        "abs_importance_diff": abs(imp_diff),
        "importance_gap_pct_tor_vs_nyc": ((tv - nv) / (nv + 1e-9)) * 100,
        "in_toronto_topN": feature in tor_set,
        "in_nyc_topN": feature in nyc_set
    }

# ----------------------------
# 3) Build rows (JIRA categories)
# ----------------------------
rows = []
for f in common:
    rows.append(make_row("common", f))

for f in tor_only:
    rows.append(make_row("toronto_specific", f))

for f in nyc_only:
    rows.append(make_row("nyc_specific", f))

driver_table = (
    pd.DataFrame(rows)
    .sort_values(["category", "abs_importance_diff"], ascending=[True, False])
    .reset_index(drop=True)
)

# ----------------------------
# 4) Save required output CSV
# ----------------------------
out_driver_csv = os.path.join(OUT_DIR, "cross_city_shap_driver_table.csv")
driver_table.to_csv(out_driver_csv, index=False)

print("✅ Subtask 1 Output Saved:")
print(out_driver_csv)

display(driver_table.head(20))


# %% [markdown]
# #### 3. SUBTASK 2: Comparative Visualization & Technical Write-Up

# %% [markdown]
# #### 3.1 Clean Comparative Plot (Presentation-ready)
# Output: cross_city_shap_comparison_clean.png

# %%
def plot_clean_side_by_side(tor_top, nyc_top, out_dir, top_n=10):
    tor_map = dict(zip(tor_top["feature"], tor_top["mean_abs_shap"]))
    nyc_map = dict(zip(nyc_top["feature"], nyc_top["mean_abs_shap"]))

    features = sorted(set(tor_map.keys()).union(set(nyc_map.keys())))

    tor_vals = np.array([tor_map.get(f, 0.0) for f in features])
    nyc_vals = np.array([nyc_map.get(f, 0.0) for f in features])

    order = np.argsort(-np.maximum(tor_vals, nyc_vals))
    features = [features[i] for i in order][:top_n]
    tor_vals = tor_vals[order][:top_n]
    nyc_vals = nyc_vals[order][:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    axes[0].barh(features[::-1], tor_vals[::-1])
    axes[0].set_title("Toronto — SHAP Feature Importance", fontsize=14, weight="bold")
    axes[0].set_xlabel("Mean |SHAP|", fontsize=12)

    axes[1].barh(features[::-1], nyc_vals[::-1])
    axes[1].set_title("NYC — SHAP Feature Importance", fontsize=14, weight="bold")
    axes[1].set_xlabel("Mean |SHAP|", fontsize=12)

    plt.suptitle("Cross-City SHAP Driver Comparison", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(out_dir, f"cross_city_shap_comparison_clean_top{top_n}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    return plot_path

plot_path = plot_clean_side_by_side(tor_top, nyc_top, OUT_DIR, top_n=TOP_N)
print("Comparative plot saved:", plot_path)


# %% [markdown]
# #### 3.2 Short Technical Write-Up (NO policy interpretation)
# Output: cross_city_shap_writeup.md

# %%
# =========================
# SUBTASK 2 — Write-up (technical only, US5.3-safe)
# Auto-adds:
# - "All Top-N drivers are shared" when true
# - Largest magnitude differences from driver_table
# =========================

def fmt_list(xs):
    return ", ".join(xs) if xs else "None"

# Detect "all shared" condition
all_shared_flag = (len(tor_only) == 0 and len(nyc_only) == 0 and len(common) == TOP_N)

all_shared_line = (
    f" All Top-{TOP_N} drivers are shared across Toronto and NYC (same feature set; order/magnitude may differ)."
    if all_shared_flag
    else f" Top-{TOP_N} drivers differ across cities (shared + city-specific drivers exist)."
)

# Largest magnitude differences (technical: mean|SHAP| gap)
# Using the columns we created in Subtask 1 table
# abs_importance_diff and importance_gap_pct_tor_vs_nyc
top_diffs = (
    driver_table.sort_values("abs_importance_diff", ascending=False)
    .head(3)[["feature", "toronto_mean_abs_shap", "nyc_mean_abs_shap", "abs_importance_diff", "importance_gap_pct_tor_vs_nyc"]]
)

diff_bullets = []
for _, r in top_diffs.iterrows():
    diff_bullets.append(
        f"- {r['feature']}: "
        f"Toronto={r['toronto_mean_abs_shap']:.4f}, "
        f"NYC={r['nyc_mean_abs_shap']:.4f}, "
        f"|Δ|={r['abs_importance_diff']:.4f}, "
        f"Gap% (Tor vs NYC)={r['importance_gap_pct_tor_vs_nyc']:.1f}%"
    )

diff_section = "\n".join(diff_bullets) if diff_bullets else "- None"

writeup = f"""# US5.3 — Cross-City SHAP Explainability Comparison (Predictive)

## Inputs (from US5.2)
- toronto_shap_importance.csv
- nyc_shap_importance.csv  
(No new SHAP computation performed.)

## Summary (Top-{TOP_N})
{all_shared_line}

## Top-{TOP_N} Drivers (Toronto)
{fmt_list(list(tor_top["feature"]))}

## Top-{TOP_N} Drivers (NYC)
{fmt_list(list(nyc_top["feature"]))}

## Shared vs City-Specific Drivers (Top-{TOP_N})
**Common Drivers:** {fmt_list(common)}  
**Toronto-Specific Drivers:** {fmt_list(tor_only)}  
**NYC-Specific Drivers:** {fmt_list(nyc_only)}  

## Largest Cross-City Magnitude Differences (Mean |SHAP|)
Story (technical):
Even when the same drivers appear in both cities, models may differ in how *strongly* they rely on each feature.
Below are the top differences in mean |SHAP| magnitude:

{diff_section}

## Technical Notes
- Shared drivers suggest structural similarity in predictive patterns across cities.
- Rank and magnitude differences are captured in `cross_city_shap_driver_table.csv`.
- Operational/policy interpretation is intentionally deferred to US6.3.

## Outputs Produced (US5.3)
- cross_city_shap_driver_table.csv
- cross_city_shap_comparison_clean_top{TOP_N}.png
- cross_city_shap_writeup_top{TOP_N}.md
"""

md_path = os.path.join(OUT_DIR, f"cross_city_shap_writeup_top{TOP_N}.md")
with open(md_path, "w") as f:
    f.write(writeup)

print(writeup)
print("\n Write-up saved:", md_path)

# %% [markdown]
# #### 3.3 Definition of Done Summary

# %%
print(f"""
US5.3 — Definition of Done Checklist

Subtask 1:
✔ Feature naming consistency validated
✔ Top-{TOP_N} features per city selected
✔ Common / Toronto-specific / NYC-specific drivers identified
✔ cross_city_shap_driver_table.csv generated
   -> {out_driver_csv}

Subtask 2:
✔ Comparative plot generated (presentation-ready)
   -> {plot_path}
✔ Technical write-up prepared (no policy interpretation)
   -> {md_path}

All outputs saved under:
{OUT_DIR}

Ready for US6.3 operational interpretation.
""")
