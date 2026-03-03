# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
# #!pip install lightgbm shap

# %% [markdown]
# %md
# #### Config + outputs folder

# %%
import os

CITY = "Toronto"
TABLE = "workspace.capstone_project.toronto_model_ready"
LABEL = "delay_indicator"

cwd = os.getcwd()
print("Current working directory:", cwd)

marker = "DAMO_699-4-Capstone-Project"
if marker not in cwd:
    raise ValueError(
        f"Not inside project folder '{marker}'. Move/open this notebook inside the project folder.\n"
        f"cwd={cwd}\n"
        f"Expected to find '{marker}' in the path."
    )

project_root = cwd[:cwd.index(marker) + len(marker)]

#  Use your desired folder name: "output" (as you showed)
OUT_DIR = os.path.join(project_root, "output", "shap", CITY.lower())
os.makedirs(OUT_DIR, exist_ok=True)

print(" Saving outputs to:", OUT_DIR)

# %% [markdown]
# #### 1. Load data + label sanity check

# %%
from pyspark.sql.functions import col

df = (
    spark.table(TABLE)
    .filter(col(LABEL).isNotNull())
    .withColumn(LABEL, col(LABEL).cast("int"))
)

print(f"""
Step 1 — Data sanity ({CITY})

Story:
Before we explain a model, we confirm the target has both outcomes (0 and 1).
If we only have one class, there is no meaningful signal to learn or explain.
""")

df.groupBy(LABEL).count().orderBy(LABEL).show()

labels = [r[LABEL] for r in df.select(LABEL).distinct().collect()]
if len(labels) < 2:
    raise ValueError(f"{CITY}: only one class found in {LABEL}: {labels}")

print(f"{CITY} label check passed — both classes exist.")

# %% [markdown]
# #### 2. Train SHAP-friendly LightGBM model (Toronto)

# %%
import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 42

CATEGORICAL = ["incident_category", "season", "unified_call_source", "location_area"]
NUMERIC     = ["hour", "day_of_week", "month", "year", "unified_alarm_level",
               "calls_past_30min", "calls_past_60min"]

existing = set(df.columns)
cat_cols = [c for c in CATEGORICAL if c in existing]
num_cols = [c for c in NUMERIC if c in existing]

if len(cat_cols) + len(num_cols) == 0:
    raise ValueError(f"No usable feature columns found in {TABLE}. Available columns: {df.columns}")

df_model = df.select(*(num_cols + cat_cols + [LABEL]))

print(f"""
Step 2 — Train SHAP-friendly model ({CITY})

Story:
We train a model using real feature names (not hashed),
so that SHAP explanations are readable for stakeholders.
""")
print("Numeric:", num_cols)
print("Categorical:", cat_cols)

train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=SEED)
train_df = train_df.sample(False, 0.35, seed=SEED).limit(200_000)
test_df  = test_df.limit(8_000)

train_pdf = train_df.toPandas()
test_pdf  = test_df.toPandas()

for c in cat_cols:
    train_pdf[c] = train_pdf[c].astype("category")
    test_pdf[c]  = test_pdf[c].astype("category")

X_train = train_pdf[num_cols + cat_cols]
y_train = train_pdf[LABEL].astype(int)

X_exp = test_pdf[num_cols + cat_cols]
y_exp = test_pdf[LABEL].astype(int)

model = lgb.LGBMClassifier(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED
)

# important stability improvement for categorical splits
model.fit(X_train, y_train, categorical_feature=cat_cols)

print(f"""
Model trained ({CITY})

Story:
Now we compute SHAP values to explain which features push delay risk up/down.
""")

# %% [markdown]
# #### 3. Compute SHAP values (Toronto) + shape check

# %%
import shap
import numpy as np

print(f"""
Step 3 — Compute SHAP values ({CITY})

Story:
Because LightGBM uses categorical splits, we must use:
feature_perturbation="tree_path_dependent"
(and no background data).
""")

lgbm_for_shap = model.booster_ if hasattr(model, "booster_") else model

explainer = shap.TreeExplainer(
    lgbm_for_shap,
    feature_perturbation="tree_path_dependent"
)

shap_values = explainer.shap_values(X_exp)

# binary classifier: [class0, class1]
shap_matrix = shap_values[1] if isinstance(shap_values, list) else shap_values
shap_matrix = np.asarray(shap_matrix)

print("SHAP shape:", shap_matrix.shape)
print("X shape   :", X_exp.shape)

if shap_matrix.shape != X_exp.shape:
    raise ValueError(f"Shape mismatch. SHAP={shap_matrix.shape}, X={X_exp.shape}")

toronto_shap_values = shap_matrix

print(f"""
SHAP computed + validated ({CITY})

Story:
We can now tell a clear story of what increases delay risk and what reduces it.
""")

# %% [markdown]
# #### 3.1 Feature importance ranking + CSV export (JIRA schema)

# %%
print(f"""
Step 4 — Feature importance ranking ({CITY})

Story:
We summarize many SHAP explanations into one global ranking using mean(|SHAP|).
This gives the “top drivers” list required by JIRA.
""")

mean_abs = np.mean(np.abs(shap_matrix), axis=0)
imp = pd.DataFrame({"feature": X_exp.columns, "mean_abs_shap": mean_abs})
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp["rank"] = np.arange(1, len(imp) + 1)

csv_path = os.path.join(OUT_DIR, "toronto_shap_importance.csv")
imp[["feature", "mean_abs_shap", "rank"]].to_csv(csv_path, index=False)

print("Saved CSV:", csv_path)
display(spark.createDataFrame(imp.head(12)))

# %% [markdown]
# #### 3.2 Reusable plot functions (RETURN FIG) + save_fig

# %%
# Plotting Helpers — single definition (use this in NYC + TRN)

import numpy as np
import matplotlib.pyplot as plt
import shap

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 320,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

def save_fig(fig, path, show=False, dpi=320):
    # layout first (safe)
    try:
        fig.tight_layout()
    except Exception:
        pass

    # critical render step (prevents blank images in Databricks)
    try:
        fig.canvas.draw()
        plt.pause(0.001)
    except Exception:
        pass

    fig.savefig(path, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)

def plot_shap_summary_pretty(shap_matrix, X, title, max_display=15):
    plt.close("all")
    shap_matrix = np.asarray(shap_matrix)
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_matrix, X, show=False, max_display=max_display)
    plt.title(title)
    return plt.gcf()

def plot_shap_bar_pretty(shap_matrix, X, title, max_display=15):
    plt.close("all")
    shap_matrix = np.asarray(shap_matrix)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X, plot_type="bar", show=False, max_display=max_display)
    plt.title(title)
    return plt.gcf()

def plot_shap_dependence_pretty(shap_matrix, X, feature, title, interaction_index=None):
    plt.close("all")
    shap_matrix = np.asarray(shap_matrix)
    shap.dependence_plot(
        feature,
        shap_matrix,
        X,
        interaction_index=interaction_index,
        show=False
    )
    plt.title(title)
    return plt.gcf()


# %% [markdown]
# #### Toronto: SHAP Explainability Summary (Global & Distributional View)
#
# The SHAP analysis shows that **location_area** is the dominant driver of predicted delay risk in Toronto, with the largest mean absolute SHAP value and the widest spread of impacts across incidents. This indicates that geographic differences play the strongest role in shaping response-time delay predictions.
#
# **Hour of day** is the second most influential factor, confirming that temporal conditions meaningfully affect delay risk. The beeswarm distribution further suggests that certain hours consistently increase or decrease predicted risk rather than producing uniform effects.
#
# Operational context variables — particularly **unified_call_source** and **incident_category** — also rank highly in global importance, indicating that both incident characteristics and reporting pathways influence how quickly incidents are resolved.
#
# Short-term workload indicators (**calls_past_30min** and, to a lesser extent, **calls_past_60min**) contribute measurable but smaller effects, suggesting that recent demand levels modify delay risk without dominating model behavior.
#
# Calendar-related variables (**month**, **day_of_week**, **season**, and **year**) show comparatively modest influence, with SHAP values clustered near zero. These features act primarily as secondary adjustments rather than primary determinants.
#
# Overall, combining global importance rankings with SHAP distribution patterns indicates that Toronto delay risk is primarily structured by **spatial factors and time-of-day conditions**, while **incident context and short-term demand** function as meaningful but secondary modifiers of prediction outcomes.

# %%
TOP_N = 15

print(f"""
Step 6 — Global plots ({CITY})

Story:
1) Beeswarm summary: direction + spread of influence.
2) Bar plot (Top {TOP_N}): clean executive view.
""")

summary_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_summary_pretty_top{TOP_N}.png")
bar_path     = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_bar_pretty_top{TOP_N}.png")

fig1 = plot_shap_summary_pretty(
    shap_matrix, X_exp,
    f"{CITY} — SHAP Summary (Top {TOP_N} drivers)",
    max_display=TOP_N
)
save_fig(fig1, summary_path)

fig2 = plot_shap_bar_pretty(
    shap_matrix, X_exp,
    f"{CITY} — Global Feature Importance (mean |SHAP|, Top {TOP_N})",
    max_display=TOP_N
)
save_fig(fig2, bar_path)

top3 = imp["feature"].head(3).tolist()
print(f"""
{CITY} Story headline:
The model’s strongest drivers of delay risk are:
1) {top3[0]}
2) {top3[1]}
3) {top3[2]}
""")

print("Saved:", summary_path)
print("Saved:", bar_path)

# %% [markdown]
# #### Location-Based Effects (`location_area`)
#
# The SHAP dependence analysis shows strong variation in delay risk across geographic areas.
#
# - Several locations consistently produce **positive SHAP values**, indicating structurally higher predicted delay risk.
# - Other areas show strongly negative SHAP values, suggesting systematically faster response outcomes.
# - The separation between location groups is large and stable, implying persistent spatial differences rather than random variation.
#
# These results indicate that operational conditions tied to geographic areas play a major role in shaping delay risk.
#

# %% [markdown]
# #### 3.3 Dependence plots (top 2 drivers)

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

shap_matrix = np.asarray(shap_matrix)

print(f"""
Step 7 — Dependence plots ({CITY})

Story:
We zoom into the strongest drivers to see how they behave:
- Does risk increase steadily?
- Is there a threshold/tipping point?
""")

def categorical_topk_boxplot(X, shap_matrix, feature, top_k=10, title=None):
    if feature not in X.columns:
        raise ValueError(f"Missing column in X_exp: {feature}")

    j = list(X.columns).index(feature)
    dfp = pd.DataFrame({feature: X[feature].astype(str), "shap": shap_matrix[:, j]})

    top_cats = (
        dfp.groupby(feature)["shap"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    dfp = dfp[dfp[feature].isin(top_cats)].copy()
    dfp[feature] = pd.Categorical(dfp[feature], categories=top_cats, ordered=True)

    plt.close("all")
    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()

    data = [dfp.loc[dfp[feature] == c, "shap"].values for c in top_cats]
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(top_cats, rotation=45, ha="right")

    ax.set_ylabel("SHAP value")
    ax.set_xlabel(feature)
    ax.set_title(title if title else f"{CITY} — SHAP Dependence (Top {top_k} {feature})")

    fig.set_size_inches(14, 6)
    return fig

# ---- Plot 1: hour colored by calls_past_60min (matches your Toronto screenshot style) ----
f = "hour"
color_by = "calls_past_60min"

if f not in X_exp.columns:
    raise ValueError(f"Missing: {f}")
if color_by not in X_exp.columns:
    raise ValueError(f"Missing: {color_by}")

path1 = os.path.join(OUT_DIR, "toronto_dependence_hour.png")

fig1 = plot_shap_dependence_pretty(
    shap_matrix=shap_matrix,
    X=X_exp,
    feature=f,
    title="Toronto — SHAP Dependence: hour",
    interaction_index=color_by
)
save_fig(fig1, path1, show=False)
print("Saved:", path1, "bytes:", os.path.getsize(path1))

# ---- Plot 2: location_area (Top-10 boxplot; same approach as NYC) ----
path2 = os.path.join(OUT_DIR, "toronto_dependence_location_area.png")

fig2 = categorical_topk_boxplot(
    X=X_exp,
    shap_matrix=shap_matrix,
    feature="location_area",
    top_k=10,
    title="Toronto — SHAP Dependence (Top 10 location_area)"
)
save_fig(fig2, path2, show=False)
print("Saved:", path2, "bytes:", os.path.getsize(path2))

# %% [markdown]
#
# #### Temporal Effects (`hour`)
#
# The dependence plot for `hour` reveals a nonlinear relationship between time-of-day and predicted delay risk.
#
# - Certain hours are associated with higher SHAP values, meaning increased probability of delay.
# - Earlier or lower-activity periods tend to reduce predicted risk.
# - Coloring by recent workload (`calls_past_60min`) suggests interaction effects, where higher demand amplifies delay risk during sensitive hours.
#
# This indicates that delay risk is influenced by **time-of-day combined with operational workload**, rather than time alone.

# %% [markdown]
# #### Key Insight
#
# Dependence analysis confirms that:
#
# - Delay risk varies systematically across locations.
# - Temporal effects exist but interact with workload intensity.
# - Model predictions are driven by contextual operational conditions rather than single variables in isolation.

# %% [markdown]
# ### 3.4 Waterfall “case study” (most professional storytelling plot)

# %%
# ============================================
# Step 8 — Waterfall case study (Toronto)
# ============================================

import numpy as np
import shap
import matplotlib.pyplot as plt
import os

print(f"""
Step 8 — Waterfall case study ({CITY})

Story:
This is the most presentation-ready local explanation.
We pick one high-risk incident and show which features pushed risk up/down.
""")

# 1) Pick a high-risk row
probs = model.predict_proba(X_exp)[:, 1]
idx_high = int(np.argmax(probs))
x_row = X_exp.iloc[idx_high:idx_high+1]
p = float(probs[idx_high])

# 2) Get SHAP values for that single row
sv = explainer.shap_values(x_row)

# Normalize to numpy
sv_arr = np.asarray(sv) if not isinstance(sv, list) else None

# 3) Extract class-1 SHAP values safely
# Common cases:
# - list: [class0, class1] -> take class1 row 0
# - ndarray 3D: (n_rows, n_features, n_classes) -> take class1
# - ndarray 2D: (n_rows, n_features) -> take row 0
if isinstance(sv, list):
    vals = np.asarray(sv[1])[0]
else:
    if sv_arr.ndim == 3:
        # assume last dim is classes
        vals = sv_arr[0, :, 1] if sv_arr.shape[-1] >= 2 else sv_arr[0, :, 0]
    else:
        vals = sv_arr[0]

vals = np.asarray(vals).reshape(-1)

# 4) Base value safely (class-1 if available)
ev = explainer.expected_value
if isinstance(ev, (list, np.ndarray)):
    base = float(ev[1]) if len(ev) >= 2 else float(ev[0])
else:
    base = float(ev)

# 5) Build Explanation + plot
exp = shap.Explanation(
    values=vals,
    base_values=base,
    data=x_row.iloc[0].values,
    feature_names=list(x_row.columns)
)

fig = plt.figure(figsize=(12, 6))
shap.plots.waterfall(exp, max_display=12, show=False)
plt.title(f"{CITY} — Waterfall Explanation (High-risk case, p={p:.3f})")
plt.tight_layout()

wf_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_waterfall_highrisk.png")
save_fig(fig, wf_path)

print(f"""
Waterfall saved: {wf_path}

Story:
Predicted delay risk ≈ {p:.3f}.
Positive bars increase risk; negative bars reduce risk.
This is the best plot for stakeholder storytelling.
""")

# %% [markdown]
# #### Toronto — SHAP Waterfall Case Study (High-Risk Incident)
#
# The SHAP waterfall plot provides a local explanation for a single high-risk incident with a predicted delay probability of **0.863**. The visualization decomposes the model prediction into feature-level contributions, showing how each factor increased or reduced delay risk relative to the baseline expectation.
#
# ##### Main Risk Drivers
#
# The prediction is primarily driven upward by contextual and operational factors:
#
# - **Location area** is the strongest contributor, producing the largest positive impact on delay risk.
# - **Incident category** and **call source** substantially increase predicted risk, indicating that incident context strongly influences response outcomes.
# - **Hour of occurrence** also contributes positively, suggesting elevated risk during specific time periods.
# - Temporal attributes such as **month** and **year** add smaller incremental effects.
#
# Together, these factors shift the prediction significantly above the model’s baseline risk level.
#
# ##### Mitigating Factors
#
# A small number of variables slightly reduce predicted risk:
#
# - Lower short-term workload (`calls_past_30min`)
# - Day-of-week effects
#
# However, their influence is minor compared with the dominant positive drivers.
#
# ##### Interpretation
#
# This case study demonstrates that high delay risk emerges from the **combined effect of location, incident context, and timing**, rather than a single variable alone. The waterfall explanation illustrates how the model aggregates multiple operational signals to produce a final risk estimate, improving transparency of individual predictions.

# %% [markdown]
# ### 3.5 Save SHAP sample parquet + notes.md (JIRA requirement)

# %%
# ============================================
# Toronto — Fix: Step 9 Save artifacts (parquet fallback to csv)
# File: notebooks/shap/trn_shap_model.ipynb
# ============================================

TOPK = 12
SAMPLE_ROWS = min(5000, X_exp.shape[0])

top_feats = imp["feature"].head(TOPK).tolist()
idx = [X_exp.columns.get_loc(f) for f in top_feats]

shap_top = shap_matrix[:SAMPLE_ROWS, :][:, idx]
Xs = X_exp.head(SAMPLE_ROWS)[top_feats].copy()
Ss = pd.DataFrame(shap_top, columns=[f"shap_{f}" for f in top_feats])
out_pdf = pd.concat([Xs.reset_index(drop=True), Ss.reset_index(drop=True)], axis=1)

parquet_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_values_sample.parquet")
csv_fallback = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_values_sample.csv")

try:
    out_pdf.to_parquet(parquet_path, index=False)
    print("Saved:", parquet_path)
except Exception as e:
    out_pdf.to_csv(csv_fallback, index=False)
    print("Parquet write failed, saved CSV instead.")
    print("Reason:", repr(e))
    print("Saved:", csv_fallback)

notes_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_notes.md")

top5 = imp.head(5)
lines = []
lines.append(f"# {CITY} — SHAP Interpretation Notes\n")
lines.append("## Top drivers\n")
for _, r in top5.iterrows():
    lines.append(f"- **{r['feature']}** (mean|SHAP|={r['mean_abs_shap']:.6f})")
lines.append("\n## Key patterns\n")
lines.append("- Summary plot: direction + spread of feature impact.")
lines.append("- Bar plot: global importance ranking.")
lines.append("- Dependence plots: feature behavior and thresholds.")
lines.append("- Waterfall: one incident explanation for storytelling.")
lines.append("- Positive SHAP increases delay risk; negative SHAP decreases delay risk.")

with open(notes_path, "w") as f:
    f.write("\n".join(lines))

print("Saved:", notes_path)

# %% [markdown]
# ## 3.6 City-specific Drivers (TRN)

# %% [markdown]
# #### City-Level Drivers (Top-N for This Notebook)
#
# In cross-city comparison (US5.3), Top-10 drivers often appear identical because both datasets were harmonized into a shared feature schema (time, demand, location, alarm, call source, incident category). That means the most dominant predictors tend to overlap.
#
# To better surface *city-specific* patterns, this notebook also reports **Top-{TOP_CITY_N}** drivers for the current city. City-specific differences are more likely to appear in mid-tier drivers (Top-15/Top-20) through:
# - rank shifts,
# - differences in mean(|SHAP|) magnitude,
# - and sensitivity differences (how strongly the feature moves delay risk).
#
# This Top-{TOP_CITY_N} driver list is saved as a separate CSV for clean reuse in reporting and cross-city analysis.

# %%
# --------------------------------------------
# City-specific driver threshold 
# --------------------------------------------
TOP_CITY_N = 15 

# %%
# ============================================
# City-specific driver list (Toronto) — Top-N for this notebook
# ============================================

top_city = imp["feature"].head(TOP_CITY_N).tolist()

top_city_path = os.path.join(OUT_DIR, f"{CITY.lower()}_top{TOP_CITY_N}_drivers.csv")
(
    imp.loc[imp["feature"].isin(top_city), ["feature", "mean_abs_shap", "rank"]]
      .sort_values("rank")
      .to_csv(top_city_path, index=False)
)

print(f"Top-{TOP_CITY_N} Toronto drivers saved to:")
print(top_city_path)

print(f"Top-{TOP_CITY_N} Toronto drivers:")
print(top_city)
