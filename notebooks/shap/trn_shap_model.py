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
# #### Config + outputs folder (same structure as earlier)

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
# #### Train SHAP-friendly LightGBM model (Toronto)

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
#  Professional plot styling + reusable plot helpers (JIRA requirement)

# %%
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

def save_fig(fig, path, show=True):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=320)
    if show:
        plt.show()
    plt.close(fig)

def plot_shap_summary_pretty(shap_matrix, X, title, max_display=15):
    fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_matrix, X, show=False, max_display=max_display)
    plt.title(title)
    return fig

def plot_shap_bar_pretty(shap_matrix, X, title, max_display=15):
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X, plot_type="bar", show=False, max_display=max_display)
    plt.title(title)
    return fig

def plot_shap_dependence_pretty(shap_matrix, X, feature, title, interaction_index=None):
    fig = plt.figure(figsize=(11, 6))
    shap.dependence_plot(feature, shap_matrix, X, interaction_index=interaction_index, show=False)
    plt.title(title)
    return fig

print("""
Plot functions ready (JIRA compliant)

Story:
These functions RETURN figure objects, making them reusable for:
- reports
- dashboards
- cross-city comparison
""")

# %% [markdown]
# #### Reusable plot functions (RETURN FIG) + save_fig

# %%
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

def save_fig(fig, path, show=True):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=320)
    if show:
        plt.show()
    plt.close(fig)

def plot_shap_summary_pretty(shap_matrix, X, title, max_display=15):
    fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_matrix, X, show=False, max_display=max_display)
    plt.title(title)
    return fig

def plot_shap_bar_pretty(shap_matrix, X, title, max_display=15):
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X, plot_type="bar", show=False, max_display=max_display)
    plt.title(title)
    return fig

def plot_shap_dependence_pretty(shap_matrix, X, feature, title, interaction_index=None):
    fig = plt.figure(figsize=(11, 6))
    shap.dependence_plot(feature, shap_matrix, X, interaction_index=interaction_index, show=False)
    plt.title(title)
    return fig

print("""
Plot functions ready (JIRA compliant)

Story:
These functions RETURN figure objects, making them reusable for:
- reports
- dashboards
- cross-city comparison
""")

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

# %% [markdown]
# #### 3.3 Dependence plots (top 2 drivers) + save + story

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

# %%
top2 = imp["feature"].head(2).tolist()

print(f"""
Step 7 — Dependence plots ({CITY})

Story:
We zoom into the strongest drivers to see how they behave:
- Does risk increase steadily?
- Is there a threshold/tipping point?
Top drivers: {top2}
""")

for f in top2:
    dep_path = os.path.join(OUT_DIR, f"{CITY.lower()}_dependence_{f}.png")

    # high-cardinality categorical: matplotlib-only boxplot
    if f == "location_area":
        print("Detected high-cardinality feature → using Top 10 aggregation (matplotlib only).")

        shap_df = pd.DataFrame({
            f: X_exp[f].astype(str),
            "shap": shap_matrix[:, list(X_exp.columns).index(f)]
        })

        area_importance = (
            shap_df.groupby(f)["shap"]
            .apply(lambda x: np.mean(np.abs(x)))
            .sort_values(ascending=False)
        )

        top_areas = area_importance.head(10).index.tolist()
        shap_top = shap_df[shap_df[f].isin(top_areas)].copy()

        # Order categories by importance (makes plot readable)
        shap_top[f] = pd.Categorical(shap_top[f], categories=top_areas, ordered=True)

        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()

        data = [shap_top.loc[shap_top[f] == a, "shap"].values for a in top_areas]
        ax.boxplot(data, showfliers=False)

        ax.set_xticklabels(top_areas, rotation=45, ha="right")
        ax.set_title(f"{CITY} — SHAP Dependence (Top 10 {f})")
        ax.set_ylabel("SHAP Value")
        ax.set_xlabel(f)

    else:
        fig = plot_shap_dependence_pretty(
            shap_matrix, X_exp, f,
            f"{CITY} — SHAP Dependence: {f}"
        )

    save_fig(fig, dep_path)

    print(f"""
Story for {f}:
This plot explains whether increasing {f} pushes delay risk up or down.
Look for thresholds (sharp bends) that might suggest operational tipping points.
""")
    print("Saved:", dep_path)

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
print(f"""
Step 8 — Waterfall case study ({CITY})

Story:
We pick one high-risk incident and show a “receipt”:
what pushed risk up, what pushed it down, and where the final prediction landed.
""")

probs = model.predict_proba(X_exp)[:, 1]
idx_high = int(np.argmax(probs))
x_row = X_exp.iloc[idx_high:idx_high+1]
p = float(probs[idx_high])

sv = explainer.shap_values(x_row)
vals = sv[1][0] if isinstance(sv, list) else sv[0]

base = float(
    explainer.expected_value[1]
    if isinstance(explainer.expected_value, (list, np.ndarray))
    else explainer.expected_value
)

exp = shap.Explanation(
    values=vals,
    base_values=base,
    data=x_row.iloc[0].values,
    feature_names=list(x_row.columns)
)

fig = plt.figure(figsize=(12, 6))
shap.plots.waterfall(exp, max_display=12, show=False)
plt.title(f"{CITY} — Waterfall Explanation (High-risk case, p={p:.3f})")

wf_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_waterfall_highrisk.png")
save_fig(fig, wf_path)

print(f"""
📖 {CITY} Story:
This incident has predicted delay risk ≈ {p:.3f}.
- Right-side bars increased risk (drivers of delay).
- Left-side bars reduced risk (protective factors).
""")
print(" Saved:", wf_path)

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
print(f"""
Step 9 — Save reusable artifacts ({CITY})

Story:
We save a Git-friendly SHAP sample package and a notes file
so the results can be reused in reporting and dashboards (US5.3, US6.3).
""")

TOPK = 12
SAMPLE_ROWS = min(5000, X_exp.shape[0])

top_feats = imp["feature"].head(TOPK).tolist()
idx = [X_exp.columns.get_loc(f) for f in top_feats]

shap_top = shap_matrix[:SAMPLE_ROWS, :][:, idx]
Xs = X_exp.head(SAMPLE_ROWS)[top_feats].copy()
Ss = pd.DataFrame(shap_top, columns=[f"shap_{f}" for f in top_feats])
out_pdf = pd.concat([Xs.reset_index(drop=True), Ss.reset_index(drop=True)], axis=1)

parquet_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_values_sample.parquet")
out_pdf.to_parquet(parquet_path, index=False)

notes_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_notes.md")
top5 = imp.head(5)

lines = []
lines.append(f"# {CITY} — SHAP Interpretation Notes\n")
lines.append("## Top drivers\n")
for _, r in top5.iterrows():
    lines.append(f"- **{r['feature']}** (mean|SHAP|={r['mean_abs_shap']:.6f})")
lines.append("\n## Key patterns\n")
lines.append("- Beeswarm: direction + spread of feature impact.")
lines.append("- Bar chart: global importance ranking.")
lines.append("- Waterfall: one incident explanation (best for storytelling).")
lines.append("- Positive SHAP → higher delay risk; negative SHAP → lower delay risk.")

with open(notes_path, "w") as f:
    f.write("\n".join(lines))

print("Saved:", parquet_path)
print("Saved:", notes_path)

print(f"""
{CITY} Wrap-up Story:
We translated model predictions into explanations.
Now {CITY} has professional plots + ranking + reusable SHAP artifacts saved in output/.
""")
