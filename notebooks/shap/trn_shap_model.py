# ============================================
# New Cell
# ============================================

!pip install lightgbm shap

# ============================================
# New Cell
# ============================================

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

# ============================================
# New Cell
# ============================================

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

# ============================================
# New Cell
# ============================================

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

# ============================================
# New Cell
# ============================================

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

# ============================================
# New Cell
# ============================================

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

# ============================================
# New Cell
# ============================================

# ============================================
# Toronto — Fix: Keep only ONE save_fig definition
# File: notebooks/shap/trn_shap_model.ipynb
# ============================================

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
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_matrix, X, show=False, max_display=max_display)
    plt.title(title)
    return plt.gcf()

def plot_shap_bar_pretty(shap_matrix, X, title, max_display=15):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X, plot_type="bar", show=False, max_display=max_display)
    plt.title(title)
    return plt.gcf()

def plot_shap_dependence_pretty(shap_matrix, X, feature, title):
    plt.figure(figsize=(11, 6))
    shap.dependence_plot(feature, shap_matrix, X, show=False)
    plt.title(title)
    return plt.gcf()

print("Toronto plotting helpers ready (single save_fig, return fig).")

# ============================================
# New Cell
# ============================================

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

# ============================================
# New Cell
# ============================================

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

# ============================================
# New Cell
# ============================================

# ============================================
# Toronto — Fix: Step 8 Waterfall (robust to list/2D/3D SHAP)
# File: notebooks/shap/trn_shap_model.ipynb
# ============================================

import numpy as np
import shap
import matplotlib.pyplot as plt

print(f"""
Step 8 — Waterfall case study ({CITY})

We pick one high-risk incident and show the SHAP breakdown:
what pushed risk up, what pushed it down, and the final predicted probability.
""")

probs = model.predict_proba(X_exp)[:, 1]
idx_high = int(np.argmax(probs))

x_row = X_exp.iloc[idx_high:idx_high+1]
p = float(probs[idx_high])

sv = explainer.shap_values(x_row)

# Normalize to numpy
if isinstance(sv, list):
    vals = sv[1]
else:
    vals = sv
vals = np.asarray(vals)

# Handle 3D / 2D
if vals.ndim == 3:
    if vals.shape[-1] == 2:
        vals = vals[0, :, 1]
    elif vals.shape[1] == 2:
        vals = vals[0, 1, :]
    else:
        vals = vals[0, :]
elif vals.ndim == 2:
    vals = vals[0]

ev = explainer.expected_value
if isinstance(ev, (list, np.ndarray)):
    base = float(ev[1]) if len(ev) > 1 else float(ev[0])
else:
    base = float(ev)

exp = shap.Explanation(
    values=vals,
    base_values=base,
    data=x_row.iloc[0].values,
    feature_names=list(x_row.columns)
)

plt.figure(figsize=(12, 6))
shap.plots.waterfall(exp, max_display=12, show=False)
plt.title(f"{CITY} — Waterfall Explanation (High-risk case, p={p:.3f})")

wf_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_waterfall_highrisk.png")
save_fig(plt.gcf(), wf_path)

print("Saved:", wf_path)

# ============================================
# New Cell
# ============================================

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

