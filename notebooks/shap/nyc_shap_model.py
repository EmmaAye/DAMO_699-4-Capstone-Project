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

CITY = "NYC"
TABLE = "workspace.capstone_project.nyc_model_ready"
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

# use your real folder name here:
OUT_DIR = os.path.join(project_root, "output", "shap", CITY.lower())
os.makedirs(OUT_DIR, exist_ok=True)

print(" Saving outputs to:", OUT_DIR)

# %% [markdown]
# Load data + label sanity check

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

print(f" {CITY} label check passed — both classes exist.")

# %% [markdown]
# Train LightGBM model (SHAP-friendly surrogate)

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

# small stability improvement for categorical splits
model.fit(X_train, y_train, categorical_feature=cat_cols)

print(f"""
Model trained ({CITY})

Story:
Now we compute SHAP values to explain which features push delay risk up/down.
""")

# %% [markdown]
# Compute SHAP (FIXED: categorical splits) + validate shapes

# %%
# ============================================
# Cell 3 — Compute SHAP values (NYC) [SAFE + robust across SHAP versions]
# ============================================

import numpy as np
import shap

print(f"""
Step 3 — Compute SHAP values ({CITY})

Story:
Because LightGBM can use categorical splits, we use:
feature_perturbation="tree_path_dependent"
and we DO NOT pass background data.
""")

# Use Booster if model is sklearn wrapper (LGBMClassifier)
lgbm_for_shap = model.booster_ if hasattr(model, "booster_") else model

explainer = shap.TreeExplainer(
    lgbm_for_shap,
    feature_perturbation="tree_path_dependent"
)

shap_values = explainer.shap_values(X_exp)

# Robust handling
if isinstance(shap_values, list):
    # binary classifier -> [class0, class1]
    shap_matrix = shap_values[1]
else:
    shap_matrix = shap_values

shap_matrix = np.asarray(shap_matrix)

# Rare: some SHAP versions return 3D arrays; reduce if needed
if shap_matrix.ndim == 3:
    # try common pattern: (n_samples, n_features, n_classes)
    shap_matrix = shap_matrix[:, :, 1]

print("SHAP shape:", shap_matrix.shape)
print("X shape   :", X_exp.shape)

if shap_matrix.shape != X_exp.shape:
    raise ValueError(f"Shape mismatch. SHAP={shap_matrix.shape}, X={X_exp.shape}")

nyc_shap_values = shap_matrix

print(f"""
SHAP computed + validated ({CITY})

Story:
We can now explain which features increase vs decrease delay risk.
""")

# %% [markdown]
# Feature importance ranking + CSV export (JIRA schema)

# %%
# ============================================
# Cell 4 — Feature importance ranking (NYC)
# ============================================

import os
import pandas as pd
import numpy as np

print(f"""
Step 4 — Feature importance ranking ({CITY})

Story:
We summarize many SHAP explanations into one global ranking using mean(|SHAP|).
This produces the “top drivers” list required by JIRA.
""")

mean_abs = np.mean(np.abs(shap_matrix), axis=0).astype(float)

imp = pd.DataFrame({
    "feature": list(X_exp.columns),
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

imp["rank"] = np.arange(1, len(imp) + 1)

csv_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_importance.csv")
imp[["feature", "mean_abs_shap", "rank"]].to_csv(csv_path, index=False)

print("Saved CSV:", csv_path)
display(spark.createDataFrame(imp.head(12)))

# %% [markdown]
# Professional plot styling + reusable plot helpers (JIRA requirement)

# %%
# ============================================
# Cell 5 — Reusable Plotting Functions (JIRA-Compliant: return fig)
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
    fig = plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_matrix, X, show=False, max_display=max_display)
    plt.title(title)
    return fig

def plot_shap_bar_pretty(shap_matrix, X, title, max_display=15):
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X, plot_type="bar", show=False, max_display=max_display)
    plt.title(title)
    return fig

def plot_shap_dependence_pretty(shap_matrix, X, feature, title):
    fig = plt.figure(figsize=(11, 6))
    shap.dependence_plot(feature, shap_matrix, X, show=False)
    plt.title(title)
    return fig

print("""
Plot functions ready (JIRA compliant).

Story:
These functions now RETURN figure objects,
so they can be reused cleanly in reports/dashboards.
""")

# %% [markdown]
# “Beautiful” global plots (summary + top-N bar) + storytelling

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

# 1) Beeswarm summary
fig1 = plot_shap_summary_pretty(
    shap_matrix,
    X_exp,
    f"{CITY} — SHAP Summary (Top {TOP_N} drivers)",
    max_display=TOP_N
)
save_fig(fig1, summary_path)

print("""
Story:
Each dot is an incident.
- Right → increases delay risk
- Left → decreases delay risk
Color shows low→high feature values.
""")

# 2) Bar plot
fig2 = plot_shap_bar_pretty(
    shap_matrix,
    X_exp,
    f"{CITY} — Global Feature Importance (mean |SHAP|, Top {TOP_N})",
    max_display=TOP_N
)
save_fig(fig2, bar_path)

# FIX: top3 must come from imp (feature importance table), not shap_matrix
if "imp" not in globals():
    # fallback: compute imp quickly if the ranking cell wasn't run
    import numpy as np
    import pandas as pd
    mean_abs = np.mean(np.abs(shap_matrix), axis=0).astype(float)
    imp = pd.DataFrame({"feature": list(X_exp.columns), "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)

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
# Dependence plots (top 2 drivers) + save + story

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
    print(f"\n Generating dependence plot for: {f}")

    dep_path = os.path.join(OUT_DIR, f"{CITY.lower()}_dependence_{f}.png")

    # Capture figure then save
    fig = plot_shap_dependence_pretty(
        shap_matrix,
        X_exp,
        f,
        f"{CITY} — SHAP Dependence: {f}"
    )
    save_fig(fig, dep_path)

    print(f"""
Story for {f}:
This plot explains whether increasing {f} pushes delay risk up or down.

• Positive SHAP → increases delay probability
• Negative SHAP → reduces delay probability

Look for:
- steady increase/decrease (monotonic effect)
- sharp bends (threshold / tipping point)
- scattered clusters (nonlinear interactions)
""")
    print("Saved:", dep_path)

# %% [markdown]
# Waterfall “case study” (most professional storytelling plot)

# %%
print(f"""
Step 8 — Waterfall case study ({CITY})

Story:
This is the most presentation-ready explanation.
We pick one high-risk incident and show a “receipt”:
what pushed risk up, what pushed it down, and where the final prediction landed.
""")

import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# 1) Pick a high-risk case
probs = model.predict_proba(X_exp)[:, 1]
idx_high = int(np.argmax(probs))
x_row = X_exp.iloc[idx_high:idx_high+1]
p = float(probs[idx_high])

# 2) Compute SHAP for this single row (robust)
sv = explainer.shap_values(x_row)

if isinstance(sv, list):
    # binary: [class0, class1] -> use class1
    vals = np.asarray(sv[1])[0]
else:
    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        # (n_samples, n_features, n_classes) -> class1
        vals = sv_arr[0, :, 1]
    else:
        # (n_samples, n_features)
        vals = sv_arr[0]

# 3) Base value for class 1 (robust)
ev = explainer.expected_value
if isinstance(ev, (list, np.ndarray)):
    base = float(ev[1]) if len(ev) > 1 else float(ev[0])
else:
    base = float(ev)

# 4) Build SHAP Explanation object
exp = shap.Explanation(
    values=vals,
    base_values=base,
    data=x_row.iloc[0].values,
    feature_names=list(x_row.columns)
)

# 5) Plot + SAVE (JIRA compliant)
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

This plot is ideal for stakeholder storytelling because it shows a
clear “audit trail” of how the prediction was formed.
""")
print(" Saved:", wf_path)

# %% [markdown]
# Save SHAP sample parquet + notes.md (JIRA requirement)

# %%
print(f"""
Step 9 — Save reusable artifacts ({CITY})

Story:
We save a Git-friendly SHAP sample package and a notes file
so the results can be reused in reporting and dashboards (US5.3, US6.3).
""")

import os
import pandas as pd
import numpy as np

TOPK = 12
SAMPLE_ROWS = min(5000, X_exp.shape[0])

# --- Pick top features safely ---
top_feats = imp["feature"].head(TOPK).tolist()

# Guard: ensure all top features exist in X_exp
top_feats = [f for f in top_feats if f in X_exp.columns]
if len(top_feats) == 0:
    raise ValueError("No top features found in X_exp columns. Check imp['feature'] vs X_exp.columns.")

# Indices for shap_matrix
idx = [X_exp.columns.get_loc(f) for f in top_feats]

# --- Build SHAP sample package ---
shap_top = shap_matrix[:SAMPLE_ROWS, :][:, idx]
Xs = X_exp.head(SAMPLE_ROWS)[top_feats].copy()
Ss = pd.DataFrame(shap_top, columns=[f"shap_{f}" for f in top_feats])

out_pdf = pd.concat([Xs.reset_index(drop=True), Ss.reset_index(drop=True)], axis=1)

# Use CITY in filenames (works for Toronto + NYC)
base_name = f"{CITY.lower()}_shap_values_sample"

# Prefer parquet (best), fallback to csv if parquet engine missing
parquet_path = os.path.join(OUT_DIR, f"{base_name}.parquet")
csv_path     = os.path.join(OUT_DIR, f"{base_name}.csv")

saved_paths = []

try:
    out_pdf.to_parquet(parquet_path, index=False)
    saved_paths.append(parquet_path)
except Exception as e:
    print(" Parquet save failed (likely missing pyarrow/fastparquet). Falling back to CSV.")
    print("   Error:", str(e)[:200], "...")
    out_pdf.to_csv(csv_path, index=False)
    saved_paths.append(csv_path)

# --- Notes file (technical, JIRA-safe) ---
notes_path = os.path.join(OUT_DIR, f"{CITY.lower()}_shap_notes.md")
top5 = imp.head(5)

lines = []
lines.append(f"# {CITY} — SHAP Interpretation Notes\n")
lines.append("## Top drivers (mean |SHAP|)\n")
for _, r in top5.iterrows():
    lines.append(f"- **{r['feature']}** (mean|SHAP|={float(r['mean_abs_shap']):.6f})")

lines.append("\n## Key technical patterns\n")
lines.append("- Beeswarm: direction + spread of feature impact across incidents.")
lines.append("- Bar chart: global importance ranking by mean(|SHAP|).")
lines.append("- Dependence: how each top feature changes risk (thresholds/nonlinearity).")
lines.append("- Waterfall: one high-risk case explanation (best for storytelling).")
lines.append("- Interpretation rule: **positive SHAP → higher delay risk**, **negative SHAP → lower delay risk**.")

with open(notes_path, "w") as f:
    f.write("\n".join(lines))

saved_paths.append(notes_path)

print("\n Saved artifacts:")
for p in saved_paths:
    print(" -", p)

print(f"""
{CITY} Wrap-up Story:
We translated model predictions into explanations.
Now {CITY} has professional plots + ranking + reusable SHAP artifacts saved in the project outputs folder.
""")
