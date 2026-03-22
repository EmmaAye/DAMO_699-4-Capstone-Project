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
import os
import pandas as pd
import matplotlib.pyplot as plt

print("Starting RQ4 NYC comparison compilation...")

# =========================================================
# 1. PATHS
# =========================================================
base_output_dir = os.path.abspath("../../../output")
tables_dir = os.path.join(base_output_dir, "tables")
graphs_dir = os.path.join(base_output_dir, "graphs")

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# =========================================================
# 2. LOAD METRICS FILES
# =========================================================
naive_path = os.path.join(tables_dir, "rq4_nyc_naive_majority_class_metrics.csv")
lr_path = os.path.join(tables_dir, "rq4_nyc_logistic_regression_metrics.csv")
rf_path = os.path.join(tables_dir, "rq4_nyc_random_forest_metrics.csv")
gbt_path = os.path.join(tables_dir, "rq4_nyc_gbt_classifier_metrics.csv")

naive_df = pd.read_csv(naive_path)
lr_df = pd.read_csv(lr_path)
rf_df = pd.read_csv(rf_path)
gbt_df = pd.read_csv(gbt_path)

# =========================================================
# 3. COMBINE COMPARISON TABLE
# =========================================================
comparison_df = pd.concat(
    [naive_df, lr_df, rf_df, gbt_df],
    ignore_index=True
)

comparison_df = comparison_df[
    ["Model", "AUC-ROC", "PR-AUC", "Precision", "Recall", "F1-Score", "Accuracy"]
].sort_values(by="AUC-ROC", ascending=False)

comparison_path = os.path.join(tables_dir, "rq4_nyc_model_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)

print("RQ4 NYC MODEL COMPARISON TABLE")
display(comparison_df)

print(f"Comparison table saved to: {comparison_path}")

# =========================================================
# 4. LOAD ROC POINTS
# =========================================================
lr_roc_path = os.path.join(graphs_dir, "rq4_nyc_logistic_regression_roc_points.csv")
rf_roc_path = os.path.join(graphs_dir, "rq4_nyc_random_forest_roc_points.csv")
gbt_roc_path = os.path.join(graphs_dir, "rq4_nyc_gbt_classifier_roc_points.csv")

lr_roc = pd.read_csv(lr_roc_path)
rf_roc = pd.read_csv(rf_roc_path)
gbt_roc = pd.read_csv(gbt_roc_path)

auc_lookup = dict(zip(comparison_df["Model"], comparison_df["AUC-ROC"]))

# =========================================================
# 5. COMBINED ROC CURVE
# =========================================================
plt.figure(figsize=(10,6))

plt.plot(
    lr_roc["fpr"], lr_roc["tpr"],
    label=f"Logistic Regression (AUC={auc_lookup['Logistic Regression']:.3f})"
)

plt.plot(
    rf_roc["fpr"], rf_roc["tpr"],
    label=f"Random Forest (AUC={auc_lookup['Random Forest']:.3f})"
)

plt.plot(
    gbt_roc["fpr"], gbt_roc["tpr"],
    label=f"GBT Classifier (AUC={auc_lookup['GBT Classifier']:.3f})"
)

# Naive baseline
plt.plot([0,1], [0,1], "k--", label="Naive Majority Baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison of Delay Prediction Models (NYC)", fontweight = "bold")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()

combined_roc_path = os.path.join(graphs_dir, "rq4_nyc_combined_roc_curve.png")
plt.savefig(combined_roc_path, dpi=300)
plt.show()

print(f"Combined ROC curve saved to: {combined_roc_path}")

# =========================================================
# 6. MODEL PERFORMANCE BAR CHART
# =========================================================
plot_df = comparison_df.set_index("Model")[
    ["AUC-ROC","PR-AUC","Precision","Recall","F1-Score","Accuracy"]
]

plot_df.plot(kind="bar", figsize=(11,6))

plt.title("RQ4 NYC Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=20)
plt.tight_layout()

bar_chart_path = os.path.join(graphs_dir, "rq4_nyc_model_comparison_bar_chart.png")
plt.savefig(bar_chart_path, dpi=300)
plt.show()

print(f"Comparison bar chart saved to: {bar_chart_path}")

# =========================================================
# 7. BEST MODEL
# =========================================================
best_model = comparison_df.iloc[0]

print("BEST MODEL")
display(pd.DataFrame([best_model]))

best_model_summary_path = os.path.join(tables_dir, "rq4_nyc_best_model_summary.csv")
pd.DataFrame([best_model]).to_csv(best_model_summary_path, index=False)

print(f"Best model summary saved to: {best_model_summary_path}")

print("RQ4 NYC comparison complete.")
