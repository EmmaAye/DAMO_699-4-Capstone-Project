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

# =========================================================
# 1. SETUP
# =========================================================
base_output_dir = "../../../output"
tables_dir = f"{base_output_dir}/tables"
graphs_dir = f"{base_output_dir}/graphs"

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# =========================================================
# 2. LOAD METRICS FILES
# =========================================================
naive_df = pd.read_csv(f"{tables_dir}/rq1_trn_naive_baseline_metrics.csv")
lr_df = pd.read_csv(f"{tables_dir}/rq1_trn_logistic_regression_metrics.csv")
rf_df = pd.read_csv(f"{tables_dir}/rq1_trn_random_forest_metrics.csv")
gbt_df = pd.read_csv(f"{tables_dir}/rq1_trn_gbt_metrics.csv")

# =========================================================
# 3. COMBINE COMPARISON TABLE
# =========================================================
comparison_df = pd.concat(
    [naive_df, lr_df, rf_df, gbt_df],
    ignore_index=True
)

comparison_df = comparison_df[
    ["Model", "AUC-ROC", "Precision", "Recall", "F1-Score", "Accuracy"]
].sort_values(by="AUC-ROC", ascending=False)

comparison_df.to_csv(f"{tables_dir}/rq1_trn_model_comparison.csv", index=False)

print("=====================================================")
print("TORONTO MODEL COMPARISON TABLE")
print("=====================================================")
# print(comparison_df.to_string(index=False))
display(comparison_df)

# =========================================================
# 4. SAVE ROUNDED / PRETTY VERSION
# =========================================================
comparison_pretty_df = comparison_df.copy()

for c in ["AUC-ROC", "Precision", "Recall", "F1-Score", "Accuracy"]:
    comparison_pretty_df[c] = comparison_pretty_df[c].round(3)

comparison_pretty_df.to_csv(
    f"{tables_dir}/rq1_trn_model_comparison_rounded.csv",
    index=False
)

# =========================================================
# 5. LOAD ROC POINTS
# =========================================================
lr_roc = pd.read_csv(f"{graphs_dir}/rq1_trn_logistic_regression_roc_points.csv")
rf_roc = pd.read_csv(f"{graphs_dir}/rq1_trn_random_forest_roc_points.csv")
gbt_roc = pd.read_csv(f"{graphs_dir}/rq1_trn_gbt_roc_points.csv")

# Get AUC values from comparison table
auc_lookup = dict(zip(comparison_df["Model"], comparison_df["AUC-ROC"]))

# =========================================================
# 6. COMBINED ROC CURVE PLOT
# =========================================================
plt.figure(figsize=(10, 6))

plt.plot(
    lr_roc["fpr"], lr_roc["tpr"],
    label=f"Logistic Regression (AUC = {auc_lookup['Logistic Regression']:.3f})"
)

plt.plot(
    rf_roc["fpr"], rf_roc["tpr"],
    label=f"Random Forest (AUC = {auc_lookup['Random Forest']:.3f})"
)

plt.plot(
    gbt_roc["fpr"], gbt_roc["tpr"],
    label=f"GBT (AUC = {auc_lookup['GBT']:.3f})"
)

# Naive baseline / random line
plt.plot([0, 1], [0, 1], "k--", label="Naive Majority Baseline (AUC = 0.500)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Toronto Model Comparison - ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{graphs_dir}/rq1_trn_combined_roc_curve.png", dpi=300)
plt.show()

print("\nCombined ROC curve saved.")

# =========================================================
# 7. OPTIONAL: BAR CHART OF MODEL METRICS
# =========================================================
plot_df = comparison_df.set_index("Model")[["AUC-ROC", "Precision", "Recall", "F1-Score", "Accuracy"]]

plt.figure(figsize=(10, 6))
plot_df.plot(kind="bar", figsize=(10, 6))
plt.title("Toronto Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{graphs_dir}/rq1_trn_model_comparison_bar_chart.png", dpi=300)
plt.show()

print("Comparison bar chart saved.")
print("\nCompilation complete.")
