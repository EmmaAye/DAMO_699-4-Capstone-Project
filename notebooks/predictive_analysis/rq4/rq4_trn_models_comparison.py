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

print("Starting RQ4 TORONTO comparison compilation...")

base_output_dir = os.path.abspath("../../../output")
tables_dir = os.path.join(base_output_dir, "tables")
graphs_dir = os.path.join(base_output_dir, "graphs")

naive_df = pd.read_csv(os.path.join(tables_dir, "rq4_trn_naive_majority_class_metrics.csv"))
lr_df = pd.read_csv(os.path.join(tables_dir, "rq4_trn_logistic_regression_metrics.csv"))
rf_df = pd.read_csv(os.path.join(tables_dir, "rq4_trn_random_forest_metrics.csv"))
gbt_df = pd.read_csv(os.path.join(tables_dir, "rq4_trn_gbt_classifier_metrics.csv"))

comparison_df = pd.concat([naive_df, lr_df, rf_df, gbt_df], ignore_index=True)
comparison_df = comparison_df[
    ["Model", "AUC-ROC", "PR-AUC", "Precision", "Recall", "F1-Score", "Accuracy"]
].sort_values(by="AUC-ROC", ascending=False)

comparison_df.to_csv(os.path.join(tables_dir, "rq4_trn_model_comparison.csv"), index=False)

display(comparison_df.round(3))

lr_roc = pd.read_csv(os.path.join(graphs_dir, "rq4_trn_logistic_regression_roc_points.csv"))
rf_roc = pd.read_csv(os.path.join(graphs_dir, "rq4_trn_random_forest_roc_points.csv"))
gbt_roc = pd.read_csv(os.path.join(graphs_dir, "rq4_trn_gbt_classifier_roc_points.csv"))

auc_lookup = dict(zip(comparison_df["Model"], comparison_df["AUC-ROC"]))

plt.figure(figsize=(10, 6))
plt.plot(lr_roc["fpr"], lr_roc["tpr"], label=f"Logistic Regression (AUC={auc_lookup['Logistic Regression']:.3f})")
plt.plot(rf_roc["fpr"], rf_roc["tpr"], label=f"Random Forest (AUC={auc_lookup['Random Forest']:.3f})")
plt.plot(gbt_roc["fpr"], gbt_roc["tpr"], label=f"GBT Classifier (AUC={auc_lookup['GBT Classifier']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Naive Majority Class")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison of Delay Prediction Models (Toronto)",fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "rq4_trn_combined_roc_curve.png"), dpi=300)
plt.show()

plot_df = comparison_df.set_index("Model")[["AUC-ROC", "PR-AUC", "Precision", "Recall", "F1-Score", "Accuracy"]]
plot_df.plot(kind="bar", figsize=(11, 6))
plt.title("Toronto Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "rq4_trn_model_comparison_bar_chart.png"), dpi=300)
plt.show()

best_model = comparison_df.iloc[0]
display(pd.DataFrame([best_model]))
pd.DataFrame([best_model]).to_csv(os.path.join(tables_dir, "rq4_trn_best_model_summary.csv"), index=False)

print("RQ4 TORONTO comparison complete.")
