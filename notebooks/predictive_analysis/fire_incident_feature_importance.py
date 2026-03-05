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

SHAP_DIR = "../../output/shap/"
FIG_DIR  = "../../output/graphs/"
TOP_N = 15

os.makedirs(FIG_DIR, exist_ok=True)

def generate_predictive_importance_outputs(city: str, top_n: int = TOP_N):
    city = city.lower()
    print(f"\n>>> Processing Predictive Importance for {city.upper()} via SHAP")

    file_path = os.path.join(SHAP_DIR, city, f"{city}_shap_importance.csv")

    shap_df = pd.read_csv(file_path)

    # enforce schema (safer)
    required = {"feature", "mean_abs_shap"}
    if not required.issubset(set(shap_df.columns)):
        raise ValueError(f"{city}: SHAP CSV missing required columns {required}. Found={shap_df.columns.tolist()}")

    # rank descending (most important first)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = range(1, len(shap_df) + 1)

    # Top-N for plotting (reverse for horizontal plot readability)
    top_df = shap_df.head(top_n).sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"])
    plt.title(f"{city.upper()} — Predictive Feature Importance (SHAP) Top {top_n}")
    plt.xlabel("mean(|SHAP|)")
    plt.tight_layout()

    plot_path = os.path.join(FIG_DIR, f"{city}_predictive_shap_importance_top{top_n}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved plot: {plot_path}")
    print(f"Used table: {file_path} (full ranking already exported)")

if __name__ == "__main__":
    for city in ["toronto", "nyc"]:
        generate_predictive_importance_outputs(city)

    print("\n Predictive feature importance deliverables complete (plots + ranked tables from SHAP CSVs).")
