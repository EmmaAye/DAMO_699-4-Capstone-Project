import pandas as pd
import matplotlib.pyplot as plt

# 1. Define Paths (Aligning with your existing structure)
TABLE_DIR = "../../output/shap/"
GRAPH_DIR = "../../output/graphs/"

def generate_predictive_importance_outputs(city_name):
    print(f"\n>>> Processing Predictive Importance for {city_name} via SHAP")

    # 2. Load the SHAP Importance Artifact
    # The requirement specifically asks for the *_shap_importance.csv
    file_path = f"{TABLE_DIR}{city_name.lower()}/{city_name.lower()}_shap_importance.csv"
    
    try:
        # Load the SHAP values
        shap_df = pd.read_csv(file_path)
        
        # Sort by importance for the plot
        shap_df = shap_df.sort_values(by=shap_df.columns[1], ascending=True)

        # 3. Deliverable: SHAP bar plot (Top N)
        plt.figure(figsize=(10, 8))
        plt.barh(shap_df.iloc[:, 0], shap_df.iloc[:, 1], color='skyblue')
        plt.title(f"Predictive Feature Importance (SHAP): {city_name}")
        plt.xlabel("mean(|SHAP value|) (Average impact on delay_indicator)")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save Plot
        plot_name = f"{city_name.lower()}_predictive_importance_plot.png"
        plt.savefig(f"{GRAPH_DIR}{plot_name}")
        plt.close()
        
        print(f"Successfully generated plot: {plot_name}")
        print(f"Using existing table: {file_path}")

    except FileNotFoundError:
        print(f"Error: {file_path} not found. Ensure SHAP values were exported previously.")

# 4. Main Execution
if __name__ == "__main__":
    cities = ["toronto", "nyc"]
    for city in cities:
        generate_predictive_importance_outputs(city)
    
    print("\nUS Requirement Check: Predictive feature importance deliverables complete.")