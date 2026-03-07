import pandas as pd
import os

# 1. Paths
INPUT_DIR = "../../output/shap/cross_city_comparison/"
OUTPUT_DIR = "../../output/interpretations/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_interpretation_report():
    input_file = os.path.join(INPUT_DIR, "cross_city_shap_driver_table.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 2. Load and Sort
    df = pd.read_csv(input_file)
    
    # 3. Enhanced Logic based on your column headers
    # Shared: Small rank difference (|diff| <= 1)
    shared = df[df['rank_diff_tor_minus_nyc'].abs() <= 1]
    
    # Toronto-Dominant: Positive importance gap (higher in TO) and large rank diff
    # or significantly higher SHAP values
    to_dominant = df[df['importance_gap_pct_tor_vs_nyc'] > 50].sort_values(by='importance_gap_pct_tor_vs_nyc', ascending=False)
    
    # NYC-Dominant: Negative importance gap (higher in NYC)
    nyc_dominant = df[df['importance_gap_pct_tor_vs_nyc'] < -20].sort_values(by='importance_gap_pct_tor_vs_nyc')

    # 4. Generate the Report
    report_path = os.path.join(OUTPUT_DIR, "driver_interpretation_notes.txt")
    
    with open(report_path, "w") as f:
        f.write("PREDICTIVE DRIVER INTERPRETATION REPORT\n")
        f.write("Source: cross_city_shap_driver_table\n")
        f.write("="*60 + "\n\n")

        # Section 1: Shared Drivers
        f.write("### 1. SHARED DRIVERS (Symmetric Importance)\n")
        f.write("These features show consistent ranking across both cities, suggesting universal predictors of delay:\n")
        for _, row in shared.iterrows():
            f.write(f"- {row['feature']}: (TO Rank: {row['toronto_rank']}, NYC Rank: {row['nyc_rank']})\n")
        f.write("\n")

        # Section 2: Toronto-Specific Drivers
        f.write("### 2. TORONTO-SPECIFIC DRIVERS (High Gap)\n")
        f.write("Features that have a significantly higher predictive impact in Toronto:\n")
        for _, row in to_dominant.iterrows():
            if row['feature'] not in shared['feature'].values:
                f.write(f"- {row['feature']}: {row['importance_gap_pct_tor_vs_nyc']:.1f}% higher impact in Toronto.\n")
        f.write("\n")

        # Section 3: NYC-Specific Drivers
        f.write("### 3. NYC-SPECIFIC DRIVERS (High Gap)\n")
        f.write("Features that drive the model more heavily for NYC incidents:\n")
        for _, row in nyc_dominant.iterrows():
            if row['feature'] not in shared['feature'].values:
                f.write(f"- {row['feature']}: {abs(row['importance_gap_pct_tor_vs_nyc']):.1f}% higher impact in NYC.\n")
        
        # Section 4: Critical Insight (Top Divergence)
        f.write("\n### 4. CRITICAL INSIGHT: TOP DIVERGENCE\n")
        max_diff_feat = df.loc[df['abs_importance_diff'].idxmax()]
        f.write(f"The feature with the greatest absolute importance gap is '{max_diff_feat['feature']}' \n")
        f.write(f"with an absolute difference of {max_diff_feat['abs_importance_diff']:.4f} SHAP units.\n")

    print(f"Success! Detailed notes generated at: {report_path}")

if __name__ == "__main__":
    generate_interpretation_report()