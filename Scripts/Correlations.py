import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_correlations():
    print("--- Generating Correlation Analysis ---")
    
    # 1. Robust Loading
    filename = 'Universal_QSPR_Weighted_Indices_Final.csv'
    if not os.path.exists(filename):
        alt_name = 'Universal_QSPR_Weighted_Indices_Final.csv.xlsx - Universal_QSPR_Weighted_Indices.csv'
        if os.path.exists(alt_name):
            filename = alt_name
        else:
            print("❌ Error: File not found.")
            return

    try:
        df = pd.read_csv(filename)
    except:
        try:
            df = pd.read_csv(filename, encoding='latin1')
        except:
            df = pd.read_excel(filename)

    # 2. Identify Columns
    # Indices (X)
    index_cols = [c for c in df.columns if c.startswith(('SO', 'ESO', 'MESO'))]
    
    # Properties (Y) - The 6 we are studying
    # We must match the names exactly as they appear in your file
    target_cols = [
        'Boiling Point (BP)', 
        'Molar Volume (MV)', 
        'Flash Point (FP)', 
        'Polarizability (Pol)', 
        'Molar Refractivity (MR)', 
        'Density (D)'
    ]
    
    # Filter only columns that actually exist in the file
    target_cols = [c for c in target_cols if c in df.columns]

    print(f"Indices: {len(index_cols)}")
    print(f"Properties: {len(target_cols)}")

    # 3. Calculate Pearson Correlation
    # We want a matrix of [Indices] vs [Properties]
    correlation_matrix = df[index_cols + target_cols].corr(method='pearson')
    
    # We only care about Index vs Property (slice the matrix)
    # Rows = Indices, Columns = Properties
    final_matrix = correlation_matrix.loc[index_cols, target_cols]

    # 4. Save the CSV (For Supplementary Data)
    final_matrix.to_csv("Supplementary_Correlations.csv")
    print("✅ Saved 'Supplementary_Correlations.csv'")

    # 5. Generate the Heatmap (For Main Paper)
    plt.figure(figsize=(10, 12)) # Tall and narrow
    
    # Create Heatmap
    # cmap='coolwarm' makes Positive=Red, Negative=Blue
    sns.heatmap(final_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                linewidths=0.5, linecolor='white', cbar_kws={'label': 'Pearson Correlation (r)'})
    
    plt.title('Correlation between Weighted Sombor Indices\nand Physicochemical Properties', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Properties', fontsize=12, fontweight='bold')
    plt.ylabel('Topological Indices', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('Paper_Figure_3_Correlation_Heatmap.png', dpi=300)
    print("✅ Saved 'Paper_Figure_3_Correlation_Heatmap.png'")

if __name__ == "__main__":
    generate_correlations()