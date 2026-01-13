import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

def analyze_all_properties_importance():
    print("--- Step 1: Loading Data (Robust Mode) ---")
    
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

    # Identify Features (Indices)
    feature_cols = [c for c in df.columns if c.startswith(('SO', 'ESO', 'MESO'))]
    
    # Identify Targets (Properties)
    # We exclude ID, Name, SMILES, and the indices themselves
    exclude_cols = ['ID', 'Name', 'SMILES'] + feature_cols
    target_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Indices found: {len(feature_cols)}")
    print(f"Properties to analyze: {target_cols}")
    print("-" * 60)

    # --- Step 2: The Grand Loop ---
    for target_name in target_cols:
        print(f"\nProcessing: {target_name}...")
        
        # Filter valid data for this specific property
        data = df.dropna(subset=[target_name])
        
        if len(data) < 50:
            print(f"   ⚠️ Skipping (Not enough data: {len(data)} rows)")
            continue

        X = data[feature_cols]
        y = data[target_name]
        
        # Scale X (Standard practice for comparison)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # --- A. Linear Correlation (Pearson) ---
        # We use absolute value because negative correlation is just as important as positive
        correlations = data[feature_cols].corrwith(data[target_name]).abs()
        sorted_corr = correlations.sort_values(ascending=False)

        # --- B. Random Forest Importance (Non-Linear) ---
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

        # --- C. XGBoost Importance (Gradient Boosting) ---
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        xgb.fit(X_scaled, y)
        xgb_importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)

        # --- Step 3: Create the Top 10 Table ---
        top_n = 10
        
        rank_df = pd.DataFrame({
            'Rank': range(1, top_n + 1),
            
            # Linear Columns
            'Linear_Index': sorted_corr.index[:top_n],
            'Pearson_r': sorted_corr.values[:top_n].round(4),
            
            # RF Columns
            'RF_Index': rf_importances.index[:top_n],
            'RF_Importance': rf_importances.values[:top_n].round(4),
            
            # XGB Columns
            'XGB_Index': xgb_importances.index[:top_n],
            'XGB_Importance': xgb_importances.values[:top_n].round(4)
        })

        # --- Step 4: Save to CSV ---
        # Clean filename (remove special chars)
        safe_name = target_name.split('(')[0].strip().replace(' ', '_')
        out_file = f"Importance_Table_{safe_name}.csv"
        
        rank_df.to_csv(out_file, index=False)
        print(f"   ✅ Saved: {out_file}")

    print("\n" + "="*60)
    print("🎉 All Tables Generated Successfully!")
    print("You can now open these CSV files and copy them into your thesis as 'Table 3'.")

if __name__ == "__main__":
    analyze_all_properties_importance()
