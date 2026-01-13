import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

def generate_thesis_tables():
    print("--- Step 1: Loading Data (Robust Mode) ---")
    
    # 1. Find the file
    filename = 'Universal_QSPR_Weighted_Indices_Final.csv'
    if not os.path.exists(filename):
        # Check for alternative name from your uploads
        alt_name = 'Universal_QSPR_Weighted_Indices_Final.csv.xlsx - Universal_QSPR_Weighted_Indices.csv'
        if os.path.exists(alt_name):
            filename = alt_name
        else:
            print("❌ Error: File not found.")
            return

    # 2. Super Smart Loader (Handles CSV vs Excel confusion)
    try:
        # Attempt 1: Standard CSV
        df = pd.read_csv(filename, encoding='utf-8')
    except:
        try:
            # Attempt 2: Latin-1 CSV
            df = pd.read_csv(filename, encoding='latin1')
        except:
            print("⚠️ CSV read failed. Reading as Excel file...")
            try:
                # Attempt 3: Excel (This is likely what will work)
                df = pd.read_excel(filename)
            except Exception as e:
                print(f"❌ Critical Error: Could not read file. {e}")
                return

    print(f"✅ Loaded {len(df)} compounds successfully.")

    # --- Step 2: Generate Tables for Top 3 Properties ---
    # The 3 Winners we agreed on
    target_properties = [
        'Molar Refractivity (MR)', 
        'Molar Volume (MV)', 
        'Boiling Point (BP)'
    ]
    
    # Identify Features (Indices)
    feature_cols = [c for c in df.columns if c.startswith(('SO', 'ESO', 'MESO'))]
    
    # Prepare X (Features)
    X = df[feature_cols]
    # Add Intercept (Required for the equation: Y = aX + b)
    X = sm.add_constant(X)
    
    print("\n" + "="*50)
    print("GENERATING THESIS TABLES")
    print("="*50)

    for prop in target_properties:
        print(f"\n🔹 Processing: {prop}")
        
        if prop not in df.columns:
            print(f"   ⚠️ Warning: Column '{prop}' not found in file.")
            continue
            
        # Drop missing values for this specific property
        # We need to align X and y perfectly
        valid_data = df.dropna(subset=[prop])
        y = valid_data[prop]
        X_aligned = X.loc[valid_data.index]
        
        # Run OLS (MLR)
        model = sm.OLS(y, X_aligned).fit()
        
        # 1. Print Summary to Screen
        print(model.summary())
        
        # 2. Save Clean CSV for Thesis
        # We extract just the table with Coef, P-value, etc.
        results_as_html = model.summary().tables[1].as_html()
        table_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
        
        # Clean up filenames (remove special chars)
        safe_name = prop.split('(')[0].strip().replace(' ', '_')
        output_name = f"Thesis_Table_{safe_name}.csv"
        
        table_df.to_csv(output_name)
        print(f"   ✅ Saved equation table to: {output_name}")

    print("\n" + "="*50)
    print("🎉 DONE! You can now open the CSV files in Excel.")

if __name__ == "__main__":
    generate_thesis_tables()
