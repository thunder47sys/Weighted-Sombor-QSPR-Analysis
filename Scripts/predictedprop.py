import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Import Models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

def generate_supplementary_data():
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
        df = pd.read_csv(filename, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filename, encoding='latin1')
        except:
            print("⚠️ Reading as Excel...")
            df = pd.read_excel(filename)

    # Identify Features and Targets
    feature_cols = [c for c in df.columns if c.startswith(('SO', 'ESO', 'MESO'))]
    exclude_cols = ['ID', 'Name', 'SMILES'] + feature_cols
    target_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    print(f"✅ Loaded {len(df)} compounds.")
    print(f"Features: {len(feature_cols)} Indices")
    print(f"Properties to Process: {target_cols}")

    # --- Step 2: Define Models ---
    models = {
        "MLR": LinearRegression(),
        "Lasso": Lasso(alpha=0.01),
        "SVR": SVR(kernel='rbf', C=100),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, random_state=42),
        "ANN": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=800, random_state=42)
    }

    # Initialize Master DataFrame for Output
    # We start with just IDs and Names
    output_df = df[['ID', 'Name']].copy()

    # --- Step 3: The Prediction Loop ---
    print("\n" + "="*60)
    print("GENERATING SUPPLEMENTARY DATA (Actual vs Predicted)")
    print("="*60)

    for target in target_cols:
        print(f"\n🔹 Processing: {target}")
        
        # Filter valid data
        valid_idx = df.dropna(subset=[target]).index
        if len(valid_idx) < 50:
            print(f"   ⚠️ Skipping (Not enough data)")
            continue
            
        X = df.loc[valid_idx, feature_cols].values
        y = df.loc[valid_idx, target].values
        
        # 3a. Find the Best Model for this Property
        best_r2 = -float('inf')
        best_model_name = ""
        best_predictions = np.zeros(len(y))
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            # Storage for this model's predictions
            current_preds = np.zeros(len(y))
            fold_r2s = []
            
            # 5-Fold Loop (The Reference Method)
            for train_i, test_i in kf.split(X):
                X_train, X_test = X[train_i], X[test_i]
                y_train, y_test = y[train_i], y[test_i]
                
                # Scale
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                try:
                    model.fit(X_train_s, y_train)
                    p = model.predict(X_test_s)
                    current_preds[test_i] = p
                    fold_r2s.append(r2_score(y_test, p))
                except:
                    fold_r2s.append(-1)

            avg_r2 = np.mean(fold_r2s)
            
            # Check if this is the winner
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_model_name = name
                best_predictions = current_preds
        
        print(f"   🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")
        
        # 3b. Add Columns to Output DataFrame
        # We align data back to the original rows using valid_idx
        
        # Create Series to map back to original DF
        exp_series = pd.Series(y, index=valid_idx)
        pred_series = pd.Series(best_predictions, index=valid_idx)
        
        # Clean column name
        clean_name = target.split('(')[0].strip()
        
        output_df[f'{clean_name}_Exp'] = exp_series
        output_df[f'{clean_name}_Pred ({best_model_name})'] = pred_series
        output_df[f'{clean_name}_Residual'] = exp_series - pred_series

    # --- Step 4: Save ---
    outfile = "Supplementary_Data_Predictions.csv"
    output_df.to_csv(outfile, index=False)
    
    print("\n" + "="*60)
    print(f"🎉 DONE! Saved file: {outfile}")
    print("This file contains the Experimental and Predicted values for all valid properties.")

if __name__ == "__main__":
    generate_supplementary_data()
