import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def run_grand_loop_sorted():
    print("--- Step 1: Loading Data ---")
    
    # Specified Input File
    filename = 'Universal_QSPR_Weighted_Indices_Final.xlsx'
            
    if not os.path.exists(filename):
        print(f"Error: Could not find '{filename}'. Please ensure the file is in the directory.")
        return

    try:
        df = pd.read_excel(filename)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Clean data: Fill NaNs with mean
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

    # Define Targets in the order you want them to appear
    targets = ['Boiling Point (BP)', 'Molar Volume (MV)', 'Flash Point (FP)', 
               'Polarizability (Pol)', 'Molar Refractivity (MR)', 'Density (D)']
    
    # Identify indices
    feature_cols = [c for c in df.columns if any(x in c for x in ['SO', 'ESO', 'MESO'])]
    
    # Define Models
    models = {
        "MLR": LinearRegression(),
        "Lasso": Lasso(alpha=0.1),
        "SVR": SVR(kernel='rbf'),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "ANN": MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
    }

    master_results = []

    print(f"--- Step 2: Processing {len(targets)} Properties ---")

    for target in targets:
        if target not in df.columns:
            print(f"Warning: Target '{target}' not found. Skipping.")
            continue
            
        print(f"   Processing: {target}...")
        
        y = df[target].values
        X = df[feature_cols].values
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for name, model in models.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            fold_r2 = []
            fold_mae = []
            fold_rmse = []

            for train_idx, test_idx in kf.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    fold_r2.append(r2_score(y_test, pred))
                    fold_mae.append(mean_absolute_error(y_test, pred))
                    fold_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
                except:
                    fold_r2.append(0)
                    fold_mae.append(np.nan)
                    fold_rmse.append(np.nan)

            avg_r2 = np.mean(fold_r2)
            avg_mae = np.mean(fold_mae)
            avg_rmse = np.mean(fold_rmse)
            
            master_results.append({
                "Property": target,
                "Model": name,
                "R2_Score": round(avg_r2, 4),
                "MAE": round(avg_mae, 4),
                "RMSE": round(avg_rmse, 4)
            })

    # --- Step 3: Sorting Results ---
    results_df = pd.DataFrame(master_results)
    
    # Ensure properties are in the correct custom order (BP, MV, FP...)
    results_df['Property'] = pd.Categorical(results_df['Property'], categories=targets, ordered=True)
    
    # Sort by Property (Custom Order) and then by R2_Score (Descending)
    results_df = results_df.sort_values(by=['Property', 'R2_Score'], ascending=[True, False])

    # Save Results
    output_filename = "All_Properties_Grand_Results_Sorted.csv"
    results_df.to_csv(output_filename, index=False)
    
    print(f"\nCalculation Complete. Results sorted by top performance.")
    print(f"File saved to: '{output_filename}'")
    print("-" * 30)
    print(results_df.head(10)) # Preview

if __name__ == "__main__":
    run_grand_loop_sorted()
