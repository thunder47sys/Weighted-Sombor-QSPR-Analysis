import pandas as pd
import numpy as np
import math
import os

!pip install rdkit
from rdkit import Chem
from rdkit.Chem import PeriodicTable

def calculate_final_indices():
    # --- Step 1: Smart Load the File ---
    # We look for your specific file name
    possible_names = [
        'Universal_QSPR_Final_200_Purified.csv.xlsx - Universal_QSPR_Final_200_Purifi.csv',
        'Universal_QSPR_Final_200_Purified.csv',
        'Universal_QSPR_Final_With_SMILES.csv'
    ]

    input_file = None
    for name in possible_names:
        if os.path.exists(name):
            input_file = name
            break

    if input_file is None:
        print("❌ Error: Could not find the CSV file.")
        print("Please rename your file to: Universal_QSPR_Final_200_Purified.csv")
        return

    print(f"✅ Loading data from: {input_file}")

    # Handle potential encoding/format issues
    try:
        df = pd.read_csv(input_file)
    except:
        try:
            df = pd.read_csv(input_file, encoding='latin1')
        except:
            df = pd.read_excel(input_file)

    print(f"   Processing {len(df)} compounds...")
    pt = Chem.GetPeriodicTable()

    # --- Step 2: Define Atomic Properties ---
    # Includes Br and I for your marine compounds
    atom_props_dict = {
        1:  {'en': 2.20, 'ie': 13.60}, # H
        6:  {'en': 2.55, 'ie': 11.26}, # C (Reference)
        7:  {'en': 3.04, 'ie': 14.53}, # N
        8:  {'en': 3.44, 'ie': 13.61}, # O
        9:  {'en': 3.98, 'ie': 17.42}, # F
        15: {'en': 2.19, 'ie': 10.49}, # P
        16: {'en': 2.58, 'ie': 10.36}, # S
        17: {'en': 3.16, 'ie': 12.97}, # Cl
        35: {'en': 2.96, 'ie': 11.81}, # Br
        53: {'en': 2.66, 'ie': 10.45}, # I
    }

    def get_prop_val(atom, prop_name):
        idx = atom.GetAtomicNum()
        if prop_name == 'mass': return pt.GetAtomicWeight(idx)
        if prop_name == 'radius': return pt.GetRcovalent(idx)
        # Default to Carbon values if missing
        return atom_props_dict.get(idx, atom_props_dict[6]).get(prop_name, 1.0)

    # --- Step 3: Calculation Logic ---
    def calc_row(smiles, prop_name):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if not mol: return [None]*7

            # A. Weighted Degree Calculation (Carbon Reference Method)
            # Reference w_c setup
            w_c = 12.011 if prop_name == 'mass' else 0.77
            if prop_name in ['en', 'ie']:
                w_c = atom_props_dict[6][prop_name]

            degrees = {}
            for atom in mol.GetAtoms():
                d_w = 0
                idx = atom.GetIdx()
                w_i = get_prop_val(atom, prop_name)

                for neighbor in atom.GetNeighbors():
                    bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                    bo = bond.GetBondTypeAsDouble()
                    w_j = get_prop_val(neighbor, prop_name)

                    # The Formula: dw = sum( wc^2 / (BO * wi * wj) )
                    if bo != 0 and w_i != 0 and w_j != 0:
                        d_w += (w_c * w_c) / (bo * w_i * w_j)

                degrees[idx] = d_w

            # B. Geometric Sombor Indices
            SO = 0; SO3 = 0; SO4 = 0; SO5 = 0; SO6 = 0; ESO = 0; MESO = 0

            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                du = degrees[u]
                dv = degrees[v]

                # Common Terms
                term = math.sqrt(du**2 + dv**2)
                sum_sq = du**2 + dv**2
                sum_deg = du + dv
                abs_diff_sq = abs(du**2 - dv**2)
                denom_so5 = (math.sqrt(2) + 2 * term)

                # 1. SO
                SO += term

                # 2. SO3
                if sum_deg != 0:
                    SO3 += (math.sqrt(2) * math.pi * (sum_sq / sum_deg))

                # 3. SO4
                if sum_deg != 0:
                    SO4 += ((math.pi / 2) * ((sum_sq / sum_deg) ** 2))

                # 4. SO5
                if denom_so5 != 0:
                    SO5 += (2 * math.pi * (abs_diff_sq / denom_so5))

                # 5. SO6
                if denom_so5 != 0:
                    SO6 += (math.pi * ((abs_diff_sq / denom_so5) ** 2))

                # 6. ESO
                ESO += (sum_deg * term)

            # 7. MESO
            if ESO != 0:
                MESO = 1 / ESO

            return [SO, SO3, SO4, SO5, SO6, ESO, MESO]

        except Exception as e:
            return [None]*7

    # --- Step 4: Run Loops ---
    print("--- Calculating Indices ---")

    properties = ['mass', 'radius', 'en', 'ie']
    variations = ['SO', 'SO3', 'SO4', 'SO5', 'SO6', 'ESO', 'MESO']

    for p in properties:
        print(f"   Processing property: {p}...")
        # Run calculation
        results = df['SMILES'].apply(lambda x: calc_row(x, p))

        # Create columns
        cols = [f"{v}_{p}" for v in variations]
        result_df = pd.DataFrame(results.tolist(), columns=cols, index=df.index)

        # Round to 4 decimal places
        df[cols] = result_df.round(4)

    # --- Step 5: Save ---
    output_file = 'Universal_QSPR_Weighted_Indices_Final.csv'
    df.to_csv(output_file, index=False)
    print("-" * 30)
    print(f"🎉 Success! Final file saved as: {output_file}")

if __name__ == "__main__":
    calculate_final_indices()
