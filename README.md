# Linearity in Topology: Benchmarking Weighted Sombor Indices

**Author:** Muhammad Shehroze Khan, Dr. Zunaira Kosar

---

### 📌 Project Overview
This repository provides a comprehensive computational framework for Quantitative Structure-Property Relationship (QSPR) modeling using Weighted Sombor Indices. We investigate the predictive power of 28 geometric topological invariants—weighted by atomic mass, radius, electronegativity, and ionization energy—across a dataset of 200 organic compounds.

---

### 📂 Repository Structure

#### 1. /Data - The Foundation
* **Universal_QSPR_Final_200_Purified.csv**: Contains canonical SMILES, compound names, and experimental property values.
* **Universal_QSPR_Weighted_Indices_Final**: **[CRITICAL]** This is the master input file for all analysis scripts. It contains the pre-calculated 28 weighted indices for all compounds.

#### 2. /Scripts - The Computational Pipeline
> **Note:** All scripts (except index calculation) require the master data file from the `/Data` folder to execute.

* **windicesfinal.py**: The primary engine used to calculate the 28 Weighted Sombor Indices from SMILES strings.
* **Correlations.py**: Computes Pearson correlation matrices and generates heatmaps for visual analysis.
* **importance tables.py**: Calculates Gini Importance scores using tree-based ensembles (Random Forest/XGBoost).
* **predictedprop.py**: Generates side-by-side comparisons of Experimental vs. Predicted values.
* **allproperror.py**: The benchmarking core. It calculates $R^2$, MAE, and RMSE to determine the top-performing models.
* **statsdata3.py**: Provides deep statistical profiles for the three most successfully predicted properties.

#### 3. /Results - Statistical Evidence
Organized into 5 specialized subfolders:
* **/correlations**: Supplementary correlation matrices.
* **/Feature importance Tables**: Detailed property-wise importance rankings.
* **/Predicted Values of Top Indices**: Final prediction data ($y$ vs $\hat{y}$).
* **/Properties Error Measurements**: The "Grand Results" summary (Sorted performance metrics).
* **/Statistical Data of Top 3 Predicted Properties**: Focused analysis of the study's top performers.

---

### 🚀 Reproduction Guide (For Reviewers)
To reproduce the findings of this study, follow these steps:

#### **Step 1: Environment Setup**
Ensure you have Python 3.9+ installed with the following libraries:
```bash
pip install rdkit pandas numpy scikit-learn xgboost matplotlib seaborn


Step 2: Execution Order

Data Generation: Run Scripts/windicesfinal.py to understand how the descriptors were derived.

Analysis: Run any of the remaining scripts (e.g., Scripts/allproperror.py).

Important: These scripts are hard-coded to look for the input file: Data/Universal_QSPR_Weighted_Indices_Final. Ensure this file path is maintained.


📊 Summary of Findings

The study reveals that size-dependent properties (Molar Refractivity) exhibit near-perfect linearity with weighted topology, while electronic properties (Polarizability) benefit significantly from the non-linear processing of Artificial Neural Networks.

📜 Acknowledgments

This research utilizes the RDKit chemoinformatics library and builds upon the geometric approach to degree-based topological indices established in recent mathematical chemistry literature.
