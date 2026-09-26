# DHHO and S2MHHO Code

This code accompanies manuscript IPM-D-26-01690. It contains the two proposed feature-selection algorithms, S2MHHO and DHHO. Moreover, CKD is included only as a small example dataset for checking that the code runs.

## Package contents

```text
.
|-- Algorithm/
|   |-- DHHO.py
|   `-- S2MHHO.py
|-- Dataset/
|   |-- CKD.csv
|   `-- CKD-cost.csv
|-- README.md

```

## Environment

The package was verified with Python 3.8.19 on Windows. From the package root:

```bash
conda env create -f environment_win.yml
conda activate 24_Medical_Cost
```

The principal dependencies are NumPy 1.24.3, pandas 2.0.3, SciPy 1.10.1, scikit-learn 1.3.0, and XlsxWriter 3.1.1. Exact versions are listed in `environment_win.yml`.

## Run the example

Run from the package root:

```bash
python Algorithm/DHHO.py
python Algorithm/S2MHHO.py
```

Both scripts use CKD, a population size of 8, 100 iterations, and a one-versus-one RBF SVM (`C=1.0`). Results are written to `Results/DHHO/` and `Results/S2MHHO/`.

`CKD.csv` has no header. Its last column is the class label and the preceding 24 columns are features. `CKD-cost.csv` contains one cost per feature in the same order.

## Train-test split protocol

All experiments use a stratified 70:30 train-test split produced by `sklearn.model_selection.train_test_split`.

## Feature-cost vectors

Each vector follows the feature-column order of its corresponding original dataset.

- **kidney-disease (CKD)** (24 features): `[1,1,1,25,30,39,30,30,50,20,11.9,14,3.5,49,1.7,1.6,30,30,1,18.4,50,1,1,27.6]`

- **diabetes** (8 features): `[1,17.61,1,1,22.78,1,1,1]`

- **heart** (13 features): `[1,1,1,1,7.27,5.2,15.5,102.9,87.3,87.3,87.3,100.9,102.9]`

- **hepatitis** (19 features): `[1,1,1,1,1,1,1,1,1,1,1,1,1,7.27,7.27,7.27,7.27,8.3,1]`

- **liver-disorders** (6 features): `[7.27,7.27,7.27,7.27,9.86,1]`

- **hypothyroid** (21 features): `[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,22.78,11.41,14.51,11.41,1]`


## Objectives and outputs

DHHO minimizes the weighted fitness

```text
0.98 * diagnostic_error_rate
+ 0.01 * feature_subset_ratio
+ 0.01 * normalized_medical_cost
```

Its workbook reports fitness, diagnostic error rate (`f1`), feature subset ratio (`f2`), normalized medical cost (`f3`), raw cost, runtime, selected features, and convergence history.

S2MHHO minimizes three objectives: diagnostic error rate, feature subset ratio, and medical cost. Its workbook reports the objectives of the non-dominated solutions recorded at generations 30, 60, and 100.

