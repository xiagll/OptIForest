# COMP8240 Assignment 3 — Reproduction guide

This directory contains the synthetic generator, validation checks, evaluation runner, and complete recorded results for the project **Reproducing and Extending OptIForest: A Study of Benchmark Anomalies and Vehicle Insurance Claims**.

**Run every command from the repository root.** The assessment work lives on branch `assignment3-synthetic-dataset`. See the [project overview](../README.md) for research questions, published comparisons and results.

## 1. Set up Python

A Linux-based GitHub Codespace or equivalent Python 3 environment can be used. The project was executed with a Python virtual environment containing NumPy, pandas, SciPy and scikit-learn. The exact versions installed during the recorded runs were **not captured**, so this dependency list is **not an exact environment lock**.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-a3.txt
python -m pip freeze
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. To preserve precise versions for a new run, save `python -m pip freeze > environment-freeze.txt` after installation. A new environment may produce slightly different stochastic results and timing.

The upstream README contains older version pins for the upstream project; use `requirements-a3.txt` for these Assignment 3 scripts and check compatibility with your Python/NumPy version.

## 2. Files and dataset preparation

The model-ready CSVs are **headerless**, with numeric predictors and the last column reserved for binary ground truth. Labels are not used during unsupervised model fitting.

| Name in evaluation command | Input path | Rows | Positives | Predictors |
| --- | --- | ---: | ---: | ---: |
| `ad` | `data/ad.csv` | 3,279 | 459 | 1,555 |
| `synthetic_vehicle_claims_a3` | `data/synthetic_vehicle_claims_a3.csv` | 10,000 | 600 | 20 |
| `synthetic_vehicle_claims_a3_sample` | `data/synthetic_vehicle_claims_a3_sample.csv` | 3,000 | 182 | 20 |
| `vehicle_fraud` | `data/vehicle_fraud.csv` | 15,420 | 923 | 162 |
| `vehicle_fraud_sample` | `data/vehicle_fraud_sample.csv` | 3,000 | 193 | 162 |

### Original AD benchmark

The repository includes the upstream OptIForest AD benchmark as `data/ad.csv`. Its first 1,555 columns are numeric predictors, followed by the binary evaluation label. The baseline reproduction uses the original L2OPT implementation with its stated demonstration parameters.

### Existing vehicle insurance claims

The real source is the [Vehicle Claim Fraud Detection dataset (Kaggle uploader: shivamb)](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection), stored as `data/fraud_oracle.csv` in this fork. The raw table has 15,420 rows and 33 columns, including `FraudFound_P` (923 positives). The project performed no independent fraud verification.

To regenerate the numeric model input:

```bash
python preprocess_vehicle_fraud.py
```

The script drops the `PolicyNumber` identifier; replaces string `0` values in `DayOfWeekClaimed` and `MonthClaimed` using the mode; treats `Age=0` as unknown and imputes the median; treats `RepNumber` and `Year` as categories; one-hot encodes categorical fields; applies `StandardScaler` to original numeric columns; and writes `data/vehicle_fraud.csv` (162 predictors, then label).

**Interpretation:** Preprocessing is fitted to the full input because the reported experiment evaluates in-sample rankings. For a hold-out evaluation, preprocessing statistics must be fitted on the training partition only. The 3,000-row `vehicle_fraud_sample.csv` is already included; the preprocessing script does not regenerate that sample.

**Permissions:** The Kaggle dataset's licensing/redistribution permissions have not been verified for this project. Its appearance in a public repository does not establish redistribution rights; review source terms before publishing or distributing copies.

### Constructed synthetic insurance claims

The generator uses a deterministic pseudorandom state and index permutation to produce 10,000 fictional claims. **These 600 labels represent artificially injected statistical anomalies, not verified fraud**:

- `inflated_claim` (150): unusually high claim relative to vehicle value;
- `repeat_claim` (150): many earlier claims with late reporting;
- `no_evidence` (150): elevated amount with no police report or witnesses and long delay;
- `young_luxury` (150): young driver, high-value newer vehicle and substantial claim.

It writes a raw human-readable table with `anomaly_type`, a headerless model-ready matrix with 20 features and final label, and a deterministic first-3,000-record sample. IDs and anomaly type **are not model features**; numerical values use fixed scaling and the categorical fields use one-hot encoding.

```bash
python assignment3/generate_synthetic_claims.py
python assignment3/validate_synthetic_claims.py
```

Expected validation message begins with `PASS: 10,000 rows, 600 anomalies, 20 features + label`, and sample positives should be `182`. No human annotation or LLM-as-judge procedure was used.

## 3. Reproduce the original AD demonstration

```bash
python demo.py --dataset=ad --threshold=403 --branch=0
```

The authors' original demo runs 15 repetitions and uses scikit-learn's `average_precision_score` alongside ROC-AUC. The source paper's AD entry reports **77.4 ± 0.7% ROC-AUC** and **43.8 ± 3.1% AUC-PR**. The demo is separate from the three-run comparative script below.

## 4. Reproduce the paired OptIForest versus Isolation Forest experiments

From the repository root, execute each of the following:

```bash
python -m assignment3.evaluate --dataset ad --repeats 3
python -m assignment3.evaluate --dataset synthetic_vehicle_claims_a3 --repeats 3
python -m assignment3.evaluate --dataset vehicle_fraud --repeats 3
python -m assignment3.evaluate --dataset synthetic_vehicle_claims_a3_sample --repeats 3
python -m assignment3.evaluate --dataset vehicle_fraud_sample --repeats 3
```

Defaults are **100 trees, `threshold=403`, `branch=0`, seed 42**, with seed incremented for each run (42, 43, 44). The scikit-learn baseline also uses 100 estimators. Each pair of models is fitted without labels, then scored on the same data. Higher scores indicate greater anomaly suspicion; metrics use `roc_auc_score` and `average_precision_score`.

**Outputs:** For each dataset, the script writes `assignment3/results/<dataset>_comparison.csv` and `.json`. These files already contain the **recorded three-run results** used in the final report. Rerunning overwrites the corresponding result files, so retain the committed files if exact provenance matters. Timing reflects the original execution environment and should not be assumed to reproduce identically on other hardware.

The reported means ± **sample SD** (all values in %) are:

| Dataset | OptIForest ROC-AUC | Isolation Forest ROC-AUC | OptIForest AP | Isolation Forest AP |
| --- | ---: | ---: | ---: | ---: |
| AD | 77.45 ± 0.67 | 69.69 ± 1.28 | 42.34 ± 1.30 | 44.90 ± 3.91 |
| Full synthetic claims | 83.25 ± 0.58 | 96.41 ± 0.50 | 42.37 ± 3.16 | 83.94 ± 2.61 |
| Full real claims | 47.28 ± 1.77 | 48.81 ± 2.29 | 5.92 ± 0.23 | 5.94 ± 0.41 |
| Synthetic sample | 84.40 ± 0.52 | 96.96 ± 0.41 | 50.78 ± 1.48 | 85.18 ± 0.78 |
| Real sample | 48.05 ± 1.96 | 48.17 ± 3.32 | 7.04 ± 0.31 | 6.60 ± 0.49 |

Mean recorded fit+score times on the full sets (s): **AD** OptIForest 87.47, Isolation Forest 0.22; **synthetic** 120.24 and 0.31; **real** 249.01 and 0.27. Runtime differences are specific to these implementations and their execution environment, not a controlled measure of inherent algorithmic complexity.

## 5. Source-code fixes and limitations

- Replaced incompatible deprecated NumPy `np.mat` uses with `np.asmatrix` in OptIForest detector implementation for a newer NumPy environment.
- Added a guard against a zero standard deviation in the OptIForest tree calculation to avoid invalid divisions.
- The released OptIForest implementation internally appends a row index when fitting/scoring; the scikit-learn baseline does not, so the comparison is **not** an isolated branching-factor ablation.
- All scores are **in-sample unsupervised rankings**. They should not be interpreted as measured performance on unseen insurance claims.
- Three repetitions provide limited variability information, not statistical significance.
- Synthetic anomalies are simplified and rule-injected; real fraud may not be statistically isolated in the available features. Datasets differ in feature dimensionality and target semantics.
- No row-order perturbation, group fairness audit, held-out testing, or anomaly-type breakdown is claimed as completed work.
- Actual `pip freeze` output from the original execution was not preserved. `requirements-a3.txt` lists dependencies only, not a verified lockfile.

## 6. Original work and attribution

Xiang, H., Zhang, X., Hu, H., Qi, L., Dou, W., Dras, M., Beheshti, A., and Xu, X. (2023). *OptIForest: Optimal Isolation Forest for Anomaly Detection*. IJCAI-23. [Paper DOI](https://doi.org/10.24963/ijcai.2023/264) | [Original code](https://github.com/xiagll/OptIForest).

Baseline: Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). *Isolation Forest*, IEEE ICDM.
