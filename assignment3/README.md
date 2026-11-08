# Assignment 3 — Synthetic insurance claims

This branch contains reproducible tools to extend the original OptIForest replication. The synthetic dataset contains **injected statistical anomalies, not confirmed insurance fraud**.

## Generate and validate

Run from the repository root in your GitHub Codespace:

```bash
python assignment3/generate_synthetic_claims.py
python assignment3/validate_synthetic_claims.py
```

The generator writes:
- `data/synthetic_vehicle_claims_a3_raw.csv` — human-readable 10,000 claims, labels and anomaly-type annotations.
- `data/synthetic_vehicle_claims_a3.csv` — numeric, headerless data with **20 features and final label** for OptIForest.
- `data/synthetic_vehicle_claims_a3_sample.csv` — deterministic 3,000-row sample.

The dataset includes 9,400 normal cases and 600 injected anomalies (150 per type): inflated claim, repeated claims, no evidence, and young driver/luxury vehicle. Identifiers and anomaly types are excluded from model features to prevent leakage. Numerical variables use fixed scaling, and categorical variables use one-hot encoding. All records are fictional.

## Experiments

```bash
python demo.py --dataset=synthetic_vehicle_claims_a3_sample --threshold=403 --branch=0
python assignment3/evaluate.py --dataset synthetic_vehicle_claims_a3_sample --repeats 1
python assignment3/evaluate.py --dataset synthetic_vehicle_claims_a3 --repeats 3
python assignment3/evaluate.py --dataset vehicle_fraud --repeats 3
```

The comparison script reports ROC-AUC and **average precision (AP)** for OptIForest and standard IsolationForest, saving JSON/CSV to `assignment3/results/`. The full-data runs may take substantial time. Run them before reporting any scores; no evaluation result is implied by the scripts alone.

## Limitations

Anomalies are injected by rules and may be easy to detect; these are not verified fraud cases. The generator is not fitted to real claim distributions, and some anomalous-looking claims may be legitimate. Report this limitation, inspect errors, and compare synthetic and real vehicle-fraud performance separately.
