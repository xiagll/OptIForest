# OptIForest — COMP8240 Assignment 3 Replication and Extension

**Reproducing and Extending OptIForest: A Study of Benchmark Anomalies and Vehicle Insurance Claims**

This branch contains the computational work for the COMP8240 (Applications of Data Science) project at Macquarie University. It reproduces one result from [Xiang et al. (IJCAI 2023)](https://doi.org/10.24963/ijcai.2023/264), then evaluates the released OptIForest implementation alongside scikit-learn's Isolation Forest on two insurance-claims datasets.

**This is the submission branch:** `assignment3-synthetic-dataset`. The default `main` branch retains the original project code. The [original authors' repository](https://github.com/xiagll/OptIForest) is credited here; this is a fork with documented compatibility, preprocessing, synthetic-data and evaluation additions.

## Project questions

1. Can the released OptIForest implementation approximately reproduce the published result on the **AD** benchmark?
2. How do OptIForest and standard Isolation Forest rank **rule-injected synthetic anomalies** and **fraud-labelled real vehicle claims**?

A high anomaly score is **not proof of insurance fraud**. The synthetic and real positive labels represent different concepts.

## Datasets and completed experiments

| Dataset | Records | Positive cases | Model predictors | Purpose |
| --- | ---: | ---: | ---: | --- |
| Original AD (`data/ad.csv`) | 3,279 | 459 (14.00%) | 1,555 | Original-paper replication |
| Synthetic claims (`data/synthetic_vehicle_claims_a3.csv`) | 10,000 | 600 (6.00%) | 20 | Controlled, injected anomalies |
| Real vehicle claims (`data/vehicle_fraud.csv`) | 15,420 | 923 (5.99%) | 162 | Recorded fraud labels |
| Synthetic claims sample | 3,000 | 182 (6.07%) | 20 | Smaller-sample check |
| Real claims sample | 3,000 | 193 (6.43%) | 162 | Smaller-sample check |

All five evaluations were run with **100 trees and seeds 42, 43, 44**, using the saved full datasets and sample datasets. Metrics below are **mean ± sample standard deviation (%) over three runs**:

| Dataset | OptIForest ROC-AUC | Isolation Forest ROC-AUC | OptIForest AP | Isolation Forest AP |
| --- | ---: | ---: | ---: | ---: |
| AD | 77.45 ± 0.67 | 69.69 ± 1.28 | 42.34 ± 1.30 | 44.90 ± 3.91 |
| Synthetic claims | 83.25 ± 0.58 | 96.41 ± 0.50 | 42.37 ± 3.16 | 83.94 ± 2.61 |
| Real vehicle claims | 47.28 ± 1.77 | 48.81 ± 2.29 | 5.92 ± 0.23 | 5.94 ± 0.41 |

The published paper reports approximately **77.4 ± 0.7% ROC-AUC** and **43.8 ± 3.1% AUC-PR** on AD (15 repetitions). The original released demo uses `average_precision_score`. The present replication used three seeds and sample SD, so it is an **approximate, narrower replication**, not a full reproduction of all original experiments.

**Important evaluation qualification:** Both unsupervised models fit on `X` and score those *same rows*. These results describe **in-sample anomaly ranking**, not held-out fraud prediction. The original OptIForest implementation also adds row indices internally, unlike the scikit-learn baseline; this is a comparison of two implementations, not a controlled branching-factor ablation. Differences between the synthetic and real datasets cannot be attributed solely to fraud realism.

## Start here

For setup, dataset generation, preprocessing, complete reproduction commands, output locations and limitations, see **[assignment3/README.md](assignment3/README.md)**.

The evaluation output is preserved in **[assignment3/results/](assignment3/results/)** as per-run CSV and JSON, including original AD and the two smaller-sample checks.

### Key files

- `demo.py` — original demonstration entry point.
- `detectors/` — OptIForest implementation with documented NumPy and stability repairs.
- `preprocess_vehicle_fraud.py` — prepares the real vehicle-claims CSV.
- `assignment3/generate_synthetic_claims.py` — deterministic generation of fictional claims.
- `assignment3/validate_synthetic_claims.py` — checks labels, dimensions and one-hot data.
- `assignment3/evaluate.py` — paired unsupervised evaluations and run-level metrics.
- `requirements-a3.txt` — current dependency names; **not a lockfile of the tested environment**.

## Data provenance and responsible use

- Original benchmark: `data/ad.csv` from the upstream OptIForest distribution; see the [upstream README](https://github.com/xiagll/OptIForest).
- Existing real claims: [Vehicle Claim Fraud Detection on Kaggle (shivamb)](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection), loaded as `data/fraud_oracle.csv` in this project. Its licence and redistribution permissions have **not been independently verified**; review the source terms before redistributing the raw data.
- Synthetic data: entirely fictional deterministic records. Each of four anomaly types has 150 examples; labels denote **injected statistical irregularities**, never validated fraudulent acts.

## Original project attribution

OptIForest: Optimal Isolation Forest for Anomaly Detection — Haolong Xiang, Xuyun Zhang, Hongsheng Hu, Lianyong Qi, Wanchun Dou, Mark Dras, Amin Beheshti and Xiaolong Xu, IJCAI 2023, [DOI](https://doi.org/10.24963/ijcai.2023/264).
