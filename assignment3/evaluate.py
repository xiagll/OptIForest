"""Evaluate OptIForest and IsolationForest with ROC-AUC and average precision.

From repository root:
    python assignment3/evaluate.py --dataset synthetic_vehicle_claims_a3_sample
    python assignment3/evaluate.py --dataset vehicle_fraud --repeats 3

Full datasets may run slowly.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from detectors import OptIForest

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="synthetic_vehicle_claims_a3_sample")
parser.add_argument("--trees", type=int, default=100)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--threshold", type=int, default=403)
parser.add_argument("--branch", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
if args.trees < 1 or args.repeats < 1:
    parser.error("--trees and --repeats must be positive")
path = Path("data") / f"{args.dataset}.csv"
if not path.exists():
    parser.error(f"Missing {path}. Generate the synthetic data first.")
data = pd.read_csv(path, header=None).to_numpy(dtype=float)
X, y = data[:, :-1], data[:, -1].astype(int)
if set(np.unique(y)) != {0, 1}:
    parser.error("Expected both binary labels 0 and 1 in final column.")

results = []
for run in range(args.repeats):
    seed = args.seed + run
    for name in ("OptIForest", "IsolationForest"):
        random.seed(seed)
        np.random.seed(seed)
        start = time.perf_counter()
        if name == "OptIForest":
            model = OptIForest("L2OPT", args.trees, args.threshold, args.branch)
            model.fit(X)
            train_time = time.perf_counter() - start
            scores = -model.decision_function(X)
        else:
            model = IsolationForest(n_estimators=args.trees, random_state=seed)
            model.fit(X)
            train_time = time.perf_counter() - start
            scores = -model.score_samples(X)
        test_time = time.perf_counter() - start - train_time
        result = dict(dataset=args.dataset, method=name, run=run + 1, seed=seed,
                      n_rows=len(y), n_anomalies=int(y.sum()), trees=args.trees,
                      roc_auc=float(roc_auc_score(y, scores)),
                      average_precision=float(average_precision_score(y, scores)),
                      train_seconds=round(train_time, 3),
                      test_seconds=round(test_time, 3))
        results.append(result)
        print(f"{name} run {run+1}: ROC-AUC={result['roc_auc']:.4f} "
              f"AP={result['average_precision']:.4f}", flush=True)

out = Path("assignment3/results")
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.dataset}_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
with (out / f"{args.dataset}_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
print(f"Saved metrics to {out}")
