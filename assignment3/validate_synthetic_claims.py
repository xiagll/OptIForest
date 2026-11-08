"""Validate generated A3 synthetic data. Run from repository root."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path("data")
with (ROOT / "synthetic_vehicle_claims_a3_raw.csv").open(newline="") as f:
    raw = list(csv.DictReader(f))
with (ROOT / "synthetic_vehicle_claims_a3.csv").open(newline="") as f:
    processed = list(csv.reader(f))
with (ROOT / "synthetic_vehicle_claims_a3_sample.csv").open(newline="") as f:
    sample = list(csv.reader(f))
assert len(raw) == len(processed) == 10000
assert len(sample) == 3000
assert len({r["record_id"] for r in raw}) == 10000
assert all(len(row) == 21 for row in processed + sample)
assert all(all(cell != "" and float(cell) == float(cell) for cell in row) for row in processed)
assert all(sum(map(int, row[11:14])) == 1 and
           sum(map(int, row[14:17])) == 1 and
           sum(map(int, row[17:20])) == 1 for row in processed)
assert all(row["label"] == processed[i][-1] for i, row in enumerate(raw))
assert sample == processed[:3000]
assert Counter(r["anomaly_type"] for r in raw) == {
    "normal": 9400, "inflated_claim": 150, "repeat_claim": 150,
    "no_evidence": 150, "young_luxury": 150,
}
print("PASS: 10,000 rows, 600 anomalies, 20 features + label, valid one-hot columns.")
print("Sample anomalies:", sum(int(row[-1]) for row in sample))
