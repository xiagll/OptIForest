"""Generate 10,000 fictional insurance claims for COMP8240 Assignment 3.

Run from repository root: python assignment3/generate_synthetic_claims.py
Labels mark injected statistical anomalies, NOT verified insurance fraud.
"""
from __future__ import annotations
import csv
from pathlib import Path

TOTAL = 10_000
ANOMALIES = 600
SEED = 0xA5A5A5A5
AREAS = ("Urban", "Suburban", "Rural")
POLICIES = ("Comprehensive", "ThirdParty", "Collision")
SEVERITIES = ("Minor", "Moderate", "Major")
TYPES = ("inflated_claim", "repeat_claim", "no_evidence", "young_luxury")
NUMERIC_FIELDS = (
    "driver_age", "vehicle_age", "vehicle_value_aud", "claim_amount_aud",
    "deductible_aud", "prior_claims", "claim_delay_days", "incident_hour",
    "police_report", "witness_present", "driver_rating",
)
RAW_FIELDS = ("record_id",) + NUMERIC_FIELDS + (
    "accident_area", "policy_type", "accident_severity", "label", "anomaly_type",
)
SCALES = (100, 25, 150000, 180000, 2000, 10, 90, 23, 1, 1, 5)


def make_row(index: int) -> dict:
    state = (SEED ^ (((index + 1) * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 4294967296

    age = 18 + int(rnd() * 58)
    vehicle_age = int(rnd() * 21)
    value = max(4000, int(68000 - 2600 * vehicle_age + (rnd() - 0.5) * 18000))
    amount = max(300, int(value * (0.025 + 0.28 * rnd()) + 800 * rnd()))
    deductible = (500, 750, 1000, 1500)[int(rnd() * 4)]
    prior_draw = rnd()
    prior = 0 if prior_draw < 0.65 else 1 if prior_draw < 0.88 else 2 if prior_draw < 0.97 else 3
    delay = int(rnd() * 15)
    hour = int(rnd() * 24)
    police = int(rnd() < 0.72)
    witness = int(rnd() < 0.65)
    rating = 1 + int(rnd() * 5)
    area = AREAS[int(rnd() * 3)]
    policy = POLICIES[int(rnd() * 3)]
    severity = SEVERITIES[int(rnd() * 3)]

    label = int(index < ANOMALIES)
    anomaly_type = "normal"
    if label:
        anomaly_type = TYPES[index % len(TYPES)]
        if anomaly_type == "inflated_claim":
            amount = int(value * (1.15 + 0.70 * rnd()))
        elif anomaly_type == "repeat_claim":
            prior = 6 + int(rnd() * 5)
            delay = 35 + int(rnd() * 45)
            amount = int(value * (0.35 + 0.50 * rnd()))
            police = 0
        elif anomaly_type == "no_evidence":
            amount = int(value * (0.55 + 0.55 * rnd()))
            police = 0
            witness = 0
            hour = int(rnd() * 5)
            delay = 20 + int(rnd() * 45)
        else:
            age = 18 + int(rnd() * 4)
            vehicle_age = int(rnd() * 3)
            value = 90000 + int(rnd() * 60000)
            amount = int(value * (0.35 + 0.45 * rnd()))

    return dict(zip(RAW_FIELDS, (
        f"SYN{index + 1:05d}", age, vehicle_age, value, amount, deductible,
        prior, delay, hour, police, witness, rating, area, policy, severity,
        label, anomaly_type,
    )))


def numeric_row(row: dict) -> list[str]:
    """Fixed-range scaling + one-hot encoding; excludes ID and anomaly type."""
    features = [f"{int(row[name]) / scale:.6f}" for name, scale in zip(NUMERIC_FIELDS, SCALES)]
    for field, values in (
        ("accident_area", AREAS),
        ("policy_type", POLICIES),
        ("accident_severity", SEVERITIES),
    ):
        features.extend("1" if row[field] == value else "0" for value in values)
    return features + [str(row["label"])]


def generate(output_dir: Path = Path("data")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "synthetic_vehicle_claims_a3_raw.csv"
    model_path = output_dir / "synthetic_vehicle_claims_a3.csv"
    sample_path = output_dir / "synthetic_vehicle_claims_a3_sample.csv"
    with (raw_path.open("w", newline="") as raw_file,
          model_path.open("w", newline="") as model_file,
          sample_path.open("w", newline="") as sample_file):
        raw_writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS, lineterminator="\n")
        model_writer = csv.writer(model_file, lineterminator="\n")
        sample_writer = csv.writer(sample_file, lineterminator="\n")
        raw_writer.writeheader()
        for output_index in range(TOTAL):
            source_index = (7919 * output_index + 1237) % TOTAL
            row = make_row(source_index)
            raw_writer.writerow(row)
            features = numeric_row(row)
            model_writer.writerow(features)
            if output_index < 3000:
                sample_writer.writerow(features)
    print(f"Generated {TOTAL:,} fictional claims, including {ANOMALIES:,} anomalies (6%).")
    print(f"Raw: {raw_path} | Model: {model_path} | Sample: {sample_path}")


if __name__ == "__main__":
    generate()
