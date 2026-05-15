"""Analyze output-length prediction risk for scheduling.

Unlike MAE-only evaluation, this script treats conservative prediction as a
scheduling policy. It reports coverage, underestimate severity, and
overestimate overhead for the primary prediction and any percentile ablations
stored by scripts/evaluate_output_prediction.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Evaluation JSONL from evaluate_output_prediction.py")
    parser.add_argument(
        "--group-by",
        default="predicted_category,dataset",
        help="Comma-separated fields for grouped metrics. Use empty string for overall only.",
    )
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if actual_length(record) is None:
                raise ValueError(f"{path}:{line_number} is missing actual output length")
            records.append(record)
    return records


def actual_length(record: dict[str, Any]) -> int | None:
    for key in ("actual_output_length", "ground_truth_output_length"):
        if record.get(key) is not None:
            return int(record[key])
    return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def metric_summary(records: list[dict[str, Any]], prediction_key: str = "predicted_output_length") -> dict[str, Any]:
    pairs: list[tuple[int, int]] = []
    for record in records:
        predicted = record.get(prediction_key)
        actual = actual_length(record)
        if predicted is not None and actual is not None:
            pairs.append((int(predicted), int(actual)))

    if not pairs:
        return {"n": 0}

    under = [actual - predicted for predicted, actual in pairs if predicted < actual]
    over = [predicted - actual for predicted, actual in pairs if predicted >= actual]
    signed = [predicted - actual for predicted, actual in pairs]
    abs_errors = [abs(value) for value in signed]
    actuals = [actual for _, actual in pairs]
    predictions = [predicted for predicted, _ in pairs]

    return {
        "n": len(pairs),
        "actual_mean": mean(actuals),
        "predicted_mean": mean(predictions),
        "coverage": len(over) / len(pairs),
        "underestimate_rate": len(under) / len(pairs),
        "underestimate_mean": mean(under) if under else 0.0,
        "underestimate_p90": percentile(under, 0.90) if under else 0.0,
        "overestimate_mean": mean(over) if over else 0.0,
        "overestimate_p90": percentile(over, 0.90) if over else 0.0,
        "mae": mean(abs_errors),
        "median_abs_error": median(abs_errors),
        "signed_bias_pred_minus_actual": mean(signed),
    }


def records_with_ablation(records: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        prediction = record.get("prediction_ablation", {}).get(label, {})
        predicted = prediction.get("predicted_output_length")
        if predicted is None:
            continue
        copy = dict(record)
        copy["predicted_output_length"] = predicted
        output.append(copy)
    return output


def print_summary(label: str, records: list[dict[str, Any]]) -> None:
    summary = metric_summary(records)
    if summary["n"] == 0:
        print(f"{label}: n=0")
        return
    profile_percentiles = {
        record.get("profile_percentile")
        for record in records
        if record.get("profile_percentile") is not None
    }
    profile_text = (
        f"profile_percentile={next(iter(profile_percentiles))} "
        if len(profile_percentiles) == 1
        else ""
    )
    print(
        f"{label}: n={summary['n']} "
        f"{profile_text}"
        f"actual_mean={fmt(summary['actual_mean'])} "
        f"predicted_mean={fmt(summary['predicted_mean'])} "
        f"coverage={fmt(summary['coverage'] * 100)}% "
        f"under_rate={fmt(summary['underestimate_rate'] * 100)}% "
        f"under_mean={fmt(summary['underestimate_mean'])} "
        f"under_p90={fmt(summary['underestimate_p90'])} "
        f"over_mean={fmt(summary['overestimate_mean'])} "
        f"over_p90={fmt(summary['overestimate_p90'])} "
        f"mae={fmt(summary['mae'])} "
        f"bias_pred_minus_actual={fmt(summary['signed_bias_pred_minus_actual'])}"
    )


def grouped(records: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(field) or "unknown")].append(record)
    return dict(sorted(groups.items()))


def ablation_labels(records: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for record in records:
        labels.update(record.get("prediction_ablation", {}).keys())
    return sorted(labels)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    group_fields = [field.strip() for field in args.group_by.split(",") if field.strip()]

    print("primary_prediction")
    print_summary("overall", records)
    for field in group_fields:
        print(f"by_{field}")
        for value, group_records in grouped(records, field).items():
            print_summary(f"  {value}", group_records)

    labels = ablation_labels(records)
    if labels:
        print("percentile_ablation")
        for label in labels:
            ablation_records = records_with_ablation(records, label)
            print_summary(label, ablation_records)
            for field in group_fields:
                print(f"{label}_by_{field}")
                for value, group_records in grouped(ablation_records, field).items():
                    print_summary(f"  {value}", group_records)


if __name__ == "__main__":
    main()
