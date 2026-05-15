"""Compute quick metrics from serving pipeline JSONL logs.

This script is intentionally small: it validates that logs are readable and
prints basic latency, TTFT, queue wait, output length, and per-dataset summaries.
It can grow into the full evaluation script later.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    """Parse the path to the JSONL log file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/run.jsonl")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict]:
    """Load one JSON object per completed request from a log file."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def numeric(records: list[dict], key: str) -> list[float]:
    """Extract numeric values for one metric, skipping missing entries."""
    return [float(record[key]) for record in records if record.get(key) is not None]


def print_stats(label: str, values: list[float]) -> None:
    """Print count, mean, p50, p95, min, and max for one metric."""
    if not values:
        print(f"{label}: n=0")
        return
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    print(
        f"{label}: n={len(values)} mean={mean(values):.4f} "
        f"p50={median(values):.4f} p95={ordered[p95_index]:.4f} "
        f"min={min(values):.4f} max={max(values):.4f}"
    )


def p90(values: list[float]) -> float | None:
    """Return the nearest-rank p90 value for a numeric list."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))
    return ordered[index]


def nested_numeric(records: list[dict], parent: str, key: str) -> list[float]:
    """Extract numeric values from a nested dict field."""
    values: list[float] = []
    for record in records:
        container = record.get(parent)
        if isinstance(container, dict) and container.get(key) is not None:
            values.append(float(container[key]))
    return values


def prediction_errors(
    records: list[dict],
    actual_key: str,
    predicted_parent: str,
    predicted_key: str,
) -> tuple[list[float], list[float]]:
    """Return absolute errors and percentage errors for prediction quality."""
    absolute_errors: list[float] = []
    percentage_errors: list[float] = []
    for record in records:
        predicted = record.get(predicted_parent, {}).get(predicted_key)
        actual = record.get(actual_key)
        if predicted is None or actual is None:
            continue
        predicted_value = float(predicted)
        actual_value = float(actual)
        error = abs(predicted_value - actual_value)
        absolute_errors.append(error)
        if actual_value != 0:
            percentage_errors.append(error / abs(actual_value))
    return absolute_errors, percentage_errors


def print_prediction_metrics(label: str, absolute_errors: list[float], percentage_errors: list[float]) -> None:
    """Print MAE, P90 absolute error, and MAPE when available."""
    if not absolute_errors:
        print(f"{label}: n=0")
        return
    p90_abs = p90(absolute_errors)
    mape = mean(percentage_errors) * 100 if percentage_errors else None
    mape_text = f"{mape:.2f}%" if mape is not None else "n/a"
    print(
        f"{label}: n={len(absolute_errors)} "
        f"mae={mean(absolute_errors):.4f} p90_abs={p90_abs:.4f} mape={mape_text}"
    )


def throughput(records: list[dict]) -> float:
    """Compute completed requests per second over the observed run window."""
    completions = numeric(records, "completion_time")
    arrivals = numeric(records, "arrival_time")
    if not completions or not arrivals:
        return 0.0
    duration = max(completions) - min(arrivals)
    if duration <= 0:
        return float(len(records))
    return len(records) / duration


def main() -> None:
    """Load logs and print a compact summary."""
    args = parse_args()
    records = load_records(args.log)
    print(f"requests: {len(records)}")
    print(f"throughput_requests_per_second: {throughput(records):.4f}")
    print_stats("latency_seconds", numeric(records, "latency_seconds"))
    print_stats("ttft_seconds", numeric(records, "ttft_seconds"))
    print_stats("queue_wait_seconds", numeric(records, "queue_wait_seconds"))
    print_stats("output_token_count", numeric(records, "output_token_count"))

    prefill_abs, prefill_pct = prediction_errors(
        records,
        "actual_prefill_cost",
        "features",
        "predicted_prefill_cost",
    )
    output_abs, output_pct = prediction_errors(
        records,
        "output_token_count",
        "features",
        "predicted_output_length",
    )
    print_prediction_metrics("prefill_estimation_error", prefill_abs, prefill_pct)
    print_prediction_metrics("output_length_estimation_error", output_abs, output_pct)

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_dataset[record.get("dataset") or "unknown"].append(record)

    print("per_dataset_latency:")
    for dataset, dataset_records in sorted(by_dataset.items()):
        latencies = numeric(dataset_records, "latency_seconds")
        value = mean(latencies) if latencies else 0.0
        print(f"  {dataset}: n={len(dataset_records)} mean_latency={value:.4f}")


if __name__ == "__main__":
    main()
