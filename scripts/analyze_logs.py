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
