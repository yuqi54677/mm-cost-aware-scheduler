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
from statistics import mean

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
    """Print count, mean, min, and max for one metric."""
    if not values:
        print(f"{label}: n=0")
        return
    print(f"{label}: n={len(values)} mean={mean(values):.4f} min={min(values):.4f} max={max(values):.4f}")


def main() -> None:
    """Load logs and print a compact summary."""
    args = parse_args()
    records = load_records(args.log)
    print(f"requests: {len(records)}")
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
