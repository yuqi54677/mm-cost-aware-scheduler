"""Compare profile examples against an evaluation workload.

This helps catch dataset sampling drift: profile generation and evaluation may
use the same dataset names and seed while still selecting different examples if
they are run from different candidate pools.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-summary", required=True, help="JSON summary from build_output_length_profile.py")
    parser.add_argument("--workload", required=True, help="Evaluation workload JSONL")
    return parser.parse_args()


def load_profile_examples(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.get("examples", []))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(record: dict[str, Any]) -> str:
    return str(record.get("request_id") or record.get("id") or record.get("prompt"))


def by_dataset(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("dataset") or "unknown")].append(record)
    return dict(sorted(grouped.items()))


def main() -> None:
    args = parse_args()
    profile = load_profile_examples(args.profile_summary)
    workload = load_jsonl(args.workload)
    profile_keys = {key(record) for record in profile}
    workload_keys = {key(record) for record in workload}
    overlap = profile_keys & workload_keys

    print(f"profile_examples: {len(profile)}")
    print(f"workload_rows: {len(workload)}")
    print(f"overlap: {len(overlap)}")
    print(f"profile_only: {len(profile_keys - workload_keys)}")
    print(f"workload_only: {len(workload_keys - profile_keys)}")

    print("by_dataset")
    profile_groups = by_dataset(profile)
    workload_groups = by_dataset(workload)
    for dataset in sorted(set(profile_groups) | set(workload_groups)):
        pkeys = {key(record) for record in profile_groups.get(dataset, [])}
        wkeys = {key(record) for record in workload_groups.get(dataset, [])}
        print(
            f"  {dataset}: profile={len(pkeys)} workload={len(wkeys)} "
            f"overlap={len(pkeys & wkeys)}"
        )

    if profile_keys - workload_keys:
        print("sample_profile_only")
        for item in sorted(profile_keys - workload_keys)[:10]:
            print(f"  {item}")
    if workload_keys - profile_keys:
        print("sample_workload_only")
        for item in sorted(workload_keys - profile_keys)[:10]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
