"""Build a small normalized profiling dataset.

This script reuses the dataset-specific normalization loaders from
assemble_datasets.py, randomly samples examples from each dataset, and writes a
single JSON array. The output is intended for building output-length profiles
without mixing profiling examples into the evaluation set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from assemble_datasets import iter_demo_records, iter_normalized_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="profiling_dataset.json",
        help="Output JSON file containing a normalized profiling set",
    )
    parser.add_argument(
        "--datasets",
        default="coco,textvqa,mmmu",
        help="Comma-separated dataset names to include",
    )
    parser.add_argument(
        "--examples-per-dataset",
        type=int,
        default=15,
        help="Number of random examples to keep per dataset",
    )
    parser.add_argument(
        "--candidate-limit-per-dataset",
        type=int,
        default=200,
        help="Number of normalized candidates to load before random sampling",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable Hugging Face streaming and build local Arrow caches",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for non-streaming loads",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demo examples instead of Hugging Face datasets",
    )
    return parser.parse_args()


def collect_candidates(args: argparse.Namespace, dataset_names: list[str]) -> list[dict[str, Any]]:
    if args.demo:
        return list(
            iter_demo_records(
                dataset_names=dataset_names,
                limit_per_dataset=args.candidate_limit_per_dataset,
                split=args.split,
            )
        )

    return list(
        iter_normalized_records(
            dataset_names=dataset_names,
            limit_per_dataset=args.candidate_limit_per_dataset,
            split=args.split,
            image_dir=Path(args.image_dir),
            streaming=not args.no_streaming,
            cache_dir=args.hf_cache_dir,
        )
    )


def sample_by_dataset(
    records: list[dict[str, Any]],
    dataset_names: list[str],
    examples_per_dataset: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("dataset") or "unknown")].append(record)

    sampled: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    for dataset in dataset_names:
        candidates = grouped.get(dataset, [])
        sample_size = min(examples_per_dataset, len(candidates))
        chosen = rng.sample(candidates, sample_size)
        sampled.extend(chosen)
        summary[dataset] = {
            "candidate_count": len(candidates),
            "requested": examples_per_dataset,
            "selected": sample_size,
        }

    sampled.sort(key=lambda record: (str(record.get("dataset")), str(record.get("id"))))
    return sampled, summary


def write_json(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    candidates = collect_candidates(args, dataset_names)
    sampled, summary = sample_by_dataset(
        records=candidates,
        dataset_names=dataset_names,
        examples_per_dataset=args.examples_per_dataset,
        rng=rng,
    )
    write_json(args.output, sampled)

    print(f"Wrote {len(sampled)} normalized profiling examples to {args.output}")
    for dataset, counts in summary.items():
        print(
            f"{dataset}: selected={counts['selected']} "
            f"candidates={counts['candidate_count']} requested={counts['requested']}"
        )


if __name__ == "__main__":
    main()
