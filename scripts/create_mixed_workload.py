"""Create randomized mixed workloads from several dataset sources.

This script is meant to reproduce the shape of workloads/stress_mixed.jsonl:
sample N requests per dataset, mix them in random order, and assign randomized
arrival times. Real multimodal datasets are loaded through assemble_datasets.py;
the special dataset name "text-only" generates local synthetic text prompts.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_datasets import iter_demo_records, iter_normalized_records  # noqa: E402


TEXT_ONLY_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What are the main causes of climate change?",
    "Summarize the benefits of caching in distributed systems.",
    "Write a concise explanation of gradient descent.",
    "Describe how TCP congestion control works.",
    "What tradeoffs exist between latency and throughput?",
    "Explain why batching can improve serving efficiency.",
    "Give three examples of useful database indexes.",
    "Describe the purpose of a load balancer.",
    "Explain the difference between precision and recall.",
    "What is the role of regularization in machine learning?",
    "Summarize how attention works in transformer models.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output workload JSONL path")
    parser.add_argument(
        "--dataset-counts",
        required=True,
        help="Comma-separated counts, e.g. coco=150,textvqa=150,text-only=60",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-limit-per-dataset",
        type=int,
        default=None,
        help="Candidates loaded before sampling. Defaults to max(count * 4, count).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demo rows for supported multimodal datasets.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable Hugging Face streaming and build local Arrow caches.",
    )
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=10.0,
        help="Mean arrivals per second for Poisson interarrival times.",
    )
    parser.add_argument(
        "--arrival-process",
        choices=["poisson", "uniform"],
        default="poisson",
        help="How to randomize arrival times.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="For uniform arrivals, sample arrival_time values in [0, duration].",
    )
    return parser.parse_args()


def parse_dataset_counts(raw: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid dataset count '{part}'. Expected name=count.")
        name, count = part.split("=", 1)
        name = name.strip()
        value = int(count)
        if not name:
            raise ValueError("Dataset name cannot be empty.")
        if value < 0:
            raise ValueError(f"Count for dataset '{name}' must be non-negative.")
        counts[name] = value
    if not counts:
        raise ValueError("--dataset-counts must include at least one dataset.")
    return counts


def load_dataset_candidates(
    dataset: str,
    count: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if count == 0:
        return []

    if dataset == "text-only":
        return text_only_candidates(count)

    limit = args.candidate_limit_per_dataset
    if limit is None:
        limit = max(count * 4, count)
    if limit < count:
        raise ValueError(
            f"--candidate-limit-per-dataset ({limit}) must be >= requested "
            f"count ({count}) for dataset '{dataset}'."
        )

    iterator = (
        iter_demo_records([dataset], limit, args.split)
        if args.demo
        else iter_normalized_records(
            dataset_names=[dataset],
            limit_per_dataset=limit,
            split=args.split,
            image_dir=Path(args.image_dir),
            streaming=not args.no_streaming,
            cache_dir=args.hf_cache_dir,
        )
    )
    return list(iterator)


def text_only_candidates(count: int) -> list[dict[str, Any]]:
    repeats = (count // len(TEXT_ONLY_PROMPTS)) + 1
    prompts = (TEXT_ONLY_PROMPTS * repeats)[:count]
    return [
        {
            "id": f"text-{index}",
            "dataset": "text-only",
            "source": "synthetic",
            "prompt": prompt,
            "image_path": None,
            "metadata": {},
        }
        for index, prompt in enumerate(prompts)
    ]


def sample_records(
    dataset_counts: dict[str, int],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset, count in dataset_counts.items():
        candidates = load_dataset_candidates(dataset, count, args)
        if len(candidates) < count:
            raise ValueError(
                f"Dataset '{dataset}' only produced {len(candidates)} candidates; "
                f"requested {count}."
            )
        selected.extend(rng.sample(candidates, count))
    rng.shuffle(selected)
    return selected


def random_arrivals(count: int, args: argparse.Namespace, rng: random.Random) -> list[float]:
    if count == 0:
        return []
    if args.arrival_process == "uniform":
        duration = args.duration_seconds
        if duration is None:
            duration = max(1.0, count / max(args.arrival_rate, 1e-9))
        if duration < 0:
            raise ValueError("--duration-seconds must be non-negative.")
        arrivals = [rng.uniform(0.0, duration) for _ in range(count)]
        arrivals.sort()
        arrivals[0] = 0.0
        return arrivals

    if args.arrival_rate <= 0:
        raise ValueError("--arrival-rate must be positive for Poisson arrivals.")
    arrivals = [0.0]
    current = 0.0
    for _ in range(1, count):
        current += rng.expovariate(args.arrival_rate)
        arrivals.append(current)
    return arrivals


def workload_record(record: dict[str, Any], arrival_time: float) -> dict[str, Any]:
    output: dict[str, Any] = {
        "arrival_time": round(arrival_time, 6),
        "dataset": record.get("dataset"),
        "image_path": record.get("image_path"),
        "prompt": record["prompt"],
        "request_id": str(record.get("id") or record.get("request_id")),
        "source": record.get("source"),
    }

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    metadata = dict(metadata)
    for key in ("answer", "category"):
        if key in record:
            metadata[key] = record.get(key)
    if metadata:
        output["metadata"] = metadata
    return output


def write_jsonl(records: list[dict[str, Any]], arrivals: list[float], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, arrival_time in zip(records, arrivals, strict=True):
            handle.write(json.dumps(workload_record(record, arrival_time), sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset_counts = parse_dataset_counts(args.dataset_counts)
    records = sample_records(dataset_counts, args, rng)
    arrivals = random_arrivals(len(records), args, rng)
    write_jsonl(records, arrivals, args.output)
    print(f"Wrote {len(records)} mixed workload records to {args.output}")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count}")


if __name__ == "__main__":
    main()
