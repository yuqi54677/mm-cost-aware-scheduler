"""
Create replay workloads from normalized local JSONL datasets.

Input:
    A JSONL file from scripts/assemble_datasets.py, with fields like:
    id, dataset, source, prompt, image_path, answer, category, metadata.

Output:
    A JSONL workload for scripts/run_workload.py. Each row is one request with
    request_id, prompt, image_path, dataset/source labels, optional metadata, and
    optional arrival_time.

Author: Otto
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse workload-generation configuration.

    Defaults match the normalized schema emitted by assemble_datasets.py.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Normalized JSONL dataset")
    parser.add_argument("--output", required=True, help="Generated workload JSONL")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--dataset", default=None, help="Dataset label to write on every row")
    parser.add_argument("--dataset-field", default="dataset", help="Copy dataset label from this input field")
    parser.add_argument("--source", default=None, help="Source label to write on every row")
    parser.add_argument("--source-field", default="source", help="Copy source label from this input field")
    parser.add_argument("--request-id-field", default="id", help="Copy request ID from this input field")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--arrival-process",
        choices=["none", "fixed-gap", "poisson"],
        default="poisson",
        help="How to synthesize arrival_time values",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=0.1,
        help="Interarrival gap for fixed-gap workloads",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Mean arrivals per second for Poisson workloads",
    )
    parser.add_argument(
        "--metadata-fields",
        default="answer,category",
        help="Comma-separated input fields to copy into the metadata object",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load normalized dataset records from JSONL.

    Input:
        path: local JSONL file, usually produced by assemble_datasets.py.
    Output:
        List of JSON object records.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number}")
            records.append(record)
    return records


def arrival_times(
    count: int,
    process: str,
    rng: random.Random,
    gap_seconds: float,
    rate: float,
) -> list[float | None]:
    """Generate synthetic request arrival times.

    Input:
        count: number of requests.
        process: none, fixed-gap, or poisson.
        rng: seeded random generator for reproducibility.
        gap_seconds: deterministic interarrival time for fixed-gap.
        rate: mean arrivals per second for Poisson.
    Output:
        Arrival times in seconds from experiment start, or None values when
        arrival simulation is disabled.
    """
    if process == "none":
        return [None] * count

    if count == 0:
        return []

    if process == "fixed-gap":
        if gap_seconds < 0:
            raise ValueError("--gap-seconds must be non-negative")
        return [index * gap_seconds for index in range(count)]

    if rate <= 0:
        raise ValueError("--rate must be positive for Poisson arrivals")

    offsets = [0.0]
    current = 0.0
    for _ in range(1, count):
        current += rng.expovariate(rate)
        offsets.append(current)
    return offsets


def copy_label(record: dict[str, Any], literal: str | None, field: str | None) -> Any:
    """Choose a literal label first, otherwise copy a label from the input record."""
    if literal is not None:
        return literal
    if field is not None:
        return record.get(field)
    return None


def build_workload_record(
    record: dict[str, Any],
    index: int,
    args: argparse.Namespace,
    arrival_time: float | None,
    metadata_fields: list[str],
) -> dict[str, Any]:
    """Convert one normalized dataset record into one replay request.

    Input:
        record: normalized dataset row.
        index: one-based row index after optional shuffle/limit.
        args: workload-generation options.
        arrival_time: synthetic arrival time for this request.
        metadata_fields: top-level fields to preserve inside workload metadata.
    Output:
        One JSON-serializable request row for run_workload.py.
    """
    if args.prompt_field not in record:
        raise ValueError(f"Input record {index} is missing prompt field '{args.prompt_field}'")

    request_id = record.get(args.request_id_field) if args.request_id_field else None
    generated_request_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{Path(args.input).resolve()}:{args.seed}:{index}",
    )
    output: dict[str, Any] = {
        "request_id": str(request_id) if request_id is not None else str(generated_request_id),
        "prompt": record[args.prompt_field],
        "image_path": record.get(args.image_field),
    }

    dataset = copy_label(record, args.dataset, args.dataset_field)
    source = copy_label(record, args.source, args.source_field)
    if dataset is not None:
        output["dataset"] = dataset
    if source is not None:
        output["source"] = source
    if arrival_time is not None:
        output["arrival_time"] = round(float(arrival_time), 6)

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    metadata = dict(metadata)
    for field in metadata_fields:
        if field in record:
            metadata[field] = record.get(field)
    if metadata:
        output["metadata"] = metadata

    return output


def main() -> None:
    """Generate a workload file from normalized data and synthetic arrivals."""
    args = parse_args()
    rng = random.Random(args.seed)
    records = load_jsonl(args.input)

    if args.shuffle:
        rng.shuffle(records)
    if args.limit is not None:
        records = records[: args.limit]

    metadata_fields = [field.strip() for field in args.metadata_fields.split(",") if field.strip()]
    offsets = arrival_times(
        count=len(records),
        process=args.arrival_process,
        rng=rng,
        gap_seconds=args.gap_seconds,
        rate=args.rate,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, (record, offset) in enumerate(zip(records, offsets, strict=True), start=1):
            workload_record = build_workload_record(record, index, args, offset, metadata_fields)
            handle.write(json.dumps(workload_record, sort_keys=True) + "\n")

    print(f"Wrote {len(records)} workload records to {output_path}")


if __name__ == "__main__":
    main()
