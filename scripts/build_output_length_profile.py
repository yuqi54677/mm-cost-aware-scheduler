"""Build an output-length profile from sampled ground-truth examples.

The profile builder is separate from evaluation. It samples examples from the
requested datasets, classifies each prompt, measures ground-truth answer length,
and writes:
  1. a compact length profile JSON used by evaluate_output_prediction.py
  2. a summary JSON listing the exact examples selected
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmserve_skeleton.analyzer import OutputCategoryClassifier, OutputLengthProfile  # noqa: E402
from mmserve_skeleton.models import MMRequest  # noqa: E402


ANSWER_FIELDS = [
    "answer",
    "answers",
    "ground_truth",
    "ground_truth_answer",
    "target",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input normalized dataset/workload JSONL file(s)",
    )
    parser.add_argument(
        "--datasets",
        default="coco,textvqa,mmmu",
        help="Comma-separated dataset labels to sample",
    )
    parser.add_argument(
        "--examples-per-dataset",
        type=int,
        default=15,
        help="Random examples to sample from each dataset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classifier",
        choices=["keyword", "dataset-keyword"],
        default="dataset-keyword",
        help="Classifier used to assign profile categories",
    )
    parser.add_argument(
        "--profile-output",
        default="profiles/output_length_profile.json",
        help="Output length profile JSON used by evaluation",
    )
    parser.add_argument(
        "--summary-output",
        default="profiles/output_length_profile_examples.json",
        help="Summary JSON showing selected examples",
    )
    parser.add_argument(
        "--answer-field",
        default=None,
        help="Override the ground-truth answer field name",
    )
    return parser.parse_args()


def load_jsonl(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "prompt" not in record:
                    raise ValueError(f"{path}:{line_number} is missing required field 'prompt'")
                records.append(record)
    return records


def find_ground_truth_answer(record: dict[str, Any], answer_field: str | None) -> Any:
    fields = [answer_field] if answer_field else ANSWER_FIELDS
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}

    for field in fields:
        if field and record.get(field) is not None:
            return record[field]
        if field and metadata.get(field) is not None:
            return metadata[field]
    return None


def answer_to_text(answer: Any) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, list):
        return " ".join(str(item) for item in answer)
    if isinstance(answer, dict):
        return json.dumps(answer, sort_keys=True)
    return str(answer)


def count_answer_tokens(answer: Any) -> int | None:
    text = answer_to_text(answer)
    if text is None:
        return None
    return len(re.findall(r"\S+", text))


def group_by_dataset(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("dataset") or "unknown")].append(record)
    return grouped


def sample_records(
    records: list[dict[str, Any]],
    datasets: list[str],
    examples_per_dataset: int,
    rng: random.Random,
    answer_field: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    grouped = group_by_dataset(records)
    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}

    for dataset in datasets:
        candidates = [
            record
            for record in grouped.get(dataset, [])
            if count_answer_tokens(find_ground_truth_answer(record, answer_field)) is not None
        ]
        sample_size = min(examples_per_dataset, len(candidates))
        selected.extend(rng.sample(candidates, sample_size))
        counts[dataset] = {
            "available_with_answer": len(candidates),
            "requested": examples_per_dataset,
            "selected": sample_size,
        }
    return selected, counts


def request_from_record(record: dict[str, Any], index: int) -> MMRequest:
    request_id = record.get("request_id", record.get("id", f"profile-{index}"))
    return MMRequest(
        request_id=str(request_id),
        arrival_time=float(record.get("arrival_time", 0.0)),
        prompt=record["prompt"],
        image_path=record.get("image_path"),
        dataset=record.get("dataset"),
        source=record.get("source"),
        metadata=dict(record.get("metadata") or {}),
    )


def build_profile(
    records: list[dict[str, Any]],
    classifier: OutputCategoryClassifier,
    answer_field: str | None,
) -> tuple[OutputLengthProfile, list[dict[str, Any]]]:
    profile = OutputLengthProfile(min_samples=1)
    examples: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        request = request_from_record(record, index)
        category = classifier.predict(request)
        answer = find_ground_truth_answer(record, answer_field)
        answer_text = answer_to_text(answer)
        output_length = count_answer_tokens(answer)
        profile.observe(category, output_length)
        examples.append(
            {
                "request_id": request.request_id,
                "dataset": request.dataset,
                "source": request.source,
                "predicted_category": category,
                "ground_truth_output_length": output_length,
                "prompt": request.prompt,
                "ground_truth_answer": truncate(answer_text),
            }
        )

    return profile, examples


def truncate(text: str | None, limit: int = 300) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    records = load_jsonl(args.input)
    selected, sample_counts = sample_records(
        records=records,
        datasets=datasets,
        examples_per_dataset=args.examples_per_dataset,
        rng=rng,
        answer_field=args.answer_field,
    )
    classifier = OutputCategoryClassifier(strategy=args.classifier.replace("-", "_"))
    profile, examples = build_profile(selected, classifier, args.answer_field)

    profile.to_json(args.profile_output)
    write_json(
        args.summary_output,
        {
            "inputs": [str(path) for path in args.input],
            "seed": args.seed,
            "classifier": args.classifier,
            "datasets": datasets,
            "sample_counts": sample_counts,
            "profile_summary": profile.summary(),
            "examples": examples,
        },
    )

    print(f"Wrote length profile to {args.profile_output}")
    print(f"Wrote selected-example summary to {args.summary_output}")
    for dataset, counts in sample_counts.items():
        print(
            f"{dataset}: selected={counts['selected']} "
            f"available_with_answer={counts['available_with_answer']} "
            f"requested={counts['requested']}"
        )


if __name__ == "__main__":
    main()
