"""Evaluate output-length predictions against ground-truth answers.

This is a lightweight alternative to evaluate_output_prediction.py. It does not
run a model backend. Instead, it runs the normal metadata/analyzer path and
compares predicted output length with the token length of a ground-truth answer
found in either the top-level record or record["metadata"].
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmserve_skeleton.analyzer import (  # noqa: E402
    MultimodalRequestAnalyzer,
    OutputCategoryClassifier,
    OutputLengthPredictor,
    PrefillCostEstimator,
)
from mmserve_skeleton.models import MMRequest  # noqa: E402
from mmserve_skeleton.preprocessing import MetadataExtractor  # noqa: E402


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
        required=True,
        help="Input normalized dataset JSONL or replay workload JSONL",
    )
    parser.add_argument(
        "--output",
        default="logs/output_length_ground_truth_eval.jsonl",
        help="Per-request evaluation JSONL",
    )
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--classifier",
        choices=["keyword", "dataset-keyword"],
        default="keyword",
        help="Category classifier used by the output length predictor",
    )
    parser.add_argument(
        "--predictor-config",
        default=None,
        help="Optional JSON config for OutputLengthPredictor.from_json",
    )
    parser.add_argument(
        "--prefill-profile",
        default=None,
        help="Optional JSON profile for PrefillCostEstimator.from_profile",
    )
    parser.add_argument(
        "--answer-field",
        default=None,
        help="Override the ground-truth answer field name",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number}")
            if "prompt" not in record:
                raise ValueError(f"Input line {line_number} is missing required field 'prompt'")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def build_metadata_extractor(args: argparse.Namespace) -> MetadataExtractor:
    strategy = args.classifier.replace("-", "_")
    classifier = OutputCategoryClassifier(strategy=strategy)
    if args.predictor_config:
        output_predictor = OutputLengthPredictor.from_json(args.predictor_config)
        output_predictor.classifier = classifier
    else:
        output_predictor = OutputLengthPredictor(classifier=classifier)

    prefill_estimator = (
        PrefillCostEstimator.from_profile(args.prefill_profile)
        if args.prefill_profile
        else PrefillCostEstimator()
    )
    analyzer = MultimodalRequestAnalyzer(
        prefill_estimator=prefill_estimator,
        output_predictor=output_predictor,
    )
    return MetadataExtractor(analyzer=analyzer)


def request_from_record(record: dict[str, Any], index: int) -> MMRequest:
    metadata = dict(record.get("metadata") or {})
    request_id = record.get("request_id", record.get("id", f"gt-eval-{index}"))
    return MMRequest(
        request_id=str(request_id),
        arrival_time=float(record.get("arrival_time", time.time())),
        prompt=record["prompt"],
        image_path=record.get("image_path"),
        dataset=record.get("dataset"),
        source=record.get("source"),
        metadata=metadata,
    )


def find_ground_truth_answer(
    record: dict[str, Any],
    answer_field: str | None = None,
) -> Any:
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
    tokens = re.findall(r"\S+", text)
    return len(tokens)


def evaluate_record(
    record: dict[str, Any],
    index: int,
    metadata_extractor: MetadataExtractor,
    classifier_name: str,
    answer_field: str | None,
) -> dict[str, Any] | None:
    answer = find_ground_truth_answer(record, answer_field)
    actual = count_answer_tokens(answer)
    if actual is None:
        return None

    request = request_from_record(record, index)
    metadata_extractor.enrich(request)
    predicted = request.features.predicted_output_length

    absolute_error = abs(predicted - actual) if predicted is not None else None
    signed_error = predicted - actual if predicted is not None else None
    absolute_percentage_error = (
        absolute_error / actual if absolute_error is not None and actual else None
    )

    return {
        "request_id": request.request_id,
        "dataset": request.dataset,
        "source": request.source,
        "classifier": classifier_name,
        "predicted_category": request.features.predicted_category,
        "prompt_length": request.features.text_length,
        "image_width": request.features.image_width,
        "image_height": request.features.image_height,
        "resolution_bucket": request.features.resolution_bucket,
        "predicted_prefill_cost": request.features.predicted_prefill_cost,
        "predicted_output_length": predicted,
        "ground_truth_output_length": actual,
        "absolute_error": absolute_error,
        "signed_error": signed_error,
        "absolute_percentage_error": absolute_percentage_error,
        "ground_truth_answer": truncate(answer_to_text(answer)),
        "metadata": request.metadata,
    }


def truncate(text: str | None, limit: int = 300) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def numeric(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(mean([error * error for error in errors]))


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def share_within_relative_error(records: list[dict[str, Any]], threshold: float) -> float:
    eligible = [
        record
        for record in records
        if record.get("absolute_percentage_error") is not None
    ]
    if not eligible:
        return 0.0
    good = [
        record
        for record in eligible
        if float(record["absolute_percentage_error"]) <= threshold
    ]
    return len(good) / len(eligible)


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def print_error_summary(label: str, records: list[dict[str, Any]], skipped: int) -> None:
    actuals = numeric(records, "ground_truth_output_length")
    predictions = numeric(records, "predicted_output_length")
    absolute_errors = numeric(records, "absolute_error")
    signed_errors = numeric(records, "signed_error")
    percentage_errors = numeric(records, "absolute_percentage_error")

    print(f"{label}: n={len(records)} skipped_missing_answer={skipped}")
    if not records:
        return

    mape = mean(percentage_errors) * 100 if percentage_errors else None
    corr = correlation(predictions, actuals)
    print(f"  ground_truth_mean={format_float(mean(actuals) if actuals else None)}")
    print(f"  predicted_mean={format_float(mean(predictions) if predictions else None)}")
    print(f"  mae={format_float(mean(absolute_errors) if absolute_errors else None)} rmse={format_float(rmse(absolute_errors))}")
    print(
        f"  p50_abs={format_float(median(absolute_errors) if absolute_errors else None)} "
        f"p90_abs={format_float(percentile(absolute_errors, 0.90))}"
    )
    print(f"  bias_pred_minus_ground_truth={format_float(mean(signed_errors) if signed_errors else None)} mape={format_float(mape)}%")
    print(
        f"  within_10pct={format_float(share_within_relative_error(records, 0.10) * 100)}% "
        f"within_25pct={format_float(share_within_relative_error(records, 0.25) * 100)}%"
    )
    print(f"  pearson_pred_ground_truth={format_float(corr)}")


def print_group_summaries(records: list[dict[str, Any]], group_key: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(group_key) or "unknown")].append(record)

    print(f"by_{group_key}:")
    for group, group_records in sorted(groups.items()):
        errors = numeric(group_records, "absolute_error")
        signed = numeric(group_records, "signed_error")
        actuals = numeric(group_records, "ground_truth_output_length")
        print(
            f"  {group}: n={len(group_records)} "
            f"ground_truth_mean={format_float(mean(actuals) if actuals else None)} "
            f"mae={format_float(mean(errors) if errors else None)} "
            f"bias={format_float(mean(signed) if signed else None)}"
        )


def write_records(path: str | Path, records: list[dict[str, Any]], reset: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and output_path.exists():
        output_path.unlink()
    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    inputs = load_jsonl(args.input, args.limit)
    metadata_extractor = build_metadata_extractor(args)

    records: list[dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(inputs, start=1):
        evaluated = evaluate_record(
            record=record,
            index=index,
            metadata_extractor=metadata_extractor,
            classifier_name=args.classifier,
            answer_field=args.answer_field,
        )
        if evaluated is None:
            skipped += 1
        else:
            records.append(evaluated)

    write_records(args.output, records, args.reset_output)
    print(f"Wrote {len(records)} evaluation records to {args.output}")
    print_error_summary("ground_truth_output_length_prediction", records, skipped)
    print_group_summaries(records, "predicted_category")
    print_group_summaries(records, "dataset")


if __name__ == "__main__":
    main()
