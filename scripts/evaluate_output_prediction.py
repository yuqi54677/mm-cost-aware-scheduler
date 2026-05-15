"""Evaluate predicted output length against backend-generated output length.

This script runs the same request feature/analyzer path used by the serving
pipeline, executes each request on the selected backend, records predictions and
actual output lengths, and prints aggregate error analysis.
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
from mmserve_skeleton.backend import MockBackend, VLLMBackend  # noqa: E402
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


class JSONLengthProfile:
    """Read-only length profile loaded from a prebuilt JSON file."""

    def __init__(self, samples: dict[str, list[int]], min_samples: int = 1) -> None:
        self.samples = {
            str(category): sorted(int(value) for value in values)
            for category, values in samples.items()
        }
        self.min_samples = min_samples

    @classmethod
    def from_json(cls, path: str | Path, min_samples: int = 1) -> "JSONLengthProfile":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        samples = data.get("samples", data)
        return cls(samples=samples, min_samples=int(data.get("min_samples", min_samples)))

    def percentile(self, category: str, fraction: float) -> int | None:
        values = self.samples.get(category, [])
        if len(values) < self.min_samples:
            return None
        index = min(len(values) - 1, int(fraction * (len(values) - 1)))
        return values[index]

    def p90(self, category: str) -> int | None:
        return self.percentile(category, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, help="Input workload JSONL")
    parser.add_argument(
        "--output",
        default="logs/output_length_eval.jsonl",
        help="Per-request evaluation JSONL",
    )
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only evaluate this dataset label. Omit to evaluate all datasets.",
    )
    parser.add_argument(
        "--target",
        choices=["ground-truth", "inference"],
        default="inference",
        help="Compare against ground-truth answer length or backend inference output length",
    )
    parser.add_argument(
        "--answer-field",
        default=None,
        help="For --target ground-truth, override the answer field name",
    )
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
        "--length-profile",
        default=None,
        help="Optional prebuilt JSON profile of observed output lengths by predicted category",
    )
    parser.add_argument(
        "--profile-min-samples",
        type=int,
        default=1,
        help="Minimum samples required before using a category profile percentile",
    )
    parser.add_argument(
        "--profile-percentile",
        type=float,
        default=0.90,
        help="Primary profile percentile used for predicted_output_length.",
    )
    parser.add_argument(
        "--ablation-percentiles",
        default="0.50,0.90",
        help="Comma-separated profile percentiles to report without rerunning inference.",
    )
    parser.add_argument(
        "--prefill-profile",
        default=None,
        help="Optional JSON profile for PrefillCostEstimator.from_profile",
    )
    parser.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument(
        "--system-prompt",
        default=(
            "Answer concisely. For questions, provide only the final answer "
            "unless more detail is explicitly requested."
        ),
        help="System instruction applied by the vLLM chat template.",
    )
    return parser.parse_args()


def load_workload(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "prompt" not in record:
                raise ValueError(f"Workload line {line_number} is missing required field 'prompt'")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def filter_records(
    records: list[dict[str, Any]],
    dataset: str | None,
) -> list[dict[str, Any]]:
    if dataset is None:
        return records
    return [record for record in records if record.get("dataset") == dataset]


def build_backend(args: argparse.Namespace):
    if args.backend == "vllm":
        return VLLMBackend(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
            system_prompt=args.system_prompt,
        )
    return MockBackend()


def build_output_length_profile(args: argparse.Namespace) -> JSONLengthProfile | None:
    if not args.length_profile:
        return None
    return JSONLengthProfile.from_json(
        args.length_profile,
        min_samples=args.profile_min_samples,
    )


def build_metadata_extractor(
    args: argparse.Namespace,
    length_profile: JSONLengthProfile | None = None,
) -> MetadataExtractor:
    strategy = args.classifier.replace("-", "_")
    classifier = OutputCategoryClassifier(strategy=strategy)
    if args.predictor_config:
        output_predictor = OutputLengthPredictor.from_json(args.predictor_config)
        output_predictor.classifier = classifier
    else:
        output_predictor = OutputLengthPredictor(classifier=classifier)
    output_predictor.length_profile = length_profile
    output_predictor.length_profile_percentile = args.profile_percentile

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
    arrival_time = float(record.get("arrival_time", time.time()))
    if arrival_time < 1_000_000_000:
        arrival_time = time.time()
    return MMRequest(
        request_id=str(record.get("request_id", f"eval-{index}")),
        arrival_time=arrival_time,
        prompt=record["prompt"],
        image_path=record.get("image_path"),
        dataset=record.get("dataset"),
        source=record.get("source"),
        metadata=dict(record.get("metadata") or {}),
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
    return len(re.findall(r"\S+", text))


def prediction_error_record(
    *,
    predicted: int | None,
    actual: int,
) -> dict[str, float | int | None]:
    absolute_error = abs(predicted - actual) if predicted is not None else None
    signed_error = predicted - actual if predicted is not None else None
    absolute_percentage_error = (
        absolute_error / actual if absolute_error is not None and actual else None
    )
    return {
        "absolute_error": absolute_error,
        "signed_error": signed_error,
        "absolute_percentage_error": absolute_percentage_error,
    }


def base_evaluation_record(
    request: MMRequest,
    classifier_name: str,
    target: str,
    system_prompt: str,
) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "dataset": request.dataset,
        "source": request.source,
        "classifier": classifier_name,
        "target": target,
        "predicted_category": request.features.predicted_category,
        "prompt_length": request.features.text_length,
        "image_width": request.features.image_width,
        "image_height": request.features.image_height,
        "resolution_bucket": request.features.resolution_bucket,
        "predicted_prefill_cost": request.features.predicted_prefill_cost,
        "predicted_output_length": request.features.predicted_output_length,
        "metadata": request.metadata,
        "system_prompt": system_prompt,
    }


def parse_percentiles(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value > 1.0:
            value /= 100.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Percentile must be in [0, 1] or [0, 100], got {part}")
        values.append(value)
    return values


def percentile_label(fraction: float) -> str:
    return f"p{int(round(fraction * 100)):02d}"


def ablation_prediction_record(
    request: MMRequest,
    actual: int,
    length_profile: JSONLengthProfile | None,
    percentiles: list[float],
) -> dict[str, dict[str, float | int | bool | None]]:
    category = request.features.predicted_category
    output: dict[str, dict[str, float | int | bool | None]] = {}
    for fraction in percentiles:
        profile_value = (
            length_profile.percentile(category, fraction)
            if length_profile is not None and category is not None
            else None
        )
        predicted = int(profile_value) if profile_value is not None else None
        error = prediction_error_record(predicted=predicted, actual=actual)
        output[percentile_label(fraction)] = {
            "percentile": fraction,
            "used_profile": profile_value is not None,
            "predicted_output_length": predicted,
            "underestimated": predicted is not None and predicted < actual,
            **error,
        }
    return output


def evaluate_record(
    record: dict[str, Any],
    index: int,
    metadata_extractor: MetadataExtractor,
    backend: Any,
    classifier_name: str,
    system_prompt: str,
    length_profile: JSONLengthProfile | None,
    ablation_percentiles: list[float],
) -> dict[str, Any]:
    request = request_from_record(record, index)
    metadata_extractor.enrich(request)
    predicted = request.features.predicted_output_length
    predicted_category = request.features.predicted_category

    started = time.time()
    result = backend.run_request(request)
    completed = time.time()
    actual = result.output_token_count

    output = base_evaluation_record(
        request=request,
        classifier_name=classifier_name,
        target="inference",
        system_prompt=system_prompt,
    )
    output.update(prediction_error_record(predicted=predicted, actual=actual))
    output.update(
        {
        "actual_output_length": actual,
        "inference_seconds": completed - started,
        "generated_text": truncate(result.generated_text),
        "backend_raw": result.raw,
        "prediction_ablation": ablation_prediction_record(
            request=request,
            actual=actual,
            length_profile=length_profile,
            percentiles=ablation_percentiles,
        ),
        }
    )
    return output


def evaluate_ground_truth_record(
    record: dict[str, Any],
    index: int,
    metadata_extractor: MetadataExtractor,
    classifier_name: str,
    answer_field: str | None,
    system_prompt: str,
    length_profile: JSONLengthProfile | None,
    ablation_percentiles: list[float],
) -> dict[str, Any] | None:
    answer = find_ground_truth_answer(record, answer_field)
    actual = count_answer_tokens(answer)
    if actual is None:
        return None

    request = request_from_record(record, index)
    metadata_extractor.enrich(request)
    predicted = request.features.predicted_output_length

    output = base_evaluation_record(
        request=request,
        classifier_name=classifier_name,
        target="ground-truth",
        system_prompt=system_prompt,
    )
    output.update(prediction_error_record(predicted=predicted, actual=actual))
    output.update(
        {
            "ground_truth_output_length": actual,
            "ground_truth_answer": truncate(answer_to_text(answer)),
            "prediction_ablation": ablation_prediction_record(
                request=request,
                actual=actual,
                length_profile=length_profile,
                percentiles=ablation_percentiles,
            ),
        }
    )
    return output


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


def print_error_summary(label: str, records: list[dict[str, Any]]) -> None:
    actual_key = actual_length_key(records)
    actuals = numeric(records, actual_key)
    predictions = numeric(records, "predicted_output_length")
    absolute_errors = numeric(records, "absolute_error")
    signed_errors = numeric(records, "signed_error")
    percentage_errors = numeric(records, "absolute_percentage_error")

    if not records:
        print(f"{label}: n=0")
        return

    mae = mean(absolute_errors) if absolute_errors else None
    bias = mean(signed_errors) if signed_errors else None
    mape = mean(percentage_errors) * 100 if percentage_errors else None
    within_10 = share_within_relative_error(records, 0.10)
    within_25 = share_within_relative_error(records, 0.25)
    corr = correlation(predictions, actuals)

    print(f"{label}: n={len(records)} actual_key={actual_key}")
    print(f"  actual_mean={format_float(mean(actuals) if actuals else None)}")
    print(f"  predicted_mean={format_float(mean(predictions) if predictions else None)}")
    print(f"  mae={format_float(mae)} rmse={format_float(rmse(absolute_errors))}")
    print(
        f"  p50_abs={format_float(median(absolute_errors) if absolute_errors else None)} "
        f"p90_abs={format_float(percentile(absolute_errors, 0.90))}"
    )
    print(f"  bias_pred_minus_actual={format_float(bias)} mape={format_float(mape)}%")
    print(f"  within_10pct={format_float(within_10 * 100)}% within_25pct={format_float(within_25 * 100)}%")
    print(f"  pearson_pred_actual={format_float(corr)}")


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


def print_group_summaries(records: list[dict[str, Any]], group_key: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(group_key) or "unknown")].append(record)

    print(f"by_{group_key}:")
    for group, group_records in sorted(groups.items()):
        errors = numeric(group_records, "absolute_error")
        signed = numeric(group_records, "signed_error")
        actuals = numeric(group_records, actual_length_key(group_records))
        print(
            f"  {group}: n={len(group_records)} "
            f"actual_mean={format_float(mean(actuals) if actuals else None)} "
            f"mae={format_float(mean(errors) if errors else None)} "
            f"bias={format_float(mean(signed) if signed else None)}"
        )


def actual_length_key(records: list[dict[str, Any]]) -> str:
    if records and records[0].get("target") == "ground-truth":
        return "ground_truth_output_length"
    return "actual_output_length"


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
    workload = filter_records(load_workload(args.workload), args.dataset)
    if args.limit is not None:
        workload = workload[: args.limit]
    backend = build_backend(args) if args.target == "inference" else None
    length_profile = build_output_length_profile(args)
    metadata_extractor = build_metadata_extractor(args, length_profile)
    ablation_percentiles = parse_percentiles(args.ablation_percentiles)

    records: list[dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(workload, start=1):
        if args.target == "ground-truth":
            evaluated = evaluate_ground_truth_record(
                record=record,
                index=index,
                metadata_extractor=metadata_extractor,
                classifier_name=args.classifier,
                answer_field=args.answer_field,
                system_prompt=args.system_prompt,
                length_profile=length_profile,
                ablation_percentiles=ablation_percentiles,
            )
            if evaluated is None:
                skipped += 1
            else:
                records.append(evaluated)
        else:
            evaluated = evaluate_record(
                record=record,
                index=index,
                metadata_extractor=metadata_extractor,
                backend=backend,
                classifier_name=args.classifier,
                system_prompt=args.system_prompt,
                length_profile=length_profile,
                ablation_percentiles=ablation_percentiles,
            )
            records.append(evaluated)

    write_records(args.output, records, args.reset_output)
    dataset_label = args.dataset or "all"
    print(
        f"Wrote {len(records)} evaluation records to {args.output} "
        f"(dataset={dataset_label}, target={args.target}, skipped={skipped})"
    )
    print_error_summary("overall_output_length_prediction", records)
    print_group_summaries(records, "predicted_category")
    print_group_summaries(records, "dataset")


if __name__ == "__main__":
    main()
