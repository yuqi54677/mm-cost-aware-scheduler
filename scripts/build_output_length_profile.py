"""Build an output-length profile from model-generated outputs.

The profile builder samples examples from COCO, TextVQA, and MMMU, runs real
model inference, records generated output lengths by predicted category, and
writes:
  1. a compact length profile JSON used by evaluate_output_prediction.py
  2. a summary JSON listing selected examples, generated text, and lengths
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_datasets import iter_normalized_records  # noqa: E402
from mmserve_skeleton.analyzer import OutputCategoryClassifier, OutputLengthProfile  # noqa: E402
from mmserve_skeleton.backend import VLLMBackend  # noqa: E402
from mmserve_skeleton.models import MMRequest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Optional normalized dataset/workload JSONL file(s). If omitted, load datasets directly.",
    )
    parser.add_argument(
        "--datasets",
        default="coco,textvqa,mmmu",
        help="Comma-separated dataset labels to sample",
    )
    parser.add_argument(
        "--examples-per-dataset",
        type=int,
        default=10,
        help="Random examples to sample from each dataset",
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
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument(
        "--classifier",
        choices=["keyword", "dataset-keyword"],
        default="dataset-keyword",
        help="Classifier used to assign profile categories",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="vLLM model used for profiling inference",
    )
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
    parser.add_argument(
        "--profile-output",
        default="profiles/output_length_profile.json",
        help="Output length profile JSON used by evaluation",
    )
    parser.add_argument(
        "--summary-output",
        default="profiles/output_length_profile_examples.json",
        help="Summary JSON showing selected examples and generated outputs",
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


def load_candidates(args: argparse.Namespace, dataset_names: list[str]) -> list[dict[str, Any]]:
    if args.input:
        return load_jsonl(args.input)

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
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    grouped = group_by_dataset(records)
    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}

    for dataset in datasets:
        candidates = grouped.get(dataset, [])
        sample_size = min(examples_per_dataset, len(candidates))
        chosen = rng.sample(candidates, sample_size)
        selected.extend(chosen)
        counts[dataset] = {
            "available": len(candidates),
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


def build_profile_from_inference(
    records: list[dict[str, Any]],
    classifier: OutputCategoryClassifier,
    backend: VLLMBackend,
) -> tuple[OutputLengthProfile, list[dict[str, Any]]]:
    profile = OutputLengthProfile(min_samples=1)
    examples: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        request = request_from_record(record, index)
        category = classifier.predict(request)
        result = backend.run_request(request)
        output_length = result.output_token_count
        profile.observe(category, output_length)
        examples.append(
            {
                "request_id": request.request_id,
                "dataset": request.dataset,
                "source": request.source,
                "predicted_category": category,
                "generated_output_length": output_length,
                "prompt": request.prompt,
                "image_path": request.image_path,
                "generated_text": truncate(result.generated_text),
                "backend_raw": result.raw,
            }
        )
        print(
            f"[{index}/{len(records)}] {request.dataset} {request.request_id} "
            f"category={category} generated_tokens={output_length}"
        )

    return profile, examples


def truncate(text: str | None, limit: int = 500) -> str | None:
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
    candidates = load_candidates(args, datasets)
    selected, sample_counts = sample_records(
        records=candidates,
        datasets=datasets,
        examples_per_dataset=args.examples_per_dataset,
        rng=rng,
    )

    classifier = OutputCategoryClassifier(strategy=args.classifier.replace("-", "_"))
    backend = VLLMBackend(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=args.vllm_enforce_eager,
        system_prompt=args.system_prompt,
    )
    profile, examples = build_profile_from_inference(selected, classifier, backend)

    profile.to_json(args.profile_output)
    write_json(
        args.summary_output,
        {
            "inputs": [str(path) for path in args.input] if args.input else None,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "classifier": args.classifier,
            "system_prompt": args.system_prompt,
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
            f"available={counts['available']} requested={counts['requested']}"
        )


if __name__ == "__main__":
    main()
