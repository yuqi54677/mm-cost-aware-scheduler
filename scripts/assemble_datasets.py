"""
Assemble raw multimodal datasets into one normalized local JSONL file.

Purpose:
    Hide dataset-specific loading and field naming here. Downstream experiment
    code should read the normalized output instead of knowing about COCO,
    TextVQA, MMMU, or any other raw dataset format.

Normalized output schema, one JSON object per line:
    {
        "id": "unique stable sample id",
        "dataset": "coco | textvqa | mmmu | ...",
        "source": "train | val | test | local split name",
        "prompt": "model input text",
        "image_path": "local/path/to/image.jpg",
        "answer": "optional ground truth or null",
        "category": "descriptive | ocr | reasoning | short | ...",
        "metadata": {
            "dataset_specific_field": "kept for later analysis"
        }
    }

Next step:
    scripts/create_workload.py reads this file and adds request IDs plus
    arrival_time values for serving replay experiments.

Author: Otto
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse assembly configuration.

    Input:
        CLI flags that describe where to write normalized records and how many
        examples to keep from each raw dataset.
    Output:
        argparse.Namespace consumed by main().
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/normalized/assembled.jsonl")
    parser.add_argument("--limit-per-dataset", type=int, default=100)
    parser.add_argument("--include", default="coco,textvqa,mmmu")
    parser.add_argument("--split", default="validation")
    return parser.parse_args()


def load_hf_dataset(dataset_name: str, split: str):
    """Load a Hugging Face dataset with a clear error when dependencies are missing."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the Hugging Face datasets package to assemble real datasets: "
            "pip install datasets"
        ) from exc
    return load_dataset(dataset_name, split=split)


def first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Return the first available value from a raw dataset row."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def stringify_answer(answer: Any) -> str | None:
    """Make answers JSONL-friendly while preserving simple labels."""
    if answer is None:
        return None
    if isinstance(answer, list):
        return ", ".join(str(item) for item in answer)
    return str(answer)


def local_image_path(value: Any) -> str | None:
    """Convert common image representations into a local path when possible."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    filename = getattr(value, "filename", None)
    return str(filename) if filename else None


def load_coco_samples(limit: int, split: str) -> Iterable[dict[str, Any]]:
    """Load and normalize COCO-style samples.

    Input:
        limit: maximum number of COCO examples to emit.
    Output:
        Iterable of records matching the normalized schema above.

    Implementation note:
        Put the dataset-specific Hugging Face/local-file loading here, then call
        normalize_record(...) for each raw row.
    """
    raw = load_hf_dataset("lmms-lab/COCO-Caption2017", split)
    for index, row in enumerate(raw):
        if index >= limit:
            break
        prompt = "Describe the image in detail."
        answer = first_present(row, ["caption", "captions", "answer"])
        yield normalize_record(
            sample_id=f"coco-{split}-{index}",
            dataset="coco",
            source=split,
            prompt=prompt,
            image_path=local_image_path(first_present(row, ["image", "image_path", "file_name"])),
            answer=stringify_answer(answer),
            category="captioning",
            metadata={"raw_keys": sorted(row.keys())},
        )


def load_textvqa_samples(limit: int, split: str) -> Iterable[dict[str, Any]]:
    """Load and normalize TextVQA/OCR-heavy samples.

    Input:
        limit: maximum number of TextVQA examples to emit.
    Output:
        Iterable of records matching the normalized schema.
    """
    raw = load_hf_dataset("lmms-lab/TextVQA", split)
    for index, row in enumerate(raw):
        if index >= limit:
            break
        question = first_present(row, ["question", "prompt"], "Read the visible text and answer.")
        yield normalize_record(
            sample_id=str(first_present(row, ["question_id", "id"], f"textvqa-{split}-{index}")),
            dataset="textvqa",
            source=split,
            prompt=str(question),
            image_path=local_image_path(first_present(row, ["image", "image_path"])),
            answer=stringify_answer(first_present(row, ["answers", "answer"])),
            category="ocr",
            metadata={"raw_keys": sorted(row.keys())},
        )


def load_mmmu_samples(limit: int, split: str) -> Iterable[dict[str, Any]]:
    """Load and normalize MMMU/reasoning-heavy samples.

    Input:
        limit: maximum number of MMMU examples to emit.
    Output:
        Iterable of records matching the normalized schema.
    """
    raw = load_hf_dataset("MMMU/MMMU", split)
    for index, row in enumerate(raw):
        if index >= limit:
            break
        question = first_present(row, ["question", "prompt"], "")
        options = first_present(row, ["options", "choices"], None)
        if options:
            question = f"{question}\nOptions: {options}"
        yield normalize_record(
            sample_id=str(first_present(row, ["id", "question_id"], f"mmmu-{split}-{index}")),
            dataset="mmmu",
            source=split,
            prompt=str(question),
            image_path=local_image_path(first_present(row, ["image", "image_1", "image_path"])),
            answer=stringify_answer(first_present(row, ["answer", "correct_answer"])),
            category=str(first_present(row, ["subject", "category"], "reasoning")),
            metadata={"raw_keys": sorted(row.keys())},
        )


def normalize_record(
    *,
    sample_id: str,
    dataset: str,
    source: str,
    prompt: str,
    image_path: str | None,
    answer: Any = None,
    category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one normalized local dataset record.

    Input:
        Dataset-specific values extracted by a loader function.
    Output:
        A common JSON-serializable record for create_workload.py.
    """
    return {
        "id": sample_id,
        "dataset": dataset,
        "source": source,
        "prompt": prompt,
        "image_path": image_path,
        "answer": answer,
        "category": category,
        "metadata": metadata or {},
    }


def iter_normalized_records(
    dataset_names: list[str],
    limit_per_dataset: int,
    split: str,
) -> Iterable[dict[str, Any]]:
    """Dispatch to each dataset loader and yield normalized records.

    Input:
        dataset_names: dataset keys requested by the user.
        limit_per_dataset: maximum examples per dataset.
    Output:
        Combined stream of normalized records.
    """
    loaders = {
        "coco": load_coco_samples,
        "textvqa": load_textvqa_samples,
        "mmmu": load_mmmu_samples,
    }
    for name in dataset_names:
        if name not in loaders:
            raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(loaders))}")
        yield from loaders[name](limit_per_dataset, split)


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    """Write normalized records to disk.

    Input:
        records: normalized dataset records.
        output_path: JSONL destination consumed by create_workload.py.
    Output:
        Number of records written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    """Assemble selected datasets into a normalized local JSONL file."""
    args = parse_args()
    dataset_names = [name.strip() for name in args.include.split(",") if name.strip()]
    count = write_jsonl(
        iter_normalized_records(dataset_names, args.limit_per_dataset, args.split),
        args.output,
    )
    print(f"Wrote {count} normalized records to {args.output}")


if __name__ == "__main__":
    main()
