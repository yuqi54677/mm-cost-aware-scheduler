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
    """Parse assembly configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/normalized/assembled.jsonl")
    parser.add_argument("--image-dir", default="data/images", help="Directory to save downloaded images")
    parser.add_argument("--limit-per-dataset", type=int, default=100)
    parser.add_argument("--include", default="coco,textvqa,mmmu")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic COCO/TextVQA/MMMU-like rows without Hugging Face datasets",
    )
    return parser.parse_args()


# Each dataset uses its own split naming convention.  Map the user-facing
# canonical names ("train", "validation", "test") to the actual HF split names.
_SPLIT_MAP: dict[str, dict[str, str]] = {
    "coco":    {"train": "train", "validation": "val",        "test": "test", "val": "val"},
    "textvqa": {"train": "train", "validation": "validation", "test": "test", "val": "validation"},
    "mmmu":    {"train": "dev",   "validation": "validation", "test": "test", "val": "validation", "dev": "dev"},
}


def resolve_split(dataset: str, requested: str) -> str:
    """Translate a canonical split name to the dataset-specific HF split name."""
    mapping = _SPLIT_MAP.get(dataset, {})
    resolved = mapping.get(requested, requested)
    return resolved


def load_hf_dataset(dataset_name: str, split: str, config: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the Hugging Face datasets package: pip install datasets"
        ) from exc
    if config:
        return load_dataset(dataset_name, config, split=split)
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


def save_pil_image(image: Any, image_dir: Path, filename: str) -> str | None:
    """Save a PIL Image object to disk and return the local path.

    HuggingFace dataset image fields are in-memory PIL Image objects with no
    filename. This function persists them so the pipeline can load them later.
    Returns None if image is already a string path or cannot be saved.
    """
    if image is None:
        return None
    if isinstance(image, str):
        return image

    # Already has a filename (e.g. a file-backed PIL image)
    filename_attr = getattr(image, "filename", None)
    if filename_attr:
        return str(filename_attr)

    try:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            image_dir.mkdir(parents=True, exist_ok=True)
            dest = image_dir / filename
            image.convert("RGB").save(dest)
            return str(dest)
    except Exception:
        pass
    return None


def load_coco_samples(limit: int, split: str, image_dir: Path) -> Iterable[dict[str, Any]]:
    """Load and normalize COCO-style samples."""
    raw = load_hf_dataset("lmms-lab/COCO-Caption2017", resolve_split("coco", split))
    for index, row in enumerate(raw):
        if index >= limit:
            break
        prompt = add_qwen_vision_tokens("Describe the image in detail.")
        answer = first_present(row, ["caption", "captions", "answer"])
        raw_image = first_present(row, ["image", "image_path", "file_name"])
        image_path = save_pil_image(raw_image, image_dir / "coco", f"{split}_{index}.jpg")
        yield normalize_record(
            sample_id=f"coco-{split}-{index}",
            dataset="coco",
            source=split,
            prompt=prompt,
            image_path=image_path,
            answer=stringify_answer(answer),
            category="captioning",
            metadata={"raw_keys": sorted(row.keys())},
        )


def load_textvqa_samples(limit: int, split: str, image_dir: Path) -> Iterable[dict[str, Any]]:
    """Load and normalize TextVQA/OCR-heavy samples."""
    raw = load_hf_dataset("lmms-lab/TextVQA", resolve_split("textvqa", split))
    for index, row in enumerate(raw):
        if index >= limit:
            break
        question = first_present(row, ["question", "prompt"], "Read the visible text and answer.")
        raw_image = first_present(row, ["image", "image_path"])
        sample_id = str(first_present(row, ["question_id", "id"], f"textvqa-{split}-{index}"))
        image_path = save_pil_image(raw_image, image_dir / "textvqa", f"{split}_{index}.jpg")
        yield normalize_record(
            sample_id=sample_id,
            dataset="textvqa",
            source=split,
            prompt=add_qwen_vision_tokens(str(question)),
            image_path=image_path,
            answer=stringify_answer(first_present(row, ["answers", "answer"])),
            category="ocr",
            metadata={"raw_keys": sorted(row.keys())},
        )


MMMU_CONFIGS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]


def load_mmmu_samples(limit: int, split: str, image_dir: Path) -> Iterable[dict[str, Any]]:
    """Load and normalize MMMU/reasoning-heavy samples across all subject configs."""
    hf_split = resolve_split("mmmu", split)
    collected = 0
    for config in MMMU_CONFIGS:
        if collected >= limit:
            break
        try:
            raw = load_hf_dataset("MMMU/MMMU", hf_split, config=config)
        except Exception as exc:
            print(f"  Skipping MMMU config '{config}': {exc}")
            continue
        for row in raw:
            if collected >= limit:
                break
            question = first_present(row, ["question", "prompt"], "")
            options = first_present(row, ["options", "choices"], None)
            if options:
                question = f"{question}\nOptions: {options}"
            raw_image = first_present(row, ["image_1", "image", "image_path"])
            sample_id = str(first_present(row, ["id", "question_id"], f"mmmu-{split}-{collected}"))
            image_path = save_pil_image(raw_image, image_dir / "mmmu", f"{split}_{collected}.jpg")
            yield normalize_record(
                sample_id=sample_id,
                dataset="mmmu",
                source=split,
                prompt=add_qwen_vision_tokens(str(question)),
                image_path=image_path,
                answer=stringify_answer(first_present(row, ["answer", "correct_answer"])),
                category=config,
                metadata={"raw_keys": sorted(row.keys()), "mmmu_config": config},
            )
            collected += 1


def add_qwen_vision_tokens(prompt: str) -> str:
    return f"<|vision_start|><|image_pad|><|vision_end|>\n{prompt}"

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
    """Create one normalized local dataset record."""
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
    image_dir: Path,
) -> Iterable[dict[str, Any]]:
    """Dispatch to each dataset loader and yield normalized records."""
    loaders = {
        "coco": load_coco_samples,
        "textvqa": load_textvqa_samples,
        "mmmu": load_mmmu_samples,
    }
    for name in dataset_names:
        if name not in loaders:
            raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(loaders))}")
        yield from loaders[name](limit_per_dataset, split, image_dir)


def iter_demo_records(dataset_names: list[str], limit_per_dataset: int, split: str) -> Iterable[dict[str, Any]]:
    """Generate richer local examples when HF datasets are not installed."""
    examples = {
        "coco": [
            ("A group of students sitting around a table with laptops.", "Describe the image in detail.", "captioning"),
            ("A city street with buses, signs, and pedestrians.", "Give a concise caption for this image.", "captioning"),
            ("A kitchen counter with several bowls and fruit.", "Describe the visible objects and scene.", "captioning"),
        ],
        "textvqa": [
            ("STOP", "What word is written on the red sign?", "ocr"),
            ("12:45", "Read the time shown on the clock.", "ocr"),
            ("SALE 50% OFF", "What discount is advertised in the image?", "ocr"),
        ],
        "mmmu": [
            ("B", "A diagram shows force applied to a block. Which option best explains the motion?\nOptions: A. no motion B. acceleration C. constant mass D. unknown", "physics"),
            ("42", "Solve the math problem shown in the image and give the final answer.", "math"),
            ("mitochondria", "The biology figure labels an organelle. Which structure produces ATP?", "biology"),
        ],
    }

    for dataset in dataset_names:
        if dataset not in examples:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(examples))}")
        for index in range(limit_per_dataset):
            answer, prompt, category = examples[dataset][index % len(examples[dataset])]
            yield normalize_record(
                sample_id=f"demo-{dataset}-{split}-{index}",
                dataset=dataset,
                source=f"demo-{split}",
                prompt=add_qwen_vision_tokens(prompt),
                image_path=None,
                answer=answer,
                category=category,
                metadata={"demo": True, "template_index": index % len(examples[dataset])},
            )


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    """Write normalized records to disk."""
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
    image_dir = Path(args.image_dir)
    dataset_names = [name.strip() for name in args.include.split(",") if name.strip()]
    records = (
        iter_demo_records(dataset_names, args.limit_per_dataset, args.split)
        if args.demo
        else iter_normalized_records(dataset_names, args.limit_per_dataset, args.split, image_dir)
    )
    count = write_jsonl(
        records,
        args.output,
    )
    print(f"Wrote {count} normalized records to {args.output}")
    if not args.demo:
        print(f"Images saved under {image_dir}/")


if __name__ == "__main__":
    main()
