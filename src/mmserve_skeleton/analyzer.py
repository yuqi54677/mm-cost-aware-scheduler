"""Multimodal request analyzer and lightweight serving-cost estimators.

The report describes three analyzer responsibilities:
1. estimate prefill cost from image resolution and text length,
2. predict output length with category-aware models,
3. refine estimates as generation progresses.

This module implements runnable fallbacks and clean extension points for the
trained FastText/QRF models and A30-profiled alpha/beta constants.
"""

from __future__ import annotations

import json
import math
from bisect import insort
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import MMRequest


CATEGORY_LENGTH_PERCENTILES = {
    "brief": {"p5": 4, "p90": 32, "p99": 64},
    "descriptive": {"p5": 16, "p90": 96, "p99": 192},
    "ocr": {"p5": 4, "p90": 64, "p99": 128},
    "reasoning": {"p5": 24, "p90": 160, "p99": 256},
}

CATEGORY_DEFAULT_P90 = {
    category: values["p90"] for category, values in CATEGORY_LENGTH_PERCENTILES.items()
}


def visual_tokens_for_resolution(width: int | None, height: int | None, patch_size: int = 14) -> int:
    """Approximate vision tokens from image resolution."""
    if width is None or height is None:
        return 0
    return math.ceil(width / patch_size) * math.ceil(height / patch_size)


@dataclass
class PrefillCostEstimator:
    """Cp(r) = alpha * visual_tokens + beta * text_tokens."""

    alpha: float = 0.001
    beta: float = 0.0002
    patch_size: int = 14

    @classmethod
    def from_profile(cls, path: str | Path) -> "PrefillCostEstimator":
        """Load A30-profiled alpha/beta values from a small JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            alpha=float(data.get("alpha", cls.alpha)),
            beta=float(data.get("beta", cls.beta)),
            patch_size=int(data.get("patch_size", cls.patch_size)),
        )

    def estimate_prefill_cost(
        self,
        image_resolution: tuple[int | None, int | None],
        text_length: int | None,
    ) -> float:
        """Estimate prefill latency/cost from resolution and text length."""
        width, height = image_resolution
        visual_tokens = visual_tokens_for_resolution(width, height, self.patch_size)
        text_tokens = text_length or 0
        return self.alpha * visual_tokens + self.beta * text_tokens

    def estimate_request(self, request: MMRequest) -> float:
        cost = self.estimate_prefill_cost(
            (request.features.image_width, request.features.image_height),
            request.features.text_length,
        )
        request.features.predicted_prefill_cost = cost
        return cost


class OutputCategoryClassifier:
    """FastText-style category classifier with deterministic fallback strategies.

    If a FastText model path is provided, load it. Otherwise use a deterministic
    classifier so the pipeline remains runnable without training artifacts.

    The default ``keyword`` strategy preserves the original behavior. The
    ``dataset_keyword`` strategy combines dataset identity with keyword matching
    so experiments can compare keyword-only classification against a simple
    dataset-prior baseline.
    """

    VALID_STRATEGIES = {"keyword", "dataset_keyword"}

    def __init__(
        self,
        model_path: str | Path | None = None,
        strategy: str = "keyword",
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown output category strategy '{strategy}'. "
                f"Available: {', '.join(sorted(self.VALID_STRATEGIES))}"
            )
        self.strategy = strategy
        self._model = None
        if model_path:
            try:
                import fasttext

                self._model = fasttext.load_model(str(model_path))
            except ImportError as exc:
                raise RuntimeError("Install fasttext to load a FastText classifier.") from exc

    def predict(self, request: MMRequest) -> str:
        if self._model is not None:
            label = self._model.predict(request.prompt, k=1)[0][0]
            return label.replace("__label__", "").lower()

        if self.strategy == "dataset_keyword":
            return self.predict_dataset_keyword(request)

        return self.predict_keyword_only(request)

    def predict_keyword_only(self, request: MMRequest) -> str:
        """Classify using only prompt keywords.

        This is the original deterministic fallback and intentionally ignores
        dataset/source labels.
        """
        text = request.prompt.lower()
        if any(word in text for word in ["read", "text", "sign", "receipt", "word", "ocr"]):
            return "ocr"
        if any(word in text for word in ["why", "explain", "solve", "reason", "calculate"]):
            return "reasoning"
        if any(word in text for word in ["describe", "caption", "detail", "summarize"]):
            return "descriptive"
        return "brief"

    def predict_dataset_keyword(self, request: MMRequest) -> str:
        """Classify with prompt keywords plus a dataset identity prior.

        OCR/reasoning keywords override dataset priors because they often signal
        the actual task more directly than a broad dataset label. The dataset
        prior fills in prompts that are too short or generic for keyword-only
        classification.
        """
        keyword_category = self.predict_keyword_only(request)
        if keyword_category in {"ocr", "reasoning"}:
            return keyword_category

        dataset_category = self._dataset_prior(request.dataset)
        if dataset_category is not None:
            return dataset_category

        return keyword_category

    def _dataset_prior(self, dataset: str | None) -> str | None:
        """Map common benchmark dataset identities to coarse task categories."""
        if not dataset:
            return None

        normalized = dataset.lower().replace("_", "").replace("-", "")
        if "textvqa" in normalized or "ocr" in normalized:
            return "ocr"
        if "coco" in normalized or "caption" in normalized:
            return "descriptive"
        if "mmmu" in normalized or "math" in normalized or "science" in normalized:
            return "reasoning"
        return None


@dataclass
class OutputLengthProfile:
    """Observed output-length distributions keyed by predicted category.

    The profile is intentionally simple and JSON-serializable. Each category
    stores a sorted list of observed token lengths, so percentile lookup is
    deterministic and updates are cheap enough for the small experiment sizes in
    this project.
    """

    samples: dict[str, list[int]] = field(default_factory=dict)
    min_samples: int = 5

    @classmethod
    def from_json(cls, path: str | Path, min_samples: int = 5) -> "OutputLengthProfile":
        profile_path = Path(path)
        if not profile_path.exists():
            return cls(min_samples=min_samples)
        with profile_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        raw_samples = data.get("samples", data)
        samples = {
            str(category): sorted(int(value) for value in values)
            for category, values in raw_samples.items()
        }
        return cls(samples=samples, min_samples=int(data.get("min_samples", min_samples)))

    def to_json(self, path: str | Path) -> None:
        profile_path = Path(path)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": 1,
                    "min_samples": self.min_samples,
                    "samples": self.samples,
                    "summary": self.summary(),
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    def observe(self, category: str | None, output_length: int | None) -> None:
        """Record one observed output length for a category."""
        if category is None or output_length is None:
            return
        values = self.samples.setdefault(category, [])
        insort(values, int(output_length))

    def percentile(self, category: str, fraction: float) -> int | None:
        """Return nearest-rank percentile for a category when enough data exists."""
        values = self.samples.get(category, [])
        if len(values) < self.min_samples:
            return None
        index = min(len(values) - 1, int(fraction * (len(values) - 1)))
        return values[index]

    def p90(self, category: str) -> int | None:
        return self.percentile(category, 0.90)

    def summary(self) -> dict[str, dict[str, int | None]]:
        output: dict[str, dict[str, int | None]] = {}
        for category, values in sorted(self.samples.items()):
            output[category] = {
                "count": len(values),
                "p50": self._percentile_from_values(values, 0.50),
                "p90": self._percentile_from_values(values, 0.90),
                "p99": self._percentile_from_values(values, 0.99),
            }
        return output

    def _percentile_from_values(self, values: list[int], fraction: float) -> int | None:
        if not values:
            return None
        index = min(len(values) - 1, int(fraction * (len(values) - 1)))
        return values[index]


@dataclass
class OutputLengthPredictor:
    """Two-stage output length estimator with QRF/model-file extension points."""

    classifier: OutputCategoryClassifier = field(default_factory=OutputCategoryClassifier)
    category_p90: dict[str, int] = field(default_factory=lambda: dict(CATEGORY_DEFAULT_P90))
    category_p5: dict[str, int] = field(
        default_factory=lambda: {
            category: values["p5"] for category, values in CATEGORY_LENGTH_PERCENTILES.items()
        }
    )
    category_p99: dict[str, int] = field(
        default_factory=lambda: {
            category: values["p99"] for category, values in CATEGORY_LENGTH_PERCENTILES.items()
        }
    )
    qrf_tables: dict[str, dict[str, float]] = field(default_factory=dict)
    length_profile: OutputLengthProfile | None = None
    length_profile_percentile: float = 0.90

    @classmethod
    def from_json(cls, path: str | Path) -> "OutputLengthPredictor":
        """Load heuristic/QRF-like per-category coefficients from JSON."""
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            category_p90={**CATEGORY_DEFAULT_P90, **data.get("category_p90", {})},
            category_p5={
                **{
                    category: values["p5"]
                    for category, values in CATEGORY_LENGTH_PERCENTILES.items()
                },
                **data.get("category_p5", {}),
            },
            category_p99={
                **{
                    category: values["p99"]
                    for category, values in CATEGORY_LENGTH_PERCENTILES.items()
                },
                **data.get("category_p99", {}),
            },
            qrf_tables=data.get("qrf_tables", {}),
        )

    def predict(self, request: MMRequest) -> int:
        category = self.classifier.predict(request)
        profile_value = (
            self.length_profile.percentile(category, self.length_profile_percentile)
            if self.length_profile
            else None
        )
        base = profile_value or self.category_p90.get(category, CATEGORY_DEFAULT_P90["descriptive"])
        lower_bound = 1 if profile_value is not None else self.category_p5.get(category, 1)
        upper_bound = (
            max(base, 1)
            if profile_value is not None
            else self.category_p99.get(category, max(base, 1))
        )
        entropy = request.features.image_entropy or 0.0
        edge_density = request.features.edge_density or 0.0
        visual_complexity_multiplier = 1.0 + math.log1p(entropy * edge_density)
        predicted = base * visual_complexity_multiplier
        value = int(round(min(max(predicted, lower_bound), upper_bound)))
        request.features.predicted_category = category
        request.features.predicted_output_length = value
        request.metadata["output_length_profile"] = {
            "used_profile": profile_value is not None,
            "profile_percentile": self.length_profile_percentile,
            "profile_value": profile_value,
            "profile_p90": self.length_profile.p90(category) if self.length_profile else None,
            "fallback_p90": self.category_p90.get(category),
        }
        return value

    def refine(self, request: MMRequest, generated_tokens: int, elapsed_seconds: float | None = None) -> int:
        """Update the output estimate during generation.

        This simple refinement keeps the estimate at least as large as observed
        output, and gently expands it when generation is still active.
        """
        old_estimate = request.features.predicted_output_length or self.predict(request)
        growth_margin = max(8, int(0.25 * old_estimate))
        refined = max(old_estimate, generated_tokens + growth_margin)
        if elapsed_seconds is not None and elapsed_seconds > 2.0:
            refined = max(refined, int(old_estimate * 1.1))
        request.features.predicted_output_length = refined
        request.metadata["output_length_refined"] = True
        return refined


@dataclass
class MultimodalRequestAnalyzer:
    """End-to-end analyzer that populates predicted request costs."""

    prefill_estimator: PrefillCostEstimator = field(default_factory=PrefillCostEstimator)
    output_predictor: OutputLengthPredictor = field(default_factory=OutputLengthPredictor)

    def analyze(self, request: MMRequest) -> MMRequest:
        prefill = self.prefill_estimator.estimate_request(request)
        output_length = self.output_predictor.predict(request)
        request.metadata["analyzer"] = {
            "prefill_formula": "alpha * visual_tokens + beta * text_tokens",
            "predicted_prefill_cost": prefill,
            "predicted_output_length": output_length,
            "predicted_category": request.features.predicted_category,
        }
        return request

    def refine_output_estimate(
        self,
        request: MMRequest,
        generated_tokens: int,
        elapsed_seconds: float | None = None,
    ) -> int:
        return self.output_predictor.refine(request, generated_tokens, elapsed_seconds)
