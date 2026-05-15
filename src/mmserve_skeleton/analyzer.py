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
    """FastText-style category classifier with a keyword fallback.

    If a FastText model path is provided, load it. Otherwise use a deterministic
    classifier so the pipeline remains runnable without training artifacts.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
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

        text = request.prompt.lower()
        if any(word in text for word in ["read", "text", "sign", "receipt", "word", "ocr"]):
            return "ocr"
        if any(word in text for word in ["why", "explain", "solve", "reason", "calculate"]):
            return "reasoning"
        if any(word in text for word in ["describe", "caption", "detail", "summarize"]):
            return "descriptive"
        return "brief"


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
        base = self.category_p90.get(category, CATEGORY_DEFAULT_P90["descriptive"])
        lower_bound = self.category_p5.get(category, 1)
        upper_bound = self.category_p99.get(category, max(base, 1))
        entropy = request.features.image_entropy or 0.0
        edge_density = request.features.edge_density or 0.0
        visual_complexity_multiplier = 1.0 + math.log1p(entropy * edge_density)
        predicted = base * visual_complexity_multiplier
        value = int(round(min(max(predicted, lower_bound), upper_bound)))
        request.features.predicted_category = category
        request.features.predicted_output_length = value
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
