"""Shared dataclasses used across the serving skeleton.

These types define the internal contracts between intake, preprocessing,
queueing, scheduling, backend execution, and logging. Keeping the request and
batch shapes centralized helps prevent field-name drift across scripts.

Author: Yuqi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestTimings:
    """Lifecycle timestamps used to compute latency, queue wait, and TTFT."""

    arrival_time: float
    queue_enter_time: float | None = None
    dispatch_time: float | None = None
    first_token_time: float | None = None
    completion_time: float | None = None


@dataclass
class RequestFeatures:
    """Cheap request features and future estimator placeholders."""

    text_length: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    resolution_bucket: str | None = None
    num_images: int = 0
    image_entropy: float | None = None
    edge_density: float | None = None
    predicted_category: str | None = None
    predicted_prefill_cost: float | None = None
    predicted_output_length: int | None = None
    image_embedding: list[float] | None = None
    text_embedding: list[float] | None = None


@dataclass
class MMRequest:
    """Internal representation of one multimodal serving request."""

    request_id: str
    arrival_time: float
    prompt: str
    image_path: str | None = None
    dataset: str | None = None
    source: str | None = None
    features: RequestFeatures = field(default_factory=RequestFeatures)
    timings: RequestTimings | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_text: str | None = None
    output_token_count: int | None = None

    def __post_init__(self) -> None:
        if self.timings is None:
            self.timings = RequestTimings(arrival_time=self.arrival_time)


@dataclass
class Batch:
    """Group of requests selected for one backend execution step."""

    batch_id: str
    requests: list[MMRequest]
    created_time: float


@dataclass
class BackendResult:
    """Backend output normalized back into pipeline-friendly fields."""

    request_id: str
    generated_text: str
    output_token_count: int
    first_token_time: float | None = None
    completion_time: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
