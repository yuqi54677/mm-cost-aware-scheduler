"""Structured JSONL logging for completed requests.

Each completed MMRequest is written as one JSON object. The log includes request
features, timing fields, output length, generated text, and metadata. These logs
are the input to analysis scripts and later estimator-training workflows.

Author: ()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MMRequest


class JSONLLogWriter:
    """Append completed request traces to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, request: MMRequest) -> None:
        """Serialize one completed request to the configured log file."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._record(request), sort_keys=True) + "\n")

    def _record(self, request: MMRequest) -> dict[str, Any]:
        """Convert the request object into the stable JSONL log schema."""
        timings = request.timings
        latency = None
        queue_wait = None
        ttft = None
        if timings.completion_time is not None:
            latency = timings.completion_time - timings.arrival_time
        if timings.queue_enter_time is not None and timings.dispatch_time is not None:
            queue_wait = timings.dispatch_time - timings.queue_enter_time
        if timings.dispatch_time is not None and timings.first_token_time is not None:
            ttft = timings.first_token_time - timings.dispatch_time

        return {
            "request_id": request.request_id,
            "dataset": request.dataset,
            "source": request.source,
            "prompt_length": request.features.text_length,
            "image_width": request.features.image_width,
            "image_height": request.features.image_height,
            "resolution_bucket": request.features.resolution_bucket,
            "num_images": request.features.num_images,
            "arrival_time": timings.arrival_time,
            "queue_enter_time": timings.queue_enter_time,
            "dispatch_time": timings.dispatch_time,
            "first_token_time": timings.first_token_time,
            "completion_time": timings.completion_time,
            "latency_seconds": latency,
            "queue_wait_seconds": queue_wait,
            "ttft_seconds": ttft,
            "output_token_count": request.output_token_count,
            "generated_text": self._truncate(request.generated_text),
            "features": {
                "predicted_prefill_cost": request.features.predicted_prefill_cost,
                "predicted_output_length": request.features.predicted_output_length,
            },
            "metadata": request.metadata,
        }

    def _truncate(self, text: str | None, limit: int = 500) -> str | None:
        """Keep generated text logs compact while preserving useful context."""
        if text is None or len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"
