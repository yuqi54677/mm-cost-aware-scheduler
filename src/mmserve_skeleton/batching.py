"""Batch construction boundary.

The queue and scheduler decide which requests should run next. This module turns
those selected requests into a Batch object. Later scheduling research can add
smarter batch shaping here, such as grouping by resolution or prompt length.

Authors:
"""

from __future__ import annotations

import time
import uuid

from mmserve_skeleton.models import Batch, MMRequest


class BatchBuilder:
    """Build backend-ready batches from scheduler-selected requests."""

    def build(self, requests: list[MMRequest]) -> Batch:
        """Create a Batch with a unique batch ID and creation timestamp."""
        return Batch(
            batch_id=str(uuid.uuid4()),
            requests=requests,
            created_time=time.time(),
        )

    def to_backend_payload(self, batch: Batch) -> list[dict[str, str | None]]:
        """Return a simple serializable view of a batch for backend adapters."""
        return [
            {
                "request_id": request.request_id,
                "prompt": request.prompt,
                "image_path": request.image_path,
            }
            for request in batch.requests
        ]
