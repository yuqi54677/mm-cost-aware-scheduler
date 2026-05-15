"""End-to-end orchestration for the serving skeleton.

ServingPipeline connects the major components:
request creation -> metadata extraction -> queue -> scheduler -> batch builder
-> backend -> log writer. The implementation is intentionally simple, but each
stage is explicit so smarter scheduling and batching logic can replace the
naive pieces later.

Author: Yuqi
"""

from __future__ import annotations

import time
import uuid

from .backend import Backend
from .batching import BatchBuilder
from .logging import JSONLLogWriter
from .models import MMRequest
from .preprocessing import MetadataExtractor
from .queue import RequestQueue
from .scheduler import FIFOScheduler


class ServingPipeline:
    """Coordinate request submission, dispatch, backend execution, and logging."""

    def __init__(
        self,
        backend: Backend,
        log_writer: JSONLLogWriter,
        scheduler: FIFOScheduler | None = None,
        queue: RequestQueue | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        batch_builder: BatchBuilder | None = None,
    ) -> None:
        """Wire pipeline components, using simple defaults for optional pieces."""
        self.backend = backend
        self.log_writer = log_writer
        self.scheduler = scheduler or FIFOScheduler()
        self.queue = queue or RequestQueue()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.batch_builder = batch_builder or BatchBuilder()

    def submit(
        self,
        prompt: str,
        image_path: str | None = None,
        dataset: str | None = None,
        source: str | None = None,
        metadata: dict | None = None,
        request_id: str | None = None,
        arrival_time: float | None = None,
    ) -> MMRequest:
        """Create an MMRequest, extract cheap metadata, and enqueue it."""
        request = MMRequest(
            request_id=request_id or str(uuid.uuid4()),
            arrival_time=time.time() if arrival_time is None else arrival_time,
            prompt=prompt,
            image_path=image_path,
            dataset=dataset,
            source=source,
            metadata=metadata or {},
        )
        self.metadata_extractor.enrich(request)
        self.queue.add(request)
        return request

    def run_once(self) -> list[MMRequest]:
        """Run one scheduling cycle: select requests, execute one batch, log results."""
        waiting = self.queue.snapshot()
        if not waiting:
            return []

        decision = self.scheduler.select(waiting)
        selected = self.queue.pop_request_ids(set(decision.request_ids))
        if not selected:
            return []

        batch = self.batch_builder.build(selected)
        dispatch_time = time.time()
        for request in batch.requests:
            request.timings.dispatch_time = dispatch_time
            request.metadata["scheduler_policy"] = decision.policy_name
            request.metadata["scheduler_decision"] = decision.metadata or {}
            request.metadata["batch_id"] = batch.batch_id

        results = self.backend.run_batch(batch)
        by_request_id = {result.request_id: result for result in results}
        for request in batch.requests:
            result = by_request_id[request.request_id]
            request.generated_text = result.generated_text
            request.output_token_count = result.output_token_count
            request.timings.first_token_time = result.first_token_time
            request.timings.completion_time = result.completion_time or time.time()
            request.metadata["backend_raw"] = result.raw
            self.log_writer.write(request)

        return batch.requests

    def drain(self) -> list[MMRequest]:
        """Run scheduling cycles until the queue is empty."""
        completed: list[MMRequest] = []
        while len(self.queue) > 0:
            completed.extend(self.run_once())
        return completed
