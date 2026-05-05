"""Replaceable scheduling policies.

Schedulers inspect the waiting queue and return request IDs to dispatch. The
current policy is FIFO, but this file is where heterogeneous-workload scheduling
policies can be added later.

Authors: Yuqi, 
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import MMRequest


@dataclass
class ScheduleDecision:
    """Scheduler output: selected request IDs plus the policy name."""

    request_ids: list[str]
    policy_name: str


class FIFOScheduler:
    """Naive scheduler that selects the oldest waiting requests."""

    policy_name = "fifo"

    def __init__(self, max_batch_size: int = 1) -> None:
        self.max_batch_size = max_batch_size

    def select(self, waiting_requests: list[MMRequest]) -> ScheduleDecision:
        """Select the oldest waiting requests up to max_batch_size."""
        selected = waiting_requests[: self.max_batch_size]
        return ScheduleDecision(
            request_ids=[request.request_id for request in selected],
            policy_name=self.policy_name,
        )


class LengthOnlyScheduler(FIFOScheduler):
    """Baseline scheduler that sorts by text length only."""

    policy_name = "length_only"

    def select(self, waiting_requests: list[MMRequest]) -> ScheduleDecision:
        selected = sorted(
            waiting_requests,
            key=lambda request: request.features.text_length or 0,
        )[: self.max_batch_size]
        return ScheduleDecision(
            request_ids=[request.request_id for request in selected],
            policy_name=self.policy_name,
        )


class GMAXScheduler(FIFOScheduler):
    """Multimodal GMAX-style scheduler using resolution tier and text length.

    Candidates are sorted by a composite key:
        (resolution_tier, text_length)

    A sliding window of size B is scanned over the sorted queue. The chosen
    window is the one with the lowest spread in predicted prefill/output cost,
    which forms batches with more similar multimodal serving cost.
    """

    policy_name = "gmax"

    def __init__(self, max_batch_size: int = 4, window_size: int | None = None) -> None:
        super().__init__(max_batch_size=max_batch_size)
        self.window_size = window_size or max_batch_size

    def select(self, waiting_requests: list[MMRequest]) -> ScheduleDecision:
        if not waiting_requests:
            return ScheduleDecision(request_ids=[], policy_name=self.policy_name)

        sorted_candidates = sorted(waiting_requests, key=self._composite_key)
        window_size = max(self.max_batch_size, self.window_size)
        window_size = min(window_size, len(sorted_candidates))

        best_window = sorted_candidates[:window_size]
        best_score = self._window_score(best_window)
        for start in range(1, len(sorted_candidates) - window_size + 1):
            window = sorted_candidates[start : start + window_size]
            score = self._window_score(window)
            if score < best_score:
                best_window = window
                best_score = score

        selected = sorted(best_window, key=self._dynamic_priority_key)[: self.max_batch_size]
        return ScheduleDecision(
            request_ids=[request.request_id for request in selected],
            policy_name=self.policy_name,
        )

    def _composite_key(self, request: MMRequest) -> tuple[int, int, float]:
        resolution_tier = {
            "none": 0,
            "small": 1,
            "medium": 2,
            "large": 3,
        }.get(request.features.resolution_bucket or "none", 0)
        return (
            resolution_tier,
            request.features.text_length or 0,
            request.arrival_time,
        )

    def _dynamic_priority_key(self, request: MMRequest) -> tuple[float, float]:
        predicted_output = request.features.predicted_output_length or 0
        predicted_prefill = request.features.predicted_prefill_cost or 0.0
        return (predicted_prefill + 0.01 * predicted_output, request.arrival_time)

    def _window_score(self, window: list[MMRequest]) -> float:
        costs = [
            (request.features.predicted_prefill_cost or 0.0)
            + 0.01 * (request.features.predicted_output_length or 0)
            for request in window
        ]
        if not costs:
            return 0.0
        return max(costs) - min(costs)
