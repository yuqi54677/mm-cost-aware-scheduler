"""Replaceable scheduling policies.

Schedulers inspect the waiting queue and return request IDs to dispatch. The
current policy is FIFO, but this file is where heterogeneous-workload scheduling
policies can be added later.

Authors: Yuqi, 
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .models import MMRequest


@dataclass
class ScheduleDecision:
    """Scheduler output: selected request IDs plus the policy name."""

    request_ids: list[str]
    policy_name: str
    metadata: dict | None = None


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
            metadata={"candidate_count": len(waiting_requests)},
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
            metadata={"candidate_count": len(waiting_requests)},
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

    def __init__(
        self,
        max_batch_size: int = 4,
        window_size: int | None = None,
        tail_slo_seconds: float | None = None,
    ) -> None:
        super().__init__(max_batch_size=max_batch_size)
        self.window_size = window_size or max_batch_size
        self.tail_slo_seconds = tail_slo_seconds

    def select(self, waiting_requests: list[MMRequest]) -> ScheduleDecision:
        if not waiting_requests:
            return ScheduleDecision(
                request_ids=[],
                policy_name=self.policy_name,
                metadata={"candidate_count": 0},
            )

        now = time.time()
        aged_requests = self._aged_requests(waiting_requests, now)
        protected = aged_requests[: self.max_batch_size]
        remaining_slots = self.max_batch_size - len(protected)
        remaining_candidates = [
            request for request in waiting_requests if request.request_id not in {r.request_id for r in protected}
        ]

        sorted_candidates = sorted(waiting_requests, key=self._composite_key)
        window_size = max(self.max_batch_size, self.window_size)
        window_size = min(window_size, len(sorted_candidates))

        best_start = 0
        best_window = sorted_candidates[:window_size]
        best_score = self._window_score(best_window)
        for start in range(1, len(sorted_candidates) - window_size + 1):
            window = sorted_candidates[start : start + window_size]
            score = self._window_score(window)
            if score < best_score:
                best_start = start
                best_window = window
                best_score = score

        selected_from_window: list[MMRequest] = []
        if remaining_slots > 0:
            window_remaining = [
                request
                for request in best_window
                if request.request_id not in {r.request_id for r in protected}
            ]
            if len(window_remaining) < remaining_slots:
                protected_ids = {request.request_id for request in protected}
                extra = [
                    request
                    for request in sorted(remaining_candidates, key=self._dynamic_priority_key)
                    if request.request_id not in protected_ids
                ]
                window_remaining.extend(extra)
            selected_from_window = sorted(window_remaining, key=self._dynamic_priority_key)[:remaining_slots]

        selected = protected + selected_from_window
        selected_costs = [self._estimated_total_cost(request) for request in selected]
        return ScheduleDecision(
            request_ids=[request.request_id for request in selected],
            policy_name=self.policy_name,
            metadata={
                "candidate_count": len(waiting_requests),
                "configured_window_size": self.window_size,
                "effective_window_size": window_size,
                "selected_window_start": best_start,
                "selected_window_score": best_score,
                "tail_slo_seconds": self.tail_slo_seconds,
                "aged_candidate_count": len(aged_requests),
                "protected_aged_count": len(protected),
                "selected_cost_min": min(selected_costs) if selected_costs else None,
                "selected_cost_max": max(selected_costs) if selected_costs else None,
                "selected_resolution_buckets": [
                    request.features.resolution_bucket for request in selected
                ],
                "selected_text_lengths": [request.features.text_length for request in selected],
            },
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
        costs = [self._estimated_total_cost(request) for request in window]
        if not costs:
            return 0.0
        return max(costs) - min(costs)

    def _estimated_total_cost(self, request: MMRequest) -> float:
        return (
            (request.features.predicted_prefill_cost or 0.0)
            + 0.01 * (request.features.predicted_output_length or 0)
        )

    def _aged_requests(self, waiting_requests: list[MMRequest], now: float) -> list[MMRequest]:
        """Return requests that exceeded the tail SLO, oldest first."""
        if self.tail_slo_seconds is None:
            return []
        aged: list[MMRequest] = []
        for request in waiting_requests:
            queue_enter_time = request.timings.queue_enter_time
            if queue_enter_time is not None and now - queue_enter_time >= self.tail_slo_seconds:
                aged.append(request)
        return sorted(aged, key=lambda request: request.timings.queue_enter_time or request.arrival_time)
