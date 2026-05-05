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
