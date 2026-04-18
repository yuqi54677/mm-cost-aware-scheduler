"""Queue manager for requests waiting to be scheduled.

The queue is deliberately separate from the scheduler. It stores all waiting
requests and exposes snapshots so different scheduling policies can inspect the
same queue state without owning storage.

Author: Yuqi
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable

from mmserve_skeleton.models import MMRequest


class RequestQueue:
    """FIFO storage for MMRequest objects waiting for backend dispatch."""

    def __init__(self) -> None:
        self._items: deque[MMRequest] = deque()

    def add(self, request: MMRequest) -> None:
        """Add one request and record when it entered the queue."""
        request.timings.queue_enter_time = time.time()
        self._items.append(request)

    def add_many(self, requests: Iterable[MMRequest]) -> None:
        """Add several requests using the same queue-entry behavior."""
        for request in requests:
            self.add(request)

    def snapshot(self) -> list[MMRequest]:
        """Return the current waiting requests without removing them."""
        return list(self._items)

    def pop_request_ids(self, request_ids: set[str]) -> list[MMRequest]:
        """Remove and return requests whose IDs were selected by the scheduler."""
        selected: list[MMRequest] = []
        remaining: deque[MMRequest] = deque()
        while self._items:
            request = self._items.popleft()
            if request.request_id in request_ids:
                selected.append(request)
            else:
                remaining.append(request)
        self._items = remaining
        return selected

    def __len__(self) -> int:
        """Return the number of waiting requests."""
        return len(self._items)
