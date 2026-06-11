from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

from .client import HappyHorseClient
from .models import TaskRecord, VideoTask


TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}


class RateLimiter:
    def __init__(
        self,
        rps: int = 20,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rps < 1:
            raise ValueError("rps must be positive")
        self.rps = rps
        self.monotonic = monotonic
        self.sleep = sleep
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = self.monotonic()
        while self._calls and now - self._calls[0] >= 1.0:
            self._calls.popleft()
        if len(self._calls) >= self.rps:
            wait = 1.0 - (now - self._calls[0])
            if wait > 0:
                self.sleep(wait)
                now = self.monotonic()
            while self._calls and now - self._calls[0] >= 1.0:
                self._calls.popleft()
        self._calls.append(now)


@dataclass
class TaskRunner:
    client: HappyHorseClient
    limiter: RateLimiter
    poll_interval_seconds: int = 15
    sleep: Callable[[float], None] = time.sleep

    def submit_tasks(self, tasks: Iterable[VideoTask]) -> list[TaskRecord]:
        return [self.client.create_task(task) for task in tasks]

    def poll_until_complete(self, records: Iterable[TaskRecord], max_wait_seconds: int = 3600) -> list[TaskRecord]:
        pending = {record.task_id: record for record in records}
        final: list[TaskRecord] = []
        start = time.monotonic()

        while pending:
            if time.monotonic() - start > max_wait_seconds:
                final.extend(pending.values())
                break

            for task_id, previous in list(pending.items()):
                self.limiter.acquire()
                current = self.client.query_task(previous.local_id, task_id)
                if current.status in TERMINAL_STATUSES:
                    final.append(current)
                    del pending[task_id]
                else:
                    pending[task_id] = current

            if pending:
                self.sleep(self.poll_interval_seconds)

        return final
