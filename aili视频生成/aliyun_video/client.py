from __future__ import annotations

import json
from typing import Any, Callable
from urllib import request

from .models import TaskRecord, VideoTask


Transport = Callable[[str, str], dict[str, Any]]


def urllib_transport(method: str, url: str, headers: dict[str, str] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = json.dumps(body or {}).encode("utf-8") if body is not None else None
    req = request.Request(url=url, method=method, headers=headers or {}, data=payload)
    with request.urlopen(req, timeout=60) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


class HappyHorseClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        create_url: str,
        query_base_url: str,
        transport: Callable[..., dict[str, Any]] = urllib_transport,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.create_url = create_url
        self.query_base_url = query_base_url.rstrip("/")
        self.transport = transport

    def _headers(self, async_task: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if async_task:
            headers["X-DashScope-Async"] = "enable"
        return headers

    def create_task(self, task: VideoTask) -> TaskRecord:
        parameters: dict[str, Any] = {
            "resolution": task.variant.resolution,
            "ratio": task.variant.ratio,
            "duration": task.duration,
            "watermark": task.watermark,
        }
        if task.seed is not None:
            parameters["seed"] = task.seed

        body = {
            "model": self.model,
            "input": {"prompt": task.prompt},
            "parameters": parameters,
        }
        data = self.transport("POST", self.create_url, headers=self._headers(async_task=True), body=body)
        output = data.get("output") or {}
        task_id = str(output.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"create task response did not include task_id: {data}")
        return TaskRecord(
            local_id=task.id,
            task_id=task_id,
            status=str(output.get("task_status") or "PENDING"),
            request_id=data.get("request_id"),
        )

    def query_task(self, local_id: str, task_id: str) -> TaskRecord:
        data = self.transport("GET", f"{self.query_base_url}/{task_id}", headers=self._headers(), body=None)
        output = data.get("output") or {}
        return TaskRecord(
            local_id=local_id,
            task_id=task_id,
            status=str(output.get("task_status") or "UNKNOWN"),
            request_id=data.get("request_id"),
            video_url=output.get("video_url"),
            code=output.get("code") or data.get("code"),
            message=output.get("message") or data.get("message"),
        )
