from __future__ import annotations

from pathlib import Path
from typing import Callable
from urllib import request

from .models import TaskRecord, VideoTask


def fetch_url(url: str) -> bytes:
    with request.urlopen(url, timeout=120) as response:
        return response.read()


def build_output_path(output_dir: Path, task: VideoTask) -> Path:
    return output_dir / f"{task.id}.mp4"


def download_video(
    output_dir: str | Path,
    task: VideoTask,
    record: TaskRecord,
    fetch: Callable[[str], bytes] = fetch_url,
) -> Path:
    if record.status != "SUCCEEDED":
        raise ValueError(f"task {record.local_id} is not successful: {record.status}")
    if not record.video_url:
        raise ValueError(f"task {record.local_id} has no video_url")

    output_path = build_output_path(Path(output_dir), task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(fetch(record.video_url))
    return output_path
