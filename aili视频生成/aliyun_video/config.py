from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .models import PromptJob, VideoVariant


CREATE_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks"

REGION_HOSTS = {
    "cn-beijing": "https://dashscope.aliyuncs.com",
    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com",
    "us-east-1": "https://dashscope-us.aliyuncs.com",
}


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    region: str
    model: str
    rps: int
    poll_interval_seconds: int
    output_dir: Path
    variants: list[VideoVariant]
    jobs: list[PromptJob]
    create_url: str
    query_base_url: str
    workspace_id: str | None = None


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read .yaml configs. Install with: pip install -r requirements.txt") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def _endpoint_urls(region: str, workspace_id: str | None) -> tuple[str, str]:
    if region == "eu-central-1":
        if not workspace_id:
            raise ValueError("workspace_id is required for eu-central-1")
        base = f"https://{workspace_id}.eu-central-1.maas.aliyuncs.com"
    else:
        try:
            base = REGION_HOSTS[region]
        except KeyError as exc:
            supported = sorted([*REGION_HOSTS.keys(), "eu-central-1"])
            raise ValueError(f"region must be one of {supported}") from exc
    return f"{base}{CREATE_PATH}", f"{base}{TASK_PATH}"


def _load_variants(raw_variants: Sequence[Mapping[str, object]]) -> list[VideoVariant]:
    return [
        VideoVariant(resolution=str(item.get("resolution", "1080P")), ratio=str(item.get("ratio", "16:9")))
        for item in raw_variants
    ]


def _load_jobs(raw_jobs: Sequence[Mapping[str, object]]) -> list[PromptJob]:
    jobs: list[PromptJob] = []
    for item in raw_jobs:
        seed_value = item.get("seed")
        seed = int(seed_value) if seed_value is not None else None
        jobs.append(
            PromptJob(
                id=str(item.get("id", "")),
                prompt=str(item.get("prompt", "")),
                duration=int(item.get("duration", 5)),
                watermark=bool(item.get("watermark", True)),
                seed=seed,
            )
        )
    return jobs


def load_config(path: str | Path, env: Mapping[str, str] | None = None) -> AppConfig:
    config_path = Path(path)
    data = _load_yaml(config_path)
    env_map = os.environ if env is None else env

    api_key = str(data.get("api_key") or env_map.get("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("api_key is required; set it in config or DASHSCOPE_API_KEY")

    region = str(data.get("region", "cn-beijing"))
    workspace_id = data.get("workspace_id")
    workspace_id_text = str(workspace_id).strip() if workspace_id is not None else None
    create_url, query_base_url = _endpoint_urls(region, workspace_id_text)

    rps = int(data.get("rps", 20))
    if rps < 1 or rps > 20:
        raise ValueError("rps must be between 1 and 20 for the default HappyHorse query limit")

    variants = _load_variants(data.get("variants", []))
    jobs = _load_jobs(data.get("jobs", []))

    return AppConfig(
        api_key=api_key,
        region=region,
        model=str(data.get("model", "happyhorse-1.0-t2v")),
        rps=rps,
        poll_interval_seconds=int(data.get("poll_interval_seconds", 15)),
        output_dir=Path(str(data.get("output_dir", "outputs"))),
        variants=variants,
        jobs=jobs,
        create_url=create_url,
        query_base_url=query_base_url,
        workspace_id=workspace_id_text,
    )
