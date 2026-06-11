from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


VALID_RESOLUTIONS = {"720P", "1080P"}
VALID_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4", "9:21", "21:9"}


def _clean_id(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def ratio_slug(ratio: str) -> str:
    return ratio.replace(":", "x")


@dataclass(frozen=True)
class VideoVariant:
    resolution: str
    ratio: str

    def __post_init__(self) -> None:
        if self.resolution not in VALID_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {sorted(VALID_RESOLUTIONS)}")
        if self.ratio not in VALID_RATIOS:
            raise ValueError(f"ratio must be one of {sorted(VALID_RATIOS)}")

    @property
    def slug(self) -> str:
        return f"{self.resolution}_{ratio_slug(self.ratio)}"


@dataclass(frozen=True)
class PromptJob:
    id: str
    prompt: str
    duration: int = 5
    watermark: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _clean_id(self.id, "id"))
        object.__setattr__(self, "prompt", _clean_id(self.prompt, "prompt"))
        if not 3 <= self.duration <= 15:
            raise ValueError("duration must be between 3 and 15 seconds")
        if self.seed is not None and not 0 <= self.seed <= 2147483647:
            raise ValueError("seed must be between 0 and 2147483647")


@dataclass(frozen=True)
class VideoTask:
    id: str
    job_id: str
    prompt: str
    variant: VideoVariant
    duration: int
    watermark: bool
    seed: int | None = None


@dataclass(frozen=True)
class TaskRecord:
    local_id: str
    task_id: str
    status: str
    request_id: str | None = None
    video_url: str | None = None
    code: str | None = None
    message: str | None = None


def expand_jobs(jobs: Iterable[PromptJob], variants: Iterable[VideoVariant]) -> list[VideoTask]:
    expanded: list[VideoTask] = []
    variant_list = list(variants)
    if not variant_list:
        raise ValueError("at least one variant is required")

    for job in jobs:
        for variant in variant_list:
            expanded.append(
                VideoTask(
                    id=f"{job.id}_{variant.slug}",
                    job_id=job.id,
                    prompt=job.prompt,
                    variant=variant,
                    duration=job.duration,
                    watermark=job.watermark,
                    seed=job.seed,
                )
            )
    return expanded
