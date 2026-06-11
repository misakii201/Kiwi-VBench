from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import VideoTask, VideoVariant
from .variants import VariantSpec, build_variant_matrix


RESULT_FIELDS = [
    "prompt_id",
    "prompt",
    "status",
    "aliyun_task_id",
    "master_video",
    "variant_video",
    "ratio",
    "quality",
    "width",
    "height",
    "duration",
    "seed",
    "error",
]


@dataclass(frozen=True)
class BatchPrompt:
    id: str
    prompt: str
    duration: int = 5
    watermark: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class MasterResult:
    task_id: str
    status: str
    master_path: Path | None = None
    error: str | None = None


MasterGenerator = Callable[[BatchPrompt, VideoTask], MasterResult]
VariantDeriver = Callable[[Path, Path, VariantSpec], None]


def _required_text(row: dict[str, str], field: str, row_number: int) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{field} is required on CSV row {row_number}")
    return value


def _optional_int(row: dict[str, str], field: str, row_number: int) -> int | None:
    value = (row.get(field) or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer on CSV row {row_number}") from exc


def _optional_bool(row: dict[str, str], field: str, default: bool) -> bool:
    value = (row.get(field) or "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{field} must be true or false")


def load_prompt_csv(path: str | Path) -> list[BatchPrompt]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV header is required")
        missing = {"id", "prompt"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        prompts: list[BatchPrompt] = []
        for index, row in enumerate(reader, start=2):
            duration = _optional_int(row, "duration", index)
            prompts.append(
                BatchPrompt(
                    id=_required_text(row, "id", index),
                    prompt=_required_text(row, "prompt", index),
                    duration=duration if duration is not None else 5,
                    watermark=_optional_bool(row, "watermark", False),
                    seed=_optional_int(row, "seed", index),
                )
            )
    return prompts


def build_master_task(prompt: BatchPrompt) -> VideoTask:
    variant = VideoVariant(resolution="1080P", ratio="1:1")
    return VideoTask(
        id=f"{prompt.id}_{variant.slug}",
        job_id=prompt.id,
        prompt=prompt.prompt,
        variant=variant,
        duration=prompt.duration,
        watermark=prompt.watermark,
        seed=prompt.seed,
    )


def variant_output_path(output_dir: Path, prompt: BatchPrompt, variant: VariantSpec) -> Path:
    return output_dir / "variants" / f"{prompt.id}_{variant.ratio_slug}_{variant.quality}.mp4"


def _empty_result(prompt: BatchPrompt, master: MasterResult, status: str, error: str | None = None) -> dict[str, str]:
    return {
        "prompt_id": prompt.id,
        "prompt": prompt.prompt,
        "status": status,
        "aliyun_task_id": master.task_id,
        "master_video": str(master.master_path or ""),
        "variant_video": "",
        "ratio": "",
        "quality": "",
        "width": "",
        "height": "",
        "duration": str(prompt.duration),
        "seed": "" if prompt.seed is None else str(prompt.seed),
        "error": error or master.error or "",
    }


def _success_result(prompt: BatchPrompt, master: MasterResult, output_path: Path, variant: VariantSpec) -> dict[str, str]:
    return {
        "prompt_id": prompt.id,
        "prompt": prompt.prompt,
        "status": "SUCCEEDED",
        "aliyun_task_id": master.task_id,
        "master_video": str(master.master_path or ""),
        "variant_video": str(output_path),
        "ratio": variant.ratio_slug,
        "quality": variant.quality,
        "width": str(variant.target_w),
        "height": str(variant.target_h),
        "duration": str(prompt.duration),
        "seed": "" if prompt.seed is None else str(prompt.seed),
        "error": "",
    }


def run_batch_variant_workflow(
    prompts: list[BatchPrompt],
    output_dir: str | Path,
    generate_master: MasterGenerator,
    derive_variant: VariantDeriver,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.csv"
    api_tasks_path = root / "api_tasks.jsonl"
    matrix = build_variant_matrix()

    with results_path.open("w", encoding="utf-8", newline="") as results_handle, api_tasks_path.open(
        "w", encoding="utf-8"
    ) as api_handle:
        writer = csv.DictWriter(results_handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for prompt in prompts:
            task = build_master_task(prompt)
            try:
                master = generate_master(prompt, task)
            except Exception as exc:
                master = MasterResult(task_id="", status="FAILED", error=str(exc))
                print(
                    json.dumps(
                        {"prompt_id": prompt.id, "task_id": master.task_id, "status": master.status, "error": master.error},
                        ensure_ascii=False,
                    ),
                    file=api_handle,
                )
                writer.writerow(_empty_result(prompt, master, "FAILED", str(exc)))
                continue
            print(
                json.dumps(
                    {
                        "prompt_id": prompt.id,
                        "task_id": master.task_id,
                        "status": master.status,
                        "master_video": str(master.master_path or ""),
                        "error": master.error or "",
                    },
                    ensure_ascii=False,
                ),
                file=api_handle,
            )

            if master.status != "SUCCEEDED" or master.master_path is None:
                writer.writerow(_empty_result(prompt, master, "FAILED"))
                continue

            for variant in matrix:
                output_path = variant_output_path(root, prompt, variant)
                try:
                    derive_variant(master.master_path, output_path, variant)
                except Exception as exc:
                    failed = _success_result(prompt, master, output_path, variant)
                    failed["status"] = "FAILED"
                    failed["error"] = str(exc)
                    writer.writerow(failed)
                    continue
                writer.writerow(_success_result(prompt, master, output_path, variant))

    return results_path


def write_sample_18_rows(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    prompt = BatchPrompt(
        id="sample_001",
        prompt="sample centered square master prompt for crop testing",
        duration=5,
        watermark=False,
        seed=123,
    )
    master_path = root / "masters" / "sample_001_1080P_1x1.mp4"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(b"sample master placeholder")

    def generate_master(batch_prompt: BatchPrompt, task: VideoTask) -> MasterResult:
        return MasterResult(task_id="sample-task-001", status="SUCCEEDED", master_path=master_path)

    def derive_sample(master: Path, output_path: Path, variant: VariantSpec) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"sample variant placeholder")

    return run_batch_variant_workflow([prompt], root, generate_master, derive_sample)
