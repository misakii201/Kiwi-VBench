from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aliyun_video.client import HappyHorseClient
from aliyun_video.config import load_config
from aliyun_video.downloader import download_video
from aliyun_video.models import TaskRecord, expand_jobs
from aliyun_video.scheduler import RateLimiter, TaskRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate basketball videos with Aliyun HappyHorse.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded tasks without calling Aliyun APIs.")
    parser.add_argument("--max-wait-seconds", type=int, default=3600, help="Maximum polling time before returning.")
    return parser.parse_args()


def record_to_json(record: TaskRecord) -> str:
    return json.dumps(record.__dict__, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    tasks = expand_jobs(config.jobs, config.variants)

    if args.dry_run:
        for task in tasks:
            print(
                json.dumps(
                    {
                        "id": task.id,
                        "prompt": task.prompt,
                        "resolution": task.variant.resolution,
                        "ratio": task.variant.ratio,
                        "duration": task.duration,
                        "watermark": task.watermark,
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    client = HappyHorseClient(
        api_key=config.api_key,
        model=config.model,
        create_url=config.create_url,
        query_base_url=config.query_base_url,
    )
    runner = TaskRunner(
        client=client,
        limiter=RateLimiter(config.rps),
        poll_interval_seconds=config.poll_interval_seconds,
    )
    submitted = runner.submit_tasks(tasks)
    results = runner.poll_until_complete(submitted, max_wait_seconds=args.max_wait_seconds)

    task_by_id = {task.id: task for task in tasks}
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    with results_path.open("a", encoding="utf-8") as handle:
        for record in results:
            print(record_to_json(record), file=handle)
            if record.status == "SUCCEEDED":
                path = download_video(output_dir, task_by_id[record.local_id], record)
                print(f"downloaded {record.local_id}: {path}")
            else:
                print(f"{record.local_id} ended with {record.status}: {record.message or record.code or ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
