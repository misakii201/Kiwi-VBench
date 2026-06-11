from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aliyun_video.batch_variants import BatchPrompt, MasterResult, load_prompt_csv, run_batch_variant_workflow, variant_output_path
from aliyun_video.client import HappyHorseClient
from aliyun_video.config import _endpoint_urls
from aliyun_video.downloader import download_video
from aliyun_video.scheduler import RateLimiter, TaskRunner
from aliyun_video.variants import build_variant_matrix, derive_variant, probe_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 1:1 Aliyun masters and derive 9 ratios x 2 qualities.")
    parser.add_argument("--csv", required=True, help="Prompt CSV with id and prompt columns.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to outputs/batch_manual.")
    parser.add_argument("--api-key", default=None, help="DashScope API key. Defaults to DASHSCOPE_API_KEY.")
    parser.add_argument("--region", default="cn-beijing", help="Aliyun region.")
    parser.add_argument("--workspace-id", default=None, help="Workspace id for eu-central-1.")
    parser.add_argument("--model", default="happyhorse-1.0-t2v", help="Aliyun HappyHorse model.")
    parser.add_argument("--rps", type=int, default=20, help="Task query requests per second.")
    parser.add_argument("--poll-interval-seconds", type=int, default=15, help="Polling sleep interval.")
    parser.add_argument("--max-wait-seconds", type=int, default=3600, help="Maximum wait for each batch polling run.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned derived outputs without API or ffmpeg calls.")
    return parser.parse_args()


def dry_run(prompts: list[BatchPrompt], output_dir: Path) -> None:
    matrix = build_variant_matrix()
    for prompt in prompts:
        for variant in matrix:
            print(
                json.dumps(
                    {
                        "prompt_id": prompt.id,
                        "prompt": prompt.prompt,
                        "variant_video": str(variant_output_path(output_dir, prompt, variant)),
                        "ratio": variant.ratio_slug,
                        "quality": variant.quality,
                        "width": variant.target_w,
                        "height": variant.target_h,
                    },
                    ensure_ascii=False,
                )
            )


def main() -> int:
    args = parse_args()
    prompts = load_prompt_csv(args.csv)
    default_name = f"outputs/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir or default_name)

    if args.dry_run:
        dry_run(prompts, output_dir)
        return 0

    import os

    api_key = (args.api_key or os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        print("api key is required; pass --api-key or set DASHSCOPE_API_KEY", file=sys.stderr)
        return 2

    create_url, query_base_url = _endpoint_urls(args.region, args.workspace_id)
    client = HappyHorseClient(
        api_key=api_key,
        model=args.model,
        create_url=create_url,
        query_base_url=query_base_url,
    )
    runner = TaskRunner(
        client=client,
        limiter=RateLimiter(args.rps),
        poll_interval_seconds=args.poll_interval_seconds,
    )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    size_cache: dict[Path, tuple[int, int]] = {}

    def generate_master(prompt, task):
        submitted = runner.submit_tasks([task])
        final = runner.poll_until_complete(submitted, max_wait_seconds=args.max_wait_seconds)[0]
        if final.status != "SUCCEEDED":
            return MasterResult(
                task_id=final.task_id,
                status=final.status,
                error=final.message or final.code or final.status,
            )
        master_path = download_video(output_dir / "masters", task, final)
        return MasterResult(task_id=final.task_id, status="SUCCEEDED", master_path=master_path)

    def derive_from_master(master_path, output_path, variant):
        if master_path not in size_cache:
            size_cache[master_path] = probe_size(ffmpeg, master_path)
        derive_variant(ffmpeg, master_path, output_path, size_cache[master_path], variant)

    results_path = run_batch_variant_workflow(prompts, output_dir, generate_master, derive_from_master)
    print(results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
