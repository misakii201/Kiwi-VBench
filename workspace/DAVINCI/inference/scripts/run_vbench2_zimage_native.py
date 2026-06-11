#!/usr/bin/env python3
"""Build VBench-2.0 tasks and run zimage native generation (480x800, 4s).

Long-lived workers: one python process per GPU, model loaded once, no torchrun.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vbench2_zimage_lib import (  # noqa: E402
    ALL_DIMENSIONS,
    DEFAULT_PROMPTS_AUG_DIR,
    DEFAULT_PROMPTS_DIR,
    REPO,
    build_vbench_tasks,
    filter_active_tasks,
    write_task_meta,
    write_tasks_jsonl,
)


def _parse_gpus(val: str | None) -> list[int]:
    if not val or not str(val).strip():
        return list(range(8))
    s = str(val).strip().lower()
    if s == "auto":
        try:
            import torch

            n = torch.cuda.device_count() if torch.cuda.is_available() else 1
            return list(range(n))
        except Exception:
            return [0]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_dimensions(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return list(ALL_DIMENSIONS)
    dims = [d.strip() for d in raw.split(",") if d.strip()]
    bad = [d for d in dims if d not in ALL_DIMENSIONS]
    if bad:
        raise SystemExit(f"invalid dimensions: {bad}")
    return dims


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="zimage ckpt_best -> VBench-2.0 standard mp4 tree (native 480x800, 4s)",
    )
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--outdir",
        default=str(REPO / "output_vbench_t2i_1w_legacy_ckptbest"),
    )
    parser.add_argument("--prompts-dir", default=str(DEFAULT_PROMPTS_DIR))
    parser.add_argument("--prompts-aug-dir", default=str(DEFAULT_PROMPTS_AUG_DIR))
    parser.add_argument("--no-aug", action="store_true", help="use raw prompts only")
    parser.add_argument("--dimensions", default="", help="comma-separated; empty=all 18")
    parser.add_argument("--limit-prompts", type=int, default=0, help="per-dimension cap, 0=all")
    parser.add_argument("--build-only", action="store_true", help="only write task jsonl")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--per-gpu-workers", type=int, default=1)
    parser.add_argument(
        "--base",
        default="/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base",
    )
    parser.add_argument(
        "--config",
        default=str(REPO / "example" / "base" / "config_infer.json"),
    )
    parser.add_argument(
        "--ckpt",
        default=str(REPO / "output_train_t2i_1w_480x800_v1" / "ckpt_best.pt"),
    )
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--br-width", type=int, default=480)
    parser.add_argument("--br-height", type=int, default=800)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prompts_dir = Path(args.prompts_dir)
    prompts_aug_dir = None if args.no_aug else Path(args.prompts_aug_dir)
    dimensions = _parse_dimensions(args.dimensions)

    all_tasks = build_vbench_tasks(
        prompts_dir=prompts_dir,
        prompts_aug_dir=prompts_aug_dir,
        dimensions=dimensions,
        limit_prompts=int(args.limit_prompts),
        use_aug=not args.no_aug,
    )
    active = filter_active_tasks(all_tasks, outdir, force=args.force)

    write_tasks_jsonl(outdir / "vbench2_tasks_all.jsonl", all_tasks)
    write_tasks_jsonl(outdir / "vbench2_tasks_active.jsonl", active)
    write_task_meta(outdir, all_tasks=all_tasks, active=active)

    logging.info(
        "tasks: all_unique=%d active=%d outdir=%s",
        len(all_tasks),
        len(active),
        outdir,
    )

    if args.build_only:
        return

    if not active:
        logging.info("所有 VBench 视频已就绪，无需生成。")
        return

    gpu_list = _parse_gpus(args.gpus)
    per_gpu = max(1, int(args.per_gpu_workers))
    num_workers = len(gpu_list) * per_gpu
    buckets: list[list] = [[] for _ in range(num_workers)]
    for i, task in enumerate(active):
        buckets[i % num_workers].append(task)

    worker_py = Path(__file__).resolve().parent / "run_vbench2_zimage_native_worker.py"
    procs: list[subprocess.Popen] = []
    wid = 0
    for gpu in gpu_list:
        for slot in range(per_gpu):
            tasks = buckets[wid]
            if not tasks:
                wid += 1
                continue
            shard = outdir / f"vbench2_tasks.gpu{gpu}.w{slot}.jsonl"
            write_tasks_jsonl(shard, tasks)
            progress = outdir / f"progress_gpu{gpu}_w{slot}.json"
            master_port = 29500 + int(gpu) * 10 + int(slot)
            cmd = [
                sys.executable,
                "-u",
                str(worker_py),
                "--repo",
                str(args.repo),
                "--base",
                str(args.base),
                "--config",
                str(args.config),
                "--ckpt",
                str(args.ckpt),
                "--outdir",
                str(outdir),
                "--tasks-file",
                str(shard),
                "--progress",
                str(progress),
                "--gpu",
                str(gpu),
                "--seed",
                str(int(args.seed)),
                "--steps",
                str(int(args.steps)),
                "--cfg",
                str(float(args.cfg)),
                "--seconds",
                str(float(args.seconds)),
                "--br-width",
                str(int(args.br_width)),
                "--br-height",
                str(int(args.br_height)),
            ]
            if args.force:
                cmd.append("--force")
            env = os.environ.copy()
            env["MASTER_PORT"] = str(master_port)
            env["PYTHONPATH"] = f"{args.repo}:{env.get('PYTHONPATH', '')}"
            logging.info(
                "start native worker gpu=%s slot=%s tasks=%d port=%s",
                gpu,
                slot,
                len(tasks),
                master_port,
            )
            procs.append(
                subprocess.Popen(
                    cmd,
                    cwd=str(args.repo),
                    env=env,
                )
            )
            wid += 1

    failed = 0
    for p in procs:
        rc = p.wait()
        if rc != 0:
            failed += 1
            logging.error("worker exit rc=%s", rc)

    if failed:
        raise SystemExit(f"{failed} worker(s) failed")

    logging.info("全部 native worker 完成: %s", outdir)


if __name__ == "__main__":
    main()
