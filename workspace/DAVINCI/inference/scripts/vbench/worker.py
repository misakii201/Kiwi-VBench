#!/usr/bin/env python3
"""Single-GPU long-lived worker: load zimage ckpt once, emit VBench-2.0 mp4s."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib import (  # noqa: E402
    REPO,
    copy_to_other_dims,
    publish_infer_mp4_to_vbench,
    read_tasks_jsonl,
    task_is_complete,
)

NEG_PROMPT = (
    "static, blurred details, subtitles, overall gray, worst quality, low quality, "
    "jpeg artifacts, ugly, deformed, disfigured, messy background"
)


def _load_progress(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"done": 0, "skipped": 0, "failed": 0, "total": 0, "items": []}


def _save_progress(path: Path, prog: dict) -> None:
    prog["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--base",
        default="/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base",
    )
    parser.add_argument(
        "--ckpt",
        default=str(REPO / "output_train_t2i_1w_480x800_v1" / "ckpt_best.pt"),
    )
    parser.add_argument(
        "--config",
        default=str(REPO / "example" / "base" / "config_infer.json"),
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--progress", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--video-cfg", type=float, default=None)
    parser.add_argument("--audio-cfg", type=float, default=None)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--br-width", type=int, default=480)
    parser.add_argument("--br-height", type=int, default=800)
    parser.add_argument("--min-mp4-bytes", type=int, default=50_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    master_port = os.environ.get("MASTER_PORT", "").strip() or str(29500 + int(args.gpu) * 10)
    os.environ["MASTER_PORT"] = master_port
    os.environ.setdefault("MAGI_NEGATIVE_PROMPT", NEG_PROMPT)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("DAVINCI_DISABLE_MAGI_COMPILE", "1")
    os.environ.setdefault("MAGI_DISABLE_COMPILE", "1")

    repo = Path(args.repo)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks_jsonl(Path(args.tasks_file))
    progress_path = Path(args.progress or (outdir / f"progress_gpu{args.gpu}.json"))

    br_w, br_h = int(args.br_width), int(args.br_height)
    seconds = float(args.seconds)
    min_bytes = int(args.min_mp4_bytes)

    prog = _load_progress(progress_path)
    prog["total"] = len(tasks)
    prog["outdir"] = str(outdir)
    prog["ckpt"] = str(args.ckpt)
    prog["tasks_file"] = str(args.tasks_file)
    prog["gpu"] = int(args.gpu)
    if not prog.get("started_at"):
        prog["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if not args.force:
        pre_done = sum(1 for t in tasks if task_is_complete(t, outdir, min_bytes=min_bytes))
        prog["done"] = pre_done
        prog["skipped"] = pre_done
        prog["failed"] = 0
    _save_progress(progress_path, prog)
    failed_this_run = 0

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from inference.test_infer_zimage_native import create_zimage_native_pipeline  # noqa: E402

    print(
        f"[worker] gpu={args.gpu} tasks={len(tasks)} outdir={outdir} "
        f"port={master_port} loading model ...",
        flush=True,
    )
    t_load = time.time()
    video_cfg = float(args.video_cfg) if args.video_cfg is not None else float(args.cfg)
    audio_cfg = float(args.audio_cfg) if args.audio_cfg is not None else float(args.cfg)
    pipeline = create_zimage_native_pipeline(
        config_load_path=str(args.config),
        base_ckpt_dir=str(args.base),
        ckpt_dir=str(args.ckpt),
        ckpt_blend_alpha=float(args.blend),
        device="cuda:0",
        amp_dtype="bf16",
        num_inference_steps=int(args.steps),
        video_cfg_scale=video_cfg,
        audio_cfg_scale=audio_cfg,
        br_width=br_w,
        br_height=br_h,
        freeze_audio=True,
    )
    print(f"[worker] model ready ({time.time() - t_load:.1f}s)", flush=True)

    for local_i, task in enumerate(tasks):
        item_key = f"{task.dimension}:{task.stem}"
        vbench = task.vbench_path(outdir)

        if not args.force and task_is_complete(task, outdir, min_bytes=min_bytes):
            print(f"[SKIP complete] {task.dimension}/{task.filename}", flush=True)
            continue

        if vbench.is_file() and vbench.stat().st_size >= min_bytes:
            copy_to_other_dims(task, outdir)
            print(f"[COPY reuse] {task.dimension}/{task.filename}", flush=True)
            prog["done"] = int(prog.get("done", 0)) + 1
            _save_progress(progress_path, prog)
            continue

        prefix = outdir / task.dimension / task.stem
        prefix.parent.mkdir(parents=True, exist_ok=True)
        task_seed = task.seed(int(args.seed))

        t0 = time.time()
        print(
            f"[RUN {local_i+1}/{len(tasks)}] {task.dimension}/{task.filename} seed={task_seed}",
            flush=True,
        )
        try:
            pipeline.run_offline(
                prompt=task.gen_prompt,
                image=None,
                audio=None,
                save_path_prefix=str(prefix),
                seed=task_seed,
                seconds=seconds,
                br_width=br_w,
                br_height=br_h,
            )
            publish_infer_mp4_to_vbench(
                task, outdir, br_w=br_w, br_h=br_h, seconds=seconds, min_bytes=min_bytes
            )
            copy_to_other_dims(task, outdir)
        except Exception as e:
            failed_this_run += 1
            elapsed = time.time() - t0
            prog["failed"] = int(prog.get("failed", 0)) + 1
            prog.setdefault("items", []).append(
                {
                    "key": item_key,
                    "status": "failed",
                    "error": str(e),
                    "elapsed_sec": round(elapsed, 1),
                }
            )
            _save_progress(progress_path, prog)
            print(f"[FAIL] {item_key}: {e}", flush=True)
            continue

        elapsed = time.time() - t0
        prog["done"] = int(prog.get("done", 0)) + 1
        prog.setdefault("items", []).append(
            {
                "key": item_key,
                "status": "ok",
                "elapsed_sec": round(elapsed, 1),
                "vbench": str(vbench),
            }
        )
        _save_progress(progress_path, prog)
        print(f"[OK] {item_key} ({elapsed:.1f}s) -> {vbench}", flush=True)

    print(
        f"DONE worker gpu={args.gpu}: done={prog.get('done')} failed_this_run={failed_this_run} "
        f"total={prog.get('total')}",
        flush=True,
    )
    sys.exit(1 if failed_this_run else 0)


if __name__ == "__main__":
    main()
