#!/usr/bin/env python3
"""Single-GPU long-lived worker: manifest-index sampling via native zimage pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NEG_PROMPT = (
    "static, blurred details, subtitles, overall gray, worst quality, low quality, "
    "jpeg artifacts, ugly, deformed, disfigured, messy background"
)


def _load_manifest(path: Path) -> list[dict]:
    items: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            items.append(json.loads(ln))
    return items


def _load_indices(args) -> list[int]:
    if args.indices_file:
        return [int(x.strip()) for x in Path(args.indices_file).read_text().split() if x.strip()]
    start = int(args.prompt_start)
    n = int(args.num_prompts)
    return list(range(start, start + n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--base", default="/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base")
    parser.add_argument("--ckpt", default=str(REPO / "output_train_t2i_1w_480x800_v1" / "ckpt_best.pt"))
    parser.add_argument("--config", default=str(REPO / "example" / "base" / "config_infer.json"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--progress", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--indices-file", default=None)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--br-width", type=int, default=480)
    parser.add_argument("--br-height", type=int, default=800)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    os.environ.setdefault("MASTER_PORT", str(29500 + int(args.gpu) * 10))
    os.environ.setdefault("MAGI_NEGATIVE_PROMPT", NEG_PROMPT)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("DAVINCI_DISABLE_MAGI_COMPILE", "1")
    os.environ.setdefault("MAGI_DISABLE_COMPILE", "1")

    repo = Path(args.repo)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(Path(args.manifest))
    indices = _load_indices(args)
    progress_path = Path(args.progress or (outdir / f"progress_gpu{args.gpu}.json"))

    prog = {"done": 0, "failed": 0, "total": len(indices), "items": []}
    if progress_path.is_file():
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    from inference.test_infer_zimage_native import create_zimage_native_pipeline  # noqa: E402

    br_w, br_h = int(args.br_width), int(args.br_height)
    print(f"[worker] gpu={args.gpu} indices={len(indices)} loading model ...", flush=True)
    t_load = time.time()
    pipeline = create_zimage_native_pipeline(
        config_load_path=str(args.config),
        base_ckpt_dir=str(args.base),
        ckpt_dir=str(args.ckpt),
        ckpt_blend_alpha=float(args.blend),
        device="cuda:0",
        br_width=br_w,
        br_height=br_h,
        num_inference_steps=int(args.steps),
        video_cfg_scale=float(args.cfg),
        audio_cfg_scale=float(args.cfg),
        freeze_audio=True,
    )
    print(f"[worker] model ready ({time.time() - t_load:.1f}s)", flush=True)

    for local_i, idx in enumerate(indices):
        if idx < 0 or idx >= len(manifest):
            print(f"[SKIP] index {idx} out of range", flush=True)
            continue
        prompt = str(manifest[idx].get("prompt", "")).strip()
        prefix = outdir / f"native_{br_w}x{br_h}_{idx:06d}"
        task_seed = int(args.seed) + idx
        t0 = time.time()
        print(f"[RUN {local_i+1}/{len(indices)}] idx={idx} seed={task_seed}", flush=True)
        try:
            pipeline.run_offline(
                prompt=prompt,
                image=None,
                audio=None,
                save_path_prefix=str(prefix),
                seed=task_seed,
                seconds=float(args.seconds),
                br_width=br_w,
                br_height=br_h,
            )
            prog["done"] = int(prog.get("done", 0)) + 1
            prog.setdefault("items", []).append(
                {"idx": idx, "status": "ok", "elapsed_sec": round(time.time() - t0, 1)}
            )
            print(f"[OK] idx={idx} ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            prog["failed"] = int(prog.get("failed", 0)) + 1
            prog.setdefault("items", []).append({"idx": idx, "status": "failed", "error": str(e)})
            print(f"[FAIL] idx={idx}: {e}", flush=True)
        progress_path.write_text(json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8")

    sys.exit(1 if int(prog.get("failed", 0)) else 0)


if __name__ == "__main__":
    main()
