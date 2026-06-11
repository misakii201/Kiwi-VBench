#!/usr/bin/env python3
"""
DAVINCI VBench-2.0 worker。

由 `run_vbench2_davinci.py` 通过 `torchrun --nproc_per_node=1` 启动；
读取分配给本 worker 的 task 桶，复用同一个 MagiPipeline 顺序生成 mp4，
并把跨维度复用的视频 copy 分发到其它维度目录。
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# 抑制 MagiCompiler 在不支持的环境/模型尺寸下报错（与 test_infer_seedance.py 默认一致）
os.environ.setdefault("DAVINCI_DISABLE_MAGI_COMPILE", "1")
os.environ.setdefault("MAGI_DISABLE_COMPILE", "1")

_SCRIPT_DIR = Path(__file__).resolve().parent
_DAVINCI_ROOT = _SCRIPT_DIR.parent.parent
if str(_DAVINCI_ROOT) not in sys.path:
    sys.path.insert(0, str(_DAVINCI_ROOT))

import yaml  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", type=str, required=True)
    parser.add_argument("--config-yaml", type=str, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args


def _setup_distributed_env():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def _ensure_ffmpeg_on_path() -> None:
    """DAVINCI pipeline 直接调 `ffmpeg`。若系统未装，回退用 imageio_ffmpeg 自带二进制。"""
    import shutil as _sh
    if _sh.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg  # type: ignore
    except Exception:
        return
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not exe or not os.path.isfile(exe):
        return
    bin_dir = os.path.dirname(sys.executable)
    link = os.path.join(bin_dir, "ffmpeg")
    try:
        if not os.path.exists(link):
            os.symlink(exe, link)
    except Exception:
        pass
    # 兜底：把 imageio_ffmpeg 的二进制目录加进 PATH（链接万一建失败也能命中）
    os.environ["PATH"] = os.path.dirname(exe) + os.pathsep + os.environ.get("PATH", "")


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s w{args.worker_id} %(message)s",
    )
    log = logging.getLogger("davinci.vbench2.worker")

    with open(args.tasks_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    tasks: list[dict] = payload["tasks"]
    base_seed: int = int(payload["base_seed"])
    out_dir = Path(payload["out_dir"]).resolve()

    with open(args.config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["model"]
    p = cfg["params"]

    _setup_distributed_env()
    _ensure_ffmpeg_on_path()

    import torch

    # 把 DAVINCI 的 example/base/config.json 喂给 parse_config（它从 sys.argv 解析 --config-load-path）
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--config-load-path", str(m["config_load_path"])]
    from inference.common import parse_config  # noqa: E402

    config = parse_config()
    sys.argv = old_argv

    # 覆盖关键参数
    config.engine_config.load = str(m["checkpoint_path"])
    config.evaluation_config.num_inference_steps = int(p.get("num_inference_steps", 32))
    config.evaluation_config.cfg_number = int(p.get("cfg_number", 2))
    if p.get("video_cfg_scale") is not None:
        config.evaluation_config.video_txt_guidance_scale = float(p["video_cfg_scale"])
    if p.get("audio_cfg_scale") is not None:
        config.evaluation_config.audio_txt_guidance_scale = float(p["audio_cfg_scale"])

    neg_prompt = p.get("negative_prompt")
    if neg_prompt:
        os.environ["MAGI_NEGATIVE_PROMPT"] = str(neg_prompt)
    if bool(p.get("t2v_freeze_audio", True)):
        os.environ.setdefault("MAGI_T2V_FREEZE_AUDIO", "1")
    else:
        os.environ.pop("MAGI_T2V_FREEZE_AUDIO", None)
    os.environ.setdefault("MAGI_DISABLE_SPATIAL_FIX", "1")
    os.environ.setdefault("MAGI_CANVAS_FIX", "0")

    ckpt_blend_alpha = float(m.get("ckpt_blend_alpha", p.get("ckpt_blend_alpha", 1.0)))

    use_sr = bool(p.get("use_sr", False))
    if use_sr:
        if not p.get("sr_model_path"):
            raise ValueError("params.use_sr=true 时必须提供 params.sr_model_path")
        config.evaluation_config.use_sr_model = True
        config.evaluation_config.sr_model_path = str(p["sr_model_path"])
        if "sr_num_inference_steps" in p:
            config.evaluation_config.sr_num_inference_steps = int(p["sr_num_inference_steps"])
        if "sr_cfg_number" in p:
            config.evaluation_config.sr_cfg_number = int(p["sr_cfg_number"])
    else:
        config.evaluation_config.use_sr_model = False

    amp_dtype = torch.bfloat16
    config.arch_config.params_dtype = amp_dtype

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from inference.infra import initialize_infra  # noqa: E402

    initialize_infra()

    from inference.test_infer_seedance import build_dit_for_infer  # noqa: E402
    from inference.pipeline.pipeline import MagiPipeline  # noqa: E402

    log.info(
        "加载 base=%s + finetune=%s (blend=%.2f, use_sr=%s, steps=%d, v_cfg=%.1f, a_cfg=%.1f) | %d 个任务",
        m.get("base_ckpt_dir"), m["checkpoint_path"], ckpt_blend_alpha, use_sr,
        config.evaluation_config.num_inference_steps,
        config.evaluation_config.video_txt_guidance_scale,
        config.evaluation_config.audio_txt_guidance_scale,
        len(tasks),
    )

    model = build_dit_for_infer(
        config.arch_config,
        config.engine_config,
        device,
        base_ckpt_dir=str(m.get("base_ckpt_dir") or "") or None,
        ckpt_blend_alpha=ckpt_blend_alpha,
        forced_load_path=str(m["checkpoint_path"]),
    )
    pipeline = MagiPipeline(model, config.evaluation_config, device=str(device))

    seconds = float(p.get("seconds", 5))
    br_width = int(p.get("br_width", 480))
    br_height = int(p.get("br_height", 272))
    sr_width = int(p["sr_width"]) if use_sr and "sr_width" in p else None
    sr_height = int(p["sr_height"]) if use_sr and "sr_height" in p else None
    output_width = int(p["output_width"]) if p.get("output_width") else None
    output_height = int(p["output_height"]) if p.get("output_height") else None
    output_upsample_mode = p.get("output_upsample_mode") or None

    is_last_rank = (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or (torch.distributed.get_rank() == torch.distributed.get_world_size() - 1)
    )

    for task_idx, task in enumerate(tasks):
        dimension = task["dimension"]
        prompt = task["prompt"]
        gen_prompt = task["gen_prompt"]
        prompt_idx = int(task["prompt_idx"])
        index = int(task["index"])
        other_dims = list(task.get("other_dims") or [])

        dim_dir = out_dir / dimension
        dim_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{prompt[:180]}-{index}.mp4"
        mp4_path = dim_dir / filename

        if mp4_path.is_file():
            for od in other_dims:
                other_path = out_dir / od / filename
                if not other_path.is_file():
                    other_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(mp4_path), str(other_path))
                    log.info("[%d/%d] 补全自动复用至 %s/%s", task_idx + 1, len(tasks), od, filename)
            log.info("[%d/%d] 已存在，跳过 %s/%s", task_idx + 1, len(tasks), dimension, filename)
            continue

        seed = (base_seed + prompt_idx * 100 + index) % (2**32)
        log.info(
            "[%d/%d] 正在生成 %s/%s | seed=%s | prompt=%r",
            task_idx + 1, len(tasks), dimension, filename, seed, gen_prompt[:80],
        )

        with tempfile.TemporaryDirectory(prefix="davinci_vbench2_") as tmp:
            tmp_prefix = str(Path(tmp) / "clip")
            try:
                save_path = pipeline.run_offline(
                    prompt=gen_prompt,
                    image=None,
                    audio=None,
                    save_path_prefix=tmp_prefix,
                    seed=seed,
                    seconds=seconds,
                    br_width=br_width,
                    br_height=br_height,
                    sr_width=sr_width,
                    sr_height=sr_height,
                    output_width=output_width,
                    output_height=output_height,
                    upsample_mode=output_upsample_mode,
                )
            except Exception as e:
                log.exception("生成失败 %s/%s: %s", dimension, filename, e)
                raise

            if is_last_rank:
                # save_path 是 {prefix}_{seconds}s_{w}x{h}[ _{sr_w}x{sr_h}].mp4
                produced = save_path if save_path and Path(save_path).is_file() else None
                if produced is None:
                    candidates = sorted(glob.glob(f"{tmp_prefix}*.mp4"))
                    if not candidates:
                        raise FileNotFoundError(f"pipeline 未产出 mp4，目录: {tmp}")
                    produced = candidates[-1]
                shutil.move(str(produced), str(mp4_path))
                for od in other_dims:
                    other_path = out_dir / od / filename
                    other_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(mp4_path), str(other_path))
                    log.info("自动复用视频分发至 %s/%s", od, filename)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    log.info("worker 任务完成")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
