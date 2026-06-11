#!/usr/bin/env python3
"""Native-resolution zimage inference (e.g. 480x800 portrait).

test_infer_seedance.py targets large / tiled decodes: --spatial_fix enables
vertical/horizontal heuristics that crop a segment then resize back to (H, W).
That only runs when frame count T >= 2, so 0.0s single-frame eval looks fine
while 4.0s clips get stretched and cropped.

This entry point disables those fixes and uses pad-based export so aspect ratio
matches training (zimage_480x800_1w).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

os.environ.setdefault("DAVINCI_DISABLE_MAGI_COMPILE", "1")
os.environ.setdefault("MAGI_DISABLE_COMPILE", "1")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.common import parse_config
from inference.test_infer_seedance import (  # noqa: E402
    _PROJECT_ROOT as _ROOT,
    _parse_amp_dtype,
    _resolve_device,
    build_dit_for_infer,
)


def apply_zimage_native_env(*, br_width: int, br_height: int) -> None:
    """Env tuned for fixed small native resolution (no large-res tiling fixes)."""
    os.environ["MAGI_ALIGN_RESOLUTION"] = "ceil"
    os.environ["MAGI_RESIZE_STRATEGY"] = "pad"
    os.environ["MAGI_PAD_MODE"] = "replicate"
    os.environ["MAGI_SPATIAL_ROPE_INTERPOLATION"] = "extra"
    os.environ["MAGI_DISABLE_AUTO_INTER"] = "1"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ["MAGI_NATIVE_BR"] = f"{int(br_width)}x{int(br_height)}"

    # Allow ablation scripts to override spatial/canvas fix via shell env.
    if os.environ.get("MAGI_NATIVE_RESPECT_FIX_ENV", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }:
        return

    os.environ["MAGI_CANVAS_FIX"] = "0"
    os.environ["MAGI_DISABLE_SPATIAL_FIX"] = "1"
    os.environ["MAGI_DISABLE_VERTICAL_FIX"] = "1"
    os.environ["MAGI_DISABLE_HORIZONTAL_FIX"] = "1"
    os.environ["MAGI_DISABLE_TILING_FIX"] = "1"


def create_zimage_native_pipeline(
    *,
    config_load_path: str,
    base_ckpt_dir: str | None,
    ckpt_dir: str,
    ckpt_blend_alpha: float = 1.0,
    device: str = "cuda:0",
    amp_dtype: str = "bf16",
    num_inference_steps: int = 32,
    video_cfg_scale: float = 6.0,
    audio_cfg_scale: float = 6.0,
    br_width: int,
    br_height: int,
    freeze_audio: bool = True,
):
    """Load DiT + MagiPipeline once (for batch / long-lived workers)."""
    from inference.infra import initialize_infra
    from inference.pipeline.pipeline import MagiPipeline

    br_w = int(br_width)
    br_h = int(br_height)
    apply_zimage_native_env(br_width=br_w, br_height=br_h)

    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise RuntimeError("Native zimage infer requires CUDA.")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    torch.cuda.set_device(
        resolved.index if resolved.index is not None else int(os.environ.get("LOCAL_RANK", "0"))
    )
    initialize_infra()

    if not os.path.exists(config_load_path):
        raise FileNotFoundError(config_load_path)

    old_argv = sys.argv
    sys.argv = [old_argv[0], "--config-load-path", config_load_path]
    config = parse_config()
    sys.argv = old_argv

    dtype = _parse_amp_dtype(amp_dtype)
    config.arch_config.params_dtype = dtype
    config.sr_arch_config.params_dtype = dtype
    config.evaluation_config.num_inference_steps = int(num_inference_steps)
    config.evaluation_config.video_txt_guidance_scale = float(video_cfg_scale)
    config.evaluation_config.audio_txt_guidance_scale = float(audio_cfg_scale)

    if ckpt_dir:
        config.engine_config.load = str(ckpt_dir)

    engine_load_path = config.engine_config.load
    if not engine_load_path:
        raise ValueError("Empty checkpoint path")

    model = build_dit_for_infer(
        config.arch_config,
        config.engine_config,
        resolved,
        base_ckpt_dir=base_ckpt_dir,
        ckpt_blend_alpha=float(ckpt_blend_alpha),
        forced_load_path=str(engine_load_path),
    )

    if not config.evaluation_config.vae_model_path:
        raise ValueError("evaluation_config.vae_model_path is empty")
    if not config.evaluation_config.audio_model_path:
        raise ValueError("evaluation_config.audio_model_path is empty")

    if freeze_audio:
        os.environ.setdefault("MAGI_T2V_FREEZE_AUDIO", "1")
    else:
        os.environ.pop("MAGI_T2V_FREEZE_AUDIO", None)

    pipeline = MagiPipeline(model, config.evaluation_config, device=str(resolved))
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Zimage native-resolution video generation")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_path_prefix", type=str, required=True)
    parser.add_argument("--config-load-path", type=str, default=None)
    parser.add_argument("--base_ckpt_dir", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_blend_alpha", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--br_width", type=int, default=480)
    parser.add_argument("--br_height", type=int, default=800)
    parser.add_argument("--output_width", type=int, default=None)
    parser.add_argument("--output_height", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=32)
    parser.add_argument("--video_cfg_scale", type=float, default=6.0)
    parser.add_argument("--audio_cfg_scale", type=float, default=6.0)
    parser.add_argument("--freeze_audio", action="store_true")
    parser.add_argument("--upsample_mode", type=str, default="bicubic", choices=["bilinear", "nearest", "bicubic"])
    args = parser.parse_args()

    br_w = int(args.br_width)
    br_h = int(args.br_height)
    out_w = int(args.output_width) if args.output_width is not None else br_w
    out_h = int(args.output_height) if args.output_height is not None else br_h

    config_path = args.config_load_path
    if not config_path:
        config_path = os.path.join(_ROOT, "example", "base", "config_infer.json")

    try:
        pipeline = create_zimage_native_pipeline(
            config_load_path=config_path,
            base_ckpt_dir=args.base_ckpt_dir,
            ckpt_dir=str(args.ckpt_dir),
            ckpt_blend_alpha=float(args.ckpt_blend_alpha),
            device=str(args.device),
            amp_dtype=str(args.amp_dtype),
            num_inference_steps=int(args.num_inference_steps),
            video_cfg_scale=float(args.video_cfg_scale),
            audio_cfg_scale=float(args.audio_cfg_scale),
            br_width=br_w,
            br_height=br_h,
            freeze_audio=bool(args.freeze_audio),
        )

        save_path_prefix = str(args.save_path_prefix)
        if not os.path.isabs(save_path_prefix):
            save_path_prefix = os.path.join(_ROOT, save_path_prefix)
        os.makedirs(os.path.dirname(save_path_prefix) or ".", exist_ok=True)

        save_path = pipeline.run_offline(
            prompt=str(args.prompt).strip(),
            image=None,
            audio=None,
            save_path_prefix=save_path_prefix,
            seed=int(args.seed),
            seconds=float(args.seconds),
            br_width=br_w,
            br_height=br_h,
            output_width=out_w if args.output_width is not None else None,
            output_height=out_h if args.output_height is not None else None,
            upsample_mode=args.upsample_mode,
        )
        print("saved_video:", save_path)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
