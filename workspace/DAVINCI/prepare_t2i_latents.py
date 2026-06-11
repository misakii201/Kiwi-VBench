#!/usr/bin/env python3
"""
Prepare T2I (single-frame image) latents for daVinci-MagiHuman base-model training.

Reads a CSV with columns: caption, media_path
  - caption  -> stored as `prompt` in manifest (used later by train_ltx.py text encoder)
  - media_path -> source image (.png/.jpg)

Outputs:
  dataset/<name>/
    manifest_train.jsonl          # image path + prompt (no latents)
    latent_manifest_train.jsonl   # latent paths + prompt + metadata
    latents/video/000000.pt       # VAE-encoded video latent, shape (48, 1, H_lat, W_lat)
    latents/audio/000000.pt       # zero placeholder, shape (1, 64)
    meta.json

Note: prompts are NOT embedded inside .pt tensors. They live in the jsonl sidecar and are
injected at training time when --use_text_encoder is enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from prepare_seedance_dataset import (
    Record,
    _encode_video_latent,
    _parse_dtype,
    _read_csv,
    _resolve_vae_checkpoint,
    _safe_mkdir,
    _split,
    _write_jsonl,
)
from inference.common import parse_config
from inference.model.vae2_2.vae2_2_model import get_vae2_2

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _load_image_frame(image_path: str) -> Image.Image:
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image extension: {image_path}")
    return Image.open(image_path).convert("RGB")


def _rel_to_output_dir(path: str, output_dir: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(output_dir))


def _expected_latent_shape(width: int, height: int, vae_stride_hw: Tuple[int, int] = (16, 16)) -> Tuple[int, int]:
    h_lat = height // vae_stride_hw[0]
    w_lat = width // vae_stride_hw[1]
    return h_lat, w_lat


def build_manifests(
    records: List[Record],
    output_dir: str,
    *,
    val_ratio: float,
    seed: int,
    input_csv: str,
    width: int,
    height: int,
    copy_csv: bool,
) -> Tuple[List[Record], List[Record], Dict[str, Any]]:
    missing = [r for r in records if not os.path.exists(r.video_path)]
    kept = [r for r in records if os.path.exists(r.video_path)]

    _safe_mkdir(output_dir)
    if copy_csv:
        shutil.copy2(input_csv, os.path.join(output_dir, os.path.basename(input_csv)))

    train_recs, val_recs = _split(kept, val_ratio, seed)
    train_items = [{"image_path": r.video_path, "prompt": r.prompt} for r in train_recs]
    val_items = [{"image_path": r.video_path, "prompt": r.prompt} for r in val_recs]
    all_items = [{"image_path": r.video_path, "prompt": r.prompt} for r in kept]

    _write_jsonl(os.path.join(output_dir, "manifest_all.jsonl"), all_items)
    _write_jsonl(os.path.join(output_dir, "manifest_train.jsonl"), train_items)
    _write_jsonl(os.path.join(output_dir, "manifest_val.jsonl"), val_items)
    if missing:
        _write_jsonl(
            os.path.join(output_dir, "missing.jsonl"),
            [{"image_path": r.video_path, "prompt": r.prompt} for r in missing],
        )

    h_lat, w_lat = _expected_latent_shape(width, height)
    meta: Dict[str, Any] = {
        "task": "t2i_single_frame",
        "input_csv": input_csv,
        "output_dir": output_dir,
        "pixel_width": width,
        "pixel_height": height,
        "expected_video_latent_shape": [48, 1, h_lat, w_lat],
        "expected_audio_latent_shape": [1, 64],
        "total_rows": len(records),
        "kept_rows": len(kept),
        "missing_rows": len(missing),
        "val_ratio": val_ratio,
        "seed": seed,
        "prompt_source": "csv.caption -> manifest.prompt (text sidecar, not in .pt)",
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return kept, missing, meta


def encode_latents(
    kept: List[Record],
    output_dir: str,
    *,
    vae_model_path: str,
    device: str,
    dtype: str,
    width: int,
    height: int,
    skip_existing: bool,
    start_index: int,
    end_index: int,
    pad_if_smaller: bool,
    pad_mode: str,
) -> List[Dict[str, Any]]:
    torch_device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    weight_dtype = _parse_dtype(dtype)
    vae_ckpt = _resolve_vae_checkpoint(vae_model_path)
    vae = get_vae2_2(vae_ckpt, device=str(torch_device), weight_dtype=weight_dtype)

    video_lat_dir = os.path.join(output_dir, "latents", "video")
    audio_lat_dir = os.path.join(output_dir, "latents", "audio")
    _safe_mkdir(video_lat_dir)
    _safe_mkdir(audio_lat_dir)

    start = max(0, int(start_index))
    end = int(end_index)
    if end < 0 or end > len(kept):
        end = len(kept)
    subset = kept[start:end]

    latent_items: List[Dict[str, Any]] = []
    for local_idx, rec in enumerate(tqdm(subset, desc="encode_t2i_latents", unit="img")):
        global_idx = start + local_idx
        base = f"{global_idx:06d}"
        video_lat_path = os.path.join(video_lat_dir, base + ".pt")
        audio_lat_path = os.path.join(audio_lat_dir, base + ".pt")

        if skip_existing and os.path.exists(video_lat_path) and os.path.exists(audio_lat_path):
            audio_lat = torch.load(audio_lat_path, map_location="cpu")
            latent_items.append(
                {
                    "video_latent_path": _rel_to_output_dir(video_lat_path, output_dir),
                    "audio_latent_path": _rel_to_output_dir(audio_lat_path, output_dir),
                    "prompt": rec.prompt,
                    "audio_len": int(audio_lat.shape[0]),
                    "video_path": rec.video_path,
                }
            )
            continue

        image = _load_image_frame(rec.video_path)
        video_lat = _encode_video_latent(
            vae,
            [image],
            device=torch_device,
            dtype=weight_dtype,
            height=height,
            width=width,
            pad_if_smaller=pad_if_smaller,
            pad_mode=pad_mode,
        )
        audio_lat = torch.zeros(1, 64, dtype=torch.float32)

        torch.save(video_lat, video_lat_path)
        torch.save(audio_lat, audio_lat_path)

        latent_items.append(
            {
                "video_latent_path": _rel_to_output_dir(video_lat_path, output_dir),
                "audio_latent_path": _rel_to_output_dir(audio_lat_path, output_dir),
                "prompt": rec.prompt,
                "audio_len": 1,
                "video_path": rec.video_path,
            }
        )

    return latent_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode T2I image dataset into training latents.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/kwkj-k8s/LTX-2/videos-kzy/5-14/1w张图片_480p对齐32/results_mapped.csv",
        help="CSV with columns: caption, media_path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "t2i_1w_480x800_f1"),
    )
    parser.add_argument("--width", type=int, default=480, help="Target image width in pixels.")
    parser.add_argument("--height", type=int, default=800, help="Target image height in pixels.")
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_items", type=int, default=0, help="Limit rows for smoke test; 0 means all.")
    parser.add_argument("--copy_csv", action="store_true")
    parser.add_argument("--manifest_only", action="store_true", help="Only build manifest/meta, skip VAE encode.")
    parser.add_argument(
        "--encode_only",
        action="store_true",
        help="Worker mode for multi-GPU sharding: only write .pt latents, do NOT touch "
        "image/latent manifests or meta.json. Run a final non-encode_only pass with "
        "--skip_existing afterwards to assemble the complete manifest.",
    )
    parser.add_argument("--config-load-path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--pad_if_smaller", action="store_true")
    parser.add_argument("--pad_mode", type=str, default="edge", choices=["edge", "reflect"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()

    records = _read_csv(args.input_csv)
    if args.max_items and args.max_items > 0:
        records = records[: args.max_items]

    if args.encode_only:
        # Worker mode: derive the same `kept` ordering as build_manifests would,
        # but never write any manifest/meta file (avoids concurrent overwrites).
        kept = [r for r in records if os.path.exists(r.video_path)]
        missing = [r for r in records if not os.path.exists(r.video_path)]
        meta = None
    else:
        kept, missing, meta = build_manifests(
            records,
            args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            input_csv=args.input_csv,
            width=args.width,
            height=args.height,
            copy_csv=args.copy_csv,
        )

    if missing:
        print(f"[warn] {len(missing)} rows missing on disk, see missing.jsonl", file=sys.stderr)

    if args.manifest_only:
        print(f"[done] manifest_only: kept={len(kept)} output_dir={args.output_dir}")
        return

    cfg = parse_config() if args.config_load_path else None
    vae_model_path = args.vae_model_path or (cfg.evaluation_config.vae_model_path if cfg is not None else "")
    if not vae_model_path:
        raise ValueError("vae_model_path is empty. Pass --vae_model_path or --config-load-path.")

    latent_items = encode_latents(
        kept,
        args.output_dir,
        vae_model_path=vae_model_path,
        device=args.device,
        dtype=args.dtype,
        width=args.width,
        height=args.height,
        skip_existing=args.skip_existing,
        start_index=args.start_index,
        end_index=args.end_index,
        pad_if_smaller=args.pad_if_smaller,
        pad_mode=args.pad_mode,
    )

    h_lat, w_lat = _expected_latent_shape(args.width, args.height)

    if args.encode_only:
        print(
            f"[done] encode_only: wrote {len(latent_items)} latents "
            f"(shard [{args.start_index}:{args.end_index}]) "
            f"pixel={args.width}x{args.height} latent=(48,1,{h_lat},{w_lat}). "
            f"Run a final --skip_existing pass (without --encode_only) to build the manifest."
        )
        return

    train_lat, val_lat = _split(latent_items, args.val_ratio, args.seed)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_all.jsonl"), latent_items)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_train.jsonl"), train_lat)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_val.jsonl"), val_lat)

    if meta is None:
        meta = {}
    meta["encoded_rows"] = len(latent_items)
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[done] encoded={len(latent_items)} "
        f"pixel={args.width}x{args.height} latent=(48,1,{h_lat},{w_lat}) "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
