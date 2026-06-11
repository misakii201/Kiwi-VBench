#!/usr/bin/env python3
"""
Encode low_res_preprocessed videos into DAVINCI training latents.

Follows the same layout as prepare_t2i_latents.py / prepare_seedance_dataset.py:
  dataset/<name>/
    manifest_*.jsonl
    latent_manifest_*.jsonl
    latents/video/000000.pt   # Wan2.2 VAE, shape (48, T_lat, H_lat, W_lat)
    latents/audio/000000.pt     # SAAudio VAE or zero placeholder, shape (T, 64)

Each video keeps its native resolution (aligned down to multiples of 16).
Default: first 121 frames (seedance-style). Audio is encoded when present,
otherwise a zero placeholder with shape (num_frames, 64) is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from prepare_seedance_dataset import (
    Record,
    _decode_video_frames,
    _encode_audio_latent,
    _encode_video_latent,
    _extract_audio_mono,
    _parse_dtype,
    _read_csv,
    _resolve_vae_checkpoint,
    _safe_mkdir,
    _split,
    _write_jsonl,
)
from inference.model.sa_audio.sa_audio_model import SAAudioFeatureExtractor
from inference.model.vae2_2.vae2_2_model import get_vae2_2

VAE_STRIDE = 16


def _align16(v: int) -> int:
    return max(VAE_STRIDE, (int(v) // VAE_STRIDE) * VAE_STRIDE)


def _ffprobe_video_size(video_path: str) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {video_path}")
    data = json.loads(proc.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    raise RuntimeError(f"No video stream: {video_path}")


def _target_size(video_path: str) -> Tuple[int, int]:
    width, height = _ffprobe_video_size(video_path)
    return _align16(width), _align16(height)


def _rel_to_output_dir(path: str, output_dir: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(output_dir))


def build_manifests(
    records: List[Record],
    output_dir: str,
    *,
    val_ratio: float,
    seed: int,
    input_csv: str,
    copy_csv: bool,
) -> Tuple[List[Record], List[Record], Dict[str, Any]]:
    missing = [r for r in records if not os.path.exists(r.video_path)]
    kept = [r for r in records if os.path.exists(r.video_path)]

    _safe_mkdir(output_dir)
    if copy_csv:
        shutil.copy2(input_csv, os.path.join(output_dir, os.path.basename(input_csv)))

    train_recs, val_recs = _split(kept, val_ratio, seed)
    train_items = [{"video_path": r.video_path, "prompt": r.prompt} for r in train_recs]
    val_items = [{"video_path": r.video_path, "prompt": r.prompt} for r in val_recs]
    all_items = [{"video_path": r.video_path, "prompt": r.prompt} for r in kept]

    _write_jsonl(os.path.join(output_dir, "manifest_all.jsonl"), all_items)
    _write_jsonl(os.path.join(output_dir, "manifest_train.jsonl"), train_items)
    _write_jsonl(os.path.join(output_dir, "manifest_val.jsonl"), val_items)
    if missing:
        _write_jsonl(
            os.path.join(output_dir, "missing.jsonl"),
            [{"video_path": r.video_path, "prompt": r.prompt} for r in missing],
        )

    meta: Dict[str, Any] = {
        "task": "low_res_video_audio",
        "input_csv": input_csv,
        "output_dir": output_dir,
        "total_rows": len(records),
        "kept_rows": len(kept),
        "missing_rows": len(missing),
        "val_ratio": val_ratio,
        "seed": seed,
        "resolution_mode": "native aligned to 16 per video",
        "prompt_source": "csv.caption -> manifest.prompt (text sidecar, not in .pt)",
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return kept, missing, meta


def _load_existing_latent_item(
    rec: Record,
    *,
    output_dir: str,
    global_idx: int,
    video_lat_path: str,
    audio_lat_path: str,
) -> Dict[str, Any]:
    audio_lat = torch.load(audio_lat_path, map_location="cpu")
    width, height = _target_size(rec.video_path)
    return {
        "video_latent_path": _rel_to_output_dir(video_lat_path, output_dir),
        "audio_latent_path": _rel_to_output_dir(audio_lat_path, output_dir),
        "prompt": rec.prompt,
        "audio_len": int(audio_lat.shape[0]),
        "video_path": rec.video_path,
        "index": global_idx,
        "pixel_width": width,
        "pixel_height": height,
    }


def encode_latents(
    kept: List[Record],
    output_dir: str,
    *,
    vae_model_path: str,
    audio_model_path: str,
    device: str,
    dtype: str,
    num_frames: int,
    skip_existing: bool,
    start_index: int,
    end_index: int,
    pad_if_smaller: bool,
    pad_mode: str,
    gpu_id: int = 0,
) -> List[Dict[str, Any]]:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        torch_device = torch.device(f"cuda:{int(gpu_id)}")
    else:
        torch_device = torch.device("cpu")

    weight_dtype = _parse_dtype(dtype)
    vae_ckpt = _resolve_vae_checkpoint(vae_model_path)
    vae = get_vae2_2(vae_ckpt, device=str(torch_device), weight_dtype=weight_dtype)
    audio_vae = SAAudioFeatureExtractor(device=str(torch_device), model_path=audio_model_path)
    target_sr = int(getattr(audio_vae, "sample_rate", 51200))
    max_audio_seconds = float(num_frames) / 25.0

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
    desc = f"encode_low_res gpu={gpu_id} [{start}:{end}]"
    for local_idx, rec in enumerate(tqdm(subset, desc=desc, unit="vid")):
        global_idx = start + local_idx
        base = f"{global_idx:06d}"
        video_lat_path = os.path.join(video_lat_dir, base + ".pt")
        audio_lat_path = os.path.join(audio_lat_dir, base + ".pt")

        if skip_existing and os.path.exists(video_lat_path) and os.path.exists(audio_lat_path):
            latent_items.append(
                _load_existing_latent_item(
                    rec,
                    output_dir=output_dir,
                    global_idx=global_idx,
                    video_lat_path=video_lat_path,
                    audio_lat_path=audio_lat_path,
                )
            )
            continue

        width, height = _target_size(rec.video_path)
        frames = _decode_video_frames(rec.video_path, num_frames=num_frames)
        video_lat = _encode_video_latent(
            vae,
            frames,
            device=torch_device,
            dtype=weight_dtype,
            height=height,
            width=width,
            pad_if_smaller=pad_if_smaller,
            pad_mode=pad_mode,
        )
        torch.save(video_lat, video_lat_path)

        audio_mono = _extract_audio_mono(rec.video_path, target_sample_rate=target_sr, max_seconds=max_audio_seconds)
        if audio_mono is None or audio_mono.numel() == 0:
            audio_lat = torch.zeros(num_frames, 64, dtype=torch.float32, device="cpu")
        else:
            audio_lat = _encode_audio_latent(audio_vae, audio_mono, device=torch_device)
        torch.save(audio_lat, audio_lat_path)

        latent_items.append(
            {
                "video_latent_path": _rel_to_output_dir(video_lat_path, output_dir),
                "audio_latent_path": _rel_to_output_dir(audio_lat_path, output_dir),
                "prompt": rec.prompt,
                "audio_len": int(audio_lat.shape[0]),
                "video_path": rec.video_path,
                "index": global_idx,
                "pixel_width": width,
                "pixel_height": height,
            }
        )

    return latent_items


def assemble_latent_manifests(
    kept: List[Record],
    output_dir: str,
    *,
    val_ratio: float,
    seed: int,
    num_frames: int,
) -> int:
    video_lat_dir = os.path.join(output_dir, "latents", "video")
    audio_lat_dir = os.path.join(output_dir, "latents", "audio")
    latent_items: List[Dict[str, Any]] = []

    for global_idx, rec in enumerate(kept):
        base = f"{global_idx:06d}"
        video_lat_path = os.path.join(video_lat_dir, base + ".pt")
        audio_lat_path = os.path.join(audio_lat_dir, base + ".pt")
        if not (os.path.exists(video_lat_path) and os.path.exists(audio_lat_path)):
            continue
        latent_items.append(
            _load_existing_latent_item(
                rec,
                output_dir=output_dir,
                global_idx=global_idx,
                video_lat_path=video_lat_path,
                audio_lat_path=audio_lat_path,
            )
        )

    train_lat, val_lat = _split(latent_items, val_ratio, seed)
    _write_jsonl(os.path.join(output_dir, "latent_manifest_all.jsonl"), latent_items)
    _write_jsonl(os.path.join(output_dir, "latent_manifest_train.jsonl"), train_lat)
    _write_jsonl(os.path.join(output_dir, "latent_manifest_val.jsonl"), val_lat)

    meta_path = os.path.join(output_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta["encoded_rows"] = len(latent_items)
    meta["num_frames"] = int(num_frames)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return len(latent_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode low_res_preprocessed videos into DAVINCI latents.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/kwkj-k8s/cy123/LF_test/low_res_preprocessed_prompt/prompts.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kwkj-k8s/cy123/workspace/DAVINCI/dataset/low_res_preprocessed",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/home/zetyun/Sora2-mini/UniAVGen",
    )
    parser.add_argument(
        "--audio_model_path",
        type=str,
        default="/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/audio_model",
    )
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--copy_csv", action="store_true")
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument(
        "--assemble_manifest_only",
        action="store_true",
        help="Scan existing .pt files and rebuild latent_manifest_*.jsonl only.",
    )
    parser.add_argument(
        "--encode_only",
        action="store_true",
        help="Worker mode: only write .pt latents for [start_index:end_index).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--pad_if_smaller", action="store_true", default=True)
    parser.add_argument("--pad_mode", type=str, default="edge", choices=["edge", "reflect"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    records = _read_csv(args.input_csv)
    if args.max_items and args.max_items > 0:
        records = records[: args.max_items]

    if args.assemble_manifest_only:
        kept = [r for r in records if os.path.exists(r.video_path)]
        encoded = assemble_latent_manifests(
            kept,
            args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_frames=args.num_frames,
        )
        print(f"[done] assemble_manifest_only: encoded_rows={encoded} output_dir={args.output_dir}")
        return

    if args.encode_only:
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
            copy_csv=args.copy_csv,
        )

    if missing:
        print(f"[warn] {len(missing)} rows missing on disk", file=sys.stderr)

    if args.manifest_only:
        print(f"[done] manifest_only: kept={len(kept)} output_dir={args.output_dir}")
        return

    latent_items = encode_latents(
        kept,
        args.output_dir,
        vae_model_path=args.vae_model_path,
        audio_model_path=args.audio_model_path,
        device=args.device,
        dtype=args.dtype,
        num_frames=args.num_frames,
        skip_existing=args.skip_existing,
        start_index=args.start_index,
        end_index=args.end_index,
        pad_if_smaller=bool(args.pad_if_smaller),
        pad_mode=args.pad_mode,
        gpu_id=args.gpu_id,
    )

    if args.encode_only:
        print(
            f"[done] encode_only gpu={args.gpu_id}: wrote {len(latent_items)} latents "
            f"shard [{args.start_index}:{args.end_index}] output_dir={args.output_dir}"
        )
        return

    train_lat, val_lat = _split(latent_items, args.val_ratio, args.seed)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_all.jsonl"), latent_items)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_train.jsonl"), train_lat)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_val.jsonl"), val_lat)

    if meta is None:
        meta = {}
    meta["encoded_rows"] = len(latent_items)
    meta["num_frames"] = int(args.num_frames)
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] encoded={len(latent_items)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
