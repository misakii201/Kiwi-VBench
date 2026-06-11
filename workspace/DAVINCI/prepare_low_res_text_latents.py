#!/usr/bin/env python3
"""
Offline T5-Gemma text encoding for DAVINCI training manifests.

Writes:
  dataset/<name>/latents/text/000000.pt   # float32, shape (1, L, 3584)
and patches manifest jsonl with:
  txt_latent_path, txt_feat_len

Aligns with video/audio latents by manifest "index" (or line order).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CPU_OFFLOAD", "1")

import torch
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding
from prepare_seedance_dataset import _safe_mkdir, _write_jsonl


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _rel_to_output_dir(path: str, output_dir: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(output_dir))


def _resolve_manifest(output_dir: str, manifest: Optional[str]) -> str:
    if manifest:
        return os.path.abspath(manifest)
    for name in ("latent_manifest_train.jsonl", "manifest_train.jsonl"):
        candidate = os.path.join(output_dir, name)
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    raise FileNotFoundError(
        f"No manifest found under {output_dir}. Pass --manifest explicitly."
    )


def _item_index(item: Dict[str, Any], line_idx: int) -> int:
    if "index" in item:
        return int(item["index"])
    return int(line_idx)


def _parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def encode_text_latents(
    items: List[Dict[str, Any]],
    *,
    output_dir: str,
    txt_model_path: str,
    target_length: int,
    device: str,
    weight_dtype: torch.dtype,
    skip_existing: bool,
    start_index: int,
    end_index: int,
) -> int:
    text_lat_dir = os.path.join(output_dir, "latents", "text")
    _safe_mkdir(text_lat_dir)

    start = max(0, int(start_index))
    end = int(end_index)
    if end < 0 or end > len(items):
        end = len(items)

    written = 0
    desc = f"encode_text [{start}:{end}] device={device}"
    for line_idx in tqdm(range(start, end), desc=desc, unit="txt"):
        item = items[line_idx]
        global_idx = _item_index(item, line_idx)
        base = f"{global_idx:06d}"
        text_lat_path = os.path.join(text_lat_dir, base + ".pt")

        if skip_existing and os.path.exists(text_lat_path):
            item["txt_latent_path"] = _rel_to_output_dir(text_lat_path, output_dir)
            if "txt_feat_len" not in item:
                cached = torch.load(text_lat_path, map_location="cpu")
                if isinstance(cached, torch.Tensor):
                    item["txt_feat_len"] = int(cached.shape[1] if cached.dim() == 3 else cached.shape[0])
            written += 1
            continue

        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            feat = torch.zeros(1, target_length, 3584, dtype=torch.float32)
            original_len = target_length
        else:
            feat, original_len = get_padded_t5_gemma_embedding(
                prompt,
                txt_model_path,
                device,
                weight_dtype,
                target_length,
            )

        torch.save(feat.cpu(), text_lat_path)
        item["txt_latent_path"] = _rel_to_output_dir(text_lat_path, output_dir)
        item["txt_feat_len"] = int(original_len)
        written += 1

    return written


def patch_manifest_txt_paths(
    items: List[Dict[str, Any]],
    *,
    output_dir: str,
    target_length: int,
) -> int:
    text_lat_dir = os.path.join(output_dir, "latents", "text")
    patched = 0
    for line_idx, item in enumerate(items):
        global_idx = _item_index(item, line_idx)
        text_lat_path = os.path.join(text_lat_dir, f"{global_idx:06d}.pt")
        if not os.path.exists(text_lat_path):
            continue
        item["txt_latent_path"] = _rel_to_output_dir(text_lat_path, output_dir)
        if "txt_feat_len" not in item:
            item["txt_feat_len"] = int(target_length)
        patched += 1
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode manifest prompts with T5-Gemma for training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kwkj-k8s/cy123/workspace/DAVINCI/dataset/low_res_preprocessed",
    )
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument(
        "--txt_model_path",
        type=str,
        default="/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/gemma",
    )
    parser.add_argument("--target_length", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument(
        "--encode_only",
        action="store_true",
        help="Only write latents/text/*.pt for [start_index:end_index); do not rewrite manifest.",
    )
    parser.add_argument(
        "--assemble_manifest_only",
        action="store_true",
        help="Scan latents/text/*.pt and patch manifest jsonl only.",
    )
    parser.add_argument(
        "--also_patch_all_manifest",
        action="store_true",
        help="Also write latent_manifest_all.jsonl when patching train manifest.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = _resolve_manifest(output_dir, args.manifest)
    items = _read_jsonl(manifest_path)

    if args.assemble_manifest_only:
        patched = patch_manifest_txt_paths(
            items,
            output_dir=output_dir,
            target_length=args.target_length,
        )
        _write_jsonl(manifest_path, items)
        if args.also_patch_all_manifest:
            all_path = os.path.join(output_dir, "latent_manifest_all.jsonl")
            if os.path.exists(all_path):
                all_items = _read_jsonl(all_path)
                patch_manifest_txt_paths(
                    all_items,
                    output_dir=output_dir,
                    target_length=args.target_length,
                )
                _write_jsonl(all_path, all_items)
        print(f"[done] assemble_manifest_only: patched={patched} manifest={manifest_path}")
        return

    weight_dtype = _parse_dtype(args.dtype)
    written = encode_text_latents(
        items,
        output_dir=output_dir,
        txt_model_path=args.txt_model_path,
        target_length=args.target_length,
        device=args.device,
        weight_dtype=weight_dtype,
        skip_existing=bool(args.skip_existing),
        start_index=args.start_index,
        end_index=args.end_index,
    )
    if args.encode_only:
        print(
            f"[done] encode_only: wrote {written} text latents "
            f"shard [{args.start_index}:{args.end_index}] output_dir={output_dir}"
        )
        return

    _write_jsonl(manifest_path, items)
    print(f"[done] encoded={written} manifest={manifest_path} output_dir={output_dir}")


if __name__ == "__main__":
    main()
