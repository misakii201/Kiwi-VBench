#!/usr/bin/env python3
"""
CloudSway MaaS_Ge_3_pro 图像冒烟测试

- 分辨率：在 RESOLUTIONS 里配置（aspectRatio + imageSize）
- 场景：在 SCENARIOS 里配置提示词，或从 prompt 文件 / jsonl 读取

文档: https://docs.cloudsway.net/zh/maasapi/api-reference/image/maas_ge_3_pro/

用法:
  export CLOUDSWAY_API_KEY="your_access_key"
  export CLOUDSWAY_IMAGE_ENDPOINT="your_endpoint_path"
  python scripts/smoke_cloudsway_image.py

  # 只跑部分场景/分辨率
  python scripts/smoke_cloudsway_image.py --scenarios dance,campus --resolutions portrait_480x800,landscape_1k
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 分辨率：在这里改
# aspectRatio 常见值: 1:1, 3:4, 4:3, 9:16, 16:9
# imageSize 常见值: 1K, 2K, 4K
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Resolution:
    name: str
    aspect_ratio: str
    image_size: str


RESOLUTIONS: dict[str, Resolution] = {
    # 接近 DAVINCI t2i 训练 480×800 竖屏
    "portrait_480x800": Resolution("portrait_480x800", "9:16", "1K"),
    "portrait_2k": Resolution("portrait_2k", "9:16", "2K"),
    # 横屏 / 方图
    "landscape_1k": Resolution("landscape_1k", "16:9", "1K"),
    "square_1k": Resolution("square_1k", "1:1", "1K"),
}


# ---------------------------------------------------------------------------
# 场景：在这里改提示词（key 仅用于输出文件名）
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, str] = {
    "dance": (
        "Full-body portrait of a young woman dancing in a bright minimalist dance studio, "
        "vertical smartphone vlog style, natural daylight, sharp focus, photorealistic."
    ),
    "campus": (
        "Cinematic over-the-shoulder shot of a university student walking on a tree-lined campus road, "
        "soft afternoon light, shallow depth of field, photorealistic drama still."
    ),
    "lab": (
        "Close-up of a graduate student in a research lab, focused curious expression, "
        "clinical task lighting, shallow depth of field, natural skin texture."
    ),
    "hallway": (
        "Environmental portrait of a student in a department hallway at dusk, "
        "warm window light, shallow depth of field, clean cinematic color grade."
    ),
}


def load_scenarios_from_prompt_dir(prompt_dir: Path) -> dict[str, str]:
    """从 example/assets/prompt01.txt 这类文件加载场景。"""
    scenarios: dict[str, str] = {}
    for path in sorted(prompt_dir.glob("prompt*.txt")):
        name = path.stem  # prompt01
        text = path.read_text(encoding="utf-8").strip()
        if text:
            scenarios[name] = text
    return scenarios


def load_scenarios_from_jsonl(jsonl_path: Path, *, limit: int = 0) -> dict[str, str]:
    """从 manifest/jsonl 读取 prompt 字段，key 用行号。"""
    scenarios: dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt", "")).strip()
            if prompt:
                scenarios[f"item_{i:04d}"] = prompt
    return scenarios


def pick_keys(all_keys: Iterable[str], selected: str | None) -> list[str]:
    if not selected:
        return list(all_keys)
    wanted = [k.strip() for k in selected.split(",") if k.strip()]
    missing = [k for k in wanted if k not in all_keys]
    if missing:
        raise ValueError(f"Unknown keys: {missing}. Available: {list(all_keys)}")
    return wanted


def extract_images(resp_json: dict) -> list[tuple[str, bytes]]:
    images: list[tuple[str, bytes]] = []
    msg = resp_json["choices"][0]["message"]

    for item in msg.get("images") or []:
        url = item.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            header, b64 = url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            images.append((mime, base64.b64decode(b64)))

    content = msg.get("content") or ""
    if isinstance(content, str):
        for m in re.finditer(r"data:(image/[^;]+);base64,([^\"'\s]+)", content):
            images.append((m.group(1), base64.b64decode(m.group(2))))
    return images


def generate_one(
    *,
    client: httpx.Client,
    api_url: str,
    api_key: str,
    scenario: str,
    prompt: str,
    resolution: Resolution,
    outdir: Path,
    save_raw_json: bool,
) -> Path | None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        "imageConfig": {
            "aspectRatio": resolution.aspect_ratio,
            "imageSize": resolution.image_size.lower(),
        },
    }

    resp = client.post(
        api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    resp.raise_for_status()
    data = resp.json()

    tag = f"{scenario}__{resolution.name}"
    if save_raw_json:
        (outdir / f"{tag}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    images = extract_images(data)
    if not images:
        print(f"[FAIL] {tag}: response has no image")
        return None

    mime, raw = images[0]
    ext = "png" if "png" in mime else "jpg"
    out_path = outdir / f"{tag}.{ext}"
    out_path.write_bytes(raw)
    print(f"[OK] {out_path}  ({resolution.aspect_ratio}, {resolution.image_size})")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CloudSway image smoke test")
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PROJECT_ROOT / "output_smoke_cloudsway_image"),
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario keys, e.g. dance,campus",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="Comma-separated resolution keys, e.g. portrait_480x800,landscape_1k",
    )
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default=None,
        help="Load scenarios from prompt*.txt instead of SCENARIOS dict",
    )
    parser.add_argument(
        "--prompt-jsonl",
        type=str,
        default=None,
        help="Load scenarios from jsonl prompt field",
    )
    parser.add_argument(
        "--jsonl-limit",
        type=int,
        default=4,
        help="Max rows when using --prompt-jsonl (0 = all)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("CLOUDSWAY_IMAGE_ENDPOINT", ""),
        help="CloudSway endpointPath (or env CLOUDSWAY_IMAGE_ENDPOINT)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("CLOUDSWAY_API_KEY", ""),
        help="CloudSway access key (or env CLOUDSWAY_API_KEY)",
    )
    parser.add_argument(
        "--save-raw-json",
        action="store_true",
        help="Also save full API response json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("Error: set CLOUDSWAY_API_KEY or pass --api-key", file=sys.stderr)
        return 1
    if not args.endpoint:
        print("Error: set CLOUDSWAY_IMAGE_ENDPOINT or pass --endpoint", file=sys.stderr)
        return 1

    if args.prompt_jsonl:
        scenarios = load_scenarios_from_jsonl(Path(args.prompt_jsonl), limit=args.jsonl_limit)
    elif args.prompt_dir:
        scenarios = load_scenarios_from_prompt_dir(Path(args.prompt_dir))
    else:
        scenarios = SCENARIOS

    if not scenarios:
        print("Error: no scenarios loaded", file=sys.stderr)
        return 1

    scenario_keys = pick_keys(scenarios.keys(), args.scenarios)
    resolution_keys = pick_keys(RESOLUTIONS.keys(), args.resolutions)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://genaiapi.cloudsway.net/v1/ai/{args.endpoint.strip()}/chat/completions"
    print(f"API: {api_url}")
    print(f"Scenarios: {scenario_keys}")
    print(f"Resolutions: {resolution_keys}")
    print(f"Output: {outdir}")

    ok = 0
    fail = 0
    with httpx.Client(timeout=300.0) as client:
        for s_key in scenario_keys:
            prompt = scenarios[s_key]
            for r_key in resolution_keys:
                resolution = RESOLUTIONS[r_key]
                try:
                    path = generate_one(
                        client=client,
                        api_url=api_url,
                        api_key=args.api_key,
                        scenario=s_key,
                        prompt=prompt,
                        resolution=resolution,
                        outdir=outdir,
                        save_raw_json=args.save_raw_json,
                    )
                    if path is None:
                        fail += 1
                    else:
                        ok += 1
                except httpx.HTTPStatusError as e:
                    fail += 1
                    body = e.response.text[:500]
                    print(f"[ERR] {s_key}/{r_key}: HTTP {e.response.status_code} {body}")
                except Exception as e:
                    fail += 1
                    print(f"[ERR] {s_key}/{r_key}: {e}")

    print(f"Done. ok={ok}, fail={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
