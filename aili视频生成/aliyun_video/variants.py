from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VariantSpec:
    ratio_slug: str
    ratio_w: int
    ratio_h: int
    quality: str
    target_w: int
    target_h: int


RATIOS: tuple[tuple[str, int, int], ...] = (
    ("16x9", 16, 9),
    ("9x16", 9, 16),
    ("1x1", 1, 1),
    ("4x3", 4, 3),
    ("3x4", 3, 4),
    ("4x5", 4, 5),
    ("5x4", 5, 4),
    ("9x21", 9, 21),
    ("21x9", 21, 9),
)

QUALITY_BASES = {
    "1080P": 1080,
    "720P": 720,
}


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def dimensions_for_ratio(ratio_w: int, ratio_h: int, base: int) -> tuple[int, int]:
    if ratio_w >= ratio_h:
        height = base
        width = _even(round(base * ratio_w / ratio_h))
    else:
        width = base
        height = _even(round(base * ratio_h / ratio_w))
    return width, height


def build_variant_matrix() -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for ratio_slug, ratio_w, ratio_h in RATIOS:
        for quality, base in QUALITY_BASES.items():
            width, height = dimensions_for_ratio(ratio_w, ratio_h, base)
            specs.append(
                VariantSpec(
                    ratio_slug=ratio_slug,
                    ratio_w=ratio_w,
                    ratio_h=ratio_h,
                    quality=quality,
                    target_w=width,
                    target_h=height,
                )
            )
    return specs


def build_filter(source_w: int, source_h: int, variant: VariantSpec) -> str:
    source_ratio = source_w / source_h
    target_ratio = variant.ratio_w / variant.ratio_h

    if source_ratio > target_ratio:
        crop_h = source_h
        crop_w = _even(round(crop_h * target_ratio))
    else:
        crop_w = source_w
        crop_h = _even(round(crop_w / target_ratio))

    crop_x = _even((source_w - crop_w) // 2)
    crop_y = _even((source_h - crop_h) // 2)
    return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={variant.target_w}:{variant.target_h}"


def parse_size_from_ffmpeg_output(output: str) -> tuple[int, int]:
    match = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})[, ]", output)
    if not match:
        raise RuntimeError("could not parse video size from ffmpeg output")
    return int(match.group(1)), int(match.group(2))


def probe_size(ffmpeg: str, input_path: Path) -> tuple[int, int]:
    command = [ffmpeg, "-i", str(input_path)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return parse_size_from_ffmpeg_output(result.stderr + result.stdout)


def derive_variant(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    source_size: tuple[int, int],
    variant: VariantSpec,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        build_filter(source_size[0], source_size[1], variant),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]
    subprocess.run(command, check=True)
