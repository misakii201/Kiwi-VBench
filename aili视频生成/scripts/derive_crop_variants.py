from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg

from aliyun_video.variants import VariantSpec, build_filter, build_variant_matrix, derive_variant, parse_size_from_ffmpeg_output, probe_size


VARIANTS = {variant.ratio_slug: variant for variant in build_variant_matrix() if variant.quality == "1080P"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive center-cropped 1080P aspect-ratio variants from one master video.")
    parser.add_argument("--input", required=True, help="Master MP4 path.")
    parser.add_argument("--output-dir", required=True, help="Directory for derived MP4 files.")
    parser.add_argument("--prefix", default="basketball_square_master", help="Output filename prefix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    source_size = probe_size(ffmpeg, input_path)

    for variant in VARIANTS.values():
        output_path = output_dir / f"{args.prefix}_{variant.ratio_slug}_{variant.target_w}x{variant.target_h}.mp4"
        derive_variant(ffmpeg, input_path, output_path, source_size, variant)
        print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
