from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aliyun_video.batch_variants import write_sample_18_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a one-prompt, 18-row sample results.csv.")
    parser.add_argument("--output-dir", default="outputs/sample_18_rows", help="Directory for sample outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path = write_sample_18_rows(args.output_dir)
    print(results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
