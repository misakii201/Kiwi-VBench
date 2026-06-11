#!/usr/bin/env python3
"""Summarize VBench-2.0 eval JSON into score_summary.md."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    out_path = Path(args.output or (eval_root / "score_summary.md"))

    rows: list[tuple[str, float | None]] = []
    for path in sorted(glob.glob(str(eval_root / "*" / "*eval_results.json"))):
        dim = Path(path).parent.name
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows.append((dim, None))
            continue
        score = None
        if isinstance(data, dict):
            for k in ("score", "overall_score", "final_score"):
                if k in data and isinstance(data[k], (int, float)):
                    score = float(data[k])
                    break
            if score is None:
                for v in data.values():
                    if isinstance(v, (int, float)):
                        score = float(v)
                        break
        rows.append((dim, score))

    lines = ["# VBench-2.0 score summary", "", f"eval_root: `{eval_root}`", ""]
    lines.append("| Dimension | Score |")
    lines.append("|-----------|-------|")
    vals = [s for _, s in rows if s is not None]
    for dim, score in rows:
        lines.append(f"| {dim} | {score:.4f} |" if score is not None else f"| {dim} | - |")
    if vals:
        lines.extend(["", f"**Mean (available dims):** {sum(vals)/len(vals):.4f}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
