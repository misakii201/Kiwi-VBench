#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "vbench" / "run.py"
    runpy.run_path(str(target), run_name="__main__")
