"""Backward-compatible re-export — prefer `inference.scripts.vbench.lib`."""
from pathlib import Path
import importlib.util

_lib_path = Path(__file__).resolve().parent / "vbench" / "lib.py"
_spec = importlib.util.spec_from_file_location("vbench_lib", _lib_path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)
