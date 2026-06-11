"""Shared helpers for zimage VBench-2.0 standard video generation (480x800, 4s)."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[3]

DEFAULT_PROMPTS_DIR = Path(
    "/kwkj-k8s/cy123/workspace/VBench/VBench-2.0/prompts/prompt"
)
DEFAULT_PROMPTS_AUG_DIR = Path(
    "/kwkj-k8s/cy123/workspace/VBench/VBench-2.0/prompts/prompt_aug/VBench2_aug_prompt"
)

ALL_DIMENSIONS = [
    "Human_Anatomy",
    "Human_Identity",
    "Human_Clothes",
    "Diversity",
    "Composition",
    "Dynamic_Spatial_Relationship",
    "Dynamic_Attribute",
    "Motion_Order_Understanding",
    "Human_Interaction",
    "Complex_Landscape",
    "Complex_Plot",
    "Camera_Motion",
    "Motion_Rationality",
    "Instance_Preservation",
    "Mechanics",
    "Thermotics",
    "Material",
    "Multi-View_Consistency",
]


@dataclass
class VBenchTask:
    dimension: str
    prompt: str
    gen_prompt: str
    prompt_idx: int
    index: int
    other_dims: list[str]

    @property
    def filename(self) -> str:
        return f"{self.prompt[:180]}-{self.index}.mp4"

    @property
    def stem(self) -> str:
        return f"{self.prompt[:180]}-{self.index}"

    def vbench_path(self, outdir: Path) -> Path:
        return outdir / self.dimension / self.filename

    def infer_glob_pattern(self, outdir: Path, br_w: int, br_h: int, seconds: float) -> str:
        return f"{self.stem}_{seconds}s_{br_w}x{br_h}.mp4"

    def infer_path(self, outdir: Path, br_w: int, br_h: int, seconds: float) -> Path:
        return outdir / self.dimension / self.infer_glob_pattern(outdir, br_w, br_h, seconds)

    def seed(self, base_seed: int) -> int:
        return int(base_seed + self.prompt_idx * 100 + self.index) % (2**32)


def _read_prompt_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build_vbench_tasks(
    *,
    prompts_dir: Path,
    prompts_aug_dir: Path | None,
    dimensions: Iterable[str],
    limit_prompts: int = 0,
    use_aug: bool = True,
) -> list[VBenchTask]:
    """Global dedup on (prompt, index); first dimension generates, others copy."""
    prompt_to_first: dict[tuple[str, int], tuple[str, str, int]] = {}
    task_map: dict[tuple[str, str, int], list[str]] = {}

    for dimension in dimensions:
        txt_file = prompts_dir / f"{dimension}.txt"
        prompts = _read_prompt_lines(txt_file)
        if not prompts:
            continue

        prompts_aug: list[str] = []
        if use_aug and prompts_aug_dir:
            prompts_aug = _read_prompt_lines(prompts_aug_dir / f"{dimension}.txt")

        if limit_prompts > 0:
            prompts = prompts[:limit_prompts]
            if prompts_aug:
                prompts_aug = prompts_aug[:limit_prompts]

        iter_count = 20 if dimension == "Diversity" else 3

        for prompt_idx, prompt in enumerate(prompts):
            gen_prompt = prompt
            if prompts_aug and prompt_idx < len(prompts_aug):
                gen_prompt = prompts_aug[prompt_idx]

            for index in range(iter_count):
                key = (prompt, index)
                if key not in prompt_to_first:
                    prompt_to_first[key] = (dimension, gen_prompt, prompt_idx)
                    task_map[(dimension, prompt, index)] = []
                else:
                    first_dim, _, _ = prompt_to_first[key]
                    task_map[(first_dim, prompt, index)].append(dimension)

    tasks: list[VBenchTask] = []
    for (dimension, prompt, index), other_dims in task_map.items():
        _, gen_prompt, prompt_idx = prompt_to_first[(prompt, index)]
        tasks.append(
            VBenchTask(
                dimension=dimension,
                prompt=prompt,
                gen_prompt=gen_prompt,
                prompt_idx=prompt_idx,
                index=index,
                other_dims=sorted(other_dims),
            )
        )
    return tasks


def task_is_complete(
    task: VBenchTask,
    outdir: Path,
    *,
    min_bytes: int = 50_000,
) -> bool:
    main = task.vbench_path(outdir)
    if not main.is_file() or main.stat().st_size < min_bytes:
        return False
    for od in task.other_dims:
        p = outdir / od / task.filename
        if not p.is_file() or p.stat().st_size < min_bytes:
            return False
    return True


def filter_active_tasks(
    tasks: list[VBenchTask],
    outdir: Path,
    *,
    force: bool = False,
    min_bytes: int = 50_000,
) -> list[VBenchTask]:
    if force:
        return tasks
    return [t for t in tasks if not task_is_complete(t, outdir, min_bytes=min_bytes)]


def copy_to_other_dims(task: VBenchTask, outdir: Path) -> None:
    src = task.vbench_path(outdir)
    for od in task.other_dims:
        dst = outdir / od / task.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)


def publish_infer_mp4_to_vbench(
    task: VBenchTask,
    outdir: Path,
    *,
    br_w: int,
    br_h: int,
    seconds: float,
    min_bytes: int = 50_000,
) -> Path:
    """Move pipeline output native_*_4.0s_*.mp4 to VBench canonical name."""
    dim_dir = outdir / task.dimension
    expected = task.infer_path(outdir, br_w, br_h, seconds)
    vbench = task.vbench_path(outdir)
    vbench.parent.mkdir(parents=True, exist_ok=True)

    src: Path | None = None
    if expected.is_file() and expected.stat().st_size >= min_bytes:
        src = expected
    else:
        hits = sorted(dim_dir.glob(f"{task.stem}_*{seconds}s_{br_w}x{br_h}.mp4"))
        for p in hits:
            if p.is_file() and p.stat().st_size >= min_bytes:
                src = p
                break

    if src is None:
        raise FileNotFoundError(f"no infer mp4 for {task.dimension}/{task.filename}")

    if vbench.resolve() != src.resolve():
        if vbench.is_file():
            vbench.unlink()
        shutil.move(str(src), str(vbench))
    return vbench


def write_tasks_jsonl(path: Path, tasks: list[VBenchTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def read_tasks_jsonl(path: Path) -> list[VBenchTask]:
    tasks: list[VBenchTask] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        tasks.append(
            VBenchTask(
                dimension=d["dimension"],
                prompt=d["prompt"],
                gen_prompt=d["gen_prompt"],
                prompt_idx=int(d["prompt_idx"]),
                index=int(d["index"]),
                other_dims=list(d.get("other_dims") or []),
            )
        )
    return tasks


def write_task_meta(outdir: Path, *, all_tasks: list[VBenchTask], active: list[VBenchTask]) -> None:
    meta = {
        "all_unique_tasks": len(all_tasks),
        "active_tasks": len(active),
        "dimensions": sorted({t.dimension for t in all_tasks}),
        "outdir": str(outdir),
    }
    (outdir / "vbench2_task_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
