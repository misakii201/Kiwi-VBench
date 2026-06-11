#!/usr/bin/env python3
"""
使用 DAVINCI (daVinci-MagiHuman) 推理流水线，按 VBench-2.0 提示词分维度批量生成评测视频。
设计上完全沿用 LTX2.3 评测脚本（run_vbench2_ltx2.py）的业务逻辑：
  - 全局提示词去重（同一 prompt 在多个维度复用，仅生成一次后跨维度 copy）
  - 断点自愈与补全：启动时扫描已存在视频，缺失则补全复用 / 仅生成尚未存在的样本
  - 子维度存储：{output_dir}/{dimension}/{prompt[:180]}-{index}.mp4
  - Diversity 维度生成 20 个样本 (index 0~19)；其它 17 个维度生成 3 个样本 (index 0~2)
  - 完美支持 Augmented Prompts（prompts_aug_dir）

与 LTX2.3 的差异：
  - 底层调用 DAVINCI 的 `MagiPipeline`，需要分布式（torchrun）启动；
    因此每个 worker 通过 subprocess 起 `torchrun --nproc_per_node=1`，
    每个 worker 独占一张 GPU 与一个 rendezvous 端口；
  - 模型在 worker 内只 build 一次，循环消化分配到的任务桶，避免重复初始化与重复 MagiCompiler 编译；
    worker 进程若 OOM/异常退出，主进程会按 vbench2.parallel.max_worker_restarts（默认 3）自动重启该槽位，已生成的 mp4 会被跳过。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_DAVINCI_ROOT = _SCRIPT_DIR.parent.parent
_WORKER_PY = _SCRIPT_DIR / "davinci_vbench2_worker.py"

ALL_DIMS = [
    "Human_Anatomy", "Human_Identity", "Human_Clothes", "Diversity", "Composition",
    "Dynamic_Spatial_Relationship", "Dynamic_Attribute", "Motion_Order_Understanding",
    "Human_Interaction", "Complex_Landscape", "Complex_Plot", "Camera_Motion",
    "Motion_Rationality", "Instance_Preservation", "Mechanics", "Thermotics",
    "Material", "Multi-View_Consistency",
]


def _load_run_context(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    vbench2_cfg = raw.get("vbench2") or {}
    prompts_dir = Path(vbench2_cfg["prompts_dir"]).expanduser().resolve()
    prompts_aug_dir_raw = vbench2_cfg.get("prompts_aug_dir")
    prompts_aug_dir = Path(prompts_aug_dir_raw).expanduser().resolve() if prompts_aug_dir_raw else None
    out_dir = Path(vbench2_cfg["output_dir"]).expanduser().resolve()
    base_seed = int(vbench2_cfg.get("base_seed", 42))
    limit_prompts = int(vbench2_cfg.get("limit_prompts", 0))

    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"未找到 VBench-2.0 prompts 目录: {prompts_dir}")

    dims = vbench2_cfg.get("dimensions") or []
    if not dims:
        dims = ALL_DIMS
    else:
        for d in dims:
            if d not in ALL_DIMS:
                raise ValueError(f"无效的 VBench-2.0 维度: {d}. 可选: {ALL_DIMS}")

    return raw, dims, prompts_dir, prompts_aug_dir, out_dir, base_seed, limit_prompts


def _parse_parallel_gpus(gpus_arg: str | None, parallel_cfg: dict | None) -> list[int] | None:
    if gpus_arg is not None and str(gpus_arg).strip():
        return [int(x.strip()) for x in str(gpus_arg).split(",") if x.strip()]
    if parallel_cfg and parallel_cfg.get("gpus") is not None:
        g = parallel_cfg["gpus"]
        if isinstance(g, list):
            return [int(x) for x in g]
        return [int(x.strip()) for x in str(g).split(",") if str(x).strip()]
    return None


def _parse_per_gpu_workers(cli_val: int | None, parallel_cfg: dict | None) -> int:
    if cli_val is not None and cli_val >= 1:
        return int(cli_val)
    if parallel_cfg and parallel_cfg.get("per_gpu_workers") is not None:
        return max(1, int(parallel_cfg["per_gpu_workers"]))
    return 1


def _parse_max_worker_restarts(cli_val: int | None, parallel_cfg: dict | None) -> int:
    """OOM 等导致 worker 退出后，额外允许的重启次数（不含首次启动）。默认 3 → 同一槽位最多 4 次进程。"""
    if cli_val is not None:
        return max(0, int(cli_val))
    if parallel_cfg and parallel_cfg.get("max_worker_restarts") is not None:
        return max(0, int(parallel_cfg["max_worker_restarts"]))
    return 3


def _build_task_universe(
    dims: list[str],
    prompts_dir: Path,
    prompts_aug_dir: Path | None,
    limit_prompts: int,
):
    """构造去重后的全部任务列表 + 跨维度复用映射。逻辑与 LTX2.3 完全一致。"""
    prompt_to_first_occurrence: dict = {}  # (prompt, index) -> (dimension, gen_prompt, prompt_idx)
    task_map: dict = {}  # (first_dimension, prompt, index) -> list[other_dimensions]

    for dimension in dims:
        txt_file = prompts_dir / f"{dimension}.txt"
        if not txt_file.is_file():
            logging.warning(f"维度提示词文件未找到，跳过: {txt_file}")
            continue
        with open(txt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        prompts_aug: list[str] = []
        if prompts_aug_dir:
            aug_txt_file = prompts_aug_dir / f"{dimension}.txt"
            if aug_txt_file.is_file():
                with open(aug_txt_file, "r", encoding="utf-8") as f:
                    prompts_aug = [line.strip() for line in f if line.strip()]

        if limit_prompts > 0:
            prompts = prompts[:limit_prompts]
            if prompts_aug:
                prompts_aug = prompts_aug[:limit_prompts]

        for prompt_idx, prompt in enumerate(prompts):
            iter_count = 20 if dimension == "Diversity" else 3
            gen_prompt = prompt
            if prompts_aug and prompt_idx < len(prompts_aug):
                gen_prompt = prompts_aug[prompt_idx]
            for index in range(iter_count):
                key = (prompt, index)
                if key not in prompt_to_first_occurrence:
                    prompt_to_first_occurrence[key] = (dimension, gen_prompt, prompt_idx)
                    task_map[(dimension, prompt, index)] = []
                else:
                    first_dim, _, _ = prompt_to_first_occurrence[key]
                    task_map[(first_dim, prompt, index)].append(dimension)

    all_tasks = []
    for (dimension, prompt, index), other_dims in task_map.items():
        _, gen_prompt, prompt_idx = prompt_to_first_occurrence[(prompt, index)]
        all_tasks.append({
            "dimension": dimension,
            "prompt": prompt,
            "gen_prompt": gen_prompt,
            "prompt_idx": prompt_idx,
            "index": index,
            "other_dims": other_dims,
        })
    return all_tasks


def _filter_active(all_tasks: list[dict], out_dir: Path) -> list[dict]:
    active: list[dict] = []
    for t in all_tasks:
        filename = f"{t['prompt'][:180]}-{t['index']}.mp4"
        main_exist = (out_dir / t["dimension"] / filename).is_file()
        all_exist = main_exist and all(
            (out_dir / od / filename).is_file() for od in t["other_dims"]
        )
        if not all_exist:
            active.append(t)
    return active


def _spawn_worker(
    *,
    worker_id: int,
    gpu: int,
    base_port: int,
    tasks_json: Path,
    config_path: Path,
    log_path: Path,
    t2v_freeze_audio: bool,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = str(base_port + worker_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    env["PYTHONPATH"] = f"{_DAVINCI_ROOT}:{env.get('PYTHONPATH', '')}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if t2v_freeze_audio:
        env["MAGI_T2V_FREEZE_AUDIO"] = "1"
    else:
        env.pop("MAGI_T2V_FREEZE_AUDIO", None)
    # 关掉 DAVINCI pipeline 内部的启发式"空间修复"——它本意是兜底 TurboVAE 偶尔
    # 产生的 vertical-stack / horizontal-repeat 输出，但对 Wan2.2 VAE 是误伤：
    # 会把一个区域裁出来再放大回原尺寸，肉眼表现为糊 + 重影。
    env.setdefault("MAGI_DISABLE_SPATIAL_FIX", "1")
    env.setdefault("MAGI_CANVAS_FIX", "0")
    env.setdefault("DAVINCI_DISABLE_MAGI_COMPILE", "1")
    env.setdefault("MAGI_DISABLE_COMPILE", "1")
    # 与官方 sr_540p / sr_1080p run.sh 对齐
    env.setdefault("CPU_OFFLOAD", "true")

    cmd = [
        "torchrun",
        "--nnodes=1",
        "--node_rank=0",
        "--nproc_per_node=1",
        f"--rdzv-endpoint=localhost:{base_port + worker_id}",
        "--rdzv-backend=c10d",
        str(_WORKER_PY),
        "--tasks-json", str(tasks_json),
        "--config-yaml", str(config_path),
        "--worker-id", str(worker_id),
    ]
    log_f = open(log_path, "ab", buffering=0)
    log_f.write(f"\n=== launch worker {worker_id} on GPU {gpu} (port {base_port + worker_id}) ===\n".encode())
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    proc._log_file = log_f  # type: ignore[attr-defined]
    return proc


def _run_worker_slot_with_retries(
    *,
    worker_id: int,
    gpu: int,
    base_port: int,
    tasks_json: Path,
    config_path: Path,
    log_path: Path,
    t2v_freeze_audio: bool,
    num_tasks: int,
    max_worker_restarts: int,
) -> int | None:
    """
    在同一 GPU 槽位上顺序启动 worker；失败则重启进程（新 CUDA context），
    已写入的 mp4 由 worker 内逻辑跳过。

    Returns:
        None 若最终成功；否则返回 worker_id（已用尽重启次数仍失败）。
    """
    max_attempts = max_worker_restarts + 1
    for attempt in range(max_attempts):
        if attempt > 0:
            logging.warning(
                "worker %d (GPU %s) 上次异常退出，正在第 %d 次进程重启（本槽位至多 %d 次重启）",
                worker_id,
                gpu,
                attempt,
                max_worker_restarts,
            )
        else:
            logging.info(
                "已启动 worker %d (GPU %s, %d tasks) 日志: %s",
                worker_id,
                gpu,
                num_tasks,
                log_path,
            )
        proc = _spawn_worker(
            worker_id=worker_id,
            gpu=int(gpu),
            base_port=base_port,
            tasks_json=tasks_json,
            config_path=config_path,
            log_path=log_path,
            t2v_freeze_audio=t2v_freeze_audio,
        )
        proc.wait()
        try:
            proc._log_file.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        if proc.returncode == 0:
            return None
        logging.error(
            "worker %d (GPU %s) 异常退出 returncode=%s pid=%s（详见 %s）",
            worker_id,
            gpu,
            proc.returncode,
            proc.pid,
            log_path,
        )
    return worker_id


def run_parallel(
    config_path: Path,
    gpu_ids: list[int],
    per_gpu_workers: int,
    max_worker_restarts: int | None = None,
) -> None:
    raw, dims, prompts_dir, prompts_aug_dir, out_dir, base_seed, limit_prompts = _load_run_context(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = _build_task_universe(dims, prompts_dir, prompts_aug_dir, limit_prompts)
    active_tasks = _filter_active(all_tasks, out_dir)

    logging.info(
        "VBench-2.0 实际去重后任务总数 %d；当前仍需生成 %d 个唯一样本。",
        len(all_tasks),
        len(active_tasks),
    )
    if not active_tasks:
        logging.info("所有视频已生成完毕，无需继续采样。")
        return

    num_workers = len(gpu_ids) * per_gpu_workers
    buckets: list[list[dict]] = [[] for _ in range(num_workers)]
    for i, task in enumerate(active_tasks):
        buckets[i % num_workers].append(task)

    parallel_cfg = (raw.get("vbench2") or {}).get("parallel") or {}
    base_port = int(parallel_cfg.get("base_master_port", 6010))
    t2v_freeze_audio = bool((raw.get("params") or {}).get("t2v_freeze_audio", True))
    restarts = (
        max(0, int(max_worker_restarts))
        if max_worker_restarts is not None
        else max(0, int(parallel_cfg.get("max_worker_restarts", 3)))
    )
    logging.info(
        "worker 异常时自动重启: 每槽位额外最多 %d 次重启（共最多 %d 次进程）",
        restarts,
        restarts + 1,
    )

    work_root = out_dir / "_workers"
    work_root.mkdir(parents=True, exist_ok=True)

    slot_specs: list[dict] = []
    wid = 0
    for gpu in gpu_ids:
        for _slot in range(per_gpu_workers):
            tasks = buckets[wid]
            if not tasks:
                wid += 1
                continue
            tasks_json = work_root / f"tasks_w{wid}.json"
            with open(tasks_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "worker_id": wid,
                        "gpu": int(gpu),
                        "base_seed": base_seed,
                        "out_dir": str(out_dir),
                        "tasks": tasks,
                    },
                    f,
                    ensure_ascii=False,
                )
            log_path = work_root / f"worker_{wid}.log"
            slot_specs.append(
                {
                    "worker_id": wid,
                    "gpu": int(gpu),
                    "tasks_json": tasks_json,
                    "log_path": log_path,
                    "num_tasks": len(tasks),
                }
            )
            wid += 1

    failed_workers: list[int] = []
    if not slot_specs:
        logging.warning("无有效 worker 槽位（任务桶均为空）。")
    else:
        max_workers = min(32, len(slot_specs))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _run_worker_slot_with_retries,
                    worker_id=s["worker_id"],
                    gpu=s["gpu"],
                    base_port=base_port,
                    tasks_json=s["tasks_json"],
                    config_path=config_path,
                    log_path=s["log_path"],
                    t2v_freeze_audio=t2v_freeze_audio,
                    num_tasks=s["num_tasks"],
                    max_worker_restarts=restarts,
                ): s["worker_id"]
                for s in slot_specs
            }
            for fut in as_completed(futures):
                bad_wid = fut.result()
                if bad_wid is not None:
                    failed_workers.append(int(bad_wid))

    if failed_workers:
        failed_workers.sort()
        raise RuntimeError(
            f"以下 worker 在耗尽重启次数后仍失败: {failed_workers}（详见 {work_root}/worker_*.log）"
        )

    logging.info("全部 worker 完成，视频生成完毕！输出根目录: %s", out_dir)


def run_single(config_path: Path, max_worker_restarts: int | None = None) -> None:
    """单卡兜底：仍然通过 torchrun 启动 1 个 worker（必须有分布式 init）。"""
    run_parallel(config_path, gpu_ids=[0], per_gpu_workers=1, max_worker_restarts=max_worker_restarts)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="DAVINCI 批量生成 VBench-2.0 评测用 mp4 视频（分维度、支持多卡并行）"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML，例如 config_davinci_vbench2.yaml")
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="物理 GPU 编号，逗号分隔，例如 0,1,2,3,4,5,6,7。不设则使用单卡模式。",
    )
    parser.add_argument(
        "--per-gpu-workers", type=int, default=None,
        help="每张物理 GPU 上启动的进程数。默认 1。",
    )
    parser.add_argument(
        "--max-worker-restarts", type=int, default=None,
        help="单个 worker 槽位 OOM/异常退出后的额外重启次数（不含首次）。不设则读 YAML vbench2.parallel.max_worker_restarts，默认 3。",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_preview = yaml.safe_load(f)
    parallel_cfg = (raw_preview.get("vbench2") or {}).get("parallel") or {}

    gpu_list = _parse_parallel_gpus(args.gpus, parallel_cfg)
    per_gpu = _parse_per_gpu_workers(args.per_gpu_workers, parallel_cfg)
    max_restarts = _parse_max_worker_restarts(args.max_worker_restarts, parallel_cfg)

    if gpu_list is None or len(gpu_list) == 0:
        logging.info("未指定 GPU，单卡模式 (GPU 0, per_gpu_workers=%d)", per_gpu)
        run_single(cfg_path, max_worker_restarts=max_restarts)
        return

    logging.info(
        "VBench-2.0 并行采样模式: GPUs=%s，每卡进程数=%s，总并发=%s",
        gpu_list, per_gpu, len(gpu_list) * per_gpu,
    )
    run_parallel(cfg_path, gpu_list, per_gpu, max_worker_restarts=max_restarts)


if __name__ == "__main__":
    main()
