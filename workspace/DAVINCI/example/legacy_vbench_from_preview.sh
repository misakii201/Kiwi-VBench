#!/usr/bin/env bash
#
# 用 legacy_zimage_inference.sh 同款推理参数，复跑 VBench preview 目录里的提示词。
# 默认读取 output_preview_t2i_1w_vbench_base/_workers/tasks_w*.json 中的
# gen_prompt / index / base_seed，输出目录结构与 VBench 一致：
#   {OUTDIR}/{dimension}/{prompt[:180]}-{index}.mp4
#
# 与 VBench worker 的差异（对齐 legacy 脚本）：
#   video_cfg / audio_cfg = 6.0（VBench 默认 5.0）
#   seconds               = 4.0（VBench 默认 5）
#   MAGI_NEGATIVE_PROMPT  = 短版
#   MAGI_DISABLE_SPATIAL_FIX=1, MAGI_CANVAS_FIX=0
#
# 用法：
#   GPU=4 bash example/legacy_vbench_from_preview.sh
#   GPUS=4,5,6,7 bash example/legacy_vbench_from_preview.sh   # 并行
#   PREVIEW_DIR=output_preview_t2i_1w_vbench_base OUTDIR=output_test_infer/legacy_vbench_ckptbest bash example/legacy_vbench_from_preview.sh
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MAGI_DISABLE_SPATIAL_FIX="${MAGI_DISABLE_SPATIAL_FIX:-1}"
export MAGI_CANVAS_FIX="${MAGI_CANVAS_FIX:-0}"

# ---- 可调参数 ----
PREVIEW_DIR="${PREVIEW_DIR:-output_preview_t2i_1w_vbench_base}"
OUTDIR="${OUTDIR:-output_test_infer/legacy_vbench_from_base_prompts}"
CKPT="${CKPT:-output_train_t2i_1w_480x800_v1/ckpt_best.pt}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base}"
CONFIG="${CONFIG:-example/base/config_infer.json}"
NEG_PROMPT="${NEG_PROMPT:-static, blurred details, subtitles, overall gray, worst quality, low quality, jpeg artifacts, ugly, deformed, disfigured, messy background}"
STEPS="${STEPS:-32}"
VIDEO_CFG="${VIDEO_CFG:-6.0}"
AUDIO_CFG="${AUDIO_CFG:-6.0}"
SECONDS_LEN="${SECONDS_LEN:-4.0}"
WIDTH="${WIDTH:-480}"
HEIGHT="${HEIGHT:-800}"
GPUS="${GPUS:-${GPU:-4}}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29710}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$OUTDIR"

TASKS_JSON="$(mktemp /tmp/legacy_vbench_tasks.XXXXXX.json)"
"$PYTHON" - "$PREVIEW_DIR" "$TASKS_JSON" <<'PY'
import glob
import json
import sys
from pathlib import Path

preview_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
workers = sorted((preview_dir / "_workers").glob("tasks_w*.json"))
if not workers:
    raise SystemExit(f"未找到任务文件: {preview_dir}/_workers/tasks_w*.json")

seen = set()
tasks = []
base_seed = 521
for wf in workers:
    payload = json.loads(wf.read_text())
    base_seed = int(payload.get("base_seed", base_seed))
    for t in payload["tasks"]:
        key = (t["dimension"], t["prompt"], int(t["index"]))
        if key in seen:
            continue
        seen.add(key)
        tasks.append({
            "dimension": t["dimension"],
            "prompt": t["prompt"],
            "gen_prompt": t["gen_prompt"],
            "prompt_idx": int(t["prompt_idx"]),
            "index": int(t["index"]),
            "base_seed": base_seed,
        })

tasks.sort(key=lambda x: (x["dimension"], x["prompt"], x["index"]))
out_path.write_text(json.dumps({"base_seed": base_seed, "tasks": tasks}, ensure_ascii=False, indent=2))
print(f"[tasks] {len(tasks)} unique tasks from {len(workers)} worker json(s), base_seed={base_seed}")
for i, t in enumerate(tasks):
    seed = (t["base_seed"] + t["prompt_idx"] * 100 + t["index"]) % (2**32)
    print(f"  [{i}] {t['dimension']}/{t['prompt'][:40]}... index={t['index']} seed={seed}")
PY

run_one_task() {
  local gpu="$1" task_idx="$2" master_port="$3"
  local dim prompt gen_prompt prompt_idx index seed filename dim_dir out_mp4 tmpdir tmp_prefix produced

  eval "$("$PYTHON" - "$TASKS_JSON" "$task_idx" <<'PY'
import json, shlex, sys
t = json.loads(open(sys.argv[1]).read())["tasks"][int(sys.argv[2])]
seed = (t["base_seed"] + t["prompt_idx"] * 100 + t["index"]) % (2**32)
fields = {
    "dim": t["dimension"],
    "prompt": t["prompt"],
    "gen_prompt": t["gen_prompt"],
    "prompt_idx": t["prompt_idx"],
    "index": t["index"],
    "seed": seed,
}
for k, v in fields.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
)"

  filename="${prompt:0:180}-${index}.mp4"
  dim_dir="${OUTDIR}/${dim}"
  out_mp4="${dim_dir}/${filename}"
  mkdir -p "$dim_dir"

  if [[ "$SKIP_EXISTING" == "1" && -f "$out_mp4" ]]; then
    echo "[skip] GPU${gpu} task${task_idx}: ${out_mp4}"
    return 0
  fi

  tmpdir="$(mktemp -d "/tmp/legacy_vbench_${task_idx}_XXXXXX")"
  tmp_prefix="${tmpdir}/clip"

  echo "=============================================================="
  echo "[run] GPU=${gpu} task=${task_idx} ${dim}/${filename}"
  echo "      seed=${seed} cfg=${VIDEO_CFG} seconds=${SECONDS_LEN}"
  echo "      prompt=${gen_prompt:0:100}..."
  echo "=============================================================="

  if CUDA_VISIBLE_DEVICES="$gpu" \
     MASTER_PORT="$master_port" \
     MAGI_NEGATIVE_PROMPT="$NEG_PROMPT" \
     MAGI_T2V_FREEZE_AUDIO=1 \
     "$PYTHON" -u inference/test_infer_seedance.py \
       --config-load-path "$CONFIG" \
       --base_ckpt_dir "$BASE_CKPT_DIR" \
       --ckpt_dir "$CKPT" \
       --ckpt_blend_alpha 1.0 \
       --generate \
       --device cuda \
       --amp_dtype bf16 \
       --num_inference_steps "$STEPS" \
       --video_cfg_scale "$VIDEO_CFG" \
       --audio_cfg_scale "$AUDIO_CFG" \
       --seconds "$SECONDS_LEN" \
       --seed "$seed" \
       --freeze_audio \
       --br_width "$WIDTH" --br_height "$HEIGHT" \
       --prompt "$gen_prompt" \
       --save_path_prefix "$tmp_prefix" \
     2>&1 | tee "${tmpdir}/infer.log"; then
    :
  else
    echo "[fail] GPU${gpu} task${task_idx}, log: ${tmpdir}/infer.log"
    return 1
  fi

  produced="$(ls -t "${tmp_prefix}"*.mp4 2>/dev/null | head -1 || true)"
  if [[ -z "$produced" || ! -f "$produced" ]]; then
    echo "[fail] 未找到输出 mp4, log: ${tmpdir}/infer.log"
    return 1
  fi
  mv -f "$produced" "$out_mp4"
  rm -rf "$tmpdir"
  echo "[ok] -> ${out_mp4}"
}

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"
NUM_TASKS="$("$PYTHON" -c "import json; print(len(json.load(open('$TASKS_JSON'))['tasks']))")"

echo "[plan] ${NUM_TASKS} tasks on GPU(s): ${GPUS}"
echo "[out]  ${OUTDIR}/"
echo "[ckpt] ${CKPT}"

pids=()
fail=0
for ((task_idx= 0; task_idx < NUM_TASKS; task_idx++)); do
  gpu="${GPU_ARR[$((task_idx % NUM_GPUS))]}"
  port=$((MASTER_PORT_BASE + task_idx))

  if [[ "$NUM_GPUS" -gt 1 ]]; then
    run_one_task "$gpu" "$task_idx" "$port" &
    pids+=($!)
    # 避免多进程同时加载大模型 OOM：每卡串行，不同卡并行
    if (( (task_idx + 1) % NUM_GPUS == 0 )); then
      for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
      done
      pids=()
    fi
  else
    run_one_task "$gpu" "$task_idx" "$port" || fail=1
  fi
done

if ((${#pids[@]} > 0)); then
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
fi

rm -f "$TASKS_JSON"

if [[ "$fail" -ne 0 ]]; then
  echo "[done] 部分任务失败，请检查日志"
  exit 1
fi
echo "[done] 全部完成: ${OUTDIR}/"
