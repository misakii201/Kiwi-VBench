#!/usr/bin/env bash
# Base 模型对照生成（与 finetune 预览相同 VBench 提示词 / seed / 分辨率）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
CONFIG="${PROJECT_ROOT}/API/Vbench/config_t2i_1w_preview_base.yaml"
OUTDIR="${PROJECT_ROOT}/output_preview_t2i_1w_vbench_base"
LOG="${OUTDIR}/generate.log"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$(dirname "$PYTHON"):${PATH}"

mkdir -p "$OUTDIR"

echo "Starting BASE model preview generation..."
echo "  config: ${CONFIG}"
echo "  output: ${OUTDIR}"
echo "  log:    ${LOG}"

nohup "$PYTHON" "${PROJECT_ROOT}/API/Vbench/run_vbench2_davinci.py" \
  --config "$CONFIG" \
  >> "$LOG" 2>&1 &

echo $! > "${OUTDIR}/generate.pid"
echo "Started (pid=$(cat "${OUTDIR}/generate.pid"))."
echo "Monitor: tail -f ${LOG}"
