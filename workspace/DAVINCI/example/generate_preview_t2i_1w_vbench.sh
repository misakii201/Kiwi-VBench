#!/usr/bin/env bash
# 用 ckpt_best.pt + VBench 提示词生成预览视频（默认 6 条）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
CONFIG="${PROJECT_ROOT}/API/Vbench/config_t2i_1w_preview.yaml"
OUTDIR="${PROJECT_ROOT}/output_preview_t2i_1w_vbench"
LOG="${OUTDIR}/generate.log"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$(dirname "$PYTHON"):${PATH}"

mkdir -p "$OUTDIR"

echo "Starting preview generation..."
echo "  config: ${CONFIG}"
echo "  output: ${OUTDIR}"
echo "  log:    ${LOG}"

nohup "$PYTHON" "${PROJECT_ROOT}/API/Vbench/run_vbench2_davinci.py" \
  --config "$CONFIG" \
  >> "$LOG" 2>&1 &

echo $! > "${OUTDIR}/generate.pid"
echo "Started (pid=$(cat "${OUTDIR}/generate.pid"))."
echo "Monitor: tail -f ${LOG}"
