#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
CONFIG="${PROJECT_ROOT}/API/Vbench/config_t2i_1w_preview_step3000.yaml"
OUTDIR="${PROJECT_ROOT}/output_preview_t2i_1w_vbench_step3000"
LOG="${OUTDIR}/generate.log"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$(dirname "$PYTHON"):${PATH}"

mkdir -p "$OUTDIR"

nohup "$PYTHON" "${PROJECT_ROOT}/API/Vbench/run_vbench2_davinci.py" \
  --config "$CONFIG" \
  >> "$LOG" 2>&1 &

echo $! > "${OUTDIR}/generate.pid"
echo "Started step3000 preview (pid=$(cat "${OUTDIR}/generate.pid"))."
echo "Output: ${OUTDIR}"
echo "Monitor: tail -f ${LOG}"
