#!/usr/bin/env bash
# 后台启动「生成 + 自动评测监控」（setsid，断开 Cursor/SSH 后续跑）
#
# 448×256（旧默认）:
#   bash inference/scripts/vbench/start_bg.sh
#
# 480×800:
#   BR_WIDTH=480 BR_HEIGHT=800 VIDEO_SPEC='480×800 / 4s' \
#     OUTDIR=output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s \
#     bash inference/scripts/vbench/start_bg.sh
#
# 强制重启:
#   RESTART=1 bash inference/scripts/vbench/start_bg.sh
#
set -euo pipefail

REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
LOG_DIR="${REPO}/inference/scripts/logs"
MONITOR_LOG="${LOG_DIR}/monitor_gen_then_eval_zimage.out"
MONITOR_PID_FILE="${LOG_DIR}/monitor_gen_then_eval_zimage.pid"

BR_WIDTH="${BR_WIDTH:-448}"
BR_HEIGHT="${BR_HEIGHT:-256}"
if [[ "${BR_WIDTH}" == "480" && "${BR_HEIGHT}" == "800" ]]; then
  DEFAULT_OUT="${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s"
else
  DEFAULT_OUT="${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_${BR_WIDTH}x${BR_HEIGHT}_4s"
fi
OUTDIR="${OUTDIR:-${DEFAULT_OUT}}"

mkdir -p "${LOG_DIR}"

stop_if_running() {
  local label="$1" pid_file="$2" pattern="$3"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid=$(cat "${pid_file}")
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[stop] ${label} PID=${pid}"
      kill "${pid}" 2>/dev/null || true
      sleep 2
      kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
  fi
  pkill -f "${pattern}" 2>/dev/null || true
}

if [[ "${RESTART:-0}" == "1" ]]; then
  echo "[$(date '+%F %T')] RESTART=1，停止旧进程 ..."
  stop_if_running "monitor" "${MONITOR_PID_FILE}" "vbench/monitor_gen_then_eval.sh"
  stop_if_running "generation" "${LOG_DIR}/vbench2_zimage_standard_parallel.pid" "vbench/run_parallel|vbench/worker|vbench/run.py"
  sleep 3
fi

if [[ -f "${MONITOR_PID_FILE}" ]]; then
  old_pid=$(cat "${MONITOR_PID_FILE}")
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "监控已在运行 PID=${old_pid}"
    echo "日志: tail -f ${MONITOR_LOG}"
    exit 0
  fi
fi

cd "${REPO}"
echo "[$(date '+%F %T')] 启动 zimage 生成+评测监控 (setsid) ${BR_WIDTH}x${BR_HEIGHT}" | tee -a "${MONITOR_LOG}"

setsid bash -c "
  export OUTDIR='${OUTDIR}'
  export BR_WIDTH=${BR_WIDTH}
  export BR_HEIGHT=${BR_HEIGHT}
  export VIDEO_SPEC='${VIDEO_SPEC:-${BR_WIDTH}×${BR_HEIGHT} / 4s}'
  export TOTAL=3209
  export POLL_SEC=60
  export MAX_RETRY_PASS=3
  exec bash inference/scripts/vbench/monitor_gen_then_eval.sh
" >>"${MONITOR_LOG}" 2>&1 < /dev/null &

echo $! >"${MONITOR_PID_FILE}"
echo "后台监控已启动 PID=$(cat "${MONITOR_PID_FILE}")"
echo "日志: tail -f ${MONITOR_LOG}"
echo "停止: RESTART=1 bash inference/scripts/vbench/start_bg.sh"
