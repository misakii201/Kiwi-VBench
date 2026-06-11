#!/usr/bin/env bash
# 仅后台启动 18 维评测
set -euo pipefail
REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
LOG_DIR="${REPO}/inference/scripts/logs"
mkdir -p "${LOG_DIR}"
EVAL_LOG="${LOG_DIR}/evaluate_zimage_vbench_standard_$(date '+%Y%m%d_%H%M%S').log"
setsid bash "${REPO}/inference/scripts/vbench/evaluate.sh" \
  >>"${EVAL_LOG}" 2>&1 < /dev/null &
echo "评测已后台启动 PID=$! 日志: ${EVAL_LOG}"
