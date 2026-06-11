#!/usr/bin/env bash
# 480×800 VBench 全量生成 + 可选评测（archive 入口，对齐旧 run_eval_zimage_480x800_ckpt_best.sh）
set -euo pipefail

REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
ACTION="${1:-start}"

case "$ACTION" in
  start|generate)
    BR_WIDTH=480 BR_HEIGHT=800 \
      OUTDIR="${OUTDIR:-${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s}" \
      bash "${REPO}/inference/scripts/vbench/run_parallel.sh"
    ;;
  eval|evaluate)
    VIDEOS_ROOT="${VIDEOS_ROOT:-${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s}" \
      bash "${REPO}/inference/scripts/vbench/evaluate.sh"
    ;;
  all)
    BR_WIDTH=480 BR_HEIGHT=800 bash "${REPO}/inference/scripts/vbench/start_bg.sh"
    ;;
  *)
    echo "Usage: $0 {start|eval|all}"
    exit 1
    ;;
esac
