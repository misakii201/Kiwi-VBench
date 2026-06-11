#!/usr/bin/env bash
# VBench-2.0 全 18 维标准评测
set -euo pipefail

REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
VIDEOS_ROOT="${VIDEOS_ROOT:-${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${VIDEOS_ROOT}/vbench_eval}"

exec env \
  VIDEOS_ROOT="${VIDEOS_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  MODEL_NAME="${MODEL_NAME:-zimage_ckptbest_vbench_standard}" \
  bash "${REPO}/example/evaluate_t2i_1w_vbench_legacy_ckptbest.sh"
