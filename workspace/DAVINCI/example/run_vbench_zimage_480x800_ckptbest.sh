#!/usr/bin/env bash
# 兼容旧 OUTDIR 命名：480×800 VBench 全量
exec env \
  BR_WIDTH=480 BR_HEIGHT=800 \
  OUTDIR="${OUTDIR:-/kwkj-k8s/cy123/workspace/DAVINCI/output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s}" \
  bash "$(dirname "$0")/legacy_vbench_full.sh" "$@"
