#!/usr/bin/env bash
#
# legacy 推理 8 卡生成 VBench 18 维视频（默认仅生成，测评需手动开启）
#
# 用法：
#   bash example/run_t2i_1w_vbench_legacy_ckptbest.sh                    # 仅生成（默认）
#   bash example/run_t2i_1w_vbench_legacy_ckptbest.sh --with-eval       # 生成 + 测评
#   EVAL_ONLY=1 bash example/run_t2i_1w_vbench_legacy_ckptbest.sh        # 仅测评
#   GENERATE_ONLY=1 bash example/run_t2i_1w_vbench_legacy_ckptbest.sh    # 同默认
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WITH_EVAL=0
for arg in "$@"; do
  [[ "$arg" == "--with-eval" ]] && WITH_EVAL=1
done

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
VBENCH_CONFIG="${VBENCH_CONFIG:-${PROJECT_ROOT}/API/Vbench/config_t2i_1w_vbench_legacy_ckptbest.yaml}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/output_vbench_t2i_1w_legacy_ckptbest}"
EVAL_OUT="${EVAL_OUT:-${PROJECT_ROOT}/output_vbench_t2i_1w_legacy_ckptbest_eval}"
LOG="${OUTDIR}/generate.log"

# 默认只生成；显式 --with-eval 或 EVAL_ONLY=1 时才跑测评
GENERATE_ONLY="${GENERATE_ONLY:-1}"
if [[ "$WITH_EVAL" == "1" ]]; then
  GENERATE_ONLY=0
fi

mkdir -p "$OUTDIR"

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "=== Phase 1: 8-GPU legacy VBench generation (worker, base+finetune load once) ==="
  echo "  config: ${VBENCH_CONFIG}"
  echo "  output: ${OUTDIR}"
  echo "  log:    ${LOG}"
  GPUS="$GPUS" VBENCH_CONFIG="$VBENCH_CONFIG" OUTDIR="$OUTDIR" \
    bash example/legacy_vbench_full.sh 2>&1 | tee -a "$LOG"
fi

if [[ "$GENERATE_ONLY" == "1" && "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "[done] 仅生成模式，跳过测评。手动测评："
  echo "  EVAL_ONLY=1 bash example/run_t2i_1w_vbench_legacy_ckptbest.sh"
  exit 0
fi

if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  echo "=== Phase 2: VBench-2.0 18-dimension evaluation (manual) ==="
fi

echo "=== Phase 2: VBench-2.0 18-dimension evaluation ==="
mkdir -p "$EVAL_OUT"
VIDEOS_ROOT="$OUTDIR" OUTPUT_ROOT="$EVAL_OUT" GPUS="$GPUS" \
  bash example/evaluate_t2i_1w_vbench_legacy_ckptbest.sh 2>&1 | tee -a "${EVAL_OUT}/eval_master.log"

echo "[done] videos: ${OUTDIR}"
echo "[done] eval:   ${EVAL_OUT}"
