#!/usr/bin/env bash
#
# legacy 推理 + VBench-2.0 全量生成（8 卡 native worker，每卡只加载一次 base+finetune）
# 底层：inference/scripts/vbench/run_parallel.sh → 长驻 native worker（无 torchrun）
#
# 用法：
#   bash example/legacy_vbench_full.sh
#   GPUS=0,1,2,3,4,5,6,7 bash example/legacy_vbench_full.sh
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/output_vbench_t2i_1w_legacy_ckptbest}"
CKPT="${CKPT:-${PROJECT_ROOT}/output_train_t2i_1w_480x800_v1/ckpt_best.pt}"
BASE="${BASE:-/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/example/base/config_infer.json}"
SEED="${SEED:-521}"

echo "[legacy_vbench_full] outdir=${OUTDIR}"
echo "[legacy_vbench_full] gpus=${GPUS}"
echo "[legacy_vbench_full] mode=native worker (base+finetune load once per GPU, no torchrun)"

exec env \
  GPUS="$GPUS" \
  OUTDIR="$OUTDIR" \
  CKPT="$CKPT" \
  BASE="$BASE" \
  CONFIG="$CONFIG" \
  SEED="$SEED" \
  BR_WIDTH="${BR_WIDTH:-480}" \
  BR_HEIGHT="${BR_HEIGHT:-800}" \
  bash "${PROJECT_ROOT}/inference/scripts/vbench/run_parallel.sh"
