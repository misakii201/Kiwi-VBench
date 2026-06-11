#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-6009}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export WORLD_SIZE="$((GPUS_PER_NODE * NNODES))"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DISTRIBUTED_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${GPUS_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

TS="$(date '+%Y%m%d_%H%M%S')"

for IDX in 01 02 03 04; do
  PROMPT_FILE="example/assets/prompt${IDX}.txt"
  IMAGE_FILE="example/assets/image${IDX}.png"
  if [[ ! -f "${IMAGE_FILE}" ]]; then
    IMAGE_FILE="example/assets/image02.png"
  fi

  python -m torch.distributed.run ${DISTRIBUTED_ARGS} inference/pipeline/entry.py \
    --config-load-path example/base/config.json \
    --prompt "$(<"${PROMPT_FILE}")" \
    --image_path "${IMAGE_FILE}" \
    --seconds 5 \
    --br_width 256 \
    --br_height 448 \
    --output_path "output_example_base_${TS}_p${IDX}" \
    2>&1 | tee "log_example_base_${TS}_p${IDX}.log"
done
