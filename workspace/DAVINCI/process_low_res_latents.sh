#!/bin/bash
# =============================================================================
# DAVINCI low_res_preprocessed 视频/音频向量化 — 6 GPU 并行 (GPU 0-5)
#
# 思路同 prepare_t2i_latents.py / prepare_seedance_dataset.py:
#   1) 生成 manifest
#   2) 6 卡分片 encode_only 写 latents/video + latents/audio
#   3) 汇总 latent_manifest_*.jsonl
#
# 用法:
#   bash /kwkj-k8s/cy123/workspace/DAVINCI/process_low_res_latents.sh
#   bash .../process_low_res_latents.sh --num-gpus 6
# =============================================================================

set -euo pipefail

DAVINCI_ROOT="/kwkj-k8s/cy123/workspace/DAVINCI"
INPUT_CSV="/kwkj-k8s/cy123/LF_test/low_res_preprocessed_prompt/prompts.csv"
VIDEO_DIR="/kwkj-k8s/video_group_raw_files/low_res_preprocessed"
OUTPUT_DIR="/kwkj-k8s/cy123/workspace/DAVINCI/dataset/low_res_preprocessed"
VAE_MODEL_PATH="/home/zetyun/Sora2-mini/UniAVGen"
AUDIO_MODEL_PATH="/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/audio_model"
NUM_GPUS=6
NUM_FRAMES=121
PROGRESS_INTERVAL=15
LOG_DIR="${OUTPUT_DIR}/_run_logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            LOG_DIR="${OUTPUT_DIR}/_run_logs"
            shift 2
            ;;
        --input_csv)
            INPUT_CSV="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1" >&2
            echo "用法: $0 [--num-gpus N] [--output_dir PATH] [--input_csv PATH]" >&2
            exit 1
            ;;
    esac
done

format_duration() {
    local total_sec=$1
    printf '%02d:%02d:%02d' $((total_sec / 3600)) $(((total_sec % 3600) / 60)) $((total_sec % 60))
}

print_progress_line() {
    local done_count="$1"
    local total="$2"
    local elapsed_sec="$3"
    local pct="0.0"
    local eta_str=""
    if [ "${total}" -gt 0 ]; then
        pct=$(awk "BEGIN {printf \"%.1f\", ${done_count}*100/${total}}")
    fi
    if [ "${done_count}" -gt 0 ] && [ "${total}" -gt "${done_count}" ]; then
        local eta_sec=$((elapsed_sec * (total - done_count) / done_count))
        eta_str=" | 预计剩余 $(format_duration "${eta_sec}")"
    fi
    local audio_count
    audio_count=$(find "${OUTPUT_DIR}/latents/audio" -name "*.pt" 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进度: video ${done_count}/${total} (${pct}%) | audio ${audio_count} | 已用时 $(format_duration "${elapsed_sec}")${eta_str}"
}

if [ ! -f "${INPUT_CSV}" ]; then
    echo "错误: CSV 不存在 - ${INPUT_CSV}" >&2
    exit 1
fi
if [ ! -d "${VIDEO_DIR}" ]; then
    echo "错误: 视频目录不存在 - ${VIDEO_DIR}" >&2
    exit 1
fi

TOTAL_ROWS=$(($(wc -l < "${INPUT_CSV}") - 1))
mkdir -p "${OUTPUT_DIR}/latents/video" "${OUTPUT_DIR}/latents/audio" "${LOG_DIR}"

EXISTING_VIDEO=$(find "${OUTPUT_DIR}/latents/video" -name "*.pt" 2>/dev/null | wc -l)
EXISTING_AUDIO=$(find "${OUTPUT_DIR}/latents/audio" -name "*.pt" 2>/dev/null | wc -l)

echo "============================================"
echo "  DAVINCI low_res 视频/音频向量化"
echo "============================================"
echo "  CSV:         ${INPUT_CSV}"
echo "  视频目录:    ${VIDEO_DIR}"
echo "  数据行数:    ${TOTAL_ROWS}"
echo "  输出目录:    ${OUTPUT_DIR}"
echo "  VAE:         ${VAE_MODEL_PATH}"
echo "  Audio VAE:   ${AUDIO_MODEL_PATH}"
echo "  GPU 数量:    ${NUM_GPUS} (GPU 0-$((NUM_GPUS - 1)))"
echo "  帧数:        ${NUM_FRAMES}"
echo "  命名规则:    latents/video/{idx:06d}.pt / latents/audio/{idx:06d}.pt"
echo "  进度刷新:    每 ${PROGRESS_INTERVAL} 秒"
echo "  日志目录:    ${LOG_DIR}"
echo "  已有 video:  ${EXISTING_VIDEO}"
echo "  已有 audio:  ${EXISTING_AUDIO}"
echo "============================================"

cd "${DAVINCI_ROOT}"

echo ""
echo "[步骤 1/3] 生成 manifest..."
python3 prepare_low_res_latents.py \
    --input_csv "${INPUT_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --manifest_only \
    --copy_csv

KEPT_ROWS=$(python3 - <<PY
import json, os
meta_path = "${OUTPUT_DIR}/meta.json"
with open(meta_path, encoding="utf-8") as f:
    print(json.load(f)["kept_rows"])
PY
)

if [ "${KEPT_ROWS}" -le 0 ]; then
    echo "错误: 没有可用视频" >&2
    exit 1
fi

CHUNK_SIZE=$(python3 - <<PY
import math
print(math.ceil(${KEPT_ROWS} / ${NUM_GPUS}))
PY
)

echo ""
echo "[步骤 2/3] 启动 ${NUM_GPUS} 个 encode worker (每卡约 ${CHUNK_SIZE} 条)..."
ENCODE_PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((GPU_ID * CHUNK_SIZE))
    END=$((START + CHUNK_SIZE))
    if [ "${START}" -ge "${KEPT_ROWS}" ]; then
        echo "  GPU ${GPU_ID}: 无数据，跳过"
        continue
    fi
    if [ "${END}" -gt "${KEPT_ROWS}" ]; then
        END=${KEPT_ROWS}
    fi
    LOG="${LOG_DIR}/gpu${GPU_ID}.log"
    echo "  GPU ${GPU_ID}: index [${START}:${END}) -> ${LOG}"
    (
        CUDA_VISIBLE_DEVICES=${GPU_ID} \
        python3 prepare_low_res_latents.py \
            --input_csv "${INPUT_CSV}" \
            --output_dir "${OUTPUT_DIR}" \
            --vae_model_path "${VAE_MODEL_PATH}" \
            --audio_model_path "${AUDIO_MODEL_PATH}" \
            --num_frames "${NUM_FRAMES}" \
            --encode_only \
            --skip_existing \
            --start_index "${START}" \
            --end_index "${END}" \
            --gpu_id 0 \
            --pad_if_smaller \
        > "${LOG}" 2>&1
    ) &
    ENCODE_PIDS+=($!)
done

echo ""
echo "  ${#ENCODE_PIDS[@]} 个 worker 已启动，实时进度如下："
START_TS=$(date +%s)
while true; do
    CUR=$(find "${OUTPUT_DIR}/latents/video" -name "*.pt" 2>/dev/null | wc -l)
    ELAPSED=$(($(date +%s) - START_TS))
    print_progress_line "${CUR}" "${KEPT_ROWS}" "${ELAPSED}"

    RUNNING=0
    for pid in "${ENCODE_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            RUNNING=1
            break
        fi
    done
    [ "${RUNNING}" -eq 0 ] && break
    sleep "${PROGRESS_INTERVAL}"
done
echo ""

FAILED=0
for GPU_ID in "${!ENCODE_PIDS[@]}"; do
    if wait "${ENCODE_PIDS[$GPU_ID]}"; then
        echo "  GPU ${GPU_ID}: 完成"
    else
        echo "  GPU ${GPU_ID}: 失败 (日志: ${LOG_DIR}/gpu${GPU_ID}.log)"
        FAILED=$((FAILED + 1))
    fi
done

if [ "${FAILED}" -gt 0 ]; then
    echo "错误: ${FAILED} 个 GPU worker 失败" >&2
    exit 1
fi

echo ""
echo "[步骤 3/3] 汇总 latent_manifest..."
python3 prepare_low_res_latents.py \
    --input_csv "${INPUT_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --assemble_manifest_only \
    --num_frames "${NUM_FRAMES}"

FINAL_VIDEO=$(find "${OUTPUT_DIR}/latents/video" -name "*.pt" 2>/dev/null | wc -l)
FINAL_AUDIO=$(find "${OUTPUT_DIR}/latents/audio" -name "*.pt" 2>/dev/null | wc -l)

echo ""
echo "============================================"
echo "  完成"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  video latents: ${FINAL_VIDEO}"
echo "  audio latents: ${FINAL_AUDIO}"
echo "  manifest:    ${OUTPUT_DIR}/latent_manifest_all.jsonl"
echo "  训练示例:"
echo "    --manifest ${OUTPUT_DIR}/latent_manifest_train.jsonl"
echo "============================================"
