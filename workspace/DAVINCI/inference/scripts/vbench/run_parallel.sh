#!/usr/bin/env bash
# zimage ckpt_best -> VBench-2.0 标准目录（长驻 native worker，无 torchrun）
#
# 448×256（旧默认，快速对比）:
#   bash inference/scripts/vbench/run_parallel.sh
#
# 480×800（竖屏全量）:
#   BR_WIDTH=480 BR_HEIGHT=800 OUTDIR=output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s \
#     bash inference/scripts/vbench/run_parallel.sh
#
# 冒烟:
#   LIMIT_PROMPTS=1 DIMENSIONS=Human_Anatomy bash inference/scripts/vbench/run_parallel.sh
#
set -euo pipefail

REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
cd "${REPO}"

PYTHON="${PYTHON:-/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export MAGI_DISABLE_SPATIAL_FIX="${MAGI_DISABLE_SPATIAL_FIX:-1}"
export MAGI_CANVAS_FIX="${MAGI_CANVAS_FIX:-0}"
export MAGI_T2V_FREEZE_AUDIO="${MAGI_T2V_FREEZE_AUDIO:-1}"

BR_WIDTH="${BR_WIDTH:-448}"
BR_HEIGHT="${BR_HEIGHT:-256}"
if [[ "${BR_WIDTH}" == "480" && "${BR_HEIGHT}" == "800" ]]; then
  DEFAULT_OUT="${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s"
else
  DEFAULT_OUT="${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_${BR_WIDTH}x${BR_HEIGHT}_4s"
fi

OUTDIR="${OUTDIR:-${DEFAULT_OUT}}"
PROMPTS_DIR="${PROMPTS_DIR:-/kwkj-k8s/cy123/workspace/VBench/VBench-2.0/prompts/prompt}"
PROMPTS_AUG_DIR="${PROMPTS_AUG_DIR:-/kwkj-k8s/cy123/workspace/VBench/VBench-2.0/prompts/prompt_aug/VBench2_aug_prompt}"
CKPT="${CKPT:-${REPO}/output_train_t2i_1w_480x800_v1/ckpt_best.pt}"
BASE="${BASE:-/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base}"
CONFIG="${CONFIG:-${REPO}/example/base/config_infer.json}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PER_GPU_WORKERS="${PER_GPU_WORKERS:-1}"
SEED="${SEED:-521}"
STEPS="${STEPS:-32}"
VIDEO_CFG="${VIDEO_CFG:-6.0}"
AUDIO_CFG="${AUDIO_CFG:-6.0}"
BLEND="${BLEND:-1.0}"
VIDEO_SECONDS="${VIDEO_SECONDS:-4.0}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-0}"
DIMENSIONS="${DIMENSIONS:-}"
LOG_DIR="${REPO}/inference/scripts/logs"
MAIN_LOG="${LOG_DIR}/vbench2_zimage_standard_parallel.log"
NEG_PROMPT="${NEG_PROMPT:-static, blurred details, subtitles, overall gray, worst quality, low quality, jpeg artifacts, ugly, deformed, disfigured, messy background}"
export MAGI_NEGATIVE_PROMPT="${MAGI_NEGATIVE_PROMPT:-$NEG_PROMPT}"

mkdir -p "${OUTDIR}" "${LOG_DIR}"

extra=()
[[ "${FORCE:-0}" == "1" ]] && extra+=(--force)
[[ "${NO_AUG:-0}" == "1" ]] && extra+=(--no-aug)
dim_args=()
[[ -n "${DIMENSIONS}" ]] && dim_args+=(--dimensions "${DIMENSIONS}")
limit_args=()
[[ "${LIMIT_PROMPTS}" != "0" ]] && limit_args+=(--limit-prompts "${LIMIT_PROMPTS}")

echo "[$(date '+%F %T')] VBench2 zimage native: ${BR_WIDTH}x${BR_HEIGHT} gpus=${GPUS} OUTDIR=${OUTDIR}" | tee -a "${MAIN_LOG}"

"${PYTHON}" -u "${REPO}/inference/scripts/vbench/run.py" \
  --repo "${REPO}" \
  --outdir "${OUTDIR}" \
  --prompts-dir "${PROMPTS_DIR}" \
  --prompts-aug-dir "${PROMPTS_AUG_DIR}" \
  --gpus "${GPUS}" \
  --per-gpu-workers "${PER_GPU_WORKERS}" \
  --base "${BASE}" \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --cfg "${VIDEO_CFG}" \
  --video-cfg "${VIDEO_CFG}" \
  --audio-cfg "${AUDIO_CFG}" \
  --blend "${BLEND}" \
  --seconds "${VIDEO_SECONDS}" \
  --br-width "${BR_WIDTH}" \
  --br-height "${BR_HEIGHT}" \
  "${dim_args[@]}" \
  "${limit_args[@]}" \
  "${extra[@]}" \
  2>&1 | tee -a "${MAIN_LOG}"

echo "[$(date '+%F %T')] DONE. Videos: ${OUTDIR}/<Dimension>/*.mp4" | tee -a "${MAIN_LOG}"
