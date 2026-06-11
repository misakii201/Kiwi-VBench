#!/usr/bin/env bash
# VBench-2.0 full run: t2i_1w ckpt_step_3600, native 480x800, 8-GPU long-lived workers
set -euo pipefail

REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
cd "${REPO}"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MAGI_DISABLE_SPATIAL_FIX=1
export MAGI_CANVAS_FIX=0
export MAGI_T2V_FREEZE_AUDIO=1
export MAGI_NEGATIVE_PROMPT="static, blurred details, subtitles, overall gray, worst quality, low quality, jpeg artifacts, ugly, deformed, disfigured, messy background"

BR_WIDTH=480
BR_HEIGHT=800
OUTDIR="${OUTDIR:-${REPO}/output_vbench_t2i_1w_step3600_480x800_4s}"
CKPT="${CKPT:-${REPO}/output_train_t2i_1w_480x800_v1/ckpt_step_3600.pt}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-521}"
STEPS=32
BLEND=0.9
VIDEO_CFG=2.5
AUDIO_CFG=1.0
VIDEO_SECONDS=4.0

# smoke: LIMIT_PROMPTS=1 DIMENSIONS=Human_Anatomy bash example/run_vbench_t2i_step3600_480x800.sh

exec env \
  BR_WIDTH="${BR_WIDTH}" BR_HEIGHT="${BR_HEIGHT}" \
  OUTDIR="${OUTDIR}" CKPT="${CKPT}" GPUS="${GPUS}" \
  SEED="${SEED}" STEPS="${STEPS}" BLEND="${BLEND}" \
  VIDEO_CFG="${VIDEO_CFG}" AUDIO_CFG="${AUDIO_CFG}" VIDEO_SECONDS="${VIDEO_SECONDS}" \
  bash "${REPO}/inference/scripts/vbench/run_parallel.sh" "$@"
