#!/usr/bin/env bash
# 用 test_infer_zimage_native.py 跑 3 条 prompt 冒烟测试
set -euo pipefail

PROJECT_ROOT="/kwkj-k8s/cy123/workspace/DAVINCI"
cd "$PROJECT_ROOT"

PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MAGI_DISABLE_SPATIAL_FIX=1
export MAGI_CANVAS_FIX=0
export MAGI_T2V_FREEZE_AUDIO=1
export MAGI_NEGATIVE_PROMPT="static, blurred details, subtitles, overall gray, worst quality, low quality, jpeg artifacts, ugly, deformed, disfigured, messy background"

OUTDIR="${OUTDIR:-output_test_infer/native_prompt_smoke}"
CKPT="${CKPT:-output_train_t2i_1w_480x800_v1/ckpt_best.pt}"
BASE="${BASE:-/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base}"
CONFIG="${CONFIG:-example/base/config_infer.json}"
MANIFEST="${MANIFEST:-dataset/t2i_1w_480x800_f1/latent_manifest_train.jsonl}"
LOGDIR="${OUTDIR}/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

PROMPT0="$("$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(json.loads(f.readline())["prompt"])
PY
)"

run_one() {
  local gpu="$1" seed="$2" tag="$3" prompt="$4"
  echo "[launch] gpu=${gpu} tag=${tag}"
  CUDA_VISIBLE_DEVICES="$gpu" MASTER_PORT="$((29500 + gpu * 10))" \
  "$PYTHON" -u inference/test_infer_zimage_native.py \
    --config-load-path "$CONFIG" \
    --base_ckpt_dir "$BASE" \
    --ckpt_dir "$CKPT" \
    --ckpt_blend_alpha 1.0 \
    --device cuda:0 \
    --amp_dtype bf16 \
    --num_inference_steps 32 \
    --video_cfg_scale 6.0 \
    --audio_cfg_scale 6.0 \
    --seconds 4.0 \
    --seed "$seed" \
    --freeze_audio \
    --br_width 480 --br_height 800 \
    --prompt "$prompt" \
    --save_path_prefix "${OUTDIR}/${tag}" \
    > "${LOGDIR}/gpu${gpu}_${tag}.log" 2>&1 &
  echo $! > "${LOGDIR}/gpu${gpu}_${tag}.pid"
}

run_one 0 42 "manifest0" "$PROMPT0"
run_one 1 521 "vbench_yoga" "A man is doing yoga."
run_one 2 621 "vbench_dog" "A dog is on the left of an apple, then the dog runs to the front of the apple."

echo "3 jobs started on GPU 0,1,2. Logs: ${LOGDIR}/"
wait
echo "=== results ==="
find "$OUTDIR" -name '*.mp4' -printf '%s %p\n' | sort -k2
