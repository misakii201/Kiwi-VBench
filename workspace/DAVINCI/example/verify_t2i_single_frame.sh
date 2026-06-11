#!/usr/bin/env bash
#
# 单帧出图 vs 多帧视频 —— 时序退化快速验证
#
# 原理（见 inference/pipeline/video_generate.py）：
#   num_frames   = round(seconds * fps) + 1
#   latent_length = (num_frames - 1) // 4 + 1
#   => seconds=0  -> num_frames=1 -> latent_length=1  （等于训练时的 T=1 单帧）
#   => seconds=4  -> 多帧视频
#
# 控制变量：同一 ckpt / 同一 prompt / 同一 seed / 同一 cfg，只改 seconds。
#   - 单帧清晰、多帧有光斑  => 时序退化（ckpt 学到了图，但多帧时序被削弱）
#   - 单帧也有光斑          => 权重/latent 真坏了，不是时序问题
#
# 用法：
#   bash example/verify_t2i_single_frame.sh                 # 验证 ckpt_best
#   CKPT=output_train_t2i_1w_480x800_v1/ckpt_step_5200.pt \
#     bash example/verify_t2i_single_frame.sh
#   WITH_BASE=1 bash example/verify_t2i_single_frame.sh     # 额外跑一份 base 做对照
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---- 可调参数 ----
CKPT="${CKPT:-output_train_t2i_1w_480x800_v1/ckpt_best.pt}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base}"
CONFIG="${CONFIG:-example/base/config_infer.json}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
VIDEO_CFG="${VIDEO_CFG:-6.0}"
AUDIO_CFG="${AUDIO_CFG:-6.0}"
STEPS="${STEPS:-32}"
WIDTH="${WIDTH:-480}"
HEIGHT="${HEIGHT:-800}"
VIDEO_SECONDS="${VIDEO_SECONDS:-4.0}"
OUTDIR="${OUTDIR:-output_verify_t2i_single_frame}"
MANIFEST="${MANIFEST:-dataset/t2i_1w_480x800_f1/latent_manifest_train.jsonl}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
WITH_BASE="${WITH_BASE:-0}"
NEG_PROMPT="${NEG_PROMPT:-static, blurred details, subtitles, overall gray, worst quality, low quality, jpeg artifacts, ugly, deformed, disfigured, messy background}"

mkdir -p "$OUTDIR"

# ---- 取训练同分布 prompt（避免 VBench 短句这种 OOD 干扰）----
PROMPT_FILE="${OUTDIR}/prompt.txt"
if [[ -n "${PROMPT:-}" ]]; then
  printf '%s' "$PROMPT" > "$PROMPT_FILE"
else
  "$PYTHON" - "$MANIFEST" "$PROMPT_INDEX" > "$PROMPT_FILE" <<'PY'
import json, sys
manifest, idx = sys.argv[1], int(sys.argv[2])
with open(manifest) as f:
    for i, line in enumerate(f):
        if i == idx:
            sys.stdout.write(json.loads(line)["prompt"])
            break
PY
fi
echo "[prompt] $(head -c 160 "$PROMPT_FILE")..."

run_one() {
  # $1=tag  $2=ckpt_arg(空=base)  $3=seconds  $4=base_dir
  local tag="$1" ckpt_arg="$2" secs="$3" base_dir="$4"
  local prefix="${OUTDIR}/${tag}"
  echo "=============================================================="
  echo "[gen] tag=${tag} seconds=${secs} ckpt=${ckpt_arg:-<base>}"
  echo "=============================================================="
  local ckpt_opt=()
  [[ -n "$ckpt_arg" ]] && ckpt_opt=(--ckpt_dir "$ckpt_arg")
  CUDA_VISIBLE_DEVICES="$GPU" \
  MAGI_T2V_FREEZE_AUDIO=1 \
  MAGI_NEGATIVE_PROMPT="$NEG_PROMPT" \
  MAGI_DISABLE_SPATIAL_FIX="${MAGI_DISABLE_SPATIAL_FIX:-1}" \
  MAGI_CANVAS_FIX="${MAGI_CANVAS_FIX:-0}" \
  "$PYTHON" -u inference/test_infer_seedance.py \
    --config-load-path "$CONFIG" \
    --base_ckpt_dir "$base_dir" \
    "${ckpt_opt[@]}" \
    --ckpt_blend_alpha 1.0 \
    --generate \
    --device cuda \
    --amp_dtype bf16 \
    --num_inference_steps "$STEPS" \
    --video_cfg_scale "$VIDEO_CFG" \
    --audio_cfg_scale "$AUDIO_CFG" \
    --freeze_audio \
    --seconds "$secs" \
    --br_width "$WIDTH" --br_height "$HEIGHT" \
    --seed "$SEED" \
    --prompt_file "$PROMPT_FILE" \
    --save_path_prefix "$prefix"

  # 把生成的 mp4 第一帧抽成 png，方便直接看单帧画质
  local mp4
  mp4="$(ls -t "${prefix}"_*.mp4 2>/dev/null | head -1 || true)"
  if [[ -n "$mp4" ]]; then
    ffmpeg -y -loglevel error -i "$mp4" -frames:v 1 "${prefix}_frame0.png" || true
    echo "[ok] ${tag}: ${mp4}"
    echo "     frame0 -> ${prefix}_frame0.png"
  else
    echo "[warn] ${tag}: 没找到输出 mp4，检查上面的日志"
  fi
}

# ---- finetune ckpt：单帧 + 多帧 ----
run_one "ft_singleframe" "$CKPT"  0.0              "$BASE_CKPT_DIR"
run_one "ft_video"       "$CKPT"  "$VIDEO_SECONDS" "$BASE_CKPT_DIR"

# ---- 可选：base 对照 ----
if [[ "$WITH_BASE" == "1" ]]; then
  run_one "base_singleframe" "" 0.0              "$BASE_CKPT_DIR"
  run_one "base_video"       "" "$VIDEO_SECONDS" "$BASE_CKPT_DIR"
fi

echo
echo "=============================================================="
echo "完成。结果在: ${OUTDIR}/"
echo "  单帧（finetune）: ${OUTDIR}/ft_singleframe_*.mp4  +  ft_singleframe_frame0.png"
echo "  多帧（finetune）: ${OUTDIR}/ft_video_*.mp4        +  ft_video_frame0.png"
echo
echo "判读："
echo "  - 单帧清晰、多帧有光斑  => 时序退化（ckpt 学到了图，多帧时序被削弱）"
echo "  - 单帧也有光斑          => 权重/latent 真坏了，不是时序问题"
echo "=============================================================="
