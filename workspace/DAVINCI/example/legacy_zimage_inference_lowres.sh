#!/usr/bin/env bash
#
# ============================================================================
# 低分辨率推理 —— 历史还原（448×256 / 4s）
# ============================================================================
#
# 与 legacy_zimage_inference.sh（480×800 训练分辨率）相对：
#   - 分辨率：448×256（对齐 example/base/run.sh / base VBench 低分规格）
#   - 其余参数相同：cfg=6.0、seconds=4、seed=42、freeze_audio、短版负向词
#
# 来源（旧项目 /kwkj-k8s/davinci/LJH/daVinci-MagiHuman2）：
#   - 单条：test_infer_zimage_native.py + br 448×256
#   - 批量 VBench：inference/scripts/run_vbench2_zimage_standard_parallel.sh
#     输出：output_test_infer/zimage_ckptbest_vbench_standard_448x256_4s/
#   - 2026-06-08 补跑 finetune ckpt 448×256 VBench（8 卡 × 3209 条）
#
# 相关批量脚本（源码在未挂载 davinci 目录，仅记录调用方式）：
#   bash inference/scripts/run_vbench2_zimage_standard_parallel.sh
#   bash inference/scripts/run_first4_zimage_4s.sh          # manifest 前 4 条
#   bash inference/scripts/run_random_zimage_sample_20.sh  # 随机 20 条
#   ./inference/scripts/run_eval_zimage_480x800_ckpt_best.sh start  # 480×800 评测
#
# ----------------------------------------------------------------------------
# 【A】原始单条命令（448×256，逐字还原思路，旧路径）
# ----------------------------------------------------------------------------
# cd /kwkj-k8s/davinci/LJH/daVinci-MagiHuman2
#
# PROMPT=$(python3 -c "import json; print(json.loads(open('dataset/zimage_480x800_1w/latent_manifest_train.jsonl').readline())['prompt'])")
#
# CUDA_VISIBLE_DEVICES=0 \
# MAGI_NEGATIVE_PROMPT="static, blurred details, subtitles, overall gray, worst quality, low quality, jpeg artifacts, ugly, deformed, disfigured, messy background" \
# python3 -u inference/test_infer_zimage_native.py \
#   --config-load-path example/base/config.json \
#   --base_ckpt_dir /kwkj-k8s/davinci/LJH/models/daVinci-MagiHuman/base \
#   --ckpt_dir output_train_zimage_480x800_from_base_v1/ckpt_best.pt \
#   --ckpt_blend_alpha 1.0 \
#   --device cuda:0 \
#   --amp_dtype bf16 \
#   --num_inference_steps 32 \
#   --video_cfg_scale 6.0 \
#   --audio_cfg_scale 6.0 \
#   --seconds 4.0 \
#   --seed 42 \
#   --freeze_audio \
#   --br_width 448 \
#   --br_height 256 \
#   --output_width 448 \
#   --output_height 256 \
#   --prompt "${PROMPT}" \
#   --save_path_prefix output_test_infer/zimage_ckptbest_448x256_4s/native_448x256_000000
#
# ----------------------------------------------------------------------------
# 【B】批量 VBench 低分（历史完整跑法）
# ----------------------------------------------------------------------------
# cd /kwkj-k8s/davinci/LJH/daVinci-MagiHuman2
# # 默认 OUTDIR=output_test_infer/zimage_ckptbest_vbench_standard_448x256_4s
# # 默认 BR_WIDTH=448 BR_HEIGHT=256 seconds=4 seed=42 cfg=6.0
# bash inference/scripts/run_vbench2_zimage_standard_parallel.sh
#
# 冒烟：
# OUTDIR="${PWD}/output_test_infer/zimage_ckptbest_vbench_standard_448x256_4s" \
# NGPUS=1 PER_GPU_WORKERS=1 LIMIT_PROMPTS=1 DIMENSIONS=Human_Anatomy FORCE=1 \
# bash inference/scripts/run_vbench2_zimage_standard_parallel.sh
#
# ============================================================================
# 【C】当前工作区可直接跑的单条低分版本
# ============================================================================
#   test_infer_zimage_native.py -> test_infer_seedance.py --generate
#
# 用法：
#   GPU=0 bash example/legacy_zimage_inference_lowres.sh
#   CKPT=output_train_t2i_1w_480x800_v1/ckpt_best.pt GPU=6 bash example/legacy_zimage_inference_lowres.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

GPU="${GPU:-0}"
CKPT="${CKPT:-output_train_t2i_1w_480x800_v1/ckpt_best.pt}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base}"
CONFIG="${CONFIG:-example/base/config_infer.json}"
MANIFEST="${MANIFEST:-dataset/t2i_1w_480x800_f1/latent_manifest_train.jsonl}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
NEG_PROMPT="${NEG_PROMPT:-static, blurred details, subtitles, overall gray, worst quality, low quality, jpeg artifacts, ugly, deformed, disfigured, messy background}"
STEPS="${STEPS:-32}"
VIDEO_CFG="${VIDEO_CFG:-6.0}"
AUDIO_CFG="${AUDIO_CFG:-6.0}"
SECONDS_LEN="${SECONDS_LEN:-4.0}"
SEED="${SEED:-42}"
WIDTH="${WIDTH:-448}"
HEIGHT="${HEIGHT:-256}"
OUTDIR="${OUTDIR:-output_test_infer/zimage_ckptbest_448x256_legacy_repro}"

mkdir -p "$OUTDIR"

PROMPT="$("$PYTHON" - "$MANIFEST" "$PROMPT_INDEX" <<'PY'
import json, sys
manifest, idx = sys.argv[1], int(sys.argv[2])
with open(manifest) as f:
    for i, line in enumerate(f):
        if i == idx:
            print(json.loads(line)["prompt"]); break
PY
)"
echo "[prompt] ${PROMPT:0:160}..."
echo "[spec] ${WIDTH}x${HEIGHT} seconds=${SECONDS_LEN} cfg=${VIDEO_CFG}"

export MAGI_DISABLE_SPATIAL_FIX="${MAGI_DISABLE_SPATIAL_FIX:-1}"
export MAGI_CANVAS_FIX="${MAGI_CANVAS_FIX:-0}"

CUDA_VISIBLE_DEVICES="$GPU" \
MAGI_NEGATIVE_PROMPT="$NEG_PROMPT" \
MAGI_T2V_FREEZE_AUDIO=1 \
"$PYTHON" -u inference/test_infer_seedance.py \
  --config-load-path "$CONFIG" \
  --base_ckpt_dir "$BASE_CKPT_DIR" \
  --ckpt_dir "$CKPT" \
  --ckpt_blend_alpha 1.0 \
  --generate \
  --device cuda \
  --amp_dtype bf16 \
  --num_inference_steps "$STEPS" \
  --video_cfg_scale "$VIDEO_CFG" \
  --audio_cfg_scale "$AUDIO_CFG" \
  --seconds "$SECONDS_LEN" \
  --seed "$SEED" \
  --freeze_audio \
  --br_width "$WIDTH" --br_height "$HEIGHT" \
  --prompt "$PROMPT" \
  --save_path_prefix "${OUTDIR}/legacy_${WIDTH}x${HEIGHT}_000000"

echo "[done] 输出在: ${OUTDIR}/"
