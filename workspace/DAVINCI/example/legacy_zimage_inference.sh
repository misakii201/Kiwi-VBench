#!/usr/bin/env bash
#
# ============================================================================
# 上次（zimage_480x800_1w）出"好"视频时用的推理命令 —— 历史还原
# ============================================================================
#
# 来源：终端 bash 历史（旧项目 /kwkj-k8s/davinci/LJH/daVinci-MagiHuman2）。
# 说明：
#   - 旧推理脚本 inference/test_infer_zimage_native.py 的【源码】在未挂载的
#     davinci 目录里，当前拿不到；但下面的【调用命令】是逐字还原的。
#   - 当前工作区已恢复 inference/test_infer_zimage_native.py（长驻 worker 同款入口）
#   - 关键差异点（相对这次 VBench 预览）：
#       guidance scale = 6.0（不是默认 5.0）
#       seconds        = 4.0（不是 5）
#       seed           = 42（不是 521）
#       prompt         = 训练同分布的长 cinematic caption（不是 VBench 短句）
#       negative prompt= 通过 MAGI_NEGATIVE_PROMPT 显式设置
#
# ----------------------------------------------------------------------------
# 【A】原始命令（逐字还原，路径指向旧 davinci 项目，当前未挂载、不可直接跑）
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
#   --br_width 480 --br_height 800 \
#   --output_width 480 --output_height 800 \
#   --prompt "${PROMPT}" \
#   --save_path_prefix output_test_infer/zimage_ckptbest_480x800_first4_4s/native_480x800_000000
#
# 其他相关历史调用（脚本体未保存，仅记录调用方式）：
#   ./inference/scripts/run_eval_zimage_480x800_ckpt_best.sh start
#   GPU=0 nohup bash .../inference/scripts/run_random_zimage_sample_20.sh >> .../random_sample_20.log 2>&1 &
#   bash .../inference/scripts/run_first4_zimage_4s.sh
#   nohup bash inference/scripts/run_vbench2_zimage_standard_parallel.sh >> .../vbench2_zimage_standard_parallel.log 2>&1 &
#
# ============================================================================
# 【B】适配当前工作区、可直接运行的版本
# ============================================================================
#   - test_infer_zimage_native.py  -> 原生推理入口（禁用 spatial fix）
#   - 旧 base 路径                  -> /kwkj-k8s/cy123/daVinci-MagiHuman-main/.../base
#   - 旧 ckpt                       -> output_train_t2i_1w_480x800_v1/ckpt_best.pt
#   - 旧 manifest                   -> dataset/t2i_1w_480x800_f1/latent_manifest_train.jsonl
#   - config.json(无 audio 路径)    -> config_infer.json（--generate 需要 audio_model_path）
#   - MAGI_NEGATIVE_PROMPT           -> video_generate.py 已支持（2026-06-10 对齐修复）
#
# 对齐确认（相对 VBench 预览 config_t2i_1w_preview*.yaml）：
#   video_cfg_scale  6.0  vs  默认 5.0   <- 本脚本已对齐 6.0
#   audio_cfg_scale  6.0  vs  默认 5.0   <- 本脚本已对齐 6.0
#   seconds          4.0  vs  5           <- 本脚本已对齐 4.0
#   seed             42   vs  521         <- 本脚本已对齐 42
#   freeze_audio     yes  vs  yes         <- 已对齐（MAGI_T2V_FREEZE_AUDIO=1）
#   negative prompt  短版 vs  硬编码长版  <- 本脚本通过 MAGI_NEGATIVE_PROMPT 对齐短版
#   prompt           训练长 caption vs VBench 短句 <- 本脚本从 manifest 取
#   spatial fix      默认开启（易裁切竖屏） <- MAGI_DISABLE_SPATIAL_FIX=1 已关闭
#   canvas fix       output_* 触发 MAGI_CANVAS_FIX=1 <- 已去掉多余 output_* 参数
#
# 用法：
#   GPU=0 bash example/legacy_zimage_inference.sh
#   CKPT=output_train_t2i_1w_480x800_v1/ckpt_step_5200.pt GPU=0 bash example/legacy_zimage_inference.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---- 可调参数（默认严格对齐旧命令）----
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
WIDTH="${WIDTH:-480}"
HEIGHT="${HEIGHT:-800}"
OUTDIR="${OUTDIR:-output_test_infer/zimage_ckptbest_480x800_legacy_repro}"

mkdir -p "$OUTDIR"

# 取训练同分布 prompt（对齐旧命令的取法）
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

# 480×800 竖屏 T2I：禁用 decode 后的 spatial fix（会误裁切画面）和 canvas fix
# 不传 output_width/height（与 br 相同时多余，且会触发 MAGI_CANVAS_FIX=1）
export MAGI_DISABLE_SPATIAL_FIX="${MAGI_DISABLE_SPATIAL_FIX:-1}"
export MAGI_CANVAS_FIX="${MAGI_CANVAS_FIX:-0}"

CUDA_VISIBLE_DEVICES="$GPU" \
MAGI_NEGATIVE_PROMPT="$NEG_PROMPT" \
MAGI_T2V_FREEZE_AUDIO=1 \
"$PYTHON" -u inference/test_infer_zimage_native.py \
  --config-load-path "$CONFIG" \
  --base_ckpt_dir "$BASE_CKPT_DIR" \
  --ckpt_dir "$CKPT" \
  --ckpt_blend_alpha 1.0 \
  --device cuda:0 \
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
