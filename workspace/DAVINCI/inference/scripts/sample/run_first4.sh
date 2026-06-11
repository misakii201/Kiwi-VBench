#!/usr/bin/env bash
# 前 4 条 manifest prompt，480×800 / 4s（长驻 native worker）
set -euo pipefail

REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
OUTDIR="${OUTDIR:-${REPO}/output_test_infer/zimage_ckptbest_480x800_first4_4s}"
LOG="${REPO}/inference/scripts/logs/first4_4s.log"
PYTHON="${PYTHON:-/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3}"

mkdir -p "$(dirname "$LOG")" "${OUTDIR}"

"${PYTHON}" -u "${REPO}/inference/scripts/sample/quick_worker.py" \
  --repo "${REPO}" \
  --base /kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base \
  --ckpt "${REPO}/output_train_t2i_1w_480x800_v1/ckpt_best.pt" \
  --manifest "${REPO}/dataset/t2i_1w_480x800_f1/latent_manifest_train.jsonl" \
  --outdir "${OUTDIR}" \
  --progress "${OUTDIR}/progress_first4_4s.json" \
  --gpu "${GPU:-0}" \
  --prompt-start 0 \
  --num-prompts 4 \
  --blend 1.0 \
  --seed 42 \
  --steps 32 \
  --cfg 6.0 \
  --seconds 4.0 \
  --br-width 480 \
  --br-height 800 \
  2>&1 | tee -a "${LOG}"
