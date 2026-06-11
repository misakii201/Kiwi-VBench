#!/usr/bin/env bash
# 随机 20 条 manifest 采样（单卡）
set -euo pipefail

REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
OUTDIR="${OUTDIR:-${REPO}/output_test_infer/zimage_ckptbest_480x800/random_sample_20}"
MANIFEST="${REPO}/dataset/t2i_1w_480x800_f1/latent_manifest_train.jsonl"
INDICES="${OUTDIR}/sampled_indices.txt"
PYTHON="${PYTHON:-/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3}"

mkdir -p "${OUTDIR}"

if [[ ! -f "${INDICES}" ]]; then
  "${PYTHON}" - <<PY
import json, random
manifest = "${MANIFEST}"
n = sum(1 for ln in open(manifest) if ln.strip())
random.seed(int("${SEED:-20260604}"))
idx = sorted(random.sample(range(n), min(20, n)))
open("${INDICES}", "w").write("\n".join(map(str, idx)) + "\n")
print("wrote", len(idx), "indices to ${INDICES}")
PY
fi

"${PYTHON}" -u "${REPO}/inference/scripts/sample/quick_worker.py" \
  --repo "${REPO}" \
  --base /kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/base \
  --ckpt "${REPO}/output_train_t2i_1w_480x800_v1/ckpt_best.pt" \
  --manifest "${MANIFEST}" \
  --outdir "${OUTDIR}" \
  --progress "${OUTDIR}/progress_random20.json" \
  --gpu "${GPU:-0}" \
  --indices-file "${INDICES}" \
  --seed "${SEED:-20260604}" \
  --seconds 4.0 \
  --br-width 480 --br-height 800
