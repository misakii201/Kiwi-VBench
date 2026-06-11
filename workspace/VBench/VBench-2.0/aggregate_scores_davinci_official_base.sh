#!/usr/bin/env bash
# 评测根目录须已含 18 维（evaluate_davinci_official_base.sh + evaluate_davinci_official_base_ip.sh）。
# 报告写入官方 base 评测目录，与 aggregate_scores_davinci_finetune.sh 分开。

set -euo pipefail
VBENCH_ROOT="/kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0"
EVAL_DIR="/kwkj-k8s/YANG_LTX/workspace/RESULT/Evaluation/0515_davinci_official_base_sr2x_704x1280"
REPORT="${EVAL_DIR}/cal_local_scores_official_base_report.txt"

cd "$VBENCH_ROOT"
python cal_local_scores.py \
  --dir "$EVAL_DIR" \
  --report "$REPORT" \
  --title "VBench-2.0 local aggregate | DaVinci official base (SR 704×1280)"
echo "Wrote: $REPORT"
