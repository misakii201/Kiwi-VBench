#!/usr/bin/env bash
# 评测根目录须已含 18 个维度子目录（先跑 evaluate_davinci.sh，再跑 evaluate_davinci_ip.sh）。
# 报告单独写入该评测目录，与官方 base 的 aggregate_scores_davinci_official_base.sh 互不覆盖。

set -euo pipefail
VBENCH_ROOT="/kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0"
EVAL_DIR="/kwkj-k8s/YANG_LTX/workspace/RESULT/Evaluation/0515_davinci_seedance_v1_step4000_sr2x_704x1280"
REPORT="${EVAL_DIR}/cal_local_scores_finetune_report.txt"

cd "$VBENCH_ROOT"
python cal_local_scores.py \
  --dir "$EVAL_DIR" \
  --report "$REPORT" \
  --title "VBench-2.0 local aggregate | DaVinci finetuned (seedance ckpt_step4000, SR 704×1280)"
echo "Wrote: $REPORT"
