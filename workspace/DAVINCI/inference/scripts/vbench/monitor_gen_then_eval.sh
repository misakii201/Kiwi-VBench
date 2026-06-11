#!/usr/bin/env bash
# 监控 zimage VBench 视频生成 → 完成后自动 18 维评测 → 汇总
#
#   cd /kwkj-k8s/cy123/workspace/DAVINCI
#   bash inference/scripts/vbench/monitor_gen_then_eval.sh
#
# 视频已齐，跳过等待直接评测:
#   SKIP_GENERATE_WAIT=1 bash inference/scripts/vbench/monitor_gen_then_eval.sh
#
set -uo pipefail

REPO="/kwkj-k8s/cy123/workspace/DAVINCI"
cd "${REPO}"

BR_WIDTH="${BR_WIDTH:-448}"
BR_HEIGHT="${BR_HEIGHT:-256}"
if [[ "${BR_WIDTH}" == "480" && "${BR_HEIGHT}" == "800" ]]; then
  DEFAULT_OUT="${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s"
else
  DEFAULT_OUT="${REPO}/output_test_infer/zimage_ckptbest_vbench_standard_${BR_WIDTH}x${BR_HEIGHT}_4s"
fi

OUTDIR="${OUTDIR:-${DEFAULT_OUT}}"
EVAL_ROOT="${EVAL_ROOT:-${OUTDIR}/vbench_eval}"
LOG_DIR="${REPO}/inference/scripts/logs"
MONITOR_LOG="${LOG_DIR}/monitor_gen_then_eval_zimage.out"
TOTAL="${TOTAL:-3209}"
POLL_SEC="${POLL_SEC:-60}"
MAX_RETRY_PASS="${MAX_RETRY_PASS:-3}"
VIDEO_SPEC="${VIDEO_SPEC:-${BR_WIDTH}×${BR_HEIGHT} / 4s}"

mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%F %T')] $*"; }

ok_count() {
  python3 - <<PY
import json, glob, os
OUT = os.environ["OUTDIR_REL"]
keys = set()
for f in glob.glob(f"{OUT}/progress_gpu*_w*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for it in d.get("items", []):
        if it.get("status") == "ok" and it.get("key"):
            keys.add(it["key"])
print(len(keys))
PY
}

failed_count() {
  python3 - <<PY
import json, glob, os
OUT = os.environ["OUTDIR_REL"]
ok, failed = set(), set()
for f in glob.glob(f"{OUT}/progress_gpu*_w*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for it in d.get("items", []):
        k = it.get("key")
        if not k:
            continue
        if it.get("status") == "ok":
            ok.add(k)
        elif it.get("status") == "failed":
            failed.add(k)
print(len(failed - ok))
PY
}

workers_running() {
  pgrep -f "vbench/worker.py.*${OUTDIR}" >/dev/null 2>&1 \
    || pgrep -f "vbench/run.py.*${OUTDIR}" >/dev/null 2>&1 \
    || pgrep -f "run_vbench2_zimage.*parallel" >/dev/null 2>&1
}

export OUTDIR_REL="${OUTDIR#${REPO}/}"

log "==== zimage 监控启动 ===="
log "OUTDIR=${OUTDIR} spec=${VIDEO_SPEC}"
log "TOTAL=${TOTAL} POLL_SEC=${POLL_SEC} MAX_RETRY_PASS=${MAX_RETRY_PASS}"

if [[ "${SKIP_GENERATE_WAIT:-0}" != "1" ]]; then
  log "阶段1: 等待视频生成完成 ..."

  if [[ "$(OUTDIR_REL="${OUTDIR_REL}" ok_count)" -lt "${TOTAL}" ]] && ! workers_running; then
    log "检测到生成未齐且 worker 未运行，启动一轮生成 ..."
    OUTDIR="${OUTDIR}" BR_WIDTH="${BR_WIDTH}" BR_HEIGHT="${BR_HEIGHT}" \
      bash "${REPO}/inference/scripts/vbench/run_parallel.sh" \
      >> "${LOG_DIR}/vbench2_zimage_standard_parallel.log" 2>&1 &
    log "生成任务已后台启动 PID=$!"
  fi

  while true; do
    cur_ok=$(OUTDIR_REL="${OUTDIR_REL}" ok_count)
    cur_fail=$(OUTDIR_REL="${OUTDIR_REL}" failed_count)
    remaining=$((TOTAL - cur_ok))
    running="no"
    workers_running && running="yes"
    log "生成进度: ok=${cur_ok}/${TOTAL} 失败=${cur_fail} 剩余=${remaining} worker运行=${running}"

    if [[ "${remaining}" -le 0 ]]; then
      log "全部 ${TOTAL} 条任务已完成"
      break
    fi

    if [[ "${running}" == "no" && "${remaining}" -gt 0 ]]; then
      for ((pass=1; pass<=MAX_RETRY_PASS; pass++)); do
        prev_ok="${cur_ok}"
        log "补跑 Pass ${pass}/${MAX_RETRY_PASS}（ok=${cur_ok} 失败=${cur_fail}）..."
        OUTDIR="${OUTDIR}" BR_WIDTH="${BR_WIDTH}" BR_HEIGHT="${BR_HEIGHT}" \
          bash "${REPO}/inference/scripts/vbench/run_parallel.sh" \
          >> "${LOG_DIR}/monitor_zimage_retry_pass${pass}.out" 2>&1 || true
        cur_ok=$(OUTDIR_REL="${OUTDIR_REL}" ok_count)
        if [[ "${cur_ok}" -ge "${TOTAL}" ]]; then
          log "补跑后全部完成 ok=${cur_ok}/${TOTAL}"
          break 2
        fi
        if [[ "${cur_ok}" -eq "${prev_ok}" ]]; then
          log "补跑 Pass ${pass} 无进展，停止补跑"
          break
        fi
      done
      final_ok=$(OUTDIR_REL="${OUTDIR_REL}" ok_count)
      if [[ "${final_ok}" -lt "${TOTAL}" ]]; then
        log "[警告] 仍未齐 ok=${final_ok}/${TOTAL}，跳过评测"
        log "手动: SKIP_GENERATE_WAIT=1 bash inference/scripts/vbench/monitor_gen_then_eval.sh"
        exit 1
      fi
      break
    fi
    sleep "${POLL_SEC}"
  done

  while workers_running; do
    log "等待 worker 进程退出 ..."
    sleep 10
  done
  log "阶段1 完成: ok=$(OUTDIR_REL="${OUTDIR_REL}" ok_count)/${TOTAL}"
else
  log "SKIP_GENERATE_WAIT=1，跳过等待"
fi

log "阶段2: 启动 VBench 18 维评测 ..."
EVAL_LOG="${LOG_DIR}/monitor_zimage_eval_$(date '+%Y%m%d_%H%M%S').out"
VIDEOS_ROOT="${OUTDIR}" OUTPUT_ROOT="${EVAL_ROOT}" \
  bash "${REPO}/inference/scripts/vbench/evaluate.sh" \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
if [[ "${eval_rc}" -ne 0 ]]; then
  log "[错误] 评测失败 exit=${eval_rc}，日志: ${EVAL_LOG}"
  exit "${eval_rc}"
fi
log "阶段2 完成"

log "阶段3: 汇总评测分数 ..."
python3 "${REPO}/inference/scripts/vbench/summarize.py" \
  --eval-root "${EVAL_ROOT}" \
  --output "${EVAL_ROOT}/score_summary.md"

log "==== 全部完成 ===="
log "视频: ${OUTDIR}/"
log "评测: ${EVAL_ROOT}/"
log "汇总: ${EVAL_ROOT}/score_summary.md"
