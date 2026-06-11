#!/usr/bin/env bash
#
# Detached training launcher — survives Cursor/SSH disconnect.
#
# Usage:
#   bash example/train_t2i_1w_480x800.sh start          # start (auto-resume if checkpoint exists)
#   bash example/train_t2i_1w_480x800.sh resume         # resume from latest checkpoint
#   bash example/train_t2i_1w_480x800.sh stop           # stop training
#   bash example/train_t2i_1w_480x800.sh status         # show running state
#
# Optional env:
#   RESUME_FROM=/path/to/ckpt_step_200.pt bash example/train_t2i_1w_480x800.sh start
#   OUTDIR=output_train_t2i_1w_480x800_v1 bash example/train_t2i_1w_480x800.sh start

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="/kwkj-k8s/YANG_LTX/Envs/davinci/bin/python3"
OUTDIR="${OUTDIR:-output_train_t2i_1w_480x800_v1}"
PIDFILE="${OUTDIR}/train.pid"
LOGFILE="${OUTDIR}/train.log"
ACTION="${1:-start}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export DAVINCI_DISABLE_MAGI_COMPILE=1
export MAGI_DISABLE_COMPILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

is_running() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE")"
  kill -0 "$pid" 2>/dev/null
}

find_latest_ckpt() {
  local latest=""
  local step_ckpt
  step_ckpt="$(ls -1 "${OUTDIR}"/ckpt_step_*.pt 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "$step_ckpt" ]]; then
    latest="$step_ckpt"
  elif [[ -f "${OUTDIR}/ckpt_best.pt" ]]; then
    latest="${OUTDIR}/ckpt_best.pt"
  fi
  printf '%s' "$latest"
}

resolve_resume_from() {
  if [[ -n "${RESUME_FROM:-}" ]]; then
    printf '%s' "$RESUME_FROM"
    return
  fi
  find_latest_ckpt
}

stop_training() {
  if ! is_running; then
    echo "Training is not running."
    rm -f "$PIDFILE"
    pkill -f "train_ltx.py.*${OUTDIR}" 2>/dev/null || true
    rm -f "${OUTDIR}"/ckpt_best.pt.tmp.* 2>/dev/null || true
    return 0
  fi

  local pid pgid
  pid="$(cat "$PIDFILE")"
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  echo "Stopping training (pid=${pid}, pgid=${pgid:-unknown})..."

  if [[ -n "$pgid" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi

  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PIDFILE"
      rm -f "${OUTDIR}"/ckpt_best.pt.tmp.* 2>/dev/null || true
      echo "Training stopped."
      return 0
    fi
    sleep 2
  done

  echo "Force killing training..."
  if [[ -n "$pgid" ]]; then
    kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  else
    kill -KILL "$pid" 2>/dev/null || true
  fi
  pkill -f "train_ltx.py.*${OUTDIR}" 2>/dev/null || true
  rm -f "$PIDFILE"
  rm -f "${OUTDIR}"/ckpt_best.pt.tmp.* 2>/dev/null || true
  echo "Training force-stopped."
}

show_status() {
  if is_running; then
    local pid
    pid="$(cat "$PIDFILE")"
    echo "Training is RUNNING (pid=${pid})"
    echo "Log: ${LOGFILE}"
    tail -3 "$LOGFILE" 2>/dev/null || true
  else
    echo "Training is NOT running."
    rm -f "$PIDFILE" 2>/dev/null || true
    local ckpt
    ckpt="$(find_latest_ckpt)"
    if [[ -n "$ckpt" ]]; then
      echo "Latest checkpoint: ${ckpt}"
    fi
  fi
}

start_training() {
  if is_running; then
    echo "Training already running (pid=$(cat "$PIDFILE"))."
    exit 1
  fi

  mkdir -p "$OUTDIR"

  local resume_args=()
  local resume_from=""
  if [[ "$ACTION" == "resume" || -n "${RESUME_FROM:-}" ]]; then
    resume_from="$(resolve_resume_from)"
    if [[ -z "$resume_from" || ! -f "$resume_from" ]]; then
      echo "No checkpoint found under ${OUTDIR}."
      exit 1
    fi
    resume_args=(--resume_from "$resume_from")
    echo "Resuming from: ${resume_from}"
  else
    resume_from="$(find_latest_ckpt)"
    if [[ -n "$resume_from" && -f "$resume_from" ]]; then
      resume_args=(--resume_from "$resume_from")
      echo "Auto-resuming from: ${resume_from}"
    fi
  fi

  echo "Starting detached training..."
  echo "  outdir: ${OUTDIR}"
  echo "  log:    ${LOGFILE}"
  echo "  pid:    ${PIDFILE}"

  nohup setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=8 \
    inference/train_ltx.py \
    --config-load-path example/base/config.json \
    --manifest dataset/t2i_1w_480x800_f1/latent_manifest_train.jsonl \
    --output_dir "$OUTDIR" \
    --use_text_encoder \
    --txt_device cpu \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --amp_dtype bf16 \
    --lr 1e-5 \
    --num_steps 8000 \
    --random_latent_frames 1 \
    --random_latent_h 50 \
    --random_latent_w 30 \
    --expected_latent_h 50 \
    --expected_latent_w 30 \
    --log_every 10 \
    --save_every 200 \
    "${resume_args[@]}" \
    >> "$LOGFILE" 2>&1 &

  echo $! > "$PIDFILE"

  sleep 2
  if is_running; then
    echo "Training started in background (pid=$(cat "$PIDFILE"))."
    echo "Monitor: tail -f ${LOGFILE}"
  else
    echo "Failed to start training. Check ${LOGFILE}"
    exit 1
  fi
}

case "$ACTION" in
  start|resume)
    start_training
    ;;
  stop)
    stop_training
    ;;
  status)
    show_status
    ;;
  *)
    echo "Unknown action: ${ACTION}"
    echo "Usage: $0 {start|resume|stop|status}"
    exit 1
    ;;
esac
