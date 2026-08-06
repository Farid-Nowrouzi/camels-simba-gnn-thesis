#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PYTHON="${REPO_ROOT}/envs/camels-gnn/bin/python"
CONTROL="scripts/validation/manage_u1000_top1000_training_scaling_matrix.py"
STATE_FILE="${REPO_ROOT}/logs/u1000_top1000_training_scaling_matrix/current_state.env"

usage() {
  cat <<'EOF'
Usage: monitor_u1000_top1000_training_scaling_matrix.sh [--watch SECONDS|--help]
Read-only matrix monitor. Watch intervals below 5 seconds are rejected.
EOF
}

interval=""
case "${1:-}" in
  "") ;;
  --help) usage; exit 0 ;;
  --watch) [[ $# -eq 2 && "$2" =~ ^[0-9]+$ && "$2" -ge 5 ]] || { usage >&2; exit 2; }; interval="$2" ;;
  *) usage >&2; exit 2 ;;
esac

show_status() {
  local MATRIX_LAUNCHER_PID="" MATRIX_START_EPOCH="" CURRENT_MODEL="" CURRENT_SEED="" CURRENT_TRAIN_COUNT="" CURRENT_EXPERIMENT="" TRAINER_PID="" RUN_START_EPOCH="" LOG_FILE="" EXPERIMENT_DIR=""
  [[ -f "${STATE_FILE}" ]] && source "${STATE_FILE}"
  local now_epoch cpu rss elapsed_run elapsed_matrix size latest_epoch train_loss val_loss best_loss gpu_util gpu_proc gpu_free
  now_epoch="$(date +%s)"; elapsed_run="n/a"; elapsed_matrix="n/a"
  [[ "${RUN_START_EPOCH}" =~ ^[0-9]+$ ]] && elapsed_run="$((now_epoch-RUN_START_EPOCH))s"
  [[ "${MATRIX_START_EPOCH}" =~ ^[0-9]+$ ]] && elapsed_matrix="$((now_epoch-MATRIX_START_EPOCH))s"
  cpu="n/a"; rss="n/a"
  if [[ "${TRAINER_PID}" =~ ^[0-9]+$ ]] && kill -0 "${TRAINER_PID}" 2>/dev/null; then
    read -r cpu rss < <(ps -p "${TRAINER_PID}" -o %cpu=,rss= 2>/dev/null || echo "n/a n/a")
  fi
  size="n/a"; [[ -n "${EXPERIMENT_DIR}" && -d "${EXPERIMENT_DIR}" ]] && size="$(du -sh "${EXPERIMENT_DIR}" 2>/dev/null | awk '{print $1}')"
  latest_epoch="n/a"; train_loss="n/a"; val_loss="n/a"; best_loss="n/a"
  if [[ -n "${EXPERIMENT_DIR}" && -f "${EXPERIMENT_DIR}/train_log.csv" ]]; then
    IFS=, read -r latest_epoch train_loss val_loss best_loss _ < <(tail -n 1 "${EXPERIMENT_DIR}/train_log.csv")
  fi
  gpu_util="n/a"; gpu_free="n/a"; gpu_proc="n/a"
  local gpu_line
  if command -v nvidia-smi >/dev/null && gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)" && [[ -n "${gpu_line}" ]]; then
    IFS=, read -r gpu_util gpu_free <<<"${gpu_line}"
    if [[ "${TRAINER_PID}" =~ ^[0-9]+$ ]]; then gpu_proc="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, -v p="${TRAINER_PID}" '$1+0==p {gsub(/ /,"",$2); print $2; found=1} END {if(!found) print "n/a"}')"; fi
  fi
  echo "UTC: $(date -u +%FT%TZ)"
  echo "launcher_pid=${MATRIX_LAUNCHER_PID:-none} model=${CURRENT_MODEL:-none} seed=${CURRENT_SEED:-none} train_count=${CURRENT_TRAIN_COUNT:-none}"
  echo "experiment=${CURRENT_EXPERIMENT:-none} trainer_pid=${TRAINER_PID:-none} elapsed_run=${elapsed_run} elapsed_matrix=${elapsed_matrix}"
  (cd "${REPO_ROOT}" && "${PYTHON}" "${CONTROL}" --status)
  echo "epoch=${latest_epoch} train_loss=${train_loss} val_loss=${val_loss} best_val_loss=${best_loss}"
  echo "cpu_percent=${cpu} host_rss_kib=${rss} gpu_util_percent=${gpu_util// /} gpu_process_memory_mib=${gpu_proc} free_gpu_memory_mib=${gpu_free// /}"
  echo "experiment_size=${size} latest_log=${LOG_FILE:-none}"
  if [[ -n "${LOG_FILE}" && -f "${LOG_FILE}" ]]; then echo "latest_log_lines:"; tail -n 8 "${LOG_FILE}"; fi
}

if [[ -z "${interval}" ]]; then show_status; exit 0; fi
while true; do clear 2>/dev/null || true; show_status; sleep "${interval}"; done
