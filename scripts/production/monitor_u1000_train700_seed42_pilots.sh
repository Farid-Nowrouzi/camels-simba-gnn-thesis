#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE="${REPO_ROOT}/logs/u1000_train700_seed42_pilots/current_state.env"
[[ -r "${STATE}" ]] || { echo "No pilot state file exists."; exit 0; }
source "${STATE}"
echo "Current model: ${CURRENT_MODEL:-unknown}"
echo "Trainer PID: ${TRAINER_PID:-none}"
if [[ -n "${TRAINER_PID:-}" && -r "/proc/${TRAINER_PID}/stat" ]]; then
  echo "Elapsed seconds: $(( $(date +%s) - START_EPOCH ))"
  ps -p "${TRAINER_PID}" -o pid=,etime=,%cpu=,%mem=,rss=,cmd=
else
  echo "Trainer process: not active"
fi
free -h
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
if [[ -n "${EXPERIMENT_DIR:-}" && -d "${EXPERIMENT_DIR}" ]]; then
  du -sh "${EXPERIMENT_DIR}"
fi
if [[ -n "${LOG_FILE:-}" && -r "${LOG_FILE}" ]]; then
  echo "Latest epoch/loss:"
  grep 'Epoch ' "${LOG_FILE}" | tail -1 || true
  echo "Latest log lines:"
  tail -20 "${LOG_FILE}"
fi
