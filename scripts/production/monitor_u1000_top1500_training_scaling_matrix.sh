#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"; PYTHON="$ROOT/envs/camels-gnn/bin/python"
CONTROL="$ROOT/scripts/validation/manage_u1000_top1500_training_scaling_matrix.py"
STATE="$ROOT/logs/u1000_top1500_training_scaling_matrix/current_state.env"
usage() { echo 'Usage: monitor_u1000_top1500_training_scaling_matrix.sh [--watch SECONDS|--help]'; }
interval=""; case "${1:-}" in "") ;; --help|-h) usage; exit 0 ;; --watch) [[ $# -eq 2 && "$2" =~ ^[0-9]+$ && "$2" -ge 5 ]] || { usage >&2; exit 2; }; interval="$2" ;; *) usage >&2; exit 2 ;; esac
show() {
  local MATRIX_LAUNCHER_PID="" MATRIX_START_EPOCH="" CURRENT_MODEL="" CURRENT_SEED="" CURRENT_TRAIN_COUNT="" CURRENT_EXPERIMENT="" TRAINER_PID="" RUN_START_EPOCH="" LOG_FILE="" EXPERIMENT_DIR=""
  [[ -f "$STATE" ]] && source "$STATE"; echo "UTC: $(date -u +%FT%TZ)"
  echo "launcher_pid=${MATRIX_LAUNCHER_PID:-none} model=${CURRENT_MODEL:-none} seed=${CURRENT_SEED:-none} train=${CURRENT_TRAIN_COUNT:-none} experiment=${CURRENT_EXPERIMENT:-none}"
  (cd "$ROOT" && "$PYTHON" "$CONTROL" --status)
  pids="$(ps -eo pid=,etime=,%cpu=,rss=,args= | awk '$0 ~ /train_(evolvegcn_h|static_gcn)/ && $0 !~ /awk/ {print}')"; echo "trainer=${pids:-none}"
  if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader 2>/dev/null || true; fi
  [[ -n "$EXPERIMENT_DIR" && -f "$EXPERIMENT_DIR/train_log.csv" ]] && tail -n 2 "$EXPERIMENT_DIR/train_log.csv"
  echo "latest_log=${LOG_FILE:-none}"; [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]] && tail -n 8 "$LOG_FILE"
}
[[ -z "$interval" ]] && { show; exit 0; }; while true; do clear 2>/dev/null || true; show; sleep "$interval"; done
