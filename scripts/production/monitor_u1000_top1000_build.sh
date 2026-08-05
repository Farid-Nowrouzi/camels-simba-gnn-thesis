#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/production/monitor_u1000_top1000_build.sh [--interval SECONDS] [--log PATH]

Read-only monitor for the U1000 Top1000 sparse builder. It never starts,
restarts, signals, or kills a process. The default interval is 30 seconds and
the newest matching build log is used when --log is omitted.
EOF
}

interval=30
log_file=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      interval="$2"
      shift 2
      ;;
    --log)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      log_file="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ "$interval" =~ ^[1-9][0-9]*$ ]] || { printf 'Interval must be a positive integer.\n' >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "$REPO_ROOT"

readonly OUTPUT_DIR="data/processed/temporal_1000u_none_top1000_periodic_knn_sparse"

while true; do
  mapfile -t pids < <(
    ps -eo pid=,args= | awk '
      $0 ~ /python/ &&
      $0 ~ /(-m[[:space:]]+src\.data\.build_temporal_sequences|src\/data\/build_temporal_sequences\.py)/ &&
      $0 !~ /--help/ {print $1}'
  )
  if [[ -z "$log_file" ]]; then
    log_file="$(find logs/dataset_builds -maxdepth 1 -type f -name 'u1000_top1000_sparse_build_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print}')"
  fi

  clear 2>/dev/null || true
  printf 'U1000 Top1000 sparse build monitor — %s UTC\n' "$(date -u '+%Y-%m-%d %H:%M:%S')"
  if [[ ${#pids[@]} -eq 0 ]]; then
    printf 'Builder PID: not running\n'
  else
    for pid in "${pids[@]}"; do
      ps -p "$pid" -o pid=,etime=,%cpu=,rss=,args= | awk '{printf "Builder PID/elapsed/CPU/RSS: %s %s %s%% %s KiB\n", $1, $2, $3, $4}'
    done
  fi
  free -h | awk 'NR == 2 {printf "Host memory: free=%s available=%s\n", $4, $7} NR == 3 {printf "Swap: used=%s total=%s\n", $3, $2}'
  printf 'Output-directory size: '
  du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}' || printf 'absent\n'
  df -h "$REPO_ROOT" | awk 'NR == 2 {printf "Free disk: %s (%s used)\n", $4, $5}'
  printf '\nLatest 20 log lines (%s):\n' "${log_file:-no log found}"
  if [[ -n "$log_file" && -r "$log_file" ]]; then
    tail -n 20 "$log_file"
  else
    printf 'No readable build log found.\n'
  fi
  sleep "$interval"
done
