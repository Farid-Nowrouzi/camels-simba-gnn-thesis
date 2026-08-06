#!/usr/bin/env bash
set -Eeuo pipefail
usage() { cat <<'EOF'
Usage: scripts/production/monitor_u1000_top1500_sparse_build.sh [--watch SECONDS|--help]
Read-only monitor; it never starts, restarts, signals, or kills the builder.
EOF
}
interval=""
case "${1:-}" in "") ;; --help|-h) usage; exit 0 ;; --watch) [[ $# -eq 2 && "$2" =~ ^[0-9]+$ && "$2" -ge 5 ]] || { usage >&2; exit 2; }; interval="$2" ;; *) usage >&2; exit 2 ;; esac
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"; cd "$ROOT"
show() {
  mapfile -t pids < <(ps -eo pid=,args= | awk '$0 ~ /python/ && $0 ~ /build_temporal_sequences/ && $0 ~ /top1500/ && $0 !~ /--help/ {print $1}')
  log="$(find logs/dataset_builds -maxdepth 1 -type f -name 'u1000_top1500_sparse_build_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {sub(/^[^ ]+ /,""); print}')"
  echo "UTC: $(date -u +%FT%TZ)"; echo "builder_pids=${pids[*]:-none}"
  for pid in "${pids[@]}"; do ps -p "$pid" -o pid=,etime=,%cpu=,rss=,args=; done
  current="n/a"; completed=0; failed=0; snapshot="n/a"
  if [[ -n "$log" && -r "$log" ]]; then
    current="$(rg -o 'LH_[0-9]+' "$log" | tail -1 || true)"
    completed="$(rg -c 'Successfully processed universe|Completed universe' "$log" || true)"
    failed="$(rg -c 'Failed universe|ERROR.*LH_' "$log" || true)"
    snapshot="$(rg -o '0\.(20000|25000|51209|75065)|1\.00000' "$log" | tail -1 || true)"
  fi
  echo "current_universe=${current:-n/a} completed_universes=${completed:-0} failed_universes=${failed:-0} current_snapshot=${snapshot:-n/a}"
  echo -n 'temporary_output_size='; du -sh data/processed/temporal_1000u_none_top1500_periodic_knn_sparse 2>/dev/null | awk '{print $1}' || echo absent
  free -h | awk 'NR==2 {print "host_memory_free=" $4 " host_memory_available=" $7}'
  df -h "$ROOT" | awk 'NR==2 {print "free_disk=" $4 " used=" $5}'
  echo "latest_log=${log:-none}"; [[ -n "$log" && -r "$log" ]] && tail -n 20 "$log"
}
[[ -z "$interval" ]] && { show; exit 0; }; while true; do clear 2>/dev/null || true; show; sleep "$interval"; done
