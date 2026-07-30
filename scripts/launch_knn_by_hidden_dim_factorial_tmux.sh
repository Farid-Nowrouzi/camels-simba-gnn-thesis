#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
EVOLVE_SPEC="${REPO_ROOT}/configs/experiment_families/canonical_evolvegcn_knn_by_hidden_dim_factorial_500u_top100.json"
STATIC_SPEC="${REPO_ROOT}/configs/experiment_families/canonical_static_gcn_knn_by_hidden_dim_factorial_500u_top100.json"
SESSION_NAME="canonical-knn-hidden-factorial"
INTERNAL_EXECUTE=false
CURRENT_PHASE="launcher-preflight"
CURRENT_EXPERIMENT="none"
CURRENT_LOG_DIRECTORY=""
WRAPPER_STATUS_PATH="${WRAPPER_STATUS_PATH:-}"

usage() {
  echo "Usage: bash scripts/launch_knn_by_hidden_dim_factorial_tmux.sh [--session NAME]"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --session)
      [[ "$#" -ge 2 ]] || {
        echo "ERROR: --session requires a value." >&2
        exit 2
      }
      SESSION_NAME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --internal-execute)
      INTERNAL_EXECUTE=true
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${SESSION_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "ERROR: session name may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
}
for required_path in "${EVOLVE_SPEC}" "${STATIC_SPEC}"; do
  [[ -f "${required_path}" ]] || {
    echo "ERROR: required family specification is missing: ${required_path}" >&2
    exit 2
  }
done

ACTIVATE_PATH="${REPO_ROOT}/envs/camels-gnn/bin/activate"
[[ -f "${ACTIVATE_PATH}" ]] || {
  echo "ERROR: documented environment activation script is missing: ${ACTIVATE_PATH}" >&2
  exit 2
}

verify_exact_splits() {
  local spec_path="$1"
  python3 - "${REPO_ROOT}" "${spec_path}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {
    42: "0f963679cd284fca861fc2c59d88bdae8e8f1f21e2cbe1bb73bd593b49056748",
    123: "853549f16ef8eb3d7f18ae850c94b13c0c8bf0e770bb99cfbffff48b03530266",
    2025: "3ce48b66c11e30bec459c52ba7f4a900809dd2b45be0995b8b56aeaefc747951",
}
all_ids = {f"LH_{index}" for index in range(500)}

for run in spec["runs"]:
    config_path = repo_root / run["experiment_path"] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = [config.get(key) for key in ("train_ids", "val_ids", "test_ids")]
    if not all(isinstance(ids, list) for ids in splits):
        raise SystemExit(f"missing split lists: {config_path}")
    flat = [item for ids in splits for item in ids]
    if [len(ids) for ids in splits] != [350, 75, 75]:
        raise SystemExit(f"invalid split sizes: {config_path}")
    if len(flat) != len(set(flat)) or set(flat) != all_ids:
        raise SystemExit(f"split overlap or coverage failure: {config_path}")
    payload = json.dumps(
        {
            "train_ids": splits[0],
            "val_ids": splits[1],
            "test_ids": splits[2],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if signature != expected[int(run["seed"])]:
        raise SystemExit(
            f"split signature mismatch: {run['experiment_name']} "
            f"got={signature} expected={expected[int(run['seed'])]}"
        )

print(f"Exact split signatures verified: {len(spec['runs'])}/{len(spec['runs'])}")
PY
}

refresh_current_experiment() {
  local candidate=""
  local log_path

  [[ -n "${CURRENT_LOG_DIRECTORY}" ]] || return 0
  for log_path in "${CURRENT_LOG_DIRECTORY}"/*.log; do
    [[ -f "${log_path}" ]] || continue
    if [[ -z "${candidate}" || "${log_path}" -nt "${candidate}" ]]; then
      candidate="${log_path}"
    fi
  done
  if [[ -n "${candidate}" ]]; then
    CURRENT_EXPERIMENT="$(basename -- "${candidate}" .log)"
  fi
}

write_wrapper_status() {
  local event="$1"
  local detail="${2:-}"
  local line

  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) event=${event} phase=${CURRENT_PHASE} experiment=${CURRENT_EXPERIMENT}${detail:+ ${detail}}"
  printf '%s\n' "${line}"
  if [[ -n "${WRAPPER_STATUS_PATH}" ]]; then
    printf '%s\n' "${line}" >>"${WRAPPER_STATUS_PATH}" || true
  fi
}

handle_err() {
  local status="$1"
  local line_number="$2"
  local failing_command="$3"

  refresh_current_experiment
  write_wrapper_status \
    "ERR" \
    "exit_status=${status} line=${line_number} command=$(printf '%q' "${failing_command}")"
}

handle_exit() {
  local status="$1"

  trap - EXIT
  refresh_current_experiment
  write_wrapper_status "EXIT" "exit_status=${status}"
}

handle_signal() {
  local signal_name="$1"
  local status="$2"

  trap - ERR HUP TERM INT
  refresh_current_experiment
  write_wrapper_status "SIGNAL" "signal=${signal_name} exit_status=${status}"
  exit "${status}"
}

install_internal_traps() {
  trap 'handle_err "$?" "$LINENO" "$BASH_COMMAND"' ERR
  trap 'handle_exit "$?"' EXIT
  trap 'handle_signal HUP 129' HUP
  trap 'handle_signal TERM 143' TERM
  trap 'handle_signal INT 130' INT
}

execute_sequence() {
  cd "${REPO_ROOT}"
  # shellcheck disable=SC1090
  source "${ACTIVATE_PATH}"
  export PYTHONUNBUFFERED=1

  CURRENT_PHASE="evolve-family-runner"
  CURRENT_EXPERIMENT="pending"
  CURRENT_LOG_DIRECTORY="${REPO_ROOT}/logs/canonical_evolvegcn_knn_hidden_factorial"
  write_wrapper_status "PHASE_START"
  python3 "${REPO_ROOT}/scripts/run_experiment_family.py" \
    --repo-root "${REPO_ROOT}" \
    --spec "${EVOLVE_SPEC}" \
    --execute
  refresh_current_experiment
  write_wrapper_status "PHASE_END" "exit_status=0"

  CURRENT_PHASE="evolve-verification"
  write_wrapper_status "PHASE_START"
  python3 "${REPO_ROOT}/scripts/verify_experiment_family.py" \
    --repo-root "${REPO_ROOT}" \
    --spec "${EVOLVE_SPEC}"
  verify_exact_splits "${EVOLVE_SPEC}"
  write_wrapper_status "PHASE_END" "exit_status=0"

  CURRENT_PHASE="static-family-runner"
  CURRENT_EXPERIMENT="pending"
  CURRENT_LOG_DIRECTORY="${REPO_ROOT}/logs/canonical_static_gcn_knn_hidden_factorial"
  write_wrapper_status "PHASE_START"
  python3 "${REPO_ROOT}/scripts/run_experiment_family.py" \
    --repo-root "${REPO_ROOT}" \
    --spec "${STATIC_SPEC}" \
    --execute
  refresh_current_experiment
  write_wrapper_status "PHASE_END" "exit_status=0"

  CURRENT_PHASE="static-verification"
  write_wrapper_status "PHASE_START"
  python3 "${REPO_ROOT}/scripts/verify_experiment_family.py" \
    --repo-root "${REPO_ROOT}" \
    --spec "${STATIC_SPEC}"
  verify_exact_splits "${STATIC_SPEC}"
  write_wrapper_status "PHASE_END" "exit_status=0"
}

if [[ "${INTERNAL_EXECUTE}" == true ]]; then
  install_internal_traps
  write_wrapper_status "WRAPPER_START"
  execute_sequence
  exit 0
fi

command -v tmux >/dev/null 2>&1 || {
  echo "ERROR: tmux is unavailable." >&2
  exit 2
}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
  exit 3
fi

# Both preflights reject incompatible completed rows and any partial output
# directory before a detached execution session is created.
"${REPO_ROOT}/envs/camels-gnn/bin/python" \
  "${REPO_ROOT}/scripts/run_experiment_family.py" \
  --repo-root "${REPO_ROOT}" \
  --spec "${EVOLVE_SPEC}" \
  --dry-run
"${REPO_ROOT}/envs/camels-gnn/bin/python" \
  "${REPO_ROOT}/scripts/run_experiment_family.py" \
  --repo-root "${REPO_ROOT}" \
  --spec "${STATIC_SPEC}" \
  --dry-run

LOG_DIRECTORY="${REPO_ROOT}/logs"
LOG_PATH="${LOG_DIRECTORY}/canonical-knn-hidden-factorial-$(date -u +%Y%m%dT%H%M%SZ).log"
WRAPPER_STATUS_PATH="${LOG_PATH%.log}.wrapper-status.log"

printf -v SESSION_COMMAND \
  'set -Eeuo pipefail; mkdir -p %q; export PYTHONUNBUFFERED=1 WRAPPER_STATUS_PATH=%q; bash %q --internal-execute 2>&1 | tee -a %q' \
  "${LOG_DIRECTORY}" \
  "${WRAPPER_STATUS_PATH}" \
  "${REPO_ROOT}/scripts/launch_knn_by_hidden_dim_factorial_tmux.sh" \
  "${LOG_PATH}"

tmux new-session -d -s "${SESSION_NAME}" \
  "bash -lc $(printf '%q' "${SESSION_COMMAND}")" \; \
  set-option -w -t "${SESSION_NAME}" remain-on-exit on

echo "Launched sequential factorial runner in tmux session: ${SESSION_NAME}"
echo "Combined log: ${LOG_PATH}"
echo "Wrapper status log: ${WRAPPER_STATUS_PATH}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Monitor: tail -f ${LOG_PATH}"
