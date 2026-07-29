#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SPEC_PATH=""
SESSION_NAME=""

usage() {
  echo "Usage: bash scripts/launch_experiment_family_tmux.sh --spec FAMILY.json --session NAME"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --spec)
      [[ "$#" -ge 2 ]] || {
        echo "ERROR: --spec requires a value." >&2
        exit 2
      }
      SPEC_PATH="$2"
      shift 2
      ;;
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
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${SPEC_PATH}" && -n "${SESSION_NAME}" ]] || {
  usage >&2
  exit 2
}

if [[ "${SPEC_PATH}" != /* ]]; then
  SPEC_PATH="${REPO_ROOT}/${SPEC_PATH}"
fi
[[ -f "${SPEC_PATH}" ]] || {
  echo "ERROR: family specification does not exist: ${SPEC_PATH}" >&2
  exit 2
}
[[ "${SESSION_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "ERROR: session name may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
}
command -v tmux >/dev/null 2>&1 || {
  echo "ERROR: tmux is unavailable." >&2
  exit 2
}

python3 "${REPO_ROOT}/scripts/run_experiment_family.py" \
  --repo-root "${REPO_ROOT}" \
  --spec "${SPEC_PATH}" \
  --dry-run

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
  exit 3
fi

ACTIVATE_PATH="${REPO_ROOT}/envs/camels-gnn/bin/activate"
[[ -f "${ACTIVATE_PATH}" ]] || {
  echo "ERROR: documented environment activation script is missing: ${ACTIVATE_PATH}" >&2
  exit 2
}

printf -v LAUNCH_COMMAND \
  'cd %q && source %q && exec python3 %q --repo-root %q --spec %q --execute' \
  "${REPO_ROOT}" \
  "${ACTIVATE_PATH}" \
  "${REPO_ROOT}/scripts/run_experiment_family.py" \
  "${REPO_ROOT}" \
  "${SPEC_PATH}"

tmux new-session -d -s "${SESSION_NAME}" "${LAUNCH_COMMAND}"

echo "Launched one detached runner in tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Monitor panes: tmux list-panes -t ${SESSION_NAME} -F '#{pane_pid} #{pane_current_command}'"
