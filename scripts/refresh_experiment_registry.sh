#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

PYTHON="${REPO_ROOT}/envs/camels-gnn/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON=python3

"${PYTHON}" "${REPO_ROOT}/scripts/build_experiment_registry.py" \
  --repo-root "${REPO_ROOT}" \
  --experiments-root experiments \
  --report-dir reports/experiment_registry
