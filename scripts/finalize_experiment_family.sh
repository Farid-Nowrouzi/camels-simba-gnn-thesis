#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/finalize_experiment_family.sh FAMILY_SPEC.json" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SPEC_INPUT="$1"

if [[ "${SPEC_INPUT}" = /* ]]; then
  SPEC_PATH="${SPEC_INPUT}"
else
  SPEC_PATH="${REPO_ROOT}/${SPEC_INPUT}"
fi

[[ -f "${SPEC_PATH}" ]] || {
  echo "ERROR: family specification does not exist: ${SPEC_PATH}" >&2
  exit 2
}

RESULT_OUTPUT="$(
  python3 - "${SPEC_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    spec = json.load(handle)
print(spec["results"]["default_output_directory"])
PY
)"

python3 "${REPO_ROOT}/scripts/verify_experiment_family.py" \
  --repo-root "${REPO_ROOT}" \
  --spec "${SPEC_PATH}"

bash "${REPO_ROOT}/scripts/refresh_experiment_registry.sh"

python3 "${REPO_ROOT}/scripts/build_experiment_family_results.py" \
  --repo-root "${REPO_ROOT}" \
  --spec "${SPEC_PATH}" \
  --output-dir "${RESULT_OUTPUT}"

echo "Finalization complete."
echo "Registry: ${REPO_ROOT}/reports/experiment_registry"
echo "Family results: ${REPO_ROOT}/${RESULT_OUTPUT}"
