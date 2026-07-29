#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

bash "${REPO_ROOT}/scripts/finalize_experiment_family.sh" \
  "configs/experiment_families/canonical_evolvegcn_universe_scaling.json"
