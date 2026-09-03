#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly PYTHON="/home/ml/thesis-camels/envs/camels-gnn/bin/python"
readonly AUDITOR="$ROOT/scripts/validation/prepare_u1000_top1500_temporal_architectures.py"
readonly CONFIG_DIR="$ROOT/configs/production/u1000_top1500_temporal_architectures"

[[ -x "$PYTHON" ]] || { echo "FAIL: missing project interpreter $PYTHON" >&2; exit 1; }
cd "$ROOT"
"$PYTHON" "$AUDITOR" --architecture transformer

for seed in 42 123 2025; do
  config="$CONFIG_DIR/gcn_temporal_transformer_seed${seed}.json"
  "$PYTHON" -m src.training.train_gcn_temporal_transformer --config "$config"
  "$PYTHON" "$AUDITOR" --validate-run "$config"
done
