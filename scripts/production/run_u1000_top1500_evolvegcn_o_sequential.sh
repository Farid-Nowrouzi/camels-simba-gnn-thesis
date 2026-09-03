#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly PYTHON="/home/ml/thesis-camels/envs/camels-gnn/bin/python"
readonly AUDITOR="$ROOT/scripts/validation/prepare_u1000_top1500_evolvegcn_o.py"
readonly CONFIG_DIR="$ROOT/configs/production/u1000_top1500_evolvegcn_o"
readonly PROTOCOL="$ROOT/reports/experiment_registry/u1000_top1500_evolvegcn_o_temporal_protocol.json"

usage() {
  echo "Usage: $0 --expected-head COMMIT --expected-protocol-sha SHA256" >&2
}

expected_head=""
expected_protocol_sha=""
while (( $# )); do
  case "$1" in
    --expected-head) expected_head="${2:-}"; shift 2 ;;
    --expected-protocol-sha) expected_protocol_sha="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$expected_head" && -n "$expected_protocol_sha" ]] || { usage; exit 2; }
[[ -x "$PYTHON" ]] || { echo "FAIL: missing project interpreter $PYTHON" >&2; exit 1; }

cd "$ROOT"
actual_head="$(git rev-parse HEAD)"
actual_protocol_sha="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
[[ "$actual_head" == "$expected_head" ]] || {
  echo "FAIL: HEAD $actual_head does not match expected $expected_head" >&2; exit 1;
}
[[ "$actual_protocol_sha" == "$expected_protocol_sha" ]] || {
  echo "FAIL: protocol SHA $actual_protocol_sha does not match expected $expected_protocol_sha" >&2; exit 1;
}

"$PYTHON" -u "$AUDITOR"
mkdir -p logs/evolvegcn_o
readonly LOG="logs/evolvegcn_o/evolvegcn_o_3seed_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "EvolveGCN-O sequential production run"
echo "HEAD: $actual_head"
echo "Protocol SHA-256: $actual_protocol_sha"
echo "Seeds: 42 -> 123 -> 2025"

for seed in 42 123 2025; do
  config="$CONFIG_DIR/evolvegcn_o_seed${seed}.json"
  echo "STARTING EVOLVEGCN-O SEED $seed"
  "$PYTHON" -u -m src.training.train_evolvegcn_o --config "$config"
  echo "VALIDATING EVOLVEGCN-O SEED $seed"
  "$PYTHON" -u "$AUDITOR" --validate-run "$config"
  echo "EVOLVEGCN-O SEED $seed COMPLETE AND VALIDATED"
done

echo "EVOLVEGCN-O THREE-SEED PRODUCTION RUN COMPLETE"
