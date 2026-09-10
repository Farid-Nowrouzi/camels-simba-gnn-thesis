#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly PYTHON="/home/ml/thesis-camels/envs/camels-gnn/bin/python"
readonly AUDITOR="$ROOT/scripts/validation/prepare_u1000_top1500_evolvegcn_o.py"
readonly CONFIG_DIR="$ROOT/configs/production/u1000_top1500_evolvegcn_o"
readonly PROTOCOL="$ROOT/reports/experiment_registry/u1000_top1500_evolvegcn_o_temporal_protocol.json"
readonly PROTOCOL_SHA="45a21cc4500199a87116f72c707967a721cf9ead832cb04cde6f2dedf112cb2e"
readonly SEED123_CHECKPOINT_SHA="14b0e22641b0e58ad744338311fea93632f2d973e6db113184f2cc3779814210"
readonly RUN42="experiments/evolvegcn_o_u1000_top1500_sparse_train700_seed42_none_h32_l2_mean_temporal_mean_linear"
readonly RUN123="experiments/evolvegcn_o_u1000_top1500_sparse_train700_seed123_none_h32_l2_mean_temporal_mean_linear"
readonly RUN2025="experiments/evolvegcn_o_u1000_top1500_sparse_train700_seed2025_none_h32_l2_mean_temporal_mean_linear"

usage() { echo "Usage: $0 --expected-head COMMIT" >&2; }
expected_head=""
while (( $# )); do
  case "$1" in
    --expected-head) expected_head="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$expected_head" ]] || { usage; exit 2; }
[[ -x "$PYTHON" ]] || { echo "FAIL: missing project interpreter $PYTHON" >&2; exit 1; }

cd "$ROOT"
mkdir -p logs/evolvegcn_o
readonly LOG="logs/evolvegcn_o/evolvegcn_o_recovery_123_2025_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

current_seed="preflight"
current_stage="initialization"
timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
on_err() {
  local status=$?
  echo "ERROR utc=$(timestamp) seed=$current_seed stage=$current_stage line=${BASH_LINENO[0]} exit=$status command=${BASH_COMMAND}" >&2
  exit "$status"
}
on_signal() {
  local signal="$1"
  echo "ERROR utc=$(timestamp) seed=$current_seed stage=$current_stage signal=$signal" >&2
  exit 128
}
trap on_err ERR
trap 'on_signal SIGHUP' HUP
trap 'on_signal SIGINT' INT
trap 'on_signal SIGTERM' TERM

declare -Ar SEED42_HASHES=(
  [config.json]="606fec27172cd4cd4b6a3d499d0e7e3a97adfee834031a8fb4bba9c48fabd66e"
  [metrics.json]="e34cd4ef877620772f4d5f7a510cc164fd5febbe2f2526f3bde0e819dfd04717"
  [train_log.csv]="8d98ee52a20dfb332b0142b535329dfe015624b0293303e3390f9ca62172197c"
  [run_metadata.json]="e4960550e1f548acab8764a0c7b00fcc2ca0401e75abd8b107651562c1c69449"
  [checkpoints/best_model.pt]="04dc60e5c70465e4d52b4f9806ea315afc6658ef5a2e1e302719082925762454"
  [predictions/val_predictions.csv]="8c4deae4d8d039ede925777f80e7c74112f2751241459a8d95316eac4bb0472d"
  [predictions/test_predictions.csv]="85d6b1b399f5e5eb92efe3d6b32231e1f0ce8ff66fa05c3015a6910045fa5f96"
)

verify_seed42() {
  local relative actual
  for relative in "${!SEED42_HASHES[@]}"; do
    [[ -f "$RUN42/$relative" ]] || { echo "FAIL: missing preserved seed42 artifact $relative" >&2; return 1; }
    actual="$(sha256sum "$RUN42/$relative" | cut -d' ' -f1)"
    [[ "$actual" == "${SEED42_HASHES[$relative]}" ]] || {
      echo "FAIL: seed42 artifact hash changed: $relative" >&2
      return 1
    }
  done
}

echo "EvolveGCN-O recovery: seed123 finalization -> seed2025 training"
echo "Started UTC: $(timestamp)"
current_stage="repository_and_protocol_binding"
actual_head="$(git rev-parse HEAD)"
[[ "$actual_head" == "$expected_head" ]] || { echo "FAIL: HEAD mismatch" >&2; exit 1; }
[[ "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" == "$PROTOCOL_SHA" ]] || {
  echo "FAIL: frozen protocol SHA mismatch" >&2; exit 1;
}

current_stage="seed42_integrity_before"
verify_seed42
"$PYTHON" -u "$AUDITOR" --validate-run "$CONFIG_DIR/evolvegcn_o_seed42.json"
[[ -d "$RUN123" ]] || { echo "FAIL: seed123 partial run is missing" >&2; exit 1; }
[[ "$(sha256sum "$RUN123/checkpoints/best_model.pt" | cut -d' ' -f1)" == "$SEED123_CHECKPOINT_SHA" ]] || {
  echo "FAIL: seed123 checkpoint hash mismatch" >&2; exit 1;
}
[[ ! -e "$RUN2025" ]] || { echo "FAIL: seed2025 canonical collision" >&2; exit 1; }

current_stage="recovery_preflight"
"$PYTHON" -u "$AUDITOR" --recovery-preflight

current_seed="123"
current_stage="inference_only_finalization"
echo "$(timestamp) seed=123 finalization_start checkpoint=$SEED123_CHECKPOINT_SHA"
"$PYTHON" -u -m src.training.train_evolvegcn_o \
  --config "$CONFIG_DIR/evolvegcn_o_seed123.json" \
  --finalize-existing-run \
  --expected-checkpoint-sha256 "$SEED123_CHECKPOINT_SHA"
echo "$(timestamp) seed=123 finalization_complete"
current_stage="official_validation"
"$PYTHON" -u "$AUDITOR" --validate-run "$CONFIG_DIR/evolvegcn_o_seed123.json"
echo "$(timestamp) seed=123 auditor_PASS"

current_seed="2025"
current_stage="collision_check"
[[ ! -e "$RUN2025" ]] || { echo "FAIL: refusing to overwrite seed2025" >&2; exit 1; }
current_stage="training"
echo "$(timestamp) seed=2025 training_process_start"
"$PYTHON" -u -m src.training.train_evolvegcn_o --config "$CONFIG_DIR/evolvegcn_o_seed2025.json"
echo "$(timestamp) seed=2025 trainer_exit_status=0"
current_stage="official_validation"
"$PYTHON" -u "$AUDITOR" --validate-run "$CONFIG_DIR/evolvegcn_o_seed2025.json"
echo "$(timestamp) seed=2025 auditor_PASS"

current_seed="final"
current_stage="three_seed_completion_audit"
verify_seed42
"$PYTHON" -u "$AUDITOR" --validate-all
echo "$(timestamp) EVOLVEGCN-O RECOVERY COMPLETE"
