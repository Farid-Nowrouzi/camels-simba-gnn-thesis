#!/usr/bin/env bash
# Validation-only split x training-RNG diagnostic. Never invokes protected test evaluation.
set -Eeuo pipefail

readonly ROOT="/home/ml/thesis-camels"
readonly PYTHON="$ROOT/envs/camels-gnn/bin/python"
readonly BRANCH="thesis-notebook19-physics-informed-invariant-gnn"
readonly DIAGNOSTIC_IMPLEMENTATION_COMMIT="0019aabd6d88afaaf1cb4a86f8ac5cf31559824a"
readonly PROTOCOL="reports/experiment_registry/u1000_top1500_periodic_invariant_gnn_protocol_freeze.json"
readonly PRODUCTION_IMPLEMENTATION="reports/experiment_registry/u1000_top1500_periodic_invariant_gnn_implementation_freeze.json"
readonly VALIDATION_TRAINING="reports/experiment_registry/u1000_top1500_periodic_invariant_gnn_validation_training_freeze.json"
readonly DIAGNOSTIC_IMPLEMENTATION="reports/experiment_registry/u1000_top1500_periodic_invariant_gnn_split_rng_diagnostic_implementation_freeze.json"
readonly DATASET="data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
readonly PROTOCOL_SHA="edff75346e61070735eeb9cae5d404d9827338f695835c139e9db37a19a1cbf6"
readonly PRODUCTION_IMPLEMENTATION_SHA="139afbdde8f5402a9be861af4d56adff8644e9843c7fa8a0d250089470ad3654"
readonly VALIDATION_TRAINING_SHA="97bb9c404f82c3f0ad860df28ba5b0f3f5e42872231aecdfe1b3fd6c7b482be8"
readonly DIAGNOSTIC_IMPLEMENTATION_SHA="4f3010d3a8fa08002d36f31b4a058efc7eb18df6e57e47776b8548fa7d2b8057"
readonly DATASET_SHA="ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
readonly LOG_DIR="outputs/logs/periodic_invariant_gnn_split_rng"
readonly LOCK_FILE="/tmp/periodic_invariant_gnn_split_rng_diagnostic.lock"

AUDIT_ONLY=false
CURRENT_CELL="preflight"
CURRENT_OUTPUT="(none)"

usage() { printf 'Usage: %s [--audit-only]\n' "$0" >&2; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha256() { sha256sum "$1" | awk '{print $1}'; }

on_failure() {
  local status="$?"
  printf '\nDIAGNOSTIC STOPPED\ncell: %s\noutput: %s\nexit status: %s\n' "$CURRENT_CELL" "$CURRENT_OUTPUT" "$status" >&2
  printf 'Partial evidence is preserved; no cell will be retried, deleted, resumed, or skipped.\n' >&2
  exit "$status"
}
trap on_failure ERR INT TERM

case "$#" in
  0) ;;
  1) [[ "$1" == "--audit-only" ]] && AUDIT_ONLY=true || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac

declare -a CELLS=(split42_rng123 split42_rng2025 split123_rng42 split123_rng2025 split2025_rng42 split2025_rng123)
declare -A SPLIT_IDS=([split42_rng123]=42 [split42_rng2025]=42 [split123_rng42]=123 [split123_rng2025]=123 [split2025_rng42]=2025 [split2025_rng123]=2025)
declare -A RNG_SEEDS=([split42_rng123]=123 [split42_rng2025]=2025 [split123_rng42]=42 [split123_rng2025]=2025 [split2025_rng42]=42 [split2025_rng123]=123)
declare -A SPLITS=(
  [42]="configs/splits/u1000_top1500_none_k8_sparse/seed42_train700.json"
  [123]="configs/splits/u1000_top1500_none_k8_sparse/seed123_train700.json"
  [2025]="configs/splits/u1000_top1500_none_k8_sparse/seed2025_train700.json"
)
declare -A SPLIT_SHAS=(
  [42]="f5556ec5c193e7cb80f2231705edbdae32d4de206889dec308a819bdde427ab7"
  [123]="18a295106ec844848053f3040d2be3cdf73a443010be31e6e2cd962362982471"
  [2025]="c233c0631b1a24d963ffccc7c6389054fddf135a7aabc4dc4ab7bf5976fab3a9"
)
declare -A OUTPUTS=(
  [split42_rng123]="experiments/diagnostic_periodic_invariant_gnn_u1000_top1500_split42_rng123"
  [split42_rng2025]="experiments/diagnostic_periodic_invariant_gnn_u1000_top1500_split42_rng2025"
  [split123_rng42]="experiments/diagnostic_periodic_invariant_gnn_u1000_top1500_split123_rng42"
  [split123_rng2025]="experiments/diagnostic_periodic_invariant_gnn_u1000_top1500_split123_rng2025"
  [split2025_rng42]="experiments/diagnostic_periodic_invariant_gnn_u1000_top1500_split2025_rng42"
  [split2025_rng123]="experiments/diagnostic_periodic_invariant_gnn_u1000_top1500_split2025_rng123"
)

require_sha() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "$path" ]] || fail "$label is missing: $path"
  actual="$(sha256 "$path")"
  [[ "$actual" == "$expected" ]] || fail "$label SHA256 mismatch: expected $expected, got $actual"
}

check_repository() {
  [[ "$(git branch --show-current)" == "$BRANCH" ]] || fail "wrong branch; require $BRANCH"
  git merge-base --is-ancestor "$DIAGNOSTIC_IMPLEMENTATION_COMMIT" HEAD || fail "diagnostic implementation commit is not an ancestor of HEAD"
  [[ -z "$(git status --short --untracked-files=no)" ]] || fail "tracked working tree is not clean"
}

check_freezes_and_inputs() {
  require_sha "$PROTOCOL" "$PROTOCOL_SHA" "protocol freeze"
  require_sha "$PRODUCTION_IMPLEMENTATION" "$PRODUCTION_IMPLEMENTATION_SHA" "production implementation freeze"
  require_sha "$VALIDATION_TRAINING" "$VALIDATION_TRAINING_SHA" "validation-training freeze"
  require_sha "$DIAGNOSTIC_IMPLEMENTATION" "$DIAGNOSTIC_IMPLEMENTATION_SHA" "diagnostic implementation freeze"
  require_sha "$DATASET" "$DATASET_SHA" "canonical dataset"
  "$PYTHON" - "$DIAGNOSTIC_IMPLEMENTATION" "$ROOT" <<'PY'
import hashlib, json, sys
freeze_path, root = map(__import__('pathlib').Path, sys.argv[1:])
freeze = json.loads(freeze_path.read_text())
if freeze.get('protected_test_evaluated') is not False or freeze.get('production_artifacts_modified') is not False:
    raise SystemExit('diagnostic implementation freeze protection fields are invalid')
entries = freeze.get('implementation_file_sha256')
if not isinstance(entries, dict) or not entries:
    raise SystemExit('diagnostic implementation freeze has no protected file mapping')
for relative, expected in entries.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'protected diagnostic file hash mismatch: {relative}')
print(f'verified {len(entries)} diagnostic implementation-protected hashes')
PY
}

check_split() {
  local split_id="$1" path="${SPLITS[$1]}" expected="${SPLIT_SHAS[$1]}"
  require_sha "$path" "$expected" "split${split_id} manifest"
  "$PYTHON" - "$path" "$split_id" <<'PY'
import json, sys
from pathlib import Path
path, expected_seed = Path(sys.argv[1]), int(sys.argv[2])
manifest = json.loads(path.read_text())
if manifest.get('seed') != expected_seed:
    raise SystemExit('split manifest seed metadata mismatch')
counts = manifest.get('counts')
if not isinstance(counts, dict) or tuple(counts.get(k) for k in ('train', 'val', 'test')) != (700, 99, 201):
    raise SystemExit('split counts are not 700/99/201')
print(f'verified split{expected_seed}: train=700 val=99 test=201')
PY
}

check_test_artifacts() {
  local output="$1" found
  [[ -e "$output" ]] || return 0
  found="$(find "$output" -type f \( -name test_predictions.csv -o -name test_metrics.json -o -name test_evaluation_started.json -o -name finalization_report.json \) -print -quit)"
  [[ -z "$found" ]] || fail "protected test artifact found: $found"
}

check_all_outputs_absent() {
  local cell output
  for cell in "${CELLS[@]}"; do
    output="${OUTPUTS[$cell]}"
    [[ ! -e "$output" ]] || fail "diagnostic output already exists; refusing overwrite/resume: $output"
    check_test_artifacts "$output"
  done
}

check_trainer_cli() {
  "$PYTHON" -m src.training.train_periodic_invariant_gnn --help | grep -F -- '--training-rng-seed' >/dev/null || fail 'trainer CLI does not expose --training-rng-seed'
}

check_gpu_compute_processes() {
  local selector first_selector gpu_rows selected_uuid apps matching
  selector="$CUDA_VISIBLE_DEVICES"; first_selector="${selector%%,*}"
  gpu_rows="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits)"
  if [[ "$first_selector" =~ ^[0-9]+$ ]]; then
    selected_uuid="$(awk -F', ' -v i="$first_selector" '$1 == i {print $2}' <<<"$gpu_rows")"
  elif [[ "$first_selector" == GPU-* ]]; then
    selected_uuid="$(awk -F', ' -v u="$first_selector" '$2 == u {print $2}' <<<"$gpu_rows")"
  else
    fail "cannot map CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES to a physical GPU"
  fi
  [[ -n "$selected_uuid" ]] || fail "selected CUDA GPU is absent from nvidia-smi inventory"
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  matching="$(awk -F', ' -v u="$selected_uuid" '$3 == u {print}' <<<"$apps")"
  [[ -z "$matching" ]] || fail "active compute process(es) detected on selected GPU UUID $selected_uuid: $matching"
  printf 'No active NVIDIA compute processes on selected GPU UUID %s.\n' "$selected_uuid"
}

check_gpu() {
  if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then export CUDA_VISIBLE_DEVICES=0; fi
  [[ -n "$CUDA_VISIBLE_DEVICES" ]] || fail 'CUDA_VISIBLE_DEVICES is explicitly empty'
  command -v nvidia-smi >/dev/null || fail 'nvidia-smi is unavailable'
  nvidia-smi >/dev/null || fail 'nvidia-smi cannot communicate with the NVIDIA driver'
  local info name
  info="$("$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('no CUDA device is available')
print(torch.cuda.get_device_name(0))
PY
)"
  name="$info"
  [[ "$name" == *"NVIDIA L40"* ]] || fail "selected GPU is not NVIDIA L40: $name"
  printf 'CUDA logical device 0: %s\n' "$name"
  check_gpu_compute_processes
}

render_command() {
  local cell="$1" split_id="${SPLIT_IDS[$1]}" rng="${RNG_SEEDS[$1]}" output="${OUTPUTS[$1]}"
  printf '%q ' "$PYTHON" -m src.training.train_periodic_invariant_gnn \
    --dataset-path "$DATASET" --split-manifest-path "${SPLITS[$split_id]}" \
    --seed "$split_id" --training-rng-seed "$rng" --output-dir "$output" \
    --protocol-freeze-path "$PROTOCOL" --epochs 300 --batch-size 8 --patience 40 --device cuda
  printf '\n'
}

postcheck_cell() {
  local cell="$1" split_id="${SPLIT_IDS[$1]}" rng="${RNG_SEEDS[$1]}" output="${OUTPUTS[$1]}"
  local artifact
  for artifact in metadata.json epoch_metrics.csv best_validation_checkpoint.pt; do
    [[ -s "$output/$artifact" ]] || fail "$cell missing required artifact: $output/$artifact"
  done
  check_test_artifacts "$output"
  "$PYTHON" - "$output" "$split_id" "$rng" "${SPLIT_SHAS[$split_id]}" "$PROTOCOL_SHA" <<'PY'
import csv, hashlib, json, math, sys
from pathlib import Path
out, split_seed, rng_seed, split_sha, protocol_sha = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
metadata = json.loads((out / 'metadata.json').read_text())
expected = {'seed': split_seed, 'split_seed': split_seed, 'training_rng_seed': rng_seed, 'parameter_count': 5270, 'split_manifest_sha256': split_sha, 'protocol_freeze_sha256': protocol_sha, 'test_status': 'pending / forbidden', 'scientific_run': True}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(f'metadata {key} mismatch: expected {value!r}, got {metadata.get(key)!r}')
if not isinstance(metadata.get('best_validation_mse'), (int, float)) or not math.isfinite(metadata['best_validation_mse']):
    raise SystemExit('best_validation_mse is not finite')
checkpoint_sha = hashlib.sha256((out / 'best_validation_checkpoint.pt').read_bytes()).hexdigest()
if metadata.get('checkpoint_sha256') != checkpoint_sha:
    raise SystemExit('checkpoint SHA does not match metadata')
with (out / 'epoch_metrics.csv').open(newline='') as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit('epoch_metrics.csv has no data rows')
try:
    epochs = [int(row['epoch']) for row in rows]
    metrics = [float(row[key]) for row in rows for key in ('train_mse', 'validation_mse')]
except (KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f'epoch metrics schema/value error: {exc}')
if epochs != list(range(1, len(rows) + 1)) or not all(math.isfinite(value) for value in metrics):
    raise SystemExit('epoch metrics are non-finite or epochs are not consecutive')
print(f'POSTCHECK {out.name}: best_epoch={metadata.get("best_epoch")} validation_mse={metadata["best_validation_mse"]} checkpoint_sha256={checkpoint_sha}')
PY
}

precheck_cell() {
  local cell="$1" split_id="${SPLIT_IDS[$1]}" output="${OUTPUTS[$1]}"
  check_repository
  check_freezes_and_inputs
  check_split "$split_id"
  check_trainer_cli
  check_gpu
  [[ ! -e "$output" ]] || fail "$cell output appeared; refusing overwrite/resume: $output"
  check_test_artifacts "$output"
}

run_cell() {
  local cell="$1" log="$LOG_DIR/${1}.log"
  CURRENT_CELL="$cell"; CURRENT_OUTPUT="${OUTPUTS[$cell]}"
  precheck_cell "$cell"
  printf '\nSTARTING %s\n' "$cell"
  render_command "$cell"
  "$PYTHON" -m src.training.train_periodic_invariant_gnn \
    --dataset-path "$DATASET" \
    --split-manifest-path "${SPLITS[${SPLIT_IDS[$cell]}]}" \
    --seed "${SPLIT_IDS[$cell]}" \
    --training-rng-seed "${RNG_SEEDS[$cell]}" \
    --output-dir "${OUTPUTS[$cell]}" \
    --protocol-freeze-path "$PROTOCOL" \
    --epochs 300 --batch-size 8 --patience 40 --device cuda 2>&1 | tee "$log"
  postcheck_cell "$cell"
  printf 'COMPLETE %s\n' "$cell"
}

cd "$ROOT"
[[ -x "$PYTHON" ]] || fail "canonical Python is missing or not executable: $PYTHON"
command -v flock >/dev/null || fail 'flock is required for safe concurrency control'
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another diagnostic launcher holds $LOCK_FILE"
git check-ignore -q "$LOG_DIR/probe" || fail "operational log directory is not ignored: $LOG_DIR"

check_repository
check_freezes_and_inputs
for split_id in 42 123 2025; do check_split "$split_id"; done
check_trainer_cli
check_gpu
check_all_outputs_absent

printf '\nExact diagnostic cell order:\n'
for cell in "${CELLS[@]}"; do render_command "$cell"; done
if "$AUDIT_ONLY"; then
  printf '\nAUDIT-ONLY COMPLETE: zero trainer commands were invoked and zero diagnostic experiment directories were created.\n'
  exit 0
fi

mkdir -p "$LOG_DIR"
for cell in "${CELLS[@]}"; do run_cell "$cell"; done
printf '\nVALIDATION-ONLY DIAGNOSTIC COMPLETE\nTEST SET ACCESSED = NO\n'
