#!/usr/bin/env bash
# Production launcher for the frozen periodic physics-informed invariant GNN.
# This launcher trains and validates only; it never evaluates the protected test set.
set -Eeuo pipefail

readonly ROOT="/home/ml/thesis-camels"
readonly PYTHON="/home/ml/thesis-camels/envs/camels-gnn/bin/python"
readonly BRANCH="thesis-notebook19-physics-informed-invariant-gnn"
readonly SCIENTIFIC_CHECKPOINT="3ec3e255c3071fe6d82114b8c96e586acdb50b46"
readonly PROTOCOL="reports/experiment_registry/u1000_top1500_periodic_invariant_gnn_protocol_freeze.json"
readonly IMPLEMENTATION_FREEZE="reports/experiment_registry/u1000_top1500_periodic_invariant_gnn_implementation_freeze.json"
readonly DATASET="data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
readonly PROTOCOL_SHA="edff75346e61070735eeb9cae5d404d9827338f695835c139e9db37a19a1cbf6"
readonly IMPLEMENTATION_SHA="139afbdde8f5402a9be861af4d56adff8644e9843c7fa8a0d250089470ad3654"
readonly DATASET_SHA="ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
readonly LOG_DIR="outputs/logs/periodic_invariant_gnn"
readonly LOCK_FILE="/tmp/periodic_invariant_gnn_production.lock"

AUDIT_ONLY=false
CURRENT_SEED="preflight"
CURRENT_OUTPUT="(none)"

usage() {
  printf 'Usage: %s [--audit-only]\n' "$0" >&2
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

on_failure() {
  local status="$?"
  printf '\nPRODUCTION STOPPED\nseed: %s\noutput: %s\nexit status: %s\n' \
    "$CURRENT_SEED" "$CURRENT_OUTPUT" "$status" >&2
  printf 'Partial evidence has been preserved; nothing was deleted or retried.\n' >&2
  exit "$status"
}
trap on_failure ERR INT TERM

case "$#" in
  0) ;;
  1) [[ "$1" == "--audit-only" ]] && AUDIT_ONLY=true || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac

declare -a SEEDS=(42 123 2025)
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
  [42]="experiments/periodic_invariant_gnn_u1000_top1500_final_train700_seed42_k8_fixedscale_h17_l3_mean"
  [123]="experiments/periodic_invariant_gnn_u1000_top1500_final_train700_seed123_k8_fixedscale_h17_l3_mean"
  [2025]="experiments/periodic_invariant_gnn_u1000_top1500_final_train700_seed2025_k8_fixedscale_h17_l3_mean"
)

sha256() { sha256sum "$1" | awk '{print $1}'; }

require_sha() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "$path" ]] || fail "$label is missing: $path"
  actual="$(sha256 "$path")"
  [[ "$actual" == "$expected" ]] || fail "$label SHA256 mismatch: expected $expected, got $actual"
}

check_repository() {
  [[ "$(git branch --show-current)" == "$BRANCH" ]] || fail "wrong branch; require $BRANCH"
  git merge-base --is-ancestor "$SCIENTIFIC_CHECKPOINT" HEAD || fail "scientific checkpoint is not an ancestor of HEAD"
  [[ -z "$(git status --short --untracked-files=no)" ]] || fail "tracked working tree is not clean"
}

check_freezes_and_inputs() {
  require_sha "$PROTOCOL" "$PROTOCOL_SHA" "protocol freeze"
  require_sha "$IMPLEMENTATION_FREEZE" "$IMPLEMENTATION_SHA" "implementation freeze"
  require_sha "$DATASET" "$DATASET_SHA" "canonical dataset"
  "$PYTHON" - "$IMPLEMENTATION_FREEZE" "$ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
freeze_path, root = map(Path, sys.argv[1:])
freeze = json.loads(freeze_path.read_text())
entries = freeze.get("implementation_file_sha256")
if not isinstance(entries, dict) or not entries:
    raise SystemExit("implementation freeze has no implementation_file_sha256 mapping")
for relative, expected in entries.items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"protected implementation file missing: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"protected implementation hash mismatch: {relative}")
print(f"verified {len(entries)} implementation-protected hashes")
PY
}

check_split() {
  local seed="$1" split="${SPLITS[$1]}" expected="${SPLIT_SHAS[$1]}"
  require_sha "$split" "$expected" "seed $seed split manifest"
  "$PYTHON" - "$split" "$seed" <<'PY'
import json, sys
from pathlib import Path
path, expected_seed = Path(sys.argv[1]), int(sys.argv[2])
d = json.loads(path.read_text())
if d.get("seed") != expected_seed:
    raise SystemExit("split seed metadata mismatch")
parts = {name: d.get(f"{name}_ids") for name in ("train", "val", "test")}
if any(not isinstance(ids, list) for ids in parts.values()):
    raise SystemExit("split manifest is missing partition IDs")
if tuple(len(parts[name]) for name in ("train", "val", "test")) != (700, 99, 201):
    raise SystemExit("split partition counts are not 700/99/201")
combined = parts["train"] + parts["val"] + parts["test"]
if len(combined) != len(set(combined)):
    raise SystemExit("split partitions are not unique and pairwise disjoint")
print(f"verified split seed {expected_seed}: train=700 val=99 test=201")
PY
}

check_all_splits() {
  local seed
  for seed in "${SEEDS[@]}"; do check_split "$seed"; done
}

check_output_absence() {
  local seed output
  for seed in "${SEEDS[@]}"; do
    output="${OUTPUTS[$seed]}"
    [[ ! -e "$output" ]] || fail "production output already exists; refusing overwrite/resume: $output"
  done
}

check_test_artifacts() {
  local output="$1"
  [[ -e "$output" ]] || return 0
  local found
  found="$(find "$output" -type f \( -name 'test_predictions.csv' -o -name 'test_metrics.json' -o -name 'test_evaluation_started.json' -o -name 'finalization_report.json' -o -iname '*test*prediction*' -o -iname '*test*metric*' -o -iname '*finalization*report*' \) -print -quit)"
  [[ -z "$found" ]] || fail "protected test artifact found in $output: $found"
}

check_all_test_artifacts() {
  local seed
  for seed in "${SEEDS[@]}"; do check_test_artifacts "${OUTPUTS[$seed]}"; done
}

check_trainer_cli() {
  "$PYTHON" -m src.training.train_periodic_invariant_gnn --help >/dev/null
}

check_gpu() {
  if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
    printf 'CUDA_VISIBLE_DEVICES was unset; using physical GPU 0 (logical CUDA device 0).\n'
  else
    [[ -n "$CUDA_VISIBLE_DEVICES" ]] || fail "CUDA_VISIBLE_DEVICES is explicitly empty and therefore ambiguous"
    printf 'Preserving user CUDA_VISIBLE_DEVICES=%s; using logical CUDA device 0.\n' "$CUDA_VISIBLE_DEVICES"
  fi
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
  nvidia-smi >/dev/null || fail "nvidia-smi cannot communicate with the NVIDIA driver"
  GPU_INFO="$("$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
if torch.cuda.device_count() < 1:
    raise SystemExit("no logical CUDA devices are visible")
p = torch.cuda.get_device_properties(0)
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print("logical_device=0")
print(f"gpu_name={torch.cuda.get_device_name(0)}")
print(f"total_memory={p.total_memory}")
PY
)"
  printf '%s\n' "$GPU_INFO"
  GPU_NAME="$(awk -F= '/^gpu_name=/{print substr($0,10)}' <<<"$GPU_INFO")"
  [[ "$GPU_NAME" == *"NVIDIA L40"* ]] || fail "selected GPU is not NVIDIA L40: $GPU_NAME"
  check_gpu_compute_processes
}

check_gpu_compute_processes() {
  # nvidia-smi reports compute-app rows with GPU UUIDs.  Map logical CUDA:0 to
  # its selected physical UUID when CUDA_VISIBLE_DEVICES is a single index/UUID.
  local selector first_selector gpu_rows selected_uuid apps matching
  selector="$CUDA_VISIBLE_DEVICES"
  first_selector="${selector%%,*}"
  gpu_rows="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits)"
  if [[ "$first_selector" =~ ^[0-9]+$ ]]; then
    selected_uuid="$(awk -F', ' -v i="$first_selector" '$1 == i {print $2}' <<<"$gpu_rows")"
  elif [[ "$first_selector" == GPU-* ]]; then
    selected_uuid="$(awk -F', ' -v u="$first_selector" '$2 == u {print $2}' <<<"$gpu_rows")"
  else
    fail "cannot unambiguously map CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES to a physical GPU for process safety"
  fi
  [[ -n "$selected_uuid" ]] || fail "selected CUDA GPU is not present in nvidia-smi inventory"
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  matching="$(awk -F', ' -v u="$selected_uuid" '$3 == u {print}' <<<"$apps")"
  [[ -z "$matching" ]] || fail "active compute process(es) detected on selected GPU UUID $selected_uuid: $matching"
  printf 'No active NVIDIA compute processes detected on selected GPU UUID %s.\n' "$selected_uuid"
}

render_command() {
  local seed="$1" split="${SPLITS[$1]}" output="${OUTPUTS[$1]}"
  printf '%q ' "$PYTHON" -m src.training.train_periodic_invariant_gnn \
    --dataset-path "$DATASET" --split-manifest-path "$split" --seed "$seed" \
    --output-dir "$output" --protocol-freeze-path "$PROTOCOL" \
    --epochs 300 --batch-size 8 --patience 40 --device cuda
  printf '\n'
}

validate_seed_artifacts() {
  local seed="$1" output="${OUTPUTS[$1]}" expected_split_sha="${SPLIT_SHAS[$1]}"
  local artifact
  for artifact in metadata.json epoch_metrics.csv best_validation_checkpoint.pt; do
    [[ -s "$output/$artifact" ]] || fail "seed $seed missing or empty required artifact: $output/$artifact"
  done
  check_test_artifacts "$output"
  "$PYTHON" - "$output" "$seed" "$expected_split_sha" "$PROTOCOL_SHA" <<'PY'
import csv, hashlib, json, math, sys
from pathlib import Path
out, expected_seed, expected_split, expected_protocol = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
m = json.loads((out / "metadata.json").read_text())
required = {
    "seed": expected_seed, "parameter_count": 5270,
    "protocol_freeze_sha256": expected_protocol,
    "split_manifest_sha256": expected_split,
    "test_status": "pending / forbidden", "scientific_run": True,
}
for key, expected in required.items():
    if m.get(key) != expected:
        raise SystemExit(f"metadata {key!r} mismatch: expected {expected!r}, got {m.get(key)!r}")
best_epoch = m.get("best_epoch")
best_mse = m.get("best_validation_mse")
if not isinstance(best_epoch, int) or best_epoch < 1:
    raise SystemExit("metadata best_epoch is invalid")
if not isinstance(best_mse, (int, float)) or not math.isfinite(best_mse):
    raise SystemExit("metadata best_validation_mse is not finite")
checkpoint = out / "best_validation_checkpoint.pt"
actual_checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if m.get("checkpoint_sha256") != actual_checkpoint_sha:
    raise SystemExit("metadata checkpoint_sha256 does not match checkpoint")
with (out / "epoch_metrics.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("epoch_metrics.csv has no data rows")
expected_epochs = list(range(1, len(rows) + 1))
try:
    epochs = [int(row["epoch"]) for row in rows]
    values = [float(row[key]) for row in rows for key in ("train_mse", "validation_mse")]
except (KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f"epoch_metrics.csv schema/value error: {exc}")
if epochs != expected_epochs or not all(math.isfinite(value) for value in values):
    raise SystemExit("epoch_metrics.csv epochs or metrics are invalid")
print(f"POSTCHECK seed={expected_seed} best_epoch={best_epoch} validation_mse={best_mse} checkpoint_sha256={actual_checkpoint_sha}")
PY
}

run_seed() {
  local seed="$1" output="${OUTPUTS[$1]}" log="$LOG_DIR/seed${1}.log"
  CURRENT_SEED="$seed"
  CURRENT_OUTPUT="$output"
  check_repository
  check_freezes_and_inputs
  check_split "$seed"
  check_gpu
  [[ ! -e "$output" ]] || fail "production output appeared before seed $seed; refusing overwrite/resume: $output"
  check_test_artifacts "$output"
  printf '\nSTARTING PRODUCTION SEED %s\n' "$seed"
  render_command "$seed"
  "$PYTHON" -m src.training.train_periodic_invariant_gnn \
    --dataset-path "$DATASET" \
    --split-manifest-path "${SPLITS[$seed]}" \
    --seed "$seed" \
    --output-dir "$output" \
    --protocol-freeze-path "$PROTOCOL" \
    --epochs 300 \
    --batch-size 8 \
    --patience 40 \
    --device cuda 2>&1 | tee "$log"
  validate_seed_artifacts "$seed"
  printf 'SEED %s COMPLETE\n' "$seed"
}

cd "$ROOT"
[[ -x "$PYTHON" ]] || fail "canonical Python is missing or not executable: $PYTHON"

if command -v flock >/dev/null; then
  exec 9>"$LOCK_FILE"
  flock -n 9 || fail "another production launcher holds $LOCK_FILE"
else
  fail "flock is required for safe production concurrency control"
fi

CURRENT_SEED="preflight"
check_repository
check_freezes_and_inputs
check_all_splits
check_trainer_cli
check_gpu
check_output_absence
check_all_test_artifacts

printf '\nProduction seed order: 42 -> 123 -> 2025\n'
printf 'Exact production commands:\n'
for seed in "${SEEDS[@]}"; do render_command "$seed"; done

if "$AUDIT_ONLY"; then
  printf '\nAUDIT-ONLY COMPLETE: no trainer was invoked and no production directory was created.\n'
  exit 0
fi

mkdir -p "$LOG_DIR"
for seed in "${SEEDS[@]}"; do run_seed "$seed"; done
check_all_test_artifacts

printf '\nPRODUCTION TRAINING COMPLETE\nTEST SET ACCESSED = NO\n'
for seed in "${SEEDS[@]}"; do
  "$PYTHON" - "${OUTPUTS[$seed]}/metadata.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"seed={m['seed']} output={sys.argv[1].rsplit('/', 1)[0]} best_epoch={m['best_epoch']} validation_mse={m['best_validation_mse']} checkpoint_sha256={m['checkpoint_sha256']}")
PY
done
