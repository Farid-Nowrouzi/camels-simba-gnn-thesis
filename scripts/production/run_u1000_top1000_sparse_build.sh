#!/usr/bin/env bash
set -Eeuo pipefail

readonly REQUIRED_BRANCH="thesis-sparse-integrity-hardening"
readonly EXPECTED_TARGET_SHA256="9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"
readonly RAW_DIR="data/raw/CAMELS_SIMBA_1000U"
readonly TARGET_FILE="outputs/target_inspection_1000u.csv"
readonly OUTPUT_DIR="data/processed/temporal_1000u_none_top1000_periodic_knn_sparse"
readonly OUTPUT_FILE="${OUTPUT_DIR}/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt"
readonly MIN_FREE_KIB=$((5 * 1024 * 1024))
readonly SNAPSHOTS=("0.20000" "0.25000" "0.51209" "0.75065" "1.00000")

usage() {
  cat <<'EOF'
Usage: scripts/production/run_u1000_top1000_sparse_build.sh [--preflight-only]

  --preflight-only  Run all shell-level safety checks and print the exact
                    production command without invoking the Python builder.
  -h, --help        Show this help and exit.
EOF
}

preflight_only=false
case "${1:-}" in
  "") ;;
  --preflight-only) preflight_only=true ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "$REPO_ROOT"

fail() {
  printf 'NO-GO: %s\n' "$*" >&2
  exit 1
}

current_branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)" || \
  fail "repository is in detached-HEAD state"
[[ "$current_branch" == "$REQUIRED_BRANCH" ]] || \
  fail "required branch is ${REQUIRED_BRANCH}; current branch is ${current_branch}"

[[ -f "$TARGET_FILE" ]] || fail "target file is missing: ${TARGET_FILE}"
actual_target_sha256="$(sha256sum "$TARGET_FILE" | awk '{print $1}')"
[[ "$actual_target_sha256" == "$EXPECTED_TARGET_SHA256" ]] || \
  fail "target SHA-256 mismatch: expected ${EXPECTED_TARGET_SHA256}, got ${actual_target_sha256}"

[[ -d "$RAW_DIR" ]] || fail "raw-data root is missing: ${RAW_DIR}"
missing_count=0
for universe_id in $(seq 0 999); do
  for snapshot in "${SNAPSHOTS[@]}"; do
    catalogue="${RAW_DIR}/LH_${universe_id}_hlist_${snapshot}.list"
    if [[ ! -r "$catalogue" ]]; then
      printf 'Missing or unreadable catalogue: %s\n' "$catalogue" >&2
      missing_count=$((missing_count + 1))
    fi
  done
done
[[ "$missing_count" -eq 0 ]] || fail "${missing_count} required raw catalogues are missing or unreadable"

catalogue_count="$(find "$RAW_DIR" -maxdepth 1 -type f -name 'LH_*_hlist_*.list' -printf '.' | wc -c)"
[[ "$catalogue_count" -eq 5000 ]] || \
  fail "expected exactly 5000 raw catalogues, found ${catalogue_count}"

metadata_file="${OUTPUT_FILE%.pt}.metadata.json"
complete_file="${OUTPUT_FILE%.pt}.complete"
lock_file="${OUTPUT_FILE}.lock"
for artifact in "$OUTPUT_FILE" "$metadata_file" "$complete_file" "$lock_file"; do
  [[ ! -e "$artifact" ]] || fail "output, partial output, or lock already exists: ${artifact}"
done
if [[ -d "$OUTPUT_DIR" ]]; then
  shopt -s nullglob
  temp_files=("${OUTPUT_DIR}/.${OUTPUT_FILE##*/}."*.tmp "${OUTPUT_DIR}/.${metadata_file##*/}."*.tmp)
  shopt -u nullglob
  [[ ${#temp_files[@]} -eq 0 ]] || fail "temporary atomic output exists: ${temp_files[*]}"
fi

free_kib="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || fail "could not determine free disk space"
(( free_kib >= MIN_FREE_KIB )) || \
  fail "less than 5 GiB is free: ${free_kib} KiB"

mapfile -t active_jobs < <(
  ps -eo pid=,args= | awk '
    $0 ~ /python/ &&
    $0 ~ /(-m[[:space:]]+src\.data\.build_temporal_sequences|src\/data\/build_temporal_sequences\.py|-m[[:space:]]+src\.training\.train_|src\/training\/train_|train_(static_gcn|evolvegcn)\.py)/ &&
    $0 !~ /--help/ {print}'
)
[[ ${#active_jobs[@]} -eq 0 ]] || {
  printf 'Conflicting builder or trainer process(es):\n%s\n' "${active_jobs[*]}" >&2
  fail "another dataset builder or model trainer is active"
}

[[ -f envs/camels-gnn/bin/activate ]] || fail "Python environment activation script is missing"
[[ -x envs/camels-gnn/bin/python ]] || fail "Python environment executable is missing"
# shellcheck disable=SC1091
source envs/camels-gnn/bin/activate

builder_help="$(python -m src.data.build_temporal_sequences --help 2>&1)" || \
  fail "builder CLI --help failed in envs/camels-gnn"
required_builder_flags=(
  --raw_dir --output_path --num_universes --num_snapshots --num_nodes
  --normalization --graph_mode --k --periodic_boundary --box_size
  --graph_storage --source_manifest_policy --targets_csv --device
)
for required_flag in "${required_builder_flags[@]}"; do
  [[ "$builder_help" == *"$required_flag"* ]] || \
    fail "builder CLI is missing reviewed flag: ${required_flag}"
done

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""

BUILD_COMMAND=(
  python -m src.data.build_temporal_sequences
  --raw_dir "$RAW_DIR"
  --output_path "$OUTPUT_FILE"
  --num_universes 1000
  --num_snapshots 5
  --num_nodes 1000
  --normalization none
  --graph_mode knn
  --k 8
  --periodic_boundary
  --box_size 25.0
  --graph_storage sparse_edge_index
  --source_manifest_policy full_sha256
  --targets_csv "$TARGET_FILE"
  --device cpu
)

printf -v command_text '%q ' "${BUILD_COMMAND[@]}"
command_text="${command_text% }"

printf 'PREFLIGHT PASS\n'
printf 'Repository:       %s\n' "$REPO_ROOT"
printf 'Branch:           %s\n' "$current_branch"
printf 'Target SHA-256:   %s\n' "$actual_target_sha256"
printf 'Raw catalogues:   %s/5000\n' "$catalogue_count"
printf 'Builder CLI:      compatible with all reviewed flags\n'
printf 'Free disk:        %s KiB\n' "$free_kib"
printf 'Output:           %s/%s\n' "$REPO_ROOT" "$OUTPUT_FILE"
printf 'Exact command:    %s\n' "$command_text"

if [[ "$preflight_only" == true ]]; then
  printf 'PREFLIGHT-ONLY COMPLETE: Python construction was not invoked.\n'
  exit 0
fi

mkdir -p logs/dataset_builds "$OUTPUT_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="logs/dataset_builds/u1000_top1000_sparse_build_${timestamp}.log"

{
  printf 'U1000 Top1000 sparse production build\n'
  printf 'Git commit: %s\n' "$(git rev-parse HEAD)"
  printf 'Branch: %s\n' "$current_branch"
  printf 'Hostname: %s\n' "$(hostname)"
  printf 'Date (UTC): %s\n' "$(date -u --iso-8601=seconds)"
  printf 'Python version: %s\n' "$(python --version 2>&1)"
  printf 'Target SHA-256: %s\n' "$actual_target_sha256"
  printf 'Free disk:\n'
  df -h "$REPO_ROOT"
  printf 'Memory:\n'
  free -h
  printf 'Thread limits: OMP=4 MKL=4 OPENBLAS=4 NUMEXPR=4\n'
  printf 'CPU priority: nice -n 10\n'
  if command -v ionice >/dev/null 2>&1; then
    printf 'I/O priority: ionice -c2 -n7\n'
  else
    printf 'I/O priority: ionice unavailable; no I/O priority wrapper\n'
  fi
  printf 'CUDA_VISIBLE_DEVICES is empty; build device is CPU\n'
  printf 'Exact command: %s\n' "$command_text"
} | tee "$log_file"

priority_command=(nice -n 10)
if command -v ionice >/dev/null 2>&1; then
  priority_command+=(ionice -c2 -n7)
fi

set +e
"${priority_command[@]}" "${BUILD_COMMAND[@]}" 2>&1 | tee -a "$log_file"
builder_status=${PIPESTATUS[0]}
set -e

if [[ "$builder_status" -eq 0 ]]; then
  printf 'SUCCESS: production dataset build completed.\n' | tee -a "$log_file"
else
  printf 'FAILURE: production dataset builder exited with status %s.\n' "$builder_status" | tee -a "$log_file" >&2
fi
printf 'Expected output: %s/%s\n' "$REPO_ROOT" "$OUTPUT_FILE" | tee -a "$log_file"
printf 'Build log: %s/%s\n' "$REPO_ROOT" "$log_file" | tee -a "$log_file"
exit "$builder_status"
