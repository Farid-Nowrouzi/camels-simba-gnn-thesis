#!/usr/bin/env bash
set -Eeuo pipefail

readonly REQUIRED_BRANCH="thesis-sparse-integrity-hardening"
readonly RAW_DIR="data/raw/CAMELS_SIMBA_1000U"
readonly TARGET_FILE="outputs/target_inspection_1000u.csv"
readonly TARGET_SHA256="9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"
readonly OUTPUT_DIR="data/processed/temporal_1000u_none_top1500_periodic_knn_sparse"
readonly OUTPUT_FILE="${OUTPUT_DIR}/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
readonly MIN_FREE_KIB=$((4 * 1024 * 1024))
readonly SNAPSHOTS=("0.20000" "0.25000" "0.51209" "0.75065" "1.00000")

usage() {
  cat <<'EOF'
Usage: scripts/production/run_u1000_top1500_sparse_build.sh [--preflight-only|--help]
  --preflight-only  Run every safety/identity check and print the exact builder command.
  --help            Show this help. This launcher never starts tmux.
EOF
}

mode="${1:-}"
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "$mode" in "") ;; --preflight-only) ;; -h|--help) usage; exit 0 ;; *) usage >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "$REPO_ROOT"
fail() { printf 'NO-GO: %s\n' "$*" >&2; exit 1; }

branch="$(git branch --show-current)"
[[ "$branch" == "$REQUIRED_BRANCH" ]] || fail "required branch ${REQUIRED_BRANCH}; found ${branch}"
[[ -d "$RAW_DIR" && -f "$TARGET_FILE" ]] || fail "raw-data root or target table missing"
actual_target="$(sha256sum "$TARGET_FILE" | awk '{print $1}')"
[[ "$actual_target" == "$TARGET_SHA256" ]] || fail "target SHA-256 mismatch"
missing=0
for universe in $(seq 0 999); do
  for snapshot in "${SNAPSHOTS[@]}"; do
    [[ -r "${RAW_DIR}/LH_${universe}_hlist_${snapshot}.list" ]] || missing=$((missing+1))
  done
done
[[ "$missing" -eq 0 ]] || fail "${missing} raw catalogues missing/unreadable"
count="$(find "$RAW_DIR" -maxdepth 1 -type f -name 'LH_*_hlist_*.list' -printf '.' | wc -c)"
[[ "$count" -eq 5000 ]] || fail "expected exactly 5,000 catalogues; found ${count}"
[[ -f reports/experiment_registry/u1000_top1500_raw_halo_count_distribution.csv ]] || fail "raw Top1500 audit missing"
[[ -x envs/camels-gnn/bin/python ]] || fail "envs/camels-gnn unavailable"
source_identity="$(envs/camels-gnn/bin/python - <<'PY'
import json
from src.data.source_manifest import verify_full_source_manifest
metadata=json.load(open('data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.metadata.json'))
manifest=metadata['source_manifest']
result=verify_full_source_manifest(manifest,source_roots={'halo_catalogue':'data/raw/CAMELS_SIMBA_1000U','target_table':'outputs'})
print(result['manifest_sha256'])
PY
)" || fail "raw/target full-SHA256 identity verification against Top1000 anchor failed"

metadata="${OUTPUT_FILE%.pt}.metadata.json"; marker="${OUTPUT_FILE%.pt}.complete"; lock="${OUTPUT_FILE}.lock"
if [[ -f "$OUTPUT_FILE" && -f "$metadata" && -f "$marker" ]]; then
  if envs/camels-gnn/bin/python scripts/validation/validate_u1000_top1500_sparse_dataset.py >/dev/null 2>&1; then
    fail "valid completed Top1500 dataset already exists; refusing overwrite"
  fi
fi
for artifact in "$OUTPUT_FILE" "$metadata" "$marker" "$lock"; do
  [[ ! -e "$artifact" ]] || fail "partial/invalid output exists and is preserved: ${artifact}"
done
if [[ -d "$OUTPUT_DIR" ]]; then
  shopt -s nullglob; partials=("${OUTPUT_DIR}/."*.tmp); shopt -u nullglob
  [[ ${#partials[@]} -eq 0 ]] || fail "atomic temporary output exists and is preserved: ${partials[*]}"
fi
free_kib="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
(( free_kib >= MIN_FREE_KIB )) || fail "less than 4 GiB free"
conflicts="$(ps -eo pid=,args= | awk '$0 ~ /python/ && $0 ~ /build_temporal_sequences/ && $0 !~ /--help/ {print}')"
[[ -z "$conflicts" ]] || fail "conflicting builder active: ${conflicts}"
[[ -x envs/camels-gnn/bin/python && -f envs/camels-gnn/bin/activate ]] || fail "envs/camels-gnn unavailable"
# shellcheck disable=SC1091
source envs/camels-gnn/bin/activate

command=(python -m src.data.build_temporal_sequences --raw_dir "$RAW_DIR" --output_path "$OUTPUT_FILE"
  --num_universes 1000 --num_snapshots 5 --num_nodes 1500 --normalization none --graph_mode knn
  --k 8 --periodic_boundary --box_size 25.0 --graph_storage sparse_edge_index
  --source_manifest_policy full_sha256 --targets_csv "$TARGET_FILE" --device cpu)
printf -v command_text '%q ' "${command[@]}"; command_text="${command_text% }"
printf 'PREFLIGHT PASS\nBranch: %s\nRaw catalogues: %s/5000\nSource manifest: %s\nTarget SHA-256: %s\nFree disk: %s KiB\nExact command: %s\n' \
  "$branch" "$count" "$source_identity" "$actual_target" "$free_kib" "$command_text"
[[ "$mode" == --preflight-only ]] && { echo 'PREFLIGHT-ONLY COMPLETE: build not started.'; exit 0; }

mkdir -p logs/dataset_builds "$OUTPUT_DIR"
log="logs/dataset_builds/u1000_top1500_sparse_build_$(date -u +%Y%m%dT%H%M%SZ).log"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
{
  echo 'U1000 Top1500 sparse production build'; echo "Git commit: $(git rev-parse HEAD)"; echo "Branch: $branch"
  echo "UTC: $(date -u --iso-8601=seconds)"; echo "Exact command: $command_text"; free -h; df -h "$REPO_ROOT"
} | tee "$log"
priority=(nice -n 10); command -v ionice >/dev/null && priority+=(ionice -c2 -n7)
set +e; "${priority[@]}" "${command[@]}" 2>&1 | tee -a "$log"; status=${PIPESTATUS[0]}; set -e
[[ "$status" -eq 0 ]] || { echo "FAIL: builder exit ${status}; partials preserved; log=${log}" | tee -a "$log" >&2; exit "$status"; }
envs/camels-gnn/bin/python scripts/validation/validate_u1000_top1500_sparse_dataset.py 2>&1 | tee -a "$log"
echo "Dataset: $OUTPUT_FILE" | tee -a "$log"; echo "Metadata: $metadata" | tee -a "$log"; echo "Marker: $marker" | tee -a "$log"
sha256sum "$OUTPUT_FILE" "$metadata" "$marker" | tee -a "$log"
