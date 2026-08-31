#!/usr/bin/env bash
set -Eeuo pipefail

readonly REQUIRED_BRANCH="thesis-notebook15-graph-representation"
readonly PYTHON="/home/ml/thesis-camels/envs/camels-gnn/bin/python"
readonly RAW_DIR="data/raw/CAMELS_SIMBA_1000U"
readonly TARGET_FILE="outputs/target_inspection_1000u.csv"
readonly TARGET_SHA256="9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2"
readonly FREEZE="reports/experiment_registry/u1000_top1500_radius_selection_freeze.json"
readonly CONTROL_METADATA="data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.metadata.json"
readonly BUILDER_MODULE="src.data.build_temporal_sequences"
readonly BUILD_LAUNCHER="scripts/production/run_u1000_top1500_radius_sparse_build.sh"
readonly VALIDATOR="scripts/validation/validate_u1000_top1500_radius_dataset.py"
readonly MIN_FREE_KIB=$((6 * 1024 * 1024))
readonly SNAPSHOTS=("0.20000" "0.25000" "0.51209" "0.75065" "1.00000")

usage() {
  printf '%s\n' \
    'Usage: scripts/production/run_u1000_top1500_radius_sparse_build.sh [--resolve-only|--preflight-only]' \
    '  --resolve-only    Resolve the frozen production identity without inspecting/building raw data.' \
    '  --preflight-only  Run all fail-closed checks and print the exact command without building.'
}

mode="build"
case "${1:-}" in
  "") ;;
  --resolve-only) mode="resolve-only" ;;
  --preflight-only) mode="preflight-only" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "$REPO_ROOT"
fail() { printf 'NO-GO: %s\n' "$*" >&2; exit 1; }
[[ -x "$PYTHON" ]] || fail "established Python interpreter unavailable: $PYTHON"
[[ -f "$FREEZE" ]] || fail "radius freeze record missing: $FREEZE"

mapfile -t frozen < <("$PYTHON" - "$FREEZE" <<'PY'
import json, math, sys
value=json.load(open(sys.argv[1], encoding='utf-8'))
radius=value.get('selected_radius')
assert isinstance(radius, (int,float)) and math.isfinite(radius) and radius > 0
assert value.get('radius_units') == 'h^-1 Mpc'
assert value.get('threshold_convention') == 'float32(minimum-image Euclidean distance) <= radius'
assert value.get('leakage_audit') == {
 'omega_m_used':False,'validation_ids_used':False,'validation_predictions_used':False,
 'validation_metrics_used':False,'test_ids_used':False,'test_predictions_used':False,
 'test_metrics_used':False,'predictive_performance_used':False}
identity=value['production_identity']
print(repr(float(radius)))
print(value['radius_token'])
print(identity['output_directory'])
print(identity['dataset_path'])
print(identity['metadata_path'])
print(identity['completion_marker_path'])
print(identity['logical_dataset_id'])
print(value['code_provenance']['implementation_git_commit'])
PY
) || fail "freeze record is malformed or violates the leakage contract"
readonly RADIUS="${frozen[0]}"
readonly RADIUS_TOKEN="${frozen[1]}"
readonly OUTPUT_DIR="${frozen[2]}"
readonly OUTPUT_FILE="${frozen[3]}"
readonly METADATA="${frozen[4]}"
readonly MARKER="${frozen[5]}"
readonly LOGICAL_DATASET_ID="${frozen[6]}"
readonly IMPLEMENTATION_COMMIT="${frozen[7]}"
readonly LOCK="${OUTPUT_FILE}.lock"

command=("$PYTHON" -m "$BUILDER_MODULE" --builder_entrypoint "$BUILDER_MODULE"
  --build_launcher_path "$BUILD_LAUNCHER" --raw_dir "$RAW_DIR" --output_path "$OUTPUT_FILE"
  --num_universes 1000 --num_snapshots 5 --num_nodes 1500 --normalization none
  --graph_mode radius --radius "$RADIUS" --periodic_boundary --box_size 25.0
  --graph_storage sparse_edge_index --source_manifest_policy full_sha256
  --targets_csv "$TARGET_FILE" --device cpu)
printf -v command_text '%q ' "${command[@]}"; command_text="${command_text% }"

print_resolution() {
  printf 'RADIUS=%s\nRADIUS_UNITS=h^-1 Mpc\nRADIUS_TOKEN=%s\nGRAPH_MODE=radius\nTOP_N=1500\nSNAPSHOTS=5\nNORMALIZATION=none\nPERIODIC=true\nBOX_SIZE=25\nGRAPH_STORAGE=sparse_edge_index\nRAW_DIR=%s\nTARGET_FILE=%s\nOUTPUT_DIR=%s\nOUTPUT_FILE=%s\nMETADATA=%s\nCOMPLETION_MARKER=%s\nLOGICAL_DATASET_ID=%s\nPYTHON=%s\nVALIDATOR=%s\nDESTINATION_ABSENT=%s\nBUILDER_INVOCATION=%s\n' \
    "$RADIUS" "$RADIUS_TOKEN" "$RAW_DIR" "$TARGET_FILE" "$OUTPUT_DIR" "$OUTPUT_FILE" \
    "$METADATA" "$MARKER" "$LOGICAL_DATASET_ID" "$PYTHON" "$VALIDATOR" \
    "$([[ ! -e "$OUTPUT_DIR" ]] && echo true || echo false)" "$command_text"
}
if [[ "$mode" == "resolve-only" ]]; then print_resolution; exit 0; fi

[[ "$(git branch --show-current)" == "$REQUIRED_BRANCH" ]] || fail "wrong branch"
git diff --quiet || fail "tracked unstaged changes exist"
git diff --cached --quiet || fail "staged changes exist"
git diff --check || fail "whitespace errors exist"
for relative in scripts/validation/calibrate_u1000_top1500_radius.py src/data/camels_graph_utils.py src/data/build_temporal_sequences.py "$BUILD_LAUNCHER"; do
  expected="$("$PYTHON" - "$FREEZE" "$relative" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])); p=sys.argv[2]; c=v['code_provenance']
mapping={c['calibration_tool_path']:c['calibration_tool_sha256'],c['graph_utility_path']:c['graph_utility_sha256'],c['builder_path']:c['builder_sha256'],c['launcher_path']:c['launcher_sha256']}
print(mapping[p])
PY
)"
  [[ "$(git show "${IMPLEMENTATION_COMMIT}:${relative}" | sha256sum | awk '{print $1}')" == "$expected" ]] || fail "frozen code hash mismatch: $relative"
done
[[ -d "$RAW_DIR" && -f "$TARGET_FILE" ]] || fail "raw-data root or target table missing"
[[ "$(sha256sum "$TARGET_FILE" | awk '{print $1}')" == "$TARGET_SHA256" ]] || fail "target SHA-256 mismatch"
[[ -f "$CONTROL_METADATA" ]] || fail "frozen k8 control metadata missing"
missing=0
for universe in $(seq 0 999); do
  for snapshot in "${SNAPSHOTS[@]}"; do
    [[ -r "${RAW_DIR}/LH_${universe}_hlist_${snapshot}.list" ]] || missing=$((missing+1))
  done
done
[[ "$missing" -eq 0 ]] || fail "$missing raw catalogues missing/unreadable"
[[ "$(find "$RAW_DIR" -maxdepth 1 -type f -name 'LH_*_hlist_*.list' -printf '.' | wc -c)" -eq 5000 ]] || fail "raw catalogue count is not 5000"
for artifact in "$OUTPUT_DIR" "$OUTPUT_FILE" "$METADATA" "$MARKER" "$LOCK"; do
  [[ ! -e "$artifact" ]] || fail "future destination collision: $artifact"
done
free_kib="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
(( free_kib >= MIN_FREE_KIB )) || fail "less than 6 GiB free"
print_resolution
printf 'PREFLIGHT PASS\nIMPLEMENTATION_COMMIT=%s\nFREE_DISK_KIB=%s\n' "$IMPLEMENTATION_COMMIT" "$free_kib"
[[ "$mode" == "preflight-only" ]] && exit 0

mkdir -p logs/dataset_builds "$OUTPUT_DIR"
log="logs/dataset_builds/u1000_top1500_radius_${RADIUS_TOKEN}_sparse_build_$(date -u +%Y%m%dT%H%M%SZ).log"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
set +e
nice -n 10 "${command[@]}" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
[[ "$status" -eq 0 ]] || { printf 'FAIL: builder exit %s; log=%s\n' "$status" "$log" >&2; exit "$status"; }
"$PYTHON" "$VALIDATOR" 2>&1 | tee -a "$log"
printf 'PASS\nDataset: %s\nMetadata: %s\nMarker: %s\n' "$OUTPUT_FILE" "$METADATA" "$MARKER" | tee -a "$log"
