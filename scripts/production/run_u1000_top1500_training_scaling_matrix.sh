#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"; cd "$ROOT"
PYTHON="${ROOT}/envs/camels-gnn/bin/python"
CONTROL="scripts/validation/manage_u1000_top1500_training_scaling_matrix.py"
DATASET="data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
LOG_DIR="logs/u1000_top1500_training_scaling_matrix"; STATE_FILE="${LOG_DIR}/current_state.env"
LOCK_FILE="/tmp/u1000_top1500_training_scaling_matrix_${UID}.lock"

usage() { cat <<'EOF'
Usage: scripts/production/run_u1000_top1500_training_scaling_matrix.sh [--preflight-only|--resume|--status|--help]
  --preflight-only  Validate gates only; never starts a trainer.
  --resume          Run planned cells sequentially in the approved order.
  --status          Print read-only lifecycle counts.
EOF
}
mode="${1:---help}"; [[ $# -eq 1 ]] || { usage >&2; exit 2; }
case "$mode" in --help) usage; exit 0 ;; --status) exec "$PYTHON" "$CONTROL" --status ;; --preflight-only|--resume) ;; *) usage >&2; exit 2 ;; esac
[[ "$(git branch --show-current)" == thesis-sparse-integrity-hardening ]] || { echo 'FAIL: wrong branch' >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "FAIL: missing $PYTHON" >&2; exit 1; }
exec 9>"$LOCK_FILE"; flock -n 9 || { echo 'FAIL: concurrent Top1500 launcher owns lock' >&2; exit 73; }
conflicts="$(ps -eo pid=,args= | awk '$0 ~ /(train_evolvegcn_h|train_static_gcn|build_temporal_sequences)/ && $0 !~ /--help/ && $0 !~ /awk/ {print}')"
[[ -z "$conflicts" ]] || { echo "FAIL: conflicting builder/trainer active: $conflicts" >&2; exit 74; }
[[ -f "$DATASET" ]] || { echo "TOP1500 BUILD REQUIRED BEFORE CUDA: missing $DATASET" >&2; exit 3; }
"$PYTHON" scripts/validation/validate_u1000_top1500_sparse_dataset.py
"$PYTHON" "$CONTROL" --preflight
dirty_science="$(git diff --name-only -- src scripts configs ':!configs/experiment_registry/u1000_top1000_training_scaling_matrix.json')"
staged_science="$(git diff --cached --name-only -- src scripts configs)"
[[ -z "$dirty_science" && -z "$staged_science" ]] || { echo "FAIL: tracked/staged scientific code is not clean: $dirty_science $staged_science" >&2; exit 75; }
[[ "$mode" == --preflight-only ]] && { echo 'PASS: preflight only; no trainer or epoch loop started'; exit 0; }

mkdir -p "$LOG_DIR"; MATRIX_START="$(date +%s)"
printf 'MATRIX_LAUNCHER_PID=%q\nMATRIX_START_EPOCH=%q\nCURRENT_MODEL=\nCURRENT_SEED=\nCURRENT_TRAIN_COUNT=\nCURRENT_EXPERIMENT=\nTRAINER_PID=\nRUN_START_EPOCH=\nLOG_FILE=\nEXPERIMENT_DIR=\n' "$$" "$MATRIX_START" > "$STATE_FILE"

run_one() {
  local model="$1" count="$2" seed="$3" ident exp manifest config status log start cmd_status
  ident="u1000-top1500-sparse-${model}-train${count}-seed${seed}"
  if [[ "$model" == evolve ]]; then
    exp="evolvegcn_h_u1000_top1500_sparse_train${count}_seed${seed}_none_h32_l2_mean_temporal_mean_linear"
  else
    exp="static_gcn_u1000_top1500_sparse_train${count}_seed${seed}_none_h32_l3_mean_mlp_final"
  fi
  manifest="configs/splits/u1000_top1500_none_k8_sparse/seed${seed}_train${count}.json"
  config="configs/production/u1000_top1500_training_scaling/${exp}.json"
  status="$("$PYTHON" - "$ident" <<'PY'
import json,sys
r=json.load(open('configs/experiment_registry/u1000_top1500_training_scaling_matrix.json'))
print(next(item['status'] for item in r['entries'] if item['canonical_experiment_id']==sys.argv[1]))
PY
)"
  if [[ "$status" == completed ]]; then "$PYTHON" "$CONTROL" --validate-run "$ident" >/dev/null; echo "SKIP validated completion: $ident"; return; fi
  [[ "$status" == planned ]] || { echo "FAIL: $ident status=$status" >&2; exit 1; }
  [[ ! -e "experiments/$exp" ]] || { echo "FAIL: preserving incomplete destination experiments/$exp" >&2; exit 1; }
  log="${LOG_DIR}/${ident}_$(date -u +%Y%m%dT%H%M%SZ).log"; start="$(date +%s)"
  "$PYTHON" "$CONTROL" --set-status "$ident" --status-value running
  dataset_id="$("$PYTHON" -c "import json;print(json.load(open('$config'))['dataset_sha256'])")"
  local cmd=("$PYTHON")
  if [[ "$model" == evolve ]]; then
    cmd+=( -m src.training.train_evolvegcn_h --dataset_path "$DATASET" --split_manifest_path "$manifest" --dataset_identity "$dataset_id"
      --experiment_name "$exp" --output_root experiments --seed "$seed" --batch_size 4 --epochs 300 --patience 40
      --learning_rate 0.001 --weight_decay 0.00001 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu
      --temporal_pooling mean --graph_pooling mean --head_type linear --add_self_loops --train_ratio 0.7 --val_ratio 0.099
      --test_ratio 0.201 --grad_clip_norm 1.0 --device cuda )
  else
    cmd+=( -m src.training.train_static_gcn --dataset_path "$DATASET" --dataset_format temporal_final_snapshot
      --split_manifest_path "$manifest" --dataset_identity "$dataset_id" --experiment_name "$exp" --output_root experiments
      --seed "$seed" --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 0.00001
      --hidden_dim 32 --num_layers 3 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7
      --val_ratio 0.099 --test_ratio 0.201 --grad_clip_norm 1.0 --device cuda )
  fi
  printf 'MATRIX_LAUNCHER_PID=%q\nMATRIX_START_EPOCH=%q\nCURRENT_MODEL=%q\nCURRENT_SEED=%q\nCURRENT_TRAIN_COUNT=%q\nCURRENT_EXPERIMENT=%q\nTRAINER_PID=foreground\nRUN_START_EPOCH=%q\nLOG_FILE=%q\nEXPERIMENT_DIR=%q\n' \
    "$$" "$MATRIX_START" "$model" "$seed" "$count" "$exp" "$start" "$log" "$ROOT/experiments/$exp" > "$STATE_FILE"
  printf -v command_text '%q ' "${cmd[@]}"; echo "Exact command: ${command_text% }" | tee "$log"
  set +e; "${cmd[@]}" 2>&1 | tee -a "$log"; cmd_status=${PIPESTATUS[0]}; set -e
  if [[ "$cmd_status" -ne 0 ]]; then "$PYTHON" "$CONTROL" --set-status "$ident" --status-value failed --reason "trainer exit $cmd_status; $log"; exit "$cmd_status"; fi
  if ! "$PYTHON" "$CONTROL" --validate-run "$ident" | tee -a "$log"; then
    "$PYTHON" "$CONTROL" --set-status "$ident" --status-value failed --reason "post-run validation failed; $log"; exit 1
  fi
  "$PYTHON" "$CONTROL" --set-status "$ident" --status-value completed
}

for seed in 42 123 2025; do
  for count in 700 450 200 100 50 20; do run_one evolve "$count" "$seed"; run_one static "$count" "$seed"; done
done
"$PYTHON" "$CONTROL" --aggregate
echo 'SUCCESS: Top1500 matrix completed 36/36 sequentially and validated'
