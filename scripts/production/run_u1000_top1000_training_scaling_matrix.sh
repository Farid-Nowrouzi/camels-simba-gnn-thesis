#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PYTHON="${REPO_ROOT}/envs/camels-gnn/bin/python"
CONTROL="scripts/validation/manage_u1000_top1000_training_scaling_matrix.py"
LOG_DIR="${REPO_ROOT}/logs/u1000_top1000_training_scaling_matrix"
STATE_FILE="${LOG_DIR}/current_state.env"
LOCK_FILE="/tmp/u1000_top1000_training_scaling_matrix_${UID}.lock"
DATASET="data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt"
IDENTITY="6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a"

usage() {
  cat <<'EOF'
Usage: run_u1000_top1000_training_scaling_matrix.sh [--preflight-only|--resume|--status|--help]
  --preflight-only  Validate all inputs/configurations/environment; never train.
  --resume          Run the remaining matrix sequentially, validating each run.
  --status          Print the read-only registry summary.
EOF
}

mode="${1:---help}"
[[ $# -eq 1 ]] || { usage >&2; exit 2; }
case "${mode}" in
  --help) usage; exit 0 ;;
  --status) cd "${REPO_ROOT}"; exec "${PYTHON}" "${CONTROL}" --status ;;
  --preflight-only|--resume) ;;
  *) usage >&2; exit 2 ;;
esac

cd "${REPO_ROOT}"
[[ -x "${PYTHON}" ]] || { echo "FAIL: missing ${PYTHON}" >&2; exit 1; }
command -v flock >/dev/null || { echo "FAIL: flock is required" >&2; exit 1; }
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "FAIL: another matrix launcher owns ${LOCK_FILE}" >&2
  command -v fuser >/dev/null && fuser "${LOCK_FILE}" 2>/dev/null || true
  exit 73
fi

conflicts="$(ps -eo pid=,cmd= | awk -v self="$$" -v parent="${PPID}" '$1 != self && $1 != parent && $0 !~ /awk -v self=/ && $0 ~ /(src\.training\.train_(evolvegcn_h|static_gcn)|build_temporal_sequences|run_u1000_top1000_sparse_build)/ {print}')"
[[ -z "${conflicts}" ]] || { echo "FAIL: conflicting trainer/builder active:${conflicts}" >&2; exit 74; }

"${PYTHON}" "${CONTROL}" --preflight
bash scripts/refresh_experiment_registry.sh
[[ "${mode}" == "--preflight-only" ]] && { echo "PASS: preflight only; no experiment directory or training artifact created"; exit 0; }

mkdir -p "${LOG_DIR}"
MATRIX_START="$(date +%s)"
printf 'MATRIX_LAUNCHER_PID=%q\nMATRIX_START_EPOCH=%q\nCURRENT_MODEL=\nCURRENT_SEED=\nCURRENT_TRAIN_COUNT=\nCURRENT_EXPERIMENT=\nTRAINER_PID=\nRUN_START_EPOCH=\nLOG_FILE=\nEXPERIMENT_DIR=\n' "$$" "${MATRIX_START}" > "${STATE_FILE}"

sample_gpu() {
  local trainer_pid="$1" output="$2"
  echo "utc,pid,used_gpu_memory_mib,gpu_utilization_percent,free_gpu_memory_mib,host_rss_kib" > "${output}"
  while kill -0 "${trainer_pid}" 2>/dev/null; do
    local used util free rss
    used="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits | awk -F, -v p="${trainer_pid}" '$1+0==p {gsub(/ /,"",$2); print $2; exit}')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    rss="$(awk '/^VmRSS:/ {print $2}' "/proc/${trainer_pid}/status" 2>/dev/null || true)"
    echo "$(date -u +%FT%TZ),${trainer_pid},${used:-},${util:-},${free:-},${rss:-}" >> "${output}"
    sleep 5
  done
}

refresh_registry() {
  "${PYTHON}" "${CONTROL}" --refresh
  bash scripts/refresh_experiment_registry.sh
}

run_one() {
  local model="$1" count="$2" seed="$3" ident name manifest timestamp log telemetry runtime start end pid sampler status max_rss
  ident="u1000-top1000-sparse-${model}-train${count}-seed${seed}"
  if [[ "${model}" == evolve ]]; then
    name="evolvegcn_h_u1000_top1000_sparse_train${count}_seed${seed}_none_h32_l2_mean_temporal_mean_linear"
  else
    name="static_gcn_u1000_top1000_sparse_train${count}_seed${seed}_none_h32_l3_mean_mlp_final"
  fi
  manifest="configs/splits/u1000_top1000_none_k8_sparse/seed${seed}_train${count}.json"
  local registry_status
  registry_status="$("${PYTHON}" - "${ident}" <<'PY'
import json,sys
r=json.load(open('configs/experiment_registry/u1000_top1000_training_scaling_matrix.json'))
print(next(e['status'] for e in r['entries'] if e['canonical_experiment_id']==sys.argv[1]))
PY
)"
  if [[ "${registry_status}" == completed ]]; then
    "${PYTHON}" "${CONTROL}" --validate-run "${ident}" >/dev/null
    echo "SKIP completed and validated: ${ident}"
    return
  fi
  [[ "${registry_status}" == planned ]] || { echo "FAIL: ${ident} status=${registry_status}, expected planned" >&2; exit 1; }
  [[ ! -e "experiments/${name}" ]] || { echo "FAIL: preserving incomplete destination experiments/${name}" >&2; exit 1; }
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"; log="${LOG_DIR}/${ident}_${timestamp}.log"; telemetry="${LOG_DIR}/${ident}_${timestamp}_gpu.csv"; runtime="${LOG_DIR}/${ident}_${timestamp}_runtime.json"
  start="$(date +%s)"
  "${PYTHON}" "${CONTROL}" --set-status "${ident}" --status-value running
  refresh_registry
  local cmd=("${PYTHON}")
  if [[ "${model}" == evolve ]]; then
    cmd+=( -m src.training.train_evolvegcn_h --dataset_path "${DATASET}" --split_manifest_path "${manifest}" --dataset_identity "${IDENTITY}" --experiment_name "${name}" --output_root experiments --seed "${seed}" --batch_size 4 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 0.00001 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu --temporal_pooling mean --graph_pooling mean --head_type linear --add_self_loops --train_ratio 0.7 --val_ratio 0.099 --test_ratio 0.201 --grad_clip_norm 1.0 --device cuda )
  else
    cmd+=( -m src.training.train_static_gcn --dataset_path "${DATASET}" --dataset_format temporal_final_snapshot --split_manifest_path "${manifest}" --dataset_identity "${IDENTITY}" --experiment_name "${name}" --output_root experiments --seed "${seed}" --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 0.00001 --hidden_dim 32 --num_layers 3 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7 --val_ratio 0.099 --test_ratio 0.201 --grad_clip_norm 1.0 --device cuda )
  fi
  "${cmd[@]}" > >(tee -a "${log}") 2>&1 & pid=$!
  printf 'MATRIX_LAUNCHER_PID=%q\nMATRIX_START_EPOCH=%q\nCURRENT_MODEL=%q\nCURRENT_SEED=%q\nCURRENT_TRAIN_COUNT=%q\nCURRENT_EXPERIMENT=%q\nTRAINER_PID=%q\nRUN_START_EPOCH=%q\nLOG_FILE=%q\nEXPERIMENT_DIR=%q\n' "$$" "${MATRIX_START}" "${model}" "${seed}" "${count}" "${name}" "${pid}" "${start}" "${log}" "${REPO_ROOT}/experiments/${name}" > "${STATE_FILE}"
  sample_gpu "${pid}" "${telemetry}" & sampler=$!
  set +e; wait "${pid}"; status=$?; set -e
  wait "${sampler}" 2>/dev/null || true; end="$(date +%s)"
  max_rss="$(awk -F, 'NR>1 && $6+0>p {p=$6+0} END {if(p) print p}' "${telemetry}")"
  "${PYTHON}" - "${runtime}" "$((end-start))" "${max_rss:-}" "${status}" "${log}" <<'PY'
import json,sys
json.dump({'runtime_seconds':int(sys.argv[2]),'max_host_rss_kib':int(sys.argv[3]) if sys.argv[3] else None,'exit_status':int(sys.argv[4]),'log_file':sys.argv[5]},open(sys.argv[1],'w'),indent=2)
PY
  if [[ ${status} -ne 0 ]]; then
    "${PYTHON}" "${CONTROL}" --set-status "${ident}" --status-value failed --reason "trainer exit ${status}; log=${log}"
    refresh_registry; exit "${status}"
  fi
  if ! "${PYTHON}" "${CONTROL}" --validate-run "${ident}" --runtime-json "${runtime}" --telemetry "${telemetry}"; then
    "${PYTHON}" "${CONTROL}" --set-status "${ident}" --status-value failed --reason "post-run artifact validation failed; log=${log}"
    refresh_registry; exit 1
  fi
  "${PYTHON}" "${CONTROL}" --set-status "${ident}" --status-value completed --runtime-json "${runtime}" --telemetry "${telemetry}"
  refresh_registry
}

for count in 450 200 100 50 20; do run_one evolve "${count}" 42; run_one static "${count}" 42; done
for seed in 123 2025; do
  for count in 700 450 200 100 50 20; do run_one evolve "${count}" "${seed}"; run_one static "${count}" "${seed}"; done
done
"${PYTHON}" "${CONTROL}" --aggregate
refresh_registry
printf 'MATRIX_LAUNCHER_PID=\nMATRIX_START_EPOCH=%q\nCURRENT_MODEL=complete\nCURRENT_SEED=\nCURRENT_TRAIN_COUNT=\nCURRENT_EXPERIMENT=\nTRAINER_PID=\nRUN_START_EPOCH=\nLOG_FILE=\nEXPERIMENT_DIR=\n' "${MATRIX_START}" > "${STATE_FILE}"
echo "SUCCESS: 36/36 matrix cells completed and validated sequentially"
